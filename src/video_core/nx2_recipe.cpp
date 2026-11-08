// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/nx2_recipe.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <map>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <fmt/format.h>

#include "common/cityhash.h"
#include "common/zstd_compression.h"
#include "shader_recompiler/frontend/ampere/instruction.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ampere/program_address_map.h"
#include "shader_recompiler/frontend/ampere/program_capture.h"
#include "shader_recompiler/frontend/ampere/program_override.h"
#include "shader_recompiler/program_header.h"

namespace VideoCommon::Nx2Recipe {
namespace {

namespace X = FileSys::Nx2ShaderExtract;
using Shader::Stage;
using Shader::Ampere::Instruction;

constexpr u64 NativeHashTag = 0x4f554e4345000002ULL; // shader_environment.cpp NativeShaderHash
constexpr size_t SphSize = sizeof(Shader::ProgramHeader);
constexpr size_t NativeHeaderSize = 128;
constexpr size_t EnvHeaderSize = 76;    // GenericEnvironment::Serialize up to the code
constexpr size_t GraphicsKeySize = 1056;
constexpr size_t KeyHashesSize = 48;    // six u64 shader hashes at the start of a graphics key
constexpr size_t StateSize = GraphicsKeySize - KeyHashesSize;
constexpr size_t ComputeKeySize = 24;
constexpr u32 Format = 1;
constexpr std::string_view RecipeMagic{"NX2RCP01"};
constexpr std::string_view PrecompiledMagic{"NX2PREP1"};

using Bytes = std::span<const u8>;

template <typename T>
T Load(Bytes bytes, size_t offset) {
    if (offset > bytes.size() || sizeof(T) > bytes.size() - offset) {
        throw std::runtime_error(fmt::format("truncated input at {:#x}", offset));
    }
    T value;
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return value;
}

struct Writer {
    std::vector<u8> bytes;
    template <typename T>
    void Put(const T& value) {
        const auto* p = reinterpret_cast<const u8*>(&value);
        bytes.insert(bytes.end(), p, p + sizeof(T));
    }
    void Put(Bytes data) {
        bytes.insert(bytes.end(), data.begin(), data.end());
    }
    void Put(const std::vector<u8>& data) {
        bytes.insert(bytes.end(), data.begin(), data.end());
    }
    void Put(const char* data, size_t n) {
        bytes.insert(bytes.end(), reinterpret_cast<const u8*>(data),
                     reinterpret_cast<const u8*>(data) + n);
    }
    void Zeros(size_t n) {
        bytes.insert(bytes.end(), n, u8{0});
    }
};

struct Reader {
    Bytes bytes;
    size_t at = 0;
    template <typename T>
    T Get() {
        const T value = Load<T>(bytes, at);
        at += sizeof(T);
        return value;
    }
    Bytes Take(size_t n) {
        if (n > bytes.size() - at) {
            throw std::runtime_error("truncated input");
        }
        const auto part = bytes.subspan(at, n);
        at += n;
        return part;
    }
};

template <typename F>
void ParallelFor(size_t count, unsigned threads, F&& body) {
    const unsigned n = threads ? threads : std::max(1u, std::thread::hardware_concurrency());
    std::atomic<size_t> next{0};
    std::vector<std::jthread> pool;
    for (unsigned t = 0; t < n; ++t) {
        pool.emplace_back([&] {
            for (size_t i; (i = next.fetch_add(1)) < count;) {
                body(i);
            }
        });
    }
}

/// Where a serialized environment's parts are (GenericEnvironment::Serialize).
struct EnvParts {
    Stage stage{};
    size_t code_end{};   ///< the code is [EnvHeaderSize, code_end)
    size_t tables_end{}; ///< then the fact tables
    size_t end{};        ///< then compute's workgroup and shared size, or the program header and
                         ///< what follows it (geometry pass-through mask, mesh registers)
};

EnvParts Parse(Bytes env, bool has_code) {
    EnvParts parts;
    const u64 code_size = has_code ? Load<u64>(env, 0) : 0;
    const u64 types = Load<u64>(env, 8), formats = Load<u64>(env, 16), cbufs = Load<u64>(env, 24),
              pointers = Load<u64>(env, 32), replacements = Load<u64>(env, 40);
    parts.stage = static_cast<Stage>(Load<u32>(env, 72));
    parts.code_end = EnvHeaderSize + code_size;
    parts.tables_end = parts.code_end + types * 8 + formats * 8 + cbufs * 12 + pointers * 9 +
                       replacements * 12;
    parts.end = env.size();
    if (code_size % 8 != 0 || parts.tables_end > parts.end) {
        throw std::runtime_error("bad shader environment");
    }
    return parts;
}

/// The pass-through geometry header a mesh program's code area holds (control +0x68 bit 7, at the
/// offset +0x64 gives), as a mesh draw's environment records it.
std::optional<std::array<u8, SphSize>> MeshHeader(const X::Program& mesh) {
    const auto& control = mesh.control;
    if (control.size() <= 0x68 || !(control[0x68] & 0x80)) {
        return std::nullopt;
    }
    const u32 at = Load<u32>(control, 0x64);
    if (u64{at} + SphSize > mesh.code.size()) {
        return std::nullopt;
    }
    std::array<u8, SphSize> header;
    std::memcpy(header.data(), mesh.code.data() + at, SphSize);
    return header;
}

// Fact set flags. Bit 0 is not used: a recipe never carries a program header.
constexpr u8 MeshHeaderFromCode = 2; ///< a mesh environment's geometry header, taken from the code
constexpr size_t PassthroughMaskSize = 32; // GenericEnvironment gp_passthrough_mask
constexpr size_t MeshRegistersSize = 12;   // GenericEnvironment mesh_registers
constexpr size_t ComputeTailSize = 16;     // workgroup size and shared memory size
constexpr size_t CodeWindow = 32;

bool Dense(Bytes window) {
    return std::count_if(window.begin(), window.end(), [](u8 b) { return b != 0; }) >=
           static_cast<std::ptrdiff_t>(window.size() / 2);
}

u64 WindowHash(const u8* data) {
    return Common::CityHash64(reinterpret_cast<const char*>(data), CodeWindow);
}

/// Whether `facts` hold any 32 bytes of `code` (at an 8-byte step of the code, and dense: at least
/// half its bytes not zero, as instructions are, where mostly zero header fields can match the
/// facts' small numbers by chance).
bool HoldsCode(Bytes facts, Bytes code) {
    std::unordered_set<u64> windows;
    for (size_t at = 0; at + CodeWindow <= code.size(); at += 8) {
        if (Dense(code.subspan(at, CodeWindow))) {
            windows.insert(WindowHash(code.data() + at));
        }
    }
    for (size_t at = 0; at + CodeWindow <= facts.size(); at += 4) {
        if (windows.contains(WindowHash(facts.data() + at))) {
            return true;
        }
    }
    return false;
}

bool AllZero(Bytes bytes) {
    return std::all_of(bytes.begin(), bytes.end(), [](u8 b) { return b == 0; });
}

struct Program {
    const X::Program* source{};
    std::optional<Lowered> lowered;
    std::string error;
};

std::vector<u8> Compress(Bytes body) {
    Writer out;
    out.Put(RecipeMagic.data(), RecipeMagic.size());
    out.Put(static_cast<u32>(body.size()));
    out.Put(u32{0});
    out.Put(Common::Compression::CompressDataZSTD(body.data(), body.size(), 19));
    return std::move(out.bytes);
}

std::vector<u8> Decompress(Bytes recipe) {
    if (recipe.size() < 16 || std::memcmp(recipe.data(), RecipeMagic.data(), 8) != 0) {
        throw std::runtime_error("not a precompile recipe");
    }
    const u32 size = Load<u32>(recipe, 8);
    auto body = Common::Compression::DecompressDataZSTD(recipe.subspan(16));
    if (body.size() != size) {
        throw std::runtime_error("damaged precompile recipe");
    }
    return body;
}

std::vector<CodeMatch> FindCodeInBody(Bytes body, const X::Result& extracted, unsigned threads) {
    std::unordered_map<u64, size_t> windows;
    for (size_t at = 0; at + CodeWindow <= body.size(); ++at) {
        if (Dense(body.subspan(at, CodeWindow))) {
            windows.try_emplace(WindowHash(body.data() + at), at);
        }
    }
    std::vector<std::vector<CodeMatch>> found(extracted.programs.size());
    ParallelFor(extracted.programs.size(), threads, [&](size_t i) {
        const auto& p = extracted.programs[i];
        for (const Bytes part : {Bytes{p.control}, Bytes{p.code}}) {
            for (size_t at = 0; at + CodeWindow <= part.size(); at += 4) {
                const auto it = windows.find(WindowHash(part.data() + at));
                if (it != windows.end() &&
                    std::memcmp(body.data() + it->second, part.data() + at, CodeWindow) == 0 &&
                    Dense(part.subspan(at, CodeWindow))) {
                    found[i].push_back({p.fingerprint, at, it->second});
                }
            }
        }
    });
    std::vector<CodeMatch> matches;
    for (auto& f : found) {
        matches.insert(matches.end(), f.begin(), f.end());
    }
    return matches;
}

} // Anonymous namespace

std::vector<CodeMatch> FindGameCode(std::span<const u8> recipe, const X::Result& programs,
                                    unsigned threads) {
    return FindCodeInBody(Decompress(recipe), programs, threads);
}

template <typename T>
static void PutRaw(Writer& w, const T* data, size_t n) {
    w.Put(Bytes{reinterpret_cast<const u8*>(data), n});
}

u64 NativeShaderHash(std::span<const u64> code) {
    return Common::CityHash64(reinterpret_cast<const char*>(code.data()),
                              code.size() * sizeof(u64)) ^
           NativeHashTag;
}

Shader::Stage StageOf(X::Stage stage) noexcept {
    switch (stage) {
    case X::Stage::Vertex:
        return Stage::VertexB;
    case X::Stage::TessControl:
        return Stage::TessellationControl;
    case X::Stage::TessEval:
        return Stage::TessellationEval;
    case X::Stage::Geometry:
        return Stage::Geometry;
    case X::Stage::Fragment:
        return Stage::Fragment;
    case X::Stage::Compute:
        return Stage::Compute;
    case X::Stage::Task:
        return Stage::Task;
    case X::Stage::Mesh:
        return Stage::Mesh;
    }
    return Stage::VertexB;
}

std::optional<Lowered> Lower(Stage stage, std::span<const u8> control, std::span<const u8> code_area,
                             std::string& error) {
    const bool compute = stage == Stage::Compute;
    Shader::ProgramHeader sph{};
    size_t start = 0;
    size_t available = 0;
    if (compute) {
        if (control.size() < 0x3c) {
            error = "short control section";
            return std::nullopt;
        }
        available = std::min<size_t>(Load<u32>(control, 56), code_area.size());
    } else {
        if (code_area.size() <= NativeHeaderSize) {
            error = "short code area";
            return std::nullopt;
        }
        std::memcpy(&sph, code_area.data(), SphSize);
        start = NativeHeaderSize;
        available = code_area.size() - NativeHeaderSize;
    }
    std::vector<Instruction> native;
    Shader::Ampere::ProgramCaptureState capture;
    bool complete = false;
    const size_t count = available / sizeof(Instruction);
    for (u32 index = 0; index < Shader::Ampere::ProgramCaptureState::MaximumInstructions; ++index) {
        if (index >= count) {
            break; // the runtime reads on past the code area; the game file holds only the area
        }
        Instruction instruction;
        std::memcpy(&instruction, code_area.data() + start + index * sizeof(Instruction),
                    sizeof(Instruction));
        native.push_back(instruction);
        const auto status = capture.Observe(instruction, index);
        if (status == Shader::Ampere::ProgramCaptureState::Result::InvalidTarget) {
            error = "branch outside capture bounds";
            return std::nullopt;
        }
        if (status == Shader::Ampere::ProgramCaptureState::Result::Complete) {
            complete = true;
            break;
        }
    }
    if (!complete) {
        error = "capture incomplete inside the code area";
        return std::nullopt;
    }
    std::vector<u32> traps;
    for (u32 i = 0; i < native.size(); ++i) {
        if (native[i].Bits(0, 12) == 0x95c) {
            traps.push_back(i * sizeof(Instruction));
        }
    }
    if (!traps.empty()) {
        error = "trap guards"; // checked against live constants; never serialized
        return std::nullopt;
    }
    const auto lowered = Shader::Ampere::LowerToMaxwell(
        native, {.geometry = false,
                 .tensor = true,
                 .native_code_address = 0x100000000ULL + start,
                 .verified_traps = traps,
                 .tessellation_stage = Shader::Ampere::TessellationStage::None});
    if (!lowered) {
        error = fmt::format("lowering failed at {:#x}: {}", lowered.error_offset, lowered.error);
        return std::nullopt;
    }
    Lowered result;
    result.native_key = Shader::Ampere::NativeProgramKey(native);
    result.texture_bound = lowered.required_texture_buffer;
    auto& code = result.code;
    code.resize(SphSize / sizeof(u64));
    std::memcpy(code.data(), &sph, SphSize);
    for (const u64 instruction : lowered.instructions) {
        if ((code.size() * sizeof(u64)) % 32 == 0) {
            code.push_back(0);
        }
        code.push_back(instruction);
    }
    code.push_back(0);
    Shader::Ampere::AppendNativeAddressMap(code, lowered.source_instruction_offsets);
    result.hash = NativeShaderHash(code);
    return result;
}

std::optional<Header> ReadHeader(std::span<const u8> recipe) {
    try {
        const auto body = Decompress(recipe);
        Reader r{body};
        if (r.Get<u32>() != Format) {
            return std::nullopt;
        }
        Header header;
        header.title_id = r.Get<u64>();
        header.game_version = r.Get<u32>();
        header.cache_version = r.Get<u32>();
        return header;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::vector<u8> MakeRecipe(std::span<const u8> precompiled, const X::Result& extracted,
                           const Header& header, Report& report, unsigned threads) {
    if (precompiled.size() < 24 ||
        std::memcmp(precompiled.data(), PrecompiledMagic.data(), 8) != 0) {
        throw std::runtime_error("not a precompiled pipeline file");
    }
    const u32 env_count = Load<u32>(precompiled, 12);
    const u32 record_count = Load<u32>(precompiled, 16);

    // Every extracted program lowered once: the shader hash finds an environment's program.
    std::vector<Program> programs(extracted.programs.size());
    ParallelFor(programs.size(), threads, [&](size_t i) {
        const auto& p = extracted.programs[i];
        programs[i].source = &p;
        programs[i].lowered = Lower(StageOf(p.stage), p.control, p.code, programs[i].error);
    });
    std::unordered_map<u64, u32> by_hash;
    for (u32 i = 0; i < programs.size(); ++i) {
        if (programs[i].lowered) {
            // Programs lowering to the same code: the lowest fingerprint, so the recipe does not
            // depend on the order the extraction found them in.
            const auto [it, added] = by_hash.try_emplace(programs[i].lowered->hash, i);
            if (!added && programs[i].source->fingerprint < programs[it->second].source->fingerprint) {
                it->second = i;
            }
        }
    }

    // Environments: their program and their facts.
    struct Env {
        s64 program = -1;
        u64 hash{};
        u8 flags{};
        std::vector<u8> facts;
        Stage stage{};
    };
    std::vector<Env> envs(env_count);
    size_t offset = 24;
    std::vector<std::pair<u64, u64>> index(env_count);
    for (auto& [at, size] : index) {
        at = Load<u64>(precompiled, offset);
        size = Load<u64>(precompiled, offset + 8);
        offset += 16;
    }
    for (u32 e = 0; e < env_count; ++e) {
        auto& env = envs[e];
        const auto blob = precompiled.subspan(index[e].first, index[e].second);
        const auto parts = Parse(blob, true);
        env.stage = parts.stage;
        std::vector<u64> code((parts.code_end - EnvHeaderSize) / 8);
        std::memcpy(code.data(), blob.data() + EnvHeaderSize, code.size() * 8);
        env.hash = NativeShaderHash(code);
        const auto found = by_hash.find(env.hash);
        if (found == by_hash.end()) {
            report.errors.push_back(fmt::format("environment {}: no program of the game has its "
                                                "code ({:016x})", e, env.hash));
            continue;
        }
        if (programs[found->second].lowered->code != code) {
            report.errors.push_back(fmt::format("environment {}: same hash, other code", e));
            continue;
        }
        // The facts are the environment without its code and program header. Anything else that
        // could be the game's (a program header other than the code's, a mesh geometry header not
        // derivable from the code, a layout this does not know) withholds the environment and the
        // records using it.
        const auto withhold = [&](std::string_view why) {
            ++report.environments_withheld;
            report.errors.push_back(fmt::format("environment {} ({:016x}) withheld: {}", e,
                                                env.hash, why));
        };
        const auto& source = *programs[found->second].source;
        Writer facts;
        facts.Put(blob.subspan(0, EnvHeaderSize));
        facts.Put(blob.subspan(parts.code_end, parts.tables_end - parts.code_end));
        auto tail = blob.subspan(parts.tables_end);
        u8 flags = 0;
        if (parts.stage == Stage::Compute) {
            if (tail.size() != ComputeTailSize) {
                withhold("unknown compute layout");
                continue;
            }
        } else {
            const bool task_mesh = parts.stage == Stage::Task || parts.stage == Stage::Mesh;
            const size_t extra = parts.stage == Stage::Geometry ? PassthroughMaskSize
                                 : task_mesh                    ? MeshRegistersSize + SphSize
                                                                : 0;
            if (tail.size() != SphSize + extra) {
                withhold("unknown graphics layout");
                continue;
            }
            if (std::memcmp(tail.data(), code.data(), SphSize) != 0) {
                withhold("its program header is not the code's");
                continue;
            }
            tail = tail.subspan(SphSize);
            if (task_mesh) {
                const auto recorded = tail.last(SphSize);
                const auto derived =
                    parts.stage == Stage::Mesh ? MeshHeader(source) : std::nullopt;
                if (derived && std::memcmp(derived->data(), recorded.data(), SphSize) == 0) {
                    tail = tail.first(tail.size() - SphSize);
                    flags |= MeshHeaderFromCode;
                } else if (!AllZero(recorded)) {
                    withhold("its mesh geometry header is not derivable from the code");
                    continue;
                }
            }
        }
        facts.Put(tail);
        if (HoldsCode(facts.bytes, source.code) || HoldsCode(facts.bytes, source.control) ||
            HoldsCode(facts.bytes, Bytes{reinterpret_cast<const u8*>(code.data()),
                                         code.size() * sizeof(u64)})) {
            withhold("its facts hold some of the program's code");
            continue;
        }
        env.program = found->second;
        env.flags = flags;
        env.facts = std::move(facts.bytes);
    }

    // Records.
    struct Record {
        u8 kind{};
        std::vector<u32> envs;
        u32 state{};
        std::array<u8, 6> slots{};
        std::array<u64, 6> xors{};
    };
    std::vector<Record> records;
    std::map<std::vector<u8>, u32> states;
    for (u32 i = 0; i < record_count; ++i) {
        Record record;
        record.kind = Load<u8>(precompiled, offset);
        const u8 count = Load<u8>(precompiled, offset + 1);
        offset += 4;
        bool ok = true;
        for (u8 k = 0; k < count; ++k) {
            const u32 id = Load<u32>(precompiled, offset);
            offset += 4;
            ok &= id < env_count && envs[id].program >= 0;
            record.envs.push_back(id);
        }
        if (record.kind == 1) {
            offset += ComputeKeySize;
        } else {
            const auto key = precompiled.subspan(offset, GraphicsKeySize);
            offset += GraphicsKeySize;
            record.slots.fill(0xff);
            size_t next = 0;
            for (size_t slot = 0; slot < 6 && ok; ++slot) {
                const u64 hash = Load<u64>(key, slot * 8);
                if (hash == 0) {
                    continue;
                }
                if (next >= record.envs.size()) {
                    ok = false;
                    break;
                }
                record.slots[slot] = static_cast<u8>(next);
                record.xors[slot] = hash ^ envs[record.envs[next]].hash;
                ++next;
            }
            const std::vector<u8> state(key.begin() + KeyHashesSize, key.end());
            record.state = states.try_emplace(state, static_cast<u32>(states.size())).first->second;
        }
        if (!ok) {
            ++report.records_dropped;
            continue;
        }
        records.push_back(std::move(record));
    }

    // Programs and fact sets that records use, renumbered.
    std::vector<s64> fact_of(env_count, -1);
    std::vector<s64> program_slot(programs.size(), -1);
    std::vector<u32> program_list, fact_list;
    for (auto& record : records) {
        for (u32& id : record.envs) {
            if (fact_of[id] < 0) {
                const u32 program = static_cast<u32>(envs[id].program);
                if (program_slot[program] < 0) {
                    program_slot[program] = static_cast<s64>(program_list.size());
                    program_list.push_back(program);
                }
                fact_of[id] = static_cast<s64>(fact_list.size());
                fact_list.push_back(id);
            }
            id = static_cast<u32>(fact_of[id]);
        }
    }

    Writer body;
    body.Put(Format);
    body.Put(header.title_id);
    body.Put(header.game_version);
    body.Put(header.cache_version);
    body.Put(static_cast<u32>(program_list.size()));
    for (const u32 p : program_list) {
        body.Put(programs[p].source->fingerprint);
        body.Put(static_cast<u8>(StageOf(programs[p].source->stage)));
        body.Zeros(3);
    }
    body.Put(static_cast<u32>(fact_list.size()));
    for (const u32 id : fact_list) {
        const auto& env = envs[id];
        body.Put(static_cast<u32>(program_slot[env.program]));
        body.Put(env.flags);
        body.Zeros(3);
        body.Put(static_cast<u32>(env.facts.size()));
        body.Put(env.facts);
    }
    std::vector<const std::vector<u8>*> state_list(states.size());
    for (const auto& [state, id] : states) {
        state_list[id] = &state;
    }
    body.Put(static_cast<u32>(state_list.size()));
    for (const auto* state : state_list) {
        body.Put(*state);
    }
    body.Put(static_cast<u32>(records.size()));
    for (const auto& record : records) {
        body.Put(record.kind);
        body.Put(static_cast<u8>(record.envs.size()));
        body.Zeros(2);
        for (const u32 id : record.envs) {
            body.Put(id);
        }
        if (record.kind == 0) {
            body.Put(record.state);
            body.Put(record.slots);
            const bool has_xor =
                std::any_of(record.xors.begin(), record.xors.end(), [](u64 x) { return x != 0; });
            body.Put(static_cast<u8>(has_xor));
            body.Zeros(1);
            if (has_xor) {
                body.Put(record.xors);
            }
        }
    }
    report.programs = static_cast<u32>(program_list.size());
    report.environments = static_cast<u32>(fact_list.size());
    report.records = static_cast<u32>(records.size());
    report.states = static_cast<u32>(state_list.size());

    // The last gate: no 32 bytes of any of the game's programs anywhere in the recipe.
    const auto matches = FindCodeInBody(body.bytes, extracted, threads);
    if (!matches.empty()) {
        throw std::runtime_error(fmt::format(
            "the recipe would hold {} pieces of the game's code (program {:016x} +{:#x}); not "
            "written",
            matches.size(), matches[0].fingerprint, matches[0].program_offset));
    }
    return Compress(body.bytes);
}

std::vector<u8> Assemble(std::span<const u8> recipe, const X::Result& extracted, u32 cache_version,
                         Report& report, unsigned threads) {
    const auto body = Decompress(recipe);
    Reader r{body};
    if (r.Get<u32>() != Format) {
        throw std::runtime_error("precompile recipe of another format");
    }
    r.Get<u64>();
    r.Get<u32>();
    r.Get<u32>();

    std::unordered_map<u64, const X::Program*> by_fingerprint;
    for (const auto& p : extracted.programs) {
        by_fingerprint.try_emplace(p.fingerprint, &p);
    }
    const u32 program_count = r.Get<u32>();
    std::vector<Program> programs(program_count);
    std::vector<Stage> stages(program_count);
    for (u32 i = 0; i < program_count; ++i) {
        const u64 fingerprint = r.Get<u64>();
        stages[i] = static_cast<Stage>(r.Get<u8>());
        r.Take(3);
        const auto it = by_fingerprint.find(fingerprint);
        if (it == by_fingerprint.end()) {
            ++report.programs_missing;
            programs[i].error = fmt::format("{:016x} is not in the game", fingerprint);
        } else {
            programs[i].source = it->second;
        }
    }
    ParallelFor(program_count, threads, [&](size_t i) {
        if (programs[i].source) {
            programs[i].lowered =
                Lower(stages[i], programs[i].source->control, programs[i].source->code,
                      programs[i].error);
        }
    });
    for (const auto& p : programs) {
        report.programs_failed += p.source && !p.lowered ? 1 : 0;
    }

    // Environments: the facts with the program's code and header put back.
    const u32 fact_count = r.Get<u32>();
    struct Env {
        s64 program = -1;
        std::vector<u8> bytes;
        u64 hash{};
        u32 shared{};
        std::array<u32, 3> workgroup{};
    };
    std::vector<Env> envs(fact_count);
    for (u32 i = 0; i < fact_count; ++i) {
        const u32 program = r.Get<u32>();
        const u8 flags = r.Get<u8>();
        r.Take(3);
        const auto facts = r.Take(r.Get<u32>());
        if (program >= program_count || !programs[program].lowered ||
            (flags & ~MeshHeaderFromCode) != 0) {
            continue;
        }
        const auto& lowered = *programs[program].lowered;
        const auto parts = Parse(facts, false);
        Writer env;
        env.Put(static_cast<u64>(lowered.code.size() * sizeof(u64)));
        env.Put(facts.subspan(8, EnvHeaderSize - 8));
        PutRaw(env, lowered.code.data(), lowered.code.size() * sizeof(u64));
        env.Put(facts.subspan(EnvHeaderSize, parts.tables_end - EnvHeaderSize));
        const auto tail = facts.subspan(parts.tables_end);
        if (parts.stage == Stage::Compute) {
            envs[i].workgroup = Load<std::array<u32, 3>>(tail, 0);
            envs[i].shared = Load<u32>(tail, 12);
        } else {
            PutRaw(env, lowered.code.data(), SphSize);
        }
        env.Put(tail);
        if (flags & MeshHeaderFromCode) {
            const auto header = MeshHeader(*programs[program].source);
            if (!header) {
                continue;
            }
            env.Put(Bytes{header->data(), header->size()});
        }
        envs[i].program = program;
        envs[i].hash = lowered.hash;
        envs[i].bytes = std::move(env.bytes);
    }

    const u32 state_count = r.Get<u32>();
    std::vector<Bytes> states(state_count);
    for (auto& state : states) {
        state = r.Take(StateSize);
    }

    // Records, and the environments they use, renumbered.
    const u32 record_count = r.Get<u32>();
    Writer records;
    std::vector<s64> env_slot(fact_count, -1);
    std::vector<u32> env_list;
    u32 written = 0;
    for (u32 i = 0; i < record_count; ++i) {
        const u8 kind = r.Get<u8>();
        const u8 count = r.Get<u8>();
        r.Take(2);
        std::vector<u32> ids(count);
        bool ok = true;
        for (auto& id : ids) {
            id = r.Get<u32>();
            ok &= id < fact_count && envs[id].program >= 0;
        }
        u32 state = 0;
        std::array<u8, 6> slots{};
        std::array<u64, 6> xors{};
        if (kind == 0) {
            state = r.Get<u32>();
            slots = r.Get<std::array<u8, 6>>();
            const bool has_xor = r.Get<u8>() != 0;
            r.Take(1);
            if (has_xor) {
                xors = r.Get<std::array<u64, 6>>();
            }
            ok &= state < state_count;
        }
        if (!ok) {
            ++report.records_dropped;
            continue;
        }
        records.Put(kind);
        records.Put(count);
        records.Zeros(2);
        for (const u32 id : ids) {
            if (env_slot[id] < 0) {
                env_slot[id] = static_cast<s64>(env_list.size());
                env_list.push_back(id);
            }
            records.Put(static_cast<u32>(env_slot[id]));
        }
        if (kind == 1) {
            const auto& env = envs[ids[0]];
            records.Put(env.hash);
            records.Put(env.shared);
            records.Put(env.workgroup);
        } else {
            for (size_t slot = 0; slot < 6; ++slot) {
                records.Put(slots[slot] < ids.size() ? envs[ids[slots[slot]]].hash ^ xors[slot]
                                                     : u64{0});
            }
            records.Put(states[state]);
        }
        ++written;
    }

    // NX2PREP1: header, environment index, records, environments.
    Writer out;
    out.Put(PrecompiledMagic.data(), PrecompiledMagic.size());
    out.Put(cache_version);
    out.Put(static_cast<u32>(env_list.size()));
    out.Put(written);
    out.Put(u32{0});
    u64 at = 24 + 16 * u64{env_list.size()} + records.bytes.size();
    for (const u32 id : env_list) {
        out.Put(at);
        out.Put(static_cast<u64>(envs[id].bytes.size()));
        at += envs[id].bytes.size();
    }
    out.Put(records.bytes);
    for (const u32 id : env_list) {
        out.Put(envs[id].bytes);
    }
    report.programs = program_count;
    report.environments = static_cast<u32>(env_list.size());
    report.records = written;
    report.states = state_count;
    for (const auto& p : programs) {
        if (!p.error.empty() && report.errors.size() < 64) {
            report.errors.push_back(p.error);
        }
    }
    return std::move(out.bytes);
}

} // namespace VideoCommon::Nx2Recipe
