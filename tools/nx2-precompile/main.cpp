// SPDX-License-Identifier: GPL-3.0-or-later

// nx2-precompile: builds Lichii disk-cache shader environments for every native program a game
// ships, so the pipelines that use them can be compiled before they are first drawn.
//
//   nx2-precompile scan --pkg DIR --out PREFIX [--threads N] [--priors FILE]
//       Captures, lowers and hashes every program in nxnative's BTCD packages exactly as
//       GenericEnvironment::LowerNativeCode does at runtime, then translates each one against a
//       static fact oracle and writes the queried facts in GenericEnvironment::Serialize layout.
//       Writes PREFIX.tsv (one line per program) and PREFIX.envs (the serialized environments).
//
//   nx2-precompile validate --pkg DIR --cache FILE --out PREFIX [--threads N] [--priors FILE]
//       For every vertex, fragment and compute environment recorded in a v53 disk cache, finds
//       the package program with the same lowered code, translates both the recorded environment
//       and the oracle environment, and compares the SPIR-V. Writes PREFIX.tsv and a summary.

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "common/cityhash.h"
#include "common/fs/path_util.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ampere/program_address_map.h"
#include "shader_recompiler/frontend/ampere/program_capture.h"
#include "shader_recompiler/frontend/ampere/program_override.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/translate_profile.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"
#include "core/file_sys/nup_container.h"
#include "core/file_sys/nx2_shader_extract.h"
#include "core/file_sys/vfs/vfs_real.h"
#include "video_core/nx2_recipe.h"
#include "video_core/pipeline_cache_files.h"

namespace {
using Json = nlohmann::json;
using Shader::Stage;
using Shader::Ampere::Instruction;

constexpr size_t SphSize = sizeof(Shader::ProgramHeader);
constexpr size_t GraphicsKeySize = 1056;
constexpr size_t ComputeKeySize = 24;

u64 CbufKey(u32 bank, u32 offset) {
    return (u64{bank} << 32) | offset;
}

u64 IndirectCbufKey(u32 bank, u32 offset, u32 indirect) {
    return (u64{1} << 63) | (u64{indirect & 0x3fffff} << 40) | CbufKey(bank, offset);
}

using VideoCommon::Nx2Recipe::NativeShaderHash;

std::vector<u8> ReadFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Cannot open " + path.string());
    const auto size = file.tellg();
    std::vector<u8> bytes(static_cast<size_t>(size));
    file.seekg(0);
    if (size > 0 && !file.read(reinterpret_cast<char*>(bytes.data()), size))
        throw std::runtime_error("Cannot read " + path.string());
    return bytes;
}

template <typename T>
T Load(std::span<const u8> bytes, size_t offset) {
    if (offset + sizeof(T) > bytes.size())
        throw std::runtime_error("Truncated input");
    T value;
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return value;
}

// ---------------------------------------------------------------------------------------------
// nxnative BTCD packages: 0x38-byte record header ("BTCDBTCD", +0x28 control length, +0x2c code
// length), then the code area, then the NVN2 control section.

struct PackageProgram {
    std::string file;
    u32 index{};
    Stage stage{};
    std::span<const u8> code; // the whole code area
    std::span<const u8> control;
};

std::optional<Stage> StageOfPackage(std::string_view name) {
    if (name.starts_with("vs_"))
        return Stage::VertexB;
    if (name.starts_with("ps_"))
        return Stage::Fragment;
    if (name.starts_with("cs_"))
        return Stage::Compute;
    if (name.starts_with("ms_"))
        return Stage::Mesh;
    if (name.starts_with("ts_"))
        return Stage::Task;
    return std::nullopt;
}

struct Packages {
    std::vector<std::vector<u8>> storage;
    std::vector<PackageProgram> programs;

    explicit Packages(const std::filesystem::path& directory) {
        if (std::filesystem::is_regular_file(directory)) {
            LoadExtracted(directory);
            return;
        }
        std::vector<std::filesystem::path> files;
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            const auto name = entry.path().filename().string();
            if (entry.is_regular_file() && name.ends_with(".bin") && !name.starts_with("._") &&
                StageOfPackage(name)) {
                files.push_back(entry.path());
            }
        }
        std::ranges::sort(files);
        for (const auto& path : files) {
            const auto name = path.filename().string();
            storage.push_back(ReadFile(path));
            const std::span<const u8> bytes{storage.back()};
            size_t offset = 0;
            u32 index = 0;
            while (offset + 0x38 <= bytes.size()) {
                if (std::memcmp(bytes.data() + offset, "BTCDBTCD", 8) != 0)
                    throw std::runtime_error(fmt::format("{}: bad record at {:#x}", name, offset));
                const u32 control_size = Load<u32>(bytes, offset + 0x28);
                const u32 code_size = Load<u32>(bytes, offset + 0x2c);
                if (offset + 0x38 + code_size + control_size > bytes.size())
                    throw std::runtime_error(fmt::format("{}: truncated record {}", name, index));
                programs.push_back({.file = name,
                                    .index = index,
                                    .stage = *StageOfPackage(name),
                                    .code = bytes.subspan(offset + 0x38, code_size),
                                    .control = bytes.subspan(offset + 0x38 + code_size, control_size)});
                offset += 0x38 + code_size + control_size;
                ++index;
            }
        }
    }

    /// `nx2-precompile extract`'s PREFIX.programs: "NXSX0001", u32 count, then per program u8
    /// stage (FileSys::Nx2ShaderExtract::Stage), u64 fingerprint, u32 control length, u32 code
    /// length, u32 origin length, the origin, the control section, the code area. Programs are
    /// numbered in file order ("programs", index).
    void LoadExtracted(const std::filesystem::path& path) {
        storage.push_back(ReadFile(path));
        const std::span<const u8> bytes{storage.back()};
        if (bytes.size() < 12 || std::memcmp(bytes.data(), "NXSX0001", 8) != 0)
            throw std::runtime_error(path.string() + " is not an nx2-precompile extract file");
        const u32 count = Load<u32>(bytes, 8);
        size_t offset = 12;
        static constexpr std::array stages{Stage::VertexB,  Stage::TessellationControl,
                                           Stage::TessellationEval, Stage::Geometry,
                                           Stage::Fragment, Stage::Compute,
                                           Stage::Task,     Stage::Mesh};
        for (u32 index = 0; index < count; ++index) {
            const u8 stage = Load<u8>(bytes, offset);
            const u32 control_size = Load<u32>(bytes, offset + 9);
            const u32 code_size = Load<u32>(bytes, offset + 13);
            const u32 origin_size = Load<u32>(bytes, offset + 17);
            const size_t control_at = offset + 21 + origin_size;
            if (stage >= stages.size() || control_at + control_size + code_size > bytes.size())
                throw std::runtime_error(fmt::format("{}: bad program {}", path.string(), index));
            programs.push_back({.file = "programs",
                                .index = index,
                                .stage = stages[stage],
                                .code = bytes.subspan(control_at + control_size, code_size),
                                .control = bytes.subspan(control_at, control_size)});
            offset = control_at + control_size + code_size;
        }
    }
};

// ---------------------------------------------------------------------------------------------
// Capture + lowering, as GenericEnvironment::LowerNativeCode does it.

using VideoCommon::Nx2Recipe::Lowered;

std::optional<Lowered> CaptureAndLower(const PackageProgram& program, std::string& error) {
    return VideoCommon::Nx2Recipe::Lower(program.stage, program.control, program.code, error);
}

// ---------------------------------------------------------------------------------------------
// Static fact oracle. Answers every query the translator makes from the program and its control
// section, records the answers like GenericEnvironment does, and serializes them in its layout.
// A wrong answer only produces SPIR-V the runtime will not ask for: the runtime translates the
// guest program with its real facts and never reads these records for drawing.

struct Priors {
    // (stage, bank, offset) -> storage pointer answer; indirect adds the table offset.
    std::map<std::tuple<u32, u32, u32>, u8> direct;
    std::map<std::tuple<u32, u32, u32, u32>, u8> indirect;
    // (stage, bank, offset) -> texture type of size queries without a sampling instruction.
    std::map<std::tuple<u32, u32, u32>, Shader::TextureType> txq_types;

    static Priors FromFile(const std::filesystem::path& path) {
        Priors priors;
        if (path.empty())
            return priors;
        const auto bytes = ReadFile(path);
        const Json json = Json::parse(bytes.begin(), bytes.end());
        const auto stage_of = [](const std::string& name) -> u32 {
            constexpr std::array names{"VertexB", "TessellationControl", "TessellationEval",
                                       "Geometry", "Fragment", "Compute", "VertexA", "Task", "Mesh"};
            for (u32 i = 0; i < names.size(); ++i)
                if (name == names[i])
                    return i;
            throw std::runtime_error("Unknown stage in priors: " + name);
        };
        for (const auto& entry : json.value("storage_pointer_direct", Json::array())) {
            priors.direct[{stage_of(entry["stage"]), entry["bank"], entry["offset"]}] =
                static_cast<u8>(entry["answer"].get<int>());
        }
        for (const auto& entry : json.value("storage_pointer_indirect", Json::array())) {
            priors.indirect[{stage_of(entry["stage"]), entry["bank"], entry["offset"],
                             entry["indirect"]}] = static_cast<u8>(entry["answer"].get<int>());
        }
        constexpr std::array type_names{"Color1D", "ColorArray1D", "Color2D", "ColorArray2D", "Color3D",
                                        "ColorCube", "ColorArrayCube", "Buffer", "Color2DRect"};
        for (const auto& entry : json.value("txq_type", Json::array())) {
            const std::string answer = entry["answer"];
            for (u32 i = 0; i < type_names.size(); ++i) {
                if (answer == type_names[i]) {
                    priors.txq_types[{stage_of(entry["stage"]), entry["bank"], entry["offset"]}] =
                        static_cast<Shader::TextureType>(i);
                }
            }
        }
        return priors;
    }
};

struct RecordedEnvironment;

// Diagnostic substitution of recorded answers (validate only).
enum RefMask : unsigned { MaskCbuf = 1, MaskTypes = 2, MaskFormats = 4, MaskPointers = 8, MaskBank1 = 16, MaskScalars = 32 };

class OracleEnvironment final : public Shader::Environment {
public:
    const RecordedEnvironment* ref{};
    unsigned ref_mask{};
    void ApplyReferenceScalars();
    /// Task and mesh programs: SET_MESH_SHADER_A/B, SET_MESH_INIT_SHADER and the header of the
    /// pass-through geometry program bound with the mesh program.
    void SetMeshState(const std::array<u32, 3>& registers, std::span<const u8> header) {
        mesh_registers = registers;
        std::memcpy(&mesh_primitive_header, header.data(), SphSize);
    }
    OracleEnvironment(const PackageProgram& program_, const Lowered& lowered, const Priors& priors_)
        : program{program_}, priors{priors_}, code{lowered.code} {
        stage = program.stage;
        is_proprietary_driver = true;
        high = static_cast<u32>((code.size() - 1) * sizeof(u64));
        if (stage == Stage::Compute) {
            start_address = SphSize;
            if (program.control.size() >= 0x68) {
                workgroup = {Load<u32>(program.control, 0x58), Load<u32>(program.control, 0x5c),
                             Load<u32>(program.control, 0x60)};
                const u32 shared = Load<u32>(program.control, 0x64);
                shared_memory = (shared + 255) / 256 * 256;
            }
            // The QMD's local memory (position + call-return stack) the program was built with;
            // matches every recorded compute environment.
            local_memory = program.control.size() >= 0x6c ? Load<u32>(program.control, 0x68) : 0;
        } else {
            start_address = 0;
            std::memcpy(&sph, program.code.data(), SphSize);
            const u64 size = sph.LocalMemorySize() + sph.common3.shader_local_memory_crs_size;
            local_memory = static_cast<u32>(size);
        }
        texture_bound = lowered.texture_bound != ~0U ? lowered.texture_bound : 2;
        if (program.control.size() >= 0x48) {
            const u32 begin = Load<u32>(program.control, 0x40);
            const u32 end = Load<u32>(program.control, 0x44);
            if (begin <= end && end <= program.code.size())
                c1 = program.code.subspan(begin, end - begin);
        }
    }

    u64 ReadInstruction(u32 address) override {
        if (address > high || address % 8)
            throw std::runtime_error(fmt::format("Instruction outside capture: {:#x}", address));
        return code[address / 8];
    }
    u32 TranslateProgramAddress(u32 address) const override {
        const auto result = Shader::Ampere::NativeProgramAddress(code, address);
        if (!result)
            throw std::runtime_error("Invalid native indirect branch target");
        return *result;
    }
    bool IsNativeProgram() const override {
        return true;
    }
    bool HasNativeInstructions() const override {
        return true;
    }
    bool IsCbufStorageBufferPointer(u32 bank, u32 offset) override {
        const u64 key = CbufKey(bank, offset);
        if (const auto it = pointers.find(key); it != pointers.end())
            return it->second != 0;
        // Like GenericEnvironment, the classification reads the pointer's two words.
        (void)ReadCbufValue(bank, offset);
        (void)ReadCbufValue(bank, offset + 4);
        u8 answer = 1;
        if (const auto it = priors.direct.find({static_cast<u32>(stage), bank, offset});
            it != priors.direct.end()) {
            answer = it->second;
        }
        if (const auto value = RefPointer(key))
            answer = *value;
        pointers.emplace(key, answer);
        return answer != 0;
    }
    bool IsCbufIndirectStorageBufferPointer(u32 bank, u32 offset, u32 indirect) override {
        const u64 key = IndirectCbufKey(bank, offset, indirect);
        if (const auto it = pointers.find(key); it != pointers.end())
            return it->second != 0;
        u8 answer = 0;
        if (IsCbufStorageBufferPointer(bank, offset)) {
            (void)ReadCbufValue(bank, offset);
            (void)ReadCbufValue(bank, offset + 4);
            answer = 1;
            if (const auto it = priors.indirect.find({static_cast<u32>(stage), bank, offset, indirect});
                it != priors.indirect.end()) {
                answer = it->second;
            }
        }
        if (const auto value = RefPointer(key))
            answer = *value;
        pointers.emplace(key, answer);
        return answer != 0;
    }
    u32 ReadCbufValue(u32 bank, u32 offset) override {
        const u64 key = CbufKey(bank, offset);
        if (const auto it = cbuf_values.find(key); it != cbuf_values.end())
            return it->second;
        u32 value = 0;
        if (const auto recorded = RefCbuf(bank, key)) {
            value = *recorded;
        } else if (bank == 1) {
            // Bank 1 is the program's own constant segment.
            if (offset % 4 == 0 && offset + 4 <= c1.size())
                std::memcpy(&value, c1.data() + offset, sizeof(value));
        } else {
            // A distinct token per word: texture handles and pointers whose value the
            // translation does not depend on.
            // Texture handles keep the image index in their low 20 bits.
            value = next_token++ & 0xfffff;
            token_slot.emplace(value, key);
        }
        cbuf_values.emplace(key, value);
        return value;
    }
    Shader::TextureType ReadTextureType(u32 handle) override {
        if (const auto it = texture_types.find(handle); it != texture_types.end())
            return it->second;
        Shader::TextureType type = Shader::TextureType::Color2D;
        if (const auto recorded = RefType(handle)) {
            type = *recorded;
        } else if (const auto slot = token_slot.find(handle & 0xfffff); slot != token_slot.end()) {
            const u32 bank = static_cast<u32>(slot->second >> 32);
            const u32 offset = static_cast<u32>(slot->second);
            if (const auto it = slot_types.find(slot->second); it != slot_types.end()) {
                // Another texture instruction of the program on the same slot gives its type.
                type = it->second;
            }
            // Learned size-query types were checked and cost more than they gained; unused.
            (void)bank;
            (void)offset;
        }
        texture_types.emplace(handle, type);
        return type;
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override {
        return texture_formats
            .try_emplace(handle, RefFormat(handle).value_or(Shader::TexturePixelFormat::A8B8G8R8_UNORM))
            .first->second;
    }
    bool IsTexturePixelFormatInteger(u32 handle) override;
    u32 ReadViewportTransformState() override {
        return viewport;
    }
    u32 TextureBoundBuffer() const override {
        return texture_bound;
    }
    std::optional<u32> RefCbuf(u32 bank, u64 key) const;
    std::optional<Shader::TextureType> RefType(u32 handle) const;
    std::optional<Shader::TexturePixelFormat> RefFormat(u32 handle) const;
    std::optional<u8> RefPointer(u64 key) const;
    u32 LocalMemorySize() const override {
        return local_memory;
    }
    u32 SharedMemorySize() const override {
        return shared_memory;
    }
    std::array<u32, 3> WorkgroupSize() const override {
        return workgroup;
    }
    bool HasHLEMacroState() const override {
        return false;
    }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}

    // GenericEnvironment::Serialize layout (cache version 53).
    std::vector<u8> Serialize() const {
        std::vector<u8> out;
        const auto put = [&](const auto& value) {
            const auto* bytes = reinterpret_cast<const u8*>(&value);
            out.insert(out.end(), bytes, bytes + sizeof(value));
        };
        const u64 code_size = code.size() * sizeof(u64);
        put(code_size);
        put(static_cast<u64>(texture_types.size()));
        put(static_cast<u64>(texture_formats.size()));
        put(static_cast<u64>(cbuf_values.size()));
        put(static_cast<u64>(pointers.size()));
        put(u64{0}); // cbuf replacements
        put(local_memory);
        put(texture_bound);
        put(start_address);
        put(u32{0});  // cached_lowest
        put(high);    // cached_highest
        put(viewport);
        put(static_cast<u32>(stage));
        const auto* code_bytes = reinterpret_cast<const u8*>(code.data());
        out.insert(out.end(), code_bytes, code_bytes + code_size);
        for (const auto& [key, type] : texture_types) {
            put(key);
            put(type);
        }
        for (const auto& [key, format] : texture_formats) {
            put(key);
            put(format);
        }
        for (const auto& [key, value] : cbuf_values) {
            put(key);
            put(value);
        }
        for (const auto& [key, value] : pointers) {
            put(key);
            put(value);
        }
        if (stage == Stage::Compute) {
            put(workgroup);
            put(shared_memory);
        } else {
            put(sph);
            if (stage == Stage::Task || stage == Stage::Mesh) {
                put(mesh_registers);
                put(mesh_primitive_header);
            }
        }
        return out;
    }

    size_t QueryCount() const {
        return texture_types.size() + texture_formats.size() + cbuf_values.size() + pointers.size();
    }

    std::map<u32, Shader::TextureType> texture_types;
    std::map<u32, Shader::TexturePixelFormat> texture_formats;
    std::map<u64, u32> cbuf_values;
    std::map<u64, u8> pointers;
    std::map<u64, Shader::TextureType> slot_types;

private:
    std::unordered_map<u32, u64> token_slot;
    const PackageProgram& program;
    const Priors& priors;
    std::vector<u64> code;
    std::span<const u8> c1;
    u32 high{}, local_memory{}, shared_memory{}, viewport{1}, texture_bound{2};
    std::array<u32, 3> workgroup{1, 1, 1};
    u32 next_token{1};
};

// ---------------------------------------------------------------------------------------------
// Recorded environments from a v53 disk cache (FileEnvironment::Deserialize layout).

struct RecordedEnvironment final : public Shader::Environment {
    std::vector<u64> code;
    u32 local_memory{}, texture_bound{}, low{}, high{}, viewport{}, shared_memory{};
    std::array<u32, 3> workgroup{};
    std::map<u32, Shader::TextureType> texture_types;
    std::map<u32, Shader::TexturePixelFormat> texture_formats;
    std::map<u64, u32> cbuf_values;
    std::map<u64, u8> pointers;
    std::map<u64, Shader::ReplaceConstant> replacements;
    std::array<u32, 3> mesh_regs{};
    std::array<u8, SphSize> mesh_header{};

    size_t Parse(std::span<const u8> bytes, size_t offset) {
        const u64 code_size = Load<u64>(bytes, offset);
        const u64 n_types = Load<u64>(bytes, offset + 8);
        const u64 n_formats = Load<u64>(bytes, offset + 16);
        const u64 n_cbuf = Load<u64>(bytes, offset + 24);
        const u64 n_pointers = Load<u64>(bytes, offset + 32);
        const u64 n_replacements = Load<u64>(bytes, offset + 40);
        local_memory = Load<u32>(bytes, offset + 48);
        texture_bound = Load<u32>(bytes, offset + 52);
        start_address = Load<u32>(bytes, offset + 56);
        low = Load<u32>(bytes, offset + 60);
        high = Load<u32>(bytes, offset + 64);
        viewport = Load<u32>(bytes, offset + 68);
        stage = static_cast<Stage>(Load<u32>(bytes, offset + 72));
        offset += 76;
        if (code_size % 8 || offset + code_size > bytes.size())
            throw std::runtime_error("Invalid recorded code size");
        code.resize(code_size / 8);
        std::memcpy(code.data(), bytes.data() + offset, code_size);
        offset += code_size;
        for (u64 i = 0; i < n_types; ++i, offset += 8)
            texture_types[Load<u32>(bytes, offset)] =
                static_cast<Shader::TextureType>(Load<u32>(bytes, offset + 4));
        for (u64 i = 0; i < n_formats; ++i, offset += 8)
            texture_formats[Load<u32>(bytes, offset)] =
                static_cast<Shader::TexturePixelFormat>(Load<u32>(bytes, offset + 4));
        for (u64 i = 0; i < n_cbuf; ++i, offset += 12)
            cbuf_values[Load<u64>(bytes, offset)] = Load<u32>(bytes, offset + 8);
        for (u64 i = 0; i < n_pointers; ++i, offset += 9)
            pointers[Load<u64>(bytes, offset)] = Load<u8>(bytes, offset + 8);
        for (u64 i = 0; i < n_replacements; ++i, offset += 12)
            replacements[Load<u64>(bytes, offset)] =
                static_cast<Shader::ReplaceConstant>(Load<u32>(bytes, offset + 8));
        if (stage == Stage::Compute) {
            workgroup = Load<std::array<u32, 3>>(bytes, offset);
            shared_memory = Load<u32>(bytes, offset + 12);
            offset += 16;
        } else {
            sph = Load<Shader::ProgramHeader>(bytes, offset);
            offset += SphSize;
            if (stage == Stage::Geometry) {
                gp_passthrough_mask = Load<std::array<u32, 8>>(bytes, offset);
                offset += 32;
            }
            if (stage == Stage::Task || stage == Stage::Mesh) {
                mesh_regs = Load<std::array<u32, 3>>(bytes, offset);
                mesh_header = Load<std::array<u8, SphSize>>(bytes, offset + 12);
                mesh_registers = mesh_regs;
                std::memcpy(&mesh_primitive_header, mesh_header.data(), SphSize);
                offset += 12 + SphSize;
            }
        }
        is_proprietary_driver = texture_bound == 2 || Shader::Ampere::HasNativeAddressMap(code);
        return offset;
    }

    template <typename M, typename K>
    static auto Find(const M& map, K key, const char* what) {
        const auto it = map.find(key);
        if (it == map.end())
            throw std::runtime_error(fmt::format("Uncached {} {:#x}", what, static_cast<u64>(key)));
        return it->second;
    }
    u64 ReadInstruction(u32 address) override {
        if (address < low || address > high)
            throw std::runtime_error("Out of bounds address");
        return code[(address - low) / 8];
    }
    u32 TranslateProgramAddress(u32 address) const override {
        if (const auto result = Shader::Ampere::NativeProgramAddress(code, address))
            return *result;
        throw std::runtime_error("Invalid cached native indirect branch address");
    }
    bool IsNativeProgram() const override {
        return Shader::Ampere::HasNativeAddressMap(code);
    }
    bool HasNativeInstructions() const override {
        return IsNativeProgram();
    }
    bool IsCbufStorageBufferPointer(u32 bank, u32 offset) override {
        return Find(pointers, CbufKey(bank, offset), "storage pointer") != 0;
    }
    bool IsCbufIndirectStorageBufferPointer(u32 bank, u32 offset, u32 indirect) override {
        return Find(pointers, IndirectCbufKey(bank, offset, indirect), "storage pointer table") != 0;
    }
    u32 ReadCbufValue(u32 bank, u32 offset) override {
        return Find(cbuf_values, CbufKey(bank, offset), "cbuf value");
    }
    Shader::TextureType ReadTextureType(u32 handle) override {
        return Find(texture_types, handle, "texture type");
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override {
        return Find(texture_formats, handle, "texture pixel format");
    }
    bool IsTexturePixelFormatInteger(u32 handle) override {
        using F = Shader::TexturePixelFormat;
        switch (ReadTexturePixelFormat(handle)) {
        case F::A8B8G8R8_SINT: case F::A8B8G8R8_UINT: case F::A2B10G10R10_UINT: case F::R8_SINT:
        case F::R8_UINT: case F::R16G16B16A16_SINT: case F::R16G16B16A16_UINT:
        case F::R32G32B32A32_UINT: case F::R32G32B32A32_SINT: case F::R32G32_SINT:
        case F::R16_UINT: case F::R16_SINT: case F::R16G16_UINT: case F::R16G16_SINT:
        case F::R8G8_SINT: case F::R8G8_UINT: case F::R32G32_UINT: case F::R32_UINT:
        case F::R32_SINT:
            return true;
        default:
            return false;
        }
    }
    u32 ReadViewportTransformState() override {
        return viewport;
    }
    u32 TextureBoundBuffer() const override {
        return texture_bound;
    }
    u32 LocalMemorySize() const override {
        return local_memory;
    }
    u32 SharedMemorySize() const override {
        return shared_memory;
    }
    std::array<u32, 3> WorkgroupSize() const override {
        return workgroup;
    }
    bool HasHLEMacroState() const override {
        return !replacements.empty();
    }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32 bank, u32 offset) override {
        const auto it = replacements.find(CbufKey(bank, offset));
        return it == replacements.end() ? std::nullopt : std::optional{it->second};
    }
    void Dump(u64, u64) override {}
};

bool IsIntegerFormat(Shader::TexturePixelFormat format) {
    using F = Shader::TexturePixelFormat;
    switch (format) {
    case F::A8B8G8R8_SINT: case F::A8B8G8R8_UINT: case F::A2B10G10R10_UINT: case F::R8_SINT:
    case F::R8_UINT: case F::R16G16B16A16_SINT: case F::R16G16B16A16_UINT:
    case F::R32G32B32A32_UINT: case F::R32G32B32A32_SINT: case F::R32G32_SINT:
    case F::R16_UINT: case F::R16_SINT: case F::R16G16_UINT: case F::R16G16_SINT:
    case F::R8G8_SINT: case F::R8G8_UINT: case F::R32G32_UINT: case F::R32_UINT:
    case F::R32_SINT:
        return true;
    default:
        return false;
    }
}

bool OracleEnvironment::IsTexturePixelFormatInteger(u32 handle) {
    return IsIntegerFormat(ReadTexturePixelFormat(handle));
}

std::optional<u32> OracleEnvironment::RefCbuf(u32 bank, u64 key) const {
    if (!ref || !(ref_mask & (bank == 1 ? MaskBank1 : MaskCbuf)))
        return std::nullopt;
    const auto it = ref->cbuf_values.find(key);
    return it == ref->cbuf_values.end() ? std::nullopt : std::optional{it->second};
}

std::optional<Shader::TextureType> OracleEnvironment::RefType(u32 handle) const {
    if (!ref || !(ref_mask & MaskTypes))
        return std::nullopt;
    const auto it = ref->texture_types.find(handle);
    return it == ref->texture_types.end() ? std::nullopt : std::optional{it->second};
}

std::optional<Shader::TexturePixelFormat> OracleEnvironment::RefFormat(u32 handle) const {
    if (!ref || !(ref_mask & MaskFormats))
        return std::nullopt;
    const auto it = ref->texture_formats.find(handle);
    return it == ref->texture_formats.end() ? std::nullopt : std::optional{it->second};
}

std::optional<u8> OracleEnvironment::RefPointer(u64 key) const {
    if (!ref || !(ref_mask & MaskPointers))
        return std::nullopt;
    const auto it = ref->pointers.find(key);
    return it == ref->pointers.end() ? std::nullopt : std::optional{it->second};
}

void OracleEnvironment::ApplyReferenceScalars() {
    if (!ref || !(ref_mask & MaskScalars))
        return;
    local_memory = ref->local_memory;
    texture_bound = ref->texture_bound;
    workgroup = ref->workgroup;
    shared_memory = ref->shared_memory;
}

struct CacheRecord {
    std::vector<RecordedEnvironment> envs;
    std::vector<u8> key;
};

std::vector<CacheRecord> ParseCache(std::span<const u8> bytes) {
    if (bytes.size() < 12 || std::memcmp(bytes.data(), "yuzucach", 8) != 0)
        throw std::runtime_error("Not a Lichii pipeline cache");
    std::vector<CacheRecord> records;
    size_t offset = 12;
    while (offset < bytes.size()) {
        CacheRecord record;
        const u32 count = Load<u32>(bytes, offset);
        offset += 4;
        if (count == 0 || count > 6)
            throw std::runtime_error("Invalid environment count");
        record.envs.resize(count);
        for (auto& env : record.envs)
            offset = env.Parse(bytes, offset);
        const size_t key_size =
            record.envs[0].ShaderStage() == Stage::Compute ? ComputeKeySize : GraphicsKeySize;
        if (offset + key_size > bytes.size())
            throw std::runtime_error("Truncated key");
        record.key.assign(bytes.begin() + offset, bytes.begin() + offset + key_size);
        offset += key_size;
        records.push_back(std::move(record));
    }
    return records;
}

// ---------------------------------------------------------------------------------------------
// Translation and SPIR-V emission with a fixed offline profile.

Shader::HostTranslateInfo HostInfo() {
    // NVIDIA values as vk_pipeline_cache.cpp sets them on an RTX 30-series device.
    Shader::HostTranslateInfo host{};
    host.min_ssbo_alignment = 16;
    host.max_per_stage_descriptor_sampled_images = 1048576;
    host.max_per_stage_resources = 4294967295U;
    host.max_descriptor_set_samplers = 1048576;
    host.max_descriptor_set_uniform_buffers = 1048576;
    host.max_descriptor_set_uniform_buffers_dynamic = 15;
    host.max_descriptor_set_storage_buffers = 1048576;
    host.max_descriptor_set_storage_buffers_dynamic = 16;
    host.max_descriptor_set_sampled_images = 1048576;
    host.max_descriptor_set_storage_images = 1048576;
    host.max_descriptor_set_input_attachements = 1048576;
    host.support_float64 = true;
    host.support_float16 = true;
    host.support_int64 = true;
    host.support_snorm_render_buffer = true;
    host.support_viewport_index_layer = true;
    host.support_viewport_mask = true;
    host.support_geometry_shader_passthrough = true;
    host.support_conditional_barrier = true;
    host.support_byte_addressed_r16 = true;
    host.ApplyDescriptorLimitPolicy();
    return host;
}

struct Translated {
    std::vector<u32> spirv;
};

Translated TranslateAndEmit(Shader::Environment& env, Stage stage) {
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
    Shader::ObjectPool<Shader::IR::Inst> inst_pool;
    Shader::ObjectPool<Shader::IR::Block> block_pool;
    const auto host = HostInfo();
    const u32 entry = stage == Stage::Compute ? env.StartAddress()
                                              : static_cast<u32>(env.StartAddress() + SphSize);
    std::optional<Shader::Maxwell::Flow::CFG> cfg_storage;
    {
        Shader::TranslateProfile::Scope scope{Shader::TranslateProfile::Step::Cfg};
        cfg_storage.emplace(env, flow_pool, Shader::Maxwell::Location{entry}, false);
    }
    auto& cfg = *cfg_storage;
    auto program = Shader::Maxwell::TranslateProgram(inst_pool, block_pool, env, cfg, host);
    Shader::RuntimeInfo runtime{};
    runtime.previous_stage_stores.mask.set();
    runtime.convert_depth_mode = true;
    for (auto& type : runtime.generic_input_types)
        type = Shader::AttributeType::Float;
    Shader::Maxwell::ConvertLegacyToGeneric(program, runtime);
    Shader::Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.unified_descriptor_binding = true;
    profile.min_ssbo_alignment = host.min_ssbo_alignment;
    profile.support_int64 = true;
    profile.support_int16 = true;
    profile.support_int8 = true;
    profile.support_vote = true;
    profile.support_shuffle_relative = true;
    profile.support_geometry_streams = true;
    profile.support_geometry_shader_passthrough = true;
    profile.support_viewport_index_layer_non_geometry = true;
    profile.support_multi_viewport = true;
    profile.max_user_clip_distances = 8;
    profile.max_subgroup_size = 32;
    profile.support_typeless_image_loads = true;
    profile.support_float_controls = true;
    Shader::Backend::Bindings bindings{};
    Shader::TranslateProfile::Scope scope{Shader::TranslateProfile::Step::EmitSpirv};
    return {Shader::Backend::SPIRV::EmitSPIRV(profile, runtime, program, bindings)};
}

struct TranslateFacts {
    std::map<u64, Shader::TextureType> slot_types; // non-2D types texture descriptors carry
    u32 generic_loads{};                           // vertex inputs the program loads
};

TranslateFacts Translate(Shader::Environment& env, Stage stage) {
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
    Shader::ObjectPool<Shader::IR::Inst> inst_pool;
    Shader::ObjectPool<Shader::IR::Block> block_pool;
    static const auto host = HostInfo();
    const u32 entry = stage == Stage::Compute ? env.StartAddress()
                                              : static_cast<u32>(env.StartAddress() + SphSize);
    std::optional<Shader::Maxwell::Flow::CFG> cfg_storage;
    {
        Shader::TranslateProfile::Scope scope{Shader::TranslateProfile::Step::Cfg};
        cfg_storage.emplace(env, flow_pool, Shader::Maxwell::Location{entry}, false);
    }
    auto& cfg = *cfg_storage;
    const auto program = Shader::Maxwell::TranslateProgram(inst_pool, block_pool, env, cfg, host);
    TranslateFacts facts;
    for (const auto& desc : program.info.texture_descriptors) {
        if (desc.type != Shader::TextureType::Color2D && desc.handle_program == ~0U)
            facts.slot_types.emplace(CbufKey(desc.cbuf_index, desc.cbuf_offset), desc.type);
    }
    for (u32 index = 0; index < 32; ++index) {
        if (program.info.loads.Generic(index))
            facts.generic_loads |= 1U << index;
    }
    return facts;
}

// Generic attributes a vertex program's header says it reads (4 bits each from 0x18).
u32 HeaderReadMask(std::span<const u8> sph) {
    u32 mask = 0;
    for (u32 i = 0; i < 16; ++i) {
        for (u32 half = 0; half < 2; ++half) {
            if ((sph[0x18 + i] >> (4 * half)) & 0xf)
                mask |= 1U << (i * 2 + half);
        }
    }
    return mask;
}

// The oracle environment of a program: a first translation finds the types the program's own
// texture instructions give each slot, the second answers size queries on those slots with them.
struct MeshState {
    std::array<u32, 3> registers{};
    std::array<u8, SphSize> header{};
};

// The mesh state of a draw with this mesh program, from its control section: A holds the output
// topology (triangles) and the maximum vertex and primitive counts, B the thread count, INIT the
// task thread count (0x20 in every recorded draw) and a count `lines` the program does not give
// (2, 9 or 25 in the recorded draws). The pass-through geometry header is in the code area.
MeshState DeriveMeshState(const PackageProgram& mesh, u32 lines) {
    MeshState state;
    const auto& ctl = mesh.control;
    const u32 threads = Load<u16>(ctl, 0x58);
    const u32 max_vertices = Load<u16>(ctl, 0x60);
    const u32 max_primitives = Load<u16>(ctl, 0x62);
    state.registers = {4U | (max_vertices << 4) | (max_primitives << 13), threads << 20,
                       0x20U | (lines << 12) | (lines << 22)};
    if (ctl.size() > 0x68 && (ctl[0x68] & 0x80)) {
        const u32 at = Load<u32>(ctl, 0x64);
        if (at + SphSize <= mesh.code.size())
            std::memcpy(state.header.data(), mesh.code.data() + at, SphSize);
    }
    return state;
}

u64 MeshStateHash(const MeshState& state) {
    std::array<u32, 3 + 20> words{state.registers[0], state.registers[1], state.registers[2]};
    std::memcpy(words.data() + 3, state.header.data(), SphSize);
    return Common::CityHash64(reinterpret_cast<const char*>(words.data()), sizeof(words)) | 1;
}

std::unique_ptr<OracleEnvironment> MakeOracle(const PackageProgram& program, const Lowered& lowered,
                                              const Priors& priors, u32* generic_loads = nullptr,
                                              const MeshState* mesh = nullptr) {
    // A task environment records the registers but not the geometry header (GraphicsEnvironment).
    const auto apply_mesh = [&](OracleEnvironment& env) {
        if (!mesh)
            return;
        MeshState state = *mesh;
        if (program.stage == Stage::Task)
            state.header = {};
        env.SetMeshState(state.registers, state.header);
    };
    auto first = std::make_unique<OracleEnvironment>(program, lowered, priors);
    apply_mesh(*first);
    auto facts = Translate(*first, program.stage);
    if (generic_loads)
        *generic_loads = facts.generic_loads;
    if (facts.slot_types.empty())
        return first;
    auto second = std::make_unique<OracleEnvironment>(program, lowered, priors);
    apply_mesh(*second);
    second->slot_types = std::move(facts.slot_types);
    (void)Translate(*second, program.stage);
    return second;
}

template <typename F>
void ParallelFor(size_t count, unsigned threads, F&& body) {
    std::atomic<size_t> next{0};
    std::vector<std::thread> workers;
    for (unsigned t = 0; t < threads; ++t) {
        workers.emplace_back([&] {
            for (size_t i; (i = next.fetch_add(1)) < count;)
                body(i);
        });
    }
    for (auto& worker : workers)
        worker.join();
}

std::string Clean(std::string text) {
    std::ranges::replace(text, '\t', ' ');
    std::ranges::replace(text, '\n', ' ');
    return text;
}

constexpr std::array StageNames{"VertexB", "TessellationControl", "TessellationEval", "Geometry",
                                "Fragment", "Compute", "VertexA", "Task", "Mesh"};

// ---------------------------------------------------------------------------------------------

int Scan(const std::filesystem::path& pkg, const std::filesystem::path& prefix, unsigned threads,
         const Priors& priors) {
    const Packages packages{pkg};
    const size_t count = packages.programs.size();
    std::vector<std::string> lines(count);
    std::vector<std::vector<u8>> envs(count);
    std::vector<u64> hashes(count);
    std::atomic<size_t> ok{0}, failed{0};
    ParallelFor(count, threads, [&](size_t i) {
        const auto& program = packages.programs[i];
        std::string error;
        std::string status = "ok";
        u64 hash = 0, native_key = 0;
        size_t words = 0, queries = 0;
        u32 loads = 0, header_reads = 0;
        try {
            const auto lowered = CaptureAndLower(program, error);
            if (!lowered) {
                status = "capture";
            } else {
                hash = lowered->hash;
                native_key = lowered->native_key;
                words = lowered->code.size();
                if (program.stage == Stage::Task || program.stage == Stage::Mesh) {
                    status = "mesh-hash-only";
                } else {
                    const auto env = MakeOracle(program, *lowered, priors, &loads);
                    if (program.stage == Stage::VertexB)
                        header_reads = HeaderReadMask(program.code);
                    envs[i] = env->Serialize();
                    queries = env->QueryCount();
                }
            }
        } catch (const std::exception& exception) {
            status = "translate";
            error = exception.what();
        }
        hashes[i] = hash;
        (status == "ok" ? ok : failed).fetch_add(1);
        lines[i] = fmt::format("{}\t{}\t{}\t{}\t{:016x}\t{:016x}\t{}\t{}\t{}\t{:08x}\t{:08x}\t{}\n",
                               program.file, program.index,
                               StageNames[static_cast<u32>(program.stage)], status, hash, native_key,
                               words, envs[i].size(), queries, loads, header_reads, Clean(error));
    });
    {
        std::ofstream tsv(prefix.string() + ".tsv", std::ios::binary | std::ios::trunc);
        tsv << "file\tindex\tstage\tstatus\tcode_hash\tnative_key\tcode_words\tenv_bytes\tqueries\t"
               "generic_loads\theader_reads\terror\n";
        for (const auto& line : lines)
            tsv << line;
    }
    {
        // NXPENV01, then per environment: u32 program ordinal, u64 code hash, u32 size, bytes.
        std::ofstream out(prefix.string() + ".envs", std::ios::binary | std::ios::trunc);
        out.write("NXPENV01", 8);
        for (size_t i = 0; i < count; ++i) {
            if (envs[i].empty())
                continue;
            const u32 ordinal = static_cast<u32>(i);
            const u32 size = static_cast<u32>(envs[i].size());
            out.write(reinterpret_cast<const char*>(&ordinal), 4);
            out.write(reinterpret_cast<const char*>(&hashes[i]), 8);
            out.write(reinterpret_cast<const char*>(&size), 4);
            out.write(reinterpret_cast<const char*>(envs[i].data()), size);
        }
    }
    fmt::print("programs {} ok {} not-ok {}\n", count, ok.load(), failed.load());
    return 0;
}

int Validate(const std::filesystem::path& pkg, const std::filesystem::path& cache_path,
             const std::filesystem::path& prefix, unsigned threads, const Priors& priors) {
    const Packages packages{pkg};
    // Lower every package program and index it by its runtime hash.
    std::vector<std::optional<Lowered>> lowered(packages.programs.size());
    ParallelFor(packages.programs.size(), threads, [&](size_t i) {
        std::string error;
        try {
            lowered[i] = CaptureAndLower(packages.programs[i], error);
        } catch (const std::exception&) {
        }
    });
    std::unordered_multimap<u64, size_t> by_hash;
    for (size_t i = 0; i < lowered.size(); ++i)
        if (lowered[i])
            by_hash.emplace(lowered[i]->hash, i);

    const auto cache_bytes = ReadFile(cache_path);
    auto records = ParseCache(cache_bytes);
    // Distinct recorded environments (stage + code + facts are what translation sees).
    struct Item {
        RecordedEnvironment* env;
        u64 hash;
    };
    std::vector<Item> items;
    std::map<std::pair<u64, std::vector<u8>>, size_t> seen;
    for (auto& record : records) {
        for (auto& env : record.envs) {
            const Stage stage = env.ShaderStage();
            const u64 hash = NativeShaderHash(env.code);
            items.push_back({&env, hash});
        }
    }
    std::vector<std::string> lines(items.size());
    std::map<std::string, size_t> summary;
    std::mutex mutex;
    ParallelFor(items.size(), threads, [&](size_t i) {
        auto& [env, hash] = items[i];
        const Stage stage = env->ShaderStage();
        std::string result;
        std::string detail;
        std::string mesh_note;
        const auto range = by_hash.equal_range(hash);
        const bool found = range.first != range.second;
        const bool mesh_stage = stage == Stage::Task || stage == Stage::Mesh;
        if (!found) {
            result = "no-package-program";
        } else {
            const size_t index = range.first->second;
            try {
                const auto recorded = TranslateAndEmit(*env, stage);
                try {
                    // Mesh state as recorded: the comparison is about the other facts.
                    MeshState recorded_mesh{env->mesh_regs, env->mesh_header};
                    const auto made = MakeOracle(packages.programs[index], *lowered[index], priors,
                                                 nullptr, mesh_stage ? &recorded_mesh : nullptr);
                    OracleEnvironment oracle{packages.programs[index], *lowered[index], priors};
                    if (mesh_stage)
                        oracle.SetMeshState(env->mesh_regs, env->mesh_header);
                    oracle.slot_types = made->slot_types;
                    const auto predicted = TranslateAndEmit(oracle, stage);
                    result = recorded.spirv == predicted.spirv ? "spirv-equal" : "spirv-differs";
                    if (stage == Stage::Mesh) {
                        const auto derived = DeriveMeshState(packages.programs[index],
                                                             (env->mesh_regs[2] >> 12) & 0x1ff);
                        mesh_note = derived.registers == env->mesh_regs &&
                                            derived.header == env->mesh_header
                                        ? "derived mesh state equal"
                                        : "derived mesh state differs";
                    }
                    if (result == "spirv-differs") {
                        // Substitute recorded answers one class at a time to find the cause.
                        constexpr std::array<std::pair<unsigned, const char*>, 7> trials{{
                            {MaskCbuf | MaskTypes | MaskFormats, "texture"},
                            {MaskCbuf | MaskFormats, "texture-format"},
                            {MaskCbuf | MaskTypes, "texture-type"},
                            {MaskPointers, "storage-pointer"},
                            {MaskBank1, "bank1"},
                            {MaskScalars, "scalars"},
                            {MaskCbuf | MaskTypes | MaskFormats | MaskPointers | MaskBank1 | MaskScalars, "all"},
                        }};
                        for (const auto& [mask, name] : trials) {
                            try {
                                OracleEnvironment trial{packages.programs[index], *lowered[index], priors};
                                trial.ref = env;
                                trial.ref_mask = mask;
                                trial.ApplyReferenceScalars();
                                if (TranslateAndEmit(trial, stage).spirv == recorded.spirv) {
                                    detail += (detail.empty() ? "" : ",") + std::string{name};
                                }
                            } catch (const std::exception&) {
                            }
                        }
                        if (detail.empty())
                            detail = "unexplained";
                    }
                } catch (const std::exception& exception) {
                    result = "oracle-failed";
                    detail = exception.what();
                }
            } catch (const std::exception& exception) {
                result = "recorded-failed";
                detail = exception.what();
            }
        }
        {
            std::scoped_lock lock{mutex};
            ++summary[fmt::format("{} {}", StageNames[static_cast<u32>(stage)], result)];
            if (!mesh_note.empty())
                ++summary[fmt::format("Mesh {}", mesh_note)];
            if (!detail.empty() && result == "spirv-differs")
                ++summary[fmt::format("{} differs: {}", StageNames[static_cast<u32>(stage)], detail)];
        }
        lines[i] = fmt::format("{}\t{:016x}\t{}\t{}\n", StageNames[static_cast<u32>(stage)], hash,
                               result, Clean(detail));
    });
    std::ofstream tsv(prefix.string() + ".tsv", std::ios::binary | std::ios::trunc);
    tsv << "stage\tcode_hash\tresult\tdetail\n";
    for (const auto& line : lines)
        tsv << line;
    Json json = Json::object();
    for (const auto& [key, value] : summary) {
        json[key] = value;
        fmt::print("{:<60} {}\n", key, value);
    }
    std::ofstream(prefix.string() + ".json", std::ios::trunc) << json.dump(1) << '\n';
    return 0;
}

// Oracle environments of task and mesh programs for given draws: each input line names a task
// program (or -) and a mesh program by shader hash, and the INIT `lines` count. Writes NXPMESH1,
// then per line: u32 line, u64 mesh state hash, u64 task hash, u64 mesh hash, u32 size and the task
// environment (size 0 without one), u32 size and the mesh environment.
int MeshEnvs(const std::filesystem::path& pkg, const std::filesystem::path& combos,
             const std::filesystem::path& prefix, unsigned threads, const Priors& priors) {
    const Packages packages{pkg};
    std::vector<std::optional<Lowered>> lowered(packages.programs.size());
    ParallelFor(packages.programs.size(), threads, [&](size_t i) {
        const auto stage = packages.programs[i].stage;
        if (stage != Stage::Task && stage != Stage::Mesh)
            return;
        std::string error;
        lowered[i] = CaptureAndLower(packages.programs[i], error);
    });
    std::unordered_map<u64, size_t> by_hash;
    for (size_t i = 0; i < lowered.size(); ++i)
        if (lowered[i])
            by_hash.try_emplace(lowered[i]->hash, i);
    struct Line {
        u64 task{}, mesh{};
        u32 lines{};
    };
    std::vector<Line> input;
    {
        std::ifstream file(combos);
        std::string task, mesh;
        u32 lines;
        while (file >> task >> mesh >> lines)
            input.push_back({task == "-" ? 0 : std::stoull(task, nullptr, 16),
                             std::stoull(mesh, nullptr, 16), lines});
    }
    struct Output {
        u64 state_hash{};
        std::vector<u8> task, mesh;
        std::string error;
    };
    std::vector<Output> out(input.size());
    ParallelFor(input.size(), threads, [&](size_t i) {
        try {
            const auto mesh_it = by_hash.find(input[i].mesh);
            if (mesh_it == by_hash.end())
                throw std::runtime_error("mesh program not found");
            const auto& mesh_program = packages.programs[mesh_it->second];
            const MeshState state = DeriveMeshState(mesh_program, input[i].lines);
            out[i].state_hash = MeshStateHash(state);
            out[i].mesh = MakeOracle(mesh_program, *lowered[mesh_it->second], priors, nullptr, &state)
                              ->Serialize();
            if (input[i].task) {
                const auto task_it = by_hash.find(input[i].task);
                if (task_it == by_hash.end())
                    throw std::runtime_error("task program not found");
                out[i].task = MakeOracle(packages.programs[task_it->second],
                                         *lowered[task_it->second], priors, nullptr, &state)
                                  ->Serialize();
            }
        } catch (const std::exception& e) {
            out[i].error = e.what();
            out[i].mesh.clear();
        }
    });
    std::ofstream file(prefix.string() + ".mesh", std::ios::binary | std::ios::trunc);
    file.write("NXPMESH1", 8);
    size_t written = 0;
    std::map<std::string, size_t> errors;
    for (size_t i = 0; i < out.size(); ++i) {
        if (out[i].mesh.empty()) {
            ++errors[out[i].error.substr(0, 60)];
            continue;
        }
        const u32 line = static_cast<u32>(i);
        const u32 task_size = static_cast<u32>(out[i].task.size());
        const u32 mesh_size = static_cast<u32>(out[i].mesh.size());
        file.write(reinterpret_cast<const char*>(&line), 4);
        file.write(reinterpret_cast<const char*>(&out[i].state_hash), 8);
        file.write(reinterpret_cast<const char*>(&input[i].task), 8);
        file.write(reinterpret_cast<const char*>(&input[i].mesh), 8);
        file.write(reinterpret_cast<const char*>(&task_size), 4);
        file.write(reinterpret_cast<const char*>(out[i].task.data()), task_size);
        file.write(reinterpret_cast<const char*>(&mesh_size), 4);
        file.write(reinterpret_cast<const char*>(out[i].mesh.data()), mesh_size);
        ++written;
    }
    fmt::print("mesh draws {} written {}\n", input.size(), written);
    for (const auto& [error, n] : errors)
        fmt::print("  {:>6}  {}\n", n, error);
    return 0;
}

// Replays every environment of a precompiled file the way the engine's FileEnvironment does and
// checks that each record's key names the code of its environments.
void PrintTranslateProfile(const std::vector<RecordedEnvironment>& envs,
                           const std::vector<u64>& env_nanoseconds,
                           const std::vector<std::string>& errors, unsigned threads,
                           std::chrono::steady_clock::duration wall) {
    namespace P = Shader::TranslateProfile;
    const auto totals = P::Take();
    u64 all = 0, translated = 0;
    std::map<std::string, std::vector<double>> by_stage;
    for (size_t i = 0; i < envs.size(); ++i) {
        if (!errors[i].empty() || env_nanoseconds[i] == 0) {
            continue;
        }
        all += env_nanoseconds[i];
        ++translated;
        by_stage[StageNames[static_cast<u32>(envs[i].ShaderStage())]].push_back(
            static_cast<double>(env_nanoseconds[i]) / 1e6);
    }
    fmt::print("translation profile: {} environments in {:.1f} s on {} threads, {:.1f} s of "
               "thread time, {:.3f} ms per environment\n",
               translated, std::chrono::duration<double>(wall).count(), threads, all / 1e9,
               translated ? all / 1e6 / static_cast<double>(translated) : 0.0);
    fmt::print("  {:34} {:>10} {:>7} {:>10} {:>9}\n", "step", "total ms", "share", "count",
               "us each");
    for (size_t step = 0; step < P::NumSteps; ++step) {
        if (totals.counts[step] == 0) {
            continue;
        }
        const double ms = static_cast<double>(totals.nanoseconds[step]) / 1e6;
        fmt::print("  {:34} {:>10.1f} {:>6.1f}% {:>10} {:>9.1f}\n",
                   P::Name(static_cast<P::Step>(step)), ms,
                   all ? 100.0 * static_cast<double>(totals.nanoseconds[step]) /
                             static_cast<double>(all)
                       : 0.0,
                   totals.counts[step],
                   1000.0 * ms / static_cast<double>(totals.counts[step]));
    }
    fmt::print("  by stage (ms per environment: mean, median, 90th, 99th, max)\n");
    for (auto& [stage, times] : by_stage) {
        std::sort(times.begin(), times.end());
        const auto at = [&](double q) {
            return times[std::min(times.size() - 1, static_cast<size_t>(q * times.size()))];
        };
        double sum = 0;
        for (const double t : times) {
            sum += t;
        }
        fmt::print("  {:10} {:>6}  {:7.3f} {:7.3f} {:7.3f} {:7.3f} {:8.3f}\n", stage,
                   times.size(), sum / static_cast<double>(times.size()), at(0.5), at(0.9),
                   at(0.99), times.back());
    }
}

int Check(const std::filesystem::path& path, unsigned threads) {
    const auto bytes = ReadFile(path);
    const std::span<const u8> file{bytes};
    if (file.size() < 24 || std::memcmp(file.data(), "NX2PREP1", 8) != 0 ||
        Load<u32>(file, 8) != VideoCommon::VulkanPipelineCacheVersion)
        throw std::runtime_error(fmt::format("Not an NX2PREP1 file of cache version {}",
                                             VideoCommon::VulkanPipelineCacheVersion));
    const u32 env_count = Load<u32>(file, 12);
    const u32 record_count = Load<u32>(file, 16);
    size_t offset = 24;
    std::vector<std::pair<u64, u64>> index(env_count);
    for (auto& entry : index) {
        entry = {Load<u64>(file, offset), Load<u64>(file, offset + 8)};
        offset += 16;
    }
    std::vector<RecordedEnvironment> envs(env_count);
    std::vector<u64> hashes(env_count);
    std::vector<std::string> errors(env_count);
    // NX2_CHECK_SPIRV=<file> (with NX2_CHECK_EMIT=1): a hash of each environment's SPIR-V, one
    // line per environment, to compare a translator change against: the same file means the
    // same SPIR-V for every environment.
    std::vector<u64> spirv_hashes(env_count);
    // NX2_CHECK_EMIT=1 also emits SPIR-V; with --threads 1 the environment being emitted is
    // written to NX2_CHECK_PROGRESS first, so a crash names it.
    const bool emit = std::getenv("NX2_CHECK_EMIT") != nullptr;
    const char* progress = std::getenv("NX2_CHECK_PROGRESS");
    // NX2_CHECK_START=<index>: resume after a crash (environments before it are not checked).
    const char* start_text = std::getenv("NX2_CHECK_START");
    const size_t start = start_text ? std::stoull(start_text) : 0;
    // NX2_CHECK_PROFILE=1: time each translation step (Shader::TranslateProfile) and each
    // environment's whole translation, by stage.
    const bool profile = std::getenv("NX2_CHECK_PROFILE") != nullptr;
    Shader::TranslateProfile::SetEnabled(profile);
    std::vector<u64> env_nanoseconds(env_count);
    const auto profile_begin = std::chrono::steady_clock::now();
    ParallelFor(env_count, threads, [&](size_t i) {
        if (i < start)
            return;
        try {
            const size_t end = envs[i].Parse(file, index[i].first);
            if (end != index[i].first + index[i].second)
                throw std::runtime_error("environment size mismatch");
            hashes[i] = NativeShaderHash(envs[i].code);
            const Stage stage = envs[i].ShaderStage();
            if (progress && threads == 1) {
                std::ofstream(progress, std::ios::trunc)
                    << i << ' ' << StageNames[static_cast<u32>(stage)] << ' '
                    << fmt::format("{:016x}", hashes[i]) << '\n';
            }
            const auto begin = std::chrono::steady_clock::now();
            if (emit) {
                const auto translated = TranslateAndEmit(envs[i], stage);
                spirv_hashes[i] = Common::CityHash64(
                    reinterpret_cast<const char*>(translated.spirv.data()),
                    translated.spirv.size() * sizeof(u32));
            } else {
                Translate(envs[i], stage);
            }
            env_nanoseconds[i] = static_cast<u64>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - begin)
                    .count());
        } catch (const std::exception& e) {
            errors[i] = e.what();
        }
    });
    if (const char* spirv_out = std::getenv("NX2_CHECK_SPIRV"); spirv_out && emit) {
        std::ofstream out(spirv_out, std::ios::trunc);
        for (size_t i = 0; i < env_count; ++i) {
            out << i << ' '
                << (errors[i].empty() ? fmt::format("{:016x}", spirv_hashes[i])
                                      : std::string{"failed"})
                << '\n';
        }
    }
    if (profile) {
        PrintTranslateProfile(envs, env_nanoseconds, errors, threads,
                              std::chrono::steady_clock::now() - profile_begin);
    }
    std::map<std::string, size_t> failures;
    size_t failed_envs = 0;
    for (const auto& error : errors) {
        if (!error.empty()) {
            ++failed_envs;
            ++failures[error.substr(0, 60)];
        }
    }
    size_t bad_records = 0, graphics = 0, compute = 0, mesh = 0;
    for (u32 r = 0; r < record_count; ++r) {
        const u8 kind = Load<u8>(file, offset);
        const u8 count = Load<u8>(file, offset + 1);
        offset += 4;
        std::vector<u32> ids(count);
        for (auto& id : ids) {
            id = Load<u32>(file, offset);
            offset += 4;
        }
        if (kind == 1) {
            ++compute;
            const u64 key_hash = Load<u64>(file, offset);
            const u32 shared = Load<u32>(file, offset + 8);
            const auto workgroup = Load<std::array<u32, 3>>(file, offset + 12);
            const auto& env = envs[ids[0]];
            if (count != 1 || key_hash != hashes[ids[0]] || shared != env.shared_memory ||
                workgroup != env.workgroup)
                ++bad_records;
            offset += ComputeKeySize;
        } else {
            ++graphics;
            const auto key_hashes = Load<std::array<u64, 6>>(file, offset);
            size_t next = 0;
            bool ok = true, has_mesh = false;
            for (size_t slot = 0; slot < 6; ++slot) {
                if (key_hashes[slot] == 0)
                    continue;
                if (next >= ids.size()) {
                    ok = false;
                    break;
                }
                const auto& env = envs[ids[next]];
                const bool mesh_stage =
                    env.ShaderStage() == Stage::Task || env.ShaderStage() == Stage::Mesh;
                has_mesh |= mesh_stage;
                // Task and mesh keys add the mesh state to the code hash.
                if (!mesh_stage && key_hashes[slot] != hashes[ids[next]])
                    ok = false;
                ++next;
            }
            mesh += has_mesh;
            if (!ok || next != ids.size())
                ++bad_records;
            offset += GraphicsKeySize;
        }
    }
    fmt::print("environments {} replay failures {}\n", env_count, failed_envs);
    for (const auto& [error, n] : failures)
        fmt::print("  {:>6}  {}\n", n, error);
    fmt::print("records {} (graphics {}, of them mesh {}; compute {}) with mismatched keys {}\n",
               record_count, graphics, mesh, compute, bad_records);
    fmt::print("trailing bytes after records: {}\n",
               offset <= file.size() && offset == (index.empty() ? offset : index.front().first)
                   ? "none"
                   : "MISMATCH");
    return failed_envs || bad_records ? 1 : 0;
}

// ---------------------------------------------------------------------------------------------
// extract: the game's own shader programs, found by content (core/file_sys/nx2_shader_extract).

/// The game's RomFS as extract reads it, and which game it is.
struct Game {
    FileSys::RealVfsFilesystem filesystem;
    std::unique_ptr<FileSys::NupContainer> container;
    std::vector<FileSys::VirtualDir> roots;
    u64 title_id{};
    u32 version{};
};

void OpenGame(Game& out, const std::filesystem::path& game, const std::filesystem::path& spill,
              const std::string& packages) {
    if (std::filesystem::is_directory(game)) {
        out.roots.push_back(out.filesystem.OpenDirectory(Common::FS::PathToUTF8String(game),
                                                         FileSys::OpenMode::Read));
        return;
    }
    const auto file =
        out.filesystem.OpenFile(Common::FS::PathToUTF8String(game), FileSys::OpenMode::Read);
    if (!file || !FileSys::NupContainer::IsNupFile(file)) {
        throw std::runtime_error("--game must be a .nup file or an unpacked RomFS directory");
    }
    std::filesystem::create_directories(spill);
    out.container = std::make_unique<FileSys::NupContainer>(
        file, FileSys::NupDecompressCache::DefaultCap, spill);
    if (out.container->GetStatus() != Loader::ResultStatus::Success) {
        throw std::runtime_error("cannot read the NUP");
    }
    FileSys::VirtualDir merged;
    for (size_t i = 0; i < out.container->GetPackages().size(); ++i) {
        const auto& package = out.container->GetPackages()[i];
        const auto romfs = out.container->GetRomFSDir(i);
        fmt::print("package {}: {:016X} v{} kind {}{}\n", i, package.title_id, package.version,
                   static_cast<int>(package.kind), romfs ? "" : " (no RomFS)");
        if (package.kind == FileSys::NupContainer::PackageKind::Base) {
            out.title_id = package.title_id;
        }
        if (package.kind != FileSys::NupContainer::PackageKind::Addon) {
            out.version = std::max(out.version, package.version);
        }
        if (!romfs) {
            continue;
        }
        if (packages == "union") {
            out.roots.push_back(romfs); // every version, as nxnative's shader_split.py reads them
        } else {
            // What the game loads: later packages replace the base's files.
            merged = merged ? FileSys::NupContainer::MergeShadowDirs(merged, romfs) : romfs;
        }
    }
    if (merged) {
        out.roots.push_back(merged);
    }
}

FileSys::Nx2ShaderExtract::Result ExtractGame(const Game& game, unsigned threads) {
    namespace X = FileSys::Nx2ShaderExtract;
    X::Progress progress;
    std::atomic<bool> done{false};
    std::jthread reporter([&] {
        for (int tick = 0; !done.load(); ++tick) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            if (tick % 50 == 49 && progress.files_total) {
                fmt::print(stderr, "progress: {}/{}\n", progress.files_done.load(),
                           progress.files_total);
            }
        }
    });
    auto result = X::Extract(game.roots, threads, &progress);
    done.store(true);
    return result;
}

int Extract(const std::filesystem::path& game_path, const std::filesystem::path& prefix,
            const std::string& packages, unsigned threads) {
    namespace X = FileSys::Nx2ShaderExtract;
    Game game;
    auto spill = prefix;
    spill += ".spill";
    OpenGame(game, game_path, spill, packages);
    const auto begin = std::chrono::steady_clock::now();
    const auto result = ExtractGame(game, threads);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();

    auto path = [&](const char* suffix) {
        auto p = prefix;
        p += suffix;
        return p;
    };
    std::ofstream bin(path(".programs"), std::ios::binary);
    const auto put = [&](const auto& value) {
        bin.write(reinterpret_cast<const char*>(&value), sizeof(value));
    };
    bin.write("NXSX0001", 8);
    put(static_cast<u32>(result.programs.size()));
    std::ofstream tsv(path(".tsv"));
    tsv << "fingerprint\tstage\tcontrol_size\tcode_size\torigin\n";
    std::map<std::string, u32> stages;
    for (const auto& program : result.programs) {
        put(static_cast<u8>(program.stage));
        put(program.fingerprint);
        put(static_cast<u32>(program.control.size()));
        put(static_cast<u32>(program.code.size()));
        put(static_cast<u32>(program.origin.size()));
        bin.write(program.origin.data(), static_cast<std::streamsize>(program.origin.size()));
        bin.write(reinterpret_cast<const char*>(program.control.data()),
                  static_cast<std::streamsize>(program.control.size()));
        bin.write(reinterpret_cast<const char*>(program.code.data()),
                  static_cast<std::streamsize>(program.code.size()));
        tsv << fmt::format("{:016x}\t{}\t{}\t{}\t{}\n", program.fingerprint,
                           X::StageName(program.stage), program.control.size(),
                           program.code.size(), program.origin);
        ++stages[X::StageName(program.stage)];
    }
    std::ofstream variations(path(".variations.tsv"));
    for (const auto& variation : result.variations) {
        variations << variation.label;
        for (const u32 index : variation.programs) {
            variations << fmt::format("\t{:016x}", result.programs[index].fingerprint);
        }
        variations << '\n';
    }
    std::ofstream errors(path(".errors.txt"));
    for (const auto& error : result.errors) {
        errors << error << '\n';
    }
    const auto& s = result.stats;
    fmt::print("files {} ({:.1f} GB read), inflated {:.1f} GB, compressed files skipped after a "
               "look {}, containers {}\n",
               s.files, s.file_bytes / 1e9, s.inflated_bytes / 1e9, s.peeked_only, s.containers);
    fmt::print("programs {} from {} occurrences, variations {}, errors {}, in {:.1f} s\n",
               result.programs.size(), s.occurrences, result.variations.size(),
               result.errors.size(), seconds);
    for (const auto& [stage, count] : stages) {
        fmt::print("  {:8} {}\n", stage, count);
    }
    return 0;
}

void PrintReport(const VideoCommon::Nx2Recipe::Report& report) {
    fmt::print("programs {} (not in the game {}, not lowered {}), environments {} (withheld as "
               "game data {}), render states {}, records {} (dropped {})\n",
               report.programs, report.programs_missing, report.programs_failed,
               report.environments, report.environments_withheld, report.states, report.records,
               report.records_dropped);
    for (size_t i = 0; i < std::min<size_t>(report.errors.size(), 8); ++i) {
        fmt::print("  {}\n", report.errors[i]);
    }
    if (report.errors.size() > 8) {
        fmt::print("  ... {} more\n", report.errors.size() - 8);
    }
}

// recipe: the precompiled file without the game's code (video_core/nx2_recipe.h).
int Recipe(const std::filesystem::path& precompiled, const std::filesystem::path& game_path,
           const std::filesystem::path& out, unsigned threads) {
    Game game;
    auto spill = out;
    spill += ".spill";
    OpenGame(game, game_path, spill, "merged");
    const auto programs = ExtractGame(game, threads);
    const auto bytes = ReadFile(precompiled);
    VideoCommon::Nx2Recipe::Report report;
    const auto recipe = VideoCommon::Nx2Recipe::MakeRecipe(
        bytes,  programs,
        {.title_id = game.title_id, .game_version = game.version,
         .cache_version = Load<u32>(bytes, 8)},
        report, threads);
    std::ofstream(out, std::ios::binary)
        .write(reinterpret_cast<const char*>(recipe.data()),
               static_cast<std::streamsize>(recipe.size()));
    PrintReport(report);
    fmt::print("recipe {} bytes (precompiled file {} bytes)\n", recipe.size(), bytes.size());
    return report.records_dropped ? 1 : 0;
}

// assemble: a recipe and the game's own programs give this build's precompiled file.
int Assemble(const std::filesystem::path& recipe_path, const std::filesystem::path& game_path,
             const std::filesystem::path& out, unsigned threads) {
    const auto recipe = ReadFile(recipe_path);
    const auto header = VideoCommon::Nx2Recipe::ReadHeader(recipe);
    if (!header) {
        throw std::runtime_error(recipe_path.string() + " is not a precompile recipe");
    }
    Game game;
    auto spill = out;
    spill += ".spill";
    OpenGame(game, game_path, spill, "merged");
    fmt::print("recipe for {:016X} v{} (facts of cache version {}); game {:016X} v{}\n",
               header->title_id, header->game_version, header->cache_version, game.title_id,
               game.version);
    const auto begin = std::chrono::steady_clock::now();
    const auto programs = ExtractGame(game, threads);
    VideoCommon::Nx2Recipe::Report report;
    const auto file = VideoCommon::Nx2Recipe::Assemble(
        recipe, programs, VideoCommon::VulkanPipelineCacheVersion, report, threads);
    std::ofstream(out, std::ios::binary)
        .write(reinterpret_cast<const char*>(file.data()), static_cast<std::streamsize>(file.size()));
    PrintReport(report);
    fmt::print("precompiled file {} bytes in {:.1f} s\n", file.size(),
               std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
    return 0;
}

// audit: whether a recipe holds any of the game's code, every program of every package and
// version in the game's files counted. Exit 1 if it does.
int Audit(const std::filesystem::path& recipe_path, const std::filesystem::path& game_path,
          unsigned threads) {
    const auto recipe = ReadFile(recipe_path);
    if (!VideoCommon::Nx2Recipe::ReadHeader(recipe)) {
        throw std::runtime_error(recipe_path.string() + " is not a precompile recipe");
    }
    Game game;
    auto spill = recipe_path;
    spill += ".spill";
    OpenGame(game, game_path, spill, "union");
    const auto programs = ExtractGame(game, threads);
    const auto matches = VideoCommon::Nx2Recipe::FindGameCode(recipe, programs, threads);
    fmt::print("{} programs checked: {} pieces of their code in the recipe\n",
               programs.programs.size(), matches.size());
    for (size_t i = 0; i < std::min<size_t>(matches.size(), 8); ++i) {
        fmt::print("  program {:016x} +{:#x} at recipe body +{:#x}\n", matches[i].fingerprint,
                   matches[i].program_offset, matches[i].recipe_offset);
    }
    return matches.empty() ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "usage: nx2-precompile scan|validate --pkg DIR [--cache FILE] --out PREFIX "
                         "[--threads N] [--priors FILE]\n";
            return 2;
        }
        const std::string command = argv[1];
        if (command == "cache-version") {
            // The pipeline cache version this tool was built with: the precompiled file's v<N>.
            std::cout << VideoCommon::VulkanPipelineCacheVersion << '\n';
            return 0;
        }
        std::filesystem::path pkg, cache, out, priors_path, check_file, game, recipe;
        std::string packages = "merged";
        unsigned threads = std::max(1U, std::thread::hardware_concurrency() - 2);
        for (int i = 2; i + 1 < argc; i += 2) {
            const std::string arg = argv[i];
            const std::string value = argv[i + 1];
            if (arg == "--pkg") pkg = value;
            else if (arg == "--cache") cache = value;
            else if (arg == "--out") out = value;
            else if (arg == "--threads") threads = static_cast<unsigned>(std::stoul(value));
            else if (arg == "--priors") priors_path = value;
            else if (arg == "--file") check_file = value;
            else if (arg == "--game") game = value;
            else if (arg == "--recipe") recipe = value;
            else if (arg == "--packages") packages = value;
            else throw std::runtime_error("Unknown option " + arg);
        }
        if (command == "recipe") {
            if (check_file.empty() || game.empty() || out.empty())
                throw std::runtime_error("recipe needs --file PRECOMPILED --game GAME --out RECIPE");
            return Recipe(check_file, game, out, threads);
        }
        if (command == "assemble") {
            if (recipe.empty() || game.empty() || out.empty())
                throw std::runtime_error("assemble needs --recipe RECIPE --game GAME --out FILE");
            return Assemble(recipe, game, out, threads);
        }
        if (command == "audit") {
            if (recipe.empty() || game.empty())
                throw std::runtime_error("audit needs --recipe RECIPE --game GAME");
            return Audit(recipe, game, threads);
        }
        if (command == "extract") {
            if (game.empty() || out.empty())
                throw std::runtime_error("extract needs --game and --out");
            return Extract(game, out, packages, threads);
        }
        if (command == "check") {
            if (check_file.empty())
                throw std::runtime_error("check needs --file");
            return Check(check_file, threads);
        }
        if (pkg.empty() || out.empty())
            throw std::runtime_error("--pkg and --out are required");
        const auto priors = Priors::FromFile(priors_path);
        if (command == "scan")
            return Scan(pkg, out, threads, priors);
        if (command == "mesh-envs") {
            if (cache.empty())
                throw std::runtime_error("mesh-envs needs --cache COMBOS");
            return MeshEnvs(pkg, cache, out, threads, priors);
        }
        if (command == "validate") {
            if (cache.empty())
                throw std::runtime_error("validate needs --cache");
            return Validate(pkg, cache, out, threads, priors);
        }
        throw std::runtime_error("Unknown command " + command);
    } catch (const std::exception& exception) {
        std::cerr << "error: " << exception.what() << '\n';
        return 1;
    }
}
