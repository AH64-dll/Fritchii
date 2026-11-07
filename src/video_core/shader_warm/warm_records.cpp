// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>
#include <fstream>
#include <unordered_set>

#include "common/cityhash.h"
#include "shader_recompiler/frontend/ampere/instruction.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ampere/program_capture.h"
#include "video_core/shader_warm/warm_records.h"

namespace VideoCommon::ShaderWarm {
namespace {

constexpr u64 AddressMapMagic = 0x50414d4e4544454EULL;
constexpr u64 HashXor = 0x4F554E4345000002ULL;
constexpr size_t EnvHeaderSize = 6 * 8 + 7 * 4;
constexpr u32 ComputeStage = 5;

template <typename T>
void Put(std::vector<u8>& out, T value) {
    const size_t at = out.size();
    out.resize(at + sizeof(T));
    std::memcpy(out.data() + at, &value, sizeof(T));
}

void PutBytes(std::vector<u8>& out, std::span<const u8> bytes) {
    out.insert(out.end(), bytes.begin(), bytes.end());
}

template <typename T>
T Get(std::span<const u8> data, size_t offset) {
    T value;
    std::memcpy(&value, data.data() + offset, sizeof(T));
    return value;
}

/// The GenericEnvironment::Serialize header: six u64 counts and seven u32 fields.
void PutEnvHeader(std::vector<u8>& out, u64 code_size, u64 cbuf_count, u32 local_memory,
                  u32 texture_bound, u32 start_address, u32 stage) {
    Put<u64>(out, code_size);
    Put<u64>(out, 0); // texture types
    Put<u64>(out, 0); // texture pixel formats
    Put<u64>(out, cbuf_count);
    Put<u64>(out, 0); // storage pointer types
    Put<u64>(out, 0); // cbuf replacements
    Put<u32>(out, local_memory);
    Put<u32>(out, texture_bound);
    Put<u32>(out, start_address);
    Put<u32>(out, 0);                                  // read lowest
    Put<u32>(out, static_cast<u32>(code_size - 8));    // read highest
    Put<u32>(out, 1);                                  // viewport transform state
    Put<u32>(out, stage);
}

struct PairHash {
    u128 a{};
    u128 b{};
    bool operator==(const PairHash&) const = default;
};

struct PairHasher {
    size_t operator()(const PairHash& h) const noexcept {
        return static_cast<size_t>(h.a[0] ^ (h.b[0] * 0x9E3779B97F4A7C15ULL));
    }
};

struct U128Hasher {
    size_t operator()(const u128& h) const noexcept {
        return static_cast<size_t>(h[0]);
    }
};

u128 Hash128(std::span<const u8> bytes) {
    return Common::CityHash128(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

size_t TailSize(u32 stage) {
    if (stage == 5) {
        return 16;
    }
    return 80 + (stage == 3 ? 32 : 0) + (stage == 7 || stage == 8 ? 92 : 0);
}

} // Anonymous namespace

std::optional<std::string> LowerProgram(std::span<const u8> program, bool graphics,
                                        LoweredCode& out) {
    using Shader::Ampere::Instruction;
    const size_t header_size = graphics ? 128 : 0;
    if (program.size() <= header_size || (program.size() - header_size) % sizeof(Instruction) ||
        (program.size() - header_size) / sizeof(Instruction) > 65536) {
        return "Native input must contain 1..65536 complete 16-byte instructions";
    }
    if (graphics) {
        const u32 type = (Get<u32>(program, 0) >> 10) & 15;
        if (type < 1 || type > 5) {
            return "Invalid native graphics header stage";
        }
    }
    std::vector<Instruction> native((program.size() - header_size) / sizeof(Instruction));
    std::memcpy(native.data(), program.data() + header_size, native.size() * sizeof(Instruction));
    Shader::Ampere::ProgramCaptureState capture;
    bool complete = false;
    for (u32 index = 0; index < native.size(); ++index) {
        const auto result = capture.Observe(native[index], index);
        if (result == Shader::Ampere::ProgramCaptureState::Result::InvalidTarget) {
            return "Native allocation has an out-of-range branch target";
        }
        if (result == Shader::Ampere::ProgramCaptureState::Result::Complete) {
            native.resize(index + 1);
            complete = true;
            break;
        }
    }
    if (!complete) {
        return "Native allocation lacks a complete runtime capture";
    }
    for (const auto& instruction : native) {
        if (instruction.Bits(0, 12) == 0x95c) {
            return "Trap guards require captured constant buffers";
        }
    }
    const auto lowered = Shader::Ampere::LowerToMaxwell(native, false, true, {},
                                                        Shader::Ampere::TessellationStage::None);
    if (!lowered) {
        return std::string(lowered.error);
    }
    const auto& offsets = lowered.source_instruction_offsets;
    if (lowered.source_size != native.size() * sizeof(Instruction) ||
        offsets.size() != native.size()) {
        return "Cannot encode the lowering as an override record";
    }
    std::vector<u64> words(10, 0);
    if (graphics) {
        std::memcpy(words.data(), program.data(), 80);
    }
    words.reserve(10 + lowered.instructions.size() * 5 / 4 + 4 + offsets.size());
    for (const u64 instruction : lowered.instructions) {
        if (words.size() % 4 == 0) {
            words.push_back(0); // Maxwell scheduling word
        }
        words.push_back(instruction);
    }
    words.push_back(0);
    if (!offsets.empty()) {
        // AppendNativeAddressMap: lowered index -> byte address inside the code.
        for (const u32 i : offsets) {
            words.push_back(0x50 + u64{i} * 8 + ((u64{i} + 1) / 3) * 8);
        }
        words.push_back(offsets.size());
        words.push_back(AddressMapMagic);
    }
    out.code.resize(words.size() * sizeof(u64));
    std::memcpy(out.code.data(), words.data(), out.code.size());
    out.texture_buffer = lowered.required_texture_buffer;
    return std::nullopt;
}

u64 StageHash(std::span<const u8> code) {
    return Common::CityHash64(reinterpret_cast<const char*>(code.data()), code.size()) ^ HashXor;
}

std::optional<ComputeControl> ReadComputeControl(std::span<const u8> control) {
    if (control.size() < 108) {
        return std::nullopt;
    }
    ComputeControl result;
    for (size_t i = 0; i < 3; ++i) {
        result.workgroup[i] = Get<u32>(control, 88 + i * 4);
    }
    result.shared_memory =
        static_cast<u32>((u64{Get<u32>(control, 100)} + 255) / 256 * 256);
    result.local_memory = Get<u32>(control, 104);
    return result;
}

std::vector<u8> ComputeEntry(const LoweredCode& lowered, const ComputeControl& control,
                             ComputeKey& key) {
    std::vector<u8> out;
    out.reserve(4 + EnvHeaderSize + lowered.code.size() + 16 + key.size());
    Put<u32>(out, 1);
    PutEnvHeader(out, lowered.code.size(), 0, control.local_memory, 0, 80, ComputeStage);
    PutBytes(out, lowered.code);
    for (const u32 size : control.workgroup) {
        Put<u32>(out, size);
    }
    Put<u32>(out, control.shared_memory);
    std::vector<u8> key_bytes;
    Put<u64>(key_bytes, StageHash(lowered.code));
    Put<u32>(key_bytes, control.shared_memory);
    for (const u32 size : control.workgroup) {
        Put<u32>(key_bytes, size);
    }
    std::memcpy(key.data(), key_bytes.data(), key.size());
    PutBytes(out, key_bytes);
    return out;
}

std::vector<u8> GraphicsEnv(u32 stage, const LoweredCode& lowered, std::span<const u8> program) {
    const auto sph = program.first(80);
    const u32 bound = lowered.texture_buffer != ~0U ? lowered.texture_buffer
                                                     : (stage == 0 ? 0U : 2U);
    // Constant buffer bank 1 = the last 256 bytes of the program allocation.
    const auto bank1 = program.last(std::min<size_t>(256, program.size()));
    const size_t cbuf_count = (bank1.size() & ~size_t{3}) / 4;
    const u32 w1 = Get<u32>(sph, 4), w2 = Get<u32>(sph, 8), w3 = Get<u32>(sph, 12);
    const u64 local_memory =
        ((u64{w1} & 0xFFFFFF) | ((u64{w2} & 0xFFFFFF) << 24)) + (u64{w3} & 0xFFFFFF);
    std::vector<u8> out;
    out.reserve(EnvHeaderSize + lowered.code.size() + cbuf_count * 12 + 80);
    PutEnvHeader(out, lowered.code.size(), cbuf_count, static_cast<u32>(local_memory), bound, 0,
                 stage);
    PutBytes(out, lowered.code);
    for (size_t i = 0; i < cbuf_count; ++i) {
        Put<u64>(out, (u64{1} << 32) | (i * 4));
        Put<u32>(out, Get<u32>(bank1, i * 4));
    }
    PutBytes(out, sph);
    return out;
}

GraphicsKey GraphicsEntryKey(u32 stage, const GraphicsKey& template_key, u64 stage_hash,
                             u64 attribute_types) {
    GraphicsKey key = template_key;
    const size_t slot = stage == 0 ? 1 : 5;
    std::memcpy(key.data() + slot * 8, &stage_hash, sizeof(stage_hash));
    if (stage == 0) {
        std::memcpy(key.data() + AttributeTypesOffset, &attribute_types, sizeof(u64));
    }
    return key;
}

RuntimeStats ReadRuntimeStats(const std::vector<std::filesystem::path>& caches) {
    struct Occurrence {
        u32 stage;
        u64 hash;
        const u8* key;
    };
    std::vector<std::vector<u8>> files;
    std::vector<Occurrence> occurrences;
    std::unordered_set<PairHash, PairHasher> seen;
    for (const auto& path : caches) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            continue;
        }
        const auto size = static_cast<size_t>(file.tellg());
        file.seekg(0);
        std::vector<u8> data(size);
        if (!file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size))) {
            continue;
        }
        if (size < 12 || std::memcmp(data.data(), "yuzucach", 8) != 0 ||
            Get<u32>(data, 8) != WarmCacheVersion) {
            continue;
        }
        const std::span<const u8> bytes{data};
        size_t cur = 12;
        struct Env {
            u32 stage;
            size_t begin;
            size_t end;
        };
        std::vector<Env> envs;
        while (cur + 4 <= size) {
            const u32 count = Get<u32>(bytes, cur);
            cur += 4;
            envs.clear();
            bool truncated = count == 0;
            for (u32 i = 0; i < count && !truncated; ++i) {
                if (cur + EnvHeaderSize > size) {
                    truncated = true;
                    break;
                }
                std::array<u64, 6> counts;
                std::memcpy(counts.data(), data.data() + cur, sizeof(counts));
                const u32 stage = Get<u32>(bytes, cur + 48 + 6 * 4);
                const u64 body = counts[0] + counts[1] * 8 + counts[2] * 8 + counts[3] * 12 +
                                 counts[4] * 9 + counts[5] * 12 + TailSize(stage);
                if (body > size - cur - EnvHeaderSize) {
                    truncated = true;
                    break;
                }
                const size_t begin = cur;
                cur += EnvHeaderSize + static_cast<size_t>(body);
                envs.push_back({stage, begin, cur});
            }
            if (truncated) {
                break;
            }
            const size_t key_size = envs[0].stage == ComputeStage ? 24 : GraphicsKeySize;
            if (key_size > size - cur) {
                break;
            }
            const u8* const key = data.data() + cur;
            cur += key_size;
            if (envs[0].stage == ComputeStage) {
                continue;
            }
            std::array<u64, 6> hashes;
            std::memcpy(hashes.data(), key, sizeof(hashes));
            size_t env_index = 0;
            for (size_t slot = 0; slot < 6 && env_index < envs.size(); ++slot) {
                if (hashes[slot] == 0) {
                    continue;
                }
                const Env& env = envs[env_index++];
                if (env.stage != 0 && env.stage != 4) {
                    continue;
                }
                const PairHash id{Hash128(bytes.subspan(env.begin, env.end - env.begin)),
                                  Hash128({key, key_size})};
                if (seen.insert(id).second) {
                    occurrences.push_back({env.stage, hashes[slot], key});
                }
            }
        }
        files.push_back(std::move(data));
    }

    RuntimeStats stats;
    stats.occurrences = occurrences.size();
    for (size_t index = 0; index < 2; ++index) {
        const u32 stage = index == 0 ? 0 : 4;
        struct Count {
            size_t count;
            size_t first;
        };
        std::unordered_map<u128, Count, U128Hasher> counts;
        GraphicsKey normalized;
        for (size_t i = 0; i < occurrences.size(); ++i) {
            if (occurrences[i].stage != stage) {
                continue;
            }
            std::memcpy(normalized.data(), occurrences[i].key, normalized.size());
            std::fill(normalized.begin() + 56, normalized.begin() + 64, u8{0});
            ++counts.try_emplace(Hash128(normalized), Count{0, i}).first->second.count;
        }
        // Counter.most_common: highest count, the first seen among equals.
        const Count* best = nullptr;
        for (const auto& [hash, count] : counts) {
            if (best == nullptr || count.count > best->count ||
                (count.count == best->count && count.first < best->first)) {
                best = &count;
            }
        }
        if (best != nullptr) {
            GraphicsKey key;
            std::memcpy(key.data(), occurrences[best->first].key, key.size());
            std::fill(key.begin() + 56, key.begin() + 64, u8{0});
            std::fill(key.begin(), key.begin() + 48, u8{0});
            stats.templates[index] = key;
        }
    }
    for (const auto& occurrence : occurrences) {
        if (occurrence.stage == 0) {
            u64 attributes;
            std::memcpy(&attributes, occurrence.key + AttributeTypesOffset, sizeof(attributes));
            stats.vertex_attributes[occurrence.hash].push_back(attributes);
        }
    }
    return stats;
}

} // namespace VideoCommon::ShaderWarm
