// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "video_core/pipeline_cache_files.h"

namespace VideoCommon::ShaderWarm {

/// Warm file records, byte for byte what tools/nx2/make_compute_warm.py and
/// tools/nx2/make_graphics_warm.py write (their headers document every field and guess).

/// Warm files and the runtime caches they are built from use the pipeline cache version: the
/// emulator loads a warm file only when it matches, so the two must move together.
constexpr u32 WarmCacheVersion = VideoCommon::VulkanPipelineCacheVersion;
/// Older runtime cache versions with the same record layout, newest first: their records still
/// give the draw state guesses when the current version has no cache yet (just after a bump).
/// Only extend this list when the record layout is unchanged (52 to 54 are identical).
constexpr std::array<u32, 2> CompatibleRuntimeCacheVersions{53, 52};
/// Whether a runtime cache file of `version` can be read for the draw state guesses.
constexpr bool IsReadableRuntimeCacheVersion(u32 version) {
    if (version == WarmCacheVersion) {
        return true;
    }
    for (const u32 compatible : CompatibleRuntimeCacheVersions) {
        if (version == compatible) {
            return true;
        }
    }
    return false;
}
constexpr size_t GraphicsKeySize = 1056;
constexpr size_t AttributeTypesOffset = 112;
/// Per slot most common nonzero vertex attribute type, a default measured on one test title (see
/// warm_defaults.cpp); used when the title has no pipeline cache.
constexpr u64 DefaultAttributeTypes = 0x555555555555f555ULL;

using GraphicsKey = std::array<u8, GraphicsKeySize>;
using ComputeKey = std::array<u8, 24>;

/// A program lowered and laid out like LowerNativeCode (video_core/shader_environment.cpp).
struct LoweredCode {
    std::vector<u8> code;
    u32 texture_buffer = ~0U;
};

/// Lowers one extracted program like `shader-sweep <program> --capture-extent --lower-only
/// --override-out` (graphics: `--format native-header`, the 128-byte header is skipped) and
/// builds the runtime code layout: 10 header words (graphics: the 80-byte SPH, compute: zero),
/// instructions with a zero scheduling word before every 4th word, one zero word, the address
/// map, its count and the magic. Returns the error text when the program cannot be lowered.
std::optional<std::string> LowerProgram(std::span<const u8> program, bool graphics,
                                        LoweredCode& out);

/// NativeShaderHash: CityHash64(code) ^ 0x4f554e4345000002.
u64 StageHash(std::span<const u8> code);

struct ComputeControl {
    std::array<u32, 3> workgroup{};
    u32 shared_memory{};
    u32 local_memory{};
};

/// Workgroup (control bytes 88, 92, 96), shared memory (byte 100 rounded up to 256) and local
/// memory (byte 104). Empty when the control is too short.
std::optional<ComputeControl> ReadComputeControl(std::span<const u8> control);

/// One compute entry (env count, env, code, compute trailer, key); `key` gets the 24-byte key.
std::vector<u8> ComputeEntry(const LoweredCode& lowered, const ComputeControl& control,
                             ComputeKey& key);

/// Env record of a single-stage graphics entry (stage 0 vertex or 4 fragment), without the
/// leading env count and the key. `program` is the extracted allocation (header + code).
std::vector<u8> GraphicsEnv(u32 stage, const LoweredCode& lowered, std::span<const u8> program);

/// The key of a graphics entry: `template_key` with this stage's hash slot (vertex 1,
/// fragment 5) and, for vertex, the attribute types.
GraphicsKey GraphicsEntryKey(u32 stage, const GraphicsKey& template_key, u64 stage_hash,
                             u64 attribute_types);

/// Draw state statistics from runtime pipeline caches recorded by the game.
struct RuntimeStats {
    /// Most common FixedPipelineState per stage (index 0 vertex, 1 fragment), hash slots and
    /// color output types zeroed. Empty when the caches have no record of that stage.
    std::array<std::optional<GraphicsKey>, 2> templates;
    /// Vertex stage hash -> attribute types (key bytes 112-119) of each unique occurrence.
    std::unordered_map<u64, std::vector<u64>> vertex_attributes;
    size_t occurrences{};
};

/// Reads v53 pipeline caches (yuzucach) like make_graphics_warm.py's runtime_records().
RuntimeStats ReadRuntimeStats(const std::vector<std::filesystem::path>& caches);

/// Built-in FixedPipelineState guesses (warm_defaults.cpp), hash slots zero.
GraphicsKey DefaultTemplate(bool vertex);

} // namespace VideoCommon::ShaderWarm
