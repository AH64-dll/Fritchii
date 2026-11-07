// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <stop_token>
#include <string>
#include <vector>

#include "common/common_types.h"

namespace Core {
class System;
}

namespace VideoCommon::ShaderWarm {

/// Where the warm-up files of a title live: next to its pipeline cache
/// (<shader dir>/<title id>/warm-compute.bin and warm-graphics.bin). The emulator loads them at
/// boot when they exist; EDEN_NX2_COMPUTE_WARM / EDEN_NX2_GRAPHICS_WARM still override.
std::filesystem::path ComputeWarmPath(u64 title_id);
std::filesystem::path GraphicsWarmPath(u64 title_id);

/// "Already warmed" marker of a warm file (same folder, extension .done: warm-compute.done),
/// written by the emulator after a warm-up ran to the end. It holds the warm file's size and
/// last-write time, its entry count, the GPU driver version and device ID and the pipeline
/// cache version; a matching marker skips that warm-up at boot.
std::filesystem::path WarmMarkerPath(const std::filesystem::path& warm_file);

/// Entry count of a warm file, written by the builder next to it (extension .count): the entry
/// count and the warm file's size. The file format has no count, so the warm-up shows exact
/// progress only with this; files from the offline tools have none.
std::filesystem::path WarmCountPath(const std::filesystem::path& warm_file);

/// The entry count recorded for `warm_file`, or 0 when there is none or it was written for a
/// file of another size.
u64 ReadWarmCount(const std::filesystem::path& warm_file);

/// Removes the title's warm files, their markers and leftover temporary files (rebuild or delete
/// from the UI). Returns false when a file exists but cannot be removed, with the reason in
/// `error` when given.
bool DeleteWarmFiles(u64 title_id, std::string* error = nullptr);

struct Progress {
    std::string stage;   ///< "Reading game files", "Extracting shaders", "Converting", "Writing"
    size_t done{};       ///< Items done in this stage
    size_t total{};      ///< Items in this stage, 0 when unknown
};

struct Result {
    bool ok{};
    std::string error;           ///< Empty when ok
    size_t compute_programs{};   ///< Entries written to the compute file
    size_t vertex_programs{};    ///< Vertex entries written to the graphics file
    size_t fragment_programs{};  ///< Fragment entries written to the graphics file
    size_t skipped{};            ///< Programs that failed to extract or convert
};

/// Builds both warm-up files for the game at `game_path` (any format the loader opens, Switch 2
/// .nup first), from the user's own copy and keys, with no external tools. Runs on the calling
/// thread; `progress` is called from it; stops early when `stop` is requested. The game's
/// existing pipeline cache, when present, is used for the draw state guesses; otherwise built-in
/// defaults are used.
Result BuildWarmFiles(Core::System& system, const std::filesystem::path& game_path, u64 title_id,
                      const std::function<void(const Progress&)>& progress,
                      std::stop_token stop);

/// Overrides for BuildWarmFilesWith (tests and tools).
struct BuildOptions {
    /// Folder for warm-compute.bin and warm-graphics.bin; empty: <shader dir>/<title id>.
    std::filesystem::path output_dir;
    /// Pipeline caches used for the draw state guesses. Unset: the title's own pipeline cache
    /// when it exists, else the built-in defaults. Set but empty: always the built-in defaults.
    std::optional<std::vector<std::filesystem::path>> runtime_caches;
    /// Conversion threads, 0: a quarter of the CPU threads, 2 to 6.
    size_t threads{};
};

/// BuildWarmFiles with explicit output folder and draw state sources.
Result BuildWarmFilesWith(Core::System& system, const std::filesystem::path& game_path,
                          u64 title_id, const BuildOptions& options,
                          const std::function<void(const Progress&)>& progress,
                          std::stop_token stop);

} // namespace VideoCommon::ShaderWarm
