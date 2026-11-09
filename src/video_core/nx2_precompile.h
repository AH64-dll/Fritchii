// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Fritchii's automatic side of zortos293/Lichii#895: finds a title's precompiled pipeline file
// (precompiled.fork-v<N>.nx2prep, as #895 places it) or a precompile recipe for it, and assembles
// the recipe with the running game's own files when the file is missing or older than the recipe.
// PipelineCache::PreparePrecompiled then builds the file's pipelines in the background.
//
// Recipes are looked for, in order:
//   1. EDEN_NX2_PRECOMPILE_RECIPE=<file>
//   2. the title's shader folder, <data>/shader/<title id>/*.nx2recipe (where the file goes)
//   3. a recipes folder next to the executable, <exe dir>/recipes/*.nx2recipe
// A recipe is used when its header names this title and this build's pipeline cache version.

#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <stop_token>
#include <vector>

#include "common/common_types.h"
#include "core/file_sys/vfs/vfs_types.h"
#include "video_core/nx2_recipe.h"

namespace VideoCommon::Nx2Precompile {

/// EDEN_NX2_PRECOMPILE: 0 (the default) off; 1 prepares a title's precompiled pipelines in the
/// background where the driver keeps a pipeline cache of its own (NVIDIA, AMD's proprietary
/// driver); 2 also on other drivers. Read when first asked, after nx2_switches.txt is applied.
[[nodiscard]] int Mode();

/// EDEN_NX2_PRECOMPILED_FORCE=1 (#895's switch): prepare again although the stamp says it is done.
[[nodiscard]] bool Forced();

/// Threads that extract, assemble and prepare: EDEN_NX2_PRECOMPILE_THREADS (1 to 16), else a
/// quarter of the host's, 1 to 6, like the background pipeline builds.
[[nodiscard]] unsigned Threads();

/// The game's RomFS as the running process reads it (a NUP's update laid over its base, or the
/// patched RomFS). Set by core when an application loads, cleared when it shuts down.
using GameFiles = std::function<std::vector<FileSys::VirtualDir>()>;
void SetGameFiles(u64 title_id, GameFiles files);

/// The running title's files, empty when another title (or none) is running. The call holds the
/// lock core's shutdown clears the provider under, so it never runs into a teardown.
[[nodiscard]] std::vector<FileSys::VirtualDir> OpenGameFiles(u64 title_id);

/// <title_dir>/precompiled.fork-v<cache_version>.nx2prep
[[nodiscard]] std::filesystem::path PrecompiledPath(const std::filesystem::path& title_dir,
                                                    u32 cache_version);

struct Recipe {
    std::filesystem::path path;
    Nx2Recipe::Header header;
    /// Its size and a hash of its bytes: what the precompiled file's note remembers, so the same
    /// recipe found elsewhere (a new build's folder) or copied again is not assembled again.
    u64 size{};
    u64 content_hash{};
};

/// The recipe for this title and cache version, searched as the file comment lists.
[[nodiscard]] std::optional<Recipe> FindRecipe(u64 title_id, const std::filesystem::path& title_dir,
                                               u32 cache_version);

/// The precompiled file to prepare: the title's own file when it is there and was not assembled
/// from an older recipe, else one assembled now from the recipe and the running game's files
/// (written through a temporary file). Empty when there is neither. Takes tens of seconds and
/// about the file's size in memory while it assembles: call it off the UI and GPU threads.
[[nodiscard]] std::filesystem::path EnsurePrecompiledFile(u64 title_id,
                                                          const std::filesystem::path& title_dir,
                                                          u32 cache_version, std::stop_token stop);

} // namespace VideoCommon::Nx2Precompile
