// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>

#include "common/common_types.h"
#include "common/scratch_buffer.h"
#include "common/thread_worker.h"

namespace Tegra::Texture {

inline constexpr size_t ParallelUnswizzleMinBytes = 2 * 1024 * 1024;

class UnswizzleWorker {
public:
    bool TryUnswizzle(std::span<u8> output, std::span<const u8> input, u32 bytes_per_pixel,
                      u32 width, u32 height, u32 depth, u32 block_height, u32 block_depth,
                      u32 stride_alignment, bool native_tiling, bool verify,
                      Common::ThreadWorker* workers = nullptr);

private:
    Common::ScratchBuffer<u8> owned;
};

/// Unswizzles one 2D block-linear level (any bytes per pixel) in chunks of whole block rows,
/// shared between the idle texture transcode workers and the caller, and returns once all of it
/// is written (EDEN_NX2_EXP8 bit 4). The caller takes every chunk no worker has started, so a
/// busy pool costs no more than unswizzling alone. False when the level is not worth splitting;
/// the caller then unswizzles it itself.
bool TryUnswizzleBands(std::span<u8> output, std::span<const u8> input, u32 bytes_per_pixel,
                       u32 width, u32 height, u32 depth, u32 block_height, u32 block_depth,
                       u32 stride_alignment, bool native_tiling, bool verify,
                       Common::ThreadWorker* workers = nullptr);

} // namespace Tegra::Texture
