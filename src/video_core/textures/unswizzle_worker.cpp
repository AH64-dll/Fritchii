// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <cstring>
#include <latch>
#include <memory>
#include <vector>

#include "common/alignment.h"
#include "common/div_ceil.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "video_core/textures/decoders.h"
#include "video_core/textures/unswizzle_worker.h"
#include "video_core/textures/workers.h"

namespace Tegra::Texture {

bool UnswizzleWorker::TryUnswizzle(std::span<u8> output, std::span<const u8> input,
                                   u32 bytes_per_pixel, u32 width, u32 height, u32 depth,
                                   u32 block_height, u32 block_depth, u32 stride_alignment,
                                   bool native_tiling, bool verify, Common::ThreadWorker* workers) {
    const size_t pitch = size_t{width} * bytes_per_pixel;
    const size_t host_bytes = pitch * height;
    if (!native_tiling || depth != 1 || block_depth != 0 ||
        host_bytes < ParallelUnswizzleMinBytes || (pitch & 3) == 0) {
        return false;
    }
    const u32 band_rows = GOB_SIZE_Y << block_height;
    const u32 first_rows = height / 2 / band_rows * band_rows;
    if (first_rows == 0) {
        return false;
    }
    const size_t stride = size_t{Common::AlignUpLog2(width, stride_alignment)} * bytes_per_pixel;
    const size_t band_bytes =
        Common::DivCeil(stride, size_t{GOB_SIZE_X}) * (size_t{GOB_SIZE} << block_height);
    const size_t guest_bytes = Common::DivCeil(height, band_rows) * band_bytes;
    if (guest_bytes > 32 * 1024 * 1024 || guest_bytes > 2 * host_bytes ||
        input.size() < guest_bytes || output.size() < host_bytes) {
        return false;
    }
    owned.resize_destructive(guest_bytes);
    std::memcpy(owned.data(), input.data(), guest_bytes);
    const std::span<const u8> snapshot{owned};
    const size_t guest_split = first_rows / band_rows * band_bytes;
    const size_t host_split = first_rows * pitch;
    std::latch completed{1};
    auto& pool = workers ? *workers : GetThreadWorkers();
    if (!pool.TryQueueWork([=, &completed] {
            SCOPE_EXIT {
                completed.count_down();
            };
            UnswizzleTexture(output.first(host_split), snapshot.first(guest_split), bytes_per_pixel,
                             width, first_rows, 1, block_height, 0, stride_alignment,
                             native_tiling);
        })) {
        return false;
    }
    {
        SCOPE_EXIT {
            completed.wait();
        };
        UnswizzleTexture(output.subspan(host_split, host_bytes - host_split),
                         snapshot.subspan(guest_split), bytes_per_pixel, width, height - first_rows,
                         1, block_height, 0, stride_alignment, native_tiling);
    }
    if (verify) {
        std::vector<u8> reference(host_bytes);
        UnswizzleTexture(reference, snapshot, bytes_per_pixel, width, height, depth, block_height,
                         block_depth, stride_alignment, native_tiling);
        if (std::memcmp(reference.data(), output.data(), host_bytes) != 0) {
            LOG_ERROR(HW_GPU,
                      "Nx2Perf unswizzle_workers: output differs for {}x{} bpp={} native={}", width,
                      height, bytes_per_pixel, native_tiling);
        }
        std::memcpy(output.data(), reference.data(), host_bytes);
    }
    return true;
}

bool TryUnswizzleBands(std::span<u8> output, std::span<const u8> input, u32 bytes_per_pixel,
                       u32 width, u32 height, u32 depth, u32 block_height, u32 block_depth,
                       u32 stride_alignment, bool native_tiling, bool verify,
                       Common::ThreadWorker* workers) {
    constexpr size_t ChunkBytes = 1024 * 1024;
    constexpr u32 MaxHelpers = 8;
    const size_t pitch = size_t{width} * bytes_per_pixel;
    const size_t host_bytes = pitch * height;
    if (depth != 1 || block_depth != 0 || host_bytes < ParallelUnswizzleMinBytes) {
        return false;
    }
    // A band is one row of blocks: contiguous in guest memory, so each chunk of bands is an
    // image of its own starting at the band's offset.
    const u32 band_rows = GOB_SIZE_Y << block_height;
    const u32 bands = Common::DivCeil(height, band_rows);
    const size_t stride = size_t{Common::AlignUpLog2(width, stride_alignment)} * bytes_per_pixel;
    const size_t band_bytes =
        Common::DivCeil(stride, size_t{GOB_SIZE_X}) * (size_t{GOB_SIZE} << block_height);
    const u32 bands_per_chunk =
        (std::max)(static_cast<u32>(ChunkBytes / (pitch * band_rows)), 1U);
    const u32 chunks = Common::DivCeil(bands, bands_per_chunk);
    if (chunks < 2 || input.size() < size_t{bands} * band_bytes || output.size() < host_bytes) {
        return false;
    }
    struct Job {
        std::span<u8> output;
        std::span<const u8> input;
        size_t pitch;
        size_t band_bytes;
        u32 bytes_per_pixel;
        u32 width;
        u32 height;
        u32 band_rows;
        u32 bands_per_chunk;
        u32 block_height;
        u32 stride_alignment;
        u32 chunks;
        bool native_tiling;
        std::atomic<u32> next{};
        std::atomic<u32> done{};

        void Run() {
            for (u32 chunk = next.fetch_add(1, std::memory_order_relaxed); chunk < chunks;
                 chunk = next.fetch_add(1, std::memory_order_relaxed)) {
                const u32 first_band = chunk * bands_per_chunk;
                const u32 first_row = first_band * band_rows;
                const u32 rows = (std::min)(height - first_row, bands_per_chunk * band_rows);
                UnswizzleTexture(output.subspan(first_row * pitch, rows * pitch),
                                 input.subspan(first_band * band_bytes), bytes_per_pixel, width,
                                 rows, 1, block_height, 0, stride_alignment, native_tiling);
                done.fetch_add(1, std::memory_order_release);
                done.notify_all();
            }
        }
    };
    // Shared with the helpers: one that starts after the caller is done finds no chunk left and
    // touches nothing else.
    const auto job = std::make_shared<Job>();
    job->output = output;
    job->input = input;
    job->pitch = pitch;
    job->band_bytes = band_bytes;
    job->bytes_per_pixel = bytes_per_pixel;
    job->width = width;
    job->height = height;
    job->band_rows = band_rows;
    job->bands_per_chunk = bands_per_chunk;
    job->block_height = block_height;
    job->stride_alignment = stride_alignment;
    job->chunks = chunks;
    job->native_tiling = native_tiling;
    auto& pool = workers ? *workers : GetThreadWorkers();
    const u32 helpers = (std::min)(chunks - 1, MaxHelpers);
    for (u32 i = 0; i < helpers; ++i) {
        if (!pool.TryQueueWork([job] { job->Run(); })) {
            break;
        }
    }
    job->Run();
    for (u32 done = job->done.load(std::memory_order_acquire); done < chunks;
         done = job->done.load(std::memory_order_acquire)) {
        job->done.wait(done, std::memory_order_acquire);
    }
    if (verify) {
        std::vector<u8> reference(host_bytes);
        UnswizzleTexture(reference, input, bytes_per_pixel, width, height, depth, block_height,
                         block_depth, stride_alignment, native_tiling);
        if (std::memcmp(reference.data(), output.data(), host_bytes) != 0) {
            LOG_ERROR(HW_GPU, "Unswizzle bands: output differs for {}x{} bpp={} native={}", width,
                      height, bytes_per_pixel, native_tiling);
        }
        std::memcpy(output.data(), reference.data(), host_bytes);
    }
    return true;
}

} // namespace Tegra::Texture
