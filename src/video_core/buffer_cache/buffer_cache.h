// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "video_core/buffer_cache/uniform_cache_history.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>
#include <fmt/ranges.h>

#include "common/range_sets.inc"
#include "video_core/buffer_cache/buffer_cache_base.h"
#include "video_core/buffer_cache/multi_range_refresh.h"
#include "video_core/buffer_cache/uniform_stream_reuse.h"
#include "video_core/guest_memory.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_bindcache.h"
#include "video_core/nx2_cmdpage_diag.h"
#include "video_core/nx2_perf.h"
#include "video_core/nx2_profile.h"
#include "video_core/texture_cache/util.h"

namespace VideoCommon {

using Core::DEVICE_PAGESIZE;

template <class P>
BufferCache<P>::BufferCache(Tegra::MaxwellDeviceMemoryManager& device_memory_, Runtime& runtime_)
    : runtime{runtime_}, device_memory{device_memory_}, memory_tracker{device_memory} {
    // A new runtime has a new scheduler (its ticks restart) and a new stream buffer: whatever the
    // stream uniform reuse remembers refers to the old ones. Before the early return below.
    VideoCommon::ResetUniformStreamReuse<Runtime>();
    // Ensure the first slot is used for the null buffer
    void(slot_buffers.insert(runtime, NullBufferParams{}));
    gpu_modified_ranges.Clear();
    inline_buffer_id = NULL_BUFFER_ID;
#ifdef YUZU_LEGACY
    immediately_free = (Settings::values.vram_usage_mode.GetValue() == Settings::VramUsageMode::Aggressive);
#endif
    if (!runtime.CanReportMemoryUsage()) {
        minimum_memory = DEFAULT_EXPECTED_MEMORY;
        critical_memory = DEFAULT_CRITICAL_MEMORY;
        return;
    }

    const s64 device_local_memory = static_cast<s64>(runtime.GetDeviceLocalMemory());
    const s64 min_spacing_expected = device_local_memory - 1_GiB;
    const s64 min_spacing_critical = device_local_memory - 512_MiB;
    const s64 mem_threshold = (std::min)(device_local_memory, TARGET_THRESHOLD);
    const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
    const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
    minimum_memory = static_cast<u64>(
        (std::max)((std::min)(device_local_memory - min_vacancy_expected, min_spacing_expected),
                 DEFAULT_EXPECTED_MEMORY));
    critical_memory = static_cast<u64>(
        (std::max)((std::min)(device_local_memory - min_vacancy_critical, min_spacing_critical),
                 DEFAULT_CRITICAL_MEMORY));
}

template <class P>
BufferCache<P>::~BufferCache() = default;

template <class P>
void BufferCache<P>::RunGarbageCollector(bool aggressive_gc) {
    const u64 ticks_to_destroy = aggressive_gc ? 60 : 120;
    int num_iterations = aggressive_gc ? 64 : 32;
    // Their GPU-written data is downloaded first, all with one wait for the GPU: one wait per
    // buffer stalled the GPU thread every frame on title 1 (an 8 GB GPU near its budget, no
    // per-fence downloads keeping the buffers clean), ~2.5% of its time.
    boost::container::small_vector<BufferId, 64> evicted;
    lru_cache.ForEachItemBelow(frame_tick - ticks_to_destroy, [&](BufferId buffer_id) {
        if (num_iterations == 0) {
            return true;
        }
        --num_iterations;
        // Below the critical threshold, buffers holding GPU-written data stay: evicting one
        // means downloading it first. The thresholds compare the whole device's use (mostly
        // textures) with fixed headroom, so on an 8 GB GPU the collector ran every frame once a
        // title 1 course was loaded, and the downloads were ~2.7% of the GPU thread.
        const Buffer& buffer = slot_buffers[buffer_id];
        if (!aggressive_gc && gpu_modified_ranges.Intersects(buffer.CpuAddr(), buffer.SizeBytes())) {
            return false;
        }
        evicted.push_back(buffer_id);
        return false;
    });
    if constexpr (USE_MEMORY_MAPS) {
        DownloadBuffersMemory(evicted);
    } else {
        for (const BufferId buffer_id : evicted) {
            DownloadBufferMemory(slot_buffers[buffer_id]);
        }
    }
    for (const BufferId buffer_id : evicted) {
        DeleteBuffer(buffer_id);
    }
}

template <class P>
void BufferCache<P>::TickFrame() {
    // Homebrew console apps don't create or bind any channels, so this will be nullptr.
    if (!channel_state) {
        return;
    }
    runtime.TickFrame(slot_buffers);
    native_indirect_uniforms_this_frame.clear();
    bank_pages_this_frame.clear();
    Nx2BufCache::Report();
    if (frame_tick % 256 == 0) {
        // Hot pages that stopped changing (or stopped being used) are protected again.
        for (auto it = hot_pages.begin(); it != hot_pages.end();) {
            if (it->second->changed_frame + 256 < frame_tick) {
                memory_tracker.MarkRegionHot(it->first, DEVICE_PAGESIZE, false);
                Nx2BufCache::Note(Nx2BufCache::Stat::HotDemote);
                it = hot_pages.erase(it);
            } else {
                ++it;
            }
        }
        if (page_heat.size() > 16384) {
            page_heat.clear();
        }
    }
    if (frame_tick % 600 == 0 && !uploaded_buffer_ranges.empty()) {
        for (const auto& [address, size] : uploaded_buffer_ranges) {
            device_memory.HintCold(address, size);
        }
        uploaded_buffer_ranges.clear();
    }
    if (static u64 frames{}; ++frames % 600 == 0) {
        u64 total{};
        size_t count{};
        std::vector<std::pair<u64, VAddr>> largest;
        for (const auto [buffer_id, buffer] : slot_buffers) {
            total += buffer->SizeBytes();
            ++count;
            largest.emplace_back(buffer->SizeBytes(), buffer->CpuAddr());
        }
        std::partial_sort(largest.begin(), largest.begin() + std::min<size_t>(6, largest.size()),
                          largest.end(), std::greater<>{});
        LOG_INFO(HW_GPU, "Buffer cache: {} buffers {} MiB; largest {}", count, total >> 20,
                 fmt::join(largest.begin(), largest.begin() + std::min<size_t>(6, largest.size()), ", "));
    }

    // Calculate hits and shots and move hit bits to the right
    const u32 hits = std::reduce(channel_state->uniform_cache_hits.begin(),
                                 channel_state->uniform_cache_hits.end());
    const u32 shots = std::reduce(channel_state->uniform_cache_shots.begin(),
                                  channel_state->uniform_cache_shots.end());
    AdvanceUniformCacheHistory(channel_state->uniform_cache_hits);
    AdvanceUniformCacheHistory(channel_state->uniform_cache_shots);

    const bool skip_preferred = hits * 256 < shots * 251;
    channel_state->uniform_buffer_skip_cache_size = skip_preferred ? DEFAULT_SKIP_CACHE_SIZE : 0;

    // If we can obtain the memory info, use it instead of the estimate.
    if (runtime.CanReportMemoryUsage()) {
        total_used_memory = runtime.GetDeviceMemoryUsage();
    }
    if (gpu_memory != nullptr && gpu_memory->UsesNativeTiling()) {
        // Switch 2 titles fill the device with textures (their collector watches the budget);
        // title 1's buffers were ~200 MiB of an 8 GB GPU's ~6.8 GB in use, yet this collector
        // ran aggressively every frame on the device-wide figure, evicting up to 64 buffers (and
        // downloading the GPU-written ones, ~5% of the GPU thread) that were recreated later.
        // Collect when the buffers themselves are large. At 1 GiB it held geometry of scenes
        // long gone: title 1's title screen grew to ~0.9 GB of buffers in ten minutes.
        if (buffer_bytes >= 512_MiB) {
            RunGarbageCollector(buffer_bytes >= 1_GiB);
        }
    } else if (total_used_memory >= minimum_memory) {
        RunGarbageCollector(total_used_memory >= critical_memory);
    }
    ++frame_tick;
    delayed_destruction_ring.Tick();

    for (auto& buffer : async_buffers_death_ring) {
        runtime.FreeDeferredStagingBuffer(buffer);
    }
    async_buffers_death_ring.clear();
}

template <class P>
void BufferCache<P>::UnmapGPUMemory(size_t as_id, GPUVAddr gpu_addr, size_t size) {
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        virtual_ranges.Unmap(as_id, gpu_addr, size);
    }
}

template <class P>
void BufferCache<P>::WriteMemory(DAddr device_addr, u64 size, bool keep_page_gpu_data) {
    Nx2BufCache::Note(Nx2BufCache::Stat::WriteMemoryCalls);
    Nx2NoteListHostWrite(device_addr, size);
    // The written range lies within its pages: when they hold no GPU-written data, both lookups
    // below find nothing, so one lookup answers them.
    if (keep_page_gpu_data) {
        const DAddr page_begin = Common::AlignDown(device_addr, DEVICE_PAGESIZE);
        const DAddr page_end = Common::AlignUp(device_addr + size, DEVICE_PAGESIZE);
        if (!IsRegionGpuModified(page_begin, page_end - page_begin)) {
            if (!Nx2Perf::Verify()) {
                // Pages all CPU-dirty already: the mark below would change no tracker state, and
                // no SYNC_SKIP entry that is still valid covers them (EDEN_NX2_DIRTY_WRITE_SKIP).
                if (Nx2BufCache::g_dirty_write_skip &&
                    memory_tracker.IsRegionFullyCpuModified(device_addr, size)) {
                    Nx2BufCache::Note(Nx2BufCache::Stat::DirtyWriteSkip);
                    MarkVertexBuffersStale(device_addr, size);
                    return;
                }
                memory_tracker.MarkRegionAsCpuModified(device_addr, size);
                NoteCpuDirty(device_addr, size);
                MarkVertexBuffersStale(device_addr, size);
                return;
            }
            static std::atomic<u32> logged{};
            if (IsRegionGpuModified(device_addr, size) &&
                logged.fetch_add(1, std::memory_order_relaxed) < 64) {
                LOG_ERROR(HW_GPU, "Nx2Perf texture-writes: GPU-written data in {:#x}+{:#x} outside its pages",
                          device_addr, size);
            }
        }
    }
    if (IsRegionGpuModified(device_addr, size)) {
        ClearDownload(device_addr, size);
        gpu_modified_ranges.Subtract(device_addr, size);
    }
    // The tracker marks whole pages CPU-modified, which drops their GPU-modified state: GPU-written
    // bytes elsewhere on these pages would never be downloaded, and the page's next upload would
    // send the stale guest copy over them. Hyrule Warriors: Age of Imprisonment writes small
    // structures next to the draw arguments its compute dispatches fill in; the draws then read
    // leftover bytes as vertex counts. Download those bytes first (the CPU's own were subtracted
    // above, so they aren't overwritten).
    const DAddr page_begin = Common::AlignDown(device_addr, DEVICE_PAGESIZE);
    const DAddr page_end = Common::AlignUp(device_addr + size, DEVICE_PAGESIZE);
    if (keep_page_gpu_data && gpu_modified_ranges.Intersects(page_begin, page_end - page_begin)) {
        // Switch 2 titles: upload the CPU's bytes into the buffer instead, as CachedWriteMemory
        // does, so the page stays GPU-owned and nothing waits for the host GPU.
        static const int forced = [] {
            const char* value = std::getenv("EDEN_NX2_INLINE_PAGE_WRITES");
            return value == nullptr ? -1 : (value[0] == '0' ? 0 : 1);
        }();
        const bool inline_writes =
            forced >= 0 ? forced == 1 : Tegra::g_nx2_ampere.load(std::memory_order_relaxed);
        if (inline_writes && IsRegionRegistered(device_addr, size)) {
            tmp_buffer.resize_destructive(size);
            device_memory.ReadBlockUnsafe(device_addr, tmp_buffer.data(), size);
            InlineMemoryImplementation(device_addr, size, tmp_buffer);
            return;
        }
        DownloadMemory(page_begin, page_end - page_begin);
    }
    memory_tracker.MarkRegionAsCpuModified(device_addr, size);
    NoteCpuDirty(device_addr, size);
    MarkVertexBuffersStale(device_addr, size);
}

template <class P>
void BufferCache<P>::CachedWriteMemory(DAddr device_addr, u64 size) {
    const bool is_dirty = IsRegionRegistered(device_addr, size);
    if (!is_dirty) {
        return;
    }
    DAddr aligned_start = Common::AlignDown(device_addr, DEVICE_PAGESIZE);
    DAddr aligned_end = Common::AlignUp(device_addr + size, DEVICE_PAGESIZE);
    if (!IsRegionGpuModified(aligned_start, aligned_end - aligned_start)) {
        WriteMemory(device_addr, size);
        return;
    }

    Nx2NoteListHostWrite(device_addr, size);
    tmp_buffer.resize_destructive(size);
    device_memory.ReadBlockUnsafe(device_addr, tmp_buffer.data(), size);

    InlineMemoryImplementation(device_addr, size, tmp_buffer);
}

template <class P>
bool BufferCache<P>::OnCPUWrite(DAddr device_addr, u64 size) {
    const bool is_dirty = IsRegionRegistered(device_addr, size);
    if (!is_dirty) {
        return false;
    }
    if (memory_tracker.IsRegionGpuModified(device_addr, size)) {
        return true;
    }
    WriteMemory(device_addr, size);
    return false;
}

template <class P>
std::optional<VideoCore::RasterizerDownloadArea> BufferCache<P>::GetFlushArea(DAddr device_addr,
                                                                              u64 size) {
    std::optional<VideoCore::RasterizerDownloadArea> area{};
    area.emplace();
    DAddr device_addr_start_aligned = Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE);
    DAddr device_addr_end_aligned = Common::AlignUp(device_addr + size, Core::DEVICE_PAGESIZE);
    area->start_address = device_addr_start_aligned;
    area->end_address = device_addr_end_aligned;
    if (memory_tracker.IsRegionPreflushable(device_addr, size)) {
        area->preemtive = true;
        return area;
    };
    area->preemtive = !IsRegionGpuModified(device_addr_start_aligned,
                                           device_addr_end_aligned - device_addr_start_aligned);
    // A preflushable mark persists and downloads later GPU writes at every fence. Native titles
    // instead keep read traps and flush on demand.
    const bool preflush = !Tegra::g_nx2_ampere.load(std::memory_order_relaxed);
    if (!area->preemtive) {
        if (Nx2DownloadTrace::g_enabled) [[unlikely]] {
            Nx2DownloadTrace::Marked(device_addr_start_aligned,
                                     device_addr_end_aligned - device_addr_start_aligned, preflush);
        }
        if (preflush) {
            memory_tracker.MarkRegionAsPreflushable(
                device_addr_start_aligned, device_addr_end_aligned - device_addr_start_aligned);
        } else {
            if constexpr (USE_MEMORY_MAPS) {
                RollCpuReadRanges();
                cpu_read_ranges.Add(area->start_address, area->end_address - area->start_address);
                // The GPU writes here are already being copied back (Prefetch): the reader waits
                // for them to land instead of flushing (Nx2Reactive::WaitLanding).
                if (Nx2Reactive::Prefetch() && Nx2Reactive::WaitLanding() &&
                    GpuWritesDownloading(area->start_address,
                                         area->end_address - area->start_address)) {
                    area->wait_landing = true;
                } else {
                    download_cpu_read_ranges = true;
                }
            }
            // The core consumed this trap before GetFlushArea. Forget it so the next GPU write
            // really rearms the core instead of being skipped by the deduplication cache.
            read_trapped_ranges.Subtract(
                device_addr_start_aligned, device_addr_end_aligned - device_addr_start_aligned);
        }
    } else {
        if constexpr (USE_MEMORY_MAPS) {
            // A page read lately stays read (its GPU writes keep being copied back).
            if (Nx2Reactive::Prefetch()) {
                RollCpuReadRanges();
                if (cpu_read_ranges_previous.Intersects(area->start_address,
                                                        area->end_address - area->start_address)) {
                    cpu_read_ranges.Add(area->start_address,
                                        area->end_address - area->start_address);
                }
            }
        }
        // The core keeps this area and has no mark. The next GPU write here must arm the range
        // again, which ends the area and sets the trap.
        read_trapped_ranges.Subtract(device_addr_start_aligned,
                                     device_addr_end_aligned - device_addr_start_aligned);
    }
    return area;
}

template <class P>
void BufferCache<P>::DownloadMemory(DAddr device_addr, u64 size) {
    if constexpr (USE_MEMORY_MAPS) {
        if (download_cpu_read_ranges) {
            download_cpu_read_ranges = false;
            boost::container::small_vector<std::pair<DAddr, u64>, 16> ranges{{device_addr, size}};
            const auto add_read_range = [&](DAddr lower, DAddr upper) {
                gpu_modified_ranges.ForEachInRange(lower, upper - lower,
                                                   [&](DAddr start, DAddr end) {
                                                       ranges.emplace_back(start, end - start);
                                                   });
            };
            cpu_read_ranges.ForEach(add_read_range);
            cpu_read_ranges_previous.ForEach(add_read_range);
            DownloadRangesMemory(ranges);
            return;
        }
    }
    ForEachBufferInRange(device_addr, size, [&](BufferId, Buffer& buffer) {
        const DAddr start = (std::max)(device_addr, buffer.CpuAddr());
        const DAddr end = (std::min)(device_addr + size, buffer.CpuAddr() + buffer.SizeBytes());
        if (start < end) {
            DownloadBufferMemory(buffer, start, end - start);
        }
    });
}

template <class P>
void BufferCache<P>::DownloadRangesMemory(std::span<const std::pair<DAddr, u64>> ranges) {
    constexpr bool has_mapped = requires(Buffer& buffer) { buffer.Mapped(); };
    if constexpr (!USE_MEMORY_MAPS) {
        for (const auto& [device_addr, size] : ranges) {
            DownloadMemory(device_addr, size);
        }
    } else {
        struct Download {
            BufferId buffer_id;
            BufferCopy copy;
        };
        boost::container::small_vector<Download, 16> staged;
        boost::container::small_vector<Download, 16> mapped;
        u64 total_size_bytes = 0;
        for (const auto& [device_addr, size] : ranges) {
            ForEachBufferInRange(device_addr, size, [&](BufferId buffer_id, Buffer& buffer) {
                const DAddr buffer_addr = buffer.CpuAddr();
                const DAddr begin = (std::max)(device_addr, buffer_addr);
                const DAddr end = (std::min)(device_addr + size, buffer_addr + buffer.SizeBytes());
                if (begin >= end) {
                    return;
                }
                bool is_mapped = false;
                if constexpr (has_mapped) {
                    is_mapped = !buffer.Mapped().empty();
                }
                memory_tracker.ForEachDownloadRangeAndClear(
                    begin, end - begin, [&](u64 device_addr_out, u64 range_size) {
                        gpu_modified_ranges.ForEachInRange(
                            device_addr_out, range_size, [&](DAddr start, DAddr range_end) {
                                const BufferCopy copy{
                                    .src_offset = start - buffer_addr,
                                    .dst_offset = total_size_bytes,
                                    .size = range_end - start,
                                };
                                if (is_mapped) {
                                    mapped.push_back({buffer_id, copy});
                                    return;
                                }
                                staged.push_back({buffer_id, copy});
                                total_size_bytes += Common::AlignUp(range_end - start, 64);
                            });
                        ClearDownload(device_addr_out, range_size);
                        gpu_modified_ranges.Subtract(device_addr_out, range_size);
                    });
            });
        }
        if (staged.empty() && mapped.empty()) {
            return;
        }
        const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();
        if (staged.empty()) {
            if constexpr (has_mapped) {
                runtime.FinishForHostRead();
            }
        } else {
            auto staging = runtime.DownloadStagingBuffer(total_size_bytes);
            for (Download& download : staged) {
                download.copy.dst_offset += staging.offset;
                Buffer& buffer = slot_buffers[download.buffer_id];
                buffer.MarkUsage(download.copy.src_offset, download.copy.size);
                runtime.CopyBuffer(staging.buffer, buffer, std::span{&download.copy, 1}, true);
            }
            if constexpr (has_mapped) {
                runtime.FinishForHostRead();
            } else {
                runtime.Finish();
            }
            for (const Download& download : staged) {
                WriteDownloadedMemory(
                    cpu_written,
                    slot_buffers[download.buffer_id].CpuAddr() + download.copy.src_offset,
                    staging.mapped_span.data() + (download.copy.dst_offset - staging.offset),
                    download.copy.size);
            }
        }
        if constexpr (has_mapped) {
            for (const Download& download : mapped) {
                Buffer& buffer = slot_buffers[download.buffer_id];
                WriteDownloadedMemory(cpu_written, buffer.CpuAddr() + download.copy.src_offset,
                                      buffer.Mapped().data() + download.copy.src_offset,
                                      download.copy.size);
            }
        }
    }
}

template <class P>
void BufferCache<P>::ClearDownload(DAddr device_addr, u64 size) {
    for (auto& ranges : pending_download_ranges) {
        ranges.Subtract(device_addr, size);
    }
    uncommitted_gpu_modified_ranges.Subtract(device_addr, size);
    for (auto& interval_set : committed_gpu_modified_ranges) {
        interval_set.Subtract(device_addr, size);
    }
}

template <class P>
bool BufferCache<P>::DMACopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount) {
    const std::optional<DAddr> cpu_src_address = gpu_memory->GpuToCpuAddress(src_address);
    const std::optional<DAddr> cpu_dest_address = gpu_memory->GpuToCpuAddress(dest_address);
    if (!cpu_src_address || !cpu_dest_address) {
        return false;
    }
    const bool source_dirty = IsRegionRegistered(*cpu_src_address, amount);
    const bool dest_dirty = IsRegionRegistered(*cpu_dest_address, amount);
    if (!source_dirty && !dest_dirty) {
        return false;
    }

    ClearDownload(*cpu_dest_address, amount);
    MarkVertexBuffersStale(*cpu_dest_address, amount);

    BufferId buffer_a;
    BufferId buffer_b;
    do {
        channel_state->has_deleted_buffers = false;
        buffer_a = FindBuffer(*cpu_src_address, static_cast<u32>(amount), false);
        buffer_b = FindBuffer(*cpu_dest_address, static_cast<u32>(amount), false);
    } while (channel_state->has_deleted_buffers);
    auto& src_buffer = slot_buffers[buffer_a];
    auto& dest_buffer = slot_buffers[buffer_b];
    SynchronizeBuffer(src_buffer, *cpu_src_address, static_cast<u32>(amount));
    SynchronizeBuffer(dest_buffer, *cpu_dest_address, static_cast<u32>(amount));
    std::array copies{BufferCopy{
        .src_offset = src_buffer.Offset(*cpu_src_address),
        .dst_offset = dest_buffer.Offset(*cpu_dest_address),
        .size = amount,
    }};

    boost::container::small_vector<std::pair<DAddr, size_t>, 4> tmp_intervals;
    auto mirror = [&](DAddr base_address, DAddr base_address_end) {
        const u64 size = base_address_end - base_address;
        const DAddr diff = base_address - *cpu_src_address;
        const DAddr new_base_address = *cpu_dest_address + diff;
        tmp_intervals.push_back({new_base_address, size});
        QueueFenceDownload(new_base_address, size);
    };
    gpu_modified_ranges.ForEachInRange(*cpu_src_address, amount, mirror);
    // This subtraction in this order is important for overlapping copies.
    gpu_modified_ranges.Subtract(*cpu_dest_address, amount);
    const bool has_new_downloads = tmp_intervals.size() != 0;
    for (const auto& pair : tmp_intervals) {
        gpu_modified_ranges.Add(pair.first, pair.second);
        gpu_modified_serial.fetch_add(1, std::memory_order_relaxed);
        if (!Tegra::g_nx2_ampere.load(std::memory_order_relaxed)) {
            nx2_gpu_ranges_unmarked = true;
        } else {
            Tegra::Nx2MarkGpuWritten(pair.first, pair.second);
            Nx2NoteListWrite(buffer_b, pair.first, pair.second);
        }
    }
    const auto& copy = copies[0];
    src_buffer.MarkUsage(copy.src_offset, copy.size);
    dest_buffer.MarkUsage(copy.dst_offset, copy.size);
    InvalidateBankShadows(*cpu_dest_address, amount);
    InvalidateHotShadows(*cpu_dest_address, amount);
    runtime.CopyBuffer(dest_buffer, src_buffer, copies, true);
    if (has_new_downloads) {
        memory_tracker.MarkRegionAsGpuModified(*cpu_dest_address, amount);
        DemoteHotPages(*cpu_dest_address, amount);
        const bool should_sync = Settings::IsGPUFenceBehaviorBalanced() || Settings::IsGPUFenceBehaviorAccurate();
        if (should_sync) {
            runtime.Finish();
        }
    }

    Tegra::Memory::DeviceGuestMemoryScoped<u8, Tegra::Memory::GuestMemoryFlags::UnsafeReadWrite>
        tmp(device_memory, *cpu_src_address, amount, &tmp_buffer);
    tmp.SetAddressAndSize(*cpu_dest_address, amount);
    return true;
}

template <class P>
bool BufferCache<P>::DMAClear(GPUVAddr dst_address, u64 amount, u32 value) {
    const std::optional<DAddr> cpu_dst_address = gpu_memory->GpuToCpuAddress(dst_address);
    if (!cpu_dst_address) {
        return false;
    }
    const bool dest_dirty = IsRegionRegistered(*cpu_dst_address, amount);
    if (!dest_dirty) {
        return false;
    }

    const size_t size = amount * sizeof(u32);
    ClearDownload(*cpu_dst_address, size);
    gpu_modified_ranges.Subtract(*cpu_dst_address, size);

    const BufferId buffer = FindBuffer(*cpu_dst_address, static_cast<u32>(size), false);
    Buffer& dest_buffer = slot_buffers[buffer];
    const u32 offset = dest_buffer.Offset(*cpu_dst_address);
    InvalidateBankShadows(*cpu_dst_address, size);
    InvalidateHotShadows(*cpu_dst_address, size);
    runtime.ClearBuffer(dest_buffer, offset, size, value);
    dest_buffer.MarkUsage(offset, size);
    return true;
}

template <class P>
std::pair<typename P::Buffer*, u32> BufferCache<P>::ObtainBuffer(GPUVAddr gpu_addr, u32 size,
                                                                 ObtainBufferSynchronize sync_info,
                                                                 ObtainBufferOperation post_op) {
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (!device_addr) {
        return {&slot_buffers[NULL_BUFFER_ID], 0};
    }
    return ObtainCPUBuffer(*device_addr, size, sync_info, post_op);
}

template <class P>
std::pair<typename P::Buffer*, u32> BufferCache<P>::ObtainCPUBuffer(
    DAddr device_addr, u32 size, ObtainBufferSynchronize sync_info, ObtainBufferOperation post_op) {
    const BufferId buffer_id = FindBuffer(device_addr, size, false);
    Buffer& buffer = slot_buffers[buffer_id];

    // synchronize op
    switch (sync_info) {
    case ObtainBufferSynchronize::FullSynchronize:
        SynchronizeBuffer(buffer, device_addr, size);
        break;
    default:
        break;
    }

    switch (post_op) {
    case ObtainBufferOperation::MarkAsWritten:
        MarkWrittenBuffer(buffer_id, device_addr, size);
        buffer.MarkUsage(buffer.Offset(device_addr), size);
        break;
    case ObtainBufferOperation::DiscardWrite: {
        ClearDownload(device_addr, size);
        gpu_modified_ranges.Subtract(device_addr, size);
        break;
    }
    default:
        break;
    }

    return {&buffer, buffer.Offset(device_addr)};
}

template <class P>
void BufferCache<P>::BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr,
                                               u32 size) {
    UniformBindMemo& memo = uniform_bind_memo[stage][index];
    Binding& current = channel_state->uniform_buffers[stage][index];
    // EDEN_NX2_UBO_REBIND: the binding this slot's last bind set, bound again with the same GPU
    // range while the mappings (map serial) and the set of buffers (binding_epoch: DeleteBuffer
    // resets bindings to deleted buffers, any buffer creation or join bumps it) are unchanged.
    // The translation gives the same device range again and the buffer it resolved to is still
    // what FindBuffer returns: keep it, with its buffer_id, instead of translating again here and
    // looking the buffer up again in UpdateUniformBuffers. The contents are read at the draw as
    // before (BindHostGraphicsUniformBuffer). FindBuffer waits for the buffer's writes with
    // accurate GPU fences: keep that.
    if (Nx2BufCache::g_ubo_rebind && memo.epoch == binding_epoch && memo.gpu_addr == gpu_addr &&
        memo.size == size && memo.channel == static_cast<const void*>(channel_state) &&
        current.device_addr == memo.device_addr && current.size == size &&
        memo.as_id == gpu_memory->GetID() && memo.map_serial == gpu_memory->MapSerial() &&
        !Settings::IsGPUFenceBehaviorAccurate()) {
        Nx2BufCache::Note(Nx2BufCache::Stat::UboRebindKept);
        // As a bind UpdateUniformBuffers resolves again: OpenGL binds the host buffer again.
        if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
            channel_state->dirty_uniform_buffers[stage] |= 1U << index;
        }
        return;
    }
    // The serial before the translation: a map in between makes the memo miss next time.
    const u64 map_serial = gpu_memory->MapSerial();
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (!device_addr) {
        current = NULL_BINDING;
        memo.epoch = 0;
        return;
    }
    current = Binding{
        .device_addr = *device_addr,
        .size = size,
        .buffer_id = BufferId{},
    };
    memo = UniformBindMemo{
        .epoch = binding_epoch,
        .map_serial = map_serial,
        .as_id = gpu_memory->GetID(),
        .channel = channel_state,
        .gpu_addr = gpu_addr,
        .device_addr = *device_addr,
        .size = size,
    };
}

template <class P>
void BufferCache<P>::DisableGraphicsUniformBuffer(size_t stage, u32 index) {
    uniform_bind_memo[stage][index].epoch = 0;
    auto& current = channel_state->uniform_buffers[stage][index];
    if (current.device_addr == 0 && current.size == 0 && current.buffer_id == NULL_BUFFER_ID)
        [[likely]] {
        return;
    }
    current = NULL_BINDING;
}

template <class P>
void BufferCache<P>::UpdateGraphicsBuffers(bool is_indexed) {
    do {
        channel_state->has_deleted_buffers = false;
        DoUpdateGraphicsBuffers(is_indexed);
    } while (channel_state->has_deleted_buffers);
}

template <class P>
void BufferCache<P>::UpdateComputeBuffers() {
    do {
        channel_state->has_deleted_buffers = false;
        DoUpdateComputeBuffers();
    } while (channel_state->has_deleted_buffers);
}

template <class P>
void BufferCache<P>::BindHostGeometryBuffers(bool is_indexed) {
    if (is_indexed) {
        BindHostIndexBuffer();
    } else if constexpr (!HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT) {
        const auto& draw_state = maxwell3d->draw_manager.draw_state;
        if (draw_state.topology == Maxwell::PrimitiveTopology::Quads ||
            draw_state.topology == Maxwell::PrimitiveTopology::QuadStrip) {
            runtime.BindQuadIndexBuffer(draw_state.topology, draw_state.vertex_buffer.first,
                                        draw_state.vertex_buffer.count);
        }
    }
    BindHostVertexBuffers();
    BindHostTransformFeedbackBuffers();
    if (current_draw_indirect) {
        BindHostDrawIndirectBuffers();
    }
}

template <class P>
void BufferCache<P>::BindHostStageBuffers(size_t stage) {
    BindHostGraphicsUniformBuffers(stage);
    if (channel_state->enabled_storage_buffers[stage] != 0) {
        BindHostGraphicsStorageBuffers(stage);
    }
    if (channel_state->enabled_texture_buffers[stage] != 0) {
        BindHostGraphicsTextureBuffers(stage);
    }
}

template <class P>
void BufferCache<P>::BindHostComputeBuffers() {
    BindHostComputeUniformBuffers();
    if (channel_state->enabled_compute_storage_buffers != 0) {
        BindHostComputeStorageBuffers();
    }
    if (channel_state->enabled_compute_texture_buffers != 0) {
        BindHostComputeTextureBuffers();
    }
    if (any_buffer_uploaded) {
        runtime.PostCopyBarrier();
        any_buffer_uploaded = false;
    }
}

template <class P>
void BufferCache<P>::SetUniformBuffersState(const std::array<u32, NUM_STAGES>& mask,
                                            const UniformBufferSizes* sizes) {
    if (channel_state->enabled_uniform_buffer_masks == mask &&
        channel_state->uniform_buffer_sizes == sizes) [[likely]] {
        return;
    }
    if (channel_state->enabled_uniform_buffer_masks != mask) {
        channel_state->fast_bound_uniform_buffers.fill(0);
        if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
            channel_state->dirty_uniform_buffers.fill(~u32{0});
            channel_state->uniform_buffer_binding_sizes.fill({});
        }
        channel_state->enabled_uniform_buffer_masks = mask;
    }
    channel_state->uniform_buffer_sizes = sizes;
}

template <class P>
void BufferCache<P>::SetComputeUniformBufferState(u32 mask,
                                                  const ComputeUniformBufferSizes* sizes) {
    channel_state->enabled_compute_uniform_buffer_mask = mask;
    channel_state->compute_uniform_buffer_sizes = sizes;
}

template <class P>
void BufferCache<P>::UnbindGraphicsStorageBuffers(size_t stage) {
    if (channel_state->enabled_storage_buffers[stage] == 0) {
        return;
    }
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers()) {
            channel_state->total_graphics_storage_buffers -=
                static_cast<u32>(std::popcount(channel_state->enabled_storage_buffers[stage]));
        }
    }
    channel_state->enabled_storage_buffers[stage] = 0;
    channel_state->written_storage_buffers[stage] = 0;
}

template <class P>
bool BufferCache<P>::BindGraphicsStorageBuffer(size_t stage, size_t ssbo_index, u32 cbuf_index,
                                               u32 cbuf_offset, bool is_written,
                                               u32 indirect_offset) {
    if (stage >= NUM_STAGES || ssbo_index >= NUM_STORAGE_BUFFERS) {
        LOG_ERROR(HW_GPU, "Invalid graphics storage buffer binding: stage {}, index {}",
                  stage, ssbo_index);
        return false;
    }
    const bool already_enabled =
        ((channel_state->enabled_storage_buffers[stage] >> ssbo_index) & 1U) != 0;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            const u32 max_bindings = runtime.GetMaxDynamicStorageBuffers();
            if (channel_state->total_graphics_storage_buffers >= max_bindings) {
                LOG_WARNING(HW_GPU,
                            "Skipping graphics storage buffer {} due to driver limit {}",
                            ssbo_index, max_bindings);
                return false;
            }
        }
    }
    channel_state->enabled_storage_buffers[stage] |= u64{1} << ssbo_index;
    channel_state->written_storage_buffers[stage] |= u64{is_written ? 1U : 0U} << ssbo_index;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            ++channel_state->total_graphics_storage_buffers;
        }
    }

    const auto& cbufs = maxwell3d->state.shader_stages[stage];
    GPUVAddr ssbo_addr = cbufs.const_buffers[cbuf_index].address + cbuf_offset;
    GPUVAddr holder{};
    if (indirect_offset != ~0U) {
        // The pointer is an entry of a table the constant buffer points to.
        holder = gpu_memory->Read<u64>(ssbo_addr);
        if (holder == 0) {
            channel_state->storage_buffers[stage][ssbo_index] = NULL_BINDING;
            return false;
        }
        ssbo_addr = holder + indirect_offset;
    }
    channel_state->storage_buffers[stage][ssbo_index] =
        StorageBufferBinding(ssbo_addr, cbuf_index, is_written,
                             maxwell3d->GetNativeMacroEngine().native_turing ||
                                 indirect_offset != ~0U,
                             indirect_offset != ~0U);
    if (indirect_offset != ~0U) {
        TraceTablePointer(false, stage, ssbo_index, cbuf_index, cbuf_offset, indirect_offset, holder,
                          channel_state->storage_buffers[stage][ssbo_index]);
    }
    return (channel_state->storage_buffers[stage][ssbo_index].buffer_id != NULL_BUFFER_ID);
}

template <class P>
void BufferCache<P>::UnbindGraphicsTextureBuffers(size_t stage) {
    if (channel_state->enabled_texture_buffers[stage] == 0) {
        return;
    }
    channel_state->enabled_texture_buffers[stage] = 0;
    channel_state->written_texture_buffers[stage] = 0;
    channel_state->image_texture_buffers[stage] = 0;
}

template <class P>
void BufferCache<P>::BindGraphicsTextureBuffer(size_t stage, size_t tbo_index, GPUVAddr gpu_addr,
                                               u32 size, PixelFormat format, bool is_written,
                                               bool is_image) {
    channel_state->enabled_texture_buffers[stage] |= 1U << tbo_index;
    channel_state->written_texture_buffers[stage] |= (is_written ? 1U : 0U) << tbo_index;
    if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
        channel_state->image_texture_buffers[stage] |= (is_image ? 1U : 0U) << tbo_index;
    }
    channel_state->texture_buffers[stage][tbo_index] =
        GetTextureBufferBinding(gpu_addr, size, format);
}

template <class P>
void BufferCache<P>::UnbindComputeStorageBuffers() {
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers()) {
            channel_state->total_compute_storage_buffers -=
                static_cast<u32>(std::popcount(channel_state->enabled_compute_storage_buffers));
        }
    }
    channel_state->enabled_compute_storage_buffers = 0;
    channel_state->written_compute_storage_buffers = 0;
    channel_state->native_compute_storage_buffers = 0;
    channel_state->image_compute_texture_buffers = 0;
}

template <class P>
void BufferCache<P>::BindComputeStorageBuffer(size_t ssbo_index, u32 cbuf_index, u32 cbuf_offset,
                                              bool is_written, u32 indirect_offset,
                                              const Shader::StorageWriteExtent& write_extent) {
    if (ssbo_index >= channel_state->compute_storage_buffers.size()) [[unlikely]] {
        LOG_ERROR(HW_GPU, "Storage buffer index {} exceeds maximum storage buffer count",
                  ssbo_index);
        return;
    }
    const bool already_enabled =
        ((channel_state->enabled_compute_storage_buffers >> ssbo_index) & 1U) != 0;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            const u32 max_bindings = runtime.GetMaxDynamicStorageBuffers();
            if (channel_state->total_compute_storage_buffers >= max_bindings) {
                LOG_WARNING(HW_GPU,
                            "Skipping compute storage buffer {} due to driver limit {}",
                            ssbo_index, max_bindings);
                return;
            }
        }
    }
    channel_state->enabled_compute_storage_buffers |= u64{1} << ssbo_index;
    channel_state->written_compute_storage_buffers |= u64{is_written ? 1U : 0U} << ssbo_index;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            ++channel_state->total_compute_storage_buffers;
        }
    }

    const auto& launch_desc = kepler_compute->launch_description;
    if (((launch_desc.const_buffer_enable_mask >> cbuf_index) & 1) == 0) {
        LOG_WARNING(HW_GPU, "Skipped binding SSBO: cbuf index {} is not enabled", cbuf_index);
        return;
    }
    ASSERT(((launch_desc.const_buffer_enable_mask >> cbuf_index) & 1) != 0);

    const auto& cbufs = launch_desc.const_buffer_config;
    GPUVAddr ssbo_addr = cbufs[cbuf_index].Address() + cbuf_offset;
    GPUVAddr holder{};
    const bool table_pointer = indirect_offset != ~0U;
    if (table_pointer) {
        holder = gpu_memory->Read<u64>(ssbo_addr);
        ssbo_addr = holder + indirect_offset;
    }
    const bool native_pointer = kepler_compute->IsNativeMode();
    channel_state->native_compute_storage_buffers =
        (channel_state->native_compute_storage_buffers & ~(u64{1} << ssbo_index)) |
        (u64{native_pointer ? 1U : 0U} << ssbo_index);
    const Binding binding =
        table_pointer && holder == 0
            ? NULL_BINDING
            : StorageBufferBinding(ssbo_addr, cbuf_index, is_written, native_pointer,
                                   table_pointer);
    channel_state->compute_storage_buffers[ssbo_index] = binding;
    if (table_pointer && holder != 0) {
        TraceTablePointer(true, 0, ssbo_index, cbuf_index, cbuf_offset, indirect_offset, holder,
                          binding);
    }

    // How much of the binding the dispatch marks GPU-written. A native pointer has no size, so the
    // binding is up to 8 MiB past it, and marking all of it defers the CPU's later writes to those
    // pages until the next fence and has fences copy the over-marked range back.
    u64 mark = binding.size;
    // EDEN_NX2_WRITE_EXTENT=0 marks whole bindings again (as native_compute_write_mark Full).
    static const bool write_extent_enabled = [] {
        const char* value = std::getenv("EDEN_NX2_WRITE_EXTENT");
        return value == nullptr || value[0] != '0';
    }();
    if (native_pointer && is_written && binding.size != 0 && write_extent_enabled) {
        switch (Settings::values.native_compute_write_mark.GetValue()) {
        case Settings::NativeComputeWriteMark::Auto: {
            const auto& qmd = launch_desc;
            const std::array<u32, 3> block{qmd.block_dim_x.Value(), qmd.block_dim_y.Value(),
                                           qmd.block_dim_z.Value()};
            std::optional<std::array<u32, 3>> grid;
            if (!kepler_compute->GetIndirectComputeAddress()) {
                grid = std::array<u32, 3>{qmd.grid_dim_x.Value(), qmd.grid_dim_y.Value(),
                                          qmd.grid_dim_z.Value()};
            }
            if (const auto extent = Shader::ResolveStorageWriteExtent(write_extent, block, grid)) {
                // The extent counts from the pointer, the binding from the pointer aligned down.
                const GPUVAddr pointer = gpu_memory->Read<u64>(ssbo_addr);
                const u64 pad = pointer >= binding.gpu_addr ? pointer - binding.gpu_addr : 0;
                mark = std::min<u64>(binding.size, pad + *extent);
            }
            break;
        }
        case Settings::NativeComputeWriteMark::Full:
            break;
        case Settings::NativeComputeWriteMark::Cap64K:
            mark = std::min<u64>(binding.size, 64_KiB);
            break;
        }
    }
    channel_state->compute_storage_write_marks[ssbo_index] = static_cast<u32>(mark);
}

template <class P>
void BufferCache<P>::UnbindComputeTextureBuffers() {
    channel_state->enabled_compute_texture_buffers = 0;
    channel_state->written_compute_texture_buffers = 0;
    channel_state->image_compute_texture_buffers = 0;
}

template <class P>
void BufferCache<P>::BindComputeTextureBuffer(size_t tbo_index, GPUVAddr gpu_addr, u32 size,
                                              PixelFormat format, bool is_written, bool is_image) {
    if (tbo_index >= channel_state->compute_texture_buffers.size()) [[unlikely]] {
        LOG_ERROR(HW_GPU, "Texture buffer index {} exceeds maximum texture buffer count",
                  tbo_index);
        return;
    }
    channel_state->enabled_compute_texture_buffers |= 1U << tbo_index;
    channel_state->written_compute_texture_buffers |= (is_written ? 1U : 0U) << tbo_index;
    if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
        channel_state->image_compute_texture_buffers |= (is_image ? 1U : 0U) << tbo_index;
    }
    channel_state->compute_texture_buffers[tbo_index] =
        GetTextureBufferBinding(gpu_addr, size, format);
}

template <class P>
void BufferCache<P>::FlushCachedWrites() {
    memory_tracker.FlushCachedWrites();
    ++cpu_dirty_epoch;
}

template <class P>
bool BufferCache<P>::HasUncommittedFlushes() const noexcept {
    return !uncommitted_gpu_modified_ranges.Empty() || !committed_gpu_modified_ranges.empty();
}

template <class P>
void BufferCache<P>::AccumulateFlushes() {
    if (uncommitted_gpu_modified_ranges.Empty()) {
        return;
    }
    committed_gpu_modified_ranges.emplace_back(std::move(uncommitted_gpu_modified_ranges));
}

template <class P>
bool BufferCache<P>::ShouldWaitAsyncFlushes() const noexcept {
    return (!async_buffers.empty() && async_buffers.front().has_value());
}

template <class P>
void BufferCache<P>::CommitAsyncFlushesHigh() {
    AccumulateFlushes();

    if (committed_gpu_modified_ranges.empty()) {
        async_buffers.emplace_back(std::optional<Async_Buffer>{});
        return;
    }

    auto it = committed_gpu_modified_ranges.begin();
    while (it != committed_gpu_modified_ranges.end()) {
        auto& current_intervals = *it;
        auto next_it = std::next(it);
        while (next_it != committed_gpu_modified_ranges.end()) {
            next_it->ForEach([&current_intervals](DAddr start, DAddr end) {
                current_intervals.Subtract(start, end - start);
            });
            next_it++;
        }
        it++;
    }

    boost::container::small_vector<std::pair<BufferCopy, BufferId>, 16> downloads;
    u64 total_size_bytes = 0;
    u64 largest_copy = 0;
    for (const Common::RangeSet<DAddr>& range_set : committed_gpu_modified_ranges) {
        range_set.ForEach([&](DAddr interval_lower, DAddr interval_upper) {
            const std::size_t size = interval_upper - interval_lower;
            const DAddr device_addr = interval_lower;
            ForEachBufferInRange(device_addr, size, [&](BufferId buffer_id, Buffer& buffer) {
                const DAddr buffer_start = buffer.CpuAddr();
                const DAddr buffer_end = buffer_start + buffer.SizeBytes();
                const DAddr new_start = (std::max)(buffer_start, device_addr);
                const DAddr new_end = (std::min)(buffer_end, device_addr + size);
                memory_tracker.ForEachDownloadRange(
                    new_start, new_end - new_start, false,
                    [&](u64 device_addr_out, u64 range_size) {
                        const DAddr buffer_addr = buffer.CpuAddr();
                        const auto add_download = [&](DAddr start, DAddr end) {
                            const u64 new_offset = start - buffer_addr;
                            const u64 new_size = end - start;
                            downloads.push_back({
                                BufferCopy{
                                    .src_offset = new_offset,
                                    .dst_offset = total_size_bytes,
                                    .size = new_size,
                                },
                                buffer_id,
                            });
                            // Align up to avoid cache conflicts
                            constexpr u64 align = 64ULL;
                            constexpr u64 mask = ~(align - 1ULL);
                            total_size_bytes += (new_size + align - 1) & mask;
                            largest_copy = (std::max)(largest_copy, new_size);
                        };

                        gpu_modified_ranges.ForEachInRange(device_addr_out, range_size,
                                                           add_download);
                    });
            });
        });
    }
    committed_gpu_modified_ranges.clear();
    if (downloads.empty()) {
        async_buffers.emplace_back(std::optional<Async_Buffer>{});
        return;
    }
    Nx2Profile::Add(Nx2Profile::Counter::DownloadBytes, total_size_bytes);
    Nx2Profile::Add(Nx2Profile::Counter::DownloadCopies, downloads.size());
    auto download_staging = runtime.DownloadStagingBuffer(total_size_bytes, true);
    boost::container::small_vector<BufferCopy, 4> normalized_copies;
    Common::RangeSet<DAddr> download_ranges;
    runtime.PreCopyBarrier();
    for (auto& [copy, buffer_id] : downloads) {
        copy.dst_offset += download_staging.offset;
        const std::array copies{copy};
        BufferCopy second_copy{copy};
        Buffer& buffer = slot_buffers[buffer_id];
        second_copy.src_offset = static_cast<size_t>(buffer.CpuAddr()) + copy.src_offset;
        const DAddr orig_device_addr = static_cast<DAddr>(second_copy.src_offset);
        if (Nx2Profile::Enabled()) {
            u64 on_dma_pages = 0;
            const auto& pages = Nx2Profile::DmaReadPages();
            for (DAddr page = orig_device_addr >> 12; page <= (orig_device_addr + copy.size - 1) >> 12;
                 ++page) {
                if (pages.contains(page)) {
                    const DAddr begin = (std::max)(orig_device_addr, page << 12);
                    const DAddr end = (std::min)(orig_device_addr + copy.size, (page + 1) << 12);
                    on_dma_pages += end - begin;
                }
            }
            Nx2Profile::Add(Nx2Profile::Counter::DownloadBytesOnDmaPages, on_dma_pages);
        }
        if (Nx2DownloadTrace::g_enabled) [[unlikely]] {
            Nx2DownloadTrace::Copied(orig_device_addr, copy.size);
        }
        download_ranges.Add(orig_device_addr, copy.size);
        buffer.MarkUsage(copy.src_offset, copy.size);
        runtime.CopyBuffer(download_staging.buffer, buffer, copies, false);
        normalized_copies.push_back(second_copy);
    }
    runtime.PostCopyBarrier();
    pending_downloads.emplace_back(std::move(normalized_copies));
    pending_download_ranges.emplace_back(std::move(download_ranges));
    async_buffers.emplace_back(download_staging);
}

template <class P>
void BufferCache<P>::CommitAsyncFlushes() {
    CommitAsyncFlushesHigh();
}

template <class P>
void BufferCache<P>::PopAsyncFlushes() {
    PopAsyncBuffers();
}

template <class P>
void BufferCache<P>::ClaimCpuWrites(std::span<const std::pair<DAddr, u64>> ranges) {
    for (const auto& [device_addr, size] : ranges) {
        if (IsRegionGpuModified(device_addr, size)) {
            ClearDownload(device_addr, size);
            gpu_modified_ranges.Subtract(device_addr, size);
        }
    }
}

template <class P>
Common::RangeSet<DAddr> BufferCache<P>::PendingCpuWriteRanges() {
    Common::RangeSet<DAddr> ranges;
    if (pending_cpu_writes) {
        pending_cpu_writes(
            [&ranges](DAddr device_addr, size_t size) { ranges.Add(device_addr, size); });
    }
    return ranges;
}

template <class P>
void BufferCache<P>::WriteDownloadedMemory(const Common::RangeSet<DAddr>& cpu_written,
                                           DAddr device_addr, const u8* data, u64 size) {
    if (!cpu_written.Intersects(device_addr, size)) {
        device_memory.WriteBlockUnsafe(device_addr, data, size);
        return;
    }
    Common::RangeSet<DAddr> parts;
    parts.Add(device_addr, size);
    cpu_written.ForEachInRange(device_addr, size, [&parts](DAddr start, DAddr end) {
        parts.Subtract(start, end - start);
    });
    parts.ForEach([&](DAddr start, DAddr end) {
        device_memory.WriteBlockUnsafe(start, data + (start - device_addr), end - start);
    });
}

template <class P>
void BufferCache<P>::PopAsyncBuffers() {
    if (async_buffers.empty()) {
        return;
    }
    if (!async_buffers.front().has_value()) {
        async_buffers.pop_front();
        return;
    }
    auto& downloads = pending_downloads.front();
    auto& ranges = pending_download_ranges.front();
    auto& async_buffer = async_buffers.front();
    u8* base = async_buffer->mapped_span.data();
    const size_t base_offset = async_buffer->offset;
    // This runs on the fencing thread while the CPU keeps writing: CPU writes collected since the
    // copy was taken are newer than it, and the GPU thread hasn't dropped them from `ranges` yet.
    const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();
    for (const auto& copy : downloads) {
        const DAddr device_addr = static_cast<DAddr>(copy.src_offset);
        const u64 dst_offset = copy.dst_offset - base_offset;
        const u8* read_mapped_memory = base + dst_offset;
        // Only what no write replaced since the copy was taken: the rest is older than guest
        // memory (CPU writes) or than the GPU's copy (GPU writes, still to be downloaded).
        ranges.ForEachInRange(device_addr, copy.size, [&](DAddr start, DAddr end) {
            WriteDownloadedMemory(cpu_written, start, &read_mapped_memory[start - device_addr],
                                  end - start);
            gpu_modified_ranges.Subtract(start, end - start);
        });
    }
    Nx2Reactive::NotifyLanded();
    async_buffers_death_ring.emplace_back(*async_buffer);
    async_buffers.pop_front();
    pending_downloads.pop_front();
    pending_download_ranges.pop_front();
}

template <class P>
bool BufferCache<P>::IsRegionGpuModified(DAddr addr, size_t size) {
    // Pages the GPU never wrote hold no gpu_modified_ranges data: the GPU-written page bitmap
    // answers small ranges without the tree lookup (EDEN_NX2_WRITE_GPU_BITMAP). Every range
    // added to the set marks its pages there in Switch 2 sessions; the bits never clear.
    if (Nx2BufCache::g_write_gpu_bitmap && size <= 64_KiB && !nx2_gpu_ranges_unmarked &&
        Tegra::g_nx2_ampere.load(std::memory_order_relaxed) &&
        !Tegra::Nx2MayBeGpuWritten(addr, size)) {
        return false;
    }
    // One lookup instead of visiting every range inside (~4% of the GPU thread for title 1's
    // inline uploads once GPU-written ranges are no longer cleared by downloads).
    return gpu_modified_ranges.Intersects(addr, size);
}

template <class P>
bool BufferCache<P>::IsRegionRegistered(DAddr addr, size_t size) {
    const DAddr end_addr = addr + size;
    const u64 page_end = Common::DivCeil(end_addr, CACHING_PAGESIZE);
    for (u64 page = addr >> CACHING_PAGEBITS; page < page_end;) {
        const BufferId buffer_id = page_table[page];
        if (!buffer_id) {
            ++page;
            continue;
        }
        Buffer& buffer = slot_buffers[buffer_id];
        const DAddr buf_start_addr = buffer.CpuAddr();
        const DAddr buf_end_addr = buf_start_addr + buffer.SizeBytes();
        if (buf_start_addr < end_addr && addr < buf_end_addr) {
            return true;
        }
        page = Common::DivCeil(end_addr, CACHING_PAGESIZE);
    }
    return false;
}

template <class P>
bool BufferCache<P>::IsRegionCpuModified(DAddr addr, size_t size) {
    return memory_tracker.IsRegionCpuModified(addr, size);
}

template <class P>
void BufferCache<P>::BindHostIndexBuffer() {
    Buffer& buffer = slot_buffers[channel_state->index_buffer.buffer_id];
    TouchBuffer(buffer, channel_state->index_buffer.buffer_id);
    const u32 offset = buffer.Offset(channel_state->index_buffer.device_addr);
    const u32 size = channel_state->index_buffer.size;
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    if constexpr (requires { runtime.BindHostUint8IndexBuffer(std::span<const u8>{}); }) {
        if (draw_state.index_buffer.format == Maxwell::IndexFormat::UnsignedByte &&
            draw_state.topology != Maxwell::PrimitiveTopology::Quads &&
            draw_state.topology != Maxwell::PrimitiveTopology::QuadStrip &&
            runtime.WidensUint8IndicesOnHost()) {
            std::span<const u8> indices;
            const auto& index_buffer = draw_state.index_buffer;
            const u32 guest_size = (std::min)(
                static_cast<u32>(index_buffer.EndAddress() - index_buffer.StartAddress()),
                index_buffer.first + index_buffer.count);
            if (!draw_state.inline_index_draw_indexes.empty()) {
                indices = std::span{draw_state.inline_index_draw_indexes.data(), size};
            } else if (guest_size != 0 &&
                       !IsRegionGpuModified(channel_state->index_buffer.device_addr, guest_size)) {
                guest_uint8_indices.resize(guest_size);
                device_memory.ReadBlockUnsafe(channel_state->index_buffer.device_addr,
                                              guest_uint8_indices.data(), guest_size);
                indices = guest_uint8_indices;
            }
            if (!indices.empty()) {
                runtime.BindHostUint8IndexBuffer(indices);
                return;
            }
        }
    }
    if (draw_state.inline_index_draw_indexes.empty()) {
        SynchronizeBuffer(buffer, channel_state->index_buffer.device_addr, size);
    } else {
        if constexpr (USE_MEMORY_MAPS_FOR_UPLOADS) {
            auto upload_staging = runtime.UploadStagingBuffer(size);
            std::array<BufferCopy, 1> copies{{BufferCopy{.src_offset = upload_staging.offset, .dst_offset = 0, .size = size}}};
            std::memcpy(upload_staging.mapped_span.data(), draw_state.inline_index_draw_indexes.data(), size);
            runtime.CopyBuffer(buffer, upload_staging.buffer, copies, true);
        } else {
            buffer.ImmediateUpload(0, draw_state.inline_index_draw_indexes);
        }
    }
    if constexpr (HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT) {
        const u32 new_offset = offset + draw_state.index_buffer.first * u32(draw_state.index_buffer.FormatSizeInBytes());
        runtime.BindIndexBuffer(buffer, new_offset, size);
    } else {
        buffer.MarkUsage(offset, size);
        runtime.BindIndexBuffer(draw_state.topology, draw_state.index_buffer.format, draw_state.index_buffer.first, draw_state.index_buffer.count, buffer, offset, size);
    }
}

template <class P>
void BufferCache<P>::BindHostVertexBuffer(u32 index, Buffer& buffer, u32 offset, u32 size,
                                          u32 stride) {
    if constexpr (IS_OPENGL) {
        runtime.BindVertexBuffer(index, buffer, offset, size, stride);
    } else {
        runtime.BindVertexBuffer(index, buffer.Handle(), offset, size, stride);
    }
}

template <class P>
Binding& BufferCache<P>::VertexBufferSlot(u32 index) {
    ASSERT(index < NUM_VERTEX_BUFFERS);
    return v_buffer[index];
}

template <class P>
const Binding& BufferCache<P>::VertexBufferSlot(u32 index) const {
    ASSERT(index < NUM_VERTEX_BUFFERS);
    return v_buffer[index];
}

template <class P>
void BufferCache<P>::UpdateVertexBufferSlot(u32 index, const Binding& binding) {
    Binding& slot = VertexBufferSlot(index);
    if (slot.device_addr != binding.device_addr || slot.size != binding.size) {
        ++vertex_buffers_serial;
    }
    slot = binding;
    if (binding.buffer_id != NULL_BUFFER_ID && binding.size != 0) {
        enabled_vertex_buffers_mask |= (1u << index);
    } else {
        enabled_vertex_buffers_mask &= ~(1u << index);
    }
}

template <class P>
void BufferCache<P>::MarkVertexBuffersStale(DAddr device_addr, u64 size) {
    if constexpr (requires { runtime.IsVertexRepacked(u32{}); }) {
        u32 mask = enabled_vertex_buffers_mask;
        while (mask != 0) {
            const u32 index = std::countr_zero(mask);
            mask &= mask - 1;
            const Binding& binding = v_buffer[index];
            if (device_addr < binding.device_addr + binding.size &&
                binding.device_addr < device_addr + size) {
                stale_vertex_buffers_mask |= 1U << index;
            }
        }
    }
}

template <class P>
void BufferCache<P>::RefreshRepackedVertexBuffers() {
    if constexpr (requires { runtime.IsVertexRepacked(u32{}); }) {
        u32 mask = std::exchange(stale_vertex_buffers_mask, 0) & enabled_vertex_buffers_mask;
        auto& flags = maxwell3d->dirty.flags;
        while (mask != 0) {
            const u32 index = std::countr_zero(mask);
            mask &= mask - 1;
            if (runtime.IsVertexRepacked(index)) {
                flags[Dirty::VertexBuffer0 + index] = true;
            }
        }
    }
}

template <class P>
void BufferCache<P>::BindHostVertexBuffers() {
    RefreshRepackedVertexBuffers();

#ifdef __ANDROID__
    const bool use_optimized_vertex_buffers = Settings::values.use_optimized_vertex_buffers.GetValue();
#else
    constexpr bool use_optimized_vertex_buffers = true;
#endif

    if (use_optimized_vertex_buffers) {
        auto& flags = maxwell3d->dirty.flags;
        u32 enabled_mask = enabled_vertex_buffers_mask;
        HostBindings<Buffer> bindings{};
        u32 last_index = (std::numeric_limits<u32>::max)();
        const auto flush_bindings = [&]() {
            if (bindings.buffers.empty()) {
                return;
            }
            bindings.max_index = bindings.min_index + static_cast<u32>(bindings.buffers.size());
            runtime.BindVertexBuffers(bindings);
            bindings = HostBindings<Buffer>{};
            last_index = (std::numeric_limits<u32>::max)();
        };
        while (enabled_mask != 0) {
            const u32 index = std::countr_zero(enabled_mask);
            enabled_mask &= (enabled_mask - 1);
            const Binding& binding = VertexBufferSlot(index);
            Buffer& buffer = slot_buffers[binding.buffer_id];
            TouchBuffer(buffer, binding.buffer_id);
            SynchronizeBuffer(buffer, binding.device_addr, binding.size);
            if (!flags[Dirty::VertexBuffer0 + index]) {
                flush_bindings();
                continue;
            }
            flags[Dirty::VertexBuffer0 + index] = false;
            const u32 stride = maxwell3d->regs.vertex_streams[index].stride;
            const u32 offset = buffer.Offset(binding.device_addr);
            buffer.MarkUsage(offset, binding.size);
            if (!bindings.buffers.empty() && index != last_index + 1) {
                flush_bindings();
            }
            if (bindings.buffers.empty()) {
                bindings.min_index = index;
            }
            bindings.buffers.push_back(&buffer);
            bindings.offsets.push_back(offset);
            bindings.sizes.push_back(binding.size);
            bindings.strides.push_back(stride);
            last_index = index;
        }
        flush_bindings();
    } else {
        HostBindings<typename P::Buffer> host_bindings;
        bool any_valid{false};
        auto& flags = maxwell3d->dirty.flags;
        for (u32 index = 0; index < NUM_VERTEX_BUFFERS; ++index) {
            const Binding& binding = channel_state->vertex_buffers[index];
            Buffer& buffer = slot_buffers[binding.buffer_id];
            TouchBuffer(buffer, binding.buffer_id);
            SynchronizeBuffer(buffer, binding.device_addr, binding.size);
            if (!flags[Dirty::VertexBuffer0 + index]) {
                continue;
            }
            flags[Dirty::VertexBuffer0 + index] = false;

            host_bindings.min_index = (std::min)(host_bindings.min_index, index);
            host_bindings.max_index = (std::max)(host_bindings.max_index, index);
            any_valid = true;
        }

        if (any_valid) {
            host_bindings.max_index++;
            for (u32 index = host_bindings.min_index; index < host_bindings.max_index; index++) {
                flags[Dirty::VertexBuffer0 + index] = false;

                const Binding& binding = channel_state->vertex_buffers[index];
                Buffer& buffer = slot_buffers[binding.buffer_id];

                const u32 stride = maxwell3d->regs.vertex_streams[index].stride;
                const u32 offset = buffer.Offset(binding.device_addr);
                buffer.MarkUsage(offset, binding.size);

                host_bindings.buffers.push_back(&buffer);
                host_bindings.offsets.push_back(offset);
                host_bindings.sizes.push_back(binding.size);
                host_bindings.strides.push_back(stride);
            }
            runtime.BindVertexBuffers(host_bindings);
        }
    }
}

template <class P>
void BufferCache<P>::BindHostDrawIndirectBuffers() {
    const auto bind_buffer = [this](const Binding& binding) {
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        SynchronizeBuffer(buffer, binding.device_addr, binding.size);
    };
    if (current_draw_indirect->include_count) {
        bind_buffer(channel_state->count_buffer_binding);
    }
    bind_buffer(channel_state->indirect_buffer_binding);
}

template <class P>
void BufferCache<P>::BindHostGraphicsUniformBuffers(size_t stage) {
    u32 dirty = ~0U;
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        dirty = std::exchange(channel_state->dirty_uniform_buffers[stage], 0);
    }
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_uniform_buffer_masks[stage], [&](u32 index) {
        const bool needs_bind = ((dirty >> index) & 1) != 0;
        BindHostGraphicsUniformBuffer(stage, index, binding_index, needs_bind);
        if constexpr (NEEDS_BIND_UNIFORM_INDEX) {
            ++binding_index;
        }
    });
}

template <class P>
void BufferCache<P>::BindHostGraphicsUniformBuffer(size_t stage, u32 index, u32 binding_index, bool needs_bind) {
    ++channel_state->uniform_cache_shots[0];
    // Bank 24 (root table 0, NVN2's push constants) is read from bank 17, which such games do not
    // bind: stream the table's current words as that buffer.
    if (index == 17 && maxwell3d->root_tables.loaded && channel_state->uniform_buffers[stage][index].size == 0) {
        const auto& words = maxwell3d->root_tables.tables[0];
        const u32 size = (*channel_state->uniform_buffer_sizes)[stage][index];
        const u32 copied = (std::min)(size, static_cast<u32>(sizeof(words)));
        if constexpr (IS_OPENGL) {
            std::array<u8, sizeof(words)> data{};
            std::memcpy(data.data(), words.data(), copied);
            runtime.BindFastUniformBuffer(stage, binding_index, copied);
            runtime.PushFastUniformBuffer(stage, binding_index, std::span<const u8>(data.data(), copied));
        } else {
            const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
            std::memset(span.data(), 0, span.size());
            std::memcpy(span.data(), words.data(), copied);
        }
        channel_state->fast_bound_uniform_buffers[stage] |= 1u << binding_index;
        channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
        return;
    }
    const Binding& binding = channel_state->uniform_buffers[stage][index];
    const DAddr device_addr = binding.device_addr;
    const u32 size = (std::min)(binding.size, (*channel_state->uniform_buffer_sizes)[stage][index]);
    Buffer& buffer = slot_buffers[binding.buffer_id];
    TouchBuffer(buffer, binding.buffer_id);
    const bool has_host_buffer = binding.buffer_id != NULL_BUFFER_ID;
    const u32 offset = has_host_buffer ? buffer.Offset(device_addr) : 0;
    const bool needs_alignment_stream = [&]() {
        if constexpr (IS_OPENGL) {
            return false;
        } else {
            if (!has_host_buffer) {
                return false;
            }
            const u32 alignment = runtime.GetUniformBufferAlignment();
            return alignment > 1 && (offset % alignment) != 0;
        }
    }();
    const bool use_fast_buffer = needs_alignment_stream
        || (has_host_buffer && size <= channel_state->uniform_buffer_skip_cache_size
            && !memory_tracker.IsRegionGpuModified(device_addr, size));
    // Native programs keep tables next to their code in a 64 KiB bank read with indirect offsets
    // (Splatoon Raiders: the dialogue box corners in c1). Writes to that memory can go unnoticed by
    // the memory tracker, leaving stale host contents, so upload such a bank again once per frame.
    // A bind that streams the buffer copies guest memory itself: the upload waits for the frame's
    // first bind that reads the host buffer (EDEN_NX2_BUFCACHE_BANK).
    const bool is_native_bank =
        has_host_buffer && (*channel_state->uniform_buffer_sizes)[stage][index] == 0x10000;
    const bool bank_streamed = use_fast_buffer && Nx2BufCache::g_bank_skip;
    if (is_native_bank && bank_streamed) {
        Nx2BufCache::Note(Nx2BufCache::Stat::BankSkip);
    }
    const auto upload_bank = [&] {
        if (!(is_native_bank && !bank_streamed && size != 0 && gpu_memory != nullptr &&
              gpu_memory->UsesNativeTiling() &&
              (Nx2Perf::Verify() || !native_indirect_uniforms_this_frame.contains(device_addr)) &&
              !memory_tracker.IsRegionGpuModified(device_addr, size) &&
              native_indirect_uniforms_this_frame.insert(device_addr))) {
            return;
        }
        // The bank size is only an upper limit: stop at the end of the mapped memory.
        u64 mapped_size = 0;
        while (mapped_size < size &&
               device_memory.template GetPointer<u8>(device_addr + mapped_size) != nullptr) {
            mapped_size += Core::DEVICE_PAGESIZE - ((device_addr + mapped_size) & Core::DEVICE_PAGEMASK);
        }
        const u64 upload_size = (std::min<u64>)(mapped_size, size);
        if (upload_size != 0) {
            if constexpr (USE_MEMORY_MAPS && USE_MEMORY_MAPS_FOR_UPLOADS) {
                if (Nx2BufCache::g_bank_shadow) {
                    UploadNativeBank(buffer, device_addr, upload_size);
                } else {
                    std::array<BufferCopy, 1> copies{BufferCopy{
                        .src_offset = 0,
                        .dst_offset = buffer.Offset(device_addr),
                        .size = upload_size,
                    }};
                    UploadMemory(buffer, upload_size, upload_size, copies);
                    InvalidateHotShadows(device_addr, upload_size);
                    Nx2BufCache::Note(Nx2BufCache::Stat::BankUpload);
                    Nx2BufCache::Note(Nx2BufCache::Stat::BankUploadBytes, upload_size);
                }
            } else {
                std::array<BufferCopy, 1> copies{BufferCopy{
                    .src_offset = 0,
                    .dst_offset = buffer.Offset(device_addr),
                    .size = upload_size,
                }};
                UploadMemory(buffer, upload_size, upload_size, copies);
                InvalidateHotShadows(device_addr, upload_size);
            }
        }
    };
    // A classic bind syncs its range first: a page the tracker saw written is uploaded once and
    // the bank compare then finds it equal to its shadow (EDEN_NX2_BANK_SYNC_FIRST with
    // EDEN_NX2_BANK_SYNC_REFRESH), instead of the compare uploading it, the sync again and next
    // frame's compare a third time.
    const bool bank_after_sync = !use_fast_buffer && Nx2BufCache::g_bank_sync_first;
    if (!bank_after_sync) {
        upload_bank();
    }
    Nx2BufCache::Note(use_fast_buffer ? Nx2BufCache::Stat::UboStream : Nx2BufCache::Stat::UboClassic);
    if (use_fast_buffer) {
        if constexpr (IS_OPENGL) {
            if (runtime.HasFastBufferSubData()) {
                // Fast path for Nvidia
                const bool should_fast_bind =
                    !HasFastUniformBufferBound(stage, binding_index) ||
                    channel_state->uniform_buffer_binding_sizes[stage][binding_index] != size;
                if (should_fast_bind) {
                    // We only have to bind when the currently bound buffer is not the fast version
                    channel_state->fast_bound_uniform_buffers[stage] |= 1u << binding_index;
                    channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
                    runtime.BindFastUniformBuffer(stage, binding_index, size);
                }
                const auto span = ImmediateBufferWithData(device_addr, size);
                runtime.PushFastUniformBuffer(stage, binding_index, span);
                return;
            }
        }
        channel_state->fast_bound_uniform_buffers[stage] |= 1u << binding_index;
        channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
        // Stream buffer path to avoid stalling on non-Nvidia drivers or Vulkan
        bool streamed = false;
        if constexpr (!IS_OPENGL && VideoCommon::StreamUniformReusable<Runtime>) {
            // The same bytes as this slot's previous bind, in the same command buffer: bind that
            // stream allocation again instead of allocating and copying (uniform_stream_reuse.h).
            // EDEN_NX2_UBOREUSE=0 turns it off.
            if (VideoCommon::UniformStreamReuseEnabled()) {
                VideoCommon::UniformReuseResult reuse_result{};
                streamed = VideoCommon::BindSmallUniformReusing(runtime, device_memory, stage,
                                                                index, binding_index, device_addr,
                                                                size, &reuse_result);
                NoteUniformReuse(reuse_result, size);
            }
        }
        if (!streamed) {
            const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
            device_memory.ReadBlockUnsafe(device_addr, span.data(), size);
            if (size > DEFAULT_SKIP_CACHE_SIZE) {
                Nx2BufCache::Note(Nx2BufCache::Stat::UboLarge);
                Nx2BufCache::Note(Nx2BufCache::Stat::UboLargeBytes, size);
            }
        }
        return;
    }
    if (is_native_bank) {
        Nx2BufCache::Note(Nx2BufCache::Stat::UboClassicBank);
    } else if (size <= channel_state->uniform_buffer_skip_cache_size) {
        Nx2BufCache::Note(Nx2BufCache::Stat::UboClassicSmall);
    }
    // Classic cached path
    if (SynchronizeBuffer(buffer, device_addr, size)) {
        ++channel_state->uniform_cache_hits[0];
    } else {
        Nx2BufCache::Note(Nx2BufCache::Stat::UboClassicUpload);
    }
    if (bank_after_sync) {
        upload_bank();
    }
    // Skip binding if it's not needed and if the bound buffer is not the fast version
    // This exists to avoid instances where the fast buffer is bound and a GPU write happens
    needs_bind |= HasFastUniformBufferBound(stage, binding_index);
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        needs_bind |= channel_state->uniform_buffer_binding_sizes[stage][binding_index] != size;
    }
    if (!needs_bind) {
        return;
    }
    if constexpr (IS_OPENGL) {
        // Mark the index as dirty if offset doesn't match
        const bool is_copy_bind = offset != 0 && !runtime.SupportsNonZeroUniformOffset();
        channel_state->dirty_uniform_buffers[stage] |= (is_copy_bind ? 1U : 0U) << index;
    }
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
    }
    buffer.MarkUsage(offset, size);
    if constexpr (NEEDS_BIND_UNIFORM_INDEX) {
        runtime.BindUniformBuffer(stage, binding_index, buffer, offset, size);
    } else {
        runtime.BindUniformBuffer(buffer, offset, size);
    }
    channel_state->fast_bound_uniform_buffers[stage] &= ~(1u << binding_index);
}

template <class P>
void BufferCache<P>::NoteUniformReuse(VideoCommon::UniformReuseResult result, u32 size) {
    if (!Nx2Profile::Enabled()) [[likely]] {
        return;
    }
    switch (result) {
    case VideoCommon::UniformReuseResult::SlotHit:
        Nx2BufCache::Note(Nx2BufCache::Stat::UboSlotHit);
        break;
    case VideoCommon::UniformReuseResult::RangeHit:
        Nx2BufCache::Note(Nx2BufCache::Stat::UboRangeHit);
        break;
    case VideoCommon::UniformReuseResult::Recopied:
        Nx2BufCache::Note(Nx2BufCache::Stat::UboRecopied);
        Nx2BufCache::Note(Nx2BufCache::Stat::UboCopiedBytes, size);
        break;
    case VideoCommon::UniformReuseResult::Copied:
        Nx2BufCache::Note(Nx2BufCache::Stat::UboCopied);
        Nx2BufCache::Note(Nx2BufCache::Stat::UboCopiedBytes, size);
        break;
    case VideoCommon::UniformReuseResult::None:
        break;
    }
}

template <class P>
void BufferCache<P>::RefreshMultiRangeSegments(std::vector<MultiRangeSegment>& pool) {
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        VideoCommon::RefreshMultiRangeSegments(
            std::span{pool}, [this](BufferId id) { return slot_buffers.contains(id); },
            [this](DAddr address, u32 size) {
                return FindBuffer(address, size, runtime.PrefersSparseSources());
            });
    }
}

template <class P>
void BufferCache<P>::ResolveMultiRangeStorage(Binding& binding, bool is_written,
                                              std::vector<MultiRangeSegment>& pool) {
    binding.segment_first = 0;
    binding.segment_count = 0;
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        if (binding.gpu_addr == 0 || binding.size == 0) {
            return;
        }
        if (is_written && !runtime.PrefersSparseSources()) {
            return;
        }
        // One device range (the mapping runs, no page walk) is one segment: nothing to gather.
        // Native storage bindings are clamped to such a range, so Query only walked 8 MiB of
        // pages for each new address (EDEN_NX2_BUFCACHE_CONTIG).
        if (Nx2BufCache::g_contig_resolve &&
            gpu_memory->MaxContinuousRange(binding.gpu_addr, binding.size) >= binding.size) {
            Nx2BufCache::Note(Nx2BufCache::Stat::ResolveContig);
            return;
        }
        Nx2BufCache::Note(Nx2BufCache::Stat::ResolveQuery);
        const VirtualSegments* found =
            virtual_ranges.Query(*gpu_memory, binding.gpu_addr, binding.size);
        if (!found || found->size() < 2) {
            return;
        }
        const VirtualSegments segments = *found;
        const u32 first = static_cast<u32>(pool.size());
        const bool prefer_sparse = runtime.PrefersSparseSources();
        for (const VirtualSegment& segment : segments) {
            const BufferId buffer_id =
                FindBuffer(segment.device_addr, segment.size, prefer_sparse);
            if (!buffer_id) {
                pool.resize(first);
                return;
            }
            pool.push_back(MultiRangeSegment{
                .buffer_id = buffer_id,
                .device_addr = segment.device_addr,
                .size = segment.size,
            });
        }
        binding.segment_first = first;
        binding.segment_count = static_cast<u32>(segments.size());
    }
}

template <class P>
bool BufferCache<P>::BindMultiRangeStorage(const Binding& binding, bool is_written,
                                           std::span<const MultiRangeSegment> pool,
                                           u64 write_mark_limit) {
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        if (binding.segment_count < 2) {
            return false;
        }
        if (binding.segment_first > pool.size() ||
            binding.segment_count > pool.size() - binding.segment_first) {
            return false;
        }
        const auto segments = pool.subspan(binding.segment_first, binding.segment_count);
        if (!MultiRangeSegmentsLive(segments,
                                    [this](BufferId id) { return slot_buffers.contains(id); })) {
            return false;
        }
        const u64 key = (static_cast<u64>(gpu_memory->GetID()) << 48) ^ binding.gpu_addr;
        runtime.ResetMultiRange();
        for (u32 index = 0; index < binding.segment_count; ++index) {
            const MultiRangeSegment& segment = pool[binding.segment_first + index];
            Buffer& buffer = slot_buffers[segment.buffer_id];
            TouchBuffer(buffer, segment.buffer_id);
            SynchronizeBuffer(buffer, segment.device_addr, segment.size);
            const u32 offset = buffer.Offset(segment.device_addr);
            buffer.MarkUsage(offset, segment.size);
            if (is_written && write_mark_limit != 0) {
                // write_mark_limit: the native write mark (BindComputeStorageBuffer), spread over
                // the segments in order.
                const u32 mark = static_cast<u32>(std::min<u64>(segment.size, write_mark_limit));
                MarkWrittenBuffer(segment.buffer_id, segment.device_addr, mark);
                if (write_mark_limit != ~u64{0}) {
                    write_mark_limit -= mark;
                }
            }
            runtime.PushMultiRangeSource(buffer, offset, segment.size);
        }
        return BindFreshMultiRangeStorage(runtime, key, is_written);
    } else {
        return false;
    }
}

template <class P>
void BufferCache<P>::BindHostGraphicsStorageBuffers(size_t stage) {
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_storage_buffers[stage], [&](u32 index) {
        const Binding& binding = channel_state->storage_buffers[stage][index];
        const bool is_written = ((channel_state->written_storage_buffers[stage] >> index) & 1) != 0;
        if (BindMultiRangeStorage(binding, is_written, graphics_segments)) {
            return;
        }
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        const u32 size = binding.size;
        SynchronizeBuffer(buffer, binding.device_addr, size);

        const u32 offset = buffer.Offset(binding.device_addr);
        buffer.MarkUsage(offset, size);

        if (is_written) {
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size);
        }

        if constexpr (NEEDS_BIND_STORAGE_INDEX) {
            runtime.BindStorageBuffer(stage, binding_index, buffer, offset, size, is_written);
            ++binding_index;
        } else {
            runtime.BindStorageBuffer(buffer, offset, size, is_written);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostGraphicsTextureBuffers(size_t stage) {
    ForEachEnabledBit(channel_state->enabled_texture_buffers[stage], [&](u32 index) {
        const TextureBufferBinding& binding = channel_state->texture_buffers[stage][index];
        Buffer& buffer = slot_buffers[binding.buffer_id];
        const u32 size = binding.size;
        SynchronizeBuffer(buffer, binding.device_addr, size);

        const bool is_written = ((channel_state->written_texture_buffers[stage] >> index) & 1) != 0;
        if (is_written) {
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size);
        }

        const u32 offset = buffer.Offset(binding.device_addr);
        const PixelFormat format = binding.format;
        buffer.MarkUsage(offset, size);
        if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
            if (((channel_state->image_texture_buffers[stage] >> index) & 1) != 0) {
                runtime.BindImageBuffer(buffer, offset, size, format);
            } else {
                runtime.BindTextureBuffer(buffer, offset, size, format);
            }
        } else {
            runtime.BindTextureBuffer(buffer, offset, size, format);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostTransformFeedbackBuffers() {
    if (maxwell3d->regs.transform_feedback_enabled == 0) {
        return;
    }
    HostBindings<typename P::Buffer> host_bindings;
    for (u32 index = 0; index < NUM_TRANSFORM_FEEDBACK_BUFFERS; ++index) {
        const Binding& binding = channel_state->transform_feedback_buffers[index];
        const auto& control = maxwell3d->regs.transform_feedback.controls[index];
        const bool has_layout = control.varying_count != 0 || control.stride != 0;

        Buffer* host_buffer = &slot_buffers[NULL_BUFFER_ID];
        u32 offset = 0;
        u32 size = 0;

        if (has_layout && binding.buffer_id != NULL_BUFFER_ID && binding.size != 0) {
            Buffer& buffer = slot_buffers[binding.buffer_id];
            TouchBuffer(buffer, binding.buffer_id);
            size = binding.size;
            SynchronizeBuffer(buffer, binding.device_addr, size);
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size);
            offset = buffer.Offset(binding.device_addr);
            buffer.MarkUsage(offset, size);
            host_buffer = &buffer;
        }

        host_bindings.buffers.push_back(host_buffer);
        host_bindings.offsets.push_back(offset);
        host_bindings.sizes.push_back(size);
    }
    runtime.BindTransformFeedbackBuffers(host_bindings);
}

template <class P>
void BufferCache<P>::BindHostComputeUniformBuffers() {
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        // Mark all uniform buffers as dirty
        channel_state->dirty_uniform_buffers.fill(~u32{0});
        channel_state->fast_bound_uniform_buffers.fill(0);
    }
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_compute_uniform_buffer_mask, [&](u32 index) {
        const Binding& binding = channel_state->compute_uniform_buffers[index];
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        const u32 size =
            (std::min)(binding.size, (*channel_state->compute_uniform_buffer_sizes)[index]);
        const bool has_host_buffer = binding.buffer_id != NULL_BUFFER_ID;
        const u32 offset = has_host_buffer ? buffer.Offset(binding.device_addr) : 0;
        const bool needs_alignment_stream = [&]() {
            if constexpr (IS_OPENGL) {
                return false;
            } else {
                if (!has_host_buffer) {
                    return false;
                }
                const u32 alignment = runtime.GetUniformBufferAlignment();
                return alignment > 1 && (offset % alignment) != 0;
            }
        }();
        // Small buffers the GPU has not written are streamed, as graphics ones are: synchronizing
        // them made every dispatch re-protect pages an inline upload had just unprotected (two
        // VirtualProtect calls per dispatch, ~4,000 dispatches a second in title 1).
        const bool use_fast_buffer =
            needs_alignment_stream ||
            (has_host_buffer && size <= channel_state->uniform_buffer_skip_cache_size &&
             !memory_tracker.IsRegionGpuModified(binding.device_addr, size));
        if constexpr (!IS_OPENGL) {
            if (use_fast_buffer) {
                // The same bytes as this slot's previous dispatch in the same command buffer:
                // bind that stream allocation again (EDEN_NX2_COMPUTE_BINDCACHE).
                if constexpr (VideoCommon::StreamUniformReusable<Runtime>) {
                    if (Nx2BufCache::g_compute_bindcache &&
                        VideoCommon::UniformStreamReuseEnabled() &&
                        VideoCommon::BindSmallUniformReusing(runtime, device_memory, 5, index,
                                                             binding_index, binding.device_addr,
                                                             size)) {
                        return;
                    }
                }
                const std::span<u8> span =
                    runtime.BindMappedUniformBuffer(0, binding_index, size);
                device_memory.ReadBlockUnsafe(binding.device_addr, span.data(), size);
                return;
            }
        }

        SynchronizeBuffer(buffer, binding.device_addr, size);

        buffer.MarkUsage(offset, size);
        if constexpr (NEEDS_BIND_UNIFORM_INDEX) {
            runtime.BindComputeUniformBuffer(binding_index, buffer, offset, size);
            ++binding_index;
        } else {
            runtime.BindUniformBuffer(buffer, offset, size);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostComputeStorageBuffers() {
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_compute_storage_buffers, [&](u32 index) {
        const Binding& binding = channel_state->compute_storage_buffers[index];
        const bool is_written =
            ((channel_state->written_compute_storage_buffers >> index) & 1) != 0;
        // A native program's binding is min(layout, 8 MiB) because its pointer has no size, and
        // marking all of it GPU-written defers the CPU's later writes to those pages. Mark only
        // what BindComputeStorageBuffer worked out the dispatch can write; the binding keeps its
        // size so the shader can still read all of it.
        const bool native =
            ((channel_state->native_compute_storage_buffers >> index) & 1) != 0;
        const u64 write_mark_limit =
            native ? u64{channel_state->compute_storage_write_marks[index]} : ~u64{0};
        if (BindMultiRangeStorage(binding, is_written, compute_segments, write_mark_limit)) {
            return;
        }
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        const u32 size = binding.size;
        SynchronizeBuffer(buffer, binding.device_addr, size);

        const u32 offset = buffer.Offset(binding.device_addr);
        buffer.MarkUsage(offset, size);

        const u32 mark =
            is_written ? static_cast<u32>(std::min<u64>(size, write_mark_limit)) : 0;
        if (mark != 0) {
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, mark, true);
        }

        if constexpr (NEEDS_BIND_STORAGE_INDEX) {
            runtime.BindComputeStorageBuffer(binding_index, buffer, offset, size, is_written);
            ++binding_index;
        } else {
            runtime.BindStorageBuffer(buffer, offset, size, is_written);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostComputeTextureBuffers() {
    ForEachEnabledBit(channel_state->enabled_compute_texture_buffers, [&](u32 index) {
        const TextureBufferBinding& binding = channel_state->compute_texture_buffers[index];
        Buffer& buffer = slot_buffers[binding.buffer_id];
        const u32 size = binding.size;
        SynchronizeBuffer(buffer, binding.device_addr, size);

        const bool is_written =
            ((channel_state->written_compute_texture_buffers >> index) & 1) != 0;
        if (is_written) {
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size);
        }

        const u32 offset = buffer.Offset(binding.device_addr);
        const PixelFormat format = binding.format;
        buffer.MarkUsage(offset, size);
        if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
            if (((channel_state->image_compute_texture_buffers >> index) & 1) != 0) {
                runtime.BindImageBuffer(buffer, offset, size, format);
            } else {
                runtime.BindTextureBuffer(buffer, offset, size, format);
            }
        } else {
            runtime.BindTextureBuffer(buffer, offset, size, format);
        }
    });
}

template <class P>
void BufferCache<P>::DoUpdateGraphicsBuffers(bool is_indexed) {
    graphics_segments.clear();
    BufferOperations([&]() {
        if (is_indexed) {
            UpdateIndexBuffer();
        }
        UpdateVertexBuffers();
        UpdateTransformFeedbackBuffers();
        for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
            UpdateUniformBuffers(stage);
            UpdateStorageBuffers(stage);
            UpdateTextureBuffers(stage);
        }
        if (current_draw_indirect) {
            UpdateDrawIndirect();
        }
        RefreshMultiRangeSegments(graphics_segments);
    });
}

template <class P>
void BufferCache<P>::DoUpdateComputeBuffers() {
    compute_segments.clear();
    BufferOperations([&]() {
        UpdateComputeUniformBuffers();
        UpdateComputeStorageBuffers();
        UpdateComputeTextureBuffers();
        RefreshMultiRangeSegments(compute_segments);
    });
}

template <class P>
void BufferCache<P>::UpdateIndexBuffer() {
    // We have to check for the dirty flags and index count
    // The index count is currently changed without updating the dirty flags
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    const auto& index_buffer_ref = draw_state.index_buffer;
    auto& flags = maxwell3d->dirty.flags;
    if (!flags[Dirty::IndexBuffer]) {
        return;
    }
    flags[Dirty::IndexBuffer] = false;
    if (!draw_state.inline_index_draw_indexes.empty()) [[unlikely]] {
        auto inline_index_size = static_cast<u32>(draw_state.inline_index_draw_indexes.size());
        u32 buffer_size = Common::AlignUp(inline_index_size, CACHING_PAGESIZE);
        if (inline_buffer_id == NULL_BUFFER_ID) [[unlikely]] {
            inline_buffer_id = CreateBuffer(0, buffer_size, false);
        }
        if (slot_buffers[inline_buffer_id].SizeBytes() < buffer_size) [[unlikely]] {
            ++binding_epoch;
            slot_buffers.erase(inline_buffer_id);
            inline_buffer_id = CreateBuffer(0, buffer_size, false);
        }
        channel_state->index_buffer = Binding{
            .device_addr = 0,
            .size = inline_index_size,
            .buffer_id = inline_buffer_id,
        };
        return;
    }

    const GPUVAddr gpu_addr_begin = index_buffer_ref.StartAddress();
    const GPUVAddr gpu_addr_end = index_buffer_ref.EndAddress();
    const u32 draw_size = (index_buffer_ref.count + index_buffer_ref.first) * u32(index_buffer_ref.FormatSizeInBytes());
    if (ReuseGeometry(index_memo, gpu_addr_begin, gpu_addr_end, draw_size)) {
        channel_state->index_buffer = index_memo.binding;
        return;
    }
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr_begin);
    const u32 address_size = static_cast<u32>(gpu_addr_end - gpu_addr_begin);
    const u32 size = (std::min)(address_size, draw_size);
    if (size == 0 || !device_addr) {
        channel_state->index_buffer = NULL_BINDING;
        return;
    }
    channel_state->index_buffer = Binding{
        .device_addr = *device_addr,
        .size = size,
        .buffer_id = FindBuffer(*device_addr, size, false),
    };
    StoreGeometry(index_memo, gpu_addr_begin, gpu_addr_end, draw_size, channel_state->index_buffer);
}

template <class P>
bool BufferCache<P>::ReuseGeometry(const GeometryMemo& memo, GPUVAddr begin, GPUVAddr end,
                                   u64 extra) const {
    // FindBuffer waits for the buffer's writes with accurate GPU fences: keep that.
    return Nx2BufCache::g_geometry_memo && memo.epoch == binding_epoch && memo.begin == begin &&
           memo.end == end && memo.extra == extra &&
           memo.channel == static_cast<const void*>(channel_state) &&
           memo.as_id == gpu_memory->GetID() && memo.map_serial == gpu_memory->MapSerial() &&
           !Settings::IsGPUFenceBehaviorAccurate();
}

template <class P>
void BufferCache<P>::StoreGeometry(GeometryMemo& memo, GPUVAddr begin, GPUVAddr end, u64 extra,
                                   const Binding& binding) {
    memo = GeometryMemo{
        .epoch = binding_epoch,
        .map_serial = gpu_memory->MapSerial(),
        .as_id = gpu_memory->GetID(),
        .channel = channel_state,
        .begin = begin,
        .end = end,
        .extra = extra,
        .binding = binding,
    };
}

template <class P>
void BufferCache<P>::UpdateVertexBuffers() {
    auto& flags = maxwell3d->dirty.flags;
    if (!maxwell3d->dirty.flags[Dirty::VertexBuffers]) {
        return;
    }
    flags[Dirty::VertexBuffers] = false;

    for (u32 index = 0; index < NUM_VERTEX_BUFFERS; ++index) {
        UpdateVertexBuffer(index);
    }
}

template <class P>
void BufferCache<P>::UpdateVertexBuffer(u32 index) {
    if (!maxwell3d->dirty.flags[Dirty::VertexBuffer0 + index]) {
        return;
    }
    const auto& array = maxwell3d->regs.vertex_streams[index];
    const auto& limit = maxwell3d->regs.vertex_stream_limits[index];
    const GPUVAddr gpu_addr_begin = array.Address();
    const bool native = maxwell3d->GetNativeMacroEngine().native_turing;
    const u64 native_size = native ? maxwell3d->regs.NativeVertexStreamSize(index) : 0;
    if (native && array.enable != 0 && native_size > (std::numeric_limits<u32>::max)()) {
        throw std::runtime_error("Native vertex stream exceeds supported 32-bit buffer size");
    }
    const GPUVAddr gpu_addr_end = native ? gpu_addr_begin + native_size : limit.Address() + 1;
    GeometryMemo& memo = vertex_memo[index];
    if (array.enable != 0 && ReuseGeometry(memo, gpu_addr_begin, gpu_addr_end, 1)) {
        channel_state->vertex_buffers[index] = memo.binding;
        UpdateVertexBufferSlot(index, memo.binding);
        return;
    }
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr_begin);
    const u32 address_size = static_cast<u32>(gpu_addr_end - gpu_addr_begin);
    u32 size = address_size; // TODO: Analyze stride and number of vertices
    if (array.enable == 0 || size == 0 || !device_addr) {
        channel_state->vertex_buffers[index] = NULL_BINDING;
        UpdateVertexBufferSlot(index, NULL_BINDING);
        return;
    }
    if (!gpu_memory->IsWithinGPUAddressRange(gpu_addr_end) || size >= 64_MiB) {
        size = static_cast<u32>(gpu_memory->MaxContinuousRange(gpu_addr_begin, size));
    }
    const BufferId buffer_id = FindBuffer(*device_addr, size, false);
    const Binding binding{
        .device_addr = *device_addr,
        .size = size,
        .buffer_id = buffer_id,
    };
    channel_state->vertex_buffers[index] = binding;
    UpdateVertexBufferSlot(index, binding);
    StoreGeometry(memo, gpu_addr_begin, gpu_addr_end, 1, binding);
}

template <class P>
void BufferCache<P>::UpdateDrawIndirect() {
    const auto update = [this](GPUVAddr gpu_addr, size_t size, Binding& binding) {
        const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
        if (!device_addr) {
            binding = NULL_BINDING;
            return;
        }
        binding = Binding{
            .device_addr = *device_addr,
            .size = static_cast<u32>(size),
            .buffer_id = FindBuffer(*device_addr, static_cast<u32>(size), false),
        };
    };
    if (current_draw_indirect->include_count) {
        update(current_draw_indirect->count_start_address, sizeof(u32),
               channel_state->count_buffer_binding);
    }
    update(current_draw_indirect->indirect_start_address, current_draw_indirect->buffer_size,
           channel_state->indirect_buffer_binding);
}

template <class P>
void BufferCache<P>::UpdateUniformBuffers(size_t stage) {
    if (channel_state->enabled_uniform_buffer_masks[stage] == 0) [[likely]] {
        return;
    }
    ForEachEnabledBit(channel_state->enabled_uniform_buffer_masks[stage], [&](u32 index) {
        Binding& binding = channel_state->uniform_buffers[stage][index];
        if (binding.buffer_id) {
            // Already updated
            return;
        }
        // Mark as dirty
        if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
            channel_state->dirty_uniform_buffers[stage] |= 1U << index;
        }
        // Resolve buffer
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    });
}

template <class P>
void BufferCache<P>::UpdateStorageBuffers(size_t stage) {
    if (channel_state->enabled_storage_buffers[stage] == 0) [[likely]] {
        return;
    }
    const u64 enabled = channel_state->enabled_storage_buffers[stage];
    const u64 written = channel_state->written_storage_buffers[stage];
    StorageResolveCache& cache = graphics_storage_resolve[stage];
    if (ReuseStorageResolve(cache, channel_state->storage_buffers[stage], enabled, written,
                            graphics_segments)) {
        return;
    }
    const u64 start_epoch = binding_epoch;
    ForEachEnabledBit(enabled, [&](u32 index) {
        // Resolve buffer
        Binding& binding = channel_state->storage_buffers[stage][index];
        const bool is_written = ((written >> index) & 1) != 0;
        ResolveStorageBinding(stage, index, binding, is_written, graphics_segments);
    });
    StoreStorageResolve(cache, channel_state->storage_buffers[stage], enabled, written,
                        graphics_segments, start_epoch);
}

template <class P>
void BufferCache<P>::ResolveStorageBinding(size_t memo_stage, u32 index, Binding& binding,
                                           bool is_written, std::vector<MultiRangeSegment>& pool) {
    StorageBindingMemo& memo = storage_binding_memo[memo_stage][index];
    // FindBuffer waits for the buffer's writes with accurate GPU fences: keep that.
    const bool use_memo = Nx2BufCache::g_storage_memo && BindCacheEnabled() &&
                          !Settings::IsGPUFenceBehaviorAccurate();
    // Before the lookup: a mapping change during it leaves a memo that never matches.
    const u64 map_serial = gpu_memory->MapSerial();
    const size_t as_id = gpu_memory->GetID();
    if (use_memo && memo.epoch == binding_epoch && memo.device_addr == binding.device_addr &&
        memo.gpu_addr == binding.gpu_addr && memo.size == binding.size &&
        memo.is_written == is_written && memo.map_serial == map_serial && memo.as_id == as_id &&
        memo.channel == static_cast<const void*>(channel_state)) {
        // What FindBuffer and a ResolveMultiRangeStorage finding one range gave last time.
        binding.buffer_id = memo.buffer_id;
        binding.segment_first = 0;
        binding.segment_count = 0;
        Nx2BufCache::Note(Nx2BufCache::Stat::StorageMemoHit);
        return;
    }
    binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    ResolveMultiRangeStorage(binding, is_written, pool);
    memo.epoch = 0;
    if (use_memo && binding.segment_count == 0) {
        // binding_epoch after the lookup: a buffer it created is part of what the memo holds,
        // and any later creation or deletion drops the memo.
        memo = StorageBindingMemo{
            .epoch = binding_epoch,
            .map_serial = map_serial,
            .as_id = as_id,
            .channel = channel_state,
            .device_addr = binding.device_addr,
            .gpu_addr = binding.gpu_addr,
            .size = binding.size,
            .is_written = is_written,
            .buffer_id = binding.buffer_id,
        };
    }
}

template <class P>
bool BufferCache<P>::ReuseStorageResolve(StorageResolveCache& cache, std::span<Binding> bindings,
                                         u64 enabled, u64 written,
                                         std::vector<MultiRangeSegment>& pool) {
    // FindBuffer waits for the buffer's writes with accurate GPU fences: keep that.
    if (!BindCacheEnabled() || cache.epoch != binding_epoch || cache.enabled != enabled ||
        cache.written != written || cache.channel != static_cast<const void*>(channel_state) ||
        cache.as_id != gpu_memory->GetID() || cache.map_serial != gpu_memory->MapSerial() ||
        Settings::IsGPUFenceBehaviorAccurate()) {
        return false;
    }
    bool same = true;
    ForEachEnabledBit(enabled, [&](u32 index) {
        const Binding& now = bindings[index];
        const Binding& was = cache.bindings[index];
        same &= now.device_addr == was.device_addr && now.gpu_addr == was.gpu_addr &&
                now.size == was.size;
    });
    if (!same) {
        return false;
    }
    const u32 base = static_cast<u32>(pool.size());
    pool.insert(pool.end(), cache.segments.begin(), cache.segments.end());
    ForEachEnabledBit(enabled, [&](u32 index) {
        Binding& binding = bindings[index];
        const Binding& was = cache.bindings[index];
        binding.buffer_id = was.buffer_id;
        binding.segment_first = was.segment_count != 0 ? base + was.segment_first : 0;
        binding.segment_count = was.segment_count;
    });
    return true;
}

template <class P>
void BufferCache<P>::StoreStorageResolve(StorageResolveCache& cache,
                                         std::span<const Binding> bindings, u64 enabled,
                                         u64 written, const std::vector<MultiRangeSegment>& pool,
                                         u64 start_epoch) {
    cache.epoch = 0;
    // A buffer created or deleted while resolving may have dropped one of these bindings' buffer
    // (DeleteBuffer clears it and the caller resolves again): keep nothing then.
    if (!BindCacheEnabled() || binding_epoch != start_epoch) {
        return;
    }
    cache.segments.clear();
    bool valid = true;
    ForEachEnabledBit(enabled, [&](u32 index) {
        Binding binding = bindings[index];
        if (binding.segment_count != 0) {
            if (binding.segment_first > pool.size() ||
                binding.segment_count > pool.size() - binding.segment_first) {
                valid = false;
                return;
            }
            const auto first = pool.begin() + binding.segment_first;
            binding.segment_first = static_cast<u32>(cache.segments.size());
            cache.segments.insert(cache.segments.end(), first, first + binding.segment_count);
        }
        cache.bindings[index] = binding;
    });
    if (!valid) {
        return;
    }
    cache.enabled = enabled;
    cache.written = written;
    cache.channel = channel_state;
    cache.as_id = gpu_memory->GetID();
    cache.map_serial = gpu_memory->MapSerial();
    cache.epoch = binding_epoch;
}

template <class P>
void BufferCache<P>::UpdateTextureBuffers(size_t stage) {
    if (channel_state->enabled_texture_buffers[stage] == 0) [[likely]] {
        return;
    }
    ForEachEnabledBit(channel_state->enabled_texture_buffers[stage], [&](u32 index) {
        Binding& binding = channel_state->texture_buffers[stage][index];
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    });
}

template <class P>
void BufferCache<P>::UpdateTransformFeedbackBuffers() {
    if (maxwell3d->regs.transform_feedback_enabled == 0) {
        return;
    }
    for (u32 index = 0; index < NUM_TRANSFORM_FEEDBACK_BUFFERS; ++index) {
        UpdateTransformFeedbackBuffer(index);
    }
}

template <class P>
void BufferCache<P>::UpdateTransformFeedbackBuffer(u32 index) {
    const auto& binding = maxwell3d->regs.transform_feedback.buffers[index];
    const GPUVAddr gpu_addr = binding.Address() + binding.start_offset;
    const u32 size = binding.size;
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (binding.enable == 0 || size == 0 || !device_addr) {
        channel_state->transform_feedback_buffers[index] = NULL_BINDING;
        return;
    }
    const BufferId buffer_id = FindBuffer(*device_addr, size, false);
    channel_state->transform_feedback_buffers[index] = Binding{
        .device_addr = *device_addr,
        .size = size,
        .buffer_id = buffer_id,
    };
}

template <class P>
void BufferCache<P>::UpdateComputeUniformBuffers() {
    ForEachEnabledBit(channel_state->enabled_compute_uniform_buffer_mask, [&](u32 index) {
        Binding& binding = channel_state->compute_uniform_buffers[index];
        binding = NULL_BINDING;
        const auto& launch_desc = kepler_compute->launch_description;
        const bool enabled = ((launch_desc.const_buffer_enable_mask >> index) & 1) != 0;
        const GPUVAddr cbuf_addr = enabled ? launch_desc.const_buffer_config[index].Address() : 0;
        const u32 cbuf_size = enabled ? launch_desc.const_buffer_config[index].size : 0;
        // The same constant buffer as this slot's previous dispatch (EDEN_NX2_COMPUTE_BINDCACHE).
        GeometryMemo& memo = compute_uniform_memo[index];
        if (Nx2BufCache::g_compute_bindcache && enabled &&
            ReuseGeometry(memo, cbuf_addr, cbuf_addr + cbuf_size, 2)) {
            binding = memo.binding;
            return;
        }
        if (enabled) {
            const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(cbuf_addr);
            if (device_addr) {
                binding.device_addr = *device_addr;
                binding.size = cbuf_size;
            }
        }
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
        if (enabled) {
            StoreGeometry(memo, cbuf_addr, cbuf_addr + cbuf_size, 2, binding);
        }
    });
}

template <class P>
void BufferCache<P>::UpdateComputeStorageBuffers() {
    const u64 enabled = channel_state->enabled_compute_storage_buffers;
    const u64 written = channel_state->written_compute_storage_buffers;
    if (enabled == 0) {
        return;
    }
    if (ReuseStorageResolve(compute_storage_resolve, channel_state->compute_storage_buffers,
                            enabled, written, compute_segments)) {
        return;
    }
    const u64 start_epoch = binding_epoch;
    ForEachEnabledBit(enabled, [&](u32 index) {
        // Resolve buffer
        Binding& binding = channel_state->compute_storage_buffers[index];
        const bool is_written = ((written >> index) & 1) != 0;
        ResolveStorageBinding(NUM_STAGES, index, binding, is_written, compute_segments);
    });
    StoreStorageResolve(compute_storage_resolve, channel_state->compute_storage_buffers, enabled,
                        written, compute_segments, start_epoch);
}

template <class P>
void BufferCache<P>::UpdateComputeTextureBuffers() {
    ForEachEnabledBit(channel_state->enabled_compute_texture_buffers, [&](u32 index) {
        Binding& binding = channel_state->compute_texture_buffers[index];
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    });
}

template <class P>
void BufferCache<P>::MarkWrittenBuffer(BufferId buffer_id, DAddr device_addr, u32 size, bool needs_sync) {
    if constexpr (!IS_OPENGL) {
        if (needs_sync) {
            Buffer& buffer = slot_buffers[buffer_id];
            buffer.setWriteTick(runtime.CurrentTick());
        }
    }
    // The downloads already taken over this range hold older bytes: when their fences pop
    // they must not write them over this write's result in guest memory, nor drop the range
    // from gpu_modified_ranges (the next fence downloads it instead).
    for (auto& ranges : pending_download_ranges) {
        ranges.Subtract(device_addr, size);
    }
    memory_tracker.MarkRegionAsGpuModified(device_addr, size);
    DemoteHotPages(device_addr, size);
    MarkVertexBuffersStale(device_addr, size);
    Nx2BufCache::Note(Nx2BufCache::Stat::MarkWritten);
    // Dispatches and draws write the same bindings again and again: an interval set add that
    // changes nothing still erases and reinserts nodes (EDEN_NX2_BUFCACHE_MARK).
    if (!Nx2BufCache::g_mark_guards ||
        !Nx2BufCache::RangeCovered(gpu_modified_ranges, device_addr, size)) {
        gpu_modified_ranges.Add(device_addr, size);
        gpu_modified_serial.fetch_add(1, std::memory_order_relaxed);
    } else {
        Nx2BufCache::Note(Nx2BufCache::Stat::MarkAddSkipped);
    }
    QueueFenceDownload(device_addr, size);
    if (Tegra::g_nx2_ampere.load(std::memory_order_relaxed)) {
        Tegra::Nx2MarkGpuWritten(device_addr, size);
        Nx2NoteListWrite(buffer_id, device_addr, size);
    } else {
        nx2_gpu_ranges_unmarked = true;
    }
}

template <class P>
void BufferCache<P>::Nx2NoteListWrite(BufferId buffer_id, DAddr device_addr, u64 size) {
    if constexpr (!IS_OPENGL) {
        if (size == 0 || !Tegra::g_nx2_has_command_list_pages.load(std::memory_order_relaxed)) {
            return;
        }
        constexpr u64 page_bits = Tegra::Nx2WrittenPageBits;
        const u64 last = (std::min<u64>)((device_addr + size - 1) >> page_bits,
                                         Tegra::g_nx2_command_list_pages.size() * 64 - 1);
        for (u64 page = device_addr >> page_bits; page <= last;) {
            // Storage buffers can be large: skip 64 pages at a time where none is a list page.
            const u64 word =
                Tegra::g_nx2_command_list_pages[page / 64].load(std::memory_order_relaxed) >>
                (page % 64);
            if (word == 0) {
                page = (page / 64 + 1) * 64;
                continue;
            }
            if (word & 1) {
                const DAddr page_addr = page << page_bits;
                {
                    std::scoped_lock lock{nx2_list_mutex};
                    Nx2ListPage& entry = nx2_list_pages[page_addr];
                    ++entry.writes;
                    if (!nx2_collect_list_copies && !entry.draw_written) {
                        entry.draw_written = true;
                        LOG_INFO(HW_GPU, "Command list page {:#x} is written by draws too: its "
                                         "lists wait for the work before the last host sync",
                                 page_addr);
                    }
                }
                if (nx2_collect_list_copies) {
                    nx2_list_copy_requests.emplace_back(buffer_id, page_addr);
                }
                if (Tegra::Nx2CmdPageDiag::g_enabled) {
                    Tegra::Nx2CmdPageDiag::NoteWrite(page_addr, device_addr, size,
                                                     !nx2_collect_list_copies);
                }
            }
            ++page;
        }
    }
}

template <class P>
bool BufferCache<P>::Nx2CopyListPages(u64 tick) {
    if constexpr (IS_OPENGL) {
        return false;
    } else {
        if (nx2_list_copy_requests.empty()) {
            return false;
        }
        std::ranges::sort(nx2_list_copy_requests, {}, &std::pair<BufferId, DAddr>::second);
        constexpr DAddr page_size = DAddr{1} << Tegra::Nx2WrittenPageBits;
        bool recorded = false;
        for (size_t i = 0; i < nx2_list_copy_requests.size(); ++i) {
            const auto [buffer_id, page] = nx2_list_copy_requests[i];
            if (i + 1 < nx2_list_copy_requests.size() &&
                nx2_list_copy_requests[i + 1].second == page) {
                continue;
            }
            Buffer& buffer = slot_buffers[buffer_id];
            const DAddr begin = (std::max<DAddr>)(page, buffer.CpuAddr());
            const DAddr end =
                (std::min<DAddr>)(page + page_size, buffer.CpuAddr() + buffer.SizeBytes());
            if (begin >= end) {
                continue;
            }
            Nx2Staging staging = runtime.DownloadStagingBuffer(page_size, true);
            const BufferCopy copy{
                .src_offset = buffer.Offset(begin),
                .dst_offset = staging.offset + (begin - page),
                .size = end - begin,
            };
            buffer.MarkUsage(copy.src_offset, copy.size);
            runtime.CopyBuffer(staging.buffer, buffer, std::span{&copy, 1}, true);
            recorded = true;
            std::scoped_lock lock{nx2_list_mutex};
            Nx2ListPage& entry = nx2_list_pages[page];
            if (entry.copy) {
                nx2_retired_list_copies.emplace_back(entry.copy_tick, std::move(*entry.copy));
            }
            entry.copy = staging;
            entry.copy_writes = entry.writes;
            entry.copy_tick = tick;
            entry.copy_begin = static_cast<u32>(begin - page);
            entry.copy_end = static_cast<u32>(end - page);
        }
        nx2_list_copy_requests.clear();
        std::erase_if(nx2_retired_list_copies, [this](auto& retired) {
            if (!runtime.IsFree(retired.first)) {
                return false;
            }
            runtime.FreeDeferredStagingBuffer(retired.second);
            return true;
        });
        return recorded;
    }
}

template <class P>
auto BufferCache<P>::Nx2ReadListCopy(DAddr device_addr, std::span<u8> out, bool past_sync)
    -> Nx2ListRead {
    if constexpr (IS_OPENGL) {
        return Nx2ListRead::Unknown;
    } else {
        constexpr DAddr page_size = DAddr{1} << Tegra::Nx2WrittenPageBits;
        const DAddr page = device_addr & ~(page_size - 1);
        const u64 offset = device_addr - page;
        if (offset + out.size() > page_size) {
            return Nx2ListRead::Unknown;
        }
        const auto current = [&](const Nx2ListPage& entry) {
            return entry.copy && !entry.draw_written && entry.copy_writes == entry.writes &&
                   offset >= entry.copy_begin && offset + out.size() <= entry.copy_end;
        };
        u64 tick{};
        u64 writes{};
        {
            std::scoped_lock lock{nx2_list_mutex};
            const auto it = nx2_list_pages.find(page);
            if (it == nx2_list_pages.end()) {
                return Nx2ListRead::Unknown;
            }
            const Nx2ListPage& entry = it->second;
            if (entry.draw_written) {
                // EDEN_NX2_CMDPAGE_DRAIN=1: drain and flush for every list on such pages.
                static const bool always_drain = std::getenv("EDEN_NX2_CMDPAGE_DRAIN") != nullptr;
                if (always_drain) {
                    return Nx2ListRead::Unknown;
                }
                if (!past_sync) {
                    return Nx2ListRead::NeedsSync;
                }
                return entry.synced_writes == entry.writes ? Nx2ListRead::Synced
                                                           : Nx2ListRead::Unknown;
            }
            // Pages the CPU writes the lists of (GPU-written once, long before) stay synced.
            if (entry.synced_writes == entry.writes) {
                return Nx2ListRead::Synced;
            }
            if (!current(entry)) {
                return Nx2ListRead::Unknown;
            }
            // The lists after the first of a batch find the copy done: read it at once.
            if (runtime.IsFree(entry.copy_tick)) {
                std::memcpy(out.data(), entry.copy->mapped_span.data() + offset, out.size());
                return Nx2ListRead::Copied;
            }
            tick = entry.copy_tick;
            writes = entry.copy_writes;
        }
        // Submitted with the work that wrote the page (Nx2CopyListPages's caller flushes).
        runtime.WaitSubmittedTick(tick);
        std::scoped_lock lock{nx2_list_mutex};
        const auto it = nx2_list_pages.find(page);
        if (it == nx2_list_pages.end() || !current(it->second) || it->second.copy_tick != tick ||
            it->second.copy_writes != writes) {
            return Nx2ListRead::Unknown;
        }
        std::memcpy(out.data(), it->second.copy->mapped_span.data() + offset, out.size());
        return Nx2ListRead::Copied;
    }
}

template <class P>
void BufferCache<P>::Nx2NoteListSynced(DAddr page) {
    if constexpr (!IS_OPENGL) {
        std::scoped_lock lock{nx2_list_mutex};
        Nx2ListPage& entry = nx2_list_pages[page];
        entry.synced_writes = entry.writes;
    }
}

template <class P>
void BufferCache<P>::Nx2FlushListPages(std::span<const DAddr> pages) {
    constexpr u64 page_size = u64{1} << Tegra::Nx2WrittenPageBits;
    boost::container::small_vector<std::pair<DAddr, u64>, 16> ranges;
    for (const DAddr page : pages) {
        ranges.emplace_back(page, page_size);
    }
    boost::container::small_vector<std::pair<DAddr, u64>, 16> drawn;
    if constexpr (!IS_OPENGL) {
        std::scoped_lock lock{nx2_list_mutex};
        for (const auto& [page, entry] : nx2_list_pages) {
            if (entry.draw_written && entry.synced_writes != entry.writes) {
                ranges.emplace_back(page, page_size);
                drawn.emplace_back(page, entry.writes);
            }
        }
    }
    DownloadRangesMemory(ranges);
    if constexpr (!IS_OPENGL) {
        std::scoped_lock lock{nx2_list_mutex};
        for (const auto& [page, writes] : drawn) {
            const auto it = nx2_list_pages.find(page);
            if (it != nx2_list_pages.end() && it->second.writes == writes) {
                it->second.synced_writes = writes;
            }
        }
    }
}

template <class P>
void BufferCache<P>::Nx2NoteListHostWrite(DAddr device_addr, u64 size) {
    if constexpr (!IS_OPENGL) {
        if (size == 0 || !Tegra::g_nx2_has_command_list_pages.load(std::memory_order_relaxed)) {
            return;
        }
        constexpr u64 page_bits = Tegra::Nx2WrittenPageBits;
        const u64 last = (std::min<u64>)((device_addr + size - 1) >> page_bits,
                                         Tegra::g_nx2_command_list_pages.size() * 64 - 1);
        for (u64 page = device_addr >> page_bits; page <= last; ++page) {
            if (!((Tegra::g_nx2_command_list_pages[page / 64].load(std::memory_order_relaxed) >>
                   (page % 64)) &
                  1)) {
                continue;
            }
            std::scoped_lock lock{nx2_list_mutex};
            const auto it = nx2_list_pages.find(page << page_bits);
            if (it == nx2_list_pages.end()) {
                continue;
            }
            Nx2ListPage& entry = it->second;
            const bool synced = entry.synced_writes == entry.writes;
            ++entry.writes;
            if (synced) {
                entry.synced_writes = entry.writes;
            }
        }
    }
}

template <class P>
BufferId BufferCache<P>::FindBuffer(DAddr device_addr, u32 size, bool sparse_compatible) {
    const u64 page = device_addr >> CACHING_PAGEBITS;
    if (device_addr == 0 || page >= page_table.size()) {
        return NULL_BUFFER_ID;
    }
    const BufferId buffer_id = page_table[page];
    if (buffer_id) {
        Buffer& buffer = slot_buffers[buffer_id];
        if (buffer.IsInBounds(device_addr, size)) {
            const bool should_sync = Settings::IsGPUFenceBehaviorAccurate();
            if (should_sync) {
                SynchronizeBufferWrites(buffer);
            }
            bool usable = true;
            if constexpr (requires { buffer.IsSparseCompatible(); }) {
                if (sparse_compatible && !buffer.IsSparseCompatible()) {
                    usable = false;
                }
            }
            if (usable) {
                return buffer_id;
            }
        }
    }
    return CreateBuffer(device_addr, size, sparse_compatible);
}

template <class P>
void BufferCache<P>::SynchronizeBufferWrites(Buffer& buffer) {
    if constexpr (!IS_OPENGL) {
        const u64 buffer_tick = buffer.getWriteTick();
        if (!runtime.IsFree(buffer_tick)) {
            runtime.Wait(buffer_tick);
        }
    }
}

template <class P>
typename BufferCache<P>::OverlapResult BufferCache<P>::ResolveOverlaps(DAddr device_addr,
                                                                       u32 wanted_size) {
    static constexpr int STREAM_LEAP_THRESHOLD = 16;
    boost::container::small_vector<BufferId, 16> overlap_ids;
    DAddr begin = device_addr;
    DAddr end = device_addr + wanted_size;
    int stream_score = 0;
    bool has_stream_leap = false;
    auto expand_begin = [&](DAddr add_value) {
        static constexpr DAddr min_page = CACHING_PAGESIZE + Core::DEVICE_PAGESIZE;
        if (add_value > begin - min_page) {
            begin = min_page;
            device_addr = Core::DEVICE_PAGESIZE;
            return;
        }
        begin -= add_value;
        device_addr = begin - CACHING_PAGESIZE;
    };
    auto expand_end = [&](DAddr add_value) {
        static constexpr DAddr max_page = 1ULL << Tegra::MaxwellDeviceMemoryManager::AS_BITS;
        if (add_value > max_page - end) {
            end = max_page;
            return;
        }
        end += add_value;
    };
    if (begin == 0) {
        return OverlapResult{
            .ids = std::move(overlap_ids),
            .begin = begin,
            .end = end,
            .has_stream_leap = has_stream_leap,
        };
    }
    for (; device_addr >> CACHING_PAGEBITS < Common::DivCeil(end, CACHING_PAGESIZE);
         device_addr += CACHING_PAGESIZE) {
        const BufferId overlap_id = page_table[device_addr >> CACHING_PAGEBITS];
        if (!overlap_id) {
            continue;
        }
        Buffer& overlap = slot_buffers[overlap_id];
        if (overlap.IsPicked()) {
            continue;
        }
        overlap_ids.push_back(overlap_id);
        overlap.Pick();
        const DAddr overlap_device_addr = overlap.CpuAddr();
        const bool expands_left = overlap_device_addr < begin;
        if (expands_left) {
            begin = overlap_device_addr;
        }
        const DAddr overlap_end = overlap_device_addr + overlap.SizeBytes();
        const bool expands_right = overlap_end > end;
        if (overlap_end > end) {
            end = overlap_end;
        }
        stream_score += overlap.StreamScore();
        if (stream_score > STREAM_LEAP_THRESHOLD && !has_stream_leap) {
            // When this memory region has been joined a bunch of times, we assume it's being used
            // as a stream buffer. Increase the size to skip constantly recreating buffers.
            has_stream_leap = true;
            if (expands_right) {
                expand_begin(CACHING_PAGESIZE * 128);
            }
            if (expands_left) {
                expand_end(CACHING_PAGESIZE * 128);
            }
        }
    }
    return OverlapResult{
        .ids = std::move(overlap_ids),
        .begin = begin,
        .end = end,
        .has_stream_leap = has_stream_leap,
    };
}

template <class P>
void BufferCache<P>::JoinOverlap(BufferId new_buffer_id, BufferId overlap_id,
                                 bool accumulate_stream_score) {
    Buffer& new_buffer = slot_buffers[new_buffer_id];
    Buffer& overlap = slot_buffers[overlap_id];
    if (accumulate_stream_score) {
        new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
    }
    boost::container::small_vector<BufferCopy, 10> copies;
    const size_t dst_base_offset = overlap.CpuAddr() - new_buffer.CpuAddr();
    copies.push_back(BufferCopy{
        .src_offset = 0,
        .dst_offset = dst_base_offset,
        .size = overlap.SizeBytes(),
    });
    new_buffer.MarkUsage(copies[0].dst_offset, copies[0].size);
    runtime.CopyBuffer(new_buffer, overlap, copies, true);
#ifdef YUZU_LEGACY
    if (immediately_free)
        runtime.Finish();
#endif
    DeleteBuffer(overlap_id, true);
}

template <class P>
BufferId BufferCache<P>::CreateBuffer(DAddr device_addr, u32 wanted_size,
                                      bool sparse_compatible) {
    DAddr device_addr_end = Common::AlignUp(device_addr + wanted_size, CACHING_PAGESIZE);
    device_addr = Common::AlignDown(device_addr, CACHING_PAGESIZE);
    wanted_size = static_cast<u32>(device_addr_end - device_addr);
    const OverlapResult overlap = ResolveOverlaps(device_addr, wanted_size);
    const u32 size = static_cast<u32>(overlap.end - overlap.begin);
    const BufferId new_buffer_id =
        slot_buffers.insert(runtime, overlap.begin, size, sparse_compatible);
    ++binding_epoch;
    auto& new_buffer = slot_buffers[new_buffer_id];
    const size_t size_bytes = new_buffer.SizeBytes();
    runtime.ClearBuffer(new_buffer, 0, size_bytes, 0);
    new_buffer.MarkUsage(0, size_bytes);
    for (const BufferId overlap_id : overlap.ids) {
        JoinOverlap(new_buffer_id, overlap_id, !overlap.has_stream_leap);
    }
    Register(new_buffer_id);
    TouchBuffer(new_buffer, new_buffer_id);
    return new_buffer_id;
}

template <class P>
void BufferCache<P>::Register(BufferId buffer_id) {
    ChangeRegister<true>(buffer_id);
}

template <class P>
void BufferCache<P>::Unregister(BufferId buffer_id) {
    ChangeRegister<false>(buffer_id);
}

template <class P>
template <bool insert>
void BufferCache<P>::ChangeRegister(BufferId buffer_id) {
    Buffer& buffer = slot_buffers[buffer_id];
    const auto size = buffer.SizeBytes();
    if (insert) {
        buffer_bytes += Common::AlignUp(size, 1024);
        total_used_memory += Common::AlignUp(size, 1024);
        buffer.setLRUID(lru_cache.Insert(buffer_id, frame_tick));
    } else {
        buffer_bytes -= Common::AlignUp(size, 1024);
        total_used_memory -= Common::AlignUp(size, 1024);
        lru_cache.Free(buffer.getLRUID());
    }
    const DAddr device_addr_begin = buffer.CpuAddr();
    const DAddr device_addr_end = device_addr_begin + size;
    const u64 page_begin = device_addr_begin / CACHING_PAGESIZE;
    const u64 page_end = Common::DivCeil(device_addr_end, CACHING_PAGESIZE);
    for (u64 page = page_begin; page != page_end; ++page) {
        if constexpr (insert) {
            page_table[page] = buffer_id;
        } else {
            page_table[page] = BufferId{};
        }
    }
}

template <class P>
void BufferCache<P>::TouchBuffer(Buffer& buffer, BufferId buffer_id) noexcept {
    // The buffer's own field instead of its LRU node (a cache miss) for every bind of the frame.
    if (buffer_id != NULL_BUFFER_ID && buffer.lru_touched_tick != frame_tick) {
        buffer.lru_touched_tick = frame_tick;
        lru_cache.Touch(buffer.getLRUID(), frame_tick);
    }
}

template <class P>
bool BufferCache<P>::SynchronizeBuffer(Buffer& buffer, DAddr device_addr, u32 size) {
    // An upload over a clean word of the tracker (word_manager.h) finds and changes nothing:
    // walk only the dirty ones. Verify walks every word and checks the clean words.
    const bool skip_clean = !Nx2Perf::Verify();
    if (Nx2Perf::Verify()) {
        static std::atomic<u32> logged{};
        if (!memory_tracker.CleanWordsHold(device_addr, size) &&
            logged.fetch_add(1, std::memory_order_relaxed) < 64) {
            LOG_ERROR(HW_GPU, "Nx2Perf draw-binds: clean tracker words wrong near {:#x}+{:#x}",
                      device_addr, size);
        }
    }
    // The same range left clean by an earlier sync, and no CPU write marked in its buffer since:
    // still clean (EDEN_NX2_SYNC_SKIP). Draws sync the same multi-megabyte storage bindings
    // and vertex buffers again and again, each a walk over every tracker manager they cover.
    SyncSkip* skip_entry = nullptr;
    // A range over few 256 KiB regions is tied to the CPU-dirty marks in them, not to those in
    // its whole buffer: OoT's inline constant buffer writes mark its 100 MiB heap buffer before
    // nearly every draw (EDEN_NX2_SKIP_REGIONS).
    const bool by_regions =
        Nx2BufCache::g_skip_regions && SkipRegionCount(device_addr, size) <= MAX_SKIP_REGIONS;
    if (Nx2BufCache::g_sync_skip && skip_clean) {
        skip_entry = &sync_skip[((device_addr ^ (u64{size} << 40)) * 0x9E3779B97F4A7C15ULL) >>
                                (64 - SYNC_SKIP_BITS)];
        if (skip_entry->buffer == &buffer && skip_entry->device_addr == device_addr &&
            skip_entry->size == size && skip_entry->epoch == cpu_dirty_epoch &&
            skip_entry->binding_epoch == binding_epoch &&
            (by_regions ? skip_entry->region_sum == SkipRegionSum(device_addr, size)
                        : skip_entry->gen == buffer.cpu_dirty_gen)) {
            if (skip_entry->hot_mask == 0) {
                Nx2BufCache::Note(Nx2BufCache::Stat::SyncSkipHit);
                return true;
            }
            // The rest of the range is clean: only its hot pages (never clean) need a look.
            if (const std::optional<bool> synced =
                    SyncSkippedHotPages(buffer, device_addr, skip_entry->hot_mask)) {
                Nx2BufCache::Note(Nx2BufCache::Stat::SyncSkipHotHit);
                return *synced;
            }
        }
    }
    // The entry is gone (a CPU-dirty mark fell in a region or the buffer), but the mark may lie
    // outside this range: when no page of it is CPU-dirty or unprotected, the walk below finds
    // nothing to upload and changes no tracker state, and the range is still clean
    // (EDEN_NX2_SYNC_PRECHECK). Read only, page granular, against the walk's word granularity.
    if (Nx2BufCache::g_sync_precheck && skip_clean &&
        !memory_tracker.IsRegionUploadDirty(device_addr, size)) {
        if (skip_entry != nullptr) {
            *skip_entry = SyncSkip{
                .buffer = &buffer,
                .device_addr = device_addr,
                .size = size,
                .gen = buffer.cpu_dirty_gen,
                .epoch = cpu_dirty_epoch,
                .binding_epoch = binding_epoch,
                .region_sum = by_regions ? SkipRegionSum(device_addr, size) : 0,
                .hot_mask = 0,
            };
        }
        Nx2BufCache::Note(Nx2BufCache::Stat::SyncPrecheckClean);
        return true;
    }
    Nx2BufCache::Note(Nx2BufCache::Stat::SyncWalk);
    const auto for_each_upload_range = [&](auto&& func) {
        if (skip_clean) {
            memory_tracker.ForEachDirtyUploadRange(device_addr, size, func);
        } else {
            memory_tracker.ForEachUploadRange(device_addr, size, func);
        }
    };
    // The range is clean now unless a hot page (always dirty) lies in it. Such a range keeps its
    // entry with the hot pages listed (EDEN_NX2_HOT_SYNC_SKIP) when they all fit the mask, all
    // synced from their copies and none was promoted or demoted.
    bool range_has_hot = false;
    bool hot_skip_ok = Nx2BufCache::g_hot_sync_skip;
    u64 hot_mask = 0;
    const DAddr hot_base = Common::AlignDown(device_addr, DEVICE_PAGESIZE);
    const auto store_skip = [&] {
        if (skip_entry == nullptr || (range_has_hot && !hot_skip_ok)) {
            return;
        }
        *skip_entry = SyncSkip{
            .buffer = &buffer,
            .device_addr = device_addr,
            .size = size,
            .gen = buffer.cpu_dirty_gen,
            .epoch = cpu_dirty_epoch,
            .binding_epoch = binding_epoch,
            .region_sum = by_regions ? SkipRegionSum(device_addr, size) : 0,
            .hot_mask = range_has_hot ? hot_mask : 0,
        };
    };
#ifdef __ANDROID__
    const bool use_optimized_vertex_buffers = Settings::values.use_optimized_vertex_buffers.GetValue();
#else
    constexpr bool use_optimized_vertex_buffers = true;
#endif

    if (use_optimized_vertex_buffers) {
        upload_copies.clear();
        u64 total_size_bytes = 0;
        u64 largest_copy = 0;
        const DAddr buffer_start = buffer.cpu_addr_cached;
        const auto add_upload = [&](DAddr start, DAddr end) {
            if (start == end) return;
            const u64 range_size = end - start;
            upload_copies.push_back(BufferCopy{
                .src_offset = total_size_bytes,
                .dst_offset = start - buffer_start,
                .size = range_size,
            });
            total_size_bytes += range_size;
            largest_copy = (std::max)(largest_copy, range_size);
        };
        const auto add_dirty = [&](u64 device_addr_out, u64 range_size) {
            DAddr upload_start = device_addr_out;
            gpu_modified_ranges.ForEachInRange(device_addr_out, range_size, [&](DAddr gpu_start, DAddr gpu_end) {
                add_upload(upload_start, gpu_start);
                upload_start = gpu_end;
            });
            add_upload(upload_start, device_addr_out + range_size);
        };
        bool hot_uploaded = false;
        if (Nx2BufCache::g_hot_pages) {
            hot_changed_pages.clear();
            hot_promote_pages.clear();
            hot_demote_pages.clear();
            memory_tracker.ForEachUploadRange(
                device_addr, size, skip_clean,
                [&](u64 device_addr_out, u64 range_size) {
                    add_dirty(device_addr_out, range_size);
                    NoteDirtyPages(device_addr_out, range_size);
                },
                [&](u64 device_addr_out, u64 range_size) {
                    range_has_hot = true;
                    for (DAddr page = device_addr_out; page < device_addr_out + range_size;
                         page += DEVICE_PAGESIZE) {
                        const u64 bit = (page - hot_base) / DEVICE_PAGESIZE;
                        if (page >= hot_base && bit < 64) {
                            hot_mask |= u64{1} << bit;
                        } else {
                            hot_skip_ok = false;
                        }
                        if (!SyncHotPage(page)) {
                            hot_skip_ok = false;
                            add_dirty(page, DEVICE_PAGESIZE);
                        }
                    }
                });
            if (!hot_promote_pages.empty() || !hot_demote_pages.empty()) {
                // Promotions mark CPU-dirty and demoted pages are still dirty: walk next time.
                hot_skip_ok = false;
                ApplyHotPageChanges();
            }
            if (!hot_changed_pages.empty()) {
                UploadHotPages(buffer);
                hot_uploaded = true;
            }
        } else {
            for_each_upload_range(add_dirty);
        }
        if (!hot_promote_pages.empty()) {
            range_has_hot = true;
        }
        if (total_size_bytes == 0) {
            store_skip();
            return !hot_uploaded;
        }
        Nx2BufCache::Note(Nx2BufCache::Stat::SyncUpload);
        Nx2BufCache::Note(Nx2BufCache::Stat::SyncUploadBytes, total_size_bytes);
        const std::span<BufferCopy> copies_span(upload_copies.data(), upload_copies.size());
        bool uploaded = false;
        if constexpr (USE_MEMORY_MAPS && USE_MEMORY_MAPS_FOR_UPLOADS) {
            if (Nx2BufCache::g_bank_sync_refresh && !bank_shadows.empty()) {
                MappedUploadMemoryRefreshingBanks(buffer, total_size_bytes, copies_span);
                uploaded = true;
            }
        }
        if (!uploaded) {
            if (!bank_shadows.empty()) {
                for (const BufferCopy& copy : upload_copies) {
                    InvalidateBankShadows(buffer_start + copy.dst_offset, copy.size);
                }
            }
            UploadMemory(buffer, total_size_bytes, largest_copy, copies_span);
        }
        // Mapped uploads carry their own barriers (or run in the upload command buffer, which
        // ends with one); another barrier here only ended the render pass (~50 times a frame in
        // Pokopia).
        if constexpr (!USE_MEMORY_MAPS_FOR_UPLOADS) {
            any_buffer_uploaded = true;
        }
        store_skip();
        return false;
    } else {
        boost::container::small_vector<BufferCopy, 4> copies;
        u64 total_size_bytes = 0;
        u64 largest_copy = 0;
        DAddr buffer_start = buffer.CpuAddr();
        for_each_upload_range([&](u64 device_addr_out, u64 range_size) {
            copies.push_back(BufferCopy{
                .src_offset = total_size_bytes,
                .dst_offset = device_addr_out - buffer_start,
                .size = range_size,
            });
            total_size_bytes += range_size;
            largest_copy = (std::max)(largest_copy, range_size);
        });
        if (total_size_bytes == 0) {
            return true;
        }
        for (const BufferCopy& copy : copies) {
            InvalidateBankShadows(buffer_start + copy.dst_offset, copy.size);
        }
        const std::span<BufferCopy> copies_span(copies.data(), copies.size());
        UploadMemory(buffer, total_size_bytes, largest_copy, copies_span);
        return false;
    }
}

template <class P>
void BufferCache<P>::UploadMemory(Buffer& buffer, u64 total_size_bytes, u64 largest_copy,
                                  std::span<BufferCopy> copies) {
    if constexpr (USE_MEMORY_MAPS_FOR_UPLOADS) {
        MappedUploadMemory(buffer, total_size_bytes, copies);
    } else {
        ImmediateUploadMemory(buffer, largest_copy, copies);
    }
}

template <class P>
void BufferCache<P>::ImmediateUploadMemory([[maybe_unused]] Buffer& buffer,
                                           [[maybe_unused]] u64 largest_copy,
                                           [[maybe_unused]] std::span<const BufferCopy> copies) {
    if constexpr (!USE_MEMORY_MAPS_FOR_UPLOADS) {
        std::span<u8> immediate_buffer;
        for (const BufferCopy& copy : copies) {
            std::span<const u8> upload_span;
            const DAddr device_addr = buffer.CpuAddr() + copy.dst_offset;
            if (IsRangeGranular(device_addr, copy.size)) {
                auto* const ptr = device_memory.GetPointer<u8>(device_addr);
                if (ptr != nullptr) {
                    upload_span = std::span(ptr, copy.size);
                }
            } else {
                if (immediate_buffer.empty()) {
                    immediate_buffer = ImmediateBuffer(largest_copy);
                }
                device_memory.ReadBlockUnsafe(device_addr, immediate_buffer.data(), copy.size);
                upload_span = immediate_buffer.subspan(0, copy.size);
            }
            buffer.ImmediateUpload(copy.dst_offset, upload_span);
        }
    }
}

template <class P>
void BufferCache<P>::MappedUploadMemory([[maybe_unused]] Buffer& buffer,
                                        [[maybe_unused]] u64 total_size_bytes,
                                        [[maybe_unused]] std::span<BufferCopy> copies) {
    if constexpr (USE_MEMORY_MAPS) {
        auto upload_staging = runtime.UploadStagingBuffer(total_size_bytes);
        const std::span<u8> staging_pointer = upload_staging.mapped_span;
        for (BufferCopy& copy : copies) {
            u8* const src_pointer = staging_pointer.data() + copy.src_offset;
            const DAddr device_addr = buffer.CpuAddr() + copy.dst_offset;
            device_memory.ReadBlockUnsafe(device_addr, src_pointer, copy.size);
            // Large uploads are mostly geometry the host GPU now has its own copy of: TickFrame
            // marks their guest memory cold (see TextureCache::RefreshContents).
            if (copy.size >= 256_KiB) {
                uploaded_buffer_ranges.emplace_back(device_addr, copy.size);
            }
            // Apply the staging offset
            copy.src_offset += upload_staging.offset;
        }
        const bool can_reorder = runtime.CanReorderUpload(buffer, copies);
        runtime.CopyBuffer(buffer, upload_staging.buffer, copies, true, can_reorder);
    }
}

template <class P>
bool BufferCache<P>::InlineMemory(DAddr dest_address, size_t copy_size,
                                  std::span<const u8> inlined_buffer, bool* pages_clean) {
    // Pages without GPU-written data are the common case: test that before the buffer lookup.
    DAddr aligned_start = Common::AlignDown(dest_address, DEVICE_PAGESIZE);
    DAddr aligned_end = Common::AlignUp(dest_address + copy_size, DEVICE_PAGESIZE);
    const bool clean = !IsRegionGpuModified(aligned_start, aligned_end - aligned_start);
    if (pages_clean != nullptr) {
        *pages_clean = clean;
    }
    if (clean) {
        return false;
    }
    const bool is_dirty = IsRegionRegistered(dest_address, copy_size);
    if (!is_dirty) {
        return false;
    }

    InlineMemoryImplementation(dest_address, copy_size, inlined_buffer);

    return true;
}

template <class P>
void BufferCache<P>::InlineMemoryImplementation(DAddr dest_address, size_t copy_size,
                                                std::span<const u8> inlined_buffer) {
    ClearDownload(dest_address, copy_size);
    gpu_modified_ranges.Subtract(dest_address, copy_size);
    MarkVertexBuffersStale(dest_address, copy_size);

    BufferId buffer_id = FindBuffer(dest_address, static_cast<u32>(copy_size), false);
    auto& buffer = slot_buffers[buffer_id];
    SynchronizeBuffer(buffer, dest_address, static_cast<u32>(copy_size));
    InvalidateBankShadows(dest_address, copy_size);
    InvalidateHotShadows(dest_address, copy_size);

    if constexpr (USE_MEMORY_MAPS_FOR_UPLOADS) {
        auto upload_staging = runtime.UploadStagingBuffer(copy_size);
        std::array copies{BufferCopy{
            .src_offset = upload_staging.offset,
            .dst_offset = buffer.Offset(dest_address),
            .size = copy_size,
        }};
        u8* const src_pointer = upload_staging.mapped_span.data();
        std::memcpy(src_pointer, inlined_buffer.data(), copy_size);
        const bool can_reorder = runtime.CanReorderUpload(buffer, copies);
        runtime.CopyBuffer(buffer, upload_staging.buffer, copies, true, can_reorder);
    } else {
        buffer.ImmediateUpload(buffer.Offset(dest_address), inlined_buffer.first(copy_size));
    }
}

template <class P>
void BufferCache<P>::UploadNativeBank(Buffer& buffer, DAddr device_addr, u64 upload_size) {
    if constexpr (USE_MEMORY_MAPS && USE_MEMORY_MAPS_FOR_UPLOADS) {
        // Ocarina binds ~a hundred programs' banks a frame; they rarely change.
        if (bank_shadows.size() >= 256 && !bank_shadows.contains(device_addr)) {
            if (Nx2BufCache::g_bank_lru) {
                // Only the least recently compared bank uploads whole again, not every bank.
                auto victim = bank_shadows.begin();
                for (auto it = bank_shadows.begin(); it != bank_shadows.end(); ++it) {
                    if (it->second->compared_frame < victim->second->compared_frame) {
                        victim = it;
                    }
                }
                bank_shadows.erase(victim);
                Nx2BufCache::Note(Nx2BufCache::Stat::BankShadowDrop);
            } else {
                Nx2BufCache::Note(Nx2BufCache::Stat::BankShadowDrop, bank_shadows.size());
                bank_shadows.clear();
            }
        }
        std::unique_ptr<BankShadow>& slot = bank_shadows[device_addr];
        if (!slot) {
            slot = std::make_unique<BankShadow>();
        }
        BankShadow& shadow = *slot;
        shadow.compared_frame = frame_tick;
        Nx2BufCache::Note(Nx2BufCache::Stat::BankCompare);
        const u64 base_offset = buffer.Offset(device_addr);
        boost::container::small_vector<BufferCopy, 17> copies;
        u64 total_size = 0;
        u32 page = 0;
        // As hot pages: guest bytes over a page the GPU wrote may be older than its results there
        // (not, or no longer, in gpu_modified_ranges). Such pages keep the usual tracking.
        const bool bank_gpu_written = gpu_modified_ranges.Intersects(device_addr, upload_size) ||
                                      Tegra::Nx2MayBeGpuWritten(device_addr, upload_size);
        for (u64 offset = 0; offset < upload_size && page < 32; ++page) {
            const DAddr address = device_addr + offset;
            const u64 length = (std::min<u64>)(
                Core::DEVICE_PAGESIZE - (address & Core::DEVICE_PAGEMASK), upload_size - offset);
            const u32 bit = 1U << page;
            const u8* const guest = device_memory.template GetPointer<u8>(address);
            u8* const saved = shadow.bytes.data() + offset;
            const bool unusable =
                guest == nullptr ||
                (bank_gpu_written && (gpu_modified_ranges.Intersects(address, length) ||
                                      Tegra::Nx2MayBeGpuWritten(address, length)));
            if (!unusable && (shadow.valid & bit) != 0 && Nx2Profile::Enabled()) [[unlikely]] {
                Nx2BufCache::Note(Nx2BufCache::Stat::BankComparePages);
                if (!bank_pages_this_frame.insert(address & ~DAddr{Core::DEVICE_PAGEMASK})) {
                    Nx2BufCache::Note(Nx2BufCache::Stat::BankPageRepeat);
                }
            }
            if (unusable) {
                shadow.valid &= ~bit;
            } else if ((shadow.valid & bit) == 0 ||
                       !Nx2BufCache::BytesEqual(saved, guest, length)) {
                std::memcpy(saved, guest, length);
                shadow.valid |= bit;
                if (!copies.empty() &&
                    copies.back().dst_offset + copies.back().size == base_offset + offset) {
                    copies.back().size += length;
                } else {
                    copies.push_back(BufferCopy{
                        .src_offset = total_size,
                        .dst_offset = base_offset + offset,
                        .size = length,
                    });
                }
                total_size += length;
            }
            offset += length;
        }
        if (total_size == 0) {
            return;
        }
        // From the shadow, not guest memory again: the host buffer then holds exactly what the
        // shadow says, even if the guest writes the bank meanwhile.
        auto upload_staging = runtime.UploadStagingBuffer(total_size);
        for (BufferCopy& copy : copies) {
            std::memcpy(upload_staging.mapped_span.data() + copy.src_offset,
                        shadow.bytes.data() + (copy.dst_offset - base_offset), copy.size);
            copy.src_offset += upload_staging.offset;
        }
        const std::span<BufferCopy> copies_span(copies.data(), copies.size());
        const bool can_reorder = runtime.CanReorderUpload(buffer, copies_span);
        runtime.CopyBuffer(buffer, upload_staging.buffer, copies_span, true, can_reorder);
        for (const BufferCopy& copy : copies) {
            const u64 bank_offset = copy.dst_offset - base_offset;
            RefreshHotShadows(device_addr + bank_offset, shadow.bytes.data() + bank_offset,
                              copy.size);
        }
        Nx2BufCache::Note(Nx2BufCache::Stat::BankUpload);
        Nx2BufCache::Note(Nx2BufCache::Stat::BankUploadBytes, total_size);
    }
}

template <class P>
void BufferCache<P>::NoteCpuDirty(DAddr device_addr, u64 size) {
    ForEachBufferInRange(device_addr, size, [](BufferId, Buffer& buffer) { ++buffer.cpu_dirty_gen; });
    BumpSkipRegions(device_addr, size);
}

template <class P>
std::optional<bool> BufferCache<P>::SyncSkippedHotPages(Buffer& buffer, DAddr device_addr,
                                                        u64 hot_mask) {
    hot_changed_pages.clear();
    hot_promote_pages.clear();
    hot_demote_pages.clear();
    const DAddr base = Common::AlignDown(device_addr, DEVICE_PAGESIZE);
    for (u64 mask = hot_mask; mask != 0; mask &= mask - 1) {
        const DAddr page = base + static_cast<DAddr>(std::countr_zero(mask)) * DEVICE_PAGESIZE;
        if (!SyncHotPage(page)) {
            // No longer hot (or unmapped, or GPU-written): the walk uploads it from guest memory.
            // The copies refreshed above are not on the host yet: the walk must upload them too.
            for (const DAddr changed : hot_changed_pages) {
                if (const auto it = hot_pages.find(changed); it != hot_pages.end()) {
                    it->second->valid = false;
                }
            }
            return std::nullopt;
        }
    }
    if (!hot_demote_pages.empty()) {
        // Unchanged long enough: protected again, and still CPU-dirty for the next sync's walk.
        ApplyHotPageChanges();
    }
    if (hot_changed_pages.empty()) {
        return true;
    }
    UploadHotPages(buffer);
    return false;
}

template <class P>
void BufferCache<P>::NoteDirtyPages(DAddr device_addr, u64 size) {
    // Large ranges are first uploads and buffer rebuilds, not pages rewritten between syncs.
    if (size > 16 * DEVICE_PAGESIZE) {
        return;
    }
    for (DAddr page = device_addr; page < device_addr + size; page += DEVICE_PAGESIZE) {
        PageHeat& heat = page_heat[page];
        // Demoted for costing more compares than it saved faults: not again for a while.
        if (Nx2BufCache::g_hot_demote && frame_tick < heat.no_promote_until) {
            continue;
        }
        // Each frame without a dirty sync cools the page by one: a page dirty at the syncs of
        // every frame (or many times in one frame) heats up, one dirty every other frame doesn't.
        const u64 idle = frame_tick > heat.frame + 1 ? frame_tick - heat.frame - 1 : 0;
        heat.heat = idle >= heat.heat ? 0 : heat.heat - static_cast<u32>(idle);
        heat.frame = frame_tick;
        if (++heat.heat >= HOT_PAGE_HEAT) {
            heat.heat = 0;
            hot_promote_pages.push_back(page);
        }
    }
}

template <class P>
bool BufferCache<P>::SyncHotPage(DAddr page) {
    const auto it = hot_pages.find(page);
    if (it == hot_pages.end()) {
        return false;
    }
    HotPage& hot = *it->second;
    // A page that came to hold a command list, or that the GPU wrote, after promotion needs its
    // write notifications (see ApplyHotPageChanges).
    if (gpu_modified_ranges.Intersects(page, DEVICE_PAGESIZE) ||
        Tegra::Nx2IsCommandListPage(page) || Tegra::Nx2MayBeGpuWritten(page, DEVICE_PAGESIZE)) {
        hot_demote_pages.push_back(page);
        return false;
    }
    const u8* const guest = device_memory.template GetPointer<u8>(page);
    if (guest == nullptr) {
        hot.valid = false;
        return false;
    }
    if (hot.valid && Nx2BufCache::BytesEqual(hot.bytes.data(), guest, DEVICE_PAGESIZE)) {
        Nx2BufCache::Note(Nx2BufCache::Stat::HotSame);
        ++hot.unchanged;
        if (Nx2BufCache::g_hot_demote) {
            // Synced many times per change (a page rewritten once a frame in a bank bound at
            // every draw): the compares cost more than the fault and re-protection a change
            // costs a protected page. Protect it again and keep it out for a while.
            if (hot.unchanged >= HOT_DEMOTE_UNCHANGED) {
                hot_demote_pages.push_back(page);
                page_heat[page].no_promote_until = frame_tick + HOT_COOLDOWN_FRAMES;
                Nx2BufCache::Note(Nx2BufCache::Stat::HotCostDemote);
            }
        } else if (hot.unchanged >= 64 && frame_tick >= hot.changed_frame + 4) {
            // Unchanged for many syncs over several frames: protect it again.
            hot_demote_pages.push_back(page);
        }
        return true;
    }
    std::memcpy(hot.bytes.data(), guest, DEVICE_PAGESIZE);
    hot.valid = true;
    hot.unchanged = 0;
    hot.changed_frame = frame_tick;
    hot_changed_pages.push_back(page);
    return true;
}

template <class P>
void BufferCache<P>::ApplyHotPageChanges() {
    for (const DAddr page : hot_demote_pages) {
        if (hot_pages.erase(page) != 0) {
            memory_tracker.MarkRegionHot(page, DEVICE_PAGESIZE, false);
            // Still CPU-dirty, now as a plain page: SYNC_SKIP entries over it must walk again.
            NoteCpuDirty(page, DEVICE_PAGESIZE);
            Nx2BufCache::Note(Nx2BufCache::Stat::HotDemote);
        }
    }
    for (const DAddr page : hot_promote_pages) {
        if (hot_pages.size() >= MAX_HOT_PAGES) {
            break;
        }
        // GPU-written pages and command list pages need the write notifications. So do pages
        // the GPU has ever written (the sticky Ampere bitmap): a hot upload sends the whole guest
        // page, and on a page a dispatch writes too that put stale guest bytes over GPU results
        // that were not (or no longer) in gpu_modified_ranges. OoT's 8 KiB compute storage ring
        // (CPU-written parameters next to GPU-written state) came out 8% too bright that way.
        if (hot_pages.contains(page) || gpu_modified_ranges.Intersects(page, DEVICE_PAGESIZE) ||
            Tegra::Nx2IsCommandListPage(page) || Tegra::Nx2MayBeGpuWritten(page, DEVICE_PAGESIZE)) {
            continue;
        }
        hot_pages.emplace(page, std::make_unique<HotPage>());
        memory_tracker.MarkRegionHot(page, DEVICE_PAGESIZE, true);
        // Unprotected and dirty from now on; this sync already uploads the page.
        memory_tracker.MarkRegionAsCpuModified(page, DEVICE_PAGESIZE);
        NoteCpuDirty(page, DEVICE_PAGESIZE);
        Nx2BufCache::Note(Nx2BufCache::Stat::HotPromote);
    }
}

template <class P>
void BufferCache<P>::UploadHotPages(Buffer& buffer) {
    const DAddr buffer_start = buffer.CpuAddr();
    boost::container::small_vector<BufferCopy, 8> copies;
    u64 total_size = 0;
    for (const DAddr page : hot_changed_pages) {
        const u64 dst_offset = page - buffer_start;
        if (!copies.empty() && copies.back().dst_offset + copies.back().size == dst_offset) {
            copies.back().size += DEVICE_PAGESIZE;
        } else {
            copies.push_back(BufferCopy{
                .src_offset = total_size,
                .dst_offset = dst_offset,
                .size = DEVICE_PAGESIZE,
            });
        }
        total_size += DEVICE_PAGESIZE;
    }
    if constexpr (USE_MEMORY_MAPS_FOR_UPLOADS) {
        auto upload_staging = runtime.UploadStagingBuffer(total_size);
        u64 src_offset = 0;
        for (const DAddr page : hot_changed_pages) {
            std::memcpy(upload_staging.mapped_span.data() + src_offset,
                        hot_pages[page]->bytes.data(), DEVICE_PAGESIZE);
            src_offset += DEVICE_PAGESIZE;
        }
        for (BufferCopy& copy : copies) {
            copy.src_offset += upload_staging.offset;
        }
        const std::span<BufferCopy> copies_span(copies.data(), copies.size());
        const bool can_reorder = runtime.CanReorderUpload(buffer, copies_span);
        runtime.CopyBuffer(buffer, upload_staging.buffer, copies_span, true, can_reorder);
    } else {
        for (const DAddr page : hot_changed_pages) {
            buffer.ImmediateUpload(page - buffer_start,
                                   std::span<const u8>(hot_pages[page]->bytes.data(),
                                                       DEVICE_PAGESIZE));
        }
        any_buffer_uploaded = true;
    }
    for (const DAddr page : hot_changed_pages) {
        RefreshBankShadows(page, hot_pages[page]->bytes.data());
        // What WriteMemory does for a noticed CPU write: derived vertex data is rebuilt.
        MarkVertexBuffersStale(page, DEVICE_PAGESIZE);
    }
    if (stale_vertex_buffers_mask != 0) {
        RefreshRepackedVertexBuffers();
    }
    Nx2BufCache::Note(Nx2BufCache::Stat::HotUpload, hot_changed_pages.size());
    Nx2BufCache::Note(Nx2BufCache::Stat::SyncUpload);
    Nx2BufCache::Note(Nx2BufCache::Stat::SyncUploadBytes, total_size);
}

template <class P>
void BufferCache<P>::DemoteHotPages(DAddr device_addr, u64 size) {
    if (hot_pages.empty() || size == 0) {
        return;
    }
    boost::container::small_vector<std::pair<DAddr, u64>, 4> ranges;
    memory_tracker.ForEachHotRange(device_addr, size, [&](u64 begin, u64 length) {
        ranges.emplace_back(begin, length);
    });
    for (const auto& [begin, length] : ranges) {
        for (DAddr page = begin; page < begin + length; page += DEVICE_PAGESIZE) {
            if (hot_pages.erase(page) != 0) {
                Nx2BufCache::Note(Nx2BufCache::Stat::HotDemote);
            }
        }
        memory_tracker.MarkRegionHot(begin, length, false);
    }
}

template <class P>
void BufferCache<P>::InvalidateHotShadows(DAddr device_addr, u64 size) {
    if (hot_pages.empty() || size == 0) {
        return;
    }
    const DAddr begin = Common::AlignDown(device_addr, DEVICE_PAGESIZE);
    const DAddr end = Common::AlignUp(device_addr + size, DEVICE_PAGESIZE);
    if ((end - begin) / DEVICE_PAGESIZE <= hot_pages.size()) {
        for (DAddr page = begin; page < end; page += DEVICE_PAGESIZE) {
            if (const auto it = hot_pages.find(page); it != hot_pages.end()) {
                it->second->valid = false;
            }
        }
        return;
    }
    for (auto& [page, hot] : hot_pages) {
        if (page >= begin && page < end) {
            hot->valid = false;
        }
    }
}

template <class P>
void BufferCache<P>::RefreshBankShadows(DAddr page, const u8* bytes) {
    if (bank_shadows.empty()) {
        return;
    }
    if (!Nx2BufCache::g_shadow_refresh) {
        InvalidateBankShadows(page, DEVICE_PAGESIZE);
        return;
    }
    constexpr DAddr bank_span = 0x10000;
    auto it = bank_shadows.lower_bound(page > bank_span ? page - bank_span : 0);
    for (; it != bank_shadows.end() && it->first < page + DEVICE_PAGESIZE; ++it) {
        const DAddr bank = it->first;
        if (bank + bank_span <= page) {
            continue;
        }
        BankShadow& shadow = *it->second;
        // UploadNativeBank splits a bank at device page boundaries: a device page wholly inside
        // the bank is one shadow page, at the same offset.
        if (page >= bank && page + DEVICE_PAGESIZE <= bank + bank_span) {
            const u64 index = (page >> Core::DEVICE_PAGEBITS) - (bank >> Core::DEVICE_PAGEBITS);
            if (index < 32) {
                std::memcpy(shadow.bytes.data() + (page - bank), bytes, DEVICE_PAGESIZE);
                shadow.valid |= 1U << index;
            }
            continue;
        }
        const DAddr first = (std::max)(bank, page);
        const DAddr end = (std::min)(bank + bank_span, page + DEVICE_PAGESIZE);
        const u64 first_index = (first >> Core::DEVICE_PAGEBITS) - (bank >> Core::DEVICE_PAGEBITS);
        const u64 last_index = ((end - 1) >> Core::DEVICE_PAGEBITS) - (bank >> Core::DEVICE_PAGEBITS);
        for (u64 index = first_index; index <= last_index && index < 32; ++index) {
            shadow.valid &= ~(1U << index);
        }
    }
}

template <class P>
void BufferCache<P>::RefreshHotShadows(DAddr device_addr, const u8* bytes, u64 size) {
    if (hot_pages.empty() || size == 0) {
        return;
    }
    if (!Nx2BufCache::g_shadow_refresh) {
        InvalidateHotShadows(device_addr, size);
        return;
    }
    const DAddr begin = Common::AlignDown(device_addr, DEVICE_PAGESIZE);
    const DAddr end = Common::AlignUp(device_addr + size, DEVICE_PAGESIZE);
    for (DAddr page = begin; page < end; page += DEVICE_PAGESIZE) {
        const auto it = hot_pages.find(page);
        if (it == hot_pages.end()) {
            continue;
        }
        HotPage& hot = *it->second;
        if (page >= device_addr && page + DEVICE_PAGESIZE <= device_addr + size) {
            // The host holds these bytes now, as after UploadHotPages.
            std::memcpy(hot.bytes.data(), bytes + (page - device_addr), DEVICE_PAGESIZE);
            hot.valid = true;
        } else {
            hot.valid = false;
        }
    }
}

template <class P>
void BufferCache<P>::InvalidateBankShadows(DAddr device_addr, u64 size) {
    if (bank_shadows.empty() || size == 0) {
        return;
    }
    const DAddr end = device_addr + size;
    constexpr DAddr bank_span = 0x10000;
    auto it = bank_shadows.lower_bound(device_addr > bank_span ? device_addr - bank_span : 0);
    for (; it != bank_shadows.end() && it->first < end; ++it) {
        const DAddr bank = it->first;
        if (bank + bank_span <= device_addr) {
            continue;
        }
        const DAddr first = (std::max)(bank, device_addr);
        const u64 first_page = (first >> Core::DEVICE_PAGEBITS) - (bank >> Core::DEVICE_PAGEBITS);
        const u64 last_page = ((end - 1) >> Core::DEVICE_PAGEBITS) - (bank >> Core::DEVICE_PAGEBITS);
        for (u64 page = first_page; page <= last_page && page < 32; ++page) {
            it->second->valid &= ~(1U << page);
        }
    }
}

template <class P>
bool BufferCache<P>::OverlapsBankShadow(DAddr device_addr, u64 size) const {
    if (bank_shadows.empty() || size == 0) {
        return false;
    }
    constexpr DAddr bank_span = 0x10000;
    const auto it = bank_shadows.lower_bound(device_addr > bank_span ? device_addr - bank_span + 1 : 0);
    return it != bank_shadows.end() && it->first < device_addr + size;
}

template <class P>
void BufferCache<P>::RefreshBankShadowsFromUpload(DAddr device_addr, const u8* bytes, u64 size) {
    if (bank_shadows.empty() || size == 0) {
        return;
    }
    const DAddr end = device_addr + size;
    constexpr DAddr bank_span = 0x10000;
    auto it = bank_shadows.lower_bound(device_addr > bank_span ? device_addr - bank_span : 0);
    for (; it != bank_shadows.end() && it->first < end; ++it) {
        const DAddr bank = it->first;
        if (bank + bank_span <= device_addr) {
            continue;
        }
        BankShadow& shadow = *it->second;
        // Shadow page k of the bank (UploadNativeBank): the k-th device page from the bank's
        // first, clipped to the bank. The bank's mapped end may stop earlier; a copy covering
        // the clipped page covers that too.
        const DAddr bank_page = Common::AlignDown(bank, DEVICE_PAGESIZE);
        const DAddr first = (std::max)(bank, device_addr);
        const DAddr last = (std::min)(bank + bank_span, end) - 1;
        const u64 first_index = (first - bank_page) / DEVICE_PAGESIZE;
        const u64 last_index = (last - bank_page) / DEVICE_PAGESIZE;
        for (u64 index = first_index; index <= last_index && index < 32; ++index) {
            const DAddr page_begin = (std::max)(bank, bank_page + index * DEVICE_PAGESIZE);
            const DAddr page_end = (std::min)(bank + bank_span, bank_page + (index + 1) * DEVICE_PAGESIZE);
            const u32 bit = 1U << index;
            if (page_begin >= device_addr && page_end <= end) {
                std::memcpy(shadow.bytes.data() + (page_begin - bank), bytes + (page_begin - device_addr),
                            page_end - page_begin);
                shadow.valid |= bit;
                Nx2BufCache::Note(Nx2BufCache::Stat::BankSyncRefresh);
            } else {
                shadow.valid &= ~bit;
            }
        }
    }
}

template <class P>
void BufferCache<P>::MappedUploadMemoryRefreshingBanks([[maybe_unused]] Buffer& buffer,
                                                       [[maybe_unused]] u64 total_size_bytes,
                                                       [[maybe_unused]] std::span<BufferCopy> copies) {
    if constexpr (USE_MEMORY_MAPS) {
        auto upload_staging = runtime.UploadStagingBuffer(total_size_bytes);
        const std::span<u8> staging_pointer = upload_staging.mapped_span;
        for (BufferCopy& copy : copies) {
            u8* const src_pointer = staging_pointer.data() + copy.src_offset;
            const DAddr device_addr = buffer.CpuAddr() + copy.dst_offset;
            const bool over_bank = OverlapsBankShadow(device_addr, copy.size);
            if (over_bank && copy.size > 256_KiB) {
                InvalidateBankShadows(device_addr, copy.size);
                device_memory.ReadBlockUnsafe(device_addr, src_pointer, copy.size);
            } else if (over_bank) {
                // One guest read: the staging buffer (write-combined) and the shadows get the
                // same bytes, so the shadows hold what the host buffer will.
                if (bank_refresh_scratch.size() < copy.size) {
                    bank_refresh_scratch.resize(copy.size);
                }
                device_memory.ReadBlockUnsafe(device_addr, bank_refresh_scratch.data(), copy.size);
                std::memcpy(src_pointer, bank_refresh_scratch.data(), copy.size);
                RefreshBankShadowsFromUpload(device_addr, bank_refresh_scratch.data(), copy.size);
            } else {
                device_memory.ReadBlockUnsafe(device_addr, src_pointer, copy.size);
            }
            if (copy.size >= 256_KiB) {
                uploaded_buffer_ranges.emplace_back(device_addr, copy.size);
            }
            copy.src_offset += upload_staging.offset;
        }
        const bool can_reorder = runtime.CanReorderUpload(buffer, copies);
        runtime.CopyBuffer(buffer, upload_staging.buffer, copies, true, can_reorder);
    }
}

template <class P>
void BufferCache<P>::DownloadBufferMemory(Buffer& buffer) {
    DownloadBufferMemory(buffer, buffer.CpuAddr(), buffer.SizeBytes());
}

template <class P>
void BufferCache<P>::DownloadBuffersMemory(std::span<const BufferId> buffer_ids) {
    struct Download {
        BufferId buffer_id;
        BufferCopy copy;
    };
    std::vector<Download> downloads;
    u64 total_size_bytes = 0;
    for (const BufferId buffer_id : buffer_ids) {
        Buffer& buffer = slot_buffers[buffer_id];
        const DAddr buffer_addr = buffer.CpuAddr();
        memory_tracker.ForEachDownloadRangeAndClear(
            buffer_addr, buffer.SizeBytes(), [&](u64 device_addr_out, u64 range_size) {
                gpu_modified_ranges.ForEachInRange(device_addr_out, range_size,
                                                   [&](DAddr start, DAddr end) {
                    downloads.push_back({buffer_id, BufferCopy{
                                                        .src_offset = start - buffer_addr,
                                                        .dst_offset = total_size_bytes,
                                                        .size = end - start,
                                                    }});
                    total_size_bytes += Common::AlignUp(end - start, 64);
                });
                ClearDownload(device_addr_out, range_size);
                gpu_modified_ranges.Subtract(device_addr_out, range_size);
            });
    }
    if (total_size_bytes == 0) {
        return;
    }
    auto download_staging = runtime.DownloadStagingBuffer(total_size_bytes);
    for (Download& download : downloads) {
        download.copy.dst_offset += download_staging.offset;
        Buffer& buffer = slot_buffers[download.buffer_id];
        buffer.MarkUsage(download.copy.src_offset, download.copy.size);
        runtime.CopyBuffer(download_staging.buffer, buffer, std::span{&download.copy, 1}, true);
    }
    runtime.Finish();
    const u8* const mapped_memory = download_staging.mapped_span.data();
    const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();
    for (const Download& download : downloads) {
        const DAddr device_addr = slot_buffers[download.buffer_id].CpuAddr() + download.copy.src_offset;
        WriteDownloadedMemory(
            cpu_written, device_addr,
            mapped_memory + (download.copy.dst_offset - download_staging.offset),
            download.copy.size);
    }
}

template <class P>
void BufferCache<P>::DownloadBufferMemory(Buffer& buffer, DAddr device_addr, u64 size) {
    boost::container::small_vector<BufferCopy, 1> copies;
    u64 total_size_bytes = 0;
    u64 largest_copy = 0;
    memory_tracker.ForEachDownloadRangeAndClear(
        device_addr, size, [&](u64 device_addr_out, u64 range_size) {
            const DAddr buffer_addr = buffer.CpuAddr();
            const auto add_download = [&](DAddr start, DAddr end) {
                const u64 new_offset = start - buffer_addr;
                const u64 new_size = end - start;
                copies.push_back(BufferCopy{
                    .src_offset = new_offset,
                    .dst_offset = total_size_bytes,
                    .size = new_size,
                });
                // Align up to avoid cache conflicts
                constexpr u64 align = 64ULL;
                constexpr u64 mask = ~(align - 1ULL);
                total_size_bytes += (new_size + align - 1) & mask;
                largest_copy = (std::max)(largest_copy, new_size);
            };

            gpu_modified_ranges.ForEachInRange(device_addr_out, range_size, add_download);
            ClearDownload(device_addr_out, range_size);
            gpu_modified_ranges.Subtract(device_addr_out, range_size);
        });
    if (total_size_bytes == 0) {
        return;
    }
    const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();

    if constexpr (requires { buffer.Mapped(); }) {
        if (const auto mapped = buffer.Mapped(); !mapped.empty()) {
            runtime.FinishForHostRead();
            for (const BufferCopy& copy : copies) {
                WriteDownloadedMemory(cpu_written, buffer.CpuAddr() + copy.src_offset,
                                      mapped.data() + copy.src_offset, copy.size);
            }
            return;
        }
    }
    if constexpr (USE_MEMORY_MAPS) {
        auto download_staging = runtime.DownloadStagingBuffer(total_size_bytes);
        const u8* const mapped_memory = download_staging.mapped_span.data();
        const std::span<BufferCopy> copies_span(copies.data(), copies.data() + copies.size());
        for (BufferCopy& copy : copies) {
            // Modify copies to have the staging offset in mind
            copy.dst_offset += download_staging.offset;
            buffer.MarkUsage(copy.src_offset, copy.size);
        }
        runtime.CopyBuffer(download_staging.buffer, buffer, copies_span, true);
        runtime.Finish();
        for (const BufferCopy& copy : copies) {
            const DAddr copy_device_addr = buffer.CpuAddr() + copy.src_offset;
            // Undo the modified offset
            const u64 dst_offset = copy.dst_offset - download_staging.offset;
            const u8* copy_mapped_memory = mapped_memory + dst_offset;
            WriteDownloadedMemory(cpu_written, copy_device_addr, copy_mapped_memory, copy.size);
        }
    } else {
        const std::span<u8> immediate_buffer = ImmediateBuffer(largest_copy);
        for (const BufferCopy& copy : copies) {
            buffer.ImmediateDownload(copy.src_offset, immediate_buffer.subspan(0, copy.size));
            const DAddr copy_device_addr = buffer.CpuAddr() + copy.src_offset;
            WriteDownloadedMemory(cpu_written, copy_device_addr, immediate_buffer.data(), copy.size);
        }
    }
}

template <class P>
void BufferCache<P>::DeleteBuffer(BufferId buffer_id, bool do_not_mark) {
    ++binding_epoch;
    if constexpr (requires { runtime.OnBufferDeleted(slot_buffers[buffer_id]); }) {
        runtime.OnBufferDeleted(slot_buffers[buffer_id]);
    }
    read_trapped_ranges.Subtract(slot_buffers[buffer_id].CpuAddr(),
                                 slot_buffers[buffer_id].SizeBytes());
    if (!do_not_mark) {
        InvalidateBankShadows(slot_buffers[buffer_id].CpuAddr(), slot_buffers[buffer_id].SizeBytes());
        InvalidateHotShadows(slot_buffers[buffer_id].CpuAddr(), slot_buffers[buffer_id].SizeBytes());
    }
    cpu_read_ranges.Subtract(slot_buffers[buffer_id].CpuAddr(), slot_buffers[buffer_id].SizeBytes());
    cpu_read_ranges_previous.Subtract(slot_buffers[buffer_id].CpuAddr(),
                                      slot_buffers[buffer_id].SizeBytes());
    bool dirty_index{false};
    boost::container::small_vector<u64, NUM_VERTEX_BUFFERS> dirty_vertex_buffers;
    const auto scalar_replace = [buffer_id](Binding& binding) {
        if (binding.buffer_id == buffer_id) {
            binding.buffer_id = BufferId{};
        }
    };
    const auto replace = [scalar_replace](std::span<Binding> bindings) {
        std::ranges::for_each(bindings, scalar_replace);
    };

    if (channel_state->index_buffer.buffer_id == buffer_id) {
        channel_state->index_buffer.buffer_id = BufferId{};
        dirty_index = true;
    }

    for (u32 index = 0; index < channel_state->vertex_buffers.size(); index++) {
        auto& binding = channel_state->vertex_buffers[index];
        if (binding.buffer_id == buffer_id) {
            binding.buffer_id = BufferId{};
            UpdateVertexBufferSlot(index, binding);
            dirty_vertex_buffers.push_back(index);
        }
    }
    std::ranges::for_each(channel_state->uniform_buffers, replace);
    std::ranges::for_each(channel_state->storage_buffers, replace);
    replace(channel_state->transform_feedback_buffers);
    replace(channel_state->compute_uniform_buffers);
    replace(channel_state->compute_storage_buffers);

    // Adapted from the pre-refresh branch: merged buffers invalidate segment references too.
    InvalidateMultiRangeSegments(std::span{graphics_segments}, buffer_id);
    InvalidateMultiRangeSegments(std::span{compute_segments}, buffer_id);

    // Mark the whole buffer as CPU written to stop tracking CPU writes
    if (!do_not_mark) {
        Buffer& buffer = slot_buffers[buffer_id];
        memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddr(), buffer.SizeBytes());
        ++cpu_dirty_epoch;
    }

    Unregister(buffer_id);

#ifdef YUZU_LEGACY
    if (!do_not_mark || !immediately_free)
#endif
        delayed_destruction_ring.Push(std::move(slot_buffers[buffer_id]));

    slot_buffers.erase(buffer_id);

    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        channel_state->dirty_uniform_buffers.fill(~u32{0});
        channel_state->uniform_buffer_binding_sizes.fill({});
    }

    auto& flags = maxwell3d->dirty.flags;
    if (dirty_index) {
        flags[Dirty::IndexBuffer] = true;
    }

    if (dirty_vertex_buffers.size() > 0) {
        flags[Dirty::VertexBuffers] = true;
        for (auto index : dirty_vertex_buffers) {
            flags[Dirty::VertexBuffer0 + index] = true;
        }
    }
    channel_state->has_deleted_buffers = true;
}

template <class P>
void BufferCache<P>::TraceTablePointer(bool compute, size_t stage, size_t ssbo_index,
                                       u32 cbuf_index, u32 cbuf_offset, u32 indirect_offset,
                                       GPUVAddr holder, const Binding& binding) {
    // A table entry is read here from guest memory, and again by the shader from the holder's
    // host buffer. Logs each change of a slot's entry or size, and whether the two copies of the
    // entry can differ: GPU-written (host newer) or CPU-written and not yet uploaded (guest newer).
    static const bool enabled = std::getenv("EDEN_NX2_TABLE_POINTER_TRACE") != nullptr;
    if (!enabled) {
        return;
    }
    const GPUVAddr entry_addr = holder + indirect_offset;
    const GPUVAddr entry = gpu_memory->Read<u64>(entry_addr);
    const u64 key = (u64{compute} << 63) | (u64{stage & 7} << 60) | (u64{ssbo_index & 63} << 54) |
                    (u64{cbuf_index & 31} << 49) | (u64{cbuf_offset & 0x7ffff} << 30) |
                    (indirect_offset & 0x3fffffff);
    const std::pair<GPUVAddr, u32> state{entry, binding.size};
    const auto [it, inserted] = traced_table_pointers.try_emplace(key, state);
    if (!inserted && it->second == state) {
        return;
    }
    it->second = state;
    static constexpr u64 max_lines = 100000;
    static u64 lines{};
    if (++lines > max_lines) {
        if (lines == max_lines + 1) {
            LOG_INFO(HW_GPU, "Table pointer trace stopped after {} lines", max_lines);
        }
        return;
    }
    const std::optional<DAddr> entry_device = gpu_memory->GpuToCpuAddress(entry_addr);
    const bool gpu_modified =
        entry_device && memory_tracker.IsRegionGpuModified(*entry_device, sizeof(u64));
    const bool cpu_modified =
        entry_device && memory_tracker.IsRegionCpuModified(*entry_device, sizeof(u64));
    LOG_INFO(HW_GPU,
             "Table pointer frame={} {} stage={} ssbo={} c{}[{:#x}]+{:#x}: holder={:#x} "
             "entry={:#x} bound={:#x} size={:#x} entry_gpu_modified={} entry_cpu_modified={}",
             frame_tick, compute ? "compute" : "graphics", stage, ssbo_index, cbuf_index,
             cbuf_offset, indirect_offset, holder, entry, binding.device_addr, binding.size,
             gpu_modified, cpu_modified);
}

template <class P>
Binding BufferCache<P>::StorageBufferBinding(GPUVAddr ssbo_addr, u32 cbuf_index, bool is_written,
                                             bool native_pointer, bool table_pointer) const {
    const GPUVAddr gpu_addr = gpu_memory->Read<u64>(ssbo_addr);

    if (gpu_addr == 0) {
        return NULL_BINDING;
    }

    const bool reads_size_word = !(table_pointer && !is_written) && !native_pointer;
    const u64 next_qword = reads_size_word ? gpu_memory->Read<u64>(ssbo_addr + 8) : 0;
    const u32 alignment = runtime.GetStorageBufferAlignment();
    struct CachedBinding {
        size_t owner;
        GPUVAddr gpu_addr;
        u64 next_qword;
        u64 serial;
        u32 alignment;
        u32 flags;
        Binding binding;
    };
    constexpr size_t ways = 4;
    thread_local std::array<std::array<CachedBinding, ways>, 512> binding_cache{};
    const u32 flags =
        (is_written ? 1U : 0U) | (native_pointer ? 2U : 0U) | (table_pointer ? 4U : 0U);
    const size_t owner = gpu_memory->GetID() + 1;
    const u64 serial = gpu_memory->MapSerial();
    auto& set = binding_cache[((gpu_addr ^ next_qword ^ flags) * 0x9E3779B97F4A7C15ULL) >> 55];
    for (size_t way = 0; way < ways; ++way) {
        const CachedBinding& cached = set[way];
        if (cached.owner == owner && cached.gpu_addr == gpu_addr &&
            cached.next_qword == next_qword && cached.serial == serial &&
            cached.alignment == alignment && cached.flags == flags && !Nx2Perf::Verify()) {
            const Binding hit = cached.binding;
            std::rotate(set.begin(), set.begin() + way, set.begin() + way + 1);
            return hit;
        }
    }
    // Per-object pointers into one pool miss the cache above on every new object. A pool is
    // a pointer's kind range and device run: later pointers inside it read their layout size,
    // contiguous size and device address from it while the mappings stay (EDEN_NX2_BUFCACHE_POOLS).
    struct Pool {
        size_t owner{};
        u64 serial{};
        GPUVAddr begin{};
        GPUVAddr kind_end{};  ///< [begin, kind_end) lies in one kind map entry, not INVALID
        GPUVAddr run_end{};   ///< [begin, run_end) is one device run
        bool run_end_exact{}; ///< false: the run goes on past run_end (only 64 MiB were asked)
        DAddr device_begin{};
    };
    thread_local std::array<Pool, 8> pools{};
    thread_local u32 pool_cursor{};
    const Pool* pool = nullptr;
    if (Nx2BufCache::g_pool_ranges && !Nx2Perf::Verify()) {
        for (const Pool& entry : pools) {
            if (entry.owner == owner && entry.serial == serial && entry.begin <= gpu_addr &&
                gpu_addr < entry.kind_end && gpu_addr < entry.run_end) {
                pool = &entry;
                break;
            }
        }
        if (pool == nullptr) {
            constexpr u64 run_query = 64_MiB;
            const size_t layout = gpu_memory->GetMemoryLayoutSize(gpu_addr);
            const size_t run = layout != 0 ? gpu_memory->MaxContinuousRange(gpu_addr, run_query) : 0;
            const std::optional<DAddr> device = run != 0 ? gpu_memory->GpuToCpuAddress(gpu_addr)
                                                         : std::nullopt;
            if (device) {
                Pool& entry = pools[pool_cursor++ % pools.size()];
                entry = Pool{
                    .owner = owner,
                    .serial = serial,
                    .begin = gpu_addr,
                    .kind_end = gpu_addr + layout,
                    .run_end = gpu_addr + run,
                    .run_end_exact = run < run_query,
                    .device_begin = *device,
                };
                pool = &entry;
            }
        }
    }
    const Binding binding = [&]() -> Binding {
        u32 size = [&]() {
            const u64 layout_size = pool != nullptr ? pool->kind_end - gpu_addr
                                                    : u64{gpu_memory->GetMemoryLayoutSize(gpu_addr)};
            const u32 memory_layout_size = static_cast<u32>(
                (std::min)(layout_size, u64{(std::numeric_limits<u32>::max)()}));
            // A pointer read from a table in memory is the base of an object pool, read anywhere in
            // it: bound to the end of its mapping, as far as a storage buffer may reach (at most
            // 1 GiB). Ocarina of Time's grass instances lie 12-18 MiB past theirs (an 8 MiB cap
            // culled every blade), and its light probe records reach about 130 MiB past theirs: a
            // 64 MiB cap left the far probes outside the binding, and the fog took one overbright
            // constant from them. Written pools keep a cap: a write marks the whole binding
            // GPU-modified.
            if (table_pointer && !is_written) {
                const u64 reach = (std::min)(runtime.GetMaxStorageBufferRange(), u64{1_GiB});
                const u32 limit = static_cast<u32>(reach - (std::min)(reach, u64{64_KiB}));
                return (std::min)(memory_layout_size, limit);
            }
            // A native (Switch 2) compute program's pointer is bare: the word after it is another
            // parameter (a dimension, stride or texture handle), not an NVN buffer size. Taking it
            // as one bound Fortune's Weave's DLSS input features to 266 bytes and dropped the rest.
            if (native_pointer) {
                return (std::min)(memory_layout_size, static_cast<u32>(8_MiB));
            }
            const u32 packed_size = static_cast<u32>(next_qword);
            const bool next_qword_is_size = static_cast<u32>(next_qword >> 32) == 0 &&
                                            packed_size != 0 && packed_size <= memory_layout_size;
            if (next_qword_is_size) {
                return packed_size;
            }
            return (std::min)(memory_layout_size, static_cast<u32>(8_MiB));
        }();
        if ((native_pointer || table_pointer) && size != 0) {
            // Stop where the mapping or its device range does: the binding is one device range, and
            // a written one isn't split into separate ranges without sparse sources.
            if (pool != nullptr && (pool->run_end_exact || gpu_addr + size <= pool->run_end)) {
                size = static_cast<u32>((std::min<u64>)(size, pool->run_end - gpu_addr));
            } else {
                size = static_cast<u32>(gpu_memory->MaxContinuousRange(gpu_addr, size));
            }
        }

        // Alignment only applies to the offset of the buffer
        const GPUVAddr aligned_gpu_addr = Common::AlignDown(gpu_addr, alignment);
        const u32 aligned_size = static_cast<u32>(gpu_addr - aligned_gpu_addr) + size;

        // Inside a run, device addresses follow GPU addresses (above the device space, the
        // translation reports nothing: leave those to it).
        const auto translate = [&](GPUVAddr address) -> std::optional<DAddr> {
            if (pool != nullptr && address >= pool->begin) {
                const DAddr device = pool->device_begin + (address - pool->begin);
                if ((device >> Tegra::MaxwellDeviceMemoryManager::AS_BITS) == 0) {
                    return device;
                }
            }
            return gpu_memory->GpuToCpuAddress(address);
        };
        const std::optional<DAddr> aligned_device_addr = translate(aligned_gpu_addr);
        if (!aligned_device_addr || size == 0) {
            LOG_DEBUG(HW_GPU, "Failed to find storage buffer for cbuf index {}", cbuf_index);
            return NULL_BINDING;
        }
        const std::optional<DAddr> device_addr = translate(gpu_addr);
        ASSERT_MSG(device_addr, "Unaligned storage buffer address not found for cbuf index {}",
                   cbuf_index);
        // The end address used for size calculation does not need to be aligned
        const DAddr cpu_end = Common::AlignUp(*device_addr + size, Core::DEVICE_PAGESIZE);

        u32 binding_size = static_cast<u32>(cpu_end - *aligned_device_addr);
        if (is_written) {
            binding_size = aligned_size;
        }
        return Binding{
            .device_addr = *aligned_device_addr,
            .gpu_addr = aligned_gpu_addr,
            .size = binding_size,
            .buffer_id = BufferId{},
        };
    }();
    std::rotate(set.begin(), set.end() - 1, set.end());
    set[0] = {owner, gpu_addr, next_qword, serial, alignment, flags, binding};
    return binding;
}

template <class P>
TextureBufferBinding BufferCache<P>::GetTextureBufferBinding(GPUVAddr gpu_addr, u32 size,
                                                             PixelFormat format) {
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    TextureBufferBinding binding;
    if (!device_addr || size == 0) {
        binding.device_addr = 0;
        binding.size = 0;
        binding.buffer_id = NULL_BUFFER_ID;
        binding.format = PixelFormat::Invalid;
    } else {
        binding.device_addr = *device_addr;
        binding.size = size;
        binding.buffer_id = BufferId{};
        binding.format = format;
    }
    return binding;
}

template <class P>
std::span<const u8> BufferCache<P>::ImmediateBufferWithData(DAddr device_addr, size_t size) {
    u8* const base_pointer = device_memory.GetPointer<u8>(device_addr);
    if (IsRangeGranular(device_addr, size) ||
        base_pointer + size == device_memory.GetPointer<u8>(device_addr + size)) {
        return std::span(base_pointer, size);
    } else {
        const std::span<u8> span = ImmediateBuffer(size);
        device_memory.ReadBlockUnsafe(device_addr, span.data(), size);
        return span;
    }
}

template <class P>
std::span<u8> BufferCache<P>::ImmediateBuffer(size_t wanted_capacity) {
    immediate_buffer_alloc.resize_destructive(wanted_capacity);
    return std::span<u8>(immediate_buffer_alloc.data(), wanted_capacity);
}

template <class P>
bool BufferCache<P>::HasFastUniformBufferBound(size_t stage, u32 binding_index) const noexcept {
    return ((channel_state->fast_bound_uniform_buffers[stage] >> binding_index) & 1u) != 0;
}

template <class P>
std::pair<typename BufferCache<P>::Buffer*, u32> BufferCache<P>::GetDrawIndirectCount() {
    auto& buffer = slot_buffers[channel_state->count_buffer_binding.buffer_id];
    return std::make_pair(&buffer, buffer.Offset(channel_state->count_buffer_binding.device_addr));
}

template <class P>
std::pair<typename BufferCache<P>::Buffer*, u32> BufferCache<P>::GetDrawIndirectBuffer() {
    auto& buffer = slot_buffers[channel_state->indirect_buffer_binding.buffer_id];
    return std::make_pair(&buffer,
                          buffer.Offset(channel_state->indirect_buffer_binding.device_addr));
}

} // namespace VideoCommon
