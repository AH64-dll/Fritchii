// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "common/nx2_gpu_write_listener.h"
#include "video_core/nx2_frame_fence.h"

namespace Tegra {

// Set once a Switch 2 (Ampere, NVN2) driver is talking to the GPU. Its command streams use
// Ampere classes and the Turing+ macro ISA, which Eden's Maxwell engines don't implement.
inline std::atomic<bool> g_nx2_ampere{false};

// Device pages the GPU has written through the buffer cache (shader stores, copies), ever. Title 1
// runs command lists the GPU writes each frame (20-byte entries its culling fills in); read
// without a flush they are stale and the draws behind them (trees) go missing. The bits never
// clear: the dispatch that rewrites a page may still be queued on the split's back when its front
// reads the command list. One bit per 4 KiB of the 36-bit device address space (2 MiB).
inline constexpr u64 Nx2WrittenPageBits = 12;
inline std::array<std::atomic<u64>, (u64{1} << (36 - Nx2WrittenPageBits)) / 64> g_nx2_gpu_written_pages{};

/// Bumped when a page is marked GPU-written for the first time (read by nothing since
/// EDEN_NX2_LIST_FAST went; a diagnostic count).
inline std::atomic<u64> g_nx2_new_written_pages{};

inline void Nx2MarkGpuWritten(u64 device_address, u64 size) {
    if (size == 0) {
        return;
    }
    const u64 last = std::min<u64>((device_address + size - 1) >> Nx2WrittenPageBits,
                                   g_nx2_gpu_written_pages.size() * 64 - 1);
    // A word at a time: storage buffer binds mark megabytes, mostly marked already.
    for (u64 page = device_address >> Nx2WrittenPageBits; page <= last;) {
        const u64 first_bit = page % 64;
        const u64 bits = std::min<u64>(64 - first_bit, last - page + 1);
        const u64 mask = (bits == 64 ? ~u64{0} : (u64{1} << bits) - 1) << first_bit;
        auto& word = g_nx2_gpu_written_pages[page / 64];
        if ((word.load(std::memory_order_relaxed) & mask) != mask) {
            if ((word.fetch_or(mask, std::memory_order_relaxed) & mask) != mask) {
                g_nx2_new_written_pages.fetch_add(1, std::memory_order_relaxed);
            }
        }
        page += bits;
    }
}

// GPU-written device pages the front has read command lists from (same layout as above). The
// buffer cache copies these pages into host memory right after the compute or copy work that
// writes them and submits, so the front reads the copy once the GPU is done instead of draining
// the split and flushing the caches (BufferCache::Nx2ReadListCopy).
inline std::array<std::atomic<u64>, (u64{1} << (36 - Nx2WrittenPageBits)) / 64> g_nx2_command_list_pages{};
inline std::atomic<bool> g_nx2_has_command_list_pages{false};

inline void Nx2MarkCommandListPage(u64 device_address) {
    const u64 page = device_address >> Nx2WrittenPageBits;
    if (page >= g_nx2_command_list_pages.size() * 64) {
        return;
    }
    auto& word = g_nx2_command_list_pages[page / 64];
    const u64 bit = u64{1} << (page % 64);
    if ((word.load(std::memory_order_relaxed) & bit) == 0) {
        word.fetch_or(bit, std::memory_order_relaxed);
        g_nx2_has_command_list_pages.store(true, std::memory_order_relaxed);
    }
}

[[nodiscard]] inline bool Nx2IsCommandListPage(u64 device_address) {
    const u64 page = device_address >> Nx2WrittenPageBits;
    if (page >= g_nx2_command_list_pages.size() * 64) {
        return false;
    }
    return (g_nx2_command_list_pages[page / 64].load(std::memory_order_relaxed) >> (page % 64)) & 1;
}

// Query results written when their host work completes (counter reports): per 16-byte slot
// hash, how many the back issued and how many completed. Equal counts mean the words in memory
// are final, so the split's front can read them without draining the back.
inline constexpr u64 Nx2AsyncReportSlots = u64{1} << 16;
inline std::array<std::atomic<u32>, Nx2AsyncReportSlots> g_nx2_async_reports_issued{};
inline std::array<std::atomic<u32>, Nx2AsyncReportSlots> g_nx2_async_reports_completed{};

[[nodiscard]] inline u64 Nx2AsyncReportSlot(u64 device_address) {
    return (device_address >> 4) % Nx2AsyncReportSlots;
}

[[nodiscard]] inline bool Nx2AsyncReportsSettled(u64 device_address, u64 size) {
    if (size == 0) {
        return true;
    }
    for (u64 slot = device_address >> 4; slot <= (device_address + size - 1) >> 4; ++slot) {
        const u64 index = slot % Nx2AsyncReportSlots;
        if (g_nx2_async_reports_completed[index].load(std::memory_order_acquire) !=
            g_nx2_async_reports_issued[index].load(std::memory_order_acquire)) {
            return false;
        }
    }
    return true;
}

// Bumped after every completed report; threads in Nx2WaitAsyncReportsSettled wait on it
// (zortos293/Lichii#559).
inline std::atomic<u32> g_nx2_async_reports_generation{};
inline std::atomic<u32> g_nx2_async_reports_waiters{};

/// A report was issued to device_address (the thread that owns the query cache).
inline void Nx2IssueAsyncReport(u64 device_address) {
    g_nx2_async_reports_issued[Nx2AsyncReportSlot(device_address)].fetch_add(
        1, std::memory_order_release);
}

/// A report's host work completed and its words are in memory (the fencing thread).
inline void Nx2CompleteAsyncReport(u64 device_address) {
    g_nx2_async_reports_completed[Nx2AsyncReportSlot(device_address)].fetch_add(
        1, std::memory_order_release);
    g_nx2_async_reports_generation.fetch_add(1, std::memory_order_seq_cst);
    if (g_nx2_async_reports_waiters.load(std::memory_order_seq_cst) != 0) {
        g_nx2_async_reports_generation.notify_all();
    }
}

/// One asynchronous report's issued/completed pair. Complete counts it once, and destruction
/// does if nothing did: an operation dropped without running (fences discarded when emulation
/// stops) would otherwise leave its slot unequal for good, and a later wait on that slot would
/// never end. Owned by the report's operation (shared by its copies); not thread-safe, as the
/// operation runs or is dropped on one thread at a time.
class Nx2AsyncReport {
public:
    explicit Nx2AsyncReport(u64 device_address_) : device_address{device_address_} {}
    ~Nx2AsyncReport() {
        Complete();
    }
    Nx2AsyncReport(const Nx2AsyncReport&) = delete;
    Nx2AsyncReport& operator=(const Nx2AsyncReport&) = delete;

    void Issue() {
        if (!issued) {
            issued = true;
            Nx2IssueAsyncReport(device_address);
        }
    }
    void Complete() {
        if (issued && !completed) {
            completed = true;
            Nx2CompleteAsyncReport(device_address);
        }
    }

private:
    u64 device_address;
    bool issued{};
    bool completed{};
};

/// Blocks until every report issued so far to [device_address, device_address + size) is in
/// memory. Only for the thread that issues reports, so none can be issued while it waits; the
/// fences carrying them must already be submitted. A slot shared with another address only
/// waits for that report too.
inline void Nx2WaitAsyncReportsSettled(u64 device_address, u64 size) {
    if (Nx2AsyncReportsSettled(device_address, size)) {
        return;
    }
    g_nx2_async_reports_waiters.fetch_add(1, std::memory_order_seq_cst);
    for (;;) {
        const u32 generation = g_nx2_async_reports_generation.load(std::memory_order_seq_cst);
        if (Nx2AsyncReportsSettled(device_address, size)) {
            break;
        }
        g_nx2_async_reports_generation.wait(generation, std::memory_order_seq_cst);
    }
    g_nx2_async_reports_waiters.fetch_sub(1, std::memory_order_relaxed);
}

/// A new emulation session starts: no page has been written by its GPU yet.
inline void Nx2ClearGpuWrittenPages() {
    for (auto& word : g_nx2_gpu_written_pages) {
        word.store(0, std::memory_order_relaxed);
    }
    for (auto& word : g_nx2_command_list_pages) {
        word.store(0, std::memory_order_relaxed);
    }
    g_nx2_has_command_list_pages.store(false, std::memory_order_relaxed);
}

[[nodiscard]] inline bool Nx2IsGpuWritten(u64 device_address) {
    const u64 page = device_address >> Nx2WrittenPageBits;
    if (page >= g_nx2_gpu_written_pages.size() * 64) {
        return false;
    }
    return (g_nx2_gpu_written_pages[page / 64].load(std::memory_order_relaxed) >> (page % 64)) & 1;
}

/// Guest writes of cached pages no GPU write ever marked, collected for the split's back without
/// OnCPUWrite's cache mutexes (EDEN_NX2_EXP35, Core::Memory::HandleRasterizerWrite); counted for
/// RasterizerVulkan's guest access log.
inline std::atomic<u64> g_nx2_guest_writes_collected{};
/// Guest reads of cached memory sent to the split's back as flush requests because the cache
/// mutexes stayed taken (EDEN_NX2_EXP60).
inline std::atomic<u64> g_nx2_guest_reads_handed{};

/// Whether any page of the range may have been written by the GPU (zortos293/Lichii#463).
[[nodiscard]] inline bool Nx2MayBeGpuWritten(u64 device_address, u64 size) {
    if (size == 0) {
        return false;
    }
    const u64 first = device_address >> Nx2WrittenPageBits;
    const u64 last = (device_address + size - 1) >> Nx2WrittenPageBits;
    if (last < first || last >= g_nx2_gpu_written_pages.size() * 64) {
        return true;
    }
    for (u64 page = first; page <= last; ++page) {
        if ((g_nx2_gpu_written_pages[page / 64].load(std::memory_order_relaxed) >> (page % 64)) & 1) {
            return true;
        }
    }
    return false;
}

inline constexpr size_t Nx2MaxSemaphoreRegions = 32;
inline std::array<std::atomic<u64>, Nx2MaxSemaphoreRegions> g_nx2_semaphore_regions{}; // starts

inline void Nx2AddSemaphoreRegion(u64 start) {
    for (auto& region : g_nx2_semaphore_regions) {
        if (region.load() == start) {
            return;
        }
    }
    for (auto& region : g_nx2_semaphore_regions) {
        u64 expected = 0;
        if (region.compare_exchange_strong(expected, start)) {
            return;
        }
    }
}

/// The region's memory was destroyed: stop mirroring into it.
inline void Nx2RemoveSemaphoreRegion(u64 start) {
    for (auto& region : g_nx2_semaphore_regions) {
        u64 expected = start;
        region.compare_exchange_strong(expected, 0);
    }
}

/// A new emulation session starts: no queue has a region yet.
inline void Nx2ClearSemaphoreRegions() {
    for (auto& region : g_nx2_semaphore_regions) {
        region.store(0);
    }
}

// Switch 2 sparse regions (nx2:gpu 214, word 7 bit 0x200: title 1's terrain texture arrays and
// layer pools, 170 MB-5.8 GB each, of which the game fills a little) get memory per block when the
// GPU first writes there, not in full. nx2:gpu installs the handler and the span its regions cover;
// the GPU memory manager calls Nx2BackSparseWrite before each write, with the bytes it writes
// (blocks nobody wrote that only get zeros need no memory: they read zero already).
using Nx2SparseBackingHandler = std::function<void(u64 gpu_address, u64 size, const u8* data)>;
inline std::recursive_mutex g_nx2_sparse_backing_mutex;
inline Nx2SparseBackingHandler g_nx2_sparse_backing;
inline std::atomic<u64> g_nx2_sparse_low{~u64{0}};
inline std::atomic<u64> g_nx2_sparse_high{0};

inline void Nx2SetSparseBacking(Nx2SparseBackingHandler handler) {
    std::scoped_lock lk{g_nx2_sparse_backing_mutex};
    g_nx2_sparse_backing = std::move(handler);
    g_nx2_sparse_low.store(~u64{0});
    g_nx2_sparse_high.store(0);
}

/// Widens the span that writes are checked against (a new lazily backed region).
inline void Nx2AddSparseSpan(u64 gpu_address, u64 size) {
    u64 low = g_nx2_sparse_low.load();
    while (gpu_address < low && !g_nx2_sparse_low.compare_exchange_weak(low, gpu_address)) {
    }
    u64 high = g_nx2_sparse_high.load();
    while (gpu_address + size > high &&
           !g_nx2_sparse_high.compare_exchange_weak(high, gpu_address + size)) {
    }
}

inline void Nx2BackSparseWrite(u64 gpu_address, u64 size, const void* data) {
    if (gpu_address + size <= g_nx2_sparse_low.load(std::memory_order_relaxed) ||
        gpu_address >= g_nx2_sparse_high.load(std::memory_order_relaxed)) [[likely]] {
        return;
    }
    std::scoped_lock lk{g_nx2_sparse_backing_mutex};
    if (g_nx2_sparse_backing) {
        g_nx2_sparse_backing(gpu_address, size, static_cast<const u8*>(data));
    }
}

// Which blocks of each sparse region have been written (got memory), for the texture cache: an
// image over blocks nobody wrote is all zeros and needs no device memory. Its own lock, taken last:
// the texture cache asks with its lock held, and nx2:gpu updates it holding only its region mutex.
struct Nx2SparseRegionState {
    u64 size;
    u64 block_size;
    std::vector<bool> written;
    /// Once the game maps its own memory into the region (nx2:gpu 216): which tiles it maps.
    /// Only those hold data then (writes elsewhere are dropped, as on the console).
    std::vector<bool> mapped;
};

/// The granularity of the game's sparse tiles (nx2:gpu 216 maps 64 KiB pages).
inline constexpr u64 Nx2SparseTileSize = 64ULL << 10;
inline std::mutex g_nx2_sparse_state_mutex;
inline std::map<u64, Nx2SparseRegionState> g_nx2_sparse_state;

inline void Nx2SparseRegionCreated(u64 gpu_address, u64 size, u64 block_size) {
    std::scoped_lock lk{g_nx2_sparse_state_mutex};
    g_nx2_sparse_state[gpu_address] = {size, block_size,
                                       std::vector<bool>((size + block_size - 1) / block_size)};
}

inline void Nx2SparseRegionDestroyed(u64 gpu_address) {
    std::scoped_lock lk{g_nx2_sparse_state_mutex};
    g_nx2_sparse_state.erase(gpu_address);
}

/// A new emulation session: the previous game's regions are gone (nvdrv's Module clears them).
inline void Nx2ClearSparseRegions() {
    std::scoped_lock lk{g_nx2_sparse_state_mutex};
    g_nx2_sparse_state.clear();
}

inline void Nx2SparseBlockWritten(u64 region_address, u64 block) {
    std::scoped_lock lk{g_nx2_sparse_state_mutex};
    if (const auto it = g_nx2_sparse_state.find(region_address); it != g_nx2_sparse_state.end()) {
        it->second.written[block] = true;
    }
}

/// True unless the block of the region at region_address is known to hold nothing (no write from
/// the CPU or the GPU yet).
inline bool Nx2SparseBlockIsWritten(u64 region_address, u64 block) {
    std::scoped_lock lk{g_nx2_sparse_state_mutex};
    const auto it = g_nx2_sparse_state.find(region_address);
    return it == g_nx2_sparse_state.end() || block >= it->second.written.size() ||
           it->second.written[block];
}

/// The game maps its own memory over [offset, +size) of the region at region_address (mapped), or
/// unmaps it there.
inline void Nx2SparseTilesMapped(u64 region_address, u64 offset, u64 size, bool mapped) {
    std::scoped_lock lk{g_nx2_sparse_state_mutex};
    const auto it = g_nx2_sparse_state.find(region_address);
    if (it == g_nx2_sparse_state.end() || size == 0) {
        return;
    }
    auto& region = it->second;
    if (region.mapped.empty()) {
        region.mapped.resize((region.size + Nx2SparseTileSize - 1) / Nx2SparseTileSize);
    }
    const u64 last = (std::min)((offset + size - 1) / Nx2SparseTileSize, u64{region.mapped.size() - 1});
    for (u64 tile = offset / Nx2SparseTileSize; tile <= last; ++tile) {
        region.mapped[tile] = mapped;
    }
}

/// The parts of [gpu_address, +size), which lies in one sparse region, that can hold data: the
/// tiles the game maps there if it maps any, otherwise the blocks written. As (offset from
/// gpu_address, size) runs.
inline std::vector<std::pair<u64, u64>> Nx2SparseDataRanges(u64 gpu_address, u64 size) {
    std::vector<std::pair<u64, u64>> ranges;
    std::scoped_lock lk{g_nx2_sparse_state_mutex};
    auto it = g_nx2_sparse_state.upper_bound(gpu_address);
    if (it == g_nx2_sparse_state.begin()) {
        return ranges;
    }
    --it;
    const auto& [start, region] = *it;
    if (size == 0 || gpu_address + size > start + region.size) {
        return ranges;
    }
    const bool by_tile = !region.mapped.empty();
    const u64 unit = by_tile ? Nx2SparseTileSize : region.block_size;
    const std::vector<bool>& present = by_tile ? region.mapped : region.written;
    const u64 begin = gpu_address - start;
    const u64 end = begin + size;
    for (u64 index = begin / unit; index * unit < end && index < present.size(); ++index) {
        if (!present[index]) {
            continue;
        }
        const u64 run_begin = (std::max)(index * unit, begin) - begin;
        const u64 run_end = (std::min)((index + 1) * unit, end) - begin;
        if (!ranges.empty() && ranges.back().first + ranges.back().second == run_begin) {
            ranges.back().second += run_end - run_begin;
        } else {
            ranges.emplace_back(run_begin, run_end - run_begin);
        }
    }
    return ranges;
}

/// True when [gpu_address, +size) lies in one sparse region.
inline bool Nx2InSparseRegion(u64 gpu_address, u64 size) {
    if (size == 0 || gpu_address + size <= g_nx2_sparse_low.load(std::memory_order_relaxed) ||
        gpu_address >= g_nx2_sparse_high.load(std::memory_order_relaxed)) [[likely]] {
        return false;
    }
    std::scoped_lock lk{g_nx2_sparse_state_mutex};
    auto it = g_nx2_sparse_state.upper_bound(gpu_address);
    if (it == g_nx2_sparse_state.begin()) {
        return false;
    }
    --it;
    return gpu_address + size <= it->first + it->second.size;
}

/// The GPU renders into [gpu_address, +size): count its blocks as written, so images made there
/// later get memory (their contents live in the host image, not in guest memory).
inline void Nx2SparseMarkWritten(u64 gpu_address, u64 size) {
    if (size == 0 || gpu_address + size <= g_nx2_sparse_low.load(std::memory_order_relaxed) ||
        gpu_address >= g_nx2_sparse_high.load(std::memory_order_relaxed)) [[likely]] {
        return;
    }
    std::scoped_lock lk{g_nx2_sparse_state_mutex};
    for (auto& [start, region] : g_nx2_sparse_state) {
        const u64 end = start + region.size;
        if (gpu_address >= end || gpu_address + size <= start) {
            continue;
        }
        const u64 first = ((std::max)(gpu_address, start) - start) / region.block_size;
        const u64 last = ((std::min)(gpu_address + size, end) - 1 - start) / region.block_size;
        for (u64 block = first; block <= last; ++block) {
            region.written[block] = true;
        }
    }
}

/// True when [gpu_address, +size) lies in one sparse region and none of it has been written.
inline bool Nx2SparseUnwritten(u64 gpu_address, u64 size) {
    if (size == 0 || gpu_address + size <= g_nx2_sparse_low.load(std::memory_order_relaxed) ||
        gpu_address >= g_nx2_sparse_high.load(std::memory_order_relaxed)) [[likely]] {
        return false;
    }
    std::scoped_lock lk{g_nx2_sparse_state_mutex};
    auto it = g_nx2_sparse_state.upper_bound(gpu_address);
    if (it == g_nx2_sparse_state.begin()) {
        return false;
    }
    --it;
    const auto& [start, region] = *it;
    if (gpu_address + size > start + region.size) {
        return false;
    }
    const u64 last = (gpu_address + size - 1 - start) / region.block_size;
    for (u64 block = (gpu_address - start) / region.block_size; block <= last; ++block) {
        if (region.written[block]) {
            return false;
        }
    }
    return true;
}

// GPU semaphore writes as they land in guest memory: EXP13=2 marks the release records they
// belong to (#717's landed counter, per record: VideoCommon::QmdPrediction::NoteLanded, which
// sets the hook once it records). The landing sites call it BEFORE they write the word, so a
// front that reads the new word also finds the record marked.
enum class Nx2Landing : u32 {
    QueryCompletion, ///< A query cache payload write when its fence or query completes (plain)
    QueryImmediate,  ///< A query cache payload write as the back runs it (plain)
    QmdMax,          ///< A QMD max release: max(word, payload)
    QmdReduction,    ///< A QMD release reduction's result
    ReportReduction, ///< A compute report semaphore reduction's result
};
inline std::atomic<void (*)(u64, u32, Nx2Landing)> g_nx2_landed_hook{};

/// Whether anything listens for landings (EXP13=2 recorded once this session).
[[nodiscard]] inline bool Nx2LandedHooked() {
    return g_nx2_landed_hook.load(std::memory_order_relaxed) != nullptr;
}

/// A GPU semaphore write of `value` to the u32 at `device_address` is about to land.
inline void Nx2NoteLanded(u64 device_address, u32 value, Nx2Landing kind) {
    if (const auto hook = g_nx2_landed_hook.load(std::memory_order_acquire)) [[unlikely]] {
        hook(device_address, value, kind);
    }
}

/// Channels suspended on a semaphore acquire. A console channel polls and sees every release;
/// Eden writes releases in bursts when fences signal and resumes waiters only between command
/// lists, so an acquire-equal could miss its value. Writes latch the waiters they satisfy.
class Nx2SemaphoreWaiters {
public:
    // Puller GpuSemaphoreOperation values.
    static constexpr u32 AcquireEqual = 0x1;
    static constexpr u32 AcquireGequal = 0x4;
    static constexpr u32 AcquireMask = 0x8;

    static Nx2SemaphoreWaiters& Get() {
        static Nx2SemaphoreWaiters waiters;
        return waiters;
    }

    void Add(const void* owner, u64 device_address, u32 value, u32 operation) {
        std::scoped_lock lk{mutex};
        std::erase_if(waiters, [owner](const Waiter& w) { return w.owner == owner; });
        waiters.push_back({owner, device_address, value, operation, false});
        count.store(waiters.size(), std::memory_order_release);
    }

    void Remove(const void* owner) {
        if (count.load(std::memory_order_acquire) == 0) {
            return;
        }
        std::scoped_lock lk{mutex};
        std::erase_if(waiters, [owner](const Waiter& w) { return w.owner == owner; });
        count.store(waiters.size(), std::memory_order_release);
    }

    [[nodiscard]] bool Latched(const void* owner) {
        if (count.load(std::memory_order_acquire) == 0) {
            return false;
        }
        std::scoped_lock lk{mutex};
        return std::ranges::any_of(waiters,
                                   [owner](const Waiter& w) { return w.owner == owner && w.latched; });
    }

    /// A GPU write of `value` to the u32 at `device_address`.
    void OnWrite(u64 device_address, u32 value) {
        // (Common::Nx2GpuWrite::Notify, for Ocarina of Time's present poll, and Nx2NoteLanded
        // are called by each write site itself.)
        if (count.load(std::memory_order_acquire) == 0) {
            return;
        }
        std::scoped_lock lk{mutex};
        for (auto& w : waiters) {
            if (w.address == device_address && !w.latched && Matches(w, value)) {
                w.latched = true;
            }
        }
    }

private:
    struct Waiter {
        const void* owner;
        u64 address;
        u32 value;
        u32 operation;
        bool latched;
    };

    static bool Matches(const Waiter& w, u32 current) {
        switch (w.operation) {
        case AcquireEqual:
            return current == w.value;
        case AcquireGequal:
            return current >= w.value;
        case AcquireMask:
            return (current & w.value) != 0;
        default:
            return false;
        }
    }

    std::mutex mutex;
    std::vector<Waiter> waiters;
    std::atomic<size_t> count{};
};

// Render-frame clock (EDEN_NX2_RENDER_CLOCK=1, fable_answer_16 1b and 17): one count per guest
// frame, bumped where the frame's last GPU work is, so the texture cache's relocation rules can
// tell frames apart where the host present tick (mid-stream) cannot. Sources:
// - by default the flip's frame fence (Nx2FrameFenceClock, video_core/nx2_frame_fence.h),
//   bumped on the split's back in stream order (FenceManager::SignalSyncPoint);
// - EDEN_NX2_RENDER_CLOCK_SYNCPT=<id>: every back signal of that syncpoint (manual override);
// - EDEN_NX2_RENDER_CLOCK=2: a write to the present poll's learned word
//   (core/hle/kernel/svc/svc_nx2_oot_present.cpp OnGpuWrite, needs EXP24=1). A guest thread
//   writes that word, so it never fires (fable_answer_17 0); kept to compare.
// Nothing counts while the texture cache does not want the clock (EDEN_NX2_RENDER_CLOCK or
// EXP43=4).
inline std::atomic<bool> g_nx2_render_clock_wanted{false};
inline std::atomic<u64> g_nx2_render_frame{};
// Read on first use, not at static initialization: that runs before main() applies
// nx2_switches.txt, so the file's value would be lost.
inline int Nx2RenderClockSyncpt() {
    static const int syncpt = [] {
        const char* value = std::getenv("EDEN_NX2_RENDER_CLOCK_SYNCPT");
        return value != nullptr && value[0] != '\0' ? std::atoi(value) : -1;
    }();
    return syncpt;
}
/// EDEN_NX2_RENDER_CLOCK: 0 unset, 1 the clock, 2 the clock on the poll-word source.
inline int Nx2RenderClockEnv() {
    static const int env = [] {
        // On (1) by default since 2026-11-08: flip-fence clock at 1.00 per host frame, 0 tombstone
        // hits under EXP43=4, 11 loads 0 washed out. EDEN_NX2_RENDER_CLOCK=0 turns it off.
        const char* value = std::getenv("EDEN_NX2_RENDER_CLOCK");
        return value == nullptr ? 1 : value[0] == '1' ? 1 : value[0] == '2' ? 2 : 0;
    }();
    return env;
}
inline bool Nx2RenderClockWordSource() {
    return Nx2RenderClockSyncpt() < 0 && Nx2RenderClockEnv() == 2;
}
inline bool Nx2RenderClockFlipSource() {
    return Nx2RenderClockSyncpt() < 0 && Nx2RenderClockEnv() != 2;
}
// Syncpoint signals by id while the clock is wanted (logged with the clock).
inline constexpr size_t Nx2SyncpointSignalSlots = 192;
inline std::array<std::atomic<u32>, Nx2SyncpointSignalSlots> g_nx2_syncpoint_signals{};
// Numbers the nx2:syncpt sessions (the frame fence's slot key is (session << 32) | slot).
inline std::atomic<u32> g_nx2_syncpt_sessions{};

/// The back signals syncpoint `id`, which then reads `new_value` (FenceManager::SignalSyncPoint,
/// before the increment; only called while the clock is wanted).
inline void Nx2NoteSyncpointSignal(u32 id, u32 new_value) {
    if (!g_nx2_render_clock_wanted.load(std::memory_order_relaxed) ||
        id >= Nx2SyncpointSignalSlots) {
        return;
    }
    g_nx2_syncpoint_signals[id].fetch_add(1, std::memory_order_relaxed);
    if (const int override_id = Nx2RenderClockSyncpt(); override_id >= 0) {
        if (static_cast<int>(id) == override_id) {
            g_nx2_render_frame.fetch_add(1, std::memory_order_relaxed);
        }
    } else if (id == g_nx2_frame_fence.syncpoint.load(std::memory_order_relaxed) &&
               Nx2RenderClockFlipSource()) {
        if (g_nx2_frame_fence.NoteSignal(new_value)) {
            g_nx2_render_frame.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

} // namespace Tegra
