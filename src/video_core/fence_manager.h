// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#include <queue>

#include "common/common_types.h"
#include "common/logging.h"
#include "common/settings.h"
#include "common/thread.h"
#include "video_core/delayed_destruction_ring.h"
#include "video_core/gpu.h"
#include "video_core/host1x/host1x.h"
#include "video_core/host1x/syncpoint_manager.h"
#include "video_core/nx2_profile.h"
#include "video_core/nx2_query_fast.h"
#include "video_core/nx2_reactive.h"
#include "video_core/rasterizer_interface.h"

namespace VideoCommon {

/// EDEN_NX2_GPU_PROFILE: what the fence manager does with each SignalFence, logged per second
/// every 10 s by the thread that signals.
namespace Nx2FenceStats {
enum Stat : u32 {
    Signals,         ///< SignalFence calls
    Skipped,         ///< Empty fences not made (SkipEmptyFence)
    Real,            ///< Fences with downloads or query results: they wait for the host GPU
    Stubbed,         ///< Fences with only queued work (no host GPU wait)
    ForcedSkip,      ///< Of Stubbed: the 64th empty fence in a row
    QueryWork,       ///< Of Stubbed: query host syncs
    QueuedOperations, ///< Of Stubbed: queued operations
    Accurate,        ///< Of Stubbed: accurate GPU fences (callbacks wait for the fence)
    Folded,          ///< Of Stubbed: work added to the newest queued fence (EDEN_NX2_FENCE_FOLD)
    Inlined,         ///< Of Stubbed: work run at once, nothing queued (EDEN_NX2_FENCE_FOLD)
    Syncpoints,      ///< SignalSyncPoint calls
    SyncpointGuestNs, ///< Time in the guest increment's actions (the present's composite too)
    Count,
};
inline std::array<std::atomic<u64>, Count> g_stats{};
inline std::atomic<u64> g_window_start{};

inline void Add(Stat stat, u64 amount = 1) {
    g_stats[stat].fetch_add(amount, std::memory_order_relaxed);
}

inline void MaybeReport() {
    const u64 now = Nx2Profile::Now();
    u64 start = g_window_start.load(std::memory_order_relaxed);
    if (start == 0) {
        g_window_start.compare_exchange_strong(start, now);
        return;
    }
    if (now - start < 10'000'000'000ULL || !g_window_start.compare_exchange_strong(start, now)) {
        return;
    }
    const double seconds = static_cast<double>(now - start) / 1e9;
    std::array<double, Count> rate{};
    for (u32 i = 0; i < Count; ++i) {
        rate[i] = static_cast<double>(g_stats[i].exchange(0, std::memory_order_relaxed)) / seconds;
    }
    LOG_INFO(HW_GPU,
             "Fences per second: {:.0f} signals, {:.0f} skipped, {:.0f} waiting for the host "
             "GPU, {:.0f} stubbed (64th skip {:.0f}, query syncs {:.0f}, queued operations {:.0f}, "
             "accurate {:.0f}; folded {:.0f}, inline {:.0f}) | syncpoint increments {:.0f}, guest "
             "actions {:.1f} ms",
             rate[Signals], rate[Skipped], rate[Real], rate[Stubbed], rate[ForcedSkip],
             rate[QueryWork], rate[QueuedOperations], rate[Accurate], rate[Folded], rate[Inlined],
             rate[Syncpoints], rate[SyncpointGuestNs] / 1e6);
}
} // namespace Nx2FenceStats

class FenceBase {
public:
    explicit FenceBase(bool is_stubbed_) : is_stubbed{is_stubbed_} {}

    bool IsStubbed() const {
        return is_stubbed;
    }

protected:
    bool is_stubbed;
};

template <typename Traits>
class FenceManager {
    using TFence = typename Traits::FenceType;
    using TTextureCache = typename Traits::TextureCacheType;
    using TBufferCache = typename Traits::BufferCacheType;
    using TQueryCache = typename Traits::QueryCacheType;
    static constexpr bool can_async_check = Traits::HAS_ASYNC_CHECK;

public:
    /// Notify the fence manager about a new frame
    void TickFrame() {
        std::unique_lock lock(ring_guard);
        delayed_destruction_ring.Tick();
        Nx2Reactive::g_frame.fetch_add(1, std::memory_order_relaxed);
    }

    // Unlike other fences, this one doesn't
    void SignalOrdering() {
        if constexpr (!can_async_check) {
            TryReleasePendingFences<false>();
        }
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.AccumulateFlushes();
    }

    void SignalReference() {
        std::function<void()> do_nothing([] {});
        SignalFence(std::move(do_nothing));
    }

    void SyncOperation(std::function<void()>&& func) {
        // Not only the GPU thread queues these (query cache work runs elsewhere too), while
        // SignalFence moves the list into pending_operations: unguarded, the list was corrupted
        // and the fencing thread crashed walking it. Its own lock, held for no callbacks.
        std::scoped_lock lock{uncommitted_guard};
        uncommitted_operations.emplace_back(std::move(func));
    }

    void SignalFence(std::function<void()>&& func) {
        Nx2Profile::Scope profile_scope{Nx2Profile::Zone::Fence};
        const bool profiling = Nx2Profile::Enabled();
        if (profiling) [[unlikely]] {
            Nx2FenceStats::Add(Nx2FenceStats::Signals);
            Nx2FenceStats::MaybeReport();
        }
        // A guest read waits for downloads to land (Nx2Reactive::WaitLanding): submit the
        // deferred work now rather than at the end of the batching window.
        if (Nx2Reactive::GetLanding().waiters.load(std::memory_order_relaxed) != 0) {
            SubmitDeferredWork();
        }
        const bool delay_fence = Settings::IsGPUFenceBehaviorDefault() ? Settings::IsGPULevelHigh() : Settings::IsGPUFenceBehaviorBalanced() || Settings::IsGPUFenceBehaviorAccurate();
        if constexpr (!can_async_check) {
            TryReleasePendingFences<false>();
        }
        auto [should_flush, committed_flushes] = CommitCacheFlushes();
        skip_refusal.store(Nx2FenceStats::Accurate, std::memory_order_relaxed);
        if (!delay_fence && committed_flushes == 0 && SkipEmptyFence()) {
            Nx2Profile::Scope skipped_scope{Nx2Profile::Zone::FenceSkipped};
            if (profiling) [[unlikely]] {
                Nx2FenceStats::Add(Nx2FenceStats::Skipped);
            }
            func();
            rasterizer.InvalidateGPUCache();
            return;
        }
        skipped_fences = 0;
        should_flush |= query_cache.HasUncommittedFlushes();
        query_cache.CommitAsyncFlushes();
        const bool stubbed = !should_flush && committed_flushes == 0;
        if (profiling) [[unlikely]] {
            Nx2FenceStats::Add(stubbed ? Nx2FenceStats::Stubbed : Nx2FenceStats::Real);
            if (stubbed) {
                Nx2FenceStats::Add(skip_refusal.load(std::memory_order_relaxed));
            }
        }
        if constexpr (can_async_check) {
            if (stubbed && FenceFold()) {
                // As the made fence below: its callback runs now, before its queued work.
                if (!delay_fence) {
                    func();
                }
                FoldStubbedFence(delay_fence, func);
                if (delay_fence) {
                    SubmitDeferredWork();
                }
                rasterizer.InvalidateGPUCache();
                return;
            }
        }
        TFence new_fence = CreateFence(!should_flush);
        if constexpr (can_async_check) {
            guard.lock();
        }
        {
            std::scoped_lock lock{uncommitted_guard};
            if (delay_fence) {
                uncommitted_operations.emplace_back(std::move(func));
            }
            pending_operations.emplace_back(std::move(uncommitted_operations));
            uncommitted_operations.clear(); // moved-from: make it empty explicitly
        }
        pending_flushes.push_back(committed_flushes);
        if constexpr (can_async_check) {
            pending_query_pops.push_back(1);
        }
        QueueFence(new_fence);
        if (!delay_fence) {
            func();
        }
        fences.push(std::move(new_fence));
        if (should_flush) {
            rasterizer.FlushCommands();
        }
        if constexpr (can_async_check) {
            guard.unlock();
            cv.notify_all();
        }
        if (delay_fence) {
            SubmitDeferredWork();
        }
        rasterizer.InvalidateGPUCache();
    }

    void SignalSyncPoint(u32 value) {
        // The id comes from a command list (the 3D engine's sync method, the puller's fence
        // action). Past the syncpoint table, the increment would walk memory after it as an
        // action list; command words read wrong carry any id (Splatoon Raiders on Linux).
        if (value >= Tegra::Host1x::SyncpointManager::NUM_MAX_SYNCPOINTS) {
            LOG_ERROR(HW_GPU, "Signal of syncpoint {} past the table, ignored", value);
            return;
        }
        // The guest sees the guest value (nx2:syncpt, the present's fence wait); its actions run
        // here, the present's composite and frame tick among them. The host value goes with the
        // fence: with accurate fences it joins the newest queued fence (FoldStubbedFence).
        if (Nx2Profile::Enabled()) [[unlikely]] {
            const u64 start = Nx2Profile::Now();
            syncpoint_manager.IncrementGuest(value);
            Nx2FenceStats::Add(Nx2FenceStats::Syncpoints);
            Nx2FenceStats::Add(Nx2FenceStats::SyncpointGuestNs, Nx2Profile::Now() - start);
        } else {
            syncpoint_manager.IncrementGuest(value);
        }
        std::function<void()> func([this, value] { syncpoint_manager.IncrementHost(value); });
        SignalFence(std::move(func));
    }

    void WaitPendingFences([[maybe_unused]] bool force) {
        if constexpr (!can_async_check) {
            TryReleasePendingFences<true>();
        } else {
            if (!force) {
                return;
            }
            std::mutex wait_mutex;
            std::condition_variable wait_cv;
            std::atomic<bool> wait_finished{};
            std::function<void()> func([&] {
                std::scoped_lock lk(wait_mutex);
                wait_finished.store(true, std::memory_order_relaxed);
                wait_cv.notify_all();
            });
            SignalFence(std::move(func));
            // The fence's commands can be held back by a deferred flush that only this thread
            // submits. The fencing thread waits for that submission before releasing the fence,
            // so waiting here without submitting deadlocks (Ocarina of Time entering gameplay).
            SubmitDeferredWork();
            std::unique_lock lk(wait_mutex);
            wait_cv.wait(
                lk, [&wait_finished] { return wait_finished.load(std::memory_order_relaxed); });
        }
    }

protected:
    explicit FenceManager(VideoCore::RasterizerInterface& rasterizer_, Tegra::GPU& gpu_,
                          TTextureCache& texture_cache_, TBufferCache& buffer_cache_,
                          TQueryCache& query_cache_)
        : rasterizer{rasterizer_}, gpu{gpu_}, syncpoint_manager{gpu.Host1x().GetSyncpointManager()},
          texture_cache{texture_cache_}, buffer_cache{buffer_cache_}, query_cache{query_cache_} {
        if constexpr (can_async_check) {
            fence_thread =
                std::jthread([this](std::stop_token token) { ReleaseThreadFunc(token); });
        }
    }

    virtual ~FenceManager() {
        if constexpr (can_async_check) {
            fence_thread.request_stop();
            cv.notify_all();
            fence_thread.join();
        }
    }

    /// Creates a Fence Interface, does not create a backend fence if 'is_stubbed' is
    /// true
    virtual TFence CreateFence(bool is_stubbed) = 0;
    /// Queues a fence into the backend if the fence isn't stubbed.
    virtual void QueueFence(TFence& fence) = 0;
    /// Notifies that the backend fence has been signaled/reached in host GPU.
    virtual bool IsFenceSignaled(TFence& fence) const = 0;
    /// Waits until a fence has been signalled by the host GPU.
    virtual void WaitFence(TFence& fence) = 0;
    /// Submits work whose submission the backend deferred (called before blocking on a fence).
    virtual void SubmitDeferredWork() {}

    VideoCore::RasterizerInterface& rasterizer;
    Tegra::GPU& gpu;
    Tegra::Host1x::SyncpointManager& syncpoint_manager;
    TTextureCache& texture_cache;
    TBufferCache& buffer_cache;
    TQueryCache& query_cache;

private:
    template <bool force_wait>
    void TryReleasePendingFences() {
        while (!fences.empty()) {
            TFence& current_fence = fences.front();
            if (ShouldWait() && !IsFenceSignaled(current_fence)) {
                if constexpr (force_wait) {
                    WaitFence(current_fence);
                } else {
                    return;
                }
            }
            PopAsyncFlushes(pending_flushes.front());
            pending_flushes.pop_front();
            auto operations = std::move(pending_operations.front());
            pending_operations.pop_front();
            for (auto& operation : operations) {
                operation();
            }
            {
                std::unique_lock lock(ring_guard);
                delayed_destruction_ring.Push(std::move(current_fence));
            }
            fences.pop();
        }
    }

    void ReleaseThreadFunc(std::stop_token stop_token) {
        Common::SetCurrentThreadName("GPUFencingThread");
        Common::SetCurrentThreadPriority(Common::ThreadPriority::High);
        Common::SetCurrentThreadToPerformanceCores();

        // Every fence the host GPU has already passed is released together: Switch 2 titles
        // signal thousands a second and the host GPU completes them in bursts, so each cache's
        // lock is taken once per burst instead of once per fence (the GPU thread, which holds
        // them for nearly every command, spun, slept and woke this thread for each one).
        // EDEN_NX2_FENCE_BATCH=0: one fence at a time (for comparisons).
        static const size_t MaxBatch = [] {
            const char* value = std::getenv("EDEN_NX2_FENCE_BATCH");
            return value != nullptr && value[0] == '0' ? size_t{1} : size_t{64};
        }();
        struct Released {
            TFence fence;
            std::deque<std::function<void()>> operations;
            u8 flushes;
            u32 query_pops; ///< Query cache entries: one per fence, folded ones included
        };
        std::vector<Released> batch;
        batch.reserve(MaxBatch);
        const auto take_front = [&] {
            batch.push_back({std::move(fences.front()), std::move(pending_operations.front()),
                             pending_flushes.front(), pending_query_pops.front()});
            fences.pop();
            pending_operations.pop_front();
            pending_flushes.pop_front();
            pending_query_pops.pop_front();
        };
        while (!stop_token.stop_requested()) {
            {
                std::unique_lock lock(guard);
                // Work the GPU thread runs inline (FoldStubbedFence) goes before any fence
                // queued meanwhile.
                cv.wait(lock, [&] {
                    return stop_token.stop_requested() || (!fences.empty() && !inline_running);
                });
                if (stop_token.stop_requested()) [[unlikely]] {
                    return;
                }
                take_front();
                fence_thread_busy = true;
            }
            if (!batch.front().fence->IsStubbed()) {
                WaitFence(batch.front().fence);
            }
            {
                std::unique_lock lock(guard);
                while (batch.size() < MaxBatch && !fences.empty() &&
                       (fences.front()->IsStubbed() || IsFenceSignaled(fences.front()))) {
                    take_front();
                }
            }
            size_t texture_pops = 0;
            size_t buffer_pops = 0;
            for (const Released& released : batch) {
                texture_pops += (released.flushes & TextureFlushes) != 0 ? 1 : 0;
                buffer_pops += (released.flushes & BufferFlushes) != 0 ? 1 : 0;
            }
            if (texture_pops != 0) {
                std::scoped_lock lock{texture_cache.mutex};
                for (size_t i = 0; i < texture_pops; ++i) {
                    texture_cache.PopAsyncFlushes();
                }
            }
            if (buffer_pops != 0) {
                std::scoped_lock lock{buffer_cache.mutex};
                for (size_t i = 0; i < buffer_pops; ++i) {
                    buffer_cache.PopAsyncFlushes();
                }
            }
            // Queries and operations keep their per-fence order.
            for (Released& released : batch) {
                for (u32 i = 0; i < released.query_pops; ++i) {
                    query_cache.PopAsyncFlushes();
                }
                for (auto& operation : released.operations) {
                    operation();
                }
            }
            {
                std::unique_lock lock(ring_guard);
                for (Released& released : batch) {
                    delayed_destruction_ring.Push(std::move(released.fence));
                }
            }
            batch.clear();
            {
                std::scoped_lock lock(guard);
                fence_thread_busy = false;
            }
        }
    }

    /// EDEN_NX2_FENCE_FOLD: a fence with nothing to wait for on the host GPU (no downloads, no
    /// query results) carries only queued work: the query cache's slot cleanup, operations, and
    /// with accurate fences its callback (a syncpoint's host increment). The fencing thread runs
    /// that work as soon as every fence before it is released, so instead of a fence of its own
    /// (an allocation, a queue entry and a wake of the fencing thread, ~2,000 a second):
    /// - with fences queued, the work joins the newest one: the fencing thread releases a fence
    ///   without a host wait together with the one before it, in order;
    /// - with nothing queued or being released, the GPU thread runs it at once (the same order);
    /// - otherwise (the fencing thread is releasing fences it took) it gets a fence as before.
    static bool FenceFold() {
        static const bool enabled = [] {
            const char* value = std::getenv("EDEN_NX2_FENCE_FOLD");
            return value == nullptr || value[0] != '0';
        }();
        return enabled;
    }

    void FoldStubbedFence(bool delay_fence, std::function<void()>& func) {
        std::unique_lock lock{guard};
        if (!fences.empty()) {
            {
                std::scoped_lock ops_lock{uncommitted_guard};
                auto& operations = pending_operations.back();
                for (auto& operation : uncommitted_operations) {
                    operations.emplace_back(std::move(operation));
                }
                uncommitted_operations.clear();
                if (delay_fence) {
                    operations.emplace_back(std::move(func));
                }
            }
            // This fence's query cache entry (no queries: CommitAsyncFlushes found none).
            ++pending_query_pops.back();
            if (Nx2Profile::Enabled()) [[unlikely]] {
                Nx2FenceStats::Add(Nx2FenceStats::Folded);
            }
            return;
        }
        if (!fence_thread_busy && !inline_running && GpuSplit::OnBackThread()) {
            inline_running = true;
            std::deque<std::function<void()>> operations;
            {
                std::scoped_lock ops_lock{uncommitted_guard};
                operations.swap(uncommitted_operations);
            }
            lock.unlock();
            query_cache.PopAsyncFlushes();
            for (auto& operation : operations) {
                operation();
            }
            if (delay_fence) {
                func();
            }
            lock.lock();
            inline_running = false;
            const bool queued = !fences.empty();
            lock.unlock();
            if (queued) {
                cv.notify_all();
            }
            if (Nx2Profile::Enabled()) [[unlikely]] {
                Nx2FenceStats::Add(Nx2FenceStats::Inlined);
            }
            return;
        }
        TFence new_fence = CreateFence(true);
        {
            std::scoped_lock ops_lock{uncommitted_guard};
            if (delay_fence) {
                uncommitted_operations.emplace_back(std::move(func));
            }
            pending_operations.emplace_back(std::move(uncommitted_operations));
            uncommitted_operations.clear();
        }
        pending_flushes.push_back(0);
        pending_query_pops.push_back(1);
        QueueFence(new_fence);
        fences.push(std::move(new_fence));
        lock.unlock();
        cv.notify_all();
    }

    bool ShouldWait() const {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        return texture_cache.ShouldWaitAsyncFlushes() || buffer_cache.ShouldWaitAsyncFlushes() ||
               query_cache.ShouldWaitAsyncFlushes();
    }

    static constexpr u8 TextureFlushes = 1;
    static constexpr u8 BufferFlushes = 2;

    /// committed: what CommitAsyncFlushes returned for this fence.
    void PopAsyncFlushes(u8 committed) {
        if (committed & TextureFlushes) {
            std::scoped_lock lock{texture_cache.mutex};
            texture_cache.PopAsyncFlushes();
        }
        if (committed & BufferFlushes) {
            std::scoped_lock lock{buffer_cache.mutex};
            buffer_cache.PopAsyncFlushes();
        }
        query_cache.PopAsyncFlushes();
    }

    struct CacheFlushes {
        bool should_flush;  ///< Downloads to wait for: the fence must reach the host GPU
        u8 committed;       ///< TextureFlushes, BufferFlushes: the caches that committed downloads
    };

    /// Commits the texture and buffer caches' downloads for a new fence (the query cache's are
    /// committed once a fence is made). Switch 2 titles signal a fence per compute dispatch
    /// (thousands a second), mostly with nothing or only buffer data to download: each cache's
    /// lock is taken once, alone, and the fencing thread later takes only the lock of the cache
    /// with work, instead of holding both while it copies (the GPU thread's draws and uploads
    /// waited for it).
    CacheFlushes CommitCacheFlushes() {
        CacheFlushes flushes{};
        {
            std::scoped_lock lock{texture_cache.mutex};
            flushes.should_flush = texture_cache.HasUncommittedFlushes();
            if (texture_cache.HasAsyncFlushWork()) {
                texture_cache.CommitAsyncFlushes();
                flushes.committed |= TextureFlushes;
            }
        }
        {
            std::scoped_lock lock{buffer_cache.mutex};
            if (buffer_cache.HasUncommittedFlushes()) {
                buffer_cache.CommitAsyncFlushes();
                flushes.should_flush = true;
                flushes.committed |= BufferFlushes;
            }
        }
        return flushes;
    }

    /// Whether a fence with no downloads would do nothing, so need not be made: with fast GPU
    /// fences its callback runs at once anyway, and most of the per-dispatch fences of Switch 2
    /// titles have no queries or queued operations either. Making, queueing and retiring them
    /// cost the GPU thread ~5% and woke the fencing thread for each. One in 64 is still made, so
    /// the operations fences carry along (query slot cleanup) keep running.
    bool SkipEmptyFence() {
        if constexpr (requires { query_cache.HasPendingFenceWork(true); }) {
            // EDEN_NX2_FENCE_TRIM: payload releases (QMD releases, ~4,300 a second in Ocarina)
            // write guest memory at once anyway; their host buffer syncs (a render pass end and
            // two barriers each) go together with the next fence made or host wait.
            if (skipped_fences >= 64) {
                skip_refusal.store(Nx2FenceStats::ForcedSkip, std::memory_order_relaxed);
                return false;
            }
            if (query_cache.HasPendingFenceWork(Nx2QueryFast::FenceTrim())) {
                skip_refusal.store(Nx2FenceStats::QueryWork, std::memory_order_relaxed);
                return false;
            }
            {
                std::scoped_lock lock{uncommitted_guard};
                if (!uncommitted_operations.empty()) {
                    skip_refusal.store(Nx2FenceStats::QueuedOperations,
                                       std::memory_order_relaxed);
                    return false;
                }
            }
            ++skipped_fences;
            return true;
        } else {
            return false;
        }
    }

    std::queue<TFence> fences;
    std::deque<std::function<void()>> uncommitted_operations;
    std::deque<std::deque<std::function<void()>>> pending_operations;
    std::deque<u8> pending_flushes; ///< Per fence: the caches that committed downloads
    std::deque<u32> pending_query_pops; ///< Async check, per fence: query cache entries to pop
    u32 skipped_fences{}; ///< Empty fences skipped since the last one made (GPU thread only)
    /// Why the last fence was made (profile counters; any signalling thread, so atomic)
    std::atomic<Nx2FenceStats::Stat> skip_refusal{};
    bool fence_thread_busy{}; ///< Guarded by guard: the fencing thread releases fences it took
    bool inline_running{};    ///< Guarded by guard: FoldStubbedFence runs a fence's work inline

    std::mutex guard;
    std::mutex ring_guard;
    std::mutex uncommitted_guard; ///< Guards uncommitted_operations
    std::condition_variable cv;

    std::jthread fence_thread;

    DelayedDestructionRing<TFence, 8> delayed_destruction_ring;
};

} // namespace VideoCommon
