// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "common/logging.h"

// Reactive flushing of GPU-written buffer data on Switch 2 titles. Their GPU writes are not
// copied back at fences (BufferCache::AsyncBufferDownloads): a guest CPU read of a GPU-written
// page traps, drains the GPU thread and waits for the host GPU to go idle (Scheduler::Finish)
// before the data is copied back (GPU::OnCPURead). Ocarina takes ~160 of these a second, ~5 a
// frame, mostly on a few hundred pages it reads every frame.
namespace VideoCommon::Nx2Reactive {

/// EDEN_NX2_REACTIVE_PREFETCH (on unless 0): GPU writes over pages the guest CPU has read GPU
/// data from (BufferCache::cpu_read_ranges) are also copied back at the next fence; their traps
/// stay armed. A read after the copy reaches guest memory goes on without a stall and sees
/// current data; an earlier one waits for the copy (WaitLanding) or flushes. A pop skips the
/// guest CPU writes not yet applied (BufferCache::PendingCpuWriteRanges), so it cannot write the
/// GPU's older bytes over them.
inline bool Prefetch() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_REACTIVE_PREFETCH");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

/// EDEN_NX2_REACTIVE_WAIT_LANDING (on unless 0): a guest read of a GPU-written page whose GPU
/// writes all have a download recorded (Prefetch) waits for those downloads to land instead of
/// flushing: the GPU thread neither copies again nor waits for the host GPU (Scheduler::Finish).
/// After 16 ms without the copy landing the read flushes as before.
inline bool WaitLanding() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_REACTIVE_WAIT_LANDING");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

/// EDEN_NX2_REACTIVE_ASYNC_FLUSH (on unless 0): a guest read of GPU-written buffer data with no
/// download on its way has the GPU thread queue that data for download at a fence it submits at
/// once (RasterizerInterface::QueueFlushRegion), then waits for the copy to land as WaitLanding
/// does, instead of the GPU thread copying it and waiting for the host GPU to go idle
/// (Scheduler::Finish). The copy holds what the GPU wrote up to the GPU thread's position when it
/// took the request, as the flush's did, and the read goes on only once guest memory has it (no
/// stale data, unlike preflushing). Needs Prefetch and WaitLanding; image data still flushes.
inline bool AsyncFlush() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_REACTIVE_ASYNC_FLUSH");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

/// EDEN_NX2_REACTIVE_HOT_FRAMES (default 64, 0 never forgets): pages the guest CPU read GPU data
/// from keep their later GPU writes copied back (Prefetch) while they are read at least once every
/// 64 to 128 frames. A forgotten page's next read flushes as a first read does.
inline u64 HotFrames() {
    static const u64 frames = [] {
        const char* value = std::getenv("EDEN_NX2_REACTIVE_HOT_FRAMES");
        return value != nullptr ? static_cast<u64>(std::strtoull(value, nullptr, 10)) : u64{64};
    }();
    return frames;
}

/// Frames ended so far (FenceManager::TickFrame).
inline std::atomic<u64> g_frame{};

/// Wakes the reads waiting for downloads to land (WaitLanding) when a fence's downloads pop.
struct Landing {
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<u64> generation{};
    std::atomic<u32> waiters{};
};

inline Landing& GetLanding() {
    static Landing landing;
    return landing;
}

inline void NotifyLanded() {
    Landing& landing = GetLanding();
    landing.generation.fetch_add(1, std::memory_order_acq_rel);
    if (landing.waiters.load(std::memory_order_acquire) != 0) {
        std::scoped_lock lock{landing.mutex};
        landing.cv.notify_all();
    }
}

} // namespace VideoCommon::Nx2Reactive
