// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <thread>

#include "common/common_types.h"

namespace VideoCore {

struct RasterizerDownloadArea {
    VAddr start_address;
    VAddr end_address;
    bool preemtive;
};

/// What DownloadLanding publishes.
struct DownloadLandingState {
    /// Odd while a landing is published.
    std::atomic<u64> sequence{};
    std::atomic<DAddr> begin{};
    std::atomic<DAddr> end{};
    std::atomic<u64> waits{};
};

/// The buffer cache writing downloaded GPU bytes to guest memory [begin, end). A download keeps the
/// bytes of the guest CPU writes collected and not yet applied when it looks at them
/// (BufferCache::PendingCpuWriteRanges). A guest core collects a write
/// (Core::GPUDirtyMemoryManager::Collect) and stores its bytes right after, without the buffer
/// cache's lock: a write collected after the download looked had its bytes written over with the
/// GPU's older ones. Downloads now look only inside a landing, and a write collected in the
/// landing's range stores its bytes once the landing has ended (Await, from Core::Memory's
/// HandleRasterizerWrite).
///
/// Each side fences between its store (the landing, the collected write) and its load (the
/// collected writes, the landing): either the download finds the write, or the write finds the
/// landing. Downloads land under the buffer cache's lock, so one landing is published at a time.
/// (zortos293/Lichii#805 4942eb8a39.)
class DownloadLanding {
public:
    explicit DownloadLanding(DAddr begin, DAddr end) noexcept {
        state.begin.store(begin, std::memory_order_relaxed);
        state.end.store(end, std::memory_order_relaxed);
        state.sequence.fetch_add(1, std::memory_order_seq_cst);
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
    ~DownloadLanding() {
        state.sequence.fetch_add(1, std::memory_order_release);
    }
    DownloadLanding(const DownloadLanding&) = delete;
    DownloadLanding& operator=(const DownloadLanding&) = delete;

    /// A guest CPU write of [addr, addr + size) was collected: returns once its bytes may be
    /// stored.
    static void Await(DAddr addr, u64 size) noexcept {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        const u64 sequence = state.sequence.load(std::memory_order_acquire);
        if ((sequence & 1) != 0) [[unlikely]] {
            Wait(sequence, addr, size);
        }
    }

    /// Writes that waited for a landing so far.
    [[nodiscard]] static u64 Waits() noexcept {
        return state.waits.load(std::memory_order_relaxed);
    }

private:
    static void Wait(u64 sequence, DAddr addr, u64 size) noexcept {
        const DAddr begin = state.begin.load(std::memory_order_relaxed);
        const DAddr end = state.end.load(std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_acquire);
        // Ended (its writes come first, and the landings after it found this write), or it
        // writes elsewhere.
        if (state.sequence.load(std::memory_order_acquire) != sequence || addr >= end ||
            addr + size <= begin) {
            return;
        }
        state.waits.fetch_add(1, std::memory_order_relaxed);
        while (state.sequence.load(std::memory_order_acquire) == sequence) {
            std::this_thread::yield();
        }
    }

    static inline DownloadLandingState state;
};

} // namespace VideoCore
