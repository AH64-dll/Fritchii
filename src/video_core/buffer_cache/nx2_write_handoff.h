// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdlib>

#include "common/common_types.h"

namespace VideoCommon {

/// EDEN_NX2_WRITE_HANDOFF: a guest write trap that finds the buffer cache's lock held (the GPU
/// thread holds it through each draw) queues its range here instead of waiting, and whoever holds
/// the lock applies the queue right after taking it and right before releasing it
/// (SpinRecursiveMutex::SetOwnerHook). Ranges are kept as GPUDirtyMemoryManager keeps them: 64-byte
/// granules of 2 KiB pages, one page per slot.
class Nx2WriteHandoff {
public:
    [[nodiscard]] static bool Enabled() {
        static const bool enabled = [] {
            const char* value = std::getenv("EDEN_NX2_WRITE_HANDOFF");
            return value == nullptr || value[0] != '0';
        }();
        return enabled;
    }

    /// Queues [address, address + size). False when it did not fit (part of it may be queued):
    /// the caller then takes the lock, which applies what was queued.
    bool Push(u64 address, u64 size) {
        if (size == 0) {
            return true;
        }
        const u64 end = address + size;
        const u64 first = address >> PageBits;
        const u64 last = (end - 1) >> PageBits;
        if (end < address || (last >> 32) != 0 || last - first > 1) {
            return false;
        }
        bool queued = true;
        for (u64 page = first; page <= last && queued; ++page) {
            const u64 page_begin = page << PageBits;
            const u64 begin = (std::max)(address, page_begin) - page_begin;
            const u64 page_end = (std::min)(end, page_begin + PageSize) - page_begin;
            const u32 low = static_cast<u32>(begin >> GranuleBits);
            const u32 high = static_cast<u32>((page_end + GranuleSize - 1) >> GranuleBits);
            const u32 mask = (high >= 32 ? ~u32{0} : (u32{1} << high) - 1) & ~((u32{1} << low) - 1);
            queued = PushPage(static_cast<u32>(page), mask);
        }
        // Flagged even when only a part went in: the caller's lock must apply that part now, not
        // a later holder after newer GPU writes.
        if (pending.load(std::memory_order_seq_cst) == 0) {
            pending.store(1, std::memory_order_seq_cst);
        }
        // The caller's check after this (OnCPUWrite) must not be satisfied by a value older than
        // the queue's.
        std::atomic_thread_fence(std::memory_order_seq_cst);
        return queued;
    }

    /// Calls func(address, size) for each queued run of granules (adjacent pages merged), in
    /// address order, and empties the queue. Runs on the lock's owner.
    template <typename Func>
    void Drain(Func&& func) {
        if (pending.load(std::memory_order_seq_cst) == 0) {
            return;
        }
        pending.store(0, std::memory_order_seq_cst);
        std::array<u64, NumSlots> taken;
        size_t count = 0;
        for (auto& slot : slots) {
            if (slot.load(std::memory_order_relaxed) == 0) {
                continue;
            }
            if (const u64 value = slot.exchange(0, std::memory_order_seq_cst); value != 0) {
                taken[count++] = value;
            }
        }
        std::sort(taken.begin(), taken.begin() + count);
        u64 run_begin = 0;
        u64 run_end = 0;
        for (size_t index = 0; index < count; ++index) {
            const u64 page_begin = (taken[index] >> 32) << PageBits;
            u32 mask = static_cast<u32>(taken[index]);
            while (mask != 0) {
                const u32 low = static_cast<u32>(std::countr_zero(mask));
                const u32 length = static_cast<u32>(std::countr_one(mask >> low));
                const u64 begin = page_begin + (u64{low} << GranuleBits);
                const u64 end = begin + (u64{length} << GranuleBits);
                mask = low + length >= 32 ? 0 : mask & ~(((u32{1} << length) - 1) << low);
                if (run_end == begin && run_end != 0) {
                    run_end = end;
                    continue;
                }
                if (run_end != run_begin) {
                    func(run_begin, run_end - run_begin);
                }
                run_begin = begin;
                run_end = end;
            }
        }
        if (run_end != run_begin) {
            func(run_begin, run_end - run_begin);
        }
    }

    static constexpr u64 PageBits = 11;
    static constexpr u64 PageSize = u64{1} << PageBits;

private:
    static constexpr u64 GranuleBits = 6;
    static constexpr u64 GranuleSize = u64{1} << GranuleBits;
    static constexpr size_t NumSlots = 32;

    /// A slot holds (page << 32) | granule mask; 0 is free (a queued mask is never 0).
    bool PushPage(u32 page, u32 mask) {
        const u64 key = u64{page} << 32;
        for (auto& slot : slots) {
            u64 value = slot.load(std::memory_order_seq_cst);
            while (value != 0 && (value & ~u64{0xffffffff}) == key) {
                if ((value | mask) == value ||
                    slot.compare_exchange_weak(value, value | mask, std::memory_order_seq_cst,
                                               std::memory_order_seq_cst)) {
                    return true;
                }
            }
        }
        for (auto& slot : slots) {
            u64 expected = 0;
            if (slot.load(std::memory_order_relaxed) == 0 &&
                slot.compare_exchange_strong(expected, key | mask, std::memory_order_seq_cst,
                                             std::memory_order_relaxed)) {
                return true;
            }
        }
        return false;
    }

    std::array<std::atomic<u64>, NumSlots> slots{};
    std::atomic<u32> pending{};
};

} // namespace VideoCommon
