// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <functional>
#include <mutex>
#include <utility>
#include <vector>

#include "core/device_memory_manager.h"

namespace Core {

class GPUDirtyMemoryManager {
public:
    GPUDirtyMemoryManager() : current{default_transform} {
        back_buffer.reserve(256);
        front_buffer.reserve(256);
    }

    ~GPUDirtyMemoryManager() = default;

    void Collect(PAddr address, size_t size) {
        TransformAddress t = BuildTransform(address, size);
        TransformAddress tmp, original;
        do {
            tmp = current.load(std::memory_order_acquire);
            original = tmp;
            if (tmp.address != t.address) {
                if (IsValid(tmp.address)) {
                    std::scoped_lock lk(guard);
                    back_buffer.emplace_back(tmp);
                    has_back.store(true, std::memory_order_release);
                    current.exchange(t, std::memory_order_seq_cst);
                    NoteAnyWrites();
                    return;
                }
                tmp.address = t.address;
                tmp.mask = 0;
            }
            if ((tmp.mask | t.mask) == tmp.mask) {
                return;
            }
            tmp.mask |= t.mask;
        } while (!current.compare_exchange_weak(original, tmp, std::memory_order_seq_cst,
                                                std::memory_order_relaxed));
        // The first write since the last Gather: tell InvalidateGPUCache (AnyWrites). Later
        // writes to the same transform find the flag set by this one, or a Gather after it.
        if (!IsValid(original.address)) {
            NoteAnyWrites();
        }
    }

    /// EDEN_NX2_DIRTY_ANY: whether any manager may hold collected writes, in one load instead of
    /// three per manager. Set after the store that starts a manager's batch of writes (seq_cst,
    /// as the Gather's exchange of `current`): a reader that clears it (BeginGather) and then
    /// gathers every manager either finds a batch begun before its clear or leaves the flag set.
    [[nodiscard]] static bool AnyWrites() {
        return any_writes.load(std::memory_order_seq_cst) ||
               any_gatherers.load(std::memory_order_seq_cst) != 0;
    }

    /// Starts a gather of every manager after AnyWrites: clears the flag. Pair with EndGather
    /// once the gather has applied its writes; a reader finding the flag clear while another one
    /// gathers takes the gather path too (Gather serializes them), so it never returns before
    /// writes collected before it are applied.
    static void BeginGather() {
        any_gatherers.fetch_add(1, std::memory_order_seq_cst);
        any_writes.exchange(false, std::memory_order_seq_cst);
    }

    static void EndGather() {
        any_gatherers.fetch_sub(1, std::memory_order_seq_cst);
    }

    /// Calls `callback` for each write collected and not yet applied, without consuming them: those
    /// still collecting and those a Gather is handing out.
    void ForEachPending(const std::function<void(PAddr, size_t)>& callback) {
        std::vector<TransformAddress> pending;
        {
            std::scoped_lock gather_lk(gather_guard);
            std::scoped_lock lk(guard);
            pending = back_buffer;
            pending.insert(pending.end(), front_buffer.begin(), front_buffer.end());
            const TransformAddress t = current.load(std::memory_order_acquire);
            if (IsValid(t.address)) {
                pending.push_back(t);
            }
        }
        for (const auto& transform : pending) {
            ForEachRun(transform, callback);
        }
    }

    void Gather(std::function<void(PAddr, size_t)>& callback) {
        // FlushRegion also gathers off the GPU thread (host1x video decode reads); serialize
        // gatherers so one cannot swap or clear front_buffer while another iterates it.
        std::scoped_lock gather_lk(gather_guard);
        {
            std::scoped_lock lk(guard);
            TransformAddress t = current.exchange(default_transform, std::memory_order_seq_cst);
            front_buffer.swap(back_buffer);
            has_back.store(false, std::memory_order_relaxed);
            if (IsValid(t.address)) {
                front_buffer.emplace_back(t);
            }
            has_front.store(!front_buffer.empty(), std::memory_order_release);
        }
        for (auto& transform : front_buffer) {
            ForEachRun(transform, callback);
        }
        {
            std::scoped_lock lk(guard);
            front_buffer.clear();
            has_front.store(false, std::memory_order_release);
        }
    }

    /// Whether Gather can find collected writes, without its locks (a write collected meanwhile
    /// is gathered by the next one, as if it came right after this Gather).
    bool MayHaveWrites() {
        return IsValid(current.load(std::memory_order_acquire).address) ||
               has_back.load(std::memory_order_acquire) ||
               has_front.load(std::memory_order_acquire);
    }

    /// Returns true when collected writes not yet gathered overlap [address, address + size).
    bool Overlaps(PAddr address, size_t size) {
        const PAddr end = address + size;
        const auto overlaps = [&](TransformAddress t) {
            if (!IsValid(t.address)) {
                return false;
            }
            const PAddr begin = static_cast<PAddr>(t.address) << page_bits;
            if (address >= begin + page_size || end <= begin) {
                return false;
            }
            const size_t minor_bit = address > begin ? (address - begin) >> align_bits : 0;
            const size_t top_bit = ((std::min)(end - begin, PAddr{page_size}) + align_mask) >> align_bits;
            return (t.mask & CreateMask<u32>(top_bit, minor_bit)) != 0;
        };
        std::scoped_lock lk(guard);
        if (overlaps(current.load(std::memory_order_acquire))) {
            return true;
        }
        for (const TransformAddress& t : back_buffer) {
            if (overlaps(t)) {
                return true;
            }
        }
        // Writes a Gather is still applying count until their callbacks have run.
        for (const TransformAddress& t : front_buffer) {
            if (overlaps(t)) {
                return true;
            }
        }
        return false;
    }

private:
    struct alignas(8) TransformAddress {
        u32 address;
        u32 mask;
    };

    template <typename Func>
    static void ForEachRun(const TransformAddress& transform, Func&& callback) {
        size_t offset = 0;
        u64 mask = transform.mask;
        while (mask != 0) {
            const size_t empty_bits = std::countr_zero(mask);
            offset += empty_bits << align_bits;
            mask = mask >> empty_bits;
            const size_t continuous_bits = std::countr_one(mask);
            callback((static_cast<PAddr>(transform.address) << page_bits) + offset,
                     continuous_bits << align_bits);
            mask = continuous_bits < align_size ? (mask >> continuous_bits) : 0;
            offset += continuous_bits << align_bits;
        }
    }

    constexpr static size_t page_bits = DEVICE_PAGEBITS - 1;
    constexpr static size_t page_size = 1ULL << page_bits;
    constexpr static size_t page_mask = page_size - 1;

    constexpr static size_t align_bits = 6U;
    constexpr static size_t align_size = 1U << align_bits;
    constexpr static size_t align_mask = align_size - 1;
    constexpr static TransformAddress default_transform = {.address = ~0U, .mask = 0U};

    bool IsValid(PAddr address) {
        return address < (1ULL << 39);
    }

    static void NoteAnyWrites() {
        // Read first: the flag is set nearly always while cores write, and a load keeps the
        // line shared between them.
        if (!any_writes.load(std::memory_order_seq_cst)) {
            any_writes.store(true, std::memory_order_seq_cst);
        }
    }

    // Shared by every manager (one per core and the handoff's); starts set so the first
    // InvalidateGPUCache gathers.
    static inline std::atomic<bool> any_writes{true};
    static inline std::atomic<u32> any_gatherers{};

    template <typename T>
    T CreateMask(size_t top_bit, size_t minor_bit) {
        T mask = ~T(0);
        mask <<= (sizeof(T) * 8 - top_bit);
        mask >>= (sizeof(T) * 8 - top_bit);
        mask >>= minor_bit;
        mask <<= minor_bit;
        return mask;
    }

    TransformAddress BuildTransform(PAddr address, size_t size) {
        const size_t minor_address = address & page_mask;
        const size_t minor_bit = minor_address >> align_bits;
        const size_t top_bit = (minor_address + size + align_mask) >> align_bits;
        TransformAddress result{};
        result.address = static_cast<u32>(address >> page_bits);
        result.mask = CreateMask<u32>(top_bit, minor_bit);
        return result;
    }

    std::atomic<TransformAddress> current{};
    std::atomic<bool> has_back{}; ///< back_buffer holds writes
    std::atomic<bool> has_front{}; ///< front_buffer holds writes Gather is still applying
    std::mutex guard;
    std::mutex gather_guard;
    std::vector<TransformAddress> back_buffer;
    std::vector<TransformAddress> front_buffer;
};

} // namespace Core
