// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <type_traits>

#include "common/common_types.h"

namespace VideoCommon {

/// What a buffer cache runtime offers for stream uniform reuse.
///  - StreamUniformRef: a small trivially copyable description of one stream allocation (the
///    host buffer, its offset and whatever else the runtime needs to add it to a descriptor
///    payload again, e.g. the device address of the descriptor buffer path).
///  - LastStreamUniformRef(): the allocation made by the latest BindMappedUniformBuffer call.
///  - RebindStreamUniform(ref, size): adds `ref` to the descriptor payload exactly as
///    BindMappedUniformBuffer added the allocation the first time.
template <class R>
concept StreamUniformReusable = requires(R& runtime, u32 size) {
    typename R::StreamUniformRef;
    requires std::is_trivially_copyable_v<typename R::StreamUniformRef>;
    { runtime.LastStreamUniformRef() } -> std::same_as<typename R::StreamUniformRef>;
    runtime.RebindStreamUniform(runtime.LastStreamUniformRef(), size);
    { runtime.CurrentTick() } -> std::convertible_to<u64>;
};

/// EDEN_NX2_UBOREUSE=0 turns the reuse off (every small uniform buffer is copied, as before).
/// Read once at static init: asked for every streamed uniform buffer.
inline const bool g_uniform_stream_reuse = [] {
    const char* const value = std::getenv("EDEN_NX2_UBOREUSE");
    return value == nullptr || value[0] != '0';
}();

[[nodiscard]] inline bool UniformStreamReuseEnabled() noexcept {
    return g_uniform_stream_reuse;
}

/// EDEN_NX2_OPT_UBO_INPLACE=0: copy the guest bytes to a scratch buffer and compare that copy
/// (as before), instead of comparing the guest memory where it is when it is contiguous.
inline const bool g_uniform_stream_in_place = [] {
    const char* const value = std::getenv("EDEN_NX2_OPT_UBO_INPLACE");
    return value == nullptr || value[0] != '0';
}();

[[nodiscard]] inline bool UniformStreamInPlaceEnabled() noexcept {
    return g_uniform_stream_in_place;
}

/// EDEN_NX2_UBO_RANGE_REUSE=0: a stream copy is bound again only by the (stage, constant buffer)
/// slot that made it. With it on, a bind of the same guest range in the same command buffer
/// from another stage or slot, or from the slot again after it bound other ranges in between,
/// binds that copy too when the guest bytes still equal it.
inline const bool g_uniform_range_reuse = [] {
    const char* const value = std::getenv("EDEN_NX2_UBO_RANGE_REUSE");
    return value == nullptr || value[0] != '0';
}();

/// What BindSmallUniformReusing did.
enum class UniformReuseResult : u8 {
    None,     ///< nothing bound (the caller binds the buffer itself)
    SlotHit,  ///< the slot's copy bound again: same bytes, same command buffer
    RangeHit, ///< another bind's copy of the same range bound again (EDEN_NX2_UBO_RANGE_REUSE)
    Recopied, ///< same bytes as an older command buffer's copy: a new allocation filled from it
    Copied,   ///< guest bytes copied into a new allocation
};

template <class Ref>
class UniformStreamReuse {
public:
    static constexpr u32 MaxBytes = 4096; // BufferCache's DEFAULT_SKIP_CACHE_SIZE
    static constexpr size_t NumStages = 6; // the five graphics stages and compute
    static constexpr size_t NumSlots = 18; // Maxwell constant buffers per stage
    /// One private copy per slot (all EDEN_NX2_UBO_RANGE_REUSE=0 uses), then the range-keyed
    /// copies: NumSets sets of NumWays, by guest range.
    static constexpr size_t NumPrivate = NumStages * NumSlots;
    static constexpr size_t NumSets = 256;
    static constexpr size_t NumWays = 2;
    static constexpr size_t NumCopies = NumPrivate + NumSets * NumWays;

    /// Bytes copied to a stream allocation, and that allocation. `ref` was filled from exactly
    /// these bytes on scheduler tick `tick`; the bytes change only together with a new ref.
    struct Copy {
        Ref ref{};
        u64 tick{};
        u64 address{}; ///< the guest range copied (a lookup hint: hits compare the bytes)
        u32 size{};    ///< 0: nothing copied
        u32 serial{};  ///< changes whenever the bytes change
    };

    /// The copy a (stage, constant buffer) slot bound last.
    struct Slot {
        u32 copy{};
        u32 serial{}; ///< the copy's serial then; 0: none
    };

    UniformStreamReuse() : bytes{new u8[NumCopies * MaxBytes]} {}

    void Reset() noexcept {
        for (Copy& copy : copies) {
            copy.size = 0;
            copy.tick = 0;
            copy.address = 0;
            NextSerial(copy);
        }
        for (auto& stage : slots) {
            stage.fill(Slot{});
        }
        mru.fill(0);
    }

    [[nodiscard]] u8* Bytes(size_t copy) noexcept {
        return bytes.get() + copy * MaxBytes;
    }

    static void NextSerial(Copy& copy) noexcept {
        if (++copy.serial == 0) {
            copy.serial = 1;
        }
    }

    [[nodiscard]] static size_t Set(u64 address, u32 size) noexcept {
        return static_cast<size_t>(((address ^ (u64{size} << 40)) * 0x9E3779B97F4A7C15ULL) >> 56) %
               NumSets;
    }

    std::array<Copy, NumCopies> copies{};
    std::array<std::array<Slot, NumSlots>, NumStages> slots{};
    std::array<u8, NumSets> mru{}; ///< per set: the way used last
    std::unique_ptr<u8[]> bytes;   ///< MaxBytes per copy
    std::array<u8, MaxBytes> scratch{};
};

template <class Ref>
[[nodiscard]] inline UniformStreamReuse<Ref>& UniformStreamReuseState() noexcept {
    static UniformStreamReuse<Ref> state;
    return state;
}

/// Forgets every slot. Call it when a buffer cache (and so its runtime, scheduler and stream
/// buffer) is created: ticks restart with a new scheduler and the old references are dead.
template <class Runtime>
inline void ResetUniformStreamReuse() noexcept {
    if constexpr (StreamUniformReusable<Runtime>) {
        UniformStreamReuseState<typename Runtime::StreamUniformRef>().Reset();
    }
}

/// Binds the small uniform buffer at `device_addr` for (stage, cbuf `index`) through the stream
/// buffer, binding an earlier allocation again when it holds the same bytes and was made in the
/// current command buffer: the slot's previous copy (any address) or, with
/// EDEN_NX2_UBO_RANGE_REUSE, a copy of the same range made by any bind. Returns false (nothing
/// bound) when the buffer does not qualify, so the caller binds it the classic way.
///
/// Exact: every reuse compares the guest bytes with the copy's bytes, and the copy's allocation
/// was filled from those bytes on the current tick, so the staging pool has not recycled it. A
/// guest write racing with the compare cannot make the bound data differ from the copy.
template <class Runtime, class DeviceMemory>
[[nodiscard]] bool BindSmallUniformReusing(Runtime& runtime, DeviceMemory& device_memory,
                                           size_t stage, u32 index, u32 binding_index,
                                           u64 device_addr, u32 size,
                                           UniformReuseResult* result = nullptr) {
    using Reuse = UniformStreamReuse<typename Runtime::StreamUniformRef>;
    if (size == 0 || size > Reuse::MaxBytes || stage >= Reuse::NumStages ||
        index >= Reuse::NumSlots) {
        return false;
    }
    Reuse& reuse = UniformStreamReuseState<typename Runtime::StreamUniformRef>();
    typename Reuse::Slot& slot = reuse.slots[stage][index];
    // The bytes are compared where they are when contiguous (one copy fewer than through the
    // scratch buffer), and copied once into the copy's bytes when they differ: the stream
    // allocation is filled from those bytes, never from guest memory.
    const u8* guest = nullptr;
    if constexpr (requires(DeviceMemory& memory) {
                      { memory.GetSpan(u64{}, std::size_t{}) } -> std::convertible_to<const u8*>;
                  }) {
        if (UniformStreamInPlaceEnabled()) {
            guest = device_memory.GetSpan(device_addr, size);
        }
    }
    if (guest == nullptr) {
        device_memory.ReadBlockUnsafe(device_addr, reuse.scratch.data(), size);
        guest = reuse.scratch.data();
    }
    const u64 tick = runtime.CurrentTick();
    const auto same_bytes = [&](size_t copy_index) {
        return reuse.copies[copy_index].size == size &&
               std::memcmp(reuse.Bytes(copy_index), guest, size) == 0;
    };
    const auto rebind = [&](size_t copy_index, UniformReuseResult kind) {
        const typename Reuse::Copy& copy = reuse.copies[copy_index];
        runtime.RebindStreamUniform(copy.ref, size);
        slot = typename Reuse::Slot{static_cast<u32>(copy_index), copy.serial};
        if (result != nullptr) {
            *result = kind;
        }
        return true;
    };
    // The slot's previous copy, whatever its address.
    constexpr size_t NO_COPY = ~size_t{0};
    size_t slot_copy = NO_COPY;
    bool slot_same = false;
    if (slot.serial != 0 && reuse.copies[slot.copy].serial == slot.serial) {
        slot_copy = slot.copy;
        slot_same = same_bytes(slot_copy);
        if (slot_same && reuse.copies[slot_copy].tick == tick) {
            return rebind(slot_copy, UniformReuseResult::SlotHit);
        }
    }
    size_t target = stage * Reuse::NumSlots + index;
    if (g_uniform_range_reuse) {
        const size_t set = Reuse::Set(device_addr, size);
        const size_t base = Reuse::NumPrivate + set * Reuse::NumWays;
        for (size_t way = 0; way < Reuse::NumWays; ++way) {
            const size_t copy_index = base + way;
            const typename Reuse::Copy& copy = reuse.copies[copy_index];
            if (copy_index != slot_copy && copy.address == device_addr && copy.size == size &&
                copy.tick == tick && same_bytes(copy_index)) {
                reuse.mru[set] = static_cast<u8>(way);
                return rebind(copy_index, UniformReuseResult::RangeHit);
            }
        }
        // Replace the way holding an older copy of this range, else one from an older command
        // buffer, else the one used least recently.
        size_t victim = NO_COPY;
        for (size_t way = 0; way < Reuse::NumWays && victim == NO_COPY; ++way) {
            const typename Reuse::Copy& copy = reuse.copies[base + way];
            if (copy.address == device_addr && copy.size == size) {
                victim = way;
            }
        }
        for (size_t way = 0; way < Reuse::NumWays && victim == NO_COPY; ++way) {
            if (reuse.copies[base + way].tick != tick) {
                victim = way;
            }
        }
        if (victim == NO_COPY) {
            victim = (reuse.mru[set] + 1) % Reuse::NumWays;
        }
        reuse.mru[set] = static_cast<u8>(victim);
        target = base + victim;
    }
    if (slot_same) {
        // The slot's copy from an older command buffer holds these bytes: fill the new
        // allocation from it.
        target = slot_copy;
    } else {
        typename Reuse::Copy& fresh = reuse.copies[target];
        std::memcpy(reuse.Bytes(target), guest, size);
        fresh.size = size;
        fresh.address = device_addr;
        Reuse::NextSerial(fresh);
    }
    typename Reuse::Copy& copy = reuse.copies[target];
    const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
    std::memcpy(span.data(), reuse.Bytes(target), size);
    copy.ref = runtime.LastStreamUniformRef();
    // After the allocation: the tick the stream region is tagged with is the one current now.
    copy.tick = runtime.CurrentTick();
    slot = typename Reuse::Slot{static_cast<u32>(target), copy.serial};
    if (result != nullptr) {
        *result = slot_same ? UniformReuseResult::Recopied : UniformReuseResult::Copied;
    }
    return true;
}

} // namespace VideoCommon
