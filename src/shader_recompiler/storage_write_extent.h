// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <compare>
#include <optional>

#include "common/common_types.h"

namespace Shader {

/// How far a program's stores and atomics to one storage buffer can reach, in bytes past the
/// pointer it reads from its constant buffer. Every store ends at or below
///
///   const_end + sum(local_stride[i] * local_id[i]) + sum(group_stride[i] * group_id[i])
///
/// for every invocation, where local_id is gl_LocalInvocationID and group_id gl_WorkGroupID. The
/// bound only holds while each value the stores' offsets split into 16-bit halves (multiplications
/// lowered from native IMAD) stays below 2^16: guard_* is an upper bound of the largest of them in
/// the same form, and the extent is unknown when it can reach 2^16 for a dispatch.
///
/// A native (Switch 2) program's storage pointer carries no size, so the buffer cache binds up to
/// 8 MiB behind it; this is what it can claim the dispatch wrote instead of all of it.
struct StorageWriteExtent {
    /// False: the reach is not known (or the buffer is never written); take the whole binding.
    bool bounded{};
    /// Whether the bound needs the guard below.
    bool guarded{};
    u32 const_end{};
    std::array<u32, 3> local_stride{};
    std::array<u32, 3> group_stride{};
    u32 guard_const{};
    std::array<u32, 3> guard_local_stride{};
    std::array<u32, 3> guard_group_stride{};

    auto operator<=>(const StorageWriteExtent&) const = default;
};

/// Bytes a dispatch's stores can reach past the pointer, or nullopt when unknown. grid is nullopt
/// for an indirect dispatch, whose group count is read by the GPU.
[[nodiscard]] constexpr std::optional<u64> ResolveStorageWriteExtent(
    const StorageWriteExtent& extent, const std::array<u32, 3>& block,
    const std::optional<std::array<u32, 3>>& grid) {
    if (!extent.bounded) {
        return std::nullopt;
    }
    const auto evaluate = [&](u64 constant, const std::array<u32, 3>& local,
                              const std::array<u32, 3>& group) -> std::optional<u64> {
        // Anything past 2^62 bytes is as good as unknown, and stops the sum from overflowing.
        constexpr u64 limit = u64{1} << 62;
        u64 total = constant;
        for (size_t i = 0; i < 3; ++i) {
            if (block[i] == 0) {
                return std::nullopt;
            }
            const u64 local_max = u64{block[i]} - 1;
            if (local[i] != 0 && local_max != 0) {
                if (local_max > limit / local[i]) {
                    return std::nullopt;
                }
                total += u64{local[i]} * local_max;
            }
            if (group[i] == 0) {
                continue;
            }
            if (!grid || (*grid)[i] == 0) {
                return std::nullopt;
            }
            const u64 group_max = u64{(*grid)[i]} - 1;
            if (group_max != 0) {
                if (group_max > limit / group[i]) {
                    return std::nullopt;
                }
                total += u64{group[i]} * group_max;
            }
            if (total > limit) {
                return std::nullopt;
            }
        }
        return total;
    };
    if (extent.guarded) {
        const auto guard =
            evaluate(extent.guard_const, extent.guard_local_stride, extent.guard_group_stride);
        if (!guard || *guard >= 0x10000) {
            return std::nullopt;
        }
    }
    return evaluate(extent.const_end, extent.local_stride, extent.group_stride);
}

/// Widens an extent to also cover another (both bounds hold for each component's maximum).
constexpr void MergeStorageWriteExtent(StorageWriteExtent& base, const StorageWriteExtent& other) {
    if (!base.bounded || !other.bounded) {
        base = StorageWriteExtent{};
        return;
    }
    const auto max3 = [](std::array<u32, 3>& lhs, const std::array<u32, 3>& rhs) {
        for (size_t i = 0; i < 3; ++i) {
            lhs[i] = (std::max)(lhs[i], rhs[i]);
        }
    };
    base.const_end = (std::max)(base.const_end, other.const_end);
    max3(base.local_stride, other.local_stride);
    max3(base.group_stride, other.group_stride);
    if (other.guarded) {
        if (base.guarded) {
            base.guard_const = (std::max)(base.guard_const, other.guard_const);
            max3(base.guard_local_stride, other.guard_local_stride);
            max3(base.guard_group_stride, other.guard_group_stride);
        } else {
            base.guarded = true;
            base.guard_const = other.guard_const;
            base.guard_local_stride = other.guard_local_stride;
            base.guard_group_stride = other.guard_group_stride;
        }
    }
}

} // namespace Shader
