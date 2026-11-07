// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "common/common_types.h"

namespace Common::ThreadPinning {

/// A physical core of processor group 0.
struct Core {
    /// Its logical processors (SMT siblings share a core).
    u64 mask;
    /// Higher is faster. Equal on every core of a CPU that is not hybrid.
    u8 efficiency_class;
};

/// An L3 cache of processor group 0.
struct Cache {
    u64 mask;
    u64 size;
};

struct Topology {
    /// In the host's enumeration order.
    std::vector<Core> cores;
    std::vector<Cache> l3;
    /// In every processor group; only group 0 is pinned.
    u32 logical_processors{};
};

/// EDEN_NX2_PIN_THREADS: 0 is Off, 1 is Force, anything else (or unset) is Auto.
enum class Mode { Off, Auto, Force };
[[nodiscard]] Mode ParseMode(const char* setting);

/// Cores the GPU threads and guest cores are chosen from before Force pins.
constexpr std::size_t MinForcedCores = 4;
/// Auto needs this many cores and leaves the guest cores one logical processor per emulated core
/// (Hardware::NUM_CPU_CORES): 6 cores with SMT (8 guest processors) or 8 without (6).
constexpr std::size_t MinAutoCores = 6;
constexpr std::size_t MinAutoGuestProcessors = 6;

struct Plan {
    /// Affinity masks in processor group 0; all zero when nothing is pinned.
    u64 front{};
    u64 back{};
    u64 guest{};

    std::size_t performance_cores{};
    std::size_t efficiency_cores{};
    /// Performance cores the masks were chosen from.
    std::size_t chosen_cores{};
    /// L3 caches holding performance cores, and the size of the chosen one (0 if they span).
    std::size_t l3_caches{};
    u64 l3_size{};

    [[nodiscard]] bool Enabled() const {
        return front != 0;
    }
};

/// The GPU front and back each get a whole core of the fastest efficiency class, the guest cores
/// the other cores of that class. With several L3 caches, all of them share the largest one when
/// it has enough cores. Core 0 takes most interrupts, so it is never a GPU core.
[[nodiscard]] Plan MakePlan(const Topology& topology, Mode mode);

[[nodiscard]] std::string Describe(const Plan& plan, const Topology& topology, Mode mode);

} // namespace Common::ThreadPinning
