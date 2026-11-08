// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <mutex>
#include <span>
#include <string>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/nx2_debug_control.h"

// TEST: EDEN_NX2_LIST_VERIFY=1 (live; the LIST_VERIFY part of zortos293/Lichii#805 3a665f9f27).
// Each command list the split's front reads on GPU-written pages (DmaPusher::
// ReadGpuWrittenCommands) is read again as the front read such lists before EXP40: after every
// dispatch and copy sent so far (GpuSplit::WaitForWriters), from the pages' whole copies
// (t_reference), or drained and flushed. When the two differ it is also drained and flushed,
// which tells the stale read. "list verify" counts lists and differences every 10 s, the first 32
// differences are logged, and the front parses the words read last. Changes timing: a check, not
// a measurement.
namespace Tegra::Nx2ListVerify {

[[nodiscard]] inline bool Enabled() {
    return VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_list_verify,
                                            "EDEN_NX2_LIST_VERIFY") != 0;
}

/// The front, while it reads a list as before: Nx2ReadListCopy serves guest memory only for pages
/// flushed and written by no GPU work since, else the page's whole copy.
inline thread_local bool t_reference{};

class ReferenceScope {
public:
    ReferenceScope() {
        t_reference = true;
    }
    ~ReferenceScope() {
        t_reference = false;
    }
    ReferenceScope(const ReferenceScope&) = delete;
    ReferenceScope& operator=(const ReferenceScope&) = delete;
};

enum class Difference : u32 {
    OldStale,  ///< The drained read matches the new read: the read as before was stale
    NewStale,  ///< The drained read matches the read as before: the new read was stale
    AllDiffer, ///< All three differ
    Drained,   ///< The read as before drained and flushed: the new read was stale
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Difference::Count)> difference_names{
    "the read as before was stale",
    "the new read was stale",
    "all three differ",
    "the new read differs from the drained one",
};

struct State {
    std::mutex mutex;
    std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now();
    u64 verified{};
    u64 verified_drained{};
    u64 differ{};
    u64 differ_words{};
    std::array<u64, static_cast<size_t>(Difference::Count)> differences{};
    u64 verified_total{};
    u64 differ_total{};
    u32 differ_logged{};
};

[[nodiscard]] inline State& Get() {
    static State state;
    return state;
}

/// Logs every 10 s; the caller holds the mutex.
inline void MaybeLog(State& state) {
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - state.last_log).count();
    if (seconds < 10.0) {
        return;
    }
    LOG_INFO(HW_GPU,
             "list verify {:.1f}s: {} lists read again ({} drained and flushed), {} differ in {} "
             "words: {} {}, {} {}, {} {}, {} {}; since the start {} of {} differ",
             seconds, state.verified, state.verified_drained, state.differ, state.differ_words,
             difference_names[0], state.differences[0], difference_names[1], state.differences[1],
             difference_names[2], state.differences[2], difference_names[3], state.differences[3],
             state.differ_total, state.verified_total);
    state.last_log = now;
    state.verified = 0;
    state.verified_drained = 0;
    state.differ = 0;
    state.differ_words = 0;
    state.differences.fill(0);
}

/// The front read the list at `address` as `parsed`, then as before as `reference`
/// (`reference_drained`: by draining and flushing) and, when those differ, drained and flushed
/// as `drained` (else empty).
inline void NoteVerify(u64 address, std::span<const u32> parsed, std::span<const u32> reference,
                       std::span<const u32> drained, bool reference_drained) {
    u64 differing = 0;
    size_t first = parsed.size();
    for (size_t i = 0; i < parsed.size() && i < reference.size(); ++i) {
        if (parsed[i] != reference[i]) {
            ++differing;
            first = (std::min)(first, i);
        }
    }
    State& state = Get();
    std::scoped_lock lock{state.mutex};
    ++state.verified;
    ++state.verified_total;
    state.verified_drained += reference_drained ? 1 : 0;
    if (differing != 0) {
        Difference kind = Difference::Drained;
        if (!reference_drained) {
            if (std::ranges::equal(drained, parsed)) {
                kind = Difference::OldStale;
            } else if (std::ranges::equal(drained, reference)) {
                kind = Difference::NewStale;
            } else {
                kind = Difference::AllDiffer;
            }
        }
        ++state.differ;
        ++state.differ_total;
        state.differ_words += differing;
        ++state.differences[static_cast<size_t>(kind)];
        if (state.differ_logged < 32) {
            ++state.differ_logged;
            LOG_WARNING(HW_GPU,
                        "list verify: list at {:#x} ({} words) differs from its read as before in "
                        "{} words, the first word {}: {:#010x}, as before {:#010x}{}: {}",
                        address, parsed.size(), differing, first, parsed[first], reference[first],
                        drained.size() > first
                            ? fmt::format(", drained {:#010x}", drained[first])
                            : std::string{},
                        difference_names[static_cast<size_t>(kind)]);
        }
    }
    MaybeLog(state);
}

} // namespace Tegra::Nx2ListVerify
