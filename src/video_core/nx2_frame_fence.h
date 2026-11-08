// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <mutex>

#include "common/common_types.h"

namespace Tegra {

/// The flip-fence source (fable_answer_17 2). The flip names the frame's end: guest syncpoint
/// `id` reaching `value` (the last submit of the frame on the presenting queue). It is known late
/// (2-3 ms into the next frame), but the guest driver arms a wait on the same (id, value) through
/// nx2:syncpt TIPC 201 earlier, on one slot it reuses. The first flip whose (id, value) was armed
/// names that slot; from then on each arm on it is the next boundary, and the back bumps the
/// clock when it signals the syncpoint up to that value. A flip that comes before the back got
/// there is a boundary too (exact, it is the frame's own fence). Every flip confirms:
/// - equal: an armed value bumped (or about to) on exactly the flip's value;
/// - ahead: the flip came first and the back bumps on its value (no arm needed);
/// - late: the back passed the flip's value with no bump on it, bumped at the flip (as today, a
///   late boundary only refuses relocations). Not bumped again when a later armed value already
///   bumped (that would split the frame the back is in), counted as merged;
/// - early: an armed value bumped that no flip carried, found when a later flip passes it. That
///   is the dangerous direction (a frame split in two), so the first one turns the armed source
///   off for the session and only flips bump from then on. A bump above the flip's value is not
///   early by itself: the next frame's arm can be reached before this flip arrives, and the next
///   flip decides.
/// Guards on an arm: it must be the largest value armed on the syncpoint so far (by any slot),
/// above the last flip's value, and not reached yet. Locked: arms and flips run on guest threads,
/// signals on the GPU thread (only those of the frame syncpoint take the lock).
class Nx2FrameFenceClock {
public:
    struct Window {
        u32 syncpoint{~0U};
        u64 slot_key{~0ULL}; ///< (nx2:syncpt session << 32) | slot, ~0 while none
        bool fault{};
        u32 flips{}, equal{}, ahead{}, late{}, merged{}, early{}, repeats{};
        u32 bumps_armed{}, bumps_flip{}, bumps_late{};
        u32 refused_old{}, refused_not_largest{}, refused_reached{}, dropped{}, held{};
        u32 relearns{};
    };
    struct FlipEvents {
        bool syncpoint_learned{};
        bool slot_learned{};
        bool slot_relearned{};
        bool fault{};
        bool bump{};
        u64 slot_key{};
        u32 early_value{};
    };

    /// The frame-fence syncpoint (from the flips), ~0 while none: the back's hook reads it
    /// without the lock.
    std::atomic<u32> syncpoint{~0U};

    /// nx2:syncpt TIPC 201: `key` is (session << 32) | slot; `guest_value` the syncpoint's value
    /// now.
    void NoteArm(u64 key, u32 id, u32 value, u32 guest_value) {
        std::scoped_lock lk{mutex};
        arms[arm_next] = {key, id, value};
        arm_next = (arm_next + 1) % arms.size();
        arm_count = (std::min)(arm_count + 1, arms.size());
        if (!have_syncpoint || id != frame_syncpoint) {
            return;
        }
        See(guest_value);
        const bool largest = !have_high || !After(high, value);
        if (!have_high || After(value, high)) {
            high = value;
            have_high = true;
        }
        if (!have_slot || key != slot_key || fault) {
            return;
        }
        if (have_flip && !After(value, last_flip)) {
            ++window.refused_old;
        } else if (!largest) {
            ++window.refused_not_largest;
        } else if (!After(value, last_seen)) {
            ++window.refused_reached;
        } else if (FindCandidate(value) == candidate_count && candidate_count < candidates.size()) {
            candidates[candidate_count++] = {value, true, false};
        }
    }

    /// The back signals the frame syncpoint up to `new_value` (before the increment, so a flip
    /// woken by it already sees the bump). True when the clock bumps.
    [[nodiscard]] bool NoteSignal(u32 new_value) {
        bool bump = false;
        {
            std::scoped_lock lk{mutex};
            last_seen = new_value;
            for (size_t i = 0; i < candidate_count;) {
                const Candidate candidate = candidates[i];
                if (After(candidate.value, new_value)) {
                    ++i;
                    continue;
                }
                RemoveCandidate(i);
                if (candidate.carried) {
                    bump = true;
                    ++(candidate.armed ? window.bumps_armed : window.bumps_flip);
                } else if (!fault && unconfirmed_count < unconfirmed.size() &&
                           (!have_flip || After(candidate.value, last_flip))) {
                    unconfirmed[unconfirmed_count++] = candidate.value;
                    bump = true;
                    ++window.bumps_armed;
                } else {
                    ++window.held;
                }
            }
        }
        return bump;
    }

    /// The flip (the present), with the guest syncpoint's value now. events.bump: the clock
    /// bumps now (late).
    FlipEvents NoteFlip(u32 id, u32 value, u32 guest_value) {
        FlipEvents events{};
        bool& bump = events.bump;
        {
            std::scoped_lock lk{mutex};
            if (!have_syncpoint || id != frame_syncpoint) {
                // First flip, or the game presents from another queue: start over on it.
                Restart();
                frame_syncpoint = id;
                have_syncpoint = true;
                last_seen = guest_value;
                syncpoint.store(id, std::memory_order_relaxed);
                events.syncpoint_learned = true;
            }
            See(guest_value);
            // Which slot armed this value, newest first.
            bool slot_armed = false;
            bool other_found = false;
            u64 other_key{};
            for (size_t n = 0; n < arm_count; ++n) {
                const Arm& arm = arms[(arm_next + arms.size() - 1 - n) % arms.size()];
                if (arm.syncpoint != id || arm.value != value) {
                    continue;
                }
                if (have_slot && arm.key == slot_key) {
                    slot_armed = true;
                    break;
                }
                if (!other_found) {
                    other_found = true;
                    other_key = arm.key;
                }
            }
            if (!slot_armed && other_found) {
                if (have_slot) {
                    events.slot_relearned = true;
                    ++window.relearns;
                    DropArmedCandidates(nullptr);
                } else {
                    events.slot_learned = true;
                }
                slot_key = other_key;
                have_slot = true;
                events.slot_key = other_key;
            }
            if (have_flip && !After(value, last_flip)) {
                ++window.repeats;
                return events;
            }
            ++window.flips;
            bool later_bumped = false;
            for (size_t i = 0; i < unconfirmed_count; ++i) {
                later_bumped |= After(unconfirmed[i], value);
            }
            if (const size_t at = FindUnconfirmed(value); at != unconfirmed_count) {
                RemoveUnconfirmed(at);
                ++window.equal;
            } else if (const size_t c = FindCandidate(value); c != candidate_count) {
                candidates[c].carried = true;
                ++(candidates[c].armed ? window.equal : window.ahead);
            } else if (!After(value, last_seen)) {
                ++window.late;
                if (later_bumped) {
                    ++window.merged;
                } else {
                    bump = true;
                    ++window.bumps_late;
                }
            } else if (candidate_count < candidates.size()) {
                candidates[candidate_count++] = {value, false, true};
                ++window.ahead;
            } else {
                ++window.held; // no room: no boundary for it (frames merge, the safe direction)
            }
            // Armed bumps below this flip that no flip carried were not frame ends.
            for (size_t i = 0; i < unconfirmed_count;) {
                if (After(value, unconfirmed[i])) {
                    ++window.early;
                    events.early_value = unconfirmed[i];
                    RemoveUnconfirmed(i);
                    if (!fault) {
                        fault = true;
                        events.fault = true;
                    }
                } else {
                    ++i;
                }
            }
            // Armed values below this flip are not frame ends either: drop them before they bump.
            DropArmedCandidates(&value);
            if (fault) {
                DropArmedCandidates(nullptr);
            }
            last_flip = value;
            have_flip = true;
        }
        return events;
    }

    /// The clock was off (the hook and the notes stop with it): values pending from before would
    /// be judged against flips nobody noted, so drop them. The syncpoint and the slot stay.
    void Resume() {
        std::scoped_lock lk{mutex};
        have_flip = have_high = false;
        candidate_count = unconfirmed_count = 0;
    }

    /// The counters since the last call (TextureCache::TickRenderClock, every 600 frames).
    Window TakeWindow() {
        std::scoped_lock lk{mutex};
        Window taken = window;
        taken.syncpoint = have_syncpoint ? frame_syncpoint : ~0U;
        taken.slot_key = have_slot ? slot_key : ~0ULL;
        taken.fault = fault;
        window = {};
        return taken;
    }

private:
    struct Candidate {
        u32 value;
        bool armed;   ///< From the learned slot's arm, else from a flip
        bool carried; ///< A flip carried this value
    };
    struct Arm {
        u64 key;
        u32 syncpoint;
        u32 value;
    };

    /// Wrapping compare: a is after b.
    static bool After(u32 a, u32 b) {
        return static_cast<s32>(a - b) > 0;
    }

    /// The hook keeps last_seen exact while it runs; this catches it up after a stretch without
    /// it (the clock not wanted, EXP43 switched live). The guest value may lag the hook by one
    /// increment (the hook runs before it), so it only ever moves last_seen forward.
    void See(u32 guest_value) {
        if (After(guest_value, last_seen)) {
            last_seen = guest_value;
        }
    }

    /// Forgets the syncpoint, the slot and the pending values (the counters, the arms seen and
    /// a fault stay).
    void Restart() {
        have_syncpoint = have_slot = have_flip = have_high = false;
        frame_syncpoint = last_flip = last_seen = high = 0;
        slot_key = 0;
        candidate_count = unconfirmed_count = 0;
    }

    size_t FindCandidate(u32 value) const {
        size_t i = 0;
        while (i < candidate_count && candidates[i].value != value) {
            ++i;
        }
        return i;
    }
    void RemoveCandidate(size_t i) {
        candidates[i] = candidates[--candidate_count];
    }
    size_t FindUnconfirmed(u32 value) const {
        size_t i = 0;
        while (i < unconfirmed_count && unconfirmed[i] != value) {
            ++i;
        }
        return i;
    }
    void RemoveUnconfirmed(size_t i) {
        unconfirmed[i] = unconfirmed[--unconfirmed_count];
    }
    /// Drops armed values no flip carried, all of them or those before *below.
    void DropArmedCandidates(const u32* below) {
        for (size_t i = 0; i < candidate_count;) {
            const Candidate& candidate = candidates[i];
            if (candidate.armed && !candidate.carried &&
                (below == nullptr || After(*below, candidate.value))) {
                ++window.dropped;
                RemoveCandidate(i);
            } else {
                ++i;
            }
        }
    }

    std::mutex mutex;
    u32 frame_syncpoint{};
    bool have_syncpoint{};
    u64 slot_key{};
    bool have_slot{};
    u32 last_flip{};
    bool have_flip{};
    u32 last_seen{}; ///< The back's value of the frame syncpoint, as the hook last saw it
    u32 high{};      ///< The largest value armed on the frame syncpoint, by any slot
    bool have_high{};
    bool fault{};
    std::array<Candidate, 32> candidates{};
    size_t candidate_count{};
    std::array<u32, 16> unconfirmed{}; ///< Armed values bumped that no flip carried yet
    size_t unconfirmed_count{};
    std::array<Arm, 64> arms{}; ///< The latest arms on any slot, to find the flip's slot
    size_t arm_next{};
    size_t arm_count{};
    Window window{};
};
inline Nx2FrameFenceClock g_nx2_frame_fence;

} // namespace Tegra
