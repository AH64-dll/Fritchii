// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <mutex>

#include "common/logging.h"
#include "core/arm/arm_interface.h"
#include "core/core.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/svc.h"
#include "core/hle/kernel/svc/svc_nx2_oot_poll.h"
#include "core/hle/kernel/svc_types.h"
#include "core/memory.h"

namespace Kernel::Svc {
namespace {

// Ocarina of Time (main build E0F3CCFA...): the resource request thread (main+0xF7A0, state 3)
// walks every entry of its pending-request ring. Per entry it locks the manager mutex, looks the
// name up twice through the query engine (main+0x58EEA0, via 0x167B0 and 0xF630), checks the
// states of the entry's sub-resources and unlocks. If any entry is not ready ("all done" flag
// [sp+0x110] cleared), the walk starts again at once (main+0x14ABC, patched to reach this SVC).
//
// The walks are long: hundreds of entries, about 1-2 ms each in an OoT profile (the query engine
// is 455 ms/s while this SVC ran fewer than about 700 times a second). A fixed 500 us sleep per
// walk therefore cut the poll's duty cycle by only a third. The sleep is now proportional to the
// walk that just ended, so the thread spends at most about 1/(1+kWalkFactor) of its time
// re-checking a list where nothing changed.
//
// What may never be delayed (the SVC returns at once, exactly like the original branch):
//  - a walk that issued new loads ([sp+0x108], incremented on the load-issue paths 0x14B8C and
//    0x14DE4 before 0x149F8): the request chain is making progress;
//  - a request list that changed during the walk (count [x19+0x1E0] differs from the walk's
//    snapshot [sp+0x128], or the ring head/count moved since the walk started);
//  - a state other than 3 ([x19+0x2C]: another thread asked for a flush, cancel or stop);
//  - the loop top's exit request ([[main+0x4C7E178]]+0x1F8 != 0, 0xF8C0..0xF8D4).
// The manager mutex (x19+0x50) is not held at the SVC (it is released at 0x141B8 before both exits
// to 0x14AB8), so a sleep never blocks the threads that add or complete requests.
constexpr u64 kOotProgramId = 0x0400486024DEC000ULL;
constexpr u32 kPollSvc = 0xF3;

// main offsets, used to check that the SVC came from the patched code.
constexpr u64 kSvcReturnOffset = 0x1708C; // instruction after `svc #0xf3` in the cave
constexpr u64 kSvcOffset = 0x17088;
constexpr u32 kSvcInstruction = 0xD4001E61;      // svc #0xf3
constexpr u64 kRescanBranchOffset = 0x14ABC;
constexpr u32 kRescanBranch = 0x36012E68;        // tbz w8, #0, 0x17088
constexpr u64 kExitGotOffset = 0x4C7E178;        // adrp/ldr at the loop top 0xF8C0
constexpr u64 kExitFlagOffset = 0x1F8;

// Request manager (x19).
constexpr u64 kStateOffset = 0x2C;
constexpr u32 kStateWalking = 3;
constexpr u64 kHeadOffset = 0x1DC;
constexpr u64 kCountOffset = 0x1E0;

// The walk's stack frame (sp is fixed after the prologue's `sub sp, sp, #0xa30`).
constexpr u64 kIssuedOffset = 0x108;    // w: loads issued by this walk (reset at 0x130E8)
constexpr u64 kAllDoneOffset = 0x110;   // w: bit 0 = every entry was ready
constexpr u64 kWalkCountOffset = 0x128; // w: entries this walk visits (snapshot at 0x130E0)

constexpr s64 kMinSleepNs = 500'000;
constexpr s64 kDefaultMaxSleepNs = 4'000'000;
constexpr s64 kLimitMaxSleepNs = 16'000'000; // the game's own idle sleep in this loop
constexpr s64 kWalkFactor = 4;
// A gap longer than this since the last return from the SVC is not one walk (the thread idled in
// the 16 ms sleep or was preempted for long); such a walk gets the minimum sleep.
constexpr s64 kStaleWalkNs = 20'000'000;
constexpr s64 kLogIntervalNs = 10'000'000'000;

// EDEN_NX2_OOT_POLL_BACKOFF: "0" leaves main unpatched (patch_manager.cpp), and then this SVC never
// runs. A number above 1 is the longest sleep in us (at most 16000); "500" is the old fixed sleep.
// Anything else: 4000 us.
s64 MaxSleepNs() {
    static const s64 ns = [] {
        const char* value = std::getenv("EDEN_NX2_OOT_POLL_BACKOFF");
        s64 us = kDefaultMaxSleepNs / 1000;
        if (value != nullptr && value[0] != 0) {
            const long long parsed = std::strtoll(value, nullptr, 10);
            if (parsed > 1) {
                us = parsed < kLimitMaxSleepNs / 1000 ? parsed : kLimitMaxSleepNs / 1000;
            }
        }
        return us * 1000;
    }();
    return ns;
}

s64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

enum class Outcome : u8 {
    Sleep,
    Progress,    // the walk issued loads
    ListChanged, // requests added or removed during the walk
    State,       // state is no longer 3
    Exit,        // the loop top will stop the thread
    Frame,       // not the expected code or frame; behave as the original branch
    Count,
};

struct WalkView {
    Outcome outcome{Outcome::Frame};
    u64 manager{};
    u32 head{};
    u32 count{};
};

bool Valid(Core::Memory::Memory& memory, u64 address, u64 size) {
    return address != 0 && memory.IsValidVirtualAddressRange(address, size);
}

bool ExitPending(Core::Memory::Memory& memory, u64 main_base) {
    const u64 got = main_base + kExitGotOffset;
    if (!Valid(memory, got, 8)) {
        return false;
    }
    const u64 global = memory.Read64(got);
    if (!Valid(memory, global, 8)) {
        return false;
    }
    const u64 object = memory.Read64(global);
    if (object == 0 || !Valid(memory, object + kExitFlagOffset, 8)) {
        return false;
    }
    return memory.Read64(object + kExitFlagOffset) != 0;
}

WalkView ReadWalk(Core::Memory::Memory& memory, const ThreadContext& ctx) {
    WalkView view{};
    const u64 main_base = ctx.pc - kSvcReturnOffset;
    if (!Valid(memory, main_base + kRescanBranchOffset, 4) ||
        memory.Read32(main_base + kRescanBranchOffset) != kRescanBranch ||
        !Valid(memory, main_base + kSvcOffset, 4) ||
        memory.Read32(main_base + kSvcOffset) != kSvcInstruction) {
        return view;
    }
    const u64 manager = ctx.r[19];
    const u64 sp = ctx.sp;
    if (!Valid(memory, manager, kCountOffset + 4) || !Valid(memory, sp + kIssuedOffset, 0x24)) {
        return view;
    }
    // Only reached through `tbz w8, #0` with w8 = [sp+0x110]; anything else is not this frame.
    if ((memory.Read32(sp + kAllDoneOffset) & 1) != 0) {
        return view;
    }
    view.manager = manager;
    view.head = memory.Read32(manager + kHeadOffset);
    view.count = memory.Read32(manager + kCountOffset);
    if (memory.Read32(manager + kStateOffset) != kStateWalking) {
        view.outcome = Outcome::State;
    } else if (memory.Read32(sp + kIssuedOffset) != 0) {
        view.outcome = Outcome::Progress;
    } else if (view.count != memory.Read32(sp + kWalkCountOffset)) {
        view.outcome = Outcome::ListChanged;
    } else if (ExitPending(memory, main_base)) {
        view.outcome = Outcome::Exit;
    } else {
        view.outcome = Outcome::Sleep;
    }
    return view;
}

// One per request manager (the game has one poller thread per manager).
struct Slot {
    u64 manager{};
    s64 resume_ns{}; // when the thread last returned from this SVC (start of the next walk)
    u32 head{};
    u32 count{};
};

struct Stats {
    u64 calls{};
    u64 slept{};
    u64 measured{};
    s64 walk_ns{};
    s64 sleep_ns{};
    std::array<u64, static_cast<size_t>(Outcome::Count)> outcomes{};
};

std::mutex g_mutex;
std::array<Slot, 8> g_slots{};
Stats g_stats{};
u64 g_total_calls{};
s64 g_interval_start_ns{};

// Under g_mutex.
Slot& FindSlot(u64 manager) {
    Slot* oldest = &g_slots[0];
    for (Slot& slot : g_slots) {
        if (slot.manager == manager) {
            return slot;
        }
        if (slot.resume_ns < oldest->resume_ns) {
            oldest = &slot;
        }
    }
    *oldest = Slot{.manager = manager};
    return *oldest;
}

void LogStats(const Stats& stats, s64 interval_ns) {
    const double secs = static_cast<double>(interval_ns) / 1e9;
    const auto n = [&](Outcome o) { return stats.outcomes[static_cast<size_t>(o)]; };
    LOG_INFO(Kernel_SVC,
             "OoT resource poll backoff: {} walks in {:.1f} s ({:.0f}/s), {} slept (avg walk {} "
             "us, avg sleep {} us); at once: {} issued loads, {} list changed, {} state, {} exit, "
             "{} frame",
             stats.calls, secs, secs > 0 ? static_cast<double>(stats.calls) / secs : 0.0,
             stats.slept, stats.measured ? stats.walk_ns / static_cast<s64>(stats.measured) / 1000 : 0,
             stats.slept ? stats.sleep_ns / static_cast<s64>(stats.slept) / 1000 : 0,
             n(Outcome::Progress), n(Outcome::ListChanged), n(Outcome::State), n(Outcome::Exit),
             n(Outcome::Frame));
}

} // Anonymous namespace

bool CallNx2OotPollBackoff(Core::System& system, u32 imm) {
    if (imm != kPollSvc) {
        return false;
    }
    auto& kernel = system.Kernel();
    auto& process = GetCurrentProcess(kernel);
    if (!process.Is64Bit() || process.GetProgramId() != kOotProgramId) {
        return false;
    }
    auto& memory = process.GetMemory();

    // Read the walk's registers before sleeping: afterwards the thread may run on another core.
    WalkView view{};
    if (auto* const arm = process.GetArmInterface(kernel.CurrentPhysicalCoreIndex())) {
        ThreadContext ctx{};
        arm->GetContext(ctx);
        view = ReadWalk(memory, ctx);
    }

    const s64 now = NowNs();
    const s64 max_sleep = MaxSleepNs();
    const s64 min_sleep = std::min(kMinSleepNs, max_sleep);
    s64 sleep_ns = 0;
    bool first = false;
    bool do_log = false;
    Stats snapshot{};
    s64 interval_ns = 0;
    {
        std::scoped_lock lock{g_mutex};
        if (view.outcome != Outcome::Frame) {
            Slot& slot = FindSlot(view.manager);
            const s64 walk = slot.resume_ns != 0 ? now - slot.resume_ns : 0;
            // The ring moved under the walk (an entry completed and another came in).
            if (view.outcome == Outcome::Sleep && slot.resume_ns != 0 &&
                (slot.head != view.head || slot.count != view.count)) {
                view.outcome = Outcome::ListChanged;
            }
            if (view.outcome == Outcome::Sleep) {
                sleep_ns = min_sleep;
                if (walk > 0 && walk <= kStaleWalkNs) {
                    sleep_ns = std::clamp(walk * kWalkFactor, min_sleep, max_sleep);
                    ++g_stats.measured;
                    g_stats.walk_ns += walk;
                }
                ++g_stats.slept;
                g_stats.sleep_ns += sleep_ns;
            }
        }
        ++g_stats.outcomes[static_cast<size_t>(view.outcome)];
        ++g_stats.calls;
        first = g_total_calls++ == 0;
        if (first) {
            g_interval_start_ns = now;
        } else if (now - g_interval_start_ns >= kLogIntervalNs) {
            do_log = true;
            snapshot = g_stats;
            interval_ns = now - g_interval_start_ns;
            g_stats = Stats{};
            g_interval_start_ns = now;
        }
    }
    if (first) {
        LOG_INFO(Kernel_SVC,
                 "OoT resource poll backoff: active; a walk that issued nothing and saw no "
                 "change sleeps {}x its length, {}-{} us",
                 kWalkFactor, min_sleep / 1000, max_sleep / 1000);
    }
    if (do_log) {
        LogStats(snapshot, interval_ns);
    }

    if (sleep_ns > 0) {
        SleepThread(system, sleep_ns);
    }

    // The next walk starts now: remember when, and the ring it starts from.
    if (view.outcome != Outcome::Frame) {
        const u32 head = memory.Read32(view.manager + kHeadOffset);
        const u32 count = memory.Read32(view.manager + kCountOffset);
        const s64 resume = NowNs();
        std::scoped_lock lock{g_mutex};
        Slot& slot = FindSlot(view.manager);
        slot.resume_ns = resume;
        slot.head = head;
        slot.count = count;
    }
    return true;
}

} // namespace Kernel::Svc
