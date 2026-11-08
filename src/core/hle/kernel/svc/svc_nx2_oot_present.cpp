// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <memory>
#include <mutex>
#include <string>

#include <fmt/format.h>

#include "common/logging.h"
#include "common/nx2_gpu_write_listener.h"
#include "core/arm/arm_interface.h"
#include "core/arm/debug.h"
#include "core/core.h"
#include "core/hle/kernel/k_hardware_timer.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_scheduler.h"
#include "core/hle/kernel/k_scoped_scheduler_lock_and_sleep.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/k_thread_queue.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/svc/svc_nx2_oot_present.h"
#include "core/hle/kernel/svc_results.h"
#include "core/memory.h"
#include "video_core/nx2_debug_control.h"

namespace Kernel::Svc {
namespace {

// Ocarina of Time (main build E0F3CCFA...). Found by Hookshot (NotRexed's OoT recomp,
// host/hos_thread.c present_poll): the main thread waits for its frame's GPU work with
//     while (nvnEventGetValue(e) != 1) nn::os::SleepThread(1 ms);
// at main+0x590500, ~3.5 sleeps a frame, so each frame ends 0.5-1 ms after the GPU wrote the
// event. In the loop, `bl nn::os::SleepThread` is at main+0x59051C (returns to main+0x590520) and
// `blr` nvnEventGetValue at main+0x59052C (returns to main+0x590530, Hookshot's
// HOS_PRESENT_NOWAIT). Here that one sleep becomes a wait that the GPU's next write of a 1 (a
// semaphore release or report, Tegra::Nx2SemaphoreWaiters::OnWrite) ends early. It never lasts
// longer than the sleep it replaces, so the loop polls at least as often as before.
constexpr u64 OotProgramId = 0x0400486024DEC000ULL;
constexpr std::array<u8, 20> OotMainBuildId{
    0xE0, 0xF3, 0xCC, 0xFA, 0xF5, 0x00, 0x65, 0xAE, 0x4E, 0xA0,
    0x1E, 0xB5, 0xAE, 0xB1, 0x3B, 0x1A, 0x45, 0xDA, 0x33, 0x48,
};
constexpr u64 PollLoop = 0x590500;
constexpr u64 SleepCall = 0x59051C;
constexpr u64 SleepReturn = 0x590520;
constexpr u64 GetValueCall = 0x59052C;
constexpr u32 CmpW0One = 0x7100041F; // cmp w0, #1 (expected after the GetValue call, logged only)
constexpr u64 DumpFirst = PollLoop - 0x10;
constexpr size_t DumpWords = 20; // main+0x5904F0 .. main+0x59053C

constexpr s64 MaxPollSleepNs = 2'000'000; // the game asks for 1 ms
constexpr s64 MaxWaitNs = 1'000'000;
constexpr s64 ResleepWindowNs = 200'000;
constexpr s64 LogPeriodNs = 10'000'000'000;
constexpr int FrameWalkDepth = 4;
// Return addresses compared without any pointer authentication code above a 39-bit address.
constexpr u64 ReturnAddressMask = (u64{1} << 39) - 1;

constexpr bool IsBl(u32 word) {
    return (word & 0xFC000000U) == 0x94000000U;
}

constexpr bool IsBlr(u32 word) {
    return (word & 0xFFFFFC1FU) == 0xD63F0000U;
}

struct PresentPollState {
    std::mutex site_mutex;
    u64 checked_process_id = ~u64{}; // the process the site was checked in
    u64 site_return = 0;             // main base + SleepReturn; 0 when the site did not verify

    std::atomic<KernelCore*> kernel{};
    KThread* waiter{}; // the thread in the wait; guarded by the scheduler lock
    std::atomic<bool> has_waiter{};

    // Statistics, logged every 10 s while EXP24 is on.
    std::atomic<u64> waits{};
    std::atomic<u64> woken{};
    std::atomic<u64> timed_out{};
    std::atomic<u64> resleeps{};
    std::atomic<u64> last_waker{};
    std::atomic<s64> last_end_tick{};
    std::atomic<bool> last_woken{};
    std::atomic<s64> last_log_tick{};
};

PresentPollState& PollState() {
    static PresentPollState state;
    return state;
}

/// The present poll's wait: a sleep (same timer) that the GPU write listener may end early.
/// Every way out of the wait (GPU write, timer, termination) runs under the scheduler lock and
/// forgets the waiter, so the listener only ever ends a wait that is still this one.
class PresentPollQueue final : public KThreadQueue {
public:
    explicit PresentPollQueue(KernelCore& kernel) : KThreadQueue(kernel) {}

    void EndWait(KernelCore& kernel, KThread* waiting_thread, Result wait_result) override {
        Forget(waiting_thread);
        KThreadQueue::EndWait(kernel, waiting_thread, wait_result);
    }

    void CancelWait(KernelCore& kernel, KThread* waiting_thread, Result wait_result,
                    bool cancel_timer_task) override {
        Forget(waiting_thread);
        KThreadQueue::CancelWait(kernel, waiting_thread, wait_result, cancel_timer_task);
    }

private:
    static void Forget(KThread* waiting_thread) {
        PresentPollState& state = PollState();
        if (state.waiter == waiting_thread) {
            state.waiter = nullptr;
            state.has_waiter.store(false, std::memory_order_release);
        }
    }
};

/// Common::Nx2GpuWrite listener, on the GPU thread that wrote the word.
void OnGpuWrite(u64 device_address, u32 value) {
    if (value != 1) {
        return;
    }
    PresentPollState& state = PollState();
    if (!state.has_waiter.load(std::memory_order_acquire)) {
        return;
    }
    KernelCore* const kernel = state.kernel.load(std::memory_order_acquire);
    if (kernel == nullptr) {
        return;
    }
    KScopedSchedulerLock sl{*kernel};
    KThread* const thread = state.waiter;
    if (thread == nullptr) {
        return;
    }
    state.last_waker.store(device_address, std::memory_order_relaxed);
    thread->EndWait(*kernel, ResultSuccess);
}

/// main base + SleepReturn when this process runs the verified Ocarina of Time build, else 0.
/// Checked once per process.
u64 SiteReturnAddress(Core::System& system, KProcess& process) {
    PresentPollState& state = PollState();
    std::scoped_lock lk{state.site_mutex};
    if (state.checked_process_id == process.GetProcessId()) {
        return state.site_return;
    }
    state.checked_process_id = process.GetProcessId();
    state.site_return = 0;

    const auto& build_id = system.GetApplicationProcessBuildID();
    if (!std::equal(OotMainBuildId.begin(), OotMainBuildId.end(), build_id.begin())) {
        LOG_INFO(Kernel_SVC, "OoT present poll (EXP24): main is not build E0F3CCFA..., off");
        return 0;
    }
    const u64 main_base = GetInteger(Core::FindMainModuleEntrypoint(std::addressof(process)));
    auto& memory = process.GetMemory();
    if (!memory.IsValidVirtualAddressRange(main_base + DumpFirst, DumpWords * sizeof(u32))) {
        LOG_WARNING(Kernel_SVC, "OoT present poll (EXP24): main+{:#x} is not mapped (main {:#x}), off",
                    DumpFirst, main_base);
        return 0;
    }
    std::array<u32, DumpWords> words{};
    std::string text;
    for (size_t i = 0; i < DumpWords; ++i) {
        words[i] = memory.Read32(main_base + DumpFirst + i * sizeof(u32));
        text += fmt::format(" {:08x}", words[i]);
    }
    const auto word_at = [&](u64 offset) { return words[(offset - DumpFirst) / sizeof(u32)]; };
    const u32 sleep_call = word_at(SleepCall);
    const u32 get_value_call = word_at(GetValueCall);
    const u32 compare = word_at(GetValueCall + 4);
    LOG_INFO(Kernel_SVC, "OoT present poll (EXP24): main {:#x}, main+{:#x}:{}", main_base, DumpFirst,
             text);
    if (!IsBl(sleep_call) || !IsBlr(get_value_call)) {
        LOG_WARNING(Kernel_SVC,
                    "OoT present poll (EXP24): main+{:#x} = {:#010x} (want bl), main+{:#x} = "
                    "{:#010x} (want blr), off",
                    SleepCall, sleep_call, GetValueCall, get_value_call);
        return 0;
    }
    LOG_INFO(Kernel_SVC,
             "OoT present poll (EXP24): site verified (bl {:#010x}, blr {:#010x}, then {:#010x}{})",
             sleep_call, get_value_call, compare, compare == CmpW0One ? " = cmp w0, #1" : "");
    state.site_return = main_base + SleepReturn;
    state.kernel.store(std::addressof(system.Kernel()), std::memory_order_release);
    Common::Nx2GpuWrite::SetListener(&OnGpuWrite);
    return state.site_return;
}

/// Whether the current SVC was reached from the call that returns to `site_return`: the link
/// register, or a return address in one of the innermost frame records (nn::os::SleepThread's
/// own frames sit between the svc and the game's call).
bool CalledFromSite(KernelCore& kernel, KProcess& process, u64 site_return) {
    Core::ArmInterface* const arm = process.GetArmInterface(kernel.CurrentPhysicalCoreIndex());
    if (arm == nullptr) {
        return false;
    }
    Svc::ThreadContext ctx{};
    arm->GetContext(ctx);
    if ((ctx.lr & ReturnAddressMask) == site_return) {
        return true;
    }
    auto& memory = process.GetMemory();
    u64 fp = ctx.fp;
    for (int depth = 0; depth < FrameWalkDepth; ++depth) {
        if (fp == 0 || (fp & 7) != 0 || !memory.IsValidVirtualAddressRange(fp, 16)) {
            return false;
        }
        if ((memory.Read64(fp + 8) & ReturnAddressMask) == site_return) {
            return true;
        }
        const u64 next = memory.Read64(fp);
        if (next <= fp) {
            // The stack grows down: a caller's frame record is above its callee's.
            return false;
        }
        fp = next;
    }
    return false;
}

/// Sleeps up to `ns` (> 0), ended early by OnGpuWrite. Returns the wait's result.
Result WaitForGpuWrite(KernelCore& kernel, s64 ns) {
    KThread& thread = GetCurrentThread(kernel);
    PresentPollState& state = PollState();
    PresentPollQueue wait_queue(kernel);
    KHardwareTimer* timer{};
    s64 timeout = kernel.HardwareTimer().GetTick() + ns + 2;
    if (timeout <= 0) {
        timeout = (std::numeric_limits<s64>::max)();
    }
    {
        KScopedSchedulerLockAndSleep slp(kernel, std::addressof(timer), std::addressof(thread),
                                         timeout);
        if (thread.IsTerminationRequested()) {
            slp.CancelSleep();
            return ResultTerminationRequested;
        }
        // One waiter at a time (the game has one present poll); another thread just sleeps.
        if (state.waiter == nullptr) {
            state.waiter = std::addressof(thread);
            state.has_waiter.store(true, std::memory_order_release);
        }
        wait_queue.SetHardwareTimer(timer);
        thread.BeginWait(kernel, std::addressof(wait_queue));
        thread.SetWaitReasonForDebugging(ThreadWaitReasonForDebugging::Sleep);
    }
    return thread.GetWaitResult();
}

void MaybeLog(PresentPollState& state, s64 now) {
    s64 last = state.last_log_tick.load(std::memory_order_relaxed);
    if (last == 0) {
        state.last_log_tick.compare_exchange_strong(last, now, std::memory_order_relaxed);
        return;
    }
    if (now - last < LogPeriodNs ||
        !state.last_log_tick.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
        return;
    }
    const u64 waits = state.waits.exchange(0, std::memory_order_relaxed);
    const u64 woken = state.woken.exchange(0, std::memory_order_relaxed);
    const u64 timed_out = state.timed_out.exchange(0, std::memory_order_relaxed);
    const u64 resleeps = state.resleeps.exchange(0, std::memory_order_relaxed);
    LOG_INFO(Kernel_SVC,
             "OoT present poll (EXP24): {} waits in {:.1f} s, {} ended by a GPU write of 1 ({} "
             "polled again within 200 us: early), {} timed out (1 ms), last waking write {:#x}",
             waits, static_cast<double>(now - last) / 1e9, woken, resleeps, timed_out,
             state.last_waker.load(std::memory_order_relaxed));
}

} // Anonymous namespace

bool SleepNx2OotPresentPoll(Core::System& system, s64 ns) {
    if (ns <= 0 || ns > MaxPollSleepNs || VideoCore::Nx2Debug::Experiment(24) == 0) {
        return false;
    }
    auto& kernel = system.Kernel();
    auto& process = GetCurrentProcess(kernel);
    if (!process.Is64Bit() || process.GetProgramId() != OotProgramId) {
        return false;
    }
    const u64 site_return = SiteReturnAddress(system, process);
    if (site_return == 0 || !CalledFromSite(kernel, process, site_return)) {
        return false;
    }

    PresentPollState& state = PollState();
    const s64 start = kernel.HardwareTimer().GetTick();
    if (state.last_woken.load(std::memory_order_relaxed) &&
        start - state.last_end_tick.load(std::memory_order_relaxed) < ResleepWindowNs) {
        // The last wait ended at a GPU write, and the game found its event not yet 1.
        state.resleeps.fetch_add(1, std::memory_order_relaxed);
    }
    state.waits.fetch_add(1, std::memory_order_relaxed);
    MaybeLog(state, start);

    const Result result = WaitForGpuWrite(kernel, (std::min)(ns, MaxWaitNs));
    const bool woken = result == ResultSuccess;
    if (woken) {
        state.woken.fetch_add(1, std::memory_order_relaxed);
    } else if (result == ResultTimedOut) {
        state.timed_out.fetch_add(1, std::memory_order_relaxed);
    }
    state.last_woken.store(woken, std::memory_order_relaxed);
    state.last_end_tick.store(kernel.HardwareTimer().GetTick(), std::memory_order_relaxed);
    return true;
}

} // namespace Kernel::Svc
