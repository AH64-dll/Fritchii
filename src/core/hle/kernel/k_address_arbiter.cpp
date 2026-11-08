// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include "core/arm/exclusive_monitor.h"
#include "core/core.h"
#include "core/hle/kernel/k_address_arbiter.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_scheduler.h"
#include "core/hle/kernel/k_scoped_scheduler_lock_and_sleep.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/k_thread_queue.h"
#include "core/hle/kernel/k_typed_address.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/physical_core.h"
#include "core/hle/kernel/svc_results.h"
#include "core/memory.h"
#include "video_core/nx2_debug_control.h"

namespace Kernel {

KAddressArbiter::KAddressArbiter(Core::System& system_)
    : system{system_}
{}
KAddressArbiter::~KAddressArbiter() = default;

namespace {

bool ReadFromUser(KernelCore& kernel, s32* out, KProcessAddress address) {
    *out = GetCurrentMemory(kernel).Read32(GetInteger(address));
    return true;
}

bool DecrementIfLessThan(KernelCore& kernel, s32* out, KProcessAddress address, s32 value) {
    auto& monitor = GetCurrentProcess(kernel).GetExclusiveMonitor();
    const auto current_core = kernel.CurrentPhysicalCoreIndex();

    // NOTE: If scheduler lock is not held here, interrupt disable is required.
    // KScopedInterruptDisable di;

    // TODO(bunnei): We should call CanAccessAtomic(..) here.

    s32 current_value{};

    while (true) {
        // Load the value from the address.
        current_value =
            static_cast<s32>(monitor.ExclusiveRead32(current_core, GetInteger(address)));

        // Compare it to the desired one.
        if (current_value < value) {
            // If less than, we want to try to decrement.
            const s32 decrement_value = current_value - 1;

            // Decrement and try to store.
            if (monitor.ExclusiveWrite32(current_core, GetInteger(address),
                                         static_cast<u32>(decrement_value))) {
                break;
            }

            // If we failed to store, try again.
        } else {
            // Otherwise, clear our exclusive hold and finish
            monitor.ClearExclusive(current_core);
            break;
        }
    }

    // We're done.
    *out = current_value;
    return true;
}

bool UpdateIfEqual(KernelCore& kernel, s32* out, KProcessAddress address, s32 value,
                   s32 new_value) {
    auto& monitor = GetCurrentProcess(kernel).GetExclusiveMonitor();
    const auto current_core = kernel.CurrentPhysicalCoreIndex();

    // NOTE: If scheduler lock is not held here, interrupt disable is required.
    // KScopedInterruptDisable di;

    // TODO(bunnei): We should call CanAccessAtomic(..) here.

    s32 current_value{};

    // Load the value from the address.
    while (true) {
        current_value =
            static_cast<s32>(monitor.ExclusiveRead32(current_core, GetInteger(address)));

        // Compare it to the desired one.
        if (current_value == value) {
            // If equal, we want to try to write the new value.

            // Try to store.
            if (monitor.ExclusiveWrite32(current_core, GetInteger(address),
                                         static_cast<u32>(new_value))) {
                break;
            }

            // If we failed to store, try again.
        } else {
            // Otherwise, clear our exclusive hold and finish.
            monitor.ClearExclusive(current_core);
            break;
        }
    }

    // We're done.
    *out = current_value;
    return true;
}

class ThreadQueueImplForKAddressArbiter final : public KThreadQueue {
public:
    explicit ThreadQueueImplForKAddressArbiter(KernelCore& kernel, KAddressArbiter::ThreadTree* t)
        : KThreadQueue(kernel), m_tree(t) {}

    void CancelWait(KernelCore& kernel, KThread* waiting_thread, Result wait_result, bool cancel_timer_task) override {
        // If the thread is waiting on an address arbiter, remove it from the tree.
        if (waiting_thread->IsWaitingForAddressArbiter()) {
            m_tree->erase(m_tree->iterator_to(*waiting_thread));
            waiting_thread->ClearAddressArbiter();
        }

        // Invoke the base cancel wait handler.
        KThreadQueue::CancelWait(kernel, waiting_thread, wait_result, cancel_timer_task);
    }

private:
    KAddressArbiter::ThreadTree* m_tree{};
};

/// EDEN_NX2_ARB_SPIN (live, VideoCore::Nx2Debug::g_arb_spin; off by default, #641 uses 8):
/// microseconds a thread about to wait on an address first watches the word. Capped at 1 ms.
s64 ArbSpinMicroseconds() {
    const int value =
        VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_arb_spin, "EDEN_NX2_ARB_SPIN", 0);
    return value <= 0 ? 0 : (std::min)(s64{value}, s64{1000});
}

void SpinPause() {
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}

/// From zortos293/Lichii#641. Guest job systems hand work between threads through these waits
/// many times a frame, and the signal often comes a few microseconds later. Before the thread
/// sleeps (scheduler lock, context switch, then the signaler's EndWait, interrupt and switch
/// back), it watches the word briefly without any lock. If the word stops satisfying the wait,
/// the wait ends with ResultInvalidState, exactly what the kernel returns to a thread that entered
/// the SVC that much later. Threads whose spins rarely pay off stop spinning
/// (KThread::m_nx2_arb_spin_credit) and probe again every 16th wait.
/// The spin is bounded by time, not by a PAUSE count: PAUSE takes ~140 cycles on Intel Skylake
/// and later but ~10-60 on AMD, so the clock (and the core's interrupt flag) is checked every 16
/// pauses, which overshoots the budget by well under a microsecond on either.
template <typename StillWaits>
bool SpinBeforeWait(KernelCore& kernel, KThread* cur_thread, uint64_t addr, bool is_64_bit,
                    StillWaits&& still_waits) {
    const s64 spin_us = ArbSpinMicroseconds();
    if (spin_us == 0 || !kernel.IsMulticore() || (addr & (is_64_bit ? 7 : 3)) != 0) {
        return false;
    }
    u8& credit = cur_thread->m_nx2_arb_spin_credit;
    if (credit == 0 && (++cur_thread->m_nx2_arb_spin_probe & 15) != 0) {
        return false;
    }
    const u8* const pointer = GetCurrentMemory(kernel).GetPointerSilent(addr);
    if (pointer == nullptr) {
        return false;
    }
    const auto load_word = [&]() -> s64 {
        if (is_64_bit) {
            return *reinterpret_cast<const volatile s64*>(pointer);
        }
        return *reinterpret_cast<const volatile s32*>(pointer);
    };
    const PhysicalCore& core = kernel.CurrentPhysicalCore();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds{spin_us};
    for (u32 spin = 1;; ++spin) {
        SpinPause();
        if (!still_waits(load_word())) {
            credit = static_cast<u8>((std::min)(credit + 2, 16));
            // A terminating thread takes the normal path, which reports the termination first.
            return !cur_thread->IsTerminationRequested();
        }
        if ((spin & 15) == 0) {
            // Something else must run on this core: stop watching (not counted as a miss).
            if (core.IsInterrupted()) {
                return false;
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                break;
            }
        }
    }
    if (credit > 0) {
        --credit;
    }
    return false;
}

} // namespace

Result KAddressArbiter::Signal(uint64_t addr, s32 count) {
    // Perform signaling.
    s32 num_waiters{};
    {
        KScopedSchedulerLock sl(system.Kernel());

        auto it = m_tree.nfind_key({addr, -1});
        while ((it != m_tree.end()) && (count <= 0 || num_waiters < count) &&
               (it->GetAddressArbiterKey() == addr)) {
            // End the thread's wait.
            KThread* target_thread = std::addressof(*it);
            target_thread->EndWait(system.Kernel(), ResultSuccess);

            ASSERT(target_thread->IsWaitingForAddressArbiter());
            target_thread->ClearAddressArbiter();

            it = m_tree.erase(it);
            ++num_waiters;
        }
    }
    R_SUCCEED();
}

Result KAddressArbiter::SignalAndIncrementIfEqual(uint64_t addr, s32 value, s32 count) {
    // Perform signaling.
    s32 num_waiters{};
    {
        KScopedSchedulerLock sl(system.Kernel());

        // Check the userspace value.
        s32 user_value{};
        R_UNLESS(UpdateIfEqual(system.Kernel(), std::addressof(user_value), addr, value, value + 1),
                 ResultInvalidCurrentMemory);
        R_UNLESS(user_value == value, ResultInvalidState);

        auto it = m_tree.nfind_key({addr, -1});
        while ((it != m_tree.end()) && (count <= 0 || num_waiters < count) &&
               (it->GetAddressArbiterKey() == addr)) {
            // End the thread's wait.
            KThread* target_thread = std::addressof(*it);
            target_thread->EndWait(system.Kernel(), ResultSuccess);

            ASSERT(target_thread->IsWaitingForAddressArbiter());
            target_thread->ClearAddressArbiter();

            it = m_tree.erase(it);
            ++num_waiters;
        }
    }
    R_SUCCEED();
}

Result KAddressArbiter::SignalAndModifyByWaitingCountIfEqual(uint64_t addr, s32 value, s32 count) {
    // Perform signaling.
    s32 num_waiters{};
    {
        KScopedSchedulerLock sl(system.Kernel());

        auto it = m_tree.nfind_key({addr, -1});
        // Determine the updated value.
        s32 new_value{};
        if (count <= 0) {
            if (it != m_tree.end() && it->GetAddressArbiterKey() == addr) {
                new_value = value - 2;
            } else {
                new_value = value + 1;
            }
        } else {
            if (it != m_tree.end() && it->GetAddressArbiterKey() == addr) {
                auto tmp_it = it;
                s32 tmp_num_waiters{};
                while (++tmp_it != m_tree.end() && tmp_it->GetAddressArbiterKey() == addr) {
                    if (tmp_num_waiters++ >= count) {
                        break;
                    }
                }

                if (tmp_num_waiters < count) {
                    new_value = value - 1;
                } else {
                    new_value = value;
                }
            } else {
                new_value = value + 1;
            }
        }

        // Check the userspace value.
        s32 user_value{};
        bool succeeded{};
        if (value != new_value) {
            succeeded = UpdateIfEqual(system.Kernel(), std::addressof(user_value), addr, value, new_value);
        } else {
            succeeded = ReadFromUser(system.Kernel(), std::addressof(user_value), addr);
        }

        R_UNLESS(succeeded, ResultInvalidCurrentMemory);
        R_UNLESS(user_value == value, ResultInvalidState);

        while ((it != m_tree.end()) && (count <= 0 || num_waiters < count) &&
               (it->GetAddressArbiterKey() == addr)) {
            // End the thread's wait.
            KThread* target_thread = std::addressof(*it);
            target_thread->EndWait(system.Kernel(), ResultSuccess);

            ASSERT(target_thread->IsWaitingForAddressArbiter());
            target_thread->ClearAddressArbiter();

            it = m_tree.erase(it);
            ++num_waiters;
        }
    }
    R_SUCCEED();
}

Result KAddressArbiter::WaitIfLessThan(uint64_t addr, s32 value, bool decrement, s64 timeout) {
    // Prepare to wait.
    KThread* cur_thread = GetCurrentThreadPointer(system.Kernel());
    // EDEN_NX2_ARB_SPIN: the word stopped being less than value during a short spin, so the
    // wait fails as if entered then (no decrement, as the kernel does for a value not below).
    if (timeout != 0 && SpinBeforeWait(system.Kernel(), cur_thread, addr, false,
                                       [value](s64 user_value) { return user_value < value; })) {
        R_THROW(ResultInvalidState);
    }
    KHardwareTimer* timer{};
    ThreadQueueImplForKAddressArbiter wait_queue(system.Kernel(), std::addressof(m_tree));

    {
        KScopedSchedulerLockAndSleep slp{system.Kernel(), std::addressof(timer), cur_thread, timeout};

        // Check that the thread isn't terminating.
        if (cur_thread->IsTerminationRequested()) {
            slp.CancelSleep();
            R_THROW(ResultTerminationRequested);
        }

        // Read the value from userspace.
        s32 user_value{};
        bool succeeded{};
        if (decrement) {
            succeeded = DecrementIfLessThan(system.Kernel(), std::addressof(user_value), addr, value);
        } else {
            succeeded = ReadFromUser(system.Kernel(), std::addressof(user_value), addr);
        }

        if (!succeeded) {
            slp.CancelSleep();
            R_THROW(ResultInvalidCurrentMemory);
        }

        // Check that the value is less than the specified one.
        if (user_value >= value) {
            slp.CancelSleep();
            R_THROW(ResultInvalidState);
        }

        // Check that the timeout is non-zero.
        if (timeout == 0) {
            slp.CancelSleep();
            R_THROW(ResultTimedOut);
        }

        // Set the arbiter.
        cur_thread->SetAddressArbiter(std::addressof(m_tree), addr);
        m_tree.insert(*cur_thread);

        // Wait for the thread to finish.
        wait_queue.SetHardwareTimer(timer);
        cur_thread->BeginWait(system.Kernel(), std::addressof(wait_queue));
        cur_thread->SetWaitReasonForDebugging(ThreadWaitReasonForDebugging::Arbitration);
    }

    // Get the result.
    return cur_thread->GetWaitResult();
}

Result KAddressArbiter::WaitIfEqual(uint64_t addr, s64 value, s64 timeout, bool is_64_bit) {
    // Prepare to wait.
    KThread* cur_thread = GetCurrentThreadPointer(system.Kernel());
    // EDEN_NX2_ARB_SPIN: the word changed during a short spin, so the wait fails as if entered
    // then.
    if (timeout != 0 && SpinBeforeWait(system.Kernel(), cur_thread, addr, is_64_bit,
                                       [value](s64 user_value) { return user_value == value; })) {
        R_THROW(ResultInvalidState);
    }
    KHardwareTimer* timer{};
    ThreadQueueImplForKAddressArbiter wait_queue(system.Kernel(), std::addressof(m_tree));

    {
        KScopedSchedulerLockAndSleep slp{system.Kernel(), std::addressof(timer), cur_thread, timeout};

        // Check that the thread isn't terminating.
        if (cur_thread->IsTerminationRequested()) {
            slp.CancelSleep();
            R_THROW(ResultTerminationRequested);
        }

        // Read the value from userspace.
        s64 user_value{};
        bool succeeded{};
        if (is_64_bit) {
            auto& memory = GetCurrentMemory(system.Kernel());
            succeeded = memory.IsValidVirtualAddressRange(addr, sizeof(s64));
            if (succeeded) {
                user_value = static_cast<s64>(memory.Read64(addr));
            }
        } else {
            s32 user_value32{};
            succeeded = ReadFromUser(system.Kernel(), std::addressof(user_value32), addr);
            user_value = user_value32;
        }
        if (!succeeded) {
            slp.CancelSleep();
            R_THROW(ResultInvalidCurrentMemory);
        }

        // Check that the value is equal.
        if (value != user_value) {
            slp.CancelSleep();
            R_THROW(ResultInvalidState);
        }

        // Check that the timeout is non-zero.
        if (timeout == 0) {
            slp.CancelSleep();
            R_THROW(ResultTimedOut);
        }

        // Set the arbiter.
        cur_thread->SetAddressArbiter(std::addressof(m_tree), addr);
        m_tree.insert(*cur_thread);

        // Wait for the thread to finish.
        wait_queue.SetHardwareTimer(timer);
        cur_thread->BeginWait(system.Kernel(), std::addressof(wait_queue));
        cur_thread->SetWaitReasonForDebugging(ThreadWaitReasonForDebugging::Arbitration);
    }

    // Get the result.
    return cur_thread->GetWaitResult();
}

} // namespace Kernel
