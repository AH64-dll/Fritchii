// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <cstdlib>
#include <thread>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/debugger/debugger.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/physical_core.h"
#include "core/hle/kernel/svc.h"

namespace Kernel {

namespace {

/// EDEN_NX2_CORE_LOCK: the core context lock is a KSpinLock and Idle sleeps on an address wait.
/// The std::mutex and condition variable went through winpthreads: every contended lock, every
/// Interrupt of an idle core and every RunThread entry and exit paid semaphore round trips.
bool CoreLockEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_CORE_LOCK");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

class ScopedCoreGuard {
public:
    explicit ScopedCoreGuard(KSpinLock& fast_guard, std::mutex& guard)
        : m_fast_guard{fast_guard}, m_guard{guard}, m_fast{CoreLockEnabled()} {
        if (m_fast) {
            m_fast_guard.Lock();
        } else {
            m_guard.lock();
        }
    }

    ~ScopedCoreGuard() {
        if (m_fast) {
            m_fast_guard.Unlock();
        } else {
            m_guard.unlock();
        }
    }

    YUZU_NON_COPYABLE(ScopedCoreGuard);
    YUZU_NON_MOVEABLE(ScopedCoreGuard);

private:
    KSpinLock& m_fast_guard;
    std::mutex& m_guard;
    const bool m_fast;
};

} // Anonymous namespace

PhysicalCore::PhysicalCore(KernelCore& kernel, std::size_t core_index)
    : m_core_index{core_index}
{
    m_is_single_core = !kernel.IsMulticore();
}
PhysicalCore::~PhysicalCore() = default;

void PhysicalCore::RunThread(KernelCore& kernel, Kernel::KThread* thread) {
    auto* process = thread->GetOwnerProcess();
    auto& system = kernel.System();
    auto* interface = process->GetArmInterface(m_core_index);

    interface->Initialize();

    const auto EnterContext = [&]() {
        // Lock the core context.
        ScopedCoreGuard lk{m_fast_guard, m_guard};

        // Check if we are already interrupted. If we are, we can just stop immediately.
        if (m_is_interrupted) {
            return false;
        }

        // Mark that we are running.
        m_arm_interface = interface;
        m_current_thread = thread;

        // Acquire the lock on the thread parameters.
        // This allows us to force synchronization with Interrupt.
        interface->LockThread(thread);

        return true;
    };

    const auto ExitContext = [&]() {
        // Save the JIT context back to the thread so the debugger can
        // read the current register state. Debug halt paths (step,
        // breakpoint, watchpoint) return from RunThread without going
        // through the scheduler's Unload/SaveContext.
        if (system.DebuggerEnabled()) {
            interface->GetContext(thread->GetContext());
        }

        // Unlock the thread.
        interface->UnlockThread(thread);

        // Lock the core context.
        ScopedCoreGuard lk{m_fast_guard, m_guard};

        // On exit, we no longer are running.
        m_arm_interface = nullptr;
        m_current_thread = nullptr;
    };

    while (true) {
        // If the thread is scheduled for termination, exit.
        if (thread->HasDpc() && thread->IsTerminationRequested()) {
            thread->Exit(kernel);
        }

        // Notify the debugger and go to sleep if a step was performed
        // and this thread has been scheduled again.
        if (thread->GetStepState() == StepState::StepPerformed) {
            system.GetDebugger().NotifyThreadStopped(thread);
            thread->RequestSuspend(kernel, SuspendType::Debug);
            return;
        }

        // Otherwise, run the thread.
        Core::HaltReason hr{};
        {
            // If we were interrupted, exit immediately.
            if (!EnterContext()) {
                return;
            }

            if (thread->GetStepState() == StepState::StepPending) {
                hr = interface->StepThread(thread);

                if (True(hr & Core::HaltReason::StepThread)) {
                    thread->SetStepState(StepState::StepPerformed);
                }
            } else {
                hr = interface->RunThread(thread);
            }

            ExitContext();
        }

        // Determine why we stopped.
        // If a step completed successfully, skip other halt reason handlers
        // the step takes priority (e.g. step may also set InstructionBreakpoint
        // if the next instruction happens to be a breakpoint).
        const bool step_completed = True(hr & Core::HaltReason::StepThread)
                                    && thread->GetStepState() == StepState::StepPerformed;
        const bool supervisor_call = !step_completed && True(hr & Core::HaltReason::SupervisorCall);
        const bool prefetch_abort = !step_completed && True(hr & Core::HaltReason::PrefetchAbort);
        const bool breakpoint = !step_completed && True(hr & Core::HaltReason::InstructionBreakpoint);
        const bool data_abort = !step_completed && True(hr & Core::HaltReason::DataAbort);
        const bool interrupt = !step_completed && True(hr & Core::HaltReason::BreakLoop);

        // Since scheduling may occur here, we cannot use any cached
        // state after returning from calls we make.

        // Notify the debugger and go to sleep if a breakpoint was hit,
        // or if the thread is unable to continue for any reason.
        if (breakpoint || prefetch_abort) {
            if (breakpoint) {
                interface->RewindBreakpointInstruction();
                // RewindBreakpointInstruction sets the JIT state to the
                // saved breakpoint context. Update the thread context to
                // match, since ExitContext already saved the post-execution
                // state.
                interface->GetContext(thread->GetContext());
            }
            if (system.DebuggerEnabled()) {
                system.GetDebugger().NotifyThreadStopped(thread);
            } else {
                interface->LogBacktrace(process);
            }
            thread->RequestSuspend(kernel, SuspendType::Debug);
            return;
        }

        // Notify the debugger and go to sleep on data abort.
        if (data_abort) {
            if (system.DebuggerEnabled()) {
                system.GetDebugger().NotifyThreadWatchpoint(thread, *interface->HaltedWatchpoint());
            }
            thread->RequestSuspend(kernel, SuspendType::Debug);
            return;
        }

        // Handle system calls.
        if (supervisor_call) {
            // Perform call.
            Svc::Call(system, interface->GetSvcNumber());
            return;
        }

        // Handle external interrupt sources.
        if (interrupt || m_is_single_core) {
            return;
        }
    }
}

void PhysicalCore::LoadContext(const KThread* thread) {
    auto* const process = thread->GetOwnerProcess();
    if (!process) {
        // Kernel threads do not run on emulated CPU cores.
        return;
    }

    auto* interface = process->GetArmInterface(m_core_index);
    if (interface) {
        interface->SetContext(thread->GetContext());
        interface->SetTpidrroEl0(GetInteger(thread->GetTlsAddress()));
        interface->SetWatchpointArray(&process->GetWatchpoints());
    }
}

void PhysicalCore::LoadSvcArguments(const KProcess& process, std::span<const uint64_t, 8> args) {
    process.GetArmInterface(m_core_index)->SetSvcArguments(args);
}

void PhysicalCore::SaveContext(KThread* thread) const {
    auto* const process = thread->GetOwnerProcess();
    if (!process) {
        // Kernel threads do not run on emulated CPU cores.
        return;
    }

    auto* interface = process->GetArmInterface(m_core_index);
    if (interface) {
        interface->GetContext(thread->GetContext());
    }
}

void PhysicalCore::SaveSvcArguments(KProcess& process, std::span<uint64_t, 8> args) const {
    process.GetArmInterface(m_core_index)->GetSvcArguments(args);
}

void PhysicalCore::CloneFpuStatus(KThread* dst) const {
    auto* process = dst->GetOwnerProcess();

    Svc::ThreadContext ctx{};
    process->GetArmInterface(m_core_index)->GetContext(ctx);

    dst->GetContext().fpcr = ctx.fpcr;
    dst->GetContext().fpsr = ctx.fpsr;
}

void PhysicalCore::LogBacktrace(KernelCore& kernel) {
    auto* process = GetCurrentProcessPointer(kernel);
    if (!process) {
        return;
    }

    auto* interface = process->GetArmInterface(m_core_index);
    if (interface) {
        interface->LogBacktrace(process);
    }
}

void PhysicalCore::Idle(KernelCore& kernel) {
    // Guest job systems hand work between cores many times per frame; blocking at once adds the
    // host's thread wake-up latency to every handoff. EDEN_NX2_IDLE_SPIN_US=<n>: poll for that
    // many microseconds before blocking. Switch 2 titles poll 100 us by default outside Android
    // (title 1's frames: 42 ms without, 35 ms with 50 us, 34 ms with 100 us, 36 ms with 300 us).
    static const long long spin_setting = [] {
        const char* text = std::getenv("EDEN_NX2_IDLE_SPIN_US");
        return text ? std::strtoll(text, nullptr, 10) : -1LL;
    }();
    long long spin_us = spin_setting;
    if (spin_us < 0) {
#ifdef __ANDROID__
        spin_us = 0;
#else
        const KProcess* process = kernel.ApplicationProcess();
        spin_us = process != nullptr && process->IsSwitch2() ? 100 : 0;
#endif
    }
    const auto spin = std::chrono::microseconds{spin_us};
    if (spin.count() > 0) {
        // Poll with PAUSE instead of SwitchToThread: no syscall per iteration, and the SMT
        // sibling (often a GPU thread) keeps most of the core. EDEN_NX2_IDLE_PAUSE=0 restores
        // the yield loop.
        [[maybe_unused]] static const bool idle_pause = [] {
            const char* value = std::getenv("EDEN_NX2_IDLE_PAUSE");
            return value == nullptr || value[0] != '0';
        }();
        const auto deadline = std::chrono::steady_clock::now() + spin;
        while (!m_is_interrupted.load(std::memory_order_acquire) &&
               std::chrono::steady_clock::now() < deadline) {
#if defined(_M_X64) || defined(__x86_64__)
            if (idle_pause) {
                for (int i = 0; i < 8; ++i) {
                    _mm_pause();
                }
                continue;
            }
#endif
            std::this_thread::yield();
        }
    }
    if (CoreLockEnabled()) {
        while (m_is_interrupted.load(std::memory_order_acquire) == 0) {
            Nx2Wait::Wait(m_is_interrupted, 0);
        }
        return;
    }
    std::unique_lock lk{m_guard};
    m_on_interrupt.wait(lk, [this] { return m_is_interrupted.load() != 0; });
}

bool PhysicalCore::IsInterrupted() const {
    return m_is_interrupted.load() != 0;
}

void PhysicalCore::Interrupt() {
    if (CoreLockEnabled()) {
        Core::ArmInterface* arm_interface{};
        KThread* thread{};
        {
            ScopedCoreGuard lk{m_fast_guard, m_guard};
            arm_interface = m_arm_interface;
            thread = m_current_thread;
            m_is_interrupted.store(1, std::memory_order_seq_cst);
            if (arm_interface != nullptr) {
                arm_interface->SignalInterrupt(thread);
            }
        }
        // Wakes Idle if it sleeps; costs no syscall otherwise.
        Nx2Wait::WakeAll(m_is_interrupted);
        return;
    }

    // Lock core context.
    std::scoped_lock lk{m_guard};

    // Load members.
    auto* arm_interface = m_arm_interface;
    auto* thread = m_current_thread;

    // Add interrupt flag.
    m_is_interrupted = 1;

    // Interrupt ourselves.
    m_on_interrupt.notify_one();

    // If there is no thread running, we are done.
    if (arm_interface == nullptr) {
        return;
    }

    // Interrupt the CPU.
    arm_interface->SignalInterrupt(thread);
}

void PhysicalCore::ClearInterrupt() {
    ScopedCoreGuard lk{m_fast_guard, m_guard};
    m_is_interrupted = 0;
}

} // namespace Kernel
