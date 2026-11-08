// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/nx2_timeline.h"
#include "core/hle/kernel/k_hardware_timer.h"
#include "core/hle/kernel/k_thread_queue.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/svc_results.h"

namespace Kernel {

namespace {

/// The timeline's "guest wake" marker (EDEN_NX2_TIMELINE_GUEST): every wait ends here, under the
/// scheduler lock. `by` is the guest thread that woke it when a guest thread runs on this host
/// thread, else 0: the host thread the marker is on (the timer for a time-out, the fence thread
/// or the GPU back for a GPU event, a service thread for IPC).
void RecordWakeForTimeline(KernelCore& kernel, const KThread* waiting_thread, Result wait_result) {
    u32 result = 4; // other
    if (wait_result == ResultSuccess) {
        result = 0;
    } else if (wait_result == ResultTimedOut) {
        result = 1;
    } else if (wait_result == ResultCancelled) {
        result = 2;
    } else if (wait_result == ResultTerminationRequested) {
        result = 3;
    }
    u64 by = 0;
    if (const KThread* const current = GetCurrentThreadPointer(kernel);
        current != nullptr && !current->IsDummyThread()) {
        const KScheduler* const scheduler = kernel.CurrentScheduler();
        if (scheduler == nullptr || current != scheduler->GetIdleThread()) {
            by = current->GetThreadId();
        }
    }
    Common::Nx2Timeline::Instant("guest wake", {"tid", waiting_thread->GetThreadId()},
                                 {"result", result}, {"by", by});
}

} // Anonymous namespace

void KThreadQueue::NotifyAvailable(KernelCore& kernel, KThread* waiting_thread, KSynchronizationObject* signaled_object, Result wait_result) {
    UNREACHABLE();
}

void KThreadQueue::EndWait(KernelCore& kernel, KThread* waiting_thread, Result wait_result) {
    if (Common::Nx2Timeline::Guest()) [[unlikely]] {
        RecordWakeForTimeline(kernel, waiting_thread, wait_result);
    }

    // Set the thread's wait result.
    waiting_thread->SetWaitResult(wait_result);

    // Set the thread as runnable.
    waiting_thread->SetState(kernel, ThreadState::Runnable);

    // Clear the thread's wait queue.
    waiting_thread->ClearWaitQueue();

    // Cancel the thread task.
    if (m_hardware_timer != nullptr) {
        m_hardware_timer->CancelTask(waiting_thread);
    }
}

void KThreadQueue::CancelWait(KernelCore& kernel, KThread* waiting_thread, Result wait_result, bool cancel_timer_task) {
    if (Common::Nx2Timeline::Guest()) [[unlikely]] {
        RecordWakeForTimeline(kernel, waiting_thread, wait_result);
    }

    // Set the thread's wait result.
    waiting_thread->SetWaitResult(wait_result);

    // Set the thread as runnable.
    waiting_thread->SetState(kernel, ThreadState::Runnable);

    // Clear the thread's wait queue.
    waiting_thread->ClearWaitQueue();

    // Cancel the thread task.
    if (cancel_timer_task && m_hardware_timer != nullptr) {
        m_hardware_timer->CancelTask(waiting_thread);
    }
}

void KThreadQueueWithoutEndWait::EndWait(KernelCore& kernel, KThread* waiting_thread, Result wait_result) {
    UNREACHABLE();
}

} // namespace Kernel
