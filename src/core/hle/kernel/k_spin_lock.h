// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <mutex> // Kernel headers that use std::mutex get it through this one

#include "common/common_funcs.h"
#include "common/common_types.h"
#include "core/hle/kernel/k_scoped_lock.h"

namespace Kernel {

/// Address waits for the kernel's own locks (Common::Nx2AddressWait: WaitOnAddress on Windows).
/// EDEN_NX2_KSPIN=0 restores std::atomic waits and the short lock spin.
namespace Nx2Wait {
bool Enabled();
/// Microseconds a contended kernel lock spins before sleeping (EDEN_NX2_KSPIN_US, default 20).
s64 SpinMicroseconds();
/// Sleeps while word holds expected; may return early.
void Wait(std::atomic<u32>& word, u32 expected);
void WakeOne(std::atomic<u32>& word);
void WakeAll(std::atomic<u32>& word);
} // namespace Nx2Wait

class KSpinLock {
public:
    explicit KSpinLock() = default;

    YUZU_NON_COPYABLE(KSpinLock);
    YUZU_NON_MOVEABLE(KSpinLock);

    void Lock();
    void Unlock();
    bool TryLock();

private:
    void LockContendedNx2();

    // 0 free, 1 held, 2 held with (possible) sleepers. The scheduler lock is taken by every
    // core thousands of times a frame and held for microseconds: a contended Lock spins briefly
    // before sleeping, instead of putting the host thread to sleep at once and paying its
    // wake-up latency (std::mutex did).
    std::atomic<u32> m_state{0};
};

// TODO(bunnei): Alias for now, in case we want to implement these accurately in the future.
using KAlignedSpinLock = KSpinLock;
using KNotAlignedSpinLock = KSpinLock;

using KScopedSpinLock = KScopedLock<KSpinLock>;
using KScopedAlignedSpinLock = KScopedLock<KAlignedSpinLock>;
using KScopedNotAlignedSpinLock = KScopedLock<KNotAlignedSpinLock>;

} // namespace Kernel
