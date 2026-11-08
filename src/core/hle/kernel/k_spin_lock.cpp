// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/hle/kernel/k_spin_lock.h"

#include <chrono>
#include <cstdlib>

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include "common/nx2_address_wait.h"

namespace Kernel {

namespace {

// Spins of about 30-140 cycles each before a contended Lock sleeps (a few microseconds).
constexpr int SpinCount = 128;

void Pause() {
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}

} // Anonymous namespace

namespace Nx2Wait {

bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_KSPIN");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

s64 SpinMicroseconds() {
    static const s64 microseconds = [] {
        const char* value = std::getenv("EDEN_NX2_KSPIN_US");
        return value != nullptr ? static_cast<s64>(std::strtoll(value, nullptr, 10)) : s64{20};
    }();
    return microseconds;
}

void Wait(std::atomic<u32>& word, u32 expected) {
    Common::Nx2AddressWait::Wait(word, expected);
}

void WakeOne(std::atomic<u32>& word) {
    Common::Nx2AddressWait::WakeOne(word);
}

void WakeAll(std::atomic<u32>& word) {
    Common::Nx2AddressWait::WakeAll(word);
}

} // namespace Nx2Wait

void KSpinLock::Lock() {
    u32 expected = 0;
    if (m_state.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                        std::memory_order_relaxed)) {
        return;
    }
    if (Nx2Wait::Enabled()) {
        LockContendedNx2();
        return;
    }
    for (int spin = 0; spin < SpinCount; ++spin) {
        Pause();
        if (m_state.load(std::memory_order_relaxed) == 0) {
            expected = 0;
            if (m_state.compare_exchange_weak(expected, 1, std::memory_order_acquire,
                                              std::memory_order_relaxed)) {
                return;
            }
        }
    }
    // Mark the lock contended and sleep until it is released; taken this way it stays marked,
    // so its Unlock wakes the next sleeper.
    while (m_state.exchange(2, std::memory_order_acquire) != 0) {
        m_state.wait(2, std::memory_order_relaxed);
    }
}

void KSpinLock::LockContendedNx2() {
    // Holders keep the lock for a few microseconds; sleeping costs the waiter the host's wake-up
    // latency and the holder a wake syscall. Spin for a bounded time first.
    const auto budget = std::chrono::microseconds{Nx2Wait::SpinMicroseconds()};
    const auto deadline = std::chrono::steady_clock::now() + budget;
    for (u32 spin = 1;; ++spin) {
        Pause();
        if (m_state.load(std::memory_order_relaxed) == 0) {
            u32 expected = 0;
            if (m_state.compare_exchange_weak(expected, 1, std::memory_order_acquire,
                                              std::memory_order_relaxed)) {
                return;
            }
        }
        if ((spin & 31) == 0 && std::chrono::steady_clock::now() >= deadline) {
            break;
        }
    }
    while (m_state.exchange(2, std::memory_order_acquire) != 0) {
        Nx2Wait::Wait(m_state, 2);
    }
}

void KSpinLock::Unlock() {
    if (m_state.exchange(0, std::memory_order_release) == 2) {
        if (Nx2Wait::Enabled()) {
            Nx2Wait::WakeOne(m_state);
        } else {
            m_state.notify_one();
        }
    }
}

bool KSpinLock::TryLock() {
    u32 expected = 0;
    return m_state.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                           std::memory_order_relaxed);
}

} // namespace Kernel
