// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

#include "common/common_types.h"

/// Address waits for hot locks. On Windows, libstdc++'s std::atomic::wait goes through a
/// winpthreads mutex and condition variable shared by every address that hashes to the same slot:
/// a semaphore round trip per wait and per wake, wakers block on the slot's mutex, and every
/// sleeper of the slot wakes. WaitOnAddress wakes only the word's sleepers and costs no syscall to
/// wake a word nobody sleeps on. Elsewhere (and if the API is missing) std::atomic::wait is used.
namespace Common::Nx2AddressWait {

/// Sleeps while word holds expected; may return early.
void Wait(std::atomic<u32>& word, u32 expected);
void WakeOne(std::atomic<u32>& word);
void WakeAll(std::atomic<u32>& word);

} // namespace Common::Nx2AddressWait
