// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>

namespace Common::Windows {

/// Returns the minimum (least precise) supported timer resolution in nanoseconds.
std::chrono::nanoseconds GetMinimumTimerResolution();

/// Returns the maximum (most precise) supported timer resolution in nanoseconds.
std::chrono::nanoseconds GetMaximumTimerResolution();

/// Returns the current timer resolution in nanoseconds.
std::chrono::nanoseconds GetCurrentTimerResolution();

/**
 * Sets the current timer resolution.
 *
 * @param timer_resolution Timer resolution in nanoseconds.
 *
 * @returns The current timer resolution.
 */
std::chrono::nanoseconds SetCurrentTimerResolution(std::chrono::nanoseconds timer_resolution);

/**
 * Sets the current timer resolution to the maximum supported timer resolution.
 *
 * @returns The current timer resolution.
 */
std::chrono::nanoseconds SetCurrentTimerResolutionToMaximum();

/// Sleep for one tick of the current timer resolution.
void SleepForOneTick();

/// A waitable timer precise to ~0.5 ms whatever the system timer resolution is (Windows 10 1803
/// and later), or nullptr where it isn't available. Close with CloseWaitHandle.
void* CreateHighResolutionTimer();

/// An auto-reset event for WaitHighResolution's early wake-up. Close with CloseWaitHandle.
void* CreateWakeEvent();

/// Signals an event from CreateWakeEvent (no-op for nullptr).
void SignalWakeEvent(void* wake_event);

/// Waits `ns` on `timer` or until `wake_event` is signalled. False if the timer couldn't be set
/// (the caller then waits its usual way).
bool WaitHighResolution(void* timer, void* wake_event, std::chrono::nanoseconds ns);

void CloseWaitHandle(void* handle);

} // namespace Common::Windows
