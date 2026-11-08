// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

#include "common/common_types.h"

/// One listener for the GPU's semaphore and report writes to guest memory (the writes
/// Tegra::Nx2SemaphoreWaiters::OnWrite sees: query cache reports and payload releases, compute
/// releases). It lets the kernel wake a guest thread that polls such a word (Ocarina of Time's
/// present poll, core/hle/kernel/svc/svc_nx2_oot_present.cpp) without video_core knowing about
/// the kernel. Unset, a write costs one atomic load.
namespace Common::Nx2GpuWrite {

/// Called on the GPU thread that wrote `value` to the u32 at device address `device_address`,
/// after the write is visible in guest memory.
using Listener = void (*)(u64 device_address, u32 value);

inline std::atomic<Listener> g_listener{nullptr};

inline void SetListener(Listener listener) {
    g_listener.store(listener, std::memory_order_release);
}

inline void Notify(u64 device_address, u32 value) {
    if (const Listener listener = g_listener.load(std::memory_order_acquire); listener != nullptr)
        [[unlikely]] {
        listener(device_address, value);
    }
}

} // namespace Common::Nx2GpuWrite
