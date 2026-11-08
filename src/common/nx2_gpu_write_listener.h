// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

#include "common/common_types.h"

/// One listener for the GPU's semaphore and report writes to guest memory, called from every
/// path that writes one (query cache completions, immediate and semi-flush writes, the 3D
/// engine's stamped results, compute releases). It lets the kernel wake a guest thread that polls
/// such a word (Ocarina of Time's present poll, core/hle/kernel/svc/svc_nx2_oot_present.cpp)
/// without video_core knowing about the kernel. Unset, a write costs one atomic load.
namespace Common::Nx2GpuWrite {

/// Which path wrote the word (for the listener's statistics).
enum class Path : u32 {
    QueryCompletion, ///< query cache report written when its fence or sync operation runs
    QueryImmediate,  ///< query cache payload written at once (no fence)
    QuerySemiFlush,  ///< query cache value written by SemiFlushQueryDirty
    Stamp3D,         ///< Maxwell3D::StampQueryResult
    ComputeReduce,   ///< compute report semaphore with a reduction (ProcessNativeReportSemaphore)
    QmdReduce,       ///< QMD release with a reduction
    QmdMax,          ///< QMD max release
    Count,
};

inline constexpr const char* PathName(Path path) {
    switch (path) {
    case Path::QueryCompletion:
        return "query-completion";
    case Path::QueryImmediate:
        return "query-immediate";
    case Path::QuerySemiFlush:
        return "query-semi-flush";
    case Path::Stamp3D:
        return "3d-stamp";
    case Path::ComputeReduce:
        return "compute-reduce";
    case Path::QmdReduce:
        return "qmd-reduce";
    case Path::QmdMax:
        return "qmd-max";
    case Path::Count:
        break;
    }
    return "?";
}

/// Called on the GPU thread that wrote `value` (the low 32 bits) to the word at `host_pointer`
/// (device address `device_address`, GPU address `gpu_address` when known, else 0), after the
/// write is visible in guest memory.
using Listener = void (*)(Path path, u64 device_address, u64 gpu_address, const void* host_pointer,
                          u32 value);

inline std::atomic<Listener> g_listener{nullptr};

inline void SetListener(Listener listener) {
    g_listener.store(listener, std::memory_order_release);
}

inline void Notify(Path path, u64 device_address, u64 gpu_address, const void* host_pointer,
                   u32 value) {
    if (const Listener listener = g_listener.load(std::memory_order_acquire); listener != nullptr)
        [[unlikely]] {
        listener(path, device_address, gpu_address, host_pointer, value);
    }
}

} // namespace Common::Nx2GpuWrite
