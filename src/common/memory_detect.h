// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/common_types.h"

namespace Common {

struct MemoryInfo {
    u64 TotalPhysicalMemory{};
    u64 TotalSwapMemory{};
};

/**
 * Gets the memory info of the host system
 * @return Reference to a MemoryInfo struct with the physical and swap memory sizes in bytes
 */
[[nodiscard]] const MemoryInfo& GetMemInfo();

/// Page faults of this process so far, soft and hard (0 where unknown).
[[nodiscard]] u64 GetProcessPageFaults();

/// This process's memory and the system's commit, for the memory log line (zeros where unknown;
/// only Windows fills them in).
struct ProcessMemoryUsage {
    u64 working_set{};
    u64 peak_working_set{};
    u64 private_bytes{}; ///< Private commit (mapped sections like the guest memory are not in it)
    u64 peak_private_bytes{};
    u64 system_commit{}; ///< Commit charge of the whole system
    u64 system_commit_limit{};
    u64 available_physical{};
};
[[nodiscard]] ProcessMemoryUsage GetProcessMemoryUsage();

} // namespace Common
