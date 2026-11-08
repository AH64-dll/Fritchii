// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdlib>

namespace VideoCommon::Nx2Perf {

// Read once at startup: Verify() runs for every method on the GPU back thread (no guard to check).
inline const bool g_verify = [] {
    const char* value = std::getenv("EDEN_NX2_PERF_VERIFY");
    return value != nullptr && value[0] == '1';
}();

inline bool Verify() {
    return g_verify;
}

} // namespace VideoCommon::Nx2Perf
