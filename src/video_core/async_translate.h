// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdlib>

namespace VideoCommon {

/// EDEN_NX2_ASYNC_TRANSLATE, read on first use (main() has applied nx2_switches.txt by then):
/// 0 (default) translates every new graphics pipeline on the GPU thread, as before.
/// 1 translates the new pipelines of draws that may be skipped on the translation workers, from
///   a snapshot of what the translation reads; the draws are skipped until the pipeline is ready.
/// 2 is 1 plus a translation on the GPU thread from the live state, compared with the worker's
///   (SPIR-V and recorded reads); a worker result that differs is never used. A test mode.
/// 3 translates on the GPU thread as before, from the snapshot, and checks every recorded read
///   against the live state (the snapshot test, without workers).
[[nodiscard]] inline int AsyncTranslateMode() {
    static const int mode = [] {
        const char* const value = std::getenv("EDEN_NX2_ASYNC_TRANSLATE");
        if (value == nullptr) {
            return 0;
        }
        const int parsed = std::atoi(value);
        return parsed >= 0 && parsed <= 3 ? parsed : 0;
    }();
    return mode;
}

/// Whether translation workers read guest memory (modes 1 and 2): the GPU memory manager's
/// writers then take its page table lock.
[[nodiscard]] inline bool AsyncTranslateWorkers() {
    const int mode = AsyncTranslateMode();
    return mode == 1 || mode == 2;
}

} // namespace VideoCommon
