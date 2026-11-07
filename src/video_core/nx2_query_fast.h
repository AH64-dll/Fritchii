// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdlib>

#include "common/common_types.h"

// Ocarina of Time (0400486024DEC000) switches for occlusion queries, dispatch submissions and
// fences. Each is on unless its variable starts with '0'.
namespace VideoCommon::Nx2QueryFast {

namespace Detail {
inline int Level(const char* name, int default_level) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == 0) {
        return default_level;
    }
    return value[0] - '0';
}
} // namespace Detail

/// EDEN_NX2_OOT_QUERY_FAST: the split front handles occlusion queries whose results only GPU
/// conditions read (render_enable, macro MME DMA reads) itself: it writes a "visible" result at
/// once instead of sending the report, drops the counter resets nothing real reads, decides the
/// conditions on those results and drops render_enable writes that change nothing on the back.
/// 2: only the last (exact; reports and resets keep their real path). 0: off.
inline int Level() {
    static const int level = Detail::Level("EDEN_NX2_OOT_QUERY_FAST", 1);
    return level;
}
inline bool Enabled() {
    return Level() > 0;
}
/// Whether reports may get a constant "visible" result (EDEN_NX2_OOT_QUERY_FAST=1).
inline bool FakeResults() {
    return Level() == 1;
}

/// EDEN_NX2_DISPATCH_BATCH: command list copies after a dispatch are submitted when the front
/// waits for them (or with the next submission) instead of at once.
inline bool DispatchBatch() {
    static const bool enabled = Detail::Level("EDEN_NX2_DISPATCH_BATCH", 1) != 0;
    return enabled;
}

/// EDEN_NX2_FENCE_TRIM: a fence whose only work is syncing payload releases (QMD releases,
/// semaphores) into the host's buffers is skipped like an empty one (their guest memory is
/// written at once anyway); the syncs go with the next fence made or host wait, batched.
inline bool FenceTrim() {
    static const bool enabled = Detail::Level("EDEN_NX2_FENCE_TRIM", 1) != 0;
    return enabled;
}

/// 3D methods the front's query fast path looks at: MME_DMA_READ(_FIFOED) (0x157/0x158),
/// clear_report_value (0x54c), render_enable.mode (0x556) and report_semaphore.query (0x6c3).
[[nodiscard]] constexpr bool Watches(u32 method) {
    return method == 0x6c3 || method == 0x556 || method == 0x54c || method == 0x157 ||
           method == 0x158;
}

} // namespace VideoCommon::Nx2QueryFast
