// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"

namespace Core {
class System;
}

namespace Kernel::Svc {

/// Ocarina of Time's present poll, EDEN_NX2_EXP24 (live, off by default). Called by svcSleepThread
/// with its positive timeout. When the sleep is the one in the game's loop
/// `while (nvnEventGetValue(e) != 1) SleepThread(1 ms)` (main+0x590500, call returning to
/// main+0x590520), the thread instead waits until the GPU writes a 1 to guest memory, at most
/// min(ns, 1 ms), and true is returned. Returns false (and does nothing) for any other sleep.
bool SleepNx2OotPresentPoll(Core::System& system, s64 ns);

} // namespace Kernel::Svc
