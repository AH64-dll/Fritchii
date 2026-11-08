// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"

namespace Core {
class System;
}

namespace Kernel::Svc {

/// Ocarina of Time: svc 0xF3, placed by the built-in patch "Resource poll backoff"
/// (EDEN_NX2_OOT_POLL_BACKOFF=1 at launch, zortos293/Lichii#641) on the path where the game's
/// resource request loop (main+0xF7A0) would rescan its pending requests at once. Sleeps the
/// calling thread briefly while EDEN_NX2_EXP25 is on and leaves every register as it was.
/// Returns false (and does nothing) for any other SVC or process.
bool CallNx2OotPollBackoff(Core::System& system, u32 imm);

} // namespace Kernel::Svc
