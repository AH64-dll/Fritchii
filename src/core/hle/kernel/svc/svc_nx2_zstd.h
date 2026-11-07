// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdint>
#include <span>

#include "common/common_types.h"

namespace Core {
class System;
}

namespace Kernel::Svc {

/// Ocarina of Time: svc 0xF0-0xF2 placed by the built-in patch "Native zstd" over the game's calls
/// to ZSTD_decompressMultiFrame (with or without a ZSTD_DDict). Returns false (and does nothing)
/// for any other SVC.
bool CallNx2OotZstd(Core::System& system, u32 imm, std::span<uint64_t, 8> args);

} // namespace Kernel::Svc
