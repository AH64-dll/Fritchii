// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"

namespace VideoCore {

struct RasterizerDownloadArea {
    VAddr start_address;
    VAddr end_address;
    bool preemtive;
    /// Not preemtive, but every GPU write there has a download on its way: wait for it to land
    /// instead of flushing (Nx2Reactive::WaitLanding).
    bool wait_landing{};
};

} // namespace VideoCore
