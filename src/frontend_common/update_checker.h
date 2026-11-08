// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <optional>
#include <span>
#include <string_view>
#include "common/net/net.h"

namespace UpdateChecker {

enum class Failure {
    None,
    // GitHub refused the request because of rate limiting.
    RateLimited,
    // GitHub answered with an HTTP error.
    Http,
    // No usable answer arrived, for example when offline.
    Network,
};

struct CheckResult {
    std::optional<Common::Net::Release> update;
    Failure failure{Failure::None};
    int status{};
};

// Classify a failed release request by its HTTP status, or 0 when no response arrived.
Failure ClassifyFailure(int status);

CheckResult Check();
std::optional<Common::Net::Release> GetUpdate();
std::optional<Common::Net::Release> SelectUpdate(
    std::span<const Common::Net::Release> releases, std::string_view build_tag,
    std::string_view platform, Common::Net::LinuxPackage package = Common::Net::LinuxPackage::Any);

} // namespace UpdateChecker
