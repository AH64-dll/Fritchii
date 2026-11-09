// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "common/bit_util.h"
#include "common/common_types.h"

TEST_CASE("Log2Ceil rounds up to the next power of two", "[common][bit_util]") {
    REQUIRE(Common::Log2Ceil<size_t>(1) == 0);
    REQUIRE(Common::Log2Ceil<size_t>(2) == 1);
    REQUIRE(Common::Log2Ceil<size_t>(3) == 2);
    REQUIRE(Common::Log2Ceil<size_t>(4096) == 12);
    REQUIRE(Common::Log2Ceil<size_t>(4097) == 13);
    REQUIRE(Common::Log2Ceil<u32>(0x80000000U) == 31);
    REQUIRE(Common::Log2Ceil<u64>(u64{1} << 63) == 63);
}

TEST_CASE("Log2Ceil of zero is the smallest bucket", "[common][bit_util]") {
    // A zero-size staging request (the null image's upload) indexed the 64-entry staging
    // cache at 0xFFFFFFFF.
    STATIC_REQUIRE(Common::Log2Ceil<size_t>(0) == 0);
    STATIC_REQUIRE(Common::Log2Ceil<u32>(0) == 0);
    STATIC_REQUIRE(Common::Log2Ceil<u64>(0) == 0);
}
