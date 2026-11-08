// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <limits>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "common/paged_scalar.h"

TEST_CASE("Scalar reads translate both sides of a fragmented page boundary", "[memory][paged_scalar]") {
    std::array<u8, 0x4000> backing{};
    backing.fill(0xcd);
    // Guest page 0 is host page 2; guest page 1 is host page 0, not page 3.
    std::vector<u64> translations;
    const auto translate = [&](u64 address) -> u8* {
        translations.push_back(address);
        if (address >= 0x2000) return nullptr;
        return backing.data() + (address < 0x1000 ? 0x2000 : 0) + (address & 0xfff);
    };
    constexpr u64 expected = 0x8877665544332211ULL;
    std::array<u8, sizeof(expected)> bytes{};
    std::memcpy(bytes.data(), &expected, bytes.size());
    for (u64 offset = 0xff8; offset < 0x1000; ++offset) {
        for (size_t i = 0; i < bytes.size(); ++i) *translate(offset + i) = bytes[i];
        translations.clear();
        CHECK(Common::ReadPagedScalar<u64>(offset, translate) == expected);
        REQUIRE(translations.size() == (offset == 0xff8 ? 1U : 2U));
        CHECK(translations.front() == offset);
        if (translations.size() == 2) CHECK(translations.back() == 0x1000);
    }
}

TEST_CASE("Scalar writes do not corrupt the physical page following their first mapping", "[memory][paged_scalar]") {
    std::array<u8, 0x4000> backing{};
    const auto translate = [&](u64 address) -> u8* {
        if (address >= 0x2000) return nullptr;
        return backing.data() + (address < 0x1000 ? 0x2000 : 0) + (address & 0xfff);
    };
    constexpr u64 value = 0x8877665544332211ULL;
    std::array<u8, sizeof(value)> bytes{};
    std::memcpy(bytes.data(), &value, bytes.size());
    for (u64 offset = 0xff8; offset < 0x1000; ++offset) {
        backing.fill(0xcd);
        Common::WritePagedScalar(offset, value, translate);
        for (size_t i = 0; i < bytes.size(); ++i) CHECK(*translate(offset + i) == bytes[i]);
        CHECK(*translate(offset - 1) == 0xcd);
        CHECK(*translate(offset + sizeof(value)) == 0xcd);
        for (size_t i = 0x3000; i < 0x3010; ++i) CHECK(backing[i] == 0xcd);
    }
}

TEST_CASE("A missing scalar page reads zero and does not hide the next mapped page", "[memory][paged_scalar]") {
    std::array<u8, 0x1000> backing{};
    backing.fill(0x5a);
    for (const bool missing_first : {false, true}) {
        const auto translate = [&](u64 address) -> u8* {
            const bool first = address < 0x1000;
            if (address >= 0x2000 || first == missing_first) return nullptr;
            return backing.data() + (address & 0xfff);
        };
        const auto value = Common::ReadPagedScalar<u64>(0xffc, translate);
        CHECK(value == (missing_first ? 0x5a5a5a5a00000000ULL : 0x000000005a5a5a5aULL));
        Common::WritePagedScalar(0xffc, u64{0x1212121234343434ULL}, translate);
        const auto reread = Common::ReadPagedScalar<u64>(0xffc, translate);
        CHECK(reread == (missing_first ? 0x1212121200000000ULL : 0x0000000034343434ULL));
        backing.fill(0x5a);
    }
}

TEST_CASE("Scalar address overflow never wraps a translation back to zero", "[memory][paged_scalar]") {
    std::array<u8, 0x1000> backing{};
    backing.fill(0x5a);
    size_t translations = 0;
    const auto translate = [&](u64 address) -> u8* {
        CHECK(address == (std::numeric_limits<u64>::max)() - 3);
        ++translations;
        return backing.data() + (address & 0xfff);
    };
    CHECK(Common::ReadPagedScalar<u64>((std::numeric_limits<u64>::max)() - 3, translate) == 0x5a5a5a5aULL);
    Common::WritePagedScalar((std::numeric_limits<u64>::max)() - 3, u64{0}, translate);
    CHECK(translations == 2);
    CHECK(backing[0] == 0x5a);
    CHECK(backing.back() == 0);
}
