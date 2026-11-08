// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/pending_maps.h"

TEST_CASE("Pending GPU maps cover only the addresses they change", "[video_core][pending_maps]") {
    VideoCommon::PendingMaps maps;
    REQUIRE_FALSE(maps.Overlaps(0x10000, 0x1000));

    const u64 first = maps.Add(0x10000, 0x2000);
    CHECK(maps.Overlaps(0x10000, 4));
    CHECK(maps.Overlaps(0x11ffc, 4));
    CHECK(maps.Overlaps(0xf000, 0x1004));
    CHECK_FALSE(maps.Overlaps(0xf000, 0x1000));
    CHECK_FALSE(maps.Overlaps(0x12000, 0x1000));
    CHECK_FALSE(maps.Overlaps(0x10000, 0));

    const u64 second = maps.Add(0x40000, 0x1000);
    CHECK(first != second);
    maps.Remove(first);
    CHECK_FALSE(maps.Overlaps(0x10000, 0x2000));
    CHECK(maps.Overlaps(0x40800, 4));
    maps.Remove(second);
    CHECK_FALSE(maps.Overlaps(0x40000, 0x1000));
}

TEST_CASE("Pending GPU maps of the same range are removed one at a time",
          "[video_core][pending_maps]") {
    VideoCommon::PendingMaps maps;
    const u64 first = maps.Add(0x20000, 0x1000);
    const u64 second = maps.Add(0x20000, 0x1000);
    maps.Remove(first);
    CHECK(maps.Overlaps(0x20000, 0x1000));
    maps.Remove(second);
    CHECK_FALSE(maps.Overlaps(0x20000, 0x1000));
}
