// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>

#include <catch2/catch_test_macros.hpp>

#include "video_core/pipeline_cache_policy.h"

using namespace VideoCommon;

TEST_CASE("Vertex attribute mask follows the header's input map", "[video_core][pipeline_cache]") {
    std::array<std::array<bool, 4>, 32> inputs{};
    CHECK(VertexAttributeMask(inputs) == 0);

    inputs[0] = {true, false, false, false};
    inputs[3] = {false, false, false, true};
    inputs[31] = {false, true, true, false};
    CHECK(VertexAttributeMask(inputs) == ((u64{3} << 0) | (u64{3} << 6) | (u64{3} << 62)));

    inputs.fill({true, true, true, true});
    CHECK(VertexAttributeMask(inputs) == ~u64{0});
}

TEST_CASE("Pipeline keys keep only the attribute types active stages read",
          "[video_core][pipeline_cache]") {
    constexpr u64 types = 0x1234'5678'9abc'def0;
    constexpr u64 first_four = 0xff;

    SECTION("a VertexA program keeps every type") {
        const std::array<u64, 6> hashes{1, 2, 0, 0, 0, 3};
        const std::array<u64, 6> masks{0, 0, 0, 0, 0, 0};
        CHECK(KeyedAttributeTypes(types, hashes, masks) == types);
    }
    SECTION("the vertex program's input map masks the types") {
        const std::array<u64, 6> hashes{0, 2, 0, 0, 0, 3};
        const std::array<u64, 6> masks{0, first_four, 0, 0, 0, ~u64{0}};
        CHECK(KeyedAttributeTypes(types, hashes, masks) == (types & first_four));
    }
    SECTION("inactive stages do not mask") {
        const std::array<u64, 6> hashes{0, 2, 0, 0, 0, 0};
        const std::array<u64, 6> masks{0, ~u64{0}, 0, 0, 0, 0};
        CHECK(KeyedAttributeTypes(types, hashes, masks) == types);
    }
    SECTION("task and mesh programs drop every type") {
        const std::array<u64, 6> hashes{0, 7, 0, 8, 0, 3};
        const std::array<u64, 6> masks{0, 0, 0, 0, 0, ~u64{0}};
        CHECK(KeyedAttributeTypes(types, hashes, masks) == 0);
    }
    SECTION("garbage in unread registers no longer changes the key") {
        const std::array<u64, 6> hashes{0, 2, 0, 0, 0, 3};
        const std::array<u64, 6> masks{0, first_four, 0, 0, 0, ~u64{0}};
        const u64 other = (types & first_four) | 0xabcd'0000'0000'0000;
        CHECK(KeyedAttributeTypes(types, hashes, masks) ==
              KeyedAttributeTypes(other, hashes, masks));
    }
    SECTION("masking a stored key again changes nothing") {
        const std::array<u64, 6> hashes{0, 2, 0, 0, 0, 3};
        const std::array<u64, 6> masks{0, first_four, 0, 0, 0, ~u64{0}};
        const u64 once = KeyedAttributeTypes(types, hashes, masks);
        CHECK(KeyedAttributeTypes(once, hashes, masks) == once);
    }
}
