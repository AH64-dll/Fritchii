// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <catch2/catch_test_macros.hpp>
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

namespace {
using Shader::Ampere::Instruction;
using Shader::Ampere::LowerToMaxwell;
constexpr Instruction Exit{{0x794d, 0, 0x03800000, 0}};
u64 Copy(u32 dest, u32 source, u32 guard = 7) {
    return 0x5c98000000000000ULL | (u64{guard} << 16) | dest |
           (15ULL << 39) | (u64{source} << 20);
}
} // namespace

TEST_CASE("Fast Fusion TXD allows an odd single third destination", "[ampere][fast_fusion]") {
    Instruction txd{{0x10047b6c, 0x20800c04, 0x041e0707, 0x00014800}};
    const auto lowered = LowerToMaxwell(std::array{txd, Exit});
    INFO(lowered.error);
    REQUIRE(lowered);
    REQUIRE(lowered.required_texture_buffer == 2);
    REQUIRE(lowered.instructions.size() == 5);
    CHECK((lowered.instructions[0] & 255) == 248);
    CHECK(lowered.instructions[1] == Copy(4, 248));
    CHECK(lowered.instructions[2] == Copy(5, 249));
    CHECK(lowered.instructions[3] == Copy(7, 250));
    txd.words[2] |= 0x800; // Four channels still require an aligned second pair.
    CHECK_FALSE(LowerToMaxwell(std::array{txd, Exit}));
}

TEST_CASE("Fast Fusion formatted stores preserve channel masks and barriers", "[ampere][fast_fusion]") {
    for (u32 mask = 1; mask < 16; ++mask) {
        for (bool strong : {false, true}) {
            const Instruction store{{0x06007b9b, 0x60005904,
                                      (strong ? 0x0010a000U : 0x00100000U) | (mask << 8), 0x000fe200}};
            const auto lowered = LowerToMaxwell(std::array{store, Exit});
            INFO(lowered.error);
            REQUIRE(lowered);
            REQUIRE(lowered.instructions.size() == (strong ? 4U : 2U));
            const auto encoded = lowered.instructions[strong ? 1 : 0];
            CHECK(((encoded >> 20) & 15) == mask);
            CHECK((encoded & 255) == 4);
            CHECK(((encoded >> 8) & 255) == 6);
            if (strong) CHECK(lowered.instructions[0] == lowered.instructions[2]);
        }
    }
    CHECK_FALSE(LowerToMaxwell(std::array{Instruction{{0x06007b9b, 0x60005904, 0x0010a000, 0}}, Exit}));
}

TEST_CASE("Fast Fusion uniform shifts borrow and restore the guard predicate", "[ampere][fast_fusion]") {
    for (u32 guard : {0U, 3U, 6U, 8U, 11U, 14U}) {
        const Instruction shift{{0x3f0b0299U | (guard << 12), 5, 0x08011609, 0x000fe200}};
        const auto lowered = LowerToMaxwell(std::array{shift, Exit});
        INFO(lowered.error);
        REQUIRE(lowered);
        REQUIRE(lowered.instructions.size() == 5);
        const u32 slot = 6 - (guard & 7), mask = 1U << slot;
        CHECK(lowered.instructions[0] == (0x38e8000000070000ULL | 253 | (255ULL << 8) | (u64{mask} << 20)));
        CHECK(lowered.instructions[1] == (0x38f0000000070000ULL | (239ULL << 8) | (u64{mask} << 20)));
        CHECK(lowered.instructions[2] == (0x5c28000000000000ULL | (u64{(guard & 8) | slot} << 16) |
                                          227 | (225ULL << 8) | (221ULL << 20)));
        CHECK(lowered.instructions[3] == (0x38f0000000070000ULL | (253ULL << 8) | (u64{mask} << 20)));
    }
}

TEST_CASE("Fast Fusion uniform descriptor fetch snapshots inputs and splits results", "[ampere][fast_fusion]") {
    const std::array captures{
        Instruction{{0x04087f66, 0x30000405, 0x089e0705, 0x001f4200}},
        Instruction{{0x070a7f66, 0x30000608, 0x089e01ff, 0x000fc200}},
        Instruction{{0x16007f66, 0x3000080c, 0x089e01ff, 0x000fc200}},
        Instruction{{0x04047f66, 0x30000407, 0x089e01ff, 0x000fc200}},
    };
    for (auto fetch : captures) {
        const auto lowered = LowerToMaxwell(std::array{fetch, Exit});
        INFO(lowered.error);
        REQUIRE(lowered);
        CHECK(lowered.instructions[0] == Copy(250, fetch.Bits(24, 8)));
        CHECK(lowered.instructions[1] == Copy(251, fetch.Bits(32, 8)));
        CHECK(lowered.instructions[2] == Copy(246, 216 + fetch.Bits(40, 6)));
        CHECK(((lowered.instructions[3] >> 31) & 15) == fetch.Bits(72, 4));
        CHECK(lowered.instructions[4] == Copy(fetch.Bits(16, 8), 250));
        if (fetch.Bits(72, 4) == 7) {
            CHECK(lowered.instructions[5] == Copy(9, 251));
            CHECK(lowered.instructions[6] == Copy(5, 252));
        }
        fetch.words[2] |= 1U << 12; // Offsets are not silently ignored.
        CHECK_FALSE(LowerToMaxwell(std::array{fetch, Exit}));
    }
}

TEST_CASE("Fast Fusion U64 left shift uses the exact high word through shift 32", "[ampere][fast_fusion]") {
    for (u32 shift = 0; shift <= 32; ++shift) {
        const Instruction shf{{0xa6a57819, shift, 0x000102a5, 0x040fe200}};
        const auto lowered = LowerToMaxwell(std::array{shf, Exit});
        INFO(lowered.error);
        REQUIRE(lowered);
        CHECK(lowered.instructions[0] == (0x36f8000000070000ULL | 165 | (166ULL << 8) |
                                          (165ULL << 39) | (u64{shift} << 20)));
    }
    CHECK_FALSE(LowerToMaxwell(std::array{Instruction{{0xa6a57819, 33, 0x000102a5, 0}}, Exit}));
}

TEST_CASE("Fast Fusion constant LEA handles negation and high-word carry", "[ampere][fast_fusion]") {
    const Instruction low{{0x05057a11, 0x8000ac00, 0x078e18ff, 0x000fe200}};
    const auto lowered = LowerToMaxwell(std::array{low, Exit});
    INFO(lowered.error);
    REQUIRE(lowered);
    REQUIRE(lowered.instructions.size() == 3);
    CHECK(lowered.instructions[0] == (0x4c98000000070000ULL | 251 | (15ULL << 39) | (0xacULL << 20)));
    CHECK(lowered.instructions[1] == (0x5c18000000070000ULL | 5 | (5ULL << 8) |
                                      (251ULL << 20) | (3ULL << 39) | (1ULL << 48)));
    const std::array high_captures{
        Instruction{{0x2935ea11, 0x00009500, 0x010f0c27, 0x000fe400}},
        Instruction{{0x04333a11, 0x00005d00, 0x008f0c03, 0x000fe400}},
        Instruction{{0x20433a11, 0x00005d00, 0x028f0c1f, 0x000fe200}},
    };
    for (const auto high : high_captures) {
        const auto result = LowerToMaxwell(std::array{high, Exit});
        INFO(result.error);
        REQUIRE(result);
        const u64 predicate = u64{high.Bits(12, 4)} << 16;
        REQUIRE(result.instructions.size() == 7);
        CHECK(result.instructions[1] == (0x36f8000000000000ULL | predicate | 250 |
              (u64{high.Bits(24, 8)} << 8) | (u64{high.Bits(64, 8)} << 39) | (1ULL << 20)));
        CHECK(result.instructions[3] == (0x5ca0000000000000ULL | predicate | 252 |
              (252ULL << 8) | (255ULL << 20) | (u64{high.Bits(87, 3)} << 39)));
        CHECK(result.instructions[5] == (0x5c10000000000000ULL | predicate | high.Bits(16, 8) |
              (250ULL << 8) | (251ULL << 20)));
    }
}
