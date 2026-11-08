// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

namespace {
using Shader::Ampere::Instruction;
using Shader::Ampere::LowerToMaxwell;

constexpr Instruction Exit{{0x0000794d, 0, 0x03800000, 0}};
constexpr Instruction Clamped{{0x3f057499, 0xffffffff, 0x08010604, 0x000fe200}};
constexpr Instruction Wrapped{{0x3f057499, 0x00000001, 0x08010e04, 0x000fc800}};

TEST_CASE("Ampere captured uniform immediate high-word shifts lower", "[video_core][ampere]") {
    for (const Instruction instruction : {Clamped, Wrapped}) {
        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
        INFO(lowered.error);
        REQUIRE(lowered);
        REQUIRE(lowered.instructions.size() == 3);
        REQUIRE(lowered.instructions[0] ==
                (instruction.words[1] == 1 ? 0x010000000017f0faULL : 0x010ffffffff7f0faULL));
        REQUIRE(lowered.instructions[1] ==
                (instruction.words[1] == 1 ? 0x5c4800800dc7faddULL : 0x5c4800000dc7faddULL));
    }
}

TEST_CASE("Ampere uniform immediate high-word shifts clamp or wrap amounts", "[video_core][ampere]") {
    for (const bool wrap : {false, true}) {
        for (const u32 immediate : {0U, 1U, 0x12345678U, 0xffffffffU}) {
            auto instruction = wrap ? Wrapped : Clamped;
            instruction.words[1] = immediate;
            const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
            REQUIRE(lowered);
            for (const u32 amount : {0U, 1U, 31U, 32U, 33U, 63U, 64U, 0xffffffffU}) {
                std::array<u32, 256> registers{};
                registers[220] = amount;
                for (const u64 word : lowered.instructions) {
                    if ((word >> 56) == 0x01) {
                        registers[word & 255] = static_cast<u32>(word >> 20);
                    } else if ((word >> 48) == 0x5c48) {
                        const u32 source = registers[(word >> 8) & 255];
                        const u32 shift = registers[(word >> 20) & 255];
                        registers[word & 255] = ((word >> 39) & 1) ? source << (shift & 31)
                                                               : shift < 32 ? source << shift : 0;
                    }
                }
                const u32 shift = wrap ? amount & 31 : std::min(amount, 32U);
                const u64 wide = u64{immediate} << 32;
                const u32 expected = static_cast<u32>((wide << shift) >> 32);
                REQUIRE(registers[221] == expected);
            }
        }
    }
}

TEST_CASE("Ampere uniform immediate shifts use the uniform guard", "[video_core][ampere]") {
    for (const u32 guard : {0U, 8U}) {
        auto instruction = Clamped;
        instruction.words[0] = (instruction.words[0] & ~0xf000U) | (guard << 12);
        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
        INFO(lowered.error);
        REQUIRE(lowered);
        u32 shifts = 0;
        for (const u64 word : lowered.instructions) {
            if ((word >> 48) == 0x5c48) {
                REQUIRE(((word >> 16) & 15) == (guard | 6));
                ++shifts;
            }
        }
        REQUIRE(shifts == 1);
        REQUIRE(lowered.instructions.size() > 3);
    }
}

TEST_CASE("Ampere uniform immediate shifts reject unsupported fields", "[video_core][ampere]") {
    for (const Instruction original : {Clamped, Wrapped}) {
        for (const auto& [word, bit] : std::array{
                 std::pair{0U, 24U}, std::pair{2U, 8U}, std::pair{2U, 9U},
                 std::pair{2U, 12U}, std::pair{2U, 16U}, std::pair{2U, 27U},
                 std::pair{3U, 0U}, std::pair{3U, 30U}}) {
            auto invalid = original;
            invalid.words[word] ^= 1U << bit;
            REQUIRE_FALSE(LowerToMaxwell(std::array{invalid, Exit}));
        }
        auto invalid_register = original;
        invalid_register.words[2] = (invalid_register.words[2] & ~255U) | 64U;
        REQUIRE_FALSE(LowerToMaxwell(std::array{invalid_register, Exit}));
        auto discarded_result = original;
        discarded_result.words[0] = (discarded_result.words[0] & ~0xff0000U) | (63U << 16);
        REQUIRE_FALSE(LowerToMaxwell(std::array{discarded_result, Exit}));
    }
}

TEST_CASE("Ampere uniform immediate shifts allocate high uniform registers", "[video_core][ampere]") {
    auto instruction = Clamped;
    instruction.words[0] = (instruction.words[0] & ~0xff0000U) | (28U << 16);
    instruction.words[2] = (instruction.words[2] & ~255U) | 29U;
    const auto first = Shader::Ampere::LowerToMaxwellWith(std::array{instruction, Exit}, 0);
    REQUIRE_FALSE(first);
    REQUIRE(first.high_uniform_registers == 2);
    const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
    INFO(lowered.error);
    REQUIRE(lowered);
    REQUIRE(lowered.instructions[1] == 0x5c4800000d77fad6ULL);
    instruction.words[2] = (instruction.words[2] & ~255U) | 63U;
    const auto zero_shift = LowerToMaxwell(std::array{instruction, Exit});
    REQUIRE(zero_shift);
    REQUIRE(zero_shift.instructions[1] == 0x5c4800000ff7fad6ULL);
}
} // namespace
