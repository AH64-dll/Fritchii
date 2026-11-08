// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <utility>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

TEST_CASE("Ampere uniform vote writes mapped uniform registers", "[video_core][ampere]") {
    using Shader::Ampere::Instruction;
    using Shader::Ampere::LowerToMaxwell;
    const Instruction exit{{0x0000794d, 0, 0x03800000, 0}};
    constexpr std::array mappings{
        std::pair{15U, 231U}, std::pair{16U, 232U}, std::pair{21U, 237U},
        std::pair{22U, 240U}, std::pair{27U, 245U}, std::pair{28U, 214U},
        std::pair{29U, 215U}, std::pair{62U, 214U}, std::pair{63U, 255U},
    };

    for (const auto& [target, mapped] : mappings) {
        const Instruction vote{{0x00007886U | (target << 16), 0, 0x038e0100, 0}};
        const auto lowered = LowerToMaxwell(std::array{vote, exit});
        REQUIRE(static_cast<bool>(lowered));
        REQUIRE(lowered.instructions.size() == 2);
        REQUIRE((lowered.instructions[0] & 0xff) == mapped);
        REQUIRE(((lowered.instructions[0] >> 48) & 3) == 1);
        CHECK(lowered.instructions[0] ==
              (0x50d8000000070000ULL | mapped | (7ULL << 39) | (7ULL << 45) | (1ULL << 48)));
    }

    for (const u32 target : {64U, 127U, 254U, 255U}) {
        CAPTURE(target);
        const Instruction unsupported{{0x00007886U | (target << 16), 0, 0x038e0100, 0}};
        REQUIRE_FALSE(LowerToMaxwell(std::array{unsupported, exit}));
    }
    for (const u32 bit : {24U, 32U, 64U, 73U}) {
        CAPTURE(bit);
        Instruction unsupported{{0x001c7886, 0, 0x038e0100, 0}};
        unsupported.words[bit / 32] ^= 1U << (bit % 32);
        REQUIRE_FALSE(LowerToMaxwell(std::array{unsupported, exit}));
    }
}
