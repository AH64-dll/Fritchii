// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

namespace {
using Shader::Ampere::Instruction;
using Shader::Ampere::LowerToMaxwell;

constexpr Instruction Exit{{0x0000794d, 0, 0x03800000, 0}};
constexpr Instruction Load{{0x10007998, 0x60000000, 0x001e0f1d, 0x010f6600}};
constexpr Instruction Store{{0x0200799c, 0x60000004, 0x00100f00, 0x004fe200}};

TEST_CASE("Ampere captured bindless formatted surfaces preserve operands", "[video_core][ampere]") {
    struct Case {
        Instruction instruction;
        u64 expected;
    };
    for (const auto& test : std::array{
             Case{Load, 0xeb000e8600f71000ULL},
             Case{Store, 0xeb20000600f70204ULL},
             Case{{{0x00037998, 0xa0000000, 0x001e0119, 0x010f6200}},
                  0xeb000c8a00170003ULL},
             Case{{{0x0000799c, 0xa0000004, 0x00100f03, 0x0101e200}},
                  0xeb20018a00f70004ULL},
             Case{{{0x0000799c, 0x8000000c, 0x00100f07, 0x0041e200}},
                  0xeb20038800f7000cULL},
         }) {
        const auto lowered = LowerToMaxwell(std::array{test.instruction, Exit});
        INFO(lowered.error);
        REQUIRE(lowered);
        REQUIRE(lowered.instructions.size() == 2);
        REQUIRE(lowered.instructions.front() == test.expected);
        REQUIRE(lowered.required_texture_buffer == ~0U);
    }
}

TEST_CASE("Ampere bindless surfaces preserve predicates", "[video_core][ampere]") {
    for (const Instruction original : {Load, Store}) {
        auto guarded = original;
        guarded.words[0] = (guarded.words[0] & ~0xf000U) | 0x9000U;
        const auto plain = LowerToMaxwell(std::array{original, Exit});
        const auto lowered = LowerToMaxwell(std::array{guarded, Exit});
        REQUIRE(plain);
        REQUIRE(lowered);
        REQUIRE(lowered.instructions.front() ==
                ((plain.instructions.front() & ~(u64{0xf} << 16)) | (u64{9} << 16)));
    }
}

TEST_CASE("Ampere bindless surfaces reject unsupported fields", "[video_core][ampere]") {
    for (const Instruction original : {Load, Store}) {
        for (const auto& [word, bit] : std::array{
                 std::pair{1U, 8U}, std::pair{1U, 28U}, std::pair{2U, 12U},
                 std::pair{2U, 13U}, std::pair{2U, 20U}, std::pair{2U, 31U},
                 std::pair{3U, 0U}, std::pair{3U, 30U}}) {
            auto invalid = original;
            invalid.words[word] ^= 1U << bit;
            REQUIRE_FALSE(LowerToMaxwell(std::array{invalid, Exit}));
        }
        auto no_channels = original;
        no_channels.words[2] &= ~0xf00U;
        REQUIRE_FALSE(LowerToMaxwell(std::array{no_channels, Exit}));
        auto dimension = original;
        dimension.words[1] &= 0x1fffffffU;
        REQUIRE_FALSE(LowerToMaxwell(std::array{dimension, Exit}));
        auto handle = original;
        handle.words[2] = (handle.words[2] & ~0xffU) | 250U;
        REQUIRE_FALSE(LowerToMaxwell(std::array{handle, Exit}));
    }
    auto load_source = Load;
    load_source.words[1] |= 1;
    REQUIRE_FALSE(LowerToMaxwell(std::array{load_source, Exit}));
    auto unaligned_load = Load;
    unaligned_load.words[0] |= 1U << 16;
    REQUIRE_FALSE(LowerToMaxwell(std::array{unaligned_load, Exit}));
    auto store_destination = Store;
    store_destination.words[0] |= 1U << 16;
    REQUIRE_FALSE(LowerToMaxwell(std::array{store_destination, Exit}));
    auto store_mask = Store;
    store_mask.words[2] = (store_mask.words[2] & ~0xf00U) | 0x500U;
    REQUIRE_FALSE(LowerToMaxwell(std::array{store_mask, Exit}));
    auto store_scratch = Store;
    store_scratch.words[1] = (store_scratch.words[1] & ~0xffU) | 244U;
    REQUIRE_FALSE(LowerToMaxwell(std::array{store_scratch, Exit}));
    auto coordinate_scratch = Store;
    coordinate_scratch.words[0] = (coordinate_scratch.words[0] & 0xffffffU) | (245U << 24);
    REQUIRE_FALSE(LowerToMaxwell(std::array{coordinate_scratch, Exit}));
    const Instruction uniform{{0x00047882, 0, 0, 0x000fe200}};
    REQUIRE(LowerToMaxwell(std::array{uniform, Exit}));
    store_scratch.words[1] = (Store.words[1] & ~0xffU) | 214U;
    REQUIRE_FALSE(LowerToMaxwell(std::array{uniform, store_scratch, Exit}));
}
} // namespace
