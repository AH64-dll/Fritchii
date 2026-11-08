// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstdint>
#include <map>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/native_qmd.h"

using Tegra::NativeQmdScheduler;
using Address = NativeQmdScheduler::Address;
using Descriptor = NativeQmdScheduler::Descriptor;
using Word = NativeQmdScheduler::Word;

namespace {

constexpr Word Enable = 1U << 23;
constexpr Word OneWord = 1U << 30;
constexpr Word Reduction = 1U << 26;

struct Released {
    Address address;
    Word payload;
    Word flags;
    bool operator==(const Released&) const = default;
};

struct Launch {
    std::string_view error;
    std::vector<Released> released;
};

Descriptor MakeQmd() {
    Descriptor qmd{};
    qmd[18] = 0x30;
    return qmd;
}

void SetRelease(Descriptor& qmd, Word base, Word address, Word flags, Word payload) {
    qmd[base] = address;
    qmd[base + 1] = flags;
    qmd[base + 2] = payload;
}

Launch Schedule(NativeQmdScheduler& scheduler, Address address, const std::map<Address, Descriptor>& memory) {
    Launch launch;
    launch.error = scheduler.Signal(
        address, 3,
        [&](Address qmd_address, Descriptor& descriptor) {
            const auto it = memory.find(qmd_address);
            if (it == memory.end()) return false;
            descriptor = it->second;
            return true;
        },
        [&](Address destination, Word payload, Word flags) {
            launch.released.push_back({destination, payload, flags});
            return true;
        });
    return launch;
}

} // Anonymous namespace

TEST_CASE("An undecodable QMD release slot keeps the other releases and the dependent QMD",
          "[video_core][native_qmd]") {
    // Orbitals (#400): both channels wait on 0x1b18871010, released by slot 0 of this QMD.
    constexpr Address first = 0x1b19635400;
    constexpr Address second = 0x1b19635500;
    auto qmd = MakeQmd();
    SetRelease(qmd, 24, 0x18871010, 0x4088001b, 6);
    SetRelease(qmd, 52, 0x18871020, Enable | OneWord | (1U << 29) | 0x1b, 7);
    qmd[15] = Word(second >> 8);
    qmd[16] = 1 | (1U << 1);
    auto dependent = MakeQmd();
    SetRelease(dependent, 24, 0x18871030, Enable | OneWord | 0x1b, 8);

    NativeQmdScheduler scheduler;
    const auto launch = Schedule(scheduler, first, {{first, qmd}, {second, dependent}});
    REQUIRE(launch.error == "unsupported QMD release format");
    REQUIRE(launch.released == std::vector<Released>{
                                   {0x1b18871010, 6, 0x4088001b},
                                   {0x1b18871030, 8, Enable | OneWord | 0x1b},
                               });
}

TEST_CASE("Timestamped, reduction, signed and interrupting QMD releases are performed",
          "[video_core][native_qmd]") {
    constexpr Address address = 0x20000;
    constexpr Word four_words = Enable;
    constexpr Word increment = Enable | OneWord | Reduction | (3U << 20);
    constexpr Word signed_max_trap = Enable | OneWord | Reduction | (1U << 24) | (2U << 20) | (1U << 27);
    auto qmd = MakeQmd();
    SetRelease(qmd, 24, 0x1000, four_words, 1);
    SetRelease(qmd, 28, 0x1010, increment, 2);
    SetRelease(qmd, 52, 0x1020, signed_max_trap, 3);

    NativeQmdScheduler scheduler;
    const auto launch = Schedule(scheduler, address, {{address, qmd}});
    REQUIRE(launch.error.empty());
    REQUIRE(launch.released == std::vector<Released>{
                                   {0x1000, 1, four_words},
                                   {0x1010, 2, increment},
                                   {0x1020, 3, signed_max_trap},
                               });
}

TEST_CASE("Skipped QMD release slots are reported with their raw words", "[video_core][native_qmd]") {
    constexpr Address address = 0x30000;
    constexpr Word two_words = Enable | (2U << 30);
    constexpr Word reserved_size = Enable | (3U << 30);
    constexpr Word timestamped_reduction = Enable | Reduction;
    auto qmd = MakeQmd();
    SetRelease(qmd, 24, 0x2000, two_words, 1);
    SetRelease(qmd, 28, 0x2010, reserved_size, 2);
    SetRelease(qmd, 52, 0x2020, timestamped_reduction, 3);

    NativeQmdScheduler scheduler;
    std::vector<std::array<Word, 6>> skipped;
    scheduler.skipped_release = [&](Address qmd_address, std::size_t slot, std::span<const Word, 4> words) {
        skipped.push_back({Word(qmd_address), Word(slot), words[0], words[1], words[2], words[3]});
    };
    const auto launch = Schedule(scheduler, address, {{address, qmd}});
    REQUIRE(launch.error == "unsupported QMD release format");
    REQUIRE(launch.released.empty());
    REQUIRE(skipped == std::vector<std::array<Word, 6>>{
                           {Word(address), 0, 0x2000, two_words, 1, 0},
                           {Word(address), 1, 0x2010, reserved_size, 2, 0},
                           {Word(address), 2, 0x2020, timestamped_reduction, 3, 0},
                       });
}

TEST_CASE("Semaphore reductions follow the documented ops", "[video_core][native_qmd]") {
    using Tegra::ReduceSemaphore;
    REQUIRE(ReduceSemaphore(0, false, 5, 3) == 8);
    REQUIRE(ReduceSemaphore(1, false, 0xffffffff, 1) == 1);
    REQUIRE(ReduceSemaphore(1, true, 0xffffffff, 1) == 0xffffffff);
    REQUIRE(ReduceSemaphore(2, false, 0xffffffff, 1) == 0xffffffff);
    REQUIRE(ReduceSemaphore(2, true, 0xffffffff, 1) == 1);
    REQUIRE(ReduceSemaphore(3, false, 4, 5) == 5);
    REQUIRE(ReduceSemaphore(3, false, 5, 5) == 0);
    REQUIRE(ReduceSemaphore(4, false, 3, 5) == 2);
    REQUIRE(ReduceSemaphore(4, false, 0, 5) == 5);
    REQUIRE(ReduceSemaphore(4, false, 7, 5) == 5);
    REQUIRE(ReduceSemaphore(5, false, 0b1100, 0b1010) == 0b1000);
    REQUIRE(ReduceSemaphore(6, false, 0b1100, 0b1010) == 0b1110);
    REQUIRE(ReduceSemaphore(7, false, 0b1100, 0b1010) == 0b0110);
}
