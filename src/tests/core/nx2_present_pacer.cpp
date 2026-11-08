// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>

#include <catch2/catch_test_macros.hpp>

#include "core/hle/service/nx2_present_pacer.h"

namespace {
using namespace std::chrono_literals;
using Pacer = Service::Nx2PresentPacer;
constexpr auto Start = Pacer::Clock::time_point{1s};
constexpr auto OneVblank = 16'666'666ns;
constexpr auto TwoVblanks = 33'333'333ns;
constexpr auto ThreeVblanks = 50'000'000ns;
}

TEST_CASE("NX2 FPS limit only lengthens a vblank it is below", "[core][nx2_present_pacer]") {
    using Service::PacedInterval;
    // 0 leaves the game's own vblank and swap interval untouched.
    REQUIRE(PacedInterval(OneVblank, 0) == OneVblank);
    REQUIRE(PacedInterval(TwoVblanks, 0) == TwoVblanks);
    // 60 Hz is exactly the one-vblank interval; 120 Hz is faster than the game asked for.
    REQUIRE(PacedInterval(OneVblank, 60) == OneVblank);
    REQUIRE(PacedInterval(OneVblank, 120) == OneVblank);
    REQUIRE(PacedInterval(TwoVblanks, 60) == TwoVblanks);
    // Lower limits hold a frame for longer: 50, 30, 20 and 15 Hz from one 60 Hz vblank.
    REQUIRE(PacedInterval(OneVblank, 50) == 20'000'000ns);
    REQUIRE(PacedInterval(OneVblank, 30) == 33'333'333ns);
    REQUIRE(PacedInterval(OneVblank, 20) == 50'000'000ns);
    REQUIRE(PacedInterval(OneVblank, 15) == 66'666'666ns);
    // A game already pacing at two vblanks keeps that when the limit is not slower still.
    REQUIRE(PacedInterval(TwoVblanks, 40) == TwoVblanks);
    REQUIRE(PacedInterval(TwoVblanks, 20) == 50'000'000ns);
}

TEST_CASE("NX2 interval decreases apply to the current present", "[core][nx2_present_pacer]") {
    Pacer pacer;
    REQUIRE(pacer.Advance(Start, ThreeVblanks).reset);
    const auto step = pacer.Advance(Start + 2'679us, TwoVblanks);
    REQUIRE_FALSE(step.reset);
    REQUIRE(step.deadline == Start + TwoVblanks);
    REQUIRE(step.deadline - (Start + 2'679us) == 30'654'333ns);
    REQUIRE(pacer.Advance(step.deadline + 1ms, OneVblank).deadline == step.deadline + OneVblank);
}

TEST_CASE("NX2 interval increases apply to the current present", "[core][nx2_present_pacer]") {
    Pacer pacer;
    pacer.Advance(Start, OneVblank);
    const auto step = pacer.Advance(Start + 1ms, ThreeVblanks);
    REQUIRE_FALSE(step.reset);
    REQUIRE(step.deadline == Start + ThreeVblanks);
    REQUIRE(pacer.Advance(step.deadline + 1ms, TwoVblanks).deadline == step.deadline + TwoVblanks);
}

TEST_CASE("NX2 present period changes retain their existing phase", "[core][nx2_present_pacer]") {
    Pacer pacer;
    auto deadline = pacer.Advance(Start, OneVblank).deadline;
    for (const auto period : {TwoVblanks, OneVblank, 8'333'333ns, ThreeVblanks, OneVblank}) {
        const auto next = pacer.Advance(deadline + 1ms, period);
        REQUIRE_FALSE(next.reset);
        REQUIRE(next.deadline == deadline + period);
        deadline = next.deadline;
    }
}

TEST_CASE("NX2 late presents keep the existing bounded catch-up policy", "[core][nx2_present_pacer]") {
    Pacer pacer;
    pacer.Advance(Start, 10ms);
    const auto late = pacer.Advance(Start + 20ms, 10ms);
    REQUIRE_FALSE(late.reset);
    REQUIRE(late.deadline == Start + 10ms);
    REQUIRE(pacer.Advance(Start + 20ms, 10ms).deadline == Start + 20ms);
    const auto reset = pacer.Advance(Start + 41ms, 10ms);
    REQUIRE(reset.reset);
    REQUIRE(reset.deadline == Start + 41ms);
    REQUIRE(pacer.Advance(Start + 41ms, 10ms).deadline == Start + 51ms);
}

TEST_CASE("NX2 late period changes reanchor using the current period", "[core][nx2_present_pacer]") {
    Pacer pacer;
    pacer.Advance(Start, ThreeVblanks);
    const auto late = pacer.Advance(Start + 40ms, OneVblank);
    REQUIRE(late.reset);
    REQUIRE(late.deadline == Start + 40ms);
    const auto next = pacer.Advance(Start + 40ms, OneVblank);
    REQUIRE_FALSE(next.reset);
    REQUIRE(next.deadline == late.deadline + OneVblank);
}

TEST_CASE("NX2 fixed-period pacing matches the previous algorithm", "[core][nx2_present_pacer]") {
    for (const auto period : {OneVblank, TwoVblanks, std::chrono::nanoseconds{ThreeVblanks}}) {
        Pacer pacer;
        Pacer::Clock::time_point old_next{};
        auto now = Start;
        for (const auto advance : {0ms, 1ms, 16ms, 3ms, 50ms, 0ms, 100ms, 2ms, 40ms}) {
            now += advance;
            const bool old_reset = now > old_next + period;
            if (old_reset) {
                old_next = now;
            }
            const auto step = pacer.Advance(now, period);
            REQUIRE(step.reset == old_reset);
            REQUIRE(step.deadline == old_next);
            now = (std::max)(now, step.deadline);
            old_next += period;
        }
    }
}
