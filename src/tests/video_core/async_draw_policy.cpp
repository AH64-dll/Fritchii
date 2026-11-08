// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <limits>
#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/async_draw_policy.h"
#include "video_core/texture_cache/types.h"

TEST_CASE("Native pending draws are never discarded", "[video_core][async_draw]") {
    constexpr std::array<u32, 6> counts{0, 1, 6, 7, 4096, std::numeric_limits<u32>::max()};
    for (const bool depth : {false, true}) {
        for (const u32 indices : counts) {
            for (const u32 vertices : counts) {
                REQUIRE_FALSE(Vulkan::CanDiscardPendingDraw(true, depth, indices, vertices));
            }
        }
    }
}

/// EDEN_NX2_ASYNC_DISCARD=1: native draws use the legacy heuristic; non-native draws are unchanged.
TEST_CASE("Native pending draws follow the async heuristic when opted in",
          "[video_core][async_draw]") {
    REQUIRE(Vulkan::CanDiscardPendingDraw(true, true, 0, 0, true));
    REQUIRE(Vulkan::CanDiscardPendingDraw(true, false, 7, 7, true));
    REQUIRE_FALSE(Vulkan::CanDiscardPendingDraw(true, false, 6, 100, true));
    REQUIRE_FALSE(Vulkan::CanDiscardPendingDraw(true, false, 0, 0, true));
    // The opt-in never changes non-native draws.
    REQUIRE(Vulkan::CanDiscardPendingDraw(false, true, 0, 0, true) ==
            Vulkan::CanDiscardPendingDraw(false, true, 0, 0));
    REQUIRE_FALSE(Vulkan::CanDiscardPendingDraw(false, false, 0, 0, true));
}

/// EDEN_NX2_ASYNC_DISCARD=2: native draws into a target that is not recurring always wait, and
/// the target class never makes a draw skippable that the heuristic keeps.
TEST_CASE("Native pending draws into persistent targets wait", "[video_core][async_draw]") {
    REQUIRE_FALSE(Vulkan::CanDiscardPendingDraw(true, true, 0, 0, true, false));
    REQUIRE_FALSE(Vulkan::CanDiscardPendingDraw(true, false, 7, 7, true, false));
    REQUIRE(Vulkan::CanDiscardPendingDraw(true, true, 0, 0, true, true));
    REQUIRE_FALSE(Vulkan::CanDiscardPendingDraw(true, false, 0, 0, true, true));
    // Non-native draws keep the legacy heuristic.
    REQUIRE(Vulkan::CanDiscardPendingDraw(false, true, 0, 0, true, false));
    REQUIRE_FALSE(Vulkan::CanDiscardPendingDraw(false, false, 6, 6, true, false));
}

TEST_CASE("Render targets are recurring only after two cleared frames",
          "[video_core][async_draw]") {
    using VideoCommon::ClassifyClearedTarget;
    using VideoCommon::DrawTargetClass;
    constexpr u64 Never = ~u64{0};
    // Never cleared, or not in this render frame: persistent.
    REQUIRE(ClassifyClearedTarget(Never, Never, 10) == DrawTargetClass::Persistent);
    REQUIRE(ClassifyClearedTarget(9, 8, 10) == DrawTargetClass::Persistent);
    // Cleared now for the first time, or after a long gap: first-time.
    REQUIRE(ClassifyClearedTarget(10, Never, 10) == DrawTargetClass::FirstTime);
    REQUIRE(ClassifyClearedTarget(10, 10 - VideoCommon::RECURRING_CLEAR_WINDOW - 1, 10) ==
            DrawTargetClass::FirstTime);
    // Cleared now and in a recent earlier frame: recurring.
    REQUIRE(ClassifyClearedTarget(10, 9, 10) == DrawTargetClass::Recurring);
    REQUIRE(ClassifyClearedTarget(10, 10 - VideoCommon::RECURRING_CLEAR_WINDOW, 10) ==
            DrawTargetClass::Recurring);
    // A clock that went back (a switch of source) never counts as recurring.
    REQUIRE(ClassifyClearedTarget(10, 11, 10) == DrawTargetClass::FirstTime);
    // The draw takes the least safe class of its targets.
    REQUIRE(DrawTargetClass::Recurring < DrawTargetClass::FirstTime);
    REQUIRE(DrawTargetClass::FirstTime < DrawTargetClass::Persistent);
}

TEST_CASE("Legacy pending draws retain the async heuristic", "[video_core][async_draw]") {
    REQUIRE(Vulkan::CanDiscardPendingDraw(false, true, 0, 0));
    REQUIRE(Vulkan::CanDiscardPendingDraw(false, false, 7, 7));
    REQUIRE_FALSE(Vulkan::CanDiscardPendingDraw(false, false, 6, 100));
    REQUIRE_FALSE(Vulkan::CanDiscardPendingDraw(false, false, 100, 6));
    REQUIRE_FALSE(Vulkan::CanDiscardPendingDraw(false, false, 0, 0));
}

TEST_CASE("Deferred shader emission never makes a pending draw discardable",
          "[video_core][async_draw]") {
    constexpr std::array<u32, 6> counts{0, 1, 6, 7, 4096, std::numeric_limits<u32>::max()};
    for (const bool native : {false, true}) {
        REQUIRE_FALSE(Vulkan::DefersShaderEmission(false, native, 0, 1));
        if (!Vulkan::DefersShaderEmission(true, native, 0, 1)) {
            continue;
        }
        for (const bool depth : {false, true}) {
            for (const u32 indices : counts) {
                for (const u32 vertices : counts) {
                    REQUIRE_FALSE(Vulkan::CanDiscardPendingDraw(native, depth, indices, vertices));
                }
            }
        }
    }
    REQUIRE(Vulkan::DefersShaderEmission(true, true, 0, 1));
    REQUIRE_FALSE(Vulkan::DefersShaderEmission(true, false, 0, 1));
}

TEST_CASE("Deferred emission falls back when its worker budget is full",
          "[video_core][async_draw]") {
    for (const size_t capacity : {1U, 2U, 23U}) {
        for (size_t pending = 0; pending < capacity; ++pending) {
            REQUIRE(Vulkan::DefersShaderEmission(true, true, pending, capacity));
        }
        REQUIRE_FALSE(Vulkan::DefersShaderEmission(true, true, capacity, capacity));
        REQUIRE_FALSE(Vulkan::DefersShaderEmission(true, true, capacity + 1, capacity));
    }
    REQUIRE_FALSE(Vulkan::DefersShaderEmission(true, true, 0, 0));
}
