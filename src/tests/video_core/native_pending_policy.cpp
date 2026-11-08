// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/async_draw_policy.h"

using Vulkan::DecideNativePending;
using Vulkan::NativeDrawAction;
using Vulkan::NativePendingMode;

TEST_CASE("Wait mode never skips or times out", "[video_core][native_pending]") {
    REQUIRE(DecideNativePending(NativePendingMode::Wait, true, 8000) == NativeDrawAction::Wait);
    REQUIRE(DecideNativePending(NativePendingMode::Wait, false, 0) == NativeDrawAction::Wait);
}

TEST_CASE("Skip mode skips what the heuristic allows", "[video_core][native_pending]") {
    REQUIRE(DecideNativePending(NativePendingMode::Skip, true, 8000) == NativeDrawAction::Skip);
    REQUIRE(DecideNativePending(NativePendingMode::Skip, false, 8000) == NativeDrawAction::Wait);
}

TEST_CASE("Bounded mode waits while budget remains, then skips", "[video_core][native_pending]") {
    REQUIRE(DecideNativePending(NativePendingMode::Bounded, true, 1) ==
            NativeDrawAction::WaitTimed);
    REQUIRE(DecideNativePending(NativePendingMode::Bounded, true, 0) == NativeDrawAction::Skip);
    REQUIRE(DecideNativePending(NativePendingMode::Bounded, true, -500) == NativeDrawAction::Skip);
}

TEST_CASE("Bounded mode still waits fully for draws that cannot be skipped",
          "[video_core][native_pending]") {
    REQUIRE(DecideNativePending(NativePendingMode::Bounded, false, 8000) ==
            NativeDrawAction::Wait);
    REQUIRE(DecideNativePending(NativePendingMode::Bounded, false, 0) == NativeDrawAction::Wait);
}
