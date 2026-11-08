// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>

#include "common/common_types.h"

namespace Vulkan {

/// Whether a draw whose graphics pipeline is still building may be skipped (asynchronous
/// shaders) instead of waiting for the build.
/// @param native         The draw comes from a Switch 2 (native Turing macro) program.
/// @param depth_enabled  The depth test is on for the draw.
/// @param index_count    Index count of the draw.
/// @param vertex_count   Vertex count of the draw.
/// @param discard_native Let native draws follow the heuristic too (on unless EDEN_NX2_ASYNC_DISCARD=0).
/// @param recurring_target Every render target of the draw is cleared and redrawn every frame
///                         (EDEN_NX2_ASYNC_DISCARD=2 asks; the other modes pass true). Native
///                         draws into any other target wait.
/// @return True when the draw may be skipped.
constexpr bool CanDiscardPendingDraw(bool native, bool depth_enabled, u32 index_count,
                                    u32 vertex_count, bool discard_native = false,
                                    bool recurring_target = true) {
    // Native draws can produce persistent resources, even with depth enabled. They are skipped
    // unless EDEN_NX2_ASYNC_DISCARD=0, trading a possible missing effect for no stall while a
    // new pipeline builds. With EDEN_NX2_ASYNC_DISCARD=2 only those into targets the frame
    // clears and redraws are: a skipped draw into a layer rendered once is missing for good
    // (title screen text, chat bubbles), one into a per-frame target for a frame.
    if (native && (!discard_native || !recurring_target)) {
        return false;
    }
    // Preserve the legacy async-shader heuristic.
    return depth_enabled || (index_count > 6 && vertex_count > 6);
}

constexpr bool DefersShaderEmission(bool enabled, bool native, size_t pending, size_t capacity) {
    return enabled && native && pending < capacity;
}

} // namespace Vulkan
