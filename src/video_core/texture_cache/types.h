// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/common_funcs.h"
#include "common/common_types.h"
#include "common/slot_vector.h"

namespace VideoCommon {

constexpr size_t NUM_RT = 8;
constexpr size_t MAX_MIP_LEVELS = 16;

constexpr Common::SlotId CORRUPT_ID{0xfffffffe};

using ImageId = Common::SlotId;
using ImageMapId = Common::SlotId;
using ImageViewId = Common::SlotId;
using ImageAllocId = Common::SlotId;
using SamplerId = Common::SlotId;
using FramebufferId = Common::SlotId;

/// Fake image ID for null image views
constexpr ImageId NULL_IMAGE_ID{0};
/// Image view ID for null descriptors
constexpr ImageViewId NULL_IMAGE_VIEW_ID{0};
/// Sampler ID for bugged sampler ids
constexpr SamplerId NULL_SAMPLER_ID{0};

enum class ImageType : u32 {
    e1D,
    e2D,
    e3D,
    Linear,
    Buffer,
};

enum class ImageViewType : u32 {
    e1D,
    e2D,
    Cube,
    e3D,
    e1DArray,
    e2DArray,
    CubeArray,
    Rect,
    Buffer,
};
constexpr size_t NUM_IMAGE_VIEW_TYPES = 9;

/// What a draw's render targets are to the asynchronous shaders' skip rule
/// (EDEN_NX2_ASYNC_DISCARD=2, fable_answer_18 1b). Ordered from the safest to skip to the least:
/// a draw takes the last class of all its targets.
enum class DrawTargetClass : u8 {
    /// Fully cleared in this render frame and in one of the few before: a per-frame target, so a
    /// skipped draw is missing for a frame and drawn again in the next.
    Recurring,
    /// Fully cleared in this render frame for the first time (or not registered yet): a new
    /// per-frame target or a layer the game renders once. Its draws wait.
    FirstTime,
    /// Not cleared in this render frame (a draw over contents that stay), or no target at all
    /// (its effects are storage writes). Its draws wait.
    Persistent,
};

/// Render frames back an earlier full clear may be for a target cleared in this render frame to
/// count as recurring.
constexpr u64 RECURRING_CLEAR_WINDOW = 4;

/// The class of one target from the render frames of its last two full clears (~0 = none) and the
/// current render frame. A tick that lands inside a game frame (the host tick, fable_answer_16 1a)
/// can only push a target out of Recurring: its clear counts in the tick before the draw's, so the
/// draw waits. It never makes a target recurring, which takes two full clears in different render
/// frames within the window, so a mis-tick costs a wait and never a skip. (The render-frame clock
/// falls back to frame_tick after two host frames without a count, so it cannot stall long
/// enough to fold clears far apart into the window either.)
[[nodiscard]] constexpr DrawTargetClass ClassifyClearedTarget(u64 last_clear_frame,
                                                              u64 previous_clear_frame,
                                                              u64 render_frame) noexcept {
    constexpr u64 Never = ~u64{0};
    if (last_clear_frame == Never || last_clear_frame != render_frame) {
        return DrawTargetClass::Persistent;
    }
    if (previous_clear_frame == Never || previous_clear_frame >= render_frame ||
        render_frame - previous_clear_frame > RECURRING_CLEAR_WINDOW) {
        return DrawTargetClass::FirstTime;
    }
    return DrawTargetClass::Recurring;
}

enum class RelaxedOptions : u32 {
    Size = 1 << 0,
    Format = 1 << 1,
    Samples = 1 << 2,
    ForceBrokenViews = 1 << 3,
};
DECLARE_ENUM_FLAG_OPERATORS(RelaxedOptions)

struct Offset2D {
    constexpr auto operator<=>(const Offset2D&) const noexcept = default;

    s32 x;
    s32 y;
};

struct Offset3D {
    constexpr auto operator<=>(const Offset3D&) const noexcept = default;

    s32 x;
    s32 y;
    s32 z;
};

struct Region2D {
    constexpr auto operator<=>(const Region2D&) const noexcept = default;

    Offset2D start;
    Offset2D end;
};

struct Extent2D {
    constexpr auto operator<=>(const Extent2D&) const noexcept = default;

    u32 width;
    u32 height;
};

struct Extent3D {
    constexpr auto operator<=>(const Extent3D&) const noexcept = default;

    u32 width;
    u32 height;
    u32 depth;
};

struct SubresourceLayers {
    s32 base_level = 0;
    s32 base_layer = 0;
    s32 num_layers = 1;
};

struct SubresourceBase {
    constexpr auto operator<=>(const SubresourceBase&) const noexcept = default;

    s32 level = 0;
    s32 layer = 0;
};

struct SubresourceExtent {
    constexpr auto operator<=>(const SubresourceExtent&) const noexcept = default;

    s32 levels = 1;
    s32 layers = 1;
};

struct SubresourceRange {
    constexpr auto operator<=>(const SubresourceRange&) const noexcept = default;

    SubresourceBase base;
    SubresourceExtent extent;
};

struct ImageCopy {
    SubresourceLayers src_subresource;
    SubresourceLayers dst_subresource;
    Offset3D src_offset;
    Offset3D dst_offset;
    Extent3D extent;
};

/// A rectangle of one mip level, in texels.
struct LevelRect {
    s32 level;
    u32 x;
    u32 y;
    u32 width;
    u32 height;
};

struct BufferImageCopy {
    size_t buffer_offset;
    size_t buffer_size;
    u32 buffer_row_length;
    u32 buffer_image_height;
    SubresourceLayers image_subresource;
    Offset3D image_offset;
    Extent3D image_extent;
};

struct BufferCopy {
    u64 src_offset;
    u64 dst_offset;
    size_t size;
};

struct SwizzleParameters {
    Extent3D num_tiles;
    Extent3D block;
    size_t buffer_offset;
    s32 level;
};

} // namespace VideoCommon
