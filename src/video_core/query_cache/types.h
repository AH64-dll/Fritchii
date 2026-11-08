// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_funcs.h"
#include "common/common_types.h"

namespace VideoCommon {

enum class QueryPropertiesFlags : u32 {
    HasTimeout = 1 << 0,
    IsAFence = 1 << 1,
};
DECLARE_ENUM_FLAG_OPERATORS(QueryPropertiesFlags)

// This should always be equivalent to maxwell3d Report Semaphore Reports
enum class QueryType : u32 {
    Payload = 0, // "None" in docs, but confirmed via hardware to return the payload
    VerticesGenerated = 1,
    ZPassPixelCount = 2,
    PrimitivesGenerated = 3,
    AlphaBetaClocks = 4,
    VertexShaderInvocations = 5,
    StreamingPrimitivesNeededMinusSucceeded = 6,
    GeometryShaderInvocations = 7,
    GeometryShaderPrimitivesGenerated = 9,
    ZCullStats0 = 10,
    StreamingPrimitivesSucceeded = 11,
    ZCullStats1 = 12,
    StreamingPrimitivesNeeded = 13,
    ZCullStats2 = 14,
    ClipperInvocations = 15,
    ZCullStats3 = 16,
    ClipperPrimitivesGenerated = 17,
    VtgPrimitivesOut = 18,
    PixelShaderInvocations = 19,
    ZPassPixelCount64 = 21,
    IEEECleanColorTarget = 24,
    IEEECleanZetaTarget = 25,
    StreamingByteCount = 26,
    TessellationInitInvocations = 27,
    BoundingRectangle = 28,
    TessellationShaderInvocations = 29,
    TotalStreamingPrimitivesNeededMinusSucceeded = 30,
    TessellationShaderPrimitivesGenerated = 31,
    // max.
    MaxQueryTypes,
};

// Comparison modes for Host Conditional Rendering
enum class ComparisonMode : u32 {
    False = 0,
    True = 1,
    Conditional = 2,
    IfEqual = 3,
    IfNotEqual = 4,
    MaxComparisonMode,
};

/// EDEN_NX2_QUERY_LAZY_SYNC: a payload release already in guest memory, told to the host's
/// buffers as a guest CPU write of [address, address + size) instead of copied into them.
struct GuestWrittenRange {
    u64 address;
    u64 size;
    bool mark;       ///< In: mark it CPU-written (else only look whether it needs a copy)
    bool needs_copy; ///< Out: its pages hold GPU-written data, so the value is copied as before
};

// Reduction ops.
enum class ReductionOp : u32 {
    RedAdd = 0,
    RedMin = 1,
    RedMax = 2,
    RedInc = 3,
    RedDec = 4,
    RedAnd = 5,
    RedOr = 6,
    RedXor = 7,
    MaxReductionOp,
};

} // namespace VideoCommon
