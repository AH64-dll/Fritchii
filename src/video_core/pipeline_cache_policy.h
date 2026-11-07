// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>

#include "common/common_types.h"

namespace VideoCommon {

/// How many graphics pipelines to evict, least recently used first.
/// @param cached Graphics pipelines in the cache.
/// @param stale  Pipelines among them unused for long enough to be evicted.
/// @param limit  Pipelines kept before any is evicted.
/// @return Zero within the limit, else the excess, at most `stale`.
constexpr size_t GraphicsPipelinesToEvict(size_t cached, size_t stale, size_t limit) {
    if (cached <= limit) {
        return 0;
    }
    return (std::min)(stale, cached - limit);
}

/// Attribute mask of a native vertex program, two bits per attribute as in
/// FixedPipelineState::attribute_types.
/// @param input_generic Components of each generic attribute in the program header's input map.
constexpr u64 VertexAttributeMask(std::span<const std::array<bool, 4>> input_generic) {
    u64 mask = 0;
    for (size_t index = 0; index < input_generic.size(); ++index) {
        if (std::ranges::any_of(input_generic[index], [](bool component) { return component; })) {
            mask |= u64{3} << (index * 2);
        }
    }
    return mask;
}

/// Vertex attribute types a graphics pipeline key keeps.
/// @param attribute_types Types of every attribute register.
/// @param unique_hashes   Program hash of each stage, zero when the stage is off.
/// @param attribute_masks ShaderInfo::vertex_attribute_mask of each stage.
/// @return The types masked by every active stage, or all of them when VertexA is on.
constexpr u64 KeyedAttributeTypes(u64 attribute_types, std::span<const u64> unique_hashes,
                                  std::span<const u64> attribute_masks) {
    if (unique_hashes.empty() || unique_hashes[0] != 0) {
        return attribute_types;
    }
    for (size_t stage = 1; stage < unique_hashes.size(); ++stage) {
        if (unique_hashes[stage] != 0) {
            attribute_types &= attribute_masks[stage];
        }
    }
    return attribute_types;
}

} // namespace VideoCommon
