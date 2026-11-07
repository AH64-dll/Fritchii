// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <memory>
#include <vector>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class DescriptorPool;
class Device;
class MemoryAllocator;
class Scheduler;

struct OcarinaActorMesh {
    std::vector<u8> vertices;
    std::vector<u16> indices;
    std::array<f32, 4> current_origin;
    std::array<f32, 4> previous_origin;
    std::array<f32, 280 * 12> current_bones;
    std::array<f32, 280 * 12> previous_bones;
};

#ifdef EGLANTINE_STREAMLINE
class OcarinaLiveDlss {
public:
    OcarinaLiveDlss(const Device& device, Scheduler& scheduler, DescriptorPool& pool,
                    ComputePassDescriptorQueue& queue, MemoryAllocator& allocator);
    ~OcarinaLiveDlss();

    [[nodiscard]] bool EvaluationFailed() const;
    /// Save the 1440p DLSS copy before the game adds its HUD.
    void SaveSceneReference(VkImageView color);
    /// Composite the HUD from the guest image over the matching 4K scene.
    void Composite4k(VkImageView native_composite, u64 frame);
    VkImage CompositedImage() const;
    VkImageView CompositedView() const;

    /// Queue motion, DLSS evaluation and guest writeback for one frame. A false return leaves
    /// the guest dispatch in charge. The scheduler checks EvaluationFailed before skipping it.
    bool Record(VkImageView color, VkImageView depth, VkImageView target, VkExtent2D extent,
                VkExtent2D output_extent, VkExtent2D target_extent, const std::array<f32, 16>& current,
                const std::array<f32, 16>& previous, const OcarinaActorMesh& mesh,
                VkImage depth_image, u64 frame);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
#endif

} // namespace Vulkan
