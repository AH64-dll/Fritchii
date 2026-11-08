// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef EGLANTINE_STREAMLINE
#include "common/logging.h"
#include "video_core/host_shaders/ocarina_dlss_color_comp_spv.h"
#include "video_core/host_shaders/ocarina_dlss_output_comp_spv.h"
#include "video_core/host_shaders/ocarina_dlss_4k_composite_comp_spv.h"
#include "video_core/host_shaders/ocarina_dlss_4k_reference_comp_spv.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/renderer_vulkan/vk_streamline_dlss.h"

#include "video_core/host_shaders/ocarina_camera_motion_comp_spv.h"
#include "video_core/host_shaders/ocarina_actor_motion_frag_spv.h"
#include "video_core/host_shaders/ocarina_actor_motion_vert_spv.h"
#include "video_core/renderer_vulkan/vk_compute_pass.h"
#include "video_core/renderer_vulkan/vk_ocarina_camera_motion.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"

namespace Vulkan {
namespace {

using Matrix = std::array<f32, 16>;

/// Reproject depth from the current clip space directly into the previous camera's clip space.
std::optional<Matrix> ClipToPreviousClip(const Matrix& current, const Matrix& previous) {
    std::array<std::array<f64, 8>, 4> rows{};
    for (size_t i = 0; i < 4; ++i) {
        std::copy_n(current.begin() + i * 4, 4, rows[i].begin());
        rows[i][i + 4] = 1.0;
    }
    for (size_t col = 0; col < 4; ++col) {
        const auto pivot = std::max_element(rows.begin() + col, rows.end(),
            [col](const auto& a, const auto& b) { return std::abs(a[col]) < std::abs(b[col]); });
        std::swap(rows[col], *pivot);
        const f64 divisor = rows[col][col];
        if (!std::isfinite(divisor) || std::abs(divisor) < 1e-10) {
            return std::nullopt;
        }
        for (f64& value : rows[col]) {
            value /= divisor;
        }
        for (size_t row = 0; row < 4; ++row) {
            if (row == col) {
                continue;
            }
            const f64 scale = rows[row][col];
            for (size_t i = 0; i < 8; ++i) {
                rows[row][i] -= scale * rows[col][i];
            }
        }
    }
    Matrix result{};
    for (size_t row = 0; row < 4; ++row) {
        for (size_t col = 0; col < 4; ++col) {
            f64 value{};
            for (size_t i = 0; i < 4; ++i) {
                value += previous[row * 4 + i] * rows[i][col + 4];
            }
            result[row * 4 + col] = static_cast<f32>(value);
        }
    }
    if (!std::ranges::all_of(result, [](f32 value) { return std::isfinite(value); })) {
        return std::nullopt;
    }
    return result;
}

constexpr VkFormat MOTION_FORMAT = VK_FORMAT_R16G16_SFLOAT;
constexpr std::array BINDINGS{
    VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                 VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                 VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
};
constexpr std::array TEMPLATES{
    VkDescriptorUpdateTemplateEntry{0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                    0, sizeof(DescriptorUpdateEntry)},
    VkDescriptorUpdateTemplateEntry{1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                    sizeof(DescriptorUpdateEntry), sizeof(DescriptorUpdateEntry)},
};
constexpr DescriptorBankInfo BANK{.textures = 1, .images = 1, .score = 2};
constexpr std::array PUSH_CONSTANTS{
    VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Matrix)},
};
constexpr VkImageSubresourceRange MOTION_RANGE{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

struct ActorTransforms {
    std::array<f32, 16> current;
    std::array<f32, 16> previous;
    std::array<f32, 4> current_origin;
    std::array<f32, 4> previous_origin;
    std::array<f32, 4> extent;
};

vk::Buffer UploadActorData(MemoryAllocator& allocator, std::span<const std::byte> data,
                           VkBufferUsageFlags usage) {
    auto buffer = allocator.CreateBuffer(VkBufferCreateInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = data.size(),
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    }, MemoryUsage::Upload);
    std::memcpy(buffer.Mapped().data(), data.data(), data.size());
    buffer.Flush();
    return buffer;
}

class ActorOverlay {
public:
    ActorOverlay(const Device& device, MemoryAllocator& allocator, const OcarinaActorMesh& mesh,
                 const ActorTransforms& transforms, VkImageView depth_view, VkSampler sampler,
                 VkImageView motion_view, VkExtent2D extent_)
        : extent{extent_}, index_count{static_cast<u32>(mesh.indices.size())}, vertex_bytes{mesh.vertices.size()},
          vertices{UploadActorData(allocator, std::as_bytes(std::span{mesh.vertices}),
                                   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)},
          indices{UploadActorData(allocator, std::as_bytes(std::span{mesh.indices}),
                                  VK_BUFFER_USAGE_INDEX_BUFFER_BIT)},
          current{UploadActorData(allocator, std::as_bytes(std::span{mesh.current_bones}),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)},
          previous{UploadActorData(allocator, std::as_bytes(std::span{mesh.previous_bones}),
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)},
          uniform{UploadActorData(allocator, std::as_bytes(std::span{&transforms, 1}),
                                  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)} {
        const auto& logical = device.GetLogical();
        const std::array bindings{
            VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT},
            VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT},
            VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                                         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
            VkDescriptorSetLayoutBinding{3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                         VK_SHADER_STAGE_FRAGMENT_BIT},
        };
        set_layout = logical.CreateDescriptorSetLayout({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = static_cast<u32>(bindings.size()), .pBindings = bindings.data(),
        });
        const std::array sizes{
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
        };
        descriptor_pool = logical.CreateDescriptorPool({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1, .poolSizeCount = static_cast<u32>(sizes.size()), .pPoolSizes = sizes.data(),
        });
        const VkDescriptorSetLayout raw_layout = *set_layout;
        sets = descriptor_pool.Allocate({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = *descriptor_pool, .descriptorSetCount = 1, .pSetLayouts = &raw_layout,
        });
        const VkDescriptorSet set = sets[0];
        const std::array buffers{
            VkDescriptorBufferInfo{*current, 0, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{*previous, 0, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{*uniform, 0, VK_WHOLE_SIZE},
        };
        const VkDescriptorImageInfo depth{sampler, depth_view, VK_IMAGE_LAYOUT_GENERAL};
        const std::array writes{
            VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 0, 0, 1,
                                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buffers[0]},
            VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 1, 0, 1,
                                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buffers[1]},
            VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 2, 0, 1,
                                 VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &buffers[2]},
            VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 3, 0, 1,
                                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depth},
        };
        logical.UpdateDescriptorSets(writes, {});
        const VkPushConstantRange depth_range{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(u32)};
        layout = logical.CreatePipelineLayout({
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1, .pSetLayouts = &raw_layout,
            .pushConstantRangeCount = 1, .pPushConstantRanges = &depth_range,
        });

        const VkAttachmentDescription attachment{
            .format = MOTION_FORMAT, .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_GENERAL, .finalLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        const VkAttachmentReference color{0, VK_IMAGE_LAYOUT_GENERAL};
        const VkSubpassDescription subpass{
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1, .pColorAttachments = &color,
        };
        render_pass = logical.CreateRenderPass({
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 1, .pAttachments = &attachment,
            .subpassCount = 1, .pSubpasses = &subpass,
        });
        framebuffer = logical.CreateFramebuffer({
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = *render_pass, .attachmentCount = 1, .pAttachments = &motion_view,
            .width = extent.width, .height = extent.height, .layers = 1,
        });
        pipeline = MakePipeline(logical);
    }

    /// Reuse the mesh and update the two poses only while the vertex and index data still match.
    bool Update(const Device& device, const OcarinaActorMesh& mesh,
                const ActorTransforms& transforms, VkImageView depth_view, VkSampler sampler) {
        const auto vertex_data = std::as_bytes(std::span{mesh.vertices});
        const auto index_data = std::as_bytes(std::span{mesh.indices});
        if (vertex_data.size() != vertex_bytes || mesh.indices.size() != index_count) {
            return false;
        }
        const auto upload = [](vk::Buffer& buffer, std::span<const std::byte> data) {
            std::memcpy(buffer.Mapped().data(), data.data(), data.size());
            buffer.Flush();
        };
        upload(vertices, vertex_data);
        upload(indices, index_data);
        upload(current, std::as_bytes(std::span{mesh.current_bones}));
        upload(previous, std::as_bytes(std::span{mesh.previous_bones}));
        upload(uniform, std::as_bytes(std::span{&transforms, 1}));
        const VkDescriptorImageInfo depth{sampler, depth_view, VK_IMAGE_LAYOUT_GENERAL};
        const VkWriteDescriptorSet write{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sets[0], 3, 0, 1,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depth,
        };
        device.GetLogical().UpdateDescriptorSets(write, {});
        return true;
    }

    void Draw(vk::CommandBuffer cmd) const {
        cmd.BeginRenderPass({
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = *render_pass,
            .framebuffer = *framebuffer,
            .renderArea = {{0, 0}, extent},
        }, VK_SUBPASS_CONTENTS_INLINE);
        cmd.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, *pipeline);
        cmd.BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, *layout, 0, sets[0], {});
        // Depth matches in the captured replay leveled off at this tolerance.
        constexpr u32 depth_tolerance = 16;
        cmd.PushConstants(*layout, VK_SHADER_STAGE_FRAGMENT_BIT, depth_tolerance);
        cmd.BindVertexBuffer(0, *vertices, 0);
        cmd.BindIndexBuffer(*indices, 0, VK_INDEX_TYPE_UINT16);
        cmd.DrawIndexed(index_count, 1, 0, 0, 0);
        cmd.EndRenderPass();
    }

private:
    vk::Pipeline MakePipeline(const vk::Device& logical) const {
        const auto vertex = logical.CreateShaderModule({
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = sizeof(OCARINA_ACTOR_MOTION_VERT_SPV), .pCode = std::data(OCARINA_ACTOR_MOTION_VERT_SPV),
        });
        const auto fragment = logical.CreateShaderModule({
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = sizeof(OCARINA_ACTOR_MOTION_FRAG_SPV), .pCode = std::data(OCARINA_ACTOR_MOTION_FRAG_SPV),
        });
        const std::array stages{
            VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                            nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, *vertex, "main"},
            VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                            nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, *fragment, "main"},
        };
        const VkVertexInputBindingDescription binding{0, 48, VK_VERTEX_INPUT_RATE_VERTEX};
        const std::array attributes{
            VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
            VkVertexInputAttributeDescription{4, 0, VK_FORMAT_R8G8B8A8_UNORM, 40},
            VkVertexInputAttributeDescription{5, 0, VK_FORMAT_R8G8B8A8_UNORM, 44},
            VkVertexInputAttributeDescription{6, 0, VK_FORMAT_R16G16B16A16_UINT, 12},
            VkVertexInputAttributeDescription{7, 0, VK_FORMAT_R16G16B16A16_UINT, 20},
        };
        const VkPipelineVertexInputStateCreateInfo vertex_input{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &binding,
            .vertexAttributeDescriptionCount = static_cast<u32>(attributes.size()),
            .pVertexAttributeDescriptions = attributes.data(),
        };
        const VkPipelineInputAssemblyStateCreateInfo assembly{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        };
        const VkViewport viewport{0, static_cast<f32>(extent.height), static_cast<f32>(extent.width),
                                  -static_cast<f32>(extent.height), 0, 1};
        const VkRect2D scissor{{0, 0}, extent};
        const VkPipelineViewportStateCreateInfo viewport_state{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1, .pViewports = &viewport, .scissorCount = 1, .pScissors = &scissor,
        };
        const VkPipelineRasterizationStateCreateInfo raster{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
            .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1,
        };
        const VkPipelineMultisampleStateCreateInfo samples{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        };
        const VkPipelineColorBlendAttachmentState write{
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT,
        };
        const VkPipelineColorBlendStateCreateInfo blend{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .attachmentCount = 1, .pAttachments = &write,
        };
        return logical.CreateGraphicsPipeline({
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = static_cast<u32>(stages.size()), .pStages = stages.data(),
            .pVertexInputState = &vertex_input, .pInputAssemblyState = &assembly,
            .pViewportState = &viewport_state, .pRasterizationState = &raster,
            .pMultisampleState = &samples, .pColorBlendState = &blend,
            .layout = *layout, .renderPass = *render_pass, .basePipelineIndex = -1,
        });
    }

    VkExtent2D extent;
    u32 index_count;
    size_t vertex_bytes;
    vk::Buffer vertices, indices, current, previous, uniform;
    vk::DescriptorSetLayout set_layout;
    vk::DescriptorPool descriptor_pool;
    vk::DescriptorSets sets;
    vk::PipelineLayout layout;
    vk::RenderPass render_pass;
    vk::Framebuffer framebuffer;
    vk::Pipeline pipeline;
};

class CameraMotionPass final : public ComputePass {
public:
    CameraMotionPass(const Device& device_, Scheduler& scheduler, DescriptorPool& pool)
        : ComputePass(device_, scheduler, pool, BINDINGS, TEMPLATES, BANK, PUSH_CONSTANTS,
                      OCARINA_CAMERA_MOTION_COMP_SPV) {}

    /// Fill camera motion from depth, then overwrite visible actor pixels with skinned motion.
    void Record(Scheduler& scheduler, const void* descriptors, const Matrix& transform,
                VkImage motion, VkExtent2D size,
                const ActorOverlay* overlay) {
        scheduler.Record([this, descriptors, transform, motion, size, overlay]
                         (vk::CommandBuffer cmdbuf) {
            const VkMemoryBarrier read_depth{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
            };
            const VkImageMemoryBarrier init_motion{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = motion,
                .subresourceRange = MOTION_RANGE,
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
                                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                   0, read_depth, {}, init_motion);
            const VkDescriptorSet set = descriptor_allocator.Commit();
            device.GetLogical().UpdateDescriptorSet(set, *descriptor_template, descriptors);
            cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
            cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *layout, 0, set, {});
            cmdbuf.PushConstants(*layout, VK_SHADER_STAGE_COMPUTE_BIT, transform);
            cmdbuf.Dispatch((size.width + 7) / 8, (size.height + 7) / 8, 1);
            const VkMemoryBarrier to_overlay{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                 VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                   VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, to_overlay);
            if (overlay) {
                overlay->Draw(cmdbuf);
            }
        });
    }
};

constexpr VkImageSubresourceRange COLOR_RANGE{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
constexpr VkImageSubresourceRange DEPTH_RANGE{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};

VkImageMemoryBarrier ProbeBarrier(VkImage image, VkAccessFlags src, VkAccessFlags dst,
                                  VkImageLayout before, VkImageLayout after) {
    return {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = src,
        .dstAccessMask = dst,
        .oldLayout = before,
        .newLayout = after,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = COLOR_RANGE,
    };
}

class OcarinaColorPass final : public ComputePass {
public:
    OcarinaColorPass(const Device& device_, Scheduler& scheduler, DescriptorPool& pool,
                      std::span<const u32> code = OCARINA_DLSS_COLOR_COMP_SPV)
        : ComputePass(device_, scheduler, pool, BINDINGS, TEMPLATES, BANK, {},
                      code) {}

    void Record(Scheduler& scheduler, const void* descriptors, VkImage input,
                VkExtent2D extent, const StreamlineDlssUpscaler* require_success = nullptr) {
        scheduler.Record([this, descriptors, input, extent, require_success](vk::CommandBuffer cmd) {
            if (require_success && require_success->Failed()) {
                return;
            }
            const VkImageMemoryBarrier prepare = ProbeBarrier(input, VK_ACCESS_MEMORY_READ_BIT |
                                                                VK_ACCESS_MEMORY_WRITE_BIT,
                                                                VK_ACCESS_SHADER_WRITE_BIT,
                             VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
            const VkMemoryBarrier sample_color{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            };
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                0, sample_color, {}, prepare);
            const VkDescriptorSet set = descriptor_allocator.Commit();
            device.GetLogical().UpdateDescriptorSet(set, *descriptor_template, descriptors);
            cmd.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
            cmd.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *layout, 0, set, {});
            cmd.Dispatch((extent.width + 7) / 8, (extent.height + 7) / 8, 1);
        });
    }
    void RecordOutput(Scheduler& scheduler, const void* descriptors,
                      const StreamlineDlssUpscaler& upscaler, VkExtent2D extent, u64 frame,
                      bool reset) {
        scheduler.Record([this, descriptors, &upscaler, extent, frame, reset](vk::CommandBuffer cmd) {
            if (upscaler.Failed()) {
                return;
            }
            const VkMemoryBarrier before{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            };
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                0, before);
            const VkDescriptorSet set = descriptor_allocator.Commit();
            device.GetLogical().UpdateDescriptorSet(set, *descriptor_template, descriptors);
            cmd.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
            cmd.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *layout, 0, set, {});
            cmd.Dispatch((extent.width + 7) / 8, (extent.height + 7) / 8, 1);
            const VkMemoryBarrier after{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            };
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                0, after);
            if (reset) {
                LOG_DEBUG(Render_Vulkan, "Ocarina live DLSS replaced native frame={} history_reset={}", frame, reset);
            }
        });
    }
};

class OcarinaCompositePass final : public ComputePass {
    static constexpr auto bindings = [] {
        std::array<VkDescriptorSetLayoutBinding, 4> result{};
        for (u32 i = 0; i < 4; ++i) {
            result[i] = {i, i == 3 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE :
                                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                         1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        }
        return result;
    }();
    static constexpr auto templates = [] {
        std::array<VkDescriptorUpdateTemplateEntry, 4> result{};
        for (u32 i = 0; i < 4; ++i) {
            result[i] = {i, 0, 1, bindings[i].descriptorType,
                         i * sizeof(DescriptorUpdateEntry), sizeof(DescriptorUpdateEntry)};
        }
        return result;
    }();
    static constexpr DescriptorBankInfo bank{.textures = 3, .images = 1, .score = 4};
    static constexpr std::array push{VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(u32)}};

public:
    OcarinaCompositePass(const Device& device_, Scheduler& scheduler, DescriptorPool& pool)
        : ComputePass(device_, scheduler, pool, bindings, templates, bank, push,
                      OCARINA_DLSS_4K_COMPOSITE_COMP_SPV) {}

    void Record(Scheduler& scheduler, const void* descriptors, VkImage image,
                const StreamlineDlssUpscaler& upscaler, u64 frame) {
        scheduler.Record([this, descriptors, image, &upscaler, frame](vk::CommandBuffer cmd) {
            const auto output = ProbeBarrier(image, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                                             VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                                             VK_IMAGE_LAYOUT_GENERAL);
            const VkMemoryBarrier inputs{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            };
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                0, inputs, {}, output);
            const VkDescriptorSet set = descriptor_allocator.Commit();
            device.GetLogical().UpdateDescriptorSet(set, *descriptor_template, descriptors);
            cmd.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
            cmd.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *layout, 0, set, {});
            const u32 enabled = !upscaler.Failed();
            cmd.PushConstants(*layout, VK_SHADER_STAGE_COMPUTE_BIT, enabled);
            cmd.Dispatch(3840 / 8, 2160 / 8, 1);
            const VkMemoryBarrier complete{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            };
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                0, complete);
            if (enabled && (!logged || frame % 120 == 0)) {
                LOG_INFO(Render_Vulkan, "Ocarina DLSS presented 3840x2160 frame={}", frame);
                logged = true;
            }
        });
    }

private:
    bool logged{};
};

constexpr VkFormat LIVE_COLOR_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;

vk::Image CreateOcarinaImage(MemoryAllocator& allocator, VkExtent2D extent, VkFormat format,
                            VkImageUsageFlags usage) {
    return allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = {extent.width, extent.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    });
}

vk::ImageView CreateOcarinaView(const Device& device, VkImage image, VkFormat format) {
    return device.GetLogical().CreateImageView(VkImageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .subresourceRange = COLOR_RANGE,
    });
}

struct OcarinaFrameSlot {
    OcarinaFrameSlot(const Device& device, MemoryAllocator& allocator, VkExtent2D extent_)
        : extent{extent_}, motion{CreateOcarinaImage(
                              allocator, extent, MOTION_FORMAT, VK_IMAGE_USAGE_STORAGE_BIT |
                                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)},
          motion_view{CreateOcarinaView(device, *motion, MOTION_FORMAT)},
          color{CreateOcarinaImage(allocator, extent, LIVE_COLOR_FORMAT,
                                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)},
          color_view{CreateOcarinaView(device, *color, LIVE_COLOR_FORMAT)} {}

    VkExtent2D extent;
    u64 tick{};
    vk::Image motion;
    vk::ImageView motion_view;
    vk::Image color;
    vk::ImageView color_view;
    std::unique_ptr<ActorOverlay> overlay;
};

} // namespace

struct OcarinaLiveDlss::Impl {
    Impl(const Device& device_, Scheduler& scheduler_, DescriptorPool& pool,
         ComputePassDescriptorQueue& queue_, MemoryAllocator& allocator_)
        : device{device_}, scheduler{scheduler_}, queue{queue_}, allocator{allocator_}, pool{pool},
          upscaler{device, scheduler, allocator}, motion_pass{device, scheduler, pool},
          color_pass{device, scheduler, pool},
          output_pass{device, scheduler, pool, OCARINA_DLSS_OUTPUT_COMP_SPV} {}

    const Device& device;
    Scheduler& scheduler;
    ComputePassDescriptorQueue& queue;
    MemoryAllocator& allocator;
    DescriptorPool& pool;
    StreamlineDlssUpscaler upscaler;
    CameraMotionPass motion_pass;
    OcarinaColorPass color_pass;
    OcarinaColorPass output_pass;
    std::array<std::unique_ptr<OcarinaFrameSlot>, 4> slots;
    std::optional<u64> last_frame;
    Matrix last_view_projection{};
    vk::Image native_scene;
    vk::ImageView native_scene_view;
    vk::Image composite;
    vk::ImageView composite_view;
    std::unique_ptr<OcarinaColorPass> reference_pass;
    std::unique_ptr<OcarinaCompositePass> composite_pass;
    vk::Sampler linear_sampler;
    std::optional<u64> composited_frame;

    VkSampler LinearSampler() {
        if (!linear_sampler) {
            linear_sampler = device.GetLogical().CreateSampler(VkSamplerCreateInfo{
                .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                .magFilter = VK_FILTER_LINEAR,
                .minFilter = VK_FILTER_LINEAR,
                .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                .maxAnisotropy = 1.0f,
            });
        }
        return *linear_sampler;
    }
};

OcarinaLiveDlss::OcarinaLiveDlss(const Device& device, Scheduler& scheduler, DescriptorPool& pool,
                                 ComputePassDescriptorQueue& queue, MemoryAllocator& allocator)
    : impl{std::make_unique<Impl>(device, scheduler, pool, queue, allocator)} {}

OcarinaLiveDlss::~OcarinaLiveDlss() = default;

bool OcarinaLiveDlss::EvaluationFailed() const {
    return impl->upscaler.Failed();
}

void OcarinaLiveDlss::SaveSceneReference(VkImageView color) {
    auto& state = *impl;
    constexpr VkExtent2D extent{2560, 1440};
    if (!state.native_scene_view) {
        constexpr VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
        state.native_scene = CreateOcarinaImage(state.allocator, extent, format,
                                               VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        state.native_scene_view = CreateOcarinaView(state.device, *state.native_scene, format);
        state.reference_pass = std::make_unique<OcarinaColorPass>(
            state.device, state.scheduler, state.pool, OCARINA_DLSS_4K_REFERENCE_COMP_SPV);
    }
    state.queue.Acquire(state.scheduler, 2);
    state.queue.AddSampledImage(color, state.upscaler.PointSampler());
    state.queue.AddImage(*state.native_scene_view);
    state.reference_pass->Record(state.scheduler, state.queue.UpdateData(), *state.native_scene,
                                 extent, &state.upscaler);
    state.scheduler.InvalidateState();
}

void OcarinaLiveDlss::Composite4k(VkImageView native_composite, u64 frame) {
    auto& state = *impl;
    if (state.composited_frame == frame) {
        return;
    }
    if (!state.composite_pass) {
        constexpr VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
        state.composite = CreateOcarinaImage(state.allocator, {3840, 2160}, format,
                                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        state.composite_view = CreateOcarinaView(state.device, *state.composite, format);
        state.composite_pass = std::make_unique<OcarinaCompositePass>(
            state.device, state.scheduler, state.pool);
    }
    state.scheduler.RequestOutsideRenderPassOperationContext();
    state.queue.Acquire(state.scheduler, 4);
    const auto sampler = state.LinearSampler();
    state.queue.AddSampledImage(state.upscaler.OutputView(), sampler);
    state.queue.AddSampledImage(*state.native_scene_view, sampler);
    state.queue.AddSampledImage(native_composite, sampler);
    state.queue.AddImage(*state.composite_view);
    state.composite_pass->Record(state.scheduler, state.queue.UpdateData(), *state.composite,
                                 state.upscaler, frame);
    state.scheduler.InvalidateState();
    state.composited_frame = frame;
}

VkImage OcarinaLiveDlss::CompositedImage() const {
    return *impl->composite;
}

VkImageView OcarinaLiveDlss::CompositedView() const {
    return *impl->composite_view;
}

bool OcarinaLiveDlss::Record(VkImageView color, VkImageView depth, VkImageView target, VkExtent2D extent,
                            VkExtent2D output_extent, VkExtent2D target_extent,
                            const Matrix& current, const Matrix& previous,
                            const OcarinaActorMesh& mesh, VkImage depth_image, u64 frame) {
    auto& state = *impl;
    if (state.last_frame == frame || state.upscaler.Failed()) {
        return false;
    }
    const auto transform = ClipToPreviousClip(current, previous);
    if (!transform) {
        return false;
    }
    state.scheduler.GetMasterSemaphore().Refresh();
    const auto slot = std::ranges::find_if(state.slots, [&](const auto& entry) {
        return !entry || state.scheduler.IsFree(entry->tick);
    });
    if (slot == state.slots.end()) {
        return false;
    }
    if (!*slot || (*slot)->extent.width != extent.width || (*slot)->extent.height != extent.height) {
        *slot = std::make_unique<OcarinaFrameSlot>(state.device, state.allocator, extent);
    }
    auto& resources = **slot;
    const VkSampler sampler = state.upscaler.PointSampler();
    const ActorTransforms actor_transforms{
        current, previous, mesh.current_origin, mesh.previous_origin,
        {static_cast<f32>(extent.width), static_cast<f32>(extent.height), 0, 0},
    };
    // Without Link's pose (dialogue, cutscenes, first person) the motion comes from the camera alone.
    if (mesh.indices.empty()) {
        resources.overlay.reset();
    } else if (!resources.overlay ||
               !resources.overlay->Update(state.device, mesh, actor_transforms, depth, sampler)) {
        resources.overlay = std::make_unique<ActorOverlay>(state.device, state.allocator, mesh,
                                                            actor_transforms, depth, sampler,
                                                            *resources.motion_view, extent);
    }

    state.scheduler.RequestOutsideRenderPassOperationContext();
    state.queue.Acquire(state.scheduler, 2);
    state.queue.AddSampledImage(depth, sampler);
    state.queue.AddImage(*resources.motion_view);
    resources.tick = state.scheduler.CurrentTick();
    state.motion_pass.Record(state.scheduler, state.queue.UpdateData(), *transform,
                             *resources.motion, extent, resources.overlay.get());
    state.queue.Acquire(state.scheduler, 2);
    state.queue.AddSampledImage(color, sampler);
    state.queue.AddImage(*resources.color_view);
    state.color_pass.Record(state.scheduler, state.queue.UpdateData(), *resources.color, extent);

    const bool reset = !state.last_frame || frame != *state.last_frame + 1 ||
                       !std::ranges::equal(previous, state.last_view_projection,
                                           [](f32 a, f32 b) { return std::abs(a - b) < 0.001f; });
    if (!state.upscaler.Upscale({
            .color = {*resources.color_view, *resources.color, LIVE_COLOR_FORMAT, COLOR_RANGE,
                      VK_IMAGE_ASPECT_COLOR_BIT},
            .depth = {depth, depth_image, VK_FORMAT_D32_SFLOAT, DEPTH_RANGE, VK_IMAGE_ASPECT_DEPTH_BIT},
            .motion_vectors = {*resources.motion_view, *resources.motion, MOTION_FORMAT,
                               COLOR_RANGE, VK_IMAGE_ASPECT_COLOR_BIT},
            .render_size = extent,
            .upscale_size = output_extent,
            .jitter = {0.0f, 0.0f},
            .motion_vector_scale = {1.0f, 1.0f},
            .reset = reset,
            .depth_inverted = false,
            .hdr = false,
        })) {
        state.scheduler.InvalidateState();
        return false;
    }
    // A resize can submit the inputs before recording evaluation, so keep the later tick.
    resources.tick = state.scheduler.CurrentTick();
    if (target) {
        state.queue.Acquire(state.scheduler, 2);
        const bool same_size = output_extent.width == target_extent.width &&
                               output_extent.height == target_extent.height;
        const auto output_sampler = same_size ? sampler : state.LinearSampler();
        state.queue.AddSampledImage(state.upscaler.OutputView(), output_sampler);
        state.queue.AddImage(target);
        state.output_pass.RecordOutput(state.scheduler, state.queue.UpdateData(), state.upscaler,
                                        target_extent, frame, reset);
    }
    state.scheduler.InvalidateState();
    state.last_frame = frame;
    state.last_view_projection = current;
    return true;
}
} // namespace Vulkan
#endif
