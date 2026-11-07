// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include <boost/container/small_vector.hpp>

#include "common/common_types.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/nx2_bindmemo_stats.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/types.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

using Shader::Backend::SPIRV::NUM_TEXTURE_AND_IMAGE_SCALING_WORDS;

[[nodiscard]] inline std::optional<PixelFormat> PixelFormatFromImageFormat(
    Shader::ImageFormat format) {
    switch (format) {
    case Shader::ImageFormat::Typeless:
        return std::nullopt;
    case Shader::ImageFormat::R8_UINT:
        return PixelFormat::R8_UINT;
    case Shader::ImageFormat::R8_SINT:
        return PixelFormat::R8_SINT;
    case Shader::ImageFormat::R16_UINT:
        return PixelFormat::R16_UINT;
    case Shader::ImageFormat::R16_SINT:
        return PixelFormat::R16_SINT;
    case Shader::ImageFormat::R32_UINT:
        return PixelFormat::R32_UINT;
    case Shader::ImageFormat::R32G32_UINT:
        return PixelFormat::R32G32_UINT;
    case Shader::ImageFormat::R32G32B32A32_UINT:
        return PixelFormat::R32G32B32A32_UINT;
    }
    return std::nullopt;
}

[[nodiscard]] inline VkDeviceSize DescriptorSizeForType(const Device& device,
                                                        VkDescriptorType type) {
    const auto& props = device.DescriptorBufferProperties();
    const bool robust = device.IsRobustBufferAccessEnabled();
    switch (type) {
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        return robust ? props.robustUniformBufferDescriptorSize : props.uniformBufferDescriptorSize;
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        return robust ? props.robustStorageBufferDescriptorSize : props.storageBufferDescriptorSize;
    case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
        return robust ? props.robustUniformTexelBufferDescriptorSize
                      : props.uniformTexelBufferDescriptorSize;
    case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
        return robust ? props.robustStorageTexelBufferDescriptorSize
                      : props.storageTexelBufferDescriptorSize;
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        return props.combinedImageSamplerDescriptorSize;
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        return props.storageImageDescriptorSize;
    default:
        return 0;
    }
}

struct DescriptorBufferBinding {
    VkDescriptorType type;
    u32 count;
    VkDeviceSize offset;
    VkDeviceSize stride;
};

struct DescriptorBufferLayout {
    VkDeviceSize size{};
    boost::container::small_vector<DescriptorBufferBinding, 32> bindings;

    [[nodiscard]] bool Empty() const noexcept {
        return bindings.empty();
    }
};

inline void WriteDescriptorBuffer(const Device& device, const DescriptorBufferLayout& layout,
                                  const DescriptorUpdateEntry* payload, u8* host) {
    const vk::Device& dev = device.GetLogical();
    for (const DescriptorBufferBinding& binding : layout.bindings) {
        for (u32 index = 0; index < binding.count; ++index) {
            const DescriptorUpdateEntry& entry = *(payload++);
            const VkDescriptorAddressInfoEXT address_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
                .pNext = nullptr,
                .address = entry.address.address,
                .range = entry.address.range,
                .format = entry.address.format,
            };
            VkDescriptorGetInfoEXT get_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
                .pNext = nullptr,
                .type = binding.type,
                .data{},
            };
            switch (binding.type) {
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                get_info.data.pUniformBuffer = &address_info;
                break;
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                get_info.data.pStorageBuffer = &address_info;
                break;
            case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
                get_info.data.pUniformTexelBuffer = &address_info;
                break;
            case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                get_info.data.pStorageTexelBuffer = &address_info;
                break;
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                get_info.data.pCombinedImageSampler = &entry.image;
                break;
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                get_info.data.pStorageImage = &entry.image;
                break;
            default:
                continue;
            }
            dev.GetDescriptorEXT(get_info, binding.stride,
                                 host + binding.offset + index * binding.stride);
        }
    }
}

[[nodiscard]] inline u32 NumDescriptorEntries(const Shader::Info& info) {
    return Shader::NumDescriptors(info.constant_buffer_descriptors) +
           Shader::NumDescriptors(info.storage_buffers_descriptors) +
           Shader::NumDescriptors(info.texture_buffer_descriptors) +
           Shader::NumDescriptors(info.image_buffer_descriptors) +
           Shader::NumDescriptors(info.texture_descriptors) +
           Shader::NumDescriptors(info.image_descriptors);
}

class DescriptorLayoutBuilder {
public:
    DescriptorLayoutBuilder(const Device& device_) : device{&device_} {}

    bool CanUsePushDescriptor() const noexcept {
        if (!device->IsKhrPushDescriptorSupported() ||
            num_descriptors > device->MaxPushDescriptors()) {
            return false;
        }
        return !device->IsExtDescriptorBufferSupported() ||
               device->DescriptorBufferProperties().bufferlessPushDescriptors;
    }

    bool CanUseDescriptorBuffer() const noexcept {
        const auto& props = device->DescriptorBufferProperties();
        if (!device->IsExtDescriptorBufferSupported() || bindings.empty() ||
            !props.combinedImageSamplerDescriptorSingleArray) {
            return false;
        }
        return !props.bufferlessPushDescriptors || !CanUsePushDescriptor();
    }

    DescriptorBufferLayout MakeDescriptorBufferLayout(VkDescriptorSetLayout layout) const {
        DescriptorBufferLayout result;
        if (!layout) {
            return result;
        }
        const vk::Device& dev = device->GetLogical();
        result.size = dev.GetDescriptorSetLayoutSizeEXT(layout);
        result.bindings.reserve(bindings.size());
        for (const VkDescriptorSetLayoutBinding& entry : bindings) {
            result.bindings.push_back(DescriptorBufferBinding{
                .type = entry.descriptorType,
                .count = entry.descriptorCount,
                .offset = dev.GetDescriptorSetLayoutBindingOffsetEXT(layout, entry.binding),
                .stride = DescriptorSizeForType(*device, entry.descriptorType),
            });
        }
        return result;
    }

    vk::DescriptorSetLayout CreateDescriptorSetLayout(bool use_push_descriptor,
                                                      bool use_descriptor_buffer = false) const {
        if (bindings.empty()) {
            return nullptr;
        }
        VkDescriptorSetLayoutCreateFlags flags = 0;
        if (use_push_descriptor) {
            flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        }
        if (use_descriptor_buffer) {
            flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
        }
        boost::container::small_vector<VkDescriptorBindingFlags, 32> binding_flags;
        VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags_ci{};
        const void* pnext = nullptr;
        if (!use_push_descriptor && device->IsDescriptorBindingPartiallyBoundSupported()) {
            binding_flags.assign(bindings.size(), VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
            binding_flags_ci = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
                .pNext = nullptr,
                .bindingCount = static_cast<u32>(binding_flags.size()),
                .pBindingFlags = binding_flags.data(),
            };
            pnext = &binding_flags_ci;
        }
        return device->GetLogical().CreateDescriptorSetLayout({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = pnext,
            .flags = flags,
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        });
    }

    vk::DescriptorUpdateTemplate CreateTemplate(VkDescriptorSetLayout descriptor_set_layout,
                                                VkPipelineLayout pipeline_layout,
                                                bool use_push_descriptor) const {
        if (entries.empty()) {
            return nullptr;
        }
        const VkDescriptorUpdateTemplateType type =
            use_push_descriptor ? VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR
                                : VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET;
        return device->GetLogical().CreateDescriptorUpdateTemplate({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .descriptorUpdateEntryCount = static_cast<u32>(entries.size()),
            .pDescriptorUpdateEntries = entries.data(),
            .templateType = type,
            .descriptorSetLayout = descriptor_set_layout,
            .pipelineBindPoint =
                is_compute ? VK_PIPELINE_BIND_POINT_COMPUTE : VK_PIPELINE_BIND_POINT_GRAPHICS,
            .pipelineLayout = pipeline_layout,
            .set = 0,
        });
    }

    vk::PipelineLayout CreatePipelineLayout(VkDescriptorSetLayout descriptor_set_layout) const {
        using Shader::Backend::SPIRV::RenderAreaLayout;
        using Shader::Backend::SPIRV::RescalingLayout;
        const u32 size_offset = is_compute ? sizeof(RescalingLayout::down_factor) : 0u;
        const VkPushConstantRange range{
            .stageFlags = static_cast<VkShaderStageFlags>(
                is_compute ? VK_SHADER_STAGE_COMPUTE_BIT
                : mesh_emulation
                    ? VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT
                    : (uses_mesh ? VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_TASK_BIT_EXT |
                                       VK_SHADER_STAGE_MESH_BIT_EXT
                                 : VK_SHADER_STAGE_ALL_GRAPHICS)),
            .offset = 0,
            .size = static_cast<u32>(sizeof(RescalingLayout)) - size_offset +
                    static_cast<u32>(sizeof(RenderAreaLayout)),
        };
        return device->GetLogical().CreatePipelineLayout({
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = descriptor_set_layout ? 1U : 0U,
            .pSetLayouts = bindings.empty() ? nullptr : &descriptor_set_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &range,
        });
    }

    /// Mesh emulation: a graphics pipeline layout whose mesh program runs as a compute shader
    /// (its bindings carry VK_SHADER_STAGE_COMPUTE_BIT and so does the push constant range).
    void EnableMeshEmulation() noexcept {
        mesh_emulation = true;
    }

    /// Mesh emulation: the output storage buffer the compute shader writes and the vertex
    /// shader reads, at an explicit binding (after every program's bindings).
    void AddMeshEmulationOutput(u32 output_binding) {
        bindings.push_back({
            .binding = output_binding,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        });
        entries.push_back({
            .dstBinding = output_binding,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .offset = offset,
            .stride = sizeof(DescriptorUpdateEntry),
        });
        binding = (std::max)(binding, output_binding + 1);
        ++num_descriptors;
        offset += sizeof(DescriptorUpdateEntry);
    }

    void Add(const Shader::Info& info, VkShaderStageFlags stage) {
        is_compute |= (stage & VK_SHADER_STAGE_COMPUTE_BIT) != 0 && !mesh_emulation;
        uses_mesh |= (stage & (VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT)) != 0;

        Add(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, stage, info.constant_buffer_descriptors);
        Add(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stage, info.storage_buffers_descriptors);
        Add(VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, stage, info.texture_buffer_descriptors);
        Add(VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, stage, info.image_buffer_descriptors);
        Add(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage, info.texture_descriptors);
        Add(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage, info.image_descriptors);
    }

private:
    template <typename Descriptors>
    void Add(VkDescriptorType type, VkShaderStageFlags stage, const Descriptors& descriptors) {
        const size_t num{descriptors.size()};
        for (size_t i = 0; i < num; ++i) {
            bindings.push_back({
                .binding = binding,
                .descriptorType = type,
                .descriptorCount = descriptors[i].count,
                .stageFlags = stage,
                .pImmutableSamplers = nullptr,
            });
            entries.push_back({
                .dstBinding = binding,
                .dstArrayElement = 0,
                .descriptorCount = descriptors[i].count,
                .descriptorType = type,
                .offset = offset,
                .stride = sizeof(DescriptorUpdateEntry),
            });
            ++binding;
            num_descriptors += descriptors[i].count;
            offset += sizeof(DescriptorUpdateEntry) * descriptors[i].count;
        }
    }

    const Device* device{};
    bool is_compute{};
    bool uses_mesh{};
    bool mesh_emulation{};
    boost::container::small_vector<VkDescriptorSetLayoutBinding, 32> bindings;
    boost::container::small_vector<VkDescriptorUpdateTemplateEntry, 32> entries;
    u32 binding{};
    u32 num_descriptors{};
    size_t offset{};
};

class RescalingPushConstant {
public:
    explicit RescalingPushConstant() noexcept {}

    void PushTexture(bool is_rescaled) noexcept {
        *texture_ptr |= is_rescaled ? texture_bit : 0u;
        texture_bit <<= 1u;
        if (texture_bit == 0u) {
            texture_bit = 1u;
            ++texture_ptr;
        }
    }

    void PushImage(bool is_rescaled) noexcept {
        *image_ptr |= is_rescaled ? image_bit : 0u;
        image_bit <<= 1u;
        if (image_bit == 0u) {
            image_bit = 1u;
            ++image_ptr;
        }
    }

    const std::array<u32, NUM_TEXTURE_AND_IMAGE_SCALING_WORDS>& Data() const noexcept {
        return words;
    }

private:
    std::array<u32, NUM_TEXTURE_AND_IMAGE_SCALING_WORDS> words{};
    u32* texture_ptr{words.data()};
    u32* image_ptr{words.data() + Shader::Backend::SPIRV::NUM_TEXTURE_SCALING_WORDS};
    u32 texture_bit{1u};
    u32 image_bit{1u};
};

class RenderAreaPushConstant {
public:
    bool uses_render_area{};
    std::array<f32, 4> words{};
};

/// EDEN_NX2_BINDMEMO: what the descriptor made for a texture or image binding depends on besides
/// its view and sampler, as TextureCache::FindBinding keys it.
namespace BindingKind {
constexpr u32 TextureBuffer = 1;
constexpr u32 ImageBuffer = 2;
constexpr u32 Texture = 3;
constexpr u32 Image = 4;
constexpr u32 ClassMask = 7;
} // namespace BindingKind

/// Appends the kind of each texture/image binding of a stage, in the order its views are read.
inline void AppendBindingKinds(const Shader::Info& info, bool compute, std::vector<u32>& kinds) {
    const u32 base = compute ? TextureCache::BINDING_KIND_COMPUTE : 0U;
    for (const auto& desc : info.texture_buffer_descriptors) {
        kinds.insert(kinds.end(), desc.count, base | BindingKind::TextureBuffer);
    }
    for (const auto& desc : info.image_buffer_descriptors) {
        kinds.insert(kinds.end(), desc.count, base | BindingKind::ImageBuffer);
    }
    for (const auto& desc : info.texture_descriptors) {
        const u32 kind = base | BindingKind::Texture | (static_cast<u32>(desc.type) << 3) |
                         (desc.is_depth ? 1U << 7 : 0U);
        kinds.insert(kinds.end(), desc.count, kind);
    }
    for (const auto& desc : info.image_descriptors) {
        const u32 kind = base | BindingKind::Image | (static_cast<u32>(desc.type) << 3) |
                         (static_cast<u32>(desc.format) << 8) |
                         (desc.is_byte_addressed ? 1U << 12 : 0U) |
                         (desc.is_written ? 1U << 13 : 0U);
        kinds.insert(kinds.end(), desc.count, kind);
    }
}

/// A binding FindImageBindings found no memo for: its view index and, for a texture, its
/// sampler index (~0U otherwise).
struct BindingMiss {
    u32 view;
    u32 sampler;
};
using BindingMisses = boost::container::small_vector<BindingMiss, 32>;

/// Looks every binding up in the texture cache's memo. A hit fills its view's id, its sampler
/// (textures) and memos[index]; a miss leaves memos[index] null and joins `misses`. The caller
/// checked that `kinds` has an entry per view and `sampler_handles` one per texture.
/// `aliased_hit`: a hit's image has aliases.
inline void FindImageBindings(TextureCache& texture_cache,
                              const TextureCache::BindingLookup& lookup,
                              std::span<const u32> kinds,
                              std::span<VideoCommon::ImageViewInOut> views,
                              std::span<const u32> sampler_handles,
                              VideoCommon::SamplerId* samplers, TextureCache::BindingMemo** memos,
                              BindingMisses& misses, bool& aliased_hit) {
    u32 sampler_index = 0;
    aliased_hit = false;
    // EDEN_NX2_BINDMEMO_BATCH: every binding's tag line, then its memo and guest entry lines are
    // asked for before the first is checked, so the misses of a draw's bindings overlap.
    const bool batch = VideoCommon::Nx2BindMemo::BatchEnabled();
    boost::container::small_vector<TextureCache::BindingProbe, 32> probes;
    boost::container::small_vector<u32, 32> tsc_indices;
    if (batch) {
        probes.resize(views.size());
        tsc_indices.resize(views.size());
        for (size_t index = 0; index < views.size(); ++index) {
            const u32 kind = kinds[index];
            const bool is_texture = (kind & BindingKind::ClassMask) == BindingKind::Texture;
            const u32 tsc_index = is_texture ? sampler_handles[sampler_index++] : ~0U;
            tsc_indices[index] = tsc_index;
            probes[index] = texture_cache.ProbeBinding(views[index].index, tsc_index, kind);
        }
        for (size_t index = 0; index < views.size(); ++index) {
            texture_cache.PrefetchBinding(lookup, probes[index], views[index].index,
                                          tsc_indices[index]);
        }
        sampler_index = 0;
    }
    for (size_t index = 0; index < views.size(); ++index) {
        const u32 kind = kinds[index];
        const bool is_texture = (kind & BindingKind::ClassMask) == BindingKind::Texture;
        const u32 tsc_index = is_texture ? sampler_handles[sampler_index] : ~0U;
        TextureCache::BindingMemo* const memo =
            batch ? texture_cache.FindBinding(lookup, views[index].index, tsc_index, kind,
                                              probes[index])
                  : texture_cache.FindBinding(lookup, views[index].index, tsc_index, kind);
        memos[index] = memo;
        if (memo) {
            views[index].id = memo->view;
            aliased_hit |= memo->aliased != 0;
            if (views[index].blacklist) {
                texture_cache.NoteBlacklistedBinding(memo->image);
            }
            if (is_texture) {
                samplers[sampler_index] = memo->sampler;
            }
        } else {
            misses.push_back({static_cast<u32>(index), is_texture ? sampler_index : ~0U});
        }
        if (is_texture) {
            ++sampler_index;
        }
    }
}

/// Resolves the bindings FindImageBindings missed, as GetSamplerId and FillImageViews do: only
/// those (EDEN_NX2_BINDMEMO_PARTIAL) unless resolving them changed anything in the texture cache
/// (an image made, joined, deleted or rescaled: a hit may be stale), else every binding, which
/// leaves no memo in use. Every binding too when a hit's image has aliases: preparing a missed
/// image can make an alias of it newer, and the original order would then copy from it.
inline void ResolveMissedBindings(TextureCache& texture_cache,
                                  const TextureCache::BindingLookup& lookup,
                                  std::span<VideoCommon::ImageViewInOut> views,
                                  std::span<const u32> sampler_handles,
                                  std::span<VideoCommon::SamplerId> samplers,
                                  TextureCache::BindingMemo** memos,
                                  std::span<const BindingMiss> misses, bool aliased_hit,
                                  bool blacklist) {
    using VideoCommon::Nx2BindMemo::Note;
    using Stat = VideoCommon::Nx2BindMemo::Stat;
    const bool compute = lookup.compute;
    Note(Stat::Draws);
    if (misses.empty()) {
        Note(Stat::DrawsAllHit);
        return;
    }
    const bool all_missed = misses.size() == views.size();
    Note(all_missed ? Stat::DrawsAllMiss : Stat::DrawsPartial);
    if (!all_missed && aliased_hit) {
        Note(Stat::DrawsAliasedHit);
    } else if (!all_missed && VideoCommon::Nx2BindMemo::PartialEnabled()) {
        for (const BindingMiss& miss : misses) {
            if (miss.sampler != ~0U) {
                samplers[miss.sampler] =
                    texture_cache.GetSamplerId(sampler_handles[miss.sampler], compute);
            }
        }
        boost::container::small_vector<VideoCommon::ImageViewInOut, 32> subset;
        subset.reserve(misses.size());
        for (const BindingMiss& miss : misses) {
            subset.push_back(views[miss.view]);
        }
        texture_cache.FillImageViews(std::span(subset.data(), subset.size()), compute, blacklist);
        if (texture_cache.BindingClock() == lookup.clock) {
            for (size_t index = 0; index < misses.size(); ++index) {
                views[misses[index].view].id = subset[index].id;
            }
            Note(Stat::BindingsResolved, misses.size());
            return;
        }
        Note(Stat::DrawsResolvedAgain);
    }
    for (size_t index = 0; index < sampler_handles.size(); ++index) {
        samplers[index] = texture_cache.GetSamplerId(sampler_handles[index], compute);
    }
    texture_cache.FillImageViews(views, compute, blacklist);
    std::fill_n(memos, views.size(), nullptr);
    Note(Stat::BindingsResolved, views.size());
}

/// What PushImageDescriptors takes from the memo and where it leaves the descriptor and
/// rescaling bit of each binding it resolves, by view index, for RecordImageBindings.
struct ImageBindingRecord {
    const VideoCommon::ImageViewInOut* views_begin;
    TextureCache::BindingMemo* const* memos; ///< per view: the memo to push, or nullptr
    DescriptorUpdateEntry* descriptors;      ///< per view; nullptr: nothing is resolved
    u8* rescaled;
};

static_assert(sizeof(DescriptorUpdateEntry) <= sizeof(TextureCache::BindingMemo::payload));

/// Remembers the bindings of a draw or dispatch that weren't in the memo.
inline void RecordImageBindings(TextureCache& texture_cache,
                                const TextureCache::BindingLookup& lookup,
                                std::span<const u32> kinds,
                                std::span<const VideoCommon::ImageViewInOut> views,
                                std::span<const u32> sampler_handles,
                                std::span<const VideoCommon::SamplerId> samplers,
                                TextureCache::BindingMemo* const* memos,
                                const DescriptorUpdateEntry* descriptors, const u8* rescaled) {
    if (kinds.size() != views.size()) {
        return;
    }
    size_t sampler_index = 0;
    for (size_t index = 0; index < views.size(); ++index) {
        const u32 kind = kinds[index];
        const u32 kind_class = kind & BindingKind::ClassMask;
        const bool is_texture = kind_class == BindingKind::Texture;
        const size_t texture_index = sampler_index;
        if (is_texture) {
            ++sampler_index;
        }
        if (memos[index] != nullptr) {
            continue;
        }
        const bool has_descriptor = is_texture || kind_class == BindingKind::Image;
        u32 tsc_index = ~0U;
        VideoCommon::SamplerId sampler{};
        if (is_texture) {
            if (texture_index >= sampler_handles.size() || texture_index >= samplers.size()) {
                return;
            }
            tsc_index = sampler_handles[texture_index];
            sampler = samplers[texture_index];
        }
        texture_cache.RecordBinding(lookup, views[index].index, tsc_index, kind, views[index].id,
                                    sampler, has_descriptor ? &descriptors[index] : nullptr,
                                    has_descriptor ? sizeof(DescriptorUpdateEntry) : 0,
                                    rescaled[index] != 0);
    }
}

inline void PushImageDescriptors(TextureCache& texture_cache,
                                 GuestDescriptorQueue& guest_descriptor_queue,
                                 const Shader::Info& info, RescalingPushConstant& rescaling,
                                 const VideoCommon::SamplerId*& samplers,
                                 const VideoCommon::ImageViewInOut*& views,
                                 const ImageBindingRecord* record = nullptr) {
    const u32 num_texture_buffers = Shader::NumDescriptors(info.texture_buffer_descriptors);
    const u32 num_image_buffers = Shader::NumDescriptors(info.image_buffer_descriptors);
    views += num_texture_buffers;
    views += num_image_buffers;
    for (const auto& desc : info.texture_descriptors) {
        bool is_rescaled{};
        for (u32 index = 0; index < desc.count; ++index) {
            const VideoCommon::ImageViewInOut* const view{views++};
            const VideoCommon::SamplerId sampler_id{*(samplers++)};
            size_t view_index = 0;
            if (record) {
                view_index = static_cast<size_t>(view - record->views_begin);
                if (const TextureCache::BindingMemo* const memo = record->memos[view_index]) {
                    guest_descriptor_queue.AddPayload(memo->payload.data());
                    is_rescaled |= memo->rescaled != 0;
                    continue;
                }
            }
            const VideoCommon::ImageViewId image_view_id{view->id};
            ImageView& image_view{texture_cache.GetImageView(image_view_id)};
            VkImageView vk_image_view{image_view.Handle(desc.type)};
            if (vk_image_view == VK_NULL_HANDLE) {
                const VkImageView null_image_view{texture_cache.GetImageView(VideoCommon::NULL_IMAGE_VIEW_ID).Handle(desc.type)};
                if (null_image_view != VK_NULL_HANDLE) vk_image_view = null_image_view;
            }
            Sampler& sampler{texture_cache.GetSampler(sampler_id)};
            guest_descriptor_queue.AddSampledImage(vk_image_view,
                                                   sampler.HandleFor(image_view, desc.is_depth));
            const bool element_rescaled{texture_cache.IsRescaling(image_view)};
            is_rescaled |= element_rescaled;
            if (record && record->descriptors) {
                record->descriptors[view_index] = guest_descriptor_queue.Last();
                record->rescaled[view_index] = element_rescaled ? 1 : 0;
            }
        }
        rescaling.PushTexture(is_rescaled);
    }
    for (const auto& desc : info.image_descriptors) {
        bool is_rescaled{};
        for (u32 index = 0; index < desc.count; ++index) {
            const VideoCommon::ImageViewInOut* const view{views++};
            size_t view_index = 0;
            if (record) {
                view_index = static_cast<size_t>(view - record->views_begin);
                if (const TextureCache::BindingMemo* const memo = record->memos[view_index]) {
                    if (desc.is_written) {
                        texture_cache.MarkModification(memo->image);
                    }
                    guest_descriptor_queue.AddPayload(memo->payload.data());
                    is_rescaled |= memo->rescaled != 0;
                    continue;
                }
            }
            ImageView& image_view{texture_cache.GetImageView(view->id)};
            const VkImageView vk_image_view{
                image_view.StorageView(desc.type, desc.format, desc.is_byte_addressed)};
            if (desc.is_written) {
                texture_cache.MarkModification(image_view.image_id);
            }
            guest_descriptor_queue.AddImage(vk_image_view);
            const bool element_rescaled{texture_cache.IsRescaling(image_view)};
            is_rescaled |= element_rescaled;
            if (record && record->descriptors) {
                record->descriptors[view_index] = guest_descriptor_queue.Last();
                record->rescaled[view_index] = element_rescaled ? 1 : 0;
            }
        }
        rescaling.PushImage(is_rescaled);
    }
}

} // namespace Vulkan
