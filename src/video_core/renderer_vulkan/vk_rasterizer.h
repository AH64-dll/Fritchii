// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>

#include <boost/container/static_vector.hpp>

#include "common/common_types.h"
#include "common/dkb_cpu_trace.h"
#include "common/dkb_frame_trace.h"
#include "video_core/buffer_cache/nx2_write_handoff.h"
#include "video_core/control/channel_state_cache.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/oot_network.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_vulkan/blit_image.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_descriptor_buffer.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_fence_manager.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_query_cache.h"
#include "video_core/renderer_vulkan/vk_render_pass_cache.h"
#include "video_core/renderer_vulkan/vk_frame_dump.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Core {
class System;
}

namespace Core::Frontend {
class EmuWindow;
}

namespace Tegra {

namespace Engines {
class Maxwell3D;
}

} // namespace Tegra

namespace Vulkan {

struct FramebufferTextureInfo;

class StateTracker;

class AccelerateDMA : public Tegra::Engines::AccelerateDMAInterface {
public:
    explicit AccelerateDMA(BufferCache& buffer_cache, TextureCache& texture_cache,
                           Scheduler& scheduler);

    bool BufferCopy(GPUVAddr start_address, GPUVAddr end_address, u64 amount) override;

    bool BufferClear(GPUVAddr src_address, u64 amount, u32 value) override;

    bool ImageToBuffer(const Tegra::DMA::ImageCopy& copy_info, const Tegra::DMA::ImageOperand& src,
                       const Tegra::DMA::BufferOperand& dst) override;

    bool BufferToImage(const Tegra::DMA::ImageCopy& copy_info, const Tegra::DMA::BufferOperand& src,
                       const Tegra::DMA::ImageOperand& dst) override;

private:
    template <bool IS_IMAGE_UPLOAD>
    bool DmaBufferImageCopy(const Tegra::DMA::ImageCopy& copy_info,
                            const Tegra::DMA::BufferOperand& src,
                            const Tegra::DMA::ImageOperand& dst);

    BufferCache& buffer_cache;
    TextureCache& texture_cache;
    Scheduler& scheduler;
};

class RasterizerVulkan final : public VideoCore::RasterizerInterface,
                               protected VideoCommon::ChannelSetupCaches<VideoCommon::ChannelInfo> {
public:
    explicit RasterizerVulkan(Core::Frontend::EmuWindow& emu_window_, Tegra::GPU& gpu_,
                              Tegra::MaxwellDeviceMemoryManager& device_memory_,
                              const Device& device_, MemoryAllocator& memory_allocator_,
                              StateTracker& state_tracker_, Scheduler& scheduler_);
    ~RasterizerVulkan() override;

    void Draw(bool is_indexed, u32 instance_count) override;
    void DrawIndirect() override;
    void DrawTexture() override;
    void Clear(u32 layer_count) override;
    void DispatchCompute() override;
    void ResetCounter(VideoCommon::QueryType type) override;
    void Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
               VideoCommon::QueryPropertiesFlags flags, u32 payload, u32 subreport) override;
    void BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr, u32 size) override;
    void DisableGraphicsUniformBuffer(size_t stage, u32 index) override;
    void FlushAll() override;
    void FlushRegion(DAddr addr, u64 size,
                     VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    GpuWrittenList ReadGpuWrittenCommandList(DAddr addr, std::span<u8> out,
                                             bool past_sync) override;
    void NoteGpuWrittenCommandListSynced(DAddr page) override;
    void FlushGpuWrittenCommandListPages(std::span<const DAddr> pages) override;
    bool MustFlushRegion(DAddr addr, u64 size,
                         VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    VideoCore::RasterizerDownloadArea GetFlushArea(DAddr addr, u64 size) override;
    bool QueueFlushRegion(DAddr addr, u64 size) override;
    bool HasPendingQueries(DAddr addr, u64 size) override;
    void InvalidateRegion(DAddr addr, u64 size,
                          VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    void PrepareRegionWrite(DAddr addr, u64 size, VideoCommon::CacheType which) override;
    void InnerInvalidation(std::span<const std::pair<DAddr, std::size_t>> sequences) override;
    void OnCacheInvalidation(DAddr addr, u64 size) override;
    void OnCacheInvalidations(std::span<const std::pair<DAddr, u64>> ranges) override;
    bool TryCacheInvalidation(DAddr addr, u64 size) override;
    bool OnCPUWrite(DAddr addr, u64 size) override;
    void InvalidateGPUCache() override;
    void UnmapMemory(DAddr addr, u64 size) override;
    void ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) override;
    void SignalFence(std::function<void()>&& func) override;
    void SyncOperation(std::function<void()>&& func) override;
    void SignalSyncPoint(u32 value) override;
    void SignalReference() override;
    void ReleaseFences(bool force = true) override;
    void FlushQueryResults(DAddr addr, u64 size) override;
    void FlushAndInvalidateRegion(
        DAddr addr, u64 size, VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    void WaitForIdle() override;
    void FragmentBarrier() override;
    void TiledCacheBarrier() override;
    void FlushCommands() override;
    void TickFrame() override;
    bool AccelerateConditionalRendering() override;
    bool HasDrawTransformFeedback() override;
    bool AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Surface& src,
                               const Tegra::Engines::Fermi2D::Surface& dst,
                               const Tegra::Engines::Fermi2D::Config& copy_config) override;
    bool SupportsGameDlssNetwork() const override;
    bool UsesNvidiaDlss() const override;
    bool CanProbeGameDlssOutput() const override {
        return true;
    }
    std::optional<VideoCore::GameDlssProbe> ProbeGameDlssOutput(
        const Tegra::Engines::Fermi2D::Surface& src,
        const Tegra::Engines::Fermi2D::Surface& dst) override;
    bool AccelerateTemporalUpscale(const Tegra::Engines::Fermi2D::Surface& src,
                                   const Tegra::Engines::Fermi2D::Surface& dst,
                                   const Tegra::Engines::Fermi2D::Surface& motion_vectors,
                                   const Tegra::Texture::TICEntry* depth,
                                   std::array<f32, 2> jitter,
                                   std::array<f32, 2> motion_scale, bool use_history = true) override;
    VideoCommon::OotNetwork::BypassState* OotNetworkBypass() override;
    Tegra::Engines::AccelerateDMAInterface& AccessAccelerateDMA() override;
    void AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                  std::span<const u8> memory) override;
    void LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                           const VideoCore::DiskResourceLoadCallback& callback) override;

    void InitializeChannel(Tegra::Control::ChannelState& channel) override;

    void BindChannel(Tegra::Control::ChannelState& channel) override;

    void ReleaseChannel(s32 channel_id) override;
    std::optional<FramebufferTextureInfo> AccelerateDisplay(const Tegra::FramebufferConfig& config,
                                                            VAddr framebuffer_addr,
                                                            u32 pixel_stride);

private:
    static constexpr const u64 NEEDS_D24[] = {
        0x01006A800016E000ULL, // SSBU
        0x0100E95004038000ULL, // XC2
        0x0100A6301214E000ULL, // FE:Engage
    };
    static constexpr size_t MAX_TEXTURES = 192;
    static constexpr size_t MAX_IMAGES = 48;
    static constexpr size_t MAX_IMAGE_VIEWS = MAX_TEXTURES + MAX_IMAGES;

    static constexpr VkDeviceSize DEFAULT_BUFFER_SIZE = 4 * sizeof(float);

    template <typename Func>
    void PrepareDraw(bool is_indexed, Func&&, bool is_indirect = false);

    void CaptureIndirectTrace();
    void CollectIndirectTrace();
    void TraceDrawFailure(std::string_view reason);
    /// Remembers render target 0 of a draw with one of Ocarina of Time's tonemap shaders.
    void TrackOotTonemappedImage(const GraphicsPipeline& pipeline);
    bool TryNx2IndirectReset(bool is_indexed, u32 instance_count);

    void FlushWork();

    void TraceRenderState(bool is_indexed, const char* outcome, const Framebuffer* framebuffer);

    /// Records the copies of the command list pages the work just recorded wrote and submits, so
    /// the GPU makes them now (BufferCache::Nx2CopyListPages).
    void SubmitNx2ListCopies();

    /// Records the dynamic state of the draw being prepared with `pipeline` (null for
    /// DrawTexture's blit).
    void UpdateDynamicStates(const GraphicsPipeline* pipeline);

    void HandleTransformFeedback();

    void UpdateViewportsState(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateScissorsState(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateDepthBias(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateBlendConstants(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateDepthBounds(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateStencilFaces(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateLineWidth(Tegra::Engines::Maxwell3D::Regs& regs);

    void UpdateCullMode(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateDepthBoundsTestEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateDepthTestEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateDepthWriteEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateDepthCompareOp(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdatePrimitiveRestartEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateRasterizerDiscardEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateConservativeRasterizationMode(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateLineStippleEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateLineStipple(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateLineRasterizationMode(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateDepthBiasEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateLogicOpEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateDepthClampEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateAlphaToCoverageEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateAlphaToOneEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateFrontFace(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateStencilOp(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateStencilTestEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateLogicOp(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateBlending(Tegra::Engines::Maxwell3D::Regs& regs, const GraphicsPipeline* pipeline);
    void UpdateColorWriteEnable(Tegra::Engines::Maxwell3D::Regs& regs);

    void UpdateVertexInput(Tegra::Engines::Maxwell3D::Regs& regs);

    /// EDEN_NX2_DYNSTATE_CACHE: one slot per dynamic state whose last recorded value is cached.
    enum class DynSlot : u32 {
        Viewports,
        Scissors,
        DepthBias,
        BlendConstants,
        DepthBounds,
        LineWidth,
        CullMode,
        DepthBoundsTestEnable,
        DepthTestEnable,
        DepthWriteEnable,
        StencilTestEnable,
        DepthCompareOp,
        FrontFace,
        StencilOp,
        PrimitiveRestartEnable,
        RasterizerDiscardEnable,
        DepthBiasEnable,
        LogicOp,
        ColorWriteEnable,
        VertexInput,
    };

    /// False when `value` is what slot last recorded into the current command buffer (the
    /// record can be skipped); otherwise caches it and returns true.
    template <typename T>
    bool DynamicStateChanged(DynSlot slot, T& cached, const T& value);
    void ForgetDynamicState(DynSlot slot) noexcept {
        dyn_cache.valid &= ~(u64{1} << static_cast<u32>(slot));
    }
    void BuildDynamicStateMask();

    struct VertexInputLayout {
        boost::container::static_vector<VkVertexInputBindingDescription2EXT, 32> bindings;
        boost::container::static_vector<VkVertexInputAttributeDescription2EXT, 32> attributes;
    };
    /// Attribute formats, per binding stride, instancing and divisor, and the repacked bindings.
    using VertexInputKey = std::array<u32, 32 + 32 * 3 + 1>;
    struct VertexInputKeyHash {
        size_t operator()(const VertexInputKey& key) const noexcept;
    };
    void BuildVertexInputLayout(const Tegra::Engines::Maxwell3D::Regs& regs, u32 max_attributes,
                                u32 max_bindings, VertexInputLayout& layout) const;
    /// BufferCacheRuntime::SetVertexRepackLayout, skipped while its inputs are unchanged.
    u32 UpdateVertexRepackLayout(bool dynamic_input);

    void FrameDumpPass(const Framebuffer* framebuffer);
    /// EDEN_NX2_PENDING_WRITE_TRACE: logs bound buffer ranges with collected, unapplied CPU writes.
    void TracePendingCpuWrites(bool compute, std::span<const u64> shader_hashes);
#ifdef EGLANTINE_STREAMLINE
    bool OcarinaDlssEnabled() const;
    void TrackOcarinaCamera();
    void TrackOcarinaActor(u64 shader_hash, bool is_indexed, bool is_indirect);
    bool TryOcarinaDlss(const ComputePipeline& pipeline, bool replace);
    void ReadOcarinaActor(struct OcarinaActorMesh& mesh);
#endif
    /// Records the uniform and storage buffers of the current draw or dispatch for the frame dump.
    bool FrameDumpBuffers(bool compute, std::span<const u64> shader_hashes);

    Tegra::GPU& gpu;
    Tegra::MaxwellDeviceMemoryManager& device_memory;

    const Device& device;
    MemoryAllocator& memory_allocator;
    StateTracker& state_tracker;
    Scheduler& scheduler;

    StagingBufferPool staging_pool;
    DescriptorPool descriptor_pool;
    GuestDescriptorQueue guest_descriptor_queue;
    ComputePassDescriptorQueue compute_pass_descriptor_queue;
    DescriptorBufferRing descriptor_buffer_ring;
    BlitImageHelper blit_image;
    RenderPassCache render_pass_cache;

    TextureCacheRuntime texture_cache_runtime;
    TextureCache texture_cache;
    BufferCacheRuntime buffer_cache_runtime;
    BufferCache buffer_cache;
    VideoCommon::Nx2WriteHandoff write_handoff;
    QueryCacheRuntime query_cache_runtime;
    QueryCache query_cache;
    PipelineCache pipeline_cache;
    AccelerateDMA accelerate_dma;
    FenceManager fence_manager;
    FrameDump frame_dump;

    boost::container::static_vector<u32, MAX_IMAGE_VIEWS> image_view_indices;
    std::array<VideoCommon::ImageViewId, MAX_IMAGE_VIEWS> image_view_ids;
    boost::container::static_vector<VkSampler, MAX_TEXTURES> sampler_handles;

    u32 draw_counter = 0;
#ifdef EGLANTINE_STREAMLINE
    std::optional<u64> oot_camera_sample_frame;
    struct OcarinaMotionTrace {
        const Tegra::MemoryManager* camera_channel{};
        const Tegra::MemoryManager* depth_channel{};
        u64 depth_frame{};
        GPUVAddr depth_address{};
        u32 depth_tic{};
        VkExtent2D extent{};
        std::array<f32, 16> current{};
        std::array<f32, 16> previous{};
    } oot_motion;
    std::unique_ptr<class OcarinaLiveDlss> oot_live_dlss;
    bool oot_live_failed{};
    const char* oot_fallback_reason{};
    u32 oot_fallback_repeats{};
    std::optional<u64> oot_dlss_frame;
    std::optional<u64> oot_4k_scene_frame;
    VkImage oot_4k_guest_present{};
    struct OcarinaActorPose {
        u64 frame{};
        const Tegra::MemoryManager* channel{};
        GPUVAddr vertex{};
        GPUVAddr indices{};
        s32 base_vertex{};
        std::array<f32, 4> origin{};
        std::array<f32, 280 * 12> bones{};
    } oot_actor, oot_previous_actor;
#endif
    u64 nx2_dispatch_serial = 0;     ///< Compute dispatches recorded (EDEN_NX2_WFI_DEFER)
    u64 nx2_wfi_dispatch_serial = 0; ///< nx2_dispatch_serial at the last real wait for idle
    u64 render_trace_frame = 0;
    u32 render_trace_draws = 0;
    u32 render_trace_clears = 0;
    bool pending_native_work{};
    bool mesh_draw{}; ///< the draw being prepared uses a task/mesh pipeline
    /// Fragment output components the dynamic color write mask was last recorded with
    u32 last_color_output_mask{};
    VideoCommon::OotNetwork::BypassState oot_network_bypass;
    u32 mesh_emulation_groups{};             ///< mesh workgroups of the emulated draw
    u32 mesh_emulation_vertices_per_group{}; ///< 0 unless the draw is an emulated mesh draw
    /// Mesh emulation diagnostics with a task program: one draw's distribution arguments.
    struct {
        std::span<const u8> args;
        u64 tick{};
        u32 tasks{};
        u32 capacity{};
        bool pending{};
    } mesh_task_probe;
    /// With a task program: the vkCmdDrawIndirect arguments the distribution pass writes.
    struct {
        VkBuffer buffer{};
        VkDeviceSize offset{};
    } mesh_emulation_indirect;
    /// Mesh emulation diagnostics: one draw's output, read back once the GPU has written it.
    struct MeshEmulationProbe {
        std::span<const u8> data;
        u64 tick{};
        u64 frame{};
        u32 groups{};
        u32 block_words{};
        u32 vertex_lines{};
        u32 max_vertices{};
        u32 max_primitives{};
        std::array<u32, 4> position_lines{};
        bool pending{};
    } mesh_probe;
    void ReadMeshEmulationProbe();
    /// Mesh emulation with a task program: the output arena every task draw of a submission
    /// allocates its mesh workgroups from, and its host-visible counter of allocated words.
    struct MeshArena {
        vk::Buffer buffer;
        vk::Buffer counter;
        u64 size{};
        u64 tick{};
        bool counted{}; ///< the counter of its last submission was read
    };
    std::vector<MeshArena> mesh_arenas;
    size_t mesh_arena_current{~size_t{}};
    u64 mesh_arena_size{64ULL << 20};
    /// The current submission's arena, after its counter is reset.
    const MeshArena& AcquireMeshArena();

    /// Dynamic state last recorded into the command buffer of `epoch` (bit per DynSlot in
    /// `valid`). Viewports and scissors are keyed on the registers they are computed from.
    struct DynamicStateCache {
        u64 epoch{};
        u64 valid{};
        std::array<u32, 6 + 16 * 7> viewports{};
        std::array<u32, 6 + 16 * 3> scissors{};
        std::array<f32, 3> depth_bias{};
        std::array<f32, 4> blend_constants{};
        std::array<f32, 2> depth_bounds{};
        f32 line_width{};
        u32 cull_mode{};
        u32 depth_bounds_test{};
        u32 depth_test{};
        u32 depth_write{};
        u32 stencil_test{};
        u32 depth_compare{};
        u32 front_face{};
        std::array<u32, 9> stencil_op{};
        u32 primitive_restart{};
        u32 rasterizer_discard{};
        u32 depth_bias_enable{};
        u32 logic_op{};
        std::array<VkBool32, 8> color_write_enable{};
        const void* vertex_input{};
    };
    DynamicStateCache dyn_cache;
    /// The dirty flags UpdateDynamicStates consumes on this device: none set, nothing to record.
    Tegra::Engines::Maxwell3D::DirtyState::Flags dyn_state_mask{};
    bool dyn_state_mask_built{};
    bool dyn_state_never_skip{};

    /// EDEN_NX2_VERTEX_INPUT_CACHE: vertex input layouts, built once and recorded by pointer
    /// (immutable while the rasterizer lives).
    std::unordered_map<VertexInputKey, std::unique_ptr<VertexInputLayout>, VertexInputKeyHash>
        vertex_input_layouts;
    VertexInputKey last_vertex_input_key{};
    const VertexInputLayout* last_vertex_input_layout{};
    /// Attribute formats and strides SetVertexRepackLayout last chose its repack mask from.
    std::array<u32, 64> repack_key{};
    bool repack_key_valid{};
    bool repack_key_dynamic{};

    struct IndirectTrace {
        vk::Buffer readback;
        u64 tick{};
        u64 frame{};
        GPUVAddr address{};
        GPUVAddr count_address{};
        u32 max_draws{};
        u32 sampled_commands{};
        bool pending{};
        bool include_count{};
        bool byte_count{};
    };
    std::array<IndirectTrace, 8> indirect_traces;
    Common::DkbFrameTrace::RateLimit trace_rate;
    Common::DkbFrameTrace::RateLimit trace_cpu_rate;
    std::optional<Common::DkbCpuTrace::Snapshot> trace_cpu_snapshot;
    u64 trace_cpu_frame{};
    bool trace_sample_frame{true};
    u64 trace_frame{};
    u64 trace_window_frames{};
    u64 trace_draws{};
    u64 trace_skipped{};
    u64 trace_indirect{};
    u64 trace_zero_direct{};
    /// Vertex and fragment hashes of the draw being recorded, for GPU timing.
    std::pair<u64, u64> trace_draw_hashes{};
    u32 trace_captures{};
    u32 trace_capture_dropped{};
    u32 trace_failure_details{};
};

} // namespace Vulkan
