// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <stop_token>
#include <functional>
#include <mutex>
#include <string>
#include <type_traits>
#include "common/container/unordered_map.h"
#include <vector>

#include "common/common_types.h"
#include "common/thread_worker.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/shader_cache.h"

namespace Core {
class System;
}

namespace Shader::IR {
struct Program;
}

namespace VideoCore {
class ShaderNotify;
}

namespace Vulkan {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;

struct ComputePipelineCacheKey {
    u64 unique_hash;
    u32 shared_memory_size;
    std::array<u32, 3> workgroup_size;

    size_t Hash() const noexcept;

    bool operator==(const ComputePipelineCacheKey& rhs) const noexcept;

    bool operator!=(const ComputePipelineCacheKey& rhs) const noexcept {
        return !operator==(rhs);
    }
};
static_assert(std::has_unique_object_representations_v<ComputePipelineCacheKey>);
static_assert(std::is_trivially_copyable_v<ComputePipelineCacheKey>);
static_assert(std::is_trivially_constructible_v<ComputePipelineCacheKey>);

} // namespace Vulkan

namespace std {

template <>
struct hash<Vulkan::ComputePipelineCacheKey> {
    size_t operator()(const Vulkan::ComputePipelineCacheKey& k) const noexcept {
        return k.Hash();
    }
};

} // namespace std

namespace Vulkan {

class ComputePipeline;
class DescriptorPool;
class Device;
class PipelineStatistics;
class RenderPassCache;
class Scheduler;

using VideoCommon::ShaderInfo;

struct ShaderPools {
    void ReleaseContents() {
        flow_block.ReleaseContents();
        block.ReleaseContents();
        inst.ReleaseContents();
    }

    Shader::ObjectPool<Shader::IR::Inst> inst{8192};
    Shader::ObjectPool<Shader::IR::Block> block{32};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_block{32};
};

struct ComputePipelineCacheEntry {
    std::unique_ptr<ComputePipeline> pipeline;
    std::string failure;
    /// Constant buffer slots that were null pointers at translation (see NullStoragePointerSlots)
    std::vector<std::pair<u32, u32>> null_pointer_slots;
    /// Constant buffer slots bound as storage buffer pointers (see MappedStoragePointerKeys)
    std::vector<u64> mapped_pointers;
    u32 rebuilds{};

    [[nodiscard]] ComputePipeline* Get(bool native, GPUVAddr address,
                                       const ComputePipelineCacheKey& key) const;
};

class PipelineCache : public VideoCommon::ShaderCache {
public:
    explicit PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory_, const Device& device,
                           Scheduler& scheduler, DescriptorPool& descriptor_pool,
                           GuestDescriptorQueue& guest_descriptor_queue,
                           DescriptorBufferRing& descriptor_buffer_ring,
                           RenderPassCache& render_pass_cache, BufferCache& buffer_cache,
                           TextureCache& texture_cache, VideoCore::ShaderNotify& shader_notify_);
    ~PipelineCache();

    [[nodiscard]] GraphicsPipeline* CurrentGraphicsPipeline();

    void TickFrame();

    [[nodiscard]] ComputePipeline* CurrentComputePipeline();

    void LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                           const VideoCore::DiskResourceLoadCallback& callback);

    /// What a graphics warm-up entry came to: built, not buildable (a guess the translation
    /// rejects, which another warm-up would reject too), or refused by the driver.
    enum class WarmResult { Built, Unbuildable, DriverFailed };

    /// Graphics pipeline libraries (EDEN_NX2_GPL): builds the pre-rasterization and fragment
    /// shader libraries of the pipeline `key` would make, on the calling thread, with no draw and
    /// no pipeline kept (for a warm-up). `pools` must be the calling thread's own. Unbuildable
    /// when the device has no libraries or the pipeline takes the monolithic route, DriverFailed
    /// when a library build failed.
    /// The key may hold a single stage (no partner): a vertex-only key builds just that vertex
    /// program's pre-rasterization library (vertex SPIR-V does not depend on the fragment
    /// stage); a fragment-only key builds just the fragment library, emitted as if the partner
    /// writes every varying (the common variant). The key's state (attribute types, render
    /// target formats, depth and raster state) is what the libraries are keyed and built with.
    WarmResult WarmGraphicsLibraries(ShaderPools& pools, const GraphicsPipelineCacheKey& key,
                                     std::span<Shader::Environment* const> envs);

private:
    [[nodiscard]] GraphicsPipeline* CurrentGraphicsPipelineSlowPath();

    /// FixedPipelineState::Refresh of graphics_key, keeping only the vertex attribute types the
    /// bound programs read (see VideoCommon::KeyedAttributeTypes).
    void RefreshGraphicsKey();

    /// EDEN_NX2_PIPELINE_CLEAN / EDEN_NX2_EXP19: CurrentGraphicsPipeline without
    /// FixedPipelineState::Refresh nor the key comparison while nothing Refresh reads changed
    /// since the last lookup. `patch`: EXP19=2.
    [[nodiscard]] GraphicsPipeline* CurrentGraphicsPipelineClean(bool patch);
    /// The registers Refresh reads with these features, as (first word, words) ranges.
    void BuildCleanWatch(bool patch);
    [[nodiscard]] bool CleanInputsUnchanged() const;
    void TakeCleanSnapshot();

    struct CleanLookup {
        struct Range {
            u32 first;
            u32 count;
        };
        bool built{};
        bool patch{};                  ///< Built for CurrentGraphicsPipelineClean(true)
        std::vector<Range> ranges;     ///< Read by every Refresh
        std::vector<Range> xfb_ranges; ///< Read while transform feedback is enabled
        std::vector<u32> snapshot;     ///< ranges' words at the last Refresh
        std::vector<u32> xfb_snapshot;
        std::array<u64, 6> hashes{};   ///< graphics_key.unique_hashes at the last Refresh
        const Tegra::Engines::Maxwell3D* maxwell3d{};
        Maxwell::PrimitiveTopology topology{};
        Tegra::Engines::Maxwell3D::EngineHint engine_state{};
        /// current_pipeline when its key was graphics_key after the last Refresh, else null.
        GraphicsPipeline* pipeline{};
        u64 hits{};
        u64 patches{};
        u64 misses{};
        std::chrono::steady_clock::time_point last_log{};
    } clean;

    [[nodiscard]] GraphicsPipeline* BuiltPipeline(GraphicsPipeline* pipeline) const noexcept;

    std::unique_ptr<GraphicsPipeline> CreateGraphicsPipeline();

    std::unique_ptr<GraphicsPipeline> CreateGraphicsPipeline(
        ShaderPools& pools, const GraphicsPipelineCacheKey& key,
        std::span<Shader::Environment* const> envs, PipelineStatistics* statistics,
        bool build_in_parallel, std::shared_ptr<ShaderPools> emit_pools = nullptr,
        std::unique_ptr<EmissionBudget::Reservation> reservation = nullptr,
        bool libraries_only = false);

    /// Builds a disk-cached graphics pipeline from the first of its records (newest first) that
    /// loads. When none does, the GPU thread's build of it is stored again.
    std::unique_ptr<GraphicsPipeline> CreateCachedGraphicsPipeline(
        const GraphicsPipelineCacheKey& key,
        std::vector<std::vector<VideoCommon::FileEnvironment>>& records,
        PipelineStatistics* statistics);

    [[nodiscard]] bool StaleNullPointerSlot(const ComputePipelineCacheEntry& entry) const;

    /// Translates the current compute program, binding the pointers in keep_pointers whatever
    /// their slots hold now (see GenericEnvironment::KeepStoragePointers)
    ComputePipelineCacheEntry CreateComputePipeline(const ComputePipelineCacheKey& key,
                                                    const ShaderInfo* shader,
                                                    std::span<const u64> keep_pointers = {});

    ComputePipelineCacheEntry CreateComputePipeline(ShaderPools& pools,
                                                     const ComputePipelineCacheKey& key,
                                                     Shader::Environment& env,
                                                     PipelineStatistics* statistics,
                                                     bool build_in_parallel, bool warm = false);

    void SerializeVulkanPipelineCache(const std::filesystem::path& filename,
                                      const vk::PipelineCache& pipeline_cache, u32 cache_version);

    vk::PipelineCache LoadVulkanPipelineCache(const std::filesystem::path& filename,
                                              u32 expected_cache_version);

    void QueueVulkanPipelineCacheFlush();
    /// Writes the driver pipeline cache on the serialization thread, unless a write is queued
    /// already (`again_if_busy`: then one more follows it, for a save that must hold the latest
    /// builds). Callable from any thread.
    void QueueVulkanPipelineCacheSave(bool again_if_busy);
    /// True (and `last` set to now) when a write of the driver cache is due since `last`.
    [[nodiscard]] bool VulkanPipelineCacheSaveDue(
        std::chrono::steady_clock::time_point& last) const;

    /// The title's warm-compute.bin (ShaderWarm::ComputeWarmPath, EDEN_NX2_COMPUTE_WARM=<file>
    /// overrides): compute programs prepared from the game files are converted (missing reads
    /// guessed) and built in the background, then thrown away. Nothing uses them; they only fill
    /// the driver's shader cache, so a dispatch that later converts to the same SPIR-V finds the
    /// driver build ready instead of waiting seconds for it.
    void StartComputeWarm(u64 title_id);
    std::vector<std::jthread> compute_warm_threads;
    std::atomic<bool> compute_warm_done{true};

    /// The title's warm-graphics.bin (ShaderWarm::GraphicsWarmPath, EDEN_NX2_GRAPHICS_WARM=<file>
    /// overrides; needs graphics pipeline libraries): vertex and fragment programs prepared from
    /// the game files are built as single-stage libraries in the background after the compute
    /// warm-up, then released; the driver's disk cache keeps them, so a later pipeline with those
    /// stages costs a link.
    void StartGraphicsWarm(u64 title_id);
    std::vector<std::jthread> graphics_warm_threads;

    /// Set as soon as emulation is asked to stop (the loading stop token): the warm-ups end
    /// after their current build instead of compiling on until the cache is destroyed, which
    /// kept shutdown waiting on driver compiles.
    std::atomic<bool> warm_cancel{false};
    std::unique_ptr<std::stop_callback<std::function<void()>>> warm_stop_callback;

    /// Waits until the warm-up may build (ShaderNotify::WarmMayBuild). False when it is to stop.
    bool WaitForWarmTurn(std::stop_token stop);

    /// The pipeline cache warm-up builds go through: none by default, so the tens of thousands
    /// of warm-up programs stay in the driver's disk cache and out of vulkan_pipelines.bin (which
    /// is loaded into memory at every boot). EDEN_NX2_WARM_APP_CACHE=1 uses ours.
    [[nodiscard]] vk::PipelineCache& WarmPipelineCache();
    vk::PipelineCache no_pipeline_cache;

    const Device& device;
    Scheduler& scheduler;
    DescriptorPool& descriptor_pool;
    GuestDescriptorQueue& guest_descriptor_queue;
    DescriptorBufferRing& descriptor_buffer_ring;
    RenderPassCache& render_pass_cache;
    BufferCache& buffer_cache;
    TextureCache& texture_cache;
    VideoCore::ShaderNotify& shader_notify;
    bool use_asynchronous_shaders{};
    bool use_vulkan_pipeline_cache{};
    bool oot_visual_workaround{};
    u64 visual_workaround_title{};
    const size_t num_pipeline_workers;
    EmissionBudget emission_budget;

    GraphicsPipelineCacheKey graphics_key{};
    u64 vertex_attribute_types{}; ///< graphics_key's attribute types before the programs' mask
    GraphicsPipeline* current_pipeline{};

    /// EDEN_NX2_GPL: shader libraries shared by the graphics pipelines. Declared before the
    /// pipelines, so it outlives them.
    GraphicsPipelineLibraryCache graphics_libraries;

    /// Destroys graphics pipelines unused for a while, once the GPU is done with them.
    void EvictUnusedPipelines();
    u64 frame_count{};
    std::deque<std::pair<u64, std::unique_ptr<GraphicsPipeline>>> evicted_pipelines; ///< (tick, pipeline)
    std::unordered_set<u64> used_and_evicted; ///< Key hashes of pipelines used this session, evicted since

    ::Common::unordered_map<ComputePipelineCacheKey, ComputePipelineCacheEntry> compute_cache;
    /// Replaced compute pipelines, destroyed once the GPU and their queued build are done with
    /// them (tick, pipeline)
    std::deque<std::pair<u64, std::unique_ptr<ComputePipeline>>> retired_compute_pipelines;
    ::Common::unordered_map<GraphicsPipelineCacheKey, std::unique_ptr<GraphicsPipeline>> graphics_cache;

    ShaderPools main_pools;
    std::mutex spare_pools_mutex;
    std::vector<std::shared_ptr<ShaderPools>> spare_pools;

    Shader::Profile profile;
    Shader::HostTranslateInfo host_info;

    std::filesystem::path pipeline_cache_filename;

    /// Key hashes of the graphics pipelines in the disk cache. Only those a recent session used
    /// (used_pipelines) are built at boot; the rest are built when first needed, and not stored
    /// again. Each built pipeline holds driver memory (title 1: 6.5k cached, a quarter used).
    std::unordered_set<u64> disk_pipelines;
    /// Key hashes of disk-cached graphics pipelines none of whose records loaded (one written
    /// before a translator change can ask about slots it never classified): stored again when the
    /// GPU thread builds them.
    std::unordered_set<u64> failed_disk_pipelines;
    std::mutex failed_disk_pipelines_mutex;
    std::unordered_map<u64, u8> used_pipelines; ///< Key hash -> sessions since it was last used
    std::unordered_set<const GraphicsPipeline*> used_this_session;
    std::filesystem::path used_pipelines_filename;
    std::chrono::steady_clock::time_point used_pipelines_saved{};
    void SaveUsedPipelines();

    std::filesystem::path vulkan_pipeline_cache_filename;
    vk::PipelineCache vulkan_pipeline_cache;
    size_t pipelines_since_flush{};
    std::chrono::steady_clock::time_point last_flush{};
    std::atomic<size_t> last_cache_size{};
    std::atomic_bool flush_in_flight{};
    std::atomic_bool save_again{};

    Common::ThreadWorker workers;
    Common::ThreadWorker serialization_thread;
    DynamicFeatures dynamic_features;

    /// Cached graphics pipelines boot skipped (not in used_pipelines) are built on background
    /// threads once the game runs, so a draw that needs one finds it built instead of translating
    /// it on the GPU thread. EDEN_NX2_BACKGROUND_PIPELINES=0 turns it off.
    std::unique_ptr<GraphicsPipeline> TakeBackgroundPipeline(const GraphicsPipelineCacheKey& key);
    void StopBackgroundPipelines();
    std::mutex background_mutex;
    /// Built and not yet asked for by a draw.
    std::unordered_map<GraphicsPipelineCacheKey, std::unique_ptr<GraphicsPipeline>> background_ready;
    /// Keys the GPU thread built itself meanwhile: their background build is skipped or dropped.
    std::unordered_set<GraphicsPipelineCacheKey> background_claimed;
    std::atomic_bool background_active{};
    std::atomic_bool background_stop{};
    std::atomic<size_t> background_left{};
    /// Numbers the background threads for the EDEN_NX2_EXP5 thread limit.
    std::atomic<size_t> background_slots{};
    size_t background_taken{}; ///< Under background_mutex
    std::chrono::steady_clock::time_point background_start{};
    std::chrono::steady_clock::time_point background_last_save{}; ///< Under background_mutex
    /// Last: destroyed (its threads joined) before the state above.
    std::unique_ptr<Common::ThreadWorker> background_workers;
};

} // namespace Vulkan
