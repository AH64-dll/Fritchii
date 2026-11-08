// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstdlib>
#include <array>
#include <atomic>
#include <bit>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/container/unordered_map.h"
#include <boost/container/static_vector.hpp>
#include <boost/container/small_vector.hpp>

#include "common/common_types.h"
#include "common/spin_recursive_mutex.h"
#include "common/div_ceil.h"
#include "common/literals.h"
#include "common/lru_cache.h"
#include "common/range_sets.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/slot_vector.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/buffer_cache/buffer_base.h"
#include "video_core/buffer_cache/nx2_bufcache.h"
#include "video_core/buffer_cache/read_trap_tracker.h"
#include "video_core/buffer_cache/uniform_stream_reuse.h"
#include "video_core/buffer_cache/virtual_range_cache.h"
#include "video_core/control/channel_state_cache.h"
#include "video_core/delayed_destruction_ring.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_download_trace.h"
#include "video_core/nx2_reactive.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/types.h"

namespace VideoCommon {

using BufferId = Common::SlotId;

using VideoCore::Surface::PixelFormat;
using namespace Common::Literals;

#ifdef __APPLE__
constexpr u32 NUM_VERTEX_BUFFERS = 16;
#else
constexpr u32 NUM_VERTEX_BUFFERS = 32;
#endif
constexpr u32 NUM_TRANSFORM_FEEDBACK_BUFFERS = 4;
constexpr u32 NUM_GRAPHICS_UNIFORM_BUFFERS = 18;
constexpr u32 NUM_COMPUTE_UNIFORM_BUFFERS = 8;
// Native shaders can reference more than 16 storage buffers. Dropping a binding
// shifts all following descriptor payloads into the wrong descriptor types.
constexpr u32 NUM_STORAGE_BUFFERS = Shader::Info::MAX_SSBOS;
static_assert(NUM_STORAGE_BUFFERS <= 64, "storage binding masks must fit u64");
constexpr u32 NUM_TEXTURE_BUFFERS = 32;
constexpr u32 NUM_STAGES = 5;

static_assert(NUM_GRAPHICS_UNIFORM_BUFFERS <= 32, "fast bitmask must fit u32");

using UniformBufferSizes = std::array<std::array<u32, NUM_GRAPHICS_UNIFORM_BUFFERS>, NUM_STAGES>;
using ComputeUniformBufferSizes = std::array<u32, NUM_COMPUTE_UNIFORM_BUFFERS>;

enum class ObtainBufferSynchronize : u32 {
    NoSynchronize = 0,
    FullSynchronize = 1,
    SynchronizeNoDirty = 2,
};

enum class ObtainBufferOperation : u32 {
    DoNothing = 0,
    MarkAsWritten = 1,
    DiscardWrite = 2,
    MarkQuery = 3,
};

static constexpr BufferId NULL_BUFFER_ID{0};
static constexpr u32 DEFAULT_SKIP_CACHE_SIZE = static_cast<u32>(4_KiB);

struct Binding {
    DAddr device_addr{};
    GPUVAddr gpu_addr{};
    u32 size{};
    BufferId buffer_id;
    u32 segment_first{};
    u32 segment_count{};
};

struct MultiRangeSegment {
    BufferId buffer_id;
    DAddr device_addr{};
    u32 size{};
};

struct TextureBufferBinding : Binding {
    PixelFormat format;
};

static constexpr Binding NULL_BINDING{
    .device_addr = 0,
    .size = 0,
    .buffer_id = NULL_BUFFER_ID,
};

template <typename Buffer>
struct HostBindings {
    boost::container::static_vector<Buffer*, NUM_VERTEX_BUFFERS> buffers;
    boost::container::static_vector<u64, NUM_VERTEX_BUFFERS> offsets;
    boost::container::static_vector<u64, NUM_VERTEX_BUFFERS> sizes;
    boost::container::static_vector<u64, NUM_VERTEX_BUFFERS> strides;
    u32 min_index{NUM_VERTEX_BUFFERS};
    u32 max_index{0};
};

class BufferCacheChannelInfo : public ChannelInfo {
public:
    BufferCacheChannelInfo() = delete;
    BufferCacheChannelInfo(Tegra::Control::ChannelState& state) noexcept : ChannelInfo(state) {}
    BufferCacheChannelInfo(const BufferCacheChannelInfo& state) = delete;
    BufferCacheChannelInfo& operator=(const BufferCacheChannelInfo&) = delete;

    Binding index_buffer;
    std::array<Binding, NUM_VERTEX_BUFFERS> vertex_buffers;
    std::array<std::array<Binding, NUM_GRAPHICS_UNIFORM_BUFFERS>, NUM_STAGES> uniform_buffers;
    std::array<std::array<Binding, NUM_STORAGE_BUFFERS>, NUM_STAGES> storage_buffers;
    std::array<std::array<TextureBufferBinding, NUM_TEXTURE_BUFFERS>, NUM_STAGES> texture_buffers;
    std::array<Binding, NUM_TRANSFORM_FEEDBACK_BUFFERS> transform_feedback_buffers;
    Binding count_buffer_binding;
    Binding indirect_buffer_binding;

    std::array<Binding, NUM_COMPUTE_UNIFORM_BUFFERS> compute_uniform_buffers;
    std::array<Binding, NUM_STORAGE_BUFFERS> compute_storage_buffers;
    std::array<TextureBufferBinding, NUM_TEXTURE_BUFFERS> compute_texture_buffers;

    std::array<u32, NUM_STAGES> enabled_uniform_buffer_masks{};
    u32 enabled_compute_uniform_buffer_mask = 0;

    const UniformBufferSizes* uniform_buffer_sizes{};
    const ComputeUniformBufferSizes* compute_uniform_buffer_sizes{};

    std::array<u64, NUM_STAGES> enabled_storage_buffers{};
    std::array<u64, NUM_STAGES> written_storage_buffers{};
    u64 enabled_compute_storage_buffers = 0;
    u64 written_compute_storage_buffers = 0;
    // Bound by a native (Switch 2) compute program, whose pointers carry no size.
    u64 native_compute_storage_buffers = 0;
    // Bytes from each compute storage binding's start that a dispatch marks GPU-written.
    std::array<u32, NUM_STORAGE_BUFFERS> compute_storage_write_marks{};

    u32 total_graphics_storage_buffers = 0;
    u32 total_compute_storage_buffers = 0;

    std::array<u32, NUM_STAGES> enabled_texture_buffers{};
    std::array<u32, NUM_STAGES> written_texture_buffers{};
    std::array<u32, NUM_STAGES> image_texture_buffers{};
    u32 enabled_compute_texture_buffers = 0;
    u32 written_compute_texture_buffers = 0;
    u32 image_compute_texture_buffers = 0;

    std::array<u32, NUM_GRAPHICS_UNIFORM_BUFFERS> uniform_cache_hits{};
    std::array<u32, NUM_GRAPHICS_UNIFORM_BUFFERS> uniform_cache_shots{};

    u32 uniform_buffer_skip_cache_size = DEFAULT_SKIP_CACHE_SIZE;

    bool has_deleted_buffers = false;

    std::array<u32, NUM_STAGES> dirty_uniform_buffers{};
    std::array<u32, NUM_STAGES> fast_bound_uniform_buffers{};
    std::array<std::array<u32, NUM_GRAPHICS_UNIFORM_BUFFERS>, NUM_STAGES>
        uniform_buffer_binding_sizes{};
};

template <class P>
class BufferCache : public VideoCommon::ChannelSetupCaches<BufferCacheChannelInfo> {
    friend struct BufferCacheTestAccess;
    // Page size for caching purposes.
    // This is unrelated to the CPU page size and it can be changed as it seems optimal.
#ifdef YUZU_LEGACY
    static constexpr u32 CACHING_PAGEBITS = 12;
#else
    static constexpr u32 CACHING_PAGEBITS = 16;
#endif
    static constexpr u64 CACHING_PAGESIZE = u64{1} << CACHING_PAGEBITS;

    static constexpr bool IS_OPENGL = P::IS_OPENGL;
    static constexpr bool HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS =
        P::HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS;
    static constexpr bool HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT =
        P::HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT;
    static constexpr bool NEEDS_BIND_UNIFORM_INDEX = P::NEEDS_BIND_UNIFORM_INDEX;
    static constexpr bool NEEDS_BIND_STORAGE_INDEX = P::NEEDS_BIND_STORAGE_INDEX;
    static constexpr bool USE_MEMORY_MAPS = P::USE_MEMORY_MAPS;
    static constexpr bool SEPARATE_IMAGE_BUFFERS_BINDINGS = P::SEPARATE_IMAGE_BUFFER_BINDINGS;
    static constexpr bool USE_MEMORY_MAPS_FOR_UPLOADS = P::USE_MEMORY_MAPS_FOR_UPLOADS;

#ifdef YUZU_LEGACY
    static constexpr s64 TARGET_THRESHOLD = 3_GiB;
#else
    static constexpr s64 TARGET_THRESHOLD = 4_GiB;
#endif

    static constexpr s64 DEFAULT_EXPECTED_MEMORY = 512_MiB;
    static constexpr s64 DEFAULT_CRITICAL_MEMORY = 1_GiB;

    // Debug Flags.

    static constexpr bool DISABLE_DOWNLOADS = true;

    using Maxwell = Tegra::Engines::Maxwell3D::Regs;

    using Runtime = typename P::Runtime;
    using Buffer = typename P::Buffer;
    using Async_Buffer = typename P::Async_Buffer;
    using MemoryTracker = typename P::MemoryTracker;

    struct OverlapResult {
        boost::container::small_vector<BufferId, 16> ids;
        DAddr begin;
        DAddr end;
        bool has_stream_leap = false;
    };

public:
    explicit BufferCache(Tegra::MaxwellDeviceMemoryManager& device_memory_, Runtime& runtime_);

    ~BufferCache();

    void TickFrame();

    bool BindMultiRangeStorage(const Binding& binding, bool is_written,
                               std::span<const MultiRangeSegment> pool,
                               u64 write_mark_limit = ~u64{0});

    void ResolveMultiRangeStorage(Binding& binding, bool is_written,
                                  std::vector<MultiRangeSegment>& pool);

    void RefreshMultiRangeSegments(std::vector<MultiRangeSegment>& pool);

    void UnmapGPUMemory(size_t as_id, GPUVAddr gpu_addr, size_t size);

    /// `keep_page_gpu_data`: download GPU-written bytes sharing the written pages first (false
    /// when the memory is going away).
    void WriteMemory(DAddr device_addr, u64 size, bool keep_page_gpu_data = true);

    void CachedWriteMemory(DAddr device_addr, u64 size);

    /// Cancels pending downloads over the range, so a fence can't write them over newer bytes.
    void ClearDownload(DAddr base_addr, u64 size);

    /// CPU writes collected but not yet applied (SetPendingCpuWrites).
    [[nodiscard]] Common::RangeSet<DAddr> PendingCpuWriteRanges();

    /// Writes downloaded bytes to guest memory, except over `cpu_written`.
    void WriteDownloadedMemory(const Common::RangeSet<DAddr>& cpu_written, DAddr device_addr,
                               const u8* data, u64 size);

    bool OnCPUWrite(DAddr device_addr, u64 size);

    void DownloadMemory(DAddr device_addr, u64 size);

    std::optional<VideoCore::RasterizerDownloadArea> GetFlushArea(DAddr device_addr, u64 size);

    /// Without downloads at every fence, a guest CPU read is the only way GPU-written buffer
    /// data reaches guest memory, so reads must ask the buffer cache (GetFlushArea).
    bool FlushesOnCpuRead() const {
        return !AsyncBufferDownloads();
    }

    /// `pages_clean` (optional): set when no page the write touches holds GPU-written data.
    bool InlineMemory(DAddr dest_address, size_t copy_size, std::span<const u8> inlined_buffer,
                      bool* pages_clean = nullptr);

    void BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr, u32 size);

    void DisableGraphicsUniformBuffer(size_t stage, u32 index);

    void UpdateGraphicsBuffers(bool is_indexed);

    void UpdateComputeBuffers();

    void BindHostGeometryBuffers(bool is_indexed);

    void BindHostStageBuffers(size_t stage);

    void BindHostComputeBuffers();

    void SetUniformBuffersState(const std::array<u32, NUM_STAGES>& mask,
                                const UniformBufferSizes* sizes);

    void SetComputeUniformBufferState(u32 mask, const ComputeUniformBufferSizes* sizes);

    void UnbindGraphicsStorageBuffers(size_t stage);

    bool BindGraphicsStorageBuffer(size_t stage, size_t ssbo_index, u32 cbuf_index, u32 cbuf_offset,
                                   bool is_written, u32 indirect_offset = ~0U);

    void UnbindGraphicsTextureBuffers(size_t stage);

    void BindGraphicsTextureBuffer(size_t stage, size_t tbo_index, GPUVAddr gpu_addr, u32 size,
                                   PixelFormat format, bool is_written, bool is_image);

    void UnbindComputeStorageBuffers();

    /// write_extent: how far the program's stores reach past the pointer (native programs).
    void BindComputeStorageBuffer(size_t ssbo_index, u32 cbuf_index, u32 cbuf_offset,
                                  bool is_written, u32 indirect_offset = ~0U,
                                  const Shader::StorageWriteExtent& write_extent = {});

    void UnbindComputeTextureBuffers();

    void BindComputeTextureBuffer(size_t tbo_index, GPUVAddr gpu_addr, u32 size, PixelFormat format,
                                  bool is_written, bool is_image);

    [[nodiscard]] std::pair<Buffer*, u32> ObtainBuffer(GPUVAddr gpu_addr, u32 size,
                                                       ObtainBufferSynchronize sync_info,
                                                       ObtainBufferOperation post_op);

    [[nodiscard]] std::pair<Buffer*, u32> ObtainCPUBuffer(DAddr gpu_addr, u32 size,
                                                          ObtainBufferSynchronize sync_info,
                                                          ObtainBufferOperation post_op);
    void FlushCachedWrites();

    /// Where the CPU writes into GPU-written memory that were collected but not applied yet
    /// (WriteMemory) come from: downloads must not write the GPU's older copy over them.
    using PendingCpuWrites = std::function<void(const std::function<void(DAddr, size_t)>&)>;
    void SetPendingCpuWrites(PendingCpuWrites source) {
        pending_cpu_writes = std::move(source);
    }

    /// Takes CPU writes about to be applied one by one out of the GPU-written ranges first, so
    /// applying one doesn't download the GPU's copy over another (WriteMemory keeps the GPU data
    /// on the pages it writes).
    void ClaimCpuWrites(std::span<const std::pair<DAddr, u64>> ranges);

    /// Return true when there are uncommitted buffers to be downloaded
    [[nodiscard]] bool HasUncommittedFlushes() const noexcept;

    void AccumulateFlushes();

    /// Return true when the caller should wait for async downloads
    [[nodiscard]] bool ShouldWaitAsyncFlushes() const noexcept;

    /// Commit asynchronous downloads
    void CommitAsyncFlushes();
    void CommitAsyncFlushesHigh();

    /// Pop asynchronous downloads
    void PopAsyncFlushes();
    void PopAsyncBuffers();

    bool DMACopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount);

    bool DMAClear(GPUVAddr src_address, u64 amount, u32 value);

    /// Switch 2 GPU-written command lists (nx2.h, g_nx2_command_list_pages). While collecting (a
    /// compute dispatch or a DMA copy being recorded), writes to command list pages are noted;
    /// Nx2CopyListPages then records copies of those pages into host memory after the work.
    void Nx2CollectListCopies(bool collect) {
        nx2_collect_list_copies = collect;
        if (!collect) {
            nx2_list_copy_requests.clear();
        }
    }
    /// True when it recorded copies, which the GPU completes at `tick` (the caller submits).
    bool Nx2CopyListPages(u64 tick);
    enum class Nx2ListRead { Unknown, Copied, Synced, NeedsSync };
    /// Front thread: Synced when guest memory holds the page `device_addr` lies in (flushed and
    /// written by no GPU work since), else Copied with `out` filled from the copy of its last
    /// write once the GPU is done with it. Unknown otherwise: the caller drains and flushes.
    /// Pages draws write are only Synced or Unknown, and NeedsSync until `past_sync`: the
    /// split's back ran the work before the last host-level wait (GpuSplit::WaitForSync).
    Nx2ListRead Nx2ReadListCopy(DAddr device_addr, std::span<u8> out, bool past_sync);
    /// Front thread, the split drained: the command list page was flushed to guest memory.
    void Nx2NoteListSynced(DAddr page);

    void Nx2FlushListPages(std::span<const DAddr> pages);
    /// CPU and inline writes: they reach guest memory, so a synced page stays synced, but its copy
    /// no longer holds the page.
    void Nx2NoteListHostWrite(DAddr device_addr, u64 size);

    /// Return true when a CPU region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(DAddr addr, size_t size);

    /// Visits the uniform and storage buffers bound for the last draw or dispatch with the cached
    /// host buffer holding each range, if one does, for diagnostics. Nothing is
    /// created or synchronized. Stage 5 is compute.
    template <typename Func>
    void ForEachBoundBuffer(bool compute, Func&& func) {
        const auto visit = [&](u32 stage, bool storage, u32 index, bool written,
                               const Binding& binding, u32 size) {
            if (binding.device_addr == 0 || size == 0) {
                return;
            }
            const Buffer* host = nullptr;
            u32 offset = 0;
            if (const BufferId id = page_table[binding.device_addr >> CACHING_PAGEBITS]) {
                const Buffer& buffer = slot_buffers[id];
                if (buffer.IsInBounds(binding.device_addr, size)) {
                    host = &buffer;
                    offset = buffer.Offset(binding.device_addr);
                }
            }
            func(stage, storage, index, written, binding, size, host, offset);
        };
        if (compute) {
            const auto* const sizes = channel_state->compute_uniform_buffer_sizes;
            ForEachEnabledBit(channel_state->enabled_compute_uniform_buffer_mask, [&](u32 index) {
                const Binding& binding = channel_state->compute_uniform_buffers[index];
                visit(5, false, index, false, binding,
                      sizes ? std::min(binding.size, (*sizes)[index]) : binding.size);
            });
            ForEachEnabledBit(channel_state->enabled_compute_storage_buffers, [&](u32 index) {
                const Binding& binding = channel_state->compute_storage_buffers[index];
                visit(5, true, index, ((channel_state->written_compute_storage_buffers >> index) & 1) != 0,
                      binding, binding.size);
            });
            return;
        }
        const auto* const sizes = channel_state->uniform_buffer_sizes;
        for (u32 stage = 0; stage < NUM_STAGES; ++stage) {
            ForEachEnabledBit(channel_state->enabled_uniform_buffer_masks[stage], [&](u32 index) {
                const Binding& binding = channel_state->uniform_buffers[stage][index];
                visit(stage, false, index, false, binding,
                      sizes ? std::min(binding.size, (*sizes)[stage][index]) : binding.size);
            });
            ForEachEnabledBit(channel_state->enabled_storage_buffers[stage], [&](u32 index) {
                const Binding& binding = channel_state->storage_buffers[stage][index];
                visit(stage, true, index,
                      ((channel_state->written_storage_buffers[stage] >> index) & 1) != 0, binding,
                      binding.size);
            });
        }
    }

    /// Return true when a region is registered on the cache
    [[nodiscard]] bool IsRegionRegistered(DAddr addr, size_t size);

    /// Return true when a CPU region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(DAddr addr, size_t size);

    void SetDrawIndirect(
        const Tegra::Engines::Maxwell3D::DrawManager::IndirectParams* current_draw_indirect_) {
        current_draw_indirect = current_draw_indirect_;
    }

    [[nodiscard]] std::pair<Buffer*, u32> GetDrawIndirectCount();

    [[nodiscard]] std::pair<Buffer*, u32> GetDrawIndirectBuffer();

    template <typename Func>
    void BufferOperations(Func&& func) {
        do {
            channel_state->has_deleted_buffers = false;
            func();
        } while (channel_state->has_deleted_buffers);
    }

    Common::SpinRecursiveMutex mutex;
    Runtime& runtime;
    bool any_buffer_uploaded = false;
    /// Bumped whenever gpu_modified_ranges gains a range: a page found free of GPU-written data
    /// stays free while it is unchanged (the rasterizer's inline writes).
    std::atomic<u64> gpu_modified_serial{};
    /// Set once a range went into gpu_modified_ranges without marking its pages in
    /// Tegra::g_nx2_gpu_written_pages (outside Switch 2 sessions): the bitmap no longer bounds it.
    bool nx2_gpu_ranges_unmarked = false;
    /// Indirectly read banks of native programs already uploaded this frame.
    Nx2BufCache::FrameAddressSet native_indirect_uniforms_this_frame;

    /// EDEN_NX2_REACTIVE_ASYNC_FLUSH: queues the GPU-written parts of a range for download at the
    /// next fence instead of flushing them now. Returns whether anything was queued.
    bool Nx2QueueReadDownload(DAddr device_addr, u64 size) {
        if (!memory_tracker.IsRegionGpuModified(device_addr, size)) {
            return false; // nothing CommitAsyncFlushesHigh would copy: flush the old way
        }
        bool queued = false;
        gpu_modified_ranges.ForEachInRange(device_addr, size, [&](DAddr lower, DAddr upper) {
            uncommitted_gpu_modified_ranges.Add(lower, upper - lower);
            queued = true;
        });
        return queued;
    }

private:
    template <typename Mask, typename Func>
    static void ForEachEnabledBit(Mask enabled_mask, Func&& func) {
        for (u32 index = 0; enabled_mask != 0; ++index, enabled_mask >>= 1) {
            const int disabled_bits = std::countr_zero(enabled_mask);
            index += disabled_bits;
            enabled_mask >>= disabled_bits;
            func(index);
        }
    }

    template <typename Func>
    void ForEachBufferInRange(DAddr device_addr, u64 size, Func&& func) {
        const u64 page_end = Common::DivCeil(device_addr + size, CACHING_PAGESIZE);
        for (u64 page = device_addr >> CACHING_PAGEBITS; page < page_end;) {
            const BufferId buffer_id = page_table[page];
            if (!buffer_id) {
                ++page;
                continue;
            }
            Buffer& buffer = slot_buffers[buffer_id];
            func(buffer_id, buffer);

            const DAddr end_addr = buffer.CpuAddr() + buffer.SizeBytes();
            page = Common::DivCeil(end_addr, CACHING_PAGESIZE);
        }
    }

    static bool IsRangeGranular(DAddr device_addr, size_t size) {
        return (device_addr & ~Core::DEVICE_PAGEMASK) ==
               ((device_addr + size) & ~Core::DEVICE_PAGEMASK);
    }

    void RunGarbageCollector(bool aggressive_gc);

    void BindHostIndexBuffer();

    void BindHostVertexBuffers();

    void BindHostDrawIndirectBuffers();

    void BindHostGraphicsUniformBuffers(size_t stage);

    void BindHostGraphicsUniformBuffer(size_t stage, u32 index, u32 binding_index, bool needs_bind);

    /// Counts what a streamed uniform buffer bind did (EDEN_NX2_GPU_PROFILE).
    void NoteUniformReuse(VideoCommon::UniformReuseResult result, u32 size);

    void BindHostGraphicsStorageBuffers(size_t stage);

    void BindHostGraphicsTextureBuffers(size_t stage);

    void BindHostTransformFeedbackBuffers();

    void BindHostVertexBuffer(u32 index, Buffer& buffer, u32 offset, u32 size, u32 stride);

    void BindHostComputeUniformBuffers();

    void BindHostComputeStorageBuffers();

    void BindHostComputeTextureBuffers();

    void DoUpdateGraphicsBuffers(bool is_indexed);

    void DoUpdateComputeBuffers();

    void UpdateIndexBuffer();

    void UpdateVertexBuffers();

    void UpdateVertexBuffer(u32 index);

    void UpdateDrawIndirect();

    void UpdateUniformBuffers(size_t stage);

    void UpdateStorageBuffers(size_t stage);

    void UpdateTextureBuffers(size_t stage);

    void UpdateTransformFeedbackBuffers();

    void UpdateTransformFeedbackBuffer(u32 index);

    void UpdateComputeUniformBuffers();

    void UpdateComputeStorageBuffers();

    void UpdateComputeTextureBuffers();

    /// The buffers and multi-range segments a stage's storage bindings resolved to, kept while the
    /// bindings, the GPU mappings and the set of buffers (binding_epoch) stay the same.
    struct StorageResolveCache {
        u64 epoch{}; ///< binding_epoch when stored; 0 never matches
        u64 map_serial{};
        size_t as_id{};
        const void* channel{};
        u64 enabled{};
        u64 written{};
        std::array<Binding, NUM_STORAGE_BUFFERS> bindings{};
        std::vector<MultiRangeSegment> segments;
    };

    /// Restores the stage's resolved storage bindings from the cache when nothing they depend on
    /// changed (EDEN_NX2_BINDCACHE). False: resolve them, then StoreStorageResolve.
    bool ReuseStorageResolve(StorageResolveCache& cache, std::span<Binding> bindings, u64 enabled,
                             u64 written, std::vector<MultiRangeSegment>& pool);

    void StoreStorageResolve(StorageResolveCache& cache, std::span<const Binding> bindings,
                             u64 enabled, u64 written,
                             const std::vector<MultiRangeSegment>& pool, u64 start_epoch);

    void MarkWrittenBuffer(BufferId buffer_id, DAddr device_addr, u32 size, bool needs_sync = false);

    /// Counts a GPU write to the command list pages in the range (invalidating their copies) and,
    /// while collecting, asks for new copies.
    void Nx2NoteListWrite(BufferId buffer_id, DAddr device_addr, u64 size);

    [[nodiscard]] BufferId FindBuffer(DAddr device_addr, u32 size, bool sparse_compatible);

    void SynchronizeBufferWrites(Buffer& buffer);

    [[nodiscard]] OverlapResult ResolveOverlaps(DAddr device_addr, u32 wanted_size);

    void JoinOverlap(BufferId new_buffer_id, BufferId overlap_id, bool accumulate_stream_score);

    [[nodiscard]] BufferId CreateBuffer(DAddr device_addr, u32 wanted_size,
                                        bool sparse_compatible);

    void Register(BufferId buffer_id);

    void Unregister(BufferId buffer_id);

    template <bool insert>
    void ChangeRegister(BufferId buffer_id);

    void TouchBuffer(Buffer& buffer, BufferId buffer_id) noexcept;

    bool SynchronizeBuffer(Buffer& buffer, DAddr device_addr, u32 size);

    void UploadMemory(Buffer& buffer, u64 total_size_bytes, u64 largest_copy,
                      std::span<BufferCopy> copies);

    void ImmediateUploadMemory(Buffer& buffer, u64 largest_copy,
                               std::span<const BufferCopy> copies);

    void MappedUploadMemory(Buffer& buffer, u64 total_size_bytes, std::span<BufferCopy> copies);

    void DownloadBufferMemory(Buffer& buffer_id);

    void DownloadBufferMemory(Buffer& buffer_id, DAddr device_addr, u64 size);

    /// Downloads the GPU-written data of several buffers with one wait for the GPU.
    void DownloadBuffersMemory(std::span<const BufferId> buffer_ids);

    void DownloadRangesMemory(std::span<const std::pair<DAddr, u64>> ranges);

    void DeleteBuffer(BufferId buffer_id, bool do_not_mark = false);

    [[nodiscard]] Binding StorageBufferBinding(GPUVAddr ssbo_addr, u32 cbuf_index, bool is_written,
                                               bool native_pointer = false,
                                               bool table_pointer = false) const;

    void TraceTablePointer(bool compute, size_t stage, size_t ssbo_index, u32 cbuf_index,
                           u32 cbuf_offset, u32 indirect_offset, GPUVAddr holder,
                           const Binding& binding);

    [[nodiscard]] TextureBufferBinding GetTextureBufferBinding(GPUVAddr gpu_addr, u32 size,
                                                               PixelFormat format);

    [[nodiscard]] std::span<const u8> ImmediateBufferWithData(DAddr device_addr, size_t size);

    [[nodiscard]] std::span<u8> ImmediateBuffer(size_t wanted_capacity);

    [[nodiscard]] bool HasFastUniformBufferBound(size_t stage, u32 binding_index) const noexcept;

    [[nodiscard]] Binding& VertexBufferSlot(u32 index);

    [[nodiscard]] const Binding& VertexBufferSlot(u32 index) const;

    void UpdateVertexBufferSlot(u32 index, const Binding& binding);

    /// Flags the vertex buffers overlapping a written range so host copies derived from them are
    /// rebuilt.
    void MarkVertexBuffersStale(DAddr device_addr, u64 size);

    /// Re-dirties repacked vertex buffers whose source data was written since they were bound.
    void RefreshRepackedVertexBuffers();

    void InlineMemoryImplementation(DAddr dest_address, size_t copy_size,
                                    std::span<const u8> inlined_buffer);

    /// The guest bytes of a native 64 KiB uniform bank at its last once-a-frame upload
    /// (EDEN_NX2_BUFCACHE_BANKSHADOW). Bit k of `valid`: device page k of the bank still holds
    /// what the host buffer holds there; other host writes over the page clear it.
    struct BankShadow {
        std::array<u8, 0x10000> bytes{};
        u32 valid{};
        u64 compared_frame{}; ///< frame_tick of the last compare (EDEN_NX2_BANK_LRU)
    };

    /// The once-a-frame upload of a native uniform bank: only the pages whose guest bytes
    /// differ from what was uploaded last time.
    void UploadNativeBank(Buffer& buffer, DAddr device_addr, u64 upload_size);

    /// Host buffer writes other than UploadNativeBank over [device_addr, device_addr + size).
    void InvalidateBankShadows(DAddr device_addr, u64 size);

    /// UploadHotPages wrote `bytes` over the device page `page`: bank shadow pages that are
    /// exactly this page take them as what the host holds (EDEN_NX2_SHADOW_REFRESH), others are
    /// invalidated.
    void RefreshBankShadows(DAddr page, const u8* bytes);

    /// UploadNativeBank wrote `bytes` over [device_addr, device_addr + size): hot pages wholly
    /// inside take them as what the host holds (EDEN_NX2_SHADOW_REFRESH), others are invalidated.
    void RefreshHotShadows(DAddr device_addr, const u8* bytes, u64 size);

    /// Whether a bank shadow covers part of [device_addr, device_addr + size).
    [[nodiscard]] bool OverlapsBankShadow(DAddr device_addr, u64 size) const;

    /// A host upload wrote `bytes` over [device_addr, device_addr + size): bank shadow pages it
    /// wholly covers take them as what the host holds (EDEN_NX2_BANK_SYNC_REFRESH), pages it
    /// partly covers are invalidated.
    void RefreshBankShadowsFromUpload(DAddr device_addr, const u8* bytes, u64 size);

    /// MappedUploadMemory for SynchronizeBuffer when bank shadows exist: a copy of up to 256 KiB
    /// over a shadow reads guest memory once into a cached scratch, which feeds both the staging
    /// buffer and RefreshBankShadowsFromUpload; a larger one invalidates the shadow pages.
    void MappedUploadMemoryRefreshingBanks(Buffer& buffer, u64 total_size_bytes,
                                           std::span<BufferCopy> copies);

    std::map<DAddr, std::unique_ptr<BankShadow>> bank_shadows;
    std::vector<u8> bank_refresh_scratch;
    /// Device pages a bank compare read this frame (EDEN_NX2_GPU_PROFILE only: overlap count).
    Nx2BufCache::FrameAddressSet bank_pages_this_frame;

    /// A range SynchronizeBuffer left clean (EDEN_NX2_SYNC_SKIP): clean while no CPU-dirty mark
    /// fell in it (the 256 KiB regions it covers when it covers at most MAX_SKIP_REGIONS of them,
    /// else its buffer: Buffer::cpu_dirty_gen), no mark not tied to a range (cpu_dirty_epoch) and
    /// no buffer came or went. `hot_mask` (EDEN_NX2_HOT_SYNC_SKIP): the hot pages in the range,
    /// bit k for the k-th device page from the range's first; they are always dirty, and a hit
    /// compares them.
    struct SyncSkip {
        const Buffer* buffer{};
        DAddr device_addr{};
        u32 size{};
        u64 gen{};
        u64 epoch{};
        u64 binding_epoch{};
        u64 region_sum{};
        u64 hot_mask{};
    };
    static constexpr u32 SYNC_SKIP_BITS = 10;
    std::array<SyncSkip, size_t{1} << SYNC_SKIP_BITS> sync_skip{};
    /// Bumped by CPU-dirty marks not tied to one range (cached writes, deleted buffers).
    u64 cpu_dirty_epoch = 1;

    /// CPU-dirty marks per 256 KiB device region (EDEN_NX2_SKIP_REGIONS), regions folded onto
    /// SKIP_REGION_SLOTS counters (a shared counter only drops more entries). The counters only
    /// grow, so their sum over a range is unchanged exactly when none of them changed.
    static constexpr u32 SKIP_REGION_BITS = 18;
    static constexpr size_t SKIP_REGION_SLOTS = 4096;
    static constexpr u64 MAX_SKIP_REGIONS = 64;
    std::array<u64, SKIP_REGION_SLOTS> skip_region_gen{};

    /// The number of regions [device_addr, device_addr + size) covers (1 for an empty range).
    [[nodiscard]] static u64 SkipRegionCount(DAddr device_addr, u64 size) noexcept {
        const u64 first = device_addr >> SKIP_REGION_BITS;
        const u64 last = (device_addr + (size == 0 ? 0 : size - 1)) >> SKIP_REGION_BITS;
        return last - first + 1;
    }

    [[nodiscard]] u64 SkipRegionSum(DAddr device_addr, u64 size) const noexcept {
        const u64 first = device_addr >> SKIP_REGION_BITS;
        const u64 count = SkipRegionCount(device_addr, size);
        u64 sum = 0;
        for (u64 region = first; region != first + count; ++region) {
            sum += skip_region_gen[region % SKIP_REGION_SLOTS];
        }
        return sum;
    }

    void BumpSkipRegions(DAddr device_addr, u64 size) noexcept {
        const u64 first = device_addr >> SKIP_REGION_BITS;
        const u64 count = SkipRegionCount(device_addr, size);
        if (count >= SKIP_REGION_SLOTS) {
            for (u64& gen : skip_region_gen) {
                ++gen;
            }
            return;
        }
        for (u64 region = first; region != first + count; ++region) {
            ++skip_region_gen[region % SKIP_REGION_SLOTS];
        }
    }

    /// CPU-dirty marks over [device_addr, device_addr + size): their buffers' syncs can't be
    /// skipped until they ran again.
    void NoteCpuDirty(DAddr device_addr, u64 size);

    /// A SYNC_SKIP hit on a range holding hot pages (`hot_mask` from the range's first device
    /// page): compares and uploads them. Empty when one of them needs the full walk.
    [[nodiscard]] std::optional<bool> SyncSkippedHotPages(Buffer& buffer, DAddr device_addr,
                                                          u64 hot_mask);

    /// A page the guest rewrites between most of its syncs (EDEN_NX2_HOT_PAGES): it stays
    /// unprotected and CPU-dirty (Type::Hot in the tracker), and each sync compares its guest
    /// bytes with `bytes`, which the host buffer holds while `valid`, uploading only changes.
    struct HotPage {
        std::array<u8, Core::DEVICE_PAGESIZE> bytes{};
        bool valid{};
        u32 unchanged{};    ///< Syncs since the bytes last changed
        u64 changed_frame{};
    };
    struct PageHeat {
        u32 heat{};
        u64 frame{};
        u64 no_promote_until{}; ///< Demoted for its cost: not promoted before this frame
    };
    static constexpr size_t MAX_HOT_PAGES = 2048;
    static constexpr u32 HOT_PAGE_HEAT = 16;
    /// EDEN_NX2_HOT_DEMOTE: unchanged compares in a row that cost more than the guest fault and
    /// re-protection the hot page saves (~0.4 us a compare against several us a fault).
    static constexpr u32 HOT_DEMOTE_UNCHANGED = 8;
    static constexpr u64 HOT_COOLDOWN_FRAMES = 512;
    std::unordered_map<DAddr, std::unique_ptr<HotPage>> hot_pages;
    std::unordered_map<DAddr, PageHeat> page_heat;
    std::vector<DAddr> hot_changed_pages;
    std::vector<DAddr> hot_promote_pages;
    std::vector<DAddr> hot_demote_pages;

    /// Counts the syncs that found these (small) ranges dirty; pages dirty at most of them
    /// become hot.
    void NoteDirtyPages(DAddr device_addr, u64 size);

    /// A hot page reached by a sync: false when it must be uploaded from guest memory as usual.
    bool SyncHotPage(DAddr page);

    void ApplyHotPageChanges();

    /// Uploads hot_changed_pages from their copies.
    void UploadHotPages(Buffer& buffer);

    /// GPU writes: hot pages they touch are tracked as usual again.
    void DemoteHotPages(DAddr device_addr, u64 size);

    /// Host writes other than UploadHotPages: the copies no longer say what the host holds.
    void InvalidateHotShadows(DAddr device_addr, u64 size);

    Tegra::MaxwellDeviceMemoryManager& device_memory;

    Common::SlotVector<Buffer> slot_buffers;
#ifdef YUZU_LEGACY
    static constexpr size_t TICKS_TO_DESTROY = 6;
#else
    static constexpr size_t TICKS_TO_DESTROY = 8;
#endif
    DelayedDestructionRing<Buffer, TICKS_TO_DESTROY> delayed_destruction_ring;

    const Tegra::Engines::Maxwell3D::DrawManager::IndirectParams* current_draw_indirect{};

    u32 last_index_count = 0;

    u32 enabled_vertex_buffers_mask = 0;
    u32 stale_vertex_buffers_mask = 0;
    u64 vertex_buffers_serial = 0;
    std::array<Binding, 32> v_buffer{};

    boost::container::small_vector<BufferCopy, 4> upload_copies;

    MemoryTracker memory_tracker;
    Common::RangeSet<DAddr> uncommitted_gpu_modified_ranges;

    /// Whether GPU-written buffer ranges are copied back to guest memory at the next fence;
    /// otherwise only when something flushes them (as upstream yuzu below High GPU accuracy).
    /// Not for Switch 2 titles: title 1 had 250-330 MB/s copied back (a fence per compute
    /// dispatch, the fencing thread holding the buffer cache while it wrote them), while its CPU
    /// read GPU-written memory at most 64 times in 200 s (counted with reactive flushing) and
    /// the native MME's DMA reads never touched those pages. Without them a fence costs ~1 us
    /// instead of ~5 and draw setup ~7% less. EDEN_NX2_BUFFER_DOWNLOADS=0/1 forces it.
    bool AsyncBufferDownloads() const {
        static const int forced = [] {
            const char* value = std::getenv("EDEN_NX2_BUFFER_DOWNLOADS");
            return value == nullptr ? -1 : (value[0] == '0' ? 0 : 1);
        }();
        if (forced >= 0) {
            return forced == 1;
        }
        return gpu_memory == nullptr || !gpu_memory->UsesNativeTiling();
    }

    /// Whether a GPU write to this range is copied back at the next fence. Ranges the guest CPU
    /// has read (Preflushable, set by the first reactive flush) always are: later reads skip the
    /// flush and rely on these downloads, e.g. collision meshes skinned by compute shaders.
    bool DownloadsAtFence(DAddr device_addr, u64 size) noexcept {
        return AsyncBufferDownloads() || memory_tracker.IsRegionPreflushable(device_addr, size);
    }

    /// Queues a GPU write for download at the next fence where DownloadsAtFence holds. With only
    /// Preflushable marks, just the marked pages are queued: a write that covers one marked page
    /// of a large buffer would otherwise copy the whole buffer back at every fence.
    void QueueFenceDownload(DAddr device_addr, u64 size) {
        static const bool exact = [] {
            const char* value = std::getenv("EDEN_NX2_EXACT_PREFLUSH");
            return value == nullptr || value[0] != '0';
        }();
        if (AsyncBufferDownloads()) {
            uncommitted_gpu_modified_ranges.Add(device_addr, size);
            return;
        }
        if (!exact) {
            if (memory_tracker.IsRegionPreflushable(device_addr, size)) {
                uncommitted_gpu_modified_ranges.Add(device_addr, size);
            } else {
                PrefetchCpuReadRanges(device_addr, size);
                ArmReadTraps(device_addr, size);
            }
            return;
        }
        const DAddr end = device_addr + size;
        u64 queued = 0;
        memory_tracker.ForEachPreflushableRange(device_addr, size, [&](DAddr begin, u64 length) {
            const DAddr lower = (std::max)(begin, device_addr);
            const DAddr upper = (std::min)(begin + length, end);
            if (lower < upper) {
                uncommitted_gpu_modified_ranges.Add(lower, upper - lower);
                queued += upper - lower;
            }
        });
        if (queued < size) {
            PrefetchCpuReadRanges(device_addr, size);
            ArmReadTraps(device_addr, size);
        }
        if (Nx2DownloadTrace::g_enabled && queued != 0) [[unlikely]] {
            Nx2DownloadTrace::Queued(device_addr, size, queued);
        }
    }

    /// EDEN_NX2_REACTIVE_PREFETCH: the parts of a trapped GPU write over pages the guest CPU has
    /// read GPU data from are also copied back at the next fence. The trap stays armed, so a read
    /// before that copy reaches guest memory still flushes; a later one finds nothing to flush.
    void PrefetchCpuReadRanges(DAddr device_addr, u64 size) {
        if (!Nx2Reactive::Prefetch()) {
            return;
        }
        RollCpuReadRanges();
        if (cpu_read_ranges.Empty() && cpu_read_ranges_previous.Empty()) {
            return;
        }
        const auto queue = [&](DAddr lower, DAddr upper) {
            uncommitted_gpu_modified_ranges.Add(lower, upper - lower);
        };
        cpu_read_ranges.ForEachInRange(device_addr, size, queue);
        cpu_read_ranges_previous.ForEachInRange(device_addr, size, queue);
    }

    /// Forgets the pages the guest CPU has not read GPU data from for a while
    /// (Nx2Reactive::HotFrames): cpu_read_ranges holds this period's reads, the previous
    /// period's are kept one more period.
    void RollCpuReadRanges() {
        const u64 frames = Nx2Reactive::HotFrames();
        if (frames == 0) {
            return;
        }
        const u64 epoch = Nx2Reactive::g_frame.load(std::memory_order_relaxed) / frames;
        if (epoch == cpu_read_epoch) {
            return;
        }
        if (epoch == cpu_read_epoch + 1) {
            cpu_read_ranges_previous = std::move(cpu_read_ranges);
        } else {
            cpu_read_ranges_previous.Clear();
            cpu_read_ranges.Clear();
        }
        cpu_read_epoch = epoch;
    }

    /// Whether every GPU-written byte in [device_addr, device_addr + size) has a download
    /// recorded that is still to land (Nx2Reactive::WaitLanding).
    [[nodiscard]] bool GpuWritesDownloading(DAddr device_addr, u64 size) const {
        if (pending_download_ranges.empty()) {
            return false;
        }
        Common::RangeSet<DAddr> missing;
        gpu_modified_ranges.ForEachInRange(device_addr, size, [&](DAddr lower, DAddr upper) {
            missing.Add(lower, upper - lower);
        });
        if (missing.Empty()) {
            return false;
        }
        for (const Common::RangeSet<DAddr>& ranges : pending_download_ranges) {
            ranges.ForEachInRange(device_addr, size, [&](DAddr lower, DAddr upper) {
                missing.Subtract(lower, upper - lower);
            });
        }
        return missing.Empty();
    }

    /// GPU writes that are not downloaded at fences: trap the guest CPU's next read of their
    /// pages, which flushes them on demand (or marks them Preflushable with preflushing enabled).
    void ArmReadTraps(DAddr device_addr, u64 size) {
        const DAddr begin = Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE);
        const DAddr end = Common::AlignUp(device_addr + size, Core::DEVICE_PAGESIZE);
        if (!read_trapped_ranges.Arm(begin, end - begin)) {
            return;
        }
        device_memory.ArmReadTraps(begin, end - begin);
    }

    ReadTrapTracker read_trapped_ranges;
    /// Ranges the guest CPU read after GPU writes. A read that must flush downloads the GPU
    /// writes in all of them, so the reads that usually follow it find nothing left to flush.
    Common::RangeSet<DAddr> cpu_read_ranges;
    Common::RangeSet<DAddr> cpu_read_ranges_previous; ///< Read in the previous period only
    u64 cpu_read_epoch{};                             ///< Period of cpu_read_ranges (HotFrames)
    bool download_cpu_read_ranges{};
    Common::RangeSet<DAddr> gpu_modified_ranges;
    std::deque<Common::RangeSet<DAddr>> committed_gpu_modified_ranges;

    // Async Buffers
    std::deque<std::optional<Async_Buffer>> async_buffers;
    std::deque<boost::container::small_vector<BufferCopy, 4>> pending_downloads;
    /// For each entry of pending_downloads: the ranges it still writes to guest memory when its
    /// fence pops. A write to a range after a download was taken removes it from that download
    /// (and every other pending one): its bytes are older than the write.
    std::deque<Common::RangeSet<DAddr>> pending_download_ranges;
    PendingCpuWrites pending_cpu_writes;
    std::optional<Async_Buffer> current_buffer;

    std::deque<Async_Buffer> async_buffers_death_ring;

    size_t immediate_buffer_capacity = 0;
    Common::ScratchBuffer<u8> immediate_buffer_alloc;
    std::vector<u8> guest_uint8_indices;
    /// Guest memory large uploads came from since the last cold hint (device address, size).
    std::vector<std::pair<DAddr, u64>> uploaded_buffer_ranges;

    struct LRUItemParams {
        using ObjectType = BufferId;
        using TickType = u64;
    };
    Common::LeastRecentlyUsedCache<LRUItemParams> lru_cache;
    VirtualRangeCache virtual_ranges;
    std::vector<MultiRangeSegment> graphics_segments;
    std::vector<MultiRangeSegment> compute_segments;
    /// Bumped whenever a buffer is created, deleted or joined: resolved bindings may be stale.
    u64 binding_epoch = 1;
    std::array<StorageResolveCache, NUM_STAGES> graphics_storage_resolve{};
    StorageResolveCache compute_storage_resolve{};

    /// One storage binding's last lookup when it resolved to a single buffer (no segments), kept
    /// on the same terms as StorageResolveCache (EDEN_NX2_STORAGE_MEMO). Index NUM_STAGES is
    /// compute.
    struct StorageBindingMemo {
        u64 epoch{}; ///< binding_epoch when stored; 0 never matches
        u64 map_serial{};
        size_t as_id{};
        const void* channel{};
        DAddr device_addr{};
        GPUVAddr gpu_addr{};
        u32 size{};
        bool is_written{};
        BufferId buffer_id{};
    };
    std::array<std::array<StorageBindingMemo, NUM_STORAGE_BUFFERS>, NUM_STAGES + 1>
        storage_binding_memo{};

    /// Resolves one storage binding (FindBuffer, then ResolveMultiRangeStorage), from its memo
    /// when nothing it depends on changed.
    void ResolveStorageBinding(size_t memo_stage, u32 index, Binding& binding, bool is_written,
                               std::vector<MultiRangeSegment>& pool);

    /// A vertex or index buffer binding and what it was resolved from: the same GPU range, the
    /// same mappings and the same set of buffers resolve to it again (EDEN_NX2_BUFCACHE_GEOMEMO).
    struct GeometryMemo {
        u64 epoch{}; ///< binding_epoch when stored; 0 never matches
        u64 map_serial{};
        size_t as_id{};
        const void* channel{};
        GPUVAddr begin{};
        GPUVAddr end{};
        u64 extra{};
        Binding binding{};
    };
    std::array<GeometryMemo, NUM_VERTEX_BUFFERS> vertex_memo{};
    GeometryMemo index_memo{};
    std::array<GeometryMemo, NUM_COMPUTE_UNIFORM_BUFFERS> compute_uniform_memo{};

    /// The last BindGraphicsUniformBuffer of a (stage, slot) and what it translated to, valid
    /// while that binding is the one it set (DisableGraphicsUniformBuffer clears it) and the
    /// mappings and the set of buffers stay the same (EDEN_NX2_UBO_REBIND).
    struct UniformBindMemo {
        u64 epoch{}; ///< binding_epoch when stored; 0 never matches
        u64 map_serial{};
        size_t as_id{};
        const void* channel{};
        GPUVAddr gpu_addr{};
        DAddr device_addr{};
        u32 size{};
    };
    std::array<std::array<UniformBindMemo, NUM_GRAPHICS_UNIFORM_BUFFERS>, NUM_STAGES>
        uniform_bind_memo{};

    [[nodiscard]] bool ReuseGeometry(const GeometryMemo& memo, GPUVAddr begin, GPUVAddr end,
                                     u64 extra) const;

    void StoreGeometry(GeometryMemo& memo, GPUVAddr begin, GPUVAddr end, u64 extra,
                       const Binding& binding);
    u64 frame_tick = 0;
    /// EDEN_NX2_TABLE_POINTER_TRACE: last table entry and size bound per table pointer slot.
    std::unordered_map<u64, std::pair<GPUVAddr, u32>> traced_table_pointers;
    u64 total_used_memory = 0;
    u64 buffer_bytes = 0; ///< The buffers' own size (total_used_memory may be the device's)
    u64 minimum_memory = 0;
    u64 critical_memory = 0;
    BufferId inline_buffer_id;
#ifdef YUZU_LEGACY
    bool immediately_free = false;
#endif

    // Covers the device address space (MaxwellDeviceTraits::device_virtual_bits).
    std::array<BufferId, ((1ULL << 36) >> CACHING_PAGEBITS)> page_table;
    Common::ScratchBuffer<u8> tmp_buffer;

    /// A command list page (Nx2CopyListPages): the GPU writes noted to it and the last copy.
    using Nx2Staging = decltype(std::declval<Runtime&>().DownloadStagingBuffer(size_t{}, true));
    struct Nx2ListPage {
        u64 writes{};      ///< GPU writes noted to the page
        u64 copy_writes{}; ///< `writes` when the copy was made: the copy is current while equal
        u64 copy_tick{};   ///< GPU tick that completes the copy
        /// `writes` when the front last flushed the page: guest memory holds it while equal
        u64 synced_writes{~u64{0}};
        u32 copy_begin{};  ///< Bytes of the page the copy holds (a buffer can end inside it)
        u32 copy_end{};
        /// Written outside a compute dispatch or a DMA copy (a draw's storage writes) at least
        /// once: the front can't tell from what it sent which draws write the page, so it reads
        /// no copy. A draw's writes reach a pushbuffer fetch only after a host-level wait, so
        /// the page's lists read guest memory once the back ran the work before the last such
        /// wait and no write came since the front flushed the page (Nx2ReadListCopy).
        bool draw_written{};
        std::optional<Nx2Staging> copy;
    };
    std::mutex nx2_list_mutex; ///< nx2_list_pages: the split's front reads the copies
    std::unordered_map<DAddr, Nx2ListPage> nx2_list_pages;
    /// Copies replaced while the GPU may still write them: freed once it is past their tick.
    std::vector<std::pair<u64, Nx2Staging>> nx2_retired_list_copies;
    boost::container::small_vector<std::pair<BufferId, DAddr>, 8> nx2_list_copy_requests;
    bool nx2_collect_list_copies{};
};

} // namespace VideoCommon
