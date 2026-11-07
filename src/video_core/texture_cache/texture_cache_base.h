// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <type_traits>
// TODO: find out which don't require stable iters
#include <unordered_map>
#include "video_core/page_filter.h"
#include "common/container/unordered_map.h"
#include "common/container/unordered_set.h"
#include <vector>
#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>
#include <queue>

#include "common/common_types.h"
#include "common/spin_recursive_mutex.h"
#include "common/hash.h"
#include "common/literals.h"
#include "common/lru_cache.h"
#include <ranges>
#include "common/scratch_buffer.h"
#include "common/slot_vector.h"
#include "video_core/texture_cache/pending_download.h"
#include "common/thread_worker.h"
#include "video_core/compatible_formats.h"
#include "video_core/control/channel_state_cache.h"
#include "video_core/delayed_destruction_ring.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/game_dlss_status.h"
#include "video_core/nx2_bindmemo_stats.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/descriptor_table.h"
#include "video_core/texture_cache/image_base.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/image_view_base.h"
#include "video_core/texture_cache/render_targets.h"
#include "video_core/texture_cache/types.h"
#include "video_core/textures/texture.h"
#include "video_core/textures/unswizzle_worker.h"

namespace Tegra {
namespace Control {
struct ChannelState;
}
} // namespace Tegra

namespace VideoCommon {

using Tegra::Texture::TICEntry;
using Tegra::Texture::TSCEntry;
using VideoCore::Surface::PixelFormat;
using namespace Common::Literals;

struct ImageViewInOut {
    u32 index{};
    bool blacklist{};
    ImageViewId id{};
};

/// EDEN_NX2_TEX_PAGES: the host pointers of the guest pages one draw or dispatch reads its
/// texture handles and TIC/TSC entries from, each page translated once instead of once per read.
/// The reads then all see the mappings as they were at the draw's first read: the guest can't
/// remap memory a pending draw reads without racing the GPU itself, and a host pointer points
/// into guest RAM, which is never freed. Only resolved pages are kept as pointers; a cache lives
/// for one draw.
struct GuestPageCache {
    static constexpr size_t NUM_ENTRIES = 8;
    static constexpr GPUVAddr PAGE_BYTES = GPUVAddr{1} << 12; ///< Core::Memory::YUZU_PAGESIZE
    static constexpr GPUVAddr NO_PAGE = ~GPUVAddr{0};          ///< never a page base

    std::array<GPUVAddr, NUM_ENTRIES> pages{NO_PAGE, NO_PAGE, NO_PAGE, NO_PAGE,
                                            NO_PAGE, NO_PAGE, NO_PAGE, NO_PAGE};
    std::array<const u8*, NUM_ENTRIES> hosts{};
    u32 next{};

    /// Host memory of [addr, addr + size) when it lies in one mapped page, as
    /// MemoryManager::GetPointer(addr) gives it; nullptr otherwise. `words`: the caller stands in
    /// for MemoryManager::Read<u32>, so a page the syncpoint shim (or anything else
    /// Nx2ReadNeedsDrain names) overlaps is remembered as nullptr: read it the slow way.
    template <class MemoryManager>
    [[nodiscard]] const u8* Get(const MemoryManager& gpu_memory, GPUVAddr addr, size_t size,
                                bool words) noexcept {
        const GPUVAddr offset = addr & (PAGE_BYTES - 1);
        if (offset + size > PAGE_BYTES) {
            return nullptr;
        }
        const GPUVAddr page = addr - offset;
        for (size_t index = 0; index < NUM_ENTRIES; ++index) {
            if (pages[index] == page) {
                return hosts[index] != nullptr ? hosts[index] + offset : nullptr;
            }
        }
        Nx2BindMemo::Note(Nx2BindMemo::Stat::PageTranslations);
        const u8* host = gpu_memory.GetPointer(page);
        if (host == nullptr && !words) {
            return nullptr;
        }
        if (words && host != nullptr && gpu_memory.Nx2ReadNeedsDrain(page, PAGE_BYTES)) {
            host = nullptr;
        }
        const size_t slot = next++ % NUM_ENTRIES;
        pages[slot] = page;
        hosts[slot] = host;
        return host != nullptr ? host + offset : nullptr;
    }

    /// MemoryManager::Read<u32>(addr), from the translated page when it can.
    template <class MemoryManager>
    [[nodiscard]] u32 ReadWord(const MemoryManager& gpu_memory, GPUVAddr addr) noexcept {
        if (const u8* const host = Get(gpu_memory, addr, sizeof(u32), true)) {
            Nx2BindMemo::Note(Nx2BindMemo::Stat::HandleWords);
            u32 value;
            std::memcpy(&value, host, sizeof(value));
            return value;
        }
        Nx2BindMemo::Note(Nx2BindMemo::Stat::HandleWordsSlow);
        return gpu_memory.template Read<u32>(addr);
    }
};

struct AsyncDecodeContext {
    ImageId image_id;
    Common::ScratchBuffer<u8> decoded_data;
    boost::container::small_vector<BufferImageCopy, 16> copies;
    std::mutex mutex;
    std::atomic_bool complete;
    bool cancelled{}; ///< The image was deleted meanwhile: drop the result (GPU thread only).
};

using TextureCacheGPUMap = ::Common::unordered_map<u64, std::vector<ImageId>, Common::IdentityHash<u64>>;

class TextureCacheChannelInfo : public ChannelInfo {
public:
    TextureCacheChannelInfo() = delete;
    TextureCacheChannelInfo(Tegra::Control::ChannelState& state) noexcept;
    TextureCacheChannelInfo(const TextureCacheChannelInfo& state) = delete;
    TextureCacheChannelInfo& operator=(const TextureCacheChannelInfo&) = delete;

    DescriptorTable<TICEntry, ImageViewId> graphics_image_table;
    DescriptorTable<TSCEntry, SamplerId> graphics_sampler_table;
    DescriptorTable<TICEntry, ImageViewId> compute_image_table;
    DescriptorTable<TSCEntry, SamplerId> compute_sampler_table;

    // TODO: still relies on bad iterators :(
    std::unordered_map<TICEntry, ImageViewId> image_views;
    std::unordered_map<TSCEntry, SamplerId> samplers;

    TextureCacheGPUMap* gpu_page_table = nullptr;
    TextureCacheGPUMap* sparse_page_table = nullptr;
};

template <class P>
class TextureCache : public VideoCommon::ChannelSetupCaches<TextureCacheChannelInfo> {
    /// Address shift for caching images into a hash table
    static constexpr u64 YUZU_PAGEBITS = 20;

    /// Enables debugging features to the texture cache
    static constexpr bool ENABLE_VALIDATION = P::ENABLE_VALIDATION;
    /// Implement blits as copies between framebuffers
    static constexpr bool FRAMEBUFFER_BLITS = P::FRAMEBUFFER_BLITS;
    /// True when some copies have to be emulated
    static constexpr bool HAS_EMULATED_COPIES = P::HAS_EMULATED_COPIES;
    /// True when the API can provide info about the memory of the device.
    static constexpr bool HAS_DEVICE_MEMORY_INFO = P::HAS_DEVICE_MEMORY_INFO;
    /// True when the API can do asynchronous texture downloads.
    static constexpr bool IMPLEMENTS_ASYNC_DOWNLOADS = P::IMPLEMENTS_ASYNC_DOWNLOADS;

    static constexpr size_t UNSET_CHANNEL{(std::numeric_limits<size_t>::max)()};

#ifdef YUZU_LEGACY
    static constexpr s64 TARGET_THRESHOLD = 3_GiB;
#else
    static constexpr s64 TARGET_THRESHOLD = 4_GiB;
#endif

    // Periodic statistics (TickFrame).
    u64 gc_deletes{};
    u64 gc_costly_deletes{};
    u64 gc_download_deletes{}; ///< Evictions that downloaded the image first (a full GPU wait)
    u64 async_decodes_queued{};
    u64 images_created{};
    u64 images_uploaded{};   ///< Image contents uploaded from guest memory (last 600 frames)
    u64 uploaded_bytes{};    ///< Guest bytes those uploads read
    u64 largest_upload{};    ///< Largest of them
    u64 layer_uploads{};     ///< Of them, uploads of only the written layers
    bool native_thresholds_applied{};
    u64 native_critical_memory{};
    u64 native_expected_memory{};
    u64 budget_check_frames{};
    u32 seen_pressure_events{};   ///< Allocator pressure events already answered
    u32 seen_failed_images{};     ///< Images the allocator could not make, already collected for
    u64 pressure_margin{};        ///< Extra room under the budget after images did not fit
    u64 pressure_quiet_frames{};  ///< Frames since the last such image
    u64 slack_margin{};           ///< Extra room for allocator block space the driver counts as used
    u64 astc_opaque_images{};
    u64 astc_alpha_images{};
    std::vector<u8> astc_scan_buffer;
    u64 images_deleted{};
    std::map<u64, std::pair<u32, GPUVAddr>> created_shapes;
    bool debug_stats{}; ///< Collect what only the debug statistics lines use

    static constexpr s64 DEFAULT_EXPECTED_MEMORY = 1_GiB + 125_MiB;
    static constexpr s64 DEFAULT_CRITICAL_MEMORY = 1_GiB + 625_MiB;
    /// The most device memory the collector lets the cache fill before it collects in earnest,
    /// on every GPU (ApplyThresholdOverrides).
    static constexpr u64 MaxCriticalMemory = 5_GiB;
    static constexpr size_t GC_EMERGENCY_COUNTS = 2;

    using Runtime = typename P::Runtime;
    using Image = typename P::Image;
    using ImageAlloc = typename P::ImageAlloc;
    using ImageView = typename P::ImageView;
    using Sampler = typename P::Sampler;
    using Framebuffer = typename P::Framebuffer;
    using AsyncBuffer = typename P::AsyncBuffer;
    using BufferType = typename P::BufferType;

    static constexpr u32 LayerUploadMinSize = 64U << 20;
    static constexpr size_t LargeScratchCapacity = 64U << 20;

    /// Splits large 2D image uploads into smaller chunks, avoiding multi-hundred MiB staging buffers.
    bool UploadInChunks(Image& image);

    /// Large layered images whose refresh can decode only the layers written since the last one.
    [[nodiscard]] bool UploadsByLayer(const ImageBase& image) const noexcept;

    /// Records the layers a write of [cpu_addr, +size) touches.
    void MarkLayersWritten(Image& image, DAddr cpu_addr, size_t size);

    /// SparseTiles images: notes a written range, uploaded with the next refresh.
    void MarkRangeWritten(Image& image, DAddr cpu_addr, size_t size);

    /// SparseTiles images: binds and uploads the parts written since the last upload.
    void RefreshSparseTiles(Image& image);

    /// Image::BindTiles, counting what it binds in total_used_memory.
    bool BindSparseTiles(Image& image, std::span<const LevelRect> rects);

    /// Image::BackTilesWhole, counting what it binds in total_used_memory.
    bool BackSparseTilesWhole(Image& image);

    /// Uploads the image's dirty layers. False when it needs a full upload instead.
    bool UploadWrittenLayers(Image& image);

    /// Gives the written layers of a SparseLayers image memory and uploads them.
    void RefreshSparseLayers(Image& image);

    /// Image::BindLayers, with the memory it binds counted in total_used_memory.
    std::vector<u32> BindSparseLayers(Image& image, std::span<const u32> layers);

    /// Uploads `layers` of the image from guest memory.
    void UploadLayers(Image& image, std::span<const u32> layers);

    struct PendingUnswizzle {
        ImageId image_id;
        VideoCommon::ImageInfo info;
        size_t current_offset = 0;
        size_t total_size = 0;
        AsyncBuffer staging_buffer;
        size_t last_submitted_offset = 0;
        size_t bytes_per_slice;
        bool initialized = false;
    };

    struct BlitImages {
        ImageId dst_id;
        ImageId src_id;
        PixelFormat dst_format;
        PixelFormat src_format;
    };

public:
    /// Device pages this cache ever registered objects on (InnerInvalidation skips the rest).
    VideoCommon::PageFilter page_filter;
    /// Device pages of images that were ever GPU-modified, never cleared: GetFlushArea finds
    /// nothing to flush where none was (EDEN_NX2_TEXTURE_READ_FILTER).
    VideoCommon::PageFilter gpu_modified_filter;
    std::atomic<bool> gpu_modified_sparse{}; ///< A sparse image was: the filter says "maybe"

    [[nodiscard]] bool MayHaveGpuModifiedImages(DAddr addr, u64 size) const {
        return gpu_modified_sparse.load(std::memory_order_relaxed) ||
               gpu_modified_filter.Any(addr, size);
    }

    explicit TextureCache(Runtime&, Tegra::MaxwellDeviceMemoryManager&);

    /// Notify the cache that a new frame has been queued
    void TickFrame();

    /// Return a constant reference to the given image view id
    [[nodiscard]] const ImageView& GetImageView(ImageViewId id) const noexcept;

    /// Return a reference to the given image view id
    [[nodiscard]] ImageView& GetImageView(ImageViewId id) noexcept;

    /// Get the imageview from the graphics descriptor table in the specified index
    [[nodiscard]] ImageView& GetImageView(u32 index) noexcept;

    /// Mark an image as modified from the GPU
    void MarkModification(ImageId id) noexcept;

    /// Fill image_view_ids with the graphics/compute images in indices
    void FillImageViews(std::span<ImageViewInOut> views, bool compute, bool blacklist = true);

    /// A binding memo hit for a blacklisted (written storage) image: keeps it from climbing to a
    /// rescale, as FillImageViews' blacklist reset does on a miss.
    void NoteBlacklistedBinding(ImageId image_id) noexcept {
        if (image_id && image_id != NULL_IMAGE_ID) {
            slot_images[image_id].scale_rating = 0;
        }
    }

    /// Handle feedback loops during draws.
    void CheckFeedbackLoop(std::span<const ImageViewInOut> views);

    [[nodiscard]] bool SamplesRenderTarget(std::span<const ImageViewInOut> views);

    /// Get the sampler from the graphics/compute descriptor table in the specified index
    Sampler* GetSampler(u32 index, bool compute);

    /// Get the sampler id from the graphics/compute descriptor table in the specified index
    SamplerId GetSamplerId(u32 index, bool compute);

    /// Return a constant reference to the given sampler id
    [[nodiscard]] const Sampler& GetSampler(SamplerId id) const noexcept;

    /// Return a reference to the given sampler id
    [[nodiscard]] Sampler& GetSampler(SamplerId id) noexcept;

    /// Refresh the state for graphics/compute image view and sampler descriptors
    void SynchronizeDescriptors(bool compute);

    /// Changes whenever a pipeline's cached texture bindings (view and sampler ids, descriptor
    /// handles, rescaling bits) could be stale: images or views created, deleted, rescaled or
    /// CPU modified, descriptor pools moved, and every frame (EDEN_NX2_BINDCACHE).
    [[nodiscard]] u64 BindingsEpoch() const noexcept {
        return bindings_epoch;
    }

    /// EDEN_NX2_BINDMEMO: one texture, image or sampler binding as FillImageViews and
    /// GetSamplerId resolved it, with the guest TIC/TSC entries it was resolved from. It stands
    /// while guest memory holds the same entries, the descriptor pools stay, and nothing happened
    /// to its image that preparing the view again would act on (CPU writes, aliasing, deletion,
    /// rescaling). The backend keeps its descriptor for the binding in `payload`.
    struct alignas(64) BindingMemo {
        // Line 0: key and what the lookup compares first.
        u32 tic_handle;
        u32 tsc_index; ///< ~0U: no sampler
        u32 kind;      ///< the caller's binding kind (BINDING_KIND_COMPUTE included); 0: empty
        ImageViewId view;
        SamplerId sampler;
        ImageId image;
        u32 tic_limit;
        u32 tsc_limit;
        GPUVAddr tic_pool;
        GPUVAddr tsc_pool;
        const void* channel;
        u32 tic_entry; ///< TIC entry index read, ~0U: none (out of the pool)
        u32 tsc_entry; ///< TSC entry index read, ~0U: none
        // Line 1
        u64 epoch;
        u64 image_serial;
        u64 map_serial;
        u64 touch_tick;
        std::array<u64, 3> payload;
        u16 rescaled;
        u8 aliased;       ///< its image had aliases when recorded: the lookup checks their ticks
        u8 map_dependent; ///< a null or buffer view whose resolution depends on the mappings
        // Line 2
        TICEntry tic;
        TSCEntry tsc;
    };
    static constexpr u32 BINDING_KIND_COMPUTE = 1U << 31;

    /// What every binding of one draw or dispatch is looked up and recorded against.
    struct BindingLookup {
        const void* channel;
        GPUVAddr tic_pool;
        GPUVAddr tsc_pool;
        u32 tic_limit;
        u32 tsc_limit;
        u64 map_serial;
        u64 clock; ///< BindingClock() when the lookups began
        bool compute;
        /// The TIC and TSC pages this draw's lookups translated (EDEN_NX2_TEX_PAGES).
        mutable GuestPageCache pages{};
    };

    /// Where a binding key's memos are: computed once per binding of a draw
    /// (EDEN_NX2_BINDMEMO_BATCH).
    struct BindingProbe {
        u64 hash;
        size_t first; ///< the set's first way
        u32 tag;
    };

    /// Taken after SynchronizeDescriptors, before the draw's or dispatch's first FindBinding.
    [[nodiscard]] BindingLookup BeginBindingLookup(bool compute) const noexcept;

    /// The key's set, with its tag line asked for ahead of the lookup.
    [[nodiscard]] BindingProbe ProbeBinding(u32 tic_handle, u32 tsc_index, u32 kind) const noexcept;

    /// Asks ahead for the lines a lookup of this key will read: its candidate memos and the
    /// guest TIC and TSC entries the handle names. Changes nothing a lookup decides on.
    void PrefetchBinding(const BindingLookup& lookup, const BindingProbe& probe, u32 tic_handle,
                         u32 tsc_index) const noexcept;

    /// The memo of this binding when resolving it again would give the same view, sampler and
    /// descriptor with nothing to prepare; nullptr otherwise. Touches the image in the LRU once
    /// per frame, as preparing the view would.
    [[nodiscard]] BindingMemo* FindBinding(const BindingLookup& lookup, u32 tic_handle,
                                           u32 tsc_index, u32 kind);
    [[nodiscard]] BindingMemo* FindBinding(const BindingLookup& lookup, u32 tic_handle,
                                           u32 tsc_index, u32 kind, const BindingProbe& probe);

    /// Changes with every event that can make a binding memo stale.
    [[nodiscard]] u64 BindingClock() const noexcept {
        return binding_clock;
    }

    /// Remembers a binding FillImageViews/GetSamplerId just resolved, when nothing happened to
    /// its image since `lookup` was taken and the image needs no preparation.
    void RecordBinding(const BindingLookup& lookup, u32 tic_handle, u32 tsc_index, u32 kind,
                       ImageViewId view_id, SamplerId sampler_id, const void* payload,
                       size_t payload_size, bool rescaled);

    /// Updates the Render Targets if they can be rescaled
    /// @retval True if the Render Targets have been rescaled.
    bool RescaleRenderTargets();

    /// Update bound render targets and upload memory if necessary
    /// @param is_clear True when the render targets are being used for clears
    void UpdateRenderTargets(bool is_clear);

    /// Find a framebuffer with the currently bound render targets
    /// UpdateRenderTargets should be called before this
    Framebuffer* GetFramebuffer();

    [[nodiscard]] const RenderTargets& GetRenderTargets() const noexcept {
        return render_targets;
    }

    /// Mark images in a range as modified from the CPU
    void WriteMemory(DAddr cpu_addr, size_t size);

    /// Download contents of host images to guest memory in a region
    void DownloadMemory(DAddr cpu_addr, size_t size);

    std::optional<VideoCore::RasterizerDownloadArea> GetFlushArea(DAddr cpu_addr, u64 size);

    /// Remove images in a region
    void UnmapMemory(DAddr cpu_addr, size_t size);

    /// Remove images in a region
    void UnmapGPUMemory(size_t as_id, GPUVAddr gpu_addr, size_t size);

    /// Blit an image with the given parameters
    bool BlitImage(const Tegra::Engines::Fermi2D::Surface& dst,
                   const Tegra::Engines::Fermi2D::Surface& src,
                   const Tegra::Engines::Fermi2D::Config& copy);

    /// Stand-in for a DLSS evaluation: temporal upscale of src into dst (see the runtime's
    /// TemporalUpscale). False when unsupported; the caller retains its native dispatch.
    bool TemporalUpscale(const Tegra::Engines::Fermi2D::Surface& dst,
                         const Tegra::Engines::Fermi2D::Surface& src,
                         const Tegra::Engines::Fermi2D::Surface& motion_vectors,
                         const Tegra::Texture::TICEntry* depth, std::array<f32, 2> jitter,
                         std::array<f32, 2> motion_scale, bool use_history);

    /// Health check of a game's DLSS network output (dst) against its input (src); see the
    /// runtime's ProbeGameDlssOutput. Returns an earlier check's result when one is ready.
    std::optional<VideoCore::GameDlssProbe> ProbeGameDlssOutput(
        const Tegra::Engines::Fermi2D::Surface& dst, const Tegra::Engines::Fermi2D::Surface& src);

    /// Try to find a cached image view in the given CPU address
    [[nodiscard]] std::pair<ImageView*, bool> TryFindFramebufferImageView(
        const Tegra::FramebufferConfig& config, DAddr cpu_addr);

    /// Return true when there are uncommitted images to be downloaded
    [[nodiscard]] bool HasUncommittedFlushes() const noexcept;

    /// Whether CommitAsyncFlushes would commit anything (downloads or their buffers).
    [[nodiscard]] bool HasAsyncFlushWork() const noexcept;

    /// Return true when the caller should wait for async downloads
    [[nodiscard]] bool ShouldWaitAsyncFlushes() const noexcept;

    /// Commit asynchronous downloads
    void CommitAsyncFlushes();

    /// Pop asynchronous downloads
    void PopAsyncFlushes();

    [[nodiscard]] ImageId DmaImageId(const Tegra::DMA::ImageOperand& operand, bool is_upload);

    [[nodiscard]] std::pair<Image*, BufferImageCopy> DmaBufferImageCopy(
        const Tegra::DMA::ImageCopy& copy_info, const Tegra::DMA::BufferOperand& buffer_operand,
        const Tegra::DMA::ImageOperand& image_operand, ImageId image_id, bool modifies_image);

    void DownloadImageIntoBuffer(Image* image, BufferType buffer, size_t buffer_offset,
                                 std::span<const VideoCommon::BufferImageCopy> copies,
                                 GPUVAddr address = 0, size_t size = 0);

    /// Return true when a CPU region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(DAddr addr, size_t size);

    [[nodiscard]] bool IsRescaling() const noexcept;

    [[nodiscard]] bool IsRescaling(const ImageViewBase& image_view) const noexcept;

    /// Create channel state.
    void CreateChannel(Tegra::Control::ChannelState& channel) final override;

    /// Prepare an image to be used
    void PrepareImage(ImageId image_id, bool is_modification, bool invalidate);

    Common::SpinRecursiveMutex mutex;

private:
    /// Iterate over all page indices in a range
    template <typename Func>
    static void ForEachCPUPage(DAddr addr, size_t size, Func&& func) {
        static constexpr bool RETURNS_BOOL = std::is_same_v<std::invoke_result<Func, u64>, bool>;
        const u64 page_end = (addr + size - 1) >> YUZU_PAGEBITS;
        for (u64 page = addr >> YUZU_PAGEBITS; page <= page_end; ++page) {
            if constexpr (RETURNS_BOOL) {
                if (func(page)) {
                    break;
                }
            } else {
                func(page);
            }
        }
    }

    template <typename Func>
    static void ForEachGPUPage(GPUVAddr addr, size_t size, Func&& func) {
        static constexpr bool RETURNS_BOOL = std::is_same_v<std::invoke_result<Func, u64>, bool>;
        const u64 page_end = (addr + size - 1) >> YUZU_PAGEBITS;
        for (u64 page = addr >> YUZU_PAGEBITS; page <= page_end; ++page) {
            if constexpr (RETURNS_BOOL) {
                if (func(page)) {
                    break;
                }
            } else {
                func(page);
            }
        }
    }

    void OnGPUASRegister(size_t map_id) final override;

    /// Runs the Garbage Collector.
    void RunGarbageCollector(bool out_of_memory);

    /// Find or create an image view in the guest descriptor table
    ImageViewId VisitImageView(u32 index, bool compute);

    /// Find or create a framebuffer with the given render target parameters
    FramebufferId GetFramebufferId(const RenderTargets& key);

    /// Refresh the contents (pixel data) of an image
    [[nodiscard]] bool IsDownloadable(const ImageBase& image) const noexcept;

    void RefreshContents(Image& image, ImageId image_id);

    /// Upload data from guest to an image
    template <typename StagingBuffer>
    void UploadImageContents(Image& image, StagingBuffer& staging_buffer);

    /// Create a new image view from a guest descriptor
    [[nodiscard]] ImageViewId CreateImageView(const TICEntry& config);

    /// Find or create an image from the given parameters
    [[nodiscard]] ImageId FindOrInsertImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                            RelaxedOptions options = RelaxedOptions{});

    /// Find an image from the given parameters
    [[nodiscard]] ImageId FindImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                    RelaxedOptions options);

    /// Create an image from the given parameters
    [[nodiscard]] ImageId InsertImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                      RelaxedOptions options);

    /// Create a new image and join perfectly matching existing images
    /// Remove joined images from the cache
    [[nodiscard]] ImageId JoinImages(const ImageInfo& info, GPUVAddr gpu_addr, DAddr cpu_addr);

    [[nodiscard]] ImageId FindDMAImage(const ImageInfo& info, GPUVAddr gpu_addr);

    /// Return a blit image pair from the given guest blit parameters
    [[nodiscard]] std::optional<BlitImages> GetBlitImages(
        const Tegra::Engines::Fermi2D::Surface& dst, const Tegra::Engines::Fermi2D::Surface& src,
        const Tegra::Engines::Fermi2D::Config& copy);

    /// Find or create a sampler from a guest descriptor sampler
    [[nodiscard]] SamplerId FindSampler(const TSCEntry& config, bool compute);

    /// Find or create an image view for the given color buffer index
    [[nodiscard]] ImageViewId FindColorBuffer(size_t index);

    /// Find or create an image view for the depth buffer
    [[nodiscard]] ImageViewId FindDepthBuffer();

    /// Find or create a view for a render target with the given image parameters
    [[nodiscard]] ImageViewId FindRenderTargetView(const ImageInfo& info, GPUVAddr gpu_addr);

    /// EDEN_NX2_RT_LOOKUP_MEMO: the registers a render target lookup reads (slot kind first).
    using RenderTargetLookupKey = std::array<u32, 12>;
    /// FindRenderTargetView, answered from the memo while no image or view was created, deleted
    /// or remapped (bindings_epoch) since a lookup with the same registers found exactly one
    /// candidate image (so modification ticks did not pick it).
    [[nodiscard]] ImageViewId FindRenderTargetViewMemo(const RenderTargetLookupKey& key,
                                                       const ImageInfo& info, GPUVAddr gpu_addr);

    /// EDEN_NX2_RT_FAST: whether UpdateRenderTargets' clean path would change nothing: the same
    /// targets were prepared for a draw, no image was modified, created or deleted since, in the
    /// same frame, and none needs an upload, alias sync or sparse backing.
    [[nodiscard]] bool RenderTargetsStillPrepared() const;
    void NoteRenderTargetsPrepared();

    /// Iterates over all the images in a region calling func
    template <typename Func>
    void ForEachImageInRegion(DAddr cpu_addr, size_t size, Func&& func);

    template <typename Func>
    void ForEachImageInRegionGPU(size_t as_id, GPUVAddr gpu_addr, size_t size, Func&& func);

    template <typename Func>
    void ForEachSparseImageInRegion(size_t as_id, GPUVAddr gpu_addr, size_t size, Func&& func);

    /// Iterates over all the images in a region calling func
    template <typename Func>
    void ForEachSparseSegment(ImageBase& image, Func&& func);

    /// Find or create an image view in the given image with the passed parameters
    [[nodiscard]] ImageViewId FindOrEmplaceImageView(ImageId image_id, const ImageViewInfo& info);

    /// Register image in the page table
    void RegisterImage(ImageId image);

    /// Unregister image from the page table
    void UnregisterImage(ImageId image);

    /// Logs where UnregisterImage's image was registered when one of its pages is missing.
    void ReportMissingImagePage(ImageId image_id, const Image& image, u64 page);

    /// Track CPU reads and writes for image
    void TrackImage(ImageBase& image, ImageId image_id);

    /// Stop tracking CPU reads and writes for image
    void UntrackImage(ImageBase& image, ImageId image_id);

    /// Delete image from the cache
    void DeleteImage(ImageId image, bool immediate_delete = false);

    /// Remove image views references from the cache
    void RemoveImageViewReferences(std::span<const ImageViewId> removed_views);

    /// Per image view, the (channel, descriptor) entries of the channels' image_views maps that
    /// point to it. Deleting an image erased them by scanning every map (title 1: ~5,000 entries,
    /// 50-130 us for each deleted image, dozens a frame while the collector runs).
    std::unordered_map<ImageViewId, boost::container::small_vector<std::pair<size_t, TICEntry>, 2>>
        image_view_descriptors;

    /// Remove framebuffers using the given image views from the cache
    void RemoveFramebuffers(std::span<const ImageViewId> removed_views);

    /// Mark an image as modified from the GPU
    void MarkModification(ImageBase& image) noexcept;

    /// Marks gpu_modified_filter before an image becomes GPU-modified.
    void NoteGpuModified(const ImageBase& image) noexcept;

    /// Synchronize image aliases, copying data if needed
    void SynchronizeAliases(ImageId image_id);

    /// Prepare an image view to be used
    void PrepareImageView(ImageViewId image_view_id, bool is_modification, bool invalidate);

    /// Execute copies from one image to the other, even if they are incompatible
    void CopyImage(ImageId dst_id, ImageId src_id, std::vector<ImageCopy> copies);

    /// Bind an image view as render target, downloading resources preemtively if needed
    void BindRenderTarget(ImageViewId* old_id, ImageViewId new_id);

    /// Create a render target from a given image and image view parameters
    [[nodiscard]] std::pair<FramebufferId, ImageViewId> RenderTargetFromImage(
        ImageId, const ImageViewInfo& view_info);

    /// Returns true if the current clear parameters clear the whole image of a given image view
    [[nodiscard]] bool IsFullClear(ImageViewId id);

    /// Returns true if the current clear's render area and scissor cover a width x height image
    [[nodiscard]] bool ClearCovers(u32 width, u32 height) const;

    [[nodiscard]] std::pair<u32, u32> PrepareDmaImage(ImageId dst_id, GPUVAddr base_addr,
                                                      bool mark_as_modified);

    bool ImageCanRescale(ImageBase& image);
    void InvalidateScale(Image& image);
    bool ScaleUp(Image& image);
    bool ScaleDown(Image& image);
    u64 GetScaledImageSizeBytes(const ImageBase& image);

    /// The device memory an image counts for in total_used_memory between two driver readings:
    /// none for an unbacked image (too large to back, or over unwritten sparse blocks), what is
    /// bound to its layers for a SparseLayers one.
    u64 ImageMemoryBytes(const Image& image) const;

    void QueueAsyncDecode(Image& image, ImageId image_id);
    void TickAsyncDecode();

    void QueueAsyncUnswizzle(Image& image, ImageId image_id);
    void TickAsyncUnswizzle();

    Runtime& runtime;

    Tegra::MaxwellDeviceMemoryManager& device_memory;
    std::deque<TextureCacheGPUMap> gpu_page_table_storage;

    RenderTargets render_targets;
    u64 render_targets_serial = 0;
    struct RenderTargetLookupMemo {
        RenderTargetLookupKey key{};
        u64 epoch = 0; ///< bindings_epoch of the lookup; 0 never matches
        u64 frame_bumps = 0; ///< frame_epoch_bumps of the lookup
        const void* channel{}; ///< channel_state of the lookup (BindChannel moves no epoch)
        const void* memory{}; ///< gpu_memory the lookup resolved addresses through
        ImageViewId view{};
    };
    /// The bindings_epoch increments TickFrame made (the render target memo outlives them).
    u64 frame_epoch_bumps = 0;
    std::array<RenderTargetLookupMemo, 32> rt_lookup_memos{};
    u32 find_image_matches = 0; ///< candidates the last FindImage found
    struct RenderTargetsPrepared {
        bool valid = false;
        u64 serial = 0;
        u64 modification_tick = 0;
        u64 frame_tick = 0;
        u64 bindings_epoch = 0;
        std::array<ImageViewId, NUM_RT + 1> views{};
    } rt_prepared;
    u32 rt_active_mask = 0;
    std::array<ImageId, 8> rt_image_id{};
    ImageId rt_depth_image_id{};
    u64 texture_bindings_serial = 0;
    u64 bindings_epoch = 1; ///< See BindingsEpoch(); 0 never matches a cached binding
    /// EDEN_NX2_BINDMEMO: 4-way sets of BindingMemo with a tag per way (0: empty) and the next way
    /// each set replaces (allocated on first use), the serial of each image slot's last event,
    /// and the clock both serials and binding_memo_epoch are drawn from.
    std::vector<BindingMemo> binding_memos;
    std::vector<u32> binding_tags;
    std::vector<u8> binding_victims;
    std::vector<u64> image_binding_serials;
    u64 binding_clock = 1;
    u64 binding_memo_epoch = 1; ///< every memo: an image rescaled
    void BumpBindingSerial(ImageId image_id);
    void BumpBindingEpoch() noexcept {
        binding_memo_epoch = ++binding_clock;
    }
    u64 last_feedback_loop_serial = 0;
    u64 last_feedback_texture_serial = 0;
    bool last_feedback_loop_result = false;
    bool feedback_depth_dirty = true;
    FramebufferId last_framebuffer_id{};
    u64 last_framebuffer_serial = 0;

    ::Common::unordered_map<RenderTargets, FramebufferId> framebuffers;
    ::Common::unordered_map<u64, std::vector<ImageMapId>, Common::IdentityHash<u64>> page_table;
    ::Common::unordered_map<ImageId, boost::container::small_vector<ImageViewId, 16>> sparse_views;

    DAddr virtual_invalid_space{};

    /// Applies EDEN_NX2_TC_EXPECTED_MIB / EDEN_NX2_TC_CRITICAL_MIB and logs the thresholds.
    void ApplyThresholdOverrides(bool log = true);

    /// The CPU address an image at gpu_addr is keyed by: its backing, or a fixed stand-in.
    [[nodiscard]] std::optional<DAddr> ImageCpuAddress(GPUVAddr gpu_addr, size_t size) const;

    bool has_deleted_images = false;
    /// True while a clear binds its render targets (UpdateRenderTargets(true)), so JoinImages can
    /// skip the upload of a new target the clear overwrites whole.
    bool clearing_render_targets = false;
    bool is_rescaling = false;
    u64 total_used_memory = 0;
    u64 minimum_memory;
    u64 expected_memory;
    u64 critical_memory;
    size_t gpu_unswizzle_maxsize = 0;
    size_t swizzle_chunk_size = 0;
    u32 swizzle_slices_per_batch = 0;

    struct BufferDownload {
        GPUVAddr address;
        size_t size;
    };

    using PendingDownload = PendingImageDownload<Image>;

    Common::SlotVector<Image> slot_images;
    Common::SlotVector<ImageMapView> slot_map_views;
    Common::SlotVector<ImageView> slot_image_views;
    Common::SlotVector<ImageAlloc> slot_image_allocs;
    Common::SlotVector<Sampler> slot_samplers;
    Common::SlotVector<Framebuffer> slot_framebuffers;
    Common::SlotVector<BufferDownload> slot_buffer_downloads;

    // TODO: This data structure is not optimal and it should be reworked

    std::vector<PendingDownload> uncommitted_downloads;
    std::deque<std::vector<PendingDownload>> committed_downloads;
    std::vector<AsyncBuffer> uncommitted_async_buffers;
    std::deque<std::vector<AsyncBuffer>> async_buffers;
    std::deque<AsyncBuffer> async_buffers_death_ring;

    struct LRUItemParams {
        using ObjectType = ImageId;
        using TickType = u64;
    };
    Common::LeastRecentlyUsedCache<LRUItemParams> lru_cache;

 #ifdef YUZU_LEGACY
    static constexpr size_t TICKS_TO_DESTROY = 6;
 #else
    static constexpr size_t TICKS_TO_DESTROY = 8;
#endif
    DelayedDestructionRing<Image, TICKS_TO_DESTROY> sentenced_images;
    /// Device memory of the images in each slot of sentenced_images (same index): deleted, but
    /// counted in the driver's usage until the ring destroys them.
    std::array<u64, TICKS_TO_DESTROY> sentenced_image_bytes{};
    size_t sentenced_image_index = 0;
    /// Deleted images on runtimes that report GPU progress (CurrentTick, IsFree): destroyed, with
    /// their device memory, as soon as the GPU is past them rather than TICKS_TO_DESTROY frames
    /// later. While the collector deletes dozens a frame, eight frames of them held 1-1.5 GB of
    /// VRAM above its threshold (title 1 on an 8 GB card: 6.6-7.0 GB in use against 5 GiB).
    struct GpuSentencedImage {
        u64 tick;
        u64 bytes;
        Image image;
    };
    std::deque<GpuSentencedImage> gpu_sentenced_images;
    u64 gpu_sentenced_bytes{};

    /// Bytes of images per device memory block, for the collector (refreshed every 60 frames).
    std::unordered_map<u64, u64> block_image_bytes;
    u64 block_image_bytes_frame{~0ULL};
    DelayedDestructionRing<ImageView, TICKS_TO_DESTROY> sentenced_image_view;
    DelayedDestructionRing<Framebuffer, TICKS_TO_DESTROY> sentenced_framebuffers;

    ::Common::unordered_map<GPUVAddr, ImageAllocId> image_allocs_table;

    Common::ScratchBuffer<u8> swizzle_data_buffer;
    Common::ScratchBuffer<u8> unswizzle_data_buffer;
    Tegra::Texture::UnswizzleWorker unswizzle_worker;

    u64 modification_tick = 0;
    u64 frame_tick = 0;

    Common::ThreadWorker texture_decode_worker{1, "TextureDecoder", {},
                                               Common::ThreadPlacement::Efficiency};
    std::vector<std::unique_ptr<AsyncDecodeContext>> async_decodes;
    /// Guest memory images were uploaded from since the last cold hint (device address, size).
    std::vector<std::pair<DAddr, u64>> uploaded_ranges;

    std::deque<PendingUnswizzle> unswizzle_queue;
    u8 current_unswizzle_frame;

    // Join caching
    boost::container::small_vector<ImageId, 4> join_overlap_ids;
    ::Common::unordered_set<ImageId> join_overlaps_found;
    boost::container::small_vector<ImageId, 4> join_left_aliased_ids;
    boost::container::small_vector<ImageId, 4> join_right_aliased_ids;
    ::Common::unordered_set<ImageId> join_ignore_textures;
    boost::container::small_vector<ImageId, 4> join_bad_overlap_ids;
    struct JoinCopy {
        bool is_alias;
        ImageId id;
    };
    boost::container::small_vector<JoinCopy, 4> join_copies_to_do;
    ::Common::unordered_map<ImageId, size_t> join_alias_indices;
};

} // namespace VideoCommon
