// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <list>
#include <memory>
#include <utility>

#include "common/assert.h"
#include "common/settings.h"
#include "common/settings_enums.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/gpu_dirty_memory_manager.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/graphics_context.h"
#include "core/hle/service/nvdrv/nvdata.h"
#include "core/perf_stats.h"
#include "video_core/cdma_pusher.h"
#include "video_core/control/channel_state.h"
#include "video_core/control/scheduler.h"
#include "video_core/dma_pusher.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/kepler_memory.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/gpu.h"
#include "video_core/pending_maps.h"
#include "video_core/gpu_thread.h"
#include "video_core/gpu_timestamp.h"
#include "video_core/host1x/host1x.h"
#include "video_core/host1x/syncpoint_manager.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_profile.h"
#include "video_core/nx2_reactive.h"
#include "video_core/renderer_base.h"
#include "video_core/shader_notify.h"

namespace Tegra {

namespace {

// EDEN_NX2_OPT_TICKWORK=0: TickWork locks the sync request mutex on every call (each draw, clear
// and dispatch), as before, instead of first checking without the lock whether a request is
// queued at all.
bool TickWorkFastEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_OPT_TICKWORK");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

} // Anonymous namespace

struct GPU::Impl {
    explicit Impl(Core::System& system_, bool is_async_, bool use_nvdec_)
        : system{system_}
        , gpu_thread{system_}
        , use_nvdec{use_nvdec_}
        , shader_notify()
        , is_async{is_async_}
    {}

    ~Impl() {
        NotifyShutdown();
        // Rasterizer teardown flushes commands and invalidates the bound engine's dirty flags.
        // Keep channels, their scheduler references, and the graphics context alive until then.
        renderer.reset();
    }

    std::shared_ptr<Control::ChannelState> CreateChannel(s32 channel_id) {
        auto channel_state = std::make_shared<Tegra::Control::ChannelState>(channel_id);
        channels.emplace(channel_id, channel_state);
        scheduler.DeclareChannel(channel_state);
        return channel_state;
    }

    void BindChannel(s32 channel_id) {
        if (bound_channel != channel_id) {
            auto it = channels.find(channel_id);
            ASSERT(it != channels.end());
            bound_channel = channel_id;
            current_channel = it->second.get();
            renderer->ReadRasterizer()->BindChannel(*current_channel);
        }
    }

    std::shared_ptr<Control::ChannelState> AllocateChannel() {
        return CreateChannel(new_channel_id++);
    }

    void InitChannel(Control::ChannelState& to_init, u64 program_id) {
        to_init.Init(system, program_id);
        to_init.BindRasterizer(renderer->ReadRasterizer());
        renderer->ReadRasterizer()->InitializeChannel(to_init);
    }

    void InitAddressSpace(Tegra::MemoryManager& memory_manager) {
        memory_manager.BindRasterizer(renderer->ReadRasterizer());
    }

    void ReleaseChannel(Control::ChannelState& to_release) {
        UNIMPLEMENTED();
    }

    /// Binds a renderer to the GPU.
    void BindRenderer(std::unique_ptr<VideoCore::RendererBase> renderer_) {
        renderer = std::move(renderer_);
        system.Host1x().memory_manager.BindInterface(renderer->ReadRasterizer());
        system.Host1x().gmmu_manager.BindRasterizer(renderer->ReadRasterizer());
    }

    /// Flush all current written commands into the host GPU for execution.
    void FlushCommands() {
        renderer->ReadRasterizer()->FlushCommands();
    }

    /// Synchronizes CPU writes with Host GPU memory.
    void InvalidateGPUCache() {
        // Called at every fence (~11,000 a second in Ocarina), mostly with nothing collected:
        // skip the six cores' locks then (EDEN_NX2_DIRTY_FASTPATH=0 always gathers).
        static const bool fast_path = [] {
            const char* value = std::getenv("EDEN_NX2_DIRTY_FASTPATH");
            return value == nullptr || value[0] != '0';
        }();
        // EDEN_NX2_DIRTY_ANY: one shared flag instead of three atomics per core's manager, on
        // lines the cores write (18 loads per fence, ~18,000 fences a second).
        static const bool any_flag = [] {
            const char* value = std::getenv("EDEN_NX2_DIRTY_ANY");
            return value == nullptr || value[0] != '0';
        }();
        const bool use_any_flag = fast_path && any_flag;
        if (use_any_flag) {
            if (!Core::GPUDirtyMemoryManager::AnyWrites()) {
                return;
            }
        } else if (fast_path) {
            bool may_have_writes = false;
            for (auto& manager : system.GetGPUDirtyMemoryManager()) {
                may_have_writes |= manager.MayHaveWrites();
            }
            if (!may_have_writes) {
                return;
            }
        }
        std::vector<std::pair<DAddr, u64>> writes;
        std::function<void(PAddr, size_t)> callback_writes(
            [&writes](PAddr address, size_t size) { writes.emplace_back(address, size); });
        if (use_any_flag) {
            // Until EndGather, a concurrent caller finding the flag clear gathers too, so it
            // never returns before the writes collected before it are applied.
            Core::GPUDirtyMemoryManager::BeginGather();
        }
        system.GatherGPUDirtyMemory(callback_writes);
        if (!writes.empty()) {
            renderer->ReadRasterizer()->OnCacheInvalidations(writes);
        }
        if (use_any_flag) {
            Core::GPUDirtyMemoryManager::EndGather();
        }
    }

    void ForEachPendingCpuWrite(const std::function<void(DAddr, size_t)>& callback) {
        for (auto& manager : system.GetGPUDirtyMemoryManager()) {
            manager.ForEachPending(callback);
        }
    }

    /// Signal the ending of command list.
    void OnCommandListEnd() {
        renderer->ReadRasterizer()->ReleaseFences(false);
        Settings::UpdateGPUAccuracy();
    }

    /// Request a host GPU memory flush from the CPU.
    template <typename Func>
    [[nodiscard]] u64 RequestSyncOperation(Func&& action) {
        std::unique_lock lck{sync_request_mutex};
        const u64 fence = ++last_sync_fence;
        sync_requests.emplace_back(std::forward<Func>(action));
        has_sync_requests.store(true, std::memory_order_release);
        return fence;
    }

    /// Obtains current flush request fence id.
    [[nodiscard]] u64 CurrentSyncRequestFence() const {
        return current_sync_fence.load(std::memory_order_relaxed);
    }

    void WaitForSyncOperation(const u64 fence) {
        std::unique_lock lck{sync_request_mutex};
        sync_request_cv.wait(lck, [this, fence] { return CurrentSyncRequestFence() >= fence; });
    }

    void WaitForIdle() {
        const u64 fence = RequestSyncOperation([] {});
        gpu_thread.TickGPU(is_async);
        WaitForSyncOperation(fence);
    }

    /// Tick pending requests within the GPU.
    void TickWork() {
        // A request queued after this check is run by the next TickWork, as one queued just
        // after the loop below empties the queue is. Whoever made it and then had this thread
        // tick (WaitForIdle's TickGPU) is ordered after its store, so that tick sees it.
        if (TickWorkFastEnabled() && !has_sync_requests.load(std::memory_order_acquire)) {
            return;
        }
        VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::TickWork};
        std::unique_lock lck{sync_request_mutex};
        while (!sync_requests.empty()) {
            auto request = std::move(sync_requests.front());
            sync_requests.pop_front();
            sync_request_mutex.unlock();
            request();
            current_sync_fence.fetch_add(1, std::memory_order_release);
            sync_request_mutex.lock();
            sync_request_cv.notify_all();
        }
        // Under the lock with the queue empty: no request can be queued in between.
        has_sync_requests.store(false, std::memory_order_relaxed);
    }

    [[nodiscard]] u64 GetTicks() const {
        return ReadGpuTimestamp(system.CoreTiming(), g_nx2_ampere.load(std::memory_order_relaxed),
                                Settings::values.gpu_clock.GetValue());
    }

    void RendererFrameEndNotify() {
        system.GetPerfStats().EndGameFrame();
    }

    /// Performs any additional setup necessary in order to begin GPU emulation.
    /// This can be used to launch any necessary threads and register any necessary
    /// core timing events.
    void Start() {
        Settings::UpdateGPUAccuracy();
        gpu_thread.StartThread(*renderer, renderer->Context(), scheduler);
    }

    void NotifyShutdown() {
        gpu_thread.NotifyShutdown();
        std::unique_lock lk{sync_mutex};
        shutting_down.store(true, std::memory_order::relaxed);
        sync_cv.notify_all();
    }

    /// Obtain the CPU Context
    void ObtainContext() {
        if (!cpu_context) {
            cpu_context = renderer->GetRenderWindow().CreateSharedContext();
        }
        cpu_context->MakeCurrent();
    }

    /// Release the CPU Context
    void ReleaseContext() {
        cpu_context->DoneCurrent();
    }

    /// Push GPU command entries to be processed
    void PushGPUEntries(s32 channel, Tegra::CommandList&& entries) {
        gpu_thread.SubmitList(channel, std::move(entries), is_async);
    }

    void RunInOrder(std::function<void()>&& func) {
        gpu_thread.Run(std::move(func), is_async);
    }

    void MapInOrder(GPUVAddr gpu_addr, u64 size, std::function<void()>&& func) {
        const u64 id = pending_maps.Add(gpu_addr, size);
        gpu_thread.Run(
            [this, id, func = std::move(func)] {
                func();
                pending_maps.Remove(id);
            },
            is_async, true);
    }

    bool IsMapPending(GPUVAddr gpu_addr, u64 size) const {
        return pending_maps.Overlaps(gpu_addr, size);
    }

    /// Notify rasterizer that any caches of the specified region should be flushed to Switch memory
    void FlushRegion(DAddr addr, u64 size) {
        gpu_thread.FlushRegion(addr, size, is_async);
    }

    /// A read of GPU-written memory whose downloads are on their way (Nx2Reactive::WaitLanding):
    /// waits for them to land, without the GPU thread. Returns the area once they have, or the
    /// last answer (not preemtive) when no download covers it anymore or after 16 ms (a flush
    /// would wait for the same host GPU work); the read then flushes.
    VideoCore::RasterizerDownloadArea WaitForLanding(DAddr addr, u64 size,
                                                     VideoCore::RasterizerDownloadArea area) {
        auto& landing = VideoCommon::Nx2Reactive::GetLanding();
        VideoCore::RasterizerInterface* const rasterizer = renderer->ReadRasterizer();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{16};
        landing.waiters.fetch_add(1, std::memory_order_acq_rel);
        while (true) {
            const u64 generation = landing.generation.load(std::memory_order_acquire);
            area = rasterizer->GetFlushArea(addr, size);
            if (area.preemtive || !area.wait_landing) {
                break;
            }
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                break;
            }
            std::unique_lock lock{landing.mutex};
            landing.cv.wait_until(lock, (std::min)(deadline, now + std::chrono::milliseconds{1}),
                                  [&] {
                                      return landing.generation.load(std::memory_order_acquire) !=
                                             generation;
                                  });
        }
        landing.waiters.fetch_sub(1, std::memory_order_acq_rel);
        area.wait_landing = false;
        return area;
    }

    VideoCore::RasterizerDownloadArea OnCPURead(DAddr addr, u64 size) {
        auto raster_area = renderer->ReadRasterizer()->GetFlushArea(addr, size);
        if (raster_area.preemtive) {
            return raster_area;
        }
        if (raster_area.wait_landing) {
            raster_area = WaitForLanding(addr, size, raster_area);
            if (raster_area.preemtive) {
                return raster_area;
            }
        }
        raster_area.preemtive = true;
        // A guest CPU read of GPU-written memory (reactive flushing): count them, they decide
        // whether buffer data must be copied back at every fence (EDEN_NX2_BUFFER_DOWNLOADS).
        if (static std::atomic<u64> reads{}; (reads.fetch_add(1, std::memory_order_relaxed) & 63) == 0) {
            LOG_INFO(HW_GPU, "CPU read of GPU-written memory [{:#x}, {:#x}) ({} so far)",
                     raster_area.start_address, raster_area.end_address,
                     reads.load(std::memory_order_relaxed));
        }
        if (VideoCommon::Nx2Reactive::AsyncFlush()) {
            // The GPU thread queues the data for a download it submits at once and goes on;
            // this thread waits for the copy to land (EDEN_NX2_REACTIVE_ASYNC_FLUSH). Without a
            // download queued it flushes right there, as below.
            bool queued = false;
            const u64 async_fence = RequestSyncOperation([this, raster_area, &queued]() {
                InvalidateGPUCache();
                VideoCore::RasterizerInterface* const rasterizer = renderer->ReadRasterizer();
                const u64 area_size = raster_area.end_address - raster_area.start_address;
                queued = rasterizer->QueueFlushRegion(raster_area.start_address, area_size);
                if (!queued) {
                    rasterizer->FlushRegion(raster_area.start_address, area_size);
                }
            });
            gpu_thread.TickGPU(is_async);
            WaitForSyncOperation(async_fence);
            bool landed = !queued;
            if (queued) {
                VideoCore::RasterizerDownloadArea pending = raster_area;
                pending.preemtive = false;
                pending.wait_landing = true;
                landed = WaitForLanding(addr, size, pending).preemtive;
            }
            if (landed) {
                return raster_area;
            }
            // Not landed (a newer write or a pending CPU write kept the page GPU-written, or no
            // copy covered it): flush as before.
        }
        const u64 fence = RequestSyncOperation([this, raster_area]() {
            // Apply collected guest writes before downloading GPU data over guest memory.
            InvalidateGPUCache();
            renderer->ReadRasterizer()->FlushRegion(raster_area.start_address, raster_area.end_address - raster_area.start_address);
        });
        gpu_thread.TickGPU(is_async);
        WaitForSyncOperation(fence);
        return raster_area;
    }

    /// Notify rasterizer that any caches of the specified region should be invalidated
    void InvalidateRegion(DAddr addr, u64 size, bool preserve_gpu_writes) {
        if (preserve_gpu_writes) {
            VideoCore::RasterizerInterface* rasterizer = renderer->ReadRasterizer();
            if (!rasterizer->TryCacheInvalidation(addr, size)) {
                const u64 fence = RequestSyncOperation([rasterizer, addr, size] {
                    rasterizer->FlushRegion(addr, size, VideoCommon::CacheType::BufferCache);
                    rasterizer->OnCacheInvalidation(addr, size);
                });
                gpu_thread.TickGPU(is_async);
                WaitForSyncOperation(fence);
            }
            return;
        }
        gpu_thread.InvalidateRegion(addr, size);
    }

    bool OnCPUWrite(DAddr addr, u64 size) {
        return renderer->ReadRasterizer()->OnCPUWrite(addr, size);
    }

    /// Notify rasterizer that any caches of the specified region should be flushed and invalidated
    void FlushAndInvalidateRegion(DAddr addr, u64 size) {
        gpu_thread.FlushAndInvalidateRegion(addr, size, is_async);
    }

    void RequestComposite(std::vector<Tegra::FramebufferConfig>&& layers, std::vector<Service::Nvidia::NvFence>&& fences) {
        const size_t num_fences{fences.size()};
        size_t current_request_counter{};
        if (num_fences != 0) {
            std::unique_lock<std::mutex> lk(request_swap_mutex);
            if (free_swap_counters.empty()) {
                current_request_counter = request_swap_counters.size();
                request_swap_counters.emplace_back(num_fences);
            } else {
                current_request_counter = free_swap_counters.front();
                request_swap_counters[current_request_counter] = num_fences;
                free_swap_counters.pop_front();
            }
        }
        pending_composite_fence = RequestSyncOperation(
            [this, current_request_counter, num_fences, composite_layers = std::move(layers),
             composite_fences = std::move(fences)] {
                if (num_fences == 0) {
                    renderer->Composite(composite_layers);
                    return;
                }
                auto& syncpoint_manager = system.Host1x().GetSyncpointManager();
                const auto executer = [this, current_request_counter, composite_layers]() {
                    {
                        std::unique_lock<std::mutex> lk(request_swap_mutex);
                        if (--request_swap_counters[current_request_counter] != 0) {
                            return;
                        }
                        free_swap_counters.push_back(current_request_counter);
                    }
                    renderer->Composite(composite_layers);
                };
                for (size_t i = 0; i < num_fences; i++) {
                    syncpoint_manager.RegisterGuestAction(composite_fences[i].id,
                                                          composite_fences[i].value, executer);
                }
            });
        gpu_thread.TickGPU(is_async);
    }

    void WaitForComposite() {
        const u64 fence = pending_composite_fence;
        if (fence == 0) {
            return;
        }
        pending_composite_fence = 0;
        if (shutting_down.load(std::memory_order_relaxed)) {
            return;
        }
        WaitForSyncOperation(fence);
    }

    std::vector<u8> GetAppletCaptureBuffer() {
        std::vector<u8> out;

        const auto wait_fence =
            RequestSyncOperation([&] { out = renderer->GetAppletCaptureBuffer(); });
        gpu_thread.TickGPU(is_async);
        WaitForSyncOperation(wait_fence);

        return out;
    }

    Core::System& system;

    // Destruction of thread must be done before all (non trivial)
    // previous members has been destroyed
    VideoCommon::GPUThread::ThreadManager gpu_thread;

    std::unique_ptr<VideoCore::RendererBase> renderer;
    const bool use_nvdec;

    s32 new_channel_id{1};
    /// Shader build notifier
    VideoCore::ShaderNotify shader_notify;
    /// When true, we are about to shut down emulation session, so terminate outstanding tasks
    std::atomic_bool shutting_down{};

    std::array<std::atomic<u32>, Service::Nvidia::MaxSyncPoints> syncpoints{};

    std::array<std::list<u32>, Service::Nvidia::MaxSyncPoints> syncpt_interrupts;

    std::mutex sync_mutex;
    std::mutex device_mutex;

    std::condition_variable sync_cv;

    std::list<std::function<void()>> sync_requests;
    /// sync_requests is not empty (set and cleared under sync_request_mutex; TickWork reads it
    /// without the lock to skip the lock when nothing is queued).
    std::atomic<bool> has_sync_requests{};
    std::atomic<u64> current_sync_fence{};
    u64 last_sync_fence{};

    VideoCommon::PendingMaps pending_maps;
    std::mutex sync_request_mutex;
    std::condition_variable sync_request_cv;

    const bool is_async;

    std::unique_ptr<Core::Frontend::GraphicsContext> cpu_context;

    Tegra::Control::Scheduler scheduler;
    ::Common::unordered_map<s32, std::shared_ptr<Tegra::Control::ChannelState>> channels;
    Tegra::Control::ChannelState* current_channel;
    s32 bound_channel{-1};

    std::deque<size_t> free_swap_counters;
    std::deque<size_t> request_swap_counters;
    std::mutex request_swap_mutex;
    u64 pending_composite_fence{};
};

GPU::GPU(Core::System& system, bool is_async, bool use_nvdec)
    : impl{std::make_unique<Impl>(system, is_async, use_nvdec)}
{}

GPU::~GPU() = default;

std::shared_ptr<Control::ChannelState> GPU::AllocateChannel() {
    return impl->AllocateChannel();
}

void GPU::InitChannel(Control::ChannelState& to_init, u64 program_id) {
    impl->InitChannel(to_init, program_id);
}

void GPU::BindChannel(s32 channel_id) {
    impl->BindChannel(channel_id);
}

void GPU::ReleaseChannel(Control::ChannelState& to_release) {
    impl->ReleaseChannel(to_release);
}

void GPU::InitAddressSpace(Tegra::MemoryManager& memory_manager) {
    impl->InitAddressSpace(memory_manager);
}

void GPU::BindRenderer(std::unique_ptr<VideoCore::RendererBase> renderer) {
    impl->BindRenderer(std::move(renderer));
}

void GPU::FlushCommands() {
    impl->FlushCommands();
}

void GPU::InvalidateGPUCache() {
    impl->InvalidateGPUCache();
}

bool GPU::HasPendingCPUWrites(DAddr addr, u64 size) {
    for (auto& manager : impl->system.GetGPUDirtyMemoryManager()) {
        if (manager.Overlaps(addr, size)) {
            return true;
        }
    }
    return false;
}

void GPU::ForEachPendingCpuWrite(const std::function<void(DAddr, size_t)>& callback) {
    impl->ForEachPendingCpuWrite(callback);
}

void GPU::CollectHandoffWrite(DAddr addr, u64 size) {
    // The last manager is kept for the handoff (core.cpp). Its writers all hold the buffer cache's
    // lock, so it has one producer at a time, like a core's.
    impl->system.GetGPUDirtyMemoryManager().back().Collect(addr, size);
}

void GPU::OnCommandListEnd() {
    impl->OnCommandListEnd();
}

u64 GPU::RequestFlush(DAddr addr, std::size_t size) {
    return impl->RequestSyncOperation([this, addr, size]() {
        impl->renderer->ReadRasterizer()->FlushRegion(addr, size);
    });
}

u64 GPU::RunOnGpuThread(std::function<void()>&& action) {
    return impl->RequestSyncOperation(std::move(action));
}

u64 GPU::CurrentSyncRequestFence() const {
    return impl->CurrentSyncRequestFence();
}

void GPU::WaitForSyncOperation(u64 fence) {
    return impl->WaitForSyncOperation(fence);
}

void GPU::TickWork() {
    impl->TickWork();
}

/// Gets a mutable reference to the Host1x interface
Host1x::Host1x& GPU::Host1x() {
    return impl->system.Host1x();
}

/// Gets an immutable reference to the Host1x interface.
const Host1x::Host1x& GPU::Host1x() const {
    return impl->system.Host1x();
}

Engines::Maxwell3D& GPU::Maxwell3D() {
    return impl->current_channel->payload->maxwell_3d;
}

const Engines::Maxwell3D& GPU::Maxwell3D() const {
    return impl->current_channel->payload->maxwell_3d;
}

Engines::KeplerCompute& GPU::KeplerCompute() {
    return impl->current_channel->payload->kepler_compute;
}

const Engines::KeplerCompute& GPU::KeplerCompute() const {
    return impl->current_channel->payload->kepler_compute;
}

Tegra::DmaPusher& GPU::DmaPusher() {
    return impl->current_channel->payload->dma_pusher;
}

const Tegra::DmaPusher& GPU::DmaPusher() const {
    return impl->current_channel->payload->dma_pusher;
}

VideoCore::RendererBase& GPU::Renderer() {
    return *impl->renderer;
}

const VideoCore::RendererBase& GPU::Renderer() const {
    return *impl->renderer;
}

VideoCore::ShaderNotify& GPU::ShaderNotify() {
    return impl->shader_notify;
}

const VideoCore::ShaderNotify& GPU::ShaderNotify() const {
    return impl->shader_notify;
}

void GPU::RequestComposite(std::vector<Tegra::FramebufferConfig>&& layers,
                           std::vector<Service::Nvidia::NvFence>&& fences) {
    impl->RequestComposite(std::move(layers), std::move(fences));
}

void GPU::WaitForComposite() {
    impl->WaitForComposite();
}

std::vector<u8> GPU::GetAppletCaptureBuffer() {
    return impl->GetAppletCaptureBuffer();
}

u64 GPU::GetTicks() const {
    return impl->GetTicks();
}

bool GPU::IsAsync() const {
    return impl->is_async;
}

bool GPU::HasSuspendedChannels() const {
    return impl->scheduler.HasPendingChannelsAnyThread();
}

bool GPU::UseNvdec() const {
    return impl->use_nvdec;
}

void GPU::RendererFrameEndNotify() {
    impl->RendererFrameEndNotify();
}

void GPU::Start() {
    impl->Start();
}

void GPU::NotifyShutdown() {
    impl->NotifyShutdown();
}

void GPU::WaitForIdle() {
    impl->WaitForIdle();
}

void GPU::ObtainContext() {
    impl->ObtainContext();
}

void GPU::ReleaseContext() {
    impl->ReleaseContext();
}

void GPU::PushGPUEntries(s32 channel, Tegra::CommandList&& entries) {
    impl->PushGPUEntries(channel, std::move(entries));
}

VideoCore::RasterizerDownloadArea GPU::OnCPURead(PAddr addr, u64 size) {
    return impl->OnCPURead(addr, size);
}

void GPU::FlushRegion(DAddr addr, u64 size) {
    impl->FlushRegion(addr, size);
}

void GPU::RunInOrder(std::function<void()>&& func) {
    impl->RunInOrder(std::move(func));
}

void GPU::MapInOrder(GPUVAddr gpu_addr, u64 size, std::function<void()>&& func) {
    impl->MapInOrder(gpu_addr, size, std::move(func));
}

bool GPU::IsMapPending(GPUVAddr gpu_addr, u64 size) const {
    return impl->IsMapPending(gpu_addr, size);
}

void GPU::InvalidateRegion(DAddr addr, u64 size, bool preserve_gpu_writes) {
    impl->InvalidateRegion(addr, size, preserve_gpu_writes);
}

bool GPU::OnCPUWrite(DAddr addr, u64 size) {
    return impl->OnCPUWrite(addr, size);
}

void GPU::FlushAndInvalidateRegion(DAddr addr, u64 size) {
    impl->FlushAndInvalidateRegion(addr, size);
}

} // namespace Tegra
