// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#ifdef _MSC_VER
#include <intrin.h>
#elif defined(ARCHITECTURE_x86_64)
#include <x86intrin.h>
#endif
#include <chrono>
#include <cstdlib>
#include <string>
#include <unordered_set>

#include "common/common_types.h"
#include "video_core/gpu_split.h"
#include "common/logging.h"
#include "common/cpu_features.h"
#include "common/thread_cpu_clock.h"
#include "common/nx2_timeline.h"

// EDEN_NX2_GPU_PROFILE: time where the GPU thread spends its time. Each zone counts its own
// (exclusive) time, nested zones excluded; every 10 s the thread logs each zone's share.
namespace VideoCommon::Nx2Profile {

enum class Zone : u32 {
    Other, ///< GPU thread time outside every zone below (register writes, command parsing)
    Idle,
    PendingSleep,
    ResumePending,
    Push,
    Tick,
    Run,
    FlushInvalidate,
    Macro,
    Draw,
    Clear,
    Compute,
    FlushWork,
    PipelineLookup,
    Configure,
    SyncDescriptors,
    ReadHandles,
    FillImageViews,
    TextureBuffers,
    UpdateBuffers,
    DescriptorAcquire,
    BindStageBuffers,
    PushDescriptors,
    RenderTargets,
    ConfigureDraw,
    DynamicState,
    DrawRecord,
    TextureUpload,
    Fence,        ///< Fences made (SignalFence)
    FenceSkipped, ///< Empty fences not made
    Method,       ///< Engine method calls (what a split GPU thread's back half would run)
    SplitDrain,   ///< Split GPU thread: the front waits for the back (gpu_split.h)
    SplitWait,    ///< Split GPU thread: the back waits for work
    WriterWait,
    ListSubmitWait,
    ListGpuWait,
    MemoryQuery, ///< Device::QueryDeviceMemoryUsage, the driver's memory budget query (EXP29)
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Zone::Count)> zone_names{
    "other",         "idle",          "pending sleep",  "resume pending", "push",
    "tick",          "run",           "flush/inval",    "macro",          "draw",
    "clear",         "compute",       "flush work",     "pipeline lookup", "configure",
    "sync descriptors", "read handles", "fill image views", "texture buffers", "update buffers",
    "descriptor acquire", "bind stage buffers", "push descriptors", "render targets",
    "configure draw", "dynamic state", "draw record", "texture upload", "fence",
    "fence skipped", "method", "split drain", "split wait",
    "writer wait", "list submit wait", "list GPU wait", "memory query",
};

// Read once, when the GPU thread starts (Initialize): Enabled() runs several times per method
// write (no guard to check). Not at static initialization: nx2_switches.txt is only applied in
// main(). Off while EDEN_NX2_SAMPLE_PROFILE samples (common/nx2_sample_profile.h): the zone timers
// would show up in the samples.
inline bool g_enabled = false;

inline void Initialize() {
    g_enabled = std::getenv("EDEN_NX2_GPU_PROFILE") != nullptr &&
                std::getenv("EDEN_NX2_SAMPLE_PROFILE") == nullptr;
}

inline bool Enabled() {
    return g_enabled;
}

// Nanoseconds; from the host's own counter where the wall clock reads it directly (the zones
// run ~150,000 times a second, and QueryPerformanceCounter cost ~5% of the GPU thread).
inline u64 Now() {
    return static_cast<u64>(Common::g_wall_clock.GetTimeNS().count());
}

/// What a front macro memory read (MME DMA) had to wait for: the newest unexecuted writer.
enum class DmaWaitCause : u32 {
    ComputeCopy,       ///< Compute or copy work
    HostSync,          ///< The work before a host-level wait
    ConstBufferUpload, ///< A 3D constant buffer upload (const_buffer.buffer)
    InlineUpload,      ///< 3D inline data
    Macro3D,           ///< A 3D macro register
    Report,            ///< A 3D report (report_semaphore.query)
    DmaLaunch,         ///< 3D launch_dma
    SyncInfo,          ///< 3D sync_info
    WaitForIdle,       ///< 3D wait_for_idle
    Other3D,           ///< Another memory-writing 3D method
    OtherEngine,       ///< A compute, DMA, inline-to-memory or 2D engine method
    Deferred,          ///< Puller semaphore releases and maps
    AvoidedByRange,    ///< Not waited: the pending writes' ranges miss the read (RangeLastWrite)
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(DmaWaitCause::Count)>
    dma_wait_cause_names{"compute/copy",  "host sync",    "cbuf upload", "inline upload",
                         "3D macro",      "report",       "dma launch",  "sync info",
                         "wait for idle", "other 3D",     "other engine", "deferred",
                         "avoided (range)"};
// What the zones and drain waits measure in: the time stamp counter without the fences the wall
// clock reads it with (those fences cost ~10% of the GPU back thread with ~45,000 draws a
// second). Report converts it with the wall clock.
inline u64 Ticks() {
#ifdef ARCHITECTURE_x86_64
    return __rdtsc();
#else
    return Now();
#endif
}

struct ThreadData {
    std::array<u64, static_cast<size_t>(Zone::Count)> exclusive_ns{};
    std::array<u64, static_cast<size_t>(Zone::Count)> count{};
    std::array<u64, 64> counters{}; ///< See Counter (static_assert below)
    std::array<u64, static_cast<size_t>(DmaWaitCause::Count)> dma_wait_causes{};
    std::array<u64, static_cast<size_t>(DrainSite::Count)> drain_ns{};
    std::array<u64, static_cast<size_t>(DrainSite::Count)> drain_count{};
    u64 child_ns{};    ///< Time of the zones nested in the current one
    u64 window_start{};
    u64 window_start_ticks{};
    std::optional<u64> cpu_start_ns;
};

inline thread_local ThreadData t_data;

/// The zones a timeline records without detail: waits, whole draws, dispatches and macros.
[[nodiscard]] constexpr bool IsCoarse(Zone zone) {
    switch (zone) {
    case Zone::Idle:
    case Zone::PendingSleep:
    case Zone::ResumePending:
    case Zone::Push:
    case Zone::Macro:
    case Zone::Draw:
    case Zone::Clear:
    case Zone::Compute:
    case Zone::FlushWork:
    case Zone::TextureUpload:
    case Zone::Fence:
    case Zone::SplitDrain:
    case Zone::SplitWait:
    case Zone::WriterWait:
    case Zone::ListSubmitWait:
    case Zone::ListGpuWait:
        return true;
    default:
        return false;
    }
}

class Scope {
public:
    explicit Scope(Zone zone_)
        : zone{zone_}, enabled{Enabled()},
          timeline{Common::Nx2Timeline::Active() &&
                   (Common::Nx2Timeline::Detail() || IsCoarse(zone_))} {
        if (timeline) {
            timeline_start = Common::Nx2Timeline::Now();
        }
        if (!enabled) {
            return;
        }
        parent_child_ns = t_data.child_ns;
        t_data.child_ns = 0;
        start = Ticks();
    }

    ~Scope() {
        if (timeline) {
            Common::Nx2Timeline::Record(zone_names[static_cast<size_t>(zone)], timeline_start,
                                        Common::Nx2Timeline::Now());
        }
        if (!enabled) {
            return;
        }
        const u64 elapsed = Ticks() - start;
        auto& data = t_data;
        const auto index = static_cast<size_t>(zone);
        data.exclusive_ns[index] += elapsed - data.child_ns;
        ++data.count[index];
        data.child_ns = parent_child_ns + elapsed;
    }

    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    Zone zone;
    bool enabled;
    bool timeline;
    u64 start{};
    u64 parent_child_ns{};
    u64 timeline_start{};
};

/// Amounts counted on the GPU thread, logged per second with the zones.
enum class Counter : u32 {
    DownloadBytes,  ///< Buffer data copied back from the host GPU at fences
    DownloadCopies, ///< Buffer ranges copied back
    DmaReadBytes,   ///< Guest memory the native MME read with DMA
    DownloadBytesOnDmaPages, ///< Of DownloadBytes: on pages the MME has DMA-read
    MethodWrites,   ///< 3D register writes (Maxwell3D::CallMethod below the macro range)
    SameValueWrites, ///< Of MethodWrites: the register already held the value
    AcquireChecks,  ///< Semaphore acquires evaluated
    AcquireWaits,   ///< Of AcquireChecks: not met yet (the channel waits)
    AcquiresFast,   ///< EDEN_NX2_ACQUIRE_PRECHECK: acquires met without draining the split
    AcquiresQueued, ///< Acquires met by a release still queued for the split's back
    PtrCacheHits,   ///< EDEN_NX2_HANDLE_PTR_CACHE: handle/descriptor reads, page pointer cached
    PtrCacheMisses, ///< EDEN_NX2_HANDLE_PTR_CACHE: handle/descriptor reads, page looked up
    UboInPlaceSame, ///< EDEN_NX2_UBO_INPLACE: small uniform buffers unchanged, nothing copied
    UboInPlaceChanged, ///< EDEN_NX2_UBO_INPLACE: small uniform buffers changed, copied once
    AcquireNoRelease,  ///< Drained acquires: no release of the word was sent this session
    AcquireReleaseRan, ///< Drained acquires: the last release of the word already ran on the back
    AcquireLaterWrite, ///< Drained acquires: work sent after its release may write the word
    AcquirePayloadMiss, ///< Drained acquires: the queued release's payload does not meet it
    AcquireMetAfterDrain, ///< Drained acquires met at once after the drain
    DmaReadsForwarded, ///< MME DMA reads served from a queued constant buffer upload
    DmaReadsEarly,  ///< MME DMA reads without a drain (Maxwell3D::FrontDmaReadEarly)
    DmaReadsWaitUnknown, ///< Of those, waited for a write the front can't place
    DmaReadsWaitGpu,     ///< waited for all queued GPU work with unknown destinations
    DmaReadsWaitWrite,   ///< waited for a noted write to the words' pages
    AcquirePolls,   ///< Polls of a waiting acquire (Puller::CanResumeAcquire)
    AcquirePollsFast, ///< Of AcquirePolls: answered from the word without draining the back
    SemNoFastPath,    ///< SemaphoreReadableNow false: split off or EDEN_NX2_SEM_FAST=0
    SemGpuWrittenPage, ///< SemaphoreReadableNow false: the word's page is marked GPU-written
    SemAsyncPending,  ///< SemaphoreReadableNow false: a counter report or timed release is pending
    SemWritersWait,   ///< SemaphoreReadableNow false: WaitForWritesTo gave up
    AcquireNoDrain,   ///< Acquires of a current word not met yet that registered without a drain
    Count,
};
static_assert(static_cast<size_t>(Counter::Count) <= 64, "Counter outgrew the counters array");
static_assert(static_cast<size_t>(Counter::Count) <= std::tuple_size_v<decltype(ThreadData::counters)>);

/// Method calls on this thread by method (3D registers on the back, macros on the front), and
/// same-value writes among them; MME DMA reads by the macro that made them. The most frequent
/// are logged with the zones.
inline thread_local std::array<u32, 0x2000> t_method_count{};
inline thread_local std::array<u32, 0x2000> t_method_same{};
inline thread_local std::array<u32, 0x2000> t_dma_reads{};

inline void CountMethod(u32 method, bool same) {
    if (Enabled() && method < t_method_count.size()) {
        ++t_method_count[method];
        t_method_same[method] += same ? 1 : 0;
    }
}

inline void CountDmaRead(u32 macro) {
    if (Enabled() && macro < t_dma_reads.size()) {
        ++t_dma_reads[macro];
    }
}

inline std::string TopMethods(const std::array<u32, 0x2000>& counts,
                              const std::array<u32, 0x2000>* same, u64 window) {
    std::array<u32, 0x2000> order;
    for (u32 i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    constexpr size_t top = 24;
    std::partial_sort(order.begin(), order.begin() + top, order.end(),
                      [&counts](u32 a, u32 b) { return counts[a] > counts[b]; });
    std::string text;
    for (size_t i = 0; i < top && counts[order[i]] != 0; ++i) {
        const u32 method = order[i];
        text += fmt::format(" {:#x} {:.0f}/s", method,
                            static_cast<double>(counts[method]) * 1e9 / window);
        if (same != nullptr && (*same)[method] != 0) {
            text += fmt::format(" ({:.0f} same)", static_cast<double>((*same)[method]) * 1e9 / window);
        }
        text += ',';
    }
    if (!text.empty()) {
        text.pop_back();
    }
    return text;
}

/// Device pages (4 KiB) the native MME has DMA-read since start (GPU thread only; filled only
/// while profiling): which buffer downloads anything on the GPU thread reads back.
inline std::unordered_set<u64>& DmaReadPages() {
    static std::unordered_set<u64> pages;
    return pages;
}

inline void Add(Counter counter, u64 amount) {
    if (Enabled()) {
        t_data.counters[static_cast<size_t>(counter)] += amount;
    }
}

/// The front waited for the back since `start` (Ticks) for `site`.
inline void AddDrain(DrainSite site, u64 start) {
    if (Enabled()) {
        const auto index = static_cast<size_t>(site);
        t_data.drain_ns[index] += Ticks() - start;
        ++t_data.drain_count[index];
    }
}

inline void AddDmaWaitCause(DmaWaitCause cause) {
    if (Enabled()) {
        ++t_data.dma_wait_causes[static_cast<size_t>(cause)];
    }
}

/// Called by the GPU thread (and the split's back thread) between commands: logs and resets
/// the totals every 10 s.
inline void Report(const char* thread_name = "GPU thread") {
    if (!Enabled()) {
        return;
    }
    auto& data = t_data;
    const u64 now = Now();
    if (data.window_start == 0) {
        data.window_start = now;
        data.window_start_ticks = Ticks();
        data.cpu_start_ns = Common::CurrentThreadCpuTimeNs();
        data.child_ns = 0;
        return;
    }
    const u64 window = now - data.window_start;
    if (window < 10'000'000'000ULL) {
        return;
    }
    const auto cpu_now = Common::CurrentThreadCpuTimeNs();
    const u64 now_ticks = Ticks();
    const u64 window_ticks = (std::max<u64>)(now_ticks - data.window_start_ticks, 1);
    // Time in no zone at all is the wall time minus the zones' totals.
    u64 zoned{};
    for (size_t i = 1; i < data.exclusive_ns.size(); ++i) {
        zoned += data.exclusive_ns[i];
    }
    data.exclusive_ns[0] = window_ticks > zoned ? window_ticks - zoned : 0;
    std::string line;
    for (size_t i = 0; i < data.exclusive_ns.size(); ++i) {
        if (data.exclusive_ns[i] == 0 && data.count[i] == 0) {
            continue;
        }
        // Per second of wall time: milliseconds spent and calls made.
        const double ms = static_cast<double>(data.exclusive_ns[i]) * 1000.0 / window_ticks;
        const double calls = static_cast<double>(data.count[i]) * 1e9 / window;
        line += fmt::format(" | {} {:.1f} ms ({:.0f}/s)", zone_names[i], ms, calls);
    }
    line += fmt::format(" | buffer downloads {:.0f} KiB in {:.0f} copies ({:.0f} KiB on pages "
                        "MME DMA reads) | MME DMA reads {:.0f} KiB",
                        static_cast<double>(data.counters[0]) * 1e9 / 1024.0 / window,
                        static_cast<double>(data.counters[1]) * 1e9 / window,
                        static_cast<double>(data.counters[3]) * 1e9 / 1024.0 / window,
                        static_cast<double>(data.counters[2]) * 1e9 / 1024.0 / window);
    line += fmt::format(" | 3D register writes {:.0f} ({:.0f} same value)",
                        static_cast<double>(data.counters[4]) * 1e9 / window,
                        static_cast<double>(data.counters[5]) * 1e9 / window);
    line += fmt::format(" | acquires {:.0f} ({:.0f} waited)",
                        static_cast<double>(data.counters[6]) * 1e9 / window,
                        static_cast<double>(data.counters[7]) * 1e9 / window);
    if (const u64 queued = data.counters[static_cast<size_t>(Counter::AcquiresQueued)]; queued != 0) {
        line += fmt::format(" | acquires met by a queued release {:.0f}",
                            static_cast<double>(queued) * 1e9 / window);
    }
    {
        const auto rate = [&](Counter counter) {
            return static_cast<double>(data.counters[static_cast<size_t>(counter)]) * 1e9 / window;
        };
        if (data.counters[static_cast<size_t>(Counter::AcquireChecks)] != 0) {
            line += fmt::format(" | drained acquires: no release {:.0f}, release ran {:.0f}, "
                                "later write {:.0f}, payload miss {:.0f}, met after drain {:.0f}",
                                rate(Counter::AcquireNoRelease), rate(Counter::AcquireReleaseRan),
                                rate(Counter::AcquireLaterWrite),
                                rate(Counter::AcquirePayloadMiss),
                                rate(Counter::AcquireMetAfterDrain));
        }
    }
    if (const u64 forwarded = data.counters[static_cast<size_t>(Counter::DmaReadsForwarded)];
        forwarded != 0) {
        line += fmt::format(" | MME DMA reads from queued uploads {:.0f}",
                            static_cast<double>(forwarded) * 1e9 / window);
    }
    if (const u64 fast = data.counters[static_cast<size_t>(Counter::AcquiresFast)]; fast != 0) {
        line += fmt::format(" | acquires without a drain {:.0f}",
                            static_cast<double>(fast) * 1e9 / window);
    }
    if (const u64 hits = data.counters[static_cast<size_t>(Counter::PtrCacheHits)],
        misses = data.counters[static_cast<size_t>(Counter::PtrCacheMisses)];
        hits != 0 || misses != 0) {
        line += fmt::format(" | handle page pointers {:.0f} cached, {:.0f} looked up",
                            static_cast<double>(hits) * 1e9 / window,
                            static_cast<double>(misses) * 1e9 / window);
    }
    if (const u64 same = data.counters[static_cast<size_t>(Counter::UboInPlaceSame)],
        changed = data.counters[static_cast<size_t>(Counter::UboInPlaceChanged)];
        same != 0 || changed != 0) {
        line += fmt::format(" | uniform buffers compared in place {:.0f} unchanged, {:.0f} changed",
                            static_cast<double>(same) * 1e9 / window,
                            static_cast<double>(changed) * 1e9 / window);
    }
    if (const u64 early = data.counters[static_cast<size_t>(Counter::DmaReadsEarly)]; early != 0) {
        const auto rate = [&](Counter counter) {
            return static_cast<double>(data.counters[static_cast<size_t>(counter)]) * 1e9 / window;
        };
        line += fmt::format(" | MME DMA reads without a drain {:.0f} (waited for: unknown writes "
                            "{:.0f}, GPU work {:.0f}, writes to the words {:.0f})",
                            static_cast<double>(early) * 1e9 / window,
                            rate(Counter::DmaReadsWaitUnknown), rate(Counter::DmaReadsWaitGpu),
                            rate(Counter::DmaReadsWaitWrite));
    }
    {
        const auto rate = [&](Counter counter) {
            return static_cast<double>(data.counters[static_cast<size_t>(counter)]) * 1e9 / window;
        };
        if (data.counters[static_cast<size_t>(Counter::AcquirePolls)] != 0 ||
            data.counters[static_cast<size_t>(Counter::AcquireNoDrain)] != 0) {
            line += fmt::format(" | acquire polls {:.0f} ({:.0f} without a drain), acquires "
                                "waiting without a drain {:.0f}, semaphore word not readable: no "
                                "fast path {:.0f}, GPU-written page {:.0f}, report pending {:.0f}, "
                                "writers {:.0f}",
                                rate(Counter::AcquirePolls), rate(Counter::AcquirePollsFast),
                                rate(Counter::AcquireNoDrain), rate(Counter::SemNoFastPath),
                                rate(Counter::SemGpuWrittenPage), rate(Counter::SemAsyncPending),
                                rate(Counter::SemWritersWait));
        }
    }
    std::string drains;
    for (size_t i = 0; i < data.drain_count.size(); ++i) {
        if (data.drain_count[i] != 0) {
            drains += fmt::format(" {} {:.1f} ms ({:.0f}/s),", drain_site_names[i],
                                  static_cast<double>(data.drain_ns[i]) * 1000.0 / window_ticks,
                                  static_cast<double>(data.drain_count[i]) * 1e9 / window);
        }
    }
    if (!drains.empty()) {
        drains.pop_back();
        line += " | waits for the back by site:" + drains;
    }
    std::string causes;
    for (size_t i = 0; i < data.dma_wait_causes.size(); ++i) {
        if (data.dma_wait_causes[i] != 0) {
            causes += fmt::format(" {} {:.0f}/s,", dma_wait_cause_names[i],
                                  static_cast<double>(data.dma_wait_causes[i]) * 1e9 / window);
        }
    }
    if (!causes.empty()) {
        causes.pop_back();
        line += " | macro dma reads waited for:" + causes;
    }
    if (cpu_now && data.cpu_start_ns && *cpu_now >= *data.cpu_start_ns) {
        line += fmt::format(" | host_cpu_busy_ms_per_s={:.3f} window_wall_ms={:.3f}",
                            static_cast<double>(*cpu_now - *data.cpu_start_ns) * 1000.0 / window,
                            static_cast<double>(window) / 1e6);
    } else {
        line += fmt::format(" | host_cpu_busy_ms_per_s=unavailable window_wall_ms={:.3f}",
                            static_cast<double>(window) / 1e6);
    }
    LOG_INFO(HW_GPU, "{} per second:{}", thread_name, line);
    if (const std::string methods = TopMethods(t_method_count, &t_method_same, window); !methods.empty()) {
        LOG_INFO(HW_GPU, "{} methods per second:{}", thread_name, methods);
    }
    if (const std::string reads = TopMethods(t_dma_reads, nullptr, window); !reads.empty()) {
        LOG_INFO(HW_GPU, "{} MME DMA reads per second by macro:{}", thread_name, reads);
    }
    t_method_count = {};
    t_method_same = {};
    t_dma_reads = {};
    data = {};
    data.window_start = now;
    data.window_start_ticks = now_ticks;
    data.cpu_start_ns = cpu_now;
}

} // namespace VideoCommon::Nx2Profile
