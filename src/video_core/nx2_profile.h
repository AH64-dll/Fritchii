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
    TickWork,       ///< GPU::TickWork (sync requests: reactive flushes)
    InlineToMemory, ///< AccelerateInlineToMemory
    // The split back's own work, inside what was "other" and "method" (count: entries or calls).
    RingDecode,   ///< BackLoop's batches: entry decode and dispatch, Engine entries, method sinks
    DeltaApply,   ///< Kind::StateDelta (Maxwell3D::ApplyFrontDelta)
    CbWrite,      ///< Kind::CbWrite (Maxwell3D::BackCbWrite)
    CbBind,       ///< Kind::CbBind and bind group methods (Maxwell3D::ProcessCBBind)
    CbMulti,      ///< Constant buffer data in method runs (Maxwell3D::ProcessCBMultiData)
    DrawBatch,    ///< Kind::DrawBatch expansion (Maxwell3D::RunDrawBatch)
    DrawManager,  ///< DrawManager begin/end/deferred, outside the rasterizer's draw
    Method3D,     ///< Maxwell3D method entries ("method" keeps the other engines and runs)
    Call,         ///< Kind::Call entries, by thunk in the "calls" report line
    FinishMacro,  ///< Maxwell3D::FinishFrontMacro (a Call after each front macro that drew)
    QueryReport,  ///< report_semaphore.query (Maxwell3D::ProcessQueryGet)
    CounterReset, ///< clear_report_value (Maxwell3D::ProcessCounterReset)
    RenderEnable, ///< render_enable.mode (Maxwell3D::ProcessQueryCondition)
    SyncPoint,    ///< sync_info (Maxwell3D::ProcessSyncPoint)
    WaitForIdle,  ///< wait_for_idle (rasterizer WaitForIdle)
    CacheBarrier, ///< invalidate_texture_data_cache, fragment and tiled cache barriers
    IdleHook,     ///< The back's idle hook (Scheduler::FlushDeferred: submits)
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
    "writer wait", "list submit wait", "list GPU wait", "tick work", "inline to memory",
    "ring decode", "delta apply", "cb write", "cb bind", "cb multi", "draw batch",
    "draw manager", "method 3D", "call", "finish macro", "query report", "counter reset",
    "render enable", "syncpoint", "wait for idle", "cache barrier", "idle hook",
};
static_assert(zone_names[static_cast<size_t>(Zone::Count) - 1] != nullptr,
              "every zone needs a name");

// Read once at startup: Enabled() runs several times per method write (no guard to check).
inline const bool g_enabled = std::getenv("EDEN_NX2_GPU_PROFILE") != nullptr;

inline bool Enabled() {
    return g_enabled;
}

// Nanoseconds; from the host's own counter where the wall clock reads it directly (the zones
// run ~150,000 times a second, and QueryPerformanceCounter cost ~5% of the GPU thread).
inline u64 Now() {
    return static_cast<u64>(Common::g_wall_clock.GetTimeNS().count());
}

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
    DeltaWords,     ///< Split back: StateDelta words applied
    DeltaSameWords, ///< Of DeltaWords: the register held the value already
    CbWriteWords,   ///< Split back: words in Kind::CbWrite entries
    BatchRepeats,   ///< Split back: instances a DrawBatch counted without replaying them
    RingWords,      ///< Split back: ring words consumed
    RingEntries,    ///< Split back: ring entries executed
    FusedWrites,    ///< Split back: compute writes in Kind::Writes entries (EDEN_NX2_COMPUTE_FUSE)
    MergedUploads,  ///< Split back: inline upload flushes joined to the one before (I2M_MERGE)
    Count,
};

struct ThreadData {
    std::array<u64, static_cast<size_t>(Zone::Count)> exclusive_ns{};
    std::array<u64, static_cast<size_t>(Zone::Count)> count{};
    std::array<u64, static_cast<size_t>(Counter::Count)> counters{};
    std::array<u64, static_cast<size_t>(DrainSite::Count)> drain_ns{};
    std::array<u64, static_cast<size_t>(DrainSite::Count)> drain_count{};
    u64 child_ns{};    ///< Time of the zones nested in the current one
    u64 window_start{};
    u64 window_start_ticks{};
    std::optional<u64> cpu_start_ns;
};

inline thread_local ThreadData t_data;

class Scope {
public:
    explicit Scope(Zone zone_) : zone{zone_}, enabled{Enabled()} {
        if (!enabled) {
            return;
        }
        parent_child_ns = t_data.child_ns;
        t_data.child_ns = 0;
        start = Ticks();
    }

    ~Scope() {
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
    u64 start{};
    u64 parent_child_ns{};
};

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

/// Called by the GPU thread (and the split's back thread) between commands: logs and resets
/// the totals every 10 s. Returns whether it logged (the window just ended).
inline bool Report(const char* thread_name = "GPU thread") {
    if (!Enabled()) {
        return false;
    }
    auto& data = t_data;
    const u64 now = Now();
    if (data.window_start == 0) {
        data.window_start = now;
        data.window_start_ticks = Ticks();
        data.cpu_start_ns = Common::CurrentThreadCpuTimeNs();
        data.child_ns = 0;
        return false;
    }
    const u64 window = now - data.window_start;
    if (window < 10'000'000'000ULL) {
        return false;
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
    if (data.counters[static_cast<size_t>(Counter::RingEntries)] != 0) {
        const auto per_second = [&](Counter counter) {
            return static_cast<double>(data.counters[static_cast<size_t>(counter)]) * 1e9 / window;
        };
        line += fmt::format(" | ring {:.0f} entries {:.0f} words | delta words {:.0f} ({:.0f} "
                            "same on the back) | cb write words {:.0f} | batch repeats {:.0f} | "
                            "fused compute writes {:.0f} | merged uploads {:.0f}",
                            per_second(Counter::RingEntries), per_second(Counter::RingWords),
                            per_second(Counter::DeltaWords), per_second(Counter::DeltaSameWords),
                            per_second(Counter::CbWriteWords), per_second(Counter::BatchRepeats),
                            per_second(Counter::FusedWrites), per_second(Counter::MergedUploads));
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
    return true;
}

} // namespace VideoCommon::Nx2Profile
