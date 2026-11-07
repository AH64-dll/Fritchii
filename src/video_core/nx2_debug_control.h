// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/common_types.h"
#include "common/settings.h"

/// Requests from the debug server (EDEN_NX2_DEBUG_SERVER, docs/nx2/debug-mcp.md) to the thread
/// that owns the rasterizer. The server posts them from its own thread; RasterizerVulkan takes
/// them at the end of each presented frame, so every cache they touch is used from its owner.
/// Without the server nothing is posted and the per-frame cost is one relaxed atomic load.
namespace VideoCore::Nx2Debug {

/// EXPERIMENT (not for nx2): EDEN_NX2_DMA_WAIT_WRITERS, read at startup and settable live by the
/// debug server's `switch` method, so one session can A/B it.
[[nodiscard]] inline int EnvSwitch(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? 0 : std::atoi(value); // whole number: EDEN_NX2_DMA_SNAPSHOT=10
}
/// The switch's value, read from the environment on first use (-1: not read yet). Static
/// initialization ran before the frontend applied nx2_switches.txt, so those settings were lost.
[[nodiscard]] inline int SwitchValue(std::atomic<int>& value, const char* name,
                                    int fallback = 0) {
    int current = value.load(std::memory_order_relaxed);
    if (current < 0) [[unlikely]] {
        int unset = -1;
        const int initial = std::getenv(name) != nullptr ? EnvSwitch(name) : fallback;
        value.compare_exchange_strong(unset, initial, std::memory_order_relaxed);
        current = value.load(std::memory_order_relaxed);
    }
    return current;
}
inline std::atomic<int> g_dma_wait_writers{-1};
/// EXPERIMENT, off by default: Ocarina of Time flickers white on the bottom half of Link's house
/// ladder with it (confirmed toggling it live, 2026-11-06), for +1.4 fps. EDEN_NX2_SEM_FAST=1,
/// semaphore acquires without draining the back
/// when the value is already met and no queued write overlaps it (puller.cpp).
inline std::atomic<int> g_sem_fast{-1};
/// EXPERIMENT (not for nx2): EDEN_NX2_LIST_FAST=1, command list entries skip the wait for queued
/// compute and copies while no page was newly GPU-written for 2 s; =2 also checks each skipped
/// entry on the back and logs misses (dma_pusher.cpp).
inline std::atomic<int> g_list_fast{-1};
/// EXPERIMENT (not for nx2): EDEN_NX2_LOOP_SKIP=1, the split's front does not send the instance
/// loop writes the back would ignore (draw.end in an instance loop, unchanged vertex and index
/// buffer firsts and counts) (maxwell_3d.cpp).
inline std::atomic<int> g_loop_skip{-1};

/// EDEN_NX2_DMA_SNAPSHOT (from EvilOOT's dma-snapshot fix): macro DMA reads of pages a drained
/// read flushed whole skip the drain while nothing sent since may write them (Maxwell3D::
/// FrontDmaRead). 9 = on (default: Lake Hylia 46.5 -> 52.3 fps; verify 0 differences in 235k
/// reads), 10 = on and verify every skipped read against the drained path, 0 = off.
inline std::atomic<int> g_dma_snapshot{-1};

/// EDEN_NX2_ACCURATE_PARTS (bit mask, 0 = none, read once at boot): parts of GPU accuracy High in
/// Fast mode, the split GPU thread kept, to find which one stops a flicker that Accurate mode
/// stops. Not live: switching the counter query mode mid-game crashed (queries opened in one
/// mode were closed in the other).
/// 1 = counter reports written once their host query completes (synced), 2 = precise occlusion
/// queries, 4 = CPU reads of GPU-written images download them, 8 = CPU writes flush GPU data
/// before invalidating it, 16 = fences wait for the GPU work before them.
/// Whether GPU accuracy High is on, or the EDEN_NX2_ACCURATE_PARTS bit `part` asks for it.
[[nodiscard]] inline bool AccuratePart(int part) {
    static const int parts = EnvSwitch("EDEN_NX2_ACCURATE_PARTS");
    return Settings::IsGPULevelHigh() || (parts & part) != 0;
}

/// EDEN_NX2_HOST_PACE: the host frame pacer (vk_host_pacer.h). 1 = on (default), 0 = only
/// measure how long each frame was shown.
inline std::atomic<int> g_host_pace{-1};

/// Numbered experiment switches EDEN_NX2_EXP0 .. EDEN_NX2_EXP63, live through the debug server:
/// slots for batches of fixes built together and A/B-tested in one session
/// (E:\OotRemakeEmulation\work\EXPERIMENTS.md lists which fix uses which slot). 0 = off.
/// EXP8 (texture upload hitches), bits: 1 = keep idle upload staging buffers up to 32 MiB
/// (StagingBufferPool::ReleaseLevel), 2 = no guest upload for a new image that GPU copies
/// overwrite whole (TextureCache::JoinImages), 4 = large 2D levels untiled on the transcode
/// workers too (UnswizzleImage), 8 = guest memory of uploaded textures not hinted cold
/// (TextureCache::TickFrame).
inline constexpr int ExperimentCount = 64;
inline std::atomic<int> g_experiments[ExperimentCount]{};

inline void LoadExperiments() {
    static const bool loaded = [] {
        for (int n = 0; n < ExperimentCount; ++n) {
            const std::string name = "EDEN_NX2_EXP" + std::to_string(n);
            if (const char* value = std::getenv(name.c_str())) {
                g_experiments[n].store(std::atoi(value), std::memory_order_relaxed);
            }
        }
        return true;
    }();
    static_cast<void>(loaded);
}

/// The value of experiment switch `n` (EDEN_NX2_EXP<n>), 0 unless set.
[[nodiscard]] inline int Experiment(int n) {
    LoadExperiments();
    return g_experiments[n].load(std::memory_order_relaxed);
}

/// The live switch called `name`, or null.
[[nodiscard]] inline std::atomic<int>* FindSwitch(std::string_view name) {
    if (constexpr std::string_view prefix = "EDEN_NX2_EXP";
        name.size() > prefix.size() && name.substr(0, prefix.size()) == prefix) {
        int n = 0;
        for (const char c : name.substr(prefix.size())) {
            if (c < '0' || c > '9') {
                return nullptr;
            }
            n = n * 10 + (c - '0');
        }
        if (n >= ExperimentCount) {
            return nullptr;
        }
        LoadExperiments();
        return &g_experiments[n];
    }
    if (name == "EDEN_NX2_DMA_WAIT_WRITERS") {
        return &g_dma_wait_writers;
    }
    if (name == "EDEN_NX2_SEM_FAST") {
        return &g_sem_fast;
    }
    if (name == "EDEN_NX2_LIST_FAST") {
        return &g_list_fast;
    }
    if (name == "EDEN_NX2_DMA_SNAPSHOT") {
        return &g_dma_snapshot;
    }
    if (name == "EDEN_NX2_LOOP_SKIP") {
        return &g_loop_skip;
    }
    if (name == "EDEN_NX2_HOST_PACE") {
        return &g_host_pace;
    }
    return nullptr;
}

struct CaptureRequest {
    u64 id;
    std::filesystem::path directory;
    bool sampled;
    /// The uniform and storage buffers of every draw and dispatch, and dispatch group counts.
    bool buffers;
};

struct CaptureResult {
    std::string error;
};

struct GpuReadRequest {
    u64 id;
    u64 gpu_address;
    u64 size;
};

struct GpuReadResult {
    std::vector<u8> bytes;
    /// Tegra::MemoryManager::GetID of the address space that maps the whole range.
    u64 address_space;
    std::string error;
};

struct RejectChange {
    u64 generation;
    /// Whether a hash left the list, so programs rejected before must be lowered again.
    bool unrejected;
};

/// Opens a debug server session (clears requests and rejections) or ends it (wakes waiters).
void BeginSession();
void EndSession();

/// Rasterizer thread: whether a capture or a reject change is waiting.
[[nodiscard]] bool HasRequests() noexcept;

/// Server thread: queues a frame capture; the rasterizer takes it at the next frame end.
void RequestCapture(CaptureRequest request);
/// Server thread: withdraws a capture that was not taken yet (it timed out).
void CancelCapture(u64 id);
/// Server thread: waits for the capture's manifest to be complete. Empty on timeout or when the
/// session ends.
[[nodiscard]] std::optional<CaptureResult> WaitCapture(u64 id, std::chrono::milliseconds timeout);
/// Rasterizer thread.
[[nodiscard]] std::optional<CaptureRequest> TakeCapture();
/// Frame dump writer thread, once the manifest summary is written (or the capture failed).
void FinishCapture(u64 id, CaptureResult result);

/// Server thread: queues a read of GPU virtual memory; the rasterizer takes it at the next frame
/// end, where the GPU's own reads of its address spaces run.
void RequestGpuRead(GpuReadRequest request);
/// Server thread: withdraws a read that was not taken yet (it timed out).
void CancelGpuRead(u64 id);
/// Server thread: empty on timeout or when the session ends.
[[nodiscard]] std::optional<GpuReadResult> WaitGpuRead(u64 id, std::chrono::milliseconds timeout);
/// Rasterizer thread.
[[nodiscard]] std::optional<GpuReadRequest> TakeGpuRead();
/// Rasterizer thread.
void FinishGpuRead(u64 id, GpuReadResult result);

/// Server thread: adds or removes native program keys or shader hashes (as EDEN_NX2_REJECT_HASH
/// takes them). Returns the generation to wait for.
u64 SetRejected(std::span<const u64> hashes, bool rejected);
[[nodiscard]] std::vector<u64> RejectedHashes();
/// Lowering: whether the debug server rejects this program. Cheap while the list is empty.
[[nodiscard]] bool IsRejected(u64 hash);
[[nodiscard]] bool AnyRejected() noexcept;
/// Rasterizer thread: the pending reject change, once.
[[nodiscard]] std::optional<RejectChange> TakeRejectChange();
/// Rasterizer thread: the cached programs of this generation were dropped.
void NoteRejectApplied(u64 generation);
/// Server thread: waits until the rasterizer applied the generation.
[[nodiscard]] bool WaitRejectApplied(u64 generation, std::chrono::milliseconds timeout);

} // namespace VideoCore::Nx2Debug
