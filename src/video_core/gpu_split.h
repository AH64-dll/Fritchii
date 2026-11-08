// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <thread>
#include <type_traits>

#include "common/common_types.h"

namespace Core {
class System;
}

namespace Tegra::Engines {
class EngineInterface;
}

namespace VideoCommon {

/// Why the split's front waited for its back (EDEN_NX2_GPU_PROFILE counts each).
enum class DrainSite : u32 {
    Other,
    FetchDrain,     ///< EDEN_NX2_FETCH_DRAIN: before every entry
    Acquire,        ///< A semaphore acquire reads the value the back may release
    SemaphoreFlush, ///< The acquired semaphore is in the buffer cache
    FenceRelease,   ///< The puller's acquires release the pending fences
    ListWriters,    ///< GPU-written command lists: the dispatches and copies sent before them
    ListSync,       ///< Command lists on draw-written pages: the work before the last host sync
    ListFlush,      ///< Command lists on GPU-written pages: drain and flush the pages
    MacroDmaRead,   ///< The front's macros read memory (MME DMA)
    MacroDmaFlush,  ///< Those words are in the buffer cache
    GpSyncFence,    ///< A GP entry's sync bit (sync_memory_operations)
    Map,            ///< GPU memory maps before later command reads
    FrontStart,     ///< The front takes an engine's macros over
    FrontVerify,    ///< EDEN_NX2_FRONT_VERIFY
    NarrowMacroDmaRead, ///< MacroDmaRead narrowed (WaitForWritesTo)
    NarrowAcquire,      ///< Acquire narrowed: an acquire already met
    NarrowListWriters,  ///< ListWriters narrowed (WaitForWritersTo)
    QmdRead,            ///< EDEN_NX2_QMD_PREDICT: a QMD the front reads has a queued write
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(DrainSite::Count)> drain_site_names{
    "other",       "fetch drain",     "acquire",    "semaphore flush", "fence release",
    "list writers", "list sync",      "list flush", "macro dma read",  "macro dma flush",
    "gp sync fence", "map",           "front start", "front verify",   "macro dma read (narrow)",
    "acquire (narrow)", "list writers (narrow)", "qmd read",
};

/// What made the front note an unbounded write (BarrierOnly tells the kinds apart).
enum class UnboundedCause : u8 {
    Unknown,
    RenderEnable,     ///< 3D render_enable.mode (conditional rendering)
    Wfi3D,            ///< 3D wait_for_idle
    TexInvalidate,    ///< 3D invalidate_texture_data_cache
    Inline3D,         ///< 3D inline upload without a known destination
    Maxwell3DBack,    ///< 3D methods with the macros on the back
    ComputeMacro,     ///< A compute macro call (per parameter word)
    ComputeLaunch,    ///< A compute dispatch (QMD launch, inline QMD)
    ComputeUpload,    ///< Compute inline upload (exec/data upload)
    ComputeReport,    ///< Compute report semaphore
    ComputeOther,     ///< Other compute methods MayWriteMemory counts
    BlockLinearDma,   ///< Copy engine launch with writes the front can't bound
    OtherEngine,      ///< Fermi2D blits, KeplerMemory uploads
    PullerWfi,        ///< Puller WaitForIdle
    SyncpointAcquire, ///< Puller fence action acquire (forced fence release)
    MemOpB,
    GpSync,           ///< A GP entry's sync bit
    GpuThreadCommand, ///< GPU thread commands (maps, flushes, syncs)
    Overflow,         ///< The pending write records overflowed
    Record,           ///< A recorded write overlapping the read
    Sync,             ///< The last host-level wait
    AcquireSync,      ///< The last host-level wait: an acquire met in memory
    PredictedSync,    ///< The last host-level wait: a predicted acquire
    Count,
};

/// Which unbounded position a narrowed front read waits for (GpuSplit::WaitForWritesTo).
enum class NarrowBase : u8 {
    Read,     ///< Every unbounded write (macro DMA reads)
    Acquire,  ///< Every unbounded write and dispatch (semaphore acquires)
    Recorded, ///< Only the recorded writes (and the write floor): EDEN_NX2_NARROW_DMA_RECORDED
};

/// A queued release's value at a semaphore word (GpuSplit::PredictedRelease).
struct SemaphorePrediction {
    u32 value;
    /// A max release: the word will hold at least `value`.
    bool at_least;
    /// The end of the work that releases it (a met acquire orders what follows behind it).
    u64 position;
};

/// An EDEN_NX2_* on/off switch: '0' off, anything else on, unset `default_on`.
[[nodiscard]] inline bool Nx2Switch(const char* name, bool default_on) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == 0) {
        return default_on;
    }
    return value[0] != '0';
}

// Splits the GPU thread in two (NX2_GPU_THREAD_SPLIT.md). The GPU thread (the front) takes the
// CPU's commands, parses command lists and runs the puller; engine method calls and everything
// behind them (register state, macros, draws, the rasterizer and its caches) run in order on a
// second thread (the back), fed through a single-producer single-consumer ring. Work the front
// would do on back-owned state is deferred into the same ring; the front waits for the back to
// catch up (Drain) only where it needs a result from that state, e.g. a semaphore it acquires.
// On for Switch 2 titles with the 3D and compute engines' macros on the front, against copies of
// the registers they read (Maxwell3D and KeplerCompute Front*), which sends the writes they make.
// EDEN_NX2_SPLIT=1 keeps the macros on the back; EDEN_NX2_SPLIT=0 runs everything on one thread.
class GpuSplit {
public:
    static GpuSplit& Get() {
        return instance;
    }

    /// Whether the front sends work to the back (front thread only).
    [[nodiscard]] bool Active() const {
        return active;
    }
    /// Whether the front runs the engines' macros (EDEN_NX2_SPLIT=2).
    [[nodiscard]] bool FrontMacros() const {
        return front_macros;
    }

    /// Whether the calling thread is the back.
    [[nodiscard]] static bool OnBackThread() {
        return on_back_thread;
    }
    /// Called on the back before it sleeps waiting for work (the rasterizer submits deferred work).
    void SetIdleHook(void (*hook)(void*), void* argument) {
        idle_hook = hook;
        idle_argument = argument;
    }
    /// Any thread: whether the back thread runs (false before it starts and once it stopped).
    [[nodiscard]] bool BackRunning() const {
        return back_running.load(std::memory_order_acquire);
    }
    /// Any thread: has the back call the idle hook soon, between entries or at once if it sleeps.
    void RequestIdleHook() {
        hook_requested.store(true, std::memory_order_seq_cst);
        wake.fetch_add(1, std::memory_order_seq_cst);
        wake.notify_one();
    }

    /// Starts the back thread once the split is wanted (EDEN_NX2_SPLIT=1, a Switch 2 title,
    /// commands read without cache flushes). Called by the front between commands.
    void MaybeStart(Core::System& system);
    void Stop();

    /// An engine method write, executed as DmaPusher would (method sink, ConsumeSink, call).
    /// `segment`: the engine's current_dma_segment for the call, when the engine reads it.
    void Method(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument, bool is_last,
                const GPUVAddr* segment) {
        SelectEngine(engine);
        const size_t words = segment != nullptr ? 2 : 1;
        u64* const entry = Reserve(words);
        entry[0] = static_cast<u64>(Kind::Method) | (is_last ? IsLast : 0) |
                   (segment != nullptr ? HasSegment : 0) | (u64{method & 0xffff} << 8) |
                   (u64{argument} << 32);
        if (segment != nullptr) {
            entry[1] = *segment;
        }
        Commit(words);
    }
    void MultiMethod(Tegra::Engines::EngineInterface* engine, u32 method, const u32* words,
                     u32 amount, u32 methods_pending, const GPUVAddr* segment);
    /// A write made by a macro: CallMethod without the method sink, as the engines' macros do.
    void MacroWrite(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument) {
        SelectEngine(engine);
        u64* const entry = Reserve(1);
        entry[0] = static_cast<u64>(Kind::Method) | IsLast | Direct | (u64{method & 0xffff} << 8) |
                   (u64{argument} << 32);
        Commit(1);
    }

    // Front-owned register state (EDEN_NX2_FRONT_STATE, Maxwell3D::FrontState): state register
    // writes stay on the front until something else is sent, then go as one StateDelta entry
    // (Maxwell3D::ApplyFrontDelta); runs of identical instance-loop draws go as one DrawBatch.
    /// `count` (method | value << 32) words written by `fill(u64*)`, applied in order.
    template <typename Fill>
    void StateDelta(Tegra::Engines::EngineInterface* engine, u32 count, const Fill& fill) {
        SelectEngine(engine);
        u64* const entry = Reserve(1 + size_t{count});
        entry[0] = static_cast<u64>(Kind::StateDelta) | (u64{count} << 32);
        fill(entry + 1);
        Commit(1 + size_t{count});
    }
    /// `count` (method | argument << 32) instance-loop writes, run once, then `repeats - 1` more
    /// instances counted (Maxwell3D::RunDrawBatch).
    void DrawBatch(Tegra::Engines::EngineInterface* engine, const u64* methods, u32 count,
                   u32 repeats) {
        SelectEngine(engine);
        u64* const entry = Reserve(1 + size_t{count});
        entry[0] = static_cast<u64>(Kind::DrawBatch) | (u64{count & 0xff} << 8) |
                   (u64{repeats} << 32);
        std::memcpy(entry + 1, methods, size_t{count} * sizeof(u64));
        Commit(1 + size_t{count});
    }
    // EDEN_NX2_FRONT_CB_ENCODE (Maxwell3D::FrontCbWord): constant buffer commands.
    /// `count` (1-4) words written to the constant buffer data registers, register i being
    /// const_buffer.buffer + nibble i of `nibbles` (Maxwell3D::BackCbWrite). Never more words
    /// than the single method entries it stands for.
    void CbWrite(Tegra::Engines::EngineInterface* engine, u32 nibbles, const u32* words,
                 u32 count) {
        SelectEngine(engine);
        const size_t size = 1 + size_t{count} / 2;
        u64* const entry = Reserve(size);
        entry[0] = static_cast<u64>(Kind::CbWrite) | (u64{count & 7} << 8) |
                   (u64{nibbles & 0xffff} << 12) | (u64{words[0]} << 32);
        if (count > 1) {
            entry[size - 1] = 0;
            std::memcpy(entry + 1, words + 1, size_t{count - 1} * sizeof(u32));
        }
        Commit(size);
    }
    /// A write of `argument` to bind group `stage`'s raw_config (Maxwell3D::BackCbBind).
    void CbBind(Tegra::Engines::EngineInterface* engine, u32 stage, u32 argument) {
        SelectEngine(engine);
        u64* const entry = Reserve(1);
        entry[0] = static_cast<u64>(Kind::CbBind) | (u64{stage & 0xff} << 8) |
                   (u64{argument} << 32);
        Commit(1);
    }
    // EDEN_NX2_COMPUTE_FUSE (KeplerCompute::FrontQueueWrite): a run of compute method writes in
    // one entry instead of one entry each.
    /// Bits of a run word: method (bits 0-15) | WriteIsLast | WriteDirect | argument << 32.
    static constexpr u64 WriteIsLast = u64{1} << 16; ///< The call's is_last_call
    static constexpr u64 WriteDirect = u64{1} << 17; ///< Sent as MacroWrite (no method sink)
    /// `count` writes run in order on the back as the Method and MacroWrite entries they stand
    /// for (KeplerCompute::BackWrites). `segment`: current_dma_segment of the last of them sent
    /// with one, stored before the run.
    void Writes(Tegra::Engines::EngineInterface* engine, const u64* writes, u32 count,
                const GPUVAddr* segment) {
        SelectEngine(engine);
        const size_t head = segment != nullptr ? 2 : 1;
        u64* const entry = Reserve(head + size_t{count});
        entry[0] = static_cast<u64>(Kind::Writes) | (segment != nullptr ? HasSegment : 0) |
                   (u64{count} << 32);
        if (segment != nullptr) {
            entry[1] = *segment;
        }
        std::memcpy(entry + head, writes, size_t{count} * sizeof(u64));
        Commit(head + size_t{count});
    }
    /// Work the front holds back (a register delta, a draw batch): `hook(argument)` sends it
    /// before anything else is sent and before a Drain. Another holder's work is sent first.
    using PendingFlush = void (*)(void*);
    void SetPendingFlush(PendingFlush hook, void* argument) {
        if (pending_flush != hook || pending_flush_argument != argument) {
            RunPendingFlush();
            pending_flush = hook;
            pending_flush_argument = argument;
        }
    }
    /// Sends the held-back work now.
    void RunPendingFlush() {
        if (pending_flush != nullptr) [[unlikely]] {
            const PendingFlush hook = pending_flush;
            pending_flush = nullptr;
            hook(pending_flush_argument);
        }
    }
    /// The holder sends its work itself (it calls SetPendingFlush again for what it keeps).
    void ClearPendingFlush() {
        pending_flush = nullptr;
    }
    /// `argument` goes away: forget its held-back work.
    void DropPendingFlush(void* argument) {
        if (pending_flush_argument == argument) {
            pending_flush = nullptr;
            pending_flush_argument = nullptr;
        }
    }
    /// EDEN_NX2_NARROW_DRAIN=2.
    [[nodiscard]] bool NarrowStrict() const {
        return narrow_strict;
    }

    /// Runs `func` on the back after everything sent before it. Must be trivially copyable.
    /// Work that writes guest memory notes it after (NoteWrite, NoteUnboundedWrite).
    template <typename Func>
    void Defer(const Func& func) {
        static_assert(std::is_trivially_copyable_v<Func> && alignof(Func) <= alignof(u64));
        constexpr size_t payload = (sizeof(Func) + sizeof(u64) - 1) / sizeof(u64);
        u64* const entry = Reserve(2 + payload);
        entry[0] = static_cast<u64>(Kind::Call) | (u64{payload} << 8);
        entry[1] = reinterpret_cast<u64>(&CallThunk<Func>);
        std::memcpy(entry + 2, &func, sizeof(Func));
        Commit(2 + payload);
    }
    void DeferFunction(std::function<void()>&& func);
    /// Runs `func` on the back and waits for it: for work on back-owned state, such as a buffer
    /// cache download that records into the Vulkan scheduler. A drained back still submits from
    /// its idle hook, so the front never touches that state itself.
    template <typename Func>
    void Run(const Func& func, DrainSite site = DrainSite::Other) {
        if (on_back_thread || !active) {
            func();
            return;
        }
        Defer(func);
        Drain(site);
    }

    /// EDEN_NX2_SPLIT_CHECK=1 (debug): reports, with its stack, a thread other than the back
    /// using back-owned state (the Vulkan scheduler) while the back runs entries.
    static inline bool check_owner{};
    static void CheckOwner(const char* where);

    /// Makes what was sent visible to the back.
    void Publish();
    /// EDEN_NX2_PUBLISH_WAITS: before the front yields or sleeps, sends the held-back work and
    /// publishes, so the back doesn't idle on work the front already decoded.
    void PublishBeforeWait() {
        if (publish_waits) {
            RunPendingFlush();
            Publish();
        }
    }
    /// Waits until the back executed everything sent so far.
    void Drain(DrainSite site = DrainSite::Other);

    /// Compute or copy work was just sent: what can write GPU-written command lists.
    void NoteWriter() {
        last_writer_position = write_position;
    }
    /// Waits until the back executed the compute and copy work sent so far (not what came after).
    /// Returns whether any of it was still queued.
    bool WaitForWriters();
    /// A host-level wait was just sent (a syncpoint acquire, a host WFI, a GP entry with its sync
    /// bit; semaphore acquires drain): the guest's GPU work before it is complete before the
    /// commands after it are fetched.
    void NoteSync(UnboundedCause = UnboundedCause::Sync) {
        last_sync_position = write_position;
    }
    /// EDEN_NX2_ACQUIRE_SYNC_AT: a met acquire only orders the fetch after the work before the
    /// release that met it (sent before `position`), not after everything sent before the
    /// acquire; without it NoteSync. The sync never moves back.
    void NoteSyncAt(u64 position, UnboundedCause cause);
    /// NoteSyncAt for an acquire met in guest memory: the release that wrote it ran on the back.
    void NoteSyncDone(UnboundedCause cause);
    /// Waits until the back executed the work sent before the last NoteSync. Returns whether any
    /// of it was still queued.
    bool WaitForSync();

    // Narrow drains (EDEN_NX2_NARROW_DRAIN, on unless 0; only with the macros on the front, whose
    // register copies give the destinations): a front read of guest memory waits only for the
    // queued work that may write what it reads, not for everything queued. The front records
    // each queued write whose destination it knows (NoteWrite: 3D constant buffer data, inline
    // uploads and reports, semaphore releases, copy engine launches) and the position of the
    // latest work that may write memory it can't bound (NoteUnboundedWrite: dispatches, blits,
    // copies into images, forced fence releases, host waits, GPU thread commands). Draws only
    // write storage buffers (and transform feedback), whose ranges the front can't know either;
    // the guest orders them before a read from the command stream with a host wait (WFI, GP
    // entry sync, syncpoint acquire), which counts as an unbounded write. EDEN_NX2_NARROW_DRAIN=2
    // counts every 3D method as one too (as exact as the full drain, without the draw rule).
    // Readers still drain fully for memory GPU work wrote through the buffer cache (newer words
    // may be in its caches): MemoryManager::Nx2ReadNeedsDrain.
    [[nodiscard]] bool NarrowDrains() const {
        return narrow;
    }
    /// EDEN_NX2_NARROW_VERIFY=1 (debug): narrowed reads are compared with a full drain.
    [[nodiscard]] static bool NarrowVerify();
    /// EDEN_NX2_NARROW_VERIFY: logs, now and then, narrowed reads that differed from a full drain
    /// (`mismatches`: words that differed; `macro`: the macro that read, 0 if none).
    static void ReportNarrowCheck(DrainSite site, u32 mismatches, u32 macro, GPUVAddr address);
    /// Counts starts of the split: engines keeping front copies of their registers rebuild them.
    [[nodiscard]] u64 Session() const {
        return session;
    }
    /// The work just sent writes [address, address + size) of the GPU address space `space` (its
    /// MemoryManager) when the back runs it. `fine`: a release or report (tracked by the word, so
    /// reads of the words next to it don't wait for it); other writes are tracked by the page.
    void NoteWrite(const void* space, GPUVAddr address, u64 size, bool fine = false) {
        if (narrow && size != 0) {
            RecordWrite(space, address, size, fine);
        }
    }
    /// The work just sent may write guest memory the front can't bound. `writer`: it is compute
    /// or copy work (what WaitForWriters waits for).
    void NoteUnboundedWrite(bool writer = false, UnboundedCause cause = UnboundedCause::Unknown) {
        if (narrow) {
            unbounded_position = write_position;
            if (!BarrierOnly(cause)) {
                guest_write_position = write_position;
            }
            // EDEN_NX2_ACQUIRE_SKIP_BARRIERS (with EDEN_NX2_ACQUIRE_POLL): barriers (WFIs, texture
            // invalidates, MemOpB, GP sync) release no semaphore. They write guest memory only on
            // GPU-written pages (InvalidateGPUCache downloads GPU bytes next to CPU writes), where
            // the narrowed read drains instead. Behind a poll that wasn't met, a lower base only
            // lets a word that is met already proceed sooner; one not met still drains.
            if (!acquire_skip_barriers || !BarrierOnly(cause) ||
                cause == UnboundedCause::ComputeLaunch) {
                acquire_position = write_position;
            }
            if (writer) {
                unbounded_writer_position = write_position;
            }
        }
    }
    /// A compute dispatch was just sent. EDEN_NX2_NARROW_COMPUTE: the front end runs MME DMA
    /// reads in method order without waiting for earlier grids (barriers come as WFIs, which
    /// count as unbounded), so dispatches only hold semaphore acquires and command list reads.
    void NoteDispatch(UnboundedCause cause = UnboundedCause::ComputeLaunch) {
        if (!narrow) {
            return;
        }
        if (!narrow_compute) {
            NoteUnboundedWrite(true, cause);
            return;
        }
        acquire_position = write_position;
        unbounded_writer_position = write_position;
    }
    /// Unbounded writes that write no guest memory themselves on the back: barriers, and
    /// dispatches, whose shader stores go to the host GPU's copies (GuestReadTarget).
    [[nodiscard]] static constexpr bool BarrierOnly(UnboundedCause cause) {
        switch (cause) {
        case UnboundedCause::Wfi3D:
        case UnboundedCause::TexInvalidate:
        case UnboundedCause::PullerWfi:
        case UnboundedCause::MemOpB:
        case UnboundedCause::GpSync:
        case UnboundedCause::ComputeLaunch:
            return true;
        default:
            return false;
        }
    }
    // EDEN_NX2_QMD_PREDICT: the front runs a copy of the QMD scheduler (KeplerCompute), so it
    // knows where each QMD runs and what it releases. Those releases are noted as writes and
    // releases (NoteWrite, NoteRelease), so acquires on them are predicted, and a launch is no
    // dispatch for acquires (NoteDispatch) unless a grid ran.
    [[nodiscard]] bool QmdPredict() const {
        return qmd_predict;
    }
    /// EDEN_NX2_QMD_PREDICT: a QMD with a grid ran in the work just sent. Its releases were
    /// noted; its shader stores hold command list reads and, without EDEN_NX2_ACQUIRE_POLL
    /// (whose polled reads already don't wait for them), semaphore acquires.
    void NoteComputeGrid() {
        if (!narrow) {
            return;
        }
        unbounded_writer_position = write_position;
        if (!acquire_poll) {
            acquire_position = write_position;
        }
    }
    /// Work sent so far may have written semaphore words with values the front doesn't know
    /// (the QMD mirror started over): no release sent before now is predicted any more.
    void NoteReleaseBarrier() {
        release_barrier = write_position;
    }
    /// The position the front waits for before reading [address, address + size) of `space`
    /// so that guest memory then holds what the back reads there when it gets to the work sent
    /// so far: the recorded writes over it and the unbounded writes that write guest memory
    /// (not BarrierOnly). Also refreshes ConsumedPosition.
    [[nodiscard]] u64 GuestReadTarget(const void* space, GPUVAddr address, u64 size);
    /// What the back consumed, as last loaded.
    [[nodiscard]] u64 ConsumedPosition() const {
        return consumed_cache;
    }
    /// What the back consumed, loaded now (refreshes ConsumedPosition).
    u64 LoadConsumed() {
        return consumed_cache = consumed.load(std::memory_order_acquire);
    }
    /// The end of the work sent so far.
    [[nodiscard]] u64 WritePosition() const {
        return write_position;
    }
    /// Waits for GuestReadTarget's `target` (a QMD read).
    bool WaitForGuestRead(u64 target);
    /// EDEN_NX2_NARROW_COMPUTE: compute work is unbounded only where it launches or writes.
    [[nodiscard]] bool NarrowCompute() const {
        return narrow_compute;
    }
    /// EDEN_NX2_NARROW_RENDER_ENABLE: render_enable.mode is no unbounded write.
    [[nodiscard]] bool NarrowRenderEnable() const {
        return narrow_render_enable;
    }
    /// EDEN_NX2_NARROW_DMA_RECORDED: macro DMA reads wait only for the recorded writes.
    [[nodiscard]] bool NarrowDmaRecorded() const {
        return narrow_dma_recorded;
    }
    /// EDEN_NX2_NARROW_LIST_SYNC: GPU-written command lists wait for the work before the last
    /// host-level wait and the recorded writes over them, not for every dispatch and copy.
    [[nodiscard]] bool NarrowListSync() const {
        return narrow_list_sync;
    }
    /// EDEN_NX2_ACQUIRE_POLL: semaphore acquires read guest memory once the recorded writes over
    /// the word ran, without waiting for dispatches or other unbounded work; an acquire not met
    /// suspends the channel and is polled from guest memory (no drain).
    [[nodiscard]] bool AcquirePoll() const {
        return acquire_poll;
    }
    /// EDEN_NX2_LIST_EXACT: command lists on pages the GPU never wrote (and that held lists
    /// before) wait only for the recorded writes over their pages.
    [[nodiscard]] bool ListExact() const {
        return list_exact;
    }
    /// Any thread: GPU work releases or reports to the page of `address` (sticky until Stop):
    /// semaphore releases, reports and QMD releases.
    static void NoteGpuSemaphore(GPUVAddr address);
    /// Whether a semaphore acquire met at `address` is a host-level wait (NoteSync): the GPU
    /// releases semaphores on its page (EDEN_NX2_SYNC_GPU_SEMAPHORES; always without it). An
    /// acquire on a word only the CPU writes says nothing about the GPU work before it.
    [[nodiscard]] bool AcquireSyncs(GPUVAddr address) const;
    /// Whether GPU work may have released or reported to a page of [address, address + size)
    /// (NoteGpuSemaphore), so the back's query cache may hold results for it not yet written.
    [[nodiscard]] static bool MayHoldReports(GPUVAddr address, u64 size);

    // Semaphore prediction (EDEN_NX2_SEM_PREDICT): the values of the releases the front queued.
    // While a release is still queued, an acquire it meets is met: the back runs the work after
    // the acquire behind the release, as the hardware would once the release lands.
    [[nodiscard]] bool SemaphorePredict() const {
        return sem_predict;
    }
    /// A release of `value` at `address` of `space` (a Tegra::MemoryManager) was just sent.
    /// `at_least`: a max release (the word becomes max(word, value)).
    void NoteRelease(const void* space, GPUVAddr address, u32 value, bool at_least = false);
    /// The latest queued (or, EDEN_NX2_SEM_PREDICT_LANDED, run but not landed) release at
    /// `address` of `space`, as PredictedSemaphore, with max releases as lower bounds.
    [[nodiscard]] std::optional<SemaphorePrediction> PredictedRelease(const void* space,
                                                                      GPUVAddr address);
    /// The value of the latest still-queued release at `address` of `space`, if any.
    /// EDEN_NX2_SEM_PREDICT_LANDED: also once the back ran it while guest memory still holds the
    /// word from before it (a fence still holds the write back) or its value, and nothing the
    /// front sent since wrote the word.
    [[nodiscard]] std::optional<u32> PredictedSemaphore(const void* space, GPUVAddr address);
    /// Work that writes `address` of `space` with a value the front doesn't know was just sent.
    void ForgetRelease(const void* space, GPUVAddr address);
    /// The work just sent may draw (only EDEN_NX2_NARROW_DRAIN=2 counts it as unbounded).
    void NoteDraw() {
        if (narrow_strict) {
            unbounded_position = write_position;
            acquire_position = write_position;
        }
    }
    /// Waits until the back executed the queued work that may write [address, address + size) of
    /// `space`: the unbounded writes and the overlapping ones NoteWrite recorded. Returns whether
    /// any of it was still queued.
    bool WaitForWritesTo(const void* space, GPUVAddr address, u64 size, DrainSite site,
                         NarrowBase base = NarrowBase::Read);
    /// WaitForWriters for a command list at [address, address + size): the compute and copy work
    /// with unbounded writes, and the queued writes overlapping the list. `sync_only`
    /// (EDEN_NX2_NARROW_LIST_SYNC): the work before the last host-level wait instead of every
    /// dispatch and copy.
    bool WaitForWritersTo(const void* space, GPUVAddr address, u64 size, bool sync_only = false);

private:
    friend struct GpuSplitTestAccess;

    enum class Kind : u64 {
        Wrap,   ///< The rest of the ring is unused: continue at its start
        Engine, ///< The engine the next methods go to: word 1
        Method, ///< Bits 8-23 method, 32-63 argument; flags IsLast, HasSegment (word 1)
        Multi,  ///< Bits 8-23 method, 32-63 amount; word 1 methods pending, [segment], words
        Call,   ///< Bits 8-31 payload size in words; word 1 thunk, then the payload
        StateDelta, ///< Bits 32-63 count; then count words (method | value << 32), Maxwell3D
        DrawBatch,  ///< Bits 8-15 count, 32-63 repeats; then count words (method | arg << 32)
        CbWrite,    ///< Bits 8-10 count (1-4), 12-27 register nibbles, 32-63 word 0; then words
        CbBind,     ///< Bits 8-15 stage, 32-63 argument (Maxwell3D bind group raw_config)
        Writes,     ///< Bits 32-63 count; flag HasSegment (word 1); then count run words (KeplerCompute)
    };
    static constexpr u64 IsLast = 1 << 4;
    static constexpr u64 HasSegment = 1 << 5;
    static constexpr u64 Direct = 1 << 6; ///< Method: call it even if not executable (macros)

    static constexpr size_t RingWords = size_t{1} << 20; // 8 MiB
    static constexpr size_t RingMask = RingWords - 1;
    static constexpr u32 MaxMultiWords = 8192;
    static constexpr size_t PublishEvery = 256;

    template <typename Func>
    static void CallThunk(const u64* payload) {
        (*reinterpret_cast<const Func*>(payload))();
    }

    /// Room for an entry of `words` at the write position (never across the ring's end).
    u64* Reserve(size_t words) {
        // Held-back front state goes before any other entry (SetPendingFlush).
        RunPendingFlush();
        const size_t index = write_position & RingMask;
        if (index + words <= RingWords && write_position + words - consumed_cache <= RingWords)
            [[likely]] {
            return &ring[index];
        }
        return ReserveSlow(words);
    }
    u64* ReserveSlow(size_t words);
    void Commit(size_t words) {
        write_position += words;
        // EDEN_NX2_PUBLISH_IDLE: a back that ran out of work gets each entry at once.
        if (write_position - published_local >= publish_every ||
            (publish_idle && back_idle.load(std::memory_order_relaxed))) {
            Publish();
        }
    }
    void WaitForSpace(size_t words);
    void SelectEngine(Tegra::Engines::EngineInterface* engine) {
        if (engine != front_engine) [[unlikely]] {
            SwitchEngine(engine);
        }
    }
    void SwitchEngine(Tegra::Engines::EngineInterface* engine);
    bool WaitFor(u64 target, DrainSite site);

    /// One step of a front wait for the back to consume `target`.
    void Backoff(u32 spins, u64 target);
    /// After the back stored `consumed`: wakes a front sleeping in Backoff.
    void NotifyFront() {
        if (front_waiting.load(std::memory_order_seq_cst)) {
            front_wake.fetch_add(1, std::memory_order_seq_cst);
            front_wake.notify_one();
        }
    }

    void BackLoop(std::stop_token stop_token);
    /// Runs the entry at `entry` (of the `words` published before the ring's end), returns its size.
    size_t Execute(const u64* entry, size_t words);

    // Narrow drains: the queued writes with a known destination, indexed by where they write.
    // Each slot holds the position of the end of the latest work writing its page (4 KiB) or word
    // (4 bytes, releases and reports); it is done once `consumed` reaches it. Slots are found by
    // a hash in buckets of four; a slot still pending that has to make room raises write_floor,
    // which every read waits for. Writes over more than 64 pages go to a short list.
    struct PageSlot {
        u64 key;
        u64 coarse; ///< Page-granular writes
        u64 any;    ///< Any write (word writes too)
    };
    struct WordSlot {
        u64 key;
        u64 position;
    };
    struct HugeWrite {
        const void* space;
        GPUVAddr begin;
        GPUVAddr end;
        u64 position;
    };
    static constexpr size_t PageBuckets = 2048;  ///< 4 slots each
    static constexpr size_t WordBuckets = 2048;  ///< 4 slots each
    static constexpr size_t MaxHugeWrites = 16;
    static constexpr size_t MaxFineBytes = 64;   ///< Larger releases go by the page
    static constexpr size_t MaxSpaces = 4;       ///< Address spaces told apart (the last: the rest)
    static constexpr GPUVAddr PageBits = 12;
    static constexpr GPUVAddr WordBits = 2;
    void RecordWrite(const void* space, GPUVAddr address, u64 size, bool fine);
    [[nodiscard]] u32 SpaceId(const void* space);
    void InsertPage(u64 key, bool coarse);
    void InsertWord(u64 key);
    /// A pending slot gives way: reads wait for it from now on.
    void RaiseFloor(u64 position) {
        if (position <= consumed_cache || position <= write_floor) {
            return;
        }
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (position > consumed_cache) {
            write_floor = position;
        }
    }
    [[nodiscard]] const PageSlot* FindPage(u64 key) const;
    [[nodiscard]] const WordSlot* FindWord(u64 key) const;
    /// The position to wait for before reading [address, address + size) of `space`: at least
    /// `base`, and past the latest recorded write that may overlap it.
    [[nodiscard]] u64 PendingWriteTarget(const void* space, GPUVAddr address, u64 size,
                                         u64 base) const;
    /// Whether the front sent work after `position` that may write the word at `address`
    /// (true where the index no longer tells).
    [[nodiscard]] bool WrittenSince(const void* space, GPUVAddr address, u64 position) const;

    Core::System* system{};
    bool active{};
    bool front_macros{};
    bool narrow{};
    bool narrow_strict{};
    bool narrow_compute{};
    bool narrow_render_enable{};
    bool narrow_dma_recorded{};
    bool narrow_list_sync{};
    bool sem_predict{};
    bool sem_predict_landed{};
    bool acquire_poll{};
    bool list_exact{};
    bool sync_gpu_semaphores{};
    bool publish_idle{};
    bool qmd_predict{};
    bool acquire_sync_at{};
    bool publish_waits{};
    bool acquire_skip_barriers{};
    /// NoteGpuSemaphore's pages (page + 1, open addressing); saturated: every page counts.
    static constexpr size_t GpuSemaphoreSlots = 1024;
    static inline std::array<std::atomic<u64>, GpuSemaphoreSlots> gpu_semaphore_pages{};
    static inline std::atomic<u32> gpu_semaphore_count{};
    static inline std::atomic<bool> gpu_semaphore_saturated{};
    u64 session{};
    std::unique_ptr<u64[]> ring;

    // Front side.
    u64 write_position{};
    u64 last_writer_position{};
    u64 last_sync_position{};
    u64 published_local{};
    u64 consumed_cache{};
    /// Words between publishes: small right after a drain, while the back has nothing to do.
    size_t publish_every{PublishEvery};
    Tegra::Engines::EngineInterface* front_engine{};
    /// Held-back front work (SetPendingFlush).
    PendingFlush pending_flush{};
    void* pending_flush_argument{};
    /// Narrow drains: the end of the latest work with unbounded writes (of compute or copy work).
    u64 unbounded_position{};
    u64 unbounded_writer_position{};
    /// Unbounded writes and dispatches (semaphore acquires wait for those).
    u64 acquire_position{};
    /// The latest pending write slot that gave way (every narrowed read waits for it).
    u64 write_floor{};
    /// Unbounded writes that may write guest memory themselves (not BarrierOnly).
    u64 guest_write_position{};
    /// Releases sent before it are not predicted (NoteReleaseBarrier).
    u64 release_barrier{};
    /// The latest release the front queued per semaphore word (EDEN_NX2_SEM_PREDICT), found by a
    /// hash in buckets of four; `before`: the word in guest memory when it was sent.
    struct QueuedRelease {
        const void* space;
        GPUVAddr address;
        u64 position;
        u32 value;
        u32 before;
        /// Tegra::g_nx2_semaphore_landed slot of the word, its count when the entry started, and
        /// the releases sent since: the latest landed once the count grew by `sent`.
        u32 landed_slot;
        u32 landed_base;
        u32 sent;
        bool used;
        bool at_least; ///< A max release
    };
    static constexpr size_t ReleaseBuckets = 256; ///< 4 slots each
    [[nodiscard]] QueuedRelease* FindRelease(const void* space, GPUVAddr address);
    std::unique_ptr<QueuedRelease[]> queued_releases;
    /// The pending write index (RecordWrite).
    std::unique_ptr<PageSlot[]> page_slots;
    std::unique_ptr<WordSlot[]> word_slots;
    std::array<HugeWrite, MaxHugeWrites> huge_writes{};
    size_t next_huge{};
    std::array<const void*, MaxSpaces> spaces{};
    /// Per address space id: the latest recorded write (reads in another space wait for it).
    std::array<u64, MaxSpaces> space_last{};
    u32 used_spaces{};
    /// The latest recorded write anywhere.
    u64 last_record{};
    /// The page slot the last page write went to (runs of writes to one page).
    PageSlot* last_page_slot{};

    // Back side.
    Tegra::Engines::EngineInterface* back_engine{};

    static GpuSplit instance;
    static inline thread_local bool on_back_thread{};
    void (*idle_hook)(void*){};
    void* idle_argument{};
    alignas(64) std::atomic<bool> hook_requested{};
    std::atomic<bool> back_running{};

    // 128 bytes: x86 spatial prefetchers pull 64-byte lines in pairs. back_waiting and wake are
    // always touched together by both threads, so they share a block.
    alignas(128) std::atomic<u64> published{};
    alignas(128) std::atomic<u64> consumed{};
    /// A position the front waits for (WaitFor): the back stores `consumed` once past it.
    alignas(128) std::atomic<u64> wait_target{~u64{0}};
    /// The back found the ring empty (it spins, then sleeps): Commit publishes at once.
    alignas(128) std::atomic<bool> back_idle{};
    alignas(128) std::atomic<bool> back_waiting{};
    std::atomic<u32> wake{};
    /// The front sleeps in Backoff (#161: instead of yielding).
    alignas(128) std::atomic<bool> front_waiting{};
    std::atomic<u32> front_wake{};
    std::jthread back_thread;
};

} // namespace VideoCommon
