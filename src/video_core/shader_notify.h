// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>

#include "common/common_types.h"

namespace VideoCore {
class ShaderNotify {
public:
    /// Shader warm-up progress shown in the status bar. The three values are read separately, so
    /// a reading taken during an update can mix two updates; fine for a display.
    struct WarmProgress {
        u32 kind{};     ///< 0 none, 1 compute, 2 graphics
        size_t done{};
        size_t total{}; ///< 0 when unknown; for graphics it can be an estimate
    };

    [[nodiscard]] int ShadersBuilding() noexcept;

    /// Called from the warm-up threads. kind 0 hides the indicator.
    void SetWarmProgress(u32 kind, size_t done, size_t total) noexcept {
        if (kind == 3) {
            precompile_done.store(done, std::memory_order::relaxed);
            precompile_total.store(total, std::memory_order::relaxed);
        }
        warm_done.store(done, std::memory_order::relaxed);
        warm_total.store(total, std::memory_order::relaxed);
        warm_kind.store(kind, std::memory_order::release);
    }

    [[nodiscard]] WarmProgress GetWarmProgress() const noexcept {
        WarmProgress progress;
        progress.kind = warm_kind.load(std::memory_order::acquire);
        progress.done = warm_done.load(std::memory_order::relaxed);
        progress.total = warm_total.load(std::memory_order::relaxed);
        return progress;
    }

    /// A warm-up (compute or graphics) started or ended; WarmActive while any has work left.
    void BeginWarm() noexcept {
        warm_running.fetch_add(1, std::memory_order::relaxed);
    }
    void EndWarm() noexcept {
        warm_running.fetch_sub(1, std::memory_order::relaxed);
    }
    [[nodiscard]] bool WarmActive() const noexcept {
        return warm_running.load(std::memory_order::relaxed) > 0;
    }

    /// The warm-up never competes with the game for the CPU and the driver: it builds only while
    /// emulation is paused, while the player runs it from the menu (Prepare Shaders Now), or
    /// always with EDEN_NX2_WARM_WHILE_PLAYING=1.
    void SetEmulationPaused(bool paused) noexcept {
        emulation_paused.store(paused, std::memory_order::relaxed);
    }
    void SetWarmRequested(bool requested) noexcept {
        warm_requested.store(requested, std::memory_order::relaxed);
    }
    [[nodiscard]] bool WarmMayBuild() const noexcept;

    /// Precompiled pipelines (EDEN_NX2_PRECOMPILE, progress kind 3). A preparation counts as a
    /// warm-up while it runs. Prepare Shaders Now asks for another one with RequestPrecompile,
    /// which counts it at once so the menu sees work left; the preparation thread takes the
    /// request and ends it with PrecompileEnded.
    void SetPrecompileAvailable(bool available) noexcept {
        int expected = available ? PrecompileAbsent : PrecompileIdle;
        precompile_state.compare_exchange_strong(expected,
                                                 available ? PrecompileIdle : PrecompileAbsent);
    }
    /// The preparation thread starts a run (on its own or for a request).
    void PrecompileStarted() noexcept {
        if (precompile_state.exchange(PrecompileRunning) != PrecompileAsked) {
            BeginWarm();
        }
    }
    void PrecompileEnded() noexcept {
        precompile_state.store(PrecompileIdle);
        EndWarm();
    }
    /// False when this title has no precompiled pipelines to prepare.
    bool RequestPrecompile() noexcept {
        int expected = PrecompileIdle;
        if (precompile_state.compare_exchange_strong(expected, PrecompileAsked)) {
            BeginWarm();
            return true;
        }
        return expected == PrecompileRunning || expected == PrecompileAsked;
    }
    [[nodiscard]] bool PrecompileRequested() const noexcept {
        return precompile_state.load() == PrecompileAsked;
    }

    /// From boot until the title's precompiled pipelines are known to be prepared or absent: the
    /// file is looked for or assembled, the stamp checked, or the preparation runs. A frontend can
    /// hold the game back meanwhile (Fritchii's "Preparing shaders" screen).
    void SetPrecompilePending(bool pending) noexcept {
        precompile_pending.store(pending, std::memory_order::release);
    }
    [[nodiscard]] bool PrecompilePending() const noexcept {
        return precompile_pending.load(std::memory_order::acquire);
    }
    /// The last precompile progress (kind 3), kept apart from the warm-ups that share the
    /// indicator. total is 0 until a preparation counts its records.
    [[nodiscard]] WarmProgress GetPrecompileProgress() const noexcept {
        WarmProgress progress;
        progress.kind = 3;
        progress.done = precompile_done.load(std::memory_order::relaxed);
        progress.total = precompile_total.load(std::memory_order::relaxed);
        return progress;
    }
    /// Full speed: the preparation may use every core at normal priority (the game waits for it);
    /// otherwise it stays on its few idle-priority threads beside the game.
    void SetPrecompileFullSpeed(bool full) noexcept {
        precompile_full_speed.store(full, std::memory_order::relaxed);
    }
    [[nodiscard]] bool PrecompileFullSpeed() const noexcept {
        return precompile_full_speed.load(std::memory_order::relaxed);
    }

    void MarkShaderComplete() noexcept {
        ++num_complete;
    }

    void MarkShaderBuilding() noexcept {
        ++num_building;
    }

private:
    std::atomic_int num_building{};
    std::atomic_int num_complete{};
    std::atomic<u32> warm_kind{};
    std::atomic<size_t> warm_done{};
    std::atomic<size_t> warm_total{};
    std::atomic<int> warm_running{};
    std::atomic<bool> emulation_paused{};
    std::atomic<bool> warm_requested{};
    static constexpr int PrecompileAbsent = 0;  ///< nothing to prepare for this title
    static constexpr int PrecompileIdle = 1;    ///< prepared, or waiting for its turn
    static constexpr int PrecompileRunning = 2; ///< a preparation runs (counted in warm_running)
    static constexpr int PrecompileAsked = 3;   ///< requested (counted in warm_running)
    std::atomic<int> precompile_state{PrecompileAbsent};
    std::atomic<bool> precompile_pending{};
    std::atomic<bool> precompile_full_speed{};
    std::atomic<size_t> precompile_done{};
    std::atomic<size_t> precompile_total{};
    int report_base{};

    bool completed{};
    int num_when_completed{};
    std::chrono::steady_clock::time_point complete_time;
};
} // namespace VideoCore
