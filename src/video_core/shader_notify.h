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
    int report_base{};

    bool completed{};
    int num_when_completed{};
    std::chrono::steady_clock::time_point complete_time;
};
} // namespace VideoCore
