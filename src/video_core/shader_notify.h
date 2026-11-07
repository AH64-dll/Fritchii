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
    int report_base{};

    bool completed{};
    int num_when_completed{};
    std::chrono::steady_clock::time_point complete_time;
};
} // namespace VideoCore
