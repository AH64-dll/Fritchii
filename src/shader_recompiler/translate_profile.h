// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Time spent in each step of shader translation, summed over every thread, to find where a
// pipeline's translation goes. Off unless EDEN_NX2_TRANSLATE_PROFILE=1 is set or SetEnabled is
// called; when off a step costs one load of a flag.

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>

#include "common/common_types.h"

namespace Shader::TranslateProfile {

enum class Step : u32 {
    Cfg,
    BuildAsl,
    Blocks,
    RemoveUnreachable,
    LowerTypes,
    SsaRewrite,
    ConstantPropagation,
    Position,
    BlitConstants,
    GlobalMemoryToStorageBuffer,
    GeometryCompaction,
    Texture,
    Rescaling,
    DeadCodeElimination,
    CollectShaderInfo,
    Layer,
    VendorWorkaround,
    Interpolation,
    NvnStorageBuffers,
    LimitSampledTextureArrays,
    MergeDualVertex,
    ConvertLegacyToGeneric,
    EmitSpirv,
    SsaVisit,
    SsaSeal,
    SsaFinal,
    Count,
};

constexpr size_t NumSteps = static_cast<size_t>(Step::Count);

[[nodiscard]] const char* Name(Step step) noexcept;

[[nodiscard]] bool Enabled() noexcept;
void SetEnabled(bool enabled) noexcept;

void Add(Step step, std::chrono::nanoseconds time) noexcept;

/// Totals and counts of each step since the last call, which clears them.
struct Totals {
    std::array<u64, NumSteps> nanoseconds{};
    std::array<u64, NumSteps> counts{};
};
[[nodiscard]] Totals Take() noexcept;

/// Times the enclosing scope as `step` when profiling is on.
class Scope {
public:
    explicit Scope(Step step_) noexcept : step{step_}, enabled{Enabled()} {
        if (enabled) {
            begin = std::chrono::steady_clock::now();
        }
    }
    ~Scope() {
        if (enabled) {
            Add(step, std::chrono::steady_clock::now() - begin);
        }
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    Step step;
    bool enabled;
    std::chrono::steady_clock::time_point begin{};
};

} // namespace Shader::TranslateProfile
