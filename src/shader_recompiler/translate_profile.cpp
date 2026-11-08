// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "shader_recompiler/translate_profile.h"

#include <cstdlib>

namespace Shader::TranslateProfile {
namespace {

// -1 until the first translation reads EDEN_NX2_TRANSLATE_PROFILE (Fritchii: static init runs
// before nx2_switches.txt is applied, so the switch is read on first use).
std::atomic<int> enabled_flag{-1};

std::array<std::atomic<u64>, NumSteps> step_nanoseconds{};
std::array<std::atomic<u64>, NumSteps> step_counts{};

constexpr std::array<const char*, NumSteps> Names{
    "CFG",
    "BuildASL (decode and translate)",
    "GenerateBlocks + PostOrder",
    "RemoveUnreachableBlocks",
    "Lower fp64/fp16/int64, barriers",
    "SsaRewritePass",
    "ConstantPropagationPass",
    "PositionPass",
    "BlitConstantsPass",
    "GlobalMemoryToStorageBufferPass",
    "GeometryCompactionPass",
    "TexturePass",
    "RescalingPass",
    "DeadCodeEliminationPass",
    "CollectShaderInfoPass",
    "LayerPass",
    "VendorWorkaroundPass",
    "CollectInterpolationInfo",
    "AddNVNStorageBuffers",
    "LimitSampledTextureArrays",
    "MergeDualVertexPrograms",
    "ConvertLegacyToGeneric",
    "EmitSPIRV",
    "  SSA: visit (reads, writes)",
    "  SSA: seal (phi operands)",
    "  SSA: phi types and order",
};

} // Anonymous namespace

const char* Name(Step step) noexcept {
    return Names[static_cast<size_t>(step)];
}

bool Enabled() noexcept {
    int value = enabled_flag.load(std::memory_order_relaxed);
    if (value < 0) {
        const char* const text = std::getenv("EDEN_NX2_TRANSLATE_PROFILE");
        int expected = -1;
        enabled_flag.compare_exchange_strong(expected, text != nullptr && text[0] == '1' ? 1 : 0,
                                             std::memory_order_relaxed);
        value = enabled_flag.load(std::memory_order_relaxed);
    }
    return value != 0;
}

void SetEnabled(bool enabled) noexcept {
    enabled_flag.store(enabled ? 1 : 0, std::memory_order_relaxed);
}

void Add(Step step, std::chrono::nanoseconds time) noexcept {
    const auto index = static_cast<size_t>(step);
    step_nanoseconds[index].fetch_add(static_cast<u64>(time.count()), std::memory_order_relaxed);
    step_counts[index].fetch_add(1, std::memory_order_relaxed);
}

Totals Take() noexcept {
    Totals totals;
    for (size_t index = 0; index < NumSteps; ++index) {
        totals.nanoseconds[index] = step_nanoseconds[index].exchange(0, std::memory_order_relaxed);
        totals.counts[index] = step_counts[index].exchange(0, std::memory_order_relaxed);
    }
    return totals;
}

} // namespace Shader::TranslateProfile
