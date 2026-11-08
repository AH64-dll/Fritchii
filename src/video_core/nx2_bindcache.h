// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdlib>

namespace VideoCommon {

/// Binding cache: a draw or dispatch whose storage buffer bindings match the previous one reuses
/// the resolved buffers, and a dispatch with the previous descriptors reuses their descriptor
/// buffer allocation. EDEN_NX2_BINDCACHE=0 resolves them every time. (Textures: BindMemoEnabled.)
[[nodiscard]] inline bool BindCacheEnabled() noexcept {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_BINDCACHE");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

/// Texture binding memo (TextureCache::FindBinding): a draw or dispatch whose textures, images
/// and samplers all resolve from unchanged guest descriptors and untouched images reuses their
/// view and sampler ids and descriptors. EDEN_NX2_BINDMEMO=0 resolves them every time.
[[nodiscard]] inline bool BindMemoEnabled() noexcept {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_BINDMEMO");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

/// Pipeline transitions are found by a tag scan and kept most-used first.
/// EDEN_NX2_PIPE_TRANSITIONS=0 searches them by full key, in insertion order.
[[nodiscard]] inline bool PipelineTransitionTagsEnabled() noexcept {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_PIPE_TRANSITIONS");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

} // namespace VideoCommon
