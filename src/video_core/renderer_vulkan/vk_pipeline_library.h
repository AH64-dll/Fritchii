// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

// Graphics pipeline libraries (EDEN_NX2_GPL=1, VK_EXT_graphics_pipeline_library).
//
// A pipeline is linked from four parts: vertex input, pre-rasterization shaders, fragment shader
// and fragment output. The two shader parts are what the driver spends seconds on; they are built
// once per shader (and the little state the driver bakes into them) and shared by every pipeline
// that uses that shader, whatever its partner stage. The vertex input and fragment output parts
// are cheap and built per pipeline. A new pipeline whose shaders were seen before costs a link
// (a fraction of a millisecond with fast linking).
//
// For a shader library to be independent of its partner, each stage has its own descriptor set
// (set index = stage index, VK_PIPELINE_LAYOUT_CREATE_INDEPENDENT_SETS_BIT_EXT) with its bindings
// numbered from 0, and the fragment stage's texture and image rescaling bits start at a fixed
// index instead of after the previous stages' ones.
//
// Libraries are keyed by the SPIR-V the stage was emitted to (a 128-bit hash), the stage's
// descriptor set layout, the push constant range, the state the part bakes in, the dynamic state
// list, the render pass and the creation flags. Keying by the emitted SPIR-V is always correct;
// it is less reuse than possible when the emission depends on the partner stage or on state (see
// MakeRuntimeInfo), which can be narrowed later in the emission.
//
// Based in part on suyu-main 3157ae0ce0 and f80bbc701a (library cache with build outside the
// lock, fast link then optimized link, link statistics); unlike there, the shader libraries here
// do not depend on the partner stage.

/// The fragment stage's first texture and image rescaling bit with libraries (the push constant
/// has 128 texture and 64 image bits; the pre-rasterization stages use the ones below).
constexpr u32 GPL_FRAGMENT_TEXTURE_SCALING_BASE = 64;
constexpr u32 GPL_FRAGMENT_IMAGE_SCALING_BASE = 32;

/// EDEN_NX2_GPL_OPTIMIZE=1: every pipeline is linked optimized
/// (VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT) instead of fast linked, as on drivers
/// whose fast links are not usable (Device::CanFastLinkGraphicsLibraries). The shader libraries
/// are then created with VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT.
bool GplOptimizeEnabled();

enum class GplPart : u32 {
    PreRasterization = 0,
    Fragment = 1,
};

/// A cached shader library and the layout objects it was created with.
struct GplLibrary {
    std::vector<vk::DescriptorSetLayout> set_layouts;
    vk::PipelineLayout layout;
    vk::Pipeline pipeline;
};

/// Bytes of a library key. Only whole scalar values are added, never structs with padding.
class GplKeyWriter {
public:
    template <typename T>
        requires std::is_arithmetic_v<T> || std::is_enum_v<T> || std::is_pointer_v<T>
    void Add(T value) {
        char bytes[sizeof(T)];
        std::memcpy(bytes, &value, sizeof(T));
        data.append(bytes, sizeof(T));
    }

    void Add(const u128& value) {
        Add(value[0]);
        Add(value[1]);
    }

    void AddBytes(const std::string& bytes) {
        Add(static_cast<u64>(bytes.size()));
        data.append(bytes);
    }

    [[nodiscard]] const std::string& Data() const noexcept {
        return data;
    }

private:
    std::string data;
};

class GraphicsPipelineLibraryCache {
public:
    using Builder = std::function<GplLibrary()>;

    /// The library for `key`: built with `build` on this thread when nobody has built it yet,
    /// waited for when another thread is building it. Null when its build failed (a failed key
    /// is not retried). `name` (a shader hash) is only for the log.
    std::shared_ptr<const GplLibrary> GetOrCreate(GplPart part, const std::string& key,
                                                  const Builder& build, u64 name);

    /// While set on a thread, libraries it builds are not kept in the cache (the graphics
    /// warm-up: only the driver's disk cache keeps them, 60,000 kept libraries would not fit).
    static void SetNoRetainOnThisThread(bool no_retain) noexcept;

    /// Over `max_entries`, forgets libraries no pipeline holds any more (their pipelines were
    /// evicted), so the cache does not grow for the whole session. A forgotten one is built
    /// again if needed, mostly from the driver's disk cache.
    void TrimUnused(size_t max_entries);

    void RecordLink(bool optimized, std::chrono::nanoseconds time);
    void RecordHeldDraw(std::chrono::nanoseconds time);
    void RecordFallback();

    /// Logs the counters of the last 10 s (Nx2Perf gpl), when there were any.
    void Report();

private:
    struct Entry {
        std::mutex mutex;
        std::condition_variable condition;
        bool done{};
        std::shared_ptr<const GplLibrary> library;
    };

    std::mutex mutex;
    std::unordered_map<std::string, std::shared_ptr<Entry>> entries;

    std::array<std::atomic<u32>, 2> built{};
    std::array<std::atomic<u32>, 2> failed{};
    std::array<std::atomic<u32>, 2> hits{};
    std::array<std::atomic<u32>, 2> waited{};
    std::array<std::atomic<u64>, 2> build_ns{};
    std::array<std::atomic<u64>, 2> build_max_ns{};
    std::array<std::atomic<u32>, 2> link_count{};
    std::array<std::atomic<u64>, 2> link_ns{};
    std::array<std::atomic<u64>, 2> link_max_ns{};
    std::atomic<u64> held_ns{};
    std::atomic<u32> fallbacks{};
    std::atomic<u32> logged_builds{};
    std::atomic<size_t> total_libraries{};
    std::chrono::steady_clock::time_point window = std::chrono::steady_clock::now();
};

} // namespace Vulkan
