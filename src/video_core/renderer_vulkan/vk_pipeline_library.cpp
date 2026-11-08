// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>
#include <exception>

#include "common/logging.h"
#include "video_core/renderer_vulkan/vk_pipeline_library.h"

namespace Vulkan {

namespace {
void StoreMax(std::atomic<u64>& target, u64 value) {
    u64 previous = target.load(std::memory_order_relaxed);
    while (previous < value &&
           !target.compare_exchange_weak(previous, value, std::memory_order_relaxed)) {
    }
}

double Ms(u64 ns) {
    return static_cast<double>(ns) / 1'000'000.0;
}

constexpr std::array<const char*, 2> PART_NAMES{"pre-rasterization", "fragment"};

thread_local bool no_retain_on_thread = false;
} // Anonymous namespace

void GraphicsPipelineLibraryCache::SetNoRetainOnThisThread(bool no_retain) noexcept {
    no_retain_on_thread = no_retain;
}

bool GplOptimizeEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_GPL_OPTIMIZE");
        return value && *value && *value != '0';
    }();
    return enabled;
}

std::shared_ptr<const GplLibrary> GraphicsPipelineLibraryCache::GetOrCreate(
    GplPart part, const std::string& key, const Builder& build, u64 name) {
    const size_t index = static_cast<size_t>(part);
    std::shared_ptr<Entry> entry;
    bool owner = false;
    {
        std::scoped_lock lock{mutex};
        auto [it, inserted] = entries.try_emplace(key);
        if (inserted) {
            it->second = std::make_shared<Entry>();
            owner = true;
        }
        entry = it->second;
    }
    if (!owner) {
        std::unique_lock lock{entry->mutex};
        if (!entry->done) {
            // Another worker is building the same library: wait for it instead of building it
            // twice.
            waited[index].fetch_add(1, std::memory_order_relaxed);
            entry->condition.wait(lock, [&] { return entry->done; });
        }
        hits[index].fetch_add(1, std::memory_order_relaxed);
        return entry->library;
    }
    // Built outside the map lock: other libraries are built at the same time on other workers.
    const auto begin = std::chrono::steady_clock::now();
    std::shared_ptr<const GplLibrary> library;
    try {
        library = std::make_shared<const GplLibrary>(build());
        if (!library->pipeline) {
            library.reset();
        }
    } catch (const std::exception& exception) {
        LOG_ERROR(Render_Vulkan, "Nx2Perf gpl: {} library {:016x} failed: {}", PART_NAMES[index],
                  name, exception.what());
        library.reset();
    } catch (...) {
        LOG_ERROR(Render_Vulkan, "Nx2Perf gpl: {} library {:016x} failed", PART_NAMES[index],
                  name);
        library.reset();
    }
    const u64 ns = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now() - begin)
                                        .count());
    if (library) {
        built[index].fetch_add(1, std::memory_order_relaxed);
        total_libraries.fetch_add(1, std::memory_order_relaxed);
        build_ns[index].fetch_add(ns, std::memory_order_relaxed);
        StoreMax(build_max_ns[index], ns);
        if (ns > 2'000'000 && logged_builds.fetch_add(1, std::memory_order_relaxed) < 400) {
            LOG_INFO(Render_Vulkan, "Nx2Perf gpl: {} library {:016x} built in {:.1f} ms",
                     PART_NAMES[index], name, Ms(ns));
        }
    } else {
        failed[index].fetch_add(1, std::memory_order_relaxed);
    }
    {
        std::scoped_lock lock{entry->mutex};
        entry->library = library;
        entry->done = true;
    }
    entry->condition.notify_all();
    if (no_retain_on_thread) {
        // Threads already waiting on the entry hold it; later lookups build again, which the
        // driver's disk cache answers in about a millisecond.
        std::scoped_lock lock{mutex};
        if (const auto it = entries.find(key); it != entries.end() && it->second == entry) {
            entries.erase(it);
        }
    }
    return library;
}

void GraphicsPipelineLibraryCache::TrimUnused(size_t max_entries) {
    std::scoped_lock lock{mutex};
    if (entries.size() <= max_entries) {
        return;
    }
    size_t trimmed = 0;
    for (auto it = entries.begin(); it != entries.end() && entries.size() > max_entries;) {
        Entry& entry = *it->second;
        bool unused = false;
        {
            std::scoped_lock entry_lock{entry.mutex};
            // Failed keys stay (they are not retried); a library only the map holds is unused.
            unused = entry.done && entry.library && entry.library.use_count() == 1;
        }
        if (unused) {
            it = entries.erase(it);
            ++trimmed;
        } else {
            ++it;
        }
    }
    if (trimmed != 0) {
        LOG_INFO(Render_Vulkan, "Nx2Perf gpl: {} unused libraries released, {} left", trimmed,
                 entries.size());
    }
}

void GraphicsPipelineLibraryCache::RecordLink(bool optimized, std::chrono::nanoseconds time) {
    const size_t index = optimized ? 1 : 0;
    const u64 ns = static_cast<u64>(time.count());
    link_count[index].fetch_add(1, std::memory_order_relaxed);
    link_ns[index].fetch_add(ns, std::memory_order_relaxed);
    StoreMax(link_max_ns[index], ns);
}

void GraphicsPipelineLibraryCache::RecordHeldDraw(std::chrono::nanoseconds time) {
    held_ns.fetch_add(static_cast<u64>(time.count()), std::memory_order_relaxed);
}

void GraphicsPipelineLibraryCache::RecordFallback() {
    fallbacks.fetch_add(1, std::memory_order_relaxed);
}

void GraphicsPipelineLibraryCache::Report() {
    const auto now = std::chrono::steady_clock::now();
    if (now - window < std::chrono::seconds{10}) {
        return;
    }
    window = now;
    const auto take = [](auto& counter) { return counter.exchange(0, std::memory_order_relaxed); };
    const u32 pre_built = take(built[0]), frag_built = take(built[1]);
    const u32 pre_hits = take(hits[0]), frag_hits = take(hits[1]);
    const u32 fast_links = take(link_count[0]), optimized_links = take(link_count[1]);
    const u32 pre_failed = take(failed[0]), frag_failed = take(failed[1]);
    const u32 fallback_count = take(fallbacks);
    const u64 held = take(held_ns);
    if (pre_built == 0 && frag_built == 0 && pre_hits == 0 && frag_hits == 0 && fast_links == 0 &&
        optimized_links == 0 && held == 0 && pre_failed == 0 && frag_failed == 0 &&
        fallback_count == 0) {
        return;
    }
    const u32 pre_waited = take(waited[0]), frag_waited = take(waited[1]);
    const u64 pre_ns = take(build_ns[0]), frag_ns = take(build_ns[1]);
    const u64 pre_max = take(build_max_ns[0]), frag_max = take(build_max_ns[1]);
    const u64 fast_ns = take(link_ns[0]), optimized_ns = take(link_ns[1]);
    const u64 fast_max = take(link_max_ns[0]), optimized_max = take(link_max_ns[1]);
    const auto mean = [](u64 total, u32 count) {
        return count == 0 ? 0.0 : Ms(total) / static_cast<double>(count);
    };
    LOG_INFO(Render_Vulkan,
             "Nx2Perf gpl in the last 10 s: libraries built pre-rasterization {} ({:.0f} ms, max "
             "{:.1f} ms), fragment {} ({:.0f} ms, max {:.1f} ms); cache hits pre-rasterization {} "
             "(waited on a build {}), fragment {} (waited {}); failed {}/{}; fast links {} (mean "
             "{:.3f} ms, max {:.3f} ms), optimized links {} (mean {:.1f} ms, max {:.1f} ms); "
             "monolithic fallbacks {}; draws held {:.1f} ms; {} libraries cached",
             pre_built, Ms(pre_ns), Ms(pre_max), frag_built, Ms(frag_ns), Ms(frag_max), pre_hits,
             pre_waited, frag_hits, frag_waited, pre_failed, frag_failed, fast_links,
             mean(fast_ns, fast_links), Ms(fast_max), optimized_links,
             mean(optimized_ns, optimized_links), Ms(optimized_max), fallback_count, Ms(held),
             total_libraries.load(std::memory_order_relaxed));
}

} // namespace Vulkan
