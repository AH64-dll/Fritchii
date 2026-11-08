// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "common/thread_worker.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/shader_environment.h"

namespace Vulkan {

/// One graphics pipeline translated off the GPU thread (EDEN_NX2_ASYNC_TRANSLATE).
struct TranslationJob {
    enum class Outcome {
        Pending,
        Built,      ///< `pipeline` is the translated pipeline (its driver compile may still run)
        NoPipeline, ///< The translation made none (as the GPU thread's would have): left empty
        Retry,      ///< A guarded read gave up: translate again from a fresh snapshot
    };

    GraphicsPipelineCacheKey key{};
    /// Snapshot environments (GraphicsEnvironment::TakeSnapshot), at the program index of each
    /// stage the key uses, as ShaderCache::GraphicsEnvironments holds them.
    std::array<VideoCommon::GraphicsEnvironment, 6> envs;
    bool native{};  ///< A native draw asked for it (the emission budget's question)
    u32 attempt{};  ///< Translations of this key before this one (re-queues)
    std::chrono::steady_clock::time_point enqueued{};

    // Written by the translation thread.
    Outcome outcome{Outcome::Pending};
    std::unique_ptr<GraphicsPipeline> pipeline;
    std::chrono::steady_clock::duration translation{}; ///< Time on the translation thread
};

/// The queue of graphics pipelines translated on threads of their own. The pipeline cache owns
/// it and calls it from the GPU thread only; the translation itself is the `translate` callback
/// (PipelineCache's, which runs the prebuild's CreateGraphicsPipeline on the snapshot), so the
/// queue knows nothing of the cache's internals.
///
/// Its own threads at normal priority rather than the pipeline workers: those are at background
/// priority and their queue has no front, so a translation queued behind the boot prebuild would
/// wait for all of it. The driver compile of a translated pipeline still goes to the workers.
class TranslateQueue {
public:
    using Translate = std::function<void(TranslationJob&)>;

    TranslateQueue(size_t threads, size_t max_in_flight, Translate translate);
    /// Joins the threads (a translation in progress finishes); queued jobs are dropped.
    ~TranslateQueue();

    TranslateQueue(const TranslateQueue&) = delete;
    TranslateQueue& operator=(const TranslateQueue&) = delete;

    /// Queues the job, unless its key is queued already or too many are in flight (then false,
    /// and the caller translates on the GPU thread as before).
    bool Enqueue(std::unique_ptr<TranslationJob> job);

    /// Whether `key` is queued or translated and not taken yet.
    [[nodiscard]] bool Contains(const GraphicsPipelineCacheKey& key) const;

    /// The finished job of `key` (removed from the queue), or null while it still translates or
    /// when it is not queued.
    [[nodiscard]] std::unique_ptr<TranslationJob> Take(const GraphicsPipelineCacheKey& key);

    /// Take, waiting up to `cap` for the translation to finish.
    [[nodiscard]] std::unique_ptr<TranslationJob> Wait(const GraphicsPipelineCacheKey& key,
                                                       std::chrono::milliseconds cap);

    /// Forgets `key`: a translation still running is dropped when it ends.
    void Abandon(const GraphicsPipelineCacheKey& key);

    /// Abandons the jobs queued longer than `age` (a wedged translation or driver), returning
    /// their keys.
    [[nodiscard]] std::vector<GraphicsPipelineCacheKey> Sweep(
        std::chrono::steady_clock::duration age);

    /// Every finished job not taken yet (removed from the queue): the draws that asked for them
    /// have not come back, and the cache adopts them between frames.
    [[nodiscard]] std::vector<std::unique_ptr<TranslationJob>> TakeFinished();

    /// Destroys a finished job's pipeline on a translation thread: one never bound, whose
    /// destructor may wait for its driver compile, which the GPU thread must not.
    void Discard(std::unique_ptr<TranslationJob> job);

    [[nodiscard]] size_t InFlight() const;

private:
    struct Entry {
        u64 id{};
        std::unique_ptr<TranslationJob> job; ///< Null while a thread translates it
        bool done{};
        std::chrono::steady_clock::time_point enqueued{};
    };

    void Run(GraphicsPipelineCacheKey key, u64 id);

    mutable std::mutex mutex;
    std::condition_variable done_condition;
    std::unordered_map<GraphicsPipelineCacheKey, Entry> entries;
    u64 next_id{1};
    const size_t max_in_flight;
    Translate translate;
    /// Last: destroyed (its threads joined) before the state above.
    std::unique_ptr<Common::ThreadWorker> threads;
};

} // namespace Vulkan
