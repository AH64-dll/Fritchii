// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <utility>

#include "common/logging.h"
#include "video_core/renderer_vulkan/vk_pipeline_translate_queue.h"

namespace Vulkan {

TranslateQueue::TranslateQueue(size_t thread_count, size_t max_in_flight_, Translate translate_,
                               Retire retire_)
    : max_in_flight{max_in_flight_}, translate{std::move(translate_)}, retire{std::move(retire_)},
      threads{std::make_unique<Common::ThreadWorker>(thread_count, "VkTranslate")} {}

TranslateQueue::~TranslateQueue() {
    // A translation in progress finishes and stores its job; the queued ones are dropped. At
    // shutdown, on the GPU thread: the pipeline workers still run, so a pipeline here waits for
    // its build (or runs it) as any other pipeline the cache destroys then.
    threads.reset();
    entries.clear();
    abandoned.clear();
}

bool TranslateQueue::Enqueue(std::unique_ptr<TranslationJob> job) {
    const GraphicsPipelineCacheKey key = job->key;
    u64 id{};
    {
        std::scoped_lock lock{mutex};
        if (entries.size() >= max_in_flight || entries.contains(key)) {
            return false;
        }
        id = next_id++;
        Entry& entry = entries[key];
        entry.id = id;
        entry.enqueued = job->enqueued;
        entry.job = std::move(job);
    }
    threads->QueueWork([this, key, id] { Run(key, id); });
    return true;
}

void TranslateQueue::Run(GraphicsPipelineCacheKey key, u64 id) {
    std::unique_ptr<TranslationJob> job;
    {
        std::scoped_lock lock{mutex};
        const auto it = entries.find(key);
        if (it == entries.end() || it->second.id != id || !it->second.job) {
            return; // Abandoned before it ran
        }
        job = std::move(it->second.job);
    }
    const auto begin = std::chrono::steady_clock::now();
    try {
        translate(*job);
    } catch (const std::exception& exception) {
        // CreateGraphicsPipeline handles the translation's own errors; this is anything else
        // (out of memory, a driver error building the pipeline object).
        LOG_ERROR(Render_Vulkan, "Async translate: pipeline {:016x} failed on its thread: {}",
                  job->key.Hash(), exception.what());
        // A pipeline made before the failure stays in the job, for the GPU thread to retire.
        job->outcome = TranslationJob::Outcome::NoPipeline;
    } catch (...) {
        LOG_ERROR(Render_Vulkan, "Async translate: pipeline {:016x} failed on its thread",
                  job->key.Hash());
        job->outcome = TranslationJob::Outcome::NoPipeline;
    }
    if (job->outcome == TranslationJob::Outcome::Pending) {
        job->outcome = job->pipeline ? TranslationJob::Outcome::Built
                                     : TranslationJob::Outcome::NoPipeline;
    }
    job->translation = std::chrono::steady_clock::now() - begin;
    std::scoped_lock lock{mutex};
    const auto it = entries.find(key);
    if (it != entries.end() && it->second.id == id) {
        it->second.job = std::move(job);
        it->second.done = true;
        done_condition.notify_all();
        return;
    }
    // Abandoned while it ran (swept, or the GPU thread built it itself): never bound, kept for
    // the GPU thread to retire (RetireAbandoned) instead of destroyed here.
    abandoned.push_back(std::move(job));
}

bool TranslateQueue::Contains(const GraphicsPipelineCacheKey& key) const {
    std::scoped_lock lock{mutex};
    return entries.contains(key);
}

std::unique_ptr<TranslationJob> TranslateQueue::Take(const GraphicsPipelineCacheKey& key) {
    std::scoped_lock lock{mutex};
    const auto it = entries.find(key);
    if (it == entries.end() || !it->second.done) {
        return nullptr;
    }
    auto job = std::move(it->second.job);
    entries.erase(it);
    return job;
}

std::unique_ptr<TranslationJob> TranslateQueue::Wait(const GraphicsPipelineCacheKey& key,
                                                     std::chrono::milliseconds cap) {
    std::unique_lock lock{mutex};
    const bool finished = done_condition.wait_for(lock, cap, [&] {
        const auto it = entries.find(key);
        return it == entries.end() || it->second.done;
    });
    if (!finished) {
        return nullptr;
    }
    const auto it = entries.find(key);
    if (it == entries.end()) {
        return nullptr;
    }
    auto job = std::move(it->second.job);
    entries.erase(it);
    return job;
}

void TranslateQueue::Abandon(const GraphicsPipelineCacheKey& key) {
    std::unique_ptr<TranslationJob> job;
    {
        std::scoped_lock lock{mutex};
        const auto it = entries.find(key);
        if (it == entries.end()) {
            return;
        }
        job = std::move(it->second.job);
        entries.erase(it);
    }
    if (job) {
        Discard(std::move(job));
    }
}

std::vector<GraphicsPipelineCacheKey> TranslateQueue::Sweep(
    std::chrono::steady_clock::duration age) {
    std::vector<GraphicsPipelineCacheKey> keys;
    const auto now = std::chrono::steady_clock::now();
    {
        std::scoped_lock lock{mutex};
        for (const auto& [key, entry] : entries) {
            // A finished job is only waiting for its draw to come back; it is not wedged.
            if (!entry.done && now - entry.enqueued > age) {
                keys.push_back(key);
            }
        }
    }
    for (const auto& key : keys) {
        Abandon(key);
    }
    return keys;
}

std::vector<std::unique_ptr<TranslationJob>> TranslateQueue::TakeFinished() {
    std::vector<std::unique_ptr<TranslationJob>> jobs;
    std::scoped_lock lock{mutex};
    for (auto it = entries.begin(); it != entries.end();) {
        if (it->second.done) {
            jobs.push_back(std::move(it->second.job));
            it = entries.erase(it);
        } else {
            ++it;
        }
    }
    return jobs;
}

void TranslateQueue::RetireAbandoned() {
    std::vector<std::unique_ptr<TranslationJob>> jobs;
    {
        std::scoped_lock lock{mutex};
        jobs.swap(abandoned);
    }
    for (auto& job : jobs) {
        Discard(std::move(job));
    }
}

void TranslateQueue::Discard(std::unique_ptr<TranslationJob> job) {
    if (job && job->pipeline) {
        retire(std::move(job->pipeline));
    }
}

size_t TranslateQueue::InFlight() const {
    std::scoped_lock lock{mutex};
    return entries.size();
}

} // namespace Vulkan
