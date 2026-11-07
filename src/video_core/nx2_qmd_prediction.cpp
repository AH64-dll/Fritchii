// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <mutex>

#include "common/logging.h"
#include "video_core/nx2_qmd_prediction.h"

namespace VideoCommon::QmdPrediction {
namespace {

using Clock = std::chrono::steady_clock;

struct Prediction {
    u64 device;
    u64 position; ///< Ring position after the launch; 0 = empty
    u64 qmd;
    u32 payload;
    Writer writer;
    bool stale;
};

struct Release {
    u64 device;
    u64 sequence;
    u32 payload;
    Writer writer;
    bool reduction;
};

constexpr size_t Buckets = 512;
constexpr size_t Ways = 16;
constexpr size_t Releases = 1024;

struct State {
    std::mutex mutex;
    std::array<std::array<Prediction, Ways>, Buckets> predictions{};
    std::array<u32, Buckets> next{};
    std::array<Release, Releases> releases{};
    u64 sequence{};

    std::array<u64, static_cast<size_t>(Signal::Count)> signals{};
    u64 reductions{};
    u64 unmapped_releases{};
    std::array<u64, 2> predicted{};
    u64 predicted_stale{};
    std::array<u64, 2> matched{};
    std::array<u64, 2> differs{};
    u64 differs_stale{};
    std::array<u64, 2> unpredicted{};
    u64 unpredicted_full{};
    u64 back_reductions{};
    std::array<u64, static_cast<size_t>(Coverage::Count)> coverage{};
    std::array<u64, static_cast<size_t>(MetBy::Count)> met_by{};
    u32 differs_logged{};
    u32 unpredicted_logged{};
    Clock::time_point since{};

    void Tick() {
        const auto now = Clock::now();
        if (since == Clock::time_point{}) {
            since = now;
            return;
        }
        const double window = std::chrono::duration<double>(now - since).count();
        if (window < 5.0) {
            return;
        }
        const auto rate = [window](u64 count) { return static_cast<double>(count) / window; };
        const auto signal = [&](Signal s) { return rate(signals[static_cast<size_t>(s)]); };
        const auto covered = [&](Coverage c) { return rate(coverage[static_cast<size_t>(c)]); };
        const auto met = [&](MetBy m) { return rate(met_by[static_cast<size_t>(m)]); };
        constexpr size_t qmd = static_cast<size_t>(Writer::Qmd);
        constexpr size_t dma = static_cast<size_t>(Writer::Dma);
        LOG_INFO(HW_GPU,
                 "QMD release prediction (EXP13): {:.0f} signals/s read ({:.0f} waiting on "
                 "dependencies, {:.0f} chained, {:.0f} unmapped QMD), {:.0f} releases predicted/s "
                 "({:.0f} maybe stale QMD; not predicted: {:.0f} reductions, {:.0f} unmapped), "
                 "{:.0f} copy semaphores predicted/s; back QMD: {:.0f} matched, {:.0f} payload "
                 "differs ({:.0f} maybe stale QMD), {:.0f} unpredicted ({:.0f} table full), "
                 "{:.0f} reductions; back copies: {:.0f} matched, {:.0f} payload differs, {:.0f} "
                 "unpredicted; unmet acquires: {:.0f} would be met by a predicted queued release, "
                 "{:.0f} wrong value, {:.0f} already ran, {:.0f} no prediction; met after fences "
                 "by: {:.0f} QMD release, {:.0f} QMD reduction, {:.0f} copy semaphore, {:.0f} "
                 "other writer, {:.0f} still unmet",
                 signal(Signal::Read), signal(Signal::Waiting), signal(Signal::Chained),
                 signal(Signal::Unmapped), rate(predicted[qmd]), rate(predicted_stale),
                 rate(reductions), rate(unmapped_releases), rate(predicted[dma]),
                 rate(matched[qmd]), rate(differs[qmd]), rate(differs_stale),
                 rate(unpredicted[qmd]), rate(unpredicted_full), rate(back_reductions),
                 rate(matched[dma]), rate(differs[dma]), rate(unpredicted[dma]),
                 covered(Coverage::WouldMeet), covered(Coverage::WrongValue),
                 covered(Coverage::AlreadyRan), covered(Coverage::None), met(MetBy::QmdRelease),
                 met(MetBy::QmdReduction), met(MetBy::DmaRelease), met(MetBy::Other),
                 met(MetBy::Unmet));
        signals = {};
        reductions = unmapped_releases = predicted_stale = differs_stale = 0;
        predicted = matched = differs = unpredicted = {};
        unpredicted_full = back_reductions = 0;
        coverage = {};
        met_by = {};
        since = now;
    }
};

State& Get() {
    static State state;
    return state;
}

size_t Hash(u64 device) {
    return static_cast<size_t>((device * 0x9e3779b97f4a7c15ULL) >> 32);
}

size_t Bucket(u64 device) {
    return Hash(device) % Buckets;
}

} // Anonymous namespace

void NoteSignal(Signal signal) {
    auto& state = Get();
    std::scoped_lock lock{state.mutex};
    ++state.signals[static_cast<size_t>(signal)];
    state.Tick();
}

void NoteReduction() {
    auto& state = Get();
    std::scoped_lock lock{state.mutex};
    ++state.reductions;
}

void NoteUnmappedRelease() {
    auto& state = Get();
    std::scoped_lock lock{state.mutex};
    ++state.unmapped_releases;
}

void Predict(Writer writer, u64 device, u32 payload, u64 position, u64 qmd, bool stale) {
    auto& state = Get();
    std::scoped_lock lock{state.mutex};
    const size_t bucket = Bucket(device);
    state.predictions[bucket][state.next[bucket]++ % Ways] = {device, position, qmd, payload,
                                                              writer, stale};
    ++state.predicted[static_cast<size_t>(writer)];
    state.predicted_stale += stale ? 1 : 0;
    state.Tick();
}

void Verify(Writer writer, u64 device, u32 payload, u64 qmd, u64 entry_position, bool reduction) {
    auto& state = Get();
    std::scoped_lock lock{state.mutex};
    state.releases[Hash(device) % Releases] = {device, ++state.sequence, payload, writer,
                                                 reduction};
    if (reduction) {
        ++state.back_reductions;
        return;
    }
    // The launch the back runs is the first one sent after the entry it runs starts.
    const auto& ways = state.predictions[Bucket(device)];
    const Prediction* found = nullptr;
    u32 same_device = 0;
    u64 oldest = ~u64{0};
    for (const Prediction& way : ways) {
        oldest = (std::min)(oldest, way.position);
        if (way.position <= entry_position || way.device != device || way.writer != writer) {
            continue;
        }
        if (way.qmd != qmd) {
            ++same_device;
            continue;
        }
        if (!found || way.position < found->position ||
            (way.position == found->position && way.payload == payload)) {
            found = &way;
        }
    }
    const size_t index = static_cast<size_t>(writer);
    if (!found) {
        ++state.unpredicted[index];
        // Every way was predicted after this entry: the prediction may have been evicted.
        state.unpredicted_full += oldest > entry_position && writer == Writer::Qmd ? 1 : 0;
        if (state.unpredicted_logged < 8) {
            ++state.unpredicted_logged;
            LOG_INFO(HW_GPU,
                     "QMD release prediction (EXP13): unpredicted {} release, QMD {:#x}, device "
                     "{:#x} = {:#x}, entry {}; queued predictions there from other QMDs: {}",
                     writer == Writer::Qmd ? "QMD" : "copy", qmd, device, payload, entry_position,
                     same_device);
        }
        return;
    }
    if (found->payload == payload) {
        ++state.matched[index];
        return;
    }
    ++state.differs[index];
    state.differs_stale += found->stale ? 1 : 0;
    if (state.differs_logged < 16) {
        ++state.differs_logged;
        LOG_INFO(HW_GPU,
                 "QMD release prediction (EXP13): {} release payload differs, QMD {:#x}, device "
                 "{:#x}: predicted {:#x}, written {:#x}; QMD maybe stale {}, sent at {}, run at {}",
                 writer == Writer::Qmd ? "QMD" : "copy", qmd, device, found->payload, payload,
                 found->stale, found->position, entry_position);
    }
}

Queued QueuedPayloads(u64 device, u64 consumed) {
    auto& state = Get();
    std::scoped_lock lock{state.mutex};
    Queued queued{};
    for (const Prediction& way : state.predictions[Bucket(device)]) {
        if (way.position == 0 || way.device != device) {
            continue;
        }
        queued.any = true;
        if (way.position > consumed && queued.count < queued.payloads.size()) {
            queued.payloads[queued.count++] = way.payload;
        }
    }
    return queued;
}

void NoteAcquire(Coverage coverage) {
    auto& state = Get();
    std::scoped_lock lock{state.mutex};
    ++state.coverage[static_cast<size_t>(coverage)];
    state.Tick();
}

u64 BackSequence() {
    auto& state = Get();
    std::scoped_lock lock{state.mutex};
    return state.sequence;
}

std::optional<BackRelease> BackReleaseSince(u64 device, u64 since) {
    auto& state = Get();
    std::scoped_lock lock{state.mutex};
    const Release& release = state.releases[Hash(device) % Releases];
    if (release.device != device || release.sequence <= since) {
        return std::nullopt;
    }
    return BackRelease{release.writer, release.payload, release.reduction};
}

void NoteMetBy(MetBy met_by) {
    auto& state = Get();
    std::scoped_lock lock{state.mutex};
    ++state.met_by[static_cast<size_t>(met_by)];
}

} // namespace VideoCommon::QmdPrediction
