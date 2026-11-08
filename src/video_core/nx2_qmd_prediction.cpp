// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <mutex>
#include <span>
#include <string>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2.h"
#include "video_core/nx2_qmd_prediction.h"

namespace VideoCommon::QmdPrediction {
namespace {

using Clock = std::chrono::steady_clock;

struct Prediction {
    u64 device;
    u64 position; ///< Ring position after the launch; 0 = empty
    u64 qmd;
    u64 sequence; ///< Order of the records (positions repeat within one entry)
    u32 payload;
    u32 before;   ///< EXP13=2: the word in guest memory when it was sent
    /// EXP13=2: the landed count of the word (Tegra::g_nx2_semaphore_landed) when the chain of
    /// unlanded records it belongs to started, and its place in that chain: landed once the
    /// count grew by `landed_sent`.
    u32 landed_base;
    u32 landed_sent;
    Writer writer;
    Kind kind;
    u8 op;
    bool is_signed;
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
constexpr size_t Writers = static_cast<size_t>(Writer::Count);

/// Why an acquire got no reduction lower bound (fable_answer_6 6c, section 5).
enum class Bail : u32 {
    NoRecords,     ///< No queued reduction or max release to the word
    MirrorOff,     ///< The records are not complete for the queued work
    GpuDirty,      ///< The word's page is GPU-written
    Evicted,       ///< A record of the word's bucket was evicted while queued
    PlainQueued,   ///< A plain release to the word is queued
    PlainUnlanded, ///< A plain release to the word ran but has not landed
    OtherWriter,   ///< Another noted write to the word is queued
    Unbounded,     ///< Queued work that may write anything
    NonMonotonic,  ///< MIN, DEC, AND or XOR queued
    SignedMax,     ///< A signed MAX queued
    IncPayload,    ///< INC with a payload other than 0xffffffff queued
    Wrap,          ///< The bound would pass 0x7fffffff
    Count,
};

/// What a computed bound answers, by compare.
enum class Outcome : u32 {
    GeqMet,
    GeqShort,
    MaskMet,
    MaskShort,
    MaskNotOr, ///< Not every record folded is OR: no answer
    EqMet,
    EqDiffers,
    EqNotExact, ///< Run but unlanded records, or the word moved since they were sent
    OtherOp,
    Count,
};

/// Acquire compares (Tegra::Nx2SemaphoreWaiters' values).
enum class Compare : u32 { Equal, Gequal, Mask, Other, Count };

Compare CompareOf(u32 operation) {
    switch (operation) {
    case Tegra::Nx2SemaphoreWaiters::AcquireEqual:
        return Compare::Equal;
    case Tegra::Nx2SemaphoreWaiters::AcquireGequal:
        return Compare::Gequal;
    case Tegra::Nx2SemaphoreWaiters::AcquireMask:
        return Compare::Mask;
    default:
        return Compare::Other;
    }
}

bool Meets(Compare compare, u32 value, u32 current) {
    switch (compare) {
    case Compare::Equal:
        return current == value;
    case Compare::Gequal:
        return current >= value;
    case Compare::Mask:
        return (current & value) != 0;
    default:
        return true;
    }
}

size_t Hash(u64 device) {
    return static_cast<size_t>((device * 0x9e3779b97f4a7c15ULL) >> 32);
}

size_t Bucket(u64 device) {
    return Hash(device) % Buckets;
}

template <typename Array>
std::string Rates(const Array& counts, double window) {
    std::string text;
    for (const u64 count : counts) {
        text += fmt::format("{}{:.0f}", text.empty() ? "" : "/",
                            static_cast<double>(count) / window);
    }
    return text;
}

struct State {
    std::mutex mutex;
    std::array<std::array<Prediction, Ways>, Buckets> predictions{};
    std::array<u32, Buckets> next{};
    /// The newest position of a record evicted from each bucket (EXP13=2's bound).
    std::array<u64, Buckets> evicted{};
    std::array<Release, Releases> releases{};
    u64 sequence{};
    u64 record_sequence{};
    bool records_valid{};
    u64 records_valid_from{};

    std::array<u64, static_cast<size_t>(Signal::Count)> signals{};
    u64 reductions{};
    u64 unmapped_releases{};
    std::array<u64, Writers> predicted{};
    u64 predicted_stale{};
    u64 predicted_at_least{};
    std::array<u64, Writers> matched{};
    std::array<u64, Writers> differs{};
    u64 differs_stale{};
    std::array<u64, Writers> unpredicted{};
    u64 unpredicted_full{};
    u64 back_reductions{};
    std::array<u64, static_cast<size_t>(Coverage::Count)> coverage{};
    std::array<u64, static_cast<size_t>(MetBy::Count)> met_by{};
    std::array<u64, static_cast<size_t>(Mirror::Count)> mirror{};
    std::array<u64, static_cast<size_t>(Bail::Count)> bails{};
    std::array<u64, static_cast<size_t>(Outcome::Count)> outcomes{};
    std::array<u64, static_cast<size_t>(Compare::Count)> on_reduction_words{};
    /// Reduction records by op: QMD unsigned, QMD signed, compute reports.
    std::array<std::array<u64, 8>, 3> reduction_ops{};
    u64 bound_checked{};
    u64 bound_violated{};
    u64 bound_unlanded{};
    u32 differs_logged{};
    u32 unpredicted_logged{};
    u32 violated_logged{};
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
        const auto mirrored = [&](Mirror m) { return rate(mirror[static_cast<size_t>(m)]); };
        const auto bailed = [&](Bail b) { return rate(bails[static_cast<size_t>(b)]); };
        const auto outcome = [&](Outcome o) { return rate(outcomes[static_cast<size_t>(o)]); };
        const auto on_words = [&](Compare c) {
            return rate(on_reduction_words[static_cast<size_t>(c)]);
        };
        constexpr size_t qmd = static_cast<size_t>(Writer::Qmd);
        constexpr size_t dma = static_cast<size_t>(Writer::Dma);
        LOG_INFO(HW_GPU,
                 "QMD release prediction (EXP13={}): {:.0f} signals/s read ({:.0f} waiting on "
                 "dependencies, {:.0f} chained, {:.0f} unmapped QMD), {:.0f} releases predicted/s "
                 "({:.0f} max releases as lower bounds, {:.0f} maybe stale QMD; not predicted: "
                 "{:.0f} reductions, {:.0f} unmapped), {:.0f} copy semaphores predicted/s; back "
                 "QMD: {:.0f} matched, {:.0f} payload differs ({:.0f} maybe stale QMD), {:.0f} "
                 "unpredicted ({:.0f} table full), {:.0f} reductions; back copies: {:.0f} "
                 "matched, {:.0f} payload differs, {:.0f} unpredicted; unmet acquires: {:.0f} "
                 "would be met by a predicted queued release, {:.0f} wrong value, {:.0f} already "
                 "ran ({:.0f} landed pending, {:.0f} really wrong), {:.0f} no prediction; met "
                 "after fences by: {:.0f} QMD release, {:.0f} QMD reduction, {:.0f} copy "
                 "semaphore, {:.0f} other writer, {:.0f} still unmet; QMD copy: QMD reads/s "
                 "{:.0f} from inline QMDs, {:.0f} from memory, {:.0f} after a wait; {:.0f} "
                 "resyncs/s, {:.0f} starts, {:.0f} auto-off; reduction bound would meet: {:.0f} "
                 "GEQ, {:.0f} EQ, {:.0f} MASK",
                 Mode(), signal(Signal::Read), signal(Signal::Waiting), signal(Signal::Chained),
                 signal(Signal::Unmapped), rate(predicted[qmd]), rate(predicted_at_least),
                 rate(predicted_stale), rate(reductions), rate(unmapped_releases),
                 rate(predicted[dma]), rate(matched[qmd]), rate(differs[qmd]),
                 rate(differs_stale), rate(unpredicted[qmd]), rate(unpredicted_full),
                 rate(back_reductions), rate(matched[dma]), rate(differs[dma]),
                 rate(unpredicted[dma]), covered(Coverage::WouldMeet),
                 covered(Coverage::WrongValue), covered(Coverage::AlreadyRan),
                 covered(Coverage::LandedPending), covered(Coverage::ReallyWrong),
                 covered(Coverage::None), met(MetBy::QmdRelease), met(MetBy::QmdReduction),
                 met(MetBy::DmaRelease), met(MetBy::Other), met(MetBy::Unmet),
                 mirrored(Mirror::Overlay), mirrored(Mirror::Memory), mirrored(Mirror::Waited),
                 mirrored(Mirror::Resync), mirrored(Mirror::SessionStart),
                 mirrored(Mirror::AutoOff), outcome(Outcome::GeqMet), outcome(Outcome::EqMet),
                 outcome(Outcome::MaskMet));
        if (Mode() == 2) {
            LOG_INFO(HW_GPU,
                     "QMD release prediction (EXP13=2) reduction bound: acquires on reduction "
                     "words {:.0f} GEQ, {:.0f} EQ, {:.0f} MASK, {:.0f} other; GEQ {:.0f} met, "
                     "{:.0f} short; MASK {:.0f} met, {:.0f} short, {:.0f} not all OR; EQ {:.0f} "
                     "met, {:.0f} differs, {:.0f} not exact; {:.0f} other compares; no bound: "
                     "{:.0f} no records, {:.0f} copy off, {:.0f} GPU-written, {:.0f} evicted, "
                     "{:.0f} plain queued, {:.0f} plain not landed, {:.0f} other writer, {:.0f} "
                     "unbounded writer, {:.0f} non-monotonic op, {:.0f} signed max, {:.0f} inc "
                     "payload, {:.0f} wrap guard; safety: {} checked, {} bound violated, {} not "
                     "landed yet; records/s by op add/min/max/inc/dec/and/or/xor: QMD {}, QMD "
                     "signed {}, compute report {}",
                     on_words(Compare::Gequal), on_words(Compare::Equal), on_words(Compare::Mask),
                     on_words(Compare::Other), outcome(Outcome::GeqMet),
                     outcome(Outcome::GeqShort), outcome(Outcome::MaskMet),
                     outcome(Outcome::MaskShort), outcome(Outcome::MaskNotOr),
                     outcome(Outcome::EqMet), outcome(Outcome::EqDiffers),
                     outcome(Outcome::EqNotExact), outcome(Outcome::OtherOp),
                     bailed(Bail::NoRecords), bailed(Bail::MirrorOff), bailed(Bail::GpuDirty),
                     bailed(Bail::Evicted), bailed(Bail::PlainQueued),
                     bailed(Bail::PlainUnlanded), bailed(Bail::OtherWriter),
                     bailed(Bail::Unbounded), bailed(Bail::NonMonotonic),
                     bailed(Bail::SignedMax), bailed(Bail::IncPayload), bailed(Bail::Wrap),
                     bound_checked, bound_violated, bound_unlanded,
                     Rates(reduction_ops[0], window), Rates(reduction_ops[1], window),
                     Rates(reduction_ops[2], window));
        }
        signals = {};
        reductions = unmapped_releases = predicted_stale = predicted_at_least = 0;
        differs_stale = 0;
        predicted = matched = differs = unpredicted = {};
        unpredicted_full = back_reductions = 0;
        coverage = {};
        met_by = {};
        mirror = {};
        bails = {};
        outcomes = {};
        on_reduction_words = {};
        reduction_ops = {};
        bound_checked = bound_violated = bound_unlanded = 0;
        since = now;
    }

    /// Stores a record, chaining its landed count to the word's newest unlanded record.
    void Insert(const Prediction& record) {
        const size_t bucket = Bucket(record.device);
        auto& ways = predictions[bucket];
        const u32 landed = LandedCount(record.device);
        const Prediction* newest = nullptr;
        for (const Prediction& way : ways) {
            if (way.position != 0 && way.device == record.device &&
                (!newest || way.sequence > newest->sequence)) {
                newest = &way;
            }
        }
        Prediction stored = record;
        stored.sequence = ++record_sequence;
        if (newest && landed - newest->landed_base < newest->landed_sent) {
            stored.landed_base = newest->landed_base;
            stored.landed_sent = newest->landed_sent + 1;
        } else {
            stored.landed_base = landed;
            stored.landed_sent = 1;
        }
        Prediction& slot = ways[next[bucket]++ % Ways];
        if (slot.position != 0) {
            evicted[bucket] = (std::max)(evicted[bucket], slot.position);
        }
        slot = stored;
    }

    static u32 LandedCount(u64 device) {
        return Tegra::g_nx2_semaphore_landed[Tegra::Nx2LandedSlot(device)].load(
            std::memory_order_acquire);
    }
    static bool Landed(const Prediction& record, u32 landed) {
        return landed - record.landed_base >= record.landed_sent;
    }
};

State& Get() {
    static State state;
    return state;
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
    Prediction record{};
    record.device = device;
    record.position = position;
    record.qmd = qmd;
    record.payload = payload;
    record.writer = writer;
    record.kind = Kind::Plain;
    record.stale = stale;
    state.Insert(record);
    ++state.predicted[static_cast<size_t>(writer)];
    state.predicted_stale += stale ? 1 : 0;
    state.Tick();
}

void Note(u64 device, const Record& record, u32 before, u64 position, u64 qmd) {
    auto& state = Get();
    std::scoped_lock lock{state.mutex};
    Prediction stored{};
    stored.device = device;
    stored.position = position;
    stored.qmd = qmd;
    stored.payload = record.payload;
    stored.before = before;
    stored.writer = record.writer;
    stored.kind = record.kind;
    stored.op = static_cast<u8>(record.op & 7);
    stored.is_signed = record.is_signed;
    state.Insert(stored);
    switch (record.kind) {
    case Kind::Reduction:
        if (record.writer == Writer::Report) {
            ++state.reduction_ops[2][record.op & 7];
        } else {
            ++state.reductions;
            ++state.reduction_ops[record.is_signed ? 1 : 0][record.op & 7];
        }
        break;
    case Kind::AtLeast:
        ++state.predicted_at_least;
        [[fallthrough]];
    case Kind::Plain:
        ++state.predicted[static_cast<size_t>(record.writer)];
        break;
    }
    state.Tick();
}

void NoteMirror(Mirror event) {
    auto& state = Get();
    std::scoped_lock lock{state.mutex};
    ++state.mirror[static_cast<size_t>(event)];
}

void SetRecordsValid(bool valid, u64 position) {
    auto& state = Get();
    std::scoped_lock lock{state.mutex};
    state.records_valid = valid;
    state.records_valid_from = position;
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
        if (way.position <= entry_position || way.device != device || way.writer != writer ||
            way.kind == Kind::Reduction) {
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
        if (way.position == 0 || way.device != device || way.kind == Kind::Reduction) {
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

std::optional<Bound> CheckAcquire(const AcquireCheck& check) {
    auto& state = Get();
    std::scoped_lock lock{state.mutex};
    const Compare compare = CompareOf(check.operation);
    const u32 landed = State::LandedCount(check.device);
    const size_t bucket = Bucket(check.device);
    // The word's records, in send order.
    std::array<const Prediction*, Ways> records{};
    size_t count = 0;
    for (const Prediction& way : state.predictions[bucket]) {
        if (way.position != 0 && way.device == check.device) {
            records[count++] = &way;
        }
    }
    std::sort(records.begin(), records.begin() + static_cast<std::ptrdiff_t>(count),
              [](const Prediction* a, const Prediction* b) { return a->sequence < b->sequence; });
    // Positions grow with the send order: the queued records are the last ones.
    size_t queued_from = count;
    while (queued_from > 0 && records[queued_from - 1]->position > check.done) {
        --queued_from;
    }

    // Coverage by the plain predictions: the newest release recorded for the word decides.
    const auto would_meet = [&](const Prediction& r) {
        return r.kind == Kind::AtLeast
                   ? compare == Compare::Gequal && check.value <= r.payload
                   : Meets(compare, check.value, r.payload);
    };
    Coverage coverage = Coverage::None;
    if (count != 0 && records[count - 1]->kind != Kind::Reduction) {
        const Prediction& newest = *records[count - 1];
        if (newest.position > check.done) {
            coverage = would_meet(newest) ? Coverage::WouldMeet : Coverage::WrongValue;
        } else if (!State::Landed(newest, landed) && would_meet(newest) &&
                   (check.word == newest.before || check.word == newest.payload)) {
            coverage = Coverage::LandedPending;
        } else {
            coverage = Coverage::ReallyWrong;
        }
    }
    ++state.coverage[static_cast<size_t>(coverage)];
    if (coverage == Coverage::LandedPending || coverage == Coverage::ReallyWrong) {
        ++state.coverage[static_cast<size_t>(Coverage::AlreadyRan)];
    }
    state.Tick();

    // The reduction lower bound (6c section 3): the word only grows under the queued records.
    const auto bail = [&](Bail reason) -> std::optional<Bound> {
        ++state.bails[static_cast<size_t>(reason)];
        return std::nullopt;
    };
    bool any_monotonic_record = false;
    bool plain_queued = false;
    for (size_t i = queued_from; i < count; ++i) {
        any_monotonic_record |= records[i]->kind != Kind::Plain;
        plain_queued |= records[i]->kind == Kind::Plain;
    }
    if (!any_monotonic_record) {
        return bail(Bail::NoRecords);
    }
    ++state.on_reduction_words[static_cast<size_t>(compare)];
    if (check.gpu_dirty) {
        return bail(Bail::GpuDirty);
    }
    if (!state.records_valid || state.records_valid_from > check.done) {
        return bail(Bail::MirrorOff);
    }
    if (state.evicted[bucket] > check.done) {
        return bail(Bail::Evicted);
    }
    if (plain_queued) {
        return bail(Bail::PlainQueued);
    }
    bool unlanded = false;
    for (size_t i = 0; i < queued_from; ++i) {
        if (!State::Landed(*records[i], landed)) {
            if (records[i]->kind == Kind::Plain) {
                // It lands later and sets the word, maybe below the bound.
                return bail(Bail::PlainUnlanded);
            }
            unlanded = true;
        }
    }
    // Other writers of the word sent after `done`; this table's report reductions are noted as
    // writer ranges at their own positions.
    std::array<u64, Ways> own{};
    size_t own_count = 0;
    for (size_t i = queued_from; i < count; ++i) {
        if (records[i]->writer == Writer::Report) {
            own[own_count++] = records[i]->position;
        }
    }
    const auto writers = GpuSplit::Get().QueuedGuestWriters(
        check.done, check.address, check.device, sizeof(u32),
        std::span<const u64>{own.data(), own_count});
    if (writers.noted != 0) {
        return bail(Bail::OtherWriter);
    }
    if (writers.unbounded != 0) {
        return bail(Bail::Unbounded);
    }
    // Fold in send order from the word read before `done` was loaded: records that ran but did
    // not land are left out (they only raise the word). The 2^31 guard keeps the true sum, with
    // any unlanded contribution, from wrapping.
    constexpr u64 Guard = 0x7fffffff;
    u64 bound = check.word;
    bool all_or = true;
    if (bound > Guard) {
        return bail(Bail::Wrap);
    }
    for (size_t i = queued_from; i < count; ++i) {
        const Prediction& r = *records[i];
        if (r.kind == Kind::AtLeast) {
            bound = (std::max)(bound, u64{r.payload});
            all_or = false;
        } else {
            switch (r.op) {
            case 0: // ADD
                bound += r.payload;
                all_or = false;
                break;
            case 2: // MAX
                if (r.is_signed) {
                    return bail(Bail::SignedMax);
                }
                bound = (std::max)(bound, u64{r.payload});
                all_or = false;
                break;
            case 3: // INC: value >= payload ? 0 : value + 1
                if (r.payload != 0xffffffff) {
                    return bail(Bail::IncPayload);
                }
                bound += 1;
                all_or = false;
                break;
            case 6: // OR
                bound |= r.payload;
                break;
            default: // MIN, DEC, AND, XOR
                return bail(Bail::NonMonotonic);
            }
        }
        if (bound > Guard) {
            return bail(Bail::Wrap);
        }
    }
    const u32 value = static_cast<u32>(bound);
    Outcome outcome = Outcome::OtherOp;
    switch (compare) {
    case Compare::Gequal:
        outcome = value >= check.value ? Outcome::GeqMet : Outcome::GeqShort;
        break;
    case Compare::Mask:
        outcome = !all_or                       ? Outcome::MaskNotOr
                  : (value & check.value) != 0 ? Outcome::MaskMet
                                               : Outcome::MaskShort;
        break;
    case Compare::Equal: {
        // Exact only with every earlier record landed and the word unchanged since the first
        // queued one was sent (a landed-count collision can fake "landed"). It also relies on
        // fences running their callbacks in registration order.
        const bool exact = !unlanded && records[queued_from]->before == check.word;
        outcome = !exact                 ? Outcome::EqNotExact
                  : value == check.value ? Outcome::EqMet
                                         : Outcome::EqDiffers;
        break;
    }
    default:
        break;
    }
    ++state.outcomes[static_cast<size_t>(outcome)];
    const Prediction& newest = *records[count - 1];
    return Bound{value, check.device, newest.landed_base, newest.landed_sent};
}

void CheckBound(const Bound& bound, u32 word) {
    auto& state = Get();
    std::scoped_lock lock{state.mutex};
    const u32 landed = State::LandedCount(bound.device);
    if (landed - bound.landed_base < bound.landed_sent) {
        ++state.bound_unlanded;
        return;
    }
    ++state.bound_checked;
    if (word >= bound.value) {
        return;
    }
    ++state.bound_violated;
    if (state.violated_logged < 8) {
        ++state.violated_logged;
        LOG_WARNING(HW_GPU,
                    "QMD release prediction (EXP13=2): reduction bound violated at device {:#x}: "
                    "bound {:#x}, word {:#x} once met",
                    bound.device, bound.value, word);
    }
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
