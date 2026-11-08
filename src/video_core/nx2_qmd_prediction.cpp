// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <span>
#include <string>
#include <utility>

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
    Writer writer;
    Kind kind;
    u8 op;
    bool is_signed;
    bool stale;
    bool landed;  ///< EXP13=2: its write landed (NoteLanded, marked before the word is written)
    bool matched; ///< EXP13=2: the back made it (Verify)
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

/// A bound violation (fable_answer_6 6d): the word went below the word folded from (an
/// unrecorded writer: a plain release or the CPU), stayed short by the last records' payloads
/// (their landings missing), or anything else.
enum class Violation : u32 { WentDown, LandingsMissing, Other, Count };

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
    u64 foreign_landings{};
    u32 foreign_logged{};
    /// The newest plain record evicted before it landed, per bucket (its device; 0 for none):
    /// the word gets no bound until a plain landing there is attributed to it.
    std::array<u64, Buckets> evicted_plain{};
    u64 evicted_unlanded{};
    /// Back: reductions the copy predicted (Verify matches them too).
    u64 matched_reductions{};
    u64 differs_reductions{};
    u64 unpredicted_reductions{};
    u32 unpredicted_reductions_logged{};
    /// Records the back never made, swept once it ran well past them (Note).
    u64 phantoms{};
    u32 phantoms_logged{};
    size_t sweep_bucket{};
    /// Whole-session totals of the safety check (never reset by the 5 s report).
    u64 session_checked{};
    u64 session_violated{};
    u64 session_unlanded{};
    u64 session_foreign{};
    u64 session_phantoms{};
    /// Waits on writer ranges, by (kind, word offset in the QMD, size).
    struct RangeWait {
        u32 kind;
        s64 word;
        u64 size;
        u64 count;
    };
    std::array<RangeWait, 24> range_waits{};
    size_t range_wait_count{};
    u64 range_waits_other{};
    std::array<u64, static_cast<size_t>(GuestWriterCause::Count)> wait_causes{};
    std::array<std::pair<const char*, u64>, 8> wait_whats{};
    u64 wait_whats_other{};
    std::array<u64, static_cast<size_t>(Violation::Count)> violations{};
    std::array<u64, static_cast<size_t>(Violation::Count)> recovered{};
    std::array<u64, static_cast<size_t>(Violation::Count)> never_recovered{};
    /// Violations re-read later (at each Tick and the next CheckBound on the device).
    struct PendingViolation {
        u64 device;
        const u8* pointer;
        u32 bound;
        Violation kind;
        Clock::time_point since;
    };
    std::array<PendingViolation, 64> pending{};
    size_t pending_count{};
    Clock::time_point since{};

    /// Re-reads the pending violations (of `device` only, if given): recovered once the word
    /// reached the bound, never recovered after 100 ms.
    void ReRead(Clock::time_point now, const u64* device) {
        for (size_t i = 0; i < pending_count;) {
            PendingViolation& entry = pending[i];
            if (device != nullptr && entry.device != *device) {
                ++i;
                continue;
            }
            u32 current{};
            std::memcpy(&current, entry.pointer, sizeof(current));
            const size_t kind = static_cast<size_t>(entry.kind);
            if (current >= entry.bound) {
                ++recovered[kind];
            } else if (now - entry.since >= std::chrono::milliseconds{100}) {
                ++never_recovered[kind];
            } else {
                ++i;
                continue;
            }
            entry = pending[--pending_count];
        }
    }

    void Tick() {
        const auto now = Clock::now();
        if (pending_count != 0) {
            ReRead(now, nullptr);
        }
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
                 "unpredicted ({:.0f} table full), {:.0f} reductions ({:.0f} matched, {:.0f} "
                 "payload differs, {:.0f} unpredicted); back copies: {:.0f} "
                 "matched, {:.0f} payload differs, {:.0f} unpredicted; unmet acquires: {:.0f} "
                 "would be met by a predicted queued release, {:.0f} wrong value, {:.0f} already "
                 "ran ({:.0f} landed pending, {:.0f} really wrong), {:.0f} no prediction, {:.0f} "
                 "newest release a puller or report one; met after fences by: {:.0f} QMD "
                 "release, {:.0f} QMD reduction, {:.0f} copy semaphore, {:.0f} other writer "
                 "({:.0f} of the met by a release run before the check), {:.0f} still unmet; "
                 "QMD copy: QMD reads/s "
                 "{:.0f} from inline QMDs, {:.0f} from memory, {:.0f} after a wait; {:.0f} "
                 "inline QMDs patched by uploads/s; {:.0f} resyncs/s, {:.0f} starts, {:.0f} "
                 "auto-off; reduction bound would meet: {:.0f} "
                 "GEQ, {:.0f} EQ, {:.0f} MASK",
                 Mode(), signal(Signal::Read), signal(Signal::Waiting), signal(Signal::Chained),
                 signal(Signal::Unmapped), rate(predicted[qmd]), rate(predicted_at_least),
                 rate(predicted_stale), rate(reductions), rate(unmapped_releases),
                 rate(predicted[dma]), rate(matched[qmd]), rate(differs[qmd]),
                 rate(differs_stale), rate(unpredicted[qmd]), rate(unpredicted_full),
                 rate(back_reductions), rate(matched_reductions), rate(differs_reductions),
                 rate(unpredicted_reductions), rate(matched[dma]), rate(differs[dma]),
                 rate(unpredicted[dma]), covered(Coverage::WouldMeet),
                 covered(Coverage::WrongValue), covered(Coverage::AlreadyRan),
                 covered(Coverage::LandedPending), covered(Coverage::ReallyWrong),
                 covered(Coverage::None), covered(Coverage::OtherRelease),
                 met(MetBy::QmdRelease), met(MetBy::QmdReduction), met(MetBy::DmaRelease),
                 met(MetBy::Other), met(MetBy::BeforeCheck), met(MetBy::Unmet),
                 mirrored(Mirror::Overlay), mirrored(Mirror::Memory), mirrored(Mirror::Waited),
                 mirrored(Mirror::Patched), mirrored(Mirror::Resync), mirrored(Mirror::SessionStart),
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
                     "landed yet; violations: {} went down, {} landings missing, {} other; "
                     "re-read later: {}/{}/{} recovered, {}/{}/{} never recovered in 100 ms (went "
                     "down/landings missing/other); {} landings of unrecorded writes, {} "
                     "phantom records swept, {} records evicted before landing; session totals: "
                     "{} checked, {} violated, {} not landed yet, {} landings of unrecorded "
                     "writes, {} phantoms; records/s "
                     "by op add/min/max/inc/dec/and/or/xor: QMD {}, QMD signed {}, compute report "
                     "{}",
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
                     bound_checked, bound_violated, bound_unlanded, violations[0],
                     violations[1], violations[2], recovered[0], recovered[1], recovered[2],
                     never_recovered[0], never_recovered[1], never_recovered[2],
                     foreign_landings, phantoms, evicted_unlanded, session_checked,
                     session_violated, session_unlanded, session_foreign, session_phantoms,
                     Rates(reduction_ops[0], window),
                     Rates(reduction_ops[1], window), Rates(reduction_ops[2], window));
            const auto waited = [&](GuestWriterCause c) {
                return rate(wait_causes[static_cast<size_t>(c)]);
            };
            std::string whats;
            for (const auto& [what, count] : wait_whats) {
                if (what != nullptr && count != 0) {
                    whats += fmt::format("{}{} {:.0f}", whats.empty() ? "" : ", ", what,
                                         rate(count));
                }
            }
            if (wait_whats_other != 0) {
                whats += fmt::format("{}others {:.0f}", whats.empty() ? "" : ", ",
                                     rate(wait_whats_other));
            }
            // GpuSplit::WriterKind's order.
            constexpr std::array<const char*, 5> range_kinds{"upload", "release", "compute",
                                                             "copy", "other"};
            std::string ranges;
            for (size_t i = 0; i < range_wait_count; ++i) {
                const RangeWait& entry = range_waits[i];
                ranges += fmt::format("{}{} @{} x{} {:.0f}", ranges.empty() ? "" : ", ",
                                      entry.kind < range_kinds.size() ? range_kinds[entry.kind]
                                                                      : "?",
                                      entry.word, entry.size, rate(entry.count));
            }
            if (range_waits_other != 0) {
                ranges += fmt::format("{}others {:.0f}", ranges.empty() ? "" : ", ",
                                      rate(range_waits_other));
            }
            LOG_INFO(HW_GPU,
                     "QMD release prediction (EXP13=2) QMD reads that waited, by what set the "
                     "target (/s): {:.0f} noted write, {:.0f} writer range, {:.0f} 3D ranged "
                     "write, {:.0f} unbounded compute/copy writer, {:.0f} unranged writer, {:.0f} "
                     "queued function, {:.0f} 3D method write, {:.0f} unknown noted write ({}), "
                     "{:.0f} noted-write scan limit, {:.0f} writer-range scan limit, {:.0f} "
                     "evicted inline QMD, {:.0f} none; writer ranges waited on by (kind, word "
                     "in the QMD, bytes): {}",
                     waited(GuestWriterCause::Pending), waited(GuestWriterCause::WriterRange),
                     waited(GuestWriterCause::Ranged), waited(GuestWriterCause::UnboundedWriter),
                     waited(GuestWriterCause::Unranged), waited(GuestWriterCause::Function),
                     waited(GuestWriterCause::MethodWrite),
                     waited(GuestWriterCause::UnknownPending), whats,
                     waited(GuestWriterCause::PendingScanLimit),
                     waited(GuestWriterCause::WriterScanLimit),
                     waited(GuestWriterCause::OverlayEvicted), waited(GuestWriterCause::None),
                     ranges);
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
        foreign_landings = 0;
        evicted_unlanded = 0;
        matched_reductions = differs_reductions = unpredicted_reductions = 0;
        phantoms = 0;
        range_wait_count = 0;
        range_waits_other = 0;
        wait_causes = {};
        wait_whats = {};
        wait_whats_other = 0;
        violations = recovered = never_recovered = {};
        since = now;
    }

    /// Stores a record (not landed, not matched yet).
    void Insert(const Prediction& record) {
        const size_t bucket = Bucket(record.device);
        auto& ways = predictions[bucket];
        Prediction stored = record;
        stored.sequence = ++record_sequence;
        stored.landed = false;
        stored.matched = false;
        Prediction& slot = ways[next[bucket]++ % Ways];
        if (slot.position != 0) {
            evicted[bucket] = (std::max)(evicted[bucket], slot.position);
            if (!slot.landed) {
                ++evicted_unlanded;
                if (slot.kind == Kind::Plain) {
                    evicted_plain[bucket] = slot.device;
                }
            }
        }
        slot = stored;
    }

    [[nodiscard]] static bool Landed(const Prediction& record) {
        return record.landed;
    }

    /// The records of `device` in the table and how many of them landed.
    [[nodiscard]] std::pair<u32, u32> Counts(u64 device) const {
        std::pair<u32, u32> counts{};
        for (const Prediction& way : predictions[Bucket(device)]) {
            if (way.position != 0 && way.device == device) {
                ++counts.first;
                counts.second += way.landed ? 1 : 0;
            }
        }
        return counts;
    }

    /// Drops the records of `device`.
    void ClearWord(u64 device) {
        const size_t bucket = Bucket(device);
        for (Prediction& way : predictions[bucket]) {
            if (way.device == device) {
                way = {};
            }
        }
        if (evicted_plain[bucket] == device) {
            evicted_plain[bucket] = 0;
        }
    }

    /// Records of the writers the back checks (QMD and copy releases) that it never made, once
    /// it ran well past them (`consumed` is published every 4096 words at most): phantoms. A few
    /// buckets per call. Their words start over.
    void Sweep(u64 consumed) {
        if (!records_valid || consumed < 4096) {
            return;
        }
        for (size_t step = 0; step < 4; ++step) {
            const size_t bucket = sweep_bucket++ % Buckets;
            for (Prediction& way : predictions[bucket]) {
                if (way.position == 0 || way.matched || way.position + 4096 > consumed ||
                    way.position < records_valid_from ||
                    (way.writer != Writer::Qmd && way.writer != Writer::Dma)) {
                    continue;
                }
                ++phantoms;
                ++session_phantoms;
                if (phantoms_logged < 16) {
                    ++phantoms_logged;
                    LOG_WARNING(HW_GPU,
                                "QMD release prediction (EXP13=2): phantom {} record (the back "
                                "never made it): QMD {:#x}, device {:#x}, kind {}, payload {:#x}, "
                                "sent at {}, back at {}",
                                way.writer == Writer::Qmd ? "QMD" : "copy", way.qmd, way.device,
                                static_cast<u32>(way.kind), way.payload, way.position, consumed);
                }
                ClearWord(way.device);
            }
        }
    }

    /// Drops the records.
    void Clear() {
        predictions = {};
        next = {};
        evicted = {};
        evicted_plain = {};
        pending_count = 0;
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
    static const bool hooked = [] {
        Tegra::g_nx2_landed_hook.store(&NoteLanded, std::memory_order_release);
        return true;
    }();
    (void)hooked;
    state.Insert(stored);
    state.Sweep(GpuSplit::Get().ConsumedNow());
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
    if (!valid) {
        // The copy stopped: its records and their landed counts start over with it.
        state.Clear();
    }
}

void NoteWait(GuestWriterCause cause, const char* what) {
    auto& state = Get();
    std::scoped_lock lock{state.mutex};
    ++state.wait_causes[static_cast<size_t>(cause)];
    if (cause != GuestWriterCause::UnknownPending) {
        return;
    }
    for (auto& [name, count] : state.wait_whats) {
        if (name == nullptr || name == what ||
            (what != nullptr && std::strcmp(name, what) == 0)) {
            name = what != nullptr ? what : "unknown";
            ++count;
            return;
        }
    }
    ++state.wait_whats_other;
}

void NoteRangeWait(u32 kind, s64 offset, u64 size) {
    auto& state = Get();
    std::scoped_lock lock{state.mutex};
    const s64 word = offset >= 0 ? offset / 4 : -((-offset + 3) / 4);
    for (size_t i = 0; i < state.range_wait_count; ++i) {
        State::RangeWait& entry = state.range_waits[i];
        if (entry.kind == kind && entry.word == word && entry.size == size) {
            ++entry.count;
            return;
        }
    }
    if (state.range_wait_count < state.range_waits.size()) {
        state.range_waits[state.range_wait_count++] = {kind, word, size, 1};
    } else {
        ++state.range_waits_other;
    }
}

void NoteLanded(u64 device, u32 value, Tegra::Nx2Landing kind) {
    if (!MirrorMode()) {
        return;
    }
    auto& state = Get();
    std::scoped_lock lock{state.mutex};
    // The record it belongs to: reductions land in send order among themselves (synchronously,
    // in back order), a plain write is known by its payload, a max by a payload up to it. Kinds
    // land out of order with each other (query completions run on the fencing thread).
    Prediction* oldest = nullptr;
    bool any = false;
    const size_t bucket = Bucket(device);
    for (Prediction& way : state.predictions[bucket]) {
        if (way.position == 0 || way.device != device) {
            continue;
        }
        any = true;
        if (way.landed) {
            continue;
        }
        const bool fits = kind == Tegra::Nx2Landing::Reduction ? way.kind == Kind::Reduction
                          : kind == Tegra::Nx2Landing::Max
                              ? way.kind == Kind::AtLeast && way.payload <= value
                              : way.kind == Kind::Plain && way.payload == value;
        if (fits && (oldest == nullptr || way.sequence < oldest->sequence)) {
            oldest = &way;
        }
    }
    if (oldest != nullptr) {
        oldest->landed = true;
        return;
    }
    if (kind == Tegra::Nx2Landing::Plain && state.evicted_plain[bucket] == device) {
        state.evicted_plain[bucket] = 0; // The evicted plain record's, most likely
        return;
    }
    if (!any) {
        return; // A word without records (counter reports, CPU-polled words)
    }
    ++state.foreign_landings;
    ++state.session_foreign;
    if (state.foreign_logged < 8) {
        ++state.foreign_logged;
        std::string kinds;
        for (const Prediction& way : state.predictions[bucket]) {
            if (way.position != 0 && way.device == device) {
                kinds += fmt::format(" {}{}={:#x}{}",
                                     way.kind == Kind::Plain     ? "plain"
                                     : way.kind == Kind::AtLeast ? "max"
                                                                 : "reduction",
                                     way.writer == Writer::Qmd      ? ""
                                     : way.writer == Writer::Dma    ? "(copy)"
                                     : way.writer == Writer::Puller ? "(puller)"
                                                                    : "(report)",
                                     way.payload, way.landed ? " landed" : "");
            }
        }
        LOG_INFO(HW_GPU,
                 "QMD release prediction (EXP13=2): a landing no record fits: device {:#x}, "
                 "{} {:#x}; the word's records:{}",
                 device,
                 kind == Tegra::Nx2Landing::Plain ? "plain"
                 : kind == Tegra::Nx2Landing::Max ? "max"
                                                  : "reduction",
                 value, kinds);
    }
}

void Verify(Writer writer, u64 device, u32 payload, u64 qmd, u64 entry_position, bool reduction) {
    auto& state = Get();
    std::scoped_lock lock{state.mutex};
    state.releases[Hash(device) % Releases] = {device, ++state.sequence, payload, writer,
                                                 reduction};
    const bool mirror = MirrorMode();
    if (reduction) {
        ++state.back_reductions;
        if (!mirror) {
            return;
        }
    }
    // The launch the back runs is the first one sent after the entry it runs starts. With
    // EXP13=2 each record is the prediction of one release (matched once).
    auto& ways = state.predictions[Bucket(device)];
    Prediction* found = nullptr;
    u32 same_device = 0;
    u64 oldest = ~u64{0};
    for (Prediction& way : ways) {
        oldest = (std::min)(oldest, way.position);
        if (way.position <= entry_position || way.device != device || way.writer != writer ||
            (way.kind == Kind::Reduction) != reduction || (mirror && way.matched)) {
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
    if (reduction) {
        if (!found) {
            ++state.unpredicted_reductions;
            if (state.unpredicted_reductions_logged < 8) {
                ++state.unpredicted_reductions_logged;
                LOG_INFO(HW_GPU,
                         "QMD release prediction (EXP13=2): unpredicted QMD reduction, QMD "
                         "{:#x}, device {:#x} payload {:#x}, entry {}; queued predictions there "
                         "from other QMDs: {}",
                         qmd, device, payload, entry_position, same_device);
            }
        } else {
            found->matched = true;
            ++(found->payload == payload ? state.matched_reductions : state.differs_reductions);
        }
        return;
    }
    if (found) {
        found->matched = true;
    }
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

    // Coverage by the plain predictions: the newest release recorded for the word decides.
    const auto would_meet = [&](const Prediction& r) {
        return r.kind == Kind::AtLeast
                   ? compare == Compare::Gequal && check.value <= r.payload
                   : Meets(compare, check.value, r.payload);
    };
    Coverage coverage = Coverage::None;
    if (count != 0 && (records[count - 1]->writer == Writer::Puller ||
                       (records[count - 1]->writer == Writer::Report &&
                        records[count - 1]->kind == Kind::Plain))) {
        coverage = Coverage::OtherRelease;
    } else if (count != 0 && records[count - 1]->kind != Kind::Reduction) {
        const Prediction& newest = *records[count - 1];
        if (newest.position > check.done) {
            coverage = would_meet(newest) ? Coverage::WouldMeet : Coverage::WrongValue;
        } else if (!state.Landed(newest) && would_meet(newest) &&
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

    // The reduction lower bound (6c section 3, 6e): the word only grows under the records that
    // have not landed, whatever their position (`done` trails the back: a record past it may
    // be inside the word already, and its mark says so; the back marks before it writes).
    const auto bail = [&](Bail reason) -> std::optional<Bound> {
        ++state.bails[static_cast<size_t>(reason)];
        return std::nullopt;
    };
    bool any_monotonic_record = false;
    bool plain_queued = false;
    bool plain_unlanded = false;
    for (size_t i = 0; i < count; ++i) {
        if (State::Landed(*records[i])) {
            continue;
        }
        any_monotonic_record |= records[i]->kind != Kind::Plain;
        if (records[i]->kind == Kind::Plain) {
            (records[i]->position > check.done ? plain_queued : plain_unlanded) = true;
        }
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
    if (state.evicted[bucket] > check.done || state.evicted_plain[bucket] == check.device) {
        return bail(Bail::Evicted);
    }
    if (plain_queued) {
        return bail(Bail::PlainQueued);
    }
    if (plain_unlanded) {
        // It lands later and sets the word, maybe below the bound.
        return bail(Bail::PlainUnlanded);
    }
    // Other writers of the word sent after `done`; this table's report releases are noted as
    // writer ranges at their own positions.
    std::array<u64, Ways> own{};
    size_t own_count = 0;
    for (size_t i = 0; i < count; ++i) {
        if (records[i]->writer == Writer::Report && records[i]->position > check.done) {
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
    // Fold the unlanded records in send order from the word read before the marks (outside the
    // lock: a record marked after the read is folded, one marked before is in the word). The
    // 2^31 guard keeps the sum from wrapping.
    constexpr u64 Guard = 0x7fffffff;
    u64 bound = check.word;
    bool all_or = true;
    if (bound > Guard) {
        return bail(Bail::Wrap);
    }
    Bound result{};
    for (size_t i = 0; i < count; ++i) {
        const Prediction& r = *records[i];
        if (State::Landed(r)) {
            continue;
        }
        result.sequences[result.folded] = r.sequence;
        result.payloads[result.folded++] = r.payload;
        result.payload_sum += r.payload;
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
        // With the marks every write still to land is folded (plain ones bailed above): the
        // fold of monotonic ops is the word's final value, unless an unrecorded writer joins.
        outcome = value == check.value ? Outcome::EqMet : Outcome::EqDiffers;
        break;
    }
    default:
        break;
    }
    ++state.outcomes[static_cast<size_t>(outcome)];
    result.value = value;
    result.word = check.word;
    result.device = check.device;
    result.address = check.address;
    result.done = check.done;
    result.since = Clock::now();
    result.pointer = check.pointer;
    result.operation = check.operation;
    result.acquire_value = check.value;
    return result;
}

void CheckBound(const Bound& bound, u32 word) {
    auto& state = Get();
    std::scoped_lock lock{state.mutex};
    const auto now = Clock::now();
    if (state.pending_count != 0) {
        state.ReRead(now, &bound.device);
    }
    // Meaningful once every folded record landed (a record dropped since counts as not).
    u32 folded_landed = 0;
    for (u32 i = 0; i < bound.folded; ++i) {
        for (const Prediction& way : state.predictions[Bucket(bound.device)]) {
            if (way.position != 0 && way.sequence == bound.sequences[i]) {
                folded_landed += way.landed ? 1 : 0;
                break;
            }
        }
    }
    if (folded_landed < bound.folded) {
        ++state.bound_unlanded;
        ++state.session_unlanded;
        return;
    }
    ++state.bound_checked;
    ++state.session_checked;
    if (word >= bound.value) {
        return;
    }
    ++state.bound_violated;
    ++state.session_violated;
    const auto [word_records, word_landed] = state.Counts(bound.device);
    const u32 deficit = bound.value - word;
    Violation kind = Violation::Other;
    if (word < bound.word) {
        kind = Violation::WentDown;
    } else {
        // Short by the payloads of the last j records folded: their landings are missing.
        u64 sum = 0;
        for (u32 j = bound.folded; j > 0; --j) {
            sum += bound.payloads[j - 1];
            if (sum == deficit) {
                kind = Violation::LandingsMissing;
                break;
            }
        }
    }
    ++state.violations[static_cast<size_t>(kind)];
    if (state.violated_logged < 32) {
        ++state.violated_logged;
        constexpr std::array<const char*, 3> kinds{"went down", "landings missing", "other"};
        LOG_WARNING(HW_GPU,
                    "QMD release prediction (EXP13=2): reduction bound violated ({}) at {:#x} "
                    "(device {:#x}): acquire op {} value {:#x}; folded from {:#x} at back "
                    "position {}: bound {:#x}, word {:#x} once met, deficit {:#x}; {} records "
                    "folded, payloads summing to {:#x}; records of the word: {}, {} landed; {} "
                    "us after the acquire",
                    kinds[static_cast<size_t>(kind)], bound.address, bound.device,
                    bound.operation, bound.acquire_value, bound.word, bound.done, bound.value,
                    word, deficit, bound.folded, bound.payload_sum, word_records, word_landed,
                    std::chrono::duration_cast<std::chrono::microseconds>(now - bound.since)
                        .count());
    }
    if (bound.pointer != nullptr && state.pending_count < state.pending.size()) {
        state.pending[state.pending_count++] = {bound.device, bound.pointer, bound.value, kind,
                                                now};
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

std::optional<BackRelease> BackReleaseLatest(u64 device) {
    auto& state = Get();
    std::scoped_lock lock{state.mutex};
    const Release& release = state.releases[Hash(device) % Releases];
    if (release.device != device || release.sequence == 0) {
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
