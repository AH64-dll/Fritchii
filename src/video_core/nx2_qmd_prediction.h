// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <optional>

#include "common/common_types.h"
#include "video_core/nx2_debug_control.h"

/// EDEN_NX2_EXP13 (live, diagnostic only): whether the split's front could know the payloads of
/// native QMD completion releases and copy engine semaphores before the back runs them. The front
/// predicts them, the back checks each release it makes against the prediction, and the front's
/// unmet readable acquires ask whether a queued prediction would have met them. Nothing feeds
/// GpuSplit::queued_releases and no acquire is treated as met: counting only.
/// 1: predicted at every signal that reads a QMD, from guest memory (the first gate run).
/// 2: the front runs its own copy of the QMD scheduler (KeplerCompute's QMD mirror, from
///    zortos293/Lichii#717), so each release is predicted once, where the back makes it, from
///    the inline QMD words the front sent; max releases (bit 19) are lower bounds, reductions
///    (bit 26) and compute report reductions are recorded for the reduction lower bound
///    (fable_answer_6 6c), which is counted too.
/// One log line every 5 s (HW_GPU, Info), two with 2.
namespace VideoCommon::QmdPrediction {

[[nodiscard]] inline int Mode() {
    return VideoCore::Nx2Debug::Experiment(13);
}

[[nodiscard]] inline bool Enabled() {
    return Mode() != 0;
}

/// EXP13=2: the front's QMD scheduler copy predicts.
[[nodiscard]] inline bool MirrorMode() {
    return Mode() == 2;
}

/// Report: a compute report semaphore reduction (0x6c3 with bit 3; EXP13=2 records only).
enum class Writer : u32 { Qmd, Dma, Report, Count };

/// Front: how a QMD signal was read (Predict follows for its releases).
enum class Signal : u32 {
    Read,      ///< A schedule or decrement whose QMD was read
    Waiting,   ///< A schedule of a QMD with a dependency count
    Chained,   ///< The dependent QMD of one read, read too
    Unmapped,  ///< Its QMD is not mapped
    Count,
};
void NoteSignal(Signal signal);
/// Front: a QMD release slot that was not predicted (reduction, unmapped destination).
void NoteReduction();
void NoteUnmappedRelease();
/// Front: the release of `writer` sent at ring `position` writes `payload` at `device`. `qmd` is
/// the QMD's GPU address (0 for copies), `stale`: queued work may still write the QMD.
void Predict(Writer writer, u64 device, u32 payload, u64 position, u64 qmd, bool stale);

/// EXP13=2: how a recorded release changes its word.
enum class Kind : u8 {
    Plain,     ///< Writes the payload
    AtLeast,   ///< A QMD max release (bit 19): max(word, payload), at a fence
    Reduction, ///< ReduceSemaphore(op, is_signed, word, payload), at a fence
};
struct Record {
    Writer writer;
    Kind kind;
    u32 payload;
    u32 op;         ///< Reduction: ReduceSemaphore's op (0 add .. 7 xor)
    bool is_signed; ///< Reduction: the signed format (QMD release flags bit 24)
};
/// Front, EXP13=2: a release sent at ring `position` (of the QMD at `qmd`, 0 for others) to the
/// u32 at `device`, which held `before` in guest memory then.
void Note(u64 device, const Record& record, u32 before, u64 position, u64 qmd);

/// Front, EXP13=2: what the QMD scheduler copy did.
enum class Mirror : u32 {
    Overlay,      ///< A QMD read from the inline QMD the front sent last over it
    Memory,       ///< A QMD read from guest memory, nothing queued writing it
    Waited,       ///< A QMD read from guest memory after waiting for the work writing it
    Resync,       ///< A drain and a copy of the back's table (GPU-written page, all-zero QMD)
    SessionStart, ///< The same at the first signal of a split session or of EXP13=2
    AutoOff,      ///< Turned off: too many reads waited or resynced (EDEN_NX2_QMD_AUTO_OFF)
    Count,
};
void NoteMirror(Mirror event);
/// Front, EXP13=2: the records are complete for the work sent from `position` on (the copy just
/// started over), or not any more (`valid` false: it stopped).
void SetRecordsValid(bool valid, u64 position);

/// Back: the release just made by the entry at `entry_position`, of the QMD at `qmd`.
void Verify(Writer writer, u64 device, u32 payload, u64 qmd, u64 entry_position, bool reduction);

/// Front: the payloads of predictions for `device` still queued (position > `consumed`), and
/// whether any prediction for it is known at all.
struct Queued {
    std::array<u32, 16> payloads;
    u32 count;
    bool any;
};
[[nodiscard]] Queued QueuedPayloads(u64 device, u64 consumed);

enum class Coverage : u32 {
    WouldMeet,
    WrongValue,
    AlreadyRan,
    None,
    LandedPending, ///< EXP13=2, of AlreadyRan: run, not landed, and its payload would meet
    ReallyWrong,   ///< EXP13=2, of AlreadyRan: landed, or its payload would not meet
    Count,
};
/// Front: an unmet readable acquire before its fence release.
void NoteAcquire(Coverage coverage);

/// Front, EXP13=2: an unmet readable acquire of `operation` (Nx2SemaphoreWaiters' values) and
/// `value` on the u32 at `device` (GPU `address`). `word` was read from guest memory BEFORE
/// `done` (the back's position) was loaded; `gpu_dirty`: the word's page is GPU-written.
struct AcquireCheck {
    u64 device;
    u64 address;
    u32 operation;
    u32 value;
    u32 word;
    u64 done;
    bool gpu_dirty;
};
/// The reduction lower bound computed for an acquire, for the safety check once it is met.
struct Bound {
    u32 value;
    u64 device;
    u32 landed_base; ///< Of the newest record folded: landed once the count passed base + sent
    u32 landed_sent;
};
/// Counts the acquire's coverage (NoteAcquire) and its reduction lower bound (6c): the bound
/// when one was computed.
std::optional<Bound> CheckAcquire(const AcquireCheck& check);
/// Front: the acquire `bound` was computed for was met; `word` is its word now. Counts a bound
/// violation (word < bound) once the records folded landed.
void CheckBound(const Bound& bound, u32 word);

/// The count of releases the back made (Verify), to tell which came after a point.
[[nodiscard]] u64 BackSequence();
/// The newest release the back made for `device` after BackSequence() returned `since`.
struct BackRelease {
    Writer writer;
    u32 payload;
    bool reduction;
};
[[nodiscard]] std::optional<BackRelease> BackReleaseSince(u64 device, u64 since);

enum class MetBy : u32 { QmdRelease, QmdReduction, DmaRelease, Other, Unmet, Count };
/// Front: what met an acquire NoteAcquire counted, once its fences were released.
void NoteMetBy(MetBy met_by);

} // namespace VideoCommon::QmdPrediction
