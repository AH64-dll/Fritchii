// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <optional>
#include <string>

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
/// 3: as 2, and the predictions meet acquires (fable_answer_6 6b slice B, 6c/6e): a queued (or
///    run but not landed) plain or max release whose value meets the acquire with nothing queued
///    after it that may write the word, or a reduction lower bound (GEQ only). The feed turns
///    itself off for the rest of the session (until EXP13 leaves 3) at the first bound
///    violation, payload mismatch, unpredicted back release or reduction, phantom record or
///    lost event, with an ERROR line saying why.
/// The tables are the front's own (no lock): the back's releases and the landings reach them
/// through rings the front drains.
/// One log line every 5 s (HW_GPU, Info), three with 2, four with 3.
namespace Tegra {
enum class Nx2Landing : u32; // nx2.h
}
namespace VideoCommon {
enum class GuestWriterCause : u8; // gpu_split.h
}

namespace VideoCommon::QmdPrediction {

[[nodiscard]] inline int Mode() {
    return VideoCore::Nx2Debug::Experiment(13);
}

[[nodiscard]] inline bool Enabled() {
    return Mode() != 0;
}

/// EXP13=2, 3 or 4: the front's QMD scheduler copy predicts.
[[nodiscard]] inline bool MirrorMode() {
    const int mode = Mode();
    return mode >= 2 && mode <= 4;
}

/// EXP13=3 (the narrow feed: records of QMDs a schedule runs at once) or 4 (the full feed): the
/// predictions meet acquires (while the feed has not turned itself off).
[[nodiscard]] inline bool FeedRequested() {
    const int mode = Mode();
    return mode == 3 || mode == 4;
}

/// EXP13=2 records only: Report, a compute report semaphore (0x6c3) or a 3D payload report;
/// Puller, a puller semaphore release.
enum class Writer : u32 { Qmd, Dma, Report, Puller, Count };

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
/// EXP13=2/3: where the QMD copy read the QMD a record's release came from (fable_answer_6 6f).
enum class Source : u8 {
    Other,             ///< Not from a QMD (copy, puller and report releases: the front's own)
    InlineOverlay,     ///< The words a 0x107 sent
    PatchedOverlay,    ///< The same, with inline uploads patched over them
    OverlayFromMemory, ///< Guest memory with a small upload patched over it
    Shadow,            ///< An upload shadow's block
    Memory,            ///< Guest memory, nothing queued writing it
    Waited,            ///< Guest memory after waiting for the work writing it
    Count,
};
struct Record {
    Writer writer;
    Kind kind;
    u32 payload;
    u32 op;         ///< Reduction: ReduceSemaphore's op (0 add .. 7 xor)
    bool is_signed; ///< Reduction: the signed format (QMD release flags bit 24)
    /// The QMD read it came from: the source, the back's position then, and the inline
    /// overlay's and upload shadow block's positions (0 for none).
    Source source;
    u64 read_done;
    u64 overlay_position;
    u64 block_position;
    /// Made by a schedule (action 2 or 3) of the signal's own QMD that ran at once: no base
    /// dependency count, no deferred decrements, not a chained dependent or a decrement's run
    /// (fable_answer_6 6g (c)). The narrow feed (EXP13=3) only takes these (and non-QMD
    /// releases).
    bool simple;
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
    Patched,      ///< An inline QMD's overlay took an inline upload's words over it
    Shadow,       ///< A QMD read from an upload shadow (a compute inline upload's bytes)
    ShadowCreated,   ///< A known linear upload of 256 bytes to 4 KB got a shadow
    ShadowEvicted,   ///< ... replacing the least recently used one
    ShadowTooBig,    ///< A linear upload over 4 KB: no shadow
    ShadowNotLinear, ///< An upload of 256 bytes or more not linear and back to back: no shadow
    ShadowPatched,   ///< A shadow took a later upload's words over it
    OverlayFromMemory, ///< A small upload patched a QMD guest memory held: overlay from both
    Count,
};
void NoteMirror(Mirror event);
/// Front, EXP13=2: a QMD read waited; `cause` set its target (`what`: an unknown noted write's).
void NoteWait(GuestWriterCause cause, const char* what);
/// Front, EXP13=2: a QMD read waited on a writer range of `kind` (GpuSplit::WriterKind) that
/// starts at byte `offset` of the QMD (negative: before it) and is `size` bytes long.
void NoteRangeWait(u32 kind, s64 offset, u64 size);
/// Any thread, EXP13=2: a GPU write of `value` is about to land in the u32 at `device` through
/// `site` (called before the word is written): marks the record it belongs to, by kind (a
/// reduction the oldest unlanded reduction, a plain write the oldest unlanded plain record of
/// that payload, a max the oldest unlanded max release with a payload up to it).
void NoteLanded(u64 device, u32 value, Tegra::Nx2Landing site);
/// Front, EXP13=2: the records are complete for the work sent from `position` on (the copy just
/// started over), or not any more (`valid` false: it stopped).
void SetRecordsValid(bool valid, u64 position);
/// Front, EXP13=2-4: the copy recorded every release of the work sent up to `position` (the end
/// of the signal it just ran). Record first, compare after (6g (a)): the back's releases of
/// entries after the previous one that found no record wait for this, and the phantom sweep
/// runs here.
void Settle(u64 position);
/// Front, EXP13=2-4: where the copy's latest QMD read came from (for the unpredicted lines).
void NoteReadSource(Source source);
/// The copy's scheduler table state for a QMD, as text (the unpredicted lines).
using QmdProbe = std::string (*)(u64 qmd);
void SetQmdProbe(QmdProbe probe);

/// Back: the release just made by the entry at `entry_position`, of the QMD at `qmd`.
/// `slots`: the QMD's three release slots' flags and payload words as the back read them
/// (words 25, 26, 29, 30, 53, 54; null for copies).
void Verify(Writer writer, u64 device, u32 payload, u64 qmd, u64 entry_position, bool reduction,
            const u32* slots = nullptr);

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
    OtherRelease,  ///< EXP13=2: the newest release recorded is a puller or report one
    Count,
};
/// Front: an unmet readable acquire before its fence release.
void NoteAcquire(Coverage coverage);

/// Front, EXP13=2: an unmet readable acquire of `operation` (Nx2SemaphoreWaiters' values) and
/// `value` on the u32 at `device` (GPU `address`). `word` was read from guest memory BEFORE
/// `done` (the back's position) was loaded; `gpu_dirty`: the word's page is GPU-written;
/// `pointer`: the word's host memory, for late re-reads (may be null).
struct AcquireCheck {
    u64 device;
    u64 address;
    u32 operation;
    u32 value;
    u32 word;
    u64 done;
    bool gpu_dirty;
    const u8* pointer;
};
/// The reduction lower bound computed for an acquire, for the safety check once it is met.
struct Bound {
    u32 value; ///< The bound
    u32 word;  ///< The word it was folded from
    u64 device;
    u64 address;
    u64 done;
    std::array<u64, 32> sequences; ///< The folded records (all landed: the check is meaningful)
    u64 payload_sum;
    std::chrono::steady_clock::time_point since;
    const u8* pointer;
    u32 operation;
    u32 acquire_value;
    u32 folded;
    std::array<u32, 32> payloads; ///< The folded records' payloads, in send order
};
/// Counts the acquire's coverage (NoteAcquire) and its reduction lower bound (6c): the bound
/// when one was computed.
std::optional<Bound> CheckAcquire(const AcquireCheck& check);
/// Front: the acquire `bound` was computed for was met; `word` is its word now. Counts a bound
/// violation (word < bound) once the records folded landed.
void CheckBound(const Bound& bound, u32 word);

/// Front, EXP13=3: whether the predictions meet the unmet acquire `check` describes (`resume`:
/// a poll of a suspended one). The caller notes a host sync (GpuSplit::NoteSync) when met.
[[nodiscard]] bool Feed(const AcquireCheck& check, bool resume);

/// The count of releases the back made (Verify), to tell which came after a point.
[[nodiscard]] u64 BackSequence();
/// The newest release the back made for `device` after BackSequence() returned `since`.
struct BackRelease {
    Writer writer;
    u32 payload;
    bool reduction;
};
[[nodiscard]] std::optional<BackRelease> BackReleaseSince(u64 device, u64 since);
/// The newest release the back made for `device` (as far as its table remembers).
[[nodiscard]] std::optional<BackRelease> BackReleaseLatest(u64 device);

/// BeforeCheck counts on top: of the met ones, those whose release the back made before the
/// check (it landed after it).
enum class MetBy : u32 { QmdRelease, QmdReduction, DmaRelease, Other, Unmet, BeforeCheck, Count };
/// Front: what met an acquire NoteAcquire counted, once its fences were released.
void NoteMetBy(MetBy met_by);

} // namespace VideoCommon::QmdPrediction
