// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <optional>

#include "common/common_types.h"
#include "video_core/nx2_debug_control.h"

/// EDEN_NX2_EXP13 (live, diagnostic only): whether the split's front could know the payloads of
/// native QMD completion releases and copy engine semaphores before the back runs them. The front
/// predicts them when it sends the launch (reading the QMD from guest memory), the back checks
/// each release it makes against the prediction, and the front's unmet readable acquires ask
/// whether a queued prediction would have met them. Nothing feeds GpuSplit::queued_releases.
/// One log line every 5 s (HW_GPU, Info).
namespace VideoCommon::QmdPrediction {

enum class Writer : u32 { Qmd, Dma };

[[nodiscard]] inline bool Enabled() {
    return VideoCore::Nx2Debug::Experiment(13) != 0;
}

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

enum class Coverage : u32 { WouldMeet, WrongValue, AlreadyRan, None, Count };
/// Front: an unmet readable acquire before its fence release.
void NoteAcquire(Coverage coverage);

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
