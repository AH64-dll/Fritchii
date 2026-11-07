// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <deque>
#include <optional>
#include <string>

#include "common/common_types.h"

namespace Vulkan {

struct DisplayRefresh {
    double period_ns;
    std::string display; ///< Which display and source, for the log
};

/// The refresh period of the display a window is on (Windows; nullopt elsewhere or on failure).
/// Keeps the display's rate between queries so the display configuration is only read again
/// when the window moves to another display, its mode changes, or every 30 s.
class DisplayRefreshQuery {
public:
    /// `native_window`: the HWND on Windows; null queries the primary display.
    [[nodiscard]] std::optional<DisplayRefresh> Query(void* native_window);

private:
    std::string device;   ///< GDI name of the display the rate below belongs to
    u32 nominal_hz{};     ///< Its integer rate when the rate was read
    double rational_ns{}; ///< From the display configuration; 0: unavailable
    std::chrono::steady_clock::time_point stale_at{};
};

/// Host frame pacer (EDEN_NX2_HOST_PACE, live switch; 1 = on, the default, 0 = measure only).
///
/// A swapchain shows a frame at the first display refresh after it is ready. A 60 fps game on a
/// 120 Hz display should then show every frame for two refreshes, but when its frames reach the
/// present thread unevenly (the GPU thread's work varies frame to frame) one lands just after a
/// refresh and stays three refreshes (25 ms), the next just before one and stays one (8 ms); with
/// mailbox two frames ready within one refresh drop the first. The pacer learns when each frame
/// was shown (VK_KHR_present_wait) and the display's refresh period, and holds the next frame
/// until shortly before the refresh `k` refreshes after the previous one, `k` being the game's
/// frame period in refreshes. A frame that is ready later is presented at once.
///
/// Used by the present thread only.
class HostPacer {
public:
    using Clock = std::chrono::steady_clock;

    /// A frame reached the present thread (for the game's frame period).
    void NoteQueued(Clock::time_point queued);

    /// Whether frames are held now (switch, refresh period known).
    [[nodiscard]] bool Pacing() const {
        return pacing;
    }

    /// Re-reads the switch and the refresh period of the display `native_window` is on (see
    /// DisplayRefreshQuery) now and then.
    void Refresh(Clock::time_point now, bool allowed, void* native_window);

    /// When the next frame should be handed to the swapchain; `now` or earlier = at once.
    [[nodiscard]] Clock::time_point Target(Clock::time_point now);

    /// The frame was presented with `id` (0: not presented); `was_held` if Target made it wait.
    void NoteSubmitted(u64 id, Clock::time_point submitted_at, bool was_held);

    /// Whether the previous frame was reported shown in time. After 8 misses in a row present
    /// wait is taken as broken on this driver and the pacer turns itself off for the session.
    void NoteShownWait(bool shown);

    /// Target made the present thread wait `hold`.
    void NoteHold(std::chrono::nanoseconds hold);

    /// vkWaitForPresentKHR returned for `id` at `shown`, or the present was lost.
    void NoteShown(u64 id, Clock::time_point shown, bool was_lost);

    /// The last presented id, which the next Target wants shown first (0: none outstanding).
    [[nodiscard]] u64 Outstanding() const {
        return outstanding;
    }

    /// Time from `now` to the display's next refresh, or -1 while the refresh grid is unknown.
    [[nodiscard]] long long NsToNextRefresh(Clock::time_point now) const;

    [[nodiscard]] std::chrono::nanoseconds RefreshPeriod() const {
        return std::chrono::nanoseconds{static_cast<s64>(refresh_ns)};
    }

private:
    struct Submitted {
        u64 id;
        Clock::time_point submitted;
        std::optional<Clock::time_point> intended; // the refresh it was held for
    };

    void Report(Clock::time_point now);

    bool pacing{};
    bool broken{};           ///< Present wait does not signal here: pacing stays off
    u32 shown_wait_misses{}; ///< In a row
    double refresh_ns{1e9 / 60.0};
    bool refresh_known{};
    Clock::time_point next_refresh_query{};
    DisplayRefreshQuery refresh_query;
    std::string refresh_display;

    std::optional<Clock::time_point> last_queued;
    double content_ns{1e9 / 60.0};

    std::optional<Clock::time_point> vblank; // refresh the last shown frame appeared at
    std::optional<Clock::time_point> last_shown;
    u64 last_shown_id{};
    std::optional<Clock::time_point> pending_intended;
    double lead_ns{2.0e6};
    u32 early_streak{}; ///< Frames in a row that could have been shown a refresh earlier
    u64 outstanding{};
    std::deque<Submitted> submitted;

    // Reported every 10 s.
    Clock::time_point report_start{};
    std::array<u32, 6> refreshes_shown{}; // 0 (same refresh), 1, 2, 3, 4, 5+
    u32 frames{}, held{}, missed{}, lost{}, caught_up{};
    double interval_sum_ms{}, interval_max_ms{}, hold_sum_ms{}, latency_sum_ms{};
};

} // namespace Vulkan
