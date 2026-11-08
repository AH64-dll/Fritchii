// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <chrono>
#include <thread>

#include <catch2/catch_test_macros.hpp>

#include "video_core/nx2.h"

TEST_CASE("Nx2WaitAsyncReportsSettled returns at once without pending reports",
          "[video_core][nx2]") {
    constexpr u64 address = 0x10'0000;
    Tegra::Nx2IssueAsyncReport(address);
    Tegra::Nx2CompleteAsyncReport(address);
    Tegra::Nx2WaitAsyncReportsSettled(address, 0x20);
    REQUIRE(Tegra::Nx2AsyncReportsSettled(address, 0x20));
}

TEST_CASE("Nx2WaitAsyncReportsSettled waits for every report in the range", "[video_core][nx2]") {
    constexpr u64 address = 0x20'0000;
    Tegra::Nx2IssueAsyncReport(address);
    Tegra::Nx2IssueAsyncReport(address + 0x10);
    std::atomic<int> completed{};
    std::jthread completer([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds{20});
        completed.store(1);
        Tegra::Nx2CompleteAsyncReport(address);
        std::this_thread::sleep_for(std::chrono::milliseconds{20});
        completed.store(2);
        Tegra::Nx2CompleteAsyncReport(address + 0x10);
    });
    Tegra::Nx2WaitAsyncReportsSettled(address, 0x20);
    REQUIRE(completed.load() == 2);
}

namespace {

u32 Issued(u64 address) {
    return Tegra::g_nx2_async_reports_issued[Tegra::Nx2AsyncReportSlot(address)].load();
}

u32 Completed(u64 address) {
    return Tegra::g_nx2_async_reports_completed[Tegra::Nx2AsyncReportSlot(address)].load();
}

} // Anonymous namespace

TEST_CASE("Nx2AsyncReport completes an issued report once", "[video_core][nx2]") {
    constexpr u64 address = 0x40'0000;
    const u32 issued = Issued(address);
    const u32 completed = Completed(address);
    {
        Tegra::Nx2AsyncReport report{address};
        report.Issue();
        REQUIRE_FALSE(Tegra::Nx2AsyncReportsSettled(address, 4));
        report.Complete();
        report.Complete();
    }
    REQUIRE(Issued(address) == issued + 1);
    REQUIRE(Completed(address) == completed + 1);
}

TEST_CASE("Nx2AsyncReport dropped unrun still settles its slot", "[video_core][nx2]") {
    constexpr u64 address = 0x50'0000;
    {
        Tegra::Nx2AsyncReport report{address};
        report.Issue();
    }
    REQUIRE(Tegra::Nx2AsyncReportsSettled(address, 4));
}

TEST_CASE("Nx2AsyncReport never issued counts nothing", "[video_core][nx2]") {
    constexpr u64 address = 0x60'0000;
    const u32 completed = Completed(address);
    {
        Tegra::Nx2AsyncReport report{address};
    }
    REQUIRE(Completed(address) == completed);
    REQUIRE(Tegra::Nx2AsyncReportsSettled(address, 4));
}

TEST_CASE("Nx2AsyncReportsSettled treats an empty range as settled", "[video_core][nx2]") {
    Tegra::Nx2IssueAsyncReport(0);
    REQUIRE(Tegra::Nx2AsyncReportsSettled(0, 0));
    Tegra::Nx2WaitAsyncReportsSettled(0, 0);
    Tegra::Nx2CompleteAsyncReport(0);
}

TEST_CASE("Nx2WaitAsyncReportsSettled ignores reports outside the range", "[video_core][nx2]") {
    constexpr u64 address = 0x30'0000;
    Tegra::Nx2IssueAsyncReport(address + 0x100);
    Tegra::Nx2WaitAsyncReportsSettled(address, 0x10);
    REQUIRE_FALSE(Tegra::Nx2AsyncReportsSettled(address + 0x100, 0x10));
    Tegra::Nx2CompleteAsyncReport(address + 0x100);
}
