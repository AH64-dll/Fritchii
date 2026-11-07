// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <latch>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/alignment.h"
#include "common/div_ceil.h"
#include "common/scope_exit.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "tests/core/system_test_context.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_perf.h"
#include "video_core/texture_cache/util.h"
#include "video_core/textures/decoders.h"
#include "video_core/textures/unswizzle_worker.h"
#include "video_core/textures/workers.h"

namespace {

using namespace Tegra::Texture;

struct Layout {
    u32 width;
    u32 height;
    u32 bpp;
    u32 block_height;
    u32 alignment;

    size_t GuestBytes() const {
        const size_t stride = size_t{Common::AlignUpLog2(width, alignment)} * bpp;
        return Common::DivCeil(stride, size_t{GOB_SIZE_X}) *
               Common::DivCeil(height, GOB_SIZE_Y << block_height) *
               (size_t{GOB_SIZE} << block_height);
    }

    size_t HostBytes() const {
        return size_t{width} * height * bpp;
    }

    void Serial(std::span<u8> output, std::span<const u8> input, bool native) const {
        UnswizzleTexture(output, input, bpp, width, height, 1, block_height, 0, alignment, native);
    }

    bool Parallel(std::span<u8> output, std::span<const u8> input, bool native,
                  UnswizzleWorker& worker, Common::ThreadWorker& pool, bool verify = false) const {
        return worker.TryUnswizzle(output, input, bpp, width, height, 1, block_height, 0, alignment,
                                   native, verify, &pool);
    }
};

void Fill(std::span<u8> bytes) {
    u32 state = 0x12345678;
    for (u8& byte : bytes) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        byte = static_cast<u8>(state);
    }
}

} // namespace

TEST_CASE("Owned unswizzle matches the original across GOB layouts", "[unswizzle_workers]") {
    Common::ThreadWorker pool{1, "UnswizzleTest"};
    UnswizzleWorker worker;
    for (const bool native : {false, true}) {
        for (const u32 bpp : {1U, 2U, 4U, 8U, 16U}) {
            for (const u32 block_height : {0U, 1U, 2U, 3U, 4U, 5U}) {
                for (const u32 spacing : {0U, 2U}) {
                    const Layout layout{2049 / bpp + 1, 1025, bpp, block_height,
                                        6 - static_cast<u32>(std::countr_zero(bpp)) + spacing};
                    CAPTURE(native, bpp, block_height, spacing);
                    std::vector<u8> input(layout.GuestBytes());
                    Fill(input);
                    std::vector<u8> expected(layout.HostBytes() + 64, 0xCD);
                    std::vector<u8> actual(expected.size(), 0xCD);
                    layout.Serial(std::span{expected}.subspan(32, layout.HostBytes()), input,
                                  native);
                    const bool taken =
                        layout.Parallel(std::span{actual}.subspan(32, layout.HostBytes()), input,
                                        native, worker, pool);
                    REQUIRE(taken == (native && bpp <= 2));
                    if (!taken) {
                        REQUIRE(std::ranges::all_of(actual, [](u8 byte) { return byte == 0xCD; }));
                        layout.Serial(std::span{actual}.subspan(32, layout.HostBytes()), input,
                                      native);
                    }
                    REQUIRE(actual == expected);
                }
            }
        }
    }
}

TEST_CASE("Small and unsplittable unswizzles leave the original path untouched",
          "[unswizzle_workers]") {
    Common::ThreadWorker pool{1, "UnswizzleTest"};
    UnswizzleWorker worker;
    for (const Layout layout :
         {Layout{256, 256, 4, 4, 4}, Layout{65537, 32, 1, 5, 6}, Layout{1023, 256, 4, 0, 4},
          Layout{65, 32769, 1, 0, 8}, Layout{8193, 4096, 1, 0, 6}}) {
        std::vector<u8> input(layout.GuestBytes());
        std::vector<u8> output(layout.HostBytes(), 0xCD);
        REQUIRE_FALSE(layout.Parallel(output, input, true, worker, pool));
        REQUIRE(std::ranges::all_of(output, [](u8 byte) { return byte == 0xCD; }));
    }
    const Layout layout{1025, 1024, 2, 0, 5};
    std::vector<u8> input(layout.GuestBytes());
    std::vector<u8> output(layout.HostBytes(), 0xCD);
    REQUIRE_FALSE(
        worker.TryUnswizzle(output, input, 2, 1025, 1024, 2, 0, 1, 5, true, false, &pool));
    REQUIRE_FALSE(
        layout.Parallel(output, std::span{input}.first(input.size() - 1), true, worker, pool));
    REQUIRE_FALSE(
        layout.Parallel(std::span{output}.first(output.size() - 1), input, true, worker, pool));
    REQUIRE(std::ranges::all_of(output, [](u8 byte) { return byte == 0xCD; }));
    REQUIRE(layout.Parallel(output, input, true, worker, pool, true));
}

TEST_CASE("Banded unswizzle matches the original", "[unswizzle_workers]") {
    for (const size_t threads : {size_t{1}, size_t{4}}) {
        Common::ThreadWorker pool{threads, "UnswizzleBands"};
        for (const bool native : {false, true}) {
            for (const u32 bpp : {1U, 4U, 8U, 16U}) {
                for (const u32 block_height : {0U, 2U, 4U}) {
                    const Layout layout{4097 / bpp + 3, 1213, bpp, block_height,
                                        6 - static_cast<u32>(std::countr_zero(bpp))};
                    CAPTURE(threads, native, bpp, block_height);
                    std::vector<u8> input(layout.GuestBytes());
                    Fill(input);
                    std::vector<u8> expected(layout.HostBytes() + 64, 0xCD);
                    std::vector<u8> actual(expected.size(), 0xCD);
                    layout.Serial(std::span{expected}.subspan(32, layout.HostBytes()), input,
                                  native);
                    REQUIRE(TryUnswizzleBands(std::span{actual}.subspan(32, layout.HostBytes()),
                                              input, bpp, layout.width, layout.height, 1,
                                              block_height, 0, layout.alignment, native, false,
                                              &pool));
                    REQUIRE(actual == expected);
                }
            }
        }
    }
}

TEST_CASE("Banded unswizzle finishes alone behind a saturated pool", "[unswizzle_workers]") {
    const Layout layout{2048, 2048, 4, 4, 4};
    std::vector<u8> input(layout.GuestBytes());
    Fill(input);
    std::vector<u8> expected(layout.HostBytes());
    layout.Serial(expected, input, true);
    Common::ThreadWorker pool{1, "BandsBlocked"};
    std::promise<void> entered;
    std::promise<void> release;
    auto gate = release.get_future();
    pool.QueueWork([&] {
        entered.set_value();
        gate.wait();
    });
    entered.get_future().wait();
    std::vector<u8> actual(layout.HostBytes(), 0xCD);
    REQUIRE(TryUnswizzleBands(actual, input, 4, 2048, 2048, 1, 4, 0, 4, true, false, &pool));
    REQUIRE(actual == expected);
    release.set_value();
    pool.WaitForRequests();
}

TEST_CASE("Unswizzle leaves output untouched when a delayed worker saturates the pool",
          "[unswizzle_workers]") {
    using namespace std::chrono_literals;
    const Layout layout{1025, 1025, 2, 4, 5};
    std::vector<u8> input(layout.GuestBytes());
    Fill(input);
    std::vector<u8> expected(layout.HostBytes());
    layout.Serial(expected, input, true);
    for (int iteration = 0; iteration < 8; ++iteration) {
        Common::ThreadWorker pool{1, "DelayedUnswizzle"};
        UnswizzleWorker worker;
        std::promise<void> entered;
        std::promise<void> release;
        auto gate = release.get_future();
        pool.QueueWork([&] {
            entered.set_value();
            gate.wait();
        });
        entered.get_future().wait();
        std::vector<u8> actual(layout.HostBytes(), 0xCD);
        auto result = std::async(
            std::launch::async, [&] { return layout.Parallel(actual, input, true, worker, pool); });
        auto unblock = SCOPE_GUARD {
            release.set_value();
            pool.WaitForRequests();
        };
        REQUIRE(result.wait_for(2s) == std::future_status::ready);
        REQUIRE_FALSE(result.get());
        REQUIRE(std::ranges::all_of(actual, [](u8 byte) { return byte == 0xCD; }));
        release.set_value();
        unblock.Cancel();
        pool.WaitForRequests();
        REQUIRE(layout.Parallel(actual, input, true, worker, pool));
        REQUIRE(actual == expected);
    }
}

TEST_CASE("Texture worker admission reserves only available worker slots", "[unswizzle_workers]") {
    Common::ThreadWorker pool{2, "AdmissionTest"};
    std::latch entered{2};
    std::latch release{1};
    std::atomic<unsigned> completed{};
    auto unblock = SCOPE_GUARD {
        release.count_down();
        pool.WaitForRequests();
    };
    for (int i = 0; i < 2; ++i) {
        REQUIRE(pool.TryQueueWork([&] {
            entered.count_down();
            release.wait();
            ++completed;
        }));
    }
    entered.wait();
    REQUIRE_FALSE(pool.TryQueueWork([&] { ++completed; }));
    release.count_down();
    unblock.Cancel();
    pool.WaitForRequests();
    REQUIRE(completed == 2);
    REQUIRE(pool.TryQueueWork([&] { ++completed; }));
    pool.WaitForRequests();
    REQUIRE(completed == 3);
}

TEST_CASE("Owned snapshot survives overlapping source and destination", "[unswizzle_workers]") {
    Common::ThreadWorker pool{1, "UnswizzleTest"};
    UnswizzleWorker worker;
    for (const Layout layout :
         {Layout{2049, 1025, 1, 0, 8}, Layout{1025, 1025, 2, 4, 5}, Layout{4097, 1025, 1, 5, 6}}) {
        std::vector<u8> storage(layout.GuestBytes());
        Fill(storage);
        std::vector<u8> expected(layout.HostBytes());
        layout.Serial(expected, storage, true);
        const auto output = std::span{storage}.first(layout.HostBytes());
        REQUIRE(layout.Parallel(output, storage, true, worker, pool));
        REQUIRE(std::ranges::equal(output, expected));
    }
}

TEST_CASE("Independent owners can share the texture pool", "[unswizzle_workers]") {
    const Layout layout{1025, 1025, 2, 4, 5};
    Common::ThreadWorker pool{2, "UnswizzleTest"};
    std::vector<u8> input(layout.GuestBytes());
    Fill(input);
    std::vector<u8> expected(layout.HostBytes());
    layout.Serial(expected, input, true);
    for (int iteration = 0; iteration < 4; ++iteration) {
        std::vector<std::future<std::vector<u8>>> results;
        for (int owner = 0; owner < 4; ++owner) {
            results.push_back(std::async(std::launch::async, [&] {
                UnswizzleWorker worker;
                std::vector<u8> actual(layout.HostBytes());
                if (!layout.Parallel(actual, input, true, worker, pool)) {
                    layout.Serial(actual, input, true);
                }
                return actual;
            }));
        }
        for (auto& result : results) {
            REQUIRE(result.get() == expected);
        }
    }
}

TEST_CASE("Full upload keeps mip layer offsets and compressed edge blocks", "[unswizzle_workers]") {
    using namespace VideoCommon;
    REQUIRE_FALSE(Nx2Perf::On(Nx2Perf::Opt::SplitOverlap));
    ScopedSystemTestContext context;
    Core::System system;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device_memory{physical};
    UnswizzleWorker worker;
    for (const bool native : {false, true}) {
        Tegra::MemoryManager memory{system, device_memory, 32, 1ULL << 31, 16};
        if (native) {
            memory.EnableNativeTiling();
        }
        for (const auto format : {PixelFormat::R8_UNORM, PixelFormat::A8B8G8R8_UNORM,
                                  PixelFormat::BC1_RGBA_UNORM, PixelFormat::ASTC_2D_5X4_UNORM}) {
            ImageInfo info;
            info.type = ImageType::e2D;
            info.format = format;
            info.size = {2049, 1025, 1};
            info.block = {0, 4, 0};
            info.resources = {6, 3};
            info.tile_width_spacing = 1;
            info.layer_stride = CalculateLayerStride(info);
            CAPTURE(native, format);
            std::vector<u8> input(CalculateGuestSizeInBytes(info));
            Fill(input);
            std::vector<u8> expected(CalculateUnswizzledSizeBytes(info) + 64, 0xCD);
            std::vector<u8> actual(expected.size(), 0xCD);
            const auto original = UnswizzleImage(
                memory, 0, info, input, std::span{expected}.subspan(32, expected.size() - 64));
            const auto parallel = UnswizzleImage(
                memory, 0, info, input, std::span{actual}.subspan(32, actual.size() - 64), &worker);
            REQUIRE(actual == expected);
            REQUIRE(parallel.size() == original.size());
            for (size_t i = 0; i < original.size(); ++i) {
                CHECK(parallel[i].buffer_offset == original[i].buffer_offset);
                CHECK(parallel[i].buffer_size == original[i].buffer_size);
                CHECK(parallel[i].buffer_row_length == original[i].buffer_row_length);
                CHECK(parallel[i].buffer_image_height == original[i].buffer_image_height);
                CHECK(parallel[i].image_subresource.base_level ==
                      original[i].image_subresource.base_level);
                CHECK(parallel[i].image_subresource.base_layer ==
                      original[i].image_subresource.base_layer);
                CHECK(parallel[i].image_subresource.num_layers ==
                      original[i].image_subresource.num_layers);
                CHECK(parallel[i].image_extent == original[i].image_extent);
                CHECK(parallel[i].image_offset == original[i].image_offset);
            }
        }
    }
}

TEST_CASE("Full upload does not wait behind a saturated shared pool", "[unswizzle_workers]") {
    using namespace VideoCommon;
    using namespace std::chrono_literals;
    ScopedSystemTestContext context;
    Core::System system;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device_memory{physical};
    Tegra::MemoryManager memory{system, device_memory, 32, 1ULL << 31, 16};
    memory.EnableNativeTiling();
    ImageInfo info;
    info.type = ImageType::e2D;
    info.format = PixelFormat::R8_UNORM;
    info.size = {2049, 1025, 1};
    info.block = {0, 4, 0};
    info.resources = {1, 1};
    info.layer_stride = CalculateLayerStride(info);
    std::vector<u8> input(CalculateGuestSizeInBytes(info));
    Fill(input);
    std::vector<u8> expected(CalculateUnswizzledSizeBytes(info));
    [[maybe_unused]] const auto original = UnswizzleImage(memory, 0, info, input, expected);
    std::vector<u8> actual(expected.size(), 0xCD);
    auto& pool = GetThreadWorkers();
    const char* threads_env = std::getenv("EDEN_NX2_TRANSCODE_THREADS");
    const u32 threads =
        threads_env ? (std::max)(static_cast<u32>(std::strtoul(threads_env, nullptr, 10)), 1U)
                    : (std::max)(std::thread::hardware_concurrency() / 4, 2U);
    std::latch entered{threads};
    std::latch release{1};
    for (u32 i = 0; i < threads; ++i) {
        pool.QueueWork([&] {
            entered.count_down();
            release.wait();
        });
    }
    entered.wait();
    UnswizzleWorker worker;
    std::promise<void> started;
    auto result = std::async(std::launch::async, [&] {
        started.set_value();
        return UnswizzleImage(memory, 0, info, input, actual, &worker);
    });
    auto unblock = SCOPE_GUARD {
        release.count_down();
        pool.WaitForRequests();
    };
    started.get_future().wait();
    REQUIRE(result.wait_for(2s) == std::future_status::ready);
    release.count_down();
    unblock.Cancel();
    REQUIRE(result.get().size() == original.size());
    REQUIRE(actual == expected);
    pool.WaitForRequests();
}

TEST_CASE("Full upload end to end microbenchmark", "[.unswizzle_upload_benchmark]") {
    using namespace VideoCommon;
    using Clock = std::chrono::steady_clock;
    ScopedSystemTestContext context;
    Core::System system;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device_memory{physical};
    Tegra::MemoryManager memory{system, device_memory, 32, 1ULL << 31, 16};
    memory.EnableNativeTiling();
    UnswizzleWorker worker;
    auto& pool = GetThreadWorkers();
    std::printf("format,competing_jobs,serial_upload_us,owned_upload_us,ratio\n");
    for (const auto format :
         {PixelFormat::R8_UNORM, PixelFormat::R16_UNORM, PixelFormat::A8B8G8R8_UNORM}) {
        ImageInfo info;
        info.type = ImageType::e2D;
        info.format = format;
        info.size = {2049, 1025, 1};
        info.block = {0, 4, 0};
        info.resources = {6, 3};
        info.layer_stride = CalculateLayerStride(info);
        std::vector<u8> input(CalculateGuestSizeInBytes(info));
        Fill(input);
        std::vector<u8> expected(CalculateUnswizzledSizeBytes(info));
        std::vector<u8> actual(expected.size());
        for (const u32 competing_jobs : {0U, 4U}) {
            std::vector<double> serial_times;
            std::vector<double> parallel_times;
            const auto measure = [&](UnswizzleWorker* selected, std::span<u8> output) {
                for (u32 job = 0; job < competing_jobs; ++job) {
                    pool.QueueWork([&] {
                        const Layout busy{2049, 1025, 1, 0, 8};
                        std::vector<u8> busy_input(busy.GuestBytes(), 0x5A);
                        std::vector<u8> busy_output(busy.HostBytes());
                        for (int iteration = 0; iteration < 4; ++iteration) {
                            busy.Serial(busy_output, busy_input, true);
                        }
                    });
                }
                const auto begin = Clock::now();
                [[maybe_unused]] const auto copies =
                    UnswizzleImage(memory, 0, info, input, output, selected);
                const double us =
                    std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
                pool.WaitForRequests();
                return us;
            };
            for (int i = 0; i < 18; ++i) {
                double s, p;
                if (i % 2 == 0) {
                    s = measure(nullptr, expected);
                    p = measure(&worker, actual);
                } else {
                    p = measure(&worker, actual);
                    s = measure(nullptr, expected);
                }
                REQUIRE(actual == expected);
                if (i >= 2) {
                    serial_times.push_back(s);
                    parallel_times.push_back(p);
                }
            }
            std::ranges::sort(serial_times);
            std::ranges::sort(parallel_times);
            const double s = serial_times[serial_times.size() / 2];
            const double p = parallel_times[parallel_times.size() / 2];
            std::printf("%u,%u,%.2f,%.2f,%.3f\n", static_cast<unsigned>(format), competing_jobs, s,
                        p, p / s);
        }
    }
}

TEST_CASE("Unswizzle end to end microbenchmark", "[.unswizzle_benchmark]") {
    using Clock = std::chrono::steady_clock;
    UnswizzleWorker worker;
    const Layout startup_layout{1025, 1025, 2, 4, 5};
    std::vector<u8> startup_input(startup_layout.GuestBytes());
    std::vector<u8> startup_output(startup_layout.HostBytes());
    Fill(startup_input);
    const auto startup_begin = Clock::now();
    auto& pool = GetThreadWorkers();
    REQUIRE(startup_layout.Parallel(startup_output, startup_input, true, worker, pool));
    const auto startup_us =
        std::chrono::duration<double, std::micro>(Clock::now() - startup_begin).count();
    std::printf("first_eligible_call_including_pool_startup_us=%.2f\n", startup_us);
    std::printf(
        "width,height,bpp,block_height,native,host_bytes,guest_bytes,parallel_used,serial_us,owned_"
        "parallel_us,ratio\n");
    for (const Layout layout :
         {Layout{256, 256, 4, 4, 4}, Layout{512, 512, 4, 4, 4}, Layout{1024, 1024, 4, 4, 4},
          Layout{2048, 2048, 4, 4, 4}, Layout{2049, 1025, 1, 0, 8}, Layout{513, 257, 16, 5, 4},
          Layout{513, 1025, 16, 5, 4}, Layout{1025, 1025, 2, 4, 5}, Layout{513, 1025, 4, 4, 4},
          Layout{257, 1025, 8, 4, 3}, Layout{4097, 1025, 1, 4, 6}, Layout{2049, 1025, 2, 4, 5},
          Layout{2050, 1025, 1, 4, 6}}) {
        for (const bool native : {false, true}) {
            std::vector<u8> input(layout.GuestBytes());
            Fill(input);
            std::vector<u8> expected(layout.HostBytes());
            std::vector<u8> actual(layout.HostBytes());
            std::vector<double> serial_times;
            std::vector<double> parallel_times;
            bool parallel_used = false;
            const auto serial = [&] { layout.Serial(expected, input, native); };
            const auto parallel = [&] {
                parallel_used = layout.Parallel(actual, input, native, worker, pool);
                if (!parallel_used) {
                    layout.Serial(actual, input, native);
                }
            };
            const auto measure = [](auto&& operation) {
                const auto begin = Clock::now();
                operation();
                return std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
            };
            for (int i = 0; i < 44; ++i) {
                double s, p;
                if (i % 2 == 0) {
                    s = measure(serial);
                    p = measure(parallel);
                } else {
                    p = measure(parallel);
                    s = measure(serial);
                }
                REQUIRE(actual == expected);
                if (i >= 4) {
                    serial_times.push_back(s);
                    parallel_times.push_back(p);
                }
            }
            std::ranges::sort(serial_times);
            std::ranges::sort(parallel_times);
            const double s = serial_times[serial_times.size() / 2];
            const double p = parallel_times[parallel_times.size() / 2];
            std::printf("%u,%u,%u,%u,%d,%zu,%zu,%d,%.2f,%.2f,%.3f\n", layout.width, layout.height,
                        layout.bpp, layout.block_height, native, layout.HostBytes(),
                        layout.GuestBytes(), parallel_used, s, p, p / s);
        }
    }
}
