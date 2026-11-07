// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "common/thread_pinning.h"

namespace {

using namespace Common::ThreadPinning;

constexpr u64 MiB = 1ULL << 20;

// Windows numbers logical processors so that SMT siblings are adjacent.
struct Builder {
    Topology topology;
    u32 next = 0;

    u64 AddCores(u32 cores, u32 threads_per_core, u8 efficiency_class) {
        u64 added = 0;
        for (u32 core = 0; core < cores; ++core) {
            const u64 mask = ((1ULL << threads_per_core) - 1) << next;
            topology.cores.push_back({mask, efficiency_class});
            next += threads_per_core;
            added |= mask;
        }
        topology.logical_processors = next;
        return added;
    }

    void AddL3(u64 mask, u64 size) {
        topology.l3.push_back({mask, size});
    }
};

Topology Uniform(u32 cores, u32 threads_per_core, u64 l3_size) {
    Builder builder;
    builder.AddL3(builder.AddCores(cores, threads_per_core, 0), l3_size);
    return builder.topology;
}

// The rule before topology awareness: group 0 cores in enumeration order, cores 1 and 2 to the
// GPU threads, every other core to the guests, from 6 cores (4 when forced).
Plan Legacy(const Topology& topology, Mode mode) {
    Plan plan;
    const auto& cores = topology.cores;
    if (mode == Mode::Off || cores.size() < (mode == Mode::Force ? 4U : 6U)) {
        return plan;
    }
    plan.front = cores[1].mask;
    plan.back = cores[2].mask;
    for (const Core& core : cores) {
        plan.guest |= core.mask;
    }
    plan.guest &= ~(plan.front | plan.back);
    return plan;
}

void CheckSame(const Plan& plan, const Plan& expected) {
    CHECK(plan.front == expected.front);
    CHECK(plan.back == expected.back);
    CHECK(plan.guest == expected.guest);
}

void CheckNotPinned(const Plan& plan) {
    CHECK_FALSE(plan.Enabled());
    CHECK(plan.front == 0);
    CHECK(plan.back == 0);
    CHECK(plan.guest == 0);
}

} // Anonymous namespace

TEST_CASE("Thread pinning: EDEN_NX2_PIN_THREADS", "[common][thread_pinning]") {
    CHECK(ParseMode(nullptr) == Mode::Auto);
    CHECK(ParseMode("") == Mode::Auto);
    CHECK(ParseMode("0") == Mode::Off);
    CHECK(ParseMode("1") == Mode::Force);
    CHECK(ParseMode("2") == Mode::Auto);
}

TEST_CASE("Thread pinning: Ryzen 7 7800X3D is unchanged", "[common][thread_pinning]") {
    const Topology cpu = Uniform(8, 2, 96 * MiB);
    for (const Mode mode : {Mode::Auto, Mode::Force}) {
        const Plan plan = MakePlan(cpu, mode);
        CheckSame(plan, Legacy(cpu, mode));
        CHECK(plan.front == 0x000C);
        CHECK(plan.back == 0x0030);
        CHECK(plan.guest == 0xFFC3);
        CHECK(plan.performance_cores == 8);
        CHECK(plan.efficiency_cores == 0);
        CHECK(plan.chosen_cores == 8);
    }
    CheckNotPinned(MakePlan(cpu, Mode::Off));
}

TEST_CASE("Thread pinning: other single-L3 CPUs with SMT are unchanged",
          "[common][thread_pinning]") {
    for (const u32 cores : {6U, 8U, 12U, 16U}) {
        const Topology cpu = Uniform(cores, 2, 32 * MiB);
        CheckSame(MakePlan(cpu, Mode::Auto), Legacy(cpu, Mode::Auto));
    }
}

TEST_CASE("Thread pinning: no SMT needs 8 cores unless forced", "[common][thread_pinning]") {
    // Core i5-9400F: the guests would get 4 logical processors for 6 emulated cores.
    const Topology i5 = Uniform(6, 1, 9 * MiB);
    CheckNotPinned(MakePlan(i5, Mode::Auto));
    const Plan forced = MakePlan(i5, Mode::Force);
    CheckSame(forced, Legacy(i5, Mode::Force));
    CHECK(forced.front == 0x02);
    CHECK(forced.back == 0x04);
    CHECK(forced.guest == 0x39);

    const Topology i7 = Uniform(8, 1, 12 * MiB);
    const Plan plan = MakePlan(i7, Mode::Auto);
    CheckSame(plan, Legacy(i7, Mode::Auto));
    CHECK(plan.guest == 0xF9);
}

TEST_CASE("Thread pinning: hybrid 8P+8E leaves the E-cores alone", "[common][thread_pinning]") {
    // Core i9-12900K: 8 P-cores with SMT (class 1), then 8 E-cores (class 0), one L3.
    Builder builder;
    const u64 p = builder.AddCores(8, 2, 1);
    const u64 e = builder.AddCores(8, 1, 0);
    builder.AddL3(p | e, 30 * MiB);
    const Plan plan = MakePlan(builder.topology, Mode::Auto);
    CHECK(plan.front == 0x000C);
    CHECK(plan.back == 0x0030);
    CHECK(plan.guest == 0xFFC3);
    CHECK((plan.guest & e) == 0);
    CHECK(plan.performance_cores == 8);
    CHECK(plan.efficiency_cores == 8);
}

TEST_CASE("Thread pinning: hybrid E-cores first still pins the P-cores",
          "[common][thread_pinning]") {
    Builder builder;
    const u64 e = builder.AddCores(4, 1, 0);
    const u64 p = builder.AddCores(6, 2, 1);
    builder.AddL3(e | p, 18 * MiB);
    const Plan plan = MakePlan(builder.topology, Mode::Auto);
    // Logical processor 0 is an E-core, so the first two P-cores take the GPU threads.
    CHECK(plan.front == 0x0030);
    CHECK(plan.back == 0x00C0);
    CHECK(plan.guest == (p & ~0xF0ULL));
    CHECK((plan.guest & e) == 0);
}

TEST_CASE("Thread pinning: hybrid 4P+4E is not pinned unless forced", "[common][thread_pinning]") {
    // Core i5-12450H: 4 P-cores with SMT and 4 E-cores. E-cores do not count toward the threshold.
    Builder builder;
    const u64 p = builder.AddCores(4, 2, 1);
    const u64 e = builder.AddCores(4, 1, 0);
    builder.AddL3(p | e, 12 * MiB);
    const Plan plan = MakePlan(builder.topology, Mode::Auto);
    CheckNotPinned(plan);
    CHECK(plan.performance_cores == 4);
    CHECK(plan.efficiency_cores == 4);

    const Plan forced = MakePlan(builder.topology, Mode::Force);
    CHECK(forced.front == 0x0C);
    CHECK(forced.back == 0x30);
    CHECK(forced.guest == 0xC3);
    CHECK((forced.guest & e) == 0);
}

TEST_CASE("Thread pinning: Ryzen 9 7950X3D stays on the V-cache CCD", "[common][thread_pinning]") {
    Builder builder;
    const u64 ccd0 = builder.AddCores(8, 2, 0);
    const u64 ccd1 = builder.AddCores(8, 2, 0);
    builder.AddL3(ccd0, 96 * MiB);
    builder.AddL3(ccd1, 32 * MiB);
    const Plan plan = MakePlan(builder.topology, Mode::Auto);
    CHECK(plan.front == 0x000C);
    CHECK(plan.back == 0x0030);
    CHECK(plan.guest == 0xFFC3);
    CHECK((plan.guest & ccd1) == 0);
    CHECK(plan.chosen_cores == 8);
    CHECK(plan.l3_caches == 2);
    CHECK(plan.l3_size == 96 * MiB);
}

TEST_CASE("Thread pinning: a V-cache CCD enumerated second is preferred",
          "[common][thread_pinning]") {
    Builder builder;
    const u64 ccd0 = builder.AddCores(8, 2, 0);
    const u64 ccd1 = builder.AddCores(8, 2, 0);
    builder.AddL3(ccd0, 32 * MiB);
    builder.AddL3(ccd1, 96 * MiB);
    const Plan plan = MakePlan(builder.topology, Mode::Auto);
    // The CCD does not hold logical processor 0, so its first core can take a GPU thread.
    CHECK(plan.front == 0x0003'0000);
    CHECK(plan.back == 0x000C'0000);
    CHECK(plan.guest == 0xFFF0'0000);
    CHECK((plan.guest & ccd0) == 0);
}

TEST_CASE("Thread pinning: equal CCDs use the first", "[common][thread_pinning]") {
    // Ryzen 9 5900X: two 6-core CCDs with 32 MiB each.
    Builder builder;
    const u64 ccd0 = builder.AddCores(6, 2, 0);
    const u64 ccd1 = builder.AddCores(6, 2, 0);
    builder.AddL3(ccd0, 32 * MiB);
    builder.AddL3(ccd1, 32 * MiB);
    const Plan plan = MakePlan(builder.topology, Mode::Auto);
    CHECK(plan.front == 0x00C);
    CHECK(plan.back == 0x030);
    CHECK(plan.guest == 0xFC3);
    CHECK((plan.guest & ccd1) == 0);
}

TEST_CASE("Thread pinning: L3 caches too small to hold the plan are spanned",
          "[common][thread_pinning]") {
    // Ryzen 7 3700X: two 4-core CCXs with 16 MiB each, pinned across both as before.
    Builder builder;
    const u64 ccx0 = builder.AddCores(4, 2, 0);
    const u64 ccx1 = builder.AddCores(4, 2, 0);
    builder.AddL3(ccx0, 16 * MiB);
    builder.AddL3(ccx1, 16 * MiB);
    const Plan plan = MakePlan(builder.topology, Mode::Auto);
    CheckSame(plan, Legacy(builder.topology, Mode::Auto));
    CHECK(plan.chosen_cores == 8);
    CHECK(plan.l3_size == 0);
}

TEST_CASE("Thread pinning: nothing reported", "[common][thread_pinning]") {
    CheckNotPinned(MakePlan(Topology{}, Mode::Auto));
    CheckNotPinned(MakePlan(Topology{}, Mode::Force));
    const Topology two = Uniform(2, 2, 4 * MiB);
    CheckNotPinned(MakePlan(two, Mode::Force));
}

TEST_CASE("Thread pinning: the plan is described", "[common][thread_pinning]") {
    const Topology cpu = Uniform(8, 2, 96 * MiB);
    CHECK(Describe(MakePlan(cpu, Mode::Auto), cpu, Mode::Auto) ==
          "GPU front 0xc, GPU back 0x30, guest cores 0xffc3 (on 8 cores); 8 performance and 0 "
          "efficiency cores, 1 L3 cache");
    Builder builder;
    builder.AddL3(builder.AddCores(8, 2, 0), 96 * MiB);
    builder.AddL3(builder.AddCores(8, 2, 0), 32 * MiB);
    const Topology x3d = builder.topology;
    CHECK(Describe(MakePlan(x3d, Mode::Auto), x3d, Mode::Auto) ==
          "GPU front 0xc, GPU back 0x30, guest cores 0xffc3 (on 8 cores in one 96 MiB L3); 16 "
          "performance and 0 efficiency cores, 2 L3 caches");
    CHECK(Describe(MakePlan(cpu, Mode::Off), cpu, Mode::Off) ==
          "off (EDEN_NX2_PIN_THREADS=0); 8 performance and 0 efficiency cores, 1 L3 cache");
}
