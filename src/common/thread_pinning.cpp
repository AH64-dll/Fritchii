// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <bit>
#include <span>
#include <utility>

#include <fmt/format.h>

#include "common/thread_pinning.h"

namespace Common::ThreadPinning {

namespace {

// The first two cores that do not hold logical processor 0 go to the GPU threads.
Plan Assign(std::span<const Core> cores) {
    Plan plan;
    std::vector<u64> gpu;
    u64 all = 0;
    for (const Core& core : cores) {
        all |= core.mask;
        if ((core.mask & 1) == 0 && gpu.size() < 2) {
            gpu.push_back(core.mask);
        }
    }
    if (gpu.size() < 2) {
        return plan;
    }
    plan.front = gpu[0];
    plan.back = gpu[1];
    plan.guest = all & ~(plan.front | plan.back);
    plan.chosen_cores = cores.size();
    return plan;
}

bool Qualifies(const Plan& plan, Mode mode) {
    if (mode == Mode::Force) {
        return plan.Enabled() && plan.chosen_cores >= MinForcedCores;
    }
    return plan.Enabled() && plan.chosen_cores >= MinAutoCores &&
           static_cast<std::size_t>(std::popcount(plan.guest)) >= MinAutoGuestProcessors;
}

} // Anonymous namespace

Mode ParseMode(const char* setting) {
    if (setting == nullptr) {
        return Mode::Auto;
    }
    if (setting[0] == '0') {
        return Mode::Off;
    }
    return setting[0] == '1' ? Mode::Force : Mode::Auto;
}

Plan MakePlan(const Topology& topology, Mode mode) {
    Plan none;
    if (topology.cores.empty()) {
        return none;
    }
    const u8 fastest =
        std::ranges::max(topology.cores, {}, &Core::efficiency_class).efficiency_class;
    std::vector<Core> performance;
    for (const Core& core : topology.cores) {
        if (core.efficiency_class == fastest) {
            performance.push_back(core);
        }
    }
    none.performance_cores = performance.size();
    none.efficiency_cores = topology.cores.size() - performance.size();

    struct Group {
        u64 size;
        std::vector<Core> cores;
    };
    std::vector<Group> groups;
    for (const Cache& cache : topology.l3) {
        Group group{cache.size, {}};
        for (const Core& core : performance) {
            if ((core.mask & cache.mask) != 0) {
                group.cores.push_back(core);
            }
        }
        if (!group.cores.empty()) {
            groups.push_back(std::move(group));
        }
    }
    none.l3_caches = groups.size();
    const auto finish = [&](Plan plan, u64 l3_size) {
        plan.performance_cores = none.performance_cores;
        plan.efficiency_cores = none.efficiency_cores;
        plan.l3_caches = none.l3_caches;
        plan.l3_size = l3_size;
        return plan;
    };
    if (mode == Mode::Off) {
        return none;
    }

    // Largest L3 (an X3D V-cache CCD), then most cores, then the first enumerated.
    if (groups.size() >= 2) {
        const Group* best = &groups.front();
        for (const Group& group : groups) {
            if (group.size > best->size ||
                (group.size == best->size && group.cores.size() > best->cores.size())) {
                best = &group;
            }
        }
        if (Plan plan = Assign(best->cores); Qualifies(plan, mode)) {
            return finish(plan, best->size);
        }
        // No single L3 has enough cores (Zen 2 CCXs): span them as before.
    }
    if (Plan plan = Assign(performance); Qualifies(plan, mode)) {
        return finish(plan, groups.size() == 1 ? groups.front().size : 0);
    }
    return none;
}

std::string Describe(const Plan& plan, const Topology& topology, Mode mode) {
    std::string cpu = fmt::format("{} performance and {} efficiency cores, {} L3 cache{}",
                                  plan.performance_cores, plan.efficiency_cores, plan.l3_caches,
                                  plan.l3_caches == 1 ? "" : "s");
    u32 group0 = 0;
    for (const Core& core : topology.cores) {
        group0 += static_cast<u32>(std::popcount(core.mask));
    }
    if (topology.logical_processors > group0) {
        cpu += fmt::format(", {} of {} logical processors outside group 0 not pinned",
                           topology.logical_processors - group0, topology.logical_processors);
    }
    if (mode == Mode::Off) {
        return fmt::format("off (EDEN_NX2_PIN_THREADS=0); {}", cpu);
    }
    if (!plan.Enabled()) {
        return fmt::format("not pinned ({}); needs {} performance cores leaving {} logical "
                           "processors to the guest cores, EDEN_NX2_PIN_THREADS=1 forces from {}",
                           cpu, MinAutoCores, MinAutoGuestProcessors, MinForcedCores);
    }
    const std::string where =
        plan.l3_caches >= 2
            ? (plan.l3_size != 0 ? fmt::format(" in one {} MiB L3", plan.l3_size >> 20)
                                 : std::string{" across L3 caches"})
            : std::string{};
    return fmt::format("GPU front {:#x}, GPU back {:#x}, guest cores {:#x} (on {} cores{}); {}",
                       plan.front, plan.back, plan.guest, plan.chosen_cores, where, cpu);
}

} // namespace Common::ThreadPinning
