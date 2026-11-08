// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>

#include "common/core_pinning.h"

namespace Common {

GpuThreadPinning PlanGpuThreadPinning(std::span<const LogicalProcessor> processors) {
    GpuThreadPinning plan;
    if (processors.empty()) {
        return plan;
    }
    const u8 fastest =
        std::ranges::max(processors, {}, &LogicalProcessor::efficiency_class).efficiency_class;
    std::vector<u32> cores;
    for (const LogicalProcessor& processor : processors) {
        if (processor.efficiency_class == fastest) {
            cores.push_back(processor.core);
        }
    }
    std::ranges::sort(cores);
    cores.erase(std::unique(cores.begin(), cores.end()), cores.end());
    plan.performance_cores = cores.size();
    if (cores.size() < MinCoresForGpuThreadPinning) {
        return plan;
    }
    const u32 front_core = cores[1];
    const u32 back_core = cores[2];
    for (const LogicalProcessor& processor : processors) {
        if (processor.efficiency_class != fastest) {
            continue;
        }
        if (processor.core == front_core) {
            plan.front.push_back(processor.id);
        } else if (processor.core == back_core) {
            plan.back.push_back(processor.id);
        } else {
            plan.guest.push_back(processor.id);
        }
    }
    return plan;
}

} // namespace Common
