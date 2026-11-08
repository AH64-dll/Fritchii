// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <mutex>
#include <vector>

#include "common/common_types.h"

namespace VideoCommon {

/// GPU address ranges of the maps GPU::MapInOrder queued and the GPU thread hasn't run yet.
class PendingMaps {
public:
    u64 Add(GPUVAddr gpu_addr, u64 size) {
        std::scoped_lock lock{mutex};
        const u64 id = ++last_id;
        maps.push_back({id, gpu_addr, gpu_addr + size});
        count.store(maps.size(), std::memory_order_release);
        return id;
    }

    void Remove(u64 id) {
        std::scoped_lock lock{mutex};
        std::erase_if(maps, [id](const Map& map) { return map.id == id; });
        count.store(maps.size(), std::memory_order_release);
    }

    [[nodiscard]] bool Overlaps(GPUVAddr gpu_addr, u64 size) const {
        if (count.load(std::memory_order_acquire) == 0 || size == 0) {
            return false;
        }
        std::scoped_lock lock{mutex};
        return std::ranges::any_of(maps, [&](const Map& map) {
            return gpu_addr < map.end && map.begin < gpu_addr + size;
        });
    }

private:
    struct Map {
        u64 id;
        GPUVAddr begin;
        GPUVAddr end;
    };
    mutable std::mutex mutex;
    std::vector<Map> maps;
    u64 last_id{};
    std::atomic<size_t> count{};
};

} // namespace VideoCommon
