// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include "common/container/unordered_map.h"
#include <utility>
#include <vector>

#include "common/assert.h"
#include "common/common_types.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/query_cache/bank_base.h"
#include "video_core/query_cache/frame_trace.h"
#include "video_core/query_cache/query_base.h"
#include "video_core/query_cache/query_cache_base.h"
#include "video_core/query_cache/query_stream.h"
#include "video_core/query_cache/types.h"

namespace VideoCommon {

using Maxwell = Tegra::Engines::Maxwell3D;

/// EDEN_NX2_QUERY_LAZY_SYNC (on unless 0): payload releases (QMD releases, semaphores) whose value
/// is written to guest memory during the report reach the host's buffers as a guest CPU write
/// does (the buffer cache uploads the page before GPU work next uses it) instead of a copy at the
/// next wait for idle, which ended the render pass ~1,200 times a second in Ocarina. Pages holding
/// GPU-written data, and releases written later (high GPU accuracy, delayed fences), are still
/// copied.
inline bool LazyPayloadSync() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_QUERY_LAZY_SYNC");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

struct SyncValuesStruct {
    VAddr address;
    u64 value;
    u64 size;

    static constexpr bool GeneratesBaseBuffer = true;
};

template <typename Traits>
class GuestStreamer : public SimpleStreamer<GuestQuery> {
public:
    using RuntimeType = typename Traits::RuntimeType;

    GuestStreamer(size_t id_, RuntimeType& runtime_)
        : SimpleStreamer<GuestQuery>(id_), runtime{runtime_} {}

    virtual ~GuestStreamer() = default;

    size_t WriteCounter(VAddr address, bool has_timestamp, u32 value,
                        std::optional<u32> subreport = std::nullopt) override {
        auto new_id = BuildQuery(has_timestamp, address, static_cast<u64>(value));
        // The value is kept here too: fence releases free their slot at once, and the slot can
        // hold another release before the sync (EDEN_NX2_FENCE_TRIM batches them).
        pending_sync.push_back(PendingSync{new_id, address, static_cast<u64>(value),
                                           has_timestamp});
        return new_id;
    }

    bool HasPendingSync() const override {
        return !pending_sync.empty();
    }

    void MarkInFlight(size_t query_id) override {
        if (pending_sync.empty() || pending_sync.back().id != query_id) {
            return;
        }
        pending_sync.back().lazy = true;
        inflight = true;
        inflight_skipped = false;
    }

    bool FinishInFlight(size_t query_id, bool written) override {
        if (!inflight) {
            // A wait for idle copied it meanwhile (its pages hold GPU-written data).
            return false;
        }
        inflight = false;
        if (pending_sync.empty()) {
            return false;
        }
        PendingSync& entry = pending_sync.back();
        if (entry.id != query_id) {
            entry.lazy = false;
            return inflight_skipped;
        }
        if (!written) {
            entry.lazy = false;
            return inflight_skipped;
        }
        if (!inflight_skipped) {
            // Applied with the next wait for idle, as its copy would have been.
            return false;
        }
        // A wait for idle came between the report and its guest write: apply it now, before
        // any later GPU work, as the copy at that wait would have been.
        GuestWrittenRange range{
            .address = entry.address, .size = SyncSize(entry), .mark = true, .needs_copy = false};
        NoteGuestWritten(std::span<GuestWrittenRange>(&range, 1));
        if (range.needs_copy) {
            entry.lazy = false;
            return true;
        }
        pending_sync.pop_back();
        return false;
    }

    void ResolveLazy() override {
        lazy_ranges.clear();
        const size_t count = pending_sync.size();
        for (size_t i = 0; i < count; ++i) {
            const PendingSync& entry = pending_sync[i];
            if (!entry.lazy) {
                continue;
            }
            const bool in_flight = inflight && i + 1 == count;
            lazy_ranges.push_back(GuestWrittenRange{.address = entry.address,
                                                    .size = SyncSize(entry),
                                                    .mark = !in_flight,
                                                    .needs_copy = false});
        }
        if (lazy_ranges.empty()) {
            return;
        }
        NoteGuestWritten(lazy_ranges);
        // Addresses whose newest pending value is now marked CPU-written: an older entry for the
        // same address must not be copied after the page upload that carries the newer value.
        superseded.clear();
        for (size_t i = 0, r = 0; i < count; ++i) {
            if (!pending_sync[i].lazy) {
                continue;
            }
            const GuestWrittenRange& range = lazy_ranges[r++];
            const bool in_flight = inflight && i + 1 == count;
            if (!range.needs_copy && !in_flight) {
                superseded.emplace_back(i, pending_sync[i].address);
            }
        }
        const auto is_superseded = [&](size_t i, VAddr address) {
            return std::ranges::any_of(superseded, [&](const auto& s) {
                return s.first > i && s.second == address;
            });
        };
        size_t range_index = 0;
        size_t kept = 0;
        for (size_t i = 0; i < count; ++i) {
            PendingSync entry = pending_sync[i];
            if (entry.lazy) {
                const GuestWrittenRange& range = lazy_ranges[range_index++];
                const bool in_flight = inflight && i + 1 == count;
                if (range.needs_copy) {
                    // Pages with GPU-written data: copied by this wait as before.
                    entry.lazy = false;
                    if (in_flight) {
                        inflight = false;
                    }
                } else if (in_flight) {
                    // Its guest write is still to come: FinishInFlight applies it.
                    inflight_skipped = true;
                } else {
                    continue;
                }
            } else if (is_superseded(i, entry.address)) {
                continue; // a newer release of this address is applied as a CPU write
            }
            pending_sync[kept++] = entry;
        }
        pending_sync.resize(kept);
    }

    bool HasCopyPending() const override {
        const size_t count = pending_sync.size();
        for (size_t i = 0; i < count; ++i) {
            if (!pending_sync[i].lazy || !(inflight && inflight_skipped && i + 1 == count)) {
                return true;
            }
        }
        return false;
    }

    void SyncWrites() override {
        if (pending_sync.empty()) {
            return;
        }
        // The in-flight release a wait for idle left to its report stays pending (ResolveLazy).
        std::optional<PendingSync> kept;
        if (inflight && inflight_skipped && pending_sync.back().lazy) {
            kept = pending_sync.back();
            pending_sync.pop_back();
        } else {
            inflight = false;
        }
        std::vector<SyncValuesStruct> sync_values;
        sync_values.reserve(pending_sync.size());
        std::vector<VAddr> seen;
        seen.reserve(pending_sync.size());
        // Newest first: only the last write of an address counts (one copy command must not
        // write a range twice).
        for (auto it = pending_sync.rbegin(); it != pending_sync.rend(); ++it) {
            if (std::find(seen.begin(), seen.end(), it->address) != seen.end()) {
                continue;
            }
            seen.push_back(it->address);
            auto& query = slot_queries[it->id];
            const bool same = query.guest_address == it->address && query.value == it->value;
            if (same && (True(query.flags & QueryFlagBits::IsRewritten) ||
                         True(query.flags & QueryFlagBits::IsInvalidated))) {
                continue;
            }
            if (same) {
                query.flags |= QueryFlagBits::IsHostSynced;
            }
            sync_values.emplace_back(SyncValuesStruct{
                .address = it->address,
                .value = it->value,
                .size = static_cast<u64>(it->has_timestamp ? 8 : 4)});
        }
        pending_sync.clear();
        if (kept) {
            pending_sync.push_back(*kept);
        }
        if (sync_values.size() > 0) {
            runtime.template SyncValues<SyncValuesStruct>(sync_values);
        }
    }

private:
    struct PendingSync {
        size_t id;
        VAddr address;
        u64 value;
        bool has_timestamp;
        bool lazy{}; ///< EDEN_NX2_QUERY_LAZY_SYNC: the value is (or is about to be) in guest memory
    };

    /// The guest bytes a release wrote: the value and, for long reports, the timestamp.
    static u64 SyncSize(const PendingSync& entry) {
        return entry.has_timestamp ? 16 : 4;
    }

    void NoteGuestWritten(std::span<GuestWrittenRange> ranges) {
        if constexpr (requires { runtime.NoteGuestWrittenRanges(ranges); }) {
            runtime.NoteGuestWrittenRanges(ranges);
        } else {
            for (GuestWrittenRange& range : ranges) {
                range.needs_copy = true;
            }
        }
    }

    RuntimeType& runtime;
    std::deque<PendingSync> pending_sync;
    std::vector<GuestWrittenRange> lazy_ranges;
    std::vector<std::pair<size_t, VAddr>> superseded;
    bool inflight{};         ///< pending_sync.back() is a lazy release still in its report
    bool inflight_skipped{}; ///< A wait for idle left it to FinishInFlight
};

template <typename Traits>
class StubStreamer : public GuestStreamer<Traits> {
public:
    using RuntimeType = typename Traits::RuntimeType;

    StubStreamer(size_t id_, RuntimeType& runtime_, u32 stub_value_)
        : GuestStreamer<Traits>(id_, runtime_), stub_value{stub_value_} {}

    ~StubStreamer() override = default;

    size_t WriteCounter(VAddr address, bool has_timestamp, [[maybe_unused]] u32 value,
                        std::optional<u32> subreport = std::nullopt) override {
        size_t new_id =
            GuestStreamer<Traits>::WriteCounter(address, has_timestamp, stub_value, subreport);
        return new_id;
    }

private:
    u32 stub_value;
};

template <typename Traits>
struct QueryCacheBase<Traits>::QueryCacheBaseImpl {
    using RuntimeType = typename Traits::RuntimeType;

    QueryCacheBaseImpl(QueryCacheBase<Traits>* owner_, VideoCore::RasterizerInterface& rasterizer_,
                       Tegra::MaxwellDeviceMemoryManager& device_memory_, RuntimeType& runtime_,
                       Tegra::GPU& gpu_)
        : owner{owner_}, rasterizer{rasterizer_}, device_memory{device_memory_}, runtime{runtime_},
          gpu{gpu_} {
        streamer_mask = 0;
        for (size_t i = 0; i < static_cast<size_t>(QueryType::MaxQueryTypes); i++) {
            streamers[i] = runtime.GetStreamerInterface(static_cast<QueryType>(i));
            if (streamers[i]) {
                streamer_mask |= 1ULL << streamers[i]->GetId();
            }
        }
    }

    template <typename Func>
    void ForEachStreamerIn(u64 mask, Func&& func) {
        static constexpr bool RETURNS_BOOL =
            std::is_same_v<std::invoke_result<Func, StreamerInterface*>, bool>;
        while (mask != 0) {
            size_t position = std::countr_zero(mask);
            mask &= ~(1ULL << position);
            if constexpr (RETURNS_BOOL) {
                if (func(streamers[position])) {
                    return;
                }
            } else {
                func(streamers[position]);
            }
        }
    }

    template <typename Func>
    void ForEachStreamer(Func&& func) {
        ForEachStreamerIn(streamer_mask, func);
    }

    QueryBase* ObtainQuery(QueryCacheBase<Traits>::QueryLocation location) {
        size_t which_stream = location.stream_id.Value();
        auto* streamer = streamers[which_stream];
        if (!streamer) {
            return nullptr;
        }
        return streamer->GetQuery(location.query_id.Value());
    }

    QueryCacheBase<Traits>* owner;
    VideoCore::RasterizerInterface& rasterizer;
    Tegra::MaxwellDeviceMemoryManager& device_memory;
    RuntimeType& runtime;
    Tegra::GPU& gpu;
    std::array<StreamerInterface*, static_cast<size_t>(QueryType::MaxQueryTypes)> streamers;
    u64 streamer_mask;
    std::mutex flush_guard;
    std::deque<u64> flushes_pending;
    std::vector<QueryCacheBase<Traits>::QueryLocation> pending_unregister;
};

template <typename Traits>
QueryCacheBase<Traits>::QueryCacheBase(Tegra::GPU& gpu_,
                                       VideoCore::RasterizerInterface& rasterizer_,
                                       Tegra::MaxwellDeviceMemoryManager& device_memory_,
                                       RuntimeType& runtime_)
    : cached_queries{} {
    impl = std::make_unique<QueryCacheBase<Traits>::QueryCacheBaseImpl>(
        this, rasterizer_, device_memory_, runtime_, gpu_);
}

template <typename Traits>
QueryCacheBase<Traits>::~QueryCacheBase() = default;

template <typename Traits>
void QueryCacheBase<Traits>::CounterEnable(QueryType counter_type, bool is_enabled) {
    size_t index = static_cast<size_t>(counter_type);
    StreamerInterface* streamer = impl->streamers[index];
    if (!streamer) [[unlikely]] {
        UNREACHABLE();
        return;
    }
    if (is_enabled) {
        streamer->StartCounter();
    } else {
        streamer->PauseCounter();
    }
}

template <typename Traits>
void QueryCacheBase<Traits>::CounterClose(QueryType counter_type) {
    size_t index = static_cast<size_t>(counter_type);
    StreamerInterface* streamer = impl->streamers[index];
    if (!streamer) [[unlikely]] {
        UNREACHABLE();
        return;
    }
    streamer->CloseCounter();
}

template <typename Traits>
void QueryCacheBase<Traits>::CounterReset(QueryType counter_type) {
    size_t index = static_cast<size_t>(counter_type);
    StreamerInterface* streamer = impl->streamers[index];
    if (!streamer) [[unlikely]] {
        UNIMPLEMENTED();
        return;
    }
    streamer->ResetCounter();
}

template <typename Traits>
void QueryCacheBase<Traits>::BindToChannel(s32 id) {
    VideoCommon::ChannelSetupCaches<VideoCommon::ChannelInfo>::BindToChannel(id);
    impl->runtime.Bind3DEngine(maxwell3d);
}

template <typename Traits>
void QueryCacheBase<Traits>::CounterReport(GPUVAddr addr, QueryType counter_type,
                                           QueryPropertiesFlags flags, u32 payload, u32 subreport) {
    const bool has_timestamp = True(flags & QueryPropertiesFlags::HasTimeout);
    const bool is_fence = True(flags & QueryPropertiesFlags::IsAFence);
    const auto requested_type = counter_type;
    static const GPUVAddr watch = [] {
        const char* value = std::getenv("EDEN_NX2_SEM_WATCH");
        return value ? GPUVAddr{std::strtoull(value, nullptr, 16)} : GPUVAddr{};
    }();
    if (static const bool trace = std::getenv("EDEN_NX2_QUERY_TRACE") != nullptr;
        trace || (watch != 0 && (addr & ~GPUVAddr{0xf}) == watch)) {
        LOG_INFO(HW_GPU, "nx2 report addr={:#x} type={} flags={:#x} payload={:#x} as={} device={:#x}", addr,
                 static_cast<u32>(counter_type), static_cast<u32>(flags), payload, gpu_memory->GetID(),
                 gpu_memory->GpuToCpuAddress(addr).value_or(0));
    }
    size_t streamer_id = static_cast<size_t>(counter_type);
    auto* streamer = impl->streamers[streamer_id];
    if (streamer == nullptr) [[unlikely]] {
        counter_type = QueryType::Payload;
        payload = 1U;
        streamer_id = static_cast<size_t>(counter_type);
        streamer = impl->streamers[streamer_id];
    }
    auto cpu_addr_opt = gpu_memory->GpuToCpuAddress(addr);
    if (const u64 trace = QueryTrace::Sample(QueryTrace::Event::Request)) {
        LOG_INFO(HW_GPU,
                 "DKB_QUERY request n={} as={} gpu={:#x} device={:#x} mapped={} type={} "
                 "effective_type={} flags={:#x} payload={:#x} subreport={}",
                 trace, gpu_memory->GetID(), addr, cpu_addr_opt.value_or(0),
                 cpu_addr_opt.has_value(), static_cast<u32>(requested_type),
                 static_cast<u32>(counter_type), static_cast<u32>(flags), payload, subreport);
    }
    if (!cpu_addr_opt) [[unlikely]] {
        return;
    }
    DAddr cpu_addr = *cpu_addr_opt;
    const size_t new_query_id = streamer->WriteCounter(cpu_addr, has_timestamp, payload, subreport);
    auto* query = streamer->GetQuery(new_query_id);
    if (is_fence) {
        query->flags |= QueryFlagBits::IsFence;
    }
    QueryLocation query_location{};
    query_location.stream_id.Assign(static_cast<u32>(streamer_id));
    query_location.query_id.Assign(static_cast<u32>(new_query_id));
    const auto gen_caching_indexing = [](VAddr cur_addr) {
        return std::make_pair<u64, u32>(cur_addr >> Core::DEVICE_PAGEBITS,
                                        static_cast<u32>(cur_addr & Core::DEVICE_PAGEMASK));
    };
    u8* pointer = impl->device_memory.template GetPointer<u8>(cpu_addr);
    u8* pointer_timestamp = impl->device_memory.template GetPointer<u8>(cpu_addr + 8);
    bool is_synced = (Settings::IsGPUFenceBehaviorDefault() ? !Settings::IsGPULevelHigh() : !Settings::IsGPUFenceBehaviorBalanced() && !Settings::IsGPUFenceBehaviorAccurate()) && is_fence;
    std::function<void()> operation([this, is_synced, streamer, query_base = query, query_location,
                                     pointer, pointer_timestamp, cpu_addr, addr,
                                     asid = gpu_memory->GetID(), counter_type] {
        if (True(query_base->flags & QueryFlagBits::IsInvalidated)) {
            if (const u64 trace = QueryTrace::Sample(QueryTrace::Event::Discard)) {
                LOG_INFO(HW_GPU,
                         "DKB_QUERY discard n={} as={} gpu={:#x} device={:#x} type={} "
                         "reason=invalidated flags={:#x}",
                         trace, asid, addr, cpu_addr, static_cast<u32>(counter_type),
                         static_cast<u32>(query_base->flags));
            }
            if (!is_synced) [[likely]] {
                impl->pending_unregister.push_back(query_location);
            }
            return;
        }
        if (False(query_base->flags & QueryFlagBits::IsFinalValueSynced)) [[unlikely]] {
            if (const u64 trace = QueryTrace::Sample(QueryTrace::Event::Discard)) {
                LOG_INFO(HW_GPU,
                         "DKB_QUERY discard n={} as={} gpu={:#x} device={:#x} type={} "
                         "reason=unsynchronized flags={:#x}",
                         trace, asid, addr, cpu_addr, static_cast<u32>(counter_type),
                         static_cast<u32>(query_base->flags));
            }
            LOG_ERROR(HW_GPU,
                      "Query report value not synchronized. Consider increasing GPU accuracy.");
            if (!is_synced) [[likely]] {
                impl->pending_unregister.push_back(query_location);
            }
            return;
        }
        query_base->value += streamer->GetAmendValue();
        streamer->SetAccumulationValue(query_base->value);
        if (True(query_base->flags & QueryFlagBits::HasTimestamp)) {
            u64 timestamp = impl->gpu.GetTicks();
            std::memcpy(pointer_timestamp, &timestamp, sizeof(timestamp));
            std::memcpy(pointer, &query_base->value, sizeof(query_base->value));
        } else {
            u32 value = static_cast<u32>(query_base->value);
            std::memcpy(pointer, &value, sizeof(value));
        }
        Tegra::Nx2SemaphoreWaiters::Get().OnWrite(cpu_addr, static_cast<u32>(query_base->value));
        if (const u64 trace = QueryTrace::Sample(QueryTrace::Event::Write)) {
            const bool wide = True(query_base->flags & QueryFlagBits::HasTimestamp);
            LOG_INFO(HW_GPU,
                     "DKB_QUERY write n={} path=completion as={} gpu={:#x} device={:#x} type={} "
                     "bytes={} value={:#x} flags={:#x}",
                     trace, asid, addr, cpu_addr, static_cast<u32>(counter_type), wide ? 8 : 4,
                     wide ? query_base->value : static_cast<u32>(query_base->value),
                     static_cast<u32>(query_base->flags));
        }
        if (!is_synced) [[likely]] {
            impl->pending_unregister.push_back(query_location);
        }
    });
    // EDEN_NX2_QUERY_LAZY_SYNC: whether guest memory holds the release's value (the fence ran
    // its operation at once, or the write below).
    const bool lazy_payload = counter_type == QueryType::Payload && LazyPayloadSync();
    const auto guest_holds_value = [pointer, has_timestamp, payload] {
        if (pointer == nullptr) {
            return false;
        }
        if (has_timestamp) {
            u64 value{};
            std::memcpy(&value, pointer, sizeof(value));
            return value == static_cast<u64>(payload);
        }
        u32 value{};
        std::memcpy(&value, pointer, sizeof(value));
        return value == payload;
    };
    if (is_fence) {
        const bool lazy_fence = lazy_payload && is_synced;
        if (lazy_fence) {
            streamer->MarkInFlight(new_query_id);
        }
        impl->rasterizer.SignalFence(std::move(operation));
        if (lazy_fence && streamer->FinishInFlight(new_query_id, guest_holds_value())) {
            NotifyWFI();
        }
    } else {
        if (!Settings::IsGPULevelHigh() && counter_type == QueryType::Payload) {
            if (has_timestamp) {
                u64 timestamp = impl->gpu.GetTicks();
                u64 value = static_cast<u64>(payload);
                std::memcpy(pointer_timestamp, &timestamp, sizeof(timestamp));
                std::memcpy(pointer, &value, sizeof(value));
            } else {
                std::memcpy(pointer, &payload, sizeof(payload));
            }
            Tegra::Nx2SemaphoreWaiters::Get().OnWrite(cpu_addr, payload);
            if (const u64 trace = QueryTrace::Sample(QueryTrace::Event::Write)) {
                LOG_INFO(HW_GPU,
                         "DKB_QUERY write n={} path=immediate as={} gpu={:#x} device={:#x} type={} "
                         "bytes={} value={:#x} flags={:#x}",
                         trace, gpu_memory->GetID(), addr, cpu_addr,
                         static_cast<u32>(counter_type), has_timestamp ? 8 : 4, payload,
                         static_cast<u32>(query->flags));
            }
            if (lazy_payload) {
                // Written: applied with the next wait for idle, as its copy would have been.
                streamer->MarkInFlight(new_query_id);
                streamer->FinishInFlight(new_query_id, guest_holds_value());
            }
            streamer->Free(new_query_id);
            return;
        }
        impl->rasterizer.SyncOperation(std::move(operation));
    }
    if (is_synced) {
        streamer->Free(new_query_id);
        return;
    }
    auto [cont_addr, base] = gen_caching_indexing(cpu_addr);
    {
        std::scoped_lock lock(cache_mutex);
        auto it1 = cached_queries.try_emplace(cont_addr);
        page_filter.Mark(cpu_addr, 16);
        auto& sub_container = it1.first->second;
        auto it_current = sub_container.find(base);
        if (it_current == sub_container.end()) {
            sub_container.insert_or_assign(base, query_location);
            return;
        }
        auto* old_query = impl->ObtainQuery(it_current->second);
        old_query->flags |= QueryFlagBits::IsRewritten;
        sub_container.insert_or_assign(base, query_location);
    }
}

template <typename Traits>
void QueryCacheBase<Traits>::UnregisterPending() {
    const auto gen_caching_indexing = [](VAddr cur_addr) {
        return std::make_pair<u64, u32>(cur_addr >> Core::DEVICE_PAGEBITS,
                                        static_cast<u32>(cur_addr & Core::DEVICE_PAGEMASK));
    };
    std::scoped_lock lock(cache_mutex);
    for (QueryLocation loc : impl->pending_unregister) {
        const auto [streamer_id, query_id] = loc.unpack();
        auto* streamer = impl->streamers[streamer_id];
        if (!streamer) [[unlikely]] {
            continue;
        }
        auto* query = streamer->GetQuery(query_id);
        auto [cont_addr, base] = gen_caching_indexing(query->guest_address);
        auto it1 = cached_queries.find(cont_addr);
        if (it1 != cached_queries.end()) {
            auto it2 = it1->second.find(base);
            if (it2 != it1->second.end()) {
                if (it2->second.raw == loc.raw) {
                    it1->second.erase(it2);
                }
            }
        }
        streamer->Free(query_id);
    }
    impl->pending_unregister.clear();
}

template <typename Traits>
void QueryCacheBase<Traits>::NotifyWFI() {
    bool should_sync = false;
    impl->ForEachStreamer(
        [&should_sync](StreamerInterface* streamer) { should_sync |= streamer->HasPendingSync(); });
    if (!should_sync) {
        return;
    }
    if (LazyPayloadSync()) {
        // Payload releases already in guest memory go to the buffer cache as guest CPU writes:
        // without anything to copy, the render pass stays open (EDEN_NX2_QUERY_LAZY_SYNC).
        StreamerInterface* const payload = impl->streamers[static_cast<size_t>(QueryType::Payload)];
        if (payload != nullptr) {
            payload->ResolveLazy();
        }
        should_sync = false;
        impl->ForEachStreamer([&should_sync, payload](StreamerInterface* streamer) {
            should_sync |=
                streamer == payload ? streamer->HasCopyPending() : streamer->HasPendingSync();
        });
        if (!should_sync) {
            return;
        }
    }

    impl->ForEachStreamer([](StreamerInterface* streamer) { streamer->PresyncWrites(); });
    impl->runtime.Barriers(true);
    impl->ForEachStreamer([](StreamerInterface* streamer) { streamer->SyncWrites(); });
    impl->runtime.Barriers(false);
}

template <typename Traits>
void QueryCacheBase<Traits>::NotifySegment(bool resume) {
    if (resume) {
        impl->runtime.ResumeHostConditionalRendering();
    } else {
        impl->runtime.PauseHostConditionalRendering();
    }
}

template <typename Traits>
bool QueryCacheBase<Traits>::AccelerateHostConditionalRendering() {
    bool qc_dirty = false;
    const auto gen_lookup = [this, &qc_dirty](GPUVAddr address) -> VideoCommon::LookupData {
        auto cpu_addr_opt = gpu_memory->GpuToCpuAddress(address);
        if (!cpu_addr_opt) [[unlikely]] {
            return VideoCommon::LookupData{
                .address = 0,
                .found_query = nullptr,
            };
        }
        VAddr cpu_addr = *cpu_addr_opt;
        std::scoped_lock lock(cache_mutex);
        auto it1 = cached_queries.find(cpu_addr >> Core::DEVICE_PAGEBITS);
        if (it1 == cached_queries.end()) {
            return VideoCommon::LookupData{
                .address = cpu_addr,
                .found_query = nullptr,
            };
        }
        auto& sub_container = it1->second;
        auto it_current = sub_container.find(cpu_addr & Core::DEVICE_PAGEMASK);

        if (it_current == sub_container.end()) {
            auto it_current_2 = sub_container.find((cpu_addr & Core::DEVICE_PAGEMASK) + 4);
            if (it_current_2 == sub_container.end()) {
                return VideoCommon::LookupData{
                    .address = cpu_addr,
                    .found_query = nullptr,
                };
            }
            it_current = it_current_2;
        }
        auto* query = impl->ObtainQuery(it_current->second);
        qc_dirty |= True(query->flags & QueryFlagBits::IsHostManaged) &&
                    False(query->flags & QueryFlagBits::IsGuestSynced);
        return VideoCommon::LookupData{
            .address = cpu_addr,
            .found_query = query,
        };
    };

    auto& regs = maxwell3d->regs;
    if (regs.render_enable_override != Maxwell::Regs::RenderEnable::Override::UseRenderEnable) {
        impl->runtime.EndHostConditionalRendering();
        return false;
    }
    const ComparisonMode mode = static_cast<ComparisonMode>(regs.render_enable.mode);
    const GPUVAddr address = regs.render_enable.Address();
    switch (mode) {
    case ComparisonMode::True:
        impl->runtime.EndHostConditionalRendering();
        return false;
    case ComparisonMode::False:
        impl->runtime.EndHostConditionalRendering();
        return false;
    case ComparisonMode::Conditional: {
        VideoCommon::LookupData object_1{gen_lookup(address)};
        return impl->runtime.HostConditionalRenderingCompareValue(object_1, qc_dirty);
    }
    case ComparisonMode::IfEqual: {
        VideoCommon::LookupData object_1{gen_lookup(address)};
        VideoCommon::LookupData object_2{gen_lookup(address + 16)};
        return impl->runtime.HostConditionalRenderingCompareValues(object_1, object_2, qc_dirty,
                                                                   true);
    }
    case ComparisonMode::IfNotEqual: {
        VideoCommon::LookupData object_1{gen_lookup(address)};
        VideoCommon::LookupData object_2{gen_lookup(address + 16)};
        return impl->runtime.HostConditionalRenderingCompareValues(object_1, object_2, qc_dirty,
                                                                   false);
    }
    default:
        return false;
    }
}

// Async downloads
template <typename Traits>
void QueryCacheBase<Traits>::CommitAsyncFlushes() {
    // Make sure to have the results synced in Host.
    NotifyWFI();

    u64 mask{};
    {
        std::scoped_lock lk(impl->flush_guard);
        impl->ForEachStreamer([&mask](StreamerInterface* streamer) {
            bool local_result = streamer->HasUnsyncedQueries();
            if (local_result) {
                mask |= 1ULL << streamer->GetId();
            }
        });
        impl->flushes_pending.push_back(mask);
    }
    std::function<void()> func([this] { UnregisterPending(); });
    impl->rasterizer.SyncOperation(std::move(func));
    if (mask == 0) {
        return;
    }
    u64 ran_mask = ~mask;
    while (mask) {
        impl->ForEachStreamerIn(mask, [&mask, &ran_mask](StreamerInterface* streamer) {
            u64 dep_mask = streamer->GetDependentMask();
            if ((dep_mask & ~ran_mask) != 0) {
                return;
            }
            u64 index = streamer->GetId();
            ran_mask |= (1ULL << index);
            mask &= ~(1ULL << index);
            streamer->PushUnsyncedQueries();
        });
    }
}

template <typename Traits>
bool QueryCacheBase<Traits>::HasUncommittedFlushes() const {
    bool result = false;
    impl->ForEachStreamer([&result](StreamerInterface* streamer) {
        result |= streamer->HasUnsyncedQueries();
        return result;
    });
    return result;
}

template <typename Traits>
bool QueryCacheBase<Traits>::HasPendingFenceWork(bool ignore_payload_sync) const {
    bool result = false;
    StreamerInterface* const payload = impl->streamers[static_cast<size_t>(QueryType::Payload)];
    impl->ForEachStreamer([&](StreamerInterface* streamer) {
        result |= streamer->HasUnsyncedQueries() ||
                  ((!ignore_payload_sync || streamer != payload) && streamer->HasPendingSync());
        return result;
    });
    return result;
}

template <typename Traits>
bool QueryCacheBase<Traits>::ShouldWaitAsyncFlushes() {
    std::scoped_lock lk(impl->flush_guard);
    return !impl->flushes_pending.empty() && impl->flushes_pending.front() != 0ULL;
}

template <typename Traits>
void QueryCacheBase<Traits>::PopAsyncFlushes() {
    u64 mask;
    {
        std::scoped_lock lk(impl->flush_guard);
        mask = impl->flushes_pending.front();
        impl->flushes_pending.pop_front();
    }
    if (mask == 0) {
        return;
    }
    u64 ran_mask = ~mask;
    while (mask) {
        impl->ForEachStreamerIn(mask, [&mask, &ran_mask](StreamerInterface* streamer) {
            u64 dep_mask = streamer->GetDependenceMask();
            if ((dep_mask & ~ran_mask) != 0) {
                return;
            }
            u64 index = streamer->GetId();
            ran_mask |= (1ULL << index);
            mask &= ~(1ULL << index);
            streamer->PopUnsyncedQueries();
        });
    }
}

// Invalidation

template <typename Traits>
void QueryCacheBase<Traits>::InvalidateQuery(QueryCacheBase<Traits>::QueryLocation location) {
    auto* query_base = impl->ObtainQuery(location);
    if (!query_base) {
        return;
    }
    query_base->flags |= QueryFlagBits::IsInvalidated;
}

template <typename Traits>
bool QueryCacheBase<Traits>::IsQueryDirty(QueryCacheBase<Traits>::QueryLocation location) {
    auto* query_base = impl->ObtainQuery(location);
    if (!query_base) {
        return false;
    }
    return True(query_base->flags & QueryFlagBits::IsHostManaged) &&
           False(query_base->flags & QueryFlagBits::IsGuestSynced);
}

template <typename Traits>
bool QueryCacheBase<Traits>::SemiFlushQueryDirty(QueryCacheBase<Traits>::QueryLocation location) {
    auto* query_base = impl->ObtainQuery(location);
    if (!query_base) {
        return false;
    }
    if (True(query_base->flags & QueryFlagBits::IsFinalValueSynced) &&
        False(query_base->flags & QueryFlagBits::IsGuestSynced)) {
        auto* ptr = impl->device_memory.template GetPointer<u8>(query_base->guest_address);
        if (True(query_base->flags & QueryFlagBits::HasTimestamp)) {
            std::memcpy(ptr, &query_base->value, sizeof(query_base->value));
        } else {
            u32 value_l = static_cast<u32>(query_base->value);
            std::memcpy(ptr, &value_l, sizeof(value_l));
        }
        if (const u64 trace = QueryTrace::Sample(QueryTrace::Event::Write)) {
            const bool wide = True(query_base->flags & QueryFlagBits::HasTimestamp);
            LOG_INFO(HW_GPU,
                     "DKB_QUERY write n={} path=semi-flush device={:#x} stream={} slot={} "
                     "bytes={} value={:#x} flags={:#x}",
                     trace, query_base->guest_address, location.stream_id.Value(),
                     location.query_id.Value(), wide ? 8 : 4,
                     wide ? query_base->value : static_cast<u32>(query_base->value),
                     static_cast<u32>(query_base->flags));
        }
        return false;
    }
    return True(query_base->flags & QueryFlagBits::IsHostManaged) &&
           False(query_base->flags & QueryFlagBits::IsGuestSynced);
}

template <typename Traits>
void QueryCacheBase<Traits>::RequestGuestHostSync() {
    impl->rasterizer.ReleaseFences();
}

} // namespace VideoCommon
