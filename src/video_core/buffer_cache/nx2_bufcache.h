// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(_M_X64) || defined(__x86_64__) || defined(__SSE2__)
#include <emmintrin.h>
#define NX2_BUFCACHE_SSE2 1
#endif

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/buffer_cache/word_manager.h"
#include "video_core/nx2_profile.h"

namespace VideoCommon::Nx2BufCache {

[[nodiscard]] inline bool EnvOn(const char* name) {
    const char* const value = std::getenv(name);
    return value == nullptr || value[0] != '0';
}

/// EDEN_NX2_BUFCACHE_CONTIG=0: storage bindings always ask the virtual range cache for their
/// segments, even when the mapping runs show the range is one device range.
inline const bool g_contig_resolve = EnvOn("EDEN_NX2_BUFCACHE_CONTIG");

/// EDEN_NX2_BUFCACHE_BANK=0: a native program's 64 KiB uniform bank is re-uploaded on its first
/// bind of the frame even when that bind streams the buffer from guest memory.
inline const bool g_bank_skip = EnvOn("EDEN_NX2_BUFCACHE_BANK");

/// EDEN_NX2_BUFCACHE_BANKSHADOW=0: that once-a-frame upload copies the whole bank, unchanged
/// pages included (every such copy of a bank the frame already used ends the render pass).
inline const bool g_bank_shadow = EnvOn("EDEN_NX2_BUFCACHE_BANKSHADOW");

/// EDEN_NX2_INLINE_FAST=0: inline writes within one 4 KiB page always walk the GPU page table.
inline const bool g_inline_fast = EnvOn("EDEN_NX2_INLINE_FAST");

/// EDEN_NX2_BUFCACHE_POOLS=0: a storage pointer missing the binding cache always looks its
/// mapping up, even inside a pool (kind range and device run) a previous pointer looked up.
inline const bool g_pool_ranges = EnvOn("EDEN_NX2_BUFCACHE_POOLS");

/// EDEN_NX2_BUFCACHE_GEOMEMO=0: vertex and index buffers are looked up again whenever their
/// registers are dirty, even when they hold the same range.
inline const bool g_geometry_memo = EnvOn("EDEN_NX2_BUFCACHE_GEOMEMO");

/// EDEN_NX2_HOT_PAGES=0: pages the guest rewrites between most syncs are protected again after
/// each upload (a VirtualProtect per sync and a guest fault per write) instead of being kept
/// unprotected and compared at each sync.
inline const bool g_hot_pages = EnvOn("EDEN_NX2_HOT_PAGES");

/// EDEN_NX2_COMPUTE_BINDCACHE=0: compute dispatches look their constant buffers up and copy
/// them to the stream buffer every time, even when unchanged since the previous dispatch.
inline const bool g_compute_bindcache = EnvOn("EDEN_NX2_COMPUTE_BINDCACHE");

/// EDEN_NX2_I2M_COALESCE=0: every inline write takes the buffer cache lock to check its page
/// for GPU-written data, even right after a write that found the same page clean.
inline const bool g_i2m_coalesce = EnvOn("EDEN_NX2_I2M_COALESCE");

/// EDEN_NX2_UBO_REBIND=0: every graphics uniform buffer bind translates its GPU address again and
/// makes the next draw look its buffer up again, even when it binds the slot's current range.
inline const bool g_ubo_rebind = EnvOn("EDEN_NX2_UBO_REBIND");

/// EDEN_NX2_SYNC_SKIP=0: every sync walks the tracker over its range, even one left clean by
/// an earlier sync with no CPU write marked in the buffer since.
inline const bool g_sync_skip = EnvOn("EDEN_NX2_SYNC_SKIP");

/// EDEN_NX2_BUFCACHE_MARK=0: GPU writes re-add ranges that are already GPU-modified.
inline const bool g_mark_guards = EnvOn("EDEN_NX2_BUFCACHE_MARK");

/// EDEN_NX2_HOT_DEMOTE=0: a hot page stays hot until 64 unchanged syncs over 4 frames (which a
/// page rewritten once a frame never reaches), instead of being protected again after
/// HOT_DEMOTE_UNCHANGED unchanged syncs in a row and kept out of the hot set for
/// HOT_COOLDOWN_FRAMES frames.
inline const bool g_hot_demote = EnvOn("EDEN_NX2_HOT_DEMOTE");

/// EDEN_NX2_HOT_SYNC_SKIP=0: a range holding a hot page is walked at every sync (no SYNC_SKIP
/// entry), instead of keeping the entry and comparing only its hot pages.
inline const bool g_hot_sync_skip = EnvOn("EDEN_NX2_HOT_SYNC_SKIP");

/// EDEN_NX2_SKIP_REGIONS=0: a SYNC_SKIP entry is dropped by any CPU-dirty mark in its buffer
/// (the whole 100 MiB heap buffer each draw in OoT), instead of only by marks in the 256 KiB
/// regions the range covers.
inline const bool g_skip_regions = EnvOn("EDEN_NX2_SKIP_REGIONS");

/// EDEN_NX2_PAGE_CMP=0: hot page and bank shadow compares call memcmp instead of the inline
/// SSE2 loop.
inline const bool g_page_compare = EnvOn("EDEN_NX2_PAGE_CMP");

/// EDEN_NX2_SHADOW_REFRESH=0: a bank upload invalidates the hot page copies it overwrites (and a
/// hot upload the bank shadow pages), so the next sync uploads the page again, instead of
/// taking the uploaded bytes as the new copy.
inline const bool g_shadow_refresh = EnvOn("EDEN_NX2_SHADOW_REFRESH");

/// EDEN_NX2_STORAGE_MEMO=0: when a stage's storage bindings miss the resolve cache, every
/// binding is looked up again, instead of reusing the ones unchanged since their last lookup.
inline const bool g_storage_memo = EnvOn("EDEN_NX2_STORAGE_MEMO");

/// EDEN_NX2_SYNC_PRECHECK=0: a sync whose SYNC_SKIP entry is gone always walks the tracker,
/// instead of first checking (read only) that no page of its range is CPU-dirty or unprotected,
/// in which case the walk would find and change nothing.
inline const bool g_sync_precheck = EnvOn("EDEN_NX2_SYNC_PRECHECK");

/// EDEN_NX2_DIRTY_WRITE_SKIP=0: a write over pages that are all CPU-dirty already (constant
/// buffer and inline writes, before nearly every draw) marks them again and bumps the SYNC_SKIP
/// generations of its regions and buffers, instead of only noting vertex buffer staleness. No
/// SYNC_SKIP entry can cover a CPU-dirty page that is not hot (a walk or precheck that stores one
/// leaves its range clean, and a page turning dirty bumps the generations), and hot pages are
/// compared on every hit, so the bump only drops entries that were still right.
inline const bool g_dirty_write_skip = EnvOn("EDEN_NX2_DIRTY_WRITE_SKIP");

/// EDEN_NX2_WRITE_GPU_BITMAP=0: BufferCache::IsRegionGpuModified (every write and inline upload)
/// always asks gpu_modified_ranges (a tree lookup), instead of first asking the GPU-written page
/// bitmap, which marks every page such a range ever covered (Switch 2 sessions: both add sites
/// mark it), for ranges up to 64 KiB.
inline const bool g_write_gpu_bitmap = EnvOn("EDEN_NX2_WRITE_GPU_BITMAP");

/// EDEN_NX2_BANK_SYNC_FIRST=0: a classic bind compares (and uploads) its bank before syncing
/// it, so a page the tracker saw written is uploaded by both, instead of syncing first.
inline const bool g_bank_sync_first = EnvOn("EDEN_NX2_BANK_SYNC_FIRST");

/// EDEN_NX2_BANK_SYNC_REFRESH=0: a sync upload over a bank shadow invalidates its pages, so the
/// next bank compare uploads them again unchanged, instead of the shadow taking the bytes the
/// sync uploaded.
inline const bool g_bank_sync_refresh = EnvOn("EDEN_NX2_BANK_SYNC_REFRESH");

/// EDEN_NX2_BANK_LRU=0: a new bank beyond the shadow cap drops every bank shadow (each bank
/// then uploads whole again), instead of only the least recently compared one.
inline const bool g_bank_lru = EnvOn("EDEN_NX2_BANK_LRU");

/// Whether [a, a + size) and [b, b + size) hold the same bytes. Inline (memcmp is an out-of-line
/// ucrt call that clang does not expand for 4 KiB), 128 bytes per step with an early exit.
[[nodiscard]] inline bool BytesEqual(const u8* a, const u8* b, size_t size) {
#ifdef NX2_BUFCACHE_SSE2
    if (g_page_compare) {
        size_t offset = 0;
        for (; offset + 128 <= size; offset += 128) {
            const auto load = [&](const u8* base, size_t index) {
                return _mm_loadu_si128(reinterpret_cast<const __m128i*>(base + offset + index * 16));
            };
            __m128i acc = _mm_xor_si128(load(a, 0), load(b, 0));
            acc = _mm_or_si128(acc, _mm_xor_si128(load(a, 1), load(b, 1)));
            acc = _mm_or_si128(acc, _mm_xor_si128(load(a, 2), load(b, 2)));
            acc = _mm_or_si128(acc, _mm_xor_si128(load(a, 3), load(b, 3)));
            acc = _mm_or_si128(acc, _mm_xor_si128(load(a, 4), load(b, 4)));
            acc = _mm_or_si128(acc, _mm_xor_si128(load(a, 5), load(b, 5)));
            acc = _mm_or_si128(acc, _mm_xor_si128(load(a, 6), load(b, 6)));
            acc = _mm_or_si128(acc, _mm_xor_si128(load(a, 7), load(b, 7)));
            if (_mm_movemask_epi8(_mm_cmpeq_epi8(acc, _mm_setzero_si128())) != 0xFFFF) {
                return false;
            }
        }
        return offset == size || std::memcmp(a + offset, b + offset, size - offset) == 0;
    }
#endif
    return std::memcmp(a, b, size) == 0;
}

/// Whether one range of `set` covers [addr, addr + size) (interval sets join touching ranges).
template <typename Set, typename Addr>
[[nodiscard]] bool RangeCovered(const Set& set, Addr addr, u64 size) {
    if (size == 0) {
        return false;
    }
    u32 pieces = 0;
    bool covered = false;
    set.ForEachInRange(addr, size, [&](Addr lower, Addr upper) {
        ++pieces;
        covered = lower == addr && upper == addr + size;
    });
    return pieces == 1 && covered;
}

/// Device addresses seen this frame. clear() forgets them all without touching the table.
class FrameAddressSet {
public:
    /// True when `key` (non-zero) was not in the set this frame; it is now.
    bool insert(u64 key) {
        if (count * 2 >= table.size()) {
            Grow();
        }
        Entry& entry = Probe(key);
        if (entry.stamp == stamp && entry.key == key) {
            return false;
        }
        entry = Entry{key, stamp};
        ++count;
        return true;
    }

    [[nodiscard]] bool contains(u64 key) const {
        if (table.empty()) {
            return false;
        }
        const size_t mask = table.size() - 1;
        for (size_t index = Hash(key) & mask;; index = (index + 1) & mask) {
            const Entry& entry = table[index];
            if (entry.stamp != stamp) {
                return false;
            }
            if (entry.key == key) {
                return true;
            }
        }
    }

    void clear() {
        count = 0;
        if (++stamp == 0) {
            for (Entry& entry : table) {
                entry.stamp = 0;
            }
            stamp = 1;
        }
    }

private:
    struct Entry {
        u64 key{};
        u32 stamp{};
    };

    static size_t Hash(u64 key) {
        return static_cast<size_t>((key * 0x9E3779B97F4A7C15ULL) >> 32);
    }

    Entry& Probe(u64 key) {
        const size_t mask = table.size() - 1;
        for (size_t index = Hash(key) & mask;; index = (index + 1) & mask) {
            Entry& entry = table[index];
            if (entry.stamp != stamp || entry.key == key) {
                return entry;
            }
        }
    }

    void Grow() {
        std::vector<Entry> old = std::move(table);
        table.assign(old.empty() ? 256 : old.size() * 2, Entry{});
        const u32 live = stamp;
        stamp = 1;
        count = 0;
        for (const Entry& entry : old) {
            if (entry.stamp == live) {
                Probe(entry.key) = Entry{entry.key, stamp};
                ++count;
            }
        }
    }

    std::vector<Entry> table;
    size_t count{};
    u32 stamp{1};
};

/// Buffer cache paths taken, logged every 10 s (from TickFrame) with EDEN_NX2_GPU_PROFILE.
enum class Stat : u32 {
    UboStream,
    UboClassic,
    UboClassicUpload,
    BankUpload,
    BankUploadBytes,
    BankSkip,
    SyncUpload,
    SyncUploadBytes,
    ResolveContig,
    ResolveQuery,
    MarkWritten,
    MarkAddSkipped,
    InlineFast,
    InlineSlow,
    HotPromote,
    HotDemote,
    HotSame,
    HotUpload,
    HotCostDemote,
    SyncSkipHit,
    SyncSkipHotHit,
    SyncWalk,
    StorageMemoHit,
    UboRebindKept,
    UboSlotHit,     ///< stream: the slot's copy bound again (same bytes, same command buffer)
    UboRangeHit,    ///< stream: another bind's copy of the range bound again (UBO_RANGE_REUSE)
    UboRecopied,    ///< stream: same bytes as an older command buffer's copy, copied from it
    UboCopied,      ///< stream: guest bytes copied
    UboCopiedBytes, ///< bytes of UboRecopied and UboCopied
    UboLarge,       ///< stream binds over 4 KiB (misaligned buffers): copied every time
    UboLargeBytes,
    SyncPrecheckClean, ///< syncs a SYNC_SKIP miss sent to the walk, found clean by the precheck
    WriteMemoryCalls,  ///< BufferCache::WriteMemory calls
    DirtyWriteSkip,    ///< of those, over pages all CPU-dirty already (EDEN_NX2_DIRTY_WRITE_SKIP)
    UboClassicBank,    ///< classic binds of a native 64 KiB bank
    UboClassicSmall,   ///< classic binds within the stream size (GPU-written, or no host buffer)
    BankCompare,       ///< UploadNativeBank calls (once a frame per bank)
    BankComparePages,  ///< bank pages compared with their shadow
    BankPageRepeat,    ///< of those, pages another bank already compared this frame
    BankShadowDrop,    ///< bank shadows dropped at the cap (all of them with EDEN_NX2_BANK_LRU=0)
    BankSyncRefresh,   ///< bank shadow pages a sync upload refreshed (EDEN_NX2_BANK_SYNC_REFRESH)
    Count,
};

inline std::array<std::atomic<u64>, static_cast<size_t>(Stat::Count)> g_stats{};
inline u64 g_stats_start{};

inline void Note(Stat stat, u64 amount = 1) {
    if (Nx2Profile::Enabled()) [[unlikely]] {
        g_stats[static_cast<size_t>(stat)].fetch_add(amount, std::memory_order_relaxed);
    }
}

inline void Report() {
    if (!Nx2Profile::Enabled()) {
        return;
    }
    const u64 now = Nx2Profile::Now();
    if (g_stats_start == 0) {
        g_stats_start = now;
        return;
    }
    const u64 window = now - g_stats_start;
    if (window < 10'000'000'000ULL) {
        return;
    }
    std::array<u64, static_cast<size_t>(Stat::Count)> stats{};
    for (size_t i = 0; i < stats.size(); ++i) {
        stats[i] = g_stats[i].exchange(0, std::memory_order_relaxed);
    }
    const auto rate = [&](Stat stat) {
        return static_cast<double>(stats[static_cast<size_t>(stat)]) * 1e9 / window;
    };
    LOG_INFO(HW_GPU,
             "Buffer cache per second: UBO stream {:.0f}, classic {:.0f} ({:.0f} uploading) | "
             "bank uploads {:.0f} ({:.0f} KiB), skipped {:.0f} | sync uploads {:.0f} ({:.0f} KiB) "
             "| storage resolve contiguous {:.0f}, queried {:.0f} | GPU writes {:.0f} ({:.0f} "
             "already marked) | inline writes direct {:.0f}, walked {:.0f} | hot pages "
             "promoted {:.0f}, demoted {:.0f} ({:.0f} by cost), unchanged {:.0f}, uploaded {:.0f} | "
             "sync skipped {:.0f} ({:.0f} comparing hot pages), walked {:.0f} | storage memo "
             "hits {:.0f} | UBO rebinds kept {:.0f}",
             rate(Stat::UboStream), rate(Stat::UboClassic), rate(Stat::UboClassicUpload),
             rate(Stat::BankUpload),
             rate(Stat::BankUploadBytes) / 1024.0, rate(Stat::BankSkip), rate(Stat::SyncUpload),
             rate(Stat::SyncUploadBytes) / 1024.0, rate(Stat::ResolveContig),
             rate(Stat::ResolveQuery), rate(Stat::MarkWritten), rate(Stat::MarkAddSkipped),
             rate(Stat::InlineFast), rate(Stat::InlineSlow), rate(Stat::HotPromote),
             rate(Stat::HotDemote), rate(Stat::HotCostDemote), rate(Stat::HotSame),
             rate(Stat::HotUpload), rate(Stat::SyncSkipHit), rate(Stat::SyncSkipHotHit),
             rate(Stat::SyncWalk), rate(Stat::StorageMemoHit),
             rate(Stat::UboRebindKept));
    const auto protects = Nx2TrackerStats::Take();
    LOG_INFO(HW_GPU,
             "Buffer cache per second: sync precheck clean {:.0f} | UBO stream: slot hits {:.0f}, "
             "range hits {:.0f}, recopied {:.0f}, copied {:.0f} ({:.0f} KiB), over 4 KiB {:.0f} "
             "({:.0f} KiB) | tracker protects {:.0f} ({:.0f} pages), unprotects {:.0f} ({:.0f} "
             "pages) | WriteMemory {:.0f}, on dirty pages only {:.0f}",
             rate(Stat::SyncPrecheckClean), rate(Stat::UboSlotHit),
             rate(Stat::UboRangeHit), rate(Stat::UboRecopied), rate(Stat::UboCopied),
             rate(Stat::UboCopiedBytes) / 1024.0, rate(Stat::UboLarge),
             rate(Stat::UboLargeBytes) / 1024.0,
             static_cast<double>(protects[0]) * 1e9 / static_cast<double>(window),
             static_cast<double>(protects[1]) * 1e9 / static_cast<double>(window),
             static_cast<double>(protects[2]) * 1e9 / static_cast<double>(window),
             static_cast<double>(protects[3]) * 1e9 / static_cast<double>(window),
             rate(Stat::WriteMemoryCalls), rate(Stat::DirtyWriteSkip));
    LOG_INFO(HW_GPU,
             "Buffer cache banks per second: classic binds of banks {:.0f}, within the stream "
             "size {:.0f} | bank compares {:.0f}, pages compared {:.0f}, already compared this "
             "frame {:.0f} | shadows dropped {:.0f} | pages refreshed by syncs {:.0f}",
             rate(Stat::UboClassicBank), rate(Stat::UboClassicSmall), rate(Stat::BankCompare),
             rate(Stat::BankComparePages), rate(Stat::BankPageRepeat),
             rate(Stat::BankShadowDrop), rate(Stat::BankSyncRefresh));
    g_stats_start = now;
}

} // namespace VideoCommon::Nx2BufCache
