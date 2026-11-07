// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#define ZSTD_STATIC_LINKING_ONLY
#include <zstd.h>

#include "common/logging.h"
#include "common/scratch_buffer.h"
#include "core/arm/arm_interface.h"
#include "core/core.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/svc/svc_nx2_zstd.h"
#include "core/hle/kernel/svc_types.h"
#include "core/memory.h"

namespace Kernel::Svc {
namespace {

// Ocarina of Time (main build E0F3CCFA...): the asset streaming jobs decompress with
// ZSTD_decompressMultiFrame(dctx, dst, dstCapacity, src, srcSize, dict, dictSize, ddict) at
// main+0xEB380. Each of its three callers re-initialises a static DCtx (default parameters, format
// zstd1) right before the call and only uses the return value, so a default host DCtx decodes the
// same bytes. svc 0xF0 + i replaces the bl at kCallSites[i].
constexpr u64 kOotProgramId = 0x0400486024DEC000ULL;
constexpr u64 kMultiFrameOffset = 0xEB380;
constexpr std::array<u64, 3> kCallSites{0xBF72E0, 0xF50164, 0xE910C};
constexpr u32 kFirstSvc = 0xF0;
constexpr size_t kMaxSize = size_t{512} << 20;

// The guest ZSTD_DDict (0x6860 bytes, from ZSTD_createDDict_advanced at main+0x348B740 and
// ZSTD_initDDict_internal at main+0x348B850):
//   +0x0000 dictBuffer, +0x0008 dictContent, +0x0010 dictSize, +0x0018 entropy tables
//   (LL +0x18, OF +0x1020, ML +0x1828, HUF +0x2830, rep[3] +0x6834), +0x6840 dictID,
//   +0x6844 entropyPresent, +0x6848 customMem.
// ZSTD_decompressBegin_usingDDict (main+0x348B610) reads exactly these fields. The game creates
// its dictionaries (Pack/ZsDic.pack) with ZSTD_dlm_byRef and ZSTD_dct_auto (main+0x3B91638) and
// picks the ddict by the frame header's dictID (main+0xE8FB4 -> main+0x348C540).
constexpr u64 kDDictContent = 0x8;
constexpr u64 kDDictSize = 0x10;
constexpr u64 kDDictId = 0x6840;
constexpr u64 kDDictEntropyPresent = 0x6844;
constexpr u64 kDDictReadEnd = 0x6848;
constexpr u64 kMaxDictSize = u64{64} << 20;
constexpr size_t kMaxDictEntries = 16;
constexpr size_t kEdgeBytes = 64;
constexpr s64 kRevalidateNs = 1'000'000'000;
constexpr s64 kLogIntervalNs = 1'000'000'000;

// EDEN_NX2_OOT_ZSTD_DICT: "0" sends calls with a ddict to the game's decoder again (only calls
// without a dictionary use host libzstd then). Default on.
bool DictEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_OOT_ZSTD_DICT");
        return value == nullptr || value[0] == 0 || value[0] != '0';
    }();
    return enabled;
}

s64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

struct DCtxDeleter {
    void operator()(ZSTD_DCtx* dctx) const {
        ZSTD_freeDCtx(dctx);
    }
};

struct DDictDeleter {
    void operator()(ZSTD_DDict* ddict) const {
        ZSTD_freeDDict(ddict);
    }
};

struct Stats {
    std::atomic<u64> calls{};
    std::atomic<u64> ddict_calls{};
    std::atomic<u64> fallbacks{};
    std::atomic<u64> bytes{};
    std::atomic<u64> dict_hits{};
    std::atomic<u64> dict_builds{};
};
Stats g_stats;

// The guest ddict fields that decide how a frame decodes.
struct GuestDDict {
    u64 address{};
    u64 content{};
    u64 size{};
    u32 dict_id{};
    u32 entropy_present{};

    bool operator==(const GuestDDict&) const = default;
};

// A host copy of one guest dictionary. The host DDict references `content` (byRef), so it is
// declared after it and destroyed first. A null `ddict` means the game's decoder handles it.
struct DictEntry {
    GuestDDict key;
    std::vector<u8> content;
    std::unique_ptr<ZSTD_DDict, DDictDeleter> ddict;
    std::atomic<s64> validated_ns{};
    std::atomic<s64> last_use_ns{};
};

struct DictCache {
    std::mutex mutex;
    std::vector<std::shared_ptr<DictEntry>> entries;
};

DictCache& Cache() {
    static DictCache cache;
    return cache;
}

bool ReadGuestDDict(Core::Memory::Memory& memory, u64 address, GuestDDict& out) {
    if (!memory.IsValidVirtualAddressRange(address, kDDictReadEnd)) {
        return false;
    }
    out.address = address;
    out.content = memory.Read64(address + kDDictContent);
    out.size = memory.Read64(address + kDDictSize);
    out.dict_id = memory.Read32(address + kDDictId);
    out.entropy_present = memory.Read32(address + kDDictEntropyPresent);
    return out.content != 0 && out.size != 0 && out.size <= kMaxDictSize &&
           memory.IsValidVirtualAddressRange(out.content, out.size);
}

// Cheap per-call check: the first and last bytes of the guest dictionary still match the copy.
bool EdgesMatch(Core::Memory::Memory& memory, const DictEntry& entry) {
    const size_t size = entry.content.size();
    const size_t n = std::min(size, kEdgeBytes);
    std::array<u8, kEdgeBytes> buffer;
    if (!memory.ReadBlock(entry.key.content, buffer.data(), n) ||
        std::memcmp(buffer.data(), entry.content.data(), n) != 0) {
        return false;
    }
    if (size > n) {
        if (!memory.ReadBlock(entry.key.content + size - n, buffer.data(), n) ||
            std::memcmp(buffer.data(), entry.content.data() + size - n, n) != 0) {
            return false;
        }
    }
    return true;
}

// Full check, at most once a second per dictionary: the whole guest dictionary matches the copy.
bool ContentMatches(Core::Memory::Memory& memory, const DictEntry& entry) {
    thread_local Common::ScratchBuffer<u8> scratch;
    const size_t size = entry.content.size();
    scratch.resize_destructive(size);
    return memory.ReadBlock(entry.key.content, scratch.data(), size) &&
           std::memcmp(scratch.data(), entry.content.data(), size) == 0;
}

// Copies the guest dictionary and loads it the way the guest did: ZSTD_dct_auto with byRef gives
// entropyPresent = 1 and dictID from the header for a zstd dictionary (magic 0xEC30A437), and
// entropyPresent = 0, dictID = 0 for raw content. The host DDict gets the matching explicit type,
// and its dictID must equal the guest's. Null when the dictionary could not be read.
std::shared_ptr<DictEntry> BuildEntry(Core::Memory::Memory& memory, const GuestDDict& key,
                                      s64 now) {
    auto entry = std::make_shared<DictEntry>();
    entry->key = key;
    entry->content.resize(key.size);
    if (!memory.ReadBlock(key.content, entry->content.data(), key.size)) {
        return nullptr;
    }
    entry->validated_ns.store(now, std::memory_order_relaxed);
    entry->last_use_ns.store(now, std::memory_order_relaxed);
    const char* type_name = "unusual";
    if (key.entropy_present == 1 || (key.entropy_present == 0 && key.dict_id == 0)) {
        const bool full = key.entropy_present == 1;
        type_name = full ? "zstd dictionary" : "raw content";
        entry->ddict.reset(ZSTD_createDDict_advanced(
            entry->content.data(), entry->content.size(), ZSTD_dlm_byRef,
            full ? ZSTD_dct_fullDict : ZSTD_dct_rawContent, ZSTD_defaultCMem));
        if (entry->ddict && ZSTD_getDictID_fromDDict(entry->ddict.get()) != key.dict_id) {
            entry->ddict.reset();
        }
    }
    static std::atomic<u32> logged{};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
        LOG_INFO(Kernel_SVC,
                 "Native zstd: ddict {:#x} (content {:#x}, {} bytes, dictID {:#x}, {}): {}",
                 key.address, key.content, key.size, key.dict_id, type_name,
                 entry->ddict ? "decoding on the host" : "left to the game's decoder");
    }
    return entry;
}

// The host dictionary for a guest ddict, rebuilt when the guest ddict or its content changes.
// Null when the dictionary could not be read (not cached).
std::shared_ptr<DictEntry> GetDict(Core::Memory::Memory& memory, const GuestDDict& key, s64 now) {
    auto& cache = Cache();
    std::shared_ptr<DictEntry> entry;
    {
        std::scoped_lock lock{cache.mutex};
        for (const auto& candidate : cache.entries) {
            if (candidate->key == key) {
                entry = candidate;
                break;
            }
        }
    }
    if (entry) {
        bool valid = EdgesMatch(memory, *entry);
        if (valid && now - entry->validated_ns.load(std::memory_order_relaxed) >= kRevalidateNs) {
            valid = ContentMatches(memory, *entry);
            if (valid) {
                entry->validated_ns.store(now, std::memory_order_relaxed);
            }
        }
        if (valid) {
            entry->last_use_ns.store(now, std::memory_order_relaxed);
            return entry;
        }
    }
    auto fresh = BuildEntry(memory, key, now);
    if (!fresh) {
        return nullptr;
    }
    g_stats.dict_builds.fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lock{cache.mutex};
    std::erase_if(cache.entries, [&](const std::shared_ptr<DictEntry>& old) {
        return old->key.address == key.address || old->key == key;
    });
    if (cache.entries.size() >= kMaxDictEntries) {
        const auto oldest = std::min_element(
            cache.entries.begin(), cache.entries.end(), [](const auto& a, const auto& b) {
                return a->last_use_ns.load(std::memory_order_relaxed) <
                       b->last_use_ns.load(std::memory_order_relaxed);
            });
        cache.entries.erase(oldest);
    }
    cache.entries.push_back(fresh);
    return fresh;
}

// Only zstd1 and skippable frames, each complete: anything else (legacy formats the host library
// would accept, truncated input, trailing bytes) goes to the game's decoder.
bool OnlyStandardFrames(const u8* src, size_t size) {
    if (size == 0) {
        return false;
    }
    while (size != 0) {
        if (size < 4) {
            return false;
        }
        u32 magic;
        std::memcpy(&magic, src, sizeof(magic));
        if (magic != ZSTD_MAGICNUMBER &&
            (magic & ZSTD_MAGIC_SKIPPABLE_MASK) != ZSTD_MAGIC_SKIPPABLE_START) {
            return false;
        }
        const size_t frame_size = ZSTD_findFrameCompressedSize(src, size);
        if (ZSTD_isError(frame_size) || frame_size == 0 || frame_size > size) {
            return false;
        }
        src += frame_size;
        size -= frame_size;
    }
    return true;
}

// Decompresses on the host. Returns nullptr on success (args[0] then holds the decompressed size,
// as the function returns), or the reason the game's own function must run instead; then nothing
// was written to guest memory.
const char* HostDecompress(Core::Memory::Memory& memory, std::span<uint64_t, 8> args, s64 now) {
    const u64 dst = args[1];
    const u64 capacity = args[2];
    const u64 src = args[3];
    const u64 src_size = args[4];
    const u64 ddict_address = args[7];
    if (args[5] != 0 || args[6] != 0) {
        return "raw dict argument";
    }
    if (src_size == 0 || src_size > kMaxSize || !memory.IsValidVirtualAddressRange(src, src_size)) {
        return "source size or range";
    }
    // With a ddict, ZSTD_decompressMultiFrame is ZSTD_decompress_usingDDict: the same function
    // with the same ddict fields gives the same bytes and the same return value.
    std::shared_ptr<DictEntry> dict;
    if (ddict_address != 0) {
        if (!DictEnabled()) {
            return "ddict (EDEN_NX2_OOT_ZSTD_DICT=0)";
        }
        GuestDDict key;
        if (!ReadGuestDDict(memory, ddict_address, key)) {
            return "unreadable or unusual ddict";
        }
        dict = GetDict(memory, key, now);
        if (!dict) {
            return "dictionary content unreadable";
        }
        if (!dict->ddict) {
            return "dictionary type or dictID the host does not reproduce";
        }
    }
    thread_local Common::ScratchBuffer<u8> input;
    thread_local Common::ScratchBuffer<u8> output;
    thread_local std::unique_ptr<ZSTD_DCtx, DCtxDeleter> dctx{ZSTD_createDCtx()};
    if (!dctx) {
        return "no host DCtx";
    }
    input.resize_destructive(src_size);
    if (!memory.ReadBlock(src, input.data(), src_size)) {
        return "source unreadable";
    }
    if (!OnlyStandardFrames(input.data(), src_size)) {
        return "not only complete zstd1/skippable frames";
    }
    // Same result for any capacity >= the output, and a capacity below it fails either way.
    const unsigned long long bound = ZSTD_decompressBound(input.data(), src_size);
    if (bound == ZSTD_CONTENTSIZE_ERROR) {
        return "no decompression bound";
    }
    const u64 out_capacity = std::min<u64>(capacity, bound);
    if (out_capacity > kMaxSize) {
        return "output too large";
    }
    output.resize_destructive(std::max<size_t>(out_capacity, 1));
    const size_t result =
        dict ? ZSTD_decompress_usingDDict(dctx.get(), output.data(), out_capacity, input.data(),
                                          src_size, dict->ddict.get())
             : ZSTD_decompressDCtx(dctx.get(), output.data(), out_capacity, input.data(),
                                   src_size);
    if (ZSTD_isError(result)) {
        return ZSTD_getErrorName(result);
    }
    if (result != 0) {
        if (!memory.IsValidVirtualAddressRange(dst, result)) {
            return "destination range";
        }
        memory.WriteBlock(dst, output.data(), result);
    }
    if (dict) {
        g_stats.dict_hits.fetch_add(1, std::memory_order_relaxed);
    }
    args[0] = result;
    return nullptr;
}

// Runs the game's function as the replaced bl would have: lr = the instruction after the svc.
void RunGuestFunction(Core::System& system, KProcess& process, u32 imm) {
    auto* const arm = process.GetArmInterface(system.Kernel().CurrentPhysicalCoreIndex());
    ThreadContext ctx{};
    arm->GetContext(ctx);
    const u64 return_address = ctx.pc;
    ctx.lr = return_address;
    ctx.pc = return_address - 4 - kCallSites[imm - kFirstSvc] + kMultiFrameOffset;
    arm->SetContext(ctx);
}

// At most one counters line a second (the first call logs at once).
void MaybeLogCounters(s64 now) {
    static std::atomic<s64> next_log_ns{0};
    s64 next = next_log_ns.load(std::memory_order_relaxed);
    if (now < next ||
        !next_log_ns.compare_exchange_strong(next, now + kLogIntervalNs,
                                             std::memory_order_relaxed)) {
        return;
    }
    LOG_INFO(Kernel_SVC,
             "Native zstd: {} calls ({} with a ddict), {} MiB decompressed on the host, {} "
             "dictionary hits, {} dictionaries built, {} by the game",
             g_stats.calls.load(std::memory_order_relaxed),
             g_stats.ddict_calls.load(std::memory_order_relaxed),
             g_stats.bytes.load(std::memory_order_relaxed) >> 20,
             g_stats.dict_hits.load(std::memory_order_relaxed),
             g_stats.dict_builds.load(std::memory_order_relaxed),
             g_stats.fallbacks.load(std::memory_order_relaxed));
}

} // Anonymous namespace

bool CallNx2OotZstd(Core::System& system, u32 imm, std::span<uint64_t, 8> args) {
    if (imm < kFirstSvc || imm >= kFirstSvc + kCallSites.size()) {
        return false;
    }
    auto& process = GetCurrentProcess(system.Kernel());
    if (!process.Is64Bit() || process.GetProgramId() != kOotProgramId) {
        return false;
    }
    const s64 now = NowNs();
    g_stats.calls.fetch_add(1, std::memory_order_relaxed);
    if (args[7] != 0) {
        g_stats.ddict_calls.fetch_add(1, std::memory_order_relaxed);
    }
    const char* const reason = HostDecompress(process.GetMemory(), args, now);
    if (reason == nullptr) {
        g_stats.bytes.fetch_add(args[0], std::memory_order_relaxed);
    } else {
        if (g_stats.fallbacks.fetch_add(1, std::memory_order_relaxed) < 16) {
            LOG_WARNING(Kernel_SVC,
                        "Native zstd (svc {:#x}): src={:#x} size={:#x} dst={:#x} cap={:#x} "
                        "dict={:#x}/{:#x} ddict={:#x}: {}; running the game's decoder",
                        imm, args[3], args[4], args[1], args[2], args[5], args[6], args[7],
                        reason);
        }
        RunGuestFunction(system, process, imm);
    }
    MaybeLogCounters(now);
    return true;
}

} // namespace Kernel::Svc
