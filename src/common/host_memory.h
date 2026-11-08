// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include "common/common_funcs.h"
#include "common/common_types.h"

namespace Common {

enum class MemoryPermission : u32 {
    Read = 1 << 0,
    Write = 1 << 1,
    ReadWrite = Read | Write,
    Execute = 1 << 2,
};
DECLARE_ENUM_FLAG_OPERATORS(MemoryPermission)

/**
 * A low level linear memory buffer, which supports multiple mappings
 * Its purpose is to rebuild a given sparse memory layout, including mirrors.
 */
class HostMemory {
public:
    explicit HostMemory(size_t backing_size_, size_t virtual_size_);
    ~HostMemory();

    /**
     * Copy constructors. They shall return a copy of the buffer without the mappings.
     * TODO: Implement them with COW if needed.
     */
    HostMemory(const HostMemory& other) = delete;
    HostMemory& operator=(const HostMemory& other) = delete;

    /**
     * Move constructors. They will move the buffer and the mappings to the new object.
     */
    HostMemory(HostMemory&& other) noexcept;
    HostMemory& operator=(HostMemory&& other) noexcept;

    void Map(size_t virtual_offset, size_t host_offset, size_t length, MemoryPermission perms,
             bool separate_heap);

    void Unmap(size_t virtual_offset, size_t length, bool separate_heap);

    void Protect(size_t virtual_offset, size_t length, MemoryPermission perms);

    void EnableDirectMappedAddress();

    /// Clears guest memory being handed out. Memory still zero since it was created (or since its
    /// last full clear) is skipped: clearing it only made the host give every page real memory.
    void ClearBackingRegion(size_t physical_offset, size_t length, u32 fill_value);

    /// Memory written once and read again rarely: its pages leave the working set (backing and,
    /// when given, its view at virtual_offset), so the OS compresses or pages them out first when
    /// it needs RAM. The contents stay; using the pages brings them back.
    void HintCold(size_t physical_offset, size_t length,
                  std::optional<size_t> virtual_offset) const;
    /// Pages the kernel hands out (cleared or not).
    void MarkUsed(size_t physical_offset, size_t length);

    /// Pages given back to the kernel: their contents are no longer known to be zero.
    void MarkDirty(size_t physical_offset, size_t length);

    /// A new kernel on this memory: everything a previous one used may hold data.
    void ForgetContents();

    /// EDEN_NX2_SEC_RESERVE=1 (Windows, off by default): the guest memory section is created
    /// with SEC_RESERVE instead of SEC_COMMIT, so the commit charge follows what the guest uses
    /// instead of all of it (14 GiB) from boot. Commit commits the pages behind a range (whole
    /// 64 KiB granules, those committed already skipped); MarkUsed (so ClearBackingRegion too)
    /// and Map call it. A no-op without the switch.
    void Commit(size_t physical_offset, size_t length);

    /// Commits everything outside `lazy` (physical offset and length of the kernel's user pools,
    /// whose pages are committed as they are allocated). A no-op without EDEN_NX2_SEC_RESERVE.
    void CommitAllBut(std::span<const std::pair<size_t, size_t>> lazy);

    [[nodiscard]] bool CommitsOnUse() const noexcept {
        return commit_on_use;
    }

    /// The section's commit state, for the memory log.
    struct CommitStats {
        bool reserved{};        ///< EDEN_NX2_SEC_RESERVE is on
        u64 committed_bytes{};  ///< Committed by Commit
        u64 section_bytes{};
        u64 net_commits{};      ///< Pages the fault handler had to commit (must stay 0)
        u64 first_net_offset{}; ///< The first of those: backing offset, or virtual | 1 << 63
    };
    [[nodiscard]] static CommitStats GetCommitStats();

    [[nodiscard]] u8* BackingBasePointer() noexcept {
        return backing_base;
    }
    [[nodiscard]] const u8* BackingBasePointer() const noexcept {
        return backing_base;
    }

    [[nodiscard]] u8* VirtualBasePointer() noexcept {
        return virtual_base;
    }
    [[nodiscard]] const u8* VirtualBasePointer() const noexcept {
        return virtual_base;
    }

    [[nodiscard]] size_t VirtualSize() const noexcept {
        return virtual_size;
    }

    bool IsInVirtualRange(void* address) const noexcept {
        return address >= virtual_base && address < virtual_base + virtual_size;
    }

private:
    size_t backing_size{};
    size_t virtual_size{};

#if !(defined(__OPENORBIS__) || defined(__managarm__))
    // Low level handler for the platform dependent memory routines
    class Impl;
    std::unique_ptr<Impl> impl;
#endif
    u8* backing_base{};
    u8* virtual_base{};
    size_t virtual_base_offset{};

    static constexpr size_t GranuleShift = 16; ///< 64 KiB of backing per state byte
    static constexpr u8 GranuleUsed = 1 << 0;   ///< Handed out at least once
    static constexpr u8 GranuleDirty = 1 << 1;  ///< May hold non-zero data
    static constexpr u8 GranuleCommitted = 1 << 2; ///< EDEN_NX2_SEC_RESERVE: committed
    std::unique_ptr<std::atomic<u8>[]> granules;
    size_t num_granules{};
    // Windows requires it for kernels whom lack proper support for some functions!
    bool fallback_buffer{false};
    bool commit_on_use{false}; ///< EDEN_NX2_SEC_RESERVE (see Commit)
};

} // namespace Common
