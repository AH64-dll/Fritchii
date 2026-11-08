// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "common/alignment.h"
#include "common/common_funcs.h"
#include "common/common_types.h"
#include "common/div_ceil.h"
#include "common/logging.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/nx2_perf.h"

namespace VideoCommon {

/// Page protection changes the tracker asks for (EDEN_NX2_GPU_PROFILE), logged with the buffer
/// cache counters: protects (pages tracked again by a sync) and unprotects (CPU-dirty marks).
struct Nx2TrackerStats {
    static inline const bool enabled = std::getenv("EDEN_NX2_GPU_PROFILE") != nullptr;
    /// Protect runs, protected bytes, unprotect runs, unprotected bytes.
    static inline std::array<std::atomic<u64>, 4> counts{};

    static void Note(std::span<const std::pair<VAddr, size_t>> ranges, int delta) noexcept {
        if (!enabled) [[likely]] {
            return;
        }
        u64 bytes = 0;
        for (const auto& range : ranges) {
            bytes += range.second;
        }
        const size_t base = delta > 0 ? 0 : 2;
        counts[base].fetch_add(ranges.size(), std::memory_order_relaxed);
        counts[base + 1].fetch_add(bytes, std::memory_order_relaxed);
    }

    /// Protect runs, protected pages, unprotect runs, unprotected pages since the last call.
    static std::array<u64, 4> Take() noexcept {
        std::array<u64, 4> taken{};
        for (size_t i = 0; i < taken.size(); ++i) {
            taken[i] = counts[i].exchange(0, std::memory_order_relaxed);
        }
        taken[1] /= Core::DEVICE_PAGESIZE;
        taken[3] /= Core::DEVICE_PAGESIZE;
        return taken;
    }
};

constexpr u64 PAGES_PER_WORD = 64;
constexpr u64 BYTES_PER_PAGE = Core::DEVICE_PAGESIZE;
constexpr u64 BYTES_PER_WORD = PAGES_PER_WORD * BYTES_PER_PAGE;

enum class Type {
    CPU,
    GPU,
    CachedCPU,
    Untracked,
    Preflushable,
    Hot, ///< Kept unprotected and reported apart by ForEachCpuUploadRange (BufferCache hot pages)
    Max
};

template <class DeviceTracker, size_t stack_words, size_t size_bytes>
struct WordManager {
    using CollectedRanges = std::vector<std::pair<VAddr, size_t>>;

    static constexpr size_t num_words = Common::DivCeil(size_bytes, BYTES_PER_WORD);
    static_assert(num_words <= 64, "clean_words holds a bit per word");

    explicit WordManager(VAddr cpu_addr_, DeviceTracker& tracker_) : tracker{&tracker_}, cpu_addr{cpu_addr_} {
        std::fill_n(heap.data() + size_t(Type::CPU) * num_words, num_words, ~u64{0});
        std::fill_n(heap.data() + size_t(Type::Untracked) * num_words, num_words, ~u64{0});
        // Clean up tailing bits
        u64 const last_word_size = size_bytes % BYTES_PER_WORD;
        u64 const last_local_page = Common::DivCeil(last_word_size, BYTES_PER_PAGE);
        u64 const shift = (PAGES_PER_WORD - last_local_page) % PAGES_PER_WORD;
        u64 const last_word = (~u64{0} << shift) >> shift;
        heap[num_words * size_t(Type::CPU) + num_words - 1] = last_word;
        heap[num_words * size_t(Type::Untracked) + num_words - 1] = last_word;
        for (size_t index = 0; index < num_words; ++index) {
            NoteCleanWord(index);
        }
    }
    explicit WordManager() = default;

    static u64 ExtractBits(u64 word, size_t page_start, size_t page_end) {
        constexpr size_t number_bits = sizeof(u64) * 8;
        const size_t limit_page_end = number_bits - (std::min)(page_end, number_bits);
        u64 bits = (word >> page_start) << page_start;
        bits = (bits << limit_page_end) >> limit_page_end;
        return bits;
    }

    static std::pair<size_t, size_t> GetWordPage(VAddr address) {
        const size_t converted_address = size_t(address);
        const size_t word_number = converted_address / BYTES_PER_WORD;
        const size_t amount_pages = converted_address % BYTES_PER_WORD;
        return std::make_pair(word_number, amount_pages / BYTES_PER_PAGE);
    }

    template <typename Func>
    void IterateWords(size_t offset, size_t size, Func&& func) const {
        using FuncReturn = std::invoke_result_t<Func, std::size_t, u64>;
        const size_t start = size_t(std::max<s64>(s64(offset), 0LL));
        const size_t end = size_t(std::max<s64>(s64(offset + size), 0LL));
        if (!(start >= size_bytes || end <= start)) {
            auto [start_word, start_page] = GetWordPage(start);
            auto [end_word, end_page] = GetWordPage(end + BYTES_PER_PAGE - 1ULL);
            start_word = (std::min)(start_word, num_words);
            end_word = (std::min)(end_word, num_words);
            const size_t diff = end_word - start_word;
            end_word += (end_page + PAGES_PER_WORD - 1ULL) / PAGES_PER_WORD;
            end_word = (std::min)(end_word, num_words);
            end_page += diff * PAGES_PER_WORD;
            constexpr u64 base_mask{~0ULL};
            for (size_t word_index = start_word; word_index < end_word; word_index++) {
                const u64 mask = ExtractBits(base_mask, start_page, end_page);
                start_page = 0;
                end_page -= PAGES_PER_WORD;
                if constexpr (std::is_same_v<FuncReturn, bool>) { // bool return
                    if (func(word_index, mask))
                        return;
                } else {
                    func(word_index, mask);
                }
            }
        }
    }

    template <typename Func>
    void IteratePages(u64 mask, Func&& func) const {
        size_t offset = 0;
        while (mask != 0) {
            const size_t empty_bits = std::countr_zero(mask);
            offset += empty_bits;
            mask = mask >> empty_bits;

            const size_t continuous_bits = std::countr_one(mask);
            func(offset, continuous_bits);
            mask = continuous_bits < PAGES_PER_WORD ? (mask >> continuous_bits) : 0;
            offset += continuous_bits;
        }
    }

    /// @brief Change the state of a range of pages
    /// @param type          Type of the page
    /// @param enable        If enabling or disabling
    /// @param dirty_addr    Base address to mark or unmark as modified
    /// @param size          Size in bytes to mark or unmark as modified
    void ChangeRegionState(Type type, bool enable, u64 dirty_addr, u64 size) noexcept {
        std::span<u64> state_words = Span(type);
        [[maybe_unused]] std::span<u64> untracked_words = Span(Type::Untracked);
        [[maybe_unused]] std::span<u64> cached_words = Span(Type::CachedCPU);
        CollectedRanges ranges;
        IterateWords(dirty_addr - cpu_addr, size, [&](size_t index, u64 mask) {
            if (type == Type::CPU || type == Type::CachedCPU) {
                CollectChangedRanges(!enable, index, untracked_words[index], mask, ranges);
            }
            if (enable) {
                state_words[index] |= mask;
                if (type == Type::CPU || type == Type::CachedCPU)
                    untracked_words[index] |= mask;
                if (type == Type::CPU)
                    cached_words[index] &= ~mask;
            } else {
                if (type == Type::CPU)
                    cached_words[index] &= ~(state_words[index] & mask);
                state_words[index] &= ~mask;
                if (type == Type::CPU || type == Type::CachedCPU)
                    untracked_words[index] &= ~mask;
            }
            if (type == Type::CPU || type == Type::CachedCPU) {
                NoteCleanWord(index);
            }
        });
        if (!ranges.empty()) {
            ApplyCollectedRanges(ranges, (!enable) ? 1 : -1);
        }
    }

    /// @brief Loop over each page in the given range.
    /// Turn off those bits and notify the tracker if needed. Call the given function on each turned off range.
    /// @param type            Type of the address
    /// @param clear           Whetever to clear
    /// @param query_cpu_range Base CPU address to loop over
    /// @param size            Size in bytes of the CPU range to loop over
    /// @param func            Function to call for each turned off region
    template <typename Func>
    void ForEachModifiedRange(Type type, bool clear, VAddr query_cpu_range, s64 size, Func&& func) {
        //static_assert(type != Type::Untracked);
        std::span<u64> state_words = Span(type);
        std::span<u64> untracked_words = Span(Type::Untracked);
        std::span<u64> cached_words = Span(Type::CachedCPU);
        size_t const offset = query_cpu_range - cpu_addr;
        bool pending = false;
        size_t pending_offset{};
        size_t pending_pointer{};
        const auto release = [&]() {
            func(cpu_addr + pending_offset * BYTES_PER_PAGE,
                 (pending_pointer - pending_offset) * BYTES_PER_PAGE);
        };
        CollectedRanges ranges;
        IterateWords(offset, size, [&](size_t index, u64 mask) {
            if (type == Type::GPU)
                mask &= ~untracked_words[index];
            const u64 word = state_words[index] & mask;
            if (clear) {
                if (type == Type::CPU || type == Type::CachedCPU) {
                    CollectChangedRanges(true, index, untracked_words[index], mask, ranges);
                }
                state_words[index] &= ~mask;
                if (type == Type::CPU || type == Type::CachedCPU)
                    untracked_words[index] &= ~mask;
                if (type == Type::CPU)
                    cached_words[index] &= ~word;
                if (type == Type::CPU || type == Type::CachedCPU)
                    NoteCleanWord(index);
            }
            const size_t base_offset = index * PAGES_PER_WORD;
            IteratePages(word, [&](size_t pages_offset, size_t pages_size) {
                if (!pending) {
                    pending_offset = base_offset + pages_offset;
                    pending_pointer = base_offset + pages_offset + pages_size;
                    pending = true;
                } else if (pending_pointer == base_offset + pages_offset) {
                    pending_pointer += pages_size;
                } else {
                    func(cpu_addr + pending_offset * BYTES_PER_PAGE, (pending_pointer - pending_offset) * BYTES_PER_PAGE);
                    pending_offset = base_offset + pages_offset;
                    pending_pointer = base_offset + pages_offset + pages_size;
                }
            });
        });
        if (pending) {
            release();
        }
        if (!ranges.empty()) {
            ApplyCollectedRanges(ranges, 1);
        }
    }

    /// ForEachModifiedRange(Type::CPU, true, ...), except for Hot pages: their CPU and Untracked
    /// bits stay (no protection change) and they go to `hot_func` instead of `func`.
    template <typename Func, typename HotFunc>
    void ForEachCpuUploadRange(VAddr query_cpu_range, s64 size, Func&& func, HotFunc&& hot_func) {
        std::span<u64> state_words = Span(Type::CPU);
        std::span<u64> untracked_words = Span(Type::Untracked);
        std::span<u64> cached_words = Span(Type::CachedCPU);
        const std::span<const u64> hot_words = Span(Type::Hot);
        size_t const offset = query_cpu_range - cpu_addr;
        struct Run {
            bool pending = false;
            size_t begin{};
            size_t end{};
        };
        Run regular{};
        Run hot{};
        const auto add = [&](Run& run, size_t page, size_t pages, auto& callback) {
            if (run.pending && run.end == page) {
                run.end += pages;
                return;
            }
            if (run.pending) {
                callback(cpu_addr + run.begin * BYTES_PER_PAGE, (run.end - run.begin) * BYTES_PER_PAGE);
            }
            run = Run{true, page, page + pages};
        };
        CollectedRanges ranges;
        IterateWords(offset, size, [&](size_t index, u64 mask) {
            const u64 word = state_words[index] & mask;
            const u64 hot_word = word & hot_words[index];
            const u64 clear_mask = mask & ~hot_words[index];
            CollectChangedRanges(true, index, untracked_words[index], clear_mask, ranges);
            state_words[index] &= ~clear_mask;
            untracked_words[index] &= ~clear_mask;
            cached_words[index] &= ~word;
            NoteCleanWord(index);
            const size_t base_offset = index * PAGES_PER_WORD;
            IteratePages(word & ~hot_word, [&](size_t pages_offset, size_t pages_size) {
                add(regular, base_offset + pages_offset, pages_size, func);
            });
            IteratePages(hot_word, [&](size_t pages_offset, size_t pages_size) {
                add(hot, base_offset + pages_offset, pages_size, hot_func);
            });
        });
        if (regular.pending) {
            func(cpu_addr + regular.begin * BYTES_PER_PAGE,
                 (regular.end - regular.begin) * BYTES_PER_PAGE);
        }
        if (hot.pending) {
            hot_func(cpu_addr + hot.begin * BYTES_PER_PAGE, (hot.end - hot.begin) * BYTES_PER_PAGE);
        }
        if (!ranges.empty()) {
            ApplyCollectedRanges(ranges, 1);
        }
    }

    /// @brief Returns true when a region has been modified
    /// @param type   Type of region
    /// @param offset Offset in bytes from the start of the buffer
    /// @param size   Size in bytes of the region to query for modifications
    [[nodiscard]] bool IsRegionModified(Type type, u64 offset, u64 size) const noexcept {
        //static_assert(type != Type::Untracked);
        const std::span<const u64> state_words = Span(type);
        const std::span<const u64> untracked_words = Span(Type::Untracked);
        bool result = false;
        IterateWords(offset, size, [&](size_t index, u64 mask) {
            if (type == Type::GPU)
                mask &= ~untracked_words[index];
            return (state_words[index] & mask) != 0 ? (result = true) : false;
        });
        return result;
    }

    /// @brief Returns a begin end pair with the inclusive modified region
    /// @param offset Offset in bytes from the start of the buffer
    /// @param size   Size in bytes of the region to query for modifications
    [[nodiscard]] std::pair<u64, u64> ModifiedRegion(Type type, u64 offset, u64 size) const noexcept {
        //static_assert(type != Type::Untracked);
        const std::span<const u64> state_words = Span(type);
        const std::span<const u64> untracked_words = Span(Type::Untracked);
        u64 begin = (std::numeric_limits<u64>::max)(), end = 0;
        IterateWords(offset, size, [&](size_t index, u64 mask) {
            if (type == Type::GPU)
                mask &= ~untracked_words[index];
            const u64 word = state_words[index] & mask;
            if (word != 0) {
                const u64 local_page_begin = std::countr_zero(word);
                const u64 local_page_end = PAGES_PER_WORD - std::countl_zero(word);
                const u64 page_index = index * PAGES_PER_WORD;
                begin = (std::min)(begin, page_index + local_page_begin);
                end = page_index + local_page_end;
            }
        });
        return begin < end ? std::make_pair<u64, u64>(begin * BYTES_PER_PAGE, end * BYTES_PER_PAGE)
            : std::make_pair<u64, u64>(0, 0);
    }

    void FlushCachedWrites() noexcept {
        auto const cached_words = Span(Type::CachedCPU);
        auto const untracked_words = Span(Type::Untracked);
        auto const cpu_words = Span(Type::CPU);
        CollectedRanges ranges;
        for (u64 word_index = 0; word_index < num_words; ++word_index) {
            const u64 cached_bits = cached_words[word_index];
            CollectChangedRanges(false, word_index, untracked_words[word_index], cached_bits, ranges);
            untracked_words[word_index] |= cached_bits;
            cpu_words[word_index] |= cached_bits;
            cached_words[word_index] = 0;
            NoteCleanWord(word_index);
        }
        if (!ranges.empty()) {
            ApplyCollectedRanges(ranges, -1);
        }
    }

    /// @brief Notify tracker about changes in the CPU tracking state of a word in the buffer
    /// @param add_to_tracker If add to tracker (selects changed bits)
    /// @param word_index     Index to the word to notify to the tracker
    /// @param current_bits   Current state of the word
    /// @param new_bits       New state of the word
    /// @tparam add_to_tracker True when the tracker should start tracking the new pages
    void CollectChangedRanges(bool add_to_tracker, u64 word_index, u64 current_bits, u64 new_bits, CollectedRanges& out_ranges) const {
        u64 changed_bits = (add_to_tracker ? current_bits : ~current_bits) & new_bits;
        VAddr addr = cpu_addr + word_index * BYTES_PER_WORD;
        IteratePages(changed_bits, [&](size_t offset, size_t size) {
            out_ranges.emplace_back(addr + offset * BYTES_PER_PAGE, size * BYTES_PER_PAGE);
        });
    }

    void ApplyCollectedRanges(CollectedRanges& ranges, int delta) const {
        if (ranges.empty())
            return;
        std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        if (!Nx2Perf::Verify()) {
            CoalesceInPlace(ranges);
            Nx2TrackerStats::Note(ranges, delta);
            tracker->UpdatePagesCachedBatch(
                std::span<const std::pair<VAddr, size_t>>(ranges.data(), ranges.size()), delta);
            ranges.clear();
            return;
        }
        // Coalesce adjacent/contiguous ranges
        std::vector<std::pair<VAddr, size_t>> coalesced;
        coalesced.reserve(ranges.size());
        VAddr cur_addr = ranges[0].first;
        size_t cur_size = static_cast<size_t>(ranges[0].second);
        for (size_t i = 1; i < ranges.size(); ++i) {
            if (cur_addr + cur_size == ranges[i].first) {
                cur_size += static_cast<size_t>(ranges[i].second);
            } else {
                coalesced.emplace_back(cur_addr, cur_size);
                cur_addr = ranges[i].first;
                cur_size = static_cast<size_t>(ranges[i].second);
            }
        }
        coalesced.emplace_back(cur_addr, cur_size);
        CoalesceInPlace(ranges);
        static std::atomic<u32> logged{};
        if (!std::ranges::equal(ranges, coalesced) &&
            logged.fetch_add(1, std::memory_order_relaxed) < 64) {
            LOG_ERROR(HW_GPU, "Nx2Perf texture-writes: tracker runs from {:#x} coalesced differently",
                      coalesced.front().first);
        }
        // Use batch API to reduce lock acquisitions and contention.
        Nx2TrackerStats::Note(coalesced, delta);
        tracker->UpdatePagesCachedBatch(coalesced, delta);
        ranges.clear();
    }

    /// Merges the contiguous neighbours of sorted, disjoint runs where they lie (no copy).
    static void CoalesceInPlace(CollectedRanges& ranges) noexcept {
        size_t last = 0;
        for (size_t i = 1; i < ranges.size(); ++i) {
            if (ranges[last].first + ranges[last].second == ranges[i].first) {
                ranges[last].second += ranges[i].second;
            } else {
                ranges[++last] = ranges[i];
            }
        }
        ranges.resize(last + 1);
    }

    /// @brief Notify tracker about changes in the CPU tracking state of a word in the buffer
    /// @param add_to_tracker True when the tracker should start tracking the new pages
    /// @param word_index   Index to the word to notify to the tracker
    /// @param current_bits Current state of the word
    /// @param new_bits     New state of the word
    void NotifyRasterizer(bool add_to_tracker, u64 word_index, u64 current_bits, u64 new_bits) const {
        u64 changed_bits = (add_to_tracker ? current_bits : ~current_bits) & new_bits;
        VAddr addr = cpu_addr + word_index * BYTES_PER_WORD;
        IteratePages(changed_bits, [&](size_t offset, size_t size) {
            tracker->UpdatePagesCachedCount(addr + offset * BYTES_PER_PAGE, size * BYTES_PER_PAGE, add_to_tracker ? 1 : -1);
        });
    }

    std::span<u64> Span(Type type) noexcept {
        return std::span<u64>(heap.data() + num_words * size_t(type), num_words);
    }

    std::span<const u64> Span(Type type) const noexcept {
        return std::span<const u64>(heap.data() + num_words * size_t(type), num_words);
    }

    /// Words of [first_word, last_word] that are not clean, one bit each. A clean word has no CPU
    /// or Untracked page, so an upload over it finds and changes nothing.
    [[nodiscard]] u64 DirtyWords(size_t first_word, size_t last_word) const noexcept {
        const u64 mask = (~u64{0} >> (63 - (last_word - first_word))) << first_word;
        return mask & ~clean_words;
    }

    /// IsRegionModified(Type::GPU, ...) for the bytes [first, last] of one word.
    [[nodiscard]] bool IsWordRangeGpuModified(u64 first, u64 last) const noexcept {
        const size_t word = first / BYTES_PER_WORD;
        const u64 first_page = (first % BYTES_PER_WORD) / BYTES_PER_PAGE;
        const u64 last_page = (last % BYTES_PER_WORD) / BYTES_PER_PAGE;
        const u64 mask = (~u64{0} >> (63 - (last_page - first_page))) << first_page;
        return (heap[num_words * size_t(Type::GPU) + word] &
                ~heap[num_words * size_t(Type::Untracked) + word] & mask) != 0;
    }

    /// Whether a page of [offset, offset + size) is CPU-dirty or unprotected (what an upload walk
    /// finds or changes). Clean words are skipped through clean_words.
    [[nodiscard]] bool IsRangeUploadDirty(u64 offset, u64 size) const noexcept {
        if (size == 0) {
            return false;
        }
        const u64 end = offset + size;
        u64 dirty = DirtyWords(offset / BYTES_PER_WORD, (end - 1) / BYTES_PER_WORD);
        while (dirty != 0) {
            const size_t word = static_cast<size_t>(std::countr_zero(dirty));
            dirty &= dirty - 1;
            const u64 word_begin = word * BYTES_PER_WORD;
            const u64 begin = (std::max)(offset, word_begin) - word_begin;
            const u64 stop = (std::min)(end, word_begin + BYTES_PER_WORD) - word_begin;
            const u64 first_page = begin / BYTES_PER_PAGE;
            const u64 last_page = (stop - 1) / BYTES_PER_PAGE;
            const u64 mask = (~u64{0} >> (63 - (last_page - first_page))) << first_page;
            if (((heap[num_words * size_t(Type::CPU) + word] |
                  heap[num_words * size_t(Type::Untracked) + word]) & mask) != 0) {
                return true;
            }
        }
        return false;
    }

    /// Whether every page of [offset, offset + size) is CPU-dirty (marking it again changes no
    /// tracker state). False for an empty range.
    [[nodiscard]] bool IsRangeFullyCpuModified(u64 offset, u64 size) const noexcept {
        if (size == 0) {
            return false;
        }
        const u64 end = offset + size;
        for (u64 word = offset / BYTES_PER_WORD; word <= (end - 1) / BYTES_PER_WORD; ++word) {
            const u64 word_begin = word * BYTES_PER_WORD;
            const u64 begin = (std::max)(offset, word_begin) - word_begin;
            const u64 stop = (std::min)(end, word_begin + BYTES_PER_WORD) - word_begin;
            const u64 first_page = begin / BYTES_PER_PAGE;
            const u64 last_page = (stop - 1) / BYTES_PER_PAGE;
            const u64 mask = (~u64{0} >> (63 - (last_page - first_page))) << first_page;
            if ((heap[num_words * size_t(Type::CPU) + word] & mask) != mask) {
                return false;
            }
        }
        return true;
    }

    /// Recounts the clean words: true when clean_words matches the CPU and Untracked words.
    [[nodiscard]] bool CleanWordsHold() const noexcept {
        const std::span<const u64> cpu_words = Span(Type::CPU);
        const std::span<const u64> untracked_words = Span(Type::Untracked);
        for (size_t index = 0; index < num_words; ++index) {
            const bool clean = (cpu_words[index] | untracked_words[index]) == 0;
            if (((clean_words >> index) & 1) != u64{clean}) {
                return false;
            }
        }
        return true;
    }

    void NoteCleanWord(size_t index) noexcept {
        const bool clean = (heap[index] | heap[num_words * size_t(Type::Untracked) + index]) == 0;
        clean_words = (clean_words & ~(u64{1} << index)) | (u64{clean} << index);
    }

    std::array<u64, size_t(Type::Max) * num_words> heap = {};
    /// Bit per word, set while the word has no CPU and no Untracked page (DirtyWords). Every
    /// change to those words updates it: a per-word form of PR #161's upload_words count.
    u64 clean_words = 0;
    DeviceTracker* tracker = nullptr;
    VAddr cpu_addr = 0;
};

} // namespace VideoCommon
