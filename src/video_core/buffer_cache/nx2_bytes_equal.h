// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <cstring>

#if defined(_M_X64) || defined(__x86_64__) || defined(__SSE2__)
#include <emmintrin.h>
#define NX2_BYTES_EQUAL_SSE2 1
#endif

#include "common/common_types.h"

namespace VideoCommon {

/// Whether [a, a + size) and [b, b + size) hold the same bytes. Inline, 128 bytes per step with an
/// early exit: memcmp is an out-of-line ucrt call that clang does not expand for 4 KiB pages
/// (zortos293/Lichii#717 Nx2BufCache::BytesEqual, EDEN_NX2_PAGE_CMP). SSE2 only, so any x86-64
/// CPU; other hosts use memcmp.
[[nodiscard]] inline bool Nx2BytesEqual(const u8* a, const u8* b, std::size_t size) {
#ifdef NX2_BYTES_EQUAL_SSE2
    std::size_t offset = 0;
    for (; offset + 128 <= size; offset += 128) {
        const auto load = [offset](const u8* base, std::size_t index) {
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
#else
    return std::memcmp(a, b, size) == 0;
#endif
}

} // namespace VideoCommon
