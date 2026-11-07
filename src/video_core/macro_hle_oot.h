// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <span>

#include "common/common_types.h"

namespace Tegra::OotMacros {

// Native replacements for Ocarina of Time's hot 3D macros (NVN2 Turing MME code, run on the split
// GPU thread's front by Maxwell3D::FrontRunMacro). A replacement is found by the hash of the
// macro's code (HashMacro), so it only runs for the exact code it was written against.
// EDEN_NX2_OOT_MACROS=0 runs every macro through the JIT again.
//
// A handler makes exactly the writes the JIT makes for the code, in the same order, with the same
// values, and leaves the data RAM as the JIT would. It returns false, before any effect, for the
// calls it can't reproduce exactly (a short FIFO, an out-of-range data RAM index, a write past
// the macro methods, a loop the JIT would stop): those run through the JIT instead. The one
// exception is 0xe2c under EDEN_NX2_OCCLUSION_HLE, which writes the condition it computes.
//
// The context (Maxwell3D::OotContext) provides:
//   u32 Count() const         the load FIFO's size (MME_DMA_READ_FIFOED appends to it)
//   u32 Param(u32 i) const    its word i
//   u32* Dram()               the channel's data RAM (DataRamWords words)
//   void Emit(u32 m, u32 v)   a method write of the macro (m < 0xe00; 0x157/0x158 run the read)
//   void EmitRepeated(const std::array<u32, 2>* writes, u32 count, u32 times)
//                             `times` times the `count` (method, value) writes, in order
//   void Fail(const char* e)  a failure the JIT would report after its effects so far
//   u32 Read(u32 m) const     the front's copy of 3D register m
//   bool OcclusionHle() const EDEN_NX2_OCCLUSION_HLE (0xe2c, see there)
//   void FoldFirstInstance()  the next draw.end is an instance loop's first instance (no write;
//                             DrawManager::FoldFirstInstance on the back, EDEN_NX2_FOLD_INSTANCE)

enum class Id : u8 {
    None,
    E02,
    E04,
    E08,
    E10,
    E1C,
    E1E,
    E20,
    E28,
    E2A,
    E2C,
    E36,
    E3A,
    E3E,
    E52,
    E5C,
    E68,
    E8C,
    E92,
    E96,
    E9A,
    EA0,
    ECE,
    ED6,
};

struct Macro {
    u64 hash;
    const char* name;
    Id id;
};

/// Whether any replacement is registered (none: FrontRunMacro skips the lookup).
[[nodiscard]] bool HasHandlers();

/// Hash of a macro's code: its instructions (three words each) from `start` through the first
/// one with the end bit and its delay slot, at most 1024 instructions. Stable across builds.
[[nodiscard]] u64 HashMacro(std::span<const u32> code, u32 start);

/// The replacement for the macro with code hash `hash`, or nullptr.
[[nodiscard]] const Macro* Find(u64 hash);

// EDEN_NX2_OOT_MACROS_DUMP=1 (diagnostic): logs each macro's code once per hash, and the
// parameters and emitted writes of the first calls of the hot macros, for writing replacements.
[[nodiscard]] bool DumpEnabled();
/// Logs `method`'s code (once per hash).
void DumpCode(u32 method, u32 start, std::span<const u32> code);
/// Whether this call of `method` should be logged with LogCall.
[[nodiscard]] bool WantCallLog(u32 method);
/// `writes`: method, value pairs in emission order.
void LogCall(u32 method, u32 start, std::span<const u32> parameters, std::span<const u32> writes);

inline constexpr u32 DataRamWords = 0xc00;
inline constexpr u32 FirstMacroMethod = 0xe00;
/// Instance loops longer than this run through the JIT (its instruction limit is ~1,000,000).
inline constexpr u32 MaxInstances = 0x10000;

namespace Detail {

[[nodiscard]] constexpr s32 Signed(u32 value) {
    return static_cast<s32>(value);
}

/// Whether `writes` writes at `stride` from `first` stay below the macro methods.
[[nodiscard]] constexpr bool MethodsFit(u32 first, u64 writes, u32 stride) {
    return writes == 0 || first + (writes - 1) * stride < FirstMacroMethod;
}

// 0xe02 (start 0x21): binds a constant buffer to a stage, caching it per stage and slot.
//   r5 = p0; if (p0 >= 8 signed) { r1 = 0x480 | (p0 & 3) << 5 | (p1 & 15) << 1;
//   r5 = dram[0x520 + p0]; dram[r1] = p2; dram[r1 + 1] = p3; }  if (r5 >= 15 signed) stop;
//   8e0 = (p3 >> 12) & 0xfffff, 8e1 = p3 & 0xff, 8e2 = p2; (0x904 | (r5 & 7) << 3) = 1 | (p1 + 3) << 4
template <typename C>
bool E02(C& c) {
    if (c.Count() < 4) {
        return false;
    }
    const u32 p0 = c.Param(0), p1 = c.Param(1), p2 = c.Param(2), p3 = c.Param(3);
    u32* const dram = c.Dram();
    u32 stage = p0;
    if (Signed(p0) >= 8) {
        const u32 index = 0x520 + p0;
        if (index >= DataRamWords) {
            return false;
        }
        const u32 cache = 0x480 | ((p0 & 3) << 5) | ((p1 & 15) << 1);
        stage = dram[index];
        dram[cache] = p2;
        dram[cache + 1] = p3;
    }
    if (Signed(stage) >= 15) {
        return true;
    }
    c.Emit(0x8e0, (p3 >> 12) & 0xfffff);
    c.Emit(0x8e1, p3 & 0xff);
    c.Emit(0x8e2, p2);
    c.Emit(0x904 | ((stage & 7) << 3), 1 | ((p1 + 3) << 4));
    return true;
}

// 0xe04 (start 0x1e2): one non-instanced indexed draw.
//   586 = p0, 5f2 = p1, 5f3 = p2, 5f6 = p3, 5f7 = 0, 5f8 = p4, 585 = 0
template <typename C>
bool E04(C& c) {
    if (c.Count() < 5) {
        return false;
    }
    c.Emit(0x586, c.Param(0));
    c.Emit(0x5f2, c.Param(1));
    c.Emit(0x5f3, c.Param(2));
    c.Emit(0x5f6, c.Param(3));
    c.Emit(0x5f7, 0);
    c.Emit(0x5f8, c.Param(4));
    c.Emit(0x585, 0);
    return true;
}

// 0xe08 (start 0x8): a stage's program; caches (p1..p4) at dram[0x500 + 4 * p0].
//   b = 0x500 + 4 * p0; if (p1 == dram[b]) stop; e = dram[0x52c] & 1 << p0;
//   dram[b..b+3] = p1..p4; if (!e) stop; s = ((p4 >> 24) & 7) << 4;
//   805|s = p2 >> 24, 806|s = (p2 & 0xffffff) << 8, 807|s = (p4 >> 9) & 0x7f, 8e0 = 0x10000,
//   8e1 = p3 >> 24, 8e2 = (p3 & 0xffffff) << 8, (0x904 | ((p4 >> 28) & 7) << 3) = 0x11,
//   803|s = p4 & 0x1ff, 800|s = 1 | s
template <typename C>
bool E08(C& c) {
    if (c.Count() < 2) {
        return false;
    }
    u32* const dram = c.Dram();
    const u32 p0 = c.Param(0), p1 = c.Param(1);
    const u32 base = 0x500 + p0 * 4;
    if (base >= DataRamWords) {
        return false;
    }
    if (p1 == dram[base]) {
        return true;
    }
    if (c.Count() < 5 || base + 3 >= DataRamWords) {
        return false;
    }
    const u32 p2 = c.Param(2), p3 = c.Param(3), p4 = c.Param(4);
    const u32 enabled = dram[0x52c] & (1U << (p0 & 31));
    dram[base] = p1;
    dram[base + 1] = p2;
    dram[base + 2] = p3;
    dram[base + 3] = p4;
    if (enabled == 0) {
        return true;
    }
    const u32 stage = ((p4 >> 24) & 7) << 4;
    c.Emit(0x805 | stage, p2 >> 24);
    c.Emit(0x806 | stage, (p2 & 0xffffff) << 8);
    c.Emit(0x807 | stage, (p4 >> 9) & 0x7f);
    c.Emit(0x8e0, 0x10000);
    c.Emit(0x8e1, p3 >> 24);
    c.Emit(0x8e2, (p3 & 0xffffff) << 8);
    c.Emit(0x904 | (((p4 >> 28) & 7) << 3), 0x11);
    c.Emit(0x803 | stage, p4 & 0x1ff);
    c.Emit(0x800 | stage, 1 | stage);
    return true;
}

// 0xe10 (start 0x44): if (p0, p1, p2) != dram[0x548..0x54a]: dram[0x548..0x54a] = p0..p2,
// 4b2 = 0x19.
template <typename C>
bool E10(C& c) {
    if (c.Count() < 3) {
        return false;
    }
    u32* const dram = c.Dram();
    const u32 p0 = c.Param(0), p1 = c.Param(1), p2 = c.Param(2);
    if (((dram[0x548] ^ p0) | (dram[0x549] ^ p1) | (dram[0x54a] ^ p2)) == 0) {
        return true;
    }
    dram[0x548] = p0;
    dram[0x549] = p1;
    dram[0x54a] = p2;
    c.Emit(0x4b2, 0x19);
    return true;
}

// 0xe1c (start 0x75): selects the constant buffer cached at dram[0x53c + p0].
//   a = dram[0x53c + p0]; 8e0 = dram[0x543], 8e1 = a >> 24, 8e2 = (a & 0xffffff) << 8
template <typename C>
bool E1C(C& c) {
    if (c.Count() < 1) {
        return false;
    }
    const u32 index = 0x53c + c.Param(0);
    if (index >= DataRamWords) {
        return false;
    }
    const u32* const dram = c.Dram();
    const u32 size = dram[0x543];
    const u32 address = dram[index];
    c.Emit(0x8e0, size);
    c.Emit(0x8e1, address >> 24);
    c.Emit(0x8e2, (address & 0xffffff) << 8);
    return true;
}

// 0xe1e (start 0x79): 671 = 0, then 4d8 + i = bit 8 + i of p0 for i in 0..7.
template <typename C>
bool E1E(C& c) {
    if (c.Count() < 1) {
        return false;
    }
    const u32 p0 = c.Param(0);
    c.Emit(0x671, 0);
    for (u32 i = 0; i < 8; ++i) {
        c.Emit(0x4d8 + i, (p0 >> (8 + i)) & 1);
    }
    return true;
}

// 0xe20 (start 0x7e): 680 + 2i = p_i & 0x1fff, 681 + 2i = (p_i >> 16) & 0x1fff for i in 0..3.
template <typename C>
bool E20(C& c) {
    if (c.Count() < 4) {
        return false;
    }
    for (u32 i = 0; i < 4; ++i) {
        const u32 p = c.Param(i);
        c.Emit(0x680 + i * 2, p & 0x1fff);
        c.Emit(0x681 + i * 2, (p >> 16) & 0x1fff);
    }
    return true;
}

// 0xe28 (start 0x12e): s = dram[0x52d]; dram[0x52e] = p0; if (p0 & 1) { dram[0x52f + i] = p(1+i),
// and if (!s) 490 + i = p(1+i), for i in 0..7 }; if (!s) 82 = p0 & 1.
template <typename C>
bool E28(C& c) {
    if (c.Count() < 1) {
        return false;
    }
    const u32 p0 = c.Param(0);
    const u32 flag = p0 & 1;
    if (flag != 0 && c.Count() < 9) {
        return false;
    }
    u32* const dram = c.Dram();
    const u32 skip = dram[0x52d];
    dram[0x52e] = p0;
    if (flag != 0) {
        for (u32 i = 0; i < 8; ++i) {
            dram[0x52f + i] = c.Param(1 + i);
        }
        if (skip == 0) {
            for (u32 i = 0; i < 8; ++i) {
                c.Emit(0x490 + i, c.Param(1 + i));
            }
        }
    }
    if (skip == 0) {
        c.Emit(0x82, flag);
    }
    return true;
}

// 0xe2a (start 0x140): render_enable.mode = p0.
template <typename C>
bool E2A(C& c) {
    if (c.Count() < 1) {
        return false;
    }
    c.Emit(0x556, c.Param(0));
    return true;
}

// 0xe2c (start 0x142): conditional rendering on two 64-bit query results.
//   154 = p1, 155 = p2, 158 = 6 (MME_DMA_READ_FIFOED: six words onto the FIFO, p3..p8);
//   556 = ((p3 ^ p7) | (p4 ^ p8)) != 0 ? p0 ^ 1 : p0
template <typename C>
bool E2C(C& c) {
    if (c.Count() < 3) {
        return false;
    }
    const u32 p0 = c.Param(0);
    if (c.OcclusionHle() && c.Count() == 3 && p0 <= 1 && (c.Param(1) & ~0x1ffffffU) == 0) {
        // EDEN_NX2_OCCLUSION_HLE (from lichii-OoTPerf): the same comparison as render_enable on
        // the report address (counters at +0 and +16), decided by the query fast path, host
        // conditional rendering or the back, instead of reading the counters on the front.
        // p0 = 0 renders when they differ (IfNotEqual), p0 = 1 when they are equal (IfEqual).
        // render_enable's address is put back: the macro never writes it.
        const u32 high = c.Param(1), low = c.Param(2);
        const u32 old_high = c.Read(0x554), old_low = c.Read(0x555);
        c.Emit(0x154, high);
        c.Emit(0x155, low);
        c.Emit(0x554, high);
        c.Emit(0x555, low);
        c.Emit(0x556, p0 != 0 ? 3U : 4U);
        c.Emit(0x554, old_high);
        c.Emit(0x555, old_low);
        return true;
    }
    c.Emit(0x154, c.Param(1));
    c.Emit(0x155, c.Param(2));
    c.Emit(0x158, 6);
    if (c.Count() < 9) {
        c.Fail("parameter FIFO underflow");
        return true;
    }
    const u32 differ = (c.Param(3) ^ c.Param(7)) | (c.Param(4) ^ c.Param(8));
    c.Emit(0x556, differ != 0 ? (p0 ^ 1) : p0);
    return true;
}

// 0xe36 (start 0x180): selects the 0x900-byte constant buffer at dram[0x537], offset p0 * 4.
//   8e0 = 0x900, 8e1 = a >> 24, 8e2 = (a & 0xffffff) << 8, 8e3 = p0 << 2
template <typename C>
bool E36(C& c) {
    if (c.Count() < 1) {
        return false;
    }
    const u32 address = c.Dram()[0x537];
    c.Emit(0x8e0, 0x900);
    c.Emit(0x8e1, address >> 24);
    c.Emit(0x8e2, (address & 0xffffff) << 8);
    c.Emit(0x8e3, c.Param(0) << 2);
    return true;
}

// 0xe3a (start 0x187): s = p0 & 15; e = dram[0x52c] & 1 << s; dram[0x500 + 4s] = 0;
// if (e) (0x800 | b) = b with b = ((p0 >> 4) & 7) << 4.
template <typename C>
bool E3A(C& c) {
    if (c.Count() < 1) {
        return false;
    }
    u32* const dram = c.Dram();
    const u32 p0 = c.Param(0);
    const u32 stage = p0 & 15;
    const u32 enabled = dram[0x52c] & (1U << stage);
    dram[0x500 + stage * 4] = 0;
    if (enabled != 0) {
        const u32 bits = ((p0 >> 4) & 7) << 4;
        c.Emit(0x800 | bits, bits);
    }
    return true;
}

// 0xe3e (start 0x1a7): no parameters; 650 = bits of a data RAM word picked by dram[0x52d].
template <typename C>
bool E3E(C& c) {
    const u32* const dram = c.Dram();
    const u32 mode = dram[0x52d];
    u32 word;
    if (mode == 0) {
        word = dram[0x507];
        if (dram[0x508] != 0) {
            word = dram[0x50b];
        }
        if (dram[0x50c] != 0) {
            word = dram[0x50f];
        }
        if (dram[0x510] != 0) {
            word = dram[0x513];
        }
    } else {
        word = mode == 2 ? dram[0x523] : dram[0x51f];
    }
    const auto bit = [word](u32 from, u32 to) { return ((word >> from) & 1) << to; };
    u32 value = bit(16, 0) | bit(17, 4);
    if (((word >> 20) & 15) != 0) {
        value |= bit(20, 16) | bit(21, 20) | bit(22, 24) | bit(23, 28);
    }
    value |= bit(18, 8) | bit(19, 12);
    c.Emit(0x650, value);
    return true;
}

// The instance loop of 0xe52 and 0xea0: `instances` times draw.begin, index_buffer.count and
// draw.end, the instances after the first with draw.begin's instance id Subsequent.
template <typename C>
void DrawLoop(C& c, u32 begin, u32 count, u32 instances) {
    c.Emit(0x586, begin);
    c.Emit(0x5f8, count);
    // A loop's first instance (instance id First, as the back's register holds it): the back's
    // deferred draw of the loop covers it.
    if (instances > 1 && ((c.Read(0x586) >> 26) & 3) == 0) {
        c.FoldFirstInstance();
    }
    c.Emit(0x585, 0);
    if (instances > 1) {
        const u32 subsequent = (begin & ~(3U << 26)) | (1U << 26);
        const std::array<u32, 2> rest[3]{{0x586, subsequent}, {0x5f8, count}, {0x585, 0}};
        c.EmitRepeated(rest, 3, instances - 1);
    }
}

// 0xe52 (start 0x1f4): instanced indexed draw. p0 draw.begin, p1 count, p2 instances, p3 first,
// p4 base vertex, p5 base instance.
//   5f7 = p3, 446 = p4, 50d = p4, 50e = p5, DrawLoop(p0, p1, p2), 446 = 0, 50d = 0, 50e = 0
template <typename C>
bool E52(C& c) {
    if (c.Count() < 6) {
        return false;
    }
    const u32 instances = c.Param(2);
    if (instances > MaxInstances) {
        return false;
    }
    c.Emit(0x5f7, c.Param(3));
    c.Emit(0x446, c.Param(4));
    c.Emit(0x50d, c.Param(4));
    c.Emit(0x50e, c.Param(5));
    if (instances != 0) {
        DrawLoop(c, c.Param(0), c.Param(1), instances);
    }
    c.Emit(0x446, 0);
    c.Emit(0x50d, 0);
    c.Emit(0x50e, 0);
    return true;
}

// 0xe5c (start 0x23d): l = dram[0x53a]; t = (p1 >> 8) & 0x1ff; dram[0x538] = p0; dram[0x539] = t;
// v = p0, with bits 24-31 = (l <= t signed ? l : t) when bit 8 of l is set; d8 = v, d9 = p1 & 0xff.
template <typename C>
bool E5C(C& c) {
    if (c.Count() < 2) {
        return false;
    }
    u32* const dram = c.Dram();
    const u32 p0 = c.Param(0), p1 = c.Param(1);
    const u32 limit = dram[0x53a];
    const u32 top = (p1 >> 8) & 0x1ff;
    dram[0x538] = p0;
    dram[0x539] = top;
    u32 value = p0;
    if (((limit >> 8) & 1) != 0) {
        const u32 high = Signed(limit) <= Signed(top) ? limit : top;
        value = (p0 & 0xffffff) | ((high & 0xff) << 24);
    }
    c.Emit(0xd8, value);
    c.Emit(0xd9, p1 & 0xff);
    return true;
}

// 0xe68 (start 0x262): dram[p0] = p1.
template <typename C>
bool E68(C& c) {
    if (c.Count() < 2 || c.Param(0) >= DataRamWords) {
        return false;
    }
    c.Dram()[c.Param(0)] = c.Param(1);
    return true;
}

// 0xe8c (start 0x2f7): takes one parameter, does nothing.
template <typename C>
bool E8C(C& c) {
    return c.Count() >= 1;
}

// 0xe92 (start 0x302): o = dram[0x54b]; dram[0x54b] = p0; if (o | p0) 47d = 0.
template <typename C>
bool E92(C& c) {
    if (c.Count() < 1) {
        return false;
    }
    u32* const dram = c.Dram();
    const u32 p0 = c.Param(0);
    const u32 any = dram[0x54b] | p0;
    dram[0x54b] = p0;
    if (any != 0) {
        c.Emit(0x47d, 0);
    }
    return true;
}

// 0xe96 (start 0x308): vertex attribute formats. n = p0 pairs of parameters to 458 + i; then
// dram[0x54c] = n and the pairs the last call set beyond n (dram[0x54c] - n, signed) get
// 0x3a400040.
template <typename C>
bool E96(C& c) {
    if (c.Count() < 1) {
        return false;
    }
    u32* const dram = c.Dram();
    const u32 pairs = c.Param(0);
    const u32 pad = dram[0x54c] - pairs;
    const u64 pad_pairs = Signed(pad) > 0 ? pad : 0;
    if (u64{pairs} * 2 + 1 > c.Count() || !MethodsFit(0x458, (u64{pairs} + pad_pairs) * 2, 1)) {
        return false;
    }
    u32 method = 0x458;
    for (u32 i = 0; i < pairs * 2; ++i) {
        c.Emit(method++, c.Param(1 + i));
    }
    dram[0x54c] = pairs;
    for (u64 i = 0; i < pad_pairs * 2; ++i) {
        c.Emit(method++, 0x3a400040);
    }
    return true;
}

// 0xe9a (start 0x311): vertex streams. n0 = p0 pairs of parameters to 700 + 4i, zero for the
// pairs beyond n0 the last call set (dram[0x54d]); then n1 = p(1+2n0), b = p(2+2n0): n1 pairs of
// parameters to 703 + 4i, then 620 + j = bit j of b for j < 2n1, zero for the pairs beyond n1 the
// last call set (dram[0x54e]).
template <typename C>
bool E9A(C& c) {
    const u32 count = c.Count();
    if (count < 1) {
        return false;
    }
    u32* const dram = c.Dram();
    const u32 n0 = c.Param(0);
    if (u64{n0} * 2 + 3 > count) {
        return false;
    }
    const u32 n1 = c.Param(1 + n0 * 2);
    const u32 bits = c.Param(2 + n0 * 2);
    if (u64{n0} * 2 + 3 + u64{n1} * 2 > count) {
        return false;
    }
    const u32 pad0 = dram[0x54d] - n0;
    const u64 pad0_pairs = Signed(pad0) > 0 ? pad0 : 0;
    const u32 pad1 = dram[0x54e] - n1;
    const u64 pad1_pairs = Signed(pad1) > 0 ? pad1 : 0;
    if (!MethodsFit(0x700, (u64{n0} + pad0_pairs) * 2, 4) || !MethodsFit(0x703, u64{n1} * 2, 4) ||
        !MethodsFit(0x620, (u64{n1} + pad1_pairs) * 2, 1)) {
        return false;
    }
    u32 method = 0x700;
    for (u32 i = 0; i < n0 * 2; ++i, method += 4) {
        c.Emit(method, c.Param(1 + i));
    }
    dram[0x54d] = n0;
    for (u64 i = 0; i < pad0_pairs * 2; ++i, method += 4) {
        c.Emit(method, 0);
    }
    const u32 base = 3 + n0 * 2;
    method = 0x703;
    for (u32 i = 0; i < n1 * 2; ++i, method += 4) {
        c.Emit(method, c.Param(base + i));
    }
    method = 0x620;
    u32 rest = bits;
    for (u32 i = 0; i < n1; ++i, rest >>= 2) {
        c.Emit(method++, rest & 1);
        c.Emit(method++, (rest >> 1) & 1);
    }
    dram[0x54e] = n1;
    for (u64 i = 0; i < pad1_pairs * 2; ++i) {
        c.Emit(method++, 0);
    }
    return true;
}

// 0xea0 (start 0x1e6): instanced indexed draw with the base vertex and instance in the driver's
// constant buffer (offset 0x640). Parameters as 0xe52.
//   5f7 = p3, 446 = p4, 50d = p4, 50e = p5, 8e3 = 0x640, 8e4 = p4, 8e5 = p5, DrawLoop(p0, p1, p2),
//   446 = 0, 50d = 0, 50e = 0, 8e3 = 0x640, 8e4 = 0, 8e5 = 0
template <typename C>
bool EA0(C& c) {
    if (c.Count() < 6) {
        return false;
    }
    const u32 instances = c.Param(2);
    if (instances > MaxInstances) {
        return false;
    }
    const u32 base_vertex = c.Param(4), base_instance = c.Param(5);
    c.Emit(0x5f7, c.Param(3));
    c.Emit(0x446, base_vertex);
    c.Emit(0x50d, base_vertex);
    c.Emit(0x50e, base_instance);
    c.Emit(0x8e3, 0x640);
    c.Emit(0x8e4, base_vertex);
    c.Emit(0x8e5, base_instance);
    if (instances != 0) {
        DrawLoop(c, c.Param(0), c.Param(1), instances);
    }
    c.Emit(0x446, 0);
    c.Emit(0x50d, 0);
    c.Emit(0x50e, 0);
    c.Emit(0x8e3, 0x640);
    c.Emit(0x8e4, 0);
    c.Emit(0x8e5, 0);
    return true;
}

// 0xece (start 0x4dc): o = dram[0x54f]; dram[0x54f] = (o & 0xffff) | p0 << 16;
// 54d = (o << 16) != 0 ? 1 : p0 & 1.
template <typename C>
bool ECE(C& c) {
    if (c.Count() < 1) {
        return false;
    }
    u32* const dram = c.Dram();
    const u32 p0 = c.Param(0);
    const u32 old = dram[0x54f];
    const u32 value = (old << 16) != 0 ? 1 : (p0 & 1);
    dram[0x54f] = (old & 0xffff) | (p0 << 16);
    c.Emit(0x54d, value);
    return true;
}

// 0xed6 (start 0x4ee): s = dram[0x545] with the bits p0 >> 16 selects taken from p0;
// dram[0x545] = s; 47c = a bit of s picked by dram[0x52d] and bits 5 and 7 of s.
template <typename C>
bool ED6(C& c) {
    if (c.Count() < 1) {
        return false;
    }
    u32* const dram = c.Dram();
    const u32 p0 = c.Param(0);
    const u32 mask = p0 >> 16;
    const u32 state = (dram[0x545] & ~mask) | (p0 & mask);
    u32 value;
    if (dram[0x52d] == 0) {
        value = state & 1;
        if (((state >> 5) & 1) != 0) {
            value = (state >> 4) & 1;
        }
        if (((state >> 7) & 1) != 0) {
            value = (state >> 6) & 1;
        }
    } else {
        value = (state >> 8) & 1;
    }
    dram[0x545] = state;
    c.Emit(0x47c, value);
    return true;
}

} // namespace Detail

/// Runs replacement `id` (Find) for the call in `c`. False: run the call through the JIT.
template <typename C>
bool Run(Id id, C& c) {
    switch (id) {
    case Id::E02:
        return Detail::E02(c);
    case Id::E04:
        return Detail::E04(c);
    case Id::E08:
        return Detail::E08(c);
    case Id::E10:
        return Detail::E10(c);
    case Id::E1C:
        return Detail::E1C(c);
    case Id::E1E:
        return Detail::E1E(c);
    case Id::E20:
        return Detail::E20(c);
    case Id::E28:
        return Detail::E28(c);
    case Id::E2A:
        return Detail::E2A(c);
    case Id::E2C:
        return Detail::E2C(c);
    case Id::E36:
        return Detail::E36(c);
    case Id::E3A:
        return Detail::E3A(c);
    case Id::E3E:
        return Detail::E3E(c);
    case Id::E52:
        return Detail::E52(c);
    case Id::E5C:
        return Detail::E5C(c);
    case Id::E68:
        return Detail::E68(c);
    case Id::E8C:
        return Detail::E8C(c);
    case Id::E92:
        return Detail::E92(c);
    case Id::E96:
        return Detail::E96(c);
    case Id::E9A:
        return Detail::E9A(c);
    case Id::EA0:
        return Detail::EA0(c);
    case Id::ECE:
        return Detail::ECE(c);
    case Id::ED6:
        return Detail::ED6(c);
    case Id::None:
        break;
    }
    return false;
}

} // namespace Tegra::OotMacros
