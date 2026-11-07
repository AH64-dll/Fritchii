// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <array>
#include <initializer_list>
#include <tuple>
#include <type_traits>
#include <utility>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include "dynarmic/mcl/integer_of_size.hpp"
#include "dynarmic/backend/x64/xbyak.h"

#include "dynarmic/backend/x64/a64_emit_x64.h"
#include "dynarmic/backend/x64/abi.h"
#include "dynarmic/backend/x64/devirtualize.h"
#include "dynarmic/backend/x64/emit_x64_memory.h"
#include "dynarmic/backend/x64/exclusive_monitor_friend.h"
#include "dynarmic/backend/x64/perf_map.h"
#include "dynarmic/common/spin_lock_x64.h"
#include "dynarmic/interface/exclusive_monitor.h"

namespace Dynarmic::Backend::X64 {

using namespace Xbyak::util;

void A64EmitX64::GenMemory128Accessors() {
    code.align();
    memory_read_128 = code.getCurr<void (*)()>();
#ifdef _WIN32
    Devirtualize<&A64::UserCallbacks::MemoryRead128>(conf.callbacks).EmitCallWithReturnPointer(code, [&](Xbyak::Reg64 return_value_ptr, [[maybe_unused]] RegList args) {
        code.mov(code.ABI_PARAM3, code.ABI_PARAM2);
        code.lea(rsp, ptr[rsp - (8 + 16 + ABI_SHADOW_SPACE)]);
        code.lea(return_value_ptr, ptr[rsp + ABI_SHADOW_SPACE]);
    });
    code.movups(xmm1, xword[code.ABI_RETURN]);
    code.add(rsp, 8 + 16 + ABI_SHADOW_SPACE);
#else
    code.lea(rsp, ptr[rsp - 8]);
    Devirtualize<&A64::UserCallbacks::MemoryRead128>(conf.callbacks).EmitCall(code);
    if (code.HasHostFeature(HostFeature::SSE41)) {
        code.movq(xmm1, code.ABI_RETURN);
        code.pinsrq(xmm1, code.ABI_RETURN2, 1);
    } else {
        code.movq(xmm1, code.ABI_RETURN);
        code.movq(xmm2, code.ABI_RETURN2);
        code.punpcklqdq(xmm1, xmm2);
    }
    code.add(rsp, 8);
#endif
    code.ret();
    PerfMapRegister(memory_read_128, code.getCurr(), "a64_memory_read_128");

    code.align();
    memory_write_128 = code.getCurr<void (*)()>();
#ifdef _WIN32
    code.lea(rsp, ptr[rsp - (8 + 16 + ABI_SHADOW_SPACE)]);
    code.lea(code.ABI_PARAM3, ptr[rsp + ABI_SHADOW_SPACE]);
    code.movaps(xword[code.ABI_PARAM3], xmm1);
    Devirtualize<&A64::UserCallbacks::MemoryWrite128>(conf.callbacks).EmitCall(code);
    code.add(rsp, 8 + 16 + ABI_SHADOW_SPACE);
#else
    code.lea(rsp, ptr[rsp - 8]);
    if (code.HasHostFeature(HostFeature::SSE41)) {
        code.movq(code.ABI_PARAM3, xmm1);
        code.pextrq(code.ABI_PARAM4, xmm1, 1);
    } else {
        code.movq(code.ABI_PARAM3, xmm1);
        code.punpckhqdq(xmm1, xmm1);
        code.movq(code.ABI_PARAM4, xmm1);
    }
    Devirtualize<&A64::UserCallbacks::MemoryWrite128>(conf.callbacks).EmitCall(code);
    code.add(rsp, 8);
#endif
    code.ret();
    PerfMapRegister(memory_write_128, code.getCurr(), "a64_memory_write_128");

    code.align();
    memory_exclusive_write_128 = code.getCurr<void (*)()>();
#ifdef _WIN32
    code.lea(rsp, ptr[rsp - (8 + 32 + ABI_SHADOW_SPACE)]);
    code.lea(code.ABI_PARAM3, ptr[rsp + ABI_SHADOW_SPACE]);
    code.lea(code.ABI_PARAM4, ptr[rsp + ABI_SHADOW_SPACE + 16]);
    code.movaps(xword[code.ABI_PARAM3], xmm1);
    code.movaps(xword[code.ABI_PARAM4], xmm2);
    Devirtualize<&A64::UserCallbacks::MemoryWriteExclusive128>(conf.callbacks).EmitCall(code);
    code.add(rsp, 8 + 32 + ABI_SHADOW_SPACE);
#else
    code.lea(rsp, ptr[rsp - 8]);
    if (code.HasHostFeature(HostFeature::SSE41)) {
        code.movq(code.ABI_PARAM3, xmm1);
        code.pextrq(code.ABI_PARAM4, xmm1, 1);
        code.movq(code.ABI_PARAM5, xmm2);
        code.pextrq(code.ABI_PARAM6, xmm2, 1);
    } else {
        code.movq(code.ABI_PARAM3, xmm1);
        code.punpckhqdq(xmm1, xmm1);
        code.movq(code.ABI_PARAM4, xmm1);
        code.movq(code.ABI_PARAM5, xmm2);
        code.punpckhqdq(xmm2, xmm2);
        code.movq(code.ABI_PARAM6, xmm2);
    }
    Devirtualize<&A64::UserCallbacks::MemoryWriteExclusive128>(conf.callbacks).EmitCall(code);
    code.add(rsp, 8);
#endif
    code.ret();
    PerfMapRegister(memory_exclusive_write_128, code.getCurr(), "a64_memory_exclusive_write_128");
}

void A64EmitX64::GenFastmemFallbacks() {
    const std::initializer_list<int> idxes{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const std::array<std::pair<size_t, ArgCallback>, 4> read_callbacks{{
        {8, Devirtualize<&A64::UserCallbacks::MemoryRead>(conf.callbacks)},
        {16, Devirtualize<&A64::UserCallbacks::MemoryRead>(conf.callbacks)},
        {32, Devirtualize<&A64::UserCallbacks::MemoryRead>(conf.callbacks)},
        {64, Devirtualize<&A64::UserCallbacks::MemoryRead>(conf.callbacks)},
    }};
    const std::array<std::pair<size_t, ArgCallback>, 4> write_callbacks{{
        {8, Devirtualize<&A64::UserCallbacks::MemoryWrite>(conf.callbacks)},
        {16, Devirtualize<&A64::UserCallbacks::MemoryWrite>(conf.callbacks)},
        {32, Devirtualize<&A64::UserCallbacks::MemoryWrite>(conf.callbacks)},
        {64, Devirtualize<&A64::UserCallbacks::MemoryWrite>(conf.callbacks)},
    }};
    const std::array<std::pair<size_t, ArgCallback>, 4> exclusive_write_callbacks{{
        {8, Devirtualize<&A64::UserCallbacks::MemoryWriteExclusive8>(conf.callbacks)},
        {16, Devirtualize<&A64::UserCallbacks::MemoryWriteExclusive16>(conf.callbacks)},
        {32, Devirtualize<&A64::UserCallbacks::MemoryWriteExclusive32>(conf.callbacks)},
        {64, Devirtualize<&A64::UserCallbacks::MemoryWriteExclusive64>(conf.callbacks)},
    }};

    for (auto const ordered : {false, true}) {
        for (auto const vaddr_idx : idxes) {
            if (vaddr_idx == 4 || vaddr_idx == 15) {
                continue;
            }

            for (int value_idx : idxes) {
                code.align();
                read_fallbacks[std::make_tuple(ordered, 128, vaddr_idx, value_idx)] = code.getCurr<void (*)()>();
                ABI_PushCallerSaveRegistersAndAdjustStackExcept(code, HostLocXmmIdx(value_idx));
                if (vaddr_idx != code.ABI_PARAM2.getIdx()) {
                    code.mov(code.ABI_PARAM2, Xbyak::Reg64{vaddr_idx});
                }
                if (ordered) {
                    code.mfence();
                }
                code.call(memory_read_128);
                if (value_idx != 1) {
                    code.movaps(Xbyak::Xmm{value_idx}, xmm1);
                }
                ABI_PopCallerSaveRegistersAndAdjustStackExcept(code, HostLocXmmIdx(value_idx));
                code.ret();
                PerfMapRegister(read_fallbacks[std::make_tuple(ordered, 128, vaddr_idx, value_idx)], code.getCurr(), "a64_read_fallback_128");

                code.align();
                write_fallbacks[std::make_tuple(ordered, 128, vaddr_idx, value_idx)] = code.getCurr<void (*)()>();
                ABI_PushCallerSaveRegistersAndAdjustStack(code);
                if (vaddr_idx != code.ABI_PARAM2.getIdx()) {
                    code.mov(code.ABI_PARAM2, Xbyak::Reg64{vaddr_idx});
                }
                if (value_idx != 1) {
                    code.movaps(xmm1, Xbyak::Xmm{value_idx});
                }
                code.call(memory_write_128);
                if (ordered) {
                    code.mfence();
                }
                ABI_PopCallerSaveRegistersAndAdjustStack(code);
                code.ret();
                PerfMapRegister(write_fallbacks[std::make_tuple(ordered, 128, vaddr_idx, value_idx)], code.getCurr(), "a64_write_fallback_128");

                code.align();
                exclusive_write_fallbacks[std::make_tuple(ordered, 128, vaddr_idx, value_idx)] = code.getCurr<void (*)()>();
                ABI_PushCallerSaveRegistersAndAdjustStackExcept(code, HostLoc::RAX);
                if (value_idx != 1) {
                    code.movaps(xmm1, Xbyak::Xmm{value_idx});
                }
                if (code.HasHostFeature(HostFeature::SSE41)) {
                    code.movq(xmm2, rax);
                    code.pinsrq(xmm2, rdx, 1);
                } else {
                    code.movq(xmm2, rax);
                    code.movq(xmm0, rdx);
                    code.punpcklqdq(xmm2, xmm0);
                }
                if (vaddr_idx != code.ABI_PARAM2.getIdx()) {
                    code.mov(code.ABI_PARAM2, Xbyak::Reg64{vaddr_idx});
                }
                code.call(memory_exclusive_write_128);
                ABI_PopCallerSaveRegistersAndAdjustStackExcept(code, HostLoc::RAX);
                code.ret();
                PerfMapRegister(exclusive_write_fallbacks[std::make_tuple(ordered, 128, vaddr_idx, value_idx)], code.getCurr(), "a64_exclusive_write_fallback_128");

                if (value_idx == 4 || value_idx == 15) {
                    continue;
                }

                for (const auto& [bitsize, callback] : read_callbacks) {
                    code.align();
                    read_fallbacks[std::make_tuple(ordered, bitsize, vaddr_idx, value_idx)] = code.getCurr<void (*)()>();
                    ABI_PushCallerSaveRegistersAndAdjustStackExcept(code, HostLocRegIdx(value_idx));
                    // params = { this, vaddr, size }
                    if (vaddr_idx != code.ABI_PARAM2.getIdx()) {
                        code.mov(code.ABI_PARAM2, Xbyak::Reg64{vaddr_idx});
                    }
                    code.mov(code.ABI_PARAM3, bitsize / CHAR_BIT);
                    if (ordered) code.mfence();
                    callback.EmitCall(code);
                    if (value_idx != code.ABI_RETURN.getIdx()) {
                        code.mov(Xbyak::Reg64{value_idx}, code.ABI_RETURN);
                    }
                    ABI_PopCallerSaveRegistersAndAdjustStackExcept(code, HostLocRegIdx(value_idx));
                    code.ZeroExtendFrom(bitsize, Xbyak::Reg64{value_idx});
                    code.ret();
                    PerfMapRegister(read_fallbacks[std::make_tuple(ordered, bitsize, vaddr_idx, value_idx)], code.getCurr(), fmt::format("a64_read_fallback_{}", bitsize));
                }

                for (const auto& [bitsize, callback] : write_callbacks) {
                    code.align();
                    write_fallbacks[std::make_tuple(ordered, bitsize, vaddr_idx, value_idx)] = code.getCurr<void (*)()>();
                    ABI_PushCallerSaveRegistersAndAdjustStack(code);
                    // params = { this, vaddr, value, size }
                    if (vaddr_idx == code.ABI_PARAM3.getIdx() && value_idx == code.ABI_PARAM2.getIdx()) {
                        code.xchg(code.ABI_PARAM2, code.ABI_PARAM3);
                    } else if (vaddr_idx == code.ABI_PARAM3.getIdx()) {
                        code.mov(code.ABI_PARAM2, Xbyak::Reg64{vaddr_idx});
                        if (value_idx != code.ABI_PARAM3.getIdx()) {
                            code.mov(code.ABI_PARAM3, Xbyak::Reg64{value_idx});
                        }
                    } else {
                        if (value_idx != code.ABI_PARAM3.getIdx()) {
                            code.mov(code.ABI_PARAM3, Xbyak::Reg64{value_idx});
                        }
                        if (vaddr_idx != code.ABI_PARAM2.getIdx()) {
                            code.mov(code.ABI_PARAM2, Xbyak::Reg64{vaddr_idx});
                        }
                    }
                    code.ZeroExtendFrom(bitsize, code.ABI_PARAM3);
                    code.mov(code.ABI_PARAM4, bitsize / CHAR_BIT);
                    callback.EmitCall(code);
                    if (ordered) code.mfence();
                    ABI_PopCallerSaveRegistersAndAdjustStack(code);
                    code.ret();
                    PerfMapRegister(write_fallbacks[std::make_tuple(ordered, bitsize, vaddr_idx, value_idx)], code.getCurr(), fmt::format("a64_write_fallback_{}", bitsize));
                }

                for (const auto& [bitsize, callback] : exclusive_write_callbacks) {
                    code.align();
                    exclusive_write_fallbacks[std::make_tuple(ordered, bitsize, vaddr_idx, value_idx)] = code.getCurr<void (*)()>();
                    ABI_PushCallerSaveRegistersAndAdjustStackExcept(code, HostLoc::RAX);
                    if (vaddr_idx == code.ABI_PARAM3.getIdx() && value_idx == code.ABI_PARAM2.getIdx()) {
                        code.xchg(code.ABI_PARAM2, code.ABI_PARAM3);
                    } else if (vaddr_idx == code.ABI_PARAM3.getIdx()) {
                        code.mov(code.ABI_PARAM2, Xbyak::Reg64{vaddr_idx});
                        if (value_idx != code.ABI_PARAM3.getIdx()) {
                            code.mov(code.ABI_PARAM3, Xbyak::Reg64{value_idx});
                        }
                    } else {
                        if (value_idx != code.ABI_PARAM3.getIdx()) {
                            code.mov(code.ABI_PARAM3, Xbyak::Reg64{value_idx});
                        }
                        if (vaddr_idx != code.ABI_PARAM2.getIdx()) {
                            code.mov(code.ABI_PARAM2, Xbyak::Reg64{vaddr_idx});
                        }
                    }
                    code.ZeroExtendFrom(bitsize, code.ABI_PARAM3);
                    code.mov(code.ABI_PARAM4, rax);
                    code.ZeroExtendFrom(bitsize, code.ABI_PARAM4);
                    callback.EmitCall(code);
                    ABI_PopCallerSaveRegistersAndAdjustStackExcept(code, HostLoc::RAX);
                    code.ret();
                    PerfMapRegister(exclusive_write_fallbacks[std::make_tuple(ordered, bitsize, vaddr_idx, value_idx)], code.getCurr(), fmt::format("a64_exclusive_write_fallback_{}", bitsize));
                }
            }
        }
    }
}

#define Axx A64
#include "dynarmic/backend/x64/emit_x64_memory.cpp.inc"
#undef Axx

void A64EmitX64::EmitA64ReadMemory8(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryRead<8, &A64::UserCallbacks::MemoryRead>(ctx, inst);
}

void A64EmitX64::EmitA64ReadMemory16(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryRead<16, &A64::UserCallbacks::MemoryRead>(ctx, inst);
}

void A64EmitX64::EmitA64ReadMemory32(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryRead<32, &A64::UserCallbacks::MemoryRead>(ctx, inst);
}

void A64EmitX64::EmitA64ReadMemory64(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryRead<64, &A64::UserCallbacks::MemoryRead>(ctx, inst);
}

void A64EmitX64::EmitA64ReadMemory128(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryRead<128, &A64::UserCallbacks::MemoryRead128>(ctx, inst);
}

void A64EmitX64::EmitA64WriteMemory8(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryWrite<8, &A64::UserCallbacks::MemoryWrite>(ctx, inst);
}

void A64EmitX64::EmitA64WriteMemory16(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryWrite<16, &A64::UserCallbacks::MemoryWrite>(ctx, inst);
}

void A64EmitX64::EmitA64WriteMemory32(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryWrite<32, &A64::UserCallbacks::MemoryWrite>(ctx, inst);
}

void A64EmitX64::EmitA64WriteMemory64(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryWrite<64, &A64::UserCallbacks::MemoryWrite>(ctx, inst);
}

void A64EmitX64::EmitA64WriteMemory128(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryWrite<128, &A64::UserCallbacks::MemoryWrite>(ctx, inst);
}

void A64EmitX64::EmitA64ClearExclusive(A64EmitContext&, IR::Inst*) {
    code.mov(code.byte[code.ABI_JIT_PTR + offsetof(A64JitState, exclusive_state)], u8(0));
}

void A64EmitX64::EmitA64ExclusiveReadMemory8(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveReadMemoryInline<8, &A64::UserCallbacks::MemoryRead>(ctx, inst);
    } else {
        EmitExclusiveReadMemory<8, &A64::UserCallbacks::MemoryRead>(ctx, inst);
    }
}

void A64EmitX64::EmitA64ExclusiveReadMemory16(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveReadMemoryInline<16, &A64::UserCallbacks::MemoryRead>(ctx, inst);
    } else {
        EmitExclusiveReadMemory<16, &A64::UserCallbacks::MemoryRead>(ctx, inst);
    }
}

void A64EmitX64::EmitA64ExclusiveReadMemory32(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveReadMemoryInline<32, &A64::UserCallbacks::MemoryRead>(ctx, inst);
    } else {
        EmitExclusiveReadMemory<32, &A64::UserCallbacks::MemoryRead>(ctx, inst);
    }
}

void A64EmitX64::EmitA64ExclusiveReadMemory64(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveReadMemoryInline<64, &A64::UserCallbacks::MemoryRead>(ctx, inst);
    } else {
        EmitExclusiveReadMemory<64, &A64::UserCallbacks::MemoryRead>(ctx, inst);
    }
}

void A64EmitX64::EmitA64ExclusiveReadMemory128(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveReadMemoryInline<128, &A64::UserCallbacks::MemoryRead128>(ctx, inst);
    } else {
        EmitExclusiveReadMemory<128, &A64::UserCallbacks::MemoryRead128>(ctx, inst);
    }
}

void A64EmitX64::EmitA64ExclusiveWriteMemory8(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveWriteMemoryInline<8, &A64::UserCallbacks::MemoryWriteExclusive8>(ctx, inst);
    } else {
        EmitExclusiveWriteMemory<8, &A64::UserCallbacks::MemoryWriteExclusive8>(ctx, inst);
    }
}

void A64EmitX64::EmitA64ExclusiveWriteMemory16(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveWriteMemoryInline<16, &A64::UserCallbacks::MemoryWriteExclusive16>(ctx, inst);
    } else {
        EmitExclusiveWriteMemory<16, &A64::UserCallbacks::MemoryWriteExclusive16>(ctx, inst);
    }
}

void A64EmitX64::EmitA64ExclusiveWriteMemory32(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveWriteMemoryInline<32, &A64::UserCallbacks::MemoryWriteExclusive32>(ctx, inst);
    } else {
        EmitExclusiveWriteMemory<32, &A64::UserCallbacks::MemoryWriteExclusive32>(ctx, inst);
    }
}

void A64EmitX64::EmitA64ExclusiveWriteMemory64(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveWriteMemoryInline<64, &A64::UserCallbacks::MemoryWriteExclusive64>(ctx, inst);
    } else {
        EmitExclusiveWriteMemory<64, &A64::UserCallbacks::MemoryWriteExclusive64>(ctx, inst);
    }
}

void A64EmitX64::EmitA64ExclusiveWriteMemory128(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveWriteMemoryInline<128, &A64::UserCallbacks::MemoryWriteExclusive128>(ctx, inst);
    } else {
        EmitExclusiveWriteMemory<128, &A64::UserCallbacks::MemoryWriteExclusive128>(ctx, inst);
    }
}

namespace {

template<std::size_t bitsize>
using AtomicT = mcl::unsigned_integer_of_size<bitsize>;

template<std::size_t bitsize>
bool AtomicWriteExclusive(A64::UserCallbacks* cb, u64 vaddr, AtomicT<bitsize> value, AtomicT<bitsize> expected) {
    if constexpr (bitsize == 8) {
        return cb->MemoryWriteExclusive8(vaddr, value, expected);
    } else if constexpr (bitsize == 16) {
        return cb->MemoryWriteExclusive16(vaddr, value, expected);
    } else if constexpr (bitsize == 32) {
        return cb->MemoryWriteExclusive32(vaddr, value, expected);
    } else {
        return cb->MemoryWriteExclusive64(vaddr, value, expected);
    }
}

// op is the LSE opc field: 0 ADD, 1 CLR, 2 EOR, 3 SET, 4 SMAX, 5 SMIN, 6 UMAX, 7 UMIN, 8 SWP.
template<std::size_t bitsize>
AtomicT<bitsize> AtomicRmwResult(AtomicT<bitsize> a, AtomicT<bitsize> b, u64 op) {
    using T = AtomicT<bitsize>;
    using S = std::make_signed_t<T>;
    switch (op) {
    case 0:
        return T(a + b);
    case 1:
        return T(a & ~b);
    case 2:
        return T(a ^ b);
    case 3:
        return T(a | b);
    case 4:
        return S(a) > S(b) ? a : b;
    case 5:
        return S(a) < S(b) ? a : b;
    case 6:
        return a > b ? a : b;
    case 7:
        return a < b ? a : b;
    default:
        return b;
    }
}

// Slow paths through the memory callbacks (no fastmem, or a fastmem fault): a CAS loop. A failed
// write that leaves memory unchanged is not a race (unmapped or aborted access): stop there.
template<std::size_t bitsize>
u64 AtomicRmwFallback(const A64::UserConfig& conf, u64 vaddr, u64 operand, u64 op) {
    using T = AtomicT<bitsize>;
    for (;;) {
        const T old = T(conf.callbacks->MemoryRead(vaddr, bitsize / 8));
        if (AtomicWriteExclusive<bitsize>(conf.callbacks, vaddr, AtomicRmwResult<bitsize>(old, T(operand), op), old) ||
            T(conf.callbacks->MemoryRead(vaddr, bitsize / 8)) == old) {
            return old;
        }
    }
}

template<std::size_t bitsize>
u64 AtomicCasFallback(const A64::UserConfig& conf, u64 vaddr, u64 expected, u64 desired) {
    using T = AtomicT<bitsize>;
    for (;;) {
        const T old = T(conf.callbacks->MemoryRead(vaddr, bitsize / 8));
        if (old != T(expected)) {
            return old;
        }
        if (AtomicWriteExclusive<bitsize>(conf.callbacks, vaddr, T(desired), old) ||
            T(conf.callbacks->MemoryRead(vaddr, bitsize / 8)) == old) {
            return old;
        }
    }
}

Xbyak::Reg SizedReg(Xbyak::Reg64 reg, std::size_t bitsize) {
    switch (bitsize) {
    case 8:
        return reg.cvt8();
    case 16:
        return reg.cvt16();
    case 32:
        return reg.cvt32();
    default:
        return reg;
    }
}

Xbyak::Address SizedPtr(const Xbyak::RegExp& addr, std::size_t bitsize) {
    switch (bitsize) {
    case 8:
        return Xbyak::util::byte[addr];
    case 16:
        return word[addr];
    case 32:
        return dword[addr];
    default:
        return qword[addr];
    }
}

}  // anonymous namespace

// Reached as a call from a fastmem fault (or the address-range abort): does the whole operation
// through the callbacks and returns the old value in `result`, all other registers preserved.
void A64EmitX64::EmitAtomicFallbackStub(Xbyak::Reg64 result, std::initializer_list<Xbyak::Reg64> sources, std::optional<u32> param4, AtomicFallbackFn fallback) {
    ABI_PushCallerSaveRegistersAndAdjustStackExcept(code, HostLocRegIdx(result.getIdx()));
    // push/pop is a parallel move: every source is read before any parameter is written.
    for (const auto& source : sources) {
        code.push(source);
    }
    const std::array<Xbyak::Reg64, 3> params{code.ABI_PARAM2, code.ABI_PARAM3, code.ABI_PARAM4};
    for (size_t i = sources.size(); i > 0; --i) {
        code.pop(params[i - 1]);
    }
    code.mov(code.ABI_PARAM1, std::bit_cast<u64>(&conf));
    if (param4) {
        code.mov(code.ABI_PARAM4.cvt32(), *param4);
    }
    code.CallFunction(fallback);
    if (result.getIdx() != code.ABI_RETURN.getIdx()) {
        code.mov(result, code.ABI_RETURN);
    }
    ABI_PopCallerSaveRegistersAndAdjustStackExcept(code, HostLocRegIdx(result.getIdx()));
    code.ret();
}

template<std::size_t bitsize>
void A64EmitX64::EmitAtomicRmwMemory(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const u8 op = args[3].GetImmediateU8();
    const AtomicFallbackFn fallback = &AtomicRmwFallback<bitsize>;
    const auto fastmem_marker = ShouldFastmem(ctx, inst);

    if (!fastmem_marker) {
        ctx.reg_alloc.HostCall(code, inst, {}, args[1], args[2]);
        code.mov(code.ABI_PARAM1, std::bit_cast<u64>(&conf));
        code.mov(code.ABI_PARAM4.cvt32(), u32(op));
        code.CallFunction(fallback);
        code.ZeroExtendFrom(bitsize, code.ABI_RETURN);
        EmitCheckMemoryAbort(ctx, inst);
        return;
    }

    SharedLabel abort = ctx.GenSharedLabel(), end = ctx.GenSharedLabel();
    bool require_abort_handling = false;

    if (op == 0b0000 || op == 0b1000) {
        // LDADD: LOCK XADD. SWP: XCHG (always locked). Both leave the old value in `value`.
        const Xbyak::Reg64 vaddr = ctx.reg_alloc.UseGpr(code, args[1]);
        const Xbyak::Reg64 value = ctx.reg_alloc.UseScratchGpr(code, args[2]);
        const auto addr = EmitFastmemVAddr(code, ctx, *abort, vaddr, require_abort_handling);
        const auto location = code.getCurr();
        if (op == 0b0000) {
            code.lock();
            code.xadd(SizedPtr(addr, bitsize), SizedReg(value, bitsize));
        } else {
            code.xchg(SizedPtr(addr, bitsize), SizedReg(value, bitsize));
        }
        const auto resume = code.getCurr();
        if constexpr (bitsize < 32) {
            code.ZeroExtendFrom(bitsize, value);
        }
        code.L(*end);
        ColdEmits(code, ctx).emplace_back([=, this] {
            const auto stub = code.getCurr<void (*)()>();
            EmitAtomicFallbackStub(value, {vaddr, value}, u32(op), fallback);
            AddFastmemPatch(location, resume, stub, *fastmem_marker, conf.recompile_on_exclusive_fastmem_failure);
            if (require_abort_handling) {
                code.L(*abort);
                code.call(stub);
                code.jmp(*end, code.T_NEAR);
            }
        });
        ctx.reg_alloc.DefineValue(code, inst, value);
        EmitCheckMemoryAbort(ctx, inst);
        return;
    }

    // The others: load, compute, LOCK CMPXCHG, retry on a race. The old value ends up in rax.
    const Xbyak::Reg64 result = ctx.reg_alloc.ScratchGpr(code, HostLoc::RAX);
    const Xbyak::Reg64 vaddr = ctx.reg_alloc.UseGpr(code, args[1]);
    const Xbyak::Reg64 operand = ctx.reg_alloc.UseGpr(code, args[2]);
    const Xbyak::Reg64 desired = ctx.reg_alloc.ScratchGpr(code);
    const bool is_minmax = op >= 0b0100 && op <= 0b0111;
    const bool is_signed = op == 0b0100 || op == 0b0101;
    const Xbyak::Reg64 operand_ext = is_minmax && bitsize < 32 ? ctx.reg_alloc.ScratchGpr(code) : operand;
    const auto addr = EmitFastmemVAddr(code, ctx, *abort, vaddr, require_abort_handling);

    if (is_minmax && bitsize < 32) {
        if (is_signed) {
            code.movsx(operand_ext.cvt32(), SizedReg(operand, bitsize));
        } else {
            code.movzx(operand_ext.cvt32(), SizedReg(operand, bitsize));
        }
    }
    const Xbyak::Reg cmp_desired = bitsize == 64 ? Xbyak::Reg(desired) : Xbyak::Reg(desired.cvt32());
    const Xbyak::Reg cmp_operand = bitsize == 64 ? Xbyak::Reg(operand_ext) : Xbyak::Reg(operand_ext.cvt32());

    const auto load_location = code.getCurr();
    switch (bitsize) {
    case 8:
        code.movzx(eax, Xbyak::util::byte[addr]);
        break;
    case 16:
        code.movzx(eax, word[addr]);
        break;
    case 32:
        code.mov(eax, dword[addr]);
        break;
    default:
        code.mov(rax, qword[addr]);
        break;
    }
    Xbyak::Label loop;
    code.L(loop);
    switch (op) {
    case 0b0001:
        code.mov(desired, operand);
        code.not_(desired);
        code.and_(desired, rax);
        break;
    case 0b0010:
        code.mov(desired, rax);
        code.xor_(desired, operand);
        break;
    case 0b0011:
        code.mov(desired, rax);
        code.or_(desired, operand);
        break;
    default:
        if (bitsize == 8) {
            if (is_signed) {
                code.movsx(desired.cvt32(), al);
            } else {
                code.movzx(desired.cvt32(), al);
            }
        } else if (bitsize == 16) {
            if (is_signed) {
                code.movsx(desired.cvt32(), ax);
            } else {
                code.movzx(desired.cvt32(), ax);
            }
        } else if (bitsize == 32) {
            code.mov(desired.cvt32(), eax);
        } else {
            code.mov(desired, rax);
        }
        code.cmp(cmp_desired, cmp_operand);
        switch (op) {
        case 0b0100:
            code.cmovl(cmp_desired, cmp_operand);
            break;
        case 0b0101:
            code.cmovg(cmp_desired, cmp_operand);
            break;
        case 0b0110:
            code.cmovb(cmp_desired, cmp_operand);
            break;
        default:
            code.cmova(cmp_desired, cmp_operand);
            break;
        }
        break;
    }
    const auto cas_location = code.getCurr();
    code.lock();
    code.cmpxchg(SizedPtr(addr, bitsize), SizedReg(desired, bitsize));
    code.jnz(loop);
    code.L(*end);
    const auto resume = code.getCurr();
    ColdEmits(code, ctx).emplace_back([=, this] {
        const auto stub = code.getCurr<void (*)()>();
        EmitAtomicFallbackStub(result, {vaddr, operand}, u32(op), fallback);
        AddFastmemPatch(load_location, resume, stub, *fastmem_marker, conf.recompile_on_exclusive_fastmem_failure);
        AddFastmemPatch(cas_location, resume, stub, *fastmem_marker, conf.recompile_on_exclusive_fastmem_failure);
        if (require_abort_handling) {
            code.L(*abort);
            code.call(stub);
            code.jmp(*end, code.T_NEAR);
        }
    });
    ctx.reg_alloc.DefineValue(code, inst, result);
    EmitCheckMemoryAbort(ctx, inst);
}

template<std::size_t bitsize>
void A64EmitX64::EmitAtomicCasMemory(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const AtomicFallbackFn fallback = &AtomicCasFallback<bitsize>;
    const auto fastmem_marker = ShouldFastmem(ctx, inst);

    if (!fastmem_marker) {
        ctx.reg_alloc.HostCall(code, inst, {}, args[1], args[2], args[3]);
        code.mov(code.ABI_PARAM1, std::bit_cast<u64>(&conf));
        code.CallFunction(fallback);
        code.ZeroExtendFrom(bitsize, code.ABI_RETURN);
        EmitCheckMemoryAbort(ctx, inst);
        return;
    }

    const Xbyak::Reg64 result = ctx.reg_alloc.ScratchGpr(code, HostLoc::RAX);
    const Xbyak::Reg64 vaddr = ctx.reg_alloc.UseGpr(code, args[1]);
    const Xbyak::Reg64 expected = ctx.reg_alloc.UseGpr(code, args[2]);
    const Xbyak::Reg64 desired = ctx.reg_alloc.UseGpr(code, args[3]);

    SharedLabel abort = ctx.GenSharedLabel(), end = ctx.GenSharedLabel();
    bool require_abort_handling = false;
    const auto addr = EmitFastmemVAddr(code, ctx, *abort, vaddr, require_abort_handling);

    if constexpr (bitsize == 64) {
        code.mov(rax, expected);
    } else {
        code.mov(eax, expected.cvt32());
    }
    const auto location = code.getCurr();
    code.lock();
    code.cmpxchg(SizedPtr(addr, bitsize), SizedReg(desired, bitsize));
    const auto resume = code.getCurr();
    if constexpr (bitsize < 32) {
        code.ZeroExtendFrom(bitsize, rax);
    }
    code.L(*end);
    ColdEmits(code, ctx).emplace_back([=, this] {
        const auto stub = code.getCurr<void (*)()>();
        EmitAtomicFallbackStub(result, {vaddr, expected, desired}, std::nullopt, fallback);
        AddFastmemPatch(location, resume, stub, *fastmem_marker, conf.recompile_on_exclusive_fastmem_failure);
        if (require_abort_handling) {
            code.L(*abort);
            code.call(stub);
            code.jmp(*end, code.T_NEAR);
        }
    });
    ctx.reg_alloc.DefineValue(code, inst, result);
    EmitCheckMemoryAbort(ctx, inst);
}

void A64EmitX64::EmitA64AtomicRmwMemory8(A64EmitContext& ctx, IR::Inst* inst) {
    EmitAtomicRmwMemory<8>(ctx, inst);
}

void A64EmitX64::EmitA64AtomicRmwMemory16(A64EmitContext& ctx, IR::Inst* inst) {
    EmitAtomicRmwMemory<16>(ctx, inst);
}

void A64EmitX64::EmitA64AtomicRmwMemory32(A64EmitContext& ctx, IR::Inst* inst) {
    EmitAtomicRmwMemory<32>(ctx, inst);
}

void A64EmitX64::EmitA64AtomicRmwMemory64(A64EmitContext& ctx, IR::Inst* inst) {
    EmitAtomicRmwMemory<64>(ctx, inst);
}

void A64EmitX64::EmitA64AtomicCasMemory8(A64EmitContext& ctx, IR::Inst* inst) {
    EmitAtomicCasMemory<8>(ctx, inst);
}

void A64EmitX64::EmitA64AtomicCasMemory16(A64EmitContext& ctx, IR::Inst* inst) {
    EmitAtomicCasMemory<16>(ctx, inst);
}

void A64EmitX64::EmitA64AtomicCasMemory32(A64EmitContext& ctx, IR::Inst* inst) {
    EmitAtomicCasMemory<32>(ctx, inst);
}

void A64EmitX64::EmitA64AtomicCasMemory64(A64EmitContext& ctx, IR::Inst* inst) {
    EmitAtomicCasMemory<64>(ctx, inst);
}

void A64EmitX64::EmitCheckMemoryAbort(A64EmitContext&, IR::Inst* inst, Xbyak::Label* end) {
    if (!conf.check_halt_on_memory_access) {
        return;
    }

    Xbyak::Label skip;

    const A64::LocationDescriptor current_location{IR::LocationDescriptor{inst->GetArg(0).GetU64()}};

    code.test(dword[code.ABI_JIT_PTR + offsetof(A64JitState, halt_reason)], static_cast<u32>(HaltReason::MemoryAbort));
    if (end) {
        code.jz(*end, code.T_NEAR);
    } else {
        code.jz(skip, code.T_NEAR);
    }
    code.mov(rax, current_location.PC());
    code.mov(qword[code.ABI_JIT_PTR + offsetof(A64JitState, pc)], rax);
    code.ForceReturnFromRunCode();
    code.L(skip);
}

}  // namespace Dynarmic::Backend::X64
