// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/impl.h"

namespace {
using Shader::Ampere::Instruction;
using Shader::Ampere::LowerToMaxwell;
constexpr Instruction Exit{{0x0000794d, 0, 0x03800000, 0}};
constexpr Instruction UniformBank{{0x00007ab9, 0x00004000, 0x00000800, 0}};
constexpr Instruction LoggedMove{{0x00d87a02, 0x00006700, 0x00000f00, 0x000fc800}};
constexpr Instruction LoggedZero{{0xffea7224, 0x000000ff, 0x078e00ff, 0x000fe200}};
constexpr Instruction LoggedCompare{{0xf4007a0c, 0x00009e00, 0x03f06270, 0x000fce00}};
constexpr Instruction LoggedLoad{{0x00047ab9, 0x00006000, 0x00000000, 0x000fe200}};
constexpr Instruction LoggedMultiply{{0x0a0c02a4, 0x00000006, 0x0f8e0207, 0x000fe200}};
constexpr Instruction LoggedCopy{{0x100b7fae, 0x00000000, 0x09901d4a, 0x0003e200}};
constexpr Instruction CapturedMin{{0x79de7c17, 0x00000004, 0x0b800200, 0x040fe200}};
constexpr Instruction CapturedSpecialZero{{0x00da7805, 0x00000000, 0x0001ff00, 0x000fe200}};
constexpr Instruction CapturedMultiply{{0x03257a24, 0x00009e00, 0x078e02f4, 0x040fe200}};
constexpr Instruction CapturedSelect{{0xe4e47207, 0x000000ff, 0x01000000, 0x000fe400}};
constexpr Instruction CapturedPredicates{{0xd9007804, 0x0000000e, 0x00000000, 0x000fe200}};
constexpr Instruction CapturedHalfConversion{{0x00087305, 0x00000002, 0x00103100, 0x000e2600}};
constexpr Instruction CapturedCarry{{0x09097290, 0x00000006, 0x0ff5e03f, 0x000fe200}};
constexpr Instruction CapturedTwoCarries{{0x09097290, 0x00000004, 0x0f83e006, 0x000fe400}};
constexpr Instruction CapturedExtendedCarry{{0x08087290, 0x00000007, 0x097fe43f, 0x000fea00}};
constexpr Instruction CapturedExtendedTwoCarries{{0x08087290, 0x00000005, 0x08fe0407, 0x000fe200}};
constexpr Instruction CapturedImmediateSelect{{0xffea7807, 0x00000008, 0x04000000, 0x000fe400}};
constexpr Instruction CapturedRegisterCompare{{0xf200720c, 0x000000ff, 0x03f05270, 0x000fe200}};
constexpr Instruction CapturedHalfFloor{{0x00087305, 0x00000002, 0x00107100, 0x000e2400}};
constexpr Instruction CapturedIntegerHalf{{0x00247306, 0x0000000a, 0x00200c00, 0x000e2400}};
constexpr Instruction CapturedScopedLoad{{0x08087981, 0x0000000c, 0x0c1ebd00, 0x000ea200}};
constexpr Instruction CapturedHalfTrunc{{0x00297307, 0x00000013, 0x0010c800, 0x000fe200}};
constexpr Instruction CapturedHalfRound{{0x00317307, 0x00000030, 0x00100800, 0x000fe200}};
constexpr Instruction CapturedUniformPack{{0xff077c3e, 0x00000008, 0x080000ff, 0x002fe400}};
constexpr Instruction CapturedFloatUp{{0x000e7306, 0x00000006, 0x00209400, 0x000e3000}};
constexpr Instruction CapturedShuffleDown{{0x00077589, 0x08001f05, 0x000e0000, 0x000e2200}};
constexpr Instruction CapturedImmediateFloatUp{{0x00077906, 0x00000020, 0x00209000, 0x000e2200}};
constexpr Instruction CapturedImmediatePackedFloatTrunc{
    {0x00047845, 0x00000100, 0x0020d000, 0x000fe200}};

f64 HalfValue(u16 bits) {
    const u32 exponent = (bits >> 10) & 31, fraction = bits & 1023;
    const f64 magnitude = exponent == 31 ? (fraction ? std::numeric_limits<f64>::quiet_NaN()
                                                     : std::numeric_limits<f64>::infinity())
                                         : std::ldexp(f64(exponent ? fraction + 1024 : fraction),
                                                      exponent ? int(exponent) - 25 : -24);
    return bits & 0x8000 ? -magnitude : magnitude;
}

u32 RoundedHalfInteger(u16 bits) {
    const u32 exponent = (bits >> 10) & 31, fraction = bits & 1023;
    if (exponent == 31)
        return fraction ? 0U : bits & 0x8000 ? 0x80000000U : 0x7fffffffU;
    const u32 mantissa = exponent ? fraction + 1024 : fraction;
    u32 magnitude;
    if (exponent >= 25) {
        magnitude = mantissa << (exponent - 25);
    } else {
        const u32 shift = exponent ? 25 - exponent : 24;
        magnitude = mantissa >> shift;
        const u32 remainder = mantissa & ((1U << shift) - 1), midpoint = 1U << (shift - 1);
        magnitude += remainder > midpoint || (remainder == midpoint && (magnitude & 1));
    }
    return bits & 0x8000 ? 0U - magnitude : magnitude;
}

u16 IntegerHalf(s32 input) {
    const u32 sign = input < 0 ? 0x8000U : 0U;
    const u32 magnitude = input < 0 ? 0U - u32(input) : u32(input);
    if (magnitude == 0)
        return 0;
    if (magnitude >= 65520)
        return u16(sign | 0x7c00U);
    u32 exponent = 31U - std::countl_zero(magnitude);
    u32 mantissa = magnitude << (exponent < 10 ? 10 - exponent : 0);
    if (exponent > 10) {
        const u32 shift = exponent - 10, midpoint = 1U << (shift - 1);
        mantissa = magnitude >> shift;
        const u32 remainder = magnitude & ((1U << shift) - 1);
        mantissa += remainder > midpoint || (remainder == midpoint && (mantissa & 1));
        if (mantissa == 2048) {
            mantissa = 1024;
            ++exponent;
        }
    }
    return u16(sign | ((exponent + 15) << 10) | (mantissa & 1023));
}

u16 FloatHalf(f32 input) {
    static const auto values = [] {
        std::array<f64, 31744> result{};
        for (u32 bits = 0; bits < result.size(); ++bits)
            result[bits] = HalfValue(u16(bits));
        return result;
    }();
    if (std::isnan(input))
        return 0x7e00;
    const u32 sign = std::signbit(input) ? 0x8000U : 0U;
    const f64 magnitude = std::abs(f64(input));
    if (magnitude >= 65520.0)
        return u16(sign | 0x7c00U);
    if (magnitude > 65504.0)
        return u16(sign | 0x7bffU);
    const auto upper = std::lower_bound(values.begin(), values.end(), magnitude);
    u32 bits = u32(upper - values.begin());
    if (bits && *upper != magnitude) {
        const f64 low = magnitude - values[bits - 1], high = *upper - magnitude;
        bits -= low < high || (low == high && (bits & 1));
    }
    return u16(sign | bits);
}

u32 Address(u32 index) {
    return 0x50 + index * 8 + ((index + 1) / 3) * 8;
}

struct LoggedMachine {
    std::array<std::array<u32, 256>, 3> registers{};
    std::array<bool, 8> predicates{true, false, true, false, true, false, true, true};
    std::array<u8, 65536> constants{};
    std::unordered_map<u64, u32> global;
    std::unordered_map<u32, u32> shared;
    u32 reads{}, writes{};
    bool carry{};

    LoggedMachine() {
        for (u32 bank = 0; bank < registers.size(); ++bank)
            for (u32 reg = 0; reg < 255; ++reg)
                registers[bank][reg] = 0x9e370001U * (1 + reg + bank * 256);
        for (u32 i = 0; i < constants.size(); ++i)
            constants[i] = (i * 131 + 0x83) & 255;
    }

    u32 Read(u32 bank, u32 reg) const {
        return reg == 255 ? 0U : registers[bank][reg];
    }

    u32 Constant(u32 offset, u32 bytes) const {
        REQUIRE(offset + bytes <= constants.size());
        u32 result = 0;
        for (u32 i = 0; i < bytes; ++i)
            result |= u32{constants[offset + i]} << (i * 8);
        return result;
    }

    bool Predicate(u32 guard) const {
        return predicates[guard & 7] != bool(guard & 8);
    }

    void Run(std::span<const u64> code) {
        for (u32 pc = 0; pc < code.size(); ++pc) {
            const u64 word = code[pc];
            const auto bits = [&](u32 bit, u32 count) {
                return static_cast<u32>((word >> bit) & ((u64{1} << count) - 1));
            };
            if (!Predicate(bits(16, 4)))
                continue;
            const u32 dest = bits(0, 8), a = bits(8, 8), b = bits(20, 8);
            const auto write = [&](u32 bank, u32 value) {
                if (dest != 255)
                    registers[bank][dest] = value;
            };
            if ((word >> 48) == 0x5c98) {
                const bool banked = bits(44, 4) == 10;
                write(banked ? bits(8, 2) : 0, Read(banked ? bits(10, 2) : 0, b));
            } else if ((word >> 48) == 0x4c98) {
                REQUIRE(bits(34, 5) == 0);
                write(0, Constant(bits(20, 14) * 4, 4));
            } else if ((word & 0xfff0000000000000ULL) == 0xef90000000000000ULL) {
                REQUIRE(bits(36, 5) == 0);
                REQUIRE(bits(48, 3) == 0);
                const s32 offset = static_cast<s32>(bits(20, 16) << 16) >> 16;
                write(0, Constant(Read(0, a) + offset, 1));
            } else if ((word >> 56) == 1) {
                write(0, bits(20, 32));
            } else if ((word >> 48) == 0x38e8) {
                const u32 mask = bits(20, 7);
                u32 packed = 0;
                for (u32 i = 0; i < 7; ++i)
                    packed |= u32{predicates[i]} << i;
                write(0, (Read(0, a) & ~mask) | (packed & mask));
            } else if ((word >> 48) == 0x38f0) {
                const u32 value = Read(0, a), mask = bits(20, 7);
                for (u32 i = 0; i < 7; ++i)
                    if (mask & (1U << i))
                        predicates[i] = (value >> i) & 1;
            } else if ((word & 0xfff0000000000000ULL) == 0x5b60000000000000ULL) {
                REQUIRE((bits(49, 3) == 1 || bits(49, 3) == 5 || bits(49, 3) == 6));
                REQUIRE(bits(45, 2) == 0);
                REQUIRE(bits(0, 3) == 7);
                const bool result = bits(49, 3) == 1 ? Read(0, a) < Read(0, b)
                                    : bits(49, 3) == 5 ? Read(0, a) != Read(0, b)
                                    : bits(48, 1)    ? s32(Read(0, a)) >= s32(Read(0, b))
                                                     : Read(0, a) >= Read(0, b);
                const u32 output = bits(3, 3);
                if (output != 7)
                    predicates[output] = result && Predicate(bits(39, 4));
            } else if ((word & 0xff00000000000000ULL) == 0x5b00000000000000ULL) {
                REQUIRE(bits(50, 3) == 0);
                const u32 lhs = (Read(0, a) >> (bits(53, 1) ? 16 : 0)) & 0xffff;
                const u32 rhs = (Read(0, b) >> (bits(35, 1) ? 16 : 0)) & 0xffff;
                write(0, lhs * rhs + Read(0, bits(39, 8)));
            } else if (((word >> 48) & 0xfffe) == 0x5c20) {
                REQUIRE(bits(48, 1) == 1);
                const s32 lhs = Read(0, a), rhs = Read(0, b);
                write(0, Predicate(bits(39, 4)) ? (std::min)(lhs, rhs) : (std::max)(lhs, rhs));
            } else if ((word >> 48) == 0x5ca0) {
                write(0, Predicate(bits(39, 4)) ? Read(0, a) : Read(0, b));
            } else if ((word >> 48) == 0x5ca8) {
                REQUIRE(bits(8, 2) == 2);
                REQUIRE(bits(10, 2) == 1);
                const u16 half = Read(0, b) >> (bits(41, 1) ? 16 : 0);
                write(0, std::bit_cast<u32>(f32(HalfValue(half))));
            } else if ((word & 0xff80000000000000ULL) == 0x5800000000000000ULL) {
                REQUIRE(bits(39, 3) == 7);
                REQUIRE(bits(45, 3) == 0);
                f32 lhs = std::bit_cast<f32>(Read(0, a));
                const f32 rhs = std::bit_cast<f32>(Read(0, b));
                if (bits(54, 1))
                    lhs = std::abs(lhs);
                REQUIRE((bits(48, 4) == 1 || bits(48, 4) == 4));
                write(0, (bits(48, 4) == 1 ? lhs < rhs : lhs > rhs) ? 0xffffffffU : 0U);
            } else if (((word >> 48) & 0xfff8) == 0x5c40) {
                REQUIRE(bits(39, 2) == 0);
                REQUIRE(bits(43, 5) == 0);
                REQUIRE(bits(48, 3) == 7);
                REQUIRE(bits(41, 2) <= 1);
                write(0, bits(41, 2) ? Read(0, a) | Read(0, b) : Read(0, a) & Read(0, b));
            } else if ((word >> 48) == 0x5cb0) {
                REQUIRE(bits(8, 2) == 2);
                REQUIRE(bits(10, 2) == 2);
                REQUIRE(bits(12, 1) == 1);
                REQUIRE(bits(39, 2) <= 1);
                const f64 input = std::bit_cast<f32>(Read(0, b));
                REQUIRE(std::isfinite(input));
                REQUIRE((input >= -2147483648.0 && input < 2147483648.0));
                const f64 floor = std::floor(input), fraction = input - floor;
                const f64 rounded = fraction < 0.5                 ? floor
                                    : fraction > 0.5               ? floor + 1
                                    : std::fmod(floor, 2.0) == 0.0 ? floor
                                                                   : floor + 1;
                write(0, u32(s32(bits(39, 2) == 1 ? floor : rounded)));
            } else if (((word >> 48) & 0xfffc) == 0x5c10) {
                REQUIRE(bits(48, 2) == 0);
                const u64 sum = u64{Read(0, a)} + Read(0, b) + (bits(43, 1) && carry);
                write(0, static_cast<u32>(sum));
                if (bits(47, 1))
                    carry = sum >> 32;
            } else if ((word >> 56) == 0x1c) {
                write(0, Read(0, a) + static_cast<u32>(bits(20, 32)));
            } else if (((word >> 48) & 0xfffe) == 0x5c18) {
                write(0, (Read(0, a) << bits(39, 5)) + Read(0, b));
            } else if ((word & 0xfff0000000000000ULL) == 0xe240000000000000ULL) {
                const s32 offset = static_cast<s32>(bits(20, 24) << 8) >> 8;
                const u32 target = Address(pc) + 8 + offset;
                const u32 start = pc;
                while (pc < code.size() && Address(pc) != target)
                    ++pc;
                REQUIRE(pc > start);
                REQUIRE(pc < code.size());
                --pc;
            } else if ((word & 0xfff0000000000000ULL) == 0xeed0000000000000ULL) {
                REQUIRE(bits(45, 1) == 1);
                REQUIRE(bits(48, 3) == 6);
                const s32 offset = static_cast<s32>(bits(20, 24) << 8) >> 8;
                const u64 address = Read(0, a) | (u64{Read(0, a + 1)} << 32);
                for (u32 i = 0; i < 4; ++i) {
                    const u64 source = address + static_cast<u64>(offset) + i * 4;
                    REQUIRE(global.contains(source));
                    registers[0][dest + i] = global.at(source);
                    ++reads;
                }
            } else if ((word & 0xfff8000000000000ULL) == 0xef58000000000000ULL) {
                REQUIRE(bits(48, 3) == 6);
                const s32 offset = static_cast<s32>(bits(20, 24) << 8) >> 8;
                for (u32 i = 0; i < 4; ++i) {
                    shared[Read(0, a) + offset + i * 4] = Read(0, dest + i);
                    ++writes;
                }
            } else {
                REQUIRE(word == 0xe30000000007000fULL);
                return;
            }
        }
    }
};

class HalfEnvironment final : public Shader::Environment {
public:
    bool native = true;
    bool HasNativeInstructions() const override {
        return native;
    }
    u64 ReadInstruction(u32) override {
        throw std::logic_error("Unexpected instruction read");
    }
    u32 ReadCbufValue(u32, u32) override {
        throw std::logic_error("Unexpected constant read");
    }
    Shader::TextureType ReadTextureType(u32) override {
        return Shader::TextureType::Color2D;
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32) override {
        return false;
    }
    u32 ReadViewportTransformState() override {
        return 1;
    }
    u32 TextureBoundBuffer() const override {
        return 0;
    }
    u32 LocalMemorySize() const override {
        return 0;
    }
    u32 SharedMemorySize() const override {
        return 0;
    }
    std::array<u32, 3> WorkgroupSize() const override {
        return {1, 1, 1};
    }
    bool HasHLEMacroState() const override {
        return false;
    }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}
};

namespace IR = Shader::IR;

struct HalfIR {
    Shader::ObjectPool<IR::Inst> pool{256};
    IR::Block block{pool};
    HalfEnvironment env;
    std::array<u32, IR::NUM_REGS> registers{};
    std::unordered_map<const IR::Inst*, u64> values;

    explicit HalfIR(std::span<const u64> code) {
        Shader::Maxwell::TranslatorVisitor visitor{env, block};
        for (const u64 word : code) {
            if ((word >> 48) == 0x5ca8)
                visitor.F2F_reg(word);
            else if ((word >> 48) == 0x5cb0)
                visitor.F2I_reg(word);
            else if ((word >> 48) == 0x5cb8)
                visitor.I2F_reg(word);
            else if ((word >> 48) == 0x5c60)
                visitor.FMNMX_reg(word);
            else if ((word >> 48) == 0x5c98)
                visitor.MOV_reg(word);
            else if ((word >> 48) == 0x5c28)
                visitor.SHR_reg(word);
            else if ((word >> 48) == 0x38f8)
                visitor.SHF_r_imm(word);
            else if ((word >> 48) == 0x5c48)
                visitor.SHL_reg(word);
            else if ((word >> 48) == 0x5c18)
                visitor.ISCADD_reg(word);
            else if (((word >> 48) & 0xfffc) == 0x5c10)
                visitor.IADD_reg(word);
            else if ((word & 0xfc00000000000000ULL) == 0x1c00000000000000ULL)
                visitor.IADD32I(word);
            else if ((word >> 58) == 1)
                visitor.LOP32I(word);
            else if ((word >> 56) == 1)
                visitor.MOV32I(word);
            else if ((word & 0xff80000000000000ULL) == 0x5800000000000000ULL)
                visitor.FSET_reg(word);
            else if (((word >> 48) & 0xfff8) == 0x5c40)
                visitor.LOP_reg(word);
            else
                REQUIRE(word == 0xe30000000007000fULL);
        }
    }

    u64 Value(IR::Value value) const {
        if (!value.IsImmediate())
            return values.at(value.Inst());
        switch (value.Type()) {
        case IR::Type::U1:
            return value.U1();
        case IR::Type::U32:
            return value.U32();
        case IR::Type::F32:
            return std::bit_cast<u32>(value.F32());
        default:
            throw std::logic_error("Unexpected half IR immediate");
        }
    }

    f64 Float(IR::Value value) const {
        return value.Type() == IR::Type::F16 ? HalfValue(u16(Value(value)))
                                             : f64(std::bit_cast<f32>(u32(Value(value))));
    }

    u64 Evaluate(const IR::Inst& inst) const {
        const auto a = [&](size_t index) { return Value(inst.Arg(index)); };
        const auto f = [&](size_t index) { return Float(inst.Arg(index)); };
        const auto fp = [](f64 value) { return std::bit_cast<u32>(f32(value)); };
        const auto half_fp = [&](f64 value) -> u16 {
            if (std::isnan(value))
                return 0x7e00;
            if (std::isinf(value))
                return std::signbit(value) ? 0xfc00 : 0x7c00;
            if (value == 0)
                return std::signbit(f(0)) ? 0x8000 : 0;
            return IntegerHalf(s32(value));
        };
        switch (inst.GetOpcode()) {
        case IR::Opcode::Void:
            return 0;
        case IR::Opcode::Identity:
            return a(0);
        case IR::Opcode::GetRegister:
            return registers.at(static_cast<size_t>(inst.Arg(0).Reg()));
        case IR::Opcode::BitCastF32U32:
        case IR::Opcode::BitCastU32F32:
            return a(0);
        case IR::Opcode::UnpackFloat2x16:
            return (a(0) & 65535) | (a(0) >> 16) << 32;
        case IR::Opcode::CompositeExtractF16x2:
            return u32(a(0) >> (a(1) * 32));
        case IR::Opcode::ConvertF32F16:
            return fp(f(0));
        case IR::Opcode::ConvertF16F32:
            return std::isnan(f(0)) ? 0 : FloatHalf(f32(f(0)));
        case IR::Opcode::ConvertF32S32:
            return fp(f32(s32(a(0))));
        case IR::Opcode::ConvertU32F32:
            if (!std::isfinite(f(0)) || f(0) < 0 || f(0) >= 4294967296.0)
                throw std::domain_error("Out-of-range production IR float-to-unsigned conversion");
            return u32(f(0));
        case IR::Opcode::ConvertF16S32:
            return IntegerHalf(s32(a(0)));
        case IR::Opcode::CompositeConstructF16x2:
            return a(0) | (a(1) << 32);
        case IR::Opcode::PackFloat2x16:
            return (a(0) & 65535) | ((a(0) >> 32) << 16);
        case IR::Opcode::FPIsNan32:
            return std::isnan(f(0));
        case IR::Opcode::FPOrdGreaterThan32:
            return f(0) > f(1);
        case IR::Opcode::FPOrdLessThan32:
            return f(0) < f(1);
        case IR::Opcode::FPOrdGreaterThanEqual32:
            return f(0) >= f(1);
        case IR::Opcode::FPMin32:
            return fp(std::fmin(f(0), f(1)));
        case IR::Opcode::FPMax32:
            return fp(std::fmax(f(0), f(1)));
        case IR::Opcode::FPAbs32:
            return fp(std::abs(f(0)));
        case IR::Opcode::FPNeg32:
            return fp(-f(0));
        case IR::Opcode::FPFloor32:
            return fp(std::floor(f(0)));
        case IR::Opcode::FPCeil32:
            return fp(std::ceil(f(0)));
        case IR::Opcode::FPTrunc32:
            return fp(std::trunc(f(0)));
        case IR::Opcode::FPTrunc16:
            return half_fp(std::trunc(f(0)));
        case IR::Opcode::FPRoundEven16: {
            if (!std::isfinite(f(0)))
                return half_fp(f(0));
            const f64 low = std::floor(f(0)), fraction = f(0) - low;
            return half_fp(
                fraction < 0.5 || (fraction == 0.5 && std::fmod(low, 2.0) == 0) ? low : low + 1);
        }
        case IR::Opcode::FPClamp32:
            if (std::isnan(f(0)))
                throw std::domain_error("NaN reaches the production IR clamp");
            return fp(std::fmin(std::fmax(f(0), f(1)), f(2)));
        case IR::Opcode::FPRoundEven32: {
            const f64 low = std::floor(f(0)), fraction = f(0) - low;
            return fp(fraction < 0.5 || (fraction == 0.5 && std::fmod(low, 2.0) == 0) ? low
                                                                                      : low + 1);
        }
        case IR::Opcode::ConvertS32F32:
            if (!std::isfinite(f(0)) || f(0) < -2147483648.0 || f(0) >= 2147483648.0)
                throw std::domain_error("Out-of-range production IR float-to-integer conversion");
            return u32(s32(f(0)));
        case IR::Opcode::LogicalAnd:
            return a(0) && a(1);
        case IR::Opcode::LogicalOr:
            return a(0) || a(1);
        case IR::Opcode::LogicalNot:
            return !a(0);
        case IR::Opcode::SelectU1:
        case IR::Opcode::SelectU32:
        case IR::Opcode::SelectF32:
            return a(0) ? a(1) : a(2);
        case IR::Opcode::BitwiseAnd32:
            return a(0) & a(1);
        case IR::Opcode::BitwiseOr32:
            return a(0) | a(1);
        case IR::Opcode::BitwiseXor32:
            return a(0) ^ a(1);
        case IR::Opcode::BitwiseNot32:
            return ~u32(a(0));
        case IR::Opcode::ShiftRightLogical32:
            return u32(a(0)) >> (a(1) & 31);
        case IR::Opcode::ShiftLeftLogical32:
            return u32(a(0)) << (a(1) & 31);
        case IR::Opcode::CompositeConstructU32x2:
            return u32(a(0)) | (a(1) << 32);
        case IR::Opcode::CompositeExtractU32x2:
            return u32(a(0) >> (a(1) * 32));
        case IR::Opcode::PackUint2x32:
        case IR::Opcode::UnpackUint2x32:
            return a(0);
        case IR::Opcode::UMin32:
            return std::min(u32(a(0)), u32(a(1)));
        case IR::Opcode::ShiftRightLogical64:
            return a(0) >> (a(1) & 63);
        case IR::Opcode::IEqual:
            return a(0) == a(1);
        case IR::Opcode::INeg32:
            return 0U - u32(a(0));
        case IR::Opcode::IAdd32:
            return u32(a(0) + a(1));
        case IR::Opcode::ISub32:
            return u32(a(0) - a(1));
        case IR::Opcode::ULessThan:
            return u32(a(0)) < u32(a(1));
        case IR::Opcode::UGreaterThan:
            return u32(a(0)) > u32(a(1));
        case IR::Opcode::SLessThan:
            return s32(a(0)) < s32(a(1));
        default:
            throw std::logic_error(std::string{IR::NameOf(inst.GetOpcode())});
        }
    }

    u32 Run(u32 input, u32 source = 2, u32 destination = 8) {
        registers.fill(0xdeadbeef);
        registers[source] = input;
        values.clear();
        for (const auto& inst : block.Instructions()) {
            if (inst.GetOpcode() == IR::Opcode::SetRegister)
                registers.at(static_cast<size_t>(inst.Arg(0).Reg())) = u32(Value(inst.Arg(1)));
            else
                values.emplace(&inst, Evaluate(inst));
        }
        REQUIRE(registers[source] == input);
        return registers[destination];
    }
};
} // namespace

TEST_CASE("Ocarina packed float-to-byte conversion clamps before packing", "[video_core][ampere][oot]") {
    const Instruction convert{{0xff0a7243, 9, 0x0000c4ff, 0x000fe400}};
    const auto lowered = LowerToMaxwell(std::array{convert, Exit});
    INFO(lowered.error);
    REQUIRE(lowered);
    HalfIR translation{lowered.instructions};
    for (const f32 input : {-std::numeric_limits<f32>::infinity(), -255.0f, -0.5f, 0.0f, 0.5f,
                            1.0f, 127.75f, 255.0f, 255.75f, 256.0f,
                            std::numeric_limits<f32>::infinity(),
                            std::numeric_limits<f32>::quiet_NaN()}) {
        const u32 expected = std::isnan(input) ? 0U : u32(std::clamp(input, 0.0f, 255.0f));
        CHECK(translation.Run(std::bit_cast<u32>(input), 9, 10) == expected);
    }
}

TEST_CASE("Ocarina I2F from a uniform register selects any byte", "[video_core][ampere][oot]") {
    // I2F.F32.U8 R9, UR17.B3
    const Instruction convert{{0x00097d06, 0x30000011, 0x08001000, 0x000fe200}};
    const auto lowered = LowerToMaxwell(std::array{convert, Exit});
    INFO(lowered.error);
    REQUIRE(lowered);
    CHECK(lowered.instructions[lowered.source_instruction_offsets[0]] ==
          (0x5cb8000000070000ULL | 9 | (u64{2} << 8) | (u64{233} << 20) | (u64{3} << 41)));
}

TEST_CASE("Ocarina F2IP packs two rounded unorm bytes", "[video_core][ampere][oot]") {
    // F2IP.U8.F32 R41, R40, R39, RZ, from a compute program that scales both inputs by 255.
    const Instruction convert{{0x28297243, 0x27, 0x000004ff, 0x000fc400}};
    const auto lowered = LowerToMaxwell(std::array{convert, Exit});
    INFO(lowered.error);
    REQUIRE(lowered);
    HalfIR translation{lowered.instructions};
    const auto byte = [](f32 input) {
        return std::isnan(input) ? 0U : u32(std::nearbyintf(std::clamp(input, 0.0f, 255.0f)));
    };
    const std::array inputs{-std::numeric_limits<f32>::infinity(), -1.0f, 0.0f, 0.25f, 0.75f, 127.5f,
                            128.5f, 254.75f, 255.0f, 300.0f, std::numeric_limits<f32>::infinity(),
                            std::numeric_limits<f32>::quiet_NaN()};
    for (const f32 a : inputs) {
        for (const f32 b : inputs) {
            CAPTURE(a, b);
            translation.values.clear();
            translation.registers.fill(0xdeadbeef);
            translation.registers[40] = std::bit_cast<u32>(a);
            translation.registers[39] = std::bit_cast<u32>(b);
            for (const auto& inst : translation.block.Instructions()) {
                if (inst.GetOpcode() == IR::Opcode::SetRegister)
                    translation.registers.at(static_cast<size_t>(inst.Arg(0).Reg())) =
                        u32(translation.Value(inst.Arg(1)));
                else
                    translation.values.emplace(&inst, translation.Evaluate(inst));
            }
            CHECK(translation.registers[41] == (byte(b) | (byte(a) << 8)));
            CHECK(translation.registers[40] == std::bit_cast<u32>(a));
            CHECK(translation.registers[39] == std::bit_cast<u32>(b));
        }
    }
    // Other rounding modes and a merge register aren't known forms yet.
    auto modified = convert;
    modified.words[2] = 0x000044ff;
    CHECK_FALSE(LowerToMaxwell(std::array{modified, Exit}));
    modified = convert;
    modified.words[2] = 0x00000406;
    CHECK_FALSE(LowerToMaxwell(std::array{modified, Exit}));
}

TEST_CASE("Ocarina uniform multiply adds an immediate rather than reading a register",
          "[video_core][ampere][oot]") {
    const Instruction multiply{{0x040474a4, 0x00003c00, 0x0f8e0206, 0x000fc600}};
    const auto lowered = LowerToMaxwell(std::array{multiply, Exit});
    INFO(lowered.error);
    REQUIRE(lowered);
    for (const u32 a : {0U, 1U, 0x80000000U, 0xffffffffU}) {
        for (const u32 b : {0U, 3U, 0x40000001U, 0xffffffffU}) {
            LoggedMachine machine;
            machine.registers[0][220] = a;
            machine.registers[0][222] = b;
            const auto predicates = machine.predicates;
            machine.Run(lowered.instructions);
            CHECK(machine.registers[0][220] == a * b + 0x3c00U);
            CHECK(machine.registers[0][222] == b);
            CHECK(machine.predicates == predicates);
        }
    }
}

TEST_CASE("Ocarina 64-bit right shifts include both words and clamp at 64",
          "[video_core][ampere][oot]") {
    for (const bool uniform : {false, true}) {
        for (const u32 shift : {0U, 1U, 16U, 24U, 31U, 32U, 33U, 63U, 64U, 65U, 0xffffffffU}) {
            const Instruction instruction = uniform
                ? Instruction{{0x04077899, shift, 0x08001206, 0x002fc400}}
                : Instruction{{0x032b7219, 44, 0x0000122d, 0x000fe200}};
            const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
            INFO(lowered.error);
            REQUIRE(lowered);
            HalfIR translation{lowered.instructions};
            for (const u64 input : {0ULL, 0xffffffff00000000ULL, 0x8000000180000000ULL,
                                   0xfedcba9876543210ULL, 0xffffffffffffffffULL}) {
                CAPTURE(uniform, shift, input);
                translation.registers.fill(0xdeadbeef);
                translation.registers[uniform ? 220 : 3] = u32(input);
                translation.registers[uniform ? 222 : 45] = u32(input >> 32);
                translation.registers[44] = shift;
                translation.values.clear();
                for (const auto& inst : translation.block.Instructions()) {
                    if (inst.GetOpcode() == IR::Opcode::SetRegister)
                        translation.registers.at(static_cast<size_t>(inst.Arg(0).Reg())) =
                            u32(translation.Value(inst.Arg(1)));
                    else
                        translation.values.emplace(&inst, translation.Evaluate(inst));
                }
                CHECK(translation.registers[uniform ? 223 : 43] ==
                      (shift >= 64 ? 0U : u32(input >> shift)));
            }
        }
    }
}

TEST_CASE("Ocarina funnel shift unpacks 21-bit fields across both words",
          "[video_core][ampere][oot]") {
    // 1a55fa57a335b963 (vertex shader of the 128x128 environment capture and scene draws):
    // SHF.R.U64 R14, R14, 0x15, R17
    for (const u32 shift : {0U, 1U, 21U, 31U, 32U}) {
        const Instruction instruction{{0x0e0e7819, shift, 0x00001211, 0x020fe200}};
        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
        INFO(lowered.error);
        REQUIRE(lowered);
        HalfIR translation{lowered.instructions};
        for (const u64 input : {0ULL, 0x00000001ffe00000ULL, 0x8000000180000000ULL,
                               0xfedcba9876543210ULL, 0xffffffffffffffffULL}) {
            CAPTURE(shift, input);
            translation.registers.fill(0xdeadbeef);
            translation.registers[14] = u32(input);
            translation.registers[17] = u32(input >> 32);
            translation.values.clear();
            for (const auto& inst : translation.block.Instructions()) {
                if (inst.GetOpcode() == IR::Opcode::SetRegister)
                    translation.registers.at(static_cast<size_t>(inst.Arg(0).Reg())) =
                        u32(translation.Value(inst.Arg(1)));
                else
                    translation.values.emplace(&inst, translation.Evaluate(inst));
            }
            CHECK(translation.registers[14] == u32(input >> shift));
        }
    }
}

TEST_CASE("Ampere F2I signed byte clamps to the byte range", "[video_core][ampere]") {
    const Instruction convert{{0x00097305, 5, 0x0021e100, 0}};
    const auto lowered = LowerToMaxwell(std::array{convert, Exit});
    REQUIRE(lowered);
    HalfIR translation{lowered.instructions};
    for (const auto& [input, expected] : {
             std::pair{0.0f, 0U}, std::pair{-0.0f, 0U}, std::pair{1.9f, 1U},
             std::pair{-1.9f, 0xffffffffU}, std::pair{127.9f, 127U},
             std::pair{-127.9f, 0xffffff81U}, std::pair{128.0f, 127U},
             std::pair{-128.9f, 0xffffff80U}, std::pair{65536.0f, 127U},
             std::pair{-65536.0f, 0xffffff80U},
             std::pair{std::numeric_limits<f32>::infinity(), 127U},
             std::pair{-std::numeric_limits<f32>::infinity(), 0xffffff80U},
             std::pair{std::numeric_limits<f32>::quiet_NaN(), 0U}}) {
        CAPTURE(input);
        CHECK(translation.Run(std::bit_cast<u32>(input), 5, 9) == expected);
    }
}

TEST_CASE("Ampere signed sixteen-bit F2I truncates and saturates numerically",
          "[video_core][ampere]") {
    const Instruction convert{{0x00157305, 3, 0x0021e900, 0}};
    const auto lowered = LowerToMaxwell(std::array{convert, Exit});
    REQUIRE(lowered);
    HalfIR translation{lowered.instructions};
    for (const auto& [input, expected] : {
             std::pair{0.0f, 0U}, std::pair{1.9f, 1U}, std::pair{-1.9f, 0xffffffffU},
             std::pair{32766.9f, 32766U}, std::pair{32767.9f, 32767U},
             std::pair{32768.0f, 32767U}, std::pair{-32767.9f, 0xffff8001U},
             std::pair{-32768.9f, 0xffff8000U},
             std::pair{std::numeric_limits<f32>::infinity(), 32767U},
             std::pair{-std::numeric_limits<f32>::infinity(), 0xffff8000U}}) {
        CAPTURE(input);
        CHECK(translation.Run(std::bit_cast<u32>(input), 3, 21) == expected);
    }
}

TEST_CASE("Ampere register-absolute F2I applies absolute before rounding and saturation",
          "[video_core][ampere]") {
    const Instruction ceiling{{0x00097305, 0x40000005, 0x0020b000, 0}};
    const Instruction nearest_short{{0x00097305, 0x40000005, 0x00212800, 0}};
    const auto ceil_code = LowerToMaxwell(std::array{ceiling, Exit});
    const auto short_code = LowerToMaxwell(std::array{nearest_short, Exit});
    REQUIRE(ceil_code);
    REQUIRE(short_code);
    HalfIR ceil_translation{ceil_code.instructions};
    HalfIR short_translation{short_code.instructions};
    for (const auto& values : {
             std::array{0.0f, 0.0f, 0.0f}, std::array{-0.0f, 0.0f, 0.0f},
             std::array{-1.25f, 2.0f, 1.0f}, std::array{-1.5f, 2.0f, 2.0f},
             std::array{-2.5f, 3.0f, 2.0f}, std::array{2.5f, 3.0f, 2.0f},
             std::array{-65535.25f, 65536.0f, 65535.0f},
             std::array{-65536.0f, 65536.0f, 65535.0f}}) {
        CAPTURE(values[0]);
        const u32 input = std::bit_cast<u32>(values[0]);
        CHECK(ceil_translation.Run(input, 5, 9) == static_cast<u32>(values[1]));
        CHECK(short_translation.Run(input, 5, 9) == static_cast<u32>(values[2]));
    }
}

TEST_CASE("Fast Fusion logged high registers preserve uniform mirrors and borrowed quartets",
          "[video_core][ampere][fast_fusion_logged]") {
    for (const auto instruction : {LoggedMove, LoggedZero, LoggedCompare}) {
        DYNAMIC_SECTION("opcode " << instruction.Bits(0, 12)) {
            for (const u32 value : {0U, 0x7fffffffU, 0x80000000U, 0xffffffffU}) {
                const auto lowered = LowerToMaxwell(std::array{UniformBank, instruction, Exit});
                INFO(lowered.error);
                REQUIRE(lowered);
                LoggedMachine machine;
                machine.registers[1][244] = value;
                const auto before = machine.registers;
                const auto predicates = machine.predicates;
                machine.Run(lowered.instructions);
                for (u32 reg = 0; reg < 255; ++reg) {
                    if (reg < 246)
                        CHECK(machine.registers[0][reg] ==
                              (reg == 216 ? machine.Constant(0x100, 4) : before[0][reg]));
                    const u32 expected = instruction.words[0] == LoggedMove.words[0] && reg == 216
                                             ? machine.Constant(0x19c, 4)
                                         : instruction.words[0] == LoggedZero.words[0] && reg == 234
                                             ? 0U
                                             : before[1][reg];
                    CHECK(machine.registers[1][reg] == expected);
                }
                if (instruction.words[0] == LoggedCompare.words[0]) {
                    CHECK(machine.predicates[0] == (s32(value) >= s32(machine.Constant(0x278, 4))));
                    for (u32 p = 1; p < 8; ++p)
                        CHECK(machine.predicates[p] == predicates[p]);
                } else {
                    CHECK(machine.predicates == predicates);
                }
            }
        }
    }
}

TEST_CASE("Fast Fusion logged ULDC loads one unsigned byte without neighboring bits",
          "[video_core][ampere][fast_fusion_logged]") {
    for (u32 offset : {0x180U, 0x181U, 0x182U, 0x183U, 0x7fffU}) {
        auto load = LoggedLoad;
        load.words[1] = offset << 6;
        const auto lowered = LowerToMaxwell(std::array{load, Exit});
        INFO(lowered.error);
        REQUIRE(lowered);
        LoggedMachine machine;
        const auto before = machine.registers[0];
        machine.Run(lowered.instructions);
        for (u32 reg = 0; reg < 246; ++reg)
            CHECK(machine.registers[0][reg] ==
                  (reg == 220 ? machine.constants[offset] : before[reg]));
    }
    for (u32 offset : {0x8000U, 0xffffU}) {
        auto load = LoggedLoad;
        load.words[1] = offset << 6;
        CHECK_FALSE(LowerToMaxwell(std::array{load, Exit}));
    }
}

TEST_CASE("Fast Fusion captured register shadows distinguish pairs uniform inputs and constants",
          "[video_core][ampere][fast_fusion_logged]") {
    for (const auto instruction : {CapturedMin, CapturedSpecialZero, CapturedMultiply}) {
        DYNAMIC_SECTION("opcode " << instruction.Bits(0, 12)) {
            const auto lowered = LowerToMaxwell(std::array{UniformBank, instruction, Exit});
            INFO(lowered.error);
            REQUIRE(lowered);
            LoggedMachine machine;
            machine.registers[0][121] = 0x80000000;
            machine.registers[0][220] = 9;
            machine.registers[0][3] = 0xffff1234;
            machine.registers[1][244] = 0xfedcba98;
            const auto before = machine.registers;
            const auto predicates = machine.predicates;
            machine.Run(lowered.instructions);
            for (u32 reg = 0; reg < 255; ++reg) {
                if (reg < 246) {
                    const u32 expected =
                        reg == 216 ? machine.Constant(0x100, 4)
                        : instruction.words[0] == CapturedMultiply.words[0] && reg == 37
                            ? 0xffff1234U * machine.Constant(0x278, 4) + 0xfedcba98U
                            : before[0][reg];
                    CHECK(machine.registers[0][reg] == expected);
                }
                const u32 expected = instruction.words[0] == CapturedMin.words[0] && reg == 222
                                         ? 0x80000000U
                                     : instruction.words[0] == CapturedSpecialZero.words[0] &&
                                             (reg == 218 || reg == 219)
                                         ? 0U
                                         : before[1][reg];
                CHECK(machine.registers[1][reg] == expected);
            }
            CHECK(machine.predicates == predicates);
        }
    }
}

TEST_CASE("Fast Fusion logged uniform multiply uses UP0 and restores ordinary predicates",
          "[video_core][ampere][fast_fusion_logged]") {
    for (u32 guard : {0U, 8U}) {
        for (const bool enabled : {false, true}) {
            for (const auto values : {std::array{0U, 0U, 0xffffffffU},
                                      std::array{0x89abcdefU, 0xfedcba98U, 0x12345678U}}) {
                auto multiply = LoggedMultiply;
                multiply.words[0] |= guard << 12;
                const auto lowered = LowerToMaxwell(std::array{multiply, Exit});
                INFO(lowered.error);
                REQUIRE(lowered);
                LoggedMachine machine;
                machine.registers[0][226] = values[0];
                machine.registers[0][222] = values[1];
                machine.registers[0][223] = values[2];
                machine.registers[0][239] = enabled ? 0x55U : 0x15U;
                machine.predicates[6] = !enabled;
                const auto before = machine.registers[0];
                const auto predicates = machine.predicates;
                machine.Run(lowered.instructions);
                for (u32 reg = 0; reg < 246; ++reg)
                    CHECK(machine.registers[0][reg] == (reg == 228 && (enabled != bool(guard & 8))
                                                            ? values[0] * values[1] + values[2]
                                                            : before[reg]));
                CHECK(machine.predicates == predicates);
            }
        }
    }
}

TEST_CASE("Fast Fusion captured select and predicate writes read guest shadows not uniform mirrors",
          "[video_core][ampere][fast_fusion_logged]") {
    for (const auto instruction : {CapturedSelect, CapturedPredicates}) {
        DYNAMIC_SECTION("opcode " << instruction.Bits(0, 12)) {
            for (const bool selected : {false, true}) {
                const auto lowered = LowerToMaxwell(std::array{UniformBank, instruction, Exit});
                INFO(lowered.error);
                REQUIRE(lowered);
                LoggedMachine machine;
                machine.registers[1][217] = 0xa;
                machine.registers[1][228] = 0xdeadbeef;
                machine.predicates[2] = selected;
                const auto before = machine.registers;
                const auto predicates = machine.predicates;
                machine.Run(lowered.instructions);
                for (u32 reg = 0; reg < 255; ++reg) {
                    if (reg < 246)
                        CHECK(machine.registers[0][reg] ==
                              (reg == 216 ? machine.Constant(0x100, 4) : before[0][reg]));
                    const u32 expected =
                        instruction.words[0] == CapturedSelect.words[0] && reg == 228
                            ? (selected ? 0xdeadbeefU : 0U)
                            : before[1][reg];
                    CHECK(machine.registers[1][reg] == expected);
                }
                for (u32 p = 0; p < 8; ++p)
                    CHECK(machine.predicates[p] ==
                          (instruction.words[0] == CapturedPredicates.words[0] && p >= 1 && p <= 3
                               ? bool(0xaU & (1U << p))
                               : predicates[p]));
            }
        }
    }
}

TEST_CASE("Fast Fusion logged async copy uses its source predicate and full 64-bit address",
          "[video_core][ampere][fast_fusion_logged]") {
    for (u32 guard : {0U, 7U, 8U}) {
        for (u32 enable : {3U, 11U}) {
            for (const bool condition : {false, true}) {
                auto copy = LoggedCopy;
                copy.words[0] = (copy.words[0] & ~0xf000U) | (guard << 12);
                copy.words[2] = (copy.words[2] & ~0x07800000U) | (enable << 23);
                const auto lowered = LowerToMaxwell(std::array{copy, Exit});
                INFO(lowered.error);
                REQUIRE(lowered);
                LoggedMachine machine;
                machine.registers[0][11] = 128;
                machine.registers[0][16] = 0xfffffff0U;
                machine.registers[0][17] = 3;
                machine.predicates[0] = condition;
                machine.predicates[3] = condition;
                const auto before = machine.registers[0];
                const auto predicates = machine.predicates;
                const bool active = machine.Predicate(guard), source = machine.Predicate(enable);
                for (u32 i = 0; i < 4; ++i) {
                    if (active && source)
                        machine.global[0x3fffffff0ULL + i * 4] = 0xfeedfaceU + i;
                    machine.shared[128 + i * 4] = 0xdeadbeef;
                }
                machine.Run(lowered.instructions);
                CHECK(machine.reads == (active && source ? 4U : 0U));
                CHECK(machine.writes == (active ? 4U : 0U));
                for (u32 i = 0; i < 4; ++i)
                    CHECK(machine.shared.at(128 + i * 4) == (!active  ? 0xdeadbeefU
                                                             : source ? 0xfeedfaceU + i
                                                                      : 0U));
                for (u32 reg = 0; reg < 246; ++reg)
                    CHECK(machine.registers[0][reg] == before[reg]);
                CHECK(machine.predicates == predicates);
            }
        }
    }
}

TEST_CASE("Fast Fusion captured half conversion rounds every half value to a signed even integer",
          "[video_core][ampere][fast_fusion_logged]") {
    const auto lowered = LowerToMaxwell(std::array{CapturedHalfConversion, Exit});
    INFO(lowered.error);
    REQUIRE(lowered);
    LoggedMachine machine;
    const auto predicates = machine.predicates;
    for (u32 half = 0; half < 65536; ++half) {
        machine.registers[0][2] = 0xfc000000U | half;
        machine.Run(lowered.instructions);
        CAPTURE(half);
        CHECK(machine.registers[0][8] == RoundedHalfInteger(half));
        CHECK(machine.registers[0][2] == (0xfc000000U | half));
        CHECK(machine.predicates == predicates);
    }
}

TEST_CASE(
    "Fast Fusion captured uniform carry outputs preserve unrelated UP and ordinary predicates",
    "[video_core][ampere][fast_fusion_logged]") {
    constexpr std::array values{0U, 1U, 0x7fffffffU, 0x80000000U, 0xffffffffU};
    for (const auto instruction : {CapturedCarry, CapturedTwoCarries}) {
        DYNAMIC_SECTION("mode " << instruction.words[2]) {
            const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
            INFO(lowered.error);
            REQUIRE(lowered);
            LoggedMachine machine;
            for (u32 a : values) {
                for (u32 b : values) {
                    for (u32 c : values) {
                        for (u32 packed : {0x12340015U, 0x1234006aU}) {
                            machine.registers[0][225] = a;
                            machine.registers[0][instruction.words[1] == 6 ? 222 : 220] = b;
                            if (instruction.words[1] == 4)
                                machine.registers[0][222] = c;
                            machine.registers[0][239] = packed;
                            const auto before = machine.registers[0];
                            const auto predicates = machine.predicates;
                            const u64 sum = u64{a} + b + (instruction.words[1] == 4 ? c : 0);
                            u32 expected_predicates = packed;
                            for (u32 i = 0; i < 2; ++i) {
                                const u32 output = instruction.Bits(81 + i * 3, 3);
                                if (output == 7)
                                    continue;
                                const u32 mask = 1U << (6 - output);
                                expected_predicates = (expected_predicates & ~mask) |
                                                      (sum >= (u64{i + 1} << 32) ? mask : 0);
                            }
                            machine.Run(lowered.instructions);
                            CAPTURE(a, b, c, packed);
                            for (u32 reg = 0; reg < 246; ++reg)
                                CHECK(machine.registers[0][reg] == (reg == 225 ? u32(sum)
                                                                    : reg == 239
                                                                        ? expected_predicates
                                                                        : before[reg]));
                            CHECK(machine.predicates == predicates);
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("Ocarina call links stay proven across a predicate-to-register write",
          "[video_core][ampere][fast_fusion_logged]") {
    std::array<Instruction, 7> code{{
        {{0x00097802, 0x30, 0xf00, 0}},
        {{0xff107803, 0x40, 0, 0}}, // P2R R16, RZ, 0x40
        {{0x00007944, 0x10, 0x03c00000, 0}},
        Exit,
        CapturedImmediatePackedFloatTrunc,
        {{0x00047202, 9, 0xf00, 0}},
        {{0x04007950, 0xffffff90, 0x03c3ffff, 0}},
    }};
    const auto lowered = LowerToMaxwell(code);
    INFO(lowered.error);
    REQUIRE(lowered);
    code[1] = {{0xff097803, 0x40, 0, 0}}; // Writes the link register itself.
    const auto clobbered = LowerToMaxwell(code);
    CHECK_FALSE(clobbered);
    CHECK(clobbered.error == "native call linkage mode");
}

TEST_CASE("Ocarina uniform LEA carries through any UP and preserves the rest",
          "[video_core][ampere][fast_fusion_logged]") {
    // ULEA UR10, UR6, UR10, 2 with the carry out in UP2, from Ocarina of Time.
    constexpr Instruction captured{{0x060a7291, 0x0000000a, 0x0f84103f, 0x000fe200}};
    constexpr u32 ur6 = 222, ur10 = 226;
    constexpr std::array values{0U, 1U, 0x3fffffffU, 0x40000000U, 0xffffffffU};
    for (u32 up = 0; up < 7; ++up) {
        const u32 mask = 1U << (6 - up);
        auto instruction = captured;
        instruction.words[2] = (instruction.words[2] & ~(7U << 17)) | (up << 17);
        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
        CAPTURE(up);
        INFO(lowered.error);
        REQUIRE(lowered);
        LoggedMachine machine;
        for (u32 a : values) {
            for (u32 b : values) {
                for (u32 packed : {0x12340015U, 0x1234006aU}) {
                    machine.registers[0][ur6] = a;
                    machine.registers[0][ur10] = b;
                    machine.registers[0][239] = packed;
                    const auto before = machine.registers[0];
                    const auto predicates = machine.predicates;
                    machine.Run(lowered.instructions);
                    CAPTURE(a, b, packed);
                    const u32 sum = (a << 2) + b;
                    for (u32 reg = 0; reg < 246; ++reg)
                        CHECK(machine.registers[0][reg] ==
                              (reg == ur10  ? sum
                               : reg == 239 ? (packed & ~mask) | (sum < b ? mask : 0)
                                            : before[reg]));
                    CHECK(machine.predicates == predicates);
                }
            }
        }
    }
}

TEST_CASE("Ocarina indexed uniform constant loads keep their U8 and U16 widths",
          "[video_core][ampere][fast_fusion_logged]") {
    struct Case {
        Instruction instruction;
        u64 target, index, offset, bank, width;
    };
    // ULDC UR, c[bank][UR + offset] from Ocarina of Time's compute programs.
    const std::array cases{
        Case{{{0x11047abb, 0x01400000, 0x08000400, 0x000fe400}}, 220, 233, 0x00, 5, 2},
        Case{{{0x06077abb, 0x01002080, 0x08000000, 0x000fe400}}, 223, 222, 0x82, 4, 0},
        Case{{{0x0d047abb, 0x014000c0, 0x08000000, 0x000fe400}}, 220, 229, 0x03, 5, 0},
        Case{{{0x06087abb, 0x01002080, 0x08000000, 0x000fe400}}, 224, 222, 0x82, 4, 0},
    };
    for (const Case& test : cases) {
        CAPTURE(test.instruction.words[0]);
        const auto lowered = LowerToMaxwell(std::array{test.instruction, Exit});
        INFO(lowered.error);
        REQUIRE(lowered);
        CHECK(lowered.instructions[lowered.source_instruction_offsets[0]] ==
              (0xef90000000070000ULL | test.target | (test.index << 8) | (test.offset << 20) |
               (test.bank << 36) | (test.width << 48)));
    }
    // @UP0 with an index register and offset -1: the guard moves, the signed offset stays.
    const Instruction negative{{0x0f070abb, 0x013fffc0, 0x08000000, 0x000fe400}};
    const auto guarded = LowerToMaxwell(std::array{negative, Exit});
    INFO(guarded.error);
    REQUIRE(guarded);
    const u64 load = 0xef90000000060000ULL | 223 | (u64{231} << 8) | (u64{0xffff} << 20) | (u64{4} << 36);
    CHECK(std::ranges::find(guarded.instructions, load) != guarded.instructions.end());
    const u64 transfer = 0x38f0000000070000ULL | (u64{239} << 8) | (u64{1U << 6} << 20);
    CHECK(std::ranges::find(guarded.instructions, transfer) != guarded.instructions.end());
    // An indexed load can have any immediate: the register can align it (ULDC.32 at UR15 + 0x29).
    const Instruction indexed_word{{0x0f077abb, 0x01000a40, 0x08000800, 0x000fc600}};
    const auto word = LowerToMaxwell(std::array{indexed_word, Exit});
    INFO(word.error);
    REQUIRE(word);
    CHECK(word.instructions[word.source_instruction_offsets[0]] ==
          (0xef90000000070000ULL | 223 | (u64{231} << 8) | (u64{0x29} << 20) | (u64{4} << 36) |
           (u64{4} << 48)));
    // A direct U16 load needs an even offset, and a direct subword load the short offset range.
    Instruction misaligned{{0x00077ab9, 0x01000040, 0x00000400, 0x000fe400}};
    CHECK_FALSE(LowerToMaxwell(std::array{misaligned, Exit}));
    Instruction far{{0x00077ab9, 0x01200000, 0x00000000, 0x000fe400}};
    CHECK_FALSE(LowerToMaxwell(std::array{far, Exit}));
}

TEST_CASE("Ocarina uniform immediate add carries through any UP and preserves the rest",
          "[video_core][ampere][fast_fusion_logged]") {
    // UIADD3 UR12, UR12, 0x10 with the carry out in UP4, from Ocarina of Time.
    constexpr Instruction captured{{0x0c0c7890, 0x00000010, 0x0ff9e03f, 0x000fc400}};
    constexpr u32 ur12 = 228;
    constexpr std::array values{0U, 1U, 0xffffffefU, 0xfffffff0U, 0xffffffffU};
    for (u32 up = 0; up < 7; ++up) {
        const u32 mask = 1U << (6 - up);
        auto carry_out = captured;
        carry_out.words[2] = (carry_out.words[2] & ~(7U << 17)) | (up << 17);
        auto carry_in = captured;
        carry_in.words[2] = (carry_in.words[2] & ~(0xfU << 23)) | (7U << 17) | (1U << 10) | (up << 23);
        const auto lowered_out = LowerToMaxwell(std::array{carry_out, Exit});
        const auto lowered_in = LowerToMaxwell(std::array{carry_in, Exit});
        CAPTURE(up);
        INFO(lowered_out.error);
        INFO(lowered_in.error);
        REQUIRE(lowered_out);
        REQUIRE(lowered_in);
        LoggedMachine machine;
        for (u32 a : values) {
            for (u32 packed : {0x12340015U, 0x1234006aU}) {
                machine.registers[0][ur12] = a;
                machine.registers[0][239] = packed;
                const auto before = machine.registers[0];
                const auto predicates = machine.predicates;
                machine.Run(lowered_out.instructions);
                CAPTURE(a, packed);
                const bool carried = u64{a} + 0x10 > 0xffffffffULL;
                for (u32 reg = 0; reg < 246; ++reg)
                    CHECK(machine.registers[0][reg] ==
                          (reg == ur12  ? a + 0x10U
                           : reg == 239 ? (packed & ~mask) | (carried ? mask : 0)
                                        : before[reg]));
                CHECK(machine.predicates == predicates);

                machine.registers[0][ur12] = a;
                machine.registers[0][239] = packed;
                machine.Run(lowered_in.instructions);
                CHECK(machine.registers[0][ur12] == a + 0x10U + ((packed & mask) != 0 ? 1U : 0U));
                CHECK(machine.registers[0][239] == packed);
                CHECK(machine.predicates == predicates);
            }
        }
    }
}

TEST_CASE(
    "Fast Fusion captured half conversion keeps production IR conversions finite and in range",
    "[video_core][ampere][fast_fusion_logged]") {
    const auto lowered = LowerToMaxwell(std::array{CapturedHalfConversion, Exit});
    INFO(lowered.error);
    REQUIRE(lowered);
    HalfIR translation{lowered.instructions};
    for (u32 half = 0; half < 65536; ++half) {
        CAPTURE(half);
        CHECK(translation.Run(0xfc000000U | half) == RoundedHalfInteger(u16(half)));
    }
}

TEST_CASE(
    "Fast Fusion captured half conversion preserves its execution predicate and other registers",
    "[video_core][ampere][fast_fusion_logged]") {
    for (u32 guard : {0U, 7U, 8U}) {
        auto instruction = CapturedHalfConversion;
        instruction.words[0] = (instruction.words[0] & ~0xf000U) | (guard << 12);
        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
        INFO(lowered.error);
        REQUIRE(lowered);
        for (const bool condition : {false, true}) {
            for (u32 half : {0x0001U, 0x3e00U, 0xbe00U, 0x7bffU, 0x7c00U, 0xfc00U, 0x7e00U}) {
                LoggedMachine machine;
                machine.registers[0][2] = 0xfc000000U | half;
                machine.predicates[0] = condition;
                const auto before = machine.registers[0];
                const auto predicates = machine.predicates;
                const bool active = machine.Predicate(guard);
                machine.Run(lowered.instructions);
                for (u32 reg = 0; reg < 246; ++reg)
                    CHECK(machine.registers[0][reg] ==
                          (reg == 8 && active ? RoundedHalfInteger(u16(half)) : before[reg]));
                CHECK(machine.predicates == predicates);
            }
        }
    }
}

TEST_CASE("Fast Fusion captured upper half conversion ignores the poisoned lower half",
          "[video_core][ampere][fast_fusion_logged]") {
    auto instruction = CapturedHalfConversion;
    instruction.words[1] |= 1U << 28;
    const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
    INFO(lowered.error);
    REQUIRE(lowered);
    HalfIR translation{lowered.instructions};
    for (u32 half = 0; half < 65536; ++half) {
        CAPTURE(half);
        CHECK(translation.Run((half << 16) | 0xfc00U) == RoundedHalfInteger(u16(half)));
    }
}

TEST_CASE("Fast Fusion captured half floor converts every half pattern safely",
          "[video_core][ampere][fast_fusion_logged]") {
    for (const bool high : {false, true}) {
        auto instruction = CapturedHalfFloor;
        instruction.words[1] |= u32{high} << 28;
        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
        INFO(lowered.error);
        REQUIRE(lowered);
        HalfIR translation{lowered.instructions};
        for (u32 half = 0; half < 65536; ++half) {
            CAPTURE(high, half);
            const u32 expected = (half & 0x7c00) == 0x7c00
                                     ? RoundedHalfInteger(u16(half))
                                     : u32(s32(std::floor(HalfValue(u16(half)))));
            CHECK(translation.Run(high ? (half << 16) | 0xfc00U : 0xfc000000U | half) == expected);
        }
    }
}

TEST_CASE("Fast Fusion captured register comparison reads the high guest shadow",
          "[video_core][ampere][fast_fusion_logged]") {
    const auto lowered = LowerToMaxwell(std::array{UniformBank, CapturedRegisterCompare, Exit});
    INFO(lowered.error);
    REQUIRE(lowered);
    for (u32 input : {0U, 1U, 0x80000000U, 0xffffffffU}) {
        LoggedMachine machine;
        machine.registers[1][242] = input;
        const auto before = machine.registers;
        const auto predicates = machine.predicates;
        machine.Run(lowered.instructions);
        for (u32 reg = 0; reg < 255; ++reg) {
            if (reg < 246)
                CHECK(machine.registers[0][reg] ==
                      (reg == 216 ? machine.Constant(0x100, 4) : before[0][reg]));
            CHECK(machine.registers[1][reg] == before[1][reg]);
        }
        for (u32 p = 0; p < 8; ++p)
            CHECK(machine.predicates[p] == (p == 0 ? input != 0 : predicates[p]));
    }
}

TEST_CASE("Fast Fusion captured integer to half conversion rounds ties and clears the upper half",
          "[video_core][ampere][fast_fusion_logged]") {
    const auto lowered = LowerToMaxwell(std::array{CapturedIntegerHalf, Exit});
    INFO(lowered.error);
    REQUIRE(lowered);
    HalfIR translation{lowered.instructions};
    std::array<f64, 31744> positive;
    for (u32 bits = 0; bits < positive.size(); ++bits)
        positive[bits] = HalfValue(u16(bits));
    const auto check = [&](s32 input) {
        const f64 magnitude = std::abs(f64(input));
        u32 expected = 0;
        if (magnitude >= 65520.0) {
            expected = 0x7c00;
        } else if (magnitude > 65504.0) {
            expected = 0x7bff;
        } else {
            const auto upper = std::lower_bound(positive.begin(), positive.end(), magnitude);
            expected = u32(upper - positive.begin());
            if (expected && *upper != magnitude) {
                const f64 low_distance = magnitude - positive[expected - 1];
                const f64 high_distance = *upper - magnitude;
                expected -= low_distance < high_distance ||
                            (low_distance == high_distance && (expected & 1));
            }
        }
        expected |= input < 0 ? 0x8000U : 0U;
        CAPTURE(input);
        CHECK(translation.Run(u32(input), 10, 36) == expected);
    };
    for (s32 input = -65536; input <= 65536; ++input)
        check(input);
    for (s32 input : {(std::numeric_limits<s32>::min)(), -16777217, 16777217,
                      (std::numeric_limits<s32>::max)()})
        check(input);
}

TEST_CASE(
    "Fast Fusion captured scoped load uses the complete register pointer not its policy descriptor",
    "[video_core][ampere][fast_fusion_logged]") {
    for (s32 offset : {-32, 0, 32}) {
        auto instruction = CapturedScopedLoad;
        instruction.words[1] |= u32(offset) << 8;
        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
        INFO(lowered.error);
        REQUIRE(lowered);
        LoggedMachine machine;
        machine.registers[0][8] = 0x10;
        machine.registers[0][9] = 4;
        machine.registers[0][228] = 0xfedcba98;
        machine.registers[0][229] = 0x76543210;
        const auto before = machine.registers[0];
        const auto predicates = machine.predicates;
        for (u32 i = 0; i < 4; ++i)
            machine.global[0x400000010ULL + offset + i * 4] = 0xfeed0000U + i;
        machine.Run(lowered.instructions);
        CHECK(machine.reads == 4);
        for (u32 reg = 0; reg < 246; ++reg)
            CHECK(machine.registers[0][reg] ==
                  (reg >= 8 && reg < 12 ? 0xfeed0000U + reg - 8 : before[reg]));
        CHECK(machine.predicates == predicates);
    }
}

TEST_CASE("Fast Fusion scoped load marks production IR coherent without changing weak descriptors",
          "[video_core][ampere][fast_fusion_logged]") {
    for (const bool scoped : {false, true}) {
        auto instruction = CapturedScopedLoad;
        if (!scoped)
            instruction.words[2] = 0x0c1e1d00;
        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
        INFO(lowered.error);
        REQUIRE(lowered);
        const auto word =
            std::find_if(lowered.instructions.begin(), lowered.instructions.end(), [](u64 value) {
                return (value & 0xfff0000000000000ULL) == 0xeed0000000000000ULL;
            });
        REQUIRE(word != lowered.instructions.end());
        CHECK(((*word >> 46) & 3) == u64{scoped});
        for (const bool native : {false, true}) {
            Shader::ObjectPool<IR::Inst> pool{64};
            IR::Block block{pool};
            HalfEnvironment env;
            env.native = native;
            Shader::Maxwell::TranslatorVisitor visitor{env, block};
            visitor.LDG(*word);
            u32 loads = 0;
            for (const auto& inst : block.Instructions()) {
                if (inst.GetOpcode() == IR::Opcode::LoadGlobal128) {
                    ++loads;
                    CHECK(inst.Flags<bool>() == (scoped && native));
                }
                CHECK(inst.GetOpcode() != IR::Opcode::DeviceMemoryBarrier);
                CHECK(inst.GetOpcode() != IR::Opcode::WorkgroupMemoryBarrier);
            }
            CHECK(loads == 1);
        }
    }
}

TEST_CASE("Fast Fusion captured half rounds preserve special values and clear the upper half",
          "[video_core][ampere][fast_fusion_logged]") {
    for (const auto instruction : {CapturedHalfTrunc, CapturedHalfRound}) {
        DYNAMIC_SECTION("mode " << instruction.words[2]) {
            for (const bool high : {false, true}) {
                auto selected = instruction;
                selected.words[1] |= u32{high} << 28;
                const auto lowered = LowerToMaxwell(std::array{selected, Exit});
                INFO(lowered.error);
                REQUIRE(lowered);
                HalfIR translation{lowered.instructions};
                for (u32 half = 0; half < 65536; ++half) {
                    const u32 output =
                        translation.Run(high ? (half << 16) | 0xfc00U : 0xfc000000U | half,
                                        instruction.Bits(32, 8), instruction.Bits(16, 8));
                    CAPTURE(high, half);
                    CHECK((output >> 16) == 0);
                    if ((half & 0x7c00) == 0x7c00) {
                        if (half & 1023)
                            CHECK(std::isnan(HalfValue(u16(output))));
                        else
                            CHECK(output == half);
                    } else {
                        const f64 expected = instruction.words[2] == 0x0010c800
                                                 ? std::trunc(HalfValue(u16(half)))
                                                 : f64(s32(RoundedHalfInteger(u16(half))));
                        CHECK(HalfValue(u16(output)) == expected);
                        CHECK((output & 0x8000) == (half & 0x8000));
                    }
                }
            }
        }
    }
}

TEST_CASE(
    "Fast Fusion captured zero-high uniform pack preserves half rounding and exceptional values",
    "[video_core][ampere][fast_fusion_logged]") {
    const auto lowered = LowerToMaxwell(std::array{CapturedUniformPack, Exit});
    INFO(lowered.error);
    REQUIRE(lowered);
    HalfIR translation{lowered.instructions};
    u32 nan_checks = 0;
    bool repaired_output = false;
    for (const auto& inst : translation.block.Instructions()) {
        nan_checks += inst.GetOpcode() == IR::Opcode::FPIsNan32;
        if (inst.GetOpcode() == IR::Opcode::SetRegister && static_cast<u32>(inst.Arg(0).Reg()) == 7)
            repaired_output = inst.Arg(1).Inst()->GetOpcode() == IR::Opcode::BitwiseOr32;
    }
    CHECK(nan_checks > 0);
    CHECK(repaired_output);
    for (const auto& [input, expected] :
         {std::pair{0x00000000U, 0x0000U}, std::pair{0x80000000U, 0x8000U},
          std::pair{0x3f800000U, 0x3c00U}, std::pair{0x3fc00000U, 0x3e00U},
          std::pair{0x3f801000U, 0x3c00U}, std::pair{0x3f803000U, 0x3c02U},
          std::pair{0x477fe000U, 0x7bffU}, std::pair{0x477ff000U, 0x7c00U},
          std::pair{0x33800000U, 0x0001U}, std::pair{0x33000000U, 0x0000U},
          std::pair{0xb3800000U, 0x8001U}, std::pair{0xb3000000U, 0x8000U},
          std::pair{0x7f800000U, 0x7c00U}, std::pair{0xff800000U, 0xfc00U}}) {
        CAPTURE(input);
        CHECK(translation.Run(input, 224, 7) == expected);
    }
    for (u32 input : {0x7fc00000U, 0xffc12345U, 0x7f800001U}) {
        const u32 output = translation.Run(input, 224, 7);
        CHECK((output >> 16) == 0);
        CHECK(std::isnan(HalfValue(u16(output))));
    }
}

TEST_CASE("Fast Fusion captured signed integer to float rounds toward positive infinity",
          "[video_core][ampere][fast_fusion_logged]") {
    const auto lowered = LowerToMaxwell(std::array{CapturedFloatUp, Exit});
    INFO(lowered.error);
    REQUIRE(lowered);
    HalfIR translation{lowered.instructions};
    u32 round_trips = 0, ordinary_conversions = 0;
    for (const auto& inst : translation.block.Instructions()) {
        if (inst.GetOpcode() == IR::Opcode::ConvertU32F32) {
            ++round_trips;
            CHECK(inst.Arg(0).Inst()->GetOpcode() == IR::Opcode::FPAbs32);
        }
        if (inst.GetOpcode() == IR::Opcode::ConvertF32S32) {
            ++ordinary_conversions;
            CHECK(inst.Flags<IR::FpControl>().rounding == IR::FpRounding::DontCare);
        }
        CHECK(inst.GetOpcode() != IR::Opcode::ConvertS32F32);
    }
    CHECK(round_trips == 1);
    CHECK(ordinary_conversions == 1);
    for (const auto& [input, expected] :
         {std::pair{0, 0x00000000U}, std::pair{1, 0x3f800000U}, std::pair{-1, 0xbf800000U},
          std::pair{16777215, 0x4b7fffffU}, std::pair{16777217, 0x4b800001U},
          std::pair{16777219, 0x4b800002U}, std::pair{-16777217, 0xcb800000U},
          std::pair{-16777219, 0xcb800001U}, std::pair{67108865, 0x4c800001U},
          std::pair{-67108865, 0xcc800000U},
          std::pair{(std::numeric_limits<s32>::min)(), 0xcf000000U},
          std::pair{(std::numeric_limits<s32>::max)(), 0x4f000000U}}) {
        CAPTURE(input);
        CHECK(translation.Run(u32(input), 6, 14) == expected);
    }
}

TEST_CASE("Fast Fusion captured immediate unsigned conversions preserve rounding and predicates",
          "[video_core][ampere][fast_fusion_logged]") {
    for (const auto original : {CapturedImmediateFloatUp, CapturedImmediatePackedFloatTrunc}) {
        const bool round_up = original.Bits(0, 12) == 0x906;
        const auto captured = LowerToMaxwell(std::array{original, Exit});
        INFO(captured.error);
        REQUIRE(captured);
        REQUIRE(captured.instructions.size() == 2);
        CHECK(captured.instructions[0] ==
              (round_up ? 0x010420000007f007ULL : 0x010438000007f004ULL));
        for (const auto values :
             {std::array{0U, 0x00000000U, 0x00000000U}, std::array{1U, 0x3f800000U, 0x3f800000U},
              std::array{32U, 0x42000000U, 0x42000000U},
              std::array{16777215U, 0x4b7fffffU, 0x4b7fffffU},
              std::array{16777216U, 0x4b800000U, 0x4b800000U},
              std::array{16777217U, 0x4b800001U, 0x4b800000U},
              std::array{16777219U, 0x4b800002U, 0x4b800001U},
              std::array{33554431U, 0x4c000000U, 0x4bffffffU},
              std::array{33554433U, 0x4c000001U, 0x4c000000U},
              std::array{0x7fffffffU, 0x4f000000U, 0x4effffffU},
              std::array{0x80000000U, 0x4f000000U, 0x4f000000U},
              std::array{0x80000001U, 0x4f000001U, 0x4f000000U},
              std::array{0xffffff00U, 0x4f7fffffU, 0x4f7fffffU},
              std::array{0xffffffffU, 0x4f800000U, 0x4f7fffffU}}) {
            const u32 input = values[0], expected = values[round_up ? 1 : 2];
            for (u32 guard : {3U, 7U, 11U}) {
                auto instruction = original;
                instruction.words[0] = (instruction.words[0] & ~0xf000U) | (guard << 12);
                instruction.words[1] = input;
                const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
                INFO(lowered.error);
                REQUIRE(lowered);
                REQUIRE(lowered.instructions.size() == 2);
                CHECK((lowered.instructions[0] >> 56) == 1);
                CHECK(((lowered.instructions[0] >> 16) & 15) == guard);
                for (const bool predicate : {false, true}) {
                    LoggedMachine machine;
                    machine.predicates[3] = predicate;
                    auto registers = machine.registers;
                    const auto predicates = machine.predicates;
                    if (machine.Predicate(guard))
                        registers[0][original.Bits(16, 8)] = expected;
                    machine.Run(lowered.instructions);
                    CHECK(machine.registers == registers);
                    CHECK(machine.predicates == predicates);
                }
            }
        }
        for (u32 bit : {24U, 31U, 64U, 72U, 74U, 75U, 76U, 78U, 79U, 81U, 84U, 85U, 91U, 96U, 104U,
                        126U, 127U}) {
            CAPTURE(bit);
            auto invalid = original;
            invalid.words[bit / 32] ^= 1U << (bit % 32);
            const auto modified = LowerToMaxwell(std::array{invalid, Exit});
            if (bit == 74 || bit == 78 || bit == 79) {
                REQUIRE(modified);
                REQUIRE(modified.instructions.size() == 3);
                CHECK(modified.instructions[0] ==
                      (0x010000000007f0faULL | (u64{original.words[1]} << 20)));
                CHECK(modified.instructions[1] ==
                      (0x5cb800000fa70a00ULL | original.Bits(16, 8) |
                       (u64{invalid.Bits(74, 1)} << 13) | (u64{invalid.Bits(78, 2)} << 39)));
                HalfIR translation{modified.instructions};
                REQUIRE_FALSE(translation.block.Instructions().empty());
                if (bit == 74) {
                    CHECK(translation.Run(0, 0, original.Bits(16, 8)) ==
                          (round_up ? 0x42000000U : 0x43800000U));
                }
            } else {
                CHECK_FALSE(modified);
            }
        }
        auto scratch = original;
        scratch.words[0] = (scratch.words[0] & ~0xff0000U) | (250U << 16);
        CHECK_FALSE(LowerToMaxwell(std::array{scratch, Exit}));
        CHECK_FALSE(LowerToMaxwell(std::array{UniformBank, scratch, Exit}));
    }
}

TEST_CASE("Fast Fusion nonadjacent call links require a closed write-safe window and callee",
          "[video_core][ampere][fast_fusion_logged]") {
    const std::array<Instruction, 7> original{{
        {{0x00097802, 0x30, 0xf00, 0}},
        {{0x08087227, 3, 0x078e00ff, 0}},
        {{0x00007944, 0x10, 0x03c00000, 0}},
        Exit,
        CapturedImmediatePackedFloatTrunc,
        {{0x00047202, 9, 0xf00, 0}},
        {{0x04007950, 0xffffff90, 0x03c3ffff, 0}},
    }};
    const auto lowered = LowerToMaxwell(original);
    INFO(lowered.error);
    REQUIRE(lowered);
    CHECK(lowered.source_size == sizeof(original));
    CHECK(lowered.instructions.back() == 0xe32000000007000fULL);
    for (const auto clobber : {
             Instruction{{0x00097802, 0, 0xf00, 0}},
             Instruction{{0x00090802, 0, 0xf00, 0}},
             Instruction{{0x01087225, 2, 0x078e00ff, 0}},
             Instruction{{0x00087984, 0, 0xa00, 0}},
             Instruction{{0x00080984, 0, 0xc00, 0}},
             Instruction{{0x00087981, 0, 0x0c1e1d00, 0}},
             Instruction{{0x0008783b, 0, 0x100, 0}},
         }) {
        for (const u32 site : {1U, 4U}) {
            auto code = original;
            code[site] = clobber;
            const auto rejected = LowerToMaxwell(code);
            CHECK_FALSE(rejected);
            CHECK(rejected.error == "native call linkage mode");
        }
    }
    for (const u32 entry_opcode : {0x947U, 0x944U, 0x945U}) {
        std::vector<Instruction> code{original.begin(), original.end()};
        code.push_back(Instruction{{0x7000U | entry_opcode, 0xffffff90,
                                    entry_opcode == 0x944 ? 0x03c3ffffU : 0x0383ffffU, 0}});
        CHECK_FALSE(LowerToMaxwell(code));
    }
    for (const u32 site : {0U, 2U, 5U}) {
        auto code = original;
        code[site].words[0] &= ~0xf000U;
        CHECK_FALSE(LowerToMaxwell(code));
    }
    for (const u32 site : {1U, 4U}) {
        auto unknown = original;
        unknown[site] = Instruction{{0x02027209, 0x40000003, 0x07810200, 0}};
        const auto rejected = LowerToMaxwell(unknown);
        CHECK_FALSE(rejected);
        CHECK(rejected.error == "native call linkage mode");
    }
    auto wrong_copy = original;
    wrong_copy[5].words[1] = 8;
    CHECK_FALSE(LowerToMaxwell(wrong_copy));
    auto wrong_return = original;
    wrong_return[6].words[1] = 0xffffffa0;
    CHECK_FALSE(LowerToMaxwell(wrong_return));
    std::vector<Instruction> callee_entry{original.begin(), original.end()};
    callee_entry.push_back(Instruction{{0x00007945, 0xffffffd0, 0x0383ffff, 0}});
    CHECK_FALSE(LowerToMaxwell(callee_entry));
}

TEST_CASE("Fast Fusion adjacent callees reject implicit vector writes over their links",
          "[video_core][ampere][fast_fusion_logged]") {
    for (const auto load :
         {Instruction{{0x00087984, 0, 0xa00, 0}}, Instruction{{0x00087984, 0, 0xc00, 0}},
          Instruction{{0x00080984, 0, 0xc00, 0}}, Instruction{{0x00087981, 0, 0x0c1e1d00, 0}},
          Instruction{{0x0008783b, 0, 0x100, 0}}, Instruction{{0x0008783b, 0, 0x200, 0}}}) {
        const std::array<Instruction, 5> code{{
            {{0x00097802, 0x20, 0xf00, 0}},
            {{0x00007944, 0x10, 0x03c00000, 0}},
            Exit,
            load,
            {{0x09007950, 0xffffffb0, 0x03c3ffff, 0}},
        }};
        const auto lowered = LowerToMaxwell(code);
        CHECK_FALSE(lowered);
        CHECK(lowered.error == "native callee overwrites link register");
    }
}

TEST_CASE("Fast Fusion bounded callees reject branch and fallthrough entry",
          "[video_core][ampere][fast_fusion_logged]") {
    const std::array<Instruction, 8> branch_entry{{
        {{0x00000947, 0x40, 0x03800000, 0}},
        {{0x00097802, 0x40, 0xf00, 0}},
        {{0x08087227, 3, 0x078e00ff, 0}},
        {{0x00007944, 0x10, 0x03c00000, 0}},
        Exit,
        CapturedImmediatePackedFloatTrunc,
        {{0x00047202, 9, 0xf00, 0}},
        {{0x04007950, 0xffffff80, 0x03c3ffff, 0}},
    }};
    const std::array<Instruction, 9> fallthrough_entry{{
        {{0x00000947, 0x40, 0x03800000, 0}},
        {{0x00097802, 0x40, 0xf00, 0}},
        {{0x08087227, 3, 0x078e00ff, 0}},
        {{0x00007944, 0x20, 0x03c00000, 0}},
        Exit,
        {{0x00007918, 0, 0, 0}},
        CapturedImmediatePackedFloatTrunc,
        {{0x00047202, 9, 0xf00, 0}},
        {{0x04007950, 0xffffff70, 0x03c3ffff, 0}},
    }};
    const auto branch = LowerToMaxwell(branch_entry);
    CHECK_FALSE(branch);
    CHECK(branch.error == "native call linkage mode");
    const auto fallthrough = LowerToMaxwell(fallthrough_entry);
    CHECK_FALSE(fallthrough);
    CHECK(fallthrough.error == "native call linkage mode");
    auto closed = branch_entry;
    closed[0].words[1] = 0x30;
    const auto accepted = LowerToMaxwell(closed);
    INFO(accepted.error);
    CHECK(accepted);
}

TEST_CASE("Fast Fusion dead call intervals preserve source PCs and never accept reachable WARPSYNC",
          "[video_core][ampere][fast_fusion_logged]") {
    const Instruction warp_sync{{0x00007348, 7, 0x03800000, 0}};
    const std::array<Instruction, 13> original{{
        {{0x00097802, 0x30, 0xf00, 0}},
        {{0x08087227, 3, 0x078e00ff, 0}},
        {{0x00007944, 0x50, 0x03c00000, 0}},
        {{0x06007947, 0x33, 0x0b800000, 0}},
        {{0x00047802, 0x60, 0xf00, 0}},
        {{0x00007944, 0x50, 0x03c00000, 0}},
        {{0x00007918, 0, 0, 0}},
        Exit,
        CapturedImmediatePackedFloatTrunc,
        {{0x00047202, 9, 0xf00, 0}},
        {{0x04007950, 0xffffff50, 0x03c3ffff, 0}},
        warp_sync,
        {{0x04007950, 0xffffff30, 0x03c3ffff, 0}},
    }};
    const auto lowered = LowerToMaxwell(original);
    INFO(lowered.error);
    REQUIRE(lowered);
    CHECK(lowered.source_size == 0xb0);
    REQUIRE(lowered.source_instruction_offsets.size() == 11);
    for (u32 index = 4; index < 7; ++index)
        CHECK(lowered.source_instruction_offsets[index] == lowered.source_instruction_offsets[7]);
    u32 calls = 0;
    for (u32 index = 0; index < lowered.instructions.size(); ++index) {
        const u64 word = lowered.instructions[index];
        calls += (word & 0xfff0000000000000ULL) == 0xe260000000000000ULL;
        if ((word & 0xfff0000000000000ULL) == 0xe260000000000000ULL ||
            (word & 0xfff0000000000000ULL) == 0xe240000000000000ULL) {
            const s32 displacement = static_cast<s32>(u32(word >> 20) << 8) >> 8;
            const u32 source_target = (word >> 52) == 0xe26 ? 8U : 7U;
            CHECK(s64{Address(index)} + 8 + displacement ==
                  Address(lowered.source_instruction_offsets[source_target]));
        }
    }
    CHECK(calls == 1);
    auto guarded = original;
    guarded[3].words[0] &= ~0xf000U;
    CHECK_FALSE(LowerToMaxwell(guarded));
    for (const u32 opcode : {0x947U, 0x945U, 0x949U}) {
        std::vector<Instruction> code{original.begin(), original.end()};
        code.push_back(Instruction{{0x7000U | opcode, 0xffffff60, 0x0383ffff, 0}});
        CHECK_FALSE(LowerToMaxwell(code));
    }
    const auto reachable = LowerToMaxwell(std::array{warp_sync, Exit});
    CHECK_FALSE(reachable);
    CHECK(reachable.error == "unsupported native opcode");
}

TEST_CASE("Fast Fusion captured register-lane shuffle preserves modes masks and predicates",
          "[video_core][ampere][fast_fusion_logged]") {
    const auto captured = LowerToMaxwell(std::array{CapturedShuffleDown, Exit});
    INFO(captured.error);
    REQUIRE(captured);
    REQUIRE(captured.instructions.size() == 2);
    CHECK(captured.instructions[0] == 0xef17007ca0570007ULL);
    const std::array operations{IR::Opcode::ShuffleIndex, IR::Opcode::ShuffleUp,
                                IR::Opcode::ShuffleDown, IR::Opcode::ShuffleButterfly};
    for (u32 mode = 0; mode < operations.size(); ++mode) {
        for (u32 guard : {3U, 7U, 11U}) {
            for (u32 mask : {0x1fU, 0x1c03U}) {
                auto instruction = CapturedShuffleDown;
                instruction.words[0] = (instruction.words[0] & ~0xf000U) | (guard << 12);
                instruction.words[1] = 5U | (mask << 8) | (mode << 26);
                instruction.words[2] = 2U << 17;
                const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
                INFO(lowered.error);
                REQUIRE(lowered);
                REQUIRE(lowered.instructions.size() == 2);
                const u64 word = lowered.instructions[0];
                CHECK(((word >> 16) & 15) == guard);
                CHECK(((word >> 28) & 3) == 2);
                Shader::ObjectPool<IR::Inst> pool;
                IR::Block block{pool};
                HalfEnvironment env;
                Shader::Maxwell::TranslatorVisitor visitor{env, block};
                visitor.SHFL(word);
                const IR::Inst* shuffle = nullptr;
                u32 predicate_writes = 0, register_writes = 0;
                for (const auto& inst : block.Instructions()) {
                    if (inst.GetOpcode() == operations[mode]) {
                        shuffle = &inst;
                        for (u32 operand = 0; operand < 2; ++operand) {
                            REQUIRE(inst.Arg(operand).Inst()->GetOpcode() ==
                                    IR::Opcode::GetRegister);
                            CHECK(static_cast<u32>(inst.Arg(operand).Inst()->Arg(0).Reg()) ==
                                  (operand ? 5U : 0U));
                        }
                        for (u32 operand = 2; operand < 4; ++operand) {
                            const auto& extract = *inst.Arg(operand).Inst();
                            REQUIRE(extract.GetOpcode() == IR::Opcode::BitFieldUExtract);
                            CHECK(extract.Arg(0).U32() == mask);
                            CHECK(extract.Arg(1).U32() == (operand == 2 ? 0U : 8U));
                            CHECK(extract.Arg(2).U32() == 5);
                        }
                    } else if (inst.GetOpcode() == IR::Opcode::SetPred) {
                        ++predicate_writes;
                        CHECK(inst.Arg(0).Pred() == IR::Pred::P2);
                        const auto& bounds = *inst.Arg(1).Inst();
                        REQUIRE(bounds.GetOpcode() == IR::Opcode::GetInBoundsFromOp);
                        CHECK(bounds.Arg(0).Inst() == shuffle);
                    } else if (inst.GetOpcode() == IR::Opcode::SetRegister) {
                        ++register_writes;
                        CHECK(inst.Arg(0).Reg() == static_cast<IR::Reg>(7));
                        CHECK(inst.Arg(1).Inst() == shuffle);
                    }
                }
                REQUIRE(shuffle);
                CHECK(predicate_writes == 1);
                CHECK(register_writes == 1);
            }
        }
    }
    for (u32 bit : {53U, 54U, 55U, 56U, 57U, 60U, 61U, 62U, 63U, 64U, 72U, 80U, 84U, 91U, 96U, 104U,
                    126U, 127U}) {
        auto invalid = CapturedShuffleDown;
        invalid.words[bit / 32] |= 1U << (bit % 32);
        CHECK_FALSE(LowerToMaxwell(std::array{invalid, Exit}));
    }
    for (u32 bit : {16U, 24U, 32U}) {
        auto scratch = CapturedShuffleDown;
        scratch.words[bit / 32] =
            (scratch.words[bit / 32] & ~(255U << (bit % 32))) | (250U << (bit % 32));
        CHECK_FALSE(LowerToMaxwell(std::array{scratch, Exit}));
        CHECK_FALSE(LowerToMaxwell(std::array{UniformBank, scratch, Exit}));
    }
    auto mirrored_lane = CapturedShuffleDown;
    mirrored_lane.words[1] = (mirrored_lane.words[1] & ~255U) | 224U;
    CHECK_FALSE(LowerToMaxwell(std::array{UniformBank, mirrored_lane, Exit}));
    for (u32 bit : {16U, 24U, 32U}) {
        auto zero_register = CapturedShuffleDown;
        zero_register.words[bit / 32] |= 255U << (bit % 32);
        CHECK(LowerToMaxwell(std::array{zero_register, Exit}));
    }
}

TEST_CASE("Fast Fusion captured immediate select shadows only its register operands",
          "[video_core][ampere][fast_fusion_logged]") {
    const auto lowered = LowerToMaxwell(std::array{UniformBank, CapturedImmediateSelect, Exit});
    INFO(lowered.error);
    REQUIRE(lowered);
    for (const bool condition : {false, true}) {
        LoggedMachine machine;
        machine.predicates[0] = condition;
        const auto before = machine.registers;
        const auto predicates = machine.predicates;
        machine.Run(lowered.instructions);
        for (u32 reg = 0; reg < 255; ++reg) {
            if (reg < 246)
                CHECK(machine.registers[0][reg] ==
                      (reg == 216 ? machine.Constant(0x100, 4) : before[0][reg]));
            CHECK(machine.registers[1][reg] ==
                  (reg == 234 ? (condition ? 8U : 0U) : before[1][reg]));
        }
        CHECK(machine.predicates == predicates);
    }
}

TEST_CASE(
    "Fast Fusion captured extended uniform adds consume both UP inputs without changing predicates",
    "[video_core][ampere][fast_fusion_logged]") {
    for (const auto instruction : {CapturedExtendedCarry, CapturedExtendedTwoCarries}) {
        DYNAMIC_SECTION("mode " << instruction.words[2]) {
            const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
            INFO(lowered.error);
            REQUIRE(lowered);
            for (u32 packed = 0; packed < 128; ++packed) {
                for (u32 a : {0U, 1U, 0x7fffffffU, 0x80000000U, 0xffffffffU}) {
                    for (u32 b : {0U, 1U, 0x7fffffffU, 0x80000000U, 0xffffffffU}) {
                        LoggedMachine machine;
                        machine.registers[0][224] = a;
                        machine.registers[0][instruction.words[1] == 7 ? 223 : 221] = b;
                        machine.registers[0][239] = packed | 0xface0000U;
                        const auto before = machine.registers[0];
                        const auto predicates = machine.predicates;
                        u32 expected = a + b + (instruction.words[1] == 5 ? before[223] : 0U);
                        for (u32 bit : {77U, 87U}) {
                            const u32 input = instruction.Bits(bit, 4), index = input & 7;
                            expected += (index == 7 || bool(packed & (1U << (6 - index)))) !=
                                        bool(input & 8);
                        }
                        machine.Run(lowered.instructions);
                        CAPTURE(a, b, packed);
                        for (u32 reg = 0; reg < 246; ++reg)
                            CHECK(machine.registers[0][reg] ==
                                  (reg == 224 ? expected : before[reg]));
                        CHECK(machine.predicates == predicates);
                    }
                }
            }
        }
    }
}

TEST_CASE("Fast Fusion logged opcode support keeps nearby unsupported modes rejected",
          "[video_core][ampere][fast_fusion_logged]") {
    const auto reject = [](Instruction instruction, u32 bit) {
        instruction.words[2] ^= 1U << bit;
        CHECK_FALSE(LowerToMaxwell(std::array{UniformBank, instruction, Exit}));
    };
    reject(LoggedMove, 12);
    reject(LoggedZero, 12);
    auto compare = LoggedCompare;
    compare.words[2] |= 0xc00U;
    CHECK_FALSE(LowerToMaxwell(std::array{UniformBank, compare, Exit}));
    auto load = LoggedLoad;
    load.words[2] |= 0xe00U;
    CHECK_FALSE(LowerToMaxwell(std::array{load, Exit}));
    reject(LoggedLoad, 27);
    reject(LoggedMultiply, 10);
    reject(LoggedMultiply, 17);
    reject(CapturedMin, 10);
    reject(CapturedSpecialZero, 16);
    reject(CapturedMultiply, 10);
    reject(CapturedHalfConversion, 27);
    reject(CapturedHalfConversion, 9);
    reject(CapturedHalfConversion, 12);
    reject(CapturedHalfConversion, 15);
    reject(CapturedHalfFloor, 9);
    reject(CapturedHalfFloor, 12);
    reject(CapturedIntegerHalf, 10);
    reject(CapturedIntegerHalf, 14);
    reject(CapturedFloatUp, 15);
    for (const auto instruction : {CapturedHalfTrunc, CapturedHalfRound}) {
        reject(instruction, 14);
        auto modified = instruction;
        modified.words[1] |= 1U << 29;
        CHECK_FALSE(LowerToMaxwell(std::array{modified, Exit}));
        modified = instruction;
        modified.words[3] |= 1U;
        CHECK_FALSE(LowerToMaxwell(std::array{modified, Exit}));
    }
    for (u32 bit : {8U, 14U, 15U, 27U})
        reject(CapturedUniformPack, bit);
    auto uniform_pack = CapturedUniformPack;
    uniform_pack.words[0] &= ~0xff000000U;
    CHECK_FALSE(LowerToMaxwell(std::array{uniform_pack, Exit}));
    uniform_pack = CapturedUniformPack;
    uniform_pack.words[1] = 255;
    CHECK_FALSE(LowerToMaxwell(std::array{uniform_pack, Exit}));
    for (u32 bit : {2U, 3U, 9U, 10U, 11U, 13U, 14U, 15U, 16U, 26U})
        reject(CapturedScopedLoad, bit);
    auto scoped_load = CapturedScopedLoad;
    scoped_load.words[1] |= 1U << 6;
    CHECK_FALSE(LowerToMaxwell(std::array{scoped_load, Exit}));
    scoped_load = CapturedScopedLoad;
    scoped_load.words[1] = 13;
    CHECK_FALSE(LowerToMaxwell(std::array{scoped_load, Exit}));
    scoped_load = CapturedScopedLoad;
    scoped_load.words[0] += 1U << 16;
    CHECK_FALSE(LowerToMaxwell(std::array{scoped_load, Exit}));
    for (const auto instruction :
         {CapturedHalfConversion, CapturedHalfFloor, CapturedIntegerHalf}) {
        auto modified = instruction;
        modified.words[0] |= 1U << 24;
        CHECK_FALSE(LowerToMaxwell(std::array{modified, Exit}));
        modified = instruction;
        modified.words[1] |= 1U << 29;
        CHECK_FALSE(LowerToMaxwell(std::array{modified, Exit}));
        modified = instruction;
        modified.words[3] |= 1U;
        CHECK_FALSE(LowerToMaxwell(std::array{modified, Exit}));
    }
    for (u32 bit : {8U, 10U, 11U})
        reject(CapturedTwoCarries, bit);
    for (const auto instruction :
         {CapturedCarry, CapturedTwoCarries, CapturedExtendedCarry, CapturedExtendedTwoCarries}) {
        for (u32 bit : {8U, 9U, 11U, 12U})
            reject(instruction, bit);
        auto guarded = instruction;
        guarded.words[0] &= ~0xf000U;
        CHECK_FALSE(LowerToMaxwell(std::array{guarded, Exit}));
        auto modified = instruction;
        modified.words[1] |= 1U << 31;
        CHECK_FALSE(LowerToMaxwell(std::array{modified, Exit}));
    }
    auto duplicate_carry = CapturedTwoCarries;
    duplicate_carry.words[2] &= ~(7U << 17);
    CHECK_FALSE(LowerToMaxwell(std::array{duplicate_carry, Exit}));
    auto predicates = CapturedPredicates;
    predicates.words[1] |= 0x80U;
    CHECK_FALSE(LowerToMaxwell(std::array{UniformBank, predicates, Exit}));
    for (u32 bit : {6U, 8U, 9U, 11U, 12U, 13U, 17U, 18U, 27U})
        reject(LoggedCopy, bit);
    auto copy = LoggedCopy;
    copy.words[0] |= 1U << 24;
    CHECK_FALSE(LowerToMaxwell(std::array{copy, Exit}));
    copy = LoggedCopy;
    copy.words[0] = (copy.words[0] & ~0xff000000U) | (246U << 24);
    CHECK_FALSE(LowerToMaxwell(std::array{copy, Exit}));
    const Instruction masked_wait{{0x0000791a, 0x00008001, 0, 0}};
    const auto ordered = LowerToMaxwell(std::array{masked_wait, Exit});
    REQUIRE(ordered);
    REQUIRE(ordered.instructions.size() == 1);
    CHECK(ordered.instructions.front() == 0xe30000000007000fULL);
    const auto copying = LowerToMaxwell(std::array{LoggedCopy, masked_wait, Exit});
    REQUIRE(copying);
    CHECK(copying.instructions[copying.source_instruction_offsets[1]] == 0xef98000000070000ULL);
    CHECK(copying.source_instruction_offsets[2] == copying.source_instruction_offsets[1] + 1);
    auto invalid_wait = masked_wait;
    invalid_wait.words[1] |= 1U << 16;
    CHECK_FALSE(LowerToMaxwell(std::array{invalid_wait, Exit}));
    CHECK_FALSE(Shader::Ampere::LowerToMaxwellWith(std::array{UniformBank, LoggedMove, Exit}, 2,
                                                   false, false));
    for (u32 bit : {12U, 13U, 14U, 15U}) {
        copy = LoggedCopy;
        copy.words[1] ^= 1U << bit;
        CHECK_FALSE(LowerToMaxwell(std::array{copy, Exit}));
    }
    auto scratch = LoggedMove;
    scratch.words[0] = (scratch.words[0] & ~0xff0000U) | (246U << 16);
    CHECK_FALSE(LowerToMaxwell(std::array{scratch, Exit}));
}
