// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <unordered_map>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ir/modifiers.h"
#include "shader_recompiler/frontend/maxwell/structured_control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"

namespace {
using Shader::Ampere::Instruction;
constexpr Instruction Exit{{0x794d, 0, 0x03800000, 0}};

u32 Address(u32 index) {
    return 0x50 + index * 8 + ((index + 1) / 3) * 8;
}

Instruction Copy(u32 guard, u32 enable, s32 global_offset, s32 shared_offset, u32 shared = 6,
                 u32 global = 2) {
    return {
        {0xfae | (guard << 12) | (shared << 16) | (global << 24),
         (static_cast<u32>(global_offset) & 0xfff) | (static_cast<u32>(shared_offset / 16) << 16),
         0x0c101c08U | ((enable & 7) << 23) | ((enable >> 3) << 6), 0}};
}

struct CopyMachine {
    std::array<u32, 256> registers{};
    std::array<bool, 7> predicates{};
    std::unordered_map<u64, u32> global;
    std::unordered_map<u32, u32> shared;
    u32 reads{}, writes{};

    bool Predicate(u32 predicate) const {
        return ((predicate & 7) == 7 || predicates[predicate & 7]) != bool(predicate & 8);
    }

    void Run(std::span<const u64> code) {
        for (u32 pc = 0; pc < code.size(); ++pc) {
            const u64 word = code[pc];
            const auto bits = [&](u32 start, u32 count) {
                return static_cast<u32>((word >> start) & ((u64{1} << count) - 1));
            };
            if (!Predicate(bits(16, 4)))
                continue;
            const u32 dest = bits(0, 8), address_reg = bits(8, 8);
            const s32 offset = static_cast<s32>(bits(20, 24) << 8) >> 8;
            if ((word >> 56) == 0x01) {
                if (dest != 255)
                    registers[dest] = bits(20, 32);
            } else if ((word & 0xfff0000000000000ULL) == 0xe240000000000000ULL) {
                const u32 target = Address(pc) + 8 + offset;
                const auto start = pc;
                while (pc < code.size() && Address(pc) != target)
                    ++pc;
                REQUIRE(pc > start);
                REQUIRE(pc < code.size());
                --pc;
            } else if ((word & 0xfff0000000000000ULL) == 0xeed0000000000000ULL) {
                REQUIRE(bits(45, 1) == 1);
                REQUIRE(bits(48, 3) == 6);
                REQUIRE(dest % 4 == 0);
                REQUIRE(address_reg % 2 == 0);
                const u64 address =
                    registers[address_reg] | (u64{registers[address_reg + 1]} << 32);
                for (u32 i = 0; i < 4; ++i) {
                    // Missing memory is a test failure, including for zero-filled copies.
                    const u64 source = address + static_cast<u64>(offset) + i * 4;
                    REQUIRE(global.contains(source));
                    registers[dest + i] = global.at(source);
                    ++reads;
                }
            } else if ((word & 0xfff8000000000000ULL) == 0xef58000000000000ULL) {
                REQUIRE(bits(48, 3) == 6);
                REQUIRE(dest % 4 == 0);
                for (u32 i = 0; i < 4; ++i) {
                    shared[registers[address_reg] + static_cast<u32>(offset) + i * 4] =
                        registers[dest + i];
                    ++writes;
                }
            } else if ((word & 0xfff0000000000000ULL) == 0xe300000000000000ULL) {
                return;
            } else {
                FAIL("Unexpected copy instruction " << word);
            }
        }
    }
};

class CopyEnvironment final : public Shader::Environment {
public:
    explicit CopyEnvironment(std::span<const u64> code_) : code{code_} {
        stage = Shader::Stage::Compute;
    }
    u64 ReadInstruction(u32 address) override {
        for (u32 i = 0; i < code.size(); ++i)
            if (Address(i) == address)
                return code[i];
        FAIL("Frontend read outside lowered copy program at " << address);
        return 0;
    }
    u32 ReadCbufValue(u32, u32) override {
        FAIL("Unexpected cbuf read");
        return 0;
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
        return 4096;
    }
    std::array<u32, 3> WorkgroupSize() const override {
        return {32, 1, 1};
    }
    bool HasHLEMacroState() const override {
        return false;
    }
    bool HasNativeInstructions() const override {
        return native;
    }
    bool native = true;
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}

private:
    std::span<const u64> code;
};
} // namespace

TEST_CASE("Fire Emblem native TANH reaches SPIRV without enabling it for Maxwell", "[ampere][fe_restore]") {
    const Instruction tanh{{0x000c7308, 12, 0x2400, 0x000e3000}};
    const auto lowered = Shader::Ampere::LowerToMaxwell(std::array{tanh, Exit});
    REQUIRE(lowered);
    auto code = lowered.instructions;
    code.insert(code.begin(), 0xf0c800000007000cULL); // lane id in R12
    code.insert(code.end() - 1, 0xef5800000007000cULL | (4ULL << 48) | (255ULL << 8));
    for (const bool native : {true, false}) {
        CopyEnvironment env{code};
        env.native = native;
        Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
        Shader::Maxwell::Flow::CFG cfg{env, flow_pool, Shader::Maxwell::Location{0x50}};
        Shader::ObjectPool<Shader::IR::Inst> inst_pool;
        Shader::ObjectPool<Shader::IR::Block> block_pool;
        if (!native) {
            REQUIRE_THROWS(Shader::Maxwell::TranslateProgram(inst_pool, block_pool, env, cfg,
                                                           Shader::HostTranslateInfo{}));
            continue;
        }
        auto program = Shader::Maxwell::TranslateProgram(inst_pool, block_pool, env, cfg,
                                                               Shader::HostTranslateInfo{});
        Shader::Profile profile{};
        profile.supported_spirv = 0x00010300;
        REQUIRE(Shader::Backend::SPIRV::EmitSPIRV(profile, program).size() > 5);
    }
}

TEST_CASE("Fire Emblem library calls retain checked links and nested returns", "[ampere][fe_restore]") {
    std::array<Instruction, 7> code{{
        {{0x00377802, 0x20, 0xf00, 0}},
        {{0x7944, 0x10, 0x03c00000, 0}},
        Exit,
        {{0x00367802, 0x50, 0xf00, 0}},
        {{0x7944, 0x10, 0x03c00000, 0}},
        {{0x37007950, 0xffffffa0, 0x03c3ffff, 0}},
        {{0x36007950, 0xffffff90, 0x03c3ffff, 0}},
    }};
    const auto lowered = Shader::Ampere::LowerToMaxwell(code);
    INFO(lowered.error);
    REQUIRE(lowered);
    CHECK(lowered.source_size == sizeof(code));
    code[3].words[1] = 0x40; // Incorrect nested return PC still fails.
    CHECK_FALSE(Shader::Ampere::LowerToMaxwell(code));
}

TEST_CASE("Fire Emblem HFMA ReLU predicate gates the clamp not the FMA", "[ampere][fe_restore]") {
    // Ampere hardware oracle: P0=false preserves bc003c00 and 80008000;
    // P0=true yields 00003c00 and 00000000, without modifying P0.
    for (u32 guard = 0; guard < 16; ++guard) {
        for (u32 relu = 0; relu < 16; ++relu) {
            Instruction instruction{{0x08080831U | (guard << 12), 0x3c003c00,
                                     0x800bU | (relu << 23), 0}};
            const auto lowered = Shader::Ampere::LowerToMaxwell(std::array{instruction, Exit});
            REQUIRE(lowered);
            for (u32 flags = 0; flags < 128; ++flags) {
                const auto predicate = [&](u32 p) {
                    return ((p & 7) == 7 || ((flags >> (p & 7)) & 1)) != bool(p & 8);
                };
                bool fma = false, clamp = false, preserved = false;
                for (u32 pc = 0; pc < lowered.instructions.size(); ++pc) {
                    const u64 word = lowered.instructions[pc];
                    if (!predicate((word >> 16) & 15))
                        continue;
                    if ((word & 0xfff0000000000000ULL) == 0xe240000000000000ULL) {
                        const s32 offset = static_cast<s32>((word >> 20) << 8) >> 8;
                        const u32 target = Address(pc) + 8 + offset;
                        while (pc < lowered.instructions.size() && Address(pc) != target)
                            ++pc;
                        REQUIRE(pc < lowered.instructions.size());
                        --pc;
                    } else if ((word & 0xfff0000000000000ULL) == 0x5d00000000000000ULL) {
                        fma = true;
                        CHECK((word & 255) == 254);
                    } else if ((word & 0xfff8000000000000ULL) == 0x5c98000000000000ULL) {
                        preserved = true;
                        CHECK((word & 255) == 8);
                        CHECK(((word >> 20) & 255) == 254);
                    } else if ((word & 0xfff8000000000000ULL) == 0x38e8000000000000ULL) {
                        clamp = true;
                        break;
                    }
                }
                CHECK(fma == predicate(guard));
                CHECK(clamp == (predicate(guard) && predicate(relu)));
                CHECK(preserved == (predicate(guard) && relu != 7));
            }
        }
    }
}

TEST_CASE("Fire Emblem register shadows survive SSA and SPIRV", "[ampere][fe_restore]") {
    using Shader::Ampere::NativeRegisterCopy;
    for (const u32 high : {216U, 217U, 240U, 246U, 254U}) {
        const Instruction add{{0x00007230U | (high << 16) | (high << 24), high, 0, 0}};
        const auto pressure = Shader::Ampere::PrepareRegisterPressure(add, 216);
        REQUIRE(pressure.valid);
        REQUIRE(pressure.prefix.size() == 8);
        REQUIRE(pressure.suffix.size() == 8);
        CHECK(pressure.instruction.Bits(16, 8) < 216);
        CHECK(pressure.instruction.Bits(16, 8) == pressure.instruction.Bits(24, 8));
        CHECK(pressure.instruction.Bits(16, 8) == pressure.instruction.Bits(32, 8));
        auto lowered = Shader::Ampere::LowerToMaxwell(std::array{pressure.instruction, Exit});
        REQUIRE(lowered);
        std::vector<u64> code{0xf0c8000000070000ULL}; // lane ID, kept live through the shadows
        code.push_back(NativeRegisterCopy(1, high, 0, 0));
        code.insert(code.end(), pressure.prefix.begin(), pressure.prefix.end());
        code.insert(code.end(), lowered.instructions.begin(), lowered.instructions.end() - 1);
        code.insert(code.end(), pressure.suffix.begin(), pressure.suffix.end());
        code.push_back(NativeRegisterCopy(0, 0, 1, high));
        code.push_back(0xef58000000070000ULL | (4ULL << 48) | (255ULL << 8));
        code.push_back(lowered.instructions.back());
        CopyEnvironment env{code};
        Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
        Shader::Maxwell::Flow::CFG cfg{env, flow_pool, Shader::Maxwell::Location{0x50}};
        Shader::ObjectPool<Shader::IR::Inst> inst_pool;
        Shader::ObjectPool<Shader::IR::Block> block_pool;
        auto program = Shader::Maxwell::TranslateProgram(inst_pool, block_pool, env, cfg,
                                                               Shader::HostTranslateInfo{});
        Shader::Profile profile{};
        profile.supported_spirv = 0x00010300;
        REQUIRE(Shader::Backend::SPIRV::EmitSPIRV(profile, program).size() > 5);
    }
}

TEST_CASE("Fire Emblem conditional ReLU reaches SPIRV with P6 guards", "[ampere][fe_restore]") {
    for (const u32 guard : {6U, 7U, 14U}) {
        for (u32 relu = 0; relu < 16; ++relu) {
            const Instruction instruction{{0x08080831U | (guard << 12), 0x3c003c00,
                                           0x800bU | (relu << 23), 0}};
            const auto lowered = Shader::Ampere::LowerToMaxwell(std::array{instruction, Exit});
            REQUIRE(lowered);
            auto code = lowered.instructions;
            code.insert(code.begin(), 0xf0c8000000070008ULL);
            code.insert(code.end()-1, 0xef58000000070008ULL | (4ULL << 48) | (255ULL << 8));
            CopyEnvironment env{code};
            Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
            Shader::Maxwell::Flow::CFG cfg{env, flow_pool, Shader::Maxwell::Location{0x50}};
            Shader::ObjectPool<Shader::IR::Inst> inst_pool;
            Shader::ObjectPool<Shader::IR::Block> block_pool;
            auto program = Shader::Maxwell::TranslateProgram(inst_pool, block_pool, env, cfg,
                                                                   Shader::HostTranslateInfo{});
            Shader::Profile profile{};
            profile.supported_spirv = 0x00010300;
            REQUIRE(Shader::Backend::SPIRV::EmitSPIRV(profile, program).size() > 5);
        }
    }
}

TEST_CASE("Fire Emblem skipped clamp still restores borrowed registers", "[ampere][fe_restore]") {
    const Instruction uniform_vote{{0x00107886, 0, 0x038e0100, 0}};
    for (const u32 guard : {0U, 7U}) {
        const Instruction hfma{{0xd8d80831U | (guard << 12), 0x3c003c00, 0x800b, 0}};
        const auto lowered = Shader::Ampere::LowerToMaxwell(std::array{uniform_vote, hfma, Exit});
        INFO(lowered.error);
        REQUIRE(lowered);
        std::array<u32, Shader::IR::NUM_REGS> registers{};
        for (u32 i = 0; i < registers.size(); ++i)
            registers[i] = 0x10000 + i;
        registers[255] = 0;
        const auto before = registers;
        for (u32 pc = lowered.source_instruction_offsets[1]; pc < lowered.instructions.size(); ++pc) {
            const u64 word = lowered.instructions[pc];
            const u32 predicate = (word >> 16) & 15;
            if (((predicate & 7) == 7) == bool(predicate & 8))
                continue; // All real predicates are false.
            const u32 dest = word & 255, source = (word >> 20) & 255;
            if ((word & 0xfff0000000000000ULL) == 0xe240000000000000ULL) {
                const s32 offset = static_cast<s32>((word >> 20) << 8) >> 8;
                const u32 target = Address(pc) + 8 + offset;
                while (pc < lowered.instructions.size() && Address(pc) != target)
                    ++pc;
                REQUIRE(pc < lowered.instructions.size());
                --pc;
            } else if ((word & 0xfff8000000000000ULL) == 0x5c98000000000000ULL) {
                if (((word >> 44) & 15) == 10 && ((word >> 39) & 15) == 0) {
                    registers[dest + ((word >> 8) & 3) * 256] =
                        registers[source + ((word >> 10) & 3) * 256];
                } else {
                    registers[dest] = registers[source];
                }
            } else if ((word >> 56) == 1) {
                registers[dest] = static_cast<u32>(word >> 20);
            } else if ((word & 0xfff0000000000000ULL) == 0x5d00000000000000ULL) {
                // Inject the hardware-oracle result; test the shadow/branch protocol separately.
                registers[dest] = 0xbc003c00;
            } else if ((word & 0xfff0000000000000ULL) == 0xe300000000000000ULL) {
                break;
            } else {
                FAIL("Unexpected instruction in shadow handback " << word);
            }
        }
        for (u32 i = 0; i < 246; ++i)
            CHECK(registers[i] == before[i]);
        for (u32 i = 256; i < 512; ++i)
            CHECK(registers[i] == (i == 256 + 216 && guard == 7 ? 0xbc003c00U : before[i]));
    }
}

TEST_CASE("Ampere global to shared copy survives production control flow and IR translation",
          "[video_core][ampere][async_copy][fire_emblem]") {
    for (u32 prefix = 0; prefix < 3; ++prefix) {
        for (const u32 guard : {0U, 7U, 8U}) {
            for (const u32 enable : {4U, 12U}) {
                std::vector<Instruction> code(prefix, Instruction{{0x00ff7802, 0, 0xf00, 0}});
                code.push_back(Copy(guard, enable, -16, 32));
                code.push_back(Exit);
                const auto lowered = Shader::Ampere::LowerToMaxwell(code);
                REQUIRE(lowered);
                CopyEnvironment env{lowered.instructions};
                Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
                Shader::Maxwell::Flow::CFG cfg{env, flow_pool, Shader::Maxwell::Location{0x50}};
                Shader::ObjectPool<Shader::IR::Inst> inst_pool;
                Shader::ObjectPool<Shader::IR::Block> block_pool;
                const auto asl = Shader::Maxwell::BuildASL(inst_pool, block_pool, env, cfg,
                                                           Shader::HostTranslateInfo{});
                u32 loads = 0, stores = 0;
                for (const auto& node : asl) {
                    if (node.type != Shader::IR::AbstractSyntaxNode::Type::Block)
                        continue;
                    for (const auto& inst : *node.data.block) {
                        loads += inst.GetOpcode() == Shader::IR::Opcode::LoadGlobal128;
                        stores += inst.GetOpcode() == Shader::IR::Opcode::WriteSharedU128;
                        REQUIRE(inst.GetOpcode() != Shader::IR::Opcode::Barrier);
                    }
                }
                REQUIRE(loads == 1);
                REQUIRE(stores == 1);
            }
        }
    }
}

TEST_CASE("Ampere global to shared copy guards reads and zero fills disabled sources",
          "[video_core][ampere][async_copy][fire_emblem]") {
    for (u32 prefix = 0; prefix < 3; ++prefix) {
        for (const u32 guard : {0U, 7U, 8U, 15U}) {
            for (u32 enable = 0; enable < 16; ++enable) {
                for (const bool condition : {false, true}) {
                    for (const s32 offset : {-2048, -16, 0, 16, 2032}) {
                        std::vector<Instruction> code(prefix,
                                                      Instruction{{0x00ff7802, 0, 0xf00, 0}});
                        code.push_back(Copy(guard, enable, offset, offset));
                        code.push_back(Exit);
                        const auto lowered = Shader::Ampere::LowerToMaxwell(code);
                        INFO(lowered.error);
                        REQUIRE(lowered);
                        CopyMachine machine;
                        machine.registers[2] = 0xfffffff0U;
                        machine.registers[3] = 3;
                        machine.registers[6] = 4096;
                        machine.predicates.fill(condition);
                        const auto before = machine.registers;
                        const bool active = machine.Predicate(guard);
                        const bool enabled = machine.Predicate(enable);
                        const std::array data{0x10203040U, 0x50607080U, 0x90a0b0c0U, 0xd0e0f000U};
                        for (u32 i = 0; i < 4; ++i) {
                            if (active && enabled)
                                machine.global[0x3fffffff0ULL + static_cast<u64>(offset) + 4 * i] =
                                    data[i];
                            machine.shared[4096 + static_cast<u32>(offset) + 4 * i] = 0xfeedface;
                        }
                        machine.Run(lowered.instructions);
                        REQUIRE(machine.reads == (active && enabled ? 4U : 0U));
                        REQUIRE(machine.writes == (active ? 4U : 0U));
                        REQUIRE(machine.shared.size() == 4);
                        for (u32 i = 0; i < 4; ++i)
                            REQUIRE(machine.shared.at(4096 + static_cast<u32>(offset) + 4 * i) ==
                                    (!active   ? 0xfeedfaceU
                                     : enabled ? data[i]
                                               : 0U));
                        for (u32 i = 0; i < 246; ++i)
                            REQUIRE(machine.registers[i] == before[i]);
                        for (bool value : machine.predicates)
                            REQUIRE(value == condition);
                    }
                }
            }
        }
    }
}

TEST_CASE("Ampere global to shared copy rejects unsupported addressing and modes",
          "[video_core][ampere][async_copy][fire_emblem]") {
    const auto valid = Copy(7, 7, 0, 0);
    const auto reference = Shader::Ampere::LowerToMaxwell(std::array{valid, Exit});
    REQUIRE(reference);
    for (u32 descriptor = 0; descriptor < 64; ++descriptor) {
        CAPTURE(descriptor);
        auto copy = valid;
        copy.words[2] = (copy.words[2] & ~0x3fU) | descriptor;
        const auto lowered = Shader::Ampere::LowerToMaxwell(std::array{copy, Exit});
        if (descriptor & 1) {
            REQUIRE_FALSE(lowered);
        } else {
            REQUIRE(lowered);
            CHECK(lowered.instructions == reference.instructions);
        }
    }
    for (const u32 bit :
         {0U, 3U, 7U, 8U, 9U, 10U, 11U, 12U, 13U, 20U, 21U, 22U, 26U, 27U, 28U, 31U}) {
        auto invalid = valid;
        invalid.words[2] ^= 1U << bit;
        const auto lowered = Shader::Ampere::LowerToMaxwell(std::array{invalid, Exit});
        if (bit == 3) {
            REQUIRE(lowered);
            CHECK(lowered.instructions == reference.instructions);
        } else {
            REQUIRE_FALSE(lowered);
        }
    }
    for (const u32 bit : {12U, 13U, 14U, 15U}) {
        auto invalid = valid;
        invalid.words[1] ^= 1U << bit;
        REQUIRE_FALSE(Shader::Ampere::LowerToMaxwell(std::array{invalid, Exit}));
    }
    for (const u32 address : {1U, 245U, 246U, 254U, 255U})
        REQUIRE_FALSE(
            Shader::Ampere::LowerToMaxwell(std::array{Copy(7, 7, 0, 0, 6, address), Exit}));
    for (const u32 address : {246U, 254U, 255U})
        REQUIRE_FALSE(Shader::Ampere::LowerToMaxwell(std::array{Copy(7, 7, 0, 0, address), Exit}));
}

TEST_CASE("Ampere synchronous copy commit and scoreboard wait retain memory ordering",
          "[video_core][ampere][async_copy][fire_emblem]") {
    const Instruction commit{{0x79af, 0, 0, 0}};
    const Instruction wait{{0x791a, 0x8000, 0, 0}};
    const auto lowered =
        Shader::Ampere::LowerToMaxwell(std::array{Copy(7, 12, 0, 0), commit, wait, Exit});
    REQUIRE(lowered);
    for (u32 index : {1U, 2U})
        REQUIRE(lowered.instructions[lowered.source_instruction_offsets[index]] ==
                0xef98000000070000ULL);
    for (u32 field = 0; field < 3; ++field) {
        auto invalid = commit;
        invalid.words[field] ^= 1;
        REQUIRE_FALSE(Shader::Ampere::LowerToMaxwell(std::array{invalid, Exit}));
    }
    auto invalid_wait = wait;
    invalid_wait.words[1] = 0x7000;
    REQUIRE_FALSE(Shader::Ampere::LowerToMaxwell(std::array{invalid_wait, Exit}));
}
