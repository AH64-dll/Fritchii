#include <stdexcept>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/passes.h"

namespace {
namespace IR = Shader::IR;

class RectEnvironment final : public Shader::Environment {
public:
    bool rectangle{true};

    u64 ReadInstruction(u32) override { throw std::logic_error("Unexpected instruction read"); }
    u32 ReadCbufValue(u32 index, u32 offset) override {
        REQUIRE(index == 0);
        REQUIRE((offset == 0 || offset == 4));
        return offset;
    }
    Shader::TextureType ReadTextureType(u32 handle) override {
        return rectangle && handle == 4 ? Shader::TextureType::Color2DRect
                                        : Shader::TextureType::Color2D;
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32) override { return false; }
    u32 ReadViewportTransformState() override { return 1; }
    u32 TextureBoundBuffer() const override { return 0; }
    u32 LocalMemorySize() const override { return 0; }
    u32 SharedMemorySize() const override { return 0; }
    std::array<u32, 3> WorkgroupSize() const override { return {1, 1, 1}; }
    bool HasHLEMacroState() const override { return false; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}
};
}

TEST_CASE("Rectangle size queries use the sampled descriptor", "[shader][upstream_rect]") {
    for (const bool rectangle : {false, true}) {
        CAPTURE(rectangle);
        Shader::ObjectPool<IR::Inst> pool{256};
        IR::Block block{pool};
        IR::Program program;
        program.stage = Shader::Stage::Fragment;
        program.blocks = {&block};
        program.post_order_blocks = {&block};
        RectEnvironment env;
        env.rectangle = rectangle;
        IR::IREmitter ir{block};
        IR::TextureInstInfo flags{};
        flags.type.Assign(Shader::TextureType::Color2D);
        const auto coords{ir.CompositeConstruct(ir.Imm32(320.0f), ir.Imm32(180.0f))};
        static_cast<void>(ir.ImageSampleImplicitLod(ir.GetCbuf(ir.Imm32(0), ir.Imm32(0)),
                                                   coords, ir.Imm32(0.0f), {}, {}, flags));
        const auto sample{ir.ImageSampleImplicitLod(ir.GetCbuf(ir.Imm32(0), ir.Imm32(4)),
                                                   coords, ir.Imm32(0.0f), {}, {}, flags)};
        Shader::HostTranslateInfo host;
        host.ApplyDescriptorLimitPolicy();
        Shader::Optimization::TexturePass(env, program, host);
        REQUIRE(program.info.texture_descriptors.size() == 2);
        REQUIRE(sample.Inst()->Flags<IR::TextureInstInfo>().descriptor_index == 1);
        size_t queries{};
        for (const IR::Inst& inst : block.Instructions()) {
            if (inst.GetOpcode() != IR::Opcode::ImageQueryDimensions) {
                continue;
            }
            ++queries;
            CHECK(inst.Flags<IR::TextureInstInfo>().descriptor_index == 1);
            CHECK(inst.Arg(0).U32() == 0);
        }
        CHECK(queries == (rectangle ? 1U : 0U));
        CHECK((sample.Inst()->Arg(1) == coords) == !rectangle);
    }
}
