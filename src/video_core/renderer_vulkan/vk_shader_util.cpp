// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// For spv::HasResultAndType and spv::OpToString (FindSpirvIdProblem). Defined before anything
// can include the SPIR-V header without them.
#ifndef SPV_ENABLE_UTILITY_CODE
#define SPV_ENABLE_UTILITY_CODE
#endif
#include <spirv/unified1/spirv.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

vk::ShaderModule BuildShader(const Device& device, std::span<const u32> code) {
    return device.GetLogical().CreateShaderModule({
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = static_cast<u32>(code.size_bytes()),
        .pCode = code.data(),
    });
}

bool SpirvSelfCheck() {
    // Read at the first emitted shader, not at static init: nx2_switches.txt is applied by then.
    static const bool on = [] {
        const char* const value = std::getenv("EDEN_NX2_SPIRV_SELFCHECK");
        const bool enabled = value != nullptr && value[0] == '1';
        if (enabled) {
            LOG_INFO(Render_Vulkan, "EDEN_NX2_SPIRV_SELFCHECK=1: emitted shaders are checked for "
                                    "undefined and self-used ids before the driver gets them");
        }
        return enabled;
    }();
    return on;
}

namespace {
/// The words of an instruction that are value ids, for the opcodes the check knows: the range
/// [first, last), or {0, 0} for an opcode it does not check.
std::pair<size_t, size_t> ValueOperands(spv::Op op, size_t count) {
    const std::pair<size_t, size_t> all{3, count};
    switch (op) {
    case spv::OpLoad:
        return {3, (std::min<size_t>)(4, count)}; // The pointer; memory operands are literals
    case spv::OpStore:
        return {1, (std::min<size_t>)(3, count)}; // No result: the pointer and the object
    case spv::OpCompositeExtract:
        return {3, (std::min<size_t>)(4, count)}; // The composite; the indexes are literals
    case spv::OpCompositeInsert:
    case spv::OpVectorShuffle:
        return {3, (std::min<size_t>)(5, count)}; // Two ids, then literals
    case spv::OpBranchConditional:
    case spv::OpReturnValue:
        return {1, (std::min<size_t>)(2, count)}; // The condition or value; labels come later
    case spv::OpCompositeConstruct:
    case spv::OpAccessChain:
    case spv::OpInBoundsAccessChain:
    case spv::OpVectorExtractDynamic:
    case spv::OpVectorInsertDynamic:
        return all;
    default:
        break;
    }
    // Conversions up to OpGenericCastToPtr (OpGenericCastToPtrExplicit has a literal), and
    // OpBitcast, the arithmetic, the relational and logical operations (OpSelect included) and
    // the bit operations: every operand is an id.
    if ((op >= spv::OpConvertFToU && op <= spv::OpGenericCastToPtr) || op == spv::OpBitcast ||
        (op >= spv::OpSNegate && op <= spv::OpSMulExtended) ||
        (op >= spv::OpAny && op <= spv::OpFUnordGreaterThanEqual) ||
        (op >= spv::OpShiftRightLogical && op <= spv::OpBitCount)) {
        return all;
    }
    return {0, 0};
}
} // Anonymous namespace

std::string FindSpirvIdProblem(std::span<const u32> code) {
    if (code.size() < 5 || code[0] != spv::MagicNumber) {
        return "no SPIR-V header";
    }
    const u32 bound = code[3];
    std::vector<bool> defined(bound);
    std::vector<std::pair<u32, u32>> phi_values; // (value id, phi result), checked at the end
    for (size_t offset = 5; offset < code.size();) {
        const size_t count = code[offset] >> 16;
        const auto op = static_cast<spv::Op>(code[offset] & 0xffff);
        if (count == 0 || count > code.size() - offset) {
            return fmt::format("word {}: {} has length {}, past the end", offset,
                               spv::OpToString(op), count);
        }
        const std::span<const u32> words = code.subspan(offset, count);
        bool has_result{};
        bool has_type{};
        spv::HasResultAndType(op, &has_result, &has_type);
        const size_t result_word = has_type ? 2 : 1;
        u32 result{};
        if (has_result) {
            if (count <= result_word) {
                return fmt::format("word {}: {} has no result id", offset, spv::OpToString(op));
            }
            result = words[result_word];
            if (result == 0 || result >= bound) {
                return fmt::format("word {}: {} defines %{}, past the bound {}", offset,
                                   spv::OpToString(op), result, bound);
            }
            if (defined[result]) {
                return fmt::format("word {}: %{} = {} is defined a second time", offset, result,
                                   spv::OpToString(op));
            }
        }
        if (op == spv::OpPhi) {
            for (size_t index = 3; index < count; index += 2) {
                phi_values.emplace_back(words[index], result);
            }
        } else {
            const auto [first, last] = ValueOperands(op, count);
            if (first != 0 && has_type && (words[1] >= bound || !defined[words[1]])) {
                return fmt::format("word {}: %{} = {} has the undefined type %{}", offset, result,
                                   spv::OpToString(op), words[1]);
            }
            for (size_t index = first; index < last; ++index) {
                const u32 id = words[index];
                if (has_result && id == result) {
                    return fmt::format("word {}: %{} = {} uses itself", offset, result,
                                       spv::OpToString(op));
                }
                if (id >= bound || !defined[id]) {
                    return fmt::format("word {}: %{} = {} uses %{} before its definition", offset,
                                       result, spv::OpToString(op), id);
                }
            }
        }
        if (has_result) {
            defined[result] = true;
        }
        offset += count;
    }
    for (const auto& [id, phi] : phi_values) {
        if (id >= bound || !defined[id]) {
            return fmt::format("%{} = OpPhi uses %{}, which is never defined", phi, id);
        }
    }
    return {};
}

} // namespace Vulkan
