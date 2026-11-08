// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <span>
#include <vector>

#include "common/logging.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"

namespace Vulkan {

inline std::vector<u32> EmitPipelineShader(const Shader::Profile& profile,
                                           const Shader::RuntimeInfo& runtime_info,
                                           Shader::IR::Program& program,
                                           Shader::Backend::Bindings& binding, u64 shader_hash,
                                           std::span<const u32> reference = {}) {
    std::vector<u32> code{
        Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, binding)};
    if (!reference.empty() && !std::ranges::equal(code, reference)) {
        LOG_ERROR(Render_Vulkan,
                  "Nx2Perf async_pipelines: shader {:016x} worker SPIR-V differs; "
                  "using GPU-thread SPIR-V",
                  shader_hash);
        code.assign(reference.begin(), reference.end());
    }
    return code;
}

} // namespace Vulkan
