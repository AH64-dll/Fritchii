// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <utility>

#include "common/common_types.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/memory_manager.h"

namespace VideoCommon {

/// The texture handle a shader reads from a storage buffer, computed from the current constant
/// buffer and storage buffer words (Shader::Info::handle_programs). cbuf_address(index) is the
/// bound constant buffer's address, 0 when none is bound. storage_alignment is the host storage
/// buffer offset alignment: storage offsets in the program are relative to the buffer address
/// aligned down to it (StorageOffset), which is also where the buffer cache binds the buffer.
template <typename Memory, typename CbufAddress>
u32 EvaluateTextureHandle(const Shader::Info& info, u32 program, Memory& gpu_memory,
                          CbufAddress&& cbuf_address, u64 storage_alignment) {
    const GPUVAddr storage_mask{~(GPUVAddr{(std::max)(storage_alignment, u64{1})} - 1)};
    return Shader::EvaluateHandleProgram(
               info.handle_programs[program],
               [&](u32 index, u32 offset) -> u32 {
                   const GPUVAddr base{cbuf_address(index)};
                   return base != 0 ? gpu_memory.template Read<u32>(base + offset) : 0U;
               },
               [&](u32 binding, u32 offset) -> u32 {
                   if (binding >= info.storage_buffers_descriptors.size()) {
                       return 0U;
                   }
                   const auto& descriptor{info.storage_buffers_descriptors[binding]};
                   const GPUVAddr cbuf{cbuf_address(descriptor.cbuf_index)};
                   if (cbuf == 0) {
                       return 0U;
                   }
                   GPUVAddr base{gpu_memory.template Read<u64>(cbuf + descriptor.cbuf_offset)};
                   if (base != 0 && descriptor.indirect_offset != ~0U) {
                       // The table entry is read at its exact address. Only the buffer address
                       // it holds is aligned down, as the shader does.
                       base = gpu_memory.template Read<u64>(base + descriptor.indirect_offset);
                   }
                   base &= storage_mask;
                   return base != 0 ? gpu_memory.template Read<u32>(base + offset) : 0U;
               })
        .value_or(0U);
}

/// Whether texture descriptor `desc`'s handle word (element `index`) is where a storage buffer of
/// the same program keeps its address. The translation falls back to such a word when the shader
/// loads the handle per invocation, which no draw-time evaluation can give (a light's texture
/// read in a clustered light loop, Ocarina of Time's waterfall). Its low bits then name an
/// unrelated texture, a different one whenever the game moves the buffer: every frame for a
/// double-buffered one, so the surface flickers. Bind no texture for it instead.
template <typename Descriptor>
[[nodiscard]] bool HandleIsStoragePointer(const Shader::Info& info, const Descriptor& desc,
                                          u32 index) {
    if (desc.handle_program != ~0U || desc.has_secondary) {
        return false;
    }
    const u32 offset = desc.cbuf_offset + (index << desc.size_shift);
    return std::ranges::any_of(info.storage_buffers_descriptors, [&](const auto& storage) {
        return storage.cbuf_index == desc.cbuf_index && offset >= storage.cbuf_offset &&
               offset - storage.cbuf_offset < sizeof(u64);
    });
}

/// A texture handle no table holds: the texture cache binds its null view and sampler.
inline constexpr std::pair<u32, u32> NULL_TEXTURE_HANDLE{~0U, ~0U};

} // namespace VideoCommon
