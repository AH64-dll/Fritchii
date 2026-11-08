// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Shader::Info to bytes and back, field by field (padding never written, so two equal infos give
// equal bytes). Used to keep a pipeline's shader infos on disk: a precompiled pipeline can be
// drawn with them before its programs are translated.

#pragma once

#include <span>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/shader_info.h"

namespace Shader {

/// Appends `info` to `out`.
void SerializeInfo(const Info& info, std::vector<u8>& out);

/// Reads an info written by SerializeInfo from the front of `in`, advancing it; false when `in`
/// is too short or holds more descriptors than an info has room for.
[[nodiscard]] bool DeserializeInfo(std::span<const u8>& in, Info& info);

} // namespace Shader
