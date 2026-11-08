// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Precompile recipes: a game's precompiled pipeline file (tools/nx2-precompile) without the game's
// code, so it can be shared. A recipe holds, for every pipeline the file prepares, its render state
// and, for each of its shaders, the program's fingerprint (FileSys::Nx2ShaderExtract) and the facts
// its translation asked for (constant buffer values, texture types, storage buffer pointers, ...):
// the shader environment of GenericEnvironment::Serialize without its code and program header.
// Assembling it with the programs extracted from a player's own copy of the game puts the code back,
// lowered as GenericEnvironment::LowerNativeCode lowers it, and gives that player's precompiled file.
//
// Recipe layout: "NX2RCP01", u32 body size, u32 0, then the body as one zstd frame:
//   u32 format (1), u64 title id, u32 game version, u32 cache version of the facts
//   u32 n, then n programs: u64 fingerprint, u8 stage (Shader::Stage), 3 bytes 0
//   u32 n, then n fact sets: u32 program, u8 flags (2: a mesh environment's geometry header is the
//     one in the code; no other flag is valid), 3 bytes 0, u32 size, the facts
//
// A recipe carries none of the game's code or program headers: an environment whose program header
// is not its code's, whose mesh geometry header is neither the code's nor zero, whose layout is
// unknown, or whose facts hold any 32 bytes of its program's code is withheld, with its records.
//   u32 n, then n render states: the graphics key without its six shader hashes (1008 bytes)
//   u32 n, then n records: u8 kind (0 graphics, 1 compute), u8 shader count, u16 0, u32 fact set
//     per shader; graphics: u32 render state, u8 shader of each of the six key slots (0xff none),
//     u8 has_xor, u8 0, then if has_xor six u64 that the slots' hashes are xored with (task and
//     mesh keys add their mesh state)

#pragma once

#include <optional>
#include <span>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "core/file_sys/nx2_shader_extract.h"
#include "shader_recompiler/stage.h"

namespace VideoCommon::Nx2Recipe {

/// A native program as the runtime lowers it: program header, lowered words with scheduling slots,
/// padding and the native address map (GenericEnvironment::LowerNativeCode).
struct Lowered {
    std::vector<u64> code;
    u32 texture_bound{~0U};
    u64 hash{};       ///< the shader hash pipeline keys use
    u64 native_key{};
};

[[nodiscard]] std::optional<Lowered> Lower(Shader::Stage stage, std::span<const u8> control,
                                           std::span<const u8> code, std::string& error);

[[nodiscard]] u64 NativeShaderHash(std::span<const u64> code);

[[nodiscard]] Shader::Stage StageOf(FileSys::Nx2ShaderExtract::Stage stage) noexcept;

struct Report {
    u32 programs{};          ///< distinct programs the recipe names
    u32 programs_missing{};  ///< of them not in the game (another version of it)
    u32 programs_failed{};   ///< found but not lowered
    u32 environments{};
    u32 environments_withheld{}; ///< left out of a recipe as they would carry game data
    u32 records{};
    u32 records_dropped{};   ///< records with a missing or failed program
    u32 states{};
    std::vector<std::string> errors;
};

struct Header {
    u64 title_id{};
    u32 game_version{};
    u32 cache_version{};
};

/// A recipe from a precompiled file and the programs it was built from.
[[nodiscard]] std::vector<u8> MakeRecipe(std::span<const u8> precompiled,
                                         const FileSys::Nx2ShaderExtract::Result& programs,
                                         const Header& header, Report& report,
                                         unsigned threads = 0);

/// The recipe's header, without reading the rest.
[[nodiscard]] std::optional<Header> ReadHeader(std::span<const u8> recipe);

/// 32 bytes of a game's program (dense: at least half not zero) found in a recipe.
struct CodeMatch {
    u64 fingerprint{};
    size_t program_offset{}; ///< into the program's control section, then its code
    size_t recipe_offset{};  ///< into the recipe's uncompressed body
};

/// Any of the game's code a recipe holds. MakeRecipe refuses to make one that holds some; this
/// checks a recipe from elsewhere against a copy of the game before it is shared.
[[nodiscard]] std::vector<CodeMatch> FindGameCode(std::span<const u8> recipe,
                                                  const FileSys::Nx2ShaderExtract::Result& programs,
                                                  unsigned threads = 0);

/// A precompiled file of `cache_version` from a recipe and the programs of the player's game.
[[nodiscard]] std::vector<u8> Assemble(std::span<const u8> recipe,
                                       const FileSys::Nx2ShaderExtract::Result& programs,
                                       u32 cache_version, Report& report, unsigned threads = 0);

} // namespace VideoCommon::Nx2Recipe
