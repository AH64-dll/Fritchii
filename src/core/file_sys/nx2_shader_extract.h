// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Finds the NVN2 (Switch 2) shader programs a game ships, by their content: every RomFS file is
// read, zstd (with the game's dictionaries) and Yaz0 are undone, SARC archives are expanded, and
// BNSH (shader archives, effect bundles, layouts), agl BAHS packs, agl NDSR packs, nn::ndve movie
// files and sead's primitive drawer are parsed; other data is searched for embedded BNSH.
//
// Programs are cut as the game's GPU reads them (control section, then code area of the length
// the control section names at +0x44) and fingerprinted like nxnative's shader_split.py
// (github.com/seedds/nxnative, MIT): SHA-256 of the first 0x48 control bytes with +0x30..+0x33
// zeroed, then the code area; the first 8 bytes. Format notes are from that script and from
// tools/nx2/shader_archive.py.

#pragma once

#include <atomic>
#include <functional>
#include <span>
#include <stop_token>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "core/file_sys/vfs/vfs_types.h"

namespace FileSys::Nx2ShaderExtract {

enum class Stage : u8 { Vertex, TessControl, TessEval, Geometry, Fragment, Compute, Task, Mesh };

[[nodiscard]] const char* StageName(Stage stage) noexcept;

struct Program {
    Stage stage{};
    u64 fingerprint{};
    std::vector<u8> control;
    std::vector<u8> code;
    /// Where it was first found: "/RomFS/path[/member]#KIND@offset[#varN]".
    std::string origin;
};

/// The programs one variation of a shader program draws together (a BNSH variation, a
/// vertex/pixel pair of an NDSR pack): indices into Result::programs.
struct Variation {
    std::string label;
    std::vector<u32> programs;
};

struct Stats {
    u64 files{};
    u64 file_bytes{};          ///< bytes of RomFS files read
    u64 inflated_bytes{};      ///< bytes produced by zstd and Yaz0
    u64 peeked_only{};         ///< compressed files whose content held no container
    u64 containers{};          ///< SARC, BNSH, BAHS, NDSR, NDVE, sead files parsed
    u64 occurrences{};         ///< programs found, before removing duplicates
};

struct Result {
    std::vector<Program> programs;   ///< one per fingerprint
    std::vector<Variation> variations;
    std::vector<std::string> errors; ///< "origin: reason", for data that looked like a container
    Stats stats;
};

struct Progress {
    std::atomic<u64> files_done{};
    u64 files_total{};
};

/// Directories are scanned in order (base game first, then updates and add-ons); a program found
/// in several keeps its first origin. `threads` 0 uses the hardware's.
[[nodiscard]] Result Extract(std::span<const VirtualDir> roots, unsigned threads = 0,
                             Progress* progress = nullptr, std::stop_token stop = {});

/// The fingerprint of one program (see the file comment).
[[nodiscard]] u64 Fingerprint(std::span<const u8> control, std::span<const u8> code);

} // namespace FileSys::Nx2ShaderExtract
