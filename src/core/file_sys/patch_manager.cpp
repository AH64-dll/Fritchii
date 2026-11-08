// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <span>

#include "common/assert.h"
#include "common/hex_util.h"
#include "common/logging.h"
#include "common/settings.h"
#ifndef _WIN32
#include "common/string_util.h"
#endif

#include "core/core.h"
#include "core/file_sys/common_funcs.h"
#include "core/file_sys/content_archive.h"
#include "core/file_sys/control_metadata.h"
#include "core/file_sys/ips_layer.h"
#include "core/file_sys/patch_manager.h"
#include "core/file_sys/registered_cache.h"
#include "core/file_sys/romfs.h"
#include "core/file_sys/vfs/vfs_cached.h"
#include "core/file_sys/vfs/vfs_layered.h"
#include "core/file_sys/vfs/vfs_vector.h"
#include "core/hle/service/filesystem/filesystem.h"
#include "core/hle/service/ns/language.h"
#include "core/hle/service/set/settings_server.h"
#include "core/loader/loader.h"
#include "core/loader/nso.h"
#include "core/memory/cheat_engine.h"

namespace FileSys {
namespace {

constexpr u32 SINGLE_BYTE_MODULUS = 0x100;

constexpr std::array<const char*, 14> EXEFS_FILE_NAMES{
    "main",    "main.npdm", "rtld",    "sdk",     "subsdk0", "subsdk1", "subsdk2",
    "subsdk3", "subsdk4",   "subsdk5", "subsdk6", "subsdk7", "subsdk8", "subsdk9",
};

struct BuiltinPatchRecord {
    size_t module_offset;
    std::span<const u8> bytes;
    std::span<const u8> original;
    bool executable_padding = false;
};

struct BuiltinPatch {
    std::string_view name;
    u64 title_id;
    std::string_view module;
    std::string_view build_id;
    std::span<const BuiltinPatchRecord> records;
    size_t state_offset = 0;
    size_t state_size = 0;
    /// Applied only while this returns true (an env kill switch); null means always.
    bool (*env_enabled)() = nullptr;
};

// EDEN_NX2_<NAME>: unset or empty gives the default, otherwise on unless it starts with '0'.
bool OotEnvSwitch(const char* name, bool default_on) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == 0) {
        return default_on;
    }
    return value[0] != '0';
}

bool OotOcclusionOffEnabled() {
    static const bool enabled = OotEnvSwitch("EDEN_NX2_OOT_OCCLUSION_OFF", true);
    return enabled;
}

bool OotNativeZstdEnabled() {
    static const bool enabled = OotEnvSwitch("EDEN_NX2_OOT_NATIVE_ZSTD", true);
    return enabled;
}

bool OotFastLookupEnabled() {
    static const bool enabled = OotEnvSwitch("EDEN_NX2_OOT_FAST_LOOKUP", true);
    return enabled;
}

constexpr std::array<u8, 4> MAX_RESOLUTION_ORIGINAL{0xE8, 0x41, 0x00, 0x36};
constexpr std::array<u8, 4> MAX_RESOLUTION_BRANCH{0x0F, 0x02, 0x00, 0x14};
constexpr std::array MAX_RESOLUTION_RECORDS{
    BuiltinPatchRecord{0x805BD0, MAX_RESOLUTION_BRANCH, MAX_RESOLUTION_ORIGINAL},
};

constexpr std::array<u8, 4> DYNAMIC_FPS_ORIGINAL{0x28, 0x18, 0x20, 0x1E};
constexpr std::array<u8, 4> DYNAMIC_FPS_CALL{0x4F, 0x7D, 0xD5, 0x94};
constexpr std::array<u8, 120> DYNAMIC_FPS_ROUTINE{
    0x28, 0x18, 0x20, 0x1E, 0xCA, 0x92, 0x00, 0xB0, 0x4A, 0x01, 0x38, 0x91, 0x2B, 0xE0, 0x3B,
    0xD5, 0x4C, 0x01, 0x40, 0xF9, 0x4B, 0x01, 0x00, 0xF9, 0xCC, 0x02, 0x00, 0xB4, 0x6B, 0x01,
    // The original routine caps the measured delta at CNTFRQ / 20 (50 ms). Use /10 (100 ms)
    // so Ocarina keeps real speed down to 10 fps.
    0x0C, 0xCB, 0x0D, 0xE0, 0x3B, 0xD5, 0x4E, 0x01, 0x80, 0xD2, 0xAE, 0x09, 0xCE, 0x9A, 0x7F,
    0x01, 0x0E, 0xEB, 0xCB, 0x81, 0x8B, 0x9A, 0x4C, 0x05, 0x40, 0xF9, 0x4C, 0x00, 0x00, 0xB5,
    0xEC, 0x03, 0x0B, 0xAA, 0x6E, 0x01, 0x0C, 0xCB, 0x8C, 0x0D, 0x8E, 0x8B, 0x4C, 0x05, 0x00,
    0xF9, 0x82, 0x01, 0x23, 0x9E, 0xA3, 0x01, 0x23, 0x9E, 0x42, 0x18, 0x23, 0x1E, 0x42, 0x08,
    0x21, 0x1E, 0xC3, 0x00, 0x00, 0x1C, 0x03, 0x09, 0x23, 0x1E, 0x40, 0x20, 0x23, 0x1E, 0x4D,
    0x00, 0x00, 0x54, 0x48, 0x40, 0x20, 0x1E, 0xC0, 0x03, 0x5F, 0xD6, 0x0A, 0xD7, 0x83, 0x3F,
};
constexpr std::array DYNAMIC_FPS_RECORDS{
    BuiltinPatchRecord{0x6E6CC4, DYNAMIC_FPS_CALL, DYNAMIC_FPS_ORIGINAL},
    BuiltinPatchRecord{0x3C46200, DYNAMIC_FPS_ROUTINE, {}, true},
};

// The render pipeline's settings are mirrored into nodes (main+0x11ac..., per setting: init, then
// compare and store when it changes) after `bl getter; ldrb w8, [x0]`. `mov w8, wzr` there makes
// the setting false. Module offsets: the community IPS files' offsets are these + 0x100.
constexpr std::array<u8, 4> OOT_LDRB_W8_X0{0x08, 0x00, 0x40, 0x39};
constexpr std::array<u8, 4> OOT_MOV_W8_WZR{0xE8, 0x03, 0x1F, 0x2A};

// cLohan's OcclusionOff_ZPrePassAndSamplesQuery: CreateZPrePassOcclusionQuery (getter
// main+0xf5acc0) and CreateSamplesPassedQuery (main+0xf5bf70) off, so the game stops issuing the
// per-object occlusion queries and the conditional renders (macro 0xe2c) they feed.
constexpr std::array OOT_OCCLUSION_OFF_RECORDS{
    BuiltinPatchRecord{0x11ACE68, OOT_MOV_W8_WZR, OOT_LDRB_W8_X0},
    BuiltinPatchRecord{0x11B828C, OOT_MOV_W8_WZR, OOT_LDRB_W8_X0},
    BuiltinPatchRecord{0x11B82A4, OOT_MOV_W8_WZR, OOT_LDRB_W8_X0},
    BuiltinPatchRecord{0x11ACEC8, OOT_MOV_W8_WZR, OOT_LDRB_W8_X0},
    BuiltinPatchRecord{0x11B82DC, OOT_MOV_W8_WZR, OOT_LDRB_W8_X0},
    BuiltinPatchRecord{0x11B82F4, OOT_MOV_W8_WZR, OOT_LDRB_W8_X0},
};

// The three `bl ZSTD_decompressMultiFrame` (main+0xEB380) become svc 0xF0-0xF2, decompressed by
// the host (core/hle/kernel/svc/svc_nx2_zstd.cpp, which needs these exact sites).
constexpr std::array<u8, 4> OOT_ZSTD_CALL_1{0x28, 0xD0, 0xD3, 0x97}; // bl 0xeb380
constexpr std::array<u8, 4> OOT_ZSTD_CALL_2{0x87, 0x6C, 0xC6, 0x97}; // bl 0xeb380
constexpr std::array<u8, 4> OOT_ZSTD_CALL_3{0x9D, 0x08, 0x00, 0x94}; // bl 0xeb380
constexpr std::array<u8, 4> OOT_ZSTD_SVC_F0{0x01, 0x1E, 0x00, 0xD4}; // svc #0xf0
constexpr std::array<u8, 4> OOT_ZSTD_SVC_F1{0x21, 0x1E, 0x00, 0xD4}; // svc #0xf1
constexpr std::array<u8, 4> OOT_ZSTD_SVC_F2{0x41, 0x1E, 0x00, 0xD4}; // svc #0xf2
constexpr std::array OOT_NATIVE_ZSTD_RECORDS{
    BuiltinPatchRecord{0xBF72E0, OOT_ZSTD_SVC_F0, OOT_ZSTD_CALL_1},
    BuiltinPatchRecord{0xF50164, OOT_ZSTD_SVC_F1, OOT_ZSTD_CALL_2},
    BuiltinPatchRecord{0xE910C, OOT_ZSTD_SVC_F2, OOT_ZSTD_CALL_3},
};

// The string-keyed sorted-index lookup at main+0x58EEA0 binary-searches with an inlined strcmp
// (8-way unrolled, two branches per byte). Both hot copies (lower bound main+0x58FC64, upper bound
// main+0x591E24) are rewritten in place as an 8-bytes-per-iteration compare with the same result
// sign, the same 32 MiB limit and no loads that cross into a page the original would not read.
constexpr std::array<u8, 240> OOT_FAST_LOOKUP_A{
    0xEA, 0x03, 0x08, 0xAA, 0x4C, 0x15, 0x40, 0x38, 0x2D, 0x15, 0x40, 0x38, 0x9F, 0x01, 0x0D, 0x6B,
    0xE8, 0x97, 0x9F, 0x1A, 0x08, 0x21, 0x9F, 0x5A, 0x9F, 0x01, 0x00, 0x71, 0x80, 0x11, 0x4D, 0x7A,
    0xE1, 0xF9, 0xFF, 0x54, 0xEB, 0x63, 0x40, 0xB2, 0x7F, 0x21, 0x00, 0xF1, 0x48, 0x21, 0x7D, 0x92,
    0x08, 0x21, 0x7D, 0xD2, 0x04, 0x29, 0x40, 0xFA, 0x28, 0x21, 0x7D, 0x92, 0x08, 0x21, 0x7D, 0xD2,
    0x04, 0x19, 0x40, 0xFA, 0xC0, 0x03, 0x00, 0x54, 0x4C, 0x85, 0x40, 0xF8, 0x2D, 0x85, 0x40, 0xF8,
    0xE8, 0xC3, 0x00, 0xB2, 0x88, 0x01, 0x08, 0xCB, 0x08, 0x01, 0x2C, 0x8A, 0x08, 0xC1, 0x01, 0x92,
    0x8E, 0x01, 0x0D, 0xCA, 0xCE, 0x01, 0x08, 0xAA, 0x6B, 0x21, 0x00, 0xD1, 0x7F, 0x21, 0x00, 0xF1,
    0xC0, 0x29, 0x40, 0xFA, 0x48, 0x21, 0x7D, 0x92, 0x08, 0x21, 0x7D, 0xD2, 0x04, 0x09, 0x40, 0xFA,
    0x28, 0x21, 0x7D, 0x92, 0x08, 0x21, 0x7D, 0xD2, 0x04, 0x19, 0x40, 0xFA, 0xE1, 0xFD, 0xFF, 0x54,
    0xC8, 0x0D, 0xC0, 0xDA, 0x08, 0x11, 0xC0, 0xDA, 0x08, 0x09, 0x7D, 0x92, 0x8C, 0x25, 0xC8, 0x9A,
    0xAD, 0x25, 0xC8, 0x9A, 0x8C, 0x1D, 0x00, 0x12, 0xAD, 0x1D, 0x00, 0x12, 0x9F, 0x01, 0x0D, 0x6B,
    0xE8, 0x97, 0x9F, 0x1A, 0x08, 0x21, 0x9F, 0x5A, 0x2E, 0xF5, 0xFF, 0xB5, 0x6B, 0x01, 0x00, 0xB4,
    0x4C, 0x15, 0x40, 0x38, 0x2D, 0x15, 0x40, 0x38, 0x6B, 0x05, 0x00, 0xD1, 0x9F, 0x01, 0x0D, 0x6B,
    0xE8, 0x97, 0x9F, 0x1A, 0x08, 0x21, 0x9F, 0x5A, 0x9F, 0x01, 0x00, 0x71, 0x80, 0x11, 0x4D, 0x7A,
    0x40, 0xFA, 0xFF, 0x54, 0x9E, 0xFF, 0xFF, 0x17, 0xE8, 0x03, 0x1F, 0x2A, 0x9C, 0xFF, 0xFF, 0x17
};
constexpr std::array<u8, 240> OOT_FAST_LOOKUP_A_ORIGINAL{
    0x29, 0x0D, 0x00, 0x91, 0x0A, 0x0D, 0x00, 0x91, 0x0B, 0x08, 0xA0, 0x52, 0x4C, 0xD1, 0x5F, 0x38,
    0xAC, 0x0D, 0x00, 0x34, 0x2D, 0xD1, 0x5F, 0x38, 0x9F, 0x01, 0x0D, 0x6B, 0x6E, 0x37, 0x9B, 0x5A,
    0x08, 0x01, 0x8E, 0x1A, 0x9F, 0x01, 0x00, 0x71, 0x80, 0x11, 0x4D, 0x7A, 0x81, 0xF9, 0xFF, 0x54,
    0x4C, 0xE1, 0x5F, 0x38, 0x8C, 0x10, 0x00, 0x34, 0x2D, 0xE1, 0x5F, 0x38, 0x9F, 0x01, 0x0D, 0x6B,
    0x6E, 0x37, 0x9B, 0x5A, 0x08, 0x01, 0x8E, 0x1A, 0xAC, 0xF8, 0xFF, 0x34, 0x9F, 0x01, 0x0D, 0x6B,
    0x61, 0xF8, 0xFF, 0x54, 0x4C, 0xF1, 0x5F, 0x38, 0xAC, 0x0F, 0x00, 0x34, 0x2D, 0xF1, 0x5F, 0x38,
    0x9F, 0x01, 0x0D, 0x6B, 0x6E, 0x37, 0x9B, 0x5A, 0x08, 0x01, 0x8E, 0x1A, 0x8C, 0xF7, 0xFF, 0x34,
    0x9F, 0x01, 0x0D, 0x6B, 0x41, 0xF7, 0xFF, 0x54, 0x4C, 0x01, 0x40, 0x39, 0xCC, 0x0E, 0x00, 0x34,
    0x2D, 0x01, 0x40, 0x39, 0x9F, 0x01, 0x0D, 0x6B, 0x6E, 0x37, 0x9B, 0x5A, 0x08, 0x01, 0x8E, 0x1A,
    0x6C, 0xF6, 0xFF, 0x34, 0x9F, 0x01, 0x0D, 0x6B, 0x21, 0xF6, 0xFF, 0x54, 0x4C, 0x05, 0x40, 0x39,
    0xEC, 0x0D, 0x00, 0x34, 0x2D, 0x05, 0x40, 0x39, 0x9F, 0x01, 0x0D, 0x6B, 0x6E, 0x37, 0x9B, 0x5A,
    0x08, 0x01, 0x8E, 0x1A, 0x4C, 0xF5, 0xFF, 0x34, 0x9F, 0x01, 0x0D, 0x6B, 0x01, 0xF5, 0xFF, 0x54,
    0x4C, 0x09, 0x40, 0x39, 0x0C, 0x0D, 0x00, 0x34, 0x2D, 0x09, 0x40, 0x39, 0x9F, 0x01, 0x0D, 0x6B,
    0x6E, 0x37, 0x9B, 0x5A, 0x08, 0x01, 0x8E, 0x1A, 0x2C, 0xF4, 0xFF, 0x34, 0x9F, 0x01, 0x0D, 0x6B,
    0xE1, 0xF3, 0xFF, 0x54, 0x4C, 0x0D, 0x40, 0x39, 0x2C, 0x0C, 0x00, 0x34, 0x2D, 0x0D, 0x40, 0x39
};
constexpr std::array<u8, 244> OOT_FAST_LOOKUP_B{
    0xEA, 0x03, 0x08, 0xAA, 0x4C, 0x15, 0x40, 0x38, 0x2D, 0x15, 0x40, 0x38, 0x9F, 0x01, 0x0D, 0x6B,
    0xE8, 0x97, 0x9F, 0x1A, 0x08, 0x21, 0x9F, 0x5A, 0x9F, 0x01, 0x00, 0x71, 0x80, 0x11, 0x4D, 0x7A,
    0x01, 0x09, 0x00, 0x54, 0xEB, 0x63, 0x40, 0xB2, 0x7F, 0x21, 0x00, 0xF1, 0x48, 0x21, 0x7D, 0x92,
    0x08, 0x21, 0x7D, 0xD2, 0x04, 0x29, 0x40, 0xFA, 0x28, 0x21, 0x7D, 0x92, 0x08, 0x21, 0x7D, 0xD2,
    0x04, 0x19, 0x40, 0xFA, 0xC0, 0x03, 0x00, 0x54, 0x4C, 0x85, 0x40, 0xF8, 0x2D, 0x85, 0x40, 0xF8,
    0xE8, 0xC3, 0x00, 0xB2, 0x88, 0x01, 0x08, 0xCB, 0x08, 0x01, 0x2C, 0x8A, 0x08, 0xC1, 0x01, 0x92,
    0x8E, 0x01, 0x0D, 0xCA, 0xCE, 0x01, 0x08, 0xAA, 0x6B, 0x21, 0x00, 0xD1, 0x7F, 0x21, 0x00, 0xF1,
    0xC0, 0x29, 0x40, 0xFA, 0x48, 0x21, 0x7D, 0x92, 0x08, 0x21, 0x7D, 0xD2, 0x04, 0x09, 0x40, 0xFA,
    0x28, 0x21, 0x7D, 0x92, 0x08, 0x21, 0x7D, 0xD2, 0x04, 0x19, 0x40, 0xFA, 0xE1, 0xFD, 0xFF, 0x54,
    0xC8, 0x0D, 0xC0, 0xDA, 0x08, 0x11, 0xC0, 0xDA, 0x08, 0x09, 0x7D, 0x92, 0x8C, 0x25, 0xC8, 0x9A,
    0xAD, 0x25, 0xC8, 0x9A, 0x8C, 0x1D, 0x00, 0x12, 0xAD, 0x1D, 0x00, 0x12, 0x9F, 0x01, 0x0D, 0x6B,
    0xE8, 0x97, 0x9F, 0x1A, 0x08, 0x21, 0x9F, 0x5A, 0x4E, 0x04, 0x00, 0xB5, 0x6B, 0x01, 0x00, 0xB4,
    0x4C, 0x15, 0x40, 0x38, 0x2D, 0x15, 0x40, 0x38, 0x6B, 0x05, 0x00, 0xD1, 0x9F, 0x01, 0x0D, 0x6B,
    0xE8, 0x97, 0x9F, 0x1A, 0x08, 0x21, 0x9F, 0x5A, 0x9F, 0x01, 0x00, 0x71, 0x80, 0x11, 0x4D, 0x7A,
    0x40, 0xFA, 0xFF, 0x54, 0x17, 0x00, 0x00, 0x14, 0x08, 0x40, 0xA0, 0xD2, 0x48, 0x01, 0x08, 0xCB,
    0x9B, 0xFF, 0xFF, 0x17
};
constexpr std::array<u8, 244> OOT_FAST_LOOKUP_B_ORIGINAL{
    0x29, 0x0D, 0x00, 0x91, 0x0A, 0x0D, 0x00, 0x91, 0x0B, 0x08, 0xA0, 0x52, 0x4C, 0xD1, 0x5F, 0x38,
    0x2C, 0x0E, 0x00, 0x34, 0x2D, 0xD1, 0x5F, 0x38, 0x9F, 0x01, 0x0D, 0x6B, 0xCE, 0x36, 0x96, 0x5A,
    0x08, 0x01, 0x8E, 0x1A, 0x9F, 0x01, 0x00, 0x71, 0x80, 0x11, 0x4D, 0x7A, 0xA1, 0x08, 0x00, 0x54,
    0x4C, 0xE1, 0x5F, 0x38, 0x4C, 0x0D, 0x00, 0x34, 0x2D, 0xE1, 0x5F, 0x38, 0x9F, 0x01, 0x0D, 0x6B,
    0xCE, 0x36, 0x96, 0x5A, 0x08, 0x01, 0x8E, 0x1A, 0xCC, 0x07, 0x00, 0x34, 0x9F, 0x01, 0x0D, 0x6B,
    0x81, 0x07, 0x00, 0x54, 0x4C, 0xF1, 0x5F, 0x38, 0x6C, 0x0C, 0x00, 0x34, 0x2D, 0xF1, 0x5F, 0x38,
    0x9F, 0x01, 0x0D, 0x6B, 0xCE, 0x36, 0x96, 0x5A, 0x08, 0x01, 0x8E, 0x1A, 0xAC, 0x06, 0x00, 0x34,
    0x9F, 0x01, 0x0D, 0x6B, 0x61, 0x06, 0x00, 0x54, 0x4C, 0x01, 0x40, 0x39, 0x8C, 0x0B, 0x00, 0x34,
    0x2D, 0x01, 0x40, 0x39, 0x9F, 0x01, 0x0D, 0x6B, 0xCE, 0x36, 0x96, 0x5A, 0x08, 0x01, 0x8E, 0x1A,
    0x8C, 0x05, 0x00, 0x34, 0x9F, 0x01, 0x0D, 0x6B, 0x41, 0x05, 0x00, 0x54, 0x4C, 0x05, 0x40, 0x39,
    0xAC, 0x0A, 0x00, 0x34, 0x2D, 0x05, 0x40, 0x39, 0x9F, 0x01, 0x0D, 0x6B, 0xCE, 0x36, 0x96, 0x5A,
    0x08, 0x01, 0x8E, 0x1A, 0x6C, 0x04, 0x00, 0x34, 0x9F, 0x01, 0x0D, 0x6B, 0x21, 0x04, 0x00, 0x54,
    0x4C, 0x09, 0x40, 0x39, 0xCC, 0x09, 0x00, 0x34, 0x2D, 0x09, 0x40, 0x39, 0x9F, 0x01, 0x0D, 0x6B,
    0xCE, 0x36, 0x96, 0x5A, 0x08, 0x01, 0x8E, 0x1A, 0x4C, 0x03, 0x00, 0x34, 0x9F, 0x01, 0x0D, 0x6B,
    0x01, 0x03, 0x00, 0x54, 0x4C, 0x0D, 0x40, 0x39, 0xEC, 0x08, 0x00, 0x34, 0x2D, 0x0D, 0x40, 0x39,
    0x9F, 0x01, 0x0D, 0x6B
};
constexpr std::array OOT_FAST_LOOKUP_RECORDS{
    BuiltinPatchRecord{0x58FC64, OOT_FAST_LOOKUP_A, OOT_FAST_LOOKUP_A_ORIGINAL},
    BuiltinPatchRecord{0x591E24, OOT_FAST_LOOKUP_B, OOT_FAST_LOOKUP_B_ORIGINAL},
};

bool OotPollBackoffEnabled() {
    static const bool enabled = OotEnvSwitch("EDEN_NX2_OOT_POLL_BACKOFF", true);
    return enabled;
}

// The resource request thread (main+0xF7A0) rescans its pending requests at once when one is not
// ready (`tbz w8, #0, loop` at main+0x14ABC, w8 = "all done"). That branch now goes through the
// padding at main+0x17088: `svc #0xf3` and `b loop`. The SVC (svc_nx2_oot_poll.cpp) sleeps in
// proportion to the walk only if the walk was a pure re-check; otherwise it returns at once. This
// is the only re-walk-without-waiting in state 3: the other loop-back, after 15 loads issued in one
// walk (main+0x15158), is progress and stays immediate.
constexpr std::array<u8, 4> OOT_POLL_RESCAN_ORIGINAL{0x28, 0x70, 0x05, 0x36}; // tbz w8, #0, 0xf8c0
constexpr std::array<u8, 4> OOT_POLL_RESCAN_BRANCH{0x68, 0x2E, 0x01, 0x36};   // tbz w8, #0, 0x17088
constexpr std::array<u8, 8> OOT_POLL_PADDING_ORIGINAL{};
constexpr std::array<u8, 8> OOT_POLL_BACKOFF_CAVE{
    0x61, 0x1E, 0x00, 0xD4, // svc #0xf3
    0x0D, 0xE2, 0xFF, 0x17, // b 0xf8c0
};
constexpr std::array OOT_POLL_BACKOFF_RECORDS{
    BuiltinPatchRecord{0x14ABC, OOT_POLL_RESCAN_BRANCH, OOT_POLL_RESCAN_ORIGINAL},
    BuiltinPatchRecord{0x17088, OOT_POLL_BACKOFF_CAVE, OOT_POLL_PADDING_ORIGINAL},
};

constexpr std::array BUILTIN_PATCHES{
    // Ocarina of Time takes its "dynamic resolution disabled" path, so the 3D renders at full scale
    // instead of following the GPU-time controller down to its 480p floor.
    BuiltinPatch{
        .name = "Max Resolution (built-in)",
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = MAX_RESOLUTION_RECORDS,
    },
    BuiltinPatch{
        .name = "Dynamic FPS by Dark (built-in)",
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = DYNAMIC_FPS_RECORDS,
        .state_offset = 0x4E9FE00,
        .state_size = 16,
    },
    BuiltinPatch{
        .name = "Occlusion Off (built-in)",
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = OOT_OCCLUSION_OFF_RECORDS,
        .env_enabled = OotOcclusionOffEnabled,
    },
    BuiltinPatch{
        .name = "Native zstd (built-in)",
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = OOT_NATIVE_ZSTD_RECORDS,
        .env_enabled = OotNativeZstdEnabled,
    },
    BuiltinPatch{
        .name = "Fast lookup strcmp (built-in)",
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = OOT_FAST_LOOKUP_RECORDS,
        .env_enabled = OotFastLookupEnabled,
    },
    BuiltinPatch{
        .name = "Resource poll backoff (built-in)",
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = OOT_POLL_BACKOFF_RECORDS,
        .env_enabled = OotPollBackoffEnabled,
    },
};

enum class TitleVersionFormat : u8 {
    ThreeElements, ///< vX.Y.Z
    FourElements,  ///< vX.Y.Z.W
};

std::string FormatTitleVersion(u32 version,
                               TitleVersionFormat format = TitleVersionFormat::ThreeElements) {
    std::array<u8, sizeof(u32)> bytes{};
    bytes[0] = static_cast<u8>(version % SINGLE_BYTE_MODULUS);
    for (std::size_t i = 1; i < bytes.size(); ++i) {
        version /= SINGLE_BYTE_MODULUS;
        bytes[i] = static_cast<u8>(version % SINGLE_BYTE_MODULUS);
    }

    if (format == TitleVersionFormat::FourElements) {
        return fmt::format("v{}.{}.{}.{}", bytes[3], bytes[2], bytes[1], bytes[0]);
    }
    return fmt::format("v{}.{}.{}", bytes[3], bytes[2], bytes[1]);
}

// Returns a directory with name matching name case-insensitive. Returns nullptr if directory
// doesn't have a directory with name.
VirtualDir FindSubdirectoryCaseless(const VirtualDir dir, std::string_view name) {
#ifdef _WIN32
    return dir->GetSubdirectory(name);
#else
    const auto subdirs = dir->GetSubdirectories();
    for (const auto& subdir : subdirs) {
        std::string dir_name = Common::ToLower(subdir->GetName());
        if (dir_name == name) {
            return subdir;
        }
    }

    return nullptr;
#endif
}

std::optional<std::vector<Core::Memory::CheatEntry>> ReadCheatFileFromFolder(
    u64 title_id, const PatchManager::BuildID& build_id_, const VirtualDir& base_path, bool upper) {
    const auto build_id_raw = Common::HexToString(build_id_, upper);
    const auto build_id = build_id_raw.substr(0, sizeof(u64) * 2);
    const auto file = base_path->GetFile(fmt::format("{}.txt", build_id));

    if (file == nullptr) {
        LOG_INFO(Common_Filesystem, "No cheats file found for title_id={:016X}, build_id={}",
                 title_id, build_id);
        return std::nullopt;
    }

    std::vector<u8> data(file->GetSize());
    if (file->Read(data.data(), data.size()) != data.size()) {
        LOG_INFO(Common_Filesystem, "Failed to read cheats file for title_id={:016X}, build_id={}",
                 title_id, build_id);
        return std::nullopt;
    }

    const Core::Memory::TextCheatParser parser;
    return parser.Parse(std::string_view(reinterpret_cast<const char*>(data.data()), data.size()));
}

void AppendCommaIfNotEmpty(std::string& to, std::string_view with) {
    if (to.empty()) {
        to += with;
    } else {
        to += ", ";
        to += with;
    }
}

bool IsDirValidAndNonEmpty(const VirtualDir& dir) {
    return dir != nullptr && (!dir->GetFiles().empty() || !dir->GetSubdirectories().empty());
}

bool IsVersionedExternalUpdateDisabled(const std::vector<std::string>& disabled, u32 version) {
    const std::string disabled_key = fmt::format("Update@{}", version);
    return std::find(disabled.cbegin(), disabled.cend(), disabled_key) != disabled.cend() ||
           std::find(disabled.cbegin(), disabled.cend(), "Update") != disabled.cend();
}

std::string GetUpdateVersionStringFromSlot(const ContentProvider* provider, u64 update_tid) {
    if (provider == nullptr) {
        return {};
    }

    auto control_nca = provider->GetEntry(update_tid, ContentRecordType::Control);
    if (control_nca == nullptr ||
        control_nca->GetStatus() != Loader::ResultStatus::Success) {
        return {};
    }

    const auto romfs = control_nca->GetRomFS();
    if (romfs == nullptr) {
        return {};
    }

    const auto extracted = ExtractRomFS(romfs);
    if (extracted == nullptr) {
        return {};
    }

    auto nacp_file = extracted->GetFile("control.nacp");
    if (nacp_file == nullptr) {
        nacp_file = extracted->GetFile("Control.nacp");
    }
    if (nacp_file == nullptr) {
        return {};
    }

    NACP nacp{nacp_file};
    return nacp.GetVersionString();
}
YUZU_NO_INLINE std::optional<u64> GetParentApplicationId(const ContentProvider* provider, u64 title_id) {
    return provider == nullptr ? std::nullopt : provider->GetParentApplicationId(title_id);
}
} // Anonymous namespace

PatchManager::PatchManager(u64 title_id_,
                           const Service::FileSystem::FileSystemController& fs_controller_,
                           const ContentProvider& content_provider_)
    : title_id{title_id_}, parent_title_id{GetParentApplicationId(std::addressof(content_provider_), title_id_)}, fs_controller{fs_controller_}, content_provider{content_provider_} {}

PatchManager::~PatchManager() = default;

u64 PatchManager::GetTitleID() const {
    return title_id;
}

VirtualDir PatchManager::GetModificationLoadRoot(bool sdmc) const {
    const auto get_root = [&](u64 id) {
        return sdmc ? fs_controller.GetSDMCModificationLoadRoot(id) : fs_controller.GetModificationLoadRoot(id);
    };
    auto root = get_root(title_id);
    if (!parent_title_id) {
        return root;
    }
    std::vector<VirtualDir> roots{std::move(root), get_root(*parent_title_id)};
    std::erase(roots, nullptr);
    return LayeredVfsDirectory::MakeLayeredDirectory(std::move(roots));
}

std::vector<std::string> PatchManager::GetDisabledAddons() const {
    auto disabled = Settings::values.disabled_addons[title_id];
    if (parent_title_id) {
        const auto& shared = Settings::values.disabled_addons[*parent_title_id];
        disabled.insert(disabled.end(), shared.begin(), shared.end());
    }
    return disabled;
}

VirtualDir PatchManager::PatchExeFS(VirtualDir exefs) const {
    LOG_INFO(Loader, "Patching ExeFS for title_id={:016X}", title_id);

    if (exefs == nullptr)
        return exefs;

    const auto disabled = GetDisabledAddons();

    bool update_disabled = true;
    std::optional<u32> enabled_version;
    bool checked_external = false;
    bool checked_manual = false;

    const auto* content_union = static_cast<const ContentProviderUnion*>(&content_provider);
    const auto update_tid = GetUpdateTitleID(title_id);

    if (content_union) {
        // First, check ExternalContentProvider
        const auto* external_provider = content_union->GetExternalProvider();
        if (external_provider) {
            const auto update_versions = external_provider->ListUpdateVersions(update_tid);

            if (!update_versions.empty()) {
                checked_external = true;
                for (const auto& update_entry : update_versions) {
                    if (!IsVersionedExternalUpdateDisabled(disabled, update_entry.version)) {
                        update_disabled = false;
                        enabled_version = update_entry.version;
                        break;
                    }
                }
            }
        }

        // Also check ManualContentProvider (for Android)
        if (!checked_external) {
            const auto* manual_provider = static_cast<const ManualContentProvider*>(
                content_union->GetSlotProvider(ContentProviderUnionSlot::FrontendManual));
            if (manual_provider) {
                const auto manual_update_versions = manual_provider->ListUpdateVersions(update_tid);

                if (!manual_update_versions.empty()) {
                    checked_manual = true;
                    for (const auto& update_entry : manual_update_versions) {
                        if (!IsVersionedExternalUpdateDisabled(disabled, update_entry.version)) {
                            update_disabled = false;
                            enabled_version = update_entry.version;
                            break;
                        }
                    }
                }
            }
        }
    }

    // check for original NAND style
    // Check NAND if: no external updates exist, OR all external updates are disabled
    if (!checked_external && !checked_manual) {
        // Only enable NAND update if it exists AND is not disabled
        // We need to check if an update actually exists in the content provider
        const bool has_nand_update = content_provider.HasEntry(update_tid, ContentRecordType::Program);

        if (has_nand_update) {
            const bool nand_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update (NAND)") != disabled.cend();
            const bool sdmc_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update (SDMC)") != disabled.cend();
            const bool generic_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update") != disabled.cend();

            if (!nand_disabled && !sdmc_disabled && !generic_disabled) {
                update_disabled = false;
            }
        }
    } else if (update_disabled && content_union) {
        const bool nand_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update (NAND)") != disabled.cend();
        const bool sdmc_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update (SDMC)") != disabled.cend();

        if (!nand_disabled || !sdmc_disabled) {
            const auto nand_sdmc_entries = content_union->ListEntriesFilterOrigin(
                std::nullopt, TitleType::Update, ContentRecordType::Program, update_tid);

            for (const auto& [slot, entry] : nand_sdmc_entries) {
                if (slot == ContentProviderUnionSlot::UserNAND ||
                    slot == ContentProviderUnionSlot::SysNAND) {
                    if (!nand_disabled) {
                        update_disabled = false;
                        break;
                    }
                } else if (slot == ContentProviderUnionSlot::SDMC) {
                    if (!sdmc_disabled) {
                        update_disabled = false;
                        break;
                    }
                }
            }
        }
    }

    // Game Updates
    std::unique_ptr<NCA> update = nullptr;

    // If we have a specific enabled version from external provider, use it
    if (enabled_version.has_value() && content_union) {
        const auto* external_provider = content_union->GetExternalProvider();
        if (external_provider) {
            auto file = external_provider->GetEntryForVersion(update_tid, ContentRecordType::Program, *enabled_version);
            if (file != nullptr) {
                update = std::make_unique<NCA>(file);
            }
        }

        // Also try ManualContentProvider
        if (update == nullptr) {
            const auto* manual_provider = static_cast<const ManualContentProvider*>(
                content_union->GetSlotProvider(ContentProviderUnionSlot::FrontendManual));
            if (manual_provider) {
                auto file = manual_provider->GetEntryForVersion(update_tid, ContentRecordType::Program, *enabled_version);
                if (file != nullptr) {
                    update = std::make_unique<NCA>(file);
                }
            }
        }
    }

    // Fallback to regular content provider if no external update was loaded
    if (update == nullptr && !update_disabled) {
        update = content_provider.GetEntry(update_tid, ContentRecordType::Program);
    }

    if (!update_disabled && update != nullptr && update->GetExeFS() != nullptr) {
        LOG_INFO(Loader, "    ExeFS: Update ({}) applied successfully",
                 FormatTitleVersion(content_provider.GetEntryVersion(update_tid).value_or(0)));
        exefs = update->GetExeFS();
    }

    // LayeredExeFS
    const auto load_dir = GetModificationLoadRoot();
    const auto sdmc_load_dir = GetModificationLoadRoot(true);

    std::vector<VirtualDir> patch_dirs = {sdmc_load_dir};
    if (load_dir != nullptr) {
        const auto load_patch_dirs = load_dir->GetSubdirectories();
        patch_dirs.insert(patch_dirs.end(), load_patch_dirs.begin(), load_patch_dirs.end());
    }

    std::sort(patch_dirs.begin(), patch_dirs.end(),
              [](const VirtualDir& l, const VirtualDir& r) { return l->GetName() < r->GetName(); });

    std::vector<VirtualDir> layers;
    layers.reserve(patch_dirs.size() + 1);
    for (const auto& subdir : patch_dirs) {
        if (std::find(disabled.begin(), disabled.end(), subdir->GetName()) != disabled.end())
            continue;

        auto exefs_dir = FindSubdirectoryCaseless(subdir, "exefs");
        if (exefs_dir != nullptr)
            layers.push_back(std::move(exefs_dir));
    }
    layers.push_back(exefs);

    auto layered = LayeredVfsDirectory::MakeLayeredDirectory(std::move(layers));
    if (layered != nullptr) {
        LOG_INFO(Loader, "    ExeFS: LayeredExeFS patches applied successfully");
        exefs = std::move(layered);
    }

    if (Settings::values.dump_exefs) {
        LOG_INFO(Loader, "Dumping ExeFS for title_id={:016X}", title_id);
        const auto dump_dir = fs_controller.GetModificationDumpRoot(title_id);
        if (dump_dir != nullptr) {
            const auto exefs_dir = GetOrCreateDirectoryRelative(dump_dir, "/exefs");
            VfsRawCopyD(exefs, exefs_dir);
        }
    }

    return exefs;
}

std::vector<VirtualFile> PatchManager::CollectPatches(const std::vector<VirtualDir>& patch_dirs, const std::string& build_id) const {
    const auto disabled = GetDisabledAddons();
    const auto nso_build_id = fmt::format("{:0<64}", build_id);

    std::vector<VirtualFile> out;
    out.reserve(patch_dirs.size());
    for (const auto& subdir : patch_dirs) {
        if (std::find(disabled.cbegin(), disabled.cend(), subdir->GetName()) != disabled.cend())
            continue;

        auto exefs_dir = FindSubdirectoryCaseless(subdir, "exefs");
        if (exefs_dir != nullptr) {
            for (const auto& file : exefs_dir->GetFiles()) {
                if (file->GetExtension() == "ips") {
                    auto name = file->GetName();
                    const auto this_build_id = fmt::format("{:0<64}", name.substr(0, name.find('.')));
                    if (nso_build_id == this_build_id)
                        out.push_back(file);
                } else if (file->GetExtension() == "pchtxt") {
                    IPSwitchCompiler compiler{file};
                    const auto this_build_id = Common::HexToString(compiler.GetBuildID());
                    if (nso_build_id == this_build_id)
                        out.push_back(file);
                }
            }
        }
    }
    return out;
}

std::vector<u8> PatchManager::PatchNSO(const std::vector<u8>& nso, const std::string& name) const {
    if (nso.size() < sizeof(Loader::NSOHeader)) {
        return nso;
    }

    Loader::NSOHeader header;
    std::memcpy(&header, nso.data(), sizeof(header));

    if (header.magic != Common::MakeMagic('N', 'S', 'O', '0')) {
        return nso;
    }

    const auto build_id_raw = Common::HexToString(header.build_id);
    const auto build_id = build_id_raw.substr(0, build_id_raw.find_last_not_of('0') + 1);

    if (Settings::values.dump_nso) {
        LOG_INFO(Loader, "Dumping NSO for name={}, build_id={}, title_id={:016X}", name, build_id,
                 title_id);
        const auto dump_dir = fs_controller.GetModificationDumpRoot(title_id);
        if (dump_dir != nullptr) {
            const auto nso_dir = GetOrCreateDirectoryRelative(dump_dir, "/nso");
            const auto file = nso_dir->CreateFile(fmt::format("{}-{}.nso", name, build_id));

            file->Resize(nso.size());
            file->WriteBytes(nso);
        }
    }

    LOG_INFO(Loader, "Patching NSO for name={}, build_id={}", name, build_id);

    const auto load_dir = GetModificationLoadRoot();
    if (load_dir == nullptr) {
        LOG_ERROR(Loader, "Cannot load mods for invalid title_id={:016X}", title_id);
        return nso;
    }

    auto patch_dirs = load_dir->GetSubdirectories();
    std::sort(patch_dirs.begin(), patch_dirs.end(),
              [](const VirtualDir& l, const VirtualDir& r) { return l->GetName() < r->GetName(); });
    const auto patches = CollectPatches(patch_dirs, build_id);

    auto out = nso;
    for (const auto& patch_file : patches) {
        if (patch_file->GetExtension() == "ips") {
            LOG_INFO(Loader, "    - Applying IPS patch from mod \"{}\"",
                     patch_file->GetContainingDirectory()->GetParentDirectory()->GetName());
            const auto patched = PatchIPS(std::make_shared<VectorVfsFile>(out), patch_file);
            if (patched != nullptr)
                out = patched->ReadAllBytes();
        } else if (patch_file->GetExtension() == "pchtxt") {
            LOG_INFO(Loader, "    - Applying IPSwitch patch from mod \"{}\"",
                     patch_file->GetContainingDirectory()->GetParentDirectory()->GetName());
            const IPSwitchCompiler compiler{patch_file};
            const auto patched = compiler.Apply(std::make_shared<VectorVfsFile>(out));
            if (patched != nullptr)
                out = patched->ReadAllBytes();
        }
    }

    const auto disabled = GetDisabledAddons();
    for (const auto& patch : BUILTIN_PATCHES) {
        if (patch.title_id != title_id || patch.module != name) {
            continue;
        }
        if (std::find(disabled.cbegin(), disabled.cend(), patch.name) != disabled.cend()) {
            LOG_INFO(Loader, "    - Skipping \"{}\": disabled in Add-Ons", patch.name);
            continue;
        }
        if (patch.env_enabled != nullptr && !patch.env_enabled()) {
            LOG_INFO(Loader, "    - Skipping \"{}\": off by its EDEN_NX2_ switch", patch.name);
            continue;
        }
        if (patch.build_id != build_id) {
            LOG_INFO(Loader, "    - Skipping \"{}\": build id {} is not {}", patch.name, build_id,
                     patch.build_id);
            continue;
        }
        const auto fits = [](const std::vector<u8>& data, size_t offset, size_t size) {
            return data.size() >= sizeof(Loader::NSOHeader) &&
                   offset <= data.size() - sizeof(Loader::NSOHeader) &&
                   size <= data.size() - sizeof(Loader::NSOHeader) - offset;
        };
        if (patch.state_size != 0) {
            const auto& data = header.segments[2];
            const u64 bss_start = u64{data.location} + data.size;
            const u64 bss_end = bss_start + data.bss_size;
            if (patch.state_offset < bss_start || patch.state_offset > bss_end ||
                patch.state_size > bss_end - patch.state_offset ||
                !fits(nso, patch.state_offset, patch.state_size) ||
                !fits(out, patch.state_offset, patch.state_size)) {
                LOG_WARNING(Loader, "    - Skipping \"{}\": timer state is outside BSS",
                            patch.name);
                continue;
            }
            const size_t offset = sizeof(Loader::NSOHeader) + patch.state_offset;
            if (!std::all_of(nso.begin() + offset, nso.begin() + offset + patch.state_size,
                             [](u8 byte) { return byte == 0; }) ||
                !std::equal(nso.begin() + offset, nso.begin() + offset + patch.state_size,
                            out.begin() + offset)) {
                LOG_WARNING(Loader, "    - Skipping \"{}\": timer state is not zero-filled",
                            patch.name);
                continue;
            }
        }
        bool apply = true;
        for (const auto& record : patch.records) {
            if (!fits(nso, record.module_offset, record.bytes.size()) ||
                !fits(out, record.module_offset, record.bytes.size()) ||
                !fits(nso, record.module_offset, record.original.size())) {
                LOG_WARNING(Loader, "    - Skipping \"{}\": {}+{:#x} is past the module's end",
                            patch.name, name, record.module_offset);
                apply = false;
                break;
            }
            const size_t offset = sizeof(Loader::NSOHeader) + record.module_offset;
            if (record.executable_padding) {
                const auto& text = header.segments[0];
                const u64 text_end = u64{text.location} + text.size;
                const u64 executable_end = (text_end + 0xFFF) & ~u64{0xFFF};
                if (record.module_offset < text_end ||
                    record.module_offset + record.bytes.size() > executable_end ||
                    record.module_offset + record.bytes.size() > header.segments[1].location ||
                    !std::all_of(nso.begin() + offset, nso.begin() + offset + record.bytes.size(),
                                 [](u8 byte) { return byte == 0; })) {
                    LOG_WARNING(Loader,
                                "    - Skipping \"{}\": code cave is not executable padding",
                                patch.name);
                    apply = false;
                    break;
                }
            }
            if (!std::equal(record.original.begin(), record.original.end(), nso.begin() + offset)) {
                LOG_WARNING(Loader, "    - Skipping \"{}\": unexpected original bytes at {}+{:#x}",
                            patch.name, name, record.module_offset);
                apply = false;
                break;
            }
            if (!std::equal(nso.begin() + offset, nso.begin() + offset + record.bytes.size(),
                            out.begin() + offset)) {
                LOG_INFO(Loader, "    - Skipping \"{}\": a user mod already patched {}+{:#x}",
                         patch.name, name, record.module_offset);
                apply = false;
                break;
            }
        }
        if (!apply) {
            continue;
        }
        for (const auto& record : patch.records) {
            std::memcpy(out.data() + sizeof(Loader::NSOHeader) + record.module_offset,
                        record.bytes.data(), record.bytes.size());
        }
        LOG_INFO(Loader, "    - Applying built-in patch \"{}\" ({} records) to {}", patch.name,
                 patch.records.size(), name);
    }

    if (out.size() < sizeof(Loader::NSOHeader)) {
        return nso;
    }

    std::memcpy(out.data(), &header, sizeof(header));
    return out;
}

bool PatchManager::HasNSOPatch(const BuildID& build_id_, std::string_view name) const {
    const auto build_id_raw = Common::HexToString(build_id_);
    const auto build_id = build_id_raw.substr(0, build_id_raw.find_last_not_of('0') + 1);

    LOG_INFO(Loader, "Querying NSO patch existence for build_id={}, name={}", build_id, name);

    if (std::ranges::any_of(BUILTIN_PATCHES, [&](const BuiltinPatch& patch) {
            return patch.title_id == title_id && patch.module == name;
        })) {
        return true;
    }

    const auto load_dir = GetModificationLoadRoot();
    if (load_dir == nullptr) {
        LOG_ERROR(Loader, "Cannot load mods for invalid title_id={:016X}", title_id);
        return false;
    }

    auto patch_dirs = load_dir->GetSubdirectories();
    std::sort(patch_dirs.begin(), patch_dirs.end(),
              [](const VirtualDir& l, const VirtualDir& r) { return l->GetName() < r->GetName(); });

    return !CollectPatches(patch_dirs, build_id).empty();
}

std::vector<Core::Memory::CheatEntry> PatchManager::CreateCheatList(const BuildID& build_id_) const {
    const auto load_dir = GetModificationLoadRoot();
    if (load_dir == nullptr) {
        LOG_ERROR(Loader, "Cannot load mods for invalid title_id={:016X}", title_id);
        return {};
    }

    const auto disabled = GetDisabledAddons();
    auto patch_dirs = load_dir->GetSubdirectories();
    std::sort(patch_dirs.begin(), patch_dirs.end(), [](auto const& l, auto const& r) { return l->GetName() < r->GetName(); });

    // <mod dir> / <folder> / cheats / <build id>.txt
    std::vector<Core::Memory::CheatEntry> out;
    for (const auto& subdir : patch_dirs) {
        if (std::find(disabled.cbegin(), disabled.cend(), subdir->GetName()) == disabled.cend()) {
            if (auto cheats_dir = FindSubdirectoryCaseless(subdir, "cheats"); cheats_dir != nullptr) {
                if (auto const res = ReadCheatFileFromFolder(title_id, build_id_, cheats_dir, true))
                    std::copy(res->begin(), res->end(), std::back_inserter(out));
                if (auto const res = ReadCheatFileFromFolder(title_id, build_id_, cheats_dir, false))
                    std::copy(res->begin(), res->end(), std::back_inserter(out));
            }
        }
    }
    // Uncareless user-friendly loading of patches (must start with 'cheat_')
    // <mod dir> / <cheat file>.txt
    for (auto const& f : load_dir->GetFiles()) {
        auto const name = f->GetName();
        if (name.starts_with("cheat_") && std::find(disabled.cbegin(), disabled.cend(), name) == disabled.cend()) {
            std::vector<u8> data(f->GetSize());
            if (f->Read(data.data(), data.size()) == data.size()) {
                const Core::Memory::TextCheatParser parser;
                auto const res = parser.Parse(std::string_view(reinterpret_cast<const char*>(data.data()), data.size()));
                std::copy(res.begin(), res.end(), std::back_inserter(out));
            } else {
                LOG_INFO(Common_Filesystem, "Failed to read cheats file for title_id={:016X}", title_id);
            }
        }
    }
    return out;
}

void PatchManager::ApplyLayeredFS(VirtualFile& romfs, ContentRecordType type) const {
    const auto load_dir = GetModificationLoadRoot();
    const auto sdmc_load_dir = GetModificationLoadRoot(true);
    if ((type != ContentRecordType::Program && type != ContentRecordType::Data &&
         type != ContentRecordType::HtmlDocument) ||
        (load_dir == nullptr && sdmc_load_dir == nullptr)) {
        return;
    }

    const auto disabled = GetDisabledAddons();
    std::vector<VirtualDir> patch_dirs = load_dir->GetSubdirectories();
    if (std::find(disabled.cbegin(), disabled.cend(), "SDMC") == disabled.cend()) {
        patch_dirs.push_back(sdmc_load_dir);
    }
    std::sort(patch_dirs.begin(), patch_dirs.end(),
              [](const VirtualDir& l, const VirtualDir& r) { return l->GetName() < r->GetName(); });

    std::vector<VirtualDir> layers;
    std::vector<VirtualDir> layers_ext;
    layers.reserve(patch_dirs.size() + 1);
    layers_ext.reserve(patch_dirs.size() + 1);
    for (const auto& subdir : patch_dirs) {
        if (std::find(disabled.cbegin(), disabled.cend(), subdir->GetName()) != disabled.cend()) {
            continue;
        }

        auto romfs_dir = FindSubdirectoryCaseless(subdir, "romfs");
        if (romfs_dir != nullptr)
            layers.emplace_back(std::make_shared<CachedVfsDirectory>(std::move(romfs_dir)));

        // Support for romfslite introduced in Atmosphere 1.9.5
        auto romfslite_dir = FindSubdirectoryCaseless(subdir, "romfslite");
        if (romfslite_dir != nullptr)
            layers.emplace_back(std::make_shared<CachedVfsDirectory>(std::move(romfslite_dir)));

        auto ext_dir = FindSubdirectoryCaseless(subdir, "romfs_ext");
        if (ext_dir != nullptr)
            layers_ext.emplace_back(std::make_shared<CachedVfsDirectory>(std::move(ext_dir)));

        if (type == ContentRecordType::HtmlDocument) {
            auto manual_dir = FindSubdirectoryCaseless(subdir, "manual_html");
            if (manual_dir != nullptr)
                layers.emplace_back(std::make_shared<CachedVfsDirectory>(std::move(manual_dir)));
        }
    }

    // When there are no layers to apply, return early as there is no need to rebuild the RomFS
    if (layers.empty() && layers_ext.empty()) {
        return;
    }

    auto extracted = ExtractRomFS(romfs);
    if (extracted == nullptr) {
        return;
    }

    layers.emplace_back(std::move(extracted));

    auto layered = LayeredVfsDirectory::MakeLayeredDirectory(std::move(layers));
    auto layered_ext = LayeredVfsDirectory::MakeLayeredDirectory(std::move(layers_ext));
    if (layered == nullptr && layered_ext == nullptr) {
        return;
    }

    auto packed = CreateRomFS(std::move(layered), std::move(layered_ext));
    if (packed == nullptr) {
        return;
    }

    LOG_INFO(Loader, "    RomFS: LayeredFS patches applied successfully");
    romfs = std::move(packed);
}

VirtualFile PatchManager::PatchRomFS(const NCA* base_nca, VirtualFile base_romfs,
                                     ContentRecordType type, VirtualFile packed_update_raw,
                                     bool apply_layeredfs) const {
    const auto log_string = fmt::format("Patching RomFS for title_id={:016X}, type={:02X}",
                                        title_id, static_cast<u8>(type));
    if (type == ContentRecordType::Program || type == ContentRecordType::Data) {
        LOG_INFO(Loader, "{}", log_string);
    } else {
        LOG_DEBUG(Loader, "{}", log_string);
    }

    auto romfs = base_romfs;

    // Game Updates
    const auto update_tid = GetUpdateTitleID(title_id);
    const auto disabled = GetDisabledAddons();

    bool update_disabled = true;
    std::optional<u32> enabled_version;
    VirtualFile update_raw = nullptr;
    bool checked_external = false;
    bool checked_manual = false;

    const auto* content_union = static_cast<const ContentProviderUnion*>(&content_provider);
    if (content_union) {
        // First, check ExternalContentProvider
        const auto* external_provider = content_union->GetExternalProvider();
        if (external_provider) {
            const auto update_versions = external_provider->ListUpdateVersions(update_tid);

            if (!update_versions.empty()) {
                checked_external = true;
                for (const auto& update_entry : update_versions) {
                    if (!IsVersionedExternalUpdateDisabled(disabled, update_entry.version)) {
                        update_disabled = false;
                        enabled_version = update_entry.version;
                        update_raw = external_provider->GetEntryForVersion(update_tid, type, update_entry.version);
                        break;
                    }
                }
            }
        }

        if (!checked_external) {
            const auto* manual_provider = static_cast<const ManualContentProvider*>(
                content_union->GetSlotProvider(ContentProviderUnionSlot::FrontendManual));
            if (manual_provider) {
                const auto manual_update_versions = manual_provider->ListUpdateVersions(update_tid);

                if (!manual_update_versions.empty()) {
                    checked_manual = true;
                    for (const auto& update_entry : manual_update_versions) {
                        if (!IsVersionedExternalUpdateDisabled(disabled, update_entry.version)) {
                            update_disabled = false;
                            enabled_version = update_entry.version;
                            update_raw = manual_provider->GetEntryForVersion(update_tid, type, update_entry.version);
                            break;
                        }
                    }
                }
            }
        }
    }

    if (!checked_external && !checked_manual) {
        const bool nand_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update (NAND)") != disabled.cend();
        const bool sdmc_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update (SDMC)") != disabled.cend();
        const bool generic_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update") != disabled.cend();

        if (!nand_disabled && !sdmc_disabled && !generic_disabled) {
            update_disabled = false;
        }
        if (!update_disabled) {
            update_raw = content_provider.GetEntryRaw(update_tid, type);
        }
    } else if (update_disabled && content_union) {
        const bool nand_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update (NAND)") != disabled.cend();
        const bool sdmc_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update (SDMC)") != disabled.cend();

        if (!nand_disabled || !sdmc_disabled) {
            const auto nand_sdmc_entries = content_union->ListEntriesFilterOrigin(
                std::nullopt, TitleType::Update, type, update_tid);

            for (const auto& [slot, entry] : nand_sdmc_entries) {
                if (slot == ContentProviderUnionSlot::UserNAND ||
                    slot == ContentProviderUnionSlot::SysNAND) {
                    if (!nand_disabled) {
                        update_disabled = false;
                        update_raw = content_provider.GetEntryRaw(update_tid, type);
                        break;
                    }
                } else if (slot == ContentProviderUnionSlot::SDMC) {
                    if (!sdmc_disabled) {
                        update_disabled = false;
                        update_raw = content_provider.GetEntryRaw(update_tid, type);
                        break;
                    }
                }
            }
        }
    }

    if (!update_disabled && update_raw != nullptr && base_nca != nullptr) {
        const auto new_nca = std::make_shared<NCA>(update_raw, base_nca);
        if (new_nca->GetStatus() == Loader::ResultStatus::Success &&
            new_nca->GetRomFS() != nullptr) {
            LOG_INFO(Loader, "    RomFS: Update ({}) applied successfully",
                     enabled_version.has_value() ? FormatTitleVersion(*enabled_version) :
                     FormatTitleVersion(content_provider.GetEntryVersion(update_tid).value_or(0)));
            romfs = new_nca->GetRomFS();
        }
    } else if (!update_disabled && packed_update_raw != nullptr && base_nca != nullptr) {
        const auto new_nca = std::make_shared<NCA>(packed_update_raw, base_nca);
        if (new_nca->GetStatus() == Loader::ResultStatus::Success &&
            new_nca->GetRomFS() != nullptr) {
            LOG_INFO(Loader, "    RomFS: Update (PACKED) applied successfully");
            romfs = new_nca->GetRomFS();
        }
    }

    // LayeredFS
    if (apply_layeredfs) {
        ApplyLayeredFS(romfs, type);
    }

    return romfs;
}

std::vector<Patch> PatchManager::GetPatches(VirtualFile update_raw) const {
    if (title_id == 0) {
        return {};
    }

    std::vector<Patch> out;
    const auto& disabled = Settings::values.disabled_addons[title_id];

    // Game Updates
    const auto update_tid = GetUpdateTitleID(title_id);

    std::vector<Patch> external_update_patches;

    const auto* content_union = static_cast<const ContentProviderUnion*>(&content_provider);

    if (content_union) {
        // First, check ExternalContentProvider for updates
        const auto* external_provider = content_union->GetExternalProvider();
        if (external_provider) {
            const auto update_versions = external_provider->ListUpdateVersions(update_tid);

            for (const auto& update_entry : update_versions) {
                std::string version_str = update_entry.version_string;
                if (version_str.empty()) {
                    version_str = FormatTitleVersion(update_entry.version);
                }

                const auto update_disabled =
                    IsVersionedExternalUpdateDisabled(disabled, update_entry.version);

                Patch update_patch = {.enabled = !update_disabled,
                                      .name = "Update",
                                      .version = version_str,
                                      .type = PatchType::Update,
                                      .program_id = title_id,
                                      .title_id = update_tid,
                                      .source = PatchSource::External,
                                      .numeric_version = update_entry.version};

                external_update_patches.push_back(update_patch);
            }
        }

        const auto* manual_provider = static_cast<const ManualContentProvider*>(
            content_union->GetSlotProvider(ContentProviderUnionSlot::FrontendManual));
        if (manual_provider && external_update_patches.empty()) {
            const auto manual_update_versions = manual_provider->ListUpdateVersions(update_tid);

            for (const auto& update_entry : manual_update_versions) {
                std::string version_str = update_entry.version_string;
                if (version_str.empty()) {
                    version_str = FormatTitleVersion(update_entry.version);
                }

                const auto update_disabled =
                    IsVersionedExternalUpdateDisabled(disabled, update_entry.version);


                Patch update_patch = {.enabled = !update_disabled,
                                      .name = "Update",
                                      .version = version_str,
                                      .type = PatchType::Update,
                                      .program_id = title_id,
                                      .title_id = update_tid,
                                      .source = PatchSource::External,
                                      .numeric_version = update_entry.version};

                external_update_patches.push_back(update_patch);
            }
        }

        if (external_update_patches.size() > 1) {
            bool found_enabled = false;
            for (auto& patch : external_update_patches) {
                if (patch.enabled) {
                    if (found_enabled) {
                        patch.enabled = false;
                    } else {
                        found_enabled = true;
                    }
                }
            }
        }

        for (auto& patch : external_update_patches) {
            out.push_back(std::move(patch));
        }

        const auto all_updates = content_union->ListEntriesFilterOrigin(
            std::nullopt, std::nullopt, ContentRecordType::Program, update_tid);

        for (const auto& [slot, entry] : all_updates) {
            (void)entry;
            if (slot == ContentProviderUnionSlot::External ||
                slot == ContentProviderUnionSlot::FrontendManual) {
                continue;
            }

            PatchSource source_type = PatchSource::Unknown;
            std::string source_suffix;

            switch (slot) {
                case ContentProviderUnionSlot::UserNAND:
                case ContentProviderUnionSlot::SysNAND:
                    source_type = PatchSource::NAND;
                    source_suffix = " (NAND)";
                    break;
                case ContentProviderUnionSlot::SDMC:
                    source_type = PatchSource::SDMC;
                    source_suffix = " (SDMC)";
                    break;
                default:
                    break;
            }

            std::string version_str;
            u32 numeric_ver = 0;
            const auto* slot_provider = content_union->GetSlotProvider(slot);
            version_str = GetUpdateVersionStringFromSlot(slot_provider, update_tid);

            if (slot_provider != nullptr) {
                const auto slot_ver = slot_provider->GetEntryVersion(update_tid);
                if (slot_ver.has_value()) {
                    numeric_ver = *slot_ver;
                    if (version_str.empty() && numeric_ver != 0) {
                        version_str = FormatTitleVersion(numeric_ver);
                    }
                }
            }

            std::string patch_name = "Update" + source_suffix;

            bool update_disabled =
                std::find(disabled.cbegin(), disabled.cend(), patch_name) != disabled.cend();

            Patch update_patch = {.enabled = !update_disabled,
                                  .name = patch_name,
                                  .version = version_str,
                                  .type = PatchType::Update,
                                  .program_id = title_id,
                                  .title_id = update_tid,
                                  .source = source_type,
                                  .numeric_version = numeric_ver};

            out.push_back(update_patch);
        }
    } else {
        PatchManager update{update_tid, fs_controller, content_provider};
        const auto metadata = update.GetControlMetadata();
        const auto& nacp = metadata.first;

        bool update_disabled =
            std::find(disabled.cbegin(), disabled.cend(), "Update") != disabled.cend();
        Patch update_patch = {.enabled = !update_disabled,
                              .name = "Update",
                              .version = "",
                              .type = PatchType::Update,
                              .program_id = title_id,
                              .title_id = title_id,
                              .source = PatchSource::Unknown,
                              .numeric_version = 0};

        if (nacp != nullptr) {
            update_patch.version = nacp->GetVersionString();
            out.push_back(update_patch);
        } else {
            if (content_provider.HasEntry(update_tid, ContentRecordType::Program)) {
                const auto meta_ver = content_provider.GetEntryVersion(update_tid);
                if (meta_ver.value_or(0) == 0) {
                    out.push_back(update_patch);
                } else {
                    update_patch.version = FormatTitleVersion(*meta_ver);
                    update_patch.numeric_version = *meta_ver;
                    out.push_back(update_patch);
                }
            } else if (update_raw != nullptr) {
                update_patch.version = "PACKED";
                update_patch.source = PatchSource::Packed;
                out.push_back(update_patch);
            }
        }
    }

    // General Mods (LayeredFS and IPS)
    const auto mod_dir = fs_controller.GetModificationLoadRoot(title_id);
    if (mod_dir != nullptr) {
        for (auto const& f : mod_dir->GetFiles())
            if (auto const name = f->GetName(); name.starts_with("cheat_")) {
                auto const mod_disabled = std::find(disabled.begin(), disabled.end(), name) != disabled.end();
                out.push_back({
                    .enabled = !mod_disabled,
                    .name = name,
                    .version = "Cheats",
                    .type = PatchType::Mod,
                    .program_id = title_id,
                    .title_id = title_id,
                    .source = PatchSource::Unknown,
                    .location = f->GetFullPath(),
                });
            }

        for (const auto& mod : mod_dir->GetSubdirectories()) {
            std::string types;

            const auto exefs_dir = FindSubdirectoryCaseless(mod, "exefs");
            if (IsDirValidAndNonEmpty(exefs_dir)) {
                bool ips = false;
                bool ipswitch = false;
                bool layeredfs = false;

                for (const auto& file : exefs_dir->GetFiles()) {
                    if (file->GetExtension() == "ips") {
                        ips = true;
                    } else if (file->GetExtension() == "pchtxt") {
                        ipswitch = true;
                    } else if (std::find(EXEFS_FILE_NAMES.begin(), EXEFS_FILE_NAMES.end(),
                                         file->GetName()) != EXEFS_FILE_NAMES.end()) {
                        layeredfs = true;
                                         }
                }

                if (ips)
                    AppendCommaIfNotEmpty(types, "IPS");
                if (ipswitch)
                    AppendCommaIfNotEmpty(types, "IPSwitch");
                if (layeredfs)
                    AppendCommaIfNotEmpty(types, "LayeredExeFS");
            }
            if (IsDirValidAndNonEmpty(FindSubdirectoryCaseless(mod, "romfs")) ||
                IsDirValidAndNonEmpty(FindSubdirectoryCaseless(mod, "romfslite")))
                AppendCommaIfNotEmpty(types, "LayeredFS");
            if (IsDirValidAndNonEmpty(FindSubdirectoryCaseless(mod, "romfs_ext")))
                AppendCommaIfNotEmpty(types, "ExtLayeredFS");
            if (IsDirValidAndNonEmpty(FindSubdirectoryCaseless(mod, "cheats")))
                AppendCommaIfNotEmpty(types, "Cheats");

            if (types.empty())
                continue;

            const auto mod_disabled = std::find(disabled.begin(), disabled.end(), mod->GetName()) != disabled.end();
            out.push_back({.enabled = !mod_disabled,
                           .name = mod->GetName(),
                           .version = types,
                           .type = PatchType::Mod,
                           .program_id = title_id,
                           .title_id = title_id,
                           .source = PatchSource::Unknown,
                           .location = mod->GetFullPath()});
        }
    }

    // SDMC mod directory (RomFS LayeredFS)
    const auto sdmc_mod_dir = fs_controller.GetSDMCModificationLoadRoot(title_id);
    if (sdmc_mod_dir != nullptr) {
        std::string types;
        if (IsDirValidAndNonEmpty(FindSubdirectoryCaseless(sdmc_mod_dir, "exefs")))
            AppendCommaIfNotEmpty(types, "LayeredExeFS");
        if (IsDirValidAndNonEmpty(FindSubdirectoryCaseless(sdmc_mod_dir, "romfs")) ||
            IsDirValidAndNonEmpty(FindSubdirectoryCaseless(sdmc_mod_dir, "romfslite")))
            AppendCommaIfNotEmpty(types, "LayeredFS");
        if (IsDirValidAndNonEmpty(FindSubdirectoryCaseless(sdmc_mod_dir, "romfs_ext")))
            AppendCommaIfNotEmpty(types, "ExtLayeredFS");

        if (!types.empty()) {
            const auto mod_disabled =
                std::find(disabled.begin(), disabled.end(), "SDMC") != disabled.end();
            out.push_back({.enabled = !mod_disabled,
                           .name = "SDMC",
                           .version = types,
                           .type = PatchType::Mod,
                           .program_id = title_id,
                           .title_id = title_id,
                           .source = PatchSource::Unknown});
        }
    }

    for (const auto& patch : BUILTIN_PATCHES) {
        if (patch.title_id != title_id) {
            continue;
        }
        const auto patch_disabled =
            std::find(disabled.begin(), disabled.end(), patch.name) != disabled.end();
        out.push_back({.enabled = !patch_disabled,
                       .name = std::string{patch.name},
                       .version = "IPS",
                       .type = PatchType::Mod,
                       .program_id = title_id,
                       .title_id = title_id,
                       .source = PatchSource::Unknown});
    }

    // DLC
    std::vector<ContentProviderEntry> dlc_match;
    bool has_external_dlc = false;
    bool has_nand_dlc = false;
    bool has_sdmc_dlc = false;
    bool has_other_dlc = false;
    const auto dlc_entries_with_origin =
        content_union->ListEntriesFilterOrigin(std::nullopt, TitleType::AOC, ContentRecordType::Data);

    dlc_match.reserve(dlc_entries_with_origin.size());
    for (const auto& [slot, entry] : dlc_entries_with_origin) {
        const auto base_tid = GetBaseTitleID(entry.title_id);
        const bool matches_base = base_tid == title_id;
        if (!matches_base) {
            LOG_DEBUG(Loader, "DLC {:016X} base {:016X} doesn't match title {:016X}",
                      entry.title_id, base_tid, title_id);
            continue;
        }

        const auto* slot_provider = content_union->GetSlotProvider(slot);
        if (slot_provider == nullptr) {
            continue;
        }

        auto nca = slot_provider->GetEntry(entry);
        if (!nca) {
            LOG_DEBUG(Loader, "Failed to get NCA for DLC {:016X}", entry.title_id);
            continue;
        }

        const auto status = nca->GetStatus();
        if (status != Loader::ResultStatus::Success) {
            LOG_DEBUG(Loader, "DLC {:016X} NCA has status {}", entry.title_id,
                      static_cast<int>(status));
            continue;
        }

        switch (slot) {
            case ContentProviderUnionSlot::External:
            case ContentProviderUnionSlot::FrontendManual:
                has_external_dlc = true;
                break;
            case ContentProviderUnionSlot::UserNAND:
            case ContentProviderUnionSlot::SysNAND:
                has_nand_dlc = true;
                break;
            case ContentProviderUnionSlot::SDMC:
                has_sdmc_dlc = true;
                break;
            default:
                has_other_dlc = true;
                break;
        }
        dlc_match.push_back(entry);
    }

    if (!dlc_match.empty()) {
        // Ensure sorted so DLC IDs show in order.
        std::sort(dlc_match.begin(), dlc_match.end());

        std::string list;
        for (size_t i = 0; i < dlc_match.size() - 1; ++i)
            list += fmt::format("{}, ", dlc_match[i].title_id & 0x7FF);

        list += fmt::format("{}", dlc_match.back().title_id & 0x7FF);

        const auto dlc_disabled =
            std::find(disabled.begin(), disabled.end(), "DLC") != disabled.end();
        PatchSource dlc_source = PatchSource::Unknown;
        if (has_external_dlc && !has_nand_dlc && !has_sdmc_dlc && !has_other_dlc) {
            dlc_source = PatchSource::External;
        } else if (has_nand_dlc && !has_external_dlc && !has_sdmc_dlc && !has_other_dlc) {
            dlc_source = PatchSource::NAND;
        } else if (has_sdmc_dlc && !has_external_dlc && !has_nand_dlc && !has_other_dlc) {
            dlc_source = PatchSource::SDMC;
        }

        out.push_back({.enabled = !dlc_disabled,
                       .name = "DLC",
                       .version = std::move(list),
                       .type = PatchType::DLC,
                       .program_id = title_id,
                       .title_id = dlc_match.back().title_id,
                       .source = dlc_source});
    }

    return out;
}

std::optional<u32> PatchManager::GetGameVersion() const {
    const auto update_tid = GetUpdateTitleID(title_id);
    if (content_provider.HasEntry(update_tid, ContentRecordType::Program)) {
        const auto version = content_provider.GetEntryVersion(update_tid);
        return version || !parent_title_id ? version : content_provider.GetEntryVersion(GetUpdateTitleID(*parent_title_id));
    }

    const auto version = content_provider.GetEntryVersion(title_id);
    return version || !parent_title_id ? version : content_provider.GetEntryVersion(*parent_title_id);
}

PatchManager::Metadata PatchManager::GetControlMetadata() const {
    const auto base_control_nca = content_provider.GetEntry(title_id, ContentRecordType::Control);
    if (base_control_nca == nullptr) {
        if (parent_title_id) {
            const auto control_id = content_provider.HasEntry(*parent_title_id, ContentRecordType::Control) ? *parent_title_id : GetUpdateTitleID(*parent_title_id);
            const PatchManager parent{control_id, fs_controller, content_provider};
            return parent.GetControlMetadata();
        }
        return {};
    }

    return ParseControlNCA(*base_control_nca);
}

PatchManager::Metadata PatchManager::ParseControlNCA(const NCA& nca) const {
    const auto base_romfs = nca.GetRomFS();
    if (base_romfs == nullptr) {
        return {};
    }

    const auto romfs = PatchRomFS(&nca, base_romfs, ContentRecordType::Control);
    if (romfs == nullptr) {
        return {};
    }

    const auto extracted = ExtractRomFS(romfs);
    if (extracted == nullptr) {
        return {};
    }

    auto nacp_file = extracted->GetFile("control.nacp");
    if (nacp_file == nullptr) {
        nacp_file = extracted->GetFile("Control.nacp");
    }

    auto nacp = nacp_file == nullptr ? nullptr : std::make_unique<NACP>(nacp_file);

    // Get language code from settings
    const auto language_code = Service::Set::GetLanguageCodeFromIndex(
        static_cast<u32>(Settings::values.language_index.GetValue()));

    // Convert to application language and get priority list
    const auto application_language =
        Service::NS::ConvertToApplicationLanguage(language_code)
            .value_or(Service::NS::ApplicationLanguage::AmericanEnglish);
    const auto language_priority_list =
        Service::NS::GetApplicationLanguagePriorityList(application_language);

    // Convert to language names
    auto priority_language_names = FileSys::LANGUAGE_NAMES; // Copy
    if (language_priority_list) {
        for (size_t i = 0; i < priority_language_names.size(); ++i) {
            // Relies on FileSys::LANGUAGE_NAMES being in the same order as
            // Service::NS::ApplicationLanguage
            const auto language_index = static_cast<u8>(language_priority_list->at(i));

            if (language_index < FileSys::LANGUAGE_NAMES.size()) {
                priority_language_names[i] = FileSys::LANGUAGE_NAMES[language_index];
            } else {
                // Not a catastrophe, unlikely to happen
                LOG_WARNING(Loader, "Invalid language index {}", language_index);
            }
        }
    }

    // Get first matching icon
    VirtualFile icon_file;
    for (const auto& language : priority_language_names) {
        icon_file = extracted->GetFile(std::string("icon_").append(language).append(".dat"));
        if (icon_file != nullptr) {
            break;
        }
    }

    return {std::move(nacp), icon_file};
}

[[nodiscard]] PatchManager::Metadata PatchManager::GetMetadataFromBaseOrUpdate(Core::System& system, u64 application_id) noexcept {
    const FileSys::PatchManager pm{application_id, system.GetFileSystemController(), system.GetContentProvider()};
    auto metadata = pm.GetControlMetadata();
    if (metadata.first != nullptr)
        return metadata;
    const FileSys::PatchManager pm_update{FileSys::GetUpdateTitleID(application_id), system.GetFileSystemController(), system.GetContentProvider()};
    return pm_update.GetControlMetadata();
}

} // namespace FileSys
