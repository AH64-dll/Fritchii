// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/gpu_split.h"
#include "video_core/macro_hle_oot.h"

namespace Tegra::OotMacros {

namespace {

constexpr u32 MaxInstructions = 1024;

// The replacements in macro_hle_oot.h, by the code hash of the macro they were written from
// (EDEN_NX2_OOT_MACROS_DUMP=1 logs each macro's code and hash).
constexpr std::array macros{
    Macro{0x2ee8ad56465747bfULL, "0xe02 bind constant buffer", Id::E02},
    Macro{0x46c054102001e459ULL, "0xe04 indexed draw", Id::E04},
    Macro{0x999c63d624da37a4ULL, "0xe08 stage program", Id::E08},
    Macro{0xe25bdd1f82c86e49ULL, "0xe10", Id::E10},
    Macro{0xbe42b5cb86694207ULL, "0xe1c select constant buffer", Id::E1C},
    Macro{0xd3f63d16fa69462dULL, "0xe1e", Id::E1E},
    Macro{0x4bdbe220379623d9ULL, "0xe20", Id::E20},
    Macro{0xa5fc8a3a55ec5a41ULL, "0xe28", Id::E28},
    Macro{0xd44bd7403b220b38ULL, "0xe2a render enable", Id::E2A},
    Macro{0x3b26ab7eb3fc9863ULL, "0xe2c conditional render", Id::E2C},
    Macro{0xcecda2c386f441e1ULL, "0xe36 select constant buffer offset", Id::E36},
    Macro{0x7653928eec3315a8ULL, "0xe3a stage disable", Id::E3A},
    Macro{0x053e08f952ce26eeULL, "0xe3e", Id::E3E},
    Macro{0xd7011b3ed6907773ULL, "0xe52 instanced draw", Id::E52},
    Macro{0x4f3ca32a4d22aaf0ULL, "0xe5c", Id::E5C},
    Macro{0xdaba5f7044390cd4ULL, "0xe68 data RAM write", Id::E68},
    Macro{0x0fc20101c0d79a0aULL, "0xe8c", Id::E8C},
    Macro{0x57f1ff45827c7b87ULL, "0xe92", Id::E92},
    Macro{0x9d3c85c8f0bea13aULL, "0xe96 vertex attributes", Id::E96},
    Macro{0x0d3ef03cb86ff29bULL, "0xe9a vertex streams", Id::E9A},
    Macro{0x38b7414725298d00ULL, "0xea0 instanced draw (base in constant buffer)", Id::EA0},
    Macro{0xa53fe0ab08bad81fULL, "0xece", Id::ECE},
    Macro{0x2e38eccae827f4f1ULL, "0xed6", Id::ED6},
};

[[nodiscard]] bool IsHot(u32 method) {
    switch (method) {
    case 0xe02:
    case 0xe1c:
    case 0xe2c:
    case 0xe36:
    case 0xe3a:
        return true;
    default:
        return false;
    }
}

} // Anonymous namespace

bool HasHandlers() {
    return !macros.empty();
}

u64 HashMacro(std::span<const u32> code, u32 start) {
    // FNV-1a over the words, plus the instruction count.
    u64 hash = 0xcbf29ce484222325ULL;
    const auto mix = [&hash](u32 word) {
        for (u32 byte = 0; byte < 4; ++byte) {
            hash ^= (word >> (byte * 8)) & 0xff;
            hash *= 0x100000001b3ULL;
        }
    };
    u32 count = 0;
    bool end_next = false;
    for (u32 pc = start; count < MaxInstructions && size_t{pc} * 3 + 2 < code.size(); ++pc) {
        mix(code[pc * 3]);
        mix(code[pc * 3 + 1]);
        mix(code[pc * 3 + 2]);
        ++count;
        if (end_next) {
            break;
        }
        // Bit 0 of the instruction's high word: end after the next instruction (delay slot).
        end_next = (code[pc * 3 + 2] & 1) != 0;
    }
    mix(count);
    return hash;
}

const Macro* Find(u64 hash) {
    for (const Macro& macro : macros) {
        if (macro.hash == hash) {
            return &macro;
        }
    }
    return nullptr;
}

bool DumpEnabled() {
    static const bool enabled = VideoCommon::Nx2Switch("EDEN_NX2_OOT_MACROS_DUMP", false);
    return enabled;
}

void DumpCode(u32 method, u32 start, std::span<const u32> code) {
    static std::unordered_set<u64> dumped;
    const u64 hash = HashMacro(code, start);
    if (!dumped.insert(hash).second) {
        return;
    }
    u32 count = 0;
    bool end_next = false;
    for (u32 pc = start; count < MaxInstructions && size_t{pc} * 3 + 2 < code.size(); ++pc) {
        ++count;
        if (end_next) {
            break;
        }
        end_next = (code[pc * 3 + 2] & 1) != 0;
    }
    LOG_INFO(HW_GPU, "OoT macro code: method {:#x} start {:#x} hash {:016x} instructions {}",
             method, start, hash, count);
    // Eight instructions (24 words, in code order) a line.
    for (u32 first = 0; first < count; first += 8) {
        std::string line;
        for (u32 i = first; i < std::min(count, first + 8); ++i) {
            const u32 pc = start + i;
            line += fmt::format(" {:08x}{:08x}{:08x}", code[pc * 3 + 2], code[pc * 3 + 1],
                                code[pc * 3]);
        }
        LOG_INFO(HW_GPU, "OoT macro code: {:016x} +{:#x}:{}", hash, first, line);
    }
}

bool WantCallLog(u32 method) {
    if (!IsHot(method)) {
        return false;
    }
    static std::unordered_map<u32, u32> logged;
    return logged[method]++ < 8;
}

void LogCall(u32 method, u32 start, std::span<const u32> parameters, std::span<const u32> writes) {
    std::string params;
    for (const u32 parameter : parameters) {
        params += fmt::format(" {:x}", parameter);
    }
    std::string emitted;
    for (size_t i = 0; i + 1 < writes.size(); i += 2) {
        emitted += fmt::format(" {:x}={:x}", writes[i], writes[i + 1]);
    }
    LOG_INFO(HW_GPU, "OoT macro call: method {:#x} start {:#x} parameters{} writes{}", method,
             start, params, emitted);
}

} // namespace Tegra::OotMacros
