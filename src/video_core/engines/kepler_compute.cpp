// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <bit>
#include <bitset>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "common/assert.h"
#include "common/dkb_frame_trace.h"
#include "common/logging.h"
#include "common/settings.h"
#include "shader_recompiler/exception.h"
#include "core/core.h"
#include "video_core/gpu_split.h"
#include "video_core/fire_emblem_upscale.h"
#include "video_core/nx2_profile.h"
#include "video_core/nx2.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/game_dlss_status.h"
#include "video_core/memory_manager.h"
#include "video_core/native_query_trace.h"
#include "video_core/oot_network.h"
#ifdef ARCHITECTURE_x86_64
#include "video_core/macro_turing_jit.h"
#endif
#include "video_core/rasterizer_interface.h"
#include "video_core/textures/decoders.h"

namespace Tegra::Engines {

namespace {

// EDEN_NX2_QMD_PREDICT: the GPU thread split's front runs its own copy of the QMD scheduler. All
// channels' native compute methods pass through the front in the order the back runs them, so
// fed the same signals and the same QMD words, its copy runs the same QMDs at the same places
// and makes the same releases, which the front then notes (GpuSplit::NoteWrite, NoteRelease).
// It reads a QMD only where guest memory provably holds what the back will read (no queued
// write over it, or the inline QMD queued last over it); elsewhere it drains and copies the
// back's table. The back reports each release it makes; the front compares them with its own
// and stops predicting at the first difference.
struct FrontQmdMirror {
    struct Overlay {
        const void* space;
        GPUVAddr address;
        u64 position; ///< The end of the method that writes it (0x107)
        NativeQmdScheduler::Descriptor words;
    };
    static constexpr size_t MaxOverlays = 64;

    FrontQmdMirror() {
        scheduler.table = &table;
    }

    NativeQmdScheduler scheduler;
    NativeQmdScheduler::StateTable table;
    u64 session{~u64{0}};
    bool disabled{};
    std::array<Overlay, MaxOverlays> overlays{};
    size_t next_overlay{};
    /// Reads in the current window, those that waited for queued writes, and resyncs.
    u32 window_reads{};
    u32 window_waits{};
    u32 window_resyncs{};

    const Overlay* FindOverlay(const void* space, GPUVAddr address) const {
        for (const Overlay& overlay : overlays) {
            if (overlay.position != 0 && overlay.address == address && overlay.space == space) {
                return &overlay;
            }
        }
        return nullptr;
    }

    void AddOverlay(const void* space, GPUVAddr address, u64 position, const u32* words) {
        Overlay* slot = nullptr;
        for (Overlay& overlay : overlays) {
            if (overlay.position != 0 && overlay.address == address && overlay.space == space) {
                slot = &overlay;
                break;
            }
        }
        if (slot == nullptr) {
            slot = &overlays[next_overlay];
            next_overlay = (next_overlay + 1) % MaxOverlays;
        }
        slot->space = space;
        slot->address = address;
        slot->position = position;
        std::memcpy(slot->words.data(), words, sizeof(slot->words));
    }
};
FrontQmdMirror g_front_qmd;

// The releases each side made, in order (a seqlock per record: 0 while it is written). The front
// keeps its own; the back publishes its own, which the front compares at its next QMD signal.
struct QmdReleaseCheck {
    static constexpr size_t Size = size_t{1} << 14;
    struct Record {
        std::atomic<u64> seq;
        std::atomic<u64> address;
        std::atomic<u32> payload;
        std::atomic<u32> flags;
    };
    struct FrontRecord {
        u64 seq;
        u64 address;
        u32 payload;
        u32 flags;
        u64 position; ///< GpuSplit::WritePosition() when noted (the back made it by then)
    };
    std::array<Record, Size> back_records{};
    std::array<FrontRecord, Size> front_records{};
    /// The back records releases (off while the front copies its table).
    std::atomic<bool> active{};
    std::atomic<u64> back_seq{};
    std::atomic<u64> back_published{};
    u64 front_seq{};
    u64 front_checked{};
};
QmdReleaseCheck g_qmd_check;

void FrontRecordRelease(u64 address, u32 payload, u32 flags, u64 position) {
    auto& check = g_qmd_check;
    ++check.front_seq;
    check.front_records[(check.front_seq - 1) % QmdReleaseCheck::Size] = {
        .seq = check.front_seq,
        .address = address,
        .payload = payload,
        .flags = flags,
        .position = position};
}

// Back thread: a release ProcessNativeMethod's QMD scheduler makes.
void BackRecordRelease(u64 address, u32 payload, u32 flags) {
    auto& check = g_qmd_check;
    if (!check.active.load(std::memory_order_acquire) || !VideoCommon::GpuSplit::OnBackThread()) {
        return;
    }
    const u64 seq = check.back_seq.load(std::memory_order_relaxed) + 1;
    check.back_seq.store(seq, std::memory_order_relaxed);
    auto& record = check.back_records[(seq - 1) % QmdReleaseCheck::Size];
    record.seq.store(0, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    record.address.store(address, std::memory_order_relaxed);
    record.payload.store(payload, std::memory_order_relaxed);
    record.flags.store(flags, std::memory_order_relaxed);
    record.seq.store(seq, std::memory_order_release);
    check.back_published.store(seq, std::memory_order_release);
}

// Front: compares the releases the back published since the last check with the front's.
// Returns false at the first difference (or a release the back made that the front never did,
// or one the front made that the back skipped).
bool FrontCompareReleases(VideoCommon::GpuSplit& split) {
    auto& check = g_qmd_check;
    const u64 published = check.back_published.load(std::memory_order_acquire);
    for (u64 seq = check.front_checked + 1; seq <= published; ++seq) {
        if (seq > check.front_seq) {
            // The back ran the method before the front noted it (published at once to an idle
            // back): compare at the next check. A release the front never makes shifts every
            // later one, which the comparisons then catch.
            break;
        }
        check.front_checked = seq;
        if (published - seq >= QmdReleaseCheck::Size || check.front_seq - seq >= QmdReleaseCheck::Size) {
            continue; // Overwritten since: can't tell
        }
        const auto& record = check.back_records[(seq - 1) % QmdReleaseCheck::Size];
        const u64 first = record.seq.load(std::memory_order_acquire);
        const u64 address = record.address.load(std::memory_order_relaxed);
        const u32 payload = record.payload.load(std::memory_order_relaxed);
        const u32 flags = record.flags.load(std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_acquire);
        if (first != seq || record.seq.load(std::memory_order_relaxed) != seq) {
            continue; // Rewritten meanwhile
        }
        const auto& mine = check.front_records[(seq - 1) % QmdReleaseCheck::Size];
        if (mine.seq != seq) {
            continue;
        }
        if (mine.address != address || mine.payload != payload || mine.flags != flags) {
            LOG_ERROR(HW_GPU,
                      "QMD release prediction mismatch at release {}: front {:#x} = {:#x} "
                      "flags {:#x}, back {:#x} = {:#x} flags {:#x}; predictions off "
                      "(EDEN_NX2_QMD_PREDICT)",
                      seq, mine.address, mine.payload, mine.flags, address, payload, flags);
            return false;
        }
    }
    if (check.front_checked < check.front_seq) {
        // The back publishes a release while it runs the entry, before consumed passes it:
        // consumed past the oldest unchecked front release without it published means the
        // back skipped it. Load consumed first.
        const auto& oldest = check.front_records[check.front_checked % QmdReleaseCheck::Size];
        const u64 consumed = split.LoadConsumed();
        if (oldest.seq == check.front_checked + 1 && consumed >= oldest.position &&
            check.back_published.load(std::memory_order_acquire) < oldest.seq) {
            LOG_ERROR(HW_GPU,
                      "QMD release prediction: the back skipped release {} ({:#x}); "
                      "predictions off (EDEN_NX2_QMD_PREDICT)",
                      oldest.seq, oldest.address);
            return false;
        }
    }
    return true;
}

// EDEN_NX2_QMD_PREDICT: the front stops predicting for the rest of the run.
void DisableFrontQmd(VideoCommon::GpuSplit& split) {
    g_front_qmd.disabled = true;
    g_qmd_check.active.store(false, std::memory_order_seq_cst);
    split.NoteReleaseBarrier();
}

// The back runs everything sent (the method being noted too), then the front copies its table.
void ResyncFrontQmd(VideoCommon::GpuSplit& split) {
    auto& mirror = g_front_qmd;
    auto& check = g_qmd_check;
    check.active.store(false, std::memory_order_seq_cst);
    split.Drain(VideoCommon::DrainSite::QmdRead);
    // The back is idle: nothing else touches its table or its release count now.
    mirror.table = NativeQmdScheduler::SharedStates();
    mirror.scheduler.table = &mirror.table;
    mirror.overlays = {};
    mirror.next_overlay = 0;
    mirror.session = split.Session();
    check.back_seq.store(check.front_seq, std::memory_order_relaxed);
    check.back_published.store(check.front_seq, std::memory_order_relaxed);
    check.front_checked = check.front_seq;
    check.active.store(true, std::memory_order_seq_cst);
    // Releases the front didn't see ran meanwhile.
    split.NoteReleaseBarrier();
}

} // Anonymous namespace

namespace {

// EDEN_NX2_COMPUTE_FUSE=0: each compute method write the split's front sends is its own ring
// entry (KeplerCompute::FrontQueueWrite).
const bool g_compute_fuse = VideoCommon::Nx2Switch("EDEN_NX2_COMPUTE_FUSE", true);

// EDEN_NX2_COMPUTE_I2M_MERGE=0: inline upload flushes in a fused run go out one by one, even when
// each continues the one before within its 4 KiB page (KeplerCompute::BackWrites).
const bool g_compute_i2m_merge = VideoCommon::Nx2Switch("EDEN_NX2_COMPUTE_I2M_MERGE", true);

// EDEN_NX2_DROP_NATIVE_UPLOADS=0: native engines record every inline upload exec for the class's
// old launch method, which they never run (ProcessNativeMethod takes 0xaf): the list grew by one
// entry per exec for the whole session.
const bool g_drop_native_uploads = VideoCommon::Nx2Switch("EDEN_NX2_DROP_NATIVE_UPLOADS", true);

} // Anonymous namespace

KeplerCompute::KeplerCompute(MemoryManager& memory_manager_)
    : memory_manager{memory_manager_}
    , upload_state{memory_manager, regs.upload}
{
    // EDEN_NX2_QMD_TRACE: log the QMD scheduler's loads, dependency counts and launches.
    if (std::getenv("EDEN_NX2_QMD_TRACE")) {
        native_qmd.trace = [this](std::string_view event, u64 address, s32 value) {
            LOG_INFO(HW_GPU, "QMD trace {} {:#x} {} (engine {} as {})", event, address, value,
                     fmt::ptr(this), memory_manager.GetID());
        };
    }
    native_qmd.skipped_release = [](u64 address, std::size_t slot, std::span<const u32, 4> words) {
        static std::unordered_set<u32> logged;
        if (!logged.insert(words[1]).second) return;
        LOG_WARNING(HW_GPU, "Native QMD {:#x} skipped release slot {} with unsupported format, words={:08x}",
                    address, slot, fmt::join(words, ","));
    };
    execution_mask.reset();
    execution_mask[KEPLER_COMPUTE_REG_INDEX(exec_upload)] = true;
    execution_mask[KEPLER_COMPUTE_REG_INDEX(data_upload)] = true;
    execution_mask[KEPLER_COMPUTE_REG_INDEX(launch)] = true;
    // Everything ProcessNativeMethod acts on besides storing the value; the QMD and 0x5a
    // trace range too while native draw diagnostics trace it.
    native_plain_methods.set();
    for (const u32 method : {0x45U, 0x46U, 0x47U, 0x48U, 0xaeU, 0xafU, 0xb0U, 0x107U, 0x6c3U,
                             static_cast<u32>(KEPLER_COMPUTE_REG_INDEX(exec_upload)),
                             static_cast<u32>(KEPLER_COMPUTE_REG_INDEX(data_upload))}) {
        native_plain_methods[method] = false;
    }
    if (std::getenv("EDEN_NATIVE_DRAW_DIAGNOSTICS") != nullptr) {
        native_plain_methods[0x5a] = false;
        for (u32 method = 0xc6; method <= 0x108; ++method) {
            native_plain_methods[method] = false;
        }
    }
}

KeplerCompute::~KeplerCompute() {
    // Held writes point at this engine (EDEN_NX2_COMPUTE_FUSE).
    VideoCommon::GpuSplit::Get().DropPendingFlush(this);
}

void KeplerCompute::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
    upload_state.BindRasterizer(rasterizer);
}

void KeplerCompute::ConsumeSinkImpl(Core::System& system) {
    for (auto [method, value] : method_sink) {
        if (method < Regs::NUM_REGS) {
            regs.reg_array[method] = value;
        }
    }
    method_sink.clear();
}

// SET_REPORT_SEMAPHORE_A-D (0x6c0-0x6c3): address high/low, payload, then operation in bits 1:0
// (release / acquire / report only / trap), reduction enable bit 3 with the op in bits 11:9, and
// a one-word (no timestamp) structure in bit 28. NVN2 releases queue syncs from compute this way.
void KeplerCompute::ProcessNativeReportSemaphore(u32 argument) {
    const GPUVAddr address = (GPUVAddr{native_registers[0x6c0] & 0x1ffffff} << 32) |
                             native_registers[0x6c1];
    const u32 payload = native_registers[0x6c2];
    const u32 operation = argument & 3;
    const bool one_word = (argument >> 28) & 1;
    static u32 logged{};
    if (logged++ < 32) {
        LOG_INFO(HW_GPU, "Native compute report semaphore {:#x} = {:#x}, control {:#x}", address,
                 payload, argument);
    }
    if (operation != 0 && operation != 2) {
        LOG_WARNING(HW_GPU, "Native compute report semaphore operation {} not supported", operation);
        return;
    }
    // A report the query cache may hold until it flushes (GpuSplit::MayHoldReports).
    VideoCommon::GpuSplit::NoteGpuSemaphore(address);
    if ((argument >> 3) & 1) {
        const u32 reduction = (argument >> 9) & 7;
        rasterizer->SignalFence([this, address, payload, reduction] {
            memory_manager.Write<u32>(address, ReduceSemaphore(reduction, false,
                                                               memory_manager.Read<u32>(address), payload));
        });
        return;
    }
    rasterizer->Query(address, VideoCommon::QueryType::Payload,
                      one_word ? VideoCommon::QueryPropertiesFlags::IsAFence
                               : VideoCommon::QueryPropertiesFlags::HasTimeout,
                      payload, 0);
}

bool KeplerCompute::ProcessNativeMethod(Core::System& system, u32 method, u32 argument, bool is_last_call) {
    if (native_context && !front_macros) {
        if (method == 0x45 || method == 0x47) native_compute_uploaded = true;
        native_macro = native_compute_uploaded ? &native_context->compute_engine : &native_context->engine;
        native_macro_positions = native_compute_uploaded ? std::span<u32>{native_context->compute_positions} : std::span<u32>{native_context->positions};
    }
    if (method >= native_registers.size()) {
        LOG_ERROR(HW_GPU, "Native compute method out of range: {:#x}", method);
        return true;
    }
    if (method >= 0xe00) {
        if (!native_executing_macro) {
            if (method & 1) { LOG_ERROR(HW_GPU, "Native compute macro starts on argument method"); return true; }
            native_executing_macro = method;
        }
        native_parameters.push_back(argument);
        if (!is_last_call) return true;
        const u32 entry = ((*native_executing_macro - 0xe00) / 2) % native_macro_positions.size();
        native_executing_macro_logged = native_executing_macro;
        native_executing_macro.reset();
        static const u32 watch = [] {
            const char* value = std::getenv("EDEN_NX2_SEM_WATCH");
            return value ? static_cast<u32>(std::strtoull(value, nullptr, 16)) : 0U;
        }();
        const bool watched = watch != 0 && std::ranges::any_of(native_parameters,
            [](u32 parameter) { return (parameter & ~0xfU) == watch; });
        if (watched) {
            LOG_INFO(HW_GPU, "Semaphore watch: compute macro start={:#x} parameters={:08x}",
                     native_macro_positions[entry], fmt::join(native_parameters, ","));
        }
        // EDEN_NX2_MME_DUMP: write the compute macro code once per engine, and log each macro
        // call with its entry and parameters (diagnostics).
        if (static const bool mme_dump = std::getenv("EDEN_NX2_MME_DUMP") != nullptr; mme_dump) {
            if (!native_mme_dumped && native_macro_positions[entry] == 0xda) {
                native_mme_dumped = true;
                std::ofstream dump(fmt::format("native-compute-mme-{}.bin", fmt::ptr(this)),
                                   std::ios::binary);
                dump.write(reinterpret_cast<const char*>(native_macro->native_code.data()),
                           sizeof(native_macro->native_code));
            }
            LOG_INFO(HW_GPU, "MME call engine {} method {:#x} start={:#x} parameters={:08x}",
                     fmt::ptr(this), *native_executing_macro_logged, native_macro_positions[entry],
                     fmt::join(native_parameters, ","));
        }
        if (native_data_ram) native_macro->native_interpreter.active_dram = native_data_ram;
        const u32 dram0_before = (*native_macro->native_interpreter.active_dram)[0x27];
        const u32 start = native_macro_positions[entry];
        const auto emit = [&](u32 address, u32 value) {
            static u32 diagnostic_writes{};
            static const bool log_emits = std::getenv("EDEN_NX2_SEM_WATCH_EMITS") != nullptr;
            if (watched && log_emits) LOG_INFO(HW_GPU, "Semaphore watch: compute macro emit {:#x} = {:#x}", address, value);
            if (diagnostic_writes++ < 4096) LOG_DEBUG(HW_GPU, "Native compute MME emit method={:#x}, value={:#x}", address, value);
            if (address >= 0xe00) return false;
            VideoCommon::Nx2Profile::Scope method_scope{VideoCommon::Nx2Profile::Zone::Method};
            CallMethod(system, address, value, true);
            return true;
        };
#ifdef ARCHITECTURE_x86_64
        // Compiled macro code, as for 3D macros (macro_turing_jit.h); EDEN_NX2_MME_JIT=0 off.
        static const bool use_jit = [] {
            const char* value = std::getenv("EDEN_NX2_MME_JIT");
            return value == nullptr || value[0] != '0';
        }();
        if (use_jit && !watched) {
            if (!native_macro->native_jit) {
                native_macro->native_jit = std::make_shared<TuringMacroJit>();
            }
            native_macro->native_jit->Prepare(native_macro->native_code,
                                              native_macro->native_code_generation);
            const auto result = native_macro->native_jit->Execute(
                start, native_parameters,
                std::span<u32>{*native_macro->native_interpreter.active_dram},
                [&](u32 address) { return address < native_registers.size() ? native_registers[address] : 0U; },
                emit);
            if (!result.success) {
                native_macro->native_execution_failed = true;
                LOG_ERROR(HW_GPU, "Native MME (JIT) failed: compute start={:#x}, consumed={}/{}, error={}",
                          start, result.consumed, native_parameters.size(), result.error);
            }
            native_parameters.clear();
            return true;
        }
#endif
        // Repeated calls replay their writes, as for 3D macros (macro_memo.h).
        static const bool use_memo = [] {
            const char* value = std::getenv("EDEN_NX2_MME_MEMO");
            return value == nullptr || value[0] != '0';
        }();
        auto& memo = native_macro->native_memo;
        if (use_memo && !watched) {
            memo.Synchronize(native_macro->native_code_generation);
            if (std::remove_reference_t<decltype(memo)>::Effects effects;
                memo.Find(start, native_parameters, *native_macro->native_interpreter.active_dram,
                          effects)) {
                auto& data_ram = *native_macro->native_interpreter.active_dram;
                for (size_t i = 0; i < effects.data_ram_writes.size(); i += 2) {
                    data_ram[effects.data_ram_writes[i]] = effects.data_ram_writes[i + 1];
                }
                for (size_t i = 0; i < effects.writes.size(); i += 2) {
                    emit(effects.writes[i], effects.writes[i + 1]);
                }
                native_parameters.clear();
                return true;
            }
            native_macro->native_writes.clear();
        }
        // The interpreter consumes native_parameters as its FIFO: keep the call's own parameters.
        native_memo_parameters.assign(native_parameters.begin(), native_parameters.end());
        const u64 code_generation = native_macro->native_code_generation;
        const auto result = native_macro->native_interpreter.Execute(native_macro->native_code,
            start, native_parameters,
            [&](u32 address) { return address < native_registers.size() ? native_registers[address] : 0U; },
            [&](u32 address, u32 value) {
                if (use_memo) {
                    native_macro->native_writes.emplace_back(address, value);
                }
                return emit(address, value);
            });
        const auto& interpreter = native_macro->native_interpreter;
        if (use_memo && !watched && result.success && !interpreter.read_registers &&
            !interpreter.reread_data_ram && code_generation == native_macro->native_code_generation) {
            memo.Store(start, native_memo_parameters, interpreter.data_ram_reads,
                       interpreter.data_ram_writes, native_macro->native_writes);
        }
        if (static const bool mme_dump = std::getenv("EDEN_NX2_MME_DUMP") != nullptr;
            mme_dump && (*native_macro->native_interpreter.active_dram)[0x27] != dram0_before) {
            LOG_INFO(HW_GPU, "MME data RAM[0x27] {:#x} -> {:#x} by engine {} macro start={:#x}",
                     dram0_before, (*native_macro->native_interpreter.active_dram)[0x27], fmt::ptr(this),
                     native_macro_positions[entry]);
        }
        if (!result.success) {
            native_macro->native_execution_failed = true;
            if (static const bool diagnostics = std::getenv("EDEN_NATIVE_DRAW_DIAGNOSTICS") != nullptr;
                diagnostics) {
                std::ofstream dump("native-compute-mme.bin", std::ios::binary);
                dump.write(reinterpret_cast<const char*>(native_macro->native_code.data()), sizeof(native_macro->native_code));
            }
            LOG_ERROR(HW_GPU, "Native MME failed: compute start={:#x}, pc={:#x}, consumed={}/{}, error={}, parameters={:08x}",
                native_macro_positions[entry], result.pc, result.consumed, native_parameters.size(), result.error,
                fmt::join(native_parameters, ","));
        }
        native_parameters.clear();
        return true;
    }
    native_registers[method] = argument;
    if (method == 0x6c3) { // SET_REPORT_SEMAPHORE_D: the compute class's semaphore release
        ProcessNativeReportSemaphore(argument);
        return true;
    }
    if (method == 0x5a || (method >= 0xc6 && method <= 0x108)) {
        VideoCommon::TraceNativeQuery(12, memory_manager.GetID(), method, 0, 0, argument, 0);
    }
    if (method == 0x107) {
        const GPUVAddr address = ((GPUVAddr{native_registers[0xc6]} << 32) |
            native_registers[0xc7]) << 8;
        const GPUVAddr program = (GPUVAddr{native_registers[0xc8 + 49] & 0x1ffff} << 32) |
            native_registers[0xc8 + 48];
        const u32 control = native_registers[0xa7];
        LOG_DEBUG(HW_GPU, "Native inline QMD: address={:#x}, program={:#x}, control={:#x}, grid={}/{}/{}",
            address, program, control, native_registers[0xc8 + 12],
            native_registers[0xc8 + 13] & 0xffff, native_registers[0xc8 + 14] & 0xffff);
        if (static const bool diagnostics = std::getenv("EDEN_NATIVE_DRAW_DIAGNOSTICS") != nullptr;
            diagnostics) {
            std::ofstream dump("native-inline-qmd.bin", std::ios::binary);
            dump.write(reinterpret_cast<const char*>(native_registers.data() + 0xc8), 256);
            std::ofstream registers("native-inline-compute-registers.bin", std::ios::binary);
            registers.write(reinterpret_cast<const char*>(native_registers.data()), sizeof(native_registers));
            if (memory_manager.IsFullyMappedRange(program, 0x10000)) {
                std::vector<u8> code(0x10000);
                memory_manager.ReadBlockUnsafe(program, code.data(), code.size());
                std::ofstream shader("native-inline-compute-shader.bin", std::ios::binary);
                shader.write(reinterpret_cast<const char*>(code.data()), code.size());
            }
        }
        if (!memory_manager.IsFullyMappedRange(address, 256)) {
            native_macro->native_execution_failed = true;
            LOG_ERROR(HW_GPU, "Unmapped native inline QMD destination");
            return true;
        }
        rasterizer->AccelerateInlineToMemory(address, 256,
            std::span<const u8>{reinterpret_cast<const u8*>(native_registers.data() + 0xc8), 256});
        if ((control & (1U << 24)) || !(native_registers[0xcc] & (1U << 12))) {
            const u32 previous = native_registers[0xad];
            native_registers[0xad] = static_cast<u32>(address >> 8);
            ProcessNativeMethod(system, 0xb0, 3, true);
            native_registers[0xad] = previous;
        }
        return true;
    }
    if (front_macros && method >= 0x45 && method <= 0x48) {
        return true; // Loaded and bound by the split GPU thread's front (FrontApply).
    }
    switch (method) {
    case 0x45: native_macro->ClearCode(argument); return true;
    case 0x46: native_macro->AddCode(0, argument); return true;
    case 0x47: native_bind_pointer = argument; return true;
    case 0x48:
        if (native_bind_pointer < native_macro_positions.size()) native_macro_positions[native_bind_pointer++] = argument;
        else LOG_ERROR(HW_GPU, "Native compute macro bind out of bounds");
        return true;
    case 0xae: case 0xaf: case 0xb0: {
        const GPUVAddr qmd_address = GPUVAddr{native_registers[0xad]} << 8;
        VideoCommon::TraceNativeQuery(9, memory_manager.GetID(), qmd_address,
            memory_manager.GpuToCpuAddress(qmd_address).value_or(0), method, argument, 0);
        if (method == 0xae || (method == 0xaf && !(argument & 2)) ||
            (method == 0xb0 && ((argument & 15) == 0 || (argument & 15) == 1))) return true;
        static const bool qmd_diagnostics = std::getenv("EDEN_NATIVE_DRAW_DIAGNOSTICS") != nullptr;
        if ((argument & 15) == 12 && qmd_diagnostics) {
            static u32 captured{};
            if (captured++ == 0 && memory_manager.IsFullyMappedRange(qmd_address, 256)) {
                std::array<u32, 64> descriptor{};
                memory_manager.ReadBlockUnsafe(qmd_address, descriptor.data(), sizeof(descriptor));
                std::ofstream dump("native-increment-qmd.bin", std::ios::binary);
                dump.write(reinterpret_cast<const char*>(descriptor.data()), sizeof(descriptor));
                LOG_WARNING(HW_GPU, "Native increment dependency descriptor: address={:#x}, words={:08x}",
                    qmd_address, fmt::join(descriptor, ","));
            }
        }
        const auto error = native_qmd.Signal(qmd_address, argument,
            [&](u64 address, NativeQmdScheduler::Descriptor& descriptor) {
                if (!memory_manager.IsFullyMappedRange(address, sizeof(descriptor))) return false;
                memory_manager.ReadBlockUnsafe(address, descriptor.data(), sizeof(descriptor));
                // An all-zero QMD whose words GPU work wrote (a copy) may still be only in the
                // buffer cache: flush it and read again (EDEN_NX2_QMD_FLUSH=0: off).
                static const bool qmd_flush = VideoCommon::Nx2Switch("EDEN_NX2_QMD_FLUSH", true);
                if (qmd_flush && std::ranges::all_of(descriptor, [](u32 word) { return word == 0; })) {
                    if (const auto device = memory_manager.GpuToCpuAddress(address);
                        device && rasterizer->MustFlushRegion(*device, sizeof(descriptor),
                                                              VideoCommon::CacheType::BufferCache)) {
                        rasterizer->FlushRegion(*device, sizeof(descriptor),
                                                VideoCommon::CacheType::BufferCache);
                        memory_manager.ReadBlockUnsafe(address, descriptor.data(), sizeof(descriptor));
                        LOG_WARNING(HW_GPU, "Native QMD {:#x} read zero from guest memory; flushed "
                                    "the buffer cache", address);
                    }
                }
                if (static u32 logged{}; (descriptor[53] & (1U << 23)) && logged < 8) {
                    ++logged;
                    LOG_INFO(HW_GPU, "Native QMD {:#x} releases: {:08x} | {:08x} | {:08x}", address,
                             fmt::join(std::span{descriptor}.subspan(24, 4), ","),
                             fmt::join(std::span{descriptor}.subspan(28, 4), ","),
                             fmt::join(std::span{descriptor}.subspan(52, 4), ","));
                }
                for (u32 slot = 0; slot < 2; ++slot) {
                    const u32 flags = descriptor[25 + slot * 4];
                    if (!(flags & (1U << 23))) continue;
                    const GPUVAddr destination = (GPUVAddr{flags & 0xff} << 32) | descriptor[24 + slot * 4];
                    VideoCommon::TraceNativeQuery(7 + slot, memory_manager.GetID(), destination,
                        memory_manager.GpuToCpuAddress(destination).value_or(0),
                        descriptor[4], descriptor[26 + slot * 4], flags);
                }
                return true;
            },
            [&](u64 address, u32 payload, u32 flags) {
                const bool four_words = (flags >> 30) == 0;
                if (!memory_manager.IsFullyMappedRange(address, four_words ? 16 : sizeof(payload))) return false;
                // EDEN_NX2_QMD_PREDICT: the front checks its prediction against it.
                BackRecordRelease(address, payload, flags);
                // The split's front: acquires met on this page are host-level waits.
                VideoCommon::GpuSplit::NoteGpuSemaphore(address);
                if (flags & (1U << 26)) {
                    rasterizer->SignalFence([this, address, payload, flags] {
                        const u32 value = ReduceSemaphore((flags >> 20) & 7, (flags >> 24) & 1,
                                                          memory_manager.Read<u32>(address), payload);
                        memory_manager.Write<u32>(address, value);
                        if (const auto device = memory_manager.GpuToCpuAddress(address)) {
                            Tegra::Nx2SemaphoreWaiters::Get().OnWrite(*device, value);
                        }
                        static const u64 watch = [] {
                            const char* text = std::getenv("EDEN_NX2_SEM_WATCH");
                            return text ? u64{std::strtoull(text, nullptr, 16)} : u64{};
                        }();
                        if (watch != 0 && (address & ~u64{0xf}) == watch) {
                            LOG_INFO(HW_GPU, "Semaphore watch: QMD release reduction {:#x} -> {:#x}", address, value);
                        }
                        VideoCommon::TraceNativeQuery(1, memory_manager.GetID(), address,
                            memory_manager.GpuToCpuAddress(address).value_or(0), 0, value, 1U << 26);
                    });
                } else if (four_words) {
                    rasterizer->Query(address, VideoCommon::QueryType::Payload,
                        VideoCommon::QueryPropertiesFlags::IsAFence | VideoCommon::QueryPropertiesFlags::HasTimeout,
                        payload, 0);
                } else if (flags & (1U << 19)) {
                    // NVN2 tracks QMD completion with a max release (bit 19) into a word other
                    // releases also advance: a plain write would move it backwards and leave
                    // GEQ acquires waiting forever.
                    rasterizer->SignalFence([this, address, payload] {
                        const u32 value = (std::max)(memory_manager.Read<u32>(address), payload);
                        memory_manager.Write<u32>(address, value);
                        if (const auto device = memory_manager.GpuToCpuAddress(address)) {
                            Tegra::Nx2SemaphoreWaiters::Get().OnWrite(*device, value);
                        }
                    });
                } else {
                    static const u64 watch = [] {
                        const char* text = std::getenv("EDEN_NX2_SEM_WATCH");
                        return text ? u64{std::strtoull(text, nullptr, 16)} : u64{};
                    }();
                    if (watch != 0 && (address & ~u64{0xf}) == (watch & ~u64{0xf})) {
                        LOG_INFO(HW_GPU, "Semaphore watch: QMD release {:#x} = {:#x} flags {:#x}", address, payload, flags);
                    }
                    rasterizer->Query(address, VideoCommon::QueryType::Payload,
                        VideoCommon::QueryPropertiesFlags::IsAFence, payload, 0);
                }
                return true;
            }, [&](const auto& descriptor) { return DispatchNativeCompute(descriptor); });
        if (error.empty()) {
            LOG_DEBUG(HW_GPU, "Native QMD control action={:#x}, QMD={:#x}", argument, qmd_address);
            return true;
        }
        native_macro->native_execution_failed = true;
        std::array<u32, 128> qmd{};
        if (memory_manager.IsFullyMappedRange(qmd_address, sizeof(qmd))) {
            memory_manager.ReadBlockUnsafe(qmd_address, qmd.data(), sizeof(qmd));
        }
        // Log a failing program's first few dispatches only (QMD words 48-49 hold its code
        // address); it fails the same way every frame.
        static std::unordered_map<u64, u32> failures;
        const u64 program = (u64{qmd[49]} << 32) | qmd[48];
        if (failures[program ^ std::hash<std::string>{}(std::string{error})]++ >= 2) {
            return true;
        }
        LOG_ERROR(HW_GPU, "Native QMD failed: {}", error);
        if (qmd_diagnostics && memory_manager.IsFullyMappedRange(qmd_address, sizeof(qmd))) {
            std::ofstream dump("native-qmd.bin", std::ios::binary);
            dump.write(reinterpret_cast<const char*>(qmd.data()), sizeof(qmd));
            std::ofstream macros("native-dispatch-mme.bin", std::ios::binary);
            macros.write(reinterpret_cast<const char*>(native_macro->native_code.data()), sizeof(native_macro->native_code));
            std::ofstream registers("native-dispatch-registers.bin", std::ios::binary);
            registers.write(reinterpret_cast<const char*>(native_registers.data()), sizeof(native_registers));
        }
        LOG_ERROR(HW_GPU, "Native compute dispatch unsupported: method={:#x}, value={:#x}, QMD={:#x}, words={:08x}", method, argument,
                  qmd_address, fmt::join(qmd, ","));
        return true;
    }
    default:
        // The inline upload register block is shared with the older class.
        if (method < Regs::NUM_REGS) regs.reg_array[method] = argument;
        return method != KEPLER_COMPUTE_REG_INDEX(exec_upload) && method != KEPLER_COMPUTE_REG_INDEX(data_upload);
    }
}

void KeplerCompute::CallMethod(Core::System& system, u32 method, u32 method_argument, bool is_last_call) {
    if (front_macros && !VideoCommon::GpuSplit::OnBackThread()) {
        ReleaseFront();
    }
    if (native_mode) {
        // Most of the compute class's register writes (QMD fields replayed by its macros for
        // each of thousands of dispatches a second) only store the value: ~4% of title 1's GPU
        // thread went through ProcessNativeMethod's checks for them.
        if (method < native_plain_methods.size() && native_plain_methods[method]) {
            native_registers[method] = method_argument;
            if (method < Regs::NUM_REGS) {
                regs.reg_array[method] = method_argument;
            }
            return;
        }
        if (ProcessNativeMethod(system, method, method_argument, is_last_call)) return;
    }
    if (method >= Regs::NUM_REGS) {
        LOG_ERROR(HW_GPU, "Compute register out of range: {:#x}", method);
        return;
    }

    regs.reg_array[method] = method_argument;

    switch (method) {
    case KEPLER_COMPUTE_REG_INDEX(exec_upload): {
        // Only the launch case below reads the list, and native mode never gets there
        // (EDEN_NX2_DROP_NATIVE_UPLOADS).
        if (!native_mode || !g_drop_native_uploads) {
            UploadInfo info{.upload_address = upload_address,
                            .exec_address = upload_state.ExecTargetAddress(),
                            .copy_size = upload_state.GetUploadSize(),
                            .was_dirty = upload_dirty};
            uploads.push_back(info);
        }
        upload_state.ProcessExec(regs.exec_upload.linear != 0);
        break;
    }
    case KEPLER_COMPUTE_REG_INDEX(data_upload): {
        upload_address = current_dma_segment;
        upload_dirty = current_dirty;
        current_dirty = false;
        upload_state.ProcessData(method_argument, is_last_call);
        break;
    }
    case KEPLER_COMPUTE_REG_INDEX(launch): {
        const GPUVAddr launch_desc_loc = regs.launch_desc_loc.Address();

        for (auto& data : uploads) {
            const GPUVAddr offset = data.exec_address - launch_desc_loc;
            if (offset / sizeof(u32) == LAUNCH_REG_INDEX(grid_dim_x)) {
                const bool source_dirty = memory_manager.IsMemoryDirty(data.upload_address, data.copy_size);
                if (data.was_dirty || source_dirty) {
                    indirect_compute = {data.upload_address};
                }
            }
        }
        uploads.clear();
        ProcessLaunch();
        indirect_compute = std::nullopt;
        break;
    }
    default:
        break;
    }
}

void KeplerCompute::CallMultiMethod(Core::System& system, u32 method, const u32* base_start, u32 amount, u32 methods_pending) {
    if (native_mode) {
        for (u32 i = 0; i < amount; ++i) CallMethod(system, method, base_start[i], methods_pending - i <= 1);
        return;
    }
    switch (method) {
    case KEPLER_COMPUTE_REG_INDEX(data_upload):
        upload_address = current_dma_segment;
        upload_dirty = current_dirty;
        current_dirty = false;
        upload_state.ProcessData(base_start, amount);
        return;
    default:
        for (u32 i = 0; i < amount; i++) {
            CallMethod(system, method, base_start[i], methods_pending - i <= 1);
        }
        break;
    }
}

bool KeplerCompute::DispatchNativeCompute(const NativeQmdScheduler::Descriptor& qmd) {
    launch_description = {};
    launch_description.grid_dim_x.Assign(qmd[12]);
    launch_description.grid_dim_y.Assign(qmd[13] & 0xffff);
    launch_description.grid_dim_z.Assign(qmd[14] & 0xffff);
    launch_description.shared_alloc.Assign(qmd[17] & 0x3ffff);
    launch_description.block_dim_x.Assign(qmd[18] >> 16);
    launch_description.block_dim_y.Assign(qmd[19] & 0xffff);
    launch_description.block_dim_z.Assign(qmd[19] >> 16);
    launch_description.const_buffer_enable_mask.Assign(qmd[20] & 0xff);
    launch_description.local_pos_alloc.Assign(qmd[23] & 0xffffff);
    launch_description.local_crs_alloc.Assign(qmd[50] & 0xffffff);
    regs.code_loc.address_low = qmd[48];
    regs.code_loc.address_high = qmd[49] & 0x1ffff;
    for (u32 slot = 0; slot < NumConstBuffers; ++slot) {
        const u32 high = qmd[33 + slot * 2];
        if (high & 0x1ff00) {
            LOG_ERROR(HW_GPU, "Native compute constant address exceeds supported GPU width");
            return false;
        }
        auto& buffer = launch_description.const_buffer_config[slot];
        buffer.address_low = qmd[32 + slot * 2];
        buffer.address_high.Assign(high & 0xff);
        buffer.size.Assign(((high >> 19) & 0x1fff) << 4);
    }
    if (!launch_description.block_dim_x || !launch_description.block_dim_y || !launch_description.block_dim_z)
        return false;
    dlss_chain.output_pass = false;
    // Fortune's Weave's evaluation is 13 dispatches. Bound incomplete evaluations, including
    // skipped dispatches, so a missing output pass cannot keep the chain active indefinitely.
    constexpr u32 max_dispatches = 64;
    if (dlss_chain.active && ++dlss_chain.dispatches > max_dispatches) {
        dlss_chain.active = false;
        dlss_chain.pending.clear();
        dlss_motion_vectors.reset();
        dlss_depth.reset();
        (void)fe_upscale_jitter.Take();
    }
    bool dispatched = true;
    try {
        rasterizer->DispatchCompute();
    } catch (const VideoCommon::FireEmblemUpscale::DispatchReplaced&) {
        return true;
    } catch (const Shader::Exception& error) {
        if (Common::DkbFrameTrace::Enabled()) {
            static thread_local Common::DkbFrameTrace::RateLimit limit;
            static thread_local u64 rejected{};
            ++rejected;
            if (limit.Ready(std::chrono::steady_clock::now())) {
                LOG_INFO(HW_GPU,
                         "DKB shader rejected stage=compute address={:#x} count={} reason={}",
                         regs.code_loc.Address() + launch_description.program_start, rejected,
                         error.what());
                rejected = 0;
            }
        }
        dlss_motion_vectors.reset();
        dlss_depth.reset();
        (void)fe_upscale_jitter.Take();
        static std::unordered_set<std::string> logged;
        if (logged.insert(error.what()).second) {
            LOG_ERROR(HW_GPU, "Native compute shader rejected: {}", error.what());
        }
        dispatched = false;
        if (dlss_chain.active) {
            ++dlss_chain.rejected;
        } else if (std::string_view{error.what()}.find("matrix multiply") !=
                   std::string_view::npos) {
            DescribeNativeResources(qmd);
            LOG_ERROR(HW_GPU, "Native tensor dispatch is unsupported; no validated replacement was available");
        }
    }
    if (dlss_chain.output_pass) {
        return FinishDlssEvaluation() || dispatched;
    }
    return dispatched;
}

namespace {
constexpr u32 DLSS_NETWORK_CLEAN_EVALUATIONS = 8;
/// Health checks in a row the network must pass before it is shown (about a second: a network that
/// goes bad does so within its first few evaluations), and fail to be dropped.
constexpr u32 DLSS_NETWORK_GOOD_CHECKS = 30;
constexpr u32 DLSS_NETWORK_BAD_CHECKS = 10;
/// A check fails when more than this percentage of its blocks is mismatched (working networks:
/// at most 0.2% on Fortune's Weave's title and Kunlun Base; frozen ones: 6-10%), or when any output
/// sample is NaN or infinite.
constexpr u32 DLSS_CHECK_MISMATCH_PERCENT = 3;
} // Anonymous namespace

bool KeplerCompute::ProcessGameDlss(u64 program_id, u64 native_key) {
    using namespace VideoCommon::FireEmblemUpscale;
    const auto passes = CapturedPasses(program_id);
    if (!passes) {
        return false;
    }
    if (passes->spatial_only && (native_key == passes->prepass || native_key == passes->output)) {
        TraceDlssContract(native_key, false);
    }
    if (native_key == passes->prepass) {
        dlss_chain.spatial_only = passes->spatial_only;
        CaptureFireEmblemJitter();
        dlss_chain.active = true;
        dlss_chain.rejected = 0;
        dlss_chain.dispatches = 0;
        dlss_chain.pending.clear();
    }
    if (!dlss_chain.active) {
        return false;
    }
    dlss_chain.pending.insert(native_key);
    dlss_chain.output_pass = native_key == passes->output;
    if (dlss_chain.spatial_only) {
        return false;
    }
    const bool replacement = dlss_chain.failure != VideoCore::GameDlssReason::None ||
                             GameDlssBlocker() != VideoCore::GameDlssReason::None;
    if (dlss_chain.output_pass) {
        return replacement && FinishDlssEvaluation();
    }
    return dlss_chain.failure != VideoCore::GameDlssReason::None &&
           native_key != passes->prepass &&
           dlss_chain.programs.contains(native_key);
}

// The game's own DLSS network (its translated programs) is shown once every pass of an
// evaluation has run several times in a row and its output has passed several health checks in
// a row. Until then, when any pass can't run, and for the rest of the session once the network
// fails its checks, the replacement upscale draws over the output pass's result. The setting
// "Use the game's own DLSS" defaults off; an explicit native override enables it, while fsr/dlss
// overrides keep the replacement. The network's image sizes are fixed by the game, so it is used
// only without resolution scaling; its matrix multiplies need shaders 32 threads wide.
VideoCore::GameDlssReason KeplerCompute::GameDlssBlocker() const {
    using VideoCore::GameDlssReason;
    static const char* const option = std::getenv("EDEN_NX2_DLSS_REPLACEMENT");
    const bool forced_replacement = option && VideoCommon::FireEmblemUpscale::UseReplacement(
        VideoCommon::FireEmblemUpscale::ProgramId,
        VideoCommon::FireEmblemUpscale::OutputProgramKey, option);
    const bool forced_native = option && !forced_replacement;
    if (forced_replacement ||
        (!forced_native && !Settings::values.game_dlss_network.GetValue())) {
        return GameDlssReason::TurnedOff;
    }
    // NVIDIA DLSS, when turned on and running, comes first.
    if (rasterizer->UsesNvidiaDlss()) {
        return GameDlssReason::NvidiaDlss;
    }
    const auto& scale = Settings::values.resolution_info;
    if (scale.up_scale != 1 || scale.down_shift != 0) {
        return GameDlssReason::ResolutionScaling;
    }
    if (!rasterizer->SupportsGameDlssNetwork()) {
        return GameDlssReason::Unsupported;
    }
    return GameDlssReason::None;
}

// Checks the network's output against its input (on the GPU; the result arrives a few frames
// later) and drops the network for the session after DLSS_NETWORK_BAD_CHECKS failed checks in a
// row: a network that produces NaN, freezes on an old image or draws garbage would otherwise stay
// on screen.
void KeplerCompute::CheckDlssNetwork() {
    dlss_chain.checked = rasterizer->CanProbeGameDlssOutput();
    if (!dlss_chain.checked) {
        return;
    }
    const auto& cbuf = launch_description.const_buffer_config[0];
    const GPUVAddr cbuf_address = (GPUVAddr{cbuf.address_high} << 32) | cbuf.address_low;
    if (!(launch_description.const_buffer_enable_mask & 1) || cbuf.size.Value() < 0x34c ||
        !memory_manager.IsFullyMappedRange(cbuf_address, 0x34c)) {
        return;
    }
    const auto surface = [&](u32 offset) -> std::optional<Fermi2D::Surface> {
        const u32 tic_index = memory_manager.Read<u32>(cbuf_address + offset) & 0xfffff;
        if (tic_index > regs.tic.limit || !memory_manager.IsFullyMappedRange(
                regs.tic.Address() + u64{tic_index} * sizeof(Texture::TICEntry),
                sizeof(Texture::TICEntry))) {
            return std::nullopt;
        }
        const auto tic = GetTICEntry(tic_index);
        using RTFormat = Tegra::RenderTargetFormat;
        Fermi2D::Surface result{};
        switch (tic.format) {
        case Texture::TextureFormat::B10G11R11:
            result.format = RTFormat::B10G11R11_FLOAT;
            break;
        case Texture::TextureFormat::R16G16B16A16:
            result.format = RTFormat::R16G16B16A16_FLOAT;
            break;
        case Texture::TextureFormat::A8B8G8R8:
            result.format = RTFormat::A8B8G8R8_UNORM;
            break;
        case Texture::TextureFormat::A2B10G10R10:
            result.format = RTFormat::A2B10G10R10_UNORM;
            break;
        default:
            return std::nullopt;
        }
        if (tic.Address() == 0 || !tic.IsBlockLinear()) {
            return std::nullopt;
        }
        result.linear = Fermi2D::MemoryLayout::BlockLinear;
        result.block_width.Assign(tic.block_width);
        result.block_height.Assign(tic.block_height);
        result.block_depth.Assign(tic.block_depth);
        result.depth = 1;
        result.width = tic.Width();
        result.height = tic.Height();
        result.addr_upper = static_cast<u32>(tic.Address() >> 32);
        result.addr_lower = static_cast<u32>(tic.Address());
        return result;
    };
    const auto input = surface(0x328);
    const auto output = surface(0x348);
    if (!input || !output) {
        return;
    }
    const auto probe = rasterizer->ProbeGameDlssOutput(*input, *output);
    if (!probe) {
        return;
    }
    const bool invalid = probe->invalid_samples != 0;
    const bool mismatched = probe->mismatched * 100 > probe->blocks * DLSS_CHECK_MISMATCH_PERCENT;
    if (!invalid && !mismatched) {
        ++dlss_chain.good_checks;
        dlss_chain.bad_checks = 0;
        return;
    }
    dlss_chain.good_checks = 0;
    if (++dlss_chain.bad_checks < DLSS_NETWORK_BAD_CHECKS) {
        return;
    }
    dlss_chain.failure = invalid ? VideoCore::GameDlssReason::InvalidOutput
                                 : VideoCore::GameDlssReason::OutputMismatch;
    LOG_WARNING(HW_GPU,
                "Game DLSS network: {} in {} checks in a row (last: {}/{} blocks mismatched, worst "
                "{:.3f}, {} invalid samples); the replacement upscale draws the output from now on",
                VideoCore::GameDlssReasonText(dlss_chain.failure), dlss_chain.bad_checks,
                probe->mismatched, probe->blocks, probe->worst, probe->invalid_samples);
}

// Logs what draws the game's DLSS evaluation, when that changes.
void KeplerCompute::UpdateGameDlssStatus(bool complete, VideoCore::GameDlssReason blocker) {
    using VideoCore::GameDlssReason;
    using VideoCore::GameDlssState;
    VideoCore::GameDlssStatus shown{.state = GameDlssState::Checking};
    if (dlss_chain.native) {
        shown.state = GameDlssState::Network;
    } else if (dlss_chain.failure != GameDlssReason::None) {
        shown = {.state = GameDlssState::Replacement, .reason = dlss_chain.failure};
    } else if (blocker != GameDlssReason::None) {
        shown = {.state = GameDlssState::Replacement, .reason = blocker};
    } else if (!complete) {
        shown = {.state = GameDlssState::Replacement,
                 .reason = GameDlssReason::PassesCannotRun,
                 .rejected_passes = dlss_chain.rejected};
    }
    if (shown.state == dlss_chain.status.state && shown.reason == dlss_chain.status.reason &&
        shown.rejected_passes == dlss_chain.status.rejected_passes) {
        return;
    }
    // A failed check was logged as it happened; the network's start is logged above.
    if (shown.state == GameDlssState::Checking) {
        LOG_INFO(HW_GPU, "Game DLSS network: checking its output; the replacement upscale draws "
                         "meanwhile");
    } else if (shown.reason == GameDlssReason::PassesCannotRun) {
        LOG_INFO(HW_GPU,
                 "Game DLSS network: {} pass(es) can't run, the replacement upscale "
                 "draws the output",
                 shown.rejected_passes);
    } else if (shown.state == GameDlssState::Replacement && shown.reason != dlss_chain.failure) {
        LOG_INFO(HW_GPU, "Game DLSS network: {}, the replacement upscale draws the output",
                 VideoCore::GameDlssReasonText(shown.reason));
    }
    dlss_chain.status = shown;
}

// Ends this frame's DLSS evaluation at its output pass: either the network's own result stays,
// or the replacement upscale draws over it. Returns whether the replacement ran.
bool KeplerCompute::FinishDlssEvaluation() {
    dlss_chain.active = false;
    dlss_chain.output_pass = false;
    if (dlss_chain.spatial_only) {
        dlss_chain.pending.clear();
        using namespace VideoCommon::FireEmblemUpscale;
        if (UseReplacement(StarFoxProgramId, StarFoxOutputProgramKey,
                           std::getenv("EDEN_NX2_DLSS_REPLACEMENT"))) {
            return TryFireEmblemUpscale(false);
        }
        dlss_motion_vectors.reset();
        dlss_depth.reset();
        (void)fe_upscale_jitter.Take();
        return false;
    }
    dlss_chain.programs.insert(dlss_chain.pending.begin(), dlss_chain.pending.end());
    dlss_chain.pending.clear();
    const bool complete = dlss_chain.rejected == 0;
    dlss_chain.clean_evaluations = complete ? dlss_chain.clean_evaluations + 1 : 0;
    ++dlss_chain.evaluations;
    using VideoCore::GameDlssReason;
    const GameDlssReason blocker = GameDlssBlocker();
    const bool can_show =
        blocker == GameDlssReason::None && dlss_chain.failure == GameDlssReason::None;
    // Checked before the replacement upscale draws over the output. Checks start over whenever
    // the network can't run.
    if (complete && can_show) {
        CheckDlssNetwork();
    } else {
        dlss_chain.good_checks = 0;
        dlss_chain.bad_checks = 0;
    }
    bool native = complete && can_show && dlss_chain.failure == GameDlssReason::None;
    if (!dlss_chain.native) {
        native = native && dlss_chain.clean_evaluations >= DLSS_NETWORK_CLEAN_EVALUATIONS &&
                 (!dlss_chain.checked || dlss_chain.good_checks >= DLSS_NETWORK_GOOD_CHECKS);
    }
    if (native != dlss_chain.native) {
        dlss_chain.native = native;
        if (native) {
            LOG_INFO(HW_GPU, "Game DLSS network: every pass runs{}, its own output is shown",
                     dlss_chain.checked ? " and its output checks out" : "");
        }
    }
    UpdateGameDlssStatus(complete, blocker);
    if (native) {
        dlss_motion_vectors.reset();
        dlss_depth.reset();
        (void)fe_upscale_jitter.Take();
        return false;
    }
    return TryFireEmblemUpscale();
}

// Logs the images a rejected program's handles (constant bank 0, bytes 0x80-0x11c) point to, so a
// substitute (e.g. an upscale blit for a tensor-core pass) can find its inputs and output.
void KeplerCompute::DescribeNativeResources(const NativeQmdScheduler::Descriptor& qmd) {
    static u32 described{};
    if (described++ >= 2) return;
    const auto& cbuf = launch_description.const_buffer_config[0];
    const GPUVAddr cbuf_address = (GPUVAddr{cbuf.address_high} << 32) | cbuf.address_low;
    LOG_INFO(HW_GPU, "Native resources: cbuf0 {:#x} size {:#x}, grid {}x{}x{}, block {}x{}x{}, tic pool {:#x}",
             cbuf_address, cbuf.size.Value(), qmd[12], qmd[13] & 0xffff, qmd[14] & 0xffff,
             launch_description.block_dim_x.Value(), launch_description.block_dim_y.Value(),
             launch_description.block_dim_z.Value(), regs.tic.Address());
    for (u32 offset = 0x80; offset < 0x120; offset += 4) {
        const u32 handle = memory_manager.Read<u32>(cbuf_address + offset);
        const u32 tic_index = handle & 0xfffff;
        if (handle == 0 || tic_index > regs.tic.limit) continue;
        const auto tic = GetTICEntry(tic_index);
        LOG_INFO(HW_GPU, "  handle @{:#x} = {:#x}: tic {} address {:#x} {}x{} format {:#x} types {}/{}/{}/{}",
                 offset, handle, tic_index, tic.Address(), tic.Width(), tic.Height(),
                 static_cast<u32>(tic.format.Value()), static_cast<u32>(tic.r_type.Value()),
                 static_cast<u32>(tic.g_type.Value()), static_cast<u32>(tic.b_type.Value()),
                 static_cast<u32>(tic.a_type.Value()));
    }
}

std::optional<Texture::TICEntry> KeplerCompute::ConstBuffer0Image(u32 offset) const {
    const auto& cbuf = launch_description.const_buffer_config[0];
    const GPUVAddr cbuf_address = (GPUVAddr{cbuf.address_high} << 32) | cbuf.address_low;
    if (!(launch_description.const_buffer_enable_mask & 1) || cbuf.size.Value() < offset + 4 ||
        !memory_manager.IsFullyMappedRange(cbuf_address + offset, 4)) {
        return std::nullopt;
    }
    const u32 tic_index = memory_manager.Read<u32>(cbuf_address + offset) & 0xfffff;
    if (tic_index > regs.tic.limit ||
        !memory_manager.IsFullyMappedRange(regs.tic.Address() +
                                               u64{tic_index} * sizeof(Texture::TICEntry),
                                           sizeof(Texture::TICEntry))) {
        return std::nullopt;
    }
    return GetTICEntry(tic_index);
}

bool KeplerCompute::BypassOotNetwork(u64 program_id, u64 native_key) {
    using namespace VideoCommon::OotNetwork;
    if (!BypassEnabled(program_id)) {
        return false;
    }
    auto* const state = rasterizer->OotNetworkBypass();
    if (!state) {
        return false;
    }
    // The tonemap pass draws before the network every frame; until it has been seen (menus, the
    // first frame) the network runs as usual. Its first passes are skipped only while the output
    // pass is being replaced, so a failed replacement brings the whole network back next frame.
    Maxwell3D::Regs::RenderTargetConfig tonemapped;
    {
        std::scoped_lock lock{state->mutex};
        tonemapped = state->tonemapped;
    }
    if (tonemapped.Address() == 0) {
        return false;
    }
    if (native_key == PrepassProgramKey || native_key == NetworkProgramKey) {
        return state->output_replaced.load(std::memory_order_relaxed);
    }
    if (native_key != OutputProgramKey) {
        return false;
    }
    // The copy needs the tonemapped image to be a B10G11R11 block-linear image of the output's size.
    const auto output = ConstBuffer0Image(OutputImageHandleOffset);
    if (!output || output->format != Texture::TextureFormat::B10G11R11 || output->Address() == 0 ||
        output->Address() == tonemapped.Address() || !output->IsBlockLinear() ||
        tonemapped.format != Tegra::RenderTargetFormat::B10G11R11_FLOAT ||
        tonemapped.tile_mode.is_pitch_linear || tonemapped.width != output->Width() ||
        tonemapped.height != output->Height()) {
        state->output_replaced.store(false, std::memory_order_relaxed);
        static bool warned = false;
        if (!std::exchange(warned, true)) {
            LOG_WARNING(HW_GPU,
                        "Ocarina of Time: unexpected network images; not replaced (tonemapped {}x{}, "
                        "output {})",
                        tonemapped.width, tonemapped.height,
                        output ? fmt::format("{}x{}", output->Width(), output->Height())
                               : std::string{"unreadable"});
        }
        return false;
    }
    Fermi2D::Surface destination{};
    destination.format = Tegra::RenderTargetFormat::B10G11R11_FLOAT;
    destination.linear = Fermi2D::MemoryLayout::BlockLinear;
    destination.block_width.Assign(output->block_width);
    destination.block_height.Assign(output->block_height);
    destination.block_depth.Assign(output->block_depth);
    destination.depth = 1;
    destination.width = output->Width();
    destination.height = output->Height();
    destination.addr_upper = static_cast<u32>(output->Address() >> 32);
    destination.addr_lower = static_cast<u32>(output->Address());
    Fermi2D::Surface source = destination;
    source.block_width.Assign(tonemapped.tile_mode.block_width);
    source.block_height.Assign(tonemapped.tile_mode.block_height);
    source.block_depth.Assign(tonemapped.tile_mode.block_depth);
    source.addr_upper = tonemapped.address_high;
    source.addr_lower = tonemapped.address_low;
    const s32 width = static_cast<s32>(output->Width());
    const s32 height = static_cast<s32>(output->Height());
    const Fermi2D::Config config{
        .operation = Fermi2D::Operation::SrcCopy,
        .filter = Fermi2D::Filter::Point,
        .must_accelerate = true,
        .dst_x0 = 0,
        .dst_y0 = 0,
        .dst_x1 = width,
        .dst_y1 = height,
        .src_x0 = 0,
        .src_y0 = 0,
        .src_x1 = width,
        .src_y1 = height,
    };
    const bool copied = rasterizer->AccelerateSurfaceCopy(source, destination, config);
    state->output_replaced.store(copied, std::memory_order_relaxed);
    static bool logged = false;
    if (!std::exchange(logged, true)) {
        LOG_INFO(HW_GPU,
                 "Ocarina of Time: anti-aliasing network bypassed ({}): {:#x} -> {:#x} {}x{}",
                 copied ? "copy of the tonemapped image" : "copy failed", tonemapped.Address(),
                 output->Address(), width, height);
    }
    return copied;
}

std::atomic<u32> g_nx2_trace_draws_after_upscale{};

// Fire Emblem: Fortune's Weave's first DLSS pass (rejected, before the network) has the input colour
// at cbuf0+0x278, its motion vectors (R16G16 float, the input's size) at +0x290, its depth (Z32, the
// input's size) at +0x298 and this frame's camera jitter (x, y; input pixels, Halton bases 2 and 3
// less 0.5) at +0x1b0.
void KeplerCompute::RememberDlssMotionVectors() {
    dlss_motion_vectors.reset();
    dlss_depth.reset();
    const auto& cbuf = launch_description.const_buffer_config[0];
    const GPUVAddr cbuf_address = (GPUVAddr{cbuf.address_high} << 32) | cbuf.address_low;
    if (!(launch_description.const_buffer_enable_mask & 1) || cbuf.size.Value() < 0x294 ||
        !memory_manager.IsFullyMappedRange(cbuf_address, cbuf.size.Value())) {
        return;
    }
    const auto image = [&](u32 offset) -> std::optional<Texture::TICEntry> {
        const u32 tic_index = memory_manager.Read<u32>(cbuf_address + offset) & 0xfffff;
        if (tic_index > regs.tic.limit || !memory_manager.IsFullyMappedRange(
                regs.tic.Address() + u64{tic_index} * sizeof(Texture::TICEntry),
                sizeof(Texture::TICEntry))) {
            return std::nullopt;
        }
        return GetTICEntry(tic_index);
    };
    const auto color = image(0x278), motion = image(0x290);
    if (!color || !motion || color->format != Texture::TextureFormat::B10G11R11 ||
        motion->format != Texture::TextureFormat::R16G16 || motion->Address() == 0 ||
        motion->Width() != color->Width() || motion->Height() != color->Height() ||
        !motion->IsBlockLinear())
        return;
    dlss_motion_vectors = *motion;
    dlss_motion_color = color->Address();
    const auto depth = cbuf.size.Value() >= 0x29c ? image(0x298) : std::nullopt;
    if (depth && depth->format == Texture::TextureFormat::Z32 && depth->Address() != 0 &&
        depth->Width() == color->Width() && depth->Height() == color->Height() &&
        depth->IsBlockLinear()) {
        dlss_depth = *depth;
    } else {
        dlss_depth.reset();
    }
}

void KeplerCompute::CaptureFireEmblemJitter() {
    using namespace VideoCommon::FireEmblemUpscale;
    (void)fe_upscale_jitter.Take();
    dlss_motion_vectors.reset();
    dlss_depth.reset();
    const auto& cbuf = launch_description.const_buffer_config[0];
    const GPUVAddr address = (GPUVAddr{cbuf.address_high} << 32) | cbuf.address_low;
    if (!(launch_description.const_buffer_enable_mask & 1) || cbuf.size.Value() < 0x27c ||
        !memory_manager.IsFullyMappedRange(address, 0x27c)) {
        return;
    }
    const u32 index = memory_manager.Read<u32>(address + PrepassColorOffset) & 0xfffff;
    if (index > regs.tic.limit || !memory_manager.IsFullyMappedRange(
            regs.tic.Address() + u64{index} * sizeof(Texture::TICEntry), sizeof(Texture::TICEntry))) {
        return;
    }
    const auto color = GetTICEntry(index);
    if (!ValidInput(color, dlss_chain.spatial_only)) {
        return;
    }
    fe_upscale_jitter.Capture(color.Address(), color.Width(), color.Height(),
        {std::bit_cast<f32>(memory_manager.Read<u32>(address + JitterOffset)),
         std::bit_cast<f32>(memory_manager.Read<u32>(address + JitterOffset + sizeof(f32)))});
    if (!dlss_chain.spatial_only) {
        RememberDlssMotionVectors();
    }
}

void KeplerCompute::TraceDlssContract(u64 native_key, bool warn_unvalidated) {
    static bool reported{};
    if (warn_unvalidated && !reported) {
        reported = true;
        LOG_WARNING(HW_GPU, "Fast Fusion temporal replacement is not selected: motion, depth, jitter and reset contract is unvalidated; native dispatches retained. EDEN_NX2_DLSS_CONTRACT_TRACE enables bounded evidence capture");
    }
    static const bool enabled = std::getenv("EDEN_NX2_DLSS_CONTRACT_TRACE") != nullptr;
    static const u32 skip = [] {
        const char* value = std::getenv("EDEN_NX2_DLSS_CONTRACT_SKIP");
        return value ? static_cast<u32>(std::min<unsigned long>(std::strtoul(value, nullptr, 10), 1000000UL)) : 0U;
    }();
    if (!enabled || dlss_contract_trace_count >= skip + 24) {
        return;
    }
    if (dlss_contract_trace_count++ < skip) {
        return;
    }
    const auto& cbuf = launch_description.const_buffer_config[0];
    const GPUVAddr address = (GPUVAddr{cbuf.address_high} << 32) | cbuf.address_low;
    const u32 bytes = std::min<u32>(cbuf.size.Value(), 0x360U);
    if (!(launch_description.const_buffer_enable_mask & 1) || bytes < 0x100 ||
        !memory_manager.IsFullyMappedRange(address, bytes)) {
        return;
    }
    LOG_INFO(HW_GPU, "DLSS contract candidate {:016x}: cb0={:#x} bytes={:#x} grid={}x{}x{} block={}x{}x{} native dispatch retained",
             native_key, address, bytes, launch_description.grid_dim_x.Value(),
             launch_description.grid_dim_y.Value(), launch_description.grid_dim_z.Value(),
             launch_description.block_dim_x.Value(), launch_description.block_dim_y.Value(),
             launch_description.block_dim_z.Value());
    for (u32 offset = 0x100; offset + 16 <= bytes; offset += 16) {
        std::array<u32, 4> words{};
        memory_manager.ReadBlock(address + offset, words.data(), sizeof(words));
        LOG_INFO(HW_GPU, "DLSS contract cb0+{:03x}: {:08x}", offset, fmt::join(words, " "));
    }
    for (u32 offset = 0x278; offset + 4 <= bytes; offset += 4) {
        const u32 handle = memory_manager.Read<u32>(address + offset);
        const u32 index = handle & 0xfffff;
        const GPUVAddr tic_address = regs.tic.Address() + u64{index} * sizeof(Texture::TICEntry);
        if (!handle || index > regs.tic.limit ||
            !memory_manager.IsFullyMappedRange(tic_address, sizeof(Texture::TICEntry))) {
            continue;
        }
        const auto tic = GetTICEntry(index);
        std::array<u32, sizeof(Texture::TICEntry) / sizeof(u32)> words{};
        std::memcpy(words.data(), &tic, sizeof(tic));
        LOG_INFO(HW_GPU, "DLSS contract handle+{:03x}={:08x}: tic={} addr={:#x} {}x{} raw={:08x}",
                 offset, handle, index, tic.Address(), tic.Width(), tic.Height(),
                 fmt::join(words, " "));
    }
}

bool KeplerCompute::TryFireEmblemUpscale(bool allow_history) {
    using namespace VideoCommon::FireEmblemUpscale;
    static const bool requested_history = [] {
        const char* const option = std::getenv("EDEN_NX2_DLSS_REPLACEMENT");
        return option && (std::string_view{option} == "fsr" || std::string_view{option} == "dlss");
    }();
    const bool use_history = allow_history && requested_history;
    const size_t game_index = dlss_chain.spatial_only ? 1 : 0;
    const char* const game = dlss_chain.spatial_only ? "Star Fox" : "Fire Emblem";
    const auto jitter = fe_upscale_jitter.Take();
    const auto motion = std::exchange(dlss_motion_vectors, std::nullopt);
    const auto depth = std::exchange(dlss_depth, std::nullopt);
    if (!jitter || (use_history && (!motion || !depth))) {
        static std::array<bool, 2> reported{};
        if (!reported[game_index]) {
            reported[game_index] = true;
            LOG_WARNING(HW_GPU, "{} replacement lacks current inputs: jitter={} motion={} depth={} temporal={}; retaining native dispatch",
                        game, jitter.has_value(), motion.has_value(), depth.has_value(), use_history);
        }
        return false;
    }
    const auto& cbuf = launch_description.const_buffer_config[0];
    const GPUVAddr address = (GPUVAddr{cbuf.address_high} << 32) | cbuf.address_low;
    if (!(launch_description.const_buffer_enable_mask & 1) || cbuf.size.Value() < 0x34c ||
        !memory_manager.IsFullyMappedRange(address, 0x34c)) {
        return false;
    }
    const auto image = [&](u32 offset) -> std::optional<Texture::TICEntry> {
        const u32 index = memory_manager.Read<u32>(address + offset) & 0xfffff;
        if (index > regs.tic.limit || !memory_manager.IsFullyMappedRange(
                regs.tic.Address() + u64{index} * sizeof(Texture::TICEntry), sizeof(Texture::TICEntry))) {
            return std::nullopt;
        }
        return GetTICEntry(index);
    };
    const auto input = image(OutputColorInputOffset);
    const auto output = image(OutputColorOffset);
    if (!input || !output || !ValidPair(*input, *output, dlss_chain.spatial_only) ||
        jitter->color != input->Address() || (use_history && dlss_motion_color != input->Address()) ||
        jitter->width != input->Width() || jitter->height != input->Height()) {
        return false;
    }
    const auto surface = [](const Texture::TICEntry& tic, Tegra::RenderTargetFormat format) {
        Fermi2D::Surface result{};
        result.format = format;
        result.linear = Fermi2D::MemoryLayout::BlockLinear;
        result.block_width.Assign(tic.block_width);
        result.block_height.Assign(tic.block_height);
        result.block_depth.Assign(tic.block_depth);
        result.depth = 1;
        result.width = tic.Width();
        result.height = tic.Height();
        result.addr_upper = static_cast<u32>(tic.Address() >> 32);
        result.addr_lower = static_cast<u32>(tic.Address());
        return result;
    };
    const auto input_format = dlss_chain.spatial_only ? Tegra::RenderTargetFormat::A8B8G8R8_SRGB
                                                     : Tegra::RenderTargetFormat::B10G11R11_FLOAT;
    const auto output_format = dlss_chain.spatial_only ? Tegra::RenderTargetFormat::R16G16B16A16_FLOAT
                                                      : Tegra::RenderTargetFormat::B10G11R11_FLOAT;
    const bool done = rasterizer->AccelerateTemporalUpscale(
        surface(*input, input_format), surface(*output, output_format),
        use_history ? surface(*motion, Tegra::RenderTargetFormat::R16G16_FLOAT) : Fermi2D::Surface{},
        use_history ? &*depth : nullptr, jitter->offset,
        use_history ? std::array{1.0f, 1.0f} : std::array{0.0f, 0.0f}, use_history);
    static std::array<u32, 2> logged{};
    if (logged[game_index]++ < 8) {
        LOG_INFO(HW_GPU, "{} DLSS replacement: {} temporal={} input={:#x} {}x{} output={:#x} {}x{} jitter={},{}",
                 game, done ? "applied" : "unavailable; retaining native dispatch", use_history,
                 input->Address(), input->Width(), input->Height(), output->Address(),
                 output->Width(), output->Height(), jitter->offset[0], jitter->offset[1]);
    }
    return done;
}

void KeplerCompute::ProcessLaunch() {
    const GPUVAddr launch_desc_loc = regs.launch_desc_loc.Address();
    memory_manager.ReadBlockUnsafe(launch_desc_loc, &launch_description,
                                   LaunchParams::NUM_LAUNCH_PARAMETERS * sizeof(u32));
    rasterizer->DispatchCompute();
}

Texture::TICEntry KeplerCompute::GetTICEntry(u32 tic_index) const {
    const GPUVAddr tic_address_gpu{regs.tic.Address() + tic_index * sizeof(Texture::TICEntry)};

    Texture::TICEntry tic_entry;
    memory_manager.ReadBlockUnsafe(tic_address_gpu, &tic_entry, sizeof(Texture::TICEntry));
    return tic_entry;
}

Texture::TSCEntry KeplerCompute::GetTSCEntry(u32 tsc_index) const {
    const GPUVAddr tsc_address_gpu{regs.tsc.Address() + tsc_index * sizeof(Texture::TSCEntry)};

    Texture::TSCEntry tsc_entry;
    memory_manager.ReadBlockUnsafe(tsc_address_gpu, &tsc_entry, sizeof(Texture::TSCEntry));
    return tsc_entry;
}

// Split GPU thread with macros on its front half (gpu_split.h, EDEN_NX2_SPLIT=2): the front
// runs compute macros against its own copy of the native registers (which hold the last value
// written to each, on the back too) and loads and binds the compute macro code itself.
struct KeplerCompute::FrontState {
    std::array<u32, 0x1000> registers{};
    std::optional<u32> executing_macro;
    std::vector<u32> parameters;
    bool compute_uploaded{};
    u32 bind_pointer{};
    MacroEngine* macro{}; ///< Without a native context: the one macro engine
    /// The inline upload's exec was seen by this front (its destination is known).
    bool upload_known{};
    bool upload_linear{};
    u32 upload_copy_size{};
};

KeplerCompute::FrontState& KeplerCompute::Front(VideoCommon::GpuSplit& split) {
    if (!front) [[unlikely]] {
        // Copy the back's state and take the macros over while it is idle.
        split.Drain(VideoCommon::DrainSite::FrontStart);
        front = std::make_unique<FrontState>();
        front->registers = native_registers;
        front->executing_macro = native_executing_macro;
        front->parameters = std::move(native_parameters);
        native_parameters.clear();
        native_executing_macro.reset();
        front->compute_uploaded = native_compute_uploaded;
        front->bind_pointer = native_bind_pointer;
        front->macro = native_macro;
        front_macros = true;
    }
    return *front;
}

// As Maxwell3D::ReleaseFront: the split stopped, the engine runs its macros again.
void KeplerCompute::ReleaseFront() {
    native_executing_macro = front->executing_macro;
    native_parameters = std::move(front->parameters);
    native_compute_uploaded = front->compute_uploaded;
    native_bind_pointer = front->bind_pointer;
    native_macro = front->macro;
    front.reset();
    front_macros = false;
    LOG_INFO(HW_GPU, "GPU thread split stopped: compute macros back on the engine");
}

void KeplerCompute::FrontApply(FrontState& fs, u32 method, u32 argument) {
    // As ProcessNativeMethod does with the macro code and bindings.
    if (method == 0x45 || method == 0x47) {
        fs.compute_uploaded = true;
    }
    fs.registers[method] = argument;
    switch (method) {
    case 0x45:
        FrontMacroEngine(fs).ClearCode(argument);
        break;
    case 0x46:
        FrontMacroEngine(fs).AddCode(0, argument);
        break;
    case 0x47:
        fs.bind_pointer = argument;
        break;
    case 0x48: {
        const auto positions = FrontMacroPositions(fs);
        if (fs.bind_pointer < positions.size()) {
            positions[fs.bind_pointer++] = argument;
        } else {
            LOG_ERROR(HW_GPU, "Native compute macro bind out of bounds");
        }
        break;
    }
    default:
        break;
    }
}

MacroEngine& KeplerCompute::FrontMacroEngine(FrontState& fs) {
    if (native_context) {
        return fs.compute_uploaded ? native_context->compute_engine : native_context->engine;
    }
    return *fs.macro;
}

std::span<u32> KeplerCompute::FrontMacroPositions(FrontState& fs) {
    if (native_context) {
        return fs.compute_uploaded ? std::span<u32>{native_context->compute_positions}
                                      : std::span<u32>{native_context->positions};
    }
    return native_macro_positions;
}

void KeplerCompute::FrontCallMethod(Core::System& system, VideoCommon::GpuSplit& split,
                                    u32 method, u32 argument, bool is_last_call,
                                    const GPUVAddr* segment) {
    auto& fs = Front(split);
    if (method >= fs.registers.size()) {
        split.Method(this, method, argument, is_last_call, segment);
        if (split.NarrowCompute()) {
            split.NoteUnboundedWrite(true, VideoCommon::UnboundedCause::ComputeOther);
        }
        return;
    }
    if (method >= 0xe00) {
        if (!fs.executing_macro) {
            if (method & 1) {
                LOG_ERROR(HW_GPU, "Native compute macro starts on argument method");
                return;
            }
            fs.executing_macro = method;
        }
        fs.parameters.push_back(argument);
        if (is_last_call) {
            FrontRunMacro(system, split, fs);
        }
        return;
    }
    FrontApply(fs, method, argument);
    FrontQueueWrite(split, fs, method, argument, is_last_call, false, segment);
}

void KeplerCompute::FrontCallMultiMethod(Core::System& system, VideoCommon::GpuSplit& split,
                                         u32 method, const u32* base_start, u32 amount,
                                         u32 methods_pending, const GPUVAddr* segment) {
    auto& fs = Front(split);
    // EDEN_NX2_QMD_PREDICT: methods that signal the QMD scheduler or release go one by one, so
    // FrontNoteWrites sees each (the back runs a multi-method word by word too).
    const bool signals = method == 0xae || method == 0xaf || method == 0xb0 || method == 0x107 ||
                         method == 0x6c3;
    if ((method >= 0xe00 && method < fs.registers.size()) ||
        (signals && amount > 1 && split.QmdPredict())) {
        for (u32 i = 0; i < amount; ++i) {
            FrontCallMethod(system, split, method, base_start[i], methods_pending - i <= 1, segment);
        }
        return;
    }
    if (method < fs.registers.size()) {
        for (u32 i = 0; i < amount; ++i) {
            FrontApply(fs, method, base_start[i]);
        }
    }
    split.MultiMethod(this, method, base_start, amount, methods_pending, segment);
    if (method < fs.registers.size()) {
        FrontNoteWrites(split, fs, method);
    } else if (split.NarrowCompute()) {
        split.NoteUnboundedWrite(true, VideoCommon::UnboundedCause::ComputeOther);
    }
}

void KeplerCompute::FrontRunMacro(Core::System& system, VideoCommon::GpuSplit& split,
                                  FrontState& fs) {
    const auto positions = FrontMacroPositions(fs);
    const u32 start = positions[((*fs.executing_macro - 0xe00) / 2) % positions.size()];
    fs.executing_macro.reset();
    auto& macro = FrontMacroEngine(fs);
    if (native_data_ram) {
        macro.native_interpreter.active_dram = native_data_ram;
    }
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Macro};
    const auto read = [&fs](u32 address) {
        return address < fs.registers.size() ? fs.registers[address] : 0U;
    };
    const auto emit = [&](u32 address, u32 value) {
        if (address >= 0xe00) {
            return false;
        }
        FrontApply(fs, address, value);
        FrontQueueWrite(split, fs, address, value, true, true, nullptr);
        return true;
    };
    // EDEN_NX2_COMPUTE_FUSE: the macro's upload data words may stay held until it returns.
    front_run.in_macro = true;
#ifdef ARCHITECTURE_x86_64
    if (!macro.native_jit) {
        macro.native_jit = std::make_shared<TuringMacroJit>();
    }
    macro.native_jit->Prepare(macro.native_code, macro.native_code_generation);
    const auto result = macro.native_jit->Execute(
        start, fs.parameters, std::span<u32>{*macro.native_interpreter.active_dram}, read, emit);
#else
    const auto result =
        macro.native_interpreter.Execute(macro.native_code, start, fs.parameters, read, emit);
#endif
    if (!result.success) {
        macro.native_execution_failed = true;
        LOG_ERROR(HW_GPU, "Native MME (split front) failed: compute start={:#x}, consumed={}, error={}",
                  start, result.consumed, result.error);
    }
    fs.parameters.clear();
    // Held data words go now, with their notes: what the front does next may read what they
    // write without sending the held work first (a semaphore prediction).
    front_run.in_macro = false;
    if (front_run.note_count != 0) {
        split.RunPendingFlush();
    }
}

void KeplerCompute::FrontQueueWrite(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                                    u32 argument, bool is_last_call, bool direct,
                                    const GPUVAddr* segment) {
    using VideoCommon::GpuSplit;
    constexpr u32 exec_upload_method = KEPLER_COMPUTE_REG_INDEX(exec_upload);
    constexpr u32 data_upload_method = KEPLER_COMPUTE_REG_INDEX(data_upload);
    constexpr u32 upload_method = KEPLER_COMPUTE_REG_INDEX(upload);
    // Narrow compute only: without it DmaPusher notes every memory-writing compute method as an
    // unbounded write at the position after the call, which held work would not have reached.
    if (!g_compute_fuse || !split.NarrowCompute() || method >= native_plain_methods.size()) {
        if (direct) {
            split.MacroWrite(this, method, argument);
        } else {
            split.Method(this, method, argument, is_last_call, segment);
        }
        FrontNoteWrites(split, fs, method);
        return;
    }
    FrontWriteRun& run = front_run;
    if (run.session != split.Session()) {
        // A run left from before the split stopped never reaches the new ring.
        run.count = 0;
        run.note_count = 0;
        run.has_segment = false;
        run.writer = false;
        run.session = split.Session();
    }
    // Register stores and execs write no memory: nothing reads what they do before the run goes.
    const bool plain = native_plain_methods[method];
    const bool exec = method == exec_upload_method;
    // A data word with a known destination inside a macro call: noted when the run goes.
    const bool hold_data = method == data_upload_method && run.in_macro && fs.upload_known;
    if (run.count == FrontWriteRun::MaxWrites ||
        (hold_data && run.note_count == FrontWriteRun::MaxNotes)) {
        split.RunPendingFlush();
    }
    run.writes[run.count++] = u64{method} | (is_last_call ? GpuSplit::WriteIsLast : 0) |
                              (direct ? GpuSplit::WriteDirect : 0) | (u64{argument} << 32);
    if (segment != nullptr) {
        run.segment = *segment;
        run.has_segment = true;
    }
    run.writer |= !plain;
    // Another holder's work goes first: it was sent before this write.
    split.SetPendingFlush(&FrontFlushWrites, this);
    if (plain || exec) {
        // FrontNoteWrites only keeps the front's upload state for these.
        FrontNoteWrites(split, fs, method);
        return;
    }
    if (hold_data) {
        FrontWriteRun::UploadNote note{};
        std::memcpy(note.registers.data(), fs.registers.data() + upload_method,
                    sizeof(note.registers));
        note.linear = fs.upload_linear;
        note.copy_size = fs.upload_copy_size;
        // The words of one transfer note the same range: the last of them is enough. Only the
        // data word itself, the last register, may differ.
        if (run.note_count != 0) {
            auto& last = run.notes[run.note_count - 1];
            if (last.linear == note.linear && last.copy_size == note.copy_size &&
                std::memcmp(last.registers.data(), note.registers.data(),
                            sizeof(note.registers) - sizeof(u32)) == 0) {
                last = note;
                return;
            }
        }
        run.notes[run.note_count++] = note;
        return;
    }
    // Launches, releases, inline QMDs, macro loads, data words outside macros: the run goes now,
    // then the method is noted at the position after it, as when it was its own entry.
    split.RunPendingFlush();
    FrontNoteWrites(split, fs, method);
}

void KeplerCompute::FrontFlushWrites(void* engine) {
    // GpuSplit cleared the hook before calling: the entry below doesn't come back here.
    auto* const self = static_cast<KeplerCompute*>(engine);
    FrontWriteRun& run = self->front_run;
    const u32 count = run.count;
    if (count == 0) {
        return;
    }
    auto& split = VideoCommon::GpuSplit::Get();
    run.count = 0;
    const GPUVAddr segment = run.segment;
    split.Writes(self, run.writes.data(), count, run.has_segment ? &segment : nullptr);
    run.has_segment = false;
    if (run.writer) {
        // As DmaPusher after each memory-writing compute method: WaitForWriters covers it.
        run.writer = false;
        split.NoteWriter();
    }
    const u32 notes = run.note_count;
    run.note_count = 0;
    if (notes == 0 || !self->front) {
        return;
    }
    // FrontNoteWrites for each held data word's transfer, against the registers it saw.
    constexpr u32 upload_method = KEPLER_COMPUTE_REG_INDEX(upload);
    FrontState& fs = *self->front;
    FrontWriteRun::UploadNote saved{};
    std::memcpy(saved.registers.data(), fs.registers.data() + upload_method,
                sizeof(saved.registers));
    saved.linear = fs.upload_linear;
    saved.copy_size = fs.upload_copy_size;
    const bool saved_known = fs.upload_known;
    for (u32 i = 0; i < notes; ++i) {
        const auto& note = run.notes[i];
        std::memcpy(fs.registers.data() + upload_method, note.registers.data(),
                    sizeof(note.registers));
        fs.upload_linear = note.linear;
        fs.upload_copy_size = note.copy_size;
        fs.upload_known = true;
        self->FrontNoteWrites(split, fs, KEPLER_COMPUTE_REG_INDEX(data_upload));
    }
    std::memcpy(fs.registers.data() + upload_method, saved.registers.data(),
                sizeof(saved.registers));
    fs.upload_linear = saved.linear;
    fs.upload_copy_size = saved.copy_size;
    fs.upload_known = saved_known;
}

void KeplerCompute::BackWrites(Core::System& system, const u64* writes, u32 count) {
    using VideoCommon::GpuSplit;
    constexpr u32 exec_upload_method = KEPLER_COMPUTE_REG_INDEX(exec_upload);
    constexpr u32 data_upload_method = KEPLER_COMPUTE_REG_INDEX(data_upload);
    // EDEN_NX2_COMPUTE_I2M_MERGE: while the run only stores registers and runs inline uploads,
    // which read no memory, a flush continuing the previous one joins it (Upload::State::
    // HoldWrites); the joined bytes go before anything else runs and at the end of the run.
    const bool merge = g_compute_i2m_merge && native_mode;
    bool holding = false;
    for (u32 i = 0; i < count; ++i) {
        const u64 write = writes[i];
        const u32 method = static_cast<u32>(write & 0xffff);
        const u32 argument = static_cast<u32>(write >> 32);
        // As GpuSplit::Execute for a Method entry (MacroWrite: Direct).
        if ((write & GpuSplit::WriteDirect) == 0 && !execution_mask[method]) {
            method_sink.emplace_back(method, argument);
            continue;
        }
        if (merge) {
            const bool upload = method == exec_upload_method || method == data_upload_method ||
                                (method < native_plain_methods.size() &&
                                 native_plain_methods[method]);
            if (upload != holding) {
                holding = upload;
                upload_state.HoldWrites(holding);
            }
        }
        ConsumeSink(system);
        CallMethod(system, method, argument, (write & GpuSplit::WriteIsLast) != 0);
    }
    if (holding) {
        upload_state.HoldWrites(false);
    }
}

void KeplerCompute::FrontNoteWrites(VideoCommon::GpuSplit& split, FrontState& fs, u32 method) {
    // EDEN_NX2_NARROW_COMPUTE: instead of every compute macro word and memory-writing method
    // counting as unbounded (DmaPusher::FrontBackWrites), what ProcessNativeMethod writes.
    if (!split.NarrowCompute()) {
        return;
    }
    using VideoCommon::UnboundedCause;
    constexpr u32 exec_upload_method = KEPLER_COMPUTE_REG_INDEX(exec_upload);
    constexpr u32 data_upload_method = KEPLER_COMPUTE_REG_INDEX(data_upload);
    switch (method) {
    case 0xae:
    case 0xaf:
    case 0xb0: {
        if (FrontQmdPredicts(split)) {
            const u32 argument = fs.registers[method];
            // As ProcessNativeMethod: these signal nothing.
            if (method == 0xae || (method == 0xaf && (argument & 2) == 0) ||
                (method == 0xb0 && ((argument & 15) == 0 || (argument & 15) == 1))) {
                break;
            }
            if (FrontQmdSignal(split, GPUVAddr{fs.registers[0xad]} << 8, argument)) {
                break;
            }
        }
        split.NoteDispatch(UnboundedCause::ComputeLaunch);
        break;
    }
    case 0x107: {
        // The inline QMD goes to memory, then may launch.
        const GPUVAddr address =
            ((GPUVAddr{fs.registers[0xc6]} << 32) | fs.registers[0xc7]) << 8;
        split.NoteWrite(&memory_manager, address, 256);
        if (FrontQmdPredicts(split)) {
            // As ProcessNativeMethod: unmapped, it neither writes nor launches.
            if (!memory_manager.IsFullyMappedRange(address, 256)) {
                break;
            }
            g_front_qmd.AddOverlay(&memory_manager, address, split.WritePosition(),
                                   fs.registers.data() + 0xc8);
            if ((fs.registers[0xa7] & (1U << 24)) == 0 && (fs.registers[0xcc] & (1U << 12)) != 0) {
                break; // Launched later
            }
            if (FrontQmdSignal(split, address, 3)) {
                break;
            }
        }
        split.NoteDispatch(UnboundedCause::ComputeLaunch);
        break;
    }
    case 0x6c3: {
        // ProcessNativeReportSemaphore.
        const GPUVAddr address =
            (GPUVAddr{fs.registers[0x6c0] & 0x1ffffff} << 32) | fs.registers[0x6c1];
        const u32 argument = fs.registers[0x6c3];
        // Bit 28: a one-word structure, without the time stamp.
        split.NoteWrite(&memory_manager, address, ((argument >> 28) & 1) != 0 ? 4 : 16, true);
        const u32 operation = argument & 3;
        if ((operation == 0 || operation == 2) && ((argument >> 3) & 1) == 0) {
            split.NoteRelease(&memory_manager, address, fs.registers[0x6c2]);
        } else {
            split.ForgetRelease(&memory_manager, address);
        }
        break;
    }
    case exec_upload_method:
        // As Upload::State::ProcessExec (bit 0: pitch layout).
        fs.upload_known = true;
        fs.upload_linear = (fs.registers[exec_upload_method] & 1) != 0;
        fs.upload_copy_size = fs.registers[KEPLER_COMPUTE_REG_INDEX(upload.line_length_in)] *
                              fs.registers[KEPLER_COMPUTE_REG_INDEX(upload.line_count)];
        break;
    case data_upload_method: {
        if (!fs.upload_known) {
            split.NoteUnboundedWrite(true, UnboundedCause::ComputeUpload);
            break;
        }
        Upload::Registers upload{};
        std::memcpy(&upload, fs.registers.data() + KEPLER_COMPUTE_REG_INDEX(upload),
                    sizeof(upload));
        const auto [begin, end] = Upload::WriteRange(upload, fs.upload_linear, fs.upload_copy_size);
        split.NoteWrite(&memory_manager, begin, end - begin);
        break;
    }
    default:
        // Register stores, macro uploads and bindings.
        break;
    }
}

bool KeplerCompute::FrontQmdPredicts(VideoCommon::GpuSplit& split) {
    return split.QmdPredict() && !g_front_qmd.disabled;
}

bool KeplerCompute::FrontQmdSignal(VideoCommon::GpuSplit& split, GPUVAddr qmd_address,
                                   u32 argument) {
    auto& mirror = g_front_qmd;
    if (mirror.session != split.Session()) {
        // First signal of this split session: start from the back's table, this one included.
        ResyncFrontQmd(split);
        return true;
    }
    if (!FrontCompareReleases(split)) {
        DisableFrontQmd(split);
        return false;
    }
    bool unknown = false;
    bool grid = false;
    const auto read = [&](u64 address, NativeQmdScheduler::Descriptor& descriptor) {
        return FrontReadQmd(split, address, descriptor, unknown);
    };
    // As ProcessNativeMethod's release callback: a reduction (bit 26), else a four-word release
    // (payload and timestamp), a max (bit 19) or a plain write of the payload.
    const auto release = [&](u64 address, u32 payload, u32 flags) {
        const bool four_words = (flags >> 30) == 0;
        const u64 size = four_words ? 16 : sizeof(payload);
        if (!memory_manager.IsFullyMappedRange(address, size)) {
            return false;
        }
        FrontRecordRelease(address, payload, flags, split.WritePosition());
        split.NoteWrite(&memory_manager, address, size, true);
        if ((flags & (1U << 26)) != 0) {
            split.ForgetRelease(&memory_manager, address);
        } else {
            split.NoteRelease(&memory_manager, address, payload, (flags & (1U << 19)) != 0);
        }
        return true;
    };
    const auto dispatch = [&](const NativeQmdScheduler::Descriptor&) {
        grid = true;
        return true;
    };
    (void)mirror.scheduler.Signal(qmd_address, argument, read, release, dispatch);
    if (unknown) {
        // The back's run of this signal may differ from the copy's: take its table.
        ResyncFrontQmd(split);
        ++mirror.window_resyncs;
    } else if (grid) {
        split.NoteComputeGrid();
    }
    // QMDs the front only reads after waiting for the back (written by queued copies or
    // uploads other than inline QMDs) or on GPU-written pages (each a drain) would make
    // launches wait: the old way is cheaper then.
    // EDEN_NX2_QMD_AUTO_OFF=0 keeps predicting however many reads wait (drains still count).
    static const bool auto_off = VideoCommon::Nx2Switch("EDEN_NX2_QMD_AUTO_OFF", true);
    if (mirror.window_reads >= 256 || mirror.window_resyncs > 8) {
        if ((auto_off && mirror.window_waits * 8 > mirror.window_reads) ||
            mirror.window_resyncs > 8) {
            LOG_WARNING(HW_GPU,
                        "QMD release prediction: {} of {} QMD reads waited for queued writes, "
                        "{} drains; predictions off (EDEN_NX2_QMD_PREDICT)",
                        mirror.window_waits, mirror.window_reads, mirror.window_resyncs);
            DisableFrontQmd(split);
        }
        mirror.window_reads = 0;
        mirror.window_waits = 0;
        mirror.window_resyncs = 0;
    }
    return true;
}

bool KeplerCompute::FrontReadQmd(VideoCommon::GpuSplit& split, u64 address,
                                 NativeQmdScheduler::Descriptor& descriptor, bool& unknown) {
    constexpr u64 size = sizeof(NativeQmdScheduler::Descriptor);
    if (!memory_manager.IsFullyMappedRange(address, size)) {
        return false; // As the back ("unmapped QMD")
    }
    // GPU-written words may land from the buffer cache at any time.
    if (memory_manager.Nx2ReadNeedsDrain(address, size)) {
        unknown = true;
        return false;
    }
    auto& mirror = g_front_qmd;
    ++mirror.window_reads;
    const u64 target = split.GuestReadTarget(&memory_manager, address, size);
    const u64 consumed = split.ConsumedPosition();
    if (target > consumed) {
        // The inline QMD written last over it, with nothing queued after it that may write
        // there: the back finds exactly its words (0x107 writes all 256 bytes).
        const FrontQmdMirror::Overlay* const overlay =
            mirror.FindOverlay(&memory_manager, address);
        if (overlay != nullptr && overlay->position > consumed && target <= overlay->position) {
            descriptor = overlay->words;
        } else {
            ++mirror.window_waits;
            split.WaitForGuestRead(target);
            memory_manager.ReadBlockUnsafe(address, descriptor.data(), size);
        }
    } else {
        memory_manager.ReadBlockUnsafe(address, descriptor.data(), size);
    }
    // The back flushes the buffer cache over an all-zero QMD and reads it again.
    if (std::ranges::all_of(descriptor, [](u32 word) { return word == 0; })) {
        unknown = true;
        return false;
    }
    return true;
}

} // namespace Tegra::Engines
