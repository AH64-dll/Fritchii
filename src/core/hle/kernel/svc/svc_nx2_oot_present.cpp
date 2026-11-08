// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/logging.h"
#include "common/nx2_gpu_write_listener.h"
#include "common/nx2_timeline.h"
#include "core/arm/arm_interface.h"
#include "core/arm/debug.h"
#include "core/core.h"
#include "core/hle/kernel/k_hardware_timer.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_scheduler.h"
#include "core/hle/kernel/k_scoped_scheduler_lock_and_sleep.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/k_thread_queue.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/svc/svc_nx2_oot_present.h"
#include "core/hle/kernel/svc_results.h"
#include "core/loader/loader.h"
#include "core/memory.h"
#include "video_core/nx2.h"
#include "video_core/nx2_debug_control.h"

namespace Kernel::Svc {
namespace {

// Ocarina of Time. Found by Hookshot (NotRexed's OoT recomp, host/hos_thread.c present_poll):
// the main thread waits for its frame's GPU work with
//     while (nvnEventGetValue(e) != 1) nn::os::SleepThread(1 ms);
// ~3.5 sleeps a frame, so each frame ends 0.5-1 ms after the GPU wrote the event. In the loop a
// call to nn::os::SleepThread returns to the "sleep return" address, and nvnEventGetValue is
// called through a pointer (blr) right after. With EXP24 = 1 that one sleep becomes a wait that a
// GPU write ends early (Common::Nx2GpuWrite, from every semaphore/report write path): a write to
// the event's word once it is learned (a written word that read 1 when the next poll found the
// event set) or one the event object points at, and until then any write of a 1. It never lasts
// longer than the sleep it replaces, so the loop polls at least as often. EXP24 = 3 only shortens
// the sleep to EDEN_NX2_OOT_PRESENT_SLEEP_US (default 250 us, at most the time asked for).
//
// EXP24 = 2 finds the site for a build: it counts the short sleeps by call chain and logs the top
// ones every 10 s. EDEN_NX2_OOT_PRESENT_SITE=<ret>[:<blr>] (hex main offsets: the sleep call's
// return address, and the nvnEventGetValue blr, default ret + 0xC) tries a found site with EXP24=1
// before it goes in the table.
constexpr u64 OotProgramId = 0x0400486024DEC000ULL;

using BuildId = std::array<u8, 20>;

/// Where the present poll is in one build of main (offsets from main's start).
struct PresentSite {
    BuildId build_id;
    u64 sleep_return;   // return address of the call to nn::os::SleepThread
    u64 sleep_call;     // that call (bl)
    u64 get_value_call; // blr nvnEventGetValue
};

constexpr std::array PresentSites{
    // E0F3CCFA... (2.0.0, v131072). Hookshot's 0x80590520 (sleep return), 0x8059051C and
    // 0x8059052C are absolute addresses in its Eden layout (nx2/image.py): rtld at 0x80000000,
    // main right after rtld's 0x20000 (as in Eden: rtld@0x80fe4000, main@0x81004000), so main
    // offsets 0x570520, 0x57051C, 0x57052C. main+0x590520 held a function epilogue. Not yet
    // seen verified; EXP24=2 shows the real site.
    PresentSite{
        .build_id = {0xE0, 0xF3, 0xCC, 0xFA, 0xF5, 0x00, 0x65, 0xAE, 0x4E, 0xA0,
                     0x1E, 0xB5, 0xAE, 0xB1, 0x3B, 0x1A, 0x45, 0xDA, 0x33, 0x48},
        .sleep_return = 0x570520,
        .sleep_call = 0x57051C,
        .get_value_call = 0x57052C,
    },
};

constexpr u32 CmpW0One = 0x7100041F; // cmp w0, #1 (expected after the GetValue call, logged only)
constexpr u64 DumpBefore = 0x30;     // the site check logs sleep_return - 0x30 ..
constexpr size_t DumpWords = 20;     // .. + 0x1C (20 words)
constexpr u64 LocateDumpBefore = 0x20; // locate logs site - 0x20 ..
constexpr size_t LocateDumpWords = 16; // .. site + 0x1C (16 words)

constexpr s64 MaxPollSleepNs = 2'000'000; // the game asks for 1 ms
constexpr s64 MaxWaitNs = 1'000'000;
constexpr s64 ResleepWindowNs = 200'000;
constexpr s64 LogPeriodNs = 10'000'000'000;
constexpr size_t FrameWalkDepth = 4;
constexpr size_t LocateTop = 8;
constexpr size_t LocateDumped = 3; // chains whose site words are logged
constexpr size_t LocateMaxChains = 512;
// Return addresses compared without any pointer authentication code above a 39-bit address.
constexpr u64 ReturnAddressMask = (u64{1} << 39) - 1;

// The poll's event object (from the code before the call: ldr w8, [x22, #0x164];
// add x0, x23, x8, lsl #6): x23 + (u32 [x22 + 0x164] << 6), 64 bytes. x22 and x23 are
// callee-saved, so the sleep SVC still sees the loop's values.
constexpr size_t EventIndexOffset = 0x164;
constexpr u32 EventStrideShift = 6;
constexpr size_t EventObjectWords = 8;
constexpr u64 MinAddressField = 0x10000; // smaller fields are not taken for addresses
constexpr size_t MaxLoggedObjects = 8;

// Learning the event's word: a GPU-written word that read 1 when a poll found the event set.
constexpr size_t MaxSeenWrites = 64;  // distinct words written during one wait
constexpr size_t MaxOnesPerWait = 4;  // more candidates than this in one wait teach nothing
constexpr size_t MaxScores = 32;
constexpr size_t MaxLearned = 8;
constexpr u32 LearnSuccesses = 3;

constexpr s64 DefaultShortSleepUs = 250; // EXP24 = 3

using Common::Nx2GpuWrite::Path;
constexpr size_t PathCount = static_cast<size_t>(Path::Count);

/// LR, then the return addresses of the first FrameWalkDepth frame records; 0 = none.
using ReturnChain = std::array<u64, 1 + FrameWalkDepth>;

constexpr bool IsBl(u32 word) {
    return (word & 0xFC000000U) == 0x94000000U;
}

constexpr bool IsBlr(u32 word) {
    return (word & 0xFFFFFC1FU) == 0xD63F0000U;
}

/// A GPU write seen during a wait.
struct SeenWrite {
    const void* host;
    u64 device;
    Path path;
};

/// How a wait ended, settled at the next poll sleep: the poll after it found the event set when
/// that sleep comes later than ResleepWindowNs (else the loop polled again at once).
struct PendingOutcome {
    bool valid = false;
    bool short_sleep = false; // EXP24 = 3
    bool woken = false;
    bool matched = false;
    const void* wake_host = nullptr;
    u64 wake_device = 0;
    s64 end_tick = 0;
    std::array<SeenWrite, MaxOnesPerWait> ones{}; // words written during the wait that read 1
    size_t one_count = 0;                         // (MaxOnesPerWait + 1 = too many)
    u32 field_ones = 0;    // event object fields whose word read 1 (bit mask)
    u32 field_written = 0; // event object fields whose word the GPU wrote during the wait
};

/// A word's record: polls that found the event set right after it read 1, and wakes on it
/// that were followed by another poll at once.
struct WordScore {
    const void* host;
    u64 device;
    u32 ok;
    u32 fail;
};

struct PresentPollState {
    // The process the site was checked in (process ids restart every session, so the pointer
    // too), and what was found: main base + sleep return (0 when the site did not verify) and the
    // two call words, read again on each use in case main moved.
    std::mutex site_mutex;
    const KProcess* checked_process{};
    u64 checked_process_id = ~u64{};
    u64 site_return = 0;
    u64 sleep_call_address = 0;
    u64 get_value_call_address = 0;
    u32 sleep_call_word = 0;
    u32 get_value_call_word = 0;
    bool build_id_logged = false;

    std::atomic<KernelCore*> kernel{};
    std::atomic<bool> listener_set{}; // OnGpuWrite is installed (EXP24 = 1 or 3 found the site)
    KThread* waiter{}; // the thread in the wait; guarded by the scheduler lock
    std::atomic<bool> has_waiter{};
    // Set by the waker under the scheduler lock before it ends the wait; read by the woken thread.
    Path wake_path{};
    const void* wake_host{};
    u64 wake_device{};
    bool wake_matched{};

    // What the GPU side matches writes against, and the writes seen during the current wait;
    // guarded by wake_mutex. Never held while taking the scheduler lock.
    std::mutex wake_mutex;
    std::array<const void*, EventObjectWords> object_hosts{}; // fields that are guest addresses
    std::array<u64, EventObjectWords> object_values{};        // fields (GPU addresses?)
    std::vector<SeenWrite> seen;
    std::vector<const void*> learned;
    std::vector<u64> learned_devices; // (for the log)
    // `learned` for the render-frame clock, read without wake_mutex (written under it).
    std::array<std::atomic<const void*>, MaxLearned> learned_mirror{};

    // The polling thread's bookkeeping between waits, guarded by poll_mutex (not held in a wait).
    std::mutex poll_mutex;
    PendingOutcome pending{};
    std::vector<WordScore> scores;
    std::vector<u64> logged_objects;

    // Statistics, logged every 10 s while EXP24 is 1 or 3.
    std::atomic<u64> waits{};
    std::atomic<u64> woken{};
    std::atomic<u64> woken_matched{};
    std::atomic<u64> wake_ok{};
    std::atomic<u64> wake_matched_ok{};
    std::atomic<u64> early{};
    std::atomic<u64> timed_out{};
    std::atomic<u64> timeout_ok{};
    std::atomic<u64> short_sleeps{};
    std::atomic<u64> short_ok{};
    std::atomic<u64> no_object{};
    std::atomic<u64> last_waker{};
    std::atomic<s64> last_log_tick{};
    std::array<std::atomic<u64>, PathCount> path_writes{};
    std::array<std::atomic<u64>, PathCount> path_ones{};
    std::array<std::atomic<u64>, PathCount> path_matched{};
    std::array<std::atomic<u64>, PathCount> path_wakes{};
    std::array<std::atomic<u64>, PathCount> path_last_device{};
    std::array<std::atomic<u32>, PathCount> path_last_value{};
    // Per event object field: polls that found the event set while the field's word read 1, and
    // those of them with a GPU write to that word seen during the wait.
    std::array<std::atomic<u64>, EventObjectWords> field_one_ok{};
    std::array<std::atomic<u64>, EventObjectWords> field_written_ok{};
    std::array<u64, EventObjectWords> field_addresses{}; // (for the log; poll_mutex)
};

PresentPollState& PollState() {
    static PresentPollState state;
    return state;
}

struct ModuleRange {
    u64 base;
    u64 end;
    std::string name;
};

struct LocateEntry {
    u64 count = 0;
    s64 ns_sum = 0;
};

/// EXP24 = 2: short sleeps by call chain since the last log.
struct LocateState {
    std::mutex mutex;
    const KProcess* process{};
    u64 process_id = ~u64{};
    std::vector<ModuleRange> modules;
    std::map<ReturnChain, LocateEntry> chains;
    u64 sleeps = 0;
    u64 dropped = 0;
    s64 last_log_tick = 0;
};

LocateState& Locate() {
    static LocateState state;
    return state;
}

/// The present poll's wait: a sleep (same timer) that the GPU write listener may end early.
/// Every way out of the wait (GPU write, timer, termination) runs under the scheduler lock and
/// forgets the waiter, so the listener only ever ends a wait that is still this one.
class PresentPollQueue final : public KThreadQueue {
public:
    explicit PresentPollQueue(KernelCore& kernel) : KThreadQueue(kernel) {}

    void EndWait(KernelCore& kernel, KThread* waiting_thread, Result wait_result) override {
        Forget(waiting_thread);
        KThreadQueue::EndWait(kernel, waiting_thread, wait_result);
    }

    void CancelWait(KernelCore& kernel, KThread* waiting_thread, Result wait_result,
                    bool cancel_timer_task) override {
        Forget(waiting_thread);
        KThreadQueue::CancelWait(kernel, waiting_thread, wait_result, cancel_timer_task);
    }

private:
    static void Forget(KThread* waiting_thread) {
        PresentPollState& state = PollState();
        if (state.waiter == waiting_thread) {
            state.waiter = nullptr;
            state.has_waiter.store(false, std::memory_order_release);
        }
    }
};

template <typename Container, typename T>
bool Contains(const Container& container, const T& item) {
    return std::find(container.begin(), container.end(), item) != container.end();
}

/// Common::Nx2GpuWrite listener, on the GPU thread that wrote the word. While a poll waits, every
/// write is counted by path and remembered; a write to a matched word (one learned, or one the
/// event object points at) ends the wait, and until a word is learned so does any write of a 1.
void OnGpuWrite(Path path, u64 device_address, u64 gpu_address, const void* host_pointer,
                u32 value) {
    PresentPollState& state = PollState();
    // The render-frame clock's poll-word source (Tegra::g_nx2_render_frame, only with
    // EDEN_NX2_RENDER_CLOCK=2): a write to a learned word, whether or not a poll waits for it.
    if (host_pointer != nullptr && Tegra::Nx2RenderClockWordSource() &&
        Tegra::g_nx2_render_clock_wanted.load(std::memory_order_relaxed)) {
        for (const auto& learned : state.learned_mirror) {
            if (learned.load(std::memory_order_relaxed) == host_pointer) {
                Tegra::g_nx2_render_frame.fetch_add(1, std::memory_order_relaxed);
                break;
            }
        }
    }
    if (!state.has_waiter.load(std::memory_order_acquire)) {
        return;
    }
    const size_t index = static_cast<size_t>(path);
    if (index >= PathCount) {
        return;
    }
    state.path_writes[index].fetch_add(1, std::memory_order_relaxed);
    state.path_last_device[index].store(device_address, std::memory_order_relaxed);
    state.path_last_value[index].store(value, std::memory_order_relaxed);
    if (value == 1) {
        state.path_ones[index].fetch_add(1, std::memory_order_relaxed);
    }
    bool matched = false;
    bool strict = false;
    {
        std::scoped_lock lk{state.wake_mutex};
        if (host_pointer != nullptr) {
            matched = Contains(state.learned, host_pointer) ||
                      Contains(state.object_hosts, host_pointer);
            const auto seen_it =
                std::find_if(state.seen.begin(), state.seen.end(),
                             [host_pointer](const SeenWrite& w) { return w.host == host_pointer; });
            if (seen_it != state.seen.end()) {
                seen_it->path = path;
            } else if (state.seen.size() < MaxSeenWrites) {
                state.seen.push_back(
                    SeenWrite{.host = host_pointer, .device = device_address, .path = path});
            }
        }
        if (!matched && gpu_address >= MinAddressField) {
            matched = Contains(state.object_values, gpu_address);
        }
        strict = !state.learned.empty();
    }
    if (matched) {
        state.path_matched[index].fetch_add(1, std::memory_order_relaxed);
    }
    if (!matched && (strict || value != 1)) {
        return;
    }
    KernelCore* const kernel = state.kernel.load(std::memory_order_acquire);
    if (kernel == nullptr) {
        return;
    }
    KScopedSchedulerLock sl{*kernel};
    KThread* const thread = state.waiter;
    if (thread == nullptr) {
        return;
    }
    state.wake_path = path;
    state.wake_host = host_pointer;
    state.wake_device = device_address;
    state.wake_matched = matched;
    state.last_waker.store(device_address, std::memory_order_relaxed);
    state.path_wakes[index].fetch_add(1, std::memory_order_relaxed);
    thread->EndWait(*kernel, ResultSuccess);
}

/// The process's modules with their load addresses: the loader's own list (the NSOs it loaded,
/// by file name: rtld, main, subsdk*, sdk), else the modules found in the process's memory by
/// their path names. `out_list` describes them, for the log.
/// (Core::FindMainModuleEntrypoint took the code region start, 0x8000000, not main.)
std::vector<ModuleRange> LoadedModules(Core::System& system, KProcess& process,
                                       std::string& out_list) {
    Loader::AppLoader::Modules modules;
    const char* source = "loader:";
    if (system.GetAppLoader().ReadNSOModules(modules) != Loader::ResultStatus::Success ||
        modules.empty()) {
        modules = Core::FindModules(std::addressof(process));
        source = "memory:";
    }
    std::vector<ModuleRange> ranges;
    out_list = source;
    for (auto it = modules.begin(); it != modules.end(); ++it) {
        const auto next = std::next(it);
        const u64 end = next != modules.end()
                            ? next->first
                            : GetInteger(Core::GetModuleEnd(std::addressof(process), it->first)) + 1;
        std::string name = it->second;
        if (name.starts_with("main.")) {
            name = "main"; // a path name from memory
        }
        out_list += fmt::format(" {}@{:#x}", it->second, it->first);
        ranges.push_back(ModuleRange{.base = it->first, .end = end, .name = std::move(name)});
    }
    return ranges;
}

const ModuleRange* FindModule(const std::vector<ModuleRange>& modules, u64 address) {
    for (const ModuleRange& range : modules) {
        if (address >= range.base && address < range.end) {
            return std::addressof(range);
        }
    }
    return nullptr;
}

std::string DescribeAddress(const std::vector<ModuleRange>& modules, u64 address) {
    if (const ModuleRange* range = FindModule(modules, address)) {
        return fmt::format("{}+{:#x}", range->name, address - range->base);
    }
    return fmt::format("{:#x}", address);
}

/// The present-poll site for this build (env override first), or none.
struct SiteOffsets {
    u64 sleep_return;
    u64 get_value_call;
    bool from_env;
};

std::optional<SiteOffsets> EnvSite() {
    static const std::optional<SiteOffsets> site = []() -> std::optional<SiteOffsets> {
        const char* const text = std::getenv("EDEN_NX2_OOT_PRESENT_SITE");
        if (text == nullptr || text[0] == 0) {
            return std::nullopt;
        }
        char* end = nullptr;
        const u64 sleep_return = std::strtoull(text, &end, 16);
        if (end == text || sleep_return < 4) {
            return std::nullopt;
        }
        u64 get_value_call = sleep_return + 0xC;
        if (*end == ':') {
            const char* const second = end + 1;
            char* second_end = nullptr;
            const u64 parsed = std::strtoull(second, &second_end, 16);
            if (second_end != second) {
                get_value_call = parsed;
            }
        }
        return SiteOffsets{
            .sleep_return = sleep_return, .get_value_call = get_value_call, .from_env = true};
    }();
    return site;
}

std::optional<SiteOffsets> SiteFor(const std::array<u8, 0x20>& build_id) {
    if (const std::optional<SiteOffsets> env = EnvSite()) {
        return env;
    }
    for (const PresentSite& site : PresentSites) {
        if (std::equal(site.build_id.begin(), site.build_id.end(), build_id.begin())) {
            if (site.sleep_call + 4 != site.sleep_return) {
                return std::nullopt; // (a table typo: the call must be just before its return)
            }
            return SiteOffsets{.sleep_return = site.sleep_return,
                               .get_value_call = site.get_value_call,
                               .from_env = false};
        }
    }
    return std::nullopt;
}

std::string HexBuildId(const std::array<u8, 0x20>& build_id) {
    std::string text;
    for (size_t i = 0; i < 20; ++i) {
        text += fmt::format("{:02X}", build_id[i]);
    }
    return text;
}

/// main base + sleep return when this process runs a build with a known (or EDEN_NX2_OOT_
/// PRESENT_SITE) site whose call words check out, else 0. The module is searched once per
/// process; later calls only read the two call words again.
u64 SiteReturnAddress(Core::System& system, KProcess& process) {
    PresentPollState& state = PollState();
    std::scoped_lock lk{state.site_mutex};
    const auto& build_id = system.GetApplicationProcessBuildID();
    const std::optional<SiteOffsets> site = SiteFor(build_id);
    if (!site) {
        if (!state.build_id_logged) {
            state.build_id_logged = true;
            LOG_INFO(Kernel_SVC,
                     "OoT present poll (EXP24): no site for main build {} (EXP24=2 finds it, "
                     "EDEN_NX2_OOT_PRESENT_SITE tries it), off",
                     HexBuildId(build_id));
        }
        return 0;
    }
    auto& memory = process.GetMemory();
    if (state.checked_process == std::addressof(process) &&
        state.checked_process_id == process.GetProcessId()) {
        if (state.site_return == 0 ||
            (memory.Read32(state.sleep_call_address) == state.sleep_call_word &&
             memory.Read32(state.get_value_call_address) == state.get_value_call_word)) {
            return state.site_return;
        }
    }
    state.checked_process = std::addressof(process);
    state.checked_process_id = process.GetProcessId();
    state.site_return = 0;

    std::string module_list;
    const std::vector<ModuleRange> modules = LoadedModules(system, process, module_list);
    const auto main_module = std::find_if(modules.begin(), modules.end(),
                                          [](const ModuleRange& m) { return m.name == "main"; });
    if (main_module == modules.end()) {
        LOG_WARNING(Kernel_SVC, "OoT present poll (EXP24): no module called main ({}), off",
                    module_list);
        return 0;
    }
    const u64 main_base = main_module->base;
    LOG_INFO(Kernel_SVC, "OoT present poll (EXP24): using module main at {:#x} ({})", main_base,
             module_list);
    LOG_INFO(Kernel_SVC,
             "OoT present poll (EXP24): site main+{:#x} (sleep call main+{:#x}, blr main+{:#x}) "
             "from {}",
             site->sleep_return, site->sleep_return - 4, site->get_value_call,
             site->from_env ? "EDEN_NX2_OOT_PRESENT_SITE" : "the table");
    const u64 dump_first = main_base + site->sleep_return - DumpBefore;
    if (!memory.IsValidVirtualAddressRange(dump_first, DumpWords * sizeof(u32)) ||
        !memory.IsValidVirtualAddressRange(main_base + site->get_value_call, 2 * sizeof(u32))) {
        LOG_WARNING(Kernel_SVC, "OoT present poll (EXP24): main+{:#x} is not mapped (main {:#x}), off",
                    site->sleep_return - DumpBefore, main_base);
        return 0;
    }
    std::string text;
    for (size_t i = 0; i < DumpWords; ++i) {
        text += fmt::format(" {:08x}", memory.Read32(dump_first + i * sizeof(u32)));
    }
    LOG_INFO(Kernel_SVC, "OoT present poll (EXP24): main {:#x}, main+{:#x}:{}", main_base,
             site->sleep_return - DumpBefore, text);
    const u64 sleep_call_address = main_base + site->sleep_return - 4;
    const u64 get_value_call_address = main_base + site->get_value_call;
    const u32 sleep_call = memory.Read32(sleep_call_address);
    const u32 get_value_call = memory.Read32(get_value_call_address);
    const u32 compare = memory.Read32(get_value_call_address + 4);
    // The table's sleep call is a bl (nn::os::SleepThread through its PLT stub); a site under test
    // may call through a pointer too.
    const bool call_ok = IsBl(sleep_call) || (site->from_env && IsBlr(sleep_call));
    if (!call_ok || !IsBlr(get_value_call)) {
        LOG_WARNING(Kernel_SVC,
                    "OoT present poll (EXP24): main+{:#x} = {:#010x} (want bl), main+{:#x} = "
                    "{:#010x} (want blr), off",
                    site->sleep_return - 4, sleep_call, site->get_value_call, get_value_call);
        return 0;
    }
    LOG_INFO(Kernel_SVC,
             "OoT present poll (EXP24): site verified (bl {:#010x}, blr {:#010x}, then {:#010x}{})",
             sleep_call, get_value_call, compare, compare == CmpW0One ? " = cmp w0, #1" : "");
    state.site_return = main_base + site->sleep_return;
    state.sleep_call_address = sleep_call_address;
    state.get_value_call_address = get_value_call_address;
    state.sleep_call_word = sleep_call;
    state.get_value_call_word = get_value_call;
    state.kernel.store(std::addressof(system.Kernel()), std::memory_order_release);
    return state.site_return;
}

/// The current SVC's caller chain: LR, then the return addresses in the innermost frame records
/// (nn::os::SleepThread's own frames sit between the svc and the game's call).
/// The calling thread's registers at the SVC; false when they cannot be read.
bool ReadContext(KernelCore& kernel, KProcess& process, Svc::ThreadContext& ctx) {
    Core::ArmInterface* const arm = process.GetArmInterface(kernel.CurrentPhysicalCoreIndex());
    if (arm == nullptr) {
        return false;
    }
    arm->GetContext(ctx);
    return true;
}

ReturnChain ReturnAddresses(KProcess& process, const Svc::ThreadContext& ctx) {
    ReturnChain chain{};
    chain[0] = ctx.lr & ReturnAddressMask;
    auto& memory = process.GetMemory();
    u64 fp = ctx.fp;
    for (size_t depth = 0; depth < FrameWalkDepth; ++depth) {
        if (fp == 0 || (fp & 7) != 0 || !memory.IsValidVirtualAddressRange(fp, 16)) {
            break;
        }
        chain[1 + depth] = memory.Read64(fp + 8) & ReturnAddressMask;
        const u64 next = memory.Read64(fp);
        if (next <= fp) {
            // The stack grows down: a caller's frame record is above its callee's.
            break;
        }
        fp = next;
    }
    return chain;
}

bool CalledFromSite(KProcess& process, const Svc::ThreadContext& ctx, u64 site_return) {
    return Contains(ReturnAddresses(process, ctx), site_return);
}

/// The first address of the chain inside main (the game's call), else LR.
u64 ChainSite(const std::vector<ModuleRange>& modules, const ReturnChain& chain) {
    for (const u64 address : chain) {
        if (address == 0) {
            break;
        }
        if (const ModuleRange* range = FindModule(modules, address);
            range != nullptr && range->name == "main") {
            return address;
        }
    }
    return chain[0];
}

void LogLocate(LocateState& locate, KProcess& process, s64 elapsed) {
    std::vector<std::pair<ReturnChain, LocateEntry>> top(locate.chains.begin(),
                                                         locate.chains.end());
    std::sort(top.begin(), top.end(),
              [](const auto& a, const auto& b) { return a.second.count > b.second.count; });
    if (top.size() > LocateTop) {
        top.resize(LocateTop);
    }
    LOG_INFO(Kernel_SVC,
             "OoT present poll locate (EXP24=2): {} sleeps of at most 2 ms in {:.1f} s, {} call "
             "chains ({} sleeps not kept), top {}:",
             locate.sleeps, static_cast<double>(elapsed) / 1e9, locate.chains.size(),
             locate.dropped, top.size());
    for (size_t i = 0; i < top.size(); ++i) {
        const auto& [chain, entry] = top[i];
        std::string text = "lr " + DescribeAddress(locate.modules, chain[0]);
        for (size_t depth = 1; depth < chain.size() && chain[depth] != 0; ++depth) {
            text += " < " + DescribeAddress(locate.modules, chain[depth]);
        }
        LOG_INFO(Kernel_SVC, "OoT present poll locate #{}: {} sleeps, {} ns asked on average: {}",
                 i, entry.count, entry.ns_sum / static_cast<s64>(entry.count), text);
    }
    auto& memory = process.GetMemory();
    for (size_t i = 0; i < top.size() && i < LocateDumped; ++i) {
        const u64 site = ChainSite(locate.modules, top[i].first);
        if (site < LocateDumpBefore ||
            !memory.IsValidVirtualAddressRange(site - LocateDumpBefore,
                                               LocateDumpWords * sizeof(u32))) {
            continue;
        }
        std::string words;
        for (size_t w = 0; w < LocateDumpWords; ++w) {
            words += fmt::format(" {:08x}", memory.Read32(site - LocateDumpBefore + w * sizeof(u32)));
        }
        LOG_INFO(Kernel_SVC, "OoT present poll locate #{} site {}, words from {}:{}", i,
                 DescribeAddress(locate.modules, site),
                 DescribeAddress(locate.modules, site - LocateDumpBefore), words);
    }
}

/// EXP24 = 2: counts this short sleep by its call chain; logs the top chains every 10 s.
void LocateCount(Core::System& system, KernelCore& kernel, KProcess& process, s64 ns) {
    Svc::ThreadContext ctx{};
    if (!ReadContext(kernel, process, ctx)) {
        return;
    }
    const ReturnChain chain = ReturnAddresses(process, ctx);
    const s64 now = kernel.HardwareTimer().GetTick();
    LocateState& locate = Locate();
    std::scoped_lock lk{locate.mutex};
    if (locate.process != std::addressof(process) ||
        locate.process_id != process.GetProcessId()) {
        locate.process = std::addressof(process);
        locate.process_id = process.GetProcessId();
        std::string module_list;
        locate.modules = LoadedModules(system, process, module_list);
        locate.chains.clear();
        locate.sleeps = 0;
        locate.dropped = 0;
        locate.last_log_tick = now;
        LOG_INFO(Kernel_SVC, "OoT present poll locate (EXP24=2): modules {}", module_list);
    }
    ++locate.sleeps;
    auto it = locate.chains.find(chain);
    if (it == locate.chains.end() && locate.chains.size() < LocateMaxChains) {
        it = locate.chains.emplace(chain, LocateEntry{}).first;
    }
    if (it != locate.chains.end()) {
        ++it->second.count;
        it->second.ns_sum += ns;
    } else {
        ++locate.dropped;
    }
    if (now - locate.last_log_tick >= LogPeriodNs) {
        LogLocate(locate, process, now - locate.last_log_tick);
        locate.chains.clear();
        locate.sleeps = 0;
        locate.dropped = 0;
        locate.last_log_tick = now;
    }
}

/// Sleeps up to `ns` (> 0), ended early by OnGpuWrite. Returns the wait's result.
Result WaitForGpuWrite(KernelCore& kernel, s64 ns) {
    KThread& thread = GetCurrentThread(kernel);
    PresentPollState& state = PollState();
    PresentPollQueue wait_queue(kernel);
    KHardwareTimer* timer{};
    s64 timeout = kernel.HardwareTimer().GetTick() + ns + 2;
    if (timeout <= 0) {
        timeout = (std::numeric_limits<s64>::max)();
    }
    {
        KScopedSchedulerLockAndSleep slp(kernel, std::addressof(timer), std::addressof(thread),
                                         timeout);
        if (thread.IsTerminationRequested()) {
            slp.CancelSleep();
            return ResultTerminationRequested;
        }
        // One waiter at a time (the game has one present poll); another thread just sleeps.
        if (state.waiter == nullptr) {
            state.waiter = std::addressof(thread);
            state.has_waiter.store(true, std::memory_order_release);
        }
        wait_queue.SetHardwareTimer(timer);
        thread.BeginWait(kernel, std::addressof(wait_queue));
        thread.SetWaitReasonForDebugging(ThreadWaitReasonForDebugging::Sleep);
    }
    return thread.GetWaitResult();
}


s64 ShortSleepUs() {
    static const s64 us = [] {
        const char* const text = std::getenv("EDEN_NX2_OOT_PRESENT_SLEEP_US");
        if (text == nullptr || text[0] == 0) {
            return DefaultShortSleepUs;
        }
        const long long parsed = std::strtoll(text, nullptr, 10);
        return parsed > 0 ? static_cast<s64>(parsed) : DefaultShortSleepUs;
    }();
    return us;
}

/// Reads the poll's event object (from x22/x23) and hands its address-like fields to the GPU
/// side; clears the writes seen. Logs each new object once (up to MaxLoggedObjects).
/// Called with poll_mutex held.
void ReadEventObject(PresentPollState& state, KProcess& process, const Svc::ThreadContext& ctx) {
    auto& memory = process.GetMemory();
    std::array<const void*, EventObjectWords> hosts{};
    std::array<u64, EventObjectWords> values{};
    std::array<u64, EventObjectWords> fields{};
    const u64 x22 = ctx.r[22];
    const u64 x23 = ctx.r[23];
    u64 object = 0;
    u32 object_index = 0;
    if (memory.IsValidVirtualAddressRange(x22 + EventIndexOffset, sizeof(u32))) {
        object_index = memory.Read32(x22 + EventIndexOffset);
        object = x23 + (u64{object_index} << EventStrideShift);
    }
    if (object == 0 ||
        !memory.IsValidVirtualAddressRange(object, EventObjectWords * sizeof(u64))) {
        state.no_object.fetch_add(1, std::memory_order_relaxed);
        object = 0;
    } else {
        for (size_t k = 0; k < EventObjectWords; ++k) {
            fields[k] = memory.Read64(object + k * sizeof(u64));
            if (fields[k] < MinAddressField) {
                continue;
            }
            values[k] = fields[k];
            if ((fields[k] & 3) == 0 && memory.IsValidVirtualAddressRange(fields[k], sizeof(u32))) {
                hosts[k] = memory.GetPointerSilent(fields[k]);
            }
        }
    }
    for (size_t k = 0; k < EventObjectWords; ++k) {
        state.field_addresses[k] = hosts[k] != nullptr ? fields[k] : 0;
    }
    {
        std::scoped_lock lk{state.wake_mutex};
        state.object_hosts = hosts;
        state.object_values = values;
        state.seen.clear();
    }
    if (object == 0 || Contains(state.logged_objects, object) ||
        state.logged_objects.size() >= MaxLoggedObjects) {
        return;
    }
    state.logged_objects.push_back(object);
    std::string text;
    std::string words;
    for (size_t k = 0; k < EventObjectWords; ++k) {
        text += fmt::format(" {:016x}", fields[k]);
        if (hosts[k] != nullptr) {
            u32 word;
            std::memcpy(&word, hosts[k], sizeof(word));
            words += fmt::format(" f{} [{:#x}] = {:#x}", k, fields[k], word);
        }
    }
    LOG_INFO(Kernel_SVC,
             "OoT present poll (EXP24): event object {:#x} (x23 {:#x} + {} << 6, x22 {:#x}):{}; "
             "fields that are guest addresses:{}",
             object, x23, object_index, x22, text, words.empty() ? " none" : words);
}

/// Records a word's outcome and learns or forgets it. Called with poll_mutex held.
void Score(PresentPollState& state, const void* host, u64 device, bool ok) {
    if (host == nullptr) {
        return;
    }
    auto it = std::find_if(state.scores.begin(), state.scores.end(),
                           [host](const WordScore& s) { return s.host == host; });
    if (it == state.scores.end()) {
        if (!ok || state.scores.size() >= MaxScores) {
            return;
        }
        state.scores.push_back(WordScore{.host = host, .device = device, .ok = 0, .fail = 0});
        it = std::prev(state.scores.end());
    }
    if (ok) {
        ++it->ok;
    } else {
        ++it->fail;
    }
    const bool learn = it->ok >= LearnSuccesses && it->ok > 4 * it->fail;
    std::scoped_lock lk{state.wake_mutex};
    const auto learned_it = std::find(state.learned.begin(), state.learned.end(), host);
    if (learn && learned_it == state.learned.end() && state.learned.size() < MaxLearned) {
        state.learned.push_back(host);
        state.learned_devices.push_back(it->device);
        LOG_INFO(Kernel_SVC,
                 "OoT present poll (EXP24): learned the event word at device {:#x} ({} polls found "
                 "the event set right after it read 1, {} early wakes on it); waking on matched "
                 "words only now",
                 it->device, it->ok, it->fail);
    } else if (!learn && learned_it != state.learned.end()) {
        const auto position = learned_it - state.learned.begin();
        state.learned.erase(learned_it);
        state.learned_devices.erase(state.learned_devices.begin() + position);
        LOG_INFO(Kernel_SVC,
                 "OoT present poll (EXP24): forgot the word at device {:#x} ({} found set, {} "
                 "early wakes)",
                 it->device, it->ok, it->fail);
    }
    for (size_t i = 0; i < state.learned_mirror.size(); ++i) {
        state.learned_mirror[i].store(i < state.learned.size() ? state.learned[i] : nullptr,
                                      std::memory_order_relaxed);
    }
}

/// Settles the last wait's outcome now that the next poll sleep came. Called with poll_mutex held.
void ResolvePending(PresentPollState& state, s64 start) {
    PendingOutcome& pending = state.pending;
    if (!pending.valid) {
        return;
    }
    pending.valid = false;
    const bool found_set = start - pending.end_tick >= ResleepWindowNs;
    if (pending.short_sleep) {
        if (found_set) {
            state.short_ok.fetch_add(1, std::memory_order_relaxed);
        }
        return;
    }
    if (found_set) {
        for (size_t k = 0; k < EventObjectWords; ++k) {
            if ((pending.field_ones >> k) & 1) {
                state.field_one_ok[k].fetch_add(1, std::memory_order_relaxed);
                if ((pending.field_written >> k) & 1) {
                    state.field_written_ok[k].fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }
    if (pending.woken) {
        if (found_set) {
            state.wake_ok.fetch_add(1, std::memory_order_relaxed);
            if (pending.matched) {
                state.wake_matched_ok.fetch_add(1, std::memory_order_relaxed);
            }
        } else {
            state.early.fetch_add(1, std::memory_order_relaxed);
        }
        Score(state, pending.wake_host, pending.wake_device, found_set);
    } else if (found_set) {
        state.timeout_ok.fetch_add(1, std::memory_order_relaxed);
        if (pending.one_count <= MaxOnesPerWait) {
            for (size_t i = 0; i < pending.one_count; ++i) {
                Score(state, pending.ones[i].host, pending.ones[i].device, true);
            }
        }
    }
}

/// Notes how the wait just ended and which written words read 1 now. poll_mutex held.
void CapturePending(PresentPollState& state, bool woken, s64 end_tick) {
    PendingOutcome outcome{};
    outcome.valid = true;
    outcome.woken = woken;
    outcome.end_tick = end_tick;
    if (woken) {
        outcome.matched = state.wake_matched;
        outcome.wake_host = state.wake_host;
        outcome.wake_device = state.wake_device;
    }
    std::scoped_lock lk{state.wake_mutex};
    for (const SeenWrite& write : state.seen) {
        for (size_t k = 0; k < EventObjectWords; ++k) {
            if (state.object_hosts[k] == write.host) {
                outcome.field_written |= 1U << k;
            }
        }
        u32 word;
        std::memcpy(&word, write.host, sizeof(word));
        if (word == 1 && outcome.one_count <= MaxOnesPerWait) {
            if (outcome.one_count < MaxOnesPerWait) {
                outcome.ones[outcome.one_count] = write;
            }
            ++outcome.one_count;
        }
    }
    for (size_t k = 0; k < EventObjectWords; ++k) {
        if (state.object_hosts[k] != nullptr) {
            u32 word;
            std::memcpy(&word, state.object_hosts[k], sizeof(word));
            if (word == 1) {
                outcome.field_ones |= 1U << k;
            }
        }
    }
    state.pending = outcome;
}

/// Every 10 s: the wait outcomes, the writes by path, the event object fields. poll_mutex held.
void MaybeLog(PresentPollState& state, s64 now, int mode) {
    s64 last = state.last_log_tick.load(std::memory_order_relaxed);
    if (last == 0) {
        state.last_log_tick.store(now, std::memory_order_relaxed);
        return;
    }
    if (now - last < LogPeriodNs) {
        return;
    }
    state.last_log_tick.store(now, std::memory_order_relaxed);
    const double seconds = static_cast<double>(now - last) / 1e9;
    const auto take = [](std::atomic<u64>& counter) {
        return counter.exchange(0, std::memory_order_relaxed);
    };
    if (mode == 3) {
        LOG_INFO(Kernel_SVC,
                 "OoT present poll (EXP24=3): {} sleeps of {} us in {:.1f} s; {} polls then "
                 "found the event set",
                 take(state.short_sleeps), ShortSleepUs(), seconds, take(state.short_ok));
        return;
    }
    std::string learned_text;
    size_t learned_count = 0;
    {
        std::scoped_lock lk{state.wake_mutex};
        learned_count = state.learned_devices.size();
        for (const u64 device : state.learned_devices) {
            learned_text += fmt::format(" {:#x}", device);
        }
    }
    const u64 waits = take(state.waits);
    const u64 woken = take(state.woken);
    const u64 woken_matched = take(state.woken_matched);
    const u64 wake_ok = take(state.wake_ok);
    const u64 wake_matched_ok = take(state.wake_matched_ok);
    const u64 early = take(state.early);
    const u64 timed_out = take(state.timed_out);
    const u64 timeout_ok = take(state.timeout_ok);
    const u64 no_object = take(state.no_object);
    LOG_INFO(Kernel_SVC,
             "OoT present poll (EXP24=1): {} waits in {:.1f} s; {} woken by a GPU write ({} on a "
             "matched word): {} polls then found the event set ({} after a matched wake), {} "
             "polled again at once; {} timed out (1 ms): {} polls then found it set; {} without "
             "an event object; learned {} word(s):{}",
             waits, seconds, woken, woken_matched, wake_ok, wake_matched_ok, early, timed_out,
             timeout_ok, no_object, learned_count, learned_count == 0 ? " none" : learned_text);
    std::string paths;
    for (size_t p = 0; p < PathCount; ++p) {
        const u64 writes = take(state.path_writes[p]);
        const u64 ones = take(state.path_ones[p]);
        const u64 matched = take(state.path_matched[p]);
        const u64 wakes = take(state.path_wakes[p]);
        if (writes == 0) {
            continue;
        }
        paths += fmt::format(" {} {}/{}/{}/{} last {:#x}={:#x};",
                             Common::Nx2GpuWrite::PathName(static_cast<Path>(p)), writes, ones,
                             matched, wakes,
                             state.path_last_device[p].load(std::memory_order_relaxed),
                             state.path_last_value[p].load(std::memory_order_relaxed));
    }
    LOG_INFO(Kernel_SVC,
             "OoT present poll writes during waits (path writes/of 1/matched/woke, last "
             "device=value):{}",
             paths.empty() ? " none" : paths);
    std::string fields;
    for (size_t k = 0; k < EventObjectWords; ++k) {
        const u64 one_ok = take(state.field_one_ok[k]);
        const u64 written_ok = take(state.field_written_ok[k]);
        if (state.field_addresses[k] == 0 && one_ok == 0) {
            continue;
        }
        fields += fmt::format(" f{} {:#x}: {}/{};", k, state.field_addresses[k], one_ok,
                              written_ok);
    }
    LOG_INFO(Kernel_SVC,
             "OoT present poll event object fields (polls that found the event set with the "
             "field's word 1 / of them with a GPU write to it seen):{}",
             fields.empty() ? " none" : fields);
}

/// The poll's event object's guest address (as ReadEventObject finds it; 0 if unreadable). With
/// the timeline's guest layer, one "OoT: present poll word" marker per address-like word of it:
/// its value (a GPU address, compared with the GPU write's) and the host pointer it maps to
/// (compared with the host pointer of Common::Nx2GpuWrite::Notify's "GPU write"). Reads only.
u64 MarkEventObjectForTimeline(KProcess& process, const Svc::ThreadContext& ctx) {
    auto& memory = process.GetMemory();
    const u64 x22 = ctx.r[22];
    const u64 x23 = ctx.r[23];
    if (!memory.IsValidVirtualAddressRange(x22 + EventIndexOffset, sizeof(u32))) {
        return 0;
    }
    const u64 object = x23 + (u64{memory.Read32(x22 + EventIndexOffset)} << EventStrideShift);
    if (object == 0 ||
        !memory.IsValidVirtualAddressRange(object, EventObjectWords * sizeof(u64))) {
        return 0;
    }
    if (!Common::Nx2Timeline::Guest()) {
        return object;
    }
    for (size_t k = 0; k < EventObjectWords; ++k) {
        const u64 field = memory.Read64(object + k * sizeof(u64));
        if (field < MinAddressField) {
            continue;
        }
        const void* host = nullptr;
        if ((field & 3) == 0 && memory.IsValidVirtualAddressRange(field, sizeof(u32))) {
            host = memory.GetPointerSilent(field);
        }
        Common::Nx2Timeline::Instant("OoT: present poll word", {"field", k}, {"gpu", field},
                                     {"host", reinterpret_cast<uintptr_t>(host)});
    }
    return object;
}

} // Anonymous namespace

bool IsNx2OotPresentPollSleep(Core::System& system, s64 ns, u64* event_object) {
    if (ns <= 0 || ns > MaxPollSleepNs) {
        return false;
    }
    auto& kernel = system.Kernel();
    auto& process = GetCurrentProcess(kernel);
    if (!process.Is64Bit() || process.GetProgramId() != OotProgramId) {
        return false;
    }
    const u64 site_return = SiteReturnAddress(system, process);
    if (site_return == 0) {
        return false;
    }
    Svc::ThreadContext ctx{};
    if (!ReadContext(kernel, process, ctx) || !CalledFromSite(process, ctx, site_return)) {
        return false;
    }
    if (event_object != nullptr) {
        *event_object = MarkEventObjectForTimeline(process, ctx);
    }
    return true;
}

bool SleepNx2OotPresentPoll(Core::System& system, s64 ns) {
    if (ns <= 0 || ns > MaxPollSleepNs) {
        return false;
    }
    // 1 = wait for the GPU's write, 2 = locate the site, 3 = a shorter plain sleep.
    const int mode = VideoCore::Nx2Debug::Experiment(24);
    if (mode < 1 || mode > 3) {
        return false;
    }
    auto& kernel = system.Kernel();
    auto& process = GetCurrentProcess(kernel);
    if (!process.Is64Bit() || process.GetProgramId() != OotProgramId) {
        return false;
    }
    if (mode == 2) {
        // Locate only: the sleep itself is unchanged.
        LocateCount(system, kernel, process, ns);
        return false;
    }
    const u64 site_return = SiteReturnAddress(system, process);
    if (site_return == 0) {
        return false;
    }
    // Here, not in SiteReturnAddress: IsNx2OotPresentPollSleep finds the site too, without EXP24.
    PresentPollState& state = PollState();
    if (!state.listener_set.exchange(true, std::memory_order_acq_rel)) {
        Common::Nx2GpuWrite::SetListener(&OnGpuWrite);
    }
    Svc::ThreadContext ctx{};
    if (!ReadContext(kernel, process, ctx) || !CalledFromSite(process, ctx, site_return)) {
        return false;
    }

    {
        std::scoped_lock lk{state.poll_mutex};
        const s64 start = kernel.HardwareTimer().GetTick();
        ResolvePending(state, start);
        MaybeLog(state, start, mode);
        if (mode == 1) {
            ReadEventObject(state, process, ctx);
        }
    }

    if (mode == 3) {
        // No GPU wake: just a shorter sleep, at most the one asked for.
        state.short_sleeps.fetch_add(1, std::memory_order_relaxed);
        const s64 sleep_ns = (std::min)(ns, ShortSleepUs() * 1000);
        static_cast<void>(GetCurrentThread(kernel).Sleep(
            kernel, kernel.HardwareTimer().GetTick() + sleep_ns + 2));
        std::scoped_lock lk{state.poll_mutex};
        state.pending = PendingOutcome{};
        state.pending.valid = true;
        state.pending.short_sleep = true;
        state.pending.end_tick = kernel.HardwareTimer().GetTick();
        return true;
    }

    state.waits.fetch_add(1, std::memory_order_relaxed);
    const Result result = WaitForGpuWrite(kernel, (std::min)(ns, MaxWaitNs));
    const bool woken = result == ResultSuccess;
    if (woken) {
        state.woken.fetch_add(1, std::memory_order_relaxed);
        if (state.wake_matched) {
            state.woken_matched.fetch_add(1, std::memory_order_relaxed);
        }
    } else if (result == ResultTimedOut) {
        state.timed_out.fetch_add(1, std::memory_order_relaxed);
    }
    std::scoped_lock lk{state.poll_mutex};
    CapturePending(state, woken, kernel.HardwareTimer().GetTick());
    return true;
}

} // namespace Kernel::Svc
