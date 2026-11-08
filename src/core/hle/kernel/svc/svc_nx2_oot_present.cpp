// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
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
#include "video_core/nx2_debug_control.h"

namespace Kernel::Svc {
namespace {

// Ocarina of Time. Found by Hookshot (NotRexed's OoT recomp, host/hos_thread.c present_poll):
// the main thread waits for its frame's GPU work with
//     while (nvnEventGetValue(e) != 1) nn::os::SleepThread(1 ms);
// ~3.5 sleeps a frame, so each frame ends 0.5-1 ms after the GPU wrote the event. In the loop a
// call to nn::os::SleepThread returns to the "sleep return" address, and nvnEventGetValue is
// called through a pointer (blr) right after. Here that one sleep becomes a wait that the GPU's
// next write of a 1 (a semaphore release or report, Tegra::Nx2SemaphoreWaiters::OnWrite) ends
// early. It never lasts longer than the sleep it replaces, so the loop polls at least as often.
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

/// LR, then the return addresses of the first FrameWalkDepth frame records; 0 = none.
using ReturnChain = std::array<u64, 1 + FrameWalkDepth>;

constexpr bool IsBl(u32 word) {
    return (word & 0xFC000000U) == 0x94000000U;
}

constexpr bool IsBlr(u32 word) {
    return (word & 0xFFFFFC1FU) == 0xD63F0000U;
}

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
    KThread* waiter{}; // the thread in the wait; guarded by the scheduler lock
    std::atomic<bool> has_waiter{};

    // Statistics, logged every 10 s while EXP24 is on.
    std::atomic<u64> waits{};
    std::atomic<u64> woken{};
    std::atomic<u64> timed_out{};
    std::atomic<u64> resleeps{};
    std::atomic<u64> last_waker{};
    std::atomic<s64> last_end_tick{};
    std::atomic<bool> last_woken{};
    std::atomic<s64> last_log_tick{};
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

/// Common::Nx2GpuWrite listener, on the GPU thread that wrote the word.
void OnGpuWrite(u64 device_address, u32 value) {
    if (value != 1) {
        return;
    }
    PresentPollState& state = PollState();
    if (!state.has_waiter.load(std::memory_order_acquire)) {
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
    state.last_waker.store(device_address, std::memory_order_relaxed);
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
    Common::Nx2GpuWrite::SetListener(&OnGpuWrite);
    return state.site_return;
}

/// The current SVC's caller chain: LR, then the return addresses in the innermost frame records
/// (nn::os::SleepThread's own frames sit between the svc and the game's call).
ReturnChain ReturnAddresses(KernelCore& kernel, KProcess& process) {
    ReturnChain chain{};
    Core::ArmInterface* const arm = process.GetArmInterface(kernel.CurrentPhysicalCoreIndex());
    if (arm == nullptr) {
        return chain;
    }
    Svc::ThreadContext ctx{};
    arm->GetContext(ctx);
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

bool CalledFromSite(KernelCore& kernel, KProcess& process, u64 site_return) {
    const ReturnChain chain = ReturnAddresses(kernel, process);
    return std::find(chain.begin(), chain.end(), site_return) != chain.end();
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
    const ReturnChain chain = ReturnAddresses(kernel, process);
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

void MaybeLog(PresentPollState& state, s64 now) {
    s64 last = state.last_log_tick.load(std::memory_order_relaxed);
    if (last == 0) {
        state.last_log_tick.compare_exchange_strong(last, now, std::memory_order_relaxed);
        return;
    }
    if (now - last < LogPeriodNs ||
        !state.last_log_tick.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
        return;
    }
    const u64 waits = state.waits.exchange(0, std::memory_order_relaxed);
    const u64 woken = state.woken.exchange(0, std::memory_order_relaxed);
    const u64 timed_out = state.timed_out.exchange(0, std::memory_order_relaxed);
    const u64 resleeps = state.resleeps.exchange(0, std::memory_order_relaxed);
    LOG_INFO(Kernel_SVC,
             "OoT present poll (EXP24): {} waits in {:.1f} s, {} ended by a GPU write of 1 ({} "
             "polled again within 200 us: early), {} timed out (1 ms), last waking write {:#x}",
             waits, static_cast<double>(now - last) / 1e9, woken, resleeps, timed_out,
             state.last_waker.load(std::memory_order_relaxed));
}

} // Anonymous namespace

bool SleepNx2OotPresentPoll(Core::System& system, s64 ns) {
    if (ns <= 0 || ns > MaxPollSleepNs) {
        return false;
    }
    const int mode = VideoCore::Nx2Debug::Experiment(24);
    if (mode != 1 && mode != 2) {
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
    if (site_return == 0 || !CalledFromSite(kernel, process, site_return)) {
        return false;
    }

    PresentPollState& state = PollState();
    const s64 start = kernel.HardwareTimer().GetTick();
    if (state.last_woken.load(std::memory_order_relaxed) &&
        start - state.last_end_tick.load(std::memory_order_relaxed) < ResleepWindowNs) {
        // The last wait ended at a GPU write, and the game found its event not yet 1.
        state.resleeps.fetch_add(1, std::memory_order_relaxed);
    }
    state.waits.fetch_add(1, std::memory_order_relaxed);
    MaybeLog(state, start);

    const Result result = WaitForGpuWrite(kernel, (std::min)(ns, MaxWaitNs));
    const bool woken = result == ResultSuccess;
    if (woken) {
        state.woken.fetch_add(1, std::memory_order_relaxed);
    } else if (result == ResultTimedOut) {
        state.timed_out.fetch_add(1, std::memory_order_relaxed);
    }
    state.last_woken.store(woken, std::memory_order_relaxed);
    state.last_end_tick.store(kernel.HardwareTimer().GetTick(), std::memory_order_relaxed);
    return true;
}

} // namespace Kernel::Svc
