// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/nx2_timeline.h"

namespace Common::Nx2Timeline {
namespace {

using ArgSet = std::array<Arg, MaxArgs>;

struct Event {
    u64 begin;
    u64 duration;
    const char* name;
    u32 args; ///< 1 + the index of the event's args in its buffer's arg chunks; 0 = none
};

// At most this many spans per thread and recording (the GPU threads make ~200,000 a second,
// the front's macros over a million: consecutive spans of one name merge, see Record).
constexpr size_t MaxEvents = size_t{1} << 22;
// Spans are kept in chunks of this many: one vector grown to millions of spans was copied whole
// at each growth, under the recording thread's span (GPUBack stalled 2.5 to 15.7 ms each time,
// which showed as the longest fence-release drains in Ocarina of Time traces).
constexpr size_t ChunkEvents = size_t{1} << 16;

struct ThreadBuffer {
    std::mutex mutex;
    std::vector<std::vector<Event>> chunks;
    size_t count{};
    // The args of the events that have any, chunked like the events (drains carry them).
    std::vector<std::vector<ArgSet>> arg_chunks;
    size_t arg_count{};
    u64 thread_id{};
    std::string name;

    [[nodiscard]] Event* Last() {
        return count == 0 ? nullptr : &chunks.back().back();
    }
    void Push(const Event& event) {
        if (chunks.empty() || chunks.back().size() == ChunkEvents) {
            chunks.emplace_back().reserve(ChunkEvents);
        }
        chunks.back().push_back(event);
        ++count;
    }
    /// Stores `args` and returns the Event::args slot that refers to them.
    [[nodiscard]] u32 PushArgs(const ArgSet& args) {
        if (arg_chunks.empty() || arg_chunks.back().size() == ChunkEvents) {
            arg_chunks.emplace_back().reserve(ChunkEvents);
        }
        arg_chunks.back().push_back(args);
        ++arg_count;
        return static_cast<u32>(arg_count);
    }
    [[nodiscard]] ArgSet& ArgsOf(const Event& event) {
        const size_t index = event.args - 1;
        return arg_chunks[index / ChunkEvents][index % ChunkEvents];
    }
    void Clear() {
        chunks.clear();
        count = 0;
        arg_chunks.clear();
        arg_count = 0;
    }
};
// Spans of the same name closer than this merge into one.
constexpr u64 MergeGapNs = 2000;

std::mutex g_registry_mutex;
std::vector<std::shared_ptr<ThreadBuffer>> g_buffers;
std::atomic<bool> g_running{false};
std::mutex g_last_mutex;
std::string g_last_file;
u64 g_start_ns{};
// For merging with an ETW trace recorded on the performance counter clock (tools/nx2/
// critical_path.py, round 2): the counter read right after g_start_ns, and its frequency. An ETW
// event at counter value q is at (q - g_start_qpc) * 1e6 / g_qpc_frequency us on the timeline.
s64 g_start_qpc{};
s64 g_qpc_frequency{};

thread_local std::shared_ptr<ThreadBuffer> t_buffer;

// The guest layer's names: guest threads (by kernel thread id, per recording) and kernel objects
// (by address, kept across recordings while GuestEnabled()).
struct GuestThreadName {
    std::string name;
    u32 tries{};
};
// A thread whose nn::os name is not set yet is asked for again this many times.
constexpr u32 MaxGuestNameTries = 64;
std::mutex g_names_mutex;
std::unordered_map<u64, GuestThreadName> g_guest_thread_names;
std::unordered_map<const void*, std::string> g_object_names;

std::string CurrentThreadName() {
#ifdef _WIN32
    using GetThreadDescriptionFn = HRESULT(WINAPI*)(HANDLE, PWSTR*);
    static const auto get_description = reinterpret_cast<GetThreadDescriptionFn>(
        reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"KernelBase.dll"),
                                               "GetThreadDescription")));
    if (get_description != nullptr) {
        PWSTR wide = nullptr;
        if (SUCCEEDED(get_description(GetCurrentThread(), &wide)) && wide != nullptr) {
            std::string name;
            for (const wchar_t* c = wide; *c != 0; ++c) {
                name.push_back(*c < 128 ? static_cast<char>(*c) : '?');
            }
            LocalFree(wide);
            if (!name.empty()) {
                return name;
            }
        }
    }
#endif
    return "thread";
}

u64 CurrentThreadId() {
#ifdef _WIN32
    return GetCurrentThreadId();
#else
    return std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xffffff;
#endif
}

ThreadBuffer& Buffer() {
    if (!t_buffer) {
        t_buffer = std::make_shared<ThreadBuffer>();
        t_buffer->thread_id = CurrentThreadId();
        t_buffer->name = CurrentThreadName();
        std::scoped_lock lock{g_registry_mutex};
        g_buffers.push_back(t_buffer);
    }
    return *t_buffer;
}

void AppendEscaped(std::string& out, const std::string& text) {
    for (const char c : text) {
        if (c == '"' || c == '\\') {
            out.push_back('\\');
        }
        out.push_back(static_cast<unsigned char>(c) < 0x20 ? ' ' : c);
    }
}

void Write(const std::filesystem::path& path, double seconds) {
    std::vector<std::shared_ptr<ThreadBuffer>> buffers;
    {
        std::scoped_lock lock{g_registry_mutex};
        buffers = g_buffers;
    }
    // otherData (ignored by trace viewers): the clock, for merging an ETW trace (g_start_qpc).
#ifdef _WIN32
    const u64 process_id = GetCurrentProcessId();
#else
    const u64 process_id = 0;
#endif
    std::string json = fmt::format(
        "{{\"displayTimeUnit\":\"ms\",\"otherData\":{{\"clock\":\"steady_clock\","
        "\"start_ns\":{},\"qpc_start\":{},\"qpc_frequency\":{},\"process_id\":{},"
        "\"seconds\":{:.3f}}},\"traceEvents\":[\n",
        g_start_ns, g_start_qpc, g_qpc_frequency, process_id, seconds);
    size_t total = 0;
    bool first = true;
    const auto comma = [&] {
        if (!first) {
            json += ",\n";
        }
        first = false;
    };
    if (g_guest.load(std::memory_order_relaxed)) {
        // The guest layer's names (tools/nx2/critical_path.py reads them; perfetto shows the
        // guest threads' names as a second process).
        std::scoped_lock lock{g_names_mutex};
        comma();
        json += R"({"name":"process_name","ph":"M","pid":2,"args":{"name":"guest threads"}})";
        for (const auto& [thread_id, entry] : g_guest_thread_names) {
            if (entry.name.empty()) {
                continue;
            }
            comma();
            json += fmt::format(R"({{"name":"thread_name","ph":"M","pid":2,"tid":{},"args":{{"name":")",
                                thread_id);
            AppendEscaped(json, entry.name);
            json += "\"}}";
        }
        for (const auto& [object, name] : g_object_names) {
            comma();
            json += fmt::format(
                R"({{"name":"object_name","ph":"M","pid":1,"tid":0,"args":{{"ptr":{},"name":")",
                reinterpret_cast<uintptr_t>(object));
            AppendEscaped(json, name);
            json += "\"}}";
        }
    }
    for (const auto& buffer : buffers) {
        std::vector<std::vector<Event>> chunks;
        std::vector<std::vector<ArgSet>> arg_chunks;
        size_t count{};
        {
            std::scoped_lock lock{buffer->mutex};
            chunks.swap(buffer->chunks);
            count = std::exchange(buffer->count, 0);
            arg_chunks.swap(buffer->arg_chunks);
            buffer->arg_count = 0;
        }
        if (count == 0) {
            continue;
        }
        comma();
        json += fmt::format(R"({{"name":"thread_name","ph":"M","pid":1,"tid":{},"args":{{"name":")",
                            buffer->thread_id);
        AppendEscaped(json, buffer->name);
        json += "\"}}";
        std::vector<Event> events;
        events.reserve(count);
        for (const auto& chunk : chunks) {
            events.insert(events.end(), chunk.begin(), chunk.end());
        }
        chunks.clear();
        for (const Event& event : events) {
            comma();
            json += R"({"name":")";
            AppendEscaped(json, event.name);
            json += fmt::format(R"(","ph":"{}","pid":1,"tid":{},"ts":{:.3f})",
                                event.duration == 0 ? "i" : "X", buffer->thread_id,
                                static_cast<double>(event.begin - g_start_ns) / 1000.0);
            if (event.duration != 0) {
                json += fmt::format(R"(,"dur":{:.3f})", static_cast<double>(event.duration) / 1000.0);
            } else {
                json += R"(,"s":"t")";
            }
            if (event.args != 0) {
                const size_t slot = event.args - 1;
                const ArgSet& args = arg_chunks[slot / ChunkEvents][slot % ChunkEvents];
                json += R"(,"args":{)";
                bool first_arg = true;
                for (const Arg& arg : args) {
                    if (arg.key == nullptr) {
                        continue;
                    }
                    json += fmt::format(R"({}"{}":{})", first_arg ? "" : ",", arg.key, arg.value);
                    first_arg = false;
                }
                json += '}';
            }
            json += '}';
        }
        total += count;
    }
    json += "\n]}\n";
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(json.data(), static_cast<std::streamsize>(json.size()));
    file.close();
    LOG_INFO(Debug, "Timeline: {} spans of {} threads over {:.1f} s written to {}", total,
             buffers.size(), seconds, Common::FS::PathToUTF8String(path));
    std::scoped_lock lock{g_last_mutex};
    g_last_file = Common::FS::PathToUTF8String(path);
}

} // Anonymous namespace

u64 Now() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

namespace {

void RecordSpan(const char* name, u64 begin_ns, u64 end_ns, const ArgSet* args) {
    ThreadBuffer& buffer = Buffer();
    std::scoped_lock lock{buffer.mutex};
    if (!Active()) {
        return;
    }
    if (Event* const last = buffer.Last()) {
        const u64 last_end = last->begin + last->duration;
        if (last->name == name && last->duration != 0 && begin_ns >= last_end &&
            begin_ns - last_end < MergeGapNs && (last->args != 0) == (args != nullptr)) {
            last->duration = end_ns - last->begin;
            if (args != nullptr) {
                ArgSet& kept = buffer.ArgsOf(*last);
                for (size_t i = 0; i < MaxArgs; ++i) {
                    kept[i].value = (std::max)(kept[i].value, (*args)[i].value);
                }
            }
            return;
        }
    }
    if (buffer.count < MaxEvents) {
        const u32 slot = args != nullptr ? buffer.PushArgs(*args) : 0;
        buffer.Push({begin_ns, (std::max)(end_ns - begin_ns, u64{1}), name, slot});
    }
}

void RecordInstant(const char* name, const ArgSet* args) {
    if (!Active()) {
        return;
    }
    ThreadBuffer& buffer = Buffer();
    std::scoped_lock lock{buffer.mutex};
    if (buffer.count < MaxEvents && Active()) {
        const u32 slot = args != nullptr ? buffer.PushArgs(*args) : 0;
        buffer.Push({Now(), 0, name, slot});
    }
}

} // Anonymous namespace

void Record(const char* name, u64 begin_ns, u64 end_ns) {
    RecordSpan(name, begin_ns, end_ns, nullptr);
}

void Record(const char* name, u64 begin_ns, u64 end_ns, Arg a, Arg b, Arg c) {
    const ArgSet args{a, b, c};
    RecordSpan(name, begin_ns, end_ns, &args);
}

void Instant(const char* name) {
    RecordInstant(name, nullptr);
}

void Instant(const char* name, Arg a, Arg b, Arg c) {
    if (!Active()) {
        return;
    }
    const ArgSet args{a, b, c};
    RecordInstant(name, &args);
}

bool Start(const std::filesystem::path& path, double seconds, bool detail) {
    if (g_running.exchange(true)) {
        return false;
    }
    seconds = std::clamp(seconds, 0.1, 30.0);
    {
        std::scoped_lock lock{g_registry_mutex};
        for (const auto& buffer : g_buffers) {
            std::scoped_lock buffer_lock{buffer->mutex};
            buffer->Clear();
        }
    }
    g_start_ns = Now();
#ifdef _WIN32
    {
        LARGE_INTEGER counter{};
        LARGE_INTEGER frequency{};
        QueryPerformanceCounter(&counter);
        QueryPerformanceFrequency(&frequency);
        g_start_qpc = static_cast<s64>(counter.QuadPart);
        g_qpc_frequency = static_cast<s64>(frequency.QuadPart);
    }
#endif
    {
        std::scoped_lock lock{g_names_mutex};
        g_guest_thread_names.clear();
    }
    g_guest.store(GuestEnabled(), std::memory_order_relaxed);
    g_detail.store(detail, std::memory_order_relaxed);
    g_active.store(true, std::memory_order_release);
    LOG_INFO(Debug, "Timeline: recording {:.1f} s{}", seconds,
             g_guest.load(std::memory_order_relaxed) ? " with the guest layer" : "");
    std::thread([path, seconds] {
        std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
        g_active.store(false, std::memory_order_release);
        // Spans that began before the end finish their Record under their buffer's lock.
        std::this_thread::sleep_for(std::chrono::milliseconds{50});
        Write(path, seconds);
        g_running.store(false);
    }).detach();
    return true;
}

std::string LastFile() {
    std::scoped_lock lock{g_last_mutex};
    return g_last_file;
}

bool GuestEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_TIMELINE_GUEST");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

bool WantsGuestThreadName(u64 thread_id) {
    if (!Guest()) {
        return false;
    }
    std::scoped_lock lock{g_names_mutex};
    GuestThreadName& entry = g_guest_thread_names[thread_id];
    if (!entry.name.empty() || entry.tries >= MaxGuestNameTries) {
        return false;
    }
    ++entry.tries;
    return true;
}

void NameGuestThread(u64 thread_id, std::string_view name) {
    if (name.empty()) {
        return;
    }
    std::scoped_lock lock{g_names_mutex};
    g_guest_thread_names[thread_id].name = name;
}

void NameObject(const void* object, std::string_view name) {
    if (!GuestEnabled() || object == nullptr) {
        return;
    }
    std::scoped_lock lock{g_names_mutex};
    g_object_names[object] = name;
}

void ForgetObject(const void* object) {
    if (!GuestEnabled()) {
        return;
    }
    std::scoped_lock lock{g_names_mutex};
    g_object_names.erase(object);
}

void OnPresent() {
    static const char* const directory = std::getenv("EDEN_NX2_TIMELINE");
    if (directory == nullptr) {
        return;
    }
    static const auto first_present = std::chrono::steady_clock::now();
    static bool started = false;
    if (started) {
        return;
    }
    static const double at = [] {
        const char* value = std::getenv("EDEN_NX2_TIMELINE_AT");
        return value != nullptr ? std::atof(value) : 60.0;
    }();
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - first_present).count() <
        at) {
        return;
    }
    started = true;
    const char* value = std::getenv("EDEN_NX2_TIMELINE_SECONDS");
    const double seconds = value != nullptr ? std::atof(value) : 5.0;
    const char* detail = std::getenv("EDEN_NX2_TIMELINE_DETAIL");
    Start(std::filesystem::path{directory} /
              fmt::format("timeline_{}.json", std::chrono::duration_cast<std::chrono::seconds>(
                                                  std::chrono::system_clock::now().time_since_epoch())
                                                  .count()),
          seconds, detail != nullptr && detail[0] == '1');
}

} // namespace Common::Nx2Timeline
