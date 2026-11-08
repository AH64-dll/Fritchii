// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <filesystem>
#include <string>
#include <string_view>

#include "common/common_types.h"

// A timeline of what the emulator's threads do and wait for, for a few seconds: each thread
// records spans (begin, duration, name) into its own buffer while a recording runs, and the
// recording is written as a Chrome trace (open it in ui.perfetto.dev or chrome://tracing).
// Off unless started: EDEN_NX2_TIMELINE=<dir> records once EDEN_NX2_TIMELINE_AT seconds after
// the first present (default 60) for EDEN_NX2_TIMELINE_SECONDS (default 5); the debug server's
// `timeline` method starts one at any time. While none runs, a span costs one relaxed load.
namespace Common::Nx2Timeline {

inline std::atomic<bool> g_active{false};
/// Fine-grained spans (per-draw stages, per-method calls) are recorded only with detail on: they
/// are about ten per draw and slow the GPU threads by a third while recording.
inline std::atomic<bool> g_detail{false};

[[nodiscard]] inline bool Detail() {
    return g_detail.load(std::memory_order_relaxed);
}

[[nodiscard]] inline bool Active() {
    return g_active.load(std::memory_order_relaxed);
}

/// The guest layer (tools/nx2/critical_path.py round 2): the kernel's thread switches, waits and
/// wakes with the guest thread ids, guest thread and kernel object names, and the GPU's writes to
/// guest memory, recorded only with EDEN_NX2_TIMELINE_GUEST=1 (tens of thousands of markers a
/// second). Set when a recording starts.
inline std::atomic<bool> g_guest{false};

/// Whether this recording has the guest layer. While none runs, one relaxed load (Active).
[[nodiscard]] inline bool Guest() {
    return Active() && g_guest.load(std::memory_order_relaxed);
}

/// EDEN_NX2_TIMELINE_GUEST=1 (read once): kernel objects are named as they are created, before
/// any recording, so the names are kept while this is set.
[[nodiscard]] bool GuestEnabled();

/// Nanoseconds on the clock the spans use.
[[nodiscard]] u64 Now();

/// A named number on a span or marker, written as the event's args: a command list's sequence
/// number, a syncpoint value, a GPU tick, how far behind the back was. `key` must be a literal;
/// an Arg without a key is not written.
struct Arg {
    constexpr Arg() = default;
    template <typename T>
    constexpr Arg(const char* key_, T value_) : key{key_}, value{static_cast<s64>(value_)} {}

    const char* key{};
    s64 value{};
};
inline constexpr size_t MaxArgs = 3;

/// Records a span on the calling thread. `name` must outlive the recording (a literal).
void Record(const char* name, u64 begin_ns, u64 end_ns);

/// The same with args. Consecutive spans of one name still merge: the merged span keeps each
/// arg's largest value (the deepest phase of a wait, the furthest a drain was behind).
void Record(const char* name, u64 begin_ns, u64 end_ns, Arg a, Arg b = {}, Arg c = {});

/// Records a zero-length marker on the calling thread.
void Instant(const char* name);

/// The same with args (markers never merge).
void Instant(const char* name, Arg a, Arg b = {}, Arg c = {});

/// Starts a recording of `seconds` written to `path` (`detail`: the fine-grained spans too).
/// False if one is already running.
bool Start(const std::filesystem::path& path, double seconds, bool detail = false);

/// The file the last recording wrote ("" until one finished), and whether one is running.
[[nodiscard]] std::string LastFile();

/// Called at each guest present: starts the EDEN_NX2_TIMELINE recording when it is due.
void OnPresent();

/// Whether the guest thread `thread_id` still needs a name in this recording (Guest() only). The
/// caller then reads it from guest memory and calls NameGuestThread; a thread without a name yet
/// is asked for again a few times.
[[nodiscard]] bool WantsGuestThreadName(u64 thread_id);

/// Names a guest thread in this recording: a thread_name metadata record of process 2 (the guest's
/// threads, by kernel thread id), next to the host threads of process 1.
void NameGuestThread(u64 thread_id, std::string_view name);

/// Names a kernel object a guest thread can wait on (the KSynchronizationObject pointer its
/// "guest wait" marker carries) in every recording while GuestEnabled(); written as object_name
/// metadata. Not on a hot path: at the object's creation, or when its role changes.
void NameObject(const void* object, std::string_view name);

/// Drops an object's name (it is being destroyed: its address can be reused).
void ForgetObject(const void* object);

class Span {
public:
    explicit Span(const char* name_) : name{name_}, on{Active()} {
        if (on) {
            begin = Now();
        }
    }
    ~Span() {
        if (on) {
            Record(name, begin, Now());
        }
    }
    Span(const Span&) = delete;
    Span& operator=(const Span&) = delete;

private:
    const char* name;
    bool on;
    u64 begin{};
};

/// A Span with args. Values known only at its end (which phase a wait reached) are set with Set.
class ArgSpan {
public:
    explicit ArgSpan(const char* name_, Arg a = {}, Arg b = {}, Arg c = {})
        : name{name_}, on{Active()}, args{a, b, c} {
        if (on) {
            begin = Now();
        }
    }
    ~ArgSpan() {
        if (on) {
            Record(name, begin, Now(), args[0], args[1], args[2]);
        }
    }
    ArgSpan(const ArgSpan&) = delete;
    ArgSpan& operator=(const ArgSpan&) = delete;

    /// Replaces arg `index` (below MaxArgs).
    void Set(size_t index, Arg arg) {
        if (on && index < MaxArgs) {
            args[index] = arg;
        }
    }

    [[nodiscard]] bool On() const {
        return on;
    }

private:
    const char* name;
    bool on;
    u64 begin{};
    std::array<Arg, MaxArgs> args;
};

} // namespace Common::Nx2Timeline
