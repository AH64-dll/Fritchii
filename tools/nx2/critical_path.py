#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

"""Which chain of waits decided when each frame was presented, from an Nx2Timeline trace.

    python3 -I tools/nx2/critical_path.py timeline_<stamp>.json [--json out.json] [--top 10]

A frame is [guest present k, guest present k+1) on the presenting guest thread. From present
k+1 the walk goes backward in time: on work it stays on the thread, at a wait it jumps to the
thread the wait was waiting on (WAITS, the waited-on table: most waits here spin before they
block, so the span names, not context switches, say what a thread waited for) at the moment
that thread let it go, until it reaches present k. The segments add up to the frame time; the
part filed as "latency" (a waiter's wake after its waker's marker) or "unknown" (a wait whose
waker no marker names) is listed per thread. Per stage and wait type the ms per frame are
compared between the median and the slowest frames (p90 and up); the median frame's path must
add up to its frame time within 5% before the aggregates mean anything.

Round 2, the guest layer (a capture with EDEN_NX2_TIMELINE_GUEST=1): the kernel's "guest switch",
"guest wait" and "guest wake" markers give each guest thread (kernel thread id, nn::os name)
running / waiting / ready intervals across the cores. The frame thread is the guest thread its
core ran at each "guest present"; on it the walk follows the guest thread, not the core: a wait
jumps to the guest thread that woke it at the wake (or to the host thread that did: the fence
thread, the back), a time-out stays as a timer, a vsync slot wait is the display, and OoT's
present poll sleep resolves to the GPU's write of its event word, then to the writer. Two more
tables: ms on the path per thread and kind, and who the frame thread waited for, with the
waker's own state at the wake. Without guest markers the walk is round 1's.

An ETW trace (CSwitch, ReadyThread) could still be merged through Trace.wakers and the clock in
the trace's otherData (qpc_start, qpc_frequency); see merge_etw (not done).
"""

import argparse
import os
import bisect
import collections
import json
import math
import statistics
import sys

# Thread names as the emulator sets them (SetCurrentThreadName) -> stage.
ROLES = {
    "GPU": "front",
    "GPUBack": "back",
    "VulkanWorker": "worker",
    "GPUFencingThread": "fence",
    "VulkanPresent": "present",
    "VulkanPresentWait": "present-wait",
}

# What time outside every span is on a thread of a stage: those loops record their waits, so
# the rest is work (zones the timeline doesn't record without detail, the JIT on guest cores).
GAP_LABEL = {
    "guest": "guest: running",
    "front": "front: other work",
    "back": "back: other work (methods)",
    "worker": "worker: other work",
    "fence": "fence: other work",
    "present": "present: other work",
    "present-wait": "present-wait: other work",
}

# DrainSite names (src/video_core/gpu_split.h drain_site_names): the front waits for the back.
DRAIN_SITES = (
    "other", "fetch drain", "acquire", "semaphore flush", "fence release", "list writers",
    "list sync", "list flush", "macro dma read", "macro dma writes", "macro dma flush",
    "gp sync fence", "map", "front start", "front verify", "macro dma check",
    "macro dma dispatch", "fence writers", "qmd read", "list copy writers",
)

# The waited-on table: span name -> (kind, stage waited on, resolver). Kinds: "wait" jumps to
# the stage, "timer" (sleep granularity, a latency) and "pacing" (an intended hold) stay on the
# thread, "display" ends at the display. Span names not here are work of their thread's stage.
# Resolvers (Walker.resolve): at_end = the waited-on thread let go when the wait ended (spins);
# marker = the latest of these markers in the wait; syncpoint = the increment that reached the
# span's (syncpoint, value), else the given kind; gpu_tick = the tick's vkQueueSubmit, then the host GPU;
# gpu_any = the latest vkQueueSubmit in the wait; wake_window = a marker shortly before the end;
# poll = OoT's poll: a fence release during the sleep.
WAITS = {
    **{name: ("wait", "back", ("at_end",)) for name in DRAIN_SITES},
    "split drain": ("wait", "back", ("at_end",)),
    "writer wait": ("wait", "back", ("at_end",)),
    "list submit wait": ("wait", "back", ("at_end",)),
    "list GPU wait": ("wait", "gpu", ("gpu_any",)),
    "query DMA wait": ("wait", "gpu", ("gpu_any",)),
    "wait for host GPU": ("wait", "gpu", ("gpu_tick",)),
    "wait for Vulkan worker": ("wait", "worker", ("at_end",)),
    "idle": ("wait", "guest", ("marker", "submit list", "submit command")),
    "pending sleep": ("timer", "guest", None),
    "split wait": ("wait", "front", ("at_end",)),
    "worker: wait for work": ("wait", "back", ("at_end",)),
    "fence: wait for work": ("wait", "back", ("at_end",)),
    "fence: wait for fence": ("wait", "gpu", ("gpu_any",)),
    # The frame is queued by whoever runs the renderer's recorded present (the Vulkan worker, or
    # the back under the split when the worker runs inline): see Walker.resolve's pending rule.
    "present: wait for frame": ("wait", "back", ("marker", "present: frame queued")),
    "present: wait for free frame": ("wait", "present", ("at_end",)),
    "present: wait for frame render": ("wait", "gpu", ("gpu_any",)),
    "present-wait: wait for work": ("wait", "present", ("at_end",)),
    "present-wait: vkWaitForPresentKHR": ("display", "display", None),
    "present: wait until shown": ("pacing", "display", None),
    "present: host pacing hold": ("pacing", None, None),
    "present: pace to vblank": ("pacing", None, None),
    "present: wait for frame fence": ("wait", "back",
                                      ("syncpoint", "syncpoint: guest increment", "unknown")),
    # The ioctl itself rarely blocks (it returns Timeout and the guest blocks in "guest core idle").
    "guest: GPU event wait": ("wait", "fence", ("syncpoint", "syncpoint: host increment", "busy")),
    "guest: wait for GPU command": ("wait", "back", ("at_end",)),
    "guest core idle": ("wait", "fence", ("wake_window", "GPU event signalled")),
    "OoT: present poll sleep": ("timer", "fence", ("poll",)),
}
# Spans that label everything nested in them (the guest's other threads run inside a poll sleep).
DOMINANT = {"OoT: present poll sleep"}
FRAME_SPAN = "guest present"
EPS = 0.0005  # us: the trace's resolution


class Trace:
    def __init__(self):
        self.other = {}
        self.names = {}  # tid -> thread name
        self.spans = collections.defaultdict(list)  # tid -> [(begin, end, name, args)]
        self.instants = collections.defaultdict(list)  # name -> [(ts, tid, args)]
        self.pieces = {}  # tid -> ([starts], [(begin, end, span index)])
        self.wakers = {}  # (tid, wait end us) -> (waker tid, ready us): filled by merge_etw
        self.guest_names = {}  # kernel thread id -> nn::os name (pid-2 thread_name metadata)
        self.object_names = {}  # KSynchronizationObject pointer -> name (object_name metadata)
        self.guest = None  # GuestLayer when the trace has "guest switch" markers
        self._roles = {}
        self._tids = {}

    def role(self, tid):
        if isinstance(tid, tuple):  # ("guest", kernel thread id): the guest layer
            return "guest"
        if tid not in self._roles:
            name = self.names.get(tid, "")
            if name in ROLES:
                self._roles[tid] = ROLES[name]
            elif name.startswith("CPUCore_") or name == "CPUThread":
                self._roles[tid] = "guest"
            else:
                self._roles[tid] = "other"
        return self._roles[tid]

    def tid_of(self, role):
        """The thread of a stage (the busiest, should a name repeat)."""
        if role not in self._tids:
            tids = [tid for tid in self.spans if self.role(tid) == role]
            self._tids[role] = max(tids, key=lambda tid: len(self.spans[tid])) if tids else None
        return self._tids[role]

    def recorder(self, names):
        """The host thread (not a guest core) that records these markers most, or None."""
        key = ("recorder",) + tuple(names)
        if key not in self._tids:
            tids = collections.Counter(m[1] for name in names for m in self.instants.get(name, ()))
            tid = tids.most_common(1)[0][0] if tids else None
            self._tids[key] = tid if tid is not None and self.role(tid) not in ("guest", "other") \
                else None
        return self._tids[key]

    def qpc_to_us(self, qpc):
        """An ETW time stamp (performance counter) on the timeline's clock."""
        return (qpc - self.other["qpc_start"]) * 1e6 / self.other["qpc_frequency"]

    def label(self, tid):
        """A thread's name for the tables: host threads by name, guest threads as "guest <name>"."""
        if isinstance(tid, tuple):
            return f"guest {self.guest.name(tid[1])}" if self.guest else f"guest {tid[1]}"
        return self.names.get(tid, str(tid))


def load(path):
    trace = Trace()
    with open(path, encoding="utf-8") as file:
        first = file.readline()
        if "otherData" in first and first.rstrip().endswith("["):
            trace.other = json.loads(first.rstrip()[:-len('"traceEvents":[')].rstrip(",") + "}")[
                "otherData"]
        else:  # an older trace or another writer: read it whole
            file.seek(0)
            events = json.load(file)["traceEvents"]
            for event in events:
                add_event(trace, event)
            return finish(trace)
        for line in file:
            line = line.strip().rstrip(",")
            if line.startswith("{"):
                add_event(trace, json.loads(line))
    return finish(trace)


def add_event(trace, event):
    phase = event.get("ph")
    tid = event.get("tid", 0)
    if phase == "M" and event.get("name") == "thread_name":
        if event.get("pid", 1) == 2:  # the guest's threads, by kernel thread id
            trace.guest_names[tid] = event["args"]["name"]
        else:
            trace.names[tid] = event["args"]["name"]
    elif phase == "M" and event.get("name") == "object_name":
        trace.object_names[event["args"]["ptr"]] = event["args"]["name"]
    elif phase == "X":
        begin = event["ts"]
        trace.spans[tid].append((begin, begin + event["dur"], event["name"], event.get("args")))
    elif phase == "i":
        trace.instants[event["name"]].append((event["ts"], tid, event.get("args") or {}))


def finish(trace):
    for markers in trace.instants.values():
        markers.sort(key=lambda m: m[0])
    if trace.instants.get("guest switch"):
        # The guest layer follows the guest threads: a core's poll sleep span (recorded on the
        # core the thread resumed on, from when it began to sleep elsewhere) does not label the
        # other threads that ran on that core meanwhile. Its sleep is the thread's Sleep wait.
        polls = []
        for tid, spans in trace.spans.items():
            if trace.role(tid) == "guest":
                polls += [(s[0], s[1], tid, s[3] or {}) for s in spans if s[2] in DOMINANT]
                spans[:] = [s for s in spans if s[2] not in DOMINANT]
        trace.guest = GuestLayer(trace, sorted(polls))
    for tid, spans in trace.spans.items():
        spans.sort(key=lambda s: (s[0], -s[1]))
        trace.pieces[tid] = flatten(spans)
    trace.instant_ts = {name: [m[0] for m in markers] for name, markers in trace.instants.items()}
    trace.submit_by_tick = {}
    for ts, tid, args in trace.instants.get("vkQueueSubmit", ()):
        trace.submit_by_tick.setdefault(args.get("tick"), (ts, tid))
    trace.syncpoints = {}  # (marker name, id) -> sorted [(value, ts, tid)]
    for name in ("syncpoint: guest increment", "syncpoint: host increment"):
        for ts, tid, args in trace.instants.get(name, ()):
            trace.syncpoints.setdefault((name, args.get("id")), []).append((args.get("value", 0), ts, tid))
    for entries in trace.syncpoints.values():
        entries.sort()
    fence = trace.tid_of("fence")
    releases = [s for s in trace.spans.get(fence, ()) if s[2] == "fence: release"]
    trace.release_ends = sorted(s[1] for s in releases)
    return trace


def flatten(spans):
    """Non-overlapping pieces (begin, end, span index) covering all time; index -1 = between
    spans, -2 = before the first or after the last. The innermost span labels a piece, unless a
    DOMINANT one encloses it. Crossing spans (a guest thread resuming on another core) are cut."""
    pieces = []
    if not spans:
        return [-math.inf], [(-math.inf, math.inf, -2)]
    ends = [s[1] for s in spans]
    stack = []

    def label():
        for index in stack:
            if spans[index][2] in DOMINANT:
                return index
        return stack[-1] if stack else -1

    def emit(begin, end, index):
        if end > begin:
            pieces.append((begin, end, index))

    pieces.append((-math.inf, spans[0][0], -2))
    current = spans[0][0]
    for index, (begin, _end, _name, _args) in enumerate(spans):
        while stack and ends[stack[-1]] <= begin:
            emit(current, ends[stack[-1]], label())
            current = max(current, ends[stack[-1]])
            stack.pop()
        emit(current, begin, label())
        current = max(current, begin)
        if stack:
            ends[index] = min(ends[index], ends[stack[-1]])
        if ends[index] > begin:
            stack.append(index)
    while stack:
        emit(current, ends[stack[-1]], label())
        current = max(current, ends[stack[-1]])
        stack.pop()
    pieces.append((current, math.inf, -2))
    return [p[0] for p in pieces], pieces


# "guest wait" reason codes (KThread::RecordWaitForTimeline): ThreadWaitReasonForDebugging, plus
# 7 for an ArbitrateLock (a mutex), and 0 here for a wake whose wait was not marked (the kernel's
# own light locks and condition variables).
WAIT_REASONS = {0: "kernel wait", 1: "sleep", 2: "IPC", 3: "sync", 4: "condvar", 5: "arbitration",
                6: "suspended", 7: "mutex"}
WAKE_RESULTS = {0: "signalled", 1: "timed out", 2: "cancelled", 3: "terminated", 4: "other"}
POLL_WORDS = "OoT: present poll word"
GPU_WRITE = "GPU write: "
TIMER_THREADS = {"HostTiming"}


class Wait:
    __slots__ = ("begin", "end", "reason", "key", "result", "by", "host", "obj")

    def __init__(self, begin, reason, key):
        self.begin, self.reason, self.key = begin, reason, key
        self.end, self.result, self.by, self.host, self.obj = math.inf, None, 0, None, None


class GuestLayer:
    """Per guest thread (kernel thread id): running intervals (from "guest switch", with the host
    core thread), waits (a "guest wait" to the thread's next "guest wake", with the waker), and
    in between ready (woken or preempted, not yet switched in)."""

    def __init__(self, trace, polls):
        self.trace = trace
        # Per host core thread: the guest thread running from each switch on (0 = idle).
        switches = collections.defaultdict(list)
        for ts, tid, args in trace.instants.get("guest switch", ()):
            switches[tid].append((ts, args.get("to", 0), args.get("from", 0)))
        self.cores = {}
        running = collections.defaultdict(list)
        for host, entries in switches.items():
            entries.sort(key=lambda e: e[0])
            self.cores[host] = ([e[0] for e in entries], [e[1] for e in entries], entries[0][2])
            if entries[0][2]:
                running[entries[0][2]].append((-math.inf, entries[0][0], host))
            for (ts, to, _from), nxt in zip(entries, entries[1:] + [(math.inf, 0, 0)]):
                if to and nxt[0] > ts:
                    running[to].append((ts, nxt[0], host))
        self.running = {}
        for gid, intervals in running.items():
            intervals.sort()
            self.running[gid] = ([i[0] for i in intervals], intervals)
        # Waits per guest thread, closed by the thread's next wake.
        events = collections.defaultdict(list)
        for ts, _tid, args in trace.instants.get("guest wait", ()):
            events[args.get("tid")].append((ts, 0, args))
        for ts, _tid, args in trace.instants.get("guest wake object", ()):
            events[args.get("tid")].append((ts, 1, args))
        for ts, tid, args in trace.instants.get("guest wake", ()):
            events[args.get("tid")].append((ts, 2, dict(args, host=tid)))
        self.waits = {}
        for gid, entries in events.items():
            entries.sort(key=lambda e: (e[0], e[1]))
            waits, current, woken_by = [], None, None
            for ts, order, args in entries:
                if order == 0:
                    if current is not None:  # no wake seen: ends where the next one begins
                        current.end = ts
                        waits.append(current)
                    current = Wait(ts, args.get("reason", 0), args.get("key", 0))
                elif order == 1:
                    woken_by = args.get("object")
                else:
                    if current is None:  # an unmarked wait: from the thread's last switch-out
                        current = Wait(self._last_out(gid, ts), 0, 0)
                    current.end, current.result = ts, args.get("result", 4)
                    current.by, current.host, current.obj = args.get("by", 0), args["host"], woken_by
                    waits.append(current)
                    current, woken_by = None, None
            if current is not None:
                waits.append(current)
            self.waits[gid] = ([w.begin for w in waits], waits)
        # OoT's present poll: each sleep span with the words of its event object, and the GPU
        # writes by word (host pointer and GPU address).
        words = trace.instants.get(POLL_WORDS, [])
        word_ts = [w[0] for w in words]
        self.polls = []
        for begin, end, host, args in polls:
            lo = bisect.bisect_left(word_ts, begin - 200.0)
            hi = bisect.bisect_right(word_ts, begin + EPS)
            keys = set()
            for _ts, _tid, word in words[lo:hi]:
                for kind in ("host", "gpu"):
                    if word.get(kind):
                        keys.add((kind, word[kind]))
            self.polls.append((begin, end, host, frozenset(keys), args.get("object", 0)))
        self.poll_begins = [p[0] for p in self.polls]
        ones = collections.defaultdict(list)  # word -> sorted [(ts, writer tid)] of writes of a 1
        for name, markers in trace.instants.items():
            if name.startswith(GPU_WRITE):
                for ts, tid, args in markers:
                    if args.get("value") != 1:
                        continue
                    for kind in ("host", "gpu"):
                        if args.get(kind):
                            ones[(kind, args[kind])].append((ts, tid))
        self.ones = {}
        for key, entries in ones.items():
            entries.sort()
            self.ones[key] = ([e[0] for e in entries], entries)

    def name(self, gid):
        return self.trace.guest_names.get(gid) or f"thread {gid}"

    def _last_out(self, gid, t):
        starts, intervals = self.running.get(gid, ([], []))
        index = bisect.bisect_left(starts, t) - 1
        return intervals[index][1] if index >= 0 and intervals[index][1] <= t else -math.inf

    def on_core(self, host, t):
        """The guest thread a core's host thread ran just before t (0: idle or not known)."""
        core = self.cores.get(host)
        if core is None:
            return 0
        index = bisect.bisect_left(core[0], t) - 1
        return core[1][index] if index >= 0 else core[2]

    def state_at(self, gid, t):
        """("run", begin, host) / ("wait", Wait) / ("ready", begin, Wait or None) / ("none",)
        for the moment just before t."""
        starts, intervals = self.running.get(gid, ([], []))
        index = bisect.bisect_left(starts, t) - 1
        last_out = -math.inf
        if index >= 0:
            begin, end, host = intervals[index]
            if end >= t - EPS:
                return ("run", begin, host)
            last_out = end
        begins, waits = self.waits.get(gid, ([], []))
        index = bisect.bisect_left(begins, t) - 1
        if index >= 0:
            wait = waits[index]
            if wait.end >= t - EPS:
                return ("wait", wait)
            if wait.end > last_out:
                return ("ready", wait.end, wait)
        if last_out > -math.inf:
            return ("ready", last_out, None)
        return ("none",)

    def poll_of(self, wait):
        """The present poll sleep span that is this Sleep wait, if any."""
        index = bisect.bisect_right(self.poll_begins, wait.begin + EPS) - 1
        if index >= 0 and self.polls[index][0] >= wait.begin - 500.0:
            return self.polls[index]
        return None

    def poll_write(self, poll, low, high, after=False):
        """The latest GPU write of a 1 to one of the poll's words in [low, high] (after: the first
        one after `low` instead, within 100 ms), as (ts, writer tid)."""
        best = None
        for key in poll[3]:
            stamps, entries = self.ones.get(key, ([], []))
            if after:
                index = bisect.bisect_right(stamps, low)
                if index < len(entries) and entries[index][0] <= low + 100_000.0 and \
                        (best is None or entries[index][0] < best[0]):
                    best = entries[index]
            else:
                index = bisect.bisect_right(stamps, high) - 1
                if index >= 0 and entries[index][0] >= low and (best is None or entries[index][0] > best[0]):
                    best = entries[index]
        return best

    def object_label(self, pointer):
        return self.trace.object_names.get(pointer) or f"object {pointer:#x}"

    def wait_label(self, gid, wait):
        reason = WAIT_REASONS.get(wait.reason, f"reason {wait.reason}")
        if wait.reason == 3:
            what = f" on {self.object_label(wait.obj or wait.key)}"
        elif wait.reason in (4, 5, 7):
            what = f" on {wait.key:#x}"
        else:
            what = ""
        return f"{self.name(gid)}: {reason}{what}"

    def waker_label(self, wait):
        if wait.by:
            return f"guest {self.name(wait.by)}"
        if wait.host is None:
            return "?"
        host = self.trace.names.get(wait.host, str(wait.host))
        return f"{host} (timer)" if wait.result == 1 else host

    def is_display(self, wait):
        return wait.reason == 3 and "vsync" in self.object_label(wait.obj or wait.key)

    def waker_state(self, wait):
        """What the waker was doing when it woke the waiter: a guest waker's last wait before it
        (so a chain main -> render thread -> GPU reads as one line), a host waker's span."""
        if wait.by:
            begins, waits = self.waits.get(wait.by, ([], []))
            index = bisect.bisect_right(begins, wait.end) - 1
            while index >= 0 and waits[index].end > wait.end:
                index -= 1
            if index < 0:
                return "running"
            previous = waits[index]
            return f"running after {self.wait_label(wait.by, previous).split(': ', 1)[1]} " \
                   f"({WAKE_RESULTS.get(previous.result, '?')}, by {self.waker_label(previous)})"
        if wait.host is None:
            return "?"
        starts, pieces = self.trace.pieces.get(wait.host) or flatten([])
        piece = pieces[max(bisect.bisect_left(starts, wait.end) - 1, 0)]
        return self.trace.spans[wait.host][piece[2]][2] if piece[2] >= 0 else "between spans"


def merge_etw(trace, csv_path):
    """Round 2 (not done yet): read `tracerpt trace.etl -of CSV` output, convert time stamps with
    trace.qpc_to_us, and fill trace.wakers[(waiting tid, wait end)] = (waker tid, ready time)
    from ReadyThread for blocking waits; Walker.resolve uses a waker found there before any
    marker heuristic. CSwitch gives ready-but-not-running time per thread (scheduling latency)."""
    raise NotImplementedError(f"ETW merge is round 2 ({csv_path})")


class Walker:
    def __init__(self, trace, poll_release="latest", wake_window_us=500.0):
        self.trace = trace
        self.poll_release = poll_release
        self.wake_window = wake_window_us

    def piece_at(self, tid, t):
        """The piece of `tid` that holds the moment just before t."""
        starts, pieces = self.trace.pieces.get(tid) or flatten([])
        return pieces[max(bisect.bisect_left(starts, t) - 1, 0)]

    def latest_marker(self, names, low, high):
        best = None
        for name in names:
            stamps = self.trace.instant_ts.get(name, [])
            index = bisect.bisect_right(stamps, high) - 1
            if index >= 0 and stamps[index] >= low:
                marker = self.trace.instants[name][index]
                if best is None or marker[0] > best[0]:
                    best = marker
        return best

    def walk(self, frame_tid, start, end):
        """Segments (tid, begin, end, stage, kind, label) of the path from `end` back to `start`."""
        trace = self.trace
        segments = []
        self.edges = []  # the guest waits on the path: (waiter, label, begin, end, waker, state)
        tid, t = frame_tid, end
        visited = set()
        steps = 0

        def add(on_tid, begin, until, stage, kind, label):
            begin = max(begin, start)
            if until <= begin:
                return
            last = segments[-1] if segments else None
            if last and last[0] == on_tid and last[3:] == (stage, kind, label) and \
                    abs(last[1] - until) <= EPS:
                # The same thing split by a nested span (the walk goes backward).
                segments[-1] = (on_tid, begin, last[2], stage, kind, label)
            else:
                segments.append((on_tid, begin, until, stage, kind, label))

        guest = trace.guest

        def canonical(on_tid, at):
            # On a guest core: follow the guest thread it ran, across cores.
            if guest is not None and not isinstance(on_tid, tuple) and trace.role(on_tid) == "guest":
                running = guest.on_core(on_tid, at)
                if running:
                    return ("guest", running)
            return on_tid

        while t > start + EPS:
            steps += 1
            if steps > 2_000_000:
                break
            tid = canonical(tid, t)
            role = trace.role(tid)
            if isinstance(tid, tuple):
                step = self.walk_guest(tid, t, start, add)
                if step[0] == "stay":
                    t = step[1]
                    continue
                _how, next_tid, next_t, low, name = step
            else:
                piece_begin, _piece_end, index = self.piece_at(tid, t)
                low = max(piece_begin, start)
                if index == -2:
                    add(tid, low, t, role, "unknown", f"{role}: no events")
                    t = low
                    continue
                if index == -1:
                    add(tid, low, t, role, "busy" if role in GAP_LABEL else "unknown",
                        GAP_LABEL.get(role, f"{trace.names.get(tid, tid)}: untracked"))
                    t = low
                    continue
                span = trace.spans[tid][index]
                name = span[2]
                kind, target, resolver = WAITS.get(name, ("busy", None, None))
                if guest is not None and name == "guest core idle":
                    # The guest layer follows the threads: an idle core is nobody's wait.
                    add(tid, low, t, role, "unknown", "guest core idle (no guest thread)")
                    t = low
                    continue
                if kind != "wait" and resolver is None:
                    add(tid, low, t, "display" if kind == "display" else role, kind, name)
                    t = low
                    continue
                jump = self.resolve(tid, t, low, span, target, resolver, add)
                if jump is None:  # filed on this thread
                    t = low
                    continue
                next_tid, next_t = jump
            next_tid = canonical(next_tid, next_t)
            if next_t >= t - EPS and (next_tid, next_t) in visited:
                # Each waits on the other at this moment (a spin that ended as the other began
                # to wait): the one whose wait began later worked until then. The time from
                # there is the hand-off's latency.
                other_begin, _other_end, other_index = self.piece_at(next_tid, next_t)
                if other_index >= 0 and other_begin > low:
                    other = trace.spans[next_tid][other_index][2]
                    add(next_tid, other_begin, t, trace.role(next_tid), "latency",
                        f"{other} (hand-off)")
                    tid, t = next_tid, max(other_begin, start)
                else:
                    add(tid, low, t, role, "latency", f"{name} (hand-off)")
                    t = low
                continue
            visited.add((tid, t))
            tid, t = next_tid, next_t
        return segments

    def walk_guest(self, gid, t, start, add):
        """One step back on a guest thread: ("stay", t') after filing [t', t] on it, or
        ("jump", tid, t', low, label) to go on on the thread it waited for."""
        trace, guest = self.trace, self.trace.guest
        name = guest.name(gid[1])
        state = guest.state_at(gid[1], t)
        if state[0] == "run":
            # Running on a core: that host thread's spans say what it did (an HLE service's
            # wait, a present), its gaps are guest code.
            host = state[2]
            piece_begin, _end, index = self.piece_at(host, t)
            low = max(piece_begin, state[1], start)
            span = trace.spans[host][index] if index >= 0 else None
            kind, target, resolver = WAITS.get(span[2], ("busy", None, None)) if span else \
                ("busy", None, None)
            if span is None or span[2] == "guest core idle":
                add(gid, low, t, "guest", "busy", f"guest: {name} running")
                return ("stay", low)
            if kind != "wait" and resolver is None:
                add(gid, low, t, "display" if kind == "display" else "guest", kind, span[2])
                return ("stay", low)

            def add_on_guest(on_tid, begin, until, stage, kind_, label):
                add(gid if on_tid == host else on_tid, begin, until, stage, kind_, label)

            jump = self.resolve(host, t, low, span, target, resolver, add_on_guest)
            if jump is None:
                return ("stay", low)
            return ("jump", jump[0], jump[1], low, span[2])
        if state[0] == "ready":
            low = max(state[1], start)
            why = f"woken by {guest.waker_label(state[2])}" if state[2] else "preempted"
            add(gid, low, t, "guest", "latency", f"guest: {name} ready, not running ({why})")
            return ("stay", low)
        if state[0] == "none":
            add(gid, start, t, "guest", "unknown", f"guest: {name} no events")
            return ("stay", start)
        return self.resolve_guest_wait(gid, state[1], t, start, add)

    def resolve_guest_wait(self, gid, wait, t, start, add):
        """A guest thread's kernel wait, from its end (or from t inside it): jump to the waker at
        the wake (a guest thread, or the host thread that woke it: the fence thread, the back),
        or file it here: a time-out (timer), a vsync slot (display), OoT's present poll (the
        GPU's write of its word, then the writer)."""
        trace, guest = self.trace, self.trace.guest
        low = max(wait.begin, start)
        label = guest.wait_label(gid[1], wait)
        waker = guest.waker_label(wait)
        pending = wait.end > t + EPS  # the walk arrived inside the wait: its waker acts later

        def edge(state=None):
            self.edges.append((gid[1], label, low, t, waker,
                               state if state is not None else guest.waker_state(wait)))

        poll = guest.poll_of(wait) if wait.reason == 1 else None
        if poll is not None and poll[3]:
            write = None if pending else guest.poll_write(poll, wait.begin, wait.end)
            if write is not None:
                writer = trace.names.get(write[1], str(write[1]))
                edge(f"GPU write on {writer}")
                add(gid, max(write[0], low), t, "guest", "timer",
                    "OoT: present poll sleep (after the GPU's write)")
                if write[0] <= low:
                    return ("stay", low)
                return ("jump", write[1], write[0], low, label)
            later = guest.poll_write(poll, t, t, after=True)
            if later is not None:  # the GPU had not written the word yet: its writer works on
                edge(f"GPU not done yet (written later on {trace.names.get(later[1], later[1])})")
                return ("jump", later[1], t, low, label)
            edge("no GPU write of the word")
            add(gid, low, t, "guest", "unknown", "OoT: present poll sleep (no GPU write)")
            return ("stay", low)
        if pending:
            target = ("guest", wait.by) if wait.by else wait.host
            if target is not None and target != gid and wait.result != 1:
                edge()
                return ("jump", target, t, low, label)
            add(gid, low, t, "guest", "timer" if wait.result == 1 else "unknown",
                f"{label} (until {WAKE_RESULTS.get(wait.result, 'no wake')})")
            return ("stay", low)
        if guest.is_display(wait):
            edge()
            add(gid, low, t, "display", "display", f"{label} (vblank)")
            return ("stay", low)
        if wait.result == 1:
            edge()
            add(gid, low, t, "guest", "timer", f"{label} (timed out)")
            return ("stay", low)
        if wait.by and wait.by != gid[1]:
            edge()
            return ("jump", ("guest", wait.by), t, low, label)
        host = wait.host
        if host is None or wait.by == gid[1]:
            edge()
            add(gid, low, t, "guest", "unknown", f"{label} (waker unknown)")
            return ("stay", low)
        host_name = trace.names.get(host, str(host))
        edge()
        if host_name in TIMER_THREADS:
            add(gid, low, t, "guest", "timer", f"{label} (woken by {host_name})")
            return ("stay", low)
        if host not in trace.spans and trace.role(host) == "other":
            # A service thread without spans: its work for the request is the wait.
            add(host, low, t, "service", "busy", f"{host_name}: work for {label}")
            return ("stay", low)
        return ("jump", host, t, low, label)

    def resolve(self, tid, t, low, span, target, resolver, add):
        """Files the wait's part of [low, t] and returns where the path goes on, or None to go
        on at `low` on this thread."""
        trace = self.trace
        role = trace.role(tid)
        name, args = span[2], span[3] or {}
        how = resolver[0]
        waker = trace.wakers.get((tid, span[1]))
        if waker is not None and low <= waker[1] <= t:  # round 2: an ETW ReadyThread edge
            add(tid, waker[1], t, role, "latency", f"{name} (ready, not running)")
            return waker[0], waker[1]
        if how == "at_end":
            other = trace.tid_of(target)
            if other is None:
                add(tid, low, t, role, "unknown", f"{name} (no {target} thread)")
                return None
            return other, t
        if how in ("marker", "wake_window"):
            floor = low if how == "marker" else max(low, t - self.wake_window)
            marker = self.latest_marker(resolver[1:], floor, t)
            if marker is None and how == "marker" and span[1] > t + EPS and target != "guest":
                # Pending: the walk arrived inside the wait, and what ends it comes after t (the
                # present thread idles a whole frame for the next one). Whoever will end it is
                # what the path is on at t: the thread that records the marker, else the stage.
                other = trace.recorder(resolver[1:]) or trace.tid_of(target)
                if other is not None and other != tid:
                    return other, t
            if marker is None:
                add(tid, low, t, role, "unknown" if low <= span[0] + EPS else "latency",
                    f"{name} (waker unknown)")
                return None
            add(tid, marker[0], t, role, "latency", f"{name} (wake)")
            return marker[1], marker[0]
        if how == "syncpoint":
            entries = trace.syncpoints.get((resolver[1], args.get("syncpoint")), [])
            at = bisect.bisect_left(entries, (args.get("value", args.get("threshold", 0)),))
            if at == len(entries) or entries[at][1] > t:
                add(tid, low, t, role, resolver[2], f"{name} (value not reached)")
                return None
            _value, ts, by = entries[at]
            if ts < low:  # reached before this part of the wait: the wait's own overhead
                add(tid, low, t, role, "busy", f"{name} (already reached)")
                return None
            add(tid, ts, t, role, "latency", f"{name} (wake)")
            return by, ts
        if how in ("gpu_tick", "gpu_any"):
            if how == "gpu_tick":
                submit = trace.submit_by_tick.get(args.get("tick"))
                if submit is not None and submit[0] > t:
                    submit = None
            else:
                marker = self.latest_marker(("vkQueueSubmit",), low, t)
                submit = (marker[0], marker[1]) if marker else None
            if submit is None or submit[0] < low:
                add(tid, low, t, "gpu", "busy", f"host GPU ({name})")
                return None
            add(tid, submit[0], t, "gpu", "busy", f"host GPU ({name})")
            return submit[1], submit[0]
        if how == "poll":
            ends = trace.release_ends
            lo, hi = bisect.bisect_left(ends, span[0]), bisect.bisect_right(ends, t)
            if lo >= hi:
                add(tid, low, t, role, "unknown", f"{name} (no fence release in it)")
                return None
            release = ends[hi - 1] if self.poll_release == "latest" else ends[lo]
            add(tid, max(release, low), t, role, "timer", f"{name} (after the GPU's release)")
            if release < low:
                return None
            return trace.tid_of("fence"), release
        raise ValueError(f"unknown resolver {how}")


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(int(fraction * len(ordered)), len(ordered) - 1)] if ordered else 0.0


def analyse(trace, walker, top=10):
    # The guest's presenting thread may move between host core threads: each walk starts on
    # the thread of the present that ends the frame.
    timed = sorted((s[0], tid) for tid, spans in trace.spans.items() for s in spans
                   if s[2] == FRAME_SPAN)
    if len(timed) < 3:
        sys.exit("fewer than three guest presents in the trace")
    guest = trace.guest
    if guest is not None:
        # The frame thread: the guest thread its core ran at each present.
        timed = [(ts, ("guest", guest.on_core(tid, ts + EPS)) if guest.on_core(tid, ts + EPS)
                  else tid) for ts, tid in timed]
    present_tid = collections.Counter(tid for _ts, tid in timed).most_common(1)[0][0]
    presents = [ts for ts, _tid in timed]
    frames = []
    for number, ((start, _start_tid), (end, end_tid)) in enumerate(zip(timed, timed[1:])):
        segments = walker.walk(end_tid, start, end)
        edges = getattr(walker, "edges", [])
        frame_ms = (end - start) / 1000.0
        path_ms = sum(s[2] - s[1] for s in segments) / 1000.0
        by = collections.defaultdict(float)
        for _tid, begin, until, stage, kind, label in segments:
            by[(stage, kind, label)] += (until - begin) / 1000.0
        unexplained = sum(v for (_s, kind, _l), v in by.items() if kind in ("latency", "unknown"))
        threads = collections.defaultdict(float)  # (thread, kind) -> ms
        for tid, begin, until, _stage, kind, _label in segments:
            threads[(trace.label(tid), kind)] += (until - begin) / 1000.0
        frames.append({"index": number, "start_us": start, "frame_ms": frame_ms, "path_ms": path_ms,
                       "unexplained_ms": unexplained, "by": by, "segments": segments,
                       "threads": threads, "edges": edges, "frame_tid": end_tid})
    durations = [f["frame_ms"] for f in frames]
    median_ms, p90_ms = statistics.median(durations), percentile(durations, 0.9)
    lo, hi = percentile(durations, 0.4), percentile(durations, 0.6)
    bands = {"median band": [f for f in frames if lo <= f["frame_ms"] <= hi],
             "p90 band": [f for f in frames if f["frame_ms"] >= p90_ms]}
    # CP_WINDOWS="8.5-9.4,11.5-12.9" (seconds from the capture start): the second band is the
    # frames starting inside those windows instead of the p90 band (e.g. the frames of a dip).
    if os.environ.get("CP_WINDOWS"):
        spans = [tuple(float(x) for x in w.split("-")) for w in os.environ["CP_WINDOWS"].split(",")]
        bands["p90 band"] = [f for f in frames
                             if any(a <= f["start_us"] / 1e6 <= b for a, b in spans)]
    median_frame = min(frames, key=lambda f: abs(f["frame_ms"] - median_ms))

    def mean_by(band, key):
        totals = collections.defaultdict(float)
        for frame in band:
            for category, ms in frame["by"].items():
                totals[key(category)] += ms
        return {k: v / max(len(band), 1) for k, v in totals.items()}

    stage_kind = {b: mean_by(fs, lambda c: (c[0], c[1])) for b, fs in bands.items()}
    detail = {b: mean_by(fs, lambda c: c) for b, fs in bands.items()}
    threads = {b: mean_of(fs, "threads") for b, fs in bands.items()}
    waited = {b: waited_for(fs) for b, fs in bands.items()}
    longest = sorted(((s[2] - s[1], f["index"], s) for f in frames for s in f["segments"]),
                     key=lambda x: -x[0])[:top]
    unexplained = collections.defaultdict(float)
    examples = collections.defaultdict(list)
    for frame in frames:
        for tid, begin, until, _stage, kind, label in frame["segments"]:
            if kind in ("latency", "unknown"):
                key = (trace.label(tid), label)
                unexplained[key] += (until - begin) / 1000.0 / len(frames)
                if len(examples[key]) < 3:
                    examples[key].append(round(begin / 1000.0, 3))
    pacing = collections.defaultdict(float)  # all pacing spans, on the path or not
    for tid, spans in trace.spans.items():
        for begin, until, name, _args in spans:
            if WAITS.get(name, ("busy",))[0] == "pacing" and presents[0] <= begin < presents[-1]:
                pacing[name] += (until - begin) / 1000.0 / len(frames)
    return {"trace": trace, "present_tid": present_tid, "frames": frames, "median_ms": median_ms,
            "p90_ms": p90_ms, "bands": bands, "median_frame": median_frame,
            "stage_kind": stage_kind, "detail": detail, "longest": longest,
            "unexplained": unexplained, "examples": examples, "pacing": pacing,
            "lists": list_latency(trace), "queue": list_queue(trace, presents),
            "threads": threads, "waited": waited}


def mean_of(band, field):
    """A per-frame {key: ms} field averaged over a band's frames."""
    totals = collections.defaultdict(float)
    for frame in band:
        for key, ms in frame[field].items():
            totals[key] += ms
    return {k: v / max(len(band), 1) for k, v in totals.items()}


def waited_for(band):
    """For the frame thread, the waits on its frame's path by (wait, waker, the waker's state at
    the wake): ms per frame and waits per frame. Only the frame thread's own waits: a waker's
    waits further down the chain show in its state."""
    totals = collections.defaultdict(lambda: [0.0, 0])
    for frame in band:
        frame_tid = frame["frame_tid"]
        if not isinstance(frame_tid, tuple):
            continue
        for waiter, label, begin, until, waker, state in frame["edges"]:
            if waiter != frame_tid[1]:
                continue
            entry = totals[(label, waker, state)]
            entry[0] += (until - begin) / 1000.0
            entry[1] += 1
    n = max(len(band), 1)
    return {k: (ms / n, count / n) for k, (ms, count) in totals.items()}


def list_latency(trace):
    """Per command list (its sequence number): guest submit -> front begin -> back."""
    def by_seq(name):
        return {m[2].get("seq"): m[0] for m in trace.instants.get(name, ())}
    submit, front, back = by_seq("submit list"), by_seq("front: list begin"), by_seq("back: list")
    stages = {"guest submit -> front begin": [front[s] - submit[s] for s in front if s in submit],
              "front begin -> back": [back[s] - front[s] for s in back if s in front]}
    return {k: (len(v), statistics.median(v) / 1000.0 if v else 0.0,
                percentile(v, 0.9) / 1000.0) for k, v in stages.items()}


class Cumulative:
    """Time covered by sorted, non-overlapping intervals before any moment."""

    def __init__(self, intervals):
        self.begins = [b for b, _e in intervals]
        self.ends = [e for _b, e in intervals]
        self.before = [0.0]
        for b, e in intervals:
            self.before.append(self.before[-1] + e - b)

    def at(self, x):
        index = bisect.bisect_right(self.begins, x) - 1
        if index < 0:
            return 0.0
        return self.before[index] + min(x, self.ends[index]) - self.begins[index]

    def within(self, low, high):
        return self.at(high) - self.at(low)


QUEUE_IDLE = {"idle": "idle", "pending sleep": "idle (pending sleep)"}


def front_classes(trace):
    """The front's time as (begin, end, class): drains by site (its waits for the back), its
    waits for the host GPU, idle, and otherwise parsing: a list between its begin and end
    markers ("parsing other lists"), or a suspended list's methods once its semaphore released
    ("resume pending: resumed lists", the rest of that span "resume pending: polling")."""
    front = trace.tid_of("front")
    if front is None:
        return []
    begins = {m[2].get("seq"): m[0] for m in trace.instants.get("front: list begin", ())}
    ends = {m[2].get("seq"): m[0] for m in trace.instants.get("front: list end", ())}
    lists = sorted((begins[s], ends[s]) for s in begins if s in ends and ends[s] >= begins[s])
    spans = trace.spans[front]
    resumes = [(s[0], s[1]) for s in spans if s[2] == "resume pending"]  # never nested in one another
    resume_begins = [r[0] for r in resumes]

    def resuming(t):
        index = bisect.bisect_right(resume_begins, t) - 1
        return index >= 0 and t < resumes[index][1]

    out = []
    li = 0
    for begin, end, index in trace.pieces[front][1]:
        if index == -2 or end <= begin:
            continue
        name = spans[index][2] if index >= 0 else None
        target = WAITS.get(name, ("busy", None))[1] if name is not None else None
        if target == "back":
            out.append((begin, end, f"drain: {name}"))
            continue
        if target == "gpu":
            out.append((begin, end, f"host GPU wait: {name}"))
            continue
        if name in QUEUE_IDLE:
            out.append((begin, end, QUEUE_IDLE[name]))
            continue
        if name == "resume pending":
            out.append((begin, end, "resume pending: polling"))
            continue
        # Split the piece at list boundaries: inside a list it is parsing.
        t = begin
        while li < len(lists) and lists[li][1] <= t:
            li += 1
        k = li
        while t < end:
            if k < len(lists) and lists[k][0] <= t < lists[k][1]:
                until = min(end, lists[k][1])
                out.append((t, until, "parsing other lists"))
            else:
                until = min(end, lists[k][0]) if k < len(lists) else end
                out.append((t, until, "resume pending: resumed lists" if resuming(t)
                            else "front: between lists"))
            t = until
            while k < len(lists) and lists[k][1] <= t:
                k += 1
    return out


def list_queue(trace, presents):
    """Where each command list's guest submit -> front begin went on the front, and how many
    lists were queued (submitted, not begun) when it was submitted; per frame, the queue depth
    when the frame's last list was submitted."""
    submit = {m[2].get("seq"): m[0] for m in trace.instants.get("submit list", ())}
    begin = {m[2].get("seq"): m[0] for m in trace.instants.get("front: list begin", ())}
    if not submit or not begin:
        return None
    by_class = collections.defaultdict(list)
    for b, e, cls in front_classes(trace):
        by_class[cls].append((b, e))
    cumulative = {cls: Cumulative(v) for cls, v in by_class.items()}
    submit_ts = sorted(submit.values())
    begin_ts = sorted(begin[s] for s in begin if s in submit)

    def depth(at):
        # Lists submitted before `at` that the front had not begun by then (FIFO).
        return max(bisect.bisect_left(submit_ts, at) - bisect.bisect_left(begin_ts, at), 0)

    rows = []
    for seq, ts in submit.items():
        if seq not in begin or begin[seq] < ts:
            continue
        window = begin[seq] - ts
        parts = {cls: c.within(ts, begin[seq]) for cls, c in cumulative.items()}
        parts["front: untracked"] = max(window - sum(parts.values()), 0.0)
        rows.append({"seq": seq, "submit": ts, "window": window, "depth": depth(ts),
                     "parts": parts})
    rows.sort(key=lambda r: r["submit"])
    frames = []
    starts = [r["submit"] for r in rows]
    for start, end in zip(presents, presents[1:]):
        index = bisect.bisect_left(starts, end) - 1
        if index >= 0 and rows[index]["submit"] >= start:
            last = rows[index]
            frames.append({"depth": last["depth"], "window": last["window"],
                           "count": index - bisect.bisect_left(starts, start) + 1})
    return {"rows": rows, "frames": frames}


def list_at(trace, tid, t):
    """The command list a front or back segment worked on (its last list marker before t)."""
    name = {"front": "front: list begin", "back": "back: list"}.get(trace.role(tid))
    if name is None:
        return None
    stamps = trace.instant_ts.get(name, [])
    index = bisect.bisect_right(stamps, t) - 1
    return trace.instants[name][index][2].get("seq") if index >= 0 else None


def report(result, top_spans=25):
    trace, frames = result["trace"], result["frames"]
    out = []
    w = out.append
    w(f"Critical path over {len(frames)} frames (guest presents on "
      f"{trace.label(result['present_tid'])}, tid {result['present_tid']}"
      f"{', guest layer on' if trace.guest else ''})")
    w(f"Frame time: median {result['median_ms']:.2f} ms, p90 {result['p90_ms']:.2f} ms, "
      f"mean {statistics.mean(f['frame_ms'] for f in frames):.2f} ms")
    median = result["median_frame"]
    error = (median["path_ms"] - median["frame_ms"]) / median["frame_ms"]
    explained = 1 - median["unexplained_ms"] / median["frame_ms"]
    ok = abs(error) <= 0.05
    w(f"Validation, median frame #{median['index']} ({median['frame_ms']:.2f} ms): path sum "
      f"{median['path_ms']:.2f} ms ({error:+.1%}) {'OK' if ok else 'FAILED, do not trust the tables'}; "
      f"explained (not latency/unknown) {explained:.0%}")
    good = sum(abs(f["path_ms"] - f["frame_ms"]) <= 0.05 * f["frame_ms"] for f in frames)
    w(f"  frames whose path sums to their time within 5%: {good} / {len(frames)}")
    if trace.guest is not None and isinstance(median["frame_tid"], tuple):
        own = [e for e in median["edges"] if e[0] == median["frame_tid"][1]]
        if own:
            longest_wait = max(own, key=lambda e: e[3] - e[2])
            w(f"  median frame's longest wait on its path: {longest_wait[1]}, "
              f"{(longest_wait[3] - longest_wait[2]) / 1000.0:.2f} ms, woken by {longest_wait[4]} "
              f"({longest_wait[5]})")
        else:
            w("  median frame's path has no wait of the frame thread")
    bands = list(result["bands"])
    sizes = ", ".join(f"{b}: {len(result['bands'][b])} frames" for b in bands)
    w("")
    w(f"Per stage and kind, ms per frame ({sizes})")
    w(f"  {'stage':<13}{'kind':<10}{'median':>9}{'p90':>9}{'delta':>9}")
    keys = sorted(set().union(*(result["stage_kind"][b] for b in bands)),
                  key=lambda k: -result["stage_kind"][bands[1]].get(k, 0))
    for key in keys:
        a, b = (result["stage_kind"][band].get(key, 0.0) for band in bands)
        w(f"  {key[0]:<13}{key[1]:<10}{a:9.2f}{b:9.2f}{b - a:+9.2f}")
    w("")
    w(f"Per span on the path (top {top_spans} by the p90 band), ms per frame")
    keys = sorted(set().union(*(result["detail"][b] for b in bands)),
                  key=lambda k: -result["detail"][bands[1]].get(k, 0))[:top_spans]
    for key in keys:
        a, b = (result["detail"][band].get(key, 0.0) for band in bands)
        w(f"  {a:7.2f} {b:7.2f} {b - a:+7.2f}  {key[0]} / {key[1]} / {key[2]}")
    if trace.guest is not None:
        w("")
        w("Per thread on the path by kind, ms per frame (median band, p90 band, delta)")
        keys = sorted(set().union(*(result["threads"][b] for b in bands)),
                      key=lambda k: -result["threads"][bands[1]].get(k, 0))[:top_spans]
        for key in keys:
            a, b = (result["threads"][band].get(key, 0.0) for band in bands)
            w(f"  {a:7.2f} {b:7.2f} {b - a:+7.2f}  {key[0]} / {key[1]}")
        w("")
        w(f"Who the frame thread waited for on the path: ms per frame and waits per frame "
          f"(median band | p90 band); the waker's own state at the wake")
        keys = sorted(set().union(*(result["waited"][b] for b in bands)),
                      key=lambda k: -max(result["waited"][b].get(k, (0.0, 0.0))[0] for b in bands))
        for key in keys[:top_spans]:
            (am, ac), (bm, bc) = (result["waited"][band].get(key, (0.0, 0.0)) for band in bands)
            w(f"  {am:7.2f} {ac:5.1f} | {bm:7.2f} {bc:5.1f}  {key[0]}, woken by {key[1]}: {key[2]}")
    w("")
    w(f"{len(result['longest'])} longest segments")
    for length, frame, (tid, begin, _until, stage, kind, label) in result["longest"]:
        seq = list_at(trace, tid, begin)
        w(f"  {length / 1000.0:7.2f} ms  frame {frame:<5} at {begin / 1000.0:10.3f} ms  "
          f"{trace.label(tid)}: {stage} {kind}: {label}" + (f" (list {seq})" if seq else ""))
    w("")
    w("Unexplained (latency and unknown waits) per thread, ms per frame, first times in ms")
    for (thread, label), ms in sorted(result["unexplained"].items(), key=lambda x: -x[1])[:20]:
        w(f"  {ms:7.3f}  {thread}: {label}  at {result['examples'][(thread, label)]}")
    w("")
    w("Pacing holds in the capture (on the path or not), ms per frame")
    for name, ms in sorted(result["pacing"].items(), key=lambda x: -x[1]):
        w(f"  {ms:7.3f}  {name}")
    w("")
    w("Command lists through the stages (count, median ms, p90 ms)")
    for name, (count, med, p90) in result["lists"].items():
        w(f"  {name:<30}{count:>7}{med:9.3f}{p90:9.3f}")
    queue = result.get("queue")
    if queue and queue["rows"]:
        report_queue(queue, w)
    return "\n".join(out)


def report_queue(queue, w):
    rows = queue["rows"]
    windows = [r["window"] for r in rows]
    cut = percentile(windows, 0.9)
    slow = [r for r in rows if r["window"] >= cut]
    classes = sorted({c for r in rows for c in r["parts"]},
                     key=lambda c: -sum(r["parts"].get(c, 0.0) for r in rows))
    w("")
    w(f"Guest submit -> front begin, what the front did meanwhile ({len(rows)} lists; the "
      f"slowest tenth: {len(slow)} lists, window >= {cut / 1000.0:.2f} ms)")
    w(f"  {'ms per list':<30}{'mean':>8}{'median':>8}{'p90':>8}{'share':>8}"
      f"{'slow mean':>11}{'slow share':>12}")
    total, slow_total = sum(windows), sum(r["window"] for r in slow)
    for cls in ["window"] + classes:
        values = [r["window"] if cls == "window" else r["parts"].get(cls, 0.0) for r in rows]
        slow_values = [r["window"] if cls == "window" else r["parts"].get(cls, 0.0) for r in slow]
        if sum(values) < 1e-9 and cls != "window":
            continue
        w(f"  {cls:<30}{statistics.mean(values) / 1000.0:8.3f}"
          f"{statistics.median(values) / 1000.0:8.3f}{percentile(values, 0.9) / 1000.0:8.3f}"
          f"{sum(values) / max(total, 1e-9):8.1%}"
          f"{statistics.mean(slow_values) / 1000.0:11.3f}{sum(slow_values) / max(slow_total, 1e-9):12.1%}")
    depths = [r["depth"] for r in rows]
    w(f"  queue depth at submit (lists submitted, not begun): median {statistics.median(depths):.0f}, "
      f"p90 {percentile(depths, 0.9):.0f}, max {max(depths)}")
    frames = queue["frames"]
    if frames:
        last_depths = [f["depth"] for f in frames]
        last_windows = [f["window"] / 1000.0 for f in frames]
        counts = [f["count"] for f in frames]
        w(f"  per frame ({len(frames)}): lists submitted median {statistics.median(counts):.0f}; "
          f"queue depth when the frame's last list was submitted median "
          f"{statistics.median(last_depths):.0f}, p90 {percentile(last_depths, 0.9):.0f}, "
          f"max {max(last_depths)}; that list's submit -> front begin median "
          f"{statistics.median(last_windows):.3f} ms, p90 {percentile(last_windows, 0.9):.3f} ms")


def to_json(result):
    def band(values):
        return [{"stage": k[0], "kind": k[1], "label": k[2], "ms": round(v, 4)} if len(k) == 3
                else {"stage": k[0], "kind": k[1], "ms": round(v, 4)} for k, v in values.items()]
    trace = result["trace"]
    return {
        "clock": trace.other,
        "present_tid": trace.label(result["present_tid"]),
        "median_ms": result["median_ms"], "p90_ms": result["p90_ms"],
        "frames": [{k: f[k] for k in ("index", "start_us", "frame_ms", "path_ms", "unexplained_ms")}
                   for f in result["frames"]],
        "bands": {b: {"frames": [f["index"] for f in fs], "stage_kind": band(result["stage_kind"][b]),
                      "detail": band(result["detail"][b])} for b, fs in result["bands"].items()},
        "longest": [{"ms": length / 1000.0, "frame": frame, "thread": trace.label(s[0]),
                     "begin_ms": s[1] / 1000.0, "stage": s[3], "kind": s[4], "label": s[5]}
                    for length, frame, s in result["longest"]],
        "unexplained": [{"thread": t, "label": label, "ms_per_frame": ms}
                        for (t, label), ms in result["unexplained"].items()],
        "pacing": result["pacing"],
        "lists": {k: {"count": c, "median_ms": m, "p90_ms": p} for k, (c, m, p) in result["lists"].items()},
        "queue": None if not result.get("queue") else {
            "lists": [{"seq": r["seq"], "window_ms": r["window"] / 1000.0, "depth": r["depth"],
                       "parts_ms": {c: round(v / 1000.0, 4) for c, v in r["parts"].items() if v}}
                      for r in result["queue"]["rows"]],
            "frames": result["queue"]["frames"]},
        "threads": {b: [{"thread": k[0], "kind": k[1], "ms": round(v, 4)} for k, v in values.items()]
                    for b, values in result["threads"].items()},
        "waited": {b: [{"wait": k[0], "waker": k[1], "waker_state": k[2], "ms": round(ms, 4),
                        "per_frame": round(count, 3)} for k, (ms, count) in values.items()]
                   for b, values in result["waited"].items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("timeline")
    parser.add_argument("--json", help="also write the result as JSON here")
    parser.add_argument("--top", type=int, default=10, help="longest segments to list")
    parser.add_argument("--spans", type=int, default=25, help="span rows to list")
    parser.add_argument("--poll-release", choices=("latest", "earliest"), default="latest",
                        help="which fence release in OoT's last poll sleep ended it (latest: the "
                             "timer latency is a lower bound, earliest: an upper bound)")
    parser.add_argument("--wake-window-us", type=float, default=500.0,
                        help="how long before a guest core's wake a GPU event signal still counts")
    parser.add_argument("--etw", help="tracerpt CSV of an ETW trace of the same run (round 2)")
    args = parser.parse_args()
    trace = load(args.timeline)
    if args.etw:
        merge_etw(trace, args.etw)
    result = analyse(trace, Walker(trace, args.poll_release, args.wake_window_us), args.top)
    print(report(result, args.spans))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as file:
            json.dump(to_json(result), file, indent=1)


if __name__ == "__main__":
    main()
