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

Round 2 merges an ETW trace (CSwitch, ReadyThread) through Trace.wakers and the clock in the
trace's otherData (qpc_start, qpc_frequency); see merge_etw.
"""

import argparse
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
    "macro dma dispatch", "fence writers", "qmd read",
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
        self._roles = {}
        self._tids = {}

    def role(self, tid):
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
        trace.names[tid] = event["args"]["name"]
    elif phase == "X":
        begin = event["ts"]
        trace.spans[tid].append((begin, begin + event["dur"], event["name"], event.get("args")))
    elif phase == "i":
        trace.instants[event["name"]].append((event["ts"], tid, event.get("args") or {}))


def finish(trace):
    for tid, spans in trace.spans.items():
        spans.sort(key=lambda s: (s[0], -s[1]))
        trace.pieces[tid] = flatten(spans)
    for markers in trace.instants.values():
        markers.sort(key=lambda m: m[0])
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

        while t > start + EPS:
            steps += 1
            if steps > 2_000_000:
                break
            piece_begin, _piece_end, index = self.piece_at(tid, t)
            low = max(piece_begin, start)
            role = trace.role(tid)
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
            if kind != "wait" and resolver is None:
                add(tid, low, t, "display" if kind == "display" else role, kind, name)
                t = low
                continue
            jump = self.resolve(tid, t, low, span, target, resolver, add)
            if jump is None:  # filed on this thread
                t = low
                continue
            next_tid, next_t = jump
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
    present_tid = collections.Counter(tid for _ts, tid in timed).most_common(1)[0][0]
    presents = [ts for ts, _tid in timed]
    frames = []
    for number, ((start, _start_tid), (end, end_tid)) in enumerate(zip(timed, timed[1:])):
        segments = walker.walk(end_tid, start, end)
        frame_ms = (end - start) / 1000.0
        path_ms = sum(s[2] - s[1] for s in segments) / 1000.0
        by = collections.defaultdict(float)
        for _tid, begin, until, stage, kind, label in segments:
            by[(stage, kind, label)] += (until - begin) / 1000.0
        unexplained = sum(v for (_s, kind, _l), v in by.items() if kind in ("latency", "unknown"))
        frames.append({"index": number, "start_us": start, "frame_ms": frame_ms, "path_ms": path_ms,
                       "unexplained_ms": unexplained, "by": by, "segments": segments})
    durations = [f["frame_ms"] for f in frames]
    median_ms, p90_ms = statistics.median(durations), percentile(durations, 0.9)
    lo, hi = percentile(durations, 0.4), percentile(durations, 0.6)
    bands = {"median band": [f for f in frames if lo <= f["frame_ms"] <= hi],
             "p90 band": [f for f in frames if f["frame_ms"] >= p90_ms]}
    median_frame = min(frames, key=lambda f: abs(f["frame_ms"] - median_ms))

    def mean_by(band, key):
        totals = collections.defaultdict(float)
        for frame in band:
            for category, ms in frame["by"].items():
                totals[key(category)] += ms
        return {k: v / max(len(band), 1) for k, v in totals.items()}

    stage_kind = {b: mean_by(fs, lambda c: (c[0], c[1])) for b, fs in bands.items()}
    detail = {b: mean_by(fs, lambda c: c) for b, fs in bands.items()}
    longest = sorted(((s[2] - s[1], f["index"], s) for f in frames for s in f["segments"]),
                     key=lambda x: -x[0])[:top]
    unexplained = collections.defaultdict(float)
    examples = collections.defaultdict(list)
    for frame in frames:
        for tid, begin, until, _stage, kind, label in frame["segments"]:
            if kind in ("latency", "unknown"):
                key = (trace.names.get(tid, str(tid)), label)
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
            "lists": list_latency(trace), "queue": list_queue(trace, presents)}


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
      f"{trace.names.get(result['present_tid'], '?')}, tid {result['present_tid']})")
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
    w("")
    w(f"{len(result['longest'])} longest segments")
    for length, frame, (tid, begin, _until, stage, kind, label) in result["longest"]:
        seq = list_at(trace, tid, begin)
        w(f"  {length / 1000.0:7.2f} ms  frame {frame:<5} at {begin / 1000.0:10.3f} ms  "
          f"{trace.names.get(tid, tid)}: {stage} {kind}: {label}" + (f" (list {seq})" if seq else ""))
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
        "present_tid": result["present_tid"],
        "median_ms": result["median_ms"], "p90_ms": result["p90_ms"],
        "frames": [{k: f[k] for k in ("index", "start_us", "frame_ms", "path_ms", "unexplained_ms")}
                   for f in result["frames"]],
        "bands": {b: {"frames": [f["index"] for f in fs], "stage_kind": band(result["stage_kind"][b]),
                      "detail": band(result["detail"][b])} for b, fs in result["bands"].items()},
        "longest": [{"ms": length / 1000.0, "frame": frame, "thread": trace.names.get(s[0], s[0]),
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
