# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import os
import tempfile
import unittest

import critical_path

GUEST, FRONT, BACK, WORKER, FENCE = 10, 20, 30, 40, 50
NAMES = {GUEST: "CPUCore_0", FRONT: "GPU", BACK: "GPUBack", WORKER: "VulkanWorker",
         FENCE: "GPUFencingThread"}


def span(tid, name, begin, end, **args):
    event = {"name": name, "ph": "X", "pid": 1, "tid": tid, "ts": begin, "dur": end - begin}
    if args:
        event["args"] = args
    return event


def mark(tid, name, ts, **args):
    event = {"name": name, "ph": "i", "pid": 1, "tid": tid, "ts": ts, "s": "t"}
    if args:
        event["args"] = args
    return event


def events():
    out = [{"name": "thread_name", "ph": "M", "pid": 1, "tid": tid, "args": {"name": name}}
           for tid, name in NAMES.items()]
    for k, at in enumerate((0, 20000, 40000, 60000)):
        out.append(span(GUEST, "guest present", at, at + 100, frame=k + 1))
    # Frame 1 (20000 -> 40000): guest submits, the front parses, the back draws, the worker
    # submits, the host GPU runs, the fence thread releases, OoT's poll sleep sees it.
    out += [
        mark(GUEST, "submit list", 22400, seq=7, channel=1, entries=3),
        span(FRONT, "idle", 20100, 22500),
        mark(FRONT, "front: list begin", 22500, seq=7),
        span(FRONT, "push", 22500, 27000),
        span(BACK, "split wait", 24000, 26000, phase=2),
        mark(BACK, "back: list", 26000, seq=7),
        span(BACK, "draw", 26000, 35500),
        span(WORKER, "worker: wait for work", 30000, 35500),
        span(WORKER, "record commands + submit", 35500, 36010),
        mark(WORKER, "vkQueueSubmit", 36000, tick=99),
        span(FENCE, "fence: wait for fence", 35000, 38300),
        span(FENCE, "fence: release", 38300, 38500, fences=1),
        span(GUEST, "OoT: present poll sleep", 38000, 40000),
        span(GUEST, "guest core idle", 38200, 39000),  # inside the poll sleep: still the sleep
    ]
    # Frame 2 (40000 -> 60000): a drain that ends as the back starts to wait (a hand-off), and
    # the guest's wait for its frame fence, ended by the back's syncpoint increment.
    out += [
        span(FRONT, "idle", 40000, 42000),
        mark(GUEST, "submit list", 41900, seq=8, channel=1, entries=3),
        span(FRONT, "push", 42000, 49990),
        span(FRONT, "list sync", 49990, 52000, behind=40),
        span(BACK, "split wait", 45000, 50000, phase=2),
        span(BACK, "draw", 50000, 59000),
        mark(BACK, "syncpoint: guest increment", 59000, id=5, value=7),
        span(GUEST, "present: wait for frame fence", 55000, 60000, syncpoint=5, value=7),
    ]
    return out


def write_trace(path, trace_events):
    with open(path, "w", encoding="utf-8") as file:
        file.write('{"displayTimeUnit":"ms","otherData":{"clock":"steady_clock","start_ns":5,'
                   '"qpc_start":1000,"qpc_frequency":10000000,"process_id":1,"seconds":1.000},'
                   '"traceEvents":[\n')
        file.write(",\n".join(json.dumps(e, separators=(",", ":")) for e in trace_events))
        file.write("\n]}\n")


class CriticalPathTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.mkdtemp()
        self.path = os.path.join(directory, "timeline.json")
        write_trace(self.path, events())
        self.trace = critical_path.load(self.path)
        self.result = critical_path.analyse(self.trace, critical_path.Walker(self.trace))

    def by(self, frame):
        return self.result["frames"][frame]["by"]

    def test_clock(self):
        self.assertEqual(self.trace.other["qpc_frequency"], 10000000)
        self.assertAlmostEqual(self.trace.qpc_to_us(1000 + 10000000), 1e6)

    def test_paths_add_up(self):
        for frame in self.result["frames"]:
            self.assertAlmostEqual(frame["path_ms"], frame["frame_ms"], places=6)

    def test_chain_through_the_stages(self):
        by = self.by(1)
        self.assertAlmostEqual(by[("guest", "timer", "OoT: present poll sleep (after the GPU's release)")], 1.5)
        self.assertAlmostEqual(by[("fence", "busy", "fence: release")], 0.2)
        self.assertAlmostEqual(by[("gpu", "busy", "host GPU (fence: wait for fence)")], 2.3)
        self.assertAlmostEqual(by[("worker", "busy", "record commands + submit")], 0.5)
        self.assertAlmostEqual(by[("back", "busy", "draw")], 9.5)
        self.assertAlmostEqual(by[("front", "busy", "push")], 3.5)
        self.assertAlmostEqual(by[("front", "latency", "idle (wake)")], 0.1)
        self.assertAlmostEqual(by[("guest", "busy", "guest: running")], 2.3)
        self.assertNotIn(("guest", "wait", "guest core idle"), by)

    def test_hand_off_and_syncpoint(self):
        by = self.by(2)
        self.assertAlmostEqual(by[("guest", "latency", "present: wait for frame fence (wake)")], 1.0)
        self.assertAlmostEqual(by[("back", "busy", "draw")], 9.0)
        self.assertAlmostEqual(by[("front", "latency", "list sync (hand-off)")], 0.01)
        self.assertAlmostEqual(by[("front", "busy", "push")], 7.99)

    def test_list_latency(self):
        count, median, _p90 = self.result["lists"]["guest submit -> front begin"]
        self.assertEqual(count, 1)
        self.assertAlmostEqual(median, 0.1)

    def test_list_queue(self):
        rows = self.result["queue"]["rows"]
        self.assertEqual([r["seq"] for r in rows], [7])
        self.assertEqual(rows[0]["depth"], 0)
        self.assertAlmostEqual(rows[0]["parts"]["idle"], 100.0)
        frames = self.result["queue"]["frames"]
        self.assertEqual(frames[0]["count"], 1)

    def test_report_runs(self):
        text = critical_path.report(self.result)
        self.assertIn("Validation", text)
        self.assertIn("queue depth at submit", text)
        json.dumps(critical_path.to_json(self.result))


PRESENT = 60
PRESENT_NAMES = {**NAMES, PRESENT: "VulkanPresent"}


class PendingWaitTest(unittest.TestCase):
    """The walk reaches the present thread inside its wait for the next frame: the path is on
    the thread that will queue the frame, not on the waiting present thread."""

    def test_present_wait_goes_to_the_queuing_thread(self):
        trace_events = [{"name": "thread_name", "ph": "M", "pid": 1, "tid": tid, "args": {"name": name}}
                        for tid, name in PRESENT_NAMES.items()]
        for at in (0, 10000, 20000):
            trace_events.append(span(GUEST, "guest present", at, at + 50))
        trace_events += [
            span(BACK, "present: wait for free frame", 10000, 12000),
            span(BACK, "draw", 12000, 19000),
            mark(BACK, "syncpoint: guest increment", 19000, id=5, value=1),
            span(GUEST, "present: wait for frame fence", 15000, 20000, syncpoint=5, value=1),
            span(PRESENT, "present: wait for frame", 5000, 25000),
            span(WORKER, "worker: wait for work", 8000, 13000),
            mark(WORKER, "present: frame queued", 24000, frame=1),
        ]
        path = os.path.join(tempfile.mkdtemp(), "timeline.json")
        write_trace(path, trace_events)
        trace = critical_path.load(path)
        result = critical_path.analyse(trace, critical_path.Walker(trace))
        by = result["frames"][1]["by"]
        self.assertFalse([k for k in by if k[0] == "present"], by)
        self.assertAlmostEqual(by[("back", "busy", "draw")], 7.0)
        self.assertAlmostEqual(result["frames"][1]["path_ms"], 10.0)


if __name__ == "__main__":
    unittest.main()
