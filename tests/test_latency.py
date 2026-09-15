"""
Phase-1 Change 6: keyword-END -> trigger latency telemetry tests.

Covers the state-machine marker math (signed latency, stale/absent markers,
one-marker-one-trigger), the LatencyStats aggregator (percentiles, budget
verdict), and the end-to-end detector path with a scripted fake encoder.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.streaming.state_machine import DetectionStateMachine
from src.streaming.detector import StreamingVoiceActivator
from src.evaluation.latency import LatencyStats, LATENCY_BUDGET_MS


def drive_until_trigger(sm, score=0.95, start_ms=0.0, step_ms=50.0, limit=50):
    t = start_ms
    for _ in range(limit):
        r = sm.process_similarity(score, t)
        if r["is_activated"]:
            return r
        t += step_ms
    raise AssertionError("no trigger within limit")


class TestTriggerLatency(unittest.TestCase):
    def test_latency_is_trigger_minus_marker(self):
        sm = DetectionStateMachine(smoothing_window=1, consecutive_windows=2)
        sm.mark_keyword_end(1000.0)
        r = drive_until_trigger(sm, start_ms=0.0)
        self.assertAlmostEqual(r["trigger_latency_ms"], r["timestamp_ms"] - 1000.0)

    def test_no_marker_gives_none(self):
        sm = DetectionStateMachine(smoothing_window=1, consecutive_windows=2)
        r = drive_until_trigger(sm)
        self.assertIsNone(r["trigger_latency_ms"])

    def test_stale_marker_gives_none(self):
        sm = DetectionStateMachine(smoothing_window=1, consecutive_windows=2)
        sm.mark_keyword_end(1000.0)
        r = drive_until_trigger(sm, start_ms=20000.0)  # 19s later: stale
        self.assertIsNone(r["trigger_latency_ms"])

    def test_early_trigger_reported_signed_negative(self):
        sm = DetectionStateMachine(smoothing_window=1, consecutive_windows=2)
        sm.mark_keyword_end(5000.0)  # keyword "ends" in the future
        r = drive_until_trigger(sm, start_ms=0.0)
        self.assertLess(r["trigger_latency_ms"], 0.0)
        self.assertAlmostEqual(r["trigger_latency_ms"], r["timestamp_ms"] - 5000.0)

    def test_one_marker_one_trigger(self):
        sm = DetectionStateMachine(smoothing_window=1, consecutive_windows=2,
                                   cooldown_ms=200)
        sm.mark_keyword_end(1000.0)
        r1 = drive_until_trigger(sm, start_ms=0.0)
        self.assertIsNotNone(r1["trigger_latency_ms"])
        # second trigger without a fresh marker -> None (marker consumed)
        r2 = drive_until_trigger(sm, start_ms=5000.0)
        self.assertIsNone(r2["trigger_latency_ms"])

    def test_reset_clears_marker(self):
        sm = DetectionStateMachine(smoothing_window=1, consecutive_windows=2)
        sm.mark_keyword_end(1000.0)
        sm.reset()
        r = drive_until_trigger(sm)
        self.assertIsNone(r["trigger_latency_ms"])


class TestLatencyStats(unittest.TestCase):
    def test_empty_is_no_data(self):
        s = LatencyStats().summary()
        self.assertEqual(s["n"], 0)
        self.assertEqual(s["verdict"], "NO_DATA")
        self.assertIsNone(s["max_within_budget"])

    def test_percentiles_and_counts(self):
        st = LatencyStats()
        for v in [-100.0, 100.0, 200.0, 300.0, 400.0]:
            st.add(v)
        st.add(None)  # unmarked activations are ignored, not zero
        s = st.summary()
        self.assertEqual(s["n"], 5)
        self.assertEqual(s["early_triggers"], 1)
        self.assertEqual(s["late_triggers"], 4)
        self.assertAlmostEqual(s["p50_ms"], 200.0)
        self.assertAlmostEqual(s["mean_ms"], 180.0)
        self.assertEqual(s["budget_ms"], LATENCY_BUDGET_MS)

    def test_budget_verdict(self):
        ok = LatencyStats(budget_ms=1000.0)
        for v in [100.0, 900.0]:
            ok.add(v)
        self.assertEqual(ok.summary()["verdict"], "PASS")
        bad = LatencyStats(budget_ms=1000.0)
        for v in [100.0, 1500.0]:
            bad.add(v)
        b = bad.summary()
        self.assertEqual(b["verdict"], "FAIL")
        self.assertFalse(b["max_within_budget"])


class _FakeEncoder:
    """Scripted encoder: returns a fixed embedding regardless of input."""

    def __init__(self, emb):
        self.emb = np.asarray(emb, dtype=np.float32)

    def __call__(self, x, training=False):
        emb = self.emb

        class _Out:
            def numpy(self):
                return np.array([emb])

        return _Out()


class TestDetectorLatencyPath(unittest.TestCase):
    def test_end_to_end_marked_trigger(self):
        rng = np.random.RandomState(7)
        proto = rng.randn(32).astype(np.float32)
        proto /= np.linalg.norm(proto)
        det = StreamingVoiceActivator(encoder=_FakeEncoder(proto),
                                      target_prototype=proto)
        det.mark_keyword_end(1000.0)
        chunk = (np.sin(2 * np.pi * 440 * np.linspace(0, 0.05, 800))
                 .astype(np.float32) * 0.3)
        seen = []
        for i in range(40):
            r = det.process_chunk(chunk, i * 50.0)
            if r["is_activated"]:
                seen.append(r)
        self.assertTrue(seen, "fake-encoder stream should trigger")
        # first trigger consumes the marker; later ones (post-cooldown) are unmarked
        self.assertIsNotNone(seen[0]["trigger_latency_ms"])
        self.assertAlmostEqual(seen[0]["trigger_latency_ms"],
                               seen[0]["timestamp_ms"] - 1000.0, places=1)


if __name__ == "__main__":
    unittest.main()
