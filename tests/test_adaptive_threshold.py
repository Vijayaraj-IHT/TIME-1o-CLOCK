"""
Phase-1 Change 5: adaptive background-tracking threshold unit tests.

tau_eff = max(tau_calibrated, bg_mean + k*bg_std), capped at adapt_max.
Upward-only by design: silence can never drag the threshold below the
calibrated operating point (TPR-safe); loud/confusing background raises it
(FA-cutting). Background stats update ONLY on LISTENING + non-triggering
frames, so keyword frames can never self-suppress via the EMA.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.streaming.state_machine import DetectionStateMachine, DetectionState


def drive(sm, scores, start_ms=0.0, step_ms=50.0):
    out = []
    t = start_ms
    for s in scores:
        out.append(sm.process_similarity(float(s), t))
        t += step_ms
    return out


class TestAdaptiveThreshold(unittest.TestCase):
    def test_disabled_is_fixed_tau(self):
        sm = DetectionStateMachine(adaptive=False, smoothing_window=1)
        res = drive(sm, [0.7] * 100)
        self.assertIn("threshold_eff", res[0])
        for r in res:
            self.assertAlmostEqual(r["threshold_eff"], 0.89)

    def test_quiet_background_stays_at_tau(self):
        sm = DetectionStateMachine(smoothing_window=1)
        res = drive(sm, [0.0] * 200)
        for r in res:
            self.assertAlmostEqual(r["threshold_eff"], 0.89, places=3)

    def test_confusing_background_raises_tau_eff(self):
        # Alternating 0.87/0.79: mean 0.83, std 0.04 -> tau_eff -> 0.91.
        # Every sample stays below tau=0.89, so no trigger can pollute the test.
        sm = DetectionStateMachine(smoothing_window=1, adapt_warmup_frames=10)
        bg = [0.87, 0.79] * 150
        res = drive(sm, bg)
        self.assertGreater(res[-1]["threshold_eff"], 0.89)
        # upward-only: never below calibrated tau on any frame
        for r in res:
            self.assertGreaterEqual(r["threshold_eff"], 0.89 - 1e-9)

    def test_ceiling_caps_tau_eff(self):
        # Alternating 0.899/0.79: mean 0.8445, std ~0.0545 -> uncapped ~0.9535
        # -> pinned at adapt_max=0.95. Max sample 0.899 < tau=0.90: no trigger.
        sm = DetectionStateMachine(threshold=0.90, hysteresis=0.05,
                                   smoothing_window=1, adapt_warmup_frames=5,
                                   adapt_max=0.95)
        res = drive(sm, [0.899, 0.79] * 150)
        self.assertAlmostEqual(res[-1]["threshold_eff"], 0.95, places=2)

    def test_warmup_learns_but_does_not_apply(self):
        sm = DetectionStateMachine(threshold=0.90, hysteresis=0.05,
                                   smoothing_window=1, adapt_warmup_frames=40,
                                   adapt_alpha=0.1)
        bg = [0.88, 0.80] * 60  # learned tau_eff -> ~0.92
        res = drive(sm, bg)
        # during warmup: fixed tau despite high background ...
        for r in res[:40]:
            self.assertAlmostEqual(r["threshold_eff"], 0.90)
        # ... after warmup: learned stats apply immediately (no re-learning lag)
        self.assertGreater(res[41]["threshold_eff"], 0.90)

    def test_keyword_frames_do_not_pollute_background(self):
        sm = DetectionStateMachine(threshold=0.89, hysteresis=0.05,
                                   smoothing_window=1, consecutive_windows=2,
                                   adapt_warmup_frames=0)
        drive(sm, [0.1] * 50)  # settle background low
        mean_before = sm._bg_mean
        # trigger: VERIFYING -> ACTIVATED -> COOLDOWN (high scores must not feed EMA)
        r1 = sm.process_similarity(0.95, 5000.0)
        self.assertEqual(r1["state"], DetectionState.VERIFYING)
        r2 = sm.process_similarity(0.95, 5050.0)
        self.assertTrue(r2["is_activated"])
        self.assertAlmostEqual(sm._bg_mean, mean_before, places=9)
        # cooldown frames are frozen too
        sm.process_similarity(0.95, 5100.0)
        self.assertAlmostEqual(sm._bg_mean, mean_before, places=9)

    def test_recovers_when_background_clears(self):
        sm = DetectionStateMachine(smoothing_window=1, adapt_warmup_frames=5,
                                   adapt_alpha=0.05)
        drive(sm, [0.87, 0.79] * 100)
        raised = sm.threshold_eff
        self.assertGreater(raised, 0.89)
        drive(sm, [0.0] * 400, start_ms=20000.0)
        self.assertLess(sm.threshold_eff, raised)
        self.assertAlmostEqual(sm.threshold_eff, 0.89, places=2)

    def test_reset_clears_background(self):
        sm = DetectionStateMachine(smoothing_window=1, adapt_warmup_frames=5)
        drive(sm, [0.87, 0.79] * 60)
        self.assertGreater(sm.threshold_eff, 0.89)
        sm.reset()
        self.assertAlmostEqual(sm.threshold_eff, 0.89)
        self.assertEqual(sm._adapt_frames, 0)

    def test_vetoed_scores_are_safe_background_samples(self):
        # veto feeds -1.0; EMA must absorb it without breaking (stays >= tau)
        sm = DetectionStateMachine(smoothing_window=1, adapt_warmup_frames=5)
        res = drive(sm, [-1.0, 0.1] * 100)
        for r in res:
            self.assertGreaterEqual(r["threshold_eff"], 0.89 - 1e-9)


if __name__ == "__main__":
    unittest.main()
