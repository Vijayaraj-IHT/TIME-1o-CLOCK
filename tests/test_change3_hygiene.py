"""
Phase-1 Change 3: streaming-score & lifecycle hygiene tests.

3a (observed flag): VAD-silence frames run no inference, so they are NOT model
    observations. They still decay the smoothing average (via the appended 0.0)
    but must not teach the Change-5 background model -- otherwise silence
    dilutes tau_eff upward adaptation in exactly the noisy rooms ⑤ targets.
3b (history reset): similarity evidence is perishable. It is dropped on
    activation, on cooldown exit, and on any stream discontinuity (frame gap >
    stale_gap_ms, or a backwards clock jump: paused stream, ASR-handover
    freeze, stalled consumer). Slow background stats + the Change-6 marker are
    deliberately KEPT across gaps (same room; marker has its own freshness).
3c (quant-aware TFLite I/O): Python inference must apply scale/zero-point like
    firmware does (main.cpp), instead of a bare astype. Pass-through exact for
    float32-I/O models (all three current .tflite files, verified Sep-2026).
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.streaming.state_machine import DetectionStateMachine, DetectionState
from src.models.tflite_quant import quantize_input, dequantize_output, io_quant_params


def drive(sm, scores, start_ms=0.0, step_ms=50.0, observed=True):
    out = []
    t = start_ms
    for s in scores:
        out.append(sm.process_similarity(float(s), t, observed=observed))
        t += step_ms
    return out


class TestObservedFlag(unittest.TestCase):
    def test_unobserved_silence_still_decays_smoothing(self):
        sm = DetectionStateMachine(smoothing_window=4, consecutive_windows=4)
        r_hot = drive(sm, [0.9] * 3)  # VERIFYING, no trigger (needs 4)
        self.assertEqual(sm.state, DetectionState.VERIFYING)
        self.assertTrue(all(r["observed"] for r in r_hot))
        r_sil = drive(sm, [0.0] * 4, start_ms=150.0, observed=False)
        self.assertTrue(all(not r["observed"] for r in r_sil))
        self.assertLess(r_sil[-1]["smoothed_similarity"], 0.05)
        self.assertEqual(sm.state, DetectionState.LISTENING)

    def test_unobserved_frames_do_not_teach_background(self):
        sm = DetectionStateMachine(smoothing_window=1, adapt_warmup_frames=5)
        drive(sm, [0.0] * 200, observed=False)
        self.assertEqual(sm._adapt_frames, 0)
        self.assertAlmostEqual(sm.threshold_eff, 0.89)

    def test_observed_zeros_still_learn(self):
        sm = DetectionStateMachine(smoothing_window=1, adapt_warmup_frames=5)
        drive(sm, [0.0] * 200, observed=True)
        self.assertEqual(sm._adapt_frames, 200)

    def test_silence_does_not_dilute_noisy_room_adaptation(self):
        sm = DetectionStateMachine(smoothing_window=1, adapt_warmup_frames=5,
                                   adapt_alpha=0.05)
        drive(sm, [0.87, 0.79] * 100)  # confusing bg: tau rises
        raised = sm._effective_threshold()  # final-stats value (the member
        self.assertGreater(raised, 0.89)    # lags by one frame by design)
        drive(sm, [0.0] * 200, start_ms=10000.0, observed=False)
        self.assertAlmostEqual(sm.threshold_eff, raised, places=6)


class TestHistoryReset(unittest.TestCase):
    def test_activation_clears_history(self):
        sm = DetectionStateMachine(smoothing_window=8, consecutive_windows=2)
        sm.process_similarity(0.95, 0.0)
        r = sm.process_similarity(0.95, 50.0)
        self.assertTrue(r["is_activated"])
        self.assertEqual(sm.state, DetectionState.COOLDOWN)
        self.assertEqual(len(sm.similarity_history), 0)
        self.assertEqual(r["consecutive_count"], 2)  # frame report preserved

    def test_cooldown_exit_clears(self):
        sm = DetectionStateMachine(smoothing_window=4, consecutive_windows=2,
                                   cooldown_ms=200)
        sm.process_similarity(0.95, 0.0)
        r = sm.process_similarity(0.95, 50.0)  # trigger; cooldown until 250
        self.assertTrue(r["is_activated"])
        sm.process_similarity(0.95, 100.0)  # cooldown frame (would linger)
        self.assertEqual(len(sm.similarity_history), 1)
        r2 = sm.process_similarity(0.10, 300.0)  # past cooldown, no ts gap
        self.assertEqual(r2["state"], DetectionState.LISTENING)
        self.assertEqual(list(sm.similarity_history), [0.10])

    def test_gap_resets_stale_verifying(self):
        sm = DetectionStateMachine(smoothing_window=8, consecutive_windows=4)
        sm.process_similarity(0.95, 0.0)  # VERIFYING(1)
        self.assertEqual(sm.state, DetectionState.VERIFYING)
        r = sm.process_similarity(0.50, 2000.0)  # 2s stall: evidence is stale
        self.assertEqual(r["state"], DetectionState.LISTENING)
        self.assertEqual(sm.consecutive_count, 0)
        self.assertEqual(list(sm.similarity_history), [0.50])

    def test_backward_timestamp_resets(self):
        sm = DetectionStateMachine(smoothing_window=4, consecutive_windows=2)
        sm.process_similarity(0.95, 1000.0)
        r = sm.process_similarity(0.10, 500.0)  # clock jumped back
        self.assertEqual(r["state"], DetectionState.LISTENING)
        self.assertEqual(list(sm.similarity_history), [0.10])

    def test_gap_preserves_background_and_marker(self):
        sm = DetectionStateMachine(smoothing_window=1, adapt_warmup_frames=5,
                                   adapt_alpha=0.05)
        drive(sm, [0.87, 0.79] * 100)
        raised = sm._effective_threshold()
        frames = sm._adapt_frames
        sm.mark_keyword_end(50000.0)
        sm.process_similarity(0.10, 100000.0, observed=False)  # huge gap
        self.assertEqual(sm._adapt_frames, frames)
        self.assertAlmostEqual(sm.threshold_eff, raised, places=6)
        self.assertIsNotNone(sm._kw_end_ms)  # kept; freshness judges it

    def test_reset_clears_last_timestamp(self):
        sm = DetectionStateMachine()
        drive(sm, [0.1] * 10)
        sm.reset()
        self.assertIsNone(sm._last_ts)
        r = sm.process_similarity(0.10, 999999.0)
        self.assertEqual(r["state"], DetectionState.LISTENING)
        self.assertEqual(list(sm.similarity_history), [0.10])


class TestTfliteQuantIO(unittest.TestCase):
    def test_int8_matches_firmware_formula(self):
        rng = np.random.RandomState(0)
        x = (rng.randn(2, 98, 13, 1) * 3.0).astype(np.float32)
        scale, zp = 0.0213, -7
        q = quantize_input(x, scale, zp, np.int8)
        self.assertEqual(q.dtype, np.dtype(np.int8))
        # fw main.cpp: q = clip(round(x / scale) + zero_point)
        expect = np.clip(np.round(x / scale) + zp, -128, 127).astype(np.int8)
        np.testing.assert_array_equal(q, expect)
        back = dequantize_output(q, scale, zp)
        np.testing.assert_allclose(back, (expect.astype(np.float32) - zp) * scale,
                                   rtol=1e-6)
        # in-range values roundtrip within half an LSB
        x2 = (rng.randn(1, 98, 13, 1) * 0.5).astype(np.float32)
        back2 = dequantize_output(quantize_input(x2, scale, zp, np.int8),
                                  scale, zp)
        self.assertTrue(np.all(np.abs(back2 - x2) <= scale / 2 + 1e-6))

    def test_uint8_clips_to_range(self):
        x = np.array([-100.0, -0.01, 0.0, 1.0, 100.0], dtype=np.float32)
        q = quantize_input(x, 0.1, 128, np.uint8)
        self.assertEqual(q.dtype, np.dtype(np.uint8))
        np.testing.assert_array_equal(
            q, np.array([0, 128, 128, 138, 255], dtype=np.uint8))
        d = dequantize_output(q, 0.1, 128)
        np.testing.assert_allclose(
            d, (np.array([0, 128, 128, 138, 255], dtype=np.float32) - 128) * 0.1,
            rtol=1e-6)

    def test_float_passthrough_is_exact(self):
        rng = np.random.RandomState(1)
        x = (rng.randn(1, 98, 13, 1) * 5).astype(np.float32)
        q = quantize_input(x, 0.0, 0, np.float32)
        self.assertEqual(q.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(q, x)  # bit-exact
        d = dequantize_output(x, 0.0, 0)
        np.testing.assert_array_equal(d, x)

    def test_io_quant_params_split(self):
        d = {"index": 3, "dtype": np.dtype(np.int8), "quantization": (0.5, 4)}
        self.assertEqual(io_quant_params(d), (3, np.dtype(np.int8), 0.5, 4))
        d2 = {"index": 0, "dtype": np.float32}  # missing key -> float
        self.assertEqual(io_quant_params(d2), (0, np.float32, 0.0, 0))


class TestSilenceFrameTrigger(unittest.TestCase):
    """Change 3d: persistence can complete ON a silence (unobserved) frame --
    the decay edge still above tau_low after hot frames. Found on real zora
    audio (2nd keyword triggered at silence onset). Such triggers must
    propagate, not vanish as VAD_SKIP."""

    def test_sm_triggers_on_unobserved_frame(self):
        sm = DetectionStateMachine(smoothing_window=8, consecutive_windows=8)
        for i in range(7):  # VERIFYING counts 1..7 (smoothed 0.99)
            r = sm.process_similarity(0.99, i * 50.0)
            self.assertFalse(r["is_activated"])
        self.assertEqual(sm.state, DetectionState.VERIFYING)
        # decay edge: (7x0.99 + 0.0)/8 = 0.866 >= tau_low -> count 8 -> fire
        r = sm.process_similarity(0.0, 350.0, observed=False)
        self.assertTrue(r["is_activated"])
        self.assertEqual(sm.state, DetectionState.COOLDOWN)

    def test_detector_surfaces_decay_edge_trigger(self):
        from src.streaming.detector import StreamingVoiceActivator

        proto = np.zeros(32, dtype=np.float32)
        proto[0] = 1.0
        hi = proto.copy()  # cosine 1.0 vs proto
        lo = np.zeros(32, dtype=np.float32)
        lo[1] = 1.0  # cosine 0.0 vs proto

        class _CountEncoder:
            """First 7 inferences hot (keyword), then cold (silence)."""
            def __init__(self):
                self.n = 0
            def __call__(self, x, training=False):
                self.n += 1
                e = hi if self.n <= 7 else lo
                class _Out:
                    def numpy(_s):
                        return np.array([e])
                return _Out()

        sm = DetectionStateMachine(smoothing_window=8, consecutive_windows=8)
        det = StreamingVoiceActivator(encoder=_CountEncoder(),
                                      target_prototype=proto, state_machine=sm)
        tone = (0.3 * np.sin(2 * np.pi * 440 *
                np.linspace(0, 0.05, 800))).astype(np.float32)
        sils = np.zeros(800, dtype=np.float32)
        acts = []
        for i in range(14):
            chunk = tone if i < 7 else sils
            r = det.process_chunk(chunk, i * 50.0)
            if r["is_activated"]:
                acts.append(r)
        self.assertTrue(acts, "decay-edge trigger must fire")
        # the trigger lands in the silent region (chunk 7+, ts >= 350ms) on
        # the cold embedding (raw 0.0), via inference (VERIFYING bypass --
        # VAD hangover may still report speech, so assert on content, not VAD):
        self.assertGreaterEqual(acts[0]["timestamp_ms"], 350.0)
        self.assertAlmostEqual(acts[0]["raw_similarity"], 0.0)
        self.assertFalse(acts[0]["inference_skipped"])
        # skip path itself never fires (0.0 from LISTENING can't VERIFY) --
        # documents the invariant; detector still reports sm honestly (3d).
        self.assertFalse(any(a["inference_skipped"] for a in acts))


if __name__ == "__main__":
    unittest.main()
