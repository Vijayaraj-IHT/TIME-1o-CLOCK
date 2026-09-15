"""
Phase-1 Changes 3+4+5: operating-point sync guard.
configs/config.yaml is the single source of truth; Python defaults and BOTH
firmware flavors must match it, or calibration and deployed behavior drift.
"""
import os
import re
import sys
import unittest

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.streaming.state_machine import DetectionStateMachine

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CFG = os.path.join(REPO, "configs", "config.yaml")


def load_detection():
    with open(CFG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["detection"]


class TestOperatingPointSync(unittest.TestCase):
    def test_config_values_in_sih_range(self):
        d = load_detection()
        self.assertGreaterEqual(d["threshold"], 0.85)
        self.assertLessEqual(d["threshold"], 0.93)
        self.assertGreaterEqual(d["consecutive_windows"], 3)
        self.assertLessEqual(d["consecutive_windows"], 5)
        self.assertGreaterEqual(d["smoothing_window"], 5)
        self.assertLessEqual(d["smoothing_window"], 10)
        self.assertIn("garbage_margin", d)
        # Change 5: adaptive-threshold params present and sane
        self.assertIsInstance(d["adaptive_threshold"], bool)
        self.assertGreater(d["adapt_alpha"], 0.0)
        self.assertLessEqual(d["adapt_alpha"], 0.1)
        self.assertGreaterEqual(d["adapt_k"], 1.0)
        self.assertLessEqual(d["adapt_k"], 4.0)
        self.assertGreater(d["adapt_max"], d["threshold"])
        self.assertLessEqual(d["adapt_max"], 0.99)
        self.assertGreaterEqual(d["adapt_warmup_frames"], 0)
        self.assertLessEqual(d["adapt_warmup_frames"], 200)
        # Change 3b: discontinuity gap present and sane
        self.assertIn("stale_gap_ms", d)
        self.assertGreaterEqual(d["stale_gap_ms"], 100)
        self.assertLessEqual(d["stale_gap_ms"], 5000)

    def test_python_defaults_match_config(self):
        d = load_detection()
        sm = DetectionStateMachine()
        self.assertAlmostEqual(sm.threshold_high, float(d["threshold"]))
        self.assertAlmostEqual(sm.threshold_low, float(d["threshold"] - d["hysteresis"]))
        self.assertEqual(sm.consecutive_windows, int(d["consecutive_windows"]))
        self.assertEqual(sm.smoothing_window_len, int(d["smoothing_window"]))
        self.assertAlmostEqual(sm.cooldown_ms, float(d["cooldown_ms"]))
        self.assertEqual(sm.adaptive, bool(d["adaptive_threshold"]))
        self.assertAlmostEqual(sm.adapt_alpha, float(d["adapt_alpha"]))
        self.assertAlmostEqual(sm.adapt_k, float(d["adapt_k"]))
        self.assertAlmostEqual(sm.adapt_max, float(d["adapt_max"]))
        self.assertEqual(sm.adapt_warmup_frames, int(d["adapt_warmup_frames"]))
        self.assertAlmostEqual(sm.stale_gap_ms, float(d["stale_gap_ms"]))

    def test_firmware_defaults_match_config(self):
        d = load_detection()
        adapt_bool = "true" if d["adaptive_threshold"] else "false"
        for flavor in ("esp32", "esp32_wroom"):
            hdr = os.path.join(REPO, "src", "deployment", flavor, "activator_state_machine.h")
            text = open(hdr, encoding="utf-8").read()
            self.assertIn(f"tau_high = {d['threshold']:.2f}f", text, flavor)
            self.assertIn(f"tau_low = {d['threshold'] - d['hysteresis']:.2f}f", text, flavor)
            self.assertIn(f"persistence_count = {d['consecutive_windows']},", text, flavor)
            self.assertIn(f"smoothing_window = {d['smoothing_window']},", text, flavor)
            self.assertIn(f"cooldown_ms = {int(d['cooldown_ms'])}", text, flavor)
            self.assertIn(f"adaptive = {adapt_bool}", text, flavor)
            self.assertIn(f"adapt_alpha = {d['adapt_alpha']:.2f}f", text, flavor)
            self.assertIn(f"adapt_k = {d['adapt_k']:.1f}f", text, flavor)
            self.assertIn(f"adapt_max = {d['adapt_max']:.2f}f", text, flavor)
            self.assertIn(f"adapt_warmup_frames = {d['adapt_warmup_frames']}", text, flavor)
            self.assertIn(f"stale_gap_ms = {int(d['stale_gap_ms'])}", text, flavor)

    def test_entry_points_match_config(self):
        d = load_detection()
        main_cpp = open(os.path.join(REPO, "src", "deployment", "esp32", "main.cpp"),
                        encoding="utf-8").read()
        self.assertIn(f"TAU_HIGH = {d['threshold']:.2f}f", main_cpp)
        self.assertIn(f"PERSISTENCE_COUNT = {d['consecutive_windows']};", main_cpp)
        self.assertIn(f"SMOOTHING_WINDOW = {d['smoothing_window']};", main_cpp)
        ino = open(os.path.join(REPO, "src", "deployment", "esp32_wroom",
                                "voice_activator_esp32_wroom.ino"), encoding="utf-8").read()
        m = re.search(r"g_state_machine\(([\d.]+)f, ([\d.]+)f, (\d+), (\d+), (\d+)\)", ino)
        self.assertIsNotNone(m, ".ino state machine construction not found")
        self.assertAlmostEqual(float(m.group(1)), float(d["threshold"]), places=2)
        self.assertAlmostEqual(float(m.group(2)), float(d["threshold"] - d["hysteresis"]), places=2)
        self.assertEqual(int(m.group(3)), int(d["consecutive_windows"]))
        self.assertEqual(int(m.group(4)), int(d["smoothing_window"]))
        self.assertEqual(int(m.group(5)), int(d["cooldown_ms"]))


if __name__ == "__main__":
    unittest.main()
