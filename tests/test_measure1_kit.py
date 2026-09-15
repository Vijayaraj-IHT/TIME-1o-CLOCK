"""Tests for the MEASURE-1 kit (manifest + QC + recorder helper)."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.measure1.manifest import build_manifest
from tools.measure1.qc_audio import check_file
from tools.measure1.record import _next_var


def _write(path, audio, sr=16000):
    import soundfile as sf
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, np.asarray(audio, dtype=np.float32), sr, subtype="PCM_16")


def _tone_burst(seconds=1.0, freq=300.0, peak=0.5, edge_silence=0.2):
    sr = 16000
    n = int(seconds * sr)
    t = np.arange(n) / sr
    x = peak * np.sin(2 * np.pi * freq * t).astype(np.float64)
    m = int(edge_silence * sr)
    x[:m] = 0.0
    x[-m:] = 0.0
    return x


class TestManifest(unittest.TestCase):
    def test_inventory_counts_and_roles(self):
        with tempfile.TemporaryDirectory() as td:
            kwd = os.path.join(td, "custom_keywords", "zora")
            _write(os.path.join(kwd, "zora_priya_rate+0_var0.wav"), _tone_burst())
            _write(os.path.join(kwd, "zora_priya_rate+0_var1.wav"), _tone_burst())
            _write(os.path.join(td, "noise", "room_fan_01.wav"),
                   np.random.RandomState(0).randn(16000 * 3) * 0.02)
            _write(os.path.join(td, "negatives", "cough_001.wav"), _tone_burst())
            m = build_manifest(td)
            self.assertEqual(m["errors"], [])
            self.assertEqual(m["keywords"]["zora"]["n_clips"], 2)
            self.assertEqual(m["noise"]["n_clips"], 1)
            self.assertEqual(m["negatives"]["n_clips"], 1)
            # below-minimum warnings fire (2 < 10 clips, 3s < 60s noise)
            self.assertTrue(any("zora" in w for w in m["warnings"]))
            self.assertTrue(any("noise total" in w for w in m["warnings"]))

    def test_bad_name_and_stray_warn(self):
        with tempfile.TemporaryDirectory() as td:
            kwd = os.path.join(td, "custom_keywords", "zora")
            _write(os.path.join(kwd, "recording1.wav"), _tone_burst())
            _write(os.path.join(kwd, "sora_priya_rate+0_var0.wav"), _tone_burst())
            _write(os.path.join(td, "random.wav"), _tone_burst())
            m = build_manifest(td)
            self.assertTrue(any("recording1" in w for w in m["warnings"]))
            self.assertTrue(any("prefix" in w for w in m["warnings"]))
            self.assertTrue(any("outside the protocol" in w for w in m["warnings"]))

    def test_empty_root_errors(self):
        with tempfile.TemporaryDirectory() as td:
            m = build_manifest(td)
            self.assertTrue(any("no keyword clips" in e for e in m["errors"]))


class TestQC(unittest.TestCase):
    def _check(self, audio, role="keyword", sr=16000):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "c.wav")
            _write(p, audio, sr)
            return check_file(p, role)

    def test_clean_clip_passes(self):
        r = self._check(_tone_burst())
        self.assertEqual(r["errors"], [])
        self.assertEqual(r["warnings"], [])

    def test_wrong_sr_and_stereo_error(self):
        r = self._check(_tone_burst(), sr=8000)
        self.assertTrue(any("sample_rate" in e for e in r["errors"]))
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "s.wav")
            x = _tone_burst()
            _write(p, np.stack([x, x], axis=1))
            r = check_file(p, "keyword")
        self.assertTrue(any("channels" in e for e in r["errors"]))

    def test_silent_and_quiet_error(self):
        r = self._check(np.zeros(16000))
        self.assertTrue(any("too quiet" in e for e in r["errors"]))
        self.assertTrue(any("silent" in e for e in r["errors"]))

    def test_clipping_errors(self):
        r = self._check(np.ones(16000, dtype=np.float32))
        self.assertTrue(any("clipping" in e for e in r["errors"]))

    def test_cut_word_warns(self):
        t = np.arange(16000) / 16000.0  # tone wall-to-wall, no edge silence
        r = self._check(0.5 * np.sin(2 * np.pi * 300 * t))
        self.assertTrue(any("edge" in w for w in r["warnings"]))

    def test_short_background_warns(self):
        rng = np.random.RandomState(2)
        r = self._check(rng.randn(16000 * 2) * 0.02, role="background")
        self.assertEqual(r["errors"], [])
        self.assertTrue(any("short" in w for w in r["warnings"]))


class TestRecordHelper(unittest.TestCase):
    def test_next_var_skips_taken(self):
        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, "zora_priya_rate+0_var0.wav"), "w").close()
            open(os.path.join(td, "zora_priya_rate+0_var2.wav"), "w").close()
            self.assertEqual(_next_var(td, "zora_priya_rate+0"), 1)


if __name__ == "__main__":
    unittest.main()
