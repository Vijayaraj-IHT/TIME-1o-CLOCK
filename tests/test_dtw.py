"""Track-B DTW prototype tests: core distances, templates, streaming spotter."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dtw.dtw_core import dtw_distance, lb_keogh, dtw_envelope
from src.dtw.templates import build_template, resample_frames
from src.dtw.spot import DTWSpotter


class TestDTWCore(unittest.TestCase):
    def test_identical_is_zero(self):
        rng = np.random.RandomState(0)
        a = rng.randn(25, 13)
        self.assertAlmostEqual(dtw_distance(a, a), 0.0)

    def test_known_values(self):
        self.assertAlmostEqual(dtw_distance([[0.0]], [[1.0]]), 1.0)  # squared
        # band=0 forces the diagonal: (0-0)^2 + (0-1)^2 = 1
        self.assertAlmostEqual(dtw_distance([[0.0], [0.0]], [[0.0], [1.0]],
                                            band=0), 1.0)

    def test_symmetry(self):
        rng = np.random.RandomState(1)
        a, b = rng.randn(20, 5), rng.randn(24, 5)
        self.assertAlmostEqual(dtw_distance(a, b), dtw_distance(b, a))

    def test_band_violation_is_inf(self):
        a = np.zeros((10, 3))
        b = np.zeros((30, 3))
        self.assertEqual(dtw_distance(a, b, band=5), float("inf"))

    def test_lb_keogh_bounds_dtw(self):
        rng = np.random.RandomState(2)
        for _ in range(5):
            t = rng.randn(30, 4)
            q = t + rng.randn(30, 4) * 0.5
            up, lo = dtw_envelope(t, band=4)
            self.assertLessEqual(lb_keogh(q, up, lo), dtw_distance(q, t) + 1e-9)


class TestTemplates(unittest.TestCase):
    def test_identical_clips_centroid(self):
        rng = np.random.RandomState(3)
        m = rng.randn(40, 13)
        tpl = build_template("kw", [m, m.copy(), m.copy()])
        self.assertEqual(tpl.frame_count, 40)
        self.assertEqual(tpl.n_clips, 3)
        np.testing.assert_allclose(tpl.centroid, m, rtol=1e-9)
        # (a+a+a)/3 is not bit-exact in floating point: ~1e-30, not 0.0
        self.assertTrue(all(d < 1e-20 for d in tpl.intra_dists))

    def test_resample_length(self):
        m = np.random.RandomState(4).randn(50, 13)
        self.assertEqual(resample_frames(m, 30).shape, (30, 13))
        self.assertEqual(resample_frames(m, 60).shape, (60, 13))
        # endpoints preserved by linear interpolation
        r = resample_frames(m, 60)
        np.testing.assert_allclose(r[0], m[0], rtol=1e-9)
        np.testing.assert_allclose(r[-1], m[-1], rtol=1e-9)


class TestSpotter(unittest.TestCase):
    def _ramp_template(self):
        ramp = np.linspace(0, 1, 12)[:, None] * np.ones((1, 4))
        return build_template("ramp", [ramp, ramp + 0.01])

    def test_spots_embedded_template_once(self):
        tpl = self._ramp_template()
        sp = DTWSpotter(tpl, threshold=0.05, cooldown_frames=20)
        rng = np.random.RandomState(5)
        noise = rng.randn(30, 4) * 2 + 5.0  # far from ramp (values 0..1)
        stream = np.concatenate([noise, tpl.centroid, noise])
        trigs = [i for i, f in enumerate(stream)
                 if sp.update(f, i * 10.0)["triggered"]]
        self.assertEqual(len(trigs), 1)
        # template occupies frames 30..41; trigger at its end (open-end DTW)
        self.assertTrue(38 <= trigs[0] <= 44, trigs)

    def test_pure_noise_never_triggers(self):
        tpl = self._ramp_template()
        sp = DTWSpotter(tpl, threshold=0.05, cooldown_frames=20)
        rng = np.random.RandomState(6)
        n = sum(sp.update(f, i * 10.0)["triggered"]
                for i, f in enumerate(rng.randn(100, 4) * 2 + 5.0))
        self.assertEqual(n, 0)

    def test_double_template_two_triggers(self):
        tpl = self._ramp_template()
        sp = DTWSpotter(tpl, threshold=0.05, cooldown_frames=20)
        rng = np.random.RandomState(7)
        gap = rng.randn(40, 4) * 2 + 5.0
        stream = np.concatenate([tpl.centroid, gap, tpl.centroid, gap])
        n = sum(sp.update(f, i * 10.0)["triggered"] for i, f in enumerate(stream))
        self.assertEqual(n, 2)


if __name__ == "__main__":
    unittest.main()


class TestDurationGate(unittest.TestCase):
    def _const_template(self):
        v = np.full((20, 4), 0.25)
        return build_template("const", [v])

    def test_gate_blocks_single_frame_match(self):
        tpl = self._const_template()  # default gate: 20//8 = 2
        self.assertEqual(DTWSpotter(tpl, threshold=1.0).min_duration_frames, 2)
        rng = np.random.RandomState(9)
        w = rng.randn(50, 4) * 2 + 5.0
        v = np.full((1, 4), 0.25)
        stream = np.concatenate([w, v, w])
        gated = DTWSpotter(tpl, threshold=1.0, cooldown_frames=200)
        n_gated = sum(gated.update(f, i * 10.0)["triggered"] for i, f in enumerate(stream))
        self.assertEqual(n_gated, 0)
        # control: the degenerate match EXISTS -- ungated spotter fires on it
        free = DTWSpotter(tpl, threshold=1.0, cooldown_frames=200,
                          min_duration_frames=0)
        n_free = sum(free.update(f, i * 10.0)["triggered"] for i, f in enumerate(stream))
        self.assertGreaterEqual(n_free, 1)

    def test_gate_passes_genuine_with_margin(self):
        ramp = np.linspace(0, 1, 12)[:, None] * np.ones((1, 4))
        tpl = build_template("ramp", [ramp, ramp + 0.01])
        sp = DTWSpotter(tpl, threshold=0.05, cooldown_frames=20)
        rng = np.random.RandomState(5)
        noise = rng.randn(30, 4) * 2 + 5.0
        stream = np.concatenate([noise, tpl.centroid, noise])
        reps = [sp.update(f, i * 10.0) for i, f in enumerate(stream)]
        trigs = [r for r in reps if r["triggered"]]
        self.assertEqual(len(trigs), 1)
        self.assertGreaterEqual(trigs[0]["frames_consumed"],
                                sp.min_duration_frames)


class TestSpotterBank(unittest.TestCase):
    def _bank(self):
        # Variants must be SHAPE-distinct: pure time-scale copies of the same
        # line collapse under DTW (that is the point of warping). Real rate
        # variants are spectrally distinct (pilot: 130x intra spread), so
        # rise-vs-fall is the honest synthetic analogue.
        from src.dtw.spot import DTWSpotterBank
        rise = np.linspace(0, 1, 16)[:, None] * np.ones((1, 4))
        fall = np.linspace(1, 0, 10)[:, None] * np.ones((1, 4))
        bank = DTWSpotterBank({"rise": build_template("rise", [rise]),
                               "fall": build_template("fall", [fall])},
                              threshold=0.05, cooldown_frames=30)
        return bank, rise, fall

    def test_bank_catches_both_variants(self):
        from src.dtw.spot import DTWSpotterBank  # noqa: F401 (import guard)
        bank, rise, fall = self._bank()
        rng = np.random.RandomState(11)
        gap = rng.randn(50, 4) * 2 + 5.0
        stream = np.concatenate([gap, rise, gap, fall, gap])
        got = []
        for i, f in enumerate(stream):
            r = bank.update(f, i * 10.0)
            if r["triggered"]:
                got.append(r["template"])
        self.assertEqual(got, ["rise", "fall"])

    def test_bank_shared_cooldown_single_trigger(self):
        from src.dtw.spot import DTWSpotterBank
        ramp = np.linspace(0, 1, 12)[:, None] * np.ones((1, 4))
        tpl = build_template("ramp", [ramp])
        bank = DTWSpotterBank({"a": tpl, "b": tpl}, threshold=0.05,
                              cooldown_frames=30)
        rng = np.random.RandomState(12)
        gap = rng.randn(40, 4) * 2 + 5.0
        stream = np.concatenate([gap, ramp, gap])
        n = sum(bank.update(f, i * 10.0)["triggered"] for i, f in enumerate(stream))
        self.assertEqual(n, 1)
