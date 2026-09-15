"""
Phase-1 Change 2: garbage prototype + veto unit tests.
No data files, no TF needed (synthetic embeddings + fake encoder).
"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.prototype import (
    apply_garbage_veto,
    compute_garbage_prototype,
    compute_prototype,
)
from src.streaming.detector import StreamingVoiceActivator


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


class _FakeTensor:
    def __init__(self, emb):
        self._emb = np.asarray(emb, dtype=np.float32)

    def numpy(self):
        return np.array([self._emb])


class FakeEncoder:
    """Cycles canned embeddings; ignores input (detector calls enc(x, training=False))."""

    def __init__(self, embs):
        self.embs = [np.asarray(e, dtype=np.float32) for e in embs]
        self.i = 0

    def __call__(self, x, training=False):
        e = self.embs[self.i % len(self.embs)]
        self.i += 1
        return _FakeTensor(e)


class FakeVAD:
    def __init__(self, speech=True):
        self._speech = speech

    def is_speech(self, chunk):
        return self._speech

    def reset(self):
        pass


KW = _unit([1.0, 0.2, 0.1, 0.0] + [0.0] * 28)
GB = _unit([0.0, 1.0, 0.3, 0.1] + [0.0] * 28)


class TestGarbageMath(unittest.TestCase):
    def test_garbage_is_normalized_mean(self):
        g = compute_garbage_prototype(np.stack([GB, GB, GB]))
        self.assertAlmostEqual(float(np.linalg.norm(g)), 1.0, places=5)
        np.testing.assert_allclose(g, GB, atol=1e-5)

    def test_garbage_min_count_guard(self):
        with self.assertRaises(ValueError):
            compute_garbage_prototype(np.stack([GB, GB]), min_count=3)

    def test_veto_pass(self):
        score, vetoed = apply_garbage_veto(0.93, 0.70, 0.05)
        self.assertFalse(vetoed)
        self.assertAlmostEqual(score, 0.93)

    def test_veto_fires(self):
        score, vetoed = apply_garbage_veto(0.86, 0.91, 0.05)
        self.assertTrue(vetoed)
        self.assertAlmostEqual(score, -1.0)

    def test_veto_boundary_is_strict_less_than(self):
        # margin exactly met (0.90 - 0.85 == 0.05) => NOT vetoed
        score, vetoed = apply_garbage_veto(0.90, 0.85, 0.05)
        self.assertFalse(vetoed)
        self.assertAlmostEqual(score, 0.90)


class TestDetectorVeto(unittest.TestCase):
    def _run_chunks(self, det, n=25):
        t = np.arange(800) / 16000.0
        chunk = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        res = None
        for i in range(n):
            res = det.process_chunk(chunk, timestamp_ms=float(i * 50))
        return res

    def test_keyword_passes_with_garbage_present(self):
        det = StreamingVoiceActivator(
            FakeEncoder([KW]), KW, vad=FakeVAD(True),
            garbage_prototype=GB, garbage_margin=0.05)
        res = self._run_chunks(det)
        self.assertFalse(res["inference_skipped"])
        self.assertFalse(res["garbage_vetoed"])
        self.assertGreater(res["raw_similarity"], 0.9)

    def test_garbage_like_input_vetoed(self):
        det = StreamingVoiceActivator(
            FakeEncoder([GB]), KW, vad=FakeVAD(True),
            garbage_prototype=GB, garbage_margin=0.05)
        res = self._run_chunks(det)
        self.assertTrue(res["garbage_vetoed"])
        self.assertAlmostEqual(res["raw_similarity"], -1.0)
        self.assertFalse(res["is_activated"])

    def test_legacy_none_garbage_unchanged(self):
        det = StreamingVoiceActivator(FakeEncoder([KW]), KW, vad=FakeVAD(True))
        res = self._run_chunks(det)
        self.assertIsNone(res["garbage_similarity"])
        self.assertFalse(res["garbage_vetoed"])
        self.assertGreater(res["raw_similarity"], 0.9)


class TestGarbageExport(unittest.TestCase):
    def test_header_contains_garbage(self):
        from src.enrollment.enroll import KeywordEnrollmentManager

        mgr = KeywordEnrollmentManager(encoder=None)
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "keyword_prototype.h")
            mgr.export_prototype_c_header(
                compute_prototype(np.stack([KW, KW, KW])),
                keyword_name="TESTKW", output_filepath=out,
                is_valid=True, garbage=GB)
            content = open(out, encoding="utf-8").read()
        self.assertIn("GARBAGE_PROTOTYPE", content)
        self.assertIn("GARBAGE_MARGIN", content)
        self.assertIn("TESTKW", content)

    def test_header_zeros_when_no_garbage(self):
        from src.enrollment.enroll import KeywordEnrollmentManager

        mgr = KeywordEnrollmentManager(encoder=None)
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "keyword_prototype.h")
            mgr.export_prototype_c_header(
                compute_prototype(np.stack([KW, KW, KW])),
                keyword_name="TESTKW", output_filepath=out, is_valid=True)
            content = open(out, encoding="utf-8").read()
        self.assertIn("GARBAGE_PROTOTYPE", content)  # always emitted (fw refs it)
        self.assertIn("0.0000000f", content)

    def test_enroll_with_negatives(self):
        from src.enrollment.enroll import KeywordEnrollmentManager

        t = np.arange(16000) / 16000.0
        tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        noise = (np.random.RandomState(0).randn(16000) * 0.05).astype(np.float32)
        mgr = KeywordEnrollmentManager(FakeEncoder([KW, KW, KW, GB, GB, GB]))
        out = mgr.enroll([tone, tone, tone], keyword_name="T",
                         negative_inputs=[noise, noise, noise])
        self.assertIn("garbage_prototype", out)
        self.assertEqual(out["garbage_n"], 3)
        self.assertAlmostEqual(float(np.linalg.norm(out["garbage_prototype"])), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
