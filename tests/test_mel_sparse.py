"""
Sparse Mel filterbank header tests (TRACK_A Flash follow-up).

The firmware filterbank is run-encoded (247/10280 nonzero, one contiguous
run per mel). These tests prove the COMMITTED header bytes equal librosa
truth (the same call training uses) and that the encoding is self-consistent.
No compiler needed; the g++ bit-parity proof lives in
tools/verify_feature_parity/check_parity.py (+ dense-vs-sparse fw-vs-fw).
"""
import os
import re
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HEADER = os.path.join(REPO, "src", "deployment", "esp32", "mel_filterbank.h")


def parse_header():
    text = open(HEADER, encoding="utf-8").read()

    def arr(name, typ):
        m = re.search(name + r"\[\d+\] = \{(.*?)\};", text, re.S)
        assert m, name
        toks = [x.strip() for x in m.group(1).replace("\n", " ").split(",")
                if x.strip()]
        if typ is float:
            return np.array([float(x[:-1]) for x in toks])  # strip 'f'
        return np.array([int(x) for x in toks])

    nnz = int(re.search(r"MEL_FILTERBANK_NNZ = (\d+)", text).group(1))
    nmels = int(re.search(r"MEL_FILTERBANK_N_MELS = (\d+)", text).group(1))
    nbins = int(re.search(r"MEL_FILTERBANK_N_BINS = (\d+)", text).group(1))
    return {"nnz": nnz, "nmels": nmels, "nbins": nbins,
            "start": arr("MEL_SPARSE_START", int),
            "len": arr("MEL_SPARSE_LEN", int),
            "off": arr("MEL_SPARSE_OFF", int),
            "val": arr("MEL_SPARSE_VAL", float)}


def librosa_truth():
    import librosa
    return np.asarray(librosa.filters.mel(sr=16000, n_fft=512, n_mels=40,
                                          fmin=20, fmax=4000,
                                          htk=True, norm="slaney"),
                      dtype=np.float64)


class TestMelSparseHeader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = parse_header()
        cls.truth = librosa_truth()

    def test_constants(self):
        self.assertEqual(self.h["nmels"], 40)
        self.assertEqual(self.h["nbins"], 257)
        self.assertEqual(self.h["nnz"], 247)
        for k in ("start", "len", "off"):
            self.assertEqual(len(self.h[k]), 40)
        self.assertEqual(len(self.h["val"]), 247)

    def test_runs_cover_librosa_nonzeros_exactly(self):
        rec = np.zeros((40, 257))
        for r in range(40):
            s, ln, o = (int(self.h["start"][r]), int(self.h["len"][r]),
                        int(self.h["off"][r]))
            self.assertGreater(ln, 0)
            self.assertLessEqual(s + ln, 257)
            self.assertLessEqual(o + ln, 247)
            rec[r, s:s + ln] = self.h["val"][o:o + ln]
        # every librosa nonzero is covered, every covered cell is nonzero
        nz_ref = (self.truth != 0.0)
        nz_rec = (rec != 0.0)
        np.testing.assert_array_equal(nz_rec, nz_ref)
        # ... and the values match to float32-roundtrip level
        d = np.abs(rec - self.truth)
        self.assertLess(float(d.max()), 1e-9)

    def test_values_are_exact_float32_roundtrips(self):
        # header literals must parse back to the same float32 librosa
        # produces (no decimal truncation beyond float32 precision)
        rec = np.zeros((40, 257))
        for r in range(40):
            s, ln, o = (int(self.h["start"][r]), int(self.h["len"][r]),
                        int(self.h["off"][r]))
            rec[r, s:s + ln] = self.h["val"][o:o + ln]
        ref32 = self.truth.astype(np.float32).astype(np.float64)
        np.testing.assert_array_equal(rec.astype(np.float32).astype(np.float64),
                                      ref32)

    def test_uint8_ranges(self):
        for k in ("start", "len", "off"):
            self.assertTrue(bool((self.h[k] >= 0).all()), k)
            self.assertTrue(bool((self.h[k] < 256).all()), k)

    def test_flash_cost(self):
        total = len(self.h["val"]) * 4 + 40 * 3
        self.assertEqual(total, 1108)
        self.assertLess(total, 41120 * 0.05)  # >95% smaller than dense


if __name__ == "__main__":
    unittest.main()
