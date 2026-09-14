"""
Unit Tests for Audio Feature Extraction (MFCC & Log-Mel)
SIH Problem Statement 26172 - Milestone 5 Verification
"""

import sys
import unittest
import numpy as np

sys.path.insert(0, r"D:\SIH_Model")
from src.features.mfcc import MFCCFeatureExtractor
from src.features.logmel import LogMelFeatureExtractor

class TestAudioFeatures(unittest.TestCase):

    def setUp(self):
        self.sr = 16000
        self.duration = 1.0
        self.num_samples = int(self.sr * self.duration)
        self.mfcc_extractor = MFCCFeatureExtractor(sample_rate=self.sr, n_mfcc=13, n_mels=40)
        self.logmel_extractor = LogMelFeatureExtractor(sample_rate=self.sr, n_mels=40)

        # Synthetic test audio: 440 Hz pure sine wave
        t = np.linspace(0, self.duration, self.num_samples, endpoint=False)
        self.sine_audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        self.silence_audio = np.zeros(self.num_samples, dtype=np.float32)

    def test_mfcc_output_shape(self):
        feat = self.mfcc_extractor.extract(self.sine_audio)
        self.assertEqual(feat.ndim, 2, "MFCC feature output must be 2D")
        self.assertEqual(feat.shape[0], 98, "MFCC frames count must be 98 for 1.0s audio")
        self.assertEqual(feat.shape[1], 13, "MFCC feature coefficients must be 13")
        self.assertEqual(feat.dtype, np.float32, "MFCC dtype must be float32")

    def test_logmel_output_shape(self):
        feat = self.logmel_extractor.extract(self.sine_audio)
        self.assertEqual(feat.ndim, 2, "Log-Mel feature output must be 2D")
        self.assertEqual(feat.shape[0], 98, "Log-Mel frames count must be 98 for 1.0s audio")
        self.assertEqual(feat.shape[1], 40, "Log-Mel filterbank channels must be 40")
        self.assertEqual(feat.dtype, np.float32, "Log-Mel dtype must be float32")

    def test_numerical_stability_on_silence(self):
        # Must not produce NaNs or Infs on zero input
        mfcc_silence = self.mfcc_extractor.extract(self.silence_audio)
        logmel_silence = self.logmel_extractor.extract(self.silence_audio)

        self.assertFalse(np.isnan(mfcc_silence).any(), "NaN detected in MFCC silence extraction")
        self.assertFalse(np.isinf(mfcc_silence).any(), "Inf detected in MFCC silence extraction")
        self.assertFalse(np.isnan(logmel_silence).any(), "NaN detected in Log-Mel silence extraction")
        self.assertFalse(np.isinf(logmel_silence).any(), "Inf detected in Log-Mel silence extraction")

    def test_deterministic_output(self):
        feat1 = self.mfcc_extractor.extract(self.sine_audio)
        feat2 = self.mfcc_extractor.extract(self.sine_audio)
        np.testing.assert_array_equal(feat1, feat2, "Feature extraction must be completely deterministic")

if __name__ == "__main__":
    unittest.main()
