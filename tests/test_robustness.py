"""
Unit Tests for Robustness & False Activation Rate Testing
SIH Problem Statement 26172 - Milestone 11 Verification
"""

import os
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, r"D:\SIH_Model")
from src.evaluation.evaluate_robustness import RobustnessEvaluator
from src.streaming.detector import StreamingVoiceActivator
from src.models.tiny_cnn import build_tiny_cnn_encoder

class TestRobustnessPipeline(unittest.TestCase):

    def setUp(self):
        self.encoder = build_tiny_cnn_encoder(input_shape=(98, 13, 1), embedding_dim=32)
        self.proto = np.random.randn(32).astype(np.float32)
        self.proto /= np.linalg.norm(self.proto)
        self.activator = StreamingVoiceActivator(self.encoder, target_prototype=self.proto)
        self.evaluator = RobustnessEvaluator(self.activator)

    def test_snr_noise_mixing_accuracy(self):
        sr = 16000
        # Clean 440Hz sine wave
        t = np.linspace(0, 1.0, sr)
        clean = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5
        noise = np.random.normal(0, 0.2, sr).astype(np.float32)

        for target_snr in [20.0, 10.0, 0.0]:
            mixed = RobustnessEvaluator.mix_noise_at_snr(clean, noise, target_snr)
            # The injected noise portion is (mixed - clean)
            injected_noise = mixed - clean
            p_clean = np.mean(clean ** 2)
            p_injected = np.mean(injected_noise ** 2)
            measured_snr = 10.0 * np.log10(p_clean / p_injected)

            self.assertAlmostEqual(measured_snr, target_snr, delta=0.5,
                                   msg=f"Measured SNR {measured_snr:.2f} must match target {target_snr:.2f} dB")

    def test_false_activation_rate_telemetry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import soundfile as sf
            neg_files = []
            for i in range(5):
                p = os.path.join(tmpdir, f"neg_{i}.wav")
                sf.write(p, np.random.normal(0, 0.001, 16000).astype(np.float32), 16000)
                neg_files.append(p)

            far_res = self.evaluator.evaluate_false_activation_rate(neg_files, chunk_size_ms=50)

            self.assertEqual(far_res["total_negative_files"], 5)
            self.assertEqual(far_res["total_stream_seconds"], 5.0)
            self.assertIn("false_activation_rate_per_hour", far_res)
            self.assertIn("vad_inference_skip_percentage", far_res)

if __name__ == "__main__":
    unittest.main()
