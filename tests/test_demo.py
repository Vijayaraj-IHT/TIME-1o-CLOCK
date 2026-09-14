"""
Unit Tests for End-to-End Live Demonstration Pipeline
SIH Problem Statement 26172 - Milestone 14 Verification
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, r"D:\SIH_Model")
from src.streaming.demo_pipeline import EndToEndVoiceActivatorDemo


class TestDemoPipeline(unittest.TestCase):
    """Test suite for live demonstration pipeline and ASR handover mechanism."""

    @classmethod
    def setUpClass(cls):
        cls.model_path = r"D:\SIH_Model\models\tflite\voice_activator_int8.tflite"
        cls.demo = EndToEndVoiceActivatorDemo(
            tflite_model_path=cls.model_path,
            sample_rate=16000,
            chunk_size_ms=50
        )
        cls.enrollment_paths = [
            r"D:\SIH_Model\data\raw\custom_keywords\zora\zora_david_rate+0_var0.wav",
            r"D:\SIH_Model\data\raw\custom_keywords\zora\zora_david_rate+0_var1.wav",
            r"D:\SIH_Model\data\raw\custom_keywords\zora\zora_david_rate+1_var0.wav"
        ]

    def test_demo_initialization(self):
        """Verify demonstration engine loads quantized INT8 model properly."""
        self.assertIsNotNone(self.demo.interpreter)
        self.assertEqual(len(self.demo.input_details), 1)
        self.assertEqual(len(self.demo.output_details), 1)
        self.assertEqual(self.demo.chunk_size, 800)

    def test_dynamic_few_shot_enrollment(self):
        """Verify dynamic prototype enrollment computes unit-norm 32-D centroid."""
        res = self.demo.enroll_keyword(self.enrollment_paths, keyword_name="ZORA")
        self.assertEqual(res["keyword"], "ZORA")
        self.assertEqual(res["shots_count"], 3)
        self.assertEqual(res["prototype_dimension"], 32)
        self.assertAlmostEqual(res["prototype_norm"], 1.0, places=3)
        self.assertGreater(res["intra_similarity"], 0.90)

    def test_streaming_vad_silence_gating(self):
        """Verify silence chunks trigger VAD_SKIP to preserve CPU."""
        silence_chunk = np.zeros(800, dtype=np.float32)
        # Fill buffer
        for i in range(20):
            res = self.demo.process_chunk(silence_chunk, timestamp_ms=i * 50.0)

        # After ring buffer fills, silence must trigger VAD_SKIP
        res = self.demo.process_chunk(silence_chunk, timestamp_ms=1050.0)
        self.assertEqual(res["event"], "VAD_SKIP")
        self.assertFalse(res["vad_speech"])

    def test_asr_handover_dispatch(self):
        """Verify ASR handover packetization and duration."""
        dummy_audio = np.random.randn(32000).astype(np.float32) * 0.1
        dispatch_res = self.demo._dispatch_to_remote_asr(dummy_audio)

        self.assertIn("status", dispatch_res)
        self.assertEqual(dispatch_res["status"], "HANDOVER_DISPATCH_SUCCESS")
        self.assertEqual(dispatch_res["payload_duration_sec"], 2.0)
        self.assertEqual(dispatch_res["payload_bytes"], 64000)
        self.assertIn("Whisper", dispatch_res["asr_engine"])


if __name__ == "__main__":
    unittest.main()
