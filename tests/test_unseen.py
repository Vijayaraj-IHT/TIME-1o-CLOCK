"""
Unit Tests for Runtime Few-Shot Enrollment & Unseen Keyword Evaluation
SIH Problem Statement 26172 - Milestone 9 Verification
"""

import os
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, r"D:\SIH_Model")
from src.models.tiny_cnn import build_tiny_cnn_encoder
from src.enrollment.enroll import KeywordEnrollmentManager
from src.evaluation.evaluate_unseen import UnseenKeywordEvaluator

class TestUnseenKeywordPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.encoder = build_tiny_cnn_encoder(input_shape=(98, 13, 1), embedding_dim=32)
        cls.enrollment_manager = KeywordEnrollmentManager(cls.encoder)
        cls.evaluator = UnseenKeywordEvaluator(cls.encoder)

    def test_enrollment_prototype_norm_and_shapes(self):
        # Generate 5 synthetic 1-second audio clips (16000 samples)
        synthetic_clips = [np.random.normal(0, 0.1, 16000).astype(np.float32) for _ in range(5)]

        for k in [1, 2, 3, 5]:
            enroll_info = self.enrollment_manager.enroll(synthetic_clips[:k], keyword_name="ZORA")
            self.assertEqual(enroll_info["k_shots"], k)
            self.assertEqual(enroll_info["keyword_name"], "ZORA")

            proto = enroll_info["prototype"]
            self.assertEqual(proto.shape, (32,))
            # Must be unit norm
            proto_norm = np.linalg.norm(proto)
            self.assertAlmostEqual(proto_norm, 1.0, places=5)

            # Check intra similarity
            self.assertGreaterEqual(enroll_info["intra_similarity_mean"], -1.0)
            self.assertLessEqual(enroll_info["intra_similarity_mean"], 1.0)

    def test_export_c_header(self):
        proto = np.random.randn(32).astype(np.float32)
        proto /= np.linalg.norm(proto)

        with tempfile.TemporaryDirectory() as tmpdir:
            header_path = os.path.join(tmpdir, "test_proto.h")
            out_path = self.enrollment_manager.export_prototype_c_header(proto, "ZORA", header_path)

            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("#ifndef KEYWORD_PROTOTYPE_H_", content)
            self.assertIn('"ZORA"', content)
            self.assertIn("static const float ENROLLED_KEYWORD_PROTOTYPE[32]", content)

    def test_unseen_evaluation_shots_comparison(self):
        # Synthetic audio files
        with tempfile.TemporaryDirectory() as tmpdir:
            import soundfile as sf
            target_files = []
            for i in range(10):
                p = os.path.join(tmpdir, f"target_{i}.wav")
                sf.write(p, np.random.randn(16000).astype(np.float32) * 0.1, 16000)
                target_files.append(p)

            impostor_files = []
            for i in range(15):
                p = os.path.join(tmpdir, f"impostor_{i}.wav")
                sf.write(p, np.random.randn(16000).astype(np.float32) * 0.1, 16000)
                impostor_files.append(p)

            eval_res = self.evaluator.evaluate_shots_comparison(
                target_files=target_files,
                impostor_files=impostor_files,
                shot_counts=[1, 2, 3, 5],
                num_trials=2,
                keyword_name="ZORA"
            )

            self.assertEqual(eval_res["keyword_name"], "ZORA")
            for k in [1, 2, 3, 5]:
                key = f"{k}_shot"
                self.assertIn(key, eval_res["shot_evaluations"])
                shot_data = eval_res["shot_evaluations"][key]
                self.assertIn("separation_margin", shot_data)
                self.assertIn("roc_auc", shot_data)
                self.assertIn("equal_error_rate", shot_data)
                self.assertGreaterEqual(shot_data["roc_auc"], 0.0)
                self.assertLessEqual(shot_data["roc_auc"], 1.0)

if __name__ == "__main__":
    unittest.main()
