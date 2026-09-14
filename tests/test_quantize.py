"""
Unit Tests for Model Quantization (TFLite & INT8)
SIH Problem Statement 26172
"""

import os
import sys
import unittest
import numpy as np
import tensorflow as tf

sys.path.insert(0, r"D:\SIH_Model")
from src.models.tiny_cnn import build_tiny_cnn_encoder
from src.export.quantize import ModelQuantizer


class TestModelQuantizer(unittest.TestCase):
    """Test suite for TFLite quantization and numerical fidelity verification."""

    @classmethod
    def setUpClass(cls):
        # Build small test encoder
        cls.model = build_tiny_cnn_encoder(input_shape=(98, 13, 1), embedding_dim=32)
        cls.quantizer = ModelQuantizer(cls.model)

    def test_quantizer_initialization(self):
        """Verify ModelQuantizer properly binds to Keras model."""
        self.assertIsNotNone(self.quantizer.model)
        self.assertEqual(self.quantizer.model.input_shape, (None, 98, 13, 1))
        self.assertEqual(self.quantizer.model.output_shape, (None, 32))

    def test_representative_dataset_generator(self):
        """Verify calibration generator yields properly shaped float32 tensors."""
        gen_fn = self.quantizer.create_representative_dataset_generator(n_samples=5)
        gen = gen_fn()
        samples = list(gen)
        self.assertEqual(len(samples), 5)
        for s in samples:
            self.assertIsInstance(s, list)
            self.assertEqual(len(s), 1)
            self.assertEqual(s[0].shape, (1, 98, 13, 1))
            self.assertEqual(s[0].dtype, np.float32)

    def test_fp32_conversion(self):
        """Verify FP32 conversion produces non-empty TFLite model."""
        tflite_fp32 = self.quantizer.convert_to_fp32()
        self.assertIsInstance(tflite_fp32, bytes)
        self.assertGreater(len(tflite_fp32), 50000)

    def test_int8_quantization_and_compression(self):
        """Verify INT8 quantization achieves < 100 KB flash footprint."""
        def dummy_gen():
            for _ in range(10):
                yield [np.random.randn(1, 98, 13, 1).astype(np.float32)]

        tflite_int8 = self.quantizer.convert_to_int8(representative_dataset_gen=dummy_gen)
        self.assertIsInstance(tflite_int8, bytes)
        size_kb = len(tflite_int8) / 1024.0

        # Must meet the < 100 KB Flash budget constraint for ESP32-S3
        self.assertLess(size_kb, 100.0)
        self.assertGreater(size_kb, 10.0)

    def test_quantization_fidelity(self):
        """Verify INT8 output maintains > 0.99 cosine alignment with FP32 baseline."""
        def dummy_gen():
            for _ in range(20):
                yield [np.random.randn(1, 98, 13, 1).astype(np.float32)]

        tflite_fp32 = self.quantizer.convert_to_fp32()
        tflite_int8 = self.quantizer.convert_to_int8(representative_dataset_gen=dummy_gen)

        test_inputs = np.random.randn(5, 98, 13, 1).astype(np.float32)
        fidelity = ModelQuantizer.evaluate_quantization_fidelity(tflite_fp32, tflite_int8, test_inputs)

        self.assertIn("mean_cosine_similarity", fidelity)
        self.assertGreater(fidelity["mean_cosine_similarity"], 0.99)
        self.assertTrue(fidelity["flash_budget_met"])
        self.assertGreater(fidelity["compression_ratio_pct"], 50.0)

    def test_save_and_inference(self):
        """Verify saving TFLite model and running inference produces valid normalized embeddings."""
        tflite_fp32 = self.quantizer.convert_to_fp32()
        temp_path = r"D:\SIH_Model\models\tflite\test_temp.tflite"
        
        saved_bytes = ModelQuantizer.save_tflite_model(tflite_fp32, temp_path)
        self.assertEqual(saved_bytes, len(tflite_fp32))
        self.assertTrue(os.path.exists(temp_path))

        # Run inference
        sample_input = np.random.randn(1, 98, 13, 1).astype(np.float32)
        embedding = ModelQuantizer.run_inference(tflite_fp32, sample_input)

        self.assertEqual(embedding.shape, (1, 32))
        norm = np.linalg.norm(embedding[0])
        self.assertAlmostEqual(norm, 1.0, places=3)

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
