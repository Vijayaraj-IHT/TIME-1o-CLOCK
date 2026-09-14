"""
Unit Tests for Tiny CNN & DS-CNN Speech Embedding Models & Prototypes
SIH Problem Statement 26172 - Milestones 6 & 7 Verification
"""

import sys
import unittest
import numpy as np
import tensorflow as tf

sys.path.insert(0, r"D:\SIH_Model")
from src.models.tiny_cnn import build_tiny_cnn_encoder
from src.models.ds_cnn import build_ds_cnn_encoder
from src.models.prototype import compute_prototype, compute_cosine_similarity

class TestEmbeddingModel(unittest.TestCase):

    def setUp(self):
        self.batch_size = 8
        self.input_shape = (98, 13, 1)
        self.synthetic_features = np.random.randn(self.batch_size, *self.input_shape).astype(np.float32)

    def test_tiny_cnn_32d_output_shape(self):
        encoder_32 = build_tiny_cnn_encoder(input_shape=self.input_shape, embedding_dim=32)
        out = encoder_32(self.synthetic_features, training=False).numpy()
        self.assertEqual(out.shape, (self.batch_size, 32))

    def test_tiny_cnn_64d_output_shape(self):
        encoder_64 = build_tiny_cnn_encoder(input_shape=self.input_shape, embedding_dim=64)
        out = encoder_64(self.synthetic_features, training=False).numpy()
        self.assertEqual(out.shape, (self.batch_size, 64))

    def test_ds_cnn_32d_output_shape(self):
        ds_encoder_32 = build_ds_cnn_encoder(input_shape=self.input_shape, embedding_dim=32)
        out = ds_encoder_32(self.synthetic_features, training=False).numpy()
        self.assertEqual(out.shape, (self.batch_size, 32))

    def test_ds_cnn_64d_output_shape(self):
        ds_encoder_64 = build_ds_cnn_encoder(input_shape=self.input_shape, embedding_dim=64)
        out = ds_encoder_64(self.synthetic_features, training=False).numpy()
        self.assertEqual(out.shape, (self.batch_size, 64))

    def test_l2_normalization_unit_length(self):
        for build_fn in [build_tiny_cnn_encoder, build_ds_cnn_encoder]:
            encoder = build_fn(input_shape=self.input_shape, embedding_dim=32, l2_normalize=True)
            out = encoder(self.synthetic_features, training=False).numpy()
            norms = np.linalg.norm(out, axis=-1)
            np.testing.assert_allclose(norms, np.ones(self.batch_size), atol=1e-5,
                                       err_msg="Embedding vectors must be exactly unit norm (L2 normalized)")

    def test_prototype_computation_and_cosine_similarity(self):
        k_shots = 3
        v1 = np.random.randn(32).astype(np.float32)
        v1 /= np.linalg.norm(v1)
        v2 = v1 + np.random.normal(0, 0.05, 32).astype(np.float32)
        v2 /= np.linalg.norm(v2)
        v3 = v1 + np.random.normal(0, 0.05, 32).astype(np.float32)
        v3 /= np.linalg.norm(v3)

        embeddings = np.stack([v1, v2, v3])
        prototype = compute_prototype(embeddings)

        proto_norm = np.linalg.norm(prototype)
        self.assertAlmostEqual(proto_norm, 1.0, places=5)

        sim1 = compute_cosine_similarity(prototype, v1)
        self.assertGreater(sim1, 0.90)

        impostor = np.random.randn(32).astype(np.float32)
        impostor /= np.linalg.norm(impostor)
        sim_imp = compute_cosine_similarity(prototype, impostor)
        self.assertLess(sim_imp, sim1)

if __name__ == "__main__":
    unittest.main()
