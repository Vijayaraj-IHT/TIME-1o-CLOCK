"""
Unit Tests for Metric Learning Pipeline
SIH Problem Statement 26172 - Milestone 8 Verification
"""

import sys
import unittest
import numpy as np
import tensorflow as tf

sys.path.insert(0, r"D:\SIH_Model")
from src.features.mfcc import MFCCFeatureExtractor
from src.training.batch_generator import MetricLearningBatchGenerator
from src.training.validate import MetricValidator
from src.models.ds_cnn import build_ds_cnn_encoder
from src.models.losses import SupervisedContrastiveLoss

class TestMetricTrainingPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.train_manifest = r"D:\SIH_Model\data\metadata\train_manifest.csv"
        cls.val_manifest = r"D:\SIH_Model\data\metadata\validation_manifest.csv"
        cls.generator = MetricLearningBatchGenerator(cls.train_manifest, max_samples_per_class=10, seed=42)
        cls.encoder = build_ds_cnn_encoder(input_shape=(98, 13, 1), embedding_dim=32)

    def test_supcon_batch_shapes(self):
        num_classes = 8
        samples_per_class = 4
        X, y = self.generator.sample_supcon_batch(num_classes=num_classes, samples_per_class=samples_per_class)
        expected_batch = num_classes * samples_per_class
        self.assertEqual(X.shape, (expected_batch, 98, 13, 1))
        self.assertEqual(y.shape, (expected_batch,))
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(y.dtype, np.int32)

    def test_pairwise_batch_shapes_and_labels(self):
        batch_size = 16
        Xa, Xb, y = self.generator.sample_pairwise_batch(batch_size=batch_size)
        self.assertEqual(Xa.shape, (batch_size, 98, 13, 1))
        self.assertEqual(Xb.shape, (batch_size, 98, 13, 1))
        self.assertEqual(y.shape, (batch_size,))
        # Half positive (1.0), half negative (0.0)
        self.assertEqual(np.sum(y == 1.0), batch_size // 2)
        self.assertEqual(np.sum(y == 0.0), batch_size // 2)

    def test_gradient_step_validity(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss_fn = SupervisedContrastiveLoss(temperature=0.10)
        X, y = self.generator.sample_supcon_batch(num_classes=4, samples_per_class=4)

        initial_weights = [v.numpy().copy() for v in self.encoder.trainable_variables]

        with tf.GradientTape() as tape:
            emb = self.encoder(X, training=True)
            loss = loss_fn(y, emb)

        grads = tape.gradient(loss, self.encoder.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.encoder.trainable_variables))

        # Check loss is finite
        self.assertFalse(np.isnan(loss.numpy()))
        self.assertFalse(np.isinf(loss.numpy()))

        # Check at least some weights changed
        weights_changed = any(not np.allclose(w_init, w_new.numpy())
                              for w_init, w_new in zip(initial_weights, self.encoder.trainable_variables))
        self.assertTrue(weights_changed, "Model weights should update during training step")

    def test_l2_normalization_preserved(self):
        X, _ = self.generator.sample_supcon_batch(num_classes=4, samples_per_class=2)
        embeddings = self.encoder(X, training=False).numpy()
        norms = np.linalg.norm(embeddings, axis=-1)
        np.testing.assert_allclose(norms, np.ones(len(X)), atol=1e-5,
                                   err_msg="Embeddings must retain unit hypersphere norm after gradient updates")

if __name__ == "__main__":
    unittest.main()
