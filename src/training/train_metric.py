"""
Metric Learning Training Engine for Few-Shot Edge Voice Activator
SIH Problem Statement 26172 - Milestone 8

Trains Tiny CNN and Depthwise Separable CNN (DS-CNN) encoders using Supervised Contrastive Loss.
Evaluates cosine separation margin and prototype-based few-shot retrieval on held-out speakers.
"""

import os
import sys
import json
import time
from datetime import datetime
import numpy as np
import tensorflow as tf

# Enable deserialization for Lambda layer if needed
try:
    tf.keras.config.enable_unsafe_deserialization()
except Exception:
    pass

sys.path.insert(0, r"D:\SIH_Model")
from src.features.mfcc import MFCCFeatureExtractor
from src.models.tiny_cnn import build_tiny_cnn_encoder
from src.models.ds_cnn import build_ds_cnn_encoder
from src.models.losses import SupervisedContrastiveLoss, ContrastiveLoss
from src.training.batch_generator import MetricLearningBatchGenerator
from src.training.validate import MetricValidator

class MetricTrainer:
    def __init__(self, architecture="ds_cnn", embedding_dim=32, loss_type="supervised_contrastive",
                 temperature=0.10, margin=1.0, learning_rate=0.001, seed=42):
        self.architecture = architecture.lower()
        self.embedding_dim = embedding_dim
        self.loss_type = loss_type
        self.temperature = temperature
        self.margin = margin
        self.learning_rate = learning_rate
        self.seed = seed

        # Build model
        self.input_shape = (98, 13, 1)
        if self.architecture == "tiny_cnn":
            self.model = build_tiny_cnn_encoder(
                input_shape=self.input_shape,
                embedding_dim=embedding_dim,
                l2_normalize=True,
                model_name="tiny_cnn_encoder"
            )
        elif self.architecture in ["ds_cnn", "dscnn"]:
            self.model = build_ds_cnn_encoder(
                input_shape=self.input_shape,
                embedding_dim=embedding_dim,
                l2_normalize=True,
                model_name="ds_cnn_encoder"
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Setup Loss Function
        if self.loss_type == "supervised_contrastive":
            self.loss_fn = SupervisedContrastiveLoss(temperature=self.temperature)
        elif self.loss_type == "contrastive":
            self.loss_fn = ContrastiveLoss(margin=self.margin)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        # Setup Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    @tf.function
    def _train_step_supcon(self, x_batch, y_batch):
        with tf.GradientTape() as tape:
            embeddings = self.model(x_batch, training=True)
            loss_value = self.loss_fn(y_batch, embeddings)

        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_value

    def train(self, train_generator, val_validator, epochs=15, steps_per_epoch=50,
              num_classes_per_batch=16, samples_per_class=4, checkpoint_dir=r"D:\SIH_Model\models\checkpoints"):
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_model_path = os.path.join(checkpoint_dir, f"{self.architecture}_metric_best.keras")
        best_weights_path = os.path.join(checkpoint_dir, f"{self.architecture}_metric_best.weights.h5")
        final_model_path = os.path.join(checkpoint_dir, f"{self.architecture}_metric_final.keras")

        print("="*75)
        print(f"STARTING METRIC LEARNING TRAINING: {self.architecture.upper()}")
        print(f"  - Parameters:      {self.model.count_params():,}")
        print(f"  - Embedding Dim:   {self.embedding_dim}-D (Unit Hypersphere)")
        print(f"  - Loss Objective:  {self.loss_type} (Temp: {self.temperature})")
        print(f"  - Batch Structure: {num_classes_per_batch} classes x {samples_per_class} shots = {num_classes_per_batch * samples_per_class} clips/step")
        print(f"  - Epochs:          {epochs} ({steps_per_epoch} steps/epoch)")
        print("="*75)

        history = []
        best_separation_margin = -1.0
        start_time = time.perf_counter()

        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            step_losses = []

            for step in range(steps_per_epoch):
                x_batch, y_batch = train_generator.sample_supcon_batch(
                    num_classes=num_classes_per_batch,
                    samples_per_class=samples_per_class,
                    augment=True
                )
                loss_val = self._train_step_supcon(x_batch, y_batch)
                step_losses.append(float(loss_val.numpy()))

            avg_train_loss = float(np.mean(step_losses))
            epoch_duration = time.perf_counter() - epoch_start

            # Evaluate on validation set
            val_validator._embedding_cache.clear()  # Clear cache to reflect updated weights
            val_results = val_validator.evaluate_full(num_pairs=300, shots=3, max_queries_per_class=10)
            margin = val_results["cosine_separation"]["separation_margin"]
            top1_recall = val_results["few_shot_retrieval"]["top1_recall_accuracy"]
            auc_val = val_results["cosine_separation"]["roc_auc"]

            is_best = margin > best_separation_margin
            if is_best:
                best_separation_margin = margin
                try:
                    self.model.save(best_model_path)
                except Exception:
                    pass
                try:
                    self.model.save_weights(best_weights_path)
                except Exception:
                    pass

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 4),
                "duration_seconds": round(epoch_duration, 2),
                "val_separation_margin": round(margin, 4),
                "val_mean_pos_sim": val_results["cosine_separation"]["mean_positive_cosine"],
                "val_mean_neg_sim": val_results["cosine_separation"]["mean_negative_cosine"],
                "val_top1_recall": round(top1_recall, 4),
                "val_roc_auc": round(auc_val, 4),
                "is_best": is_best
            }
            history.append(epoch_record)

            best_marker = " [* BEST]" if is_best else ""
            print(f"Epoch {epoch:2d}/{epochs:2d} ({epoch_duration:.1f}s) | "
                  f"Loss: {avg_train_loss:.4f} | "
                  f"Val Margin: {margin:+.4f} (Pos: {epoch_record['val_mean_pos_sim']:.3f}, Neg: {epoch_record['val_mean_neg_sim']:.3f}) | "
                  f"Top-1: {top1_recall*100:.1f}% | AUC: {auc_val:.4f}{best_marker}")

        total_training_time = time.perf_counter() - start_time
        try:
            self.model.save(final_model_path)
        except Exception:
            pass

        # Load best model weights for final rigorous validation
        if os.path.exists(best_weights_path):
            self.model.load_weights(best_weights_path)
            val_validator.encoder = self.model
            val_validator._embedding_cache.clear()
        elif os.path.exists(best_model_path):
            try:
                self.model = tf.keras.models.load_model(best_model_path, compile=False, safe_mode=False)
                val_validator.encoder = self.model
                val_validator._embedding_cache.clear()
            except Exception:
                pass

        print("\nExecuting comprehensive final validation benchmark...")
        final_validation = val_validator.evaluate_full(num_pairs=1000, shots=3, max_queries_per_class=20)

        print("\n" + "="*75)
        print(f"TRAINING COMPLETE: {self.architecture.upper()}")
        print(f"  Total Duration:         {total_training_time:.1f} s")
        print(f"  Initial Train Loss:     {history[0]['train_loss']:.4f}")
        print(f"  Final Train Loss:       {history[-1]['train_loss']:.4f}")
        print(f"  Best Separation Margin: {best_separation_margin:+.4f}")
        print(f"  Validation ROC-AUC:     {final_validation['cosine_separation']['roc_auc']:.4f}")
        print(f"  Validation EER:         {final_validation['cosine_separation']['equal_error_rate']:.4f}")
        print(f"  Top-1 Recall (3-shot):  {final_validation['few_shot_retrieval']['top1_recall_accuracy']*100:.2f}%")
        print(f"  Top-5 Recall (3-shot):  {final_validation['few_shot_retrieval']['top5_recall_accuracy']*100:.2f}%")
        print(f"  Best Model Checkpoint:  {best_model_path}")
        print("="*75)

        return {
            "architecture": self.architecture,
            "parameters": self.model.count_params(),
            "embedding_dim": self.embedding_dim,
            "epochs": epochs,
            "total_time_seconds": round(total_training_time, 2),
            "history": history,
            "best_separation_margin": round(best_separation_margin, 4),
            "final_validation": final_validation,
            "best_checkpoint": best_model_path,
            "final_checkpoint": final_model_path
        }
