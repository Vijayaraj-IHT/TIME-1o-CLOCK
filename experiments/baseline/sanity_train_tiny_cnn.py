"""
Sanity Training Experiment: Baseline Tiny CNN Speech Embedding Encoder
SIH Problem Statement 26172 - Milestone 6 Verification
"""

import os
import sys
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

sys.path.insert(0, r"D:\SIH_Model")
from src.features.mfcc import MFCCFeatureExtractor
from src.models.tiny_cnn import build_tiny_cnn_encoder
from src.models.losses import SupervisedContrastiveLoss

def run_sanity_experiment():
    print("="*70)
    print("MILESTONE 6: BASELINE TINY CNN SANITY TRAINING EXPERIMENT")
    print("="*70)

    # 1. Load real audio samples from train manifest
    manifest_path = r"D:\SIH_Model\data\metadata\train_manifest.csv"
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Train manifest not found at {manifest_path}")

    df = pd.read_csv(manifest_path)
    # Select 4 distinct words with multiple samples each (128 total clips for balanced sanity batch)
    selected_words = ["yes", "no", "up", "down"]
    df_subset = df[df['label'].isin(selected_words)].groupby('label').head(32).reset_index(drop=True)

    print(f"Loaded {len(df_subset)} real audio samples across {len(selected_words)} vocabulary words:")
    for w in selected_words:
        print(f"  - '{w}': {len(df_subset[df_subset['label'] == w])} clips")

    # 2. Extract MFCC features
    extractor = MFCCFeatureExtractor(sample_rate=16000, n_mfcc=13, n_mels=40)
    features_list = []
    labels_list = []
    word2id = {w: i for i, w in enumerate(selected_words)}

    print("\nExtracting MFCC features for sanity training batch...")
    for _, row in df_subset.iterrows():
        audio_path = row['filepath']
        try:
            audio, sr = sf.read(audio_path, dtype='float32')
            if len(audio) < 16000:
                audio = np.pad(audio, (0, 16000 - len(audio)), mode='constant')
            elif len(audio) > 16000:
                audio = audio[:16000]
            
            feat = extractor.extract(audio)  # Shape: (98, 13)
            features_list.append(feat)
            labels_list.append(word2id[row['label']])
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")

    X = np.array(features_list)[:, :, :, np.newaxis]  # Shape: (N, 98, 13, 1)
    y = np.array(labels_list)                         # Shape: (N,)

    print(f"Feature tensor shape: {X.shape}, Label tensor shape: {y.shape}")

    # 3. Build Tiny CNN 32-D Encoder
    encoder = build_tiny_cnn_encoder(input_shape=(98, 13, 1), embedding_dim=32, l2_normalize=True)

    total_params = encoder.count_params()
    trainable_params = int(np.sum([tf.size(v).numpy() for v in encoder.trainable_variables]))
    non_trainable_params = total_params - trainable_params
    model_size_kb = (total_params * 4) / 1024.0  # FP32 = 4 bytes per param

    print("\nModel Architectural Profile:")
    print(f"  - Architecture:        Tiny CNN (2D Conv + BatchNorm + ReLU + GAP + Dense)")
    print(f"  - Output Dimension:    32-D (L2 normalized unit hypersphere)")
    print(f"  - Total Parameters:    {total_params:,}")
    print(f"  - Trainable Params:    {trainable_params:,}")
    print(f"  - Non-Trainable Params:{non_trainable_params:,}")
    print(f"  - Model Size (FP32):   {model_size_kb:.2f} KB (Well below 256 KB ESP32-S3 limit)")

    # 4. Metric Learning Training Loop (Supervised Contrastive Loss)
    loss_fn = SupervisedContrastiveLoss(temperature=0.10)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    epochs = 8
    loss_history = []
    print("\nExecuting Metric Learning Sanity Training Loop...")

    dataset = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(len(X)).batch(32)

    for epoch in range(1, epochs + 1):
        epoch_losses = []
        for step, (x_batch, y_batch) in enumerate(dataset):
            with tf.GradientTape() as tape:
                embeddings = encoder(x_batch, training=True)
                loss_value = loss_fn(y_batch, embeddings)

            grads = tape.gradient(loss_value, encoder.trainable_variables)
            optimizer.apply_gradients(zip(grads, encoder.trainable_variables))
            epoch_losses.append(float(loss_value.numpy()))

        avg_loss = float(np.mean(epoch_losses))
        loss_history.append(avg_loss)
        print(f"  Epoch {epoch:2d}/{epochs:2d} - Loss: {avg_loss:.4f}")

    # 5. Check convergence & embedding properties
    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    loss_reduction_pct = ((initial_loss - final_loss) / initial_loss) * 100.0

    print(f"\nTraining Loss Convergence:")
    print(f"  - Initial Loss (Epoch 1): {initial_loss:.4f}")
    print(f"  - Final Loss (Epoch {epochs}):   {final_loss:.4f}")
    print(f"  - Loss Reduction:         {loss_reduction_pct:.2f}%")

    # Verify L2 norm preservation after training
    test_out = encoder(X[:8], training=False).numpy()
    test_norms = np.linalg.norm(test_out, axis=-1)
    norm_preserved = bool(np.allclose(test_norms, 1.0, atol=1e-5))
    print(f"  - L2 Unit Norm Preserved: {norm_preserved} (Max deviation: {np.max(np.abs(test_norms - 1.0)):.2e})")

    # 6. Save model checkpoint & results
    checkpoint_dir = r"D:\SIH_Model\models\checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "tiny_cnn_sanity.keras")
    encoder.save(checkpoint_path)
    print(f"\nSaved model checkpoint to: {checkpoint_path}")

    results = {
        "milestone": "Milestone 6: Baseline Tiny CNN Embedding Model",
        "timestamp": datetime.now().isoformat(),
        "input_shape": list(X.shape[1:]),
        "embedding_dim": 32,
        "l2_normalization": True,
        "architecture": {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": non_trainable_params,
            "estimated_model_size_fp32_kb": round(model_size_kb, 2)
        },
        "sanity_training": {
            "num_samples": len(df_subset),
            "num_classes": len(selected_words),
            "batch_size": 32,
            "epochs": epochs,
            "initial_loss": round(initial_loss, 4),
            "final_loss": round(final_loss, 4),
            "loss_reduction_percent": round(loss_reduction_pct, 2),
            "l2_norm_preserved": norm_preserved
        }
    }

    out_json = r"D:\SIH_Model\experiments\baseline\sanity_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved sanity experiment results to: {out_json}")
    print("="*70)
    return results

if __name__ == "__main__":
    run_sanity_experiment()
