"""
Milestone 8: Metric Learning Training & Comparative Evaluation
SIH Problem Statement 26172

Compares Tiny CNN vs Depthwise Separable CNN (DS-CNN) under Supervised Contrastive Loss.
Evaluates cosine separation margin, ROC-AUC, EER, and few-shot prototype retrieval on held-out speakers.
"""

import os
import sys
import json
import time
from datetime import datetime
import numpy as np
import tensorflow as tf

sys.path.insert(0, r"D:\SIH_Model")
from src.features.mfcc import MFCCFeatureExtractor
from src.data.augment import AudioAugmenter
from src.training.batch_generator import MetricLearningBatchGenerator
from src.training.validate import MetricValidator
from src.training.train_metric import MetricTrainer

def run_metric_training_experiment():
    print("="*80)
    print("MILESTONE 8: METRIC LEARNING TRAINING PIPELINE BENCHMARK")
    print("Zero-Speaker-Leakage Speech Embedding Optimization for Edge Voice Activator")
    print("="*80)

    train_manifest = r"D:\SIH_Model\data\metadata\train_manifest.csv"
    val_manifest = r"D:\SIH_Model\data\metadata\validation_manifest.csv"
    checkpoint_dir = r"D:\SIH_Model\models\checkpoints"
    results_dir = r"D:\SIH_Model\experiments\baseline"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Setup Feature Extractor & Augmentation
    extractor = MFCCFeatureExtractor(sample_rate=16000, n_mfcc=13, n_mels=40)
    augmenter = AudioAugmenter(sample_rate=16000, seed=42)

    # 2. Setup Data Generator
    print("\n[Step 1/4] Initializing Training Data Generator...")
    train_gen = MetricLearningBatchGenerator(
        train_manifest,
        feature_extractor=extractor,
        augmenter=augmenter,
        seed=42
    )
    # Preload 50 clips per class to RAM for rapid CPU execution
    train_gen.preload_cache(samples_per_class=50, verbose=True)

    epochs = 12
    steps_per_epoch = 35
    num_classes_per_batch = 16
    samples_per_class = 4

    # 3. Train DS-CNN Encoder
    print("\n" + "#"*80)
    print("[Step 2/4] TRAINING DEPTHWISE SEPARABLE CNN (DS-CNN)")
    print("#"*80)
    ds_trainer = MetricTrainer(
        architecture="ds_cnn",
        embedding_dim=32,
        loss_type="supervised_contrastive",
        temperature=0.10,
        learning_rate=0.001,
        seed=42
    )
    ds_validator = MetricValidator(encoder=ds_trainer.model, val_manifest_path=val_manifest, feature_extractor=extractor)
    ds_results = ds_trainer.train(
        train_generator=train_gen,
        val_validator=ds_validator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        num_classes_per_batch=num_classes_per_batch,
        samples_per_class=samples_per_class,
        checkpoint_dir=checkpoint_dir
    )

    # 4. Train Tiny CNN Encoder
    print("\n" + "#"*80)
    print("[Step 3/4] TRAINING BASELINE TINY CNN")
    print("#"*80)
    tiny_trainer = MetricTrainer(
        architecture="tiny_cnn",
        embedding_dim=32,
        loss_type="supervised_contrastive",
        temperature=0.10,
        learning_rate=0.001,
        seed=42
    )
    tiny_validator = MetricValidator(encoder=tiny_trainer.model, val_manifest_path=val_manifest, feature_extractor=extractor)
    tiny_results = tiny_trainer.train(
        train_generator=train_gen,
        val_validator=tiny_validator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        num_classes_per_batch=num_classes_per_batch,
        samples_per_class=samples_per_class,
        checkpoint_dir=checkpoint_dir
    )

    # 5. Comparative Analysis & Summary
    print("\n" + "="*80)
    print("[Step 4/4] COMPARATIVE ARCHITECTURAL BENCHMARK: DS-CNN VS TINY CNN")
    print("="*80)

    ds_final_val = ds_results["final_validation"]
    tiny_final_val = tiny_results["final_validation"]

    ds_sep = ds_final_val["cosine_separation"]
    tiny_sep = tiny_final_val["cosine_separation"]
    ds_ret = ds_final_val["few_shot_retrieval"]
    tiny_ret = tiny_final_val["few_shot_retrieval"]

    summary_table = {
        "metric": [
            "Total Parameters",
            "FP32 Model Size (KB)",
            "Estimated INT8 Size (KB)",
            "Final Training Loss",
            "Mean Positive Cosine Sim",
            "Mean Negative Cosine Sim",
            "Separation Margin (Delta)",
            "ROC-AUC",
            "Equal Error Rate (EER)",
            "Top-1 Recall (3-shot)",
            "Top-5 Recall (3-shot)"
        ],
        "ds_cnn": [
            f"{ds_results['parameters']:,}",
            f"{(ds_results['parameters'] * 4)/1024:.2f} KB",
            f"{(ds_results['parameters'] * 1)/1024:.2f} KB",
            f"{ds_results['history'][-1]['train_loss']:.4f}",
            f"{ds_sep['mean_positive_cosine']:.4f}",
            f"{ds_sep['mean_negative_cosine']:.4f}",
            f"{ds_sep['separation_margin']:+.4f}",
            f"{ds_sep['roc_auc']:.4f}",
            f"{ds_sep['equal_error_rate']:.4f}",
            f"{ds_ret['top1_recall_accuracy']*100:.2f}%",
            f"{ds_ret['top5_recall_accuracy']*100:.2f}%"
        ],
        "tiny_cnn": [
            f"{tiny_results['parameters']:,}",
            f"{(tiny_results['parameters'] * 4)/1024:.2f} KB",
            f"{(tiny_results['parameters'] * 1)/1024:.2f} KB",
            f"{tiny_results['history'][-1]['train_loss']:.4f}",
            f"{tiny_sep['mean_positive_cosine']:.4f}",
            f"{tiny_sep['mean_negative_cosine']:.4f}",
            f"{tiny_sep['separation_margin']:+.4f}",
            f"{tiny_sep['roc_auc']:.4f}",
            f"{tiny_sep['equal_error_rate']:.4f}",
            f"{tiny_ret['top1_recall_accuracy']*100:.2f}%",
            f"{tiny_ret['top5_recall_accuracy']*100:.2f}%"
        ]
    }

    print(f"{'Metric':<28} | {'DS-CNN (Ours)':<18} | {'Tiny CNN (Baseline)':<20}")
    print("-" * 72)
    for m, d, t in zip(summary_table["metric"], summary_table["ds_cnn"], summary_table["tiny_cnn"]):
        print(f"{m:<28} | {d:<18} | {t:<20}")

    comparison_results = {
        "timestamp": datetime.now().isoformat(),
        "milestone": "Milestone 8: Metric Learning Training Pipeline",
        "training_configuration": {
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "classes_per_batch": num_classes_per_batch,
            "samples_per_class": samples_per_class,
            "total_batch_size": num_classes_per_batch * samples_per_class,
            "loss": "supervised_contrastive",
            "temperature": 0.10,
            "embedding_dim": 32,
            "normalization": "L2_hypersphere"
        },
        "ds_cnn": ds_results,
        "tiny_cnn": tiny_results,
        "summary": summary_table
    }

    out_file = os.path.join(results_dir, "metric_training_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)

    print(f"\nSaved all benchmark metrics and curves to: {out_file}")
    print("="*80)
    return comparison_results

if __name__ == "__main__":
    run_metric_training_experiment()
