"""
Milestone 9: Unseen-Word Evaluation Benchmark ('ZORA' Few-Shot Test)
SIH Problem Statement 26172

Demonstrates runtime few-shot keyword enrollment without retraining:
1. Enrolls unseen keyword 'ZORA' and held-out human word 'MARVIN'
2. Compares 1, 2, 3, and 5 enrollment shots against 200 impostor clips (negative speech + noise)
3. Quantifies separation margins, ROC-AUC, EER, and variance reduction
4. Exports enrolled prototype to ESP32-S3 C header (src/deployment/esp32/keyword_prototype.h)
"""

import os
import sys
import glob
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, r"D:\SIH_Model")
from src.models.tiny_cnn import build_tiny_cnn_encoder
from src.models.ds_cnn import build_ds_cnn_encoder
from src.features.mfcc import MFCCFeatureExtractor
from src.enrollment.enroll import KeywordEnrollmentManager
from src.evaluation.evaluate_unseen import UnseenKeywordEvaluator

def run_unseen_benchmark():
    print("="*80)
    print("MILESTONE 9: UNSEEN-WORD EVALUATION BENCHMARK ('ZORA' FEW-SHOT TEST)")
    print("Edge Voice Activator Runtime Enrollment Evaluation")
    print("="*80)

    # 1. Setup Models
    print("\n[Step 1/5] Loading Trained Speech Embedding Encoders...")
    tiny_encoder = build_tiny_cnn_encoder(input_shape=(98, 13, 1), embedding_dim=32)
    tiny_weights = r"D:\SIH_Model\models\checkpoints\tiny_cnn_metric_best.weights.h5"
    if os.path.exists(tiny_weights):
        tiny_encoder.load_weights(tiny_weights)
        print(f"  Loaded Tiny CNN weights from: {tiny_weights}")

    ds_encoder = build_ds_cnn_encoder(input_shape=(98, 13, 1), embedding_dim=32)
    ds_weights = r"D:\SIH_Model\models\checkpoints\ds_cnn_metric_best.weights.h5"
    if os.path.exists(ds_weights):
        ds_encoder.load_weights(ds_weights)
        print(f"  Loaded DS-CNN weights from: {ds_weights}")

    extractor = MFCCFeatureExtractor(sample_rate=16000, n_mfcc=13, n_mels=40)

    # 2. Gather Test Audio Corpora
    print("\n[Step 2/5] Preparing Target Keywords and Impostor Audio...")
    # Target 1: Custom Keyword "ZORA"
    zora_files = sorted(glob.glob(r"D:\SIH_Model\data\raw\custom_keywords\zora\*.wav"))
    print(f"  Target Keyword 'ZORA': {len(zora_files)} samples")

    # Target 2: Held-out Unseen Human Word "MARVIN" from Test Manifest
    test_manifest_path = r"D:\SIH_Model\data\metadata\test_manifest.csv"
    test_df = pd.read_csv(test_manifest_path)
    marvin_df = test_df[test_df["label"] == "marvin"]
    marvin_files = marvin_df["filepath"].head(30).tolist()
    print(f"  Held-out Word 'MARVIN' (Human Speakers): {len(marvin_files)} samples")

    # Negative Impostors: 150 diverse human speech words + 50 background noise clips
    impostor_df = test_df[~test_df["label"].isin(["marvin", "zora"])].sample(150, random_state=42)
    impostor_speech_files = impostor_df["filepath"].tolist()

    noise_files = sorted(glob.glob(r"D:\SIH_Model\data\raw\noise\*.wav"))[:50]
    total_impostor_files = impostor_speech_files + noise_files
    print(f"  Negative Impostor Pool: {len(total_impostor_files)} samples (150 diverse human speech + {len(noise_files)} noise)")

    # 3. Evaluate "ZORA" across K in {1, 2, 3, 5} shots
    print("\n" + "#"*80)
    print("[Step 3/5] EVALUATING UNSEEN KEYWORD 'ZORA' ACROSS 1, 2, 3, 5 SHOTS")
    print("#"*80)

    evaluator_tiny = UnseenKeywordEvaluator(tiny_encoder, extractor, seed=42)
    evaluator_ds = UnseenKeywordEvaluator(ds_encoder, extractor, seed=42)

    zora_results_tiny = evaluator_tiny.evaluate_shots_comparison(
        target_files=zora_files,
        impostor_files=total_impostor_files,
        shot_counts=(1, 2, 3, 5),
        num_trials=15,
        keyword_name="ZORA"
    )

    zora_results_ds = evaluator_ds.evaluate_shots_comparison(
        target_files=zora_files,
        impostor_files=total_impostor_files,
        shot_counts=(1, 2, 3, 5),
        num_trials=15,
        keyword_name="ZORA"
    )

    # 4. Evaluate Held-out Human Word "MARVIN" across K in {1, 2, 3, 5} shots
    print("\n" + "#"*80)
    print("[Step 4/5] EVALUATING HELD-OUT HUMAN WORD 'MARVIN' ACROSS 1, 2, 3, 5 SHOTS")
    print("#"*80)

    marvin_results_tiny = evaluator_tiny.evaluate_shots_comparison(
        target_files=marvin_files,
        impostor_files=total_impostor_files,
        shot_counts=(1, 2, 3, 5),
        num_trials=15,
        keyword_name="MARVIN"
    )

    # Print Summary Tables
    print("\n" + "="*80)
    print("FEW-SHOT SHOT-COUNT ANALYSIS FOR 'ZORA' (TINY CNN)")
    print("="*80)
    print(f"{'Shot Count (K)':<15} | {'Margin (Delta)':<15} | {'ROC-AUC':<10} | {'EER':<10} | {'TPR @ 5% FPR':<14} | {'TPR @ 1% FPR':<14}")
    print("-" * 84)
    for shot in ["1_shot", "2_shot", "3_shot", "5_shot"]:
        d = zora_results_tiny["shot_evaluations"][shot]
        print(f"{d['k_shots']:<15} | {d['separation_margin']:<+15.4f} | {d['roc_auc']:<10.4f} | {d['equal_error_rate']:<10.4f} | {d['tpr_at_5pct_fpr']*100:<13.1f}% | {d['tpr_at_1pct_fpr']*100:<13.1f}%")

    print("\n" + "="*80)
    print("FEW-SHOT SHOT-COUNT ANALYSIS FOR 'MARVIN' (HELD-OUT HUMAN SPEECH)")
    print("="*80)
    print(f"{'Shot Count (K)':<15} | {'Margin (Delta)':<15} | {'ROC-AUC':<10} | {'EER':<10} | {'TPR @ 5% FPR':<14} | {'TPR @ 1% FPR':<14}")
    print("-" * 84)
    for shot in ["1_shot", "2_shot", "3_shot", "5_shot"]:
        d = marvin_results_tiny["shot_evaluations"][shot]
        print(f"{d['k_shots']:<15} | {d['separation_margin']:<+15.4f} | {d['roc_auc']:<10.4f} | {d['equal_error_rate']:<10.4f} | {d['tpr_at_5pct_fpr']*100:<13.1f}% | {d['tpr_at_1pct_fpr']*100:<13.1f}%")

    # 5. Export Enrolled Prototype for ESP32-S3 Firmware
    print("\n[Step 5/5] Exporting Enrolled 'ZORA' Prototype Header for ESP32-S3...")
    enroll_manager = KeywordEnrollmentManager(tiny_encoder, extractor)
    # Use 3-shot enrollment (recommended trade-off between user friction and centroid stability)
    enroll_3shot = enroll_manager.enroll(zora_files[:3], keyword_name="ZORA")
    header_path = r"D:\SIH_Model\src\deployment\esp32\keyword_prototype.h"
    enroll_manager.export_prototype_c_header(enroll_3shot["prototype"], "ZORA", header_path)
    print(f"  Enrolled 3-shot 'ZORA' prototype vector:")
    print(f"    - Intra-enrollment Similarity: {enroll_3shot['intra_similarity_mean']:.4f}")
    print(f"    - Exported to: {header_path}")

    # Save complete results
    final_output = {
        "timestamp": datetime.now().isoformat(),
        "milestone": "Milestone 9: Unseen-Word Evaluation (Few-Shot ZORA Test)",
        "datasets": {
            "target_keyword_zora_samples": len(zora_files),
            "target_keyword_marvin_samples": len(marvin_files),
            "negative_impostor_samples": len(total_impostor_files),
            "noise_samples": len(noise_files)
        },
        "zora_evaluation_tiny_cnn": zora_results_tiny,
        "zora_evaluation_ds_cnn": zora_results_ds,
        "marvin_evaluation_tiny_cnn": marvin_results_tiny,
        "enrolled_3shot_zora": {
            "keyword_name": enroll_3shot["keyword_name"],
            "k_shots": enroll_3shot["k_shots"],
            "intra_similarity_mean": enroll_3shot["intra_similarity_mean"],
            "prototype_shape": list(enroll_3shot["prototype"].shape),
            "c_header_path": header_path
        }
    }

    out_json = r"D:\SIH_Model\experiments\unseen_keyword\experiment_unseen_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"\nSaved all unseen evaluation benchmarks to: {out_json}")
    print("="*80)
    return final_output

if __name__ == "__main__":
    run_unseen_benchmark()
