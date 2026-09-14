"""
Milestone 11: False Activation Rate (FAR) & Robustness Benchmark
SIH Problem Statement 26172

Comprehensive edge stress testing:
1. Continuous False Activation Rate (FAR/h) over negative speech and ambient noise
2. SNR Degradation Curve: Clean, 20 dB, 10 dB, 0 dB SNR
3. Phonetic Confuser Word Rejection: 'zero', 'four', 'no', 'go'
4. Operating Threshold Calibration and Freezing
"""

import os
import sys
import glob
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, r"D:\SIH_Model")
from src.models.tiny_cnn import build_tiny_cnn_encoder
from src.enrollment.enroll import KeywordEnrollmentManager
from src.streaming.detector import StreamingVoiceActivator
from src.streaming.state_machine import DetectionStateMachine
from src.evaluation.evaluate_robustness import RobustnessEvaluator

def run_robustness_benchmark():
    print("="*80)
    print("MILESTONE 11: FALSE ACTIVATION TESTING & ROBUSTNESS BENCHMARK")
    print("Edge Voice Activator Stress Testing under Noise, Speech, and Confusers")
    print("="*80)

    # 1. Load Model & Enrolled 3-Shot 'ZORA' Prototype
    print("\n[Step 1/5] Initializing Streaming Voice Activator...")
    encoder = build_tiny_cnn_encoder(input_shape=(98, 13, 1), embedding_dim=32)
    weights_path = r"D:\SIH_Model\models\checkpoints\tiny_cnn_metric_best.weights.h5"
    if os.path.exists(weights_path):
        encoder.load_weights(weights_path)
        print(f"  Loaded model weights from: {weights_path}")

    enroll_manager = KeywordEnrollmentManager(encoder)
    zora_shots = [
        r"D:\SIH_Model\data\raw\custom_keywords\zora\zora_david_rate+0_var0.wav",
        r"D:\SIH_Model\data\raw\custom_keywords\zora\zora_david_rate+1_var0.wav",
        r"D:\SIH_Model\data\raw\custom_keywords\zora\zora_zira_rate+0_var0.wav"
    ]
    zora_proto = enroll_manager.enroll(zora_shots, keyword_name="ZORA")["prototype"]

    state_machine = DetectionStateMachine(threshold=0.78, hysteresis=0.05, consecutive_windows=3, cooldown_ms=1500)
    activator = StreamingVoiceActivator(encoder, target_prototype=zora_proto, state_machine=state_machine)
    evaluator = RobustnessEvaluator(activator)

    # 2. Gather Test Audio Corpora
    print("\n[Step 2/5] Preparing Evaluation Audio Corpora...")
    test_manifest_path = r"D:\SIH_Model\data\metadata\test_manifest.csv"
    test_df = pd.read_csv(test_manifest_path)

    # Negative speech pool: 250 diverse human speech words (excluding zora)
    neg_speech_df = test_df[test_df["label"] != "zora"].sample(250, random_state=42)
    neg_speech_files = neg_speech_df["filepath"].tolist()

    # Ambient noise pool: 50 noise files
    noise_files = sorted(glob.glob(r"D:\SIH_Model\data\raw\noise\*.wav"))[:50]
    full_negative_pool = neg_speech_files + noise_files
    print(f"  Negative Stream Pool: {len(full_negative_pool)} files ({len(neg_speech_files)} human speech + {len(noise_files)} noise)")

    # Target 'ZORA' files: 30 files
    zora_files = sorted(glob.glob(r"D:\SIH_Model\data\raw\custom_keywords\zora\*.wav"))
    print(f"  Target 'ZORA' Audio Samples: {len(zora_files)} files")

    # Confuser words: 'zero', 'four', 'no', 'go' (20 samples each)
    confusers = {}
    for c_word in ["zero", "four", "no", "go"]:
        c_sub = test_df[test_df["label"] == c_word]
        if len(c_sub) > 0:
            confusers[c_word] = c_sub["filepath"].head(20).tolist()
    print(f"  Phonetic Confusers: {list(confusers.keys())} ({sum(len(v) for v in confusers.values())} samples)")

    # 3. Continuous False Activation Rate (FAR) Evaluation
    print("\n" + "#"*80)
    print("[Step 3/5] RUNNING CONTINUOUS FALSE ACTIVATION RATE (FAR) STREAM...")
    print("#"*80)
    far_results = evaluator.evaluate_false_activation_rate(full_negative_pool, chunk_size_ms=50)

    print(f"  Total Negative Stream Duration: {far_results['total_stream_seconds']:.1f} s ({far_results['total_stream_hours']:.3f} hours)")
    print(f"  VAD Inference Skip Rate:        {far_results['vad_inference_skip_percentage']:.2f}% (Bypassed during silence/noise)")
    print(f"  False Activation Events:        {far_results['false_activations_count']}")
    print(f"  False Activation Rate (FAR/h):  {far_results['false_activation_rate_per_hour']:.2f} FA / hour")

    # 4. SNR Noise Degradation Curve
    print("\n" + "#"*80)
    print("[Step 4/5] EVALUATING SNR ROBUSTNESS (Clean, 20 dB, 10 dB, 0 dB SNR)...")
    print("#"*80)
    snr_results = evaluator.evaluate_snr_robustness(
        target_files=zora_files,
        noise_files=noise_files,
        snr_levels=(None, 20, 10, 0),
        chunk_size_ms=50
    )

    print(f"{'Condition':<15} | {'SNR Level':<12} | {'Tested':<8} | {'Detections':<12} | {'TPR (%)':<10} | {'Mean Cosine Sim':<15}")
    print("-" * 82)
    for tag, d in snr_results.items():
        print(f"{tag:<15} | {str(d['snr_db']):<12} | {d['total_samples']:<8} | {d['successful_detections']:<12} | {d['tpr_percentage']:<9.1f}% | {d['mean_detection_similarity']:<15.4f}")

    # 5. Confuser Word Rejection & Threshold Freezing
    print("\n" + "#"*80)
    print("[Step 5/5] TESTING PHONETIC CONFUSER REJECTION & THRESHOLD CALIBRATION...")
    print("#"*80)
    confuser_results = evaluator.evaluate_confuser_words(confusers, chunk_size_ms=50)

    print(f"{'Confuser Word':<15} | {'Tested':<8} | {'False Triggers':<15} | {'Rejection Rate':<16} | {'Peak Cosine Sim':<16}")
    print("-" * 76)
    for c_word, c_data in confuser_results.items():
        print(f"{c_word:<15} | {c_data['total_tested']:<8} | {c_data['false_triggers']:<15} | {c_data['rejection_rate_pct']:<15.1f}% | {c_data['peak_similarity_max']:<16.4f}")

    # Threshold Sensitivity Sweep
    print("\nOperating Threshold Sensitivity Analysis (tau sweep):")
    threshold_sweep = []
    for test_thresh in [0.70, 0.74, 0.78, 0.82, 0.86]:
        sm_temp = DetectionStateMachine(threshold=test_thresh, hysteresis=0.05, consecutive_windows=3)
        act_temp = StreamingVoiceActivator(encoder, target_prototype=zora_proto, state_machine=sm_temp)
        eval_temp = RobustnessEvaluator(act_temp)

        # Quick test on 50 negative clips + 10 targets
        f_res = eval_temp.evaluate_false_activation_rate(full_negative_pool[:60])
        s_res = eval_temp.evaluate_snr_robustness(target_files=zora_files[:10], noise_files=noise_files, snr_levels=[20])

        fa_cnt = f_res["false_activations_count"]
        tpr = s_res["20dB_snr"]["tpr_percentage"]
        threshold_sweep.append({
            "threshold": test_thresh,
            "hysteresis_low": round(test_thresh - 0.05, 2),
            "false_alarms": fa_cnt,
            "tpr_at_20db_pct": tpr
        })
        print(f"  tau={test_thresh:.2f} (tau_low={test_thresh-0.05:.2f}) -> False Alarms: {fa_cnt}, TPR @ 20dB: {tpr:.1f}%")

    final_results = {
        "timestamp": datetime.now().isoformat(),
        "milestone": "Milestone 11: False Activation Testing & Robustness Benchmark",
        "calibrated_operating_parameters": {
            "operating_threshold": 0.78,
            "hysteresis_deadband": 0.05,
            "threshold_low": 0.73,
            "consecutive_confirmation_windows": 3,
            "cooldown_lockout_ms": 1500,
            "status": "FROZEN_FOR_FIRMWARE"
        },
        "false_activation_evaluation": far_results,
        "snr_robustness_curve": snr_results,
        "confuser_rejection": confuser_results,
        "threshold_sensitivity_sweep": threshold_sweep
    }

    out_json = r"D:\SIH_Model\experiments\threshold\robustness_benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nSaved full robustness benchmark results to: {out_json}")
    print("="*80)
    return final_results

if __name__ == "__main__":
    run_robustness_benchmark()
