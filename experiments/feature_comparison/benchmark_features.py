"""
Experiment A: MFCC vs. Log-Mel Feature Benchmark & Profiling
SIH Problem Statement 26172
"""

import os
import sys
import time
import json
from datetime import datetime
import numpy as np

sys.path.insert(0, r"D:\SIH_Model")
from src.features.mfcc import MFCCFeatureExtractor
from src.features.logmel import LogMelFeatureExtractor

def run_benchmark(num_iterations=500):
    print("="*70)
    print("EXPERIMENT A: MFCC VS. LOG-MEL FEATURE EXTRACTION BENCHMARK")
    print("="*70)

    sr = 16000
    duration = 1.0
    num_samples = int(sr * duration)
    rng = np.random.default_rng(42)
    # Generate realistic random speech-like audio
    t = np.linspace(0, duration, num_samples, endpoint=False)
    synthetic_speech = (0.4 * np.sin(2 * np.pi * 220 * t) +
                        0.2 * np.sin(2 * np.pi * 880 * t) +
                        0.1 * rng.normal(0, 0.05, num_samples)).astype(np.float32)

    # 1. MFCC Benchmark
    mfcc_extractor = MFCCFeatureExtractor(sample_rate=sr, n_mfcc=13, n_mels=40)
    # Warm-up
    for _ in range(20):
        _ = mfcc_extractor.extract(synthetic_speech)

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        mfcc_out = mfcc_extractor.extract(synthetic_speech)
    mfcc_total_time = time.perf_counter() - start_time
    mfcc_avg_latency_ms = (mfcc_total_time / num_iterations) * 1000.0

    # 2. Log-Mel Benchmark
    logmel_extractor = LogMelFeatureExtractor(sample_rate=sr, n_mels=40)
    # Warm-up
    for _ in range(20):
        _ = logmel_extractor.extract(synthetic_speech)

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        logmel_out = logmel_extractor.extract(synthetic_speech)
    logmel_total_time = time.perf_counter() - start_time
    logmel_avg_latency_ms = (logmel_total_time / num_iterations) * 1000.0

    # Calculate memory and dimensions
    mfcc_shape = list(mfcc_out.shape)
    logmel_shape = list(logmel_out.shape)

    mfcc_bytes = int(np.prod(mfcc_shape) * 4)       # float32 = 4 bytes
    logmel_bytes = int(np.prod(logmel_shape) * 4)

    memory_reduction_pct = ((logmel_bytes - mfcc_bytes) / logmel_bytes) * 100.0

    results = {
        "experiment": "Experiment A: MFCC vs Log-Mel",
        "timestamp": datetime.now().isoformat(),
        "iterations": num_iterations,
        "sample_rate": sr,
        "audio_duration_seconds": duration,
        "mfcc": {
            "output_shape": mfcc_shape,
            "coefficients": 13,
            "memory_bytes": mfcc_bytes,
            "memory_kb": round(mfcc_bytes / 1024.0, 2),
            "latency_ms_per_clip": round(mfcc_avg_latency_ms, 3)
        },
        "logmel": {
            "output_shape": logmel_shape,
            "filterbanks": 40,
            "memory_bytes": logmel_bytes,
            "memory_kb": round(logmel_bytes / 1024.0, 2),
            "latency_ms_per_clip": round(logmel_avg_latency_ms, 3)
        },
        "comparison": {
            "memory_reduction_percent": round(memory_reduction_pct, 2),
            "latency_difference_ms": round(abs(mfcc_avg_latency_ms - logmel_avg_latency_ms), 3),
            "recommended_primary_for_esp32s3": "MFCC (67.5% smaller memory footprint, reduces CNN activation memory in ESP32-S3 SRAM)"
        }
    }

    print(f"\n[1. Output Dimensions]")
    print(f"  - MFCC Shape:    {mfcc_shape} (Frames x Coefficients)")
    print(f"  - Log-Mel Shape: {logmel_shape} (Frames x Filterbanks)")

    print(f"\n[2. Memory Footprint per 1s Audio Clip]")
    print(f"  - MFCC:          {mfcc_bytes:,} bytes ({results['mfcc']['memory_kb']} KB)")
    print(f"  - Log-Mel:       {logmel_bytes:,} bytes ({results['logmel']['memory_kb']} KB)")
    print(f"  - Memory Saving: {results['comparison']['memory_reduction_percent']}% less memory with MFCC")

    print(f"\n[3. Extraction Latency]")
    print(f"  - MFCC Latency:    {results['mfcc']['latency_ms_per_clip']} ms per 1.0s clip")
    print(f"  - Log-Mel Latency: {results['logmel']['latency_ms_per_clip']} ms per 1.0s clip")

    print(f"\n[4. Microcontroller Suitability for ESP32-S3 (SRAM < 256 KB)]")
    print(f"  - Recommendation: {results['comparison']['recommended_primary_for_esp32s3']}")
    print("="*70)

    # Save results to JSON
    out_dir = r"D:\SIH_Model\experiments\feature_comparison"
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "experiment_a_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved Experiment A results to: {out_json}")
    return results

if __name__ == "__main__":
    run_benchmark()
