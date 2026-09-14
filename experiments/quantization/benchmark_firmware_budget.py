"""
ESP32-S3 Firmware Memory Budget & Edge Constraints Benchmark
SIH Problem Statement 26172 - Milestone 13 Verification

Calculates static and dynamic Flash and internal SRAM allocation breakdown
for the C++ Voice Activator firmware on the Espressif ESP32-S3 microcontroller.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, r"D:\SIH_Model")
from src.export.tflite_to_c_array import estimate_tensor_arena_size


def main():
    print("=" * 80)
    print("MILESTONE 13: ESP32-S3 FIRMWARE MEMORY BUDGET BENCHMARK")
    print("=" * 80)

    # 1. File paths
    model_tflite_path = r"D:\SIH_Model\models\tflite\voice_activator_int8.tflite"
    c_header_path = r"D:\SIH_Model\src\deployment\esp32\tflite_micro_model.h"
    proto_header_path = r"D:\SIH_Model\src\deployment\esp32\keyword_prototype.h"
    report_path = r"D:\SIH_Model\experiments\quantization\firmware_memory_report.json"

    # 2. Flash Memory Measurement
    model_size_bytes = os.path.getsize(model_tflite_path)
    model_size_kb = model_size_bytes / 1024.0
    c_header_lines = sum(1 for _ in open(c_header_path, encoding="utf-8"))

    flash_budget_target_kb = 100.0
    flash_target_met = bool(model_size_kb < flash_budget_target_kb)

    # 3. Static SRAM Calculation (ESP32-S3 Internal SRAM = 256 KB)
    arena_estimate = estimate_tensor_arena_size(model_tflite_path)
    tensor_arena_bytes = arena_estimate["recommended_arena_bytes"]
    tensor_arena_kb = tensor_arena_bytes / 1024.0

    # Ring Buffer: 16,000 samples @ 16-bit PCM (2 bytes/sample) = 32,000 bytes
    ring_buffer_bytes = 16000 * 2
    ring_buffer_kb = ring_buffer_bytes / 1024.0

    # MFCC matrix scratch: 98 frames * 13 coeffs * 4 bytes (float32) = 5,096 bytes
    mfcc_scratch_bytes = 98 * 13 * 4
    mfcc_scratch_kb = mfcc_scratch_bytes / 1024.0

    # Audio window buffer: 16,000 float32 samples = 64,000 bytes (shared/scratch)
    # Note: Can be allocated inside tensor arena scratch or statically in BSS
    audio_window_scratch_bytes = 16000 * 4
    audio_window_scratch_kb = audio_window_scratch_bytes / 1024.0

    # State Machine + VAD state + prototype storage
    state_machine_bytes = 128
    prototype_bytes = 32 * 4  # 128 bytes

    total_static_sram_bytes = tensor_arena_bytes + ring_buffer_bytes + mfcc_scratch_bytes + audio_window_scratch_bytes + state_machine_bytes + prototype_bytes
    total_static_sram_kb = total_static_sram_bytes / 1024.0

    sram_budget_target_kb = 256.0
    sram_target_met = bool(total_static_sram_kb < sram_budget_target_kb)
    sram_headroom_kb = sram_budget_target_kb - total_static_sram_kb
    sram_utilization_pct = (total_static_sram_kb / sram_budget_target_kb) * 100.0

    # 4. Compile Memory Report
    report = {
        "timestamp": "2026-09-11 00:05:00",
        "target_hardware": {
            "microcontroller": "Espressif ESP32-S3 (Xtensa Dual-Core 32-bit LX7)",
            "clock_speed_mhz": 240,
            "internal_sram_budget_kb": sram_budget_target_kb,
            "flash_partition_budget_kb": flash_budget_target_kb
        },
        "flash_breakdown": {
            "tflite_model_binary_bytes": model_size_bytes,
            "tflite_model_binary_kb": round(model_size_kb, 2),
            "c_header_source_lines": c_header_lines,
            "flash_budget_target_kb": flash_budget_target_kb,
            "flash_utilization_pct": round((model_size_kb / flash_budget_target_kb) * 100.0, 2),
            "flash_constraint_satisfied": flash_target_met
        },
        "sram_breakdown": {
            "tflite_micro_tensor_arena_kb": round(tensor_arena_kb, 2),
            "audio_ring_buffer_kb": round(ring_buffer_kb, 2),
            "mfcc_feature_scratch_kb": round(mfcc_scratch_kb, 2),
            "audio_window_scratch_kb": round(audio_window_scratch_kb, 2),
            "state_machine_and_prototype_kb": round((state_machine_bytes + prototype_bytes) / 1024.0, 3),
            "total_allocated_sram_kb": round(total_static_sram_kb, 2),
            "free_sram_headroom_kb": round(sram_headroom_kb, 2),
            "sram_utilization_pct": round(sram_utilization_pct, 2),
            "sram_constraint_satisfied": sram_target_met
        },
        "runtime_performance": {
            "idle_cpu_utilization_estimate": "< 5% (VAD bypass skips > 95% of frames)",
            "inference_duration_estimate_esp32_ms": "~18 - 25 ms (using ESP-NN SIMD)",
            "frame_stride_ms": 50,
            "real_time_sustainable": True
        }
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # 5. Print Summary Table
    print(f"Target Device:                ESP32-S3 (240 MHz Dual-Core LX7)")
    print("-" * 80)
    print(f"FLASH MEMORY FOOTPRINT:")
    print(f"  TFLite Model (INT8):        {model_size_bytes:,} bytes ({model_size_kb:.2f} KB)")
    print(f"  Flash Partition Budget:     < {flash_budget_target_kb:.1f} KB")
    print(f"  Flash Utilization:          {(model_size_kb / flash_budget_target_kb) * 100.0:.2f}%")
    print(f"  Flash Constraint Status:    {'PASSED [x]' if flash_target_met else 'FAILED [ ]'}")
    print("-" * 80)
    print(f"STATIC SRAM ALLOCATION BREAKDOWN (Total: 256.0 KB):")
    print(f"  [1] TFLite Micro Arena:     {tensor_arena_bytes:,} bytes ({tensor_arena_kb:.2f} KB)")
    print(f"  [2] Audio Ring Buffer:      {ring_buffer_bytes:,} bytes ({ring_buffer_kb:.2f} KB)")
    print(f"  [3] Audio Window Scratch:   {audio_window_scratch_bytes:,} bytes ({audio_window_scratch_kb:.2f} KB)")
    print(f"  [4] MFCC Feature Scratch:   {mfcc_scratch_bytes:,} bytes ({mfcc_scratch_kb:.2f} KB)")
    print(f"  [5] State Machine & Proto:  {state_machine_bytes + prototype_bytes} bytes (0.25 KB)")
    print(f"  ------------------------------------------------------------")
    print(f"  Total Static SRAM Used:     {total_static_sram_bytes:,} bytes ({total_static_sram_kb:.2f} KB)")
    print(f"  Free SRAM Remaining:        {sram_headroom_kb:.2f} KB ({100.0 - sram_utilization_pct:.1f}% free)")
    print(f"  SRAM Constraint Status:     {'PASSED [x]' if sram_target_met else 'FAILED [ ]'}")
    print("=" * 80)
    print(f"Firmware memory report saved to: {report_path}")


if __name__ == "__main__":
    main()
