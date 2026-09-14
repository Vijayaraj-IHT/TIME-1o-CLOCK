"""
TFLite to C Array Exporter for Embedded Edge Deployment
SIH Problem Statement 26172

Converts a compiled .tflite flatbuffer binary into a C++ header file containing
an aligned byte array suitable for direct inclusion in ESP32-S3 TFLite Micro firmware.
"""

import os
import sys
import argparse
import numpy as np
from typing import Optional, Tuple


def tflite_to_c_header(
    tflite_path: str,
    output_header_path: str,
    array_name: str = "g_voice_activator_model_data",
    include_guard: str = "TFLITE_MICRO_MODEL_H_",
    alignment: int = 16,
    bytes_per_line: int = 12
) -> Tuple[int, int]:
    """
    Converts a .tflite file into a C header with a byte array.
    
    Args:
        tflite_path: Path to input .tflite file.
        output_header_path: Path to output .h file.
        array_name: Name of the C array.
        include_guard: Preprocessor macro for header guard.
        alignment: Memory alignment in bytes (typically 16 for SIMD/ESP-NN).
        bytes_per_line: Formatting parameter for readable C code.
        
    Returns:
        Tuple[int, int]: (model_size_bytes, lines_written)
    """
    if not os.path.exists(tflite_path):
        raise FileNotFoundError(f"TFLite model not found at {tflite_path}")

    with open(tflite_path, "rb") as f:
        data = f.read()

    size_bytes = len(data)
    os.makedirs(os.path.dirname(os.path.abspath(output_header_path)), exist_ok=True)

    header_content = []
    header_content.append("/*")
    header_content.append(" * Auto-generated TFLite Micro Model Byte Array")
    header_content.append(" * SIH Problem Statement 26172 - Voice Activator for Edge Devices")
    header_content.append(f" * Source File: {os.path.basename(tflite_path)}")
    header_content.append(f" * Size: {size_bytes:,} bytes ({size_bytes / 1024.0:.2f} KB)")
    header_content.append(f" * Target: ESP32-S3 (TFLite Micro / ESP-NN)")
    header_content.append(" */")
    header_content.append("")
    header_content.append(f"#ifndef {include_guard}")
    header_content.append(f"#define {include_guard}")
    header_content.append("")
    header_content.append("#include <cstdint>")
    header_content.append("")
    header_content.append(f"// Aligned to {alignment} bytes for optimized ESP32-S3 vector extensions")
    header_content.append(f"alignas({alignment}) const unsigned char {array_name}[] = {{")

    # Format bytes in hex
    lines_of_bytes = []
    for i in range(0, size_bytes, bytes_per_line):
        chunk = data[i:i + bytes_per_line]
        hex_str = ", ".join(f"0x{b:02x}" for b in chunk)
        if i + bytes_per_line < size_bytes:
            hex_str += ","
        lines_of_bytes.append(f"    {hex_str}")

    header_content.extend(lines_of_bytes)
    header_content.append("};")
    header_content.append("")
    header_content.append(f"const unsigned int {array_name}_len = {size_bytes};")
    header_content.append("")
    header_content.append(f"#endif // {include_guard}")
    header_content.append("")

    full_text = "\n".join(header_content)
    with open(output_header_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    return size_bytes, len(header_content)


def estimate_tensor_arena_size(tflite_path: str, safety_margin_pct: float = 25.0) -> dict:
    """
    Estimates the static Tensor Arena SRAM requirement for TFLite Micro on ESP32-S3.
    Calculates maximum activation buffer memory needed during forward inference.
    """
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    tensor_details = interpreter.get_tensor_details()
    total_tensor_bytes = 0

    layer_tensors = []
    for t in tensor_details:
        shape = t["shape"]
        dtype_size = np.dtype(t["dtype"]).itemsize if "dtype" in t else 1
        num_elements = int(np.prod(shape)) if len(shape) > 0 else 1
        tensor_bytes = num_elements * dtype_size
        total_tensor_bytes += tensor_bytes
        layer_tensors.append({
            "name": t["name"],
            "shape": list(shape),
            "bytes": tensor_bytes
        })

    # For sequentially scheduled CNN layers, peak concurrent buffer size is roughly
    # the sum of the two largest adjacent layer activations plus overhead
    sorted_layers = sorted([lt["bytes"] for lt in layer_tensors], reverse=True)
    peak_concurrent_activations = sum(sorted_layers[:3]) if len(sorted_layers) >= 3 else sum(sorted_layers)

    runtime_overhead_bytes = 4096
    base_arena_estimate = peak_concurrent_activations + runtime_overhead_bytes
    recommended_arena_bytes = int(base_arena_estimate * (1.0 + safety_margin_pct / 100.0))

    # Round up to nearest 1024 bytes (1 KB boundary)
    recommended_arena_bytes = ((recommended_arena_bytes + 1023) // 1024) * 1024

    return {
        "model_file": os.path.basename(tflite_path),
        "total_tensors_count": len(tensor_details),
        "peak_concurrent_activations_bytes": peak_concurrent_activations,
        "runtime_overhead_bytes": runtime_overhead_bytes,
        "base_arena_estimate_bytes": base_arena_estimate,
        "recommended_arena_bytes": recommended_arena_bytes,
        "recommended_arena_kb": recommended_arena_bytes / 1024.0,
        "esp32_s3_sram_limit_kb": 256.0,
        "sram_budget_satisfied": bool(recommended_arena_bytes / 1024.0 < 64.0)
    }


def main():
    parser = argparse.ArgumentParser(description="Export TFLite model to C array for ESP32-S3")
    parser.add_argument(
        "--input",
        default=r"D:\SIH_Model\models\tflite\voice_activator_int8.tflite",
        help="Path to .tflite input model"
    )
    parser.add_argument(
        "--output",
        default=r"D:\SIH_Model\src\deployment\esp32\tflite_micro_model.h",
        help="Path to output C header"
    )
    parser.add_argument("--array-name", default="g_voice_activator_model_data", help="C array name")

    args = parser.parse_args()
    print("=" * 80)
    print("TFLITE TO C ARRAY EXPORTER (ESP32-S3)")
    print("=" * 80)
    print(f"Input model:   {args.input}")
    print(f"Output header: {args.output}")

    size_bytes, lines = tflite_to_c_header(args.input, args.output, args.array_name)
    print(f"Successfully exported {size_bytes:,} bytes ({size_bytes / 1024.0:.2f} KB) across {lines:,} lines.")

    # Arena estimation
    try:
        arena_info = estimate_tensor_arena_size(args.input)
        print("-" * 80)
        print("TENSOR ARENA SRAM ESTIMATE:")
        print(f"  Peak Activation Memory: {arena_info['peak_concurrent_activations_bytes']:,} bytes")
        print(f"  Recommended Tensor Arena: {arena_info['recommended_arena_bytes']:,} bytes ({arena_info['recommended_arena_kb']:.2f} KB)")
        print(f"  ESP32-S3 Internal SRAM Budget: 256.0 KB")
        print(f"  SRAM Budget Check (< 64 KB for arena): {'PASSED [x]' if arena_info['sram_budget_satisfied'] else 'FAILED [ ]'}")
        print("=" * 80)
    except Exception as e:
        print(f"Arena estimation warning: {e}")


if __name__ == "__main__":
    main()
