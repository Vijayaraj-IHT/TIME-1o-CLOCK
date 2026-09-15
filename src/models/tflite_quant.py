"""
Quantization-aware TFLite I/O helpers (Phase-1 Change 3c).

Problem: every Python inference site fed the model with
``tensor.astype(input_details[0]["dtype"])`` and read embeddings back raw.
That is correct ONLY for float32-I/O (dynamic-range) models. For a
full-integer model it silently feeds garbage (float MFCC values truncated to
int8 with no scale/zero-point) and returns un-dequantized int8 embeddings.

These helpers mirror src/deployment/esp32/main.cpp (input quant + output
dequant) exactly, driven by the model's own ``quantization`` params, so
Python and firmware can never disagree again:

    int8 in : q = clip(round(x / scale) + zero_point)   [fw: main.cpp]
    int8 out: x = (q - zero_point) * scale              [fw: main.cpp]

For float32-I/O models (scale == 0 -- all three current .tflite files, verified
Sep-2026) both helpers are exact pass-throughs: wiring them in changes nothing
today and fixes everything the day a full-integer model is deployed.
"""
import numpy as np


def _is_quantized(scale) -> bool:
    try:
        return float(scale) != 0.0
    except (TypeError, ValueError):
        return False


def quantize_input(x: np.ndarray, scale: float, zero_point: int, dtype) -> np.ndarray:
    """Quantize a float32 model input to the tensor's dtype.

    Args:
        x: float32 input batch, e.g. (1, 98, 13, 1) MFCC.
        scale, zero_point: from ``input_details["quantization"]``.
        dtype: from ``input_details["dtype"]`` (np dtype or type).

    Pass-through (values AND dtype) when scale == 0 (float model).
    """
    dt = np.dtype(dtype)
    xa = np.asarray(x, dtype=np.float32)
    if dt.kind == "f" or not _is_quantized(scale):
        return xa.astype(dt, copy=False)
    if dt.kind not in ("i", "u"):
        raise ValueError(f"quantize_input: unsupported tensor dtype {dt}")
    info = np.iinfo(dt)
    q = np.round(xa / float(scale)) + int(zero_point)
    return np.clip(q, info.min, info.max).astype(dt)


def dequantize_output(q: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Dequantize a model output tensor to float32.

    Pass-through (float32 copy) when scale == 0 (float model).
    """
    if not _is_quantized(scale):
        return np.asarray(q, dtype=np.float32)
    return (np.asarray(q, dtype=np.float32) - float(zero_point)) * float(scale)


def io_quant_params(details: dict):
    """Split a get_input/output_details() entry into (index, dtype, scale, zp)."""
    quant = details.get("quantization", (0.0, 0)) or (0.0, 0)
    scale, zp = float(quant[0]), int(quant[1])
    return details["index"], details["dtype"], scale, zp
