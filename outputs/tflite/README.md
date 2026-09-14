# SIH 26172 Voice Activator - TFLite Models

Prepared for Edge & ESP32-S3 Microcontroller Deployment.

## Available Models

1. **`voice_activator_int8.tflite`** (`56.41 KB`)
   - **Type**: Universal Metric Embedding Encoder.
   - **Input**: `(1, 98, 13, 1)` float32 (Quantized INT8 operators inside).
   - **Output**: `(1, 32)` float32 L2-normalized embedding.
   - **Tensor Arena**: `64.5 KB` SRAM required.
   - **Header File**: `tflite_micro_model.h` (16-byte aligned `alignas(16)` C array).
   - **Advantage**: Zero-retraining few-shot adaptation for ANY custom keyword enrolled by the user.

2. **`voice_activator_aswin_end_to_end_int8.tflite`** (`57.83 KB`)
   - **Type**: Dedicated Single-Keyword End-to-End Detector for **"ASWIN"**.
   - **Input**: `(1, 98, 13, 1)` float32 MFCC spectrogram.
   - **Output**: `(1, 1)` float32 scalar cosine similarity score directly.
   - **Header File**: `tflite_micro_model_aswin_e2e.h`.
   - **Advantage**: Edge firmware does not need vector math; model outputs keyword similarity directly.

3. **`voice_activator_ds_cnn_int8.tflite`** (`32.50 KB`)
   - **Type**: Depthwise-Separable CNN Encoder.
   - **Size**: Ultra-lean sub-35KB Flash footprint.

## Verification
Run tests anytime using pytest:
```bash
pytest tests/test_esp32_deployment.py -v
```
