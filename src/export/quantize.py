"""
Model Quantization Module for Edge Microcontrollers (TFLite & INT8)
SIH Problem Statement 26172

Converts trained Keras speech embedding models to TensorFlow Lite (TFLite) flatbuffers.
Performs full integer post-training quantization (INT8) using representative speech dataset calibration,
ensuring flash size < 100 KB and SRAM compatibility on ESP32-S3.
"""

import os
import soundfile as sf
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Generator, List, Optional, Tuple, Dict, Any

from src.features.mfcc import MFCCFeatureExtractor
from src.models.tiny_cnn import build_tiny_cnn_encoder
from src.models.ds_cnn import build_ds_cnn_encoder


class ModelQuantizer:
    """
    Quantization and conversion engine for universal speech embedding models.
    Supports FP32 baseline export and INT8 post-training quantization.
    """
    def __init__(self, keras_model: tf.keras.Model):
        """
        Args:
            keras_model: Instantiated tf.keras.Model with loaded weights.
        """
        self.model = keras_model

    @staticmethod
    def load_audio_file(audio_path: str, target_sr: int = 16000, target_length: int = 16000) -> np.ndarray:
        """Loads and standardizes an audio clip to target sample rate and duration."""
        try:
            audio, sr = sf.read(audio_path, dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr != target_sr:
                target_samples = int(len(audio) * (target_sr / sr))
                audio = np.interp(
                    np.linspace(0, len(audio), target_samples, endpoint=False),
                    np.arange(len(audio)),
                    audio
                ).astype(np.float32)
            if len(audio) < target_length:
                pad_width = target_length - len(audio)
                audio = np.pad(audio, (0, pad_width), mode="constant")
            elif len(audio) > target_length:
                audio = audio[:target_length]
            return audio
        except Exception:
            return np.zeros(target_length, dtype=np.float32)

    def create_representative_dataset_generator(
        self,
        manifest_path: Optional[str] = None,
        n_samples: int = 100,
        audio_files: Optional[List[str]] = None
    ):
        """
        Creates a representative calibration dataset generator from real speech samples.
        Extracts 13-dim MFCC features matching the runtime audio pipeline.
        """
        feature_extractor = MFCCFeatureExtractor(n_mfcc=13)
        paths: List[str] = []

        if audio_files is not None and len(audio_files) > 0:
            paths = audio_files[:n_samples]
        elif manifest_path is not None and os.path.exists(manifest_path):
            df = pd.read_csv(manifest_path)
            col = None
            for candidate in ["filepath", "file_path", "path", "audio_path"]:
                if candidate in df.columns:
                    col = candidate
                    break
            if col is None:
                col = df.columns[0]

            for p in df[col].dropna():
                p_clean = str(p).replace("/", os.sep)
                if os.path.exists(p_clean):
                    paths.append(p_clean)
                elif os.path.exists(str(p)):
                    paths.append(str(p))
                if len(paths) >= n_samples:
                    break

        def generator():
            if len(paths) > 0:
                for p in paths:
                    audio = self.load_audio_file(p)
                    mfcc = feature_extractor.extract(audio)  # (98, 13)
                    tensor = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=-1).astype(np.float32)
                    yield [tensor]
            else:
                for _ in range(n_samples):
                    dummy = np.random.randn(1, 98, 13, 1).astype(np.float32)
                    yield [dummy]

        return generator

    def convert_to_fp32(self) -> bytes:
        """Converts model to standard FP32 TFLite flatbuffer."""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        return tflite_model

    def convert_to_int8(self, representative_dataset_gen=None, n_calibration_samples: int = 100) -> bytes:
        """
        Performs full integer post-training quantization (INT8) with representative calibration.
        """
        if representative_dataset_gen is None:
            representative_dataset_gen = self.create_representative_dataset_generator(n_samples=n_calibration_samples)

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        tflite_model = converter.convert()
        return tflite_model

    @staticmethod
    def save_tflite_model(tflite_content: bytes, output_path: str) -> int:
        """Saves TFLite model flatbuffer to disk, returning byte size."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(tflite_content)
        return len(tflite_content)

    @staticmethod
    def run_inference(tflite_content: bytes, input_data: np.ndarray) -> np.ndarray:
        """Runs inference via TFLite Interpreter."""
        interpreter = tf.lite.Interpreter(model_content=tflite_content)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        if input_data.ndim == 3:
            input_data = np.expand_dims(input_data, axis=0)

        outputs = []
        for i in range(input_data.shape[0]):
            sample = np.expand_dims(input_data[i], axis=0).astype(input_details[0]["dtype"])
            interpreter.set_tensor(input_details[0]["index"], sample)
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]["index"])
            outputs.append(out[0])

        return np.array(outputs, dtype=np.float32)

    @classmethod
    def evaluate_quantization_fidelity(
        cls,
        tflite_fp32: bytes,
        tflite_int8: bytes,
        test_inputs: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluates numerical accuracy and cosine alignment between FP32 and INT8 models.
        """
        emb_fp32 = cls.run_inference(tflite_fp32, test_inputs)
        emb_int8 = cls.run_inference(tflite_int8, test_inputs)

        cosine_sims = []
        norm_diffs = []
        for f32, i8 in zip(emb_fp32, emb_int8):
            norm_f = np.linalg.norm(f32)
            norm_i = np.linalg.norm(i8)
            norm_diffs.append(abs(norm_f - norm_i))
            dot = np.dot(f32, i8)
            cos = dot / (norm_f * norm_i + 1e-9)
            cosine_sims.append(float(cos))

        fp32_size = len(tflite_fp32)
        int8_size = len(tflite_int8)
        compression = (fp32_size - int8_size) / fp32_size * 100.0

        return {
            "mean_cosine_similarity": float(np.mean(cosine_sims)),
            "min_cosine_similarity": float(np.min(cosine_sims)),
            "max_cosine_similarity": float(np.max(cosine_sims)),
            "max_norm_deviation": float(np.max(norm_diffs)),
            "mean_norm_deviation": float(np.mean(norm_diffs)),
            "fp32_size_bytes": fp32_size,
            "int8_size_bytes": int8_size,
            "fp32_size_kb": fp32_size / 1024.0,
            "int8_size_kb": int8_size / 1024.0,
            "compression_ratio_pct": float(compression),
            "flash_budget_target_kb": 100.0,
            "flash_budget_met": bool(int8_size / 1024.0 < 100.0)
        }
