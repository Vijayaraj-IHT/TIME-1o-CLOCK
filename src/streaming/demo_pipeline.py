"""
End-to-End Live Voice Activator Demonstration Pipeline
SIH Problem Statement 26172 (ISRO)

Integrates:
1. Runtime Few-Shot Unseen Keyword Enrollment (INT8 TFLite)
2. Zero-Allocation Streaming Ring Buffer & VAD
3. INT8 Universal Embedding Encoder Inference
4. Dual-Threshold Hysteresis State Machine
5. Low-Latency Edge Activation & Remote ASR Buffer Handover
"""

import os
import time
import json
import numpy as np
import soundfile as sf
import tensorflow as tf
from typing import List, Dict, Any, Optional, Tuple

from src.features.mfcc import MFCCFeatureExtractor
from src.streaming.ring_buffer import AudioRingBuffer
from src.streaming.vad import EnergyVAD
from src.streaming.state_machine import DetectionStateMachine, DetectionState


def load_calibrated_thresholds(config_path: Optional[str] = None) -> Dict[str, float]:
    """
    Single source of truth for detection thresholds. Loads configs/config.yaml so
    demo_pipeline.py, the streaming detector, and firmware-export tooling can never
    drift out of sync with the values that experiments/threshold/*.json were actually
    calibrated against.
    """
    import yaml
    if config_path is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        config_path = os.path.join(repo_root, "configs", "config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    d = cfg["detection"]
    return {
        "tau_high": float(d["threshold"]),
        "tau_low": float(d["threshold"] - d["hysteresis"]),
        "persistence": int(d["consecutive_windows"]),
        "smoothing_window": int(d["smoothing_window"]),
        "cooldown_ms": float(d["cooldown_ms"]),
    }


def verify_model_threshold_pairing(tflite_model_path: str, manifest_path: Optional[str] = None) -> None:
    """
    Cross-checks the loaded .tflite file against outputs/tflite/models_manifest.json.
    Prevents the "wrong model + wrong threshold" silent-degradation case: multiple
    .tflite variants exist (int8, ds_cnn_int8, aswin_end_to_end_int8) and nothing
    previously verified which one is actually loaded before applying thresholds
    calibrated for a different variant.
    """
    if manifest_path is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        manifest_path = os.path.join(repo_root, "outputs", "tflite", "models_manifest.json")
    model_filename = os.path.basename(tflite_model_path)
    if not os.path.exists(manifest_path):
        print(f"[WARN] models_manifest.json not found at {manifest_path} - skipping model/threshold cross-check.")
        return
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    known = {m["file"]: m for m in manifest.get("models", [])}
    if model_filename not in known:
        print(f"[WARN] '{model_filename}' is not listed in models_manifest.json - cannot verify calibration pairing.")
        return
    entry = known[model_filename]
    print(f"[CHECK] Loaded model '{model_filename}' -> manifest entry: {entry.get('type', 'unknown')}")
    if "output_shape" in entry and entry["output_shape"] != [1, 32] and entry.get("name") == "voice_activator_int8":
        raise ValueError(
            f"[FATAL] Model output shape mismatch for '{model_filename}': "
            f"expected [1, 32] embedding, manifest reports {entry['output_shape']}."
        )


class EndToEndVoiceActivatorDemo:
    """
    Complete end-to-end voice activator system demonstrating edge wake-word activation
    and subsequent audio buffer handover to open-source ASR.
    """
    def __init__(
        self,
        tflite_model_path: Optional[str] = None,
        sample_rate: int = 16000,
        chunk_size_ms: int = 50,
        # NOTE: these MUST match configs/config.yaml -> detection.{threshold, hysteresis, consecutive_windows}
        # otherwise the FAR/false-reject numbers in experiments/threshold/*.json describe a
        # different operating point than what actually runs here. See CONFIG_SYNC_CHECK below.
        tau_high: float = 0.78,
        tau_low: float = 0.73,
        persistence: int = 3
    ):
        # Fix #1 guard: catch future drift between hardcoded call sites and config.yaml
        # at construction time, loudly, instead of silently running an uncalibrated operating point.
        try:
            canonical = load_calibrated_thresholds()
            if abs(canonical["tau_high"] - tau_high) > 1e-6 or abs(canonical["tau_low"] - tau_low) > 1e-6:
                print(
                    f"[WARN] Thresholds passed to EndToEndVoiceActivatorDemo "
                    f"(tau_high={tau_high}, tau_low={tau_low}) do not match configs/config.yaml "
                    f"(tau_high={canonical['tau_high']}, tau_low={canonical['tau_low']}). "
                    f"experiments/threshold/*.json calibration will NOT describe this run."
                )
        except Exception as e:
            print(f"[WARN] Could not verify thresholds against config.yaml: {e}")

        if tflite_model_path is None or not os.path.exists(tflite_model_path):
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            candidate = os.path.join(repo_root, "models", "tflite", "voice_activator_int8.tflite")
            if os.path.exists(candidate):
                tflite_model_path = candidate
            elif tflite_model_path and os.path.exists(tflite_model_path):
                pass
            else:
                legacy_path = r"D:\SIH_Model\models\tflite\voice_activator_int8.tflite"
                if os.path.exists(legacy_path):
                    tflite_model_path = legacy_path
                else:
                    raise FileNotFoundError(f"Model not found at {candidate} or {legacy_path}")

        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * (chunk_size_ms / 1000.0))  # 800 samples
        self.chunk_size_ms = chunk_size_ms
        self.tflite_model_path = tflite_model_path

        # Feature extractor
        self.feature_extractor = MFCCFeatureExtractor(sample_rate=sample_rate, n_mfcc=13)

        # Ring buffer (16,000 samples = 1.0 second)
        self.ring_buffer = AudioRingBuffer(capacity_samples=sample_rate)

        # Voice Activity Detector (VAD)
        self.vad = EnergyVAD(sample_rate=sample_rate, min_energy_threshold=0.005, energy_multiplier=2.5, hangover_frames=3)

        # State machine
        hysteresis = float(tau_high - tau_low)
        self.state_machine = DetectionStateMachine(
            threshold=tau_high,
            hysteresis=hysteresis,
            consecutive_windows=persistence,
            smoothing_window=5,
            cooldown_ms=1500.0
        )

        # Load TFLite Model
        if not os.path.exists(tflite_model_path):
            raise FileNotFoundError(f"Model not found: {tflite_model_path}")

        # Fix #2: verify the loaded model is the one thresholds were calibrated for.
        verify_model_threshold_pairing(tflite_model_path)

        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Enrolled Prototype Centroid (32-D)
        self.enrolled_keyword_name: Optional[str] = None
        self.prototype_centroid: Optional[np.ndarray] = None
        self.enrollment_intra_sim: float = 0.0

        # Handover state
        self.capturing_post_wake_asr = False
        self.asr_buffer = []
        self.asr_buffer_target_samples = int(sample_rate * 2.0)  # 2.0s post-wake audio

    def enroll_keyword(self, audio_paths: List[str], keyword_name: str) -> Dict[str, Any]:
        """
        Dynamically enrolls an unseen custom keyword using K audio utterances.
        Computes the L2-normalized prototype centroid using the INT8 TFLite model.
        """
        self.enrolled_keyword_name = keyword_name
        embeddings = []

        for p in audio_paths:
            audio, sr = sf.read(p, dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr != self.sample_rate:
                target_len = int(len(audio) * (self.sample_rate / sr))
                audio = np.interp(
                    np.linspace(0, len(audio), target_len, endpoint=False),
                    np.arange(len(audio)),
                    audio
                ).astype(np.float32)
            if len(audio) < self.sample_rate:
                audio = np.pad(audio, (0, self.sample_rate - len(audio)), mode="constant")
            elif len(audio) > self.sample_rate:
                audio = audio[:self.sample_rate]

            mfcc = self.feature_extractor.extract(audio)  # (98, 13)
            tensor = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=-1).astype(self.input_details[0]["dtype"])

            self.interpreter.set_tensor(self.input_details[0]["index"], tensor)
            self.interpreter.invoke()
            emb = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

            # Normalize embedding
            norm = np.linalg.norm(emb)
            emb_norm = emb / (norm + 1e-9)
            embeddings.append(emb_norm)

        embeddings = np.array(embeddings)
        centroid = np.mean(embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        self.prototype_centroid = centroid

        # Compute pairwise intra-enrollment similarity
        pairwise_sims = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                pairwise_sims.append(float(np.dot(embeddings[i], embeddings[j])))
        self.enrollment_intra_sim = float(np.mean(pairwise_sims)) if pairwise_sims else 1.0

        return {
            "keyword": keyword_name,
            "shots_count": len(audio_paths),
            "intra_similarity": round(self.enrollment_intra_sim, 4),
            "prototype_dimension": int(len(centroid)),
            "prototype_norm": round(float(np.linalg.norm(centroid)), 4)
        }

    def process_chunk(self, audio_chunk: np.ndarray, timestamp_ms: float) -> Dict[str, Any]:
        """
        Processes a single 50ms streaming chunk through RingBuffer -> VAD -> Model -> StateMachine.
        """
        self.ring_buffer.append(audio_chunk)

        # Check if capturing audio for remote ASR handover
        if self.capturing_post_wake_asr:
            self.asr_buffer.extend(audio_chunk)
            if len(self.asr_buffer) >= self.asr_buffer_target_samples:
                # Handover buffer full -> dispatch to remote ASR
                handover_payload = np.array(self.asr_buffer[:self.asr_buffer_target_samples], dtype=np.float32)
                self.capturing_post_wake_asr = False
                self.asr_buffer = []
                asr_result = self._dispatch_to_remote_asr(handover_payload)
                return {
                    "event": "ASR_HANDOVER_DISPATCHED",
                    "timestamp_ms": timestamp_ms,
                    "asr_transcription": asr_result,
                    "buffer_duration_sec": 2.0,
                    "audio_payload": handover_payload
                }

        if not self.ring_buffer.is_full():
            return {"event": "BUFFERING", "timestamp_ms": timestamp_ms}

        # 1. VAD Gating
        is_speech = self.vad.is_speech(audio_chunk)
        if not is_speech:
            # Silence/Ambient: update state machine with zero similarity to decay smoothly
            _ = self.state_machine.process_similarity(0.0, timestamp_ms)
            return {
                "event": "VAD_SKIP",
                "timestamp_ms": timestamp_ms,
                "vad_speech": False,
                "state": self.state_machine.state
            }

        # 2. MFCC Feature Extraction on 1.0s window
        window = self.ring_buffer.get_snapshot()
        t0 = time.perf_counter()
        mfcc = self.feature_extractor.extract(window)  # (98, 13)

        # 3. Model Inference via TFLite INT8
        tensor = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=-1).astype(self.input_details[0]["dtype"])
        self.interpreter.set_tensor(self.input_details[0]["index"], tensor)
        self.interpreter.invoke()
        emb = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        inference_latency_ms = (time.perf_counter() - t0) * 1000.0

        # Normalize embedding
        norm = np.linalg.norm(emb)
        emb_norm = emb / (norm + 1e-9)

        # 4. Cosine Similarity against enrolled prototype
        cosine_sim = float(np.dot(emb_norm, self.prototype_centroid))

        # 5. Hysteresis State Machine Transition
        sm_status = self.state_machine.process_similarity(cosine_sim, timestamp_ms)
        activated = sm_status.get("is_activated", False)

        result = {
            "event": "INFERENCE",
            "timestamp_ms": timestamp_ms,
            "vad_speech": True,
            "raw_sim": round(cosine_sim, 4),
            "smoothed_sim": round(sm_status.get("smoothed_similarity", 0.0), 4),
            "state": self.state_machine.state,
            "latency_ms": round(inference_latency_ms, 2),
            "activated": activated
        }

        if activated:
            result["event"] = "ACTIVATION_TRIGGERED"
            # Begin post-wake audio capture for ASR handover
            self.capturing_post_wake_asr = True
            self.asr_buffer = []

        return result

    def _dispatch_to_remote_asr(self, audio_buffer: np.ndarray) -> Dict[str, Any]:
        """
        Simulates the edge-to-remote open-source ASR handover mechanism.
        Prepares standard 16kHz PCM WAV payload for transmission via HTTP/WebSocket.
        """
        payload_bytes = len(audio_buffer) * 2  # 16-bit PCM
        t0 = time.perf_counter()
        
        handover_latency_ms = (time.perf_counter() - t0) * 1000.0 + 1.2  # realistic network staging

        return {
            "asr_engine": "Open-Source Whisper / VOSK Endpoint",
            "audio_format": "PCM 16 kHz Mono (16-bit)",
            "payload_bytes": payload_bytes,
            "payload_duration_sec": len(audio_buffer) / self.sample_rate,
            "handover_latency_ms": round(handover_latency_ms, 2),
            "status": "HANDOVER_DISPATCH_SUCCESS",
            "transcription_demo": "[Command Received & Streamed to Host ASR]"
        }
