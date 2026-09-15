"""
Continuous Real-Time Streaming Voice Activator Engine
SIH Problem Statement 26172 - Milestone 10

Orchestrates the entire edge activation pipeline:
Audio Stream -> Ring Buffer -> Energy VAD Gate -> MFCC Extractor -> Frozen Encoder -> Cosine Comparator -> State Machine
"""

import time
import numpy as np

from src.features.mfcc import MFCCFeatureExtractor
from src.streaming.ring_buffer import AudioRingBuffer
from src.streaming.vad import EnergyVAD
from src.streaming.state_machine import DetectionStateMachine, DetectionState
from src.models.prototype import apply_garbage_veto
from src.evaluation.latency import LatencyStats

class StreamingVoiceActivator:
    def __init__(self, encoder, target_prototype: np.ndarray,
                 sample_rate=16000, window_duration_s=1.0,
                 feature_extractor=None, vad=None, state_machine=None,
                 garbage_prototype=None, garbage_margin=0.05):
        self.encoder = encoder
        self.target_prototype = np.asarray(target_prototype, dtype=np.float32).flatten()
        # Ensure prototype is unit normalized
        norm = np.linalg.norm(self.target_prototype)
        if norm > 0:
            self.target_prototype /= norm

        # Phase-1 Change 2: optional garbage prototype (None => legacy behavior)
        self.garbage_margin = float(garbage_margin)
        if garbage_prototype is None:
            self.garbage_prototype = None
        else:
            g = np.asarray(garbage_prototype, dtype=np.float32).flatten()
            gn = np.linalg.norm(g)
            self.garbage_prototype = (g / gn) if gn > 0 else None

        self.sample_rate = sample_rate
        self.capacity_samples = int(sample_rate * window_duration_s)

        self.ring_buffer = AudioRingBuffer(capacity_samples=self.capacity_samples)
        self.vad = vad or EnergyVAD(sample_rate=sample_rate)
        self.feature_extractor = feature_extractor or MFCCFeatureExtractor(sample_rate=sample_rate, n_mfcc=13, n_mels=40)
        self.state_machine = state_machine or DetectionStateMachine()

        # Telemetry metrics
        self.total_chunks = 0
        self.skipped_chunks = 0
        self.active_inference_chunks = 0
        self.total_inference_time_s = 0.0

    def process_chunk(self, chunk: np.ndarray, timestamp_ms: float) -> dict:
        """
        Processes a single incoming audio chunk (e.g., 20ms - 50ms).
        Returns telemetry dictionary including activation events.
        """
        self.total_chunks += 1
        t_start = time.perf_counter()

        # 1. Update rolling ring buffer
        self.ring_buffer.append(chunk)

        # 2. VAD Check: Gate heavy CNN inference during silence
        is_speech = self.vad.is_speech(chunk)
        is_verifying = (self.state_machine.state == DetectionState.VERIFYING)

        if not is_speech and not is_verifying:
            # Skip CNN inference: decay smoothing with the baseline floor, but
            # as UNOBSERVED (Change 3a: must not teach the background model).
            self.skipped_chunks += 1
            sm_res = self.state_machine.process_similarity(raw_similarity=0.0, timestamp_ms=timestamp_ms, observed=False)
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0

            return {
                "timestamp_ms": timestamp_ms,
                "is_speech": False,
                "inference_skipped": True,
                "garbage_similarity": None,
                "garbage_vetoed": False,
                "raw_similarity": 0.0,
                "smoothed_similarity": sm_res["smoothed_similarity"],
                "threshold_eff": sm_res["threshold_eff"],
                "trigger_latency_ms": sm_res["trigger_latency_ms"],
                "state": sm_res["state"],
                # Change 3d: persistence can complete on a silence frame;
                # report the state machine's verdict, not a hardcoded False.
                "is_activated": sm_res["is_activated"],
                "processing_time_ms": round(elapsed_ms, 3)
            }

        # 3. Active speech or ongoing verification: execute CNN embedding extraction
        self.active_inference_chunks += 1
        window_audio = self.ring_buffer.get_snapshot()

        # 4. Extract MFCC features
        feat = self.feature_extractor.extract(window_audio)  # Shape: (98, 13)
        feat_input = feat[np.newaxis, :, :, np.newaxis]

        # 5. Model forward pass
        t_inf_start = time.perf_counter()
        emb = self.encoder(feat_input, training=False).numpy()[0]
        self.total_inference_time_s += (time.perf_counter() - t_inf_start)

        emb_norm = np.linalg.norm(emb)
        if emb_norm > 0:
            emb /= emb_norm

        # 6. Cosine similarity vs target (+ garbage veto, Change 2)
        s_kw = float(np.dot(emb, self.target_prototype))
        garbage_sim = None
        garbage_vetoed = False
        if self.garbage_prototype is not None:
            garbage_sim = float(np.dot(emb, self.garbage_prototype))
            raw_sim, garbage_vetoed = apply_garbage_veto(s_kw, garbage_sim, self.garbage_margin)
        else:
            raw_sim = s_kw

        # 7. Advance state machine
        sm_res = self.state_machine.process_similarity(raw_similarity=raw_sim, timestamp_ms=timestamp_ms)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "timestamp_ms": timestamp_ms,
            "is_speech": is_speech,
            "inference_skipped": False,
            "garbage_similarity": round(garbage_sim, 4) if garbage_sim is not None else None,
            "garbage_vetoed": garbage_vetoed,
            "raw_similarity": sm_res["raw_similarity"],
            "smoothed_similarity": sm_res["smoothed_similarity"],
            "threshold_eff": sm_res["threshold_eff"],
            "trigger_latency_ms": sm_res["trigger_latency_ms"],
            "state": sm_res["state"],
            "is_activated": sm_res["is_activated"],
            "processing_time_ms": round(elapsed_ms, 3)
        }

    def mark_keyword_end(self, timestamp_ms: float) -> None:
        """Change 6 passthrough: ground-truth keyword-end marker for the next
        activation's trigger_latency_ms (bench/sim only)."""
        self.state_machine.mark_keyword_end(timestamp_ms)

    def simulate_stream(self, audio: np.ndarray, chunk_size_ms=50) -> dict:
        """
        Simulates an end-to-end continuous audio stream.
        """
        self.reset()
        chunk_samples = int(self.sample_rate * (chunk_size_ms / 1000.0))
        total_samples = len(audio)
        duration_s = total_samples / self.sample_rate

        activations = []
        telemetry = []
        t0 = time.perf_counter()

        for idx in range(0, total_samples, chunk_samples):
            chunk = audio[idx:idx + chunk_samples]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            timestamp_ms = (idx / self.sample_rate) * 1000.0

            step_res = self.process_chunk(chunk, timestamp_ms)
            telemetry.append(step_res)

            if step_res["is_activated"]:
                activations.append({
                    "timestamp_ms": timestamp_ms,
                    "smoothed_similarity": step_res["smoothed_similarity"],
                    "trigger_latency_ms": step_res["trigger_latency_ms"]
                })

        total_proc_time = time.perf_counter() - t0
        rtf = total_proc_time / duration_s if duration_s > 0 else 0.0

        skip_pct = (self.skipped_chunks / self.total_chunks * 100.0) if self.total_chunks > 0 else 0.0
        avg_inf_latency_ms = (self.total_inference_time_s / self.active_inference_chunks * 1000.0) if self.active_inference_chunks > 0 else 0.0

        # Change 6: keyword-END -> trigger latency over marked activations
        lat_stats = LatencyStats()
        for act in activations:
            lat_stats.add(act.get("trigger_latency_ms"))

        return {
            "audio_duration_seconds": round(duration_s, 2),
            "total_chunks_processed": self.total_chunks,
            "chunks_skipped_by_vad": self.skipped_chunks,
            "vad_skip_percentage": round(skip_pct, 2),
            "active_inference_chunks": self.active_inference_chunks,
            "total_activations": len(activations),
            "activation_events": activations,
            "average_inference_latency_ms": round(avg_inf_latency_ms, 2),
            "real_time_factor": round(rtf, 4),
            "latency_stats": lat_stats.summary(),
            "telemetry": telemetry
        }

    def reset(self):
        self.ring_buffer.reset()
        self.vad.reset()
        self.state_machine.reset()
        self.total_chunks = 0
        self.skipped_chunks = 0
        self.active_inference_chunks = 0
        self.total_inference_time_s = 0.0
