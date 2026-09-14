"""
Milestone 10: Streaming Detector Benchmark & Simulation
SIH Problem Statement 26172

Simulates a continuous 17-second audio stream containing:
- Silence and ambient noise
- Running tap / pink noise
- Unrelated spoken speech words
- 2 distinct target 'ZORA' keyword occurrences at t=8s and t=13s
Evaluates real-time detection latency, VAD idle CPU skip rate, and state machine transitions.
"""

import os
import sys
import json
import time
from datetime import datetime
import numpy as np
import soundfile as sf

sys.path.insert(0, r"D:\SIH_Model")
from src.models.tiny_cnn import build_tiny_cnn_encoder
from src.models.ds_cnn import build_ds_cnn_encoder
from src.enrollment.enroll import KeywordEnrollmentManager
from src.streaming.detector import StreamingVoiceActivator
from src.streaming.state_machine import DetectionStateMachine

def build_composite_audio_stream():
    """
    Constructs a continuous 17.0-second 16 kHz mono audio stream with:
    0-3s: Silence/ambient room noise
    3-5s: Running tap noise
    5-6s: Unrelated word ('stop')
    6-8s: Ambient silence
    8-9s: Target Keyword 'ZORA' (Trigger 1)
    9-13s: Unrelated word ('yes') + pink noise
    13-14s: Target Keyword 'ZORA' (Trigger 2)
    14-17s: Ambient silence/noise
    """
    sr = 16000
    timeline = []

    # Helper: read or generate 1s clip
    def get_clip(fpath, default_duration=1.0):
        if os.path.exists(fpath):
            a, _ = sf.read(fpath, dtype='float32')
            if a.ndim > 1:
                a = a[:, 0]
            if len(a) < int(sr * default_duration):
                a = np.pad(a, (0, int(sr * default_duration) - len(a)))
            elif len(a) > int(sr * default_duration):
                a = a[:int(sr * default_duration)]
            return a
        return np.random.normal(0, 0.002, int(sr * default_duration)).astype(np.float32)

    # 1. 0-3s: Silence (3s)
    silence = np.random.normal(0, 0.001, sr * 3).astype(np.float32)
    timeline.append(silence)

    # 2. 3-5s: Running tap noise (2s)
    tap_path = r"D:\SIH_Model\data\raw\speech_commands\_background_noise_\running_tap.wav"
    tap_noise = get_clip(tap_path, default_duration=2.0) * 0.4
    timeline.append(tap_noise)

    # 3. 5-6s: Unrelated human speech ('stop')
    stop_sample = r"D:\SIH_Model\data\raw\speech_commands\stop\004ae714_nohash_0.wav"
    timeline.append(get_clip(stop_sample, default_duration=1.0))

    # 4. 6-8s: Ambient silence (2s)
    timeline.append(np.random.normal(0, 0.001, sr * 2).astype(np.float32))

    # 5. 8-9s: Target Keyword 'ZORA' (Trigger 1)
    zora_1 = r"D:\SIH_Model\data\raw\custom_keywords\zora\zora_david_rate+0_var0.wav"
    timeline.append(get_clip(zora_1, default_duration=1.0))

    # 6. 9-13s: Unrelated word ('yes') + Pink noise (4s)
    yes_sample = r"D:\SIH_Model\data\raw\speech_commands\yes\004ae714_nohash_0.wav"
    pink_path = r"D:\SIH_Model\data\raw\speech_commands\_background_noise_\pink_noise.wav"
    timeline.append(get_clip(yes_sample, default_duration=1.0))
    timeline.append(get_clip(pink_path, default_duration=3.0) * 0.2)

    # 7. 13-14s: Target Keyword 'ZORA' (Trigger 2)
    zora_2 = r"D:\SIH_Model\data\raw\custom_keywords\zora\zora_zira_rate+0_var1.wav"
    timeline.append(get_clip(zora_2, default_duration=1.0))

    # 8. 14-17s: Ambient silence (3s)
    timeline.append(np.random.normal(0, 0.001, sr * 3).astype(np.float32))

    composite = np.concatenate(timeline)
    print(f"Constructed composite audio stream: {len(composite)/sr:.2f} s ({len(composite):,} samples)")
    return composite

def run_streaming_simulation():
    print("="*80)
    print("MILESTONE 10: REAL-TIME STREAMING VOICE ACTIVATOR BENCHMARK")
    print("Ring Buffer -> Energy VAD Gate -> MFCC Extractor -> Encoder -> State Machine")
    print("="*80)

    # 1. Load Trained Encoders
    tiny_encoder = build_tiny_cnn_encoder(input_shape=(98, 13, 1), embedding_dim=32)
    tiny_weights = r"D:\SIH_Model\models\checkpoints\tiny_cnn_metric_best.weights.h5"
    if os.path.exists(tiny_weights):
        tiny_encoder.load_weights(tiny_weights)

    ds_encoder = build_ds_cnn_encoder(input_shape=(98, 13, 1), embedding_dim=32)
    ds_weights = r"D:\SIH_Model\models\checkpoints\ds_cnn_metric_best.weights.h5"
    if os.path.exists(ds_weights):
        ds_encoder.load_weights(ds_weights)

    # 2. Enroll 3-Shot 'ZORA' Prototype
    enroll_manager = KeywordEnrollmentManager(tiny_encoder)
    zora_shots = [
        r"D:\SIH_Model\data\raw\custom_keywords\zora\zora_david_rate+0_var0.wav",
        r"D:\SIH_Model\data\raw\custom_keywords\zora\zora_david_rate+1_var0.wav",
        r"D:\SIH_Model\data\raw\custom_keywords\zora\zora_zira_rate+0_var0.wav"
    ]
    enroll_info = enroll_manager.enroll(zora_shots, keyword_name="ZORA")
    zora_proto = enroll_info["prototype"]
    print(f"Enrolled 3-Shot 'ZORA' Prototype Vector (L2 Norm: {np.linalg.norm(zora_proto):.4f})")

    # 3. Build Continuous Audio Stream
    audio_stream = build_composite_audio_stream()

    # 4. Run Streaming Simulation on Tiny CNN
    print("\n" + "#"*80)
    print("[Simulation 1/2] STREAMING ACTIVATOR: TINY CNN (48,912 PARAMS)")
    print("#"*80)
    chunk_ms = 50  # 50 ms chunk advance (800 samples)
    sm_tiny = DetectionStateMachine(threshold=0.78, hysteresis=0.05, consecutive_windows=3, cooldown_ms=1500)
    activator_tiny = StreamingVoiceActivator(tiny_encoder, target_prototype=zora_proto, state_machine=sm_tiny)

    sim_tiny = activator_tiny.simulate_stream(audio_stream, chunk_size_ms=chunk_ms)

    print(f"Total Stream Duration:       {sim_tiny['audio_duration_seconds']} s")
    print(f"Total Chunks Processed:      {sim_tiny['total_chunks_processed']} chunks ({chunk_ms} ms each)")
    print(f"VAD Silence/Noise Skips:     {sim_tiny['chunks_skipped_by_vad']} chunks ({sim_tiny['vad_skip_percentage']}% inference skipped)")
    print(f"Active Inference Chunks:     {sim_tiny['active_inference_chunks']}")
    print(f"Average Inference Latency:   {sim_tiny['average_inference_latency_ms']} ms / chunk")
    print(f"Real-Time Factor (RTF):      {sim_tiny['real_time_factor']}x ({(1.0/sim_tiny['real_time_factor']):.1f}x faster than real-time)")
    print(f"Total Activations Triggered: {sim_tiny['total_activations']}")

    for idx, act in enumerate(sim_tiny['activation_events'], 1):
        print(f"  [Activation {idx}] Timestamp: {act['timestamp_ms']:.1f} ms ({act['timestamp_ms']/1000.0:.2f} s) | Smoothed Sim: {act['smoothed_similarity']:.4f}")

    # Calculate detection latencies
    # Ground truth: ZORA starts at 8000ms (ends ~9000ms) and 13000ms (ends ~14000ms)
    expected_activations = [8000.0, 13000.0]
    latencies = []
    for exp, act in zip(expected_activations, sim_tiny['activation_events']):
        lat = act['timestamp_ms'] - exp
        latencies.append(round(lat, 1))

    print(f"Empirical Detection Latencies: {latencies} ms from keyword onset")

    # 5. Run Streaming Simulation on DS-CNN
    print("\n" + "#"*80)
    print("[Simulation 2/2] STREAMING ACTIVATOR: DS-CNN (14,384 PARAMS)")
    print("#"*80)
    # Re-enroll prototype with DS-CNN encoder
    ds_enroll_mgr = KeywordEnrollmentManager(ds_encoder)
    ds_proto = ds_enroll_mgr.enroll(zora_shots, keyword_name="ZORA")["prototype"]

    # Threshold for DS-CNN based on M9 EER threshold
    sm_ds = DetectionStateMachine(threshold=0.995, hysteresis=0.005, consecutive_windows=3, cooldown_ms=1500)
    activator_ds = StreamingVoiceActivator(ds_encoder, target_prototype=ds_proto, state_machine=sm_ds)
    sim_ds = activator_ds.simulate_stream(audio_stream, chunk_size_ms=chunk_ms)

    print(f"Total Activations Triggered: {sim_ds['total_activations']}")
    for idx, act in enumerate(sim_ds['activation_events'], 1):
        print(f"  [Activation {idx}] Timestamp: {act['timestamp_ms']:.1f} ms | Smoothed Sim: {act['smoothed_similarity']:.4f}")

    # Remove huge telemetry array before saving JSON
    sim_tiny_clean = {k: v for k, v in sim_tiny.items() if k != "telemetry"}
    sim_ds_clean = {k: v for k, v in sim_ds.items() if k != "telemetry"}

    results = {
        "timestamp": datetime.now().isoformat(),
        "milestone": "Milestone 10: Streaming Detector & State Machine",
        "stream_duration_seconds": 17.0,
        "chunk_size_ms": chunk_ms,
        "tiny_cnn_streaming": {
            **sim_tiny_clean,
            "detection_latencies_ms": latencies,
            "false_activations_count": max(0, sim_tiny['total_activations'] - 2)
        },
        "ds_cnn_streaming": {
            **sim_ds_clean,
            "false_activations_count": max(0, sim_ds['total_activations'] - 2)
        }
    }

    out_file = r"D:\SIH_Model\experiments\threshold\streaming_simulation_results.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved streaming simulation results to: {out_file}")
    print("="*80)
    return results

if __name__ == "__main__":
    run_streaming_simulation()
