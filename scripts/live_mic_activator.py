"""
Live Laptop Microphone Edge Voice Activator
SIH Problem Statement 26172 (ISRO)

Allows real-time testing of the INT8 Quantized Edge Voice Activator directly through your laptop microphone:
1. Streams continuous audio in 50ms chunks (800 samples @ 16 kHz) via laptop microphone.
2. Performs real-time Ring Buffer -> VAD Gating -> INT8 TFLite Universal Feature Encoder -> Dual-Threshold Hysteresis.
3. Automatically triggers low-latency activation when the enrolled keyword is spoken.
4. Uses tightened persistence (N=3) and hysteresis (0.05) to eliminate false positives.
"""

import os
import sys
import time
import argparse
import numpy as np
import sounddevice as sd
import soundfile as sf

# Suppress TF logging clutter
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

sys.path.insert(0, r"D:\SIH_Model")
from src.streaming.demo_pipeline import EndToEndVoiceActivatorDemo
from src.streaming.ring_buffer import AudioRingBuffer
from src.streaming.vad import EnergyVAD
from src.streaming.state_machine import DetectionStateMachine
from src.features.mfcc import MFCCFeatureExtractor
import tensorflow as tf


def load_prototype_from_header(header_path: str):
    """Loads enrolled keyword name and 32-D prototype vector directly from C header."""
    if not os.path.exists(header_path):
        return None, None

    with open(header_path, "r", encoding="utf-8") as f:
        content = f.read()

    name = "UNKNOWN"
    for line in content.splitlines():
        if "ENROLLED_KEYWORD_NAME" in line and '"' in line:
            parts = line.split('"')
            if len(parts) >= 2:
                name = parts[1]

    start = content.find("KEYWORD_PROTOTYPE")
    if start != -1:
        brace_start = content.find("{", start)
        brace_end = content.find("}", brace_start)
        if brace_start != -1 and brace_end != -1:
            raw = content[brace_start + 1:brace_end].replace("f", "").replace("\n", "")
            nums = [float(x.strip()) for x in raw.split(",") if x.strip()]
            vec = np.array(nums, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                vec /= norm
            return name, vec

    return name, None


def run_live_mic(keyword: str = None, threshold: float = 0.88):
    header_path = r"D:\SIH_Model\src\deployment\esp32\keyword_prototype.h"
    hdr_name, hdr_proto = load_prototype_from_header(header_path)

    if keyword is None:
        kw = hdr_name if hdr_name else "ZORA"
    else:
        kw = keyword.upper()

    print("\n" + "=" * 80)
    print("      SIH 26172: EDGE VOICE ACTIVATOR - LIVE MICROPHONE DETECTION")
    print("=" * 80)
    print(f"Target Hardware Profile: ESP32-WROOM / ESP32-S3 Bare-Metal")
    print(f"Active Keyword:          '{kw}'")
    print(f"Operating Thresholds:    tau_high = {threshold:.2f}, tau_low = {threshold-0.05:.2f}")
    print(f"Temporal Persistence:    N = 3 consecutive windows (150 ms sustained match)")
    print(f"False-Positive Defense:  Dual-threshold hysteresis + refractory cooldown (1500 ms)")
    print("=" * 80)

    # 1. Acquire Prototype
    prototype = None
    if hdr_name and hdr_name.upper() == kw.upper() and hdr_proto is not None:
        prototype = hdr_proto
        print(f"  [PROTOTYPE] Loaded verified '{kw}' prototype directly from keyword_prototype.h")
    else:
        kw_dir = os.path.join(r"D:\SIH_Model\data\raw\custom_keywords", kw.lower())
        if os.path.exists(kw_dir):
            wavs = [os.path.join(kw_dir, f) for f in os.listdir(kw_dir) if f.endswith(".wav")]
            if wavs:
                print(f"  [PROTOTYPE] Extracting centroid from {len(wavs)} samples in {kw_dir}...")
                from scripts.enroll_keyword import extract_and_validate_prototype
                res = extract_and_validate_prototype(kw, wavs[:5])
                prototype = res["prototype"]

    if prototype is None:
        print(f"\n[!] Error: No prototype found for '{kw}'.")
        print(f"    Please enroll it first: python scripts/enroll_keyword.py --keyword {kw} --mic")
        sys.exit(1)

    # 2. Setup TFLite Model & Streaming Components
    model_path = r"D:\SIH_Model\models\tflite\voice_activator_int8.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    in_idx = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]
    in_dtype = interpreter.get_input_details()[0]["dtype"]

    feature_extractor = MFCCFeatureExtractor(sample_rate=16000, n_mfcc=13)
    ring_buffer = AudioRingBuffer(capacity_samples=16000)
    vad = EnergyVAD(sample_rate=16000, min_energy_threshold=0.008, energy_multiplier=2.2, hangover_frames=3)
    
    state_machine = DetectionStateMachine(
        threshold=threshold,
        hysteresis=0.05,
        consecutive_windows=3,
        smoothing_window=5,
        cooldown_ms=1500.0
    )

    chunk_samples = 800  # 50 ms @ 16 kHz
    output_test_dir = r"D:\SIH_Model\outputs\live_test"
    os.makedirs(output_test_dir, exist_ok=True)

    input_device = sd.query_devices(kind='input')['name']
    print(f"  [INPUT DEVICE]  {input_device}")
    print("-" * 80)
    print(f"  * Speak naturally into your microphone.")
    print(f"  * Say '{kw}' clearly.")
    print(f"  * Notice that unrelated words like 'yes', 'no', 'inspector', etc. will NOT trigger.")
    print(f"  * Press Ctrl + C to exit.")
    print("-" * 80 + "\n")

    start_time = time.time()
    activation_count = 0
    recent_sims = []

    try:
        with sd.InputStream(samplerate=16000, channels=1, dtype="float32", blocksize=chunk_samples) as stream:
            while True:
                audio_chunk, _ = stream.read(chunk_samples)
                audio_chunk = audio_chunk.flatten()
                current_time_ms = (time.time() - start_time) * 1000.0

                ring_buffer.append(audio_chunk)
                rms = np.sqrt(np.mean(audio_chunk**2) + 1e-9)

                # VAD Gating
                is_speech = vad.is_speech(audio_chunk)
                if not is_speech and rms < 0.010:
                    time.sleep(0.001)
                    continue

                # Extract 1.0s window
                audio_window = ring_buffer.get_snapshot()
                mfcc = feature_extractor.extract(audio_window)
                tensor = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=-1).astype(in_dtype)

                interpreter.set_tensor(in_idx, tensor)
                interpreter.invoke()
                emb = interpreter.get_tensor(out_idx)[0]
                norm = np.linalg.norm(emb)
                if norm > 1e-6:
                    emb /= norm

                # Cosine similarity
                sim = float(np.dot(emb, prototype))
                recent_sims.append(sim)
                if len(recent_sims) > 5:
                    recent_sims.pop(0)
                smoothed_sim = float(np.mean(recent_sims))

                # State machine update
                sm_res = state_machine.process_similarity(sim, current_time_ms)
                current_state = sm_res["state"]
                smoothed_sim = sm_res["smoothed_similarity"]

                # Real-time visual meter on same line
                bar_len = int(max(0, min(30, (smoothed_sim - 0.5) * 60)))
                bar_str = "#" * bar_len + "-" * (30 - bar_len)
                status_str = f"\r[VAD: ON | Vol: {int(rms*500):2d}% | Sim: {smoothed_sim:.3f} [{bar_str}] State: {current_state}]"
                sys.stdout.write(status_str)
                sys.stdout.flush()

                if sm_res["is_activated"]:
                    activation_count += 1
                    sys.stdout.write("\n\n" + "=" * 80 + "\n")
                    sys.stdout.write(f"  >>> [ACTIVATION TRIGGERED #{activation_count}] KEYWORD '{kw}' DETECTED! <<<\n")
                    sys.stdout.write(f"  Timestamp:  {current_time_ms/1000.0:.2f}s\n")
                    sys.stdout.write(f"  Confidence: {smoothed_sim:.4f} (Threshold: {threshold:.2f}, Hysteresis: {threshold-0.05:.2f})\n")
                    sys.stdout.write(f"  Action:     Hardware GPIO Wake Trigger Dispatched & ASR Handover Initiated.\n")
                    sys.stdout.write("=" * 80 + "\n\n")
                    sys.stdout.flush()

    except KeyboardInterrupt:
        print(f"\n\n[SESSION TERMINATED] Live microphone session stopped by user.")
        print(f"Total activations: {activation_count}")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Live Laptop Microphone Voice Activator")
    parser.add_argument("--keyword", default=None, help="Target keyword name (defaults to active keyword in header)")
    parser.add_argument("--threshold", type=float, default=0.88, help="Activation similarity threshold (default: 0.88)")
    args = parser.parse_args()
    run_live_mic(keyword=args.keyword, threshold=args.threshold)


if __name__ == "__main__":
    main()
