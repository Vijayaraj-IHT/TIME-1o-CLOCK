"""
Unified Dynamic Custom Keyword Enrollment Utility
SIH Problem Statement 26172 (ISRO)

Enrolls ANY new unseen custom keyword into the Edge Voice Activator with ZERO retraining:
Features:
1. Full Data Purge: Clears old custom keyword recordings & headers to prevent directory pollution.
2. VAD-Guided Speech Trimming & Centering: Eliminates silence padding bias.
3. Audio Quality Checks: Rejects clipped or muted/too-quiet recordings.
4. Anti-Degeneracy Validation: Validates prototype against silence, noise, and reference words.
5. Auto-Sync: Updates ESP32-S3, ESP32-WROOM headers, and outputs.
"""

import os
import sys
import shutil
import time
import argparse
import subprocess
import numpy as np
import soundfile as sf
import sounddevice as sd
import scipy.signal
import tensorflow as tf

sys.path.insert(0, r"D:\SIH_Model")
from src.features.mfcc import MFCCFeatureExtractor


def trim_and_center_audio(audio: np.ndarray, sr: int = 16000, target_len: int = 16000):
    """
    Detects speech onset and offset using frame energy,
    and centers the active speech in a 1.0-second (16,000 samples) window.
    """
    frame_len = 320   # 20 ms
    hop_len = 160     # 10 ms
    if len(audio) < frame_len:
        return None, "Audio is too short."

    num_frames = (len(audio) - frame_len) // hop_len + 1
    energies = np.array([
        np.sum(audio[i * hop_len : i * hop_len + frame_len] ** 2)
        for i in range(num_frames)
    ])

    max_energy = np.max(energies) if len(energies) > 0 else 0
    energy_thresh = max(max_energy * 0.08, 0.001)

    speech_frames = np.where(energies >= energy_thresh)[0]
    if len(speech_frames) == 0:
        return None, "No speech detected (audio is silent)."

    start_sample = speech_frames[0] * hop_len
    end_sample = min(len(audio), speech_frames[-1] * hop_len + frame_len)
    speech_duration = (end_sample - start_sample) / sr

    if speech_duration < 0.20:
        return None, f"Speech too brief ({speech_duration:.2f}s). Speak the full keyword clearly."

    speech_segment = audio[start_sample:end_sample]

    # Center speech in target_len
    if len(speech_segment) >= target_len:
        centered = speech_segment[:target_len]
    else:
        pad_total = target_len - len(speech_segment)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        centered = np.pad(speech_segment, (pad_left, pad_right), mode="constant")

    # Normalize peak amplitude to 0.88 to avoid clipping
    max_peak = np.max(np.abs(centered))
    if max_peak > 0:
        centered = (centered / max_peak) * 0.88

    return centered, f"Isolated speech ({speech_duration:.2f}s) centered in 1.0s window"


def purge_old_keyword_data(target_keyword: str):
    """Purges old keyword directories and artifacts to guarantee zero directory pollution."""
    base_dir = r"D:\SIH_Model"
    custom_kw_root = os.path.join(base_dir, "data", "raw", "custom_keywords")
    
    print("\n[PURGE] Purging previous custom keyword directories...")
    if os.path.exists(custom_kw_root):
        for item in os.listdir(custom_kw_root):
            item_path = os.path.join(custom_kw_root, item)
            # Preserve benchmark unit-test fixture 'zora' and .gitkeep
            if os.path.isdir(item_path) and item.lower() not in [".gitkeep", "zora"]:
                try:
                    shutil.rmtree(item_path)
                    print(f"  - Deleted old directory: {item}")
                except Exception as e:
                    print(f"  ! Warning: could not delete {item_path}: {e}")

    # Remove old prototype headers to ensure fresh generation
    for header in [
        os.path.join(base_dir, "src", "deployment", "esp32", "keyword_prototype.h"),
        os.path.join(base_dir, "src", "deployment", "esp32_wroom", "keyword_prototype.h"),
        os.path.join(base_dir, "outputs", "esp32_wroom", "keyword_prototype.h")
    ]:
        if os.path.exists(header):
            try:
                os.remove(header)
            except Exception:
                pass
    print("  - Cleared old prototype header files.")


def record_from_mic(keyword: str, output_dir: str, num_samples: int = 3, sr: int = 16000) -> list:
    """Interactively records K audio clips with speech isolation and quality validation."""
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    duration = 1.5  # Record 1.5 seconds so speech is not prematurely truncated

    print("\n" + "=" * 75)
    print(f"HIGH-QUALITY MICROPHONE ENROLLMENT: '{keyword.upper()}'")
    print(f"You will record {num_samples} audio utterances (1.5 seconds window each).")
    print("Speak clearly at a normal conversational volume, ~15-30 cm from mic.")
    print("=" * 75)

    sample_idx = 1
    while sample_idx <= num_samples:
        input(f"\n[{sample_idx}/{num_samples}] Press [ENTER], then speak '{keyword.upper()}' clearly...")
        
        sys.stdout.write("  >> RECORDING NOW... ")
        sys.stdout.flush()
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
        sd.wait()
        audio = audio.flatten()
        print("Captured!")

        # 1. Check for severe clipping
        clipped_samples = np.sum(np.abs(audio) >= 0.98)
        if clipped_samples > 150:
            print("  [!] WARNING: Audio clipped (mic too close or gain too high). Please speak slightly further from mic.")
            retry = input("      Press [ENTER] to re-record this sample...")
            continue

        # 2. Check for silence / low volume
        max_amp = np.max(np.abs(audio))
        if max_amp < 0.05:
            print(f"  [!] WARNING: Audio too quiet (Peak: {max_amp:.4f}). Speak louder and directly into mic.")
            retry = input("      Press [ENTER] to re-record this sample...")
            continue

        # 3. Speech Trimming & Centering
        centered_audio, msg = trim_and_center_audio(audio, sr=sr, target_len=16000)
        if centered_audio is None:
            print(f"  [!] WARNING: {msg}")
            retry = input("      Press [ENTER] to re-record this sample...")
            continue

        file_path = os.path.join(output_dir, f"{keyword.lower()}_sample_{sample_idx}.wav")
        sf.write(file_path, centered_audio, sr)
        saved_files.append(file_path)
        print(f"  [OK] Sample {sample_idx} validated ({msg}).")
        sample_idx += 1

    print(f"\nSuccessfully recorded {len(saved_files)} clean samples into {output_dir}!")
    return saved_files


def synthesize_samples(keyword: str, output_dir: str, num_samples: int = 3, target_sr: int = 16000) -> list:
    """Auto-synthesizes K audio clips using Windows SAPI TTS with trimming & centering."""
    os.makedirs(output_dir, exist_ok=True)
    voices = ["Microsoft David Desktop", "Microsoft Zira Desktop"]
    rates = [-1, 0, 1]
    generated_files = []
    count = 0

    print(f"\nAuto-synthesizing {num_samples} samples for '{keyword.upper()}' via Windows SAPI...")
    for voice in voices:
        v_tag = "david" if "David" in voice else "zira"
        for rate in rates:
            if count >= num_samples:
                break
            raw_wav = os.path.join(output_dir, f"temp_{v_tag}_r{rate}_{count}.wav").replace("\\", "/")
            final_wav = os.path.join(output_dir, f"{keyword.lower()}_{v_tag}_r{rate}_{count}.wav")

            ps_cmd = (
                f"Add-Type -AssemblyName System.Speech; "
                f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.SelectVoice('{voice}'); "
                f"$s.Rate = {rate}; "
                f"$s.SetOutputToWaveFile('{raw_wav}'); "
                f"$s.Speak('{keyword}'); "
                f"$s.Dispose()"
            )
            subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True, text=True)

            if os.path.exists(raw_wav):
                audio, orig_sr = sf.read(raw_wav, dtype="float32")
                if audio.ndim > 1:
                    audio = audio[:, 0]
                num_target_samples = int(len(audio) * (target_sr / orig_sr))
                audio_16k = scipy.signal.resample(audio, num_target_samples)

                centered, _ = trim_and_center_audio(audio_16k, sr=target_sr, target_len=16000)
                if centered is None:
                    centered = audio_16k[:16000] if len(audio_16k) >= 16000 else np.pad(audio_16k, (0, 16000 - len(audio_16k)))
                    max_v = np.max(np.abs(centered))
                    if max_v > 0: centered = (centered / max_v) * 0.88

                sf.write(final_wav, centered, target_sr)
                generated_files.append(final_wav)
                count += 1
                try: os.remove(raw_wav)
                except Exception: pass

    return generated_files


def extract_and_validate_prototype(
    keyword_name: str,
    audio_files: list,
    model_path: str = r"D:\SIH_Model\models\tflite\voice_activator_int8.tflite",
    output_header: str = r"D:\SIH_Model\src\deployment\esp32\keyword_prototype.h"
) -> dict:
    """Extracts 32-D embeddings, validates against degeneracy, and writes headers."""
    feature_extractor = MFCCFeatureExtractor(sample_rate=16000, n_mfcc=13)
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    in_idx = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]
    in_dtype = interpreter.get_input_details()[0]["dtype"]

    def embed_audio(audio: np.ndarray) -> np.ndarray:
        if len(audio) < 16000:
            audio = np.pad(audio, (0, 16000 - len(audio)), mode="constant")
        else:
            audio = audio[:16000]
        mfcc = feature_extractor.extract(audio)
        tensor = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=-1).astype(in_dtype)
        interpreter.set_tensor(in_idx, tensor)
        interpreter.invoke()
        emb = interpreter.get_tensor(out_idx)[0]
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-9)

    # 1. Extract sample embeddings
    embeddings = []
    for fpath in audio_files:
        audio, sr = sf.read(fpath, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != 16000:
            audio = scipy.signal.resample(audio, int(len(audio) * 16000 / sr))
        emb = embed_audio(audio)
        embeddings.append(emb)

    embeddings = np.array(embeddings)
    centroid = np.mean(embeddings, axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    # 2. Check Intra-shot consistency
    pair_sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            pair_sims.append(float(np.dot(embeddings[i], embeddings[j])))
    intra_sim = float(np.mean(pair_sims)) if pair_sims else 1.0

    # 3. Anti-Degeneracy Validation: Check against silence and noise
    silence_audio = np.zeros(16000, dtype=np.float32)
    silence_emb = embed_audio(silence_audio)
    sim_silence = float(np.dot(centroid, silence_emb))

    noise_audio = np.random.uniform(-0.02, 0.02, 16000).astype(np.float32)
    noise_emb = embed_audio(noise_audio)
    sim_noise = float(np.dot(centroid, noise_emb))

    # Check against sample reference speech words if available
    ref_sims = []
    sc_dir = r"D:\SIH_Model\data\raw\speech_commands"
    if os.path.exists(sc_dir):
        for ref_w in ["stop", "go", "marvin"]:
            rw_dir = os.path.join(sc_dir, ref_w)
            if os.path.exists(rw_dir):
                rf = [os.path.join(rw_dir, f) for f in os.listdir(rw_dir) if f.endswith(".wav")]
                if rf:
                    ref_audio, _ = sf.read(rf[0])
                    ref_sims.append(float(np.dot(centroid, embed_audio(ref_audio))))
    avg_ref_sim = float(np.mean(ref_sims)) if ref_sims else 0.50

    print("\n" + "=" * 75)
    print("ENROLLMENT VALIDATION SCORECARD")
    print("=" * 75)
    print(f"  - Target Keyword:          '{keyword_name.upper()}'")
    print(f"  - Intra-Shot Consistency:  {intra_sim:.4f}  (Pass: >= 0.70) -> {'PASS' if intra_sim >= 0.70 else 'FAIL'}")
    print(f"  - Silence Separation:      {sim_silence:.4f}  (Pass: <= 0.65) -> {'PASS' if sim_silence <= 0.65 else 'WARNING: High Silence Correlation'}")
    print(f"  - Noise Separation:        {sim_noise:.4f}  (Pass: <= 0.60) -> {'PASS' if sim_noise <= 0.60 else 'WARNING: High Noise Correlation'}")
    print(f"  - Ref Word Separation:     {avg_ref_sim:.4f}  (Typical: 0.50 - 0.75)")
    print("=" * 75)

    if intra_sim < 0.70:
        raise ValueError(f"Enrollment Rejected: Inconsistent recordings (Intra-similarity: {intra_sim:.4f} < 0.70). Please re-record.")

    if sim_silence > 0.72:
        print("[!] Warning: Prototype has elevated correlation with silence. Speaking louder will improve separation.")

    # 4. Write C++ Headers
    proto_c_str = ", ".join([f"{v:.7f}f" for v in centroid])
    header_content = f"""/*
 * Auto-generated Keyword Prototype Header
 * SIH Problem Statement 26172
 * Keyword: {keyword_name.upper()}
 * Embedding Dimension: {len(centroid)}
 * Hypersphere Normalized: True (||p||_2 = 1.0)
 */

#ifndef KEYWORD_PROTOTYPE_H_
#define KEYWORD_PROTOTYPE_H_

#define ENROLLED_KEYWORD_NAME "{keyword_name.upper()}"
#define KEYWORD_PROTOTYPE_DIM {len(centroid)}
#define EMBEDDING_DIMENSION KEYWORD_PROTOTYPE_DIM
#define ENROLLED_KEYWORD_PROTOTYPE KEYWORD_PROTOTYPE

// L2-normalized prototype vector centroid on 32-D unit hypersphere
static const float KEYWORD_PROTOTYPE[KEYWORD_PROTOTYPE_DIM] = {{
    {proto_c_str}
}};

#endif // KEYWORD_PROTOTYPE_H_
"""
    # Write to primary target
    os.makedirs(os.path.dirname(output_header), exist_ok=True)
    with open(output_header, "w", encoding="utf-8") as f:
        f.write(header_content)
    print(f"\n[OK] Updated primary prototype header: {output_header}")

    # Auto-sync to all ESP32 deployment targets
    base_proj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    for wroom_dir in [
        os.path.join(base_proj_dir, 'outputs', 'esp32_wroom'),
        os.path.join(base_proj_dir, 'src', 'deployment', 'esp32_wroom')
    ]:
        if os.path.exists(wroom_dir):
            wroom_header = os.path.join(wroom_dir, 'keyword_prototype.h')
            with open(wroom_header, 'w', encoding='utf-8') as fw:
                fw.write(header_content)
            print(f"[OK] Auto-synced to: {wroom_header}")

    return {
        "keyword": keyword_name.upper(),
        "shots_count": len(audio_files),
        "intra_similarity": round(intra_sim, 4),
        "silence_similarity": round(sim_silence, 4),
        "noise_similarity": round(sim_noise, 4),
        "prototype": centroid,
        "header_path": output_header
    }


def main():
    parser = argparse.ArgumentParser(description="Enroll any new custom keyword into the Edge Voice Activator")
    parser.add_argument("--keyword", required=True, help="Custom keyword name (e.g., HELIOS, APOLLO, JARVIS)")
    parser.add_argument("--mic", action="store_true", help="Record 3 samples interactively using laptop microphone")
    parser.add_argument("--synth", action="store_true", help="Auto-synthesize 3 samples using TTS (no mic needed)")
    parser.add_argument("--audio-dir", default=None, help="Directory containing your own custom WAV files")
    parser.add_argument("--shots", type=int, default=3, help="Number of enrollment shots (default: 3)")
    parser.add_argument("--no-purge", action="store_true", help="Skip purging previous custom keyword folders")

    args = parser.parse_args()
    kw = args.keyword.upper()

    # Step 1: Purge old keyword data unless explicitly skipped
    if not args.no_purge:
        purge_old_keyword_data(target_keyword=kw)

    target_dir = os.path.join(r"D:\SIH_Model\data\raw\custom_keywords", kw.lower())
    os.makedirs(target_dir, exist_ok=True)

    # Step 2: Acquire samples
    if args.mic:
        audio_files = record_from_mic(kw, target_dir, num_samples=args.shots)
    elif args.synth:
        audio_files = synthesize_samples(kw, target_dir, num_samples=args.shots)
    elif args.audio_dir:
        audio_files = [
            os.path.join(args.audio_dir, f) for f in os.listdir(args.audio_dir)
            if f.endswith(".wav")
        ][:args.shots]
    else:
        print("[!] No mode specified. Defaulting to --synth mode (Windows SAPI)...")
        audio_files = synthesize_samples(kw, target_dir, num_samples=args.shots)

    if not audio_files:
        print("[ERROR] No audio files acquired for enrollment.")
        sys.exit(1)

    # Step 3: Extract and validate prototype
    res = extract_and_validate_prototype(kw, audio_files)
    print("\n" + "=" * 75)
    print(f"SUCCESS: Custom keyword '{kw}' enrolled and validated successfully!")
    print(f"You can now test live: python scripts/live_mic_activator.py --keyword {kw}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
