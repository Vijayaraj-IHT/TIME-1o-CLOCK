"""
Custom Keyword Audio Generator for Edge Voice Activator
SIH Problem Statement 26172 - Milestone 9

Synthesizes diverse acoustic utterances of unseen custom keywords (e.g., 'ZORA')
across multiple voices, speaking rates, and acoustic conditions.
Saves 16 kHz mono 16-bit PCM WAV files to D:\SIH_Model\data\raw\custom_keywords\zora.
"""

import os
import sys
import tempfile
import subprocess
import numpy as np
import soundfile as sf
import scipy.signal

def synthesize_custom_keyword(keyword="Zora", output_dir=r"D:\SIH_Model\data\raw\custom_keywords\zora"):
    os.makedirs(output_dir, exist_ok=True)
    voices = ["Microsoft David Desktop", "Microsoft Zira Desktop"]
    rates = [-2, -1, 0, 1, 2]

    print("="*70)
    print(f"GENERATING CUSTOM KEYWORD UTTERANCES: '{keyword.upper()}'")
    print(f"Target Directory: {output_dir}")
    print(f"Voices: {voices}")
    print(f"Speaking Rates: {rates}")
    print("="*70)

    count = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        for v_idx, voice in enumerate(voices):
            v_tag = "david" if "David" in voice else "zira"
            for rate in rates:
                for rep in range(3):  # 3 variations per rate/voice = 30 total
                    raw_wav = os.path.join(tmpdir, f"raw_{v_tag}_r{rate}_rep{rep}.wav").replace("\\", "/")
                    ps_cmd = (
                        f"Add-Type -AssemblyName System.Speech; "
                        f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                        f"$s.SelectVoice('{voice}'); "
                        f"$s.Rate = {rate}; "
                        f"$s.SetOutputToWaveFile('{raw_wav}'); "
                        f"$s.Speak('{keyword}'); "
                        f"$s.Dispose()"
                    )
                    res = subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True, text=True)
                    if not os.path.exists(raw_wav):
                        continue

                    try:
                        audio, orig_sr = sf.read(raw_wav, dtype='float32')
                        if audio.ndim > 1:
                            audio = audio[:, 0]

                        # Resample to 16000 Hz using scipy
                        target_sr = 16000
                        num_target_samples = int(len(audio) * (target_sr / orig_sr))
                        audio_16k = scipy.signal.resample(audio, num_target_samples)

                        # Standardize to 1.000 s (16,000 samples)
                        if len(audio_16k) > 16000:
                            diff = len(audio_16k) - 16000
                            start = diff // 2
                            audio_16k = audio_16k[start:start+16000]
                        else:
                            pad_total = 16000 - len(audio_16k)
                            pad_left = pad_total // 2
                            pad_right = pad_total - pad_left
                            audio_16k = np.pad(audio_16k, (pad_left, pad_right), mode='constant')

                        # Apply slight variations for realism
                        if rep == 1:
                            sig_pow = np.mean(audio_16k ** 2)
                            if sig_pow > 0:
                                noise_pow = sig_pow / (10 ** 2.5)
                                audio_16k = audio_16k + np.random.normal(0, np.sqrt(noise_pow), len(audio_16k)).astype(np.float32)
                        elif rep == 2:
                            shift = int(16000 * 0.03)
                            audio_16k = np.roll(audio_16k, shift)

                        audio_16k = np.clip(audio_16k, -1.0, 1.0)

                        out_name = f"zora_{v_tag}_rate{rate:+d}_var{rep}.wav"
                        out_path = os.path.join(output_dir, out_name)
                        sf.write(out_path, audio_16k, target_sr, subtype='PCM_16')
                        count += 1
                    except Exception as e:
                        print(f"Error processing sample {raw_wav}: {e}")

    print(f"Successfully generated {count} standardized 16 kHz mono WAV samples for '{keyword}'.")
    return count

if __name__ == "__main__":
    synthesize_custom_keyword()
