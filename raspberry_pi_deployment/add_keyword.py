"""Enroll a new keyword into the portable Raspberry Pi package."""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from model_runtime import KEYWORD_DIR, load_encoder, embed_audio, save_current_keyword


def record_samples(keyword, count, output_dir):
    import sounddevice as sd

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(1, count + 1):
        input(f"Press Enter, then speak '{keyword}' ({index}/{count})...")
        recording = sd.rec(16000, samplerate=16000, channels=1, dtype="float32")
        sd.wait()
        path = output_dir / f"{keyword.lower()}_{index}.wav"
        sf.write(path, recording[:, 0], 16000)
        paths.append(path)
    return paths


def collect_samples(keyword, audio_dir, count):
    paths = sorted(Path(audio_dir).glob("*.wav"))[:count]
    if len(paths) < count:
        raise ValueError(f"Expected {count} WAV files in {audio_dir}, found {len(paths)}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Enroll a custom keyword without retraining")
    parser.add_argument("--keyword", required=True, help="Keyword to enroll, for example AGNI")
    parser.add_argument("--audio-dir", help="Directory containing recorded WAV files")
    parser.add_argument("--mic", action="store_true", help="Record samples from the Pi microphone")
    parser.add_argument("--shots", type=int, default=3, help="Number of enrollment samples")
    args = parser.parse_args()

    if bool(args.audio_dir) == args.mic:
        parser.error("Choose exactly one of --audio-dir or --mic")

    keyword = args.keyword.upper()
    sample_dir = KEYWORD_DIR / keyword.lower()
    paths = record_samples(keyword, args.shots, sample_dir) if args.mic else collect_samples(keyword, args.audio_dir, args.shots)

    encoder = load_encoder()
    embeddings = np.asarray([embed_audio(encoder, sf.read(path, dtype="float32")[0]) for path in paths])
    prototype = np.mean(embeddings, axis=0)
    prototype /= np.linalg.norm(prototype) + 1e-9
    similarity = embeddings @ prototype
    if float(np.mean(similarity)) < 0.70:
        raise ValueError("Enrollment rejected: recordings are inconsistent. Record the keyword more clearly.")

    save_current_keyword(keyword, prototype, paths)
    print(f"Enrolled {keyword} using {len(paths)} samples.")
    print(f"Active keyword file: {KEYWORD_DIR / 'current_keyword.json'}")


if __name__ == "__main__":
    main()
