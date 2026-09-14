"""Run the trained Tiny CNN encoder on a WAV file or Pi microphone."""

import argparse
import time

import numpy as np

from model_runtime import embed_audio, load_audio, load_current_keyword, load_encoder


def run_file(path, threshold):
    encoder = load_encoder()
    keyword, prototype = load_current_keyword()
    score = float(np.dot(embed_audio(encoder, load_audio(path)), prototype))
    result = "DETECTED" if score >= threshold else "not detected"
    print(f"Keyword: {keyword} | similarity: {score:.4f} | {result}")


def run_microphone(threshold):
    import sounddevice as sd

    encoder = load_encoder()
    keyword, prototype = load_current_keyword()
    print(f"Listening for '{keyword}'. Press Ctrl+C to stop.")
    try:
        while True:
            audio = sd.rec(16000, samplerate=16000, channels=1, dtype="float32")
            sd.wait()
            score = float(np.dot(embed_audio(encoder, audio[:, 0]), prototype))
            if score >= threshold:
                print(f"DETECTED '{keyword}' | similarity: {score:.4f}")
            else:
                print(f"similarity: {score:.4f}")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopped.")


def main():
    parser = argparse.ArgumentParser(description="Run the trained keyword embedding model")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wav", help="Run inference on one WAV file")
    source.add_argument("--mic", action="store_true", help="Listen through the Pi microphone")
    parser.add_argument("--threshold", type=float, default=0.88)
    args = parser.parse_args()
    if args.wav:
        run_file(args.wav, args.threshold)
    else:
        run_microphone(args.threshold)


if __name__ == "__main__":
    main()
