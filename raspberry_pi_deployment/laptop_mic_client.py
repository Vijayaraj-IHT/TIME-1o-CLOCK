"""Laptop client: capture its microphone and stream audio to the Raspberry Pi."""

import argparse
import socket
import struct

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
FRAME_SAMPLES = 800
HEADER = struct.Struct("!I")


def main():
    parser = argparse.ArgumentParser(description="Stream laptop microphone audio to Raspberry Pi")
    parser.add_argument("--pi-host", required=True, help="Raspberry Pi hotspot IP address")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print(f"Connecting to Raspberry Pi at {args.pi_host}:{args.port}...")
    with socket.create_connection((args.pi_host, args.port)) as connection:
        print("Streaming laptop microphone audio. Press Ctrl+C to stop.")
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=FRAME_SAMPLES,
            ) as microphone:
                while True:
                    audio, _ = microphone.read(FRAME_SAMPLES)
                    frame = np.asarray(audio[:, 0], dtype=np.float32).tobytes()
                    connection.sendall(HEADER.pack(len(frame)) + frame)
        except KeyboardInterrupt:
            print("\nMicrophone stream stopped.")


if __name__ == "__main__":
    main()
