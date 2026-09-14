"""Receive laptop audio and run low-duty-cycle TFLite keyword detection."""

import argparse
import socket
import struct

import numpy as np

from model_runtime import AUDIO_SAMPLES, FRAME_SAMPLES, embed_audio, load_current_keyword, load_encoder

FRAME_BYTES = FRAME_SAMPLES * np.dtype(np.float32).itemsize
HEADER = struct.Struct("!I")


def receive_exact(connection, size):
    data = bytearray()
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def detect_stream(connection, threshold, interval_frames, persistence):
    interpreter = load_encoder(num_threads=2)
    keyword, prototype = load_current_keyword()
    audio_window = np.zeros(AUDIO_SAMPLES, dtype=np.float32)
    frame_count = 0
    positive_frames = 0
    print(f"Listening for '{keyword}' with TFLite runtime.")

    while True:
        header = receive_exact(connection, HEADER.size)
        if header is None:
            print("Laptop disconnected.")
            return
        frame_size = HEADER.unpack(header)[0]
        if frame_size != FRAME_BYTES:
            raise ValueError(f"Unexpected audio frame size: {frame_size}")
        raw_frame = receive_exact(connection, frame_size)
        if raw_frame is None:
            return
        frame = np.frombuffer(raw_frame, dtype=np.float32)
        audio_window[:-FRAME_SAMPLES] = audio_window[FRAME_SAMPLES:]
        audio_window[-FRAME_SAMPLES:] = frame
        frame_count += 1

        # Skip silence and run inference every interval_frames to reduce CPU load.
        rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
        if rms < 0.010 or frame_count % interval_frames:
            continue

        score = float(np.dot(embed_audio(interpreter, audio_window), prototype))
        positive_frames = positive_frames + 1 if score >= threshold else 0
        if positive_frames >= persistence:
            print(f"DETECTED: '{keyword}' similarity={score:.4f}", flush=True)
            positive_frames = 0
        else:
            print(f"similarity={score:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="TFLite Raspberry Pi keyword server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--threshold", type=float, default=0.88)
    parser.add_argument("--interval-frames", type=int, default=2)
    parser.add_argument("--persistence", type=int, default=2)
    args = parser.parse_args()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(f"Waiting for laptop audio on {args.host}:{args.port}...")
        while True:
            connection, address = server.accept()
            print(f"Connected to {address[0]}:{address[1]}")
            with connection:
                try:
                    detect_stream(connection, args.threshold, args.interval_frames, args.persistence)
                except (ConnectionError, ValueError) as error:
                    print(f"Stream stopped: {error}")


if __name__ == "__main__":
    main()
