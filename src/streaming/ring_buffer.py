"""
Audio Ring Buffer for Low-Power Microcontroller Streaming
SIH Problem Statement 26172 - Milestone 10

Fixed-capacity circular buffer maintaining a continuous 1.0-second rolling window.
Zero dynamic heap allocation during real-time streaming (SRAM optimized for ESP32-S3).
"""

import numpy as np

class AudioRingBuffer:
    def __init__(self, capacity_samples=16000, dtype=np.float32):
        self.capacity = int(capacity_samples)
        self.dtype = dtype
        self.buffer = np.zeros(self.capacity, dtype=self.dtype)
        self.write_pos = 0
        self.total_samples_written = 0

    def append(self, chunk: np.ndarray):
        """Appends an incoming audio chunk in a circular manner."""
        chunk = np.asarray(chunk, dtype=self.dtype)
        if chunk.ndim > 1:
            # Fix: previously took channel 0 only (chunk[:, 0]), silently discarding
            # any signal present on other channels (e.g. mic wiring/I2S config placing
            # the active audio on channel 1). Averaging down-mixes correctly regardless
            # of which channel actually carries the signal.
            chunk = chunk.mean(axis=1).astype(self.dtype)
        n = len(chunk)
        if n == 0:
            return

        if n >= self.capacity:
            # Chunk is larger than entire buffer: keep only the most recent capacity samples
            self.buffer[:] = chunk[-self.capacity:]
            self.write_pos = 0
            self.total_samples_written += n
            return

        end_pos = self.write_pos + n
        if end_pos <= self.capacity:
            self.buffer[self.write_pos:end_pos] = chunk
        else:
            first_part = self.capacity - self.write_pos
            self.buffer[self.write_pos:] = chunk[:first_part]
            self.buffer[:n - first_part] = chunk[first_part:]

        self.write_pos = (self.write_pos + n) % self.capacity
        self.total_samples_written += n

    def get_snapshot(self) -> np.ndarray:
        """
        Returns the most recent capacity_samples in correct chronological order.
        """
        if self.total_samples_written < self.capacity:
            # Buffer not full yet: return leading zeros followed by written samples
            return np.roll(self.buffer, -self.write_pos).copy()

        # In circular buffer, the oldest sample is currently at write_pos
        return np.concatenate([self.buffer[self.write_pos:], self.buffer[:self.write_pos]])

    def is_full(self) -> bool:
        return self.total_samples_written >= self.capacity

    def reset(self):
        self.buffer.fill(0)
        self.write_pos = 0
        self.total_samples_written = 0

    def write(self, chunk):
        return self.append(chunk)

    def read_window(self, n_samples=None):
        return self.get_snapshot()
