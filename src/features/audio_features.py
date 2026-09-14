"""
Base Audio Feature Extraction Framework
SIH Problem Statement 26172
"""

import abc
import numpy as np

class BaseAudioFeatureExtractor(abc.ABC):
    """
    Abstract base class for streaming and offline audio feature extractors.
    """
    def __init__(self, sample_rate=16000, win_length=480, hop_length=160, fft_length=512,
                 target_duration_s=1.0):
        self.sample_rate = sample_rate
        self.win_length = win_length      # 30 ms at 16 kHz
        self.hop_length = hop_length      # 10 ms at 16 kHz
        self.fft_length = fft_length      # 512 point FFT
        self.window = np.hanning(win_length).astype(np.float32)
        # Fix: every downstream consumer (enroll.py, quantize.py, the ESP32 firmware's
        # fixed-shape input tensor) assumes exactly 98 frames from a 1.0s/16kHz clip.
        # Previously only enroll.py enforced this length; any other caller (streaming
        # detector output, a future script) could pass audio of a different length and
        # get a different frame count with NO error until the TFLite model's fixed
        # input tensor rejects it at Invoke() time. Enforced once, here, for everyone.
        self.target_num_samples = int(round(sample_rate * target_duration_s))

    @abc.abstractmethod
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extracts spectral features from a raw 1D audio waveform.
        Returns: 2D feature matrix of shape (time_frames, feature_dim).
        """
        pass

    def frame_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Slices audio into overlapping windowed frames.
        Always standardizes to self.target_num_samples first, guaranteeing a constant
        frame count regardless of the caller's input length.
        """
        audio = np.ascontiguousarray(audio, dtype=np.float32)  # Fix: as_strided requires
        # a real, contiguous buffer - a view (e.g. from np.roll/a ring-buffer slice)
        # previously risked reading incorrect memory with no error raised.

        num_samples = len(audio)
        target = self.target_num_samples
        if num_samples < target:
            audio = np.pad(audio, (0, target - num_samples), mode='constant')
        elif num_samples > target:
            audio = audio[:target]
        num_samples = target

        if num_samples < self.win_length:
            audio = np.pad(audio, (0, self.win_length - num_samples), mode='constant')
            num_samples = len(audio)

        num_frames = 1 + (num_samples - self.win_length) // self.hop_length
        shape = (num_frames, self.win_length)
        strides = (audio.strides[0] * self.hop_length, audio.strides[0])
        frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
        return frames * self.window
