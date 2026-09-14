"""
MFCC (Mel-Frequency Cepstral Coefficients) Feature Extractor
SIH Problem Statement 26172
"""

import numpy as np
from scipy.fftpack import dct
from src.features.logmel import LogMelFeatureExtractor

class MFCCFeatureExtractor(LogMelFeatureExtractor):
    """
    Extracts 13-coefficient MFCC features for low-power edge microcontrollers.
    """
    def __init__(self, sample_rate=16000, win_length=480, hop_length=160, fft_length=512, n_mels=40, n_mfcc=13, fmin=20, fmax=4000):
        super().__init__(sample_rate, win_length, hop_length, fft_length, n_mels, fmin, fmax)
        self.n_mfcc = n_mfcc

        # Precompute DCT-II orthogonal projection matrix (n_mfcc, n_mels)
        n = np.arange(n_mels)
        k = np.arange(n_mfcc)[:, None]
        self.dct_basis = np.cos(np.pi * k * (2 * n + 1) / (2 * n_mels)) * np.sqrt(2.0 / n_mels)
        self.dct_basis[0] *= 1.0 / np.sqrt(2.0)  # Orthogonal normalization for DC term

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extracts MFCC features of shape (num_frames, n_mfcc).
        """
        # 1. Compute Log-Mel filterbank energies
        log_mel = super().extract(audio)  # Shape: (num_frames, n_mels)

        # 2. Apply Type-II Discrete Cosine Transform
        mfcc = np.dot(log_mel, self.dct_basis.T)
        return mfcc.astype(np.float32)
