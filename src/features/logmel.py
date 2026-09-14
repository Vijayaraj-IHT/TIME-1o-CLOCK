"""
Log-Mel Filterbank Feature Extractor
SIH Problem Statement 26172
"""

import numpy as np
import librosa
from src.features.audio_features import BaseAudioFeatureExtractor

class LogMelFeatureExtractor(BaseAudioFeatureExtractor):
    """
    Extracts Log-Mel spectrograms from raw 16 kHz audio waveforms.
    """
    def __init__(self, sample_rate=16000, win_length=480, hop_length=160, fft_length=512, n_mels=40, fmin=20, fmax=4000):
        super().__init__(sample_rate, win_length, hop_length, fft_length)
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or (sample_rate // 2)

        # Precompute Mel filterbank matrix (n_mels, fft_length // 2 + 1)
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=fft_length,
            n_mels=n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            htk=True,
            norm='slaney'
        ).astype(np.float32)

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extracts Log-Mel features of shape (num_frames, n_mels).
        """
        frames = self.frame_audio(audio)
        # Real FFT of windowed frames
        fft_complex = np.fft.rfft(frames, n=self.fft_length)
        power_spectrum = (np.abs(fft_complex) ** 2) / self.fft_length

        # Apply Mel filterbanks
        mel_energies = np.dot(power_spectrum, self.mel_basis.T)
        # Log compression with stabilization
        log_mel = np.log(np.maximum(mel_energies, 1e-6))
        return log_mel.astype(np.float32)
