"""
Audio Data Augmentation Pipeline
SIH Problem Statement 26172
"""

import numpy as np

class AudioAugmenter:
    """
    Configurable audio data augmenter for TinyML training robustness.
    """
    def __init__(self, sample_rate=16000, seed=42):
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(seed)

    def add_noise(self, audio, snr_db_low=10, snr_db_high=30):
        """Adds white Gaussian noise at random SNR."""
        snr_db = self.rng.uniform(snr_db_low, snr_db_high)
        signal_power = np.mean(audio ** 2)
        if signal_power == 0:
            return audio
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = self.rng.normal(0, np.sqrt(noise_power), len(audio)).astype(np.float32)
        return audio + noise

    def time_shift(self, audio, max_shift_ms=100):
        """
        Randomly time-shifts audio within +/- max_shift_ms, zero-padding the
        vacated region instead of wrapping.

        Fix: previously used np.roll, a *circular* shift. On a 1s clip where a short
        keyword occupies most of the window, this can wrap the word-final phoneme
        around to sample 0 - producing a temporally corrupted example that still
        carries the original class label. This directly pollutes metric learning
        (SupervisedContrastiveLoss pulls it toward clean same-class exemplars) and is
        especially damaging for few-shot enrollment, where K=1..3 raw-mean prototypes
        are highly sensitive to a single bad exemplar.
        """
        max_shift = int(self.sample_rate * (max_shift_ms / 1000.0))
        if max_shift <= 0:
            return audio
        shift = int(self.rng.integers(-max_shift, max_shift + 1))
        shifted = np.zeros_like(audio)
        if shift > 0:
            # Shift right: silence leads, tail is truncated (not wrapped to the front).
            shifted[shift:] = audio[: len(audio) - shift]
        elif shift < 0:
            # Shift left: silence trails, head is truncated (not wrapped to the back).
            shifted[: len(audio) + shift] = audio[-shift:]
        else:
            shifted = audio.copy()
        return shifted

    def apply_gain(self, audio, min_gain_db=-6, max_gain_db=6):
        """Applies random gain variation."""
        gain_db = self.rng.uniform(min_gain_db, max_gain_db)
        factor = 10 ** (gain_db / 20.0)
        return audio * factor

    def augment(self, audio, p_noise=0.5, p_shift=0.5, p_gain=0.5):
        """Applies random chain of augmentations."""
        augmented = audio.copy()
        if self.rng.random() < p_shift:
            augmented = self.time_shift(augmented)
        if self.rng.random() < p_noise:
            augmented = self.add_noise(augmented)
        if self.rng.random() < p_gain:
            augmented = self.apply_gain(augmented)
        return np.clip(augmented, -1.0, 1.0)
