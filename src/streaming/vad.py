"""
Energy Voice Activity Detector (VAD) for Edge Inference Gating
SIH Problem Statement 26172 - Milestone 10

Ultra-lightweight time-domain speech detector:
1. Calculates RMS energy and Zero Crossing Rate (ZCR)
2. Tracks dynamic background noise floor
3. Incorporates hangover smoothing to preserve word endings
4. Skips heavy CNN inference during silence to enforce < 10% idle CPU utilization
"""

import numpy as np

class EnergyVAD:
    def __init__(self, sample_rate=16000, min_energy_threshold=0.005,
                 energy_multiplier=2.5, hangover_frames=3,
                 max_background_energy=0.05, stuck_speech_limit_frames=200):
        self.sample_rate = sample_rate
        self.min_energy_threshold = float(min_energy_threshold)
        self.energy_multiplier = float(energy_multiplier)
        self.hangover_frames = int(hangover_frames)

        # Fix (VAD deadlock): background_energy previously only adapted in the
        # "rms < threshold" branch. If real ambient noise at boot already exceeded
        # min_energy_threshold * energy_multiplier, is_speech() returned True forever,
        # background_energy never updated, and full CNN inference ran continuously -
        # violating the <10% idle CPU budget with no recovery path short of reset().
        # Two independent safeguards now break that deadlock:
        self.max_background_energy = float(max_background_energy)  # hard ceiling on adaptation
        self.stuck_speech_limit_frames = int(stuck_speech_limit_frames)  # forced recalibration trip

        # Dynamic background noise floor tracker
        self.background_energy = self.min_energy_threshold
        self.hangover_counter = 0
        self._consecutive_speech_frames = 0

    def compute_energy_and_zcr(self, chunk: np.ndarray):
        """Computes RMS energy and Zero-Crossing Rate."""
        if len(chunk) == 0:
            return 0.0, 0.0
        # RMS energy
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        # ZCR
        signs = np.sign(chunk)
        signs[signs == 0] = 1
        zcr = float(np.mean(np.abs(signs[1:] - signs[:-1])) / 2.0) if len(chunk) > 1 else 0.0
        return rms, zcr

    def is_speech(self, chunk: np.ndarray) -> bool:
        """
        Determines whether the incoming chunk contains speech.
        Updates dynamic noise floor during quiet periods.
        """
        rms, zcr = self.compute_energy_and_zcr(chunk)
        threshold = max(self.background_energy * self.energy_multiplier, self.min_energy_threshold)

        if rms >= threshold:
            # Fix (VAD deadlock, part 1): if "speech" has been continuously flagged for
            # stuck_speech_limit_frames in a row, it is far more likely to be a sustained
            # stationary noise source (fan, AC, engine) than continuous human speech.
            # Force-adopt the observed level as the new floor so adaptation can resume,
            # instead of burning full CNN inference forever with no way out.
            self._consecutive_speech_frames += 1
            if self._consecutive_speech_frames >= self.stuck_speech_limit_frames:
                self.background_energy = min(rms, self.max_background_energy)
                self._consecutive_speech_frames = 0
                self.hangover_counter = self.hangover_frames
                return False  # treat this frame as the recalibration point, not speech

            self.hangover_counter = self.hangover_frames
            return True

        # Silence / ambient noise: adapt background noise floor
        self._consecutive_speech_frames = 0
        alpha = 0.05
        new_bg = (1.0 - alpha) * self.background_energy + alpha * rms
        # Fix (VAD deadlock, part 2): hard ceiling prevents the floor from ever
        # adapting so high that a spoken keyword can no longer exceed it.
        self.background_energy = min(new_bg, self.max_background_energy)

        # Hangover decay
        if self.hangover_counter > 0:
            self.hangover_counter -= 1
            return True

        return False

    def reset(self):
        self.background_energy = self.min_energy_threshold
        self.hangover_counter = 0
        self._consecutive_speech_frames = 0
