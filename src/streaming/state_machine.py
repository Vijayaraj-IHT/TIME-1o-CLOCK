"""
Detection State Machine with Hysteresis and Temporal Smoothing
SIH Problem Statement 26172 - Milestone 10

Implements the deterministic activation lifecycle:
States: LISTENING -> VERIFYING -> ACTIVATED -> COOLDOWN
Dual-threshold hysteresis (tau_high = 0.78, tau_low = 0.73)
Confirmation persistence: 3 consecutive frames
Temporal smoothing window: 5 frames
Cooldown lockout: 1500 ms
"""

from collections import deque
import numpy as np

class DetectionState:
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    VERIFYING = "VERIFYING"
    ACTIVATED = "ACTIVATED"
    COOLDOWN = "COOLDOWN"

class DetectionStateMachine:
    def __init__(self, threshold=0.78, hysteresis=0.05, consecutive_windows=3,
                 smoothing_window=5, cooldown_ms=1500):
        self.threshold_high = float(threshold)
        self.threshold_low = float(threshold - hysteresis)
        self.consecutive_windows = int(consecutive_windows)
        self.smoothing_window_len = int(smoothing_window)
        self.cooldown_ms = float(cooldown_ms)

        self.state = DetectionState.LISTENING
        self.similarity_history = deque(maxlen=self.smoothing_window_len)
        self.consecutive_count = 0
        self.cooldown_until_ms = 0.0
        self.last_activation_timestamp_ms = None
        self.activation_count = 0

    def process_similarity(self, raw_similarity: float, timestamp_ms: float) -> dict:
        """
        Updates the state machine with the latest cosine similarity score.
        Returns state dictionary with trigger events.
        """
        self.similarity_history.append(float(raw_similarity))
        smoothed_sim = float(np.mean(self.similarity_history))

        is_activated = False

        # State Handling
        if self.state == DetectionState.COOLDOWN:
            if timestamp_ms >= self.cooldown_until_ms:
                self.state = DetectionState.LISTENING
                self.consecutive_count = 0

        if self.state == DetectionState.LISTENING:
            if smoothed_sim >= self.threshold_high:
                self.state = DetectionState.VERIFYING
                self.consecutive_count = 1
            else:
                self.consecutive_count = 0

        elif self.state == DetectionState.VERIFYING:
            if smoothed_sim >= self.threshold_low:
                self.consecutive_count += 1
                if self.consecutive_count >= self.consecutive_windows:
                    # Confirmation threshold met: trigger activation!
                    self.state = DetectionState.ACTIVATED
                    is_activated = True
                    self.activation_count += 1
                    self.last_activation_timestamp_ms = timestamp_ms
                    self.cooldown_until_ms = timestamp_ms + self.cooldown_ms
                    # Transition immediately to COOLDOWN to prevent double triggers
                    self.state = DetectionState.COOLDOWN
            else:
                # Similarity dipped below tau_low before confirmation -> false alarm rejected
                self.state = DetectionState.LISTENING
                self.consecutive_count = 0

        return {
            "state": self.state,
            "timestamp_ms": timestamp_ms,
            "raw_similarity": round(raw_similarity, 4),
            "smoothed_similarity": round(smoothed_sim, 4),
            "consecutive_count": self.consecutive_count,
            "is_activated": is_activated,
            "threshold_high": self.threshold_high,
            "threshold_low": self.threshold_low
        }

    def reset(self):
        self.state = DetectionState.LISTENING
        self.similarity_history.clear()
        self.consecutive_count = 0
        self.cooldown_until_ms = 0.0

    def update(self, raw_similarity: float, timestamp_ms: float):
        """Compatibility alias returning an object with .triggered and .state."""
        res = self.process_similarity(raw_similarity, timestamp_ms)
        class Event:
            pass
        ev = Event()
        ev.triggered = res["is_activated"]
        ev.state = res["state"]
        ev.smoothed_similarity = res["smoothed_similarity"]
        return ev
