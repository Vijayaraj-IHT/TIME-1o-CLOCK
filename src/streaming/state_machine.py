"""
Detection State Machine with Hysteresis and Temporal Smoothing
SIH Problem Statement 26172 - Milestone 10

Implements the deterministic activation lifecycle:
States: LISTENING -> VERIFYING -> ACTIVATED -> COOLDOWN
Dual-threshold hysteresis (tau_high = 0.89, tau_low = 0.84)
Confirmation persistence: 4 consecutive frames
Temporal smoothing window: 8 frames (~400ms)
Cooldown lockout: 1500 ms
Phase-1 Change 5: adaptive background-tracking threshold (upward-only)
Phase-1 Change 6: keyword-END -> trigger latency marker
Phase-1 Change 3: score/lifecycle hygiene -- 3a observed-flag (VAD-silence
frames decay smoothing but never teach the background model), 3b history
reset on activation / cooldown-exit / stream-discontinuity gap.
"""

from collections import deque
import numpy as np

from src.evaluation.latency import MARKER_FRESHNESS_MS

class DetectionState:
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    VERIFYING = "VERIFYING"
    ACTIVATED = "ACTIVATED"
    COOLDOWN = "COOLDOWN"

class DetectionStateMachine:
    def __init__(self, threshold=0.89, hysteresis=0.05, consecutive_windows=4,
                 smoothing_window=8, cooldown_ms=1500,
                 adaptive=True, adapt_alpha=0.02, adapt_k=2.0,
                 adapt_max=0.95, adapt_warmup_frames=40, stale_gap_ms=500):
        self.threshold_high = float(threshold)
        self.threshold_low = float(threshold - hysteresis)
        self.consecutive_windows = int(consecutive_windows)
        self.smoothing_window_len = int(smoothing_window)
        self.cooldown_ms = float(cooldown_ms)

        # Phase-1 Change 5: adaptive background-tracking threshold. Defaults MUST
        # match configs/config.yaml -> detection.adapt_* (guarded by
        # tests/test_operating_point.py). Set adaptive=False to restore fixed-tau.
        self.adaptive = bool(adaptive)
        self.adapt_alpha = float(adapt_alpha)
        self.adapt_k = float(adapt_k)
        self.adapt_max = float(adapt_max)
        self.adapt_warmup_frames = int(adapt_warmup_frames)
        self._bg_mean = 0.0
        self._bg_ex2 = 0.0  # EMA of x^2; var = max(0, E[x^2] - mean^2)
        self._adapt_frames = 0
        self.threshold_eff = float(threshold)

        # Phase-1 Change 3b: stream-discontinuity reset threshold. Default
        # MUST match configs/config.yaml -> detection.stale_gap_ms (guarded
        # by tests/test_operating_point.py, incl. the firmware literal).
        self.stale_gap_ms = float(stale_gap_ms)
        self._last_ts = None

        self.state = DetectionState.LISTENING
        self.similarity_history = deque(maxlen=self.smoothing_window_len)
        self.consecutive_count = 0
        self.cooldown_until_ms = 0.0
        self.last_activation_timestamp_ms = None
        self.activation_count = 0
        # Phase-1 Change 6: ground-truth keyword-end marker (bench/sim only).
        self._kw_end_ms = None

    def _update_background(self, raw: float) -> None:
        """EMA update of background score stats. Call ONLY from LISTENING on
        non-triggering frames, so keyword frames can never inflate (and
        self-suppress via) the background estimate."""
        a = self.adapt_alpha
        self._bg_mean += a * (raw - self._bg_mean)
        self._bg_ex2 += a * (raw * raw - self._bg_ex2)
        self._adapt_frames += 1

    def _effective_threshold(self) -> float:
        """Upward-only adaptive threshold: never below calibrated tau (TPR-safe),
        rises in confusing/loud background (FA-cutting), capped at adapt_max."""
        if not self.adaptive or self._adapt_frames < self.adapt_warmup_frames:
            return self.threshold_high
        var = self._bg_ex2 - self._bg_mean * self._bg_mean
        std = float(np.sqrt(max(0.0, var)))
        tau_eff = self._bg_mean + self.adapt_k * std
        return max(self.threshold_high, min(self.adapt_max, tau_eff))

    def mark_keyword_end(self, timestamp_ms: float) -> None:
        """Change 6 (bench/sim only): record a ground-truth keyword-end
        timestamp so the next activation reports keyword-END -> trigger
        latency in trigger_latency_ms."""
        self._kw_end_ms = float(timestamp_ms)

    def process_similarity(self, raw_similarity: float, timestamp_ms: float,
                           observed: bool = True) -> dict:
        """
        Updates the state machine with the latest cosine similarity score.
        Returns state dictionary with trigger events.
        """
        raw = float(raw_similarity)
        observed = bool(observed)
        # Change 3b: stream-discontinuity reset. A gap (or a clock jump
        # backwards) means similarity_history holds evidence about audio
        # that is long gone (paused stream, ASR-handover freeze, stalled
        # consumer). Clear the FAST state so this frame starts clean. The
        # SLOW background stats are kept (same room, still valid) and the
        # Change-6 marker is kept (its own freshness window judges it).
        if (self._last_ts is not None
                and (timestamp_ms < self._last_ts
                     or (timestamp_ms - self._last_ts) > self.stale_gap_ms)):
            self.state = DetectionState.LISTENING
            self.consecutive_count = 0
            self.similarity_history.clear()
        self._last_ts = float(timestamp_ms)
        # Change 3b: cooldown expiry is a lifecycle transition like the gap
        # reset above: re-arm LISTENING with a clean history BEFORE this
        # frame's evidence is appended, so the exit frame decides on fresh
        # evidence only (cooldown frames are post-trigger audio, not evidence
        # for a NEW trigger).
        if (self.state == DetectionState.COOLDOWN
                and timestamp_ms >= self.cooldown_until_ms):
            self.state = DetectionState.LISTENING
            self.consecutive_count = 0
            self.similarity_history.clear()
        self.similarity_history.append(raw)
        smoothed_sim = float(np.mean(self.similarity_history))

        # Decide with the stats learned from PRIOR frames (no same-frame feedback).
        self.threshold_eff = self._effective_threshold()
        is_activated = False
        trigger_latency_ms = None

        # State Handling (cooldown expiry was handled above, pre-append, so a
        # frame arriving here in COOLDOWN is still locked out: frozen, no-op).
        if self.state == DetectionState.LISTENING:
            if smoothed_sim >= self.threshold_eff:
                self.state = DetectionState.VERIFYING
                self.consecutive_count = 1
            else:
                self.consecutive_count = 0
                # Change 3a: VAD-silence frames are NOT model observations
                # (no inference ran). The 0.0 appended above still decays
                # the smoothing average, but it must not teach the
                # background model, or silence dilutes tau_eff upward
                # adaptation in noisy rooms.
                if self.adaptive and observed:
                    self._update_background(raw)

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
                    # Change 6: keyword-END -> trigger latency vs ground-truth
                    # marker. Negative = early trigger (fired before the keyword
                    # ended: legal, reported signed). Stale/absent marker ->
                    # None, never a bogus number. One marker, one trigger.
                    if self._kw_end_ms is not None:
                        dt = float(timestamp_ms) - float(self._kw_end_ms)
                        if -MARKER_FRESHNESS_MS <= dt <= MARKER_FRESHNESS_MS:
                            trigger_latency_ms = dt
                        self._kw_end_ms = None
                    # Transition immediately to COOLDOWN to prevent double triggers
                    self.state = DetectionState.COOLDOWN
                    # Change 3b: drop the evidence that fired this trigger.
                    # consecutive_count is intentionally left for the frame
                    # report below (it proves persistence was met); every
                    # LISTENING/VERIFYING frame overwrites it before
                    # reading, so no explicit reset is needed.
                    self.similarity_history.clear()
            else:
                # Similarity dipped below tau_low before confirmation -> false alarm rejected
                self.state = DetectionState.LISTENING
                self.consecutive_count = 0

        return {
            "state": self.state,
            "timestamp_ms": timestamp_ms,
            "raw_similarity": round(raw_similarity, 4),
            "observed": observed,
            "smoothed_similarity": round(smoothed_sim, 4),
            "consecutive_count": self.consecutive_count,
            "is_activated": is_activated,
            "threshold_high": self.threshold_high,
            "threshold_low": self.threshold_low,
            "threshold_eff": round(self.threshold_eff, 4),
            "trigger_latency_ms": (round(trigger_latency_ms, 2)
                                   if trigger_latency_ms is not None else None)
        }

    def reset(self):
        self.state = DetectionState.LISTENING
        self.similarity_history.clear()
        self.consecutive_count = 0
        self.cooldown_until_ms = 0.0
        self._bg_mean = 0.0
        self._bg_ex2 = 0.0
        self._adapt_frames = 0
        self.threshold_eff = float(self.threshold_high)
        self._kw_end_ms = None
        self._last_ts = None

    def update(self, raw_similarity: float, timestamp_ms: float, observed: bool = True):
        """Compatibility alias returning an object with .triggered and .state."""
        res = self.process_similarity(raw_similarity, timestamp_ms, observed=observed)
        class Event:
            pass
        ev = Event()
        ev.triggered = res["is_activated"]
        ev.state = res["state"]
        ev.smoothed_similarity = res["smoothed_similarity"]
        ev.threshold_eff = res["threshold_eff"]
        ev.trigger_latency_ms = res["trigger_latency_ms"]
        ev.observed = res["observed"]
        return ev
