"""
Phase-1 Change 6: keyword-END -> trigger latency telemetry.

SIH-26172 scores latency from keyword END to ASR handover. The detection side
of that budget (keyword END -> ACTIVATED event) previously had NO measurement
anywhere: per-chunk inference ms existed, but detection delay did not.

- DetectionStateMachine.mark_keyword_end(ts): bench/sim marks ground-truth
  keyword-end timestamps; the next activation reports trigger_latency_ms.
- LatencyStats: aggregates signed latencies (negative = early trigger, i.e.
  the detector fired before the keyword finished) and judges the positive
  (late) side against LATENCY_BUDGET_MS.
"""
from __future__ import annotations

import numpy as np

# Detection-latency budget: keyword END -> ACTIVATED event must land within
# 1s. (Smoothing(8) + persistence(4) at 50ms/chunk already account for ~0.6s
# of structural delay; the budget leaves headroom for VAD + inference.)
LATENCY_BUDGET_MS = 1000.0

# A keyword-end marker older than this at trigger time is stale (belongs to a
# previous utterance) and is ignored instead of producing a bogus latency.
MARKER_FRESHNESS_MS = 5000.0


class LatencyStats:
    """Accumulates signed trigger latencies and reports percentiles + verdict."""

    def __init__(self, budget_ms: float = LATENCY_BUDGET_MS):
        self.budget_ms = float(budget_ms)
        self._lat = []

    def add(self, latency_ms) -> None:
        if latency_ms is None:
            return
        self._lat.append(float(latency_ms))

    def __len__(self):
        return len(self._lat)

    def summary(self) -> dict:
        n = len(self._lat)
        if n == 0:
            return {
                "n": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None,
                "p99_ms": None, "max_ms": None, "min_ms": None,
                "early_triggers": 0, "late_triggers": 0,
                "budget_ms": self.budget_ms, "max_within_budget": None,
                "verdict": "NO_DATA",
            }
        a = np.asarray(self._lat, dtype=np.float64)
        late = a[a >= 0.0]
        max_late = float(np.max(late)) if late.size else 0.0
        within = bool(max_late <= self.budget_ms)
        return {
            "n": n,
            "mean_ms": round(float(np.mean(a)), 2),
            "p50_ms": round(float(np.percentile(a, 50)), 2),
            "p95_ms": round(float(np.percentile(a, 95)), 2),
            "p99_ms": round(float(np.percentile(a, 99)), 2),
            "max_ms": round(float(np.max(a)), 2),
            "min_ms": round(float(np.min(a)), 2),
            "early_triggers": int(np.sum(a < 0.0)),
            "late_triggers": int(late.size),
            "budget_ms": self.budget_ms,
            "max_within_budget": within,
            "verdict": "PASS" if within else "FAIL",
        }
