#!/usr/bin/env python3
"""
Phase-1 Change 6: keyword-END -> trigger latency benchmark (synthetic).

SIH-26172 scores latency from keyword END to ASR handover. This bench measures
the detection side (keyword END -> ACTIVATED event) through the REAL
StreamingVoiceActivator (VAD + MFCC + state machine) with a scripted encoder,
so it runs anywhere with numpy/scipy -- no audio files, no TF, no hardware.

Scenarios (50ms chunks, default operating point tau=0.89/smooth=8/persist=4):
  clean_single   silence bg + one 800ms keyword      -> expect 1 trigger
  noisy_bg       loud confuser bg + one 800ms keyword -> expect 1 trigger (tau_eff raised)
  short_keyword  silence bg + one 400ms keyword      -> expect 0 triggers (negative control:
                                                   documents ~600ms+ minimum keyword length)
  double_keyword silence bg + two 800ms keywords    -> expect 2 triggers

Writes experiments/threshold/detection_latency_results.json and prints a table.
Exit 0 always (this is a measurement bench, not a gate); verdict lives in JSON.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.streaming.detector import StreamingVoiceActivator
from src.evaluation.latency import LatencyStats

SR = 16000
CHUNK = 800  # 50ms
KW_COS = 0.99


def _tonegen(freq_hz, amp, n, rng=None, noise_std=0.0):
    t = np.arange(n, dtype=np.float32) / SR
    x = (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    if noise_std and rng is not None:
        x = x + rng.normal(0.0, noise_std, n).astype(np.float32)
    return x


def _emb_with_cosine(proto, c, rng):
    r = rng.randn(*proto.shape).astype(np.float32)
    r = r - float(np.dot(r, proto)) * proto
    r /= (np.linalg.norm(r) + 1e-9)
    v = c * proto + np.sqrt(max(0.0, 1.0 - c * c)) * r
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


class ScriptedEncoder:
    """Encoder keyed by chunk index (bench sets it before each process_chunk,
    so VAD-skipped chunks cannot desync the script)."""

    def __init__(self, embs):
        self.embs = embs
        self.i = 0

    def set_chunk(self, i):
        self.i = i

    def __call__(self, x, training=False):
        e = self.embs[self.i]

        class _Out:
            def numpy(self):
                return np.array([e])

        return _Out()


def build_scenario(name, proto, rng):
    """Returns (chunks, embs, markers_ms, expected_triggers)."""
    sil = np.zeros(CHUNK, dtype=np.float32)
    kw_chunk = _tonegen(440.0, 0.30, CHUNK)
    loud_bg = _tonegen(300.0, 0.25, CHUNK, rng, 0.01)
    kw_emb = _emb_with_cosine(proto, KW_COS, rng)
    bg_low = _emb_with_cosine(proto, 0.15, rng)
    # confuser bg: alternating 0.87/0.79 cosine (mean 0.83, std 0.04 ->
    # learned tau_eff ~0.91; smoothed ~0.83 stays below tau: no false trigger)
    bg_hi = [_emb_with_cosine(proto, 0.87, rng), _emb_with_cosine(proto, 0.79, rng)]

    chunks, embs, markers = [], [], []

    def add(n, chunk, emb):
        for _ in range(n):
            chunks.append(chunk)
            embs.append(emb)

    def add_bg_hi(n):
        for k in range(n):
            chunks.append(loud_bg)
            embs.append(bg_hi[k % 2])

    def add_keyword(n_chunks=16):
        # Marker = ground-truth keyword END, but it must be PLACED at keyword
        # onset: the sim knows the end time in advance, and early triggers
        # (which fire mid-keyword) must still find their marker.
        onset = len(chunks) * 50.0
        add(n_chunks, kw_chunk, kw_emb)
        markers.append((onset, onset + n_chunks * 50.0))  # (place_at, kw_end)

    if name == "clean_single":
        add(40, sil, bg_low)
        add_keyword(16)
        add(40, sil, bg_low)
        exp = 1
    elif name == "noisy_bg":
        add_bg_hi(120)  # 6s: lets the background EMA converge
        add_keyword(16)
        add_bg_hi(40)
        exp = 1
    elif name == "short_keyword":
        add(40, sil, bg_low)
        add_keyword(8)  # 400ms: below the persistence horizon
        add(40, sil, bg_low)
        exp = 0
    elif name == "double_keyword":
        add(40, sil, bg_low)
        add_keyword(16)
        add(80, sil, bg_low)  # 4s gap > 1500ms cooldown
        add_keyword(16)
        add(40, sil, bg_low)
        exp = 2
    else:
        raise ValueError(name)
    return chunks, embs, markers, exp


def run_scenario(name, proto):
    rng = np.random.RandomState(11)
    chunks, embs, markers, expected = build_scenario(name, proto, rng)
    det = StreamingVoiceActivator(encoder=ScriptedEncoder(embs),
                                  target_prototype=proto)
    enc = det.encoder
    markers = list(markers)
    triggers = []
    tau_max = 0.0
    for i, ch in enumerate(chunks):
        ts = i * 50.0
        enc.set_chunk(i)
        while markers and ts >= markers[0][0]:
            det.mark_keyword_end(markers.pop(0)[1])
        r = det.process_chunk(ch, ts)
        tau_max = max(tau_max, r["threshold_eff"])
        if r["is_activated"]:
            triggers.append({"timestamp_ms": ts,
                             "trigger_latency_ms": r["trigger_latency_ms"],
                             "tau_eff_at_trigger": r["threshold_eff"]})
    st = LatencyStats()
    for t in triggers:
        st.add(t["trigger_latency_ms"])
    return {"scenario": name,
            "chunks": len(chunks),
            "expected_triggers": expected,
            "triggers": triggers,
            "tau_eff_max": round(tau_max, 4),
            "latency_stats": st.summary(),
            "count_match": len(triggers) == expected}


def main():
    rng = np.random.RandomState(3)
    proto = rng.randn(32).astype(np.float32)
    proto /= np.linalg.norm(proto)
    results = [run_scenario(n, proto) for n in
               ("clean_single", "noisy_bg", "short_keyword", "double_keyword")]
    all_match = all(r["count_match"] for r in results)
    all_verdicts = [r["latency_stats"]["verdict"] for r in results
                    if r["latency_stats"]["n"] > 0]
    verdict = ("PASS" if all_match and all(v == "PASS" for v in all_verdicts)
               else "FAIL")
    out = {"operating_point": {"threshold": 0.89, "smoothing_window": 8,
                               "consecutive_windows": 4, "adaptive": True},
           "scenarios": results, "verdict": verdict}
    out_path = os.path.join(os.path.dirname(__file__),
                             "detection_latency_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"{'scenario':<14}{'exp':>4}{'got':>4}{'tau_max':>8}  "
          f"{'latencies_ms':<24}verdict")
    for r in results:
        lats = [t["trigger_latency_ms"] for t in r["triggers"]]
        print(f"{r['scenario']:<14}{r['expected_triggers']:>4}{len(lats):>4}"
              f"{r['tau_eff_max']:>8}  {str(lats):<24}"
              f"{r['latency_stats']['verdict']}")
    print(f"OVERALL: {verdict} -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
