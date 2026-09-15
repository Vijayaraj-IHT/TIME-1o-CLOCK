#!/usr/bin/env python3
"""Track-B pilot: DTW template spotter vs neural prototype detector.

Same MFCC-13 frontend, same enrollment clips, same test clips, same stream --
only the matcher differs. PILOT GRADE: 3 real keyword clips + synthetic
negatives; thresholds set by midpoint (illustrative, not calibrated). Re-run
on MEASURE-1-scale data for the real verdict (the harness is data-driven).

  python experiments/track_b/compare_dtw_vs_neural.py

Writes experiments/track_b/track_b_pilot_results.json and prints tables.
Exit 0 always (measurement, not a gate).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.features.mfcc import MFCCFeatureExtractor
from src.dtw.dtw_core import dtw_distance, lb_keogh, dtw_envelope
from src.dtw.templates import build_template, resample_frames
from src.dtw.spot import DTWSpotter

ZDIR = os.path.join("raspberry_pi_deployment", "custom_keywords", "zora")
BAND = 8
OUT = os.path.join(os.path.dirname(__file__), "track_b_pilot_results.json")


def main():
    import soundfile as sf
    mfcc = MFCCFeatureExtractor(sample_rate=16000, n_mfcc=13)

    def wav(name):
        a, sr = sf.read(os.path.join(ZDIR, name), dtype="float32")
        assert sr == 16000 and len(a) == 16000
        return a

    z0 = wav("zora_david_rate+0_var0.wav")
    z1 = wav("zora_david_rate+0_var1.wav")
    z2 = wav("zora_david_rate+1_var0.wav")  # held-out (different rate!)
    sil = np.zeros(16000, np.float32)
    rng = np.random.RandomState(0)
    noise = (rng.randn(16000) * 0.05).astype(np.float32)
    t = np.arange(16000) / 16000.0
    tone = (0.1 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    # ---- enrollment (both paths, same clips) ----
    m0, m1 = mfcc.extract(z0), mfcc.extract(z1)
    tpl = build_template("zora", [m0, m1], band=BAND)
    print(f"[DTW] template: {tpl.frame_count} frames, "
          f"intra/step {[round(d, 4) for d in tpl.intra_dists]}")

    from scripts.calibrate_threshold import TfliteEncoder
    enc = TfliteEncoder()
    e0, e1 = enc.embed(z0), enc.embed(z1)
    proto = (e0 + e1) / np.linalg.norm(e0 + e1)

    # ---- isolated-clip comparison ----
    tests = {"z2-heldout": z2, "silence": sil, "noise": noise, "tone": tone}
    iso = {}
    up, lo = dtw_envelope(tpl.centroid, BAND)
    print(f"\n{'clip':<12}{'dtw/step':>9}{'lb_keogh':>9}{'n-cos':>8}")
    for name, audio in tests.items():
        m = mfcc.extract(audio)
        tot = dtw_distance(resample_frames(m, tpl.frame_count), tpl.centroid,
                           band=BAND)
        per_step = tot / (2 * tpl.frame_count)
        lb = lb_keogh(resample_frames(m, tpl.frame_count), up, lo) / (2 * tpl.frame_count)
        cos = float(np.dot(enc.embed(audio), proto))
        iso[name] = {"dtw_per_step": per_step, "lb_per_step": lb,
                     "neural_cosine": cos}
        print(f"{name:<12}{per_step:9.4f}{lb:9.4f}{cos:8.4f}")

    kw_d = iso["z2-heldout"]["dtw_per_step"]
    neg_best = min(iso[k]["dtw_per_step"] for k in ("silence", "noise", "tone"))
    if kw_d < neg_best:
        thr = (kw_d + neg_best) / 2.0
        sep = "separable (pilot)"
    else:
        thr = kw_d * 2.0
        sep = "OVERLAP -- pilot threshold is a guess"
    print(f"[DTW] kw={kw_d:.4f} best-neg={neg_best:.4f} -> thr={thr:.4f} [{sep}]")

    # ---- streaming comparison (same 4 s stream as the neural smoke) ----
    stream = np.concatenate([sil, z2, z2, sil])
    # frame_audio() clamps to exactly 1 s for the neural model's fixed input
    # tensor; the DTW stream needs continuous frames, so bypass the clamp.
    mfcc.target_num_samples = len(stream)
    frames = mfcc.extract(stream)
    assert len(frames) == 398, len(frames)
    def run_stream(threshold):
        spot = DTWSpotter(tpl, threshold=threshold, cooldown_frames=150)
        trigs = []
        for i, f in enumerate(frames):
            r = spot.update(f, i * 10.0)
            if r["triggered"]:
                trigs.append({"timestamp_ms": r["timestamp_ms"],
                              "cost_per_step": round(r["cost_per_step"], 4)})
        return trigs

    dtw_trigs = run_stream(thr)
    print(f"[DTW] stream triggers ({len(frames)} frames): "
          f"{[t['timestamp_ms'] for t in dtw_trigs]}")
    print(f"[DTW] threshold sweep (kw clips end @2000/3000 ms):")
    sweep = {}
    for t_thr in (2.6, 2.8, 3.0, 3.5, 4.0, 8.0, round(thr, 2)):
        tg = run_stream(t_thr)
        sweep[str(t_thr)] = [x["timestamp_ms"] for x in tg]
        print(f"        thr={t_thr:>7}: {sweep[str(t_thr)]}")

    from src.streaming.demo_pipeline import EndToEndVoiceActivatorDemo
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        npaths = []
        for nm, a in (("sil.wav", sil), ("noise.wav", noise), ("tone.wav", tone)):
            p = os.path.join(td, nm)
            sf.write(p, a, 16000)
            npaths.append(p)
        d0 = os.path.join(td, "z0.wav")
        d1 = os.path.join(td, "z1.wav")
        sf.write(d0, z0, 16000)
        sf.write(d1, z1, 16000)
        demo = EndToEndVoiceActivatorDemo()
        demo.enroll_keyword([d0, d1], "zora", negative_paths=npaths)
    n_trigs = []
    for i in range(0, len(stream) - 799, 800):
        r = demo.process_chunk(stream[i:i + 800], (i // 800) * 50.0)
        if r["event"] == "ACTIVATION_TRIGGERED":
            n_trigs.append({"timestamp_ms": r["timestamp_ms"],
                            "smoothed_sim": r["smoothed_sim"]})
    print(f"[NEURAL] stream triggers: {[t['timestamp_ms'] for t in n_trigs]}")

    # ---- edge-cost comparison (computed, ESP32-S3 float32) ----
    t_len, dim = tpl.frame_count, tpl.centroid.shape[1]
    cost = {
        "dtw_flash_per_keyword_bytes": t_len * dim * 4,
        "dtw_ram_streaming_bytes": 2 * (t_len + 1) * 6,
        "dtw_flops_per_10ms_frame": t_len * (3 * dim + 4),
        "neural_model_flash_bytes": 63 * 1024,
        "neural_arena_ram_bytes": 96 * 1024,
        "note": "MFCC frontend shared by both paths; neural inference flops "
                "are device-timed (see quantization bench), not estimated here.",
    }
    print(f"\n[EDGE-COST] DTW/kw: {cost['dtw_flash_per_keyword_bytes']} B flash, "
          f"{cost['dtw_ram_streaming_bytes']} B RAM, "
          f"{cost['dtw_flops_per_10ms_frame']} flops/frame")

    out = {"band": BAND, "template_frames": t_len,
           "dtw_threshold_per_step": thr, "separation": sep,
           "isolated": iso,
           "stream": {"dtw_triggers": dtw_trigs, "neural_triggers": n_trigs,
                      "dtw_threshold_sweep": sweep},
           "edge_cost": cost}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT} (PILOT grade: 3 clips + synthetic negatives)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
