#!/usr/bin/env python3
"""DTW threshold calibration: leave-one-out keyword scores vs negatives.

  python experiments/track_b/calibrate_dtw_threshold.py [--kw-dir D] [--neg-dir N]

Template(s) from --kw-dir clips (leave-one-out scores = honest), negatives
from --neg-dir (*.wav) or synthetic silence/noise/tone fallback. Sweeps the
threshold over all score midpoints, reports EER + full TPR/FPR table +
suggested deploy threshold, writes JSON. Data-driven: pilot-grade on the 3
repo zora clips, real once MEASURE-1 lands. No TF needed.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.features.mfcc import MFCCFeatureExtractor
from src.dtw.dtw_core import dtw_distance
from src.dtw.templates import build_template, resample_frames

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_KW = os.path.join(_REPO_ROOT := os.path.abspath(os.path.join(HERE, "..", "..")),
                          "data", "raw", "custom_keywords", "zora")


def load_wavs(paths):
    import soundfile as sf
    out = []
    for p in paths:
        a, sr = sf.read(p, dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        assert sr == 16000, f"{p}: sr={sr} (MEASURE-1 protocol: 16kHz)"
        out.append((os.path.basename(p), a.astype(np.float32)))
    return out


def synth_negatives():
    sil = np.zeros(16000, np.float32)
    rng = np.random.RandomState(0)
    noise = (rng.randn(16000) * 0.05).astype(np.float32)
    t = np.arange(16000) / 16000.0
    tone = (0.1 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    return [("synth_silence", sil), ("synth_noise", noise), ("synth_tone", tone)]


def per_step_cost(mfcc_test, template, band):
    tot = dtw_distance(resample_frames(mfcc_test, template.frame_count),
                       template.centroid, band=band)
    return tot / (2 * template.frame_count)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Calibrate DTW spotter threshold.")
    ap.add_argument("--kw-dir", default=DEFAULT_KW)
    ap.add_argument("--neg-dir", default=None)
    ap.add_argument("--band", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "dtw_threshold.json"))
    args = ap.parse_args(argv)

    mfcc = MFCCFeatureExtractor(sample_rate=16000, n_mfcc=13)
    kw_paths = sorted(glob.glob(os.path.join(args.kw_dir, "*.wav")))
    assert kw_paths, f"no wavs in {args.kw_dir}"
    kw = [(n, mfcc.extract(a)) for n, a in load_wavs(kw_paths)]
    if args.neg_dir:
        neg_paths = sorted(glob.glob(os.path.join(args.neg_dir, "*.wav")))
        assert neg_paths, f"no wavs in {args.neg_dir}"
        negs = [(n, mfcc.extract(a)) for n, a in load_wavs(neg_paths)]
        neg_src = args.neg_dir
    else:
        negs = [(n, mfcc.extract(a)) for n, a in synth_negatives()]
        neg_src = "synthetic (silence/noise/tone)"

    full_tpl = build_template("kw", [m for _, m in kw], band=args.band)
    # Honest keyword scores: leave-one-out (each clip vs template of OTHERS).
    kw_scores = []
    if len(kw) >= 2:
        for i, (n, m) in enumerate(kw):
            others = [mm for j, (_, mm) in enumerate(kw) if j != i]
            t = build_template("kw", others, band=args.band)
            kw_scores.append((n, per_step_cost(m, t, args.band)))
    else:  # fallback: train score (optimistic -- flagged in output)
        kw_scores = [(kw[0][0], per_step_cost(kw[0][1], full_tpl, args.band))]
    neg_scores = [(n, per_step_cost(m, full_tpl, args.band)) for n, m in negs]

    kw_vals = sorted(s for _, s in kw_scores)
    neg_vals = sorted(s for _, s in neg_scores)
    separable = kw_vals[-1] < neg_vals[0]
    # Sweep midpoints; EER = min |FPR - FNR| (lower cost = keyword).
    cands = sorted(set(kw_vals + neg_vals))
    grid = ([cands[0] / 2.0] + [(a + b) / 2.0 for a, b in zip(cands, cands[1:])]
            + [cands[-1] * 2.0])
    table, eer = [], None
    for thr in grid:
        tpr = sum(s < thr for s in kw_vals) / len(kw_vals)
        fpr = sum(s < thr for s in neg_vals) / len(neg_vals)
        table.append({"threshold": thr, "tpr": tpr, "fpr": fpr})
        if eer is None or abs(fpr - (1 - tpr)) < abs(eer["fpr"] - (1 - eer["tpr"])):
            eer = table[-1]
    eer = {**eer, "eer_rate": (eer["fpr"] + (1 - eer["tpr"])) / 2.0}

    print(f"keyword clips ({len(kw)}): "
          + ", ".join(f"{n}={s:.4f}" for n, s in kw_scores))
    print(f"negatives ({len(negs)}, {neg_src}): "
          + ", ".join(f"{n}={s:.4f}" for n, s in neg_scores))
    print(f"kw_max={kw_vals[-1]:.4f} neg_min={neg_vals[0]:.4f} "
          f"-> {'SEPARABLE' if separable else 'OVERLAP'}")
    print(f"EER: thr={eer['threshold']:.4f} rate={eer['eer_rate']:.4f} "
          f"(TPR={eer['tpr']:.2f} FPR={eer['fpr']:.2f})")
    print(f"suggested deploy threshold: {eer['threshold']:.4f} "
          f"(re-tune on MEASURE-1; pilot-grade now)")

    out = {"kw_dir": args.kw_dir, "neg_source": neg_src, "band": args.band,
           "template_frames": full_tpl.frame_count,
           "kw_scores": [{"clip": n, "cost_per_step": s} for n, s in kw_scores],
           "neg_scores": [{"clip": n, "cost_per_step": s} for n, s in neg_scores],
           "separable": separable,
           "kw_max": kw_vals[-1], "neg_min": neg_vals[0],
           "eer": eer, "suggested_threshold": eer["threshold"],
           "sweep": table}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
