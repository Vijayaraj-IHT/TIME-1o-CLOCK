#!/usr/bin/env python3
"""MEASURE-1 audio QC: per-file signal validation.

  python tools/measure1/qc_audio.py --root data/raw [--out qc.json]

Checks every .wav: sample rate, channels, duration, peak/RMS, clipping %,
DC offset, edge energy (cut words). Role (keyword/background/negative) comes
from the directory layout (see docs/MEASURE1_PROTOCOL.md). Exit 1 on any
ERROR; warnings pass but should be reviewed.
"""
import argparse
import json
import os
import sys

import numpy as np

SR_WANT = 16000
PEAK_MIN = 0.10        # keyword/negative clips below this are "too quiet"
RMS_MIN = 0.010        # ... or "silent"
CLIP_ERR = 0.01        # fraction of samples past 0.98 -> ERROR
CLIP_WARN = 0.001      # ... -> WARN
DC_WARN = 0.02         # |mean| above this -> WARN
KW_DUR = (0.5, 3.0)    # keyword clip duration window (WARN outside)
BG_DUR_WARN = 5.0      # background clips shorter than this -> WARN
EDGE_MS = 50           # edge-energy window for cut-word detection
EDGE_FRAC = 0.30       # edge RMS above this x overall RMS -> WARN


def check_file(path, role):
    import soundfile as sf
    errors, warnings, stats = [], [], {}
    try:
        info = sf.info(path)
    except Exception as e:
        return {"errors": [f"unreadable ({e})"], "warnings": [], "stats": {}}
    stats["sample_rate"] = info.samplerate
    stats["channels"] = info.channels
    stats["seconds"] = round(info.frames / float(info.samplerate or 1), 3)
    if info.samplerate != SR_WANT:
        errors.append(f"sample_rate {info.samplerate} != {SR_WANT} (resample!)")
    if info.channels != 1:
        errors.append(f"channels {info.channels} != 1 (downmix to mono!)")
    try:
        a, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as e:
        errors.append(f"unreadable ({e})")
        return {"errors": errors, "warnings": warnings, "stats": stats}
    x = a[:, 0].astype(np.float64)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
    stats["peak"] = round(peak, 4)
    stats["rms"] = round(rms, 5)
    stats["dc"] = round(float(np.mean(x)), 5)
    stats["clip_frac"] = round(float(np.mean(np.abs(x) > 0.98)), 5)
    dur = stats["seconds"]
    if role == "keyword":
        if not (KW_DUR[0] <= dur <= KW_DUR[1]):
            warnings.append(f"duration {dur}s outside {KW_DUR}s window")
    elif role == "background" and dur < BG_DUR_WARN:
        warnings.append(f"background clip short ({dur}s < {BG_DUR_WARN}s)")
    if role in ("keyword", "negative"):
        if peak < PEAK_MIN:
            errors.append(f"too quiet (peak {peak:.3f} < {PEAK_MIN})")
        if rms < RMS_MIN:
            errors.append(f"silent (rms {rms:.4f} < {RMS_MIN})")
    if stats["clip_frac"] >= CLIP_ERR:
        errors.append(f"clipping {stats['clip_frac'] * 100:.1f}% samples past 0.98")
    elif stats["clip_frac"] >= CLIP_WARN:
        warnings.append(f"mild clipping {stats['clip_frac'] * 100:.2f}% past 0.98")
    if abs(stats["dc"]) >= DC_WARN:
        warnings.append(f"DC offset {stats['dc']:.3f}")
    if role == "keyword" and rms > 0 and sr == SR_WANT:
        n = int(sr * EDGE_MS / 1000)
        if len(x) > 2 * n:
            e = max(float(np.sqrt(np.mean(x[:n] ** 2))),
                    float(np.sqrt(np.mean(x[-n:] ** 2))))
            if e > EDGE_FRAC * rms:
                warnings.append("edge energy high -- word may be cut at clip edge")
    return {"errors": errors, "warnings": warnings, "stats": stats}


def _role_of(rel):
    parts = rel.split("/")
    if len(parts) >= 3 and parts[0] == "custom_keywords":
        return "keyword"
    if len(parts) >= 2 and parts[0] == "noise":
        return "background"
    if len(parts) >= 2 and parts[0] == "negatives":
        return "negative"
    return "stray"


def main(argv=None):
    ap = argparse.ArgumentParser(description="MEASURE-1 audio QC.")
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if not os.path.isdir(args.root):
        print(f"ERROR: root dir missing: {args.root}", file=sys.stderr)
        return 1
    report, n_err, n_warn = {}, 0, 0
    for dirpath, _, files in os.walk(args.root):
        for fn in sorted(files):
            if not fn.lower().endswith(".wav"):
                continue
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, args.root).replace(os.sep, "/")
            r = check_file(p, _role_of(rel))
            report[rel] = r
            n_err += len(r["errors"])
            n_warn += len(r["warnings"])
            for e in r["errors"]:
                print(f"ERROR {rel}: {e}")
            for w in r["warnings"]:
                print(f"WARN {rel}: {w}")
    print(f"files={len(report)} errors={n_err} warnings={n_warn}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"root": os.path.abspath(args.root), "files": report,
                       "n_errors": n_err, "n_warnings": n_warn}, f, indent=2)
        print(f"wrote {args.out}")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
