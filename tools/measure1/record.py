#!/usr/bin/env python3
"""Prompted keyword-take recorder (PC with mic).

  python tools/measure1/record.py --keyword zora --speaker priya --takes 5

Records <takes> takes of --seconds each, auto-named per the MEASURE-1
protocol (<kw>_<speaker>_rate<R>_var<V>.wav, next free V). Needs
`pip install sounddevice` (imported lazily so the rest of the kit works
without it).
"""
import argparse
import glob
import os
import sys


def _next_var(out_dir, prefix):
    taken = set()
    for p in glob.glob(os.path.join(out_dir, prefix + "_var*.wav")):
        try:
            taken.add(int(p.rsplit("_var", 1)[1].split(".")[0]))
        except ValueError:
            pass
    v = 0
    while v in taken:
        v += 1
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description="Record MEASURE-1 keyword takes.")
    ap.add_argument("--keyword", required=True)
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--rate", default="+0", help="e.g. +0 (natural), +1 (fast), -1 (slow)")
    ap.add_argument("--takes", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=1.5)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    try:
        import sounddevice as sd
    except ImportError:
        print("ERROR: needs `pip install sounddevice` (mic recording only).",
              file=sys.stderr)
        return 1
    import soundfile as sf
    import numpy as np
    out_dir = args.out_dir or os.path.join("data", "raw", "custom_keywords",
                                           args.keyword)
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"{args.keyword}_{args.speaker}_rate{args.rate}"
    sr = 16000
    for _ in range(args.takes):
        v = _next_var(out_dir, prefix)
        fn = f"{prefix}_var{v}.wav"
        input(f"[{fn}] ENTER to record {args.seconds}s, then say '{args.keyword}' ... ")
        print("  recording...")
        rec = sd.rec(int(args.seconds * sr), samplerate=sr, channels=1, dtype="float32")
        sd.wait()
        peak = float(np.max(np.abs(rec)))
        print(f"  peak {peak:.2f}", "(TOO QUIET -- redo closer/louder)"
              if peak < 0.10 else "(level ok)" if peak < 0.98 else "(CLIPPING -- redo softer!)")
        sf.write(os.path.join(out_dir, fn), rec[:, 0], sr, subtype="PCM_16")
        print(f"  wrote {fn}")
    print("Done. Next: manifest.py + qc_audio.py (protocol section 5).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
