#!/usr/bin/env python3
"""MEASURE-1 manifest builder: inventory + layout/naming validation.

  python tools/measure1/manifest.py --root data/raw [--out manifest.json]

Walks <root>, assigns each .wav a role by directory, validates keyword
naming (<kw>_<speaker>_rate<+-N>_var<V>.wav), checks minimum counts, and
writes a JSON manifest. Exit 1 on any ERROR (warnings pass).
"""
import argparse
import json
import os
import re
import sys

KW_RE = re.compile(r"^[a-z0-9]+_[a-z0-9]+_rate[+-]?\d+_var\d+\.wav$")
MIN_KW_CLIPS = 10       # per keyword (warn below; protocol wants 45)
MIN_NOISE_SECONDS = 60  # total background (warn below; protocol wants 600+)


def _wav_seconds(path):
    import soundfile as sf
    info = sf.info(path)
    return info.frames / float(info.samplerate or 1)


def build_manifest(root):
    errors, warnings = [], []
    keywords, noise, negatives, stray = {}, [], [], []
    for dirpath, _, files in os.walk(root):
        for fn in sorted(files):
            if not fn.lower().endswith(".wav"):
                continue
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, root).replace(os.sep, "/")
            parts = rel.split("/")
            try:
                dur = _wav_seconds(p)
            except Exception as e:  # unreadable file: hard error
                errors.append(f"{rel}: unreadable ({e})")
                continue
            entry = {"path": rel, "seconds": round(dur, 3)}
            if len(parts) >= 3 and parts[0] == "custom_keywords":
                kw = parts[1]
                entry["keyword"] = kw
                keywords.setdefault(kw, []).append(entry)
                base = parts[-1]
                if not KW_RE.match(base):
                    warnings.append(f"{rel}: name does not match "
                                    f"<kw>_<speaker>_rate<+-N>_var<V>.wav")
                elif not base.startswith(kw + "_"):
                    warnings.append(f"{rel}: filename keyword prefix != dir '{kw}'")
            elif len(parts) >= 2 and parts[0] == "noise":
                entry["role"] = "background"
                noise.append(entry)
            elif len(parts) >= 2 and parts[0] == "negatives":
                entry["role"] = "negative"
                negatives.append(entry)
            else:
                stray.append(entry)
    if stray:
        warnings.append(f"{len(stray)} wav(s) outside the protocol layout "
                        f"(custom_keywords/<kw>/, noise/, negatives/)")
    for kw, clips in sorted(keywords.items()):
        if len(clips) < MIN_KW_CLIPS:
            warnings.append(f"keyword '{kw}': {len(clips)} clips "
                            f"(minimum {MIN_KW_CLIPS}, protocol wants 45)")
    tot_noise = sum(e["seconds"] for e in noise)
    if tot_noise < MIN_NOISE_SECONDS:
        warnings.append(f"noise total {tot_noise:.0f}s "
                        f"(minimum {MIN_NOISE_SECONDS}s, protocol wants 600s+)")
    if not keywords:
        errors.append("no keyword clips found under custom_keywords/<kw>/")
    manifest = {
        "root": os.path.abspath(root),
        "keywords": {k: {"n_clips": len(v), "seconds": round(sum(e["seconds"] for e in v), 1),
                         "clips": v} for k, v in sorted(keywords.items())},
        "noise": {"n_clips": len(noise), "seconds": round(tot_noise, 1), "clips": noise},
        "negatives": {"n_clips": len(negatives),
                      "seconds": round(sum(e["seconds"] for e in negatives), 1),
                      "clips": negatives},
        "stray": stray,
        "warnings": warnings,
        "errors": errors,
    }
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build + validate MEASURE-1 manifest.")
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if not os.path.isdir(args.root):
        print(f"ERROR: root dir missing: {args.root}", file=sys.stderr)
        return 1
    m = build_manifest(args.root)
    for w in m["warnings"]:
        print(f"WARN: {w}")
    for e in m["errors"]:
        print(f"ERROR: {e}")
    n_kw = sum(k["n_clips"] for k in m["keywords"].values())
    print(f"keywords={n_kw} noise={m['noise']['n_clips']} "
          f"({m['noise']['seconds']}s) negatives={m['negatives']['n_clips']} "
          f"errors={len(m['errors'])} warnings={len(m['warnings'])}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)
        print(f"wrote {args.out}")
    return 1 if m["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
