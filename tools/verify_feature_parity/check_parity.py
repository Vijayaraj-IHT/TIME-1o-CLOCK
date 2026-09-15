#!/usr/bin/env python3
"""Firmware MFCC parity gate (committed, reproducible).

Compiles tools/verify_feature_parity/verify_parity.cpp against the REAL master
firmware header, runs it on 3 signals (2 synthetic + 1 real zora recording),
and diffs the (98,13) MFCC matrix against src/features/mfcc.py on identical
input.

  python tools/verify_feature_parity/check_parity.py   # exit 0 == PASS

PASS bar (TRACK_A grade): max abs diff < 5e-5 AND relative diff < 0.5% where
|ref| > 0.1 on every signal. (Plain max-relative is meaningless for MFCC:
DCT coefficients cross zero, so any |ref|~0 bin explodes the ratio. TRACK_A
hit the same wall and reported rel only for white noise.) Reference numbers: dense-header era measured 1.91e-05 /
1.53e-05 max abs (float32 FFT-rounding level); the sparse port must match.
"""
import os
import subprocess
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from src.features.mfcc import MFCCFeatureExtractor  # noqa: E402

TOOLS = os.path.join(REPO, "tools", "verify_feature_parity")
CPP = os.path.join(TOOLS, "verify_parity.cpp")
BIN = "/tmp/verify_parity_sparse"
ABS_TOL, REL_TOL = 5e-5, 0.005


def build():
    inc = os.path.join(REPO, "src", "deployment", "esp32")
    r = subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra",
                        f"-I{inc}", CPP, "-o", BIN],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise SystemExit("g++ build failed")
    print(f"[BUILD] {BIN}")


def fw_mfcc(audio):
    assert audio.shape == (16000,) and audio.dtype == np.float32
    audio.tofile("/tmp/parity_in.bin")
    r = subprocess.run([BIN, "/tmp/parity_in.bin", "/tmp/parity_out.bin"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"fw binary failed: {r.stderr[-500:]}")
    return np.fromfile("/tmp/parity_out.bin", dtype=np.float32).reshape(98, 13)


def signals():
    rng = np.random.RandomState(0)
    t = np.arange(16000, dtype=np.float64) / 16000.0
    tones = (0.30 * np.sin(2 * np.pi * 220 * t)
             + 0.30 * np.sin(2 * np.pi * 440 * t)
             + 0.20 * np.sin(2 * np.pi * 1200 * t)
             + 0.01 * rng.randn(16000)).astype(np.float32)
    white = (0.10 * rng.randn(16000)).astype(np.float32)
    out = [("multitone+noise", tones), ("white-noise", white)]
    try:
        import soundfile as sf
        z, sr = sf.read(os.path.join(
            REPO, "raspberry_pi_deployment", "custom_keywords", "zora",
            "zora_david_rate+0_var0.wav"), dtype="float32")
        assert sr == 16000 and len(z) >= 16000
        out.append(("real-zora", np.ascontiguousarray(z[:16000])))
    except Exception as e:  # missing file or no soundfile: synthetic-only gate
        print(f"[SKIP] real-zora signal ({e})")
    return out


def main():
    build()
    ext = MFCCFeatureExtractor(sample_rate=16000, n_mfcc=13)
    fails = 0
    print(f"{'signal':<16}{'max_abs':>10}{'mean_abs':>10}{'max_rel':>9}  verdict")
    for name, audio in signals():
        ref = ext.extract(audio).astype(np.float64)
        got = fw_mfcc(audio).astype(np.float64)
        assert ref.shape == (98, 13), ref.shape
        ad = np.abs(got - ref)
        sig = np.abs(ref) > 0.1
        rdmax = float((ad[sig] / np.abs(ref[sig])).max()) if sig.any() else 0.0
        ok = ad.max() < ABS_TOL and rdmax < REL_TOL
        fails += (not ok)
        loc = tuple(int(x) for x in np.unravel_index(int(ad.argmax()), ad.shape))
        print(f"{name:<16}{ad.max():10.2e}{ad.mean():10.2e}{rdmax:9.2%}  "
              f"{'PASS' if ok else 'FAIL'}  @frame{loc}")
    print("PARITY: " + ("PASS" if not fails else "FAIL"))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
