#!/usr/bin/env python3
"""Host-vs-firmware DTW spotter parity gate.

  python tools/verify_dtw_parity/check_parity.py

1. Verifies the generated template header is fresh (--check).
2. Exports the pilot-stream MFCC frames (sil+z2+z2+sil, 398x13 float32).
3. Compiles tools/verify_dtw_parity/spotter_harness.cpp against
   src/deployment/esp32/ (dtw_spotter.h + dtw_template_zora.h) with g++.
4. Runs host spot.py AND the C++ harness with the header's operating point.
5. Requires identical trigger FRAMES + costs within 1e-3 + identical consumed.

Exit 1 on any mismatch. Needs g++ + soundfile/scipy (no TF).
"""
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

COST_TOL = 1e-3


def _header_operating_point():
    text = open(os.path.join(REPO, "src", "deployment", "esp32",
                             "dtw_template_zora.h"), encoding="utf-8").read()

    def num(name):
        m = re.search(rf"#define {name} ([0-9.]+)", text)
        assert m, name
        return float(m.group(1))

    return {"threshold": num("DTW_ZORA_THRESHOLD"),
            "cooldown": int(num("DTW_ZORA_COOLDOWN_FRAMES")),
            "min_duration": int(num("DTW_ZORA_MIN_DURATION_FRAMES"))}


def main():
    r = subprocess.run([sys.executable, os.path.join(REPO, "scripts",
                                                     "export_dtw_template.py"),
                        "--check"], capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print("FAIL: template header stale -- regenerate + --sync first.")
        return 1
    op = _header_operating_point()
    print(f"operating point from header: {op}")

    import soundfile as sf
    from src.features.mfcc import MFCCFeatureExtractor
    from src.dtw.templates import build_template
    from src.dtw.spot import DTWSpotter

    zdir = os.path.join(REPO, "raspberry_pi_deployment", "custom_keywords", "zora")

    def wav(n):
        a, sr = sf.read(os.path.join(zdir, n), dtype="float32")
        assert sr == 16000
        return a

    mfcc = MFCCFeatureExtractor(sample_rate=16000, n_mfcc=13)
    tpl = build_template("zora", [mfcc.extract(wav("zora_david_rate+0_var0.wav")),
                                  mfcc.extract(wav("zora_david_rate+0_var1.wav"))],
                         band=8)
    z2 = wav("zora_david_rate+1_var0.wav")
    sil = np.zeros(16000, np.float32)
    mfcc.target_num_samples = 64000
    frames = mfcc.extract(np.concatenate([sil, z2, z2, sil])).astype(np.float32)
    assert frames.shape[1] == 13

    host = DTWSpotter(tpl, threshold=op["threshold"],
                      cooldown_frames=op["cooldown"],
                      min_duration_frames=op["min_duration"])
    host_trigs = []
    for i, f in enumerate(frames):
        rr = host.update(f, i * 10.0)
        if rr["triggered"]:
            host_trigs.append((i, rr["cost_per_step"], rr["frames_consumed"]))

    with tempfile.TemporaryDirectory() as td:
        fbin = os.path.join(td, "frames.bin")
        frames.tofile(fbin)
        exe = os.path.join(td, "harness")
        c = subprocess.run(["g++", "-O2", "-Wall",
                            "-I", os.path.join(REPO, "src", "deployment", "esp32"),
                            os.path.join(REPO, "tools", "verify_dtw_parity",
                                         "spotter_harness.cpp"),
                            "-o", exe], capture_output=True, text=True)
        if c.returncode != 0:
            print("FAIL: g++ compile error:\n" + c.stderr)
            return 1
        if c.stderr.strip():
            print("g++ warnings:\n" + c.stderr.strip())
        h = subprocess.run([exe, fbin, "13"], capture_output=True, text=True)
        if h.returncode != 0:
            print("FAIL: harness error:\n" + h.stderr)
            return 1
        fw_trigs = []
        for line in h.stdout.strip().split("\n"):
            if line.strip():
                fi, co, nf = line.split()
                fw_trigs.append((int(fi), float(co), int(nf)))
        print("harness: " + h.stderr.strip())

    print(f"host triggers (frame,cost,consumed): {host_trigs}")
    print(f"fw   triggers (frame,cost,consumed): {fw_trigs}")
    ok = True
    if [t[0] for t in host_trigs] != [t[0] for t in fw_trigs]:
        print("FAIL: trigger frames differ.")
        ok = False
    for ( _, hc, hn), (_, fc, fn) in zip(host_trigs, fw_trigs):
        if abs(hc - fc) > COST_TOL:
            print(f"FAIL: cost {hc} vs {fc} exceeds {COST_TOL}.")
            ok = False
        if hn != fn:
            print(f"FAIL: consumed {hn} vs {fn}.")
            ok = False
    if not host_trigs:
        print("FAIL: no triggers at all -- parity vacuous.")
        ok = False
    print("PARITY PASS: C++ spotter matches host "
          f"({len(host_trigs)} triggers, |dcost| <= {COST_TOL})." if ok
          else "PARITY FAIL.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
