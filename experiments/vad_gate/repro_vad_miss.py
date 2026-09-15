#!/usr/bin/env python3
"""VAD-gate miss repro + hangover sweep (neural path).

The Track-B pilot found the neural demo firing 1/2 on sil+z2+z2+sil:
VAD_SKIP 1500-1900ms skipped keyword#1's completion windows, the 8-frame
smoothing decayed, and persistence never completed. This script reproduces
that miss systematically and sweeps VAD hangover to find the minimal fix.
Outcome (Sep-14): hangover 10 is now the default (host + firmware); this
script stays as the sweep tool (it overrides the default explicitly).

Streams (all 16kHz float32):
  S1 pilot  : sil(1s) + z2 + z2 + sil(1s)      (kw ends ~1310/~2310ms)
  S2 single : sil(0.5s) + z2 + sil(2s)         (kw end ~810ms)
  S3 noisy  : noise(1s) + z2 + sil(1.5s)       (kw end ~1310ms, bg adapt)
  S4 neg    : sil + noise + tone + sil (4s)    (FA check: expect 0 triggers)

  python experiments/vad_gate/repro_vad_miss.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.streaming.demo_pipeline import EndToEndVoiceActivatorDemo

ZDIR = os.path.join("raspberry_pi_deployment", "custom_keywords", "zora")


def load_material():
    import soundfile as sf

    def wav(name):
        a, sr = sf.read(os.path.join(ZDIR, name), dtype="float32")
        assert sr == 16000 and len(a) == 16000, name
        return a

    z0 = wav("zora_david_rate+0_var0.wav")
    z1 = wav("zora_david_rate+0_var1.wav")
    z2 = wav("zora_david_rate+1_var0.wav")
    sil = np.zeros(16000, np.float32)
    rng = np.random.RandomState(0)
    noise = (rng.randn(16000) * 0.05).astype(np.float32)
    t = np.arange(16000) / 16000.0
    tone = (0.1 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    return z0, z1, z2, sil, noise, tone


def fresh_demo(z0, z1, sil, noise, tone, hangover):
    import soundfile as sf
    td = tempfile.mkdtemp()
    paths = []
    for nm, a in (("z0.wav", z0), ("z1.wav", z1)):
        p = os.path.join(td, nm)
        sf.write(p, a, 16000)
        paths.append(p)
    npaths = []
    for nm, a in (("s.wav", sil), ("n.wav", noise), ("t.wav", tone)):
        p = os.path.join(td, nm)
        sf.write(p, a, 16000)
        npaths.append(p)
    demo = EndToEndVoiceActivatorDemo()
    demo.enroll_keyword(paths, "zora", negative_paths=npaths)
    demo.vad.hangover_frames = int(hangover)
    return demo


def run_stream(demo, stream, trace=False):
    trigs, n_inf, n_skip = [], 0, 0
    max_smooth, max_state = 0.0, ""
    for i in range(0, len(stream) - 799, 800):
        tau = (i // 800) * 50.0
        r = demo.process_chunk(stream[i:i + 800], tau)
        if r["event"] == "ACTIVATION_TRIGGERED":
            trigs.append(tau)
        elif r["event"] == "INFERENCE":
            n_inf += 1
        elif r["event"] == "VAD_SKIP":
            n_skip += 1
        s = r.get("smoothed_sim")
        if isinstance(s, float) and not np.isnan(s):
            max_smooth = max(max_smooth, s)
        if trace and (tau % 100 == 0 or r["event"] == "ACTIVATION_TRIGGERED"):
            print(f"    tau={tau:6.0f} ev={r['event']:<20} "
                  f"raw={r.get('raw_sim', float('nan'))} "
                  f"sm={r.get('smoothed_sim', float('nan'))} st={r.get('state', '')}")
    return trigs, n_inf, n_skip, round(max_smooth, 4)


def main():
    z0, z1, z2, sil, noise, tone = load_material()
    streams = {
        "S1 pilot": (np.concatenate([sil, z2, z2, sil]), [1310, 2310]),
        "S2 single": (np.concatenate([sil[:8000], z2, sil, sil[:8000]]), [810]),
        "S3 noisy": (np.concatenate([noise, z2, sil, sil[:8000]]), [1310]),
        "S4 neg": (np.concatenate([sil, noise, tone, sil]), []),
        # S5: wide spacing (2s gap > 1500ms cooldown) -> 2/2 possible
        "S5 wide": (np.concatenate([sil, z2, sil, sil, z2, sil]), [1310, 4310]),
        # S2b: S2 with warm ring (1.5s lead) -> isolates cold-start
        "S2b warm": (np.concatenate([sil, sil[:8000], z2, sil, sil[:8000]]), [1810]),
    }
    print(f"{'hang':>4} | " + " | ".join(f"{k:<28}" for k in streams) + " | inf/skip(S1)")
    for h in (3, 6, 10, 16):
        row = []
        inf1 = sk1 = 0
        for name, (audio, _) in streams.items():
            demo = fresh_demo(z0, z1, sil, noise, tone, h)
            trigs, n_inf, n_skip, mx = run_stream(demo, audio)
            row.append(f"{str(trigs):<18} mx={mx:.2f}")
            if name == "S1 pilot":
                inf1, sk1 = n_inf, n_skip
        print(f"{h:>4} | " + " | ".join(row) + f" | {inf1}/{sk1}")

    print("\nTrace S1 @hangover=3 (the miss):")
    demo = fresh_demo(z0, z1, sil, noise, tone, 3)
    run_stream(demo, streams["S1 pilot"][0], trace=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
