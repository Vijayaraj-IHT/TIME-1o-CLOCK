"""VAD-gate miss regression tests (neural path, Sep-14 fix).

Short keywords (~270ms word) complete inside post-word silence: with
hangover=3 the VAD skipped the completion windows, the 8-frame smoothing
decayed, and persistence never completed (Track-B pilot: neural 1/2 while
DTW went 2/2). Fix: hangover tail 10 chunks (500ms) on host + firmware.

Streams mirror experiments/vad_gate/repro_vad_miss.py (S1/S3/S4/S5).
Needs TF + the int8 model + data/raw zora wavs (same as test_demo.py).
"""
import os
import sys
import tempfile
import unittest

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO_ROOT)

from src.streaming.demo_pipeline import EndToEndVoiceActivatorDemo
from src.streaming.vad import EnergyVAD

ZDIR = os.path.join(_REPO_ROOT, "data", "raw", "custom_keywords", "zora")


def _wav(name):
    import soundfile as sf
    a, sr = sf.read(os.path.join(ZDIR, name), dtype="float32")
    assert sr == 16000 and len(a) == 16000, name
    return a


def _fresh_demo(hangover=10):
    import soundfile as sf
    z0 = _wav("zora_david_rate+0_var0.wav")
    z1 = _wav("zora_david_rate+0_var1.wav")
    sil = np.zeros(16000, np.float32)
    rng = np.random.RandomState(0)
    noise = (rng.randn(16000) * 0.05).astype(np.float32)
    t = np.arange(16000) / 16000.0
    tone = (0.1 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
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
    demo = EndToEndVoiceActivatorDemo(
        tflite_model_path=os.path.join(_REPO_ROOT, "models", "tflite",
                                       "voice_activator_int8.tflite"))
    demo.enroll_keyword(paths, "zora", negative_paths=npaths)
    demo.vad.hangover_frames = int(hangover)
    return demo, z0, z1, sil, noise, tone


def _run(demo, stream):
    trigs = []
    for i in range(0, len(stream) - 799, 800):
        r = demo.process_chunk(stream[i:i + 800], (i // 800) * 50.0)
        if r["event"] == "ACTIVATION_TRIGGERED":
            trigs.append(r["timestamp_ms"])
    return trigs


class TestVADGate(unittest.TestCase):
    def test_hangover_default_is_10(self):
        # Guards the fix literal: 3 systematically missed short keywords.
        self.assertEqual(EnergyVAD().hangover_frames, 10)
        demo, *_ = _fresh_demo()
        self.assertEqual(demo.vad.hangover_frames, 10)

    def test_hangover_tail_length(self):
        vad = EnergyVAD()
        speech = (np.random.RandomState(1).randn(800) * 0.1).astype(np.float32)
        self.assertTrue(vad.is_speech(speech))
        tail = [vad.is_speech(np.zeros(800, np.float32)) for _ in range(11)]
        self.assertEqual(tail, [True] * 10 + [False])

    def test_s5_wide_two_triggers(self):
        demo, _, _, sil, _, _ = _fresh_demo()
        z2 = _wav("zora_david_rate+1_var0.wav")
        trigs = _run(demo, np.concatenate([sil, z2, sil, sil, z2, sil]))
        self.assertEqual(trigs, [1650.0, 4650.0])  # kw ends ~1310/~4310

    def test_s3_noisy_single_trigger(self):
        demo, _, _, sil, noise, _ = _fresh_demo()
        z2 = _wav("zora_david_rate+1_var0.wav")
        trigs = _run(demo, np.concatenate([noise, z2, sil, sil[:8000]]))
        self.assertEqual(trigs, [1750.0])  # kw end ~1310

    def test_s1_pilot_catches_first_keyword(self):
        # kw#2 (end ~2310) falls inside the 1500ms cooldown from 1650 --
        # a spacing artifact of this stream, not a detector miss (S5 proves
        # back-to-back detection when spacing exceeds cooldown).
        demo, _, _, sil, _, _ = _fresh_demo()
        z2 = _wav("zora_david_rate+1_var0.wav")
        trigs = _run(demo, np.concatenate([sil, z2, z2, sil]))
        self.assertEqual(trigs, [1650.0])

    def test_s4_negatives_stay_silent(self):
        # Longer hangover must not buy FA: 4s of sil/noise/tone -> 0.
        demo, _, _, sil, noise, tone = _fresh_demo()
        trigs = _run(demo, np.concatenate([sil, noise, tone, sil]))
        self.assertEqual(trigs, [])


    def test_s2_cold_start_detected(self):
        # Ring pre-filled at construction: keyword at 500-1500ms (word end
        # ~810) reaches inference on zero-padded windows. Pre-fix: [].
        demo, _, _, sil, _, _ = _fresh_demo()
        z2 = _wav("zora_david_rate+1_var0.wav")
        trigs = _run(demo, np.concatenate([sil[:8000], z2, sil, sil[:8000]]))
        self.assertEqual(trigs, [1150.0])

    def test_s0_keyword_at_t0_detected(self):
        # Extreme edge: keyword starts at t=0 (word end ~310ms).
        demo, _, _, sil, _, _ = _fresh_demo()
        z2 = _wav("zora_david_rate+1_var0.wav")
        trigs = _run(demo, np.concatenate([z2, sil, sil]))
        self.assertEqual(trigs, [650.0])

if __name__ == "__main__":
    unittest.main()

