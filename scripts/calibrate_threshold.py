#!/usr/bin/env python3
"""
Phase-1 (Change 1b): per-keyword threshold auto-calibration.

Method (Kim et al. 2019 "Query-by-Example On-Device KWS" shuffled-audio idea):
  positives = cosine sims of the K enrollment clips vs their centroid
  negatives = sims of (a) shuffled-chunk fakes of each enrollment clip
                       (b) sliding windows over background/noise audio
  sweep tau in [0.50, 0.99], maximize Youden's J = TPR + TNR - 1
    (ties broken toward HIGHER tau: SIH wants near-zero FA first)
  also report the EER point (|FAR - FRR| minimal) for the viva/report.

Encoder: deployed INT8 .tflite via tf.lite.Interpreter (same bytes as firmware).
Features: src/features/mfcc.py (13 x 98, zero-pad / center-crop to 16000 samples).

Usage:
  python scripts/calibrate_threshold.py --audio-dir data/raw/custom_keywords/zora \\
      --background data/raw/noise --write-config
  python scripts/calibrate_threshold.py --self-test   # no data needed (synthetic tones)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO_ROOT)

import soundfile as sf  # noqa: E402
import yaml  # noqa: E402

from src.features.mfcc import MFCCFeatureExtractor  # noqa: E402
from src.models.prototype import compute_prototype  # noqa: E402
from src.models.tflite_quant import quantize_input, dequantize_output, io_quant_params  # noqa: E402

SR = 16000
CLIP_SAMPLES = 16000  # 1.0 s -> 98 MFCC frames (win480/hop160)
N_FRAMES, N_MFCC = 98, 13
DEFAULT_MODEL = os.path.join(_REPO_ROOT, "models", "tflite", "voice_activator_int8.tflite")
DEFAULT_CONFIG = os.path.join(_REPO_ROOT, "configs", "config.yaml")


class TfliteEncoder:
    """Thin wrapper around the deployed encoder (Change 3c: quantization-aware
    I/O; exact pass-through for the current float32-I/O model)."""

    def __init__(self, model_path=DEFAULT_MODEL):
        import tensorflow as tf  # lazy: self-test needs it too, but import errors stay local
        self._itp = tf.lite.Interpreter(model_path=model_path)
        self._itp.allocate_tensors()
        self._in = self._itp.get_input_details()[0]["index"]
        self._out = self._itp.get_output_details()[0]["index"]
        _, self._in_dtype, self._in_scale, self._in_zp = io_quant_params(
            self._itp.get_input_details()[0])
        _, _, self._out_scale, self._out_zp = io_quant_params(
            self._itp.get_output_details()[0])
        self._mfcc = MFCCFeatureExtractor(sample_rate=SR)

    def embed(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).flatten()
        if len(audio) >= CLIP_SAMPLES:  # center-crop long clips
            start = (len(audio) - CLIP_SAMPLES) // 2
            audio = audio[start:start + CLIP_SAMPLES]
        else:  # zero-pad short clips
            audio = np.pad(audio, (0, CLIP_SAMPLES - len(audio)))
        feat = self._mfcc.extract(audio)
        if feat.shape[0] >= N_FRAMES:
            feat = feat[:N_FRAMES, :]
        else:
            feat = np.pad(feat, ((0, N_FRAMES - feat.shape[0]), (0, 0)))
        x = quantize_input(feat.reshape(1, N_FRAMES, N_MFCC, 1), self._in_scale, self._in_zp, self._in_dtype)
        self._itp.set_tensor(self._in, x)
        self._itp.invoke()
        emb = dequantize_output(self._itp.get_tensor(self._out)[0], self._out_scale, self._out_zp)
        n = float(np.linalg.norm(emb))
        return emb / n if n > 0 else emb


def shuffled_fake(audio: np.ndarray, chunk_ms=100, seed=0) -> np.ndarray:
    """Kim'19 query-specific negative: split into chunks, shuffle, re-concat (same length)."""
    rng = np.random.RandomState(seed)
    chunk = int(SR * chunk_ms / 1000)
    n = len(audio) // chunk
    if n < 2:
        return audio[::-1].copy()  # too short: time-reverse instead
    pieces = [audio[i * chunk:(i + 1) * chunk] for i in range(n)]
    rng.shuffle(pieces)
    out = np.concatenate(pieces)
    tail = audio[n * chunk:]
    return np.concatenate([out, tail]) if len(tail) else out


def bg_windows(bg_audio: np.ndarray, hop_s=0.5, cap=200) -> list:
    hop = int(SR * hop_s)
    wins = []
    for start in range(0, max(len(bg_audio) - CLIP_SAMPLES + 1, 1), hop):
        wins.append(bg_audio[start:start + CLIP_SAMPLES])
        if len(wins) >= cap:
            break
    return wins


def load_wavs(paths: list) -> list:
    clips = []
    for p in paths:
        audio, sr = sf.read(p, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:  # cheap linear resample (numpy-only)
            idx = (np.arange(int(len(audio) * SR / sr)) * sr / SR).astype(int)
            audio = audio[np.clip(idx, 0, len(audio) - 1)]
        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            audio = audio / peak * 0.89  # match enroll.py peak normalization
        clips.append(audio.astype(np.float32))
    return clips


def sweep_threshold(pos: np.ndarray, neg: np.ndarray) -> dict:
    taus = np.arange(0.50, 1.00, 0.01)
    tprs = np.array([float(np.mean(pos >= t)) for t in taus])
    tnrs = np.array([float(np.mean(neg < t)) for t in taus])
    youden = tprs + tnrs - 1.0
    best = float(np.max(youden))
    cands = np.where(youden >= best - 1e-9)[0]
    tau_star = float(taus[cands[-1]])  # tie -> higher tau (FA-first)
    eer_idx = int(np.argmin(np.abs((1 - tprs) - (1 - tnrs))))
    return {
        "tau_star": round(tau_star, 2),
        "tpr_at_star": round(float(tprs[cands[-1]]), 4),
        "tnr_at_star": round(float(tnrs[cands[-1]]), 4),
        "eer_tau": round(float(taus[eer_idx]), 2),
        "eer": round(float(1 - tprs[eer_idx]), 4),
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
    }


def write_config_tau(config_path: str, tau: float) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    old = cfg["detection"]["threshold"]
    cfg["detection"]["threshold"] = float(tau)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[CONFIG] detection.threshold: {old} -> {tau} ({config_path})")


def self_test() -> int:
    """End-to-end mechanics check on synthetic tones (no data files needed)."""
    print("[SELF-TEST] synthetic tones through real encoder + MFCC ...")
    rng = np.random.RandomState(42)
    t = np.arange(CLIP_SAMPLES) / SR
    env = np.sin(np.pi * np.arange(CLIP_SAMPLES) / CLIP_SAMPLES) ** 2
    kw = [(np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 660 * t)) * env
          + 0.02 * rng.randn(CLIP_SAMPLES) for _ in range(3)]
    kw = [k / np.max(np.abs(k)) * 0.89 for k in kw]
    bg = (0.1 * rng.randn(CLIP_SAMPLES * 6)).astype(np.float32)
    enc = TfliteEncoder()
    embs = np.stack([enc.embed(k) for k in kw])
    centroid = compute_prototype(embs)
    pos = np.array([float(np.dot(centroid, e)) for e in embs])
    fakes = [shuffled_fake(k, seed=i) for i, k in enumerate(kw)]
    neg = [float(np.dot(centroid, enc.embed(f))) for f in fakes]
    neg += [float(np.dot(centroid, enc.embed(w))) for w in bg_windows(bg)]
    rep = sweep_threshold(pos, np.array(neg))
    print(json.dumps(rep, indent=2))
    assert 0.50 <= rep["tau_star"] <= 0.99, "tau* out of sweep range!"
    assert rep["n_pos"] == 3 and rep["n_neg"] == 3 + len(bg_windows(bg))
    print("[SELF-TEST] PASS (pipeline mechanics OK; values meaningless on tones)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-keyword threshold calibration (Kim'19 shuffled negatives).")
    ap.add_argument("--audio-dir", default=None, help="Dir with K enrollment .wav clips")
    ap.add_argument("--background", default=None, help="Background/noise .wav file or dir")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--write-config", action="store_true", help="Write tau* into configs/config.yaml")
    ap.add_argument("--out", default=None, help="Write JSON report here")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.audio_dir or not args.background:
        ap.error("--audio-dir and --background required (or use --self-test)")

    kw_paths = sorted(glob.glob(os.path.join(args.audio_dir, "*.wav")))
    assert kw_paths, f"no wavs in {args.audio_dir}"
    if os.path.isdir(args.background):
        bg_paths = sorted(glob.glob(os.path.join(args.background, "*.wav")))
    else:
        bg_paths = [args.background]
    assert bg_paths, "no background audio found"

    print(f"[LOAD] {len(kw_paths)} keyword clips, {len(bg_paths)} background files")
    enc = TfliteEncoder(args.model)
    kw_clips = load_wavs(kw_paths)
    embs = np.stack([enc.embed(c) for c in kw_clips])
    centroid = compute_prototype(embs)
    pos = np.array([float(np.dot(centroid, e)) for e in embs])

    neg = []
    for i, c in enumerate(kw_clips):
        neg.append(float(np.dot(centroid, enc.embed(shuffled_fake(c, seed=i)))))
    for bp in bg_paths:
        (bg_audio,) = load_wavs([bp])
        for w in bg_windows(bg_audio):
            neg.append(float(np.dot(centroid, enc.embed(w))))
    neg = np.array(neg)

    rep = sweep_threshold(pos, neg)
    rep["pos_min"] = round(float(np.min(pos)), 4)
    rep["neg_max"] = round(float(np.max(neg)), 4)
    print(json.dumps(rep, indent=2))
    if rep["neg_max"] >= rep["tau_star"]:
        print(f"[WARN] hardest negative ({rep['neg_max']}) beats tau* "
              f"-> expect residual FA; add Change-2 garbage prototype.")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
        print(f"[OUT] {args.out}")
    if args.write_config:
        write_config_tau(DEFAULT_CONFIG, rep["tau_star"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
