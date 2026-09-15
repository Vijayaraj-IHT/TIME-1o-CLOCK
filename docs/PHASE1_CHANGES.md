# Phase-1 Changes ①–⑥ (canonical definitions)

Single operating point for SIH-26172: **tau=0.89 / tau_low=0.84 / smooth=8 /
persist=4 / cooldown=1500ms / garbage-margin=0.05**, plus adaptive threshold
(⑤) and latency telemetry (⑥). `configs/config.yaml → detection` is the single
source of truth; `tests/test_operating_point.py` guards Python ↔ firmware sync.

| # | Name | One-line | Key files |
|---|------|----------|-----------|
| ① | Threshold unification + calibration | One tau (0.89) everywhere; `calibrate_threshold.py` auto-calibrates per keyword (Kim'19 shuffled negatives + Youden sweep) | `scripts/calibrate_threshold.py`, `configs/config.yaml`, `demo_pipeline.py`, fw headers |
| ② | Garbage prototype + veto | Non-keyword prototype from negatives; `kw − garbage < margin` vetoes the frame (`-1.0`) | `src/models/prototype.py`, `detector.py`, `enroll.py`, `tests/test_garbage.py` |
| ③ | Streaming-score & lifecycle hygiene | **Reconstructed Sep-14** (no original definition survived; built from the three TRACK_A leftovers): ③a VAD-silence feeds the SM as UNOBSERVED — still decays smoothing, but never teaches the ⑤ background model (silence used to dilute tau_eff in noisy rooms); ③b history/persistence reset on activation, cooldown-exit, and stream gaps > `stale_gap_ms`=500 or backwards clock jumps (bg stats + ⑥ marker deliberately kept); ③c quant-aware TFLite I/O helpers mirroring fw `main.cpp` exactly (bare `astype` silently feeds garbage to full-int8 models; exact pass-through for the current float32-I/O models, verified Sep-2026) | `state_machine.py`, `src/models/tflite_quant.py`, `activator_state_machine.h`, `tests/test_change3_hygiene.py` (16 tests) \| ③d (found on real audio): persistence can complete ON a silence frame — such triggers now propagate (demo `ACTIVATION_TRIGGERED` + ASR capture, detector `is_activated`, fw activation print) instead of vanishing as VAD_SKIP |
| ④ | Temporal integration | Smoothing 5→8 (~400ms), persistence 3→4 | `state_machine.py`, fw headers, entry points |
| ⑤ | Adaptive background-tracking threshold | `tau_eff = max(tau, bg_mean + k·bg_std)`, capped at 0.95. EMA learns ONLY on LISTENING + non-triggering frames (keyword frames can't self-suppress). Upward-only → TPR-safe; kill-switch `adaptive_threshold: false` restores fixed tau | `state_machine.py`, `activator_state_machine.h` (+12 B RAM), `tests/test_adaptive_threshold.py` \| FW PORT COMPLETED Sep-14 under ③ (the header had ⑤'s members/params but `update()` never used them — fixed tau on-device; the algorithm now mirrors Python, proven by `tools/verify_sm_parity/`: 28/28 trace lines match) |
| ⑥ | Keyword-END → trigger latency telemetry | Bench/sim marks ground-truth keyword ends; activations report signed `trigger_latency_ms` (negative = early). `LatencyStats` judges p50/p99/max vs 1000ms budget | `src/evaluation/latency.py`, `experiments/threshold/benchmark_detection_latency.py`, `tests/test_latency.py` |

## Re-validation after VAD-miss fix (sandbox, Sep-14 -- hangover 3 -> 10)

- `benchmark_detection_latency.py` (real VAD + state machine, scripted
  encoder): re-run output **byte-identical** to the committed JSON -- all 4
  scenarios PASS, short-keyword negative control still 0 triggers. The
  operating point (tau 0.89 / smooth 8 / persist 4) is unaffected.
- 60 s bursty-negative stream (1 s loud-noise bursts + 2 s silence x20)
  through the INT8 demo: **0 triggers at hangover 3 and 10**; measured CPU
  cost of the fix = +7 inferences per burst (441 -> 581 total, skip
  62.7% -> 50.8%). Silence-dominated idle is unaffected (counter stays 0).
- `benchmark_streaming_detector.py` + `benchmark_robustness_and_far.py`
  could NOT be re-run in sandbox (missing `zora_zira_*` wavs, speech
  commands, test_manifest.csv -- user-side data). Expected deltas when the
  user re-runs: VAD skip % down (~12 pts on speech-active audio), short-kw
  TPR up, FA count must NOT rise (sandbox: 0 on 60 s bursty).

## MEASURE-1 (user-side, Windows, needs audio)

1. `python scripts/calibrate_threshold.py --audio-dir data\raw\custom_keywords\zora --background data\raw\noise --write-config --out measure1_tau.json`
2. `python experiments/threshold/benchmark_robustness_and_far.py` (sweep row tau=0.78 ≈ BEFORE, tau=0.89 = AFTER)
3. `python experiments/threshold/benchmark_streaming_detector.py` (optional)
4. `python experiments/threshold/benchmark_detection_latency.py` (runs anywhere; synthetic)

Pass: FA/hr ≤ 100 (win) / ≤ 10 (dream); TPR drop ≤ 3%; latency verdict PASS.
Report back: FA/hr, TPR, τ*, garbage_negatives_n, veto counts, latency verdict.

## MEASURE-1 watch items (found during ③ verification)

- ⑤ transient: from a zero start, the EMA variance overshoots, so tau_eff
  pins at the 0.95 cap for ~200 frames (~10s) before settling (~0.91 for
  the 0.87/0.79 confuser bg; ~600 frames to fully settle). The 40-frame
  warmup does not cover it. If noisy_bg TPR sags in the first ~30s, the
  knobs are adapt_alpha/adapt_warmup_frames — measure first, then tune.
- `detection_latency_results.json` was regenerated under ③a
  (double_keyword's second-trigger tau_eff 0.9088 -> 0.89: VAD-silence no
  longer counts as background experience). Counts, latencies, verdicts
  unchanged — the delta is telemetry honesty, not detection behavior.
- Keyword-onset frames below tau (pre-VERIFYING) still feed the bg EMA in
  both Python and fw — inherent to the LISTENING+non-triggering rule (the
  keyword is unknowable that early). Upward-only + cap bound it.

## Ablation (isolating ⑤)

Set `detection.adaptive_threshold: false` in `configs/config.yaml` (or pass
`adaptive=False` / `DetectionStateMachine(adaptive=False)`) and re-run any
bench: the difference vs default is exactly Change-5's contribution.
Firmware: construct with `adaptive=false` (6th ctor arg... default `true`).

## Real-audio smoke (sandbox, Sep-14 — NOT a substitute for MEASURE-1)

Restored 3 real zora wavs from the merged main (`raspberry_pi_deployment/…`,
16kHz/1s, peak ~0.95) + real `voice_activator_int8.tflite`, 2-shot enroll:

- Embedding matrix (clean, stateless): zora↔zora 0.99+, zora↔white-noise
  0.30–0.35, zora↔silence 0.67–0.71, zora↔tone 0.64–0.67. Encoder healthy,
  no collapse. (The 0.9993 intra-sim is TTS-identical takes, not a bug.)
- Streaming 2×zora: 2nd keyword TRIGGERED at ts=2500ms = keyword END
  (ideal latency) — completing ON the silence frame (this find became ③d).
- Streaming 3s white noise + 1s silence: 0 activations (noise → ~0.0 sim).
- 1st keyword (cold start) MISSED: onset frames are mostly silence in the
  1s window → correctly vetoed (-1.0), and the ~500ms keyword ends before
  persistence completes. Consistent with the ⑥ short-keyword control
  (400ms → 0 by design). zora TPR is onset-sensitive; knobs are veto
  margin / persistence / garbage strength — MEASURE-1 data decides.

## Ablation (isolating ③)

- ③b gap reset: set `detection.stale_gap_ms` to a huge value (or construct
  with `stale_gap_ms=99999999`) — activation/cooldown clears stay, gap
  clears go. Fw: last ctor arg.
- ③a has no one-line kill-switch (it is call-site behavior); pass
  `observed=True` at the three VAD_SKIP sites to restore fake-0 learning.
- ③c: float models are unaffected by construction (pass-through is
  bit-exact); it only changes behavior for full-integer models.
- ③d has no kill-switch (it is trigger-reporting honesty); to restore the
  old miss, ignore `is_activated` on VAD_SKIP / silence frames.
