# MEASURE-1 recording protocol (user-side)

MEASURE-1 is the first measurement on REAL data: calibrate τ*, then report
FA/hr, TPR, garbage stats, veto counts, latency verdict. This doc fixes WHAT
to record, WHERE it lands, and HOW it is validated. Anything recorded outside
this layout will not feed the scripts.

## 1. Directory layout (fixed)

```
data/raw/
  custom_keywords/<keyword>/*.wav   # keyword clips (calibrate --audio-dir points here)
  noise/*.wav                       # ambient backgrounds (calibrate --background)
  negatives/*.wav                   # confusables + speech-like non-keywords
```

Already seeded: `data/raw/custom_keywords/zora/` (3 TTS clips -- pilot only,
NOT measurement data).

## 2. File naming (fixed -- the manifest parser depends on it)

- Keywords: `<keyword>_<speaker>_rate<±N>_var<V>.wav`
  e.g. `zora_priya_rate+0_var3.wav`, `zora_arun_rate-1_var0.wav`
  (`rate+0` = natural pace, `+1` = fast, `-1` = slow; `var` = take number.)
- Noise: `<place>_<idx>.wav`, e.g. `room_fan_01.wav`, `street_02.wav`
- Negatives: `<category>_<source>_<idx>.wav`,
  e.g. `confusable_sora_001.wav`, `babble_tv_003.wav`, `cough_002.wav`

## 3. Recording targets (minimum for a pilot-grade MEASURE-1)

| set | target | notes |
|---|---|---|
| keyword clips | ≥3 speakers × 3 rates × 5 takes = 45 | same mic/room mix as deployment |
| ambient noise | ≥10 min total, ≥3 places | fan, street, kitchen... |
| confusables | ≥30 clips (rhymes: sora/zoro/hour...) | same speakers as keywords |
| speech babble | ≥5 min (TV/radio/conversation) | drives FA/hr honestly |

Format: **16 kHz mono 16-bit WAV**. Keyword clips 1-2 s with ~200 ms lead
silence (word must NOT be clipped at either edge -- QC flags edge energy).

## 4. How to record (phone or PC -- no special hardware)

Phone (easiest): any voice-recorder app, hold 30-50 cm away, then convert:
```
ffmpeg -i input.m4a -ac 1 -ar 16000 -sample_fmt s16 clip.wav
```
PC with mic: `python tools/measure1/record.py --keyword zora --speaker priya --takes 5`
(prompts per take, auto-names files, needs `pip install sounddevice`).

Split long recordings into clips with any editor, or record take-by-take.

## 5. Validate before measuring (mandatory)

```
python tools/measure1/manifest.py --root data/raw --out measure1_manifest.json
python tools/measure1/qc_audio.py --root data/raw --out measure1_qc.json
```

`manifest.py` inventories + checks naming/layout/counts. `qc_audio.py`
checks every file (sample rate, mono, duration, peak/RMS, clipping %,
DC offset, edge energy) and exits non-zero on any ERROR. Fix flagged clips
(re-record or delete), re-run until clean. Attach both JSONs with results.

## 6. Run the measurement (after QC is clean)

```
python scripts/calibrate_threshold.py --audio-dir data\raw\custom_keywords\zora --background data\raw\noise --write-config --out measure1_tau.json
python experiments/threshold/benchmark_robustness_and_far.py
python experiments/threshold/benchmark_detection_latency.py
```

Report back: FA/hr, TPR, τ*, garbage_negatives_n, veto counts, latency
verdict (+ the manifest/QC JSONs). Pass: FA/hr ≤ 100 (win) / ≤ 10 (dream);
TPR drop ≤ 3%; latency verdict PASS.
