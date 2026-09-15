# Track-B: DTW Template Spotter (classical fallback / small-vocabulary path)

Status: **prototype + pilot measurements** (Sep-14). This doc records the
algorithms as implemented in `src/dtw/`, the ESP32 resource budget, and what
the pilot experiment did and did not prove. Nothing here is validated past
pilot grade (3 keyword clips + synthetic negatives); the MEASURE-1 dataset
decides the real verdict.

## 1. Why DTW exists alongside the neural path

| concern | neural (Track-A) | DTW (Track-B) |
|---|---|---|
| enrollment data | 100s of clips for metric training | 2-5 clips per keyword, no training |
| new keyword on-device | re-enroll prototype (cheap) but encoder fixed | record clips, build centroid (cheap) |
| flash | ~63 KB model (int8) | ~5 KB float32 / ~1.3 KB int8 per keyword |
| RAM (streaming) | ~96 KB arena (+frontend) | ~1.2 KB (+shared frontend) |
| rate/accent variation | learned invariance (cos 0.99 on held-out rate) | raw alignment (pilot: 130x intra cost on new rate) |
| failure mode | VAD gate can skip completion windows (observed!) | warping absorbs timing; threshold trades latency/FA |

DTW is the low-data, tiny-footprint complement: instant enrollment of new
keywords (a stated SIH nice-to-have) and a fallback if the neural path misses
its FA/accuracy budget. It is NOT a replacement unless MEASURE-1 says so.

## 2. Algorithms (as implemented)

### 2.1 Banded DTW + LB_Keogh (`src/dtw/dtw_core.py`)

- `dtw_distance(a, b, band=8)`: classic DP with squared-Euclidean local cost,
  Sakoe-Chiba band, path-length normalization (cost/steps). Returns `inf`
  when `|len(a)-len(b)| > band` (built-in duration gate for isolated clips).
- `dtw_envelope(t, band)` + `lb_keogh(q, upper, lower)`: the standard lower
  bound. Pilot: LB alone separates (`kw 0.47` vs `neg >= 40`), so LB works
  as a cheap pre-filter before full DTW (firmware: skip full column when
  LB > threshold).
- All distances in per-step units; comparable across templates lengths.

### 2.2 Centroid templates (`src/dtw/templates.py`)

- `resample_frames(m, T)`: linear interpolation to T frames (endpoints
  preserved). Enrollment clips vary in length; templates are fixed at the
  median clip length (98 frames here).
- `build_template(name, clips, band)`: centroid = mean of resampled clips +
  envelope + intra-enrollment distances (the honest spread metric: pilot
  intra = 0.0195/step on two same-rate clips).

### 2.3 Streaming subsequence spotter (`src/dtw/spot.py`)

True O(T) online subsequence DTW (open begin + open end), NOT a sliding
window re-computation:

- One cumulative-cost column + step-count column over the T template rows.
- Each 10 ms MFCC frame advances the column: `new_cost[0] = 0` every frame
  (open begin: a match may start anywhere); local cost = squared Euclidean
  over 13 MFCCs; transitions = insertion/deletion/match.
- Trigger when full-template cost-per-step < threshold. On trigger: columns
  clear (same-keyword tail must not re-fire; mirrors neural Change-3b) +
  cooldown counter (default 150 frames = 1500 ms).
- No Sakoe-Chiba band in streaming (no diagonal anchor exists for
  subsequence DTW). Pathological warp guard (min stream-frames consumed)
  is a known hardening gap -- see section 6.

## 3. Pilot measurements (all reproduced by one command)

```
python experiments/track_b/compare_dtw_vs_neural.py   # needs the 3 zora wavs + TF
```

Results JSON: `experiments/track_b/track_b_pilot_results.json`.
Template: "zora" from `zora_david_rate+0_{var0,var1}`; held-out keyword =
`zora_david_rate+1_var0` (different speaking rate -- the generalization test).
Negatives: silence / white noise / 300 Hz tone (synthetic, 1 s each).

### 3.1 Isolated clips (per-step cost; neural = cosine to prototype)

| clip | DTW/step | LB_Keogh | neural cos |
|---|---|---|---|
| z2 held-out (rate+1) | **2.5340** | 0.4701 | 0.9898 |
| silence | 152.02 | 39.76 | 0.7097 |
| noise | 197.48 | 129.04 | 0.3492 |
| tone | 187.64 | 86.66 | 0.6494 |

Separation is 60x (DTW) -- but note the rate effect: z2 costs **130x the
intra-template spread (0.0195)**. The neural embedding generalizes across
rate far better (0.99). Lesson: DTW enrollment MUST span speaking rates
(multi-rate clips, or one template per rate variant at ~5 KB each).

### 3.2 Shared 4 s stream: sil + z2 + z2 + sil

Word content measured at clip frames 4..28 (energy contour), so word#1 ends
~1310 ms, word#2 ends ~2310 ms.

| system | triggers | interpretation |
|---|---|---|
| DTW (thr 3.0, cooldown 800 ms) | **1360, 2360** | word-end + 50 ms, 2/2 |
| DTW (thr 3.0, cooldown 1500 ms) | 1360, 2870 | #2 delayed: cooldown blinded 1360-2860 |
| neural demo (thr 0.82, VAD hangover 10) | 1650 only | kw#1 caught at word-end+340ms; kw#2 (end ~2310) inside 1500ms cooldown -- spacing artifact, not a miss (S5 wide-spacing: 2/2, see VAD fix note) |

Three findings:

1. **DTW latency = word-end + ~50 ms**, stable across thresholds 2.6-77
   (operating range, not a knife-edge). Trigger sits at word end because the
   template's trailing-silence rows absorb post-word frames.
2. **Cooldown must be < keyword spacing.** 1500 ms cooldown on 1000 ms
   spacing delayed (not lost -- trailing silence saved it) the second
   detection. Firmware rule: cooldown ~= 800 ms, or suppress-until-silence.
3. **VAD-miss found here, fixed Sep-14.** At pilot time (hangover 3) the
   neural path missed keyword #1: VAD_SKIP 1500-1900ms skipped its completion
   windows. Hangover sweep (`experiments/vad_gate/repro_vad_miss.py`) proved
   it systematic for short keywords and set hangover 10 (host + firmware):
   clean/noisy/warm streams now trigger, negatives stay silent
   (`tests/test_vad_gate.py`, FIXES_APPLIED.md). The table above already
   reflects the fix. MEASURE-1 must still confirm on real speech negatives.

### 3.3 What the pilot does NOT prove

- No real negatives (no speech, no TV babble, no confusable words). 60x
  isolation margin will shrink; the honest number needs MEASURE-1 negatives.
- No threshold calibration (midpoint/EER on dev data still to do -- the
  script picks a midpoint only to demonstrate the sweep).
- No multi-speaker / far-field / noise robustness data.
- No on-device timing yet (host prototype only).

## 4. ESP32 budget (derived, T=98, dim=13, float32)

| resource | DTW/keyword | neural (reference) | SIH ceiling |
|---|---|---|---|
| flash | 98*13*4 = **5096 B** (1274 B int8-quantized) | ~63 KB model | plenty |
| RAM streaming | 2 col sets x 99 x 12 B = **2376 B** | ~96 KB arena | <256 KB |
| compute | 98 rows x (13 sub/mul/add + min) ~= **4.4 kflops** / 10 ms frame | TFLM invoke / 50 ms | <10% idle CPU |
| CPU @240 MHz | 440 kflops/s -> **<0.5%** (FPU ~1-2 flops/cycle) | device-timed separately | plenty |
| trigger latency | word-end + 50 ms (measured, host) | word-end + ~150-500 ms (smoothing+VAD) | keyword-END->ASR |

Multi-keyword: +5 KB flash each, compute xK (LB_Keogh pre-filter keeps the
average far below worst case). Template lives in flash, streamed per frame
(5 KB/frame x 100 Hz = 500 KB/s read -- trivial).

## 5. Firmware port (done Sep-14; flash after MEASURE-1 blesses the data)

- `scripts/export_dtw_template.py` generates `dtw_template_zora.h`
  (centroid + EER threshold + gate/cooldown, `--check` mode like the mel
  exporter). `src/deployment/esp32/dtw_spotter.h` is the C++ port (float32,
  zero-alloc, fixed 128-frame max); both ride the firmware sync list
  (now 8 shared headers).
- Parity gate `tools/verify_dtw_parity/check_parity.py`: recorded 398-frame
  pilot stream through host AND `g++` firmware code at the header operating
  point (thr 71.93, cd 150, gate 12): **identical trigger frames (133, 284),
  |dcost| <= 2.3e-5, identical consumed (28, 79). PARITY PASS.**
- Cost/frame on ESP32-S3: 98 rows x (13 MAC + min) ~= 4.4 kflops @100 Hz =
  0.44 MFLOPS (<0.5% CPU); 5096 B flash + 2.4 KB RAM per keyword.
- Open: LB_Keogh per-frame pre-filter for multi-keyword banks (envelope in
  flash ~10 KB, or cached on the fly); int8 template quantization if flash
  pressure; cooldown 800 ms + suppress-until-silence at integration time.

## 6. Hardening (done Sep-14) + remaining gaps

- **Duration gate DONE** (`src/dtw/spot.py`): third column counts consumed
  stream frames; triggers need `consumed >= T//8`. Measured on the pilot:
  genuine matches consume **31 frames** (T=98); degenerate single-frame
  matches consume 1. T//8=12 sits 2.6x below genuine and 2.4x+ above
  degenerate (a first-cut T//3=32 sat exactly ON genuine=31 -- zero margin,
  rejected by measurement). The pathological case is fenced by
  `test_gate_blocks_single_frame_match` (with an ungated control proving the
  degenerate match exists). Genuine latency unchanged (1360/2360).
- **Multi-rate enrollment: bank CODE done, data pending** (`DTWSpotterBank`):
  one centroid per variant, ORed triggers, shared cooldown + column reset
  (one keyword = one trigger). Synthetic proof: rise-vs-fall variants each
  trigger exactly once with correct labels. Design rule learned the hard
  way: members must be SHAPE-distinct -- pure time-scale copies collapse
  under warping (caught by a swapped-label test failure). Real rate proof
  needs MEASURE-1 (>= 2 clips/rate); the 130x pilot gap justifies the slot.
- **Threshold calibration: EER script done**,
  `experiments/track_b/calibrate_dtw_threshold.py` (no TF; LOO keyword
  scores vs --neg-dir or synthetic fallback). Pilot:
  kw LOO 1.98/2.06/2.53 vs neg 141-192, SEPARABLE, EER 0.0 @ thr 71.9
  (`experiments/track_b/dtw_threshold.json`). The streaming sweep (stable
  2.6-77) validates the operating RANGE, not just the point. Re-tune on
  MEASURE-1 with real negatives.
- **Still open: confusable-word rejection.** DTW has no garbage model;
  consider anti-templates (closest-negative distance must exceed a margin)
  if MEASURE-1 FA bites.

## 7. File map + rerun

- `src/dtw/dtw_core.py` -- banded DTW, envelope, LB_Keogh
- `src/dtw/templates.py` -- resample + centroid builder
- `src/dtw/spot.py` -- streaming spotter
- `tests/test_dtw.py` -- 10 tests (core math, templates, spotter incl. double-trigger)
- `experiments/track_b/compare_dtw_vs_neural.py` -- pilot harness (isolated + stream + sweep + JSON)
- `experiments/track_b/track_b_pilot_results.json` -- measured pilot numbers

```bash
python -m pytest tests/test_dtw.py -q          # 10/10
python experiments/track_b/compare_dtw_vs_neural.py   # pilot, writes JSON
```
