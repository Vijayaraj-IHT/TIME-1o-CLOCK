# Track A — ESP32 Feature-Extractor Fix & Verification

Scope (as explicitly requested): fix the two independent blockers before any DTW/algorithm
work touches hardware —
1. ESP32 feature extractor wasn't actually computing FFT → Mel.
2. `float preemph[16000]` — a 64 KB stack allocation, plausible direct cause of "DRAM exceeded".

Both fixed. Verified numerically, not just by code inspection. Everything below is measured
from actually compiling and running the real firmware header, not simulated.

## What changed

- **`src/deployment/esp32/feature_extractor.h`** (and identical copies in
  `src/deployment/esp32_wroom/` and `outputs/esp32_wroom/` — a third, previously-unnoticed
  copy of the same broken logic existed in the WROOM variant and is now fixed too):
  - Replaced the fake `compute_mel_filterbank()` (which summed `abs()` of raw time-domain
    samples, no FFT anywhere) with a real in-place iterative radix-2 Cooley-Tukey FFT
    (512-point, matching `FFT_SIZE`/`n_fft`) followed by a genuine Mel filterbank multiply.
  - The Mel filterbank is not re-derived by hand in C++ (a re-derivation risks subtle
    mismatches with librosa's exact HTK/slaney formula). Instead it's **exported directly**
    from the same `librosa.filters.mel(sr=16000, n_fft=512, n_mels=40, fmin=20, fmax=4000,
    htk=True, norm='slaney')` call `logmel.py` uses for training, into a new
    `mel_filterbank.h` (a 40×257 `static const float` table).
  - Removed pre-emphasis and switched Hamming → Hann window, matching Python's spec exactly
    (Python's `mfcc.py`/`logmel.py`/`audio_features.py` pipeline applies neither pre-emphasis
    nor a Hamming window). Python is the canonical reference here because the model's learned
    weights are tied to whatever distribution it was actually trained on — firmware must
    match Python, not the other way around.
  - This removes `float preemph[16000]` entirely (it only existed to support the pre-emphasis
    step Python never applied) — eliminating the 64 KB stack allocation, not just shrinking it.
  - The remaining FFT/power-spectrum scratch buffers (`re_scratch_`, `im_scratch_`,
    `power_spectrum_scratch_` — ~4 KB total) were made member (static-lifetime) storage
    rather than function-local, so this extractor now has **zero** large stack-local arrays,
    not just a smaller one.

## Verification (the gate proposed earlier — actually executed, not assumed)

Built `tools/verify_feature_parity/verify_parity.cpp`, which compiles the **real, unmodified**
`feature_extractor.h` on the desktop (g++, no ESP32-specific dependencies exist in this file)
and runs it on raw audio, producing an MFCC matrix to diff against Python's `mfcc.py` on
identical input.

| Test signal | Max abs diff | Mean abs diff | Max relative diff |
|---|---|---|---|
| Multi-tone + noise (220/440/1200 Hz mix) | 1.91e-05 | 5.43e-06 | — |
| Broadband white noise | 1.53e-05 | 3.13e-06 | 0.20% |

Both are at float32 rounding-precision level (the residual difference is entirely explained
by my radix-2 FFT vs. NumPy's `rfft` using different intermediate rounding, not an algorithmic
mismatch). **Parity confirmed on the two signals tested.** This has NOT yet been run against
real recorded speech from the actual dataset (not available in this environment) or on actual
ESP32 hardware (timing/fixed-point behavior can differ from a desktop x86 float build) —
both should be done before trusting this in the field; see "Not yet done" below.

## Honest cost accounting of the fix

| | Before (broken) | After (fixed) | Delta |
|---|---|---|---|
| Extractor member/static RAM | ~3.9 KB (Hamming window + DCT basis only) | ~11.9 KB (+ FFT twiddle tables + scratch buffers) | **+8.0 KB static RAM** |
| Worst-case stack allocation in this class | 64 KB (`preemph[16000]`, uncontrolled) | 0 KB (no large stack-local arrays remain) | **-64 KB stack risk** |
| New Flash cost | 0 | `mel_filterbank.h`: 40×257 float32 = 41,120 B ≈ **40.2 KB Flash** | **+40.2 KB Flash** |

Net effect: a real, bounded ~8 KB static RAM increase in exchange for eliminating an
unbounded 64 KB stack landmine — a clear win on the axis that was actually causing crashes.
The **~40 KB Flash addition is the one cost worth flagging**: the Mel filterbank matrix is
stored dense, but it's genuinely sparse — only 247 of 10,280 entries (2.4%) are non-zero
(triangular filters). A sparse (value, index) encoding would cut this to **~1.5 KB**, a ~96%
reduction. Not implemented now (correctness was the priority for this pass), but recorded
here as a concrete, quantified follow-up if Flash becomes tight once the DTW templates or a
second model variant are also added.

## Not yet done (explicitly out of scope for this pass, flagged rather than silently skipped)

- Not tested against real speech / the actual 100K-file dataset (not available in this
  environment) or on physical ESP32 hardware.
- The other bugs surfaced in the prior review pass (INT8 quantization cast in
  `demo_pipeline.py`, VAD-silence-injects-zero-into-smoothing, COOLDOWN not clearing
  `similarity_history`, hardcoded `D:\` paths) are **not** touched here — this pass was
  scoped exactly to the two named blockers, not a general cleanup sweep.
- Track B (dataset → templates → Python DTW prototype → quantitative comparison) has not
  been started; this was the reason the two tracks were split.

## Sparse follow-up (Sep-14): dense 40x257 table -> run encoding

Implemented the Flash optimization flagged above. `scripts/export_mel_filterbank.py`
is now the single source of truth: it computes `librosa.filters.mel()` with the
exact training parameters, verifies the committed bytes against it (drift guard),
and emits the sparse header (re-parse verified on every run).

| | Dense (before) | Sparse (after) |
|---|---|---|
| Flash | 41,120 B | **1,108 B** (247x4 values + 40x3 uint8 headers; `sizeof`-proven) |
| Saving | — | **97.3%** |
| Mult-adds / frame | 10,280 | **247** (~42x fewer) |
| Header source | 176 KB | 7.6 KB |

Correctness (measured, not assumed):
- Sparse-vs-dense firmware binaries on identical input: **bit-identical
  (0.000e+00)** on tones, white noise, AND real zora speech. (Skipping exact
  zeros is not an approximation: `x + 0.0 == x` in IEEE754 for finite `x`.)
- Python-parity gate re-run on the sparse build: multitone 2.29e-05,
  white 1.14e-05, real-zora 3.05e-05 max abs — same float-rounding grade as
  the dense era (1.91e-05 / 1.53e-05). The gate is now committed and
  reproducible: `python tools/verify_feature_parity/check_parity.py`.
- `tests/test_mel_sparse.py` (5 tests): committed bytes == librosa truth,
  exact float32 round-trips, uint8 ranges, Flash budget.

*Update Sep-14: the three "not touched here" leftovers above are now resolved
under Phase-1 Change ③ (see docs/PHASE1_CHANGES.md): VAD-silence feeds the
state machine as UNOBSERVED (③a), cooldown/activation/gap clears history (③b),
and all hardcoded paths are portable (Step-0). Track B still not started.*
