# Loophole Fixes Applied — SIH Problem Statement 26172

Every fix below is a targeted code change, not a rewrite. Inline `// Fix:` / `# Fix:` comments
mark every change at its exact location for viva/judging traceability.

## 1. Threshold drift across config.yaml / demo_pipeline.py / firmware (highest priority)
**Found:** THREE different operating points existed simultaneously:
- `configs/config.yaml`: τ_high=0.78, τ_low=0.73, persistence=3 (this is what `experiments/threshold/*.json` was calibrated against)
- `src/streaming/demo_pipeline.py`: τ_high=0.88, τ_low=0.84, persistence=4
- `src/deployment/esp32/activator_state_machine.h`: τ_high=0.88, τ_low=0.84, persistence=4
- `src/deployment/esp32_wroom/voice_activator_esp32_wroom.ino`: τ_high=0.82, τ_low=0.78, persistence=3

**Fix:** All four now use the config.yaml-calibrated values (0.78 / 0.73 / persistence=3 / smoothing=5 / cooldown=1500ms).
Added `load_calibrated_thresholds()` in `demo_pipeline.py` — the runtime now loads config.yaml directly and
**warns loudly** if any hardcoded value is ever passed that disagrees with it, so this class of bug cannot
silently reappear.

## 2. No model/threshold pairing check
**Found:** `outputs/tflite/` has 3 different `.tflite` variants; nothing verified which one was loaded before applying thresholds.
**Fix:** `verify_model_threshold_pairing()` in `demo_pipeline.py` cross-checks the loaded model filename against
`outputs/tflite/models_manifest.json` and fails fast on a shape/identity mismatch.

## 3. VAD adaptation deadlock (`src/streaming/vad.py`)
**Found:** if ambient noise at boot already exceeded the initial threshold, `background_energy` never entered
its adaptation branch — permanent false "speech detected", continuous CNN inference, violating the <10% idle
CPU budget with no recovery path.
**Fix:** added `max_background_energy` ceiling and a `stuck_speech_limit_frames` forced-recalibration trip that
adopts the observed level as the new floor after sustained "speech" (almost certainly stationary noise, not a
human talking for 10+ seconds straight).

## 4. Ring buffer / loader channel down-mix (4 files)
**Found:** `audio[:, 0]` / `chunk[:, 0]` took channel 0 only on stereo input, silently dropping any signal
wired to another channel — in `ring_buffer.py`, `enroll.py`, `batch_generator.py`, `validate.py`, `evaluate_robustness.py`.
**Fix:** changed to `.mean(axis=1)` everywhere — correct regardless of which channel carries the mic signal.

## 5. Feature-length enforcement (`src/features/audio_features.py`)
**Found:** only `enroll.py` padded/truncated audio to exactly 16000 samples before extraction. Any other
caller producing a slightly different length got a frame count ≠ 98, which the fixed-shape TFLite input
tensor would reject at `Invoke()` — a fatal, un-caught on-device crash.
**Fix:** `frame_audio()` now standardizes every input to `target_num_samples` (1.0s @ sample_rate) itself, so
every caller gets a guaranteed-constant frame count. Also forces the input to a contiguous array before
`as_strided` framing (a non-contiguous view previously risked reading incorrect memory silently).

## 6. Augmentation wraparound corruption (`src/data/augment.py`)
**Found:** `time_shift()` used `np.roll` — a circular shift that can wrap a keyword's ending phoneme to the
start of the clip, producing a corrupted-but-still-labeled training example. Especially damaging for
few-shot enrollment where K=1..3 raw-mean prototypes are highly sensitive to one bad exemplar.
**Fix:** rewritten as a true shift with zero-padding of the vacated region; content is truncated, never wrapped.

## 7. Silent degenerate prototype (`src/models/prototype.py`)
**Found:** a zero-norm mean embedding (degenerate/antipodal encoder output) returned an invalid zero vector
silently. Every future cosine similarity against it is 0 — the keyword becomes permanently undetectable with
no error surfaced.
**Fix:** raises an explicit `ValueError` instead, forcing re-enrollment rather than silent failure.

## 8. Ungated enrollment quality (`src/enrollment/enroll.py`)
**Found:** `intra_similarity_min` was computed but never acted on — an inconsistent K-shot enrollment (bad
volume/pronunciation/wrong word) still got exported to firmware, with failure only surfacing live on stage.
**Fix:** added `MIN_ACCEPTABLE_INTRA_SIMILARITY = 0.5` gate; `enroll()` now returns `is_valid` and warns
loudly on failure. `export_prototype_c_header()` now **refuses** to export an invalid prototype unless
`force=True` is explicitly passed.

## 9. Unhandled empty/singleton classes (`src/training/batch_generator.py`)
**Found:** a manifest class with zero usable records crashed deep inside `sample_supcon_batch()` with an
opaque stack trace mid-training, not at load time.
**Fix:** validated at `__init__` — raises immediately with the offending class names, and separately warns
(non-fatally) about single-recording classes that will be replicated to fill a batch.

## 10. ESP32 firmware: unchecked interpreter after allocation failure (`main.cpp` + `.ino`)
**Found:** `AllocateTensors()` failure only printed an error and `return`'d from `setup()` — but `loop()`/
`app_main()`'s processing loop kept running afterward, invoking a half-initialized `MicroInterpreter`
(undefined behavior on real hardware).
**Fix:** added a `g_system_ready` flag, set only on verified successful allocation; `process_audio_chunk()`
now hard no-ops until it's true. Also bumped `TENSOR_ARENA_SIZE` 64KB → 96KB (the manifest reports the
primary model needs 63KB — the old allocation left under 2% headroom).

## Verified (smoke-tested, see below)
- MFCC/LogMel extractor returns exactly (98, 13) for input lengths 8000/16000/24000/15999 samples.
- Ring buffer correctly averages a 2-channel [1.0, 3.0] input to 2.0, not 1.0.
- VAD recovers from a simulated stuck-loud-noise deadlock within `stuck_speech_limit_frames`.
- `time_shift()` no longer wraps signal content circularly.
- `compute_prototype()` raises on a degenerate zero-norm input instead of returning it.

## Not fixed in this pass (flagged, out of scope for a Python-side patch)
- **ESP32 ring buffer read/write race condition**: `g_ring_buffer`/`g_state_machine` are global, unsynchronized
  state. If I2S DMA fill and the inference loop run on separate FreeRTOS tasks/cores, a torn read during
  `read_window()` is possible. Needs a board-specific mutex/critical-section strategy (`portENTER_CRITICAL` or
  a lock-free double-buffer) matched to your actual task/core layout — flagging rather than guessing at your
  RTOS configuration.
- **INT8 calibration coverage**: whether `quantize.py`'s representative dataset reflects real on-device
  acoustic conditions (vs. clean studio/read-speech source data) should be re-verified empirically by rerunning
  `experiments/quantization/benchmark_quantization.py` against real mic-captured calibration clips before the
  final demo.
