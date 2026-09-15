#!/usr/bin/env python3
"""Regenerate src/deployment/esp32/mel_filterbank.h from librosa (sparse form).

The firmware Mel filterbank is 97.6% zeros (triangular filters: 247/10280
nonzero, one contiguous run per mel). The dense 40x257 float32 table cost
41,120 B of Flash; the sparse run encoding costs 1,108 B (247x4 values +
40x3 uint8 run headers) AND cuts the per-frame multiply-adds 10,280 -> 247.

This script is the single source of truth for the header bytes:
  1. Computes librosa.filters.mel() with the exact training parameters.
  2. Verifies it matches the CURRENT committed header (drift guard).
  3. Emits the sparse header with exact float32 round-trip literals.
  4. Re-parses what it wrote and proves sparse == librosa == old header.

Usage: python scripts/export_mel_filterbank.py [--check]
  --check: verify only (exit 1 on any mismatch), do not write.
After regenerating: run scripts/sync_firmware_headers.py --sync, rebuild
tools/verify_feature_parity, and run tools/verify_feature_parity/check_parity.py.
"""
import os
import re
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HEADER = os.path.join(REPO, "src", "deployment", "esp32", "mel_filterbank.h")

SR, N_FFT, N_MELS = 16000, 512, 40
FMIN, FMAX = 20, 4000


def librosa_truth():
    import librosa
    m = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS,
                            fmin=FMIN, fmax=FMAX, htk=True, norm="slaney")
    return np.asarray(m, dtype=np.float64)


def parse_dense_header(text):
    m = re.search(r"MEL_FILTERBANK\[40\]\[257\] = \{(.*?)\};", text, re.S)
    if not m:
        return None
    toks = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?f", m.group(1))
    return np.array([float(x[:-1]) for x in toks]).reshape(40, 257)


def parse_sparse_header(text):
    def arr(name, n, typ):
        mm = re.search(name + r"\[\d+\] = \{(.*?)\};", text, re.S)
        assert mm, f"{name} not found in header"
        return np.array([typ(x) for x in mm.group(1).replace("\n", " ").split(",")
                         if x.strip()], dtype=np.float64)
    n = int(re.search(r"MEL_FILTERBANK_NNZ = (\d+)", text).group(1))
    return (n, arr("MEL_SPARSE_START", 40, int), arr("MEL_SPARSE_LEN", 40, int),
            arr("MEL_SPARSE_OFF", 40, int),
            np.array([float(x[:-1]) for x in
                      re.search(r"MEL_SPARSE_VAL\[\d+\] = \{(.*?)\};", text, re.S)
                      .group(1).replace("\n", " ").split(",") if x.strip()]))


def runs_of(mat):
    runs = []
    for r in range(mat.shape[0]):
        idx = np.where(mat[r] != 0.0)[0]
        assert len(idx) > 0, f"mel {r}: empty row?!"
        assert idx[-1] - idx[0] + 1 == len(idx), \
            f"mel {r}: not a single run -- encoding assumption broken!"
        runs.append((int(idx[0]), len(idx)))
    return runs


def flit(v):
    return repr(float(np.float32(v))) + "f"


def emit(mat, runs):
    nnz = sum(n for _, n in runs)
    starts = [s for s, _ in runs]
    lens = [n for _, n in runs]
    assert max(starts) < 256 and max(lens) < 256, "uint8 overflow"
    offs, acc, vals = [], 0, []
    for r, (s, n) in enumerate(runs):
        offs.append(acc)
        vals.extend(float(np.float32(v)) for v in mat[r, s:s + n])
        acc += n
    assert max(offs) < 256, "uint8 offset overflow"
    assert len(vals) == nnz

    def block(name, ctype, items, per_line):
        lines = [f"static const {ctype} {name}[{len(items)}] = {{"]
        for i in range(0, len(items), per_line):
            lines.append("    " + ", ".join(items[i:i + per_line]) + ",")
        lines.append("};")
        return "\n".join(lines)

    head = """/*
 * Mel filterbank matrix, SPARSE run-encoded, EXPORTED DIRECTLY from
 * librosa.filters.mel(sr=16000, n_fft=512, n_mels=40, fmin=20, fmax=4000,
 * htk=True, norm='slaney') -- the exact call mfcc.py/logmel.py use.
 *
 * Each triangular filter is one contiguous run of nonzero bins (verified by
 * scripts/export_mel_filterbank.py, which also verifies these bytes against
 * librosa on every regen). Flash cost: 247x4 + 40x3 = 1,108 B, vs 41,120 B
 * for the old dense table (97.3%% smaller); per-frame mult-adds 10,280 -> 247.
 * DO NOT HAND-EDIT. Regenerate with: python scripts/export_mel_filterbank.py
 *
 * Shape: 40 mels x 257 bins (logical). SIH Problem Statement 26172
 */
#ifndef MEL_FILTERBANK_H_
#define MEL_FILTERBANK_H_

#include <cstdint>

static constexpr int MEL_FILTERBANK_N_MELS = 40;
static constexpr int MEL_FILTERBANK_N_BINS = 257;
static constexpr int MEL_FILTERBANK_NNZ = %d;
""" % nnz
    body = "\n".join([
        block("MEL_SPARSE_START", "uint8_t", [str(s) for s in starts], 20),
        block("MEL_SPARSE_LEN", "uint8_t", [str(n) for n in lens], 20),
        block("MEL_SPARSE_OFF", "uint8_t", [str(o) for o in offs], 20),
        block("MEL_SPARSE_VAL", "float", [flit(v) for v in vals], 4),
    ])
    return head + "\n" + body + "\n\n#endif // MEL_FILTERBANK_H_\n"


def main():
    check_only = "--check" in sys.argv
    truth = librosa_truth()
    assert truth.shape == (40, 257), truth.shape
    old_text = open(HEADER, encoding="utf-8").read()
    dense = parse_dense_header(old_text)
    if dense is not None:
        # Drift guard: committed dense bytes must equal librosa (%.8e rounding).
        d = float(np.max(np.abs(dense - truth)))
        print(f"[CHECK] dense header vs librosa: max abs diff {d:.3e}")
        assert d < 1e-9, f"dense header drifted from librosa! ({d:.3e})"
        ref = dense
    else:
        # Header already sparse (re-regen): verify it instead.
        n, starts, lens, offs, vals = parse_sparse_header(old_text)
        rec = np.zeros((40, 257))
        for r in range(40):
            s, ln, o = int(starts[r]), int(lens[r]), int(offs[r])
            rec[r, s:s + ln] = vals[o:o + ln]
        d = float(np.max(np.abs(rec - truth)))
        print(f"[CHECK] sparse header vs librosa: max abs diff {d:.3e}")
        assert d < 1e-9, f"sparse header drifted from librosa! ({d:.3e})"
        ref = rec
    runs = runs_of(ref)
    nnz = sum(n for _, n in runs)
    print(f"[INFO] runs: 40 single contiguous, nnz={nnz}, "
          f"flash {nnz * 4 + 40 * 3} B (dense was 41120 B)")
    new_text = emit(truth, runs_of(truth))
    # Prove what we emit: re-parse and compare bit-for-bit (float32).
    n2, s2, l2, o2, v2 = parse_sparse_header(new_text)
    rec2 = np.zeros((40, 257), dtype=np.float32)
    for r in range(40):
        s, ln, o = int(s2[r]), int(l2[r]), int(o2[r])
        rec2[r, s:s + ln] = np.float32(v2[o:o + ln])
    d2 = float(np.max(np.abs(rec2.astype(np.float64) - truth)))
    print(f"[CHECK] emitted sparse vs librosa: max abs diff {d2:.3e}")
    assert d2 < 1e-9
    if check_only:
        print("[OK] header matches librosa (nothing written).")
        return 0
    open(HEADER, "w", encoding="utf-8").write(new_text)
    print(f"[WRITE] {HEADER} ({len(new_text)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
