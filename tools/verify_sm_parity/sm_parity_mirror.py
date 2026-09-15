#!/usr/bin/env python3
"""Python mirror of tools/verify_sm_parity/sm_parity.cpp.

Replays the IDENTICAL frame scripts through src/streaming/state_machine.py,
prints the identical trace format, and -- when given the C++ binary's stdout
as argv[1] -- diffs the two with tolerance (float32-vs-float64 needs it):

  g++ -std=c++17 -Wall -Wextra -Isrc/deployment/esp32 \\
      tools/verify_sm_parity/sm_parity.cpp -o /tmp/sm_parity
  /tmp/sm_parity > /tmp/fw.trace
  python tools/verify_sm_parity/sm_parity_mirror.py /tmp/fw.trace  # exit 0 == parity

KNOWN folded difference: fw prints st=2 (ACTIVATED) on trigger frames, Python
prints st=3 (COOLDOWN); the comparator folds fw 2->3 when trg=1. tau compared
with abs tol 2e-4; everything else must match exactly.
"""
import os
import re
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from src.streaming.state_machine import DetectionStateMachine  # noqa: E402

ST = {"LISTENING": 0, "VERIFYING": 1, "ACTIVATED": 2, "COOLDOWN": 3}
TAU_TOL = 2e-4


def trace(lines, tag, t, sm, trg):
    lines.append("%s t=%u tau=%.4f st=%d trg=%d"
                 % (tag, t, sm.threshold_eff, ST[sm.state], 1 if trg else 0))


def run():
    lines = []
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    # A: fixed-tau trigger + clean re-arm
    sm = DetectionStateMachine(adaptive=False)
    trigs = []
    t = 0.0
    for _ in range(6):
        r = sm.process_similarity(0.95, t)
        trace(lines, "A1", int(t), sm, r["is_activated"])
        if r["is_activated"]:
            trigs.append(int(t))
        t += 50.0
    for _ in range(30):
        r = sm.process_similarity(0.10, t)
        if int(t) in (600, 1600, 1650, 1700):
            trace(lines, "A2", int(t), sm, r["is_activated"])
        if r["is_activated"]:
            trigs.append(int(t))
        t += 50.0
    for _ in range(12):
        r = sm.process_similarity(0.95, t)
        trace(lines, "A3", int(t), sm, r["is_activated"])
        if r["is_activated"]:
            trigs.append(int(t))
        t += 50.0
    check(trigs == [150, 2300], f"A: triggers {trigs} != [150, 2300]")

    # B: adaptive rise / warmup (600 frames: the EMA transient pins at the cap
    # for ~200 frames before settling; see the NOTE in sm_parity.cpp).
    sm = DetectionStateMachine()
    any_trg = False
    taus = {}
    for i in range(600):
        s = 0.87 if i % 2 == 0 else 0.79
        r = sm.process_similarity(s, i * 50.0)
        any_trg = any_trg or r["is_activated"]
        if i in (0, 39, 40, 119, 599):
            taus[i] = sm.threshold_eff
    lines.append("B tau0=%.4f tau39=%.4f tau40=%.4f tau119=%.4f tau599=%.4f"
                 % (taus[0], taus[39], taus[40], taus[119], taus[599]))
    check(taus[0] == 0.89, "B: frame 0 at calibrated tau")
    check(taus[39] == 0.89, "B: warmup frame 39 still fixed tau")
    check(taus[40] > 0.89, "B: tau rises right after warmup")
    check(taus[119] > 0.89, "B: tau stays raised during transient")
    check(0.90 < taus[599] < 0.93, "B: tau settles ~0.91")
    check(not any_trg, "B: confusing bg never triggers")

    # C: unobserved frames do not advance warmup
    sm = DetectionStateMachine()
    t = 0.0
    for _ in range(200):
        sm.process_similarity(0.0, t, observed=False)
        t += 50.0
    for i in range(10):
        s = 0.87 if i % 2 == 0 else 0.79
        sm.process_similarity(s, 10000.0 + i * 50.0, observed=True)
    lines.append("C tau_after_10obs=%.4f" % sm.threshold_eff)
    check(sm.threshold_eff == 0.89, "C: still in warmup after 10 observed")
    for i in range(40):
        s = 0.87 if i % 2 == 0 else 0.79
        sm.process_similarity(s, 11000.0 + i * 50.0, observed=True)
    lines.append("C tau_after_50obs=%.4f" % sm.threshold_eff)
    check(sm.threshold_eff > 0.89, "C: 50 observed frames leave warmup")

    # D: gap + backwards-clock reset
    sm = DetectionStateMachine()
    sm.process_similarity(0.95, 0.0)
    check(sm.state == "VERIFYING", "D: hot start VERIFYING")
    r = sm.process_similarity(0.50, 2000.0)
    trace(lines, "D1", 2000, sm, r["is_activated"])
    check(sm.state == "LISTENING", "D: gap -> LISTENING")
    r = sm.process_similarity(0.10, 500.0)
    trace(lines, "D2", 500, sm, r["is_activated"])
    check(sm.state == "LISTENING", "D: backwards -> LISTENING")

    # E: cooldown-exit frame decides on fresh evidence
    sm = DetectionStateMachine(threshold=0.89, hysteresis=0.05,
                               consecutive_windows=2, smoothing_window=4,
                               cooldown_ms=200)
    sm.process_similarity(0.95, 0.0)
    r = sm.process_similarity(0.95, 50.0)
    check(r["is_activated"], "E: trigger @50ms")
    sm.process_similarity(0.95, 100.0)
    r = sm.process_similarity(0.85, 300.0)
    trace(lines, "E1", 300, sm, r["is_activated"])
    check(sm.state == "LISTENING", "E: exit decides fresh")

    # F: Change 3d -- persistence completes ON an unobserved frame
    sm = DetectionStateMachine(smoothing_window=8, consecutive_windows=8)
    for i in range(7):
        r = sm.process_similarity(0.99, i * 50.0, observed=True)
        check(not r["is_activated"], "F: no early trigger")
    check(sm.state == "VERIFYING", "F: staged VERIFYING")
    r = sm.process_similarity(0.0, 350.0, observed=False)
    trace(lines, "F1", 350, sm, r["is_activated"])
    check(r["is_activated"], "F: unobserved frame completes persistence")


    return lines, fails


def fold_fw(line):
    # fw transient ACTIVATED(2) on trigger frames == Python COOLDOWN(3)
    return re.sub(r"st=2 trg=1", "st=3 trg=1", line)


def compare(py_lines, fw_text):
    fw_lines = [fold_fw(line) for line in fw_text.splitlines()
                if re.match(r"^[ABCDEF][123]?\s", line)]
    if len(py_lines) != len(fw_lines):
        return [f"line count: py={len(py_lines)} fw={len(fw_lines)}"]
    diffs = []
    for i, (p, f) in enumerate(zip(py_lines, fw_lines)):
        pt, ft = p.split(), f.split()
        if [x for x in pt if not x.startswith("tau=")] != \
                [x for x in ft if not x.startswith("tau=")]:
            diffs.append(f"line {i}: {p} != {f}")
            continue
        for x, y in zip(pt, ft):
            if x.startswith("tau=") and y.startswith("tau="):
                if abs(float(x[4:]) - float(y[4:])) > TAU_TOL:
                    diffs.append(f"line {i} tau: {p} != {f}")
    return diffs


def main():
    py_lines, fails = run()
    for line in py_lines:
        print(line)
    for f in fails:
        print(f"ASSERT-FAIL: {f}")
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as fh:
            fw_text = fh.read()
        if "SM_PARITY_CPP: PASS" not in fw_text:
            print("SM_PARITY_PY: FAIL (fw binary did not PASS)")
            return 1
        diffs = compare(py_lines, fw_text)
        for d in diffs:
            print(f"PARITY-DIFF: {d}")
        if fails or diffs:
            print("SM_PARITY_PY: FAIL")
            return 1
        print(f"SM_PARITY_PY: PASS ({len(py_lines)} trace lines match fw)")
        return 0
    print("SM_PARITY_PY: FAIL" if fails else "SM_PARITY_PY: PASS (no fw trace given)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
