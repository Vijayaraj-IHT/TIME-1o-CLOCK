#!/usr/bin/env python3
"""
Step-0 firmware dedup guard.

src/deployment/esp32/       = MASTER (ESP-IDF / ESP32-S3 entry: main.cpp)
src/deployment/esp32_wroom/ = Arduino flavor (.ino + platformio.ini)

The five component headers are shared by design; historically they were
copy-pasted, which caused silent drift (TRACK_A_VERIFICATION.md: three copies
of the same broken FFT logic). Arduino IDE requires headers inside the sketch
folder, so a common-include folder is NOT an option for the .ino flavor.

Policy: `esp32/` is the single source of truth for shared headers.
  - `python scripts/sync_firmware_headers.py --check`  -> exit 1 on drift (for tests/CI)
  - `python scripts/sync_firmware_headers.py --sync`   -> copy master -> wroom
"""
import hashlib
import os
import shutil
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MASTER = os.path.join(_REPO_ROOT, "src", "deployment", "esp32")
WROOM = os.path.join(_REPO_ROOT, "src", "deployment", "esp32_wroom")

# Shared by design. Entry points (main.cpp / .ino / platformio.ini / README) are per-flavor.
SHARED_HEADERS = [
    "activator_state_machine.h",
    "audio_ring_buffer.h",
    "dtw_spotter.h",
    "dtw_template_zora.h",
    "feature_extractor.h",
    "keyword_prototype.h",
    "mel_filterbank.h",
    "tflite_micro_model.h",
]


def _md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def drift():
    problems = []
    for h in SHARED_HEADERS:
        a, b = os.path.join(MASTER, h), os.path.join(WROOM, h)
        if not os.path.exists(a):
            problems.append(f"MASTER missing: {h}")
        elif not os.path.exists(b):
            problems.append(f"WROOM missing: {h}")
        elif _md5(a) != _md5(b):
            problems.append(f"DRIFT: {h}")
    return problems


def main(argv):
    if "--sync" in argv:
        for h in SHARED_HEADERS:
            shutil.copy2(os.path.join(MASTER, h), os.path.join(WROOM, h))
            print(f"[SYNC] {h}: esp32/ -> esp32_wroom/")
        print("[OK] firmware headers in sync.")
        return 0
    problems = drift()
    if problems:
        print("[DRIFT] firmware header copies differ:")
        for p in problems:
            print(f"  - {p}")
        print("Run with --sync to copy master (esp32/) -> esp32_wroom/.")
        return 1
    print(f"[OK] {len(SHARED_HEADERS)} shared headers identical across firmware flavors.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
