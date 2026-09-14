"""
Orchestration Script: Check and Download All Datasets
SIH Problem Statement 26172
"""

import os
import sys
import platform
import argparse

# Add src to Python path
sys.path.insert(0, r"D:\SIH_Model")

from src.data.download_speech_commands import get_dataset_status as get_sc_status
from src.data.download_common_voice import get_common_voice_status as get_cv_status

def check_python_environment():
    print("--- Python Environment Verification ---")
    print(f"Python Version:    {platform.python_version()} ({platform.architecture()[0]})")
    print(f"Executable:        {sys.executable}")
    
    # Check key packages
    packages_to_check = [
        "numpy", "scipy", "yaml", "librosa", "soundfile",
        "tensorflow", "tensorflow_datasets", "sklearn", "pandas"
    ]
    print("\nPackage Availability:")
    for pkg in packages_to_check:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "Available (no version)")
            print(f"  [x] {pkg:22}: {version}")
        except ImportError:
            print(f"  [ ] {pkg:22}: NOT INSTALLED")

def report_dataset_summary():
    print("\n--- Dataset Status Summary ---")
    sc_status = get_sc_status()
    cv_status = get_cv_status()
    
    print("1. Speech Commands:")
    print(f"   - Status:      {sc_status['status']}")
    print(f"   - Path:        {sc_status['directory']}")
    print(f"   - Samples:     {sc_status['sample_count']}")
    
    print("2. Mozilla Common Voice:")
    print(f"   - Status:      {cv_status['status']}")
    print(f"   - Path:        {cv_status['directory']}")
    print(f"   - Clips:       {cv_status['clip_count']}")

def main():
    parser = argparse.ArgumentParser(description="Dataset Download and Verification Orchestration")
    parser.add_argument("--check-only", action="store_true", default=True, help="Check status without downloading")
    args = parser.parse_args()

    check_python_environment()
    report_dataset_summary()
    return 0

if __name__ == "__main__":
    sys.exit(main())
