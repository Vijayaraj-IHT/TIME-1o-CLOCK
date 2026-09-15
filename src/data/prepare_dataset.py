"""
Unified Dataset Preparation and Manifest Generation
SIH Problem Statement 26172
"""

import os
import sys
import pandas as pd

# Repo root from this file location (portable; was a hardcoded Windows path).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)
from src.data.metadata import validate_metadata_integrity, save_manifests

def combine_and_generate_manifests(sc_df, cv_df, output_dir=os.path.join(_REPO_ROOT, "data", "metadata")):
    print("\n" + "="*60)
    print("COMBINING DATASETS AND GENERATING FINAL MANIFESTS")
    print("="*60)

    combined_df = pd.concat([sc_df, cv_df], ignore_index=True)
    print(f"Total Combined Samples: {len(combined_df):,}")

    stats = validate_metadata_integrity(combined_df)
    save_manifests(combined_df, output_dir)

    print(f"\nManifest Generation Summary:")
    print(f"  Total Audio Files:    {stats['total_records']:,}")
    print(f"  Total Unique Speakers: {stats['total_speakers']:,}")
    print(f"  Train Manifest:       {stats['train_samples']:,} samples ({stats['train_speakers']:,} speakers)")
    print(f"  Validation Manifest:  {stats['val_samples']:,} samples ({stats['val_speakers']:,} speakers)")
    print(f"  Test Manifest:        {stats['test_samples']:,} samples ({stats['test_speakers']:,} speakers)")
    print(f"  Zero-Leakage Status:  {'VERIFIED (0 Leakage)' if not stats['leakage_detected'] else 'WARNING LEAKAGE'}")
    print("="*60)
    return combined_df
