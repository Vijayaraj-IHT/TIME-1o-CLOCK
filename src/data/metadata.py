"""
Metadata Management Schema and Manifest Generation
SIH Problem Statement 26172
"""

import os
import pandas as pd

METADATA_COLUMNS = [
    "dataset",
    "source",
    "filepath",
    "speaker_id",
    "clip_id",
    "label",
    "language",
    "duration_seconds",
    "sample_rate_original",
    "sample_rate_processed",
    "channels",
    "split",
    "is_positive",
    "is_negative",
    "augmentation_type",
    "quality_status"
]

def create_metadata_dataframe(records=None):
    """
    Creates a pandas DataFrame initialized with the standard project schema.
    """
    if records:
        df = pd.DataFrame(records)
        for col in METADATA_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[METADATA_COLUMNS]
    return pd.DataFrame(columns=METADATA_COLUMNS)

def validate_metadata_integrity(df):
    """
    Validates that metadata complies with zero speaker leakage rules and required fields.
    """
    missing_cols = [c for c in METADATA_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Metadata missing required columns: {missing_cols}")

    # Check speaker leakage between train and test
    train_speakers = set(df[df['split'] == 'train']['speaker_id'].dropna().unique())
    val_speakers = set(df[df['split'] == 'validation']['speaker_id'].dropna().unique())
    test_speakers = set(df[df['split'] == 'test']['speaker_id'].dropna().unique())

    leakage_train_val = train_speakers.intersection(val_speakers)
    leakage_train_test = train_speakers.intersection(test_speakers)
    leakage_val_test = val_speakers.intersection(test_speakers)

    return {
        "total_records": len(df),
        "total_speakers": df['speaker_id'].nunique(),
        "train_samples": len(df[df['split'] == 'train']),
        "val_samples": len(df[df['split'] == 'validation']),
        "test_samples": len(df[df['split'] == 'test']),
        "train_speakers": len(train_speakers),
        "val_speakers": len(val_speakers),
        "test_speakers": len(test_speakers),
        "leakage_train_val": len(leakage_train_val),
        "leakage_train_test": len(leakage_train_test),
        "leakage_val_test": len(leakage_val_test),
        "leakage_detected": bool(leakage_train_val or leakage_train_test or leakage_val_test)
    }

def save_manifests(combined_df, metadata_dir=r"D:\SIH_Model\data\metadata"):
    """
    Saves combined metadata and individual split manifests.
    """
    os.makedirs(metadata_dir, exist_ok=True)
    combined_path = os.path.join(metadata_dir, "combined_metadata.csv")
    combined_df.to_csv(combined_path, index=False)

    for split in ["train", "validation", "test"]:
        split_df = combined_df[combined_df['split'] == split]
        manifest_path = os.path.join(metadata_dir, f"{split}_manifest.csv")
        split_df.to_csv(manifest_path, index=False)
        print(f"Saved {split} manifest: {len(split_df):,} records -> {manifest_path}")

    return combined_path
