"""
Mozilla Common Voice Preprocessing & Metadata Generation
SIH Problem Statement 26172
"""

import os
import sys
import pandas as pd
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, r"D:\SIH_Model")
from src.data.metadata import create_metadata_dataframe, validate_metadata_integrity
from src.data.split_dataset import partition_speakers_deterministic

def prepare_common_voice(raw_dir=r"D:\SIH_Model\data\raw\common_voice",
                         output_csv=r"D:\SIH_Model\data\metadata\common_voice_metadata.csv"):
    print("\n" + "="*60)
    print("PREPARING MOZILLA COMMON VOICE METADATA")
    print("="*60)

    clips_dir = os.path.join(raw_dir, "clips")
    meta_tsv = os.path.join(raw_dir, "subset_metadata.tsv")
    if not os.path.exists(meta_tsv):
        meta_tsv = os.path.join(raw_dir, "validated.tsv")

    if not os.path.exists(meta_tsv) or not os.path.exists(clips_dir):
        print("Common Voice metadata or clips directory missing.")
        return create_metadata_dataframe()

    meta_df = pd.read_csv(meta_tsv, sep='\t')
    available_clips = set(os.listdir(clips_dir))

    # Filter records where audio file actually exists
    meta_df = meta_df[meta_df['path'].isin(available_clips)]
    print(f"Matched {len(meta_df):,} audio clips with metadata.")

    # Partition speakers deterministically (80% train, 10% val, 10% test)
    speaker_ids = meta_df['client_id'].unique()
    train_spk, val_spk, test_spk = partition_speakers_deterministic(speaker_ids, train_ratio=0.80, val_ratio=0.10, seed=42)

    records = []
    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Processing Common Voice metadata"):
        path_clip = os.path.join(clips_dir, row['path']).replace("\\", "/")
        spk = str(row['client_id'])

        if spk in train_spk:
            split = "train"
        elif spk in val_spk:
            split = "validation"
        else:
            split = "test"

        records.append({
            "dataset": "mozilla_common_voice_7.0",
            "source": "mozilla_common_voice",
            "filepath": path_clip,
            "speaker_id": spk,
            "clip_id": f"cv_{row['path']}",
            "label": str(row.get('sentence', 'speech')).strip(),
            "language": "en",
            "duration_seconds": 1.0,  # Standardized input window
            "sample_rate_original": 48000,
            "sample_rate_processed": 16000,
            "channels": 1,
            "split": split,
            "is_positive": False,  # General conversational/diverse speech acts as negative / embedding contrast
            "is_negative": True,
            "augmentation_type": "none",
            "quality_status": "clean"
        })

    df = create_metadata_dataframe(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    stats = validate_metadata_integrity(df)
    print(f"\nCommon Voice Metadata Prepared: {len(df):,} records saved to {output_csv}")
    print(f"  Train samples:      {stats['train_samples']:,} ({stats['train_speakers']:,} speakers)")
    print(f"  Validation samples: {stats['val_samples']:,} ({stats['val_speakers']:,} speakers)")
    print(f"  Test samples:       {stats['test_samples']:,} ({stats['test_speakers']:,} speakers)")
    print(f"  Speaker Leakage:    {'NONE (Clean)' if not stats['leakage_detected'] else 'DETECTED'}")
    print("="*60)
    return df

if __name__ == "__main__":
    prepare_common_voice()
