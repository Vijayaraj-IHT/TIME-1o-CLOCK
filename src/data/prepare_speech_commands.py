"""
Speech Commands Preprocessing & Metadata Generation
SIH Problem Statement 26172
"""

import os
import sys
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, r"D:\SIH_Model")
from src.data.metadata import create_metadata_dataframe, validate_metadata_integrity
from src.data.split_dataset import assign_speech_commands_split

def prepare_speech_commands(raw_dir=r"D:\SIH_Model\data\raw\speech_commands",
                            output_csv=r"D:\SIH_Model\data\metadata\speech_commands_metadata.csv"):
    print("\n" + "="*60)
    print("PREPARING SPEECH COMMANDS DATASET METADATA")
    print("="*60)
    
    val_file = os.path.join(raw_dir, "validation_list.txt")
    test_file = os.path.join(raw_dir, "testing_list.txt")

    val_set = set([line.strip().replace("\\", "/") for line in open(val_file).readlines()]) if os.path.exists(val_file) else set()
    test_set = set([line.strip().replace("\\", "/") for line in open(test_file).readlines()]) if os.path.exists(test_file) else set()

    categories = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d)) and not d.startswith('.')]
    records = []

    for cat in tqdm(categories, desc="Scanning Speech Commands categories"):
        cat_dir = os.path.join(raw_dir, cat)
        wav_files = [f for f in os.listdir(cat_dir) if f.lower().endswith('.wav')]

        is_noise = (cat == "_background_noise_")

        for f in wav_files:
            file_path = os.path.join(cat_dir, f)
            rel_path = f"{cat}/{f}".replace("\\", "/")

            if is_noise:
                speaker_id = "background_env"
                clip_id = f"noise_{f}"
                split = "train"  # Background noise is used for data augmentation
                is_pos = False
                is_neg = True
            else:
                parts = f.split("_nohash_")
                speaker_id = parts[0] if len(parts) == 2 else "unknown"
                clip_id = f"{cat}_{f}"
                split = assign_speech_commands_split(file_path, val_set, test_set)
                is_pos = True
                is_neg = False

            records.append({
                "dataset": "speech_commands_v0.02",
                "source": "tensorflow_speech_commands",
                "filepath": file_path.replace("\\", "/"),
                "speaker_id": speaker_id,
                "clip_id": clip_id,
                "label": cat,
                "language": "en",
                "duration_seconds": 1.0,  # Standard Speech Commands clip duration
                "sample_rate_original": 16000,
                "sample_rate_processed": 16000,
                "channels": 1,
                "split": split,
                "is_positive": is_pos,
                "is_negative": is_neg,
                "augmentation_type": "none",
                "quality_status": "clean"
            })

    df = create_metadata_dataframe(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    stats = validate_metadata_integrity(df)
    print(f"\nSpeech Commands Metadata Prepared: {len(df):,} records saved to {output_csv}")
    print(f"  Train samples:      {stats['train_samples']:,} ({stats['train_speakers']:,} speakers)")
    print(f"  Validation samples: {stats['val_samples']:,} ({stats['val_speakers']:,} speakers)")
    print(f"  Test samples:       {stats['test_samples']:,} ({stats['test_speakers']:,} speakers)")
    print(f"  Speaker Leakage:    {'NONE (Clean)' if not stats['leakage_detected'] else 'DETECTED'}")
    print("="*60)
    return df

if __name__ == "__main__":
    prepare_speech_commands()
