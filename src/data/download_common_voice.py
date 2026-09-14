"""
Mozilla Common Voice Acquisition and Ingestion Pipeline
SIH Problem Statement 26172

Language-aware archive extraction, metadata parsing, subset filtering,
and speaker-independent manifest generation.
"""

import os
import sys
import tarfile
import argparse
import pandas as pd
import soundfile as sf
import yaml
from tqdm import tqdm

def load_config(config_path=r"D:\SIH_Model\configs\config.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

def get_common_voice_status(data_dir=r"D:\SIH_Model\data\raw\common_voice"):
    """
    Inspects whether Mozilla Common Voice audio clips and TSV metadata are present.
    """
    if not os.path.exists(data_dir):
        return {
            "status": "NOT DOWNLOADED YET",
            "directory": data_dir,
            "exists": False,
            "clip_count": 0,
            "tsv_files": [],
            "speaker_count": 0
        }
        
    tsv_files = [f for f in os.listdir(data_dir) if f.endswith('.tsv') or f.endswith('.csv')]
    clips_dir = os.path.join(data_dir, "clips")
    clip_count = 0
    if os.path.exists(clips_dir) and os.path.isdir(clips_dir):
        clip_count = len([f for f in os.listdir(clips_dir) if f.endswith('.mp3') or f.endswith('.wav')])
        
    if clip_count == 0 and not tsv_files:
        return {
            "status": "NOT DOWNLOADED YET",
            "directory": data_dir,
            "exists": True,
            "clip_count": 0,
            "tsv_files": [],
            "speaker_count": 0
        }
        
    # Read speaker count from validated.tsv or subset_metadata.tsv
    speaker_count = 0
    meta_tsv = os.path.join(data_dir, "subset_metadata.tsv")
    if not os.path.exists(meta_tsv):
        meta_tsv = os.path.join(data_dir, "validated.tsv")
    if os.path.exists(meta_tsv):
        try:
            df = pd.read_csv(meta_tsv, sep='\t', usecols=['client_id'])
            speaker_count = df['client_id'].nunique()
        except Exception:
            pass

    return {
        "status": f"INGESTED ({clip_count} audio clips, {len(tsv_files)} metadata files, {speaker_count} speakers)",
        "directory": data_dir,
        "exists": True,
        "clip_count": clip_count,
        "tsv_files": tsv_files,
        "speaker_count": speaker_count
    }

def ingest_archive(archive_path, target_dir=r"D:\SIH_Model\data\raw\common_voice", language="en", max_samples=5000):
    """
    Extracts an official Common Voice tar.gz archive for a specific language,
    parses TSV metadata, and extracts the configured subset.
    """
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    print(f"\nVerifying archive: {archive_path} ({os.path.getsize(archive_path):,} bytes)...")
    print(f"Target Language: '{language}', Max Samples: {max_samples}")

    os.makedirs(target_dir, exist_ok=True)
    clips_target = os.path.join(target_dir, "clips")
    os.makedirs(clips_target, exist_ok=True)

    with tarfile.open(archive_path, 'r:*') as tar:
        members = tar.getmembers()
        print(f"Total archive entries: {len(members):,}")

        # Filter language-specific members
        lang_pattern = f"/{language}/"
        lang_members = [m for m in members if lang_pattern in m.name or m.name.startswith(f"{language}/")]
        if not lang_members:
            # Fallback: check if language prefix in folder names
            lang_members = [m for m in members if f"-{language}/" in m.name]
        
        print(f"Found {len(lang_members):,} entries for language '{language}'.")

        # 1. Extract TSV metadata files for target language
        tsv_members = [m for m in lang_members if m.name.endswith('.tsv')]
        print(f"Extracting {len(tsv_members)} metadata TSV files for '{language}'...")
        for m in tsv_members:
            dest_name = os.path.basename(m.name)
            tar.makefile(m, os.path.join(target_dir, dest_name))

        # 2. Parse validated.tsv to select high-quality subset
        val_tsv_path = os.path.join(target_dir, "validated.tsv")
        if not os.path.exists(val_tsv_path):
            train_tsv = os.path.join(target_dir, "train.tsv")
            val_tsv_path = train_tsv if os.path.exists(train_tsv) else None

        selected_filenames = set()
        if val_tsv_path and os.path.exists(val_tsv_path):
            df = pd.read_csv(val_tsv_path, sep='\t')
            print(f"Loaded {len(df):,} records from {os.path.basename(val_tsv_path)}.")
            
            # Filter high quality: upvotes >= downvotes
            if 'upvotes' in df.columns and 'downvotes' in df.columns:
                df = df[df['upvotes'] >= df['downvotes']]
                print(f"Validated high-quality records: {len(df):,}")

            # Select max_samples
            subset_df = df.head(max_samples)
            selected_filenames = set(subset_df['path'].tolist())
            print(f"Selected subset: {len(selected_filenames)} clips across {subset_df['client_id'].nunique()} unique speakers.")
            
            # Save subset metadata
            subset_tsv = os.path.join(target_dir, "subset_metadata.tsv")
            subset_df.to_csv(subset_tsv, sep='\t', index=False)
        else:
            print(f"Warning: No TSV metadata found. Extracting first {max_samples} audio files.")

        # 3. Extract audio clips for the selected subset
        extracted_count = 0
        clip_members = [m for m in lang_members if m.name.endswith('.mp3') or m.name.endswith('.wav')]
        
        for m in tqdm(clip_members, desc=f"Extracting '{language}' audio subset"):
            base = os.path.basename(m.name)
            if selected_filenames:
                if base in selected_filenames:
                    tar.makefile(m, os.path.join(clips_target, base))
                    extracted_count += 1
            else:
                if extracted_count < max_samples:
                    tar.makefile(m, os.path.join(clips_target, base))
                    extracted_count += 1

            if selected_filenames and extracted_count >= len(selected_filenames):
                break

    print(f"\nExtraction complete! Successfully saved {extracted_count} audio clips to {clips_target}.")
    return inspect_common_voice(target_dir)

def inspect_common_voice(data_dir=r"D:\SIH_Model\data\raw\common_voice"):
    status = get_common_voice_status(data_dir)
    print("\n" + "="*60)
    print("MOZILLA COMMON VOICE DATASET INSPECTION REPORT")
    print("="*60)
    print(f"Target Directory:    {status['directory']}")
    print(f"Status:              {status['status']}")
    print(f"Total Audio Clips:   {status['clip_count']}")
    print(f"Metadata Files:      {status['tsv_files']}")
    print(f"Unique Speakers:     {status['speaker_count']}")

    clips_dir = os.path.join(data_dir, "clips")
    if os.path.exists(clips_dir):
        clips = [f for f in os.listdir(clips_dir) if f.endswith('.mp3') or f.endswith('.wav')]
        if clips:
            sample_clip = os.path.join(clips_dir, clips[0])
            try:
                info = sf.info(sample_clip)
                print(f"\nSample Clip Profile ({clips[0]}):")
                print(f"  Sample Rate:   {info.samplerate} Hz")
                print(f"  Channels:      {info.channels}")
                print(f"  Duration:      {info.duration:.3f} s")
                print(f"  Format:        {info.format} / {info.subtype}")
            except Exception as e:
                print(f"  Sample clip inspection note: {e}")

    print("="*60)
    return status

def main():
    parser = argparse.ArgumentParser(description="Acquire, ingest, and inspect Mozilla Common Voice dataset.")
    parser.add_argument("--check-only", action="store_true", help="Only check status")
    parser.add_argument("--archive-path", type=str, default=None, help="Path to downloaded official Common Voice tar.gz archive")
    parser.add_argument("--data-dir", default=r"D:\SIH_Model\data\raw\common_voice", help="Target raw data directory")
    parser.add_argument("--language", default=None, help="Target language code (e.g., 'en')")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of audio samples to extract")
    args = parser.parse_args()

    config = load_config()
    cv_cfg = config.get("common_voice", {})
    lang = args.language or cv_cfg.get("language", "en")
    max_samples = args.max_samples or cv_cfg.get("max_samples", 5000)

    if args.archive_path:
        ingest_archive(args.archive_path, target_dir=args.data_dir, language=lang, max_samples=max_samples)
        return 0

    status = get_common_voice_status(args.data_dir)
    if status["clip_count"] > 0:
        inspect_common_voice(args.data_dir)
        return 0

    inspect_common_voice(args.data_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())
