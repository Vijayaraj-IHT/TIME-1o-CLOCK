"""
Speech Commands Dataset Acquisition, Verification, and Inspection Pipeline
SIH Problem Statement 26172
"""

import os
import sys
import tarfile
import urllib.request
import soundfile as sf
from tqdm import tqdm

OFFICIAL_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

def get_dataset_status(data_dir=r"D:\SIH_Model\data\raw\speech_commands"):
    """
    Inspects whether the Speech Commands dataset is downloaded and extracted.
    """
    if not os.path.exists(data_dir):
        return {
            "status": "NOT DOWNLOADED YET",
            "directory": data_dir,
            "exists": False,
            "sample_count": 0,
            "categories": [],
            "speaker_count": 0
        }
    
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
    total_wavs = 0
    categories = []
    speakers = set()

    for item in subdirs:
        sub_path = os.path.join(data_dir, item)
        wav_files = [f for f in os.listdir(sub_path) if f.lower().endswith('.wav')]
        if wav_files:
            total_wavs += len(wav_files)
            categories.append(item)
            if item != "_background_noise_":
                for f in wav_files:
                    parts = f.split("_nohash_")
                    if len(parts) == 2:
                        speakers.add(parts[0])

    if total_wavs == 0:
        return {
            "status": "NOT DOWNLOADED YET",
            "directory": data_dir,
            "exists": True,
            "sample_count": 0,
            "categories": [],
            "speaker_count": 0
        }

    return {
        "status": f"DOWNLOADED ({total_wavs} samples, {len(categories)} categories)",
        "directory": data_dir,
        "exists": True,
        "sample_count": total_wavs,
        "categories": sorted(categories),
        "speaker_count": len(speakers)
    }

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, target_path):
    print(f"Downloading {url} to {target_path}...")
    temp_target = target_path + ".part"
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(target_path)) as t:
        urllib.request.urlretrieve(url, filename=temp_target, reporthook=t.update_to)
        
    os.replace(temp_target, target_path)
    print(f"Download completed: {target_path}")

def extract_archive(archive_path, extract_dir):
    print(f"Extracting {archive_path} to {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(archive_path, 'r:gz') as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting audio files"):
            tar.extract(member, path=extract_dir)
    print(f"Extraction completed successfully.")

def inspect_dataset(data_dir):
    status = get_dataset_status(data_dir)
    print("\n" + "="*60)
    print("SPEECH COMMANDS DATASET INSPECTION REPORT")
    print("="*60)
    print(f"Target Directory:    {status['directory']}")
    print(f"Status:              {status['status']}")
    print(f"Total Audio Samples: {status['sample_count']}")
    print(f"Total Categories:    {len(status['categories'])}")
    print(f"Unique Speakers:     {status['speaker_count']}")
    
    # Check official split files
    val_file = os.path.join(data_dir, "validation_list.txt")
    test_file = os.path.join(data_dir, "testing_list.txt")
    val_count = len(open(val_file).readlines()) if os.path.exists(val_file) else 0
    test_count = len(open(test_file).readlines()) if os.path.exists(test_file) else 0
    
    print(f"Official Validation Samples (validation_list.txt): {val_count}")
    print(f"Official Testing Samples (testing_list.txt):       {test_count}")
    print(f"Official Training Samples (Remaining):             {status['sample_count'] - val_count - test_count}")
    
    # Inspect a sample audio file properties
    sample_wav = None
    for cat in status['categories']:
        if cat != "_background_noise_":
            cat_dir = os.path.join(data_dir, cat)
            wavs = [f for f in os.listdir(cat_dir) if f.endswith('.wav')]
            if wavs:
                sample_wav = os.path.join(cat_dir, wavs[0])
                break
                
    if sample_wav:
        info = sf.info(sample_wav)
        print(f"\nSample Audio Profile ({os.path.basename(sample_wav)}):")
        print(f"  Sample Rate:   {info.samplerate} Hz")
        print(f"  Channels:      {info.channels}")
        print(f"  Duration:      {info.duration:.3f} s")
        print(f"  Format:        {info.format} / {info.subtype}")
    
    print("="*60)
    return status

def download_speech_commands(data_dir=r"D:\SIH_Model\data\raw\speech_commands", force=False):
    status = get_dataset_status(data_dir)
    if status["sample_count"] > 10000 and not force:
        print(f"Speech Commands already downloaded and verified ({status['sample_count']} samples).")
        return inspect_dataset(data_dir)
        
    archive_path = os.path.join(os.path.dirname(data_dir), "speech_commands_v0.02.tar.gz")
    
    # Check if archive exists and is complete (~2.4 GB)
    if not os.path.exists(archive_path) or os.path.getsize(archive_path) < 2400000000:
        download_file(OFFICIAL_URL, archive_path)
    else:
        print(f"Archive already present: {archive_path} ({os.path.getsize(archive_path):,} bytes)")
        
    extract_archive(archive_path, data_dir)
    return inspect_dataset(data_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Acquire and inspect Speech Commands v0.02")
    parser.add_argument("--check-only", action="store_true", help="Inspect without downloading")
    parser.add_argument("--force", action="store_true", help="Force redownload and re-extraction")
    parser.add_argument("--data-dir", default=r"D:\SIH_Model\data\raw\speech_commands", help="Target data directory")
    args = parser.parse_args()

    if args.check_only:
        inspect_dataset(args.data_dir)
    else:
        download_speech_commands(args.data_dir, force=args.force)
