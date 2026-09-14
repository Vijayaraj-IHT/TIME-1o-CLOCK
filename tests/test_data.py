"""
Unit Tests for Data Configuration, Acquisition, and Manifests
SIH Problem Statement 26172 - Milestones 1, 2, 3 & 4 Verification
"""

import os
import sys
import unittest
import pandas as pd

sys.path.insert(0, r"D:\SIH_Model")

class TestProjectStructureAndConfig(unittest.TestCase):

    def setUp(self):
        self.root = r"D:\SIH_Model"

    def test_required_directories_exist(self):
        required_dirs = [
            "configs",
            "data/raw/speech_commands",
            "data/raw/common_voice",
            "data/raw/custom_keywords",
            "data/raw/noise",
            "data/processed/train",
            "data/processed/validation",
            "data/processed/test",
            "data/metadata",
            "src/data",
            "src/features",
            "src/models",
            "src/training",
            "src/evaluation",
            "src/enrollment",
            "src/streaming",
            "src/export",
            "src/deployment/esp32",
            "models/checkpoints",
            "models/saved",
            "models/tflite",
            "experiments/baseline",
            "experiments/embedding_size",
            "experiments/feature_comparison",
            "experiments/threshold",
            "experiments/quantization",
            "experiments/unseen_keyword",
            "tests",
            "scripts"
        ]
        for rel_path in required_dirs:
            full_path = os.path.join(self.root, rel_path.replace("/", os.sep))
            self.assertTrue(os.path.isdir(full_path), f"Directory missing: {full_path}")

    def test_required_root_files_exist(self):
        required_files = [
            "configs/config.yaml",
            "requirements.txt",
            ".gitignore",
            "LICENSES.md",
            "README.md",
            "src/__init__.py",
            "src/data/__init__.py",
            "src/data/download_speech_commands.py",
            "src/data/download_common_voice.py",
            "scripts/download_all_datasets.py",
            "scripts/prepare_all_data.py"
        ]
        for rel_path in required_files:
            full_path = os.path.join(self.root, rel_path.replace("/", os.sep))
            self.assertTrue(os.path.isfile(full_path), f"File missing: {full_path}")

    def test_config_yaml_content_and_schema(self):
        config_path = os.path.join(self.root, "configs", "config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()

        required_keys = [
            "project:", "sample_rate:", "clip_duration_ms:", "features:",
            "type:", "n_mfcc:", "n_mels:", "embedding:", "dimension:",
            "training:", "batch_size:", "dataset:", "common_voice:",
            "enrollment:", "detection:", "similarity_metric:", "threshold:",
            "deployment:", "quantization:", "max_ram_kb:", "max_idle_cpu_percent:"
        ]
        for key in required_keys:
            self.assertIn(key, content, f"Config key '{key}' not found in config.yaml")

    def test_speech_commands_downloaded_and_valid(self):
        from src.data.download_speech_commands import get_dataset_status as get_sc_status
        sc_status = get_sc_status()
        self.assertTrue(sc_status["exists"], "Speech Commands directory missing")
        self.assertGreater(sc_status["sample_count"], 100000, "Speech Commands sample count below 100,000")
        self.assertEqual(len(sc_status["categories"]), 36, "Speech Commands categories count must be 36")
        self.assertGreater(sc_status["speaker_count"], 2500, "Speech Commands unique speaker count below 2,500")

    def test_common_voice_ingested_and_valid(self):
        from src.data.download_common_voice import get_common_voice_status as get_cv_status
        cv_status = get_cv_status()
        self.assertTrue(cv_status["exists"], "Common Voice raw directory missing")
        self.assertGreaterEqual(cv_status["clip_count"], 5000, "Common Voice clips count below 5,000")
        self.assertGreaterEqual(cv_status["speaker_count"], 1000, "Common Voice speaker count below 1,000")

    def test_manifests_and_zero_speaker_leakage(self):
        metadata_dir = os.path.join(self.root, "data", "metadata")
        manifest_files = [
            "speech_commands_metadata.csv",
            "common_voice_metadata.csv",
            "combined_metadata.csv",
            "train_manifest.csv",
            "validation_manifest.csv",
            "test_manifest.csv"
        ]
        for f in manifest_files:
            p = os.path.join(metadata_dir, f)
            self.assertTrue(os.path.isfile(p), f"Manifest missing: {p}")

        # Load manifests and check zero speaker leakage
        train_df = pd.read_csv(os.path.join(metadata_dir, "train_manifest.csv"))
        val_df = pd.read_csv(os.path.join(metadata_dir, "validation_manifest.csv"))
        test_df = pd.read_csv(os.path.join(metadata_dir, "test_manifest.csv"))

        train_spks = set(train_df['speaker_id'].dropna().unique())
        val_spks = set(val_df['speaker_id'].dropna().unique())
        test_spks = set(test_df['speaker_id'].dropna().unique())

        self.assertEqual(len(train_spks.intersection(val_spks)), 0, "Speaker leakage detected between train and val")
        self.assertEqual(len(train_spks.intersection(test_spks)), 0, "Speaker leakage detected between train and test")
        self.assertEqual(len(val_spks.intersection(test_spks)), 0, "Speaker leakage detected between val and test")

        total_samples = len(train_df) + len(val_df) + len(test_df)
        self.assertEqual(total_samples, 110835, "Total combined manifest sample count mismatch")

if __name__ == "__main__":
    unittest.main()
