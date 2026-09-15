"""
Metric Learning Batch Generator for Few-Shot Edge Voice Activator
SIH Problem Statement 26172 - Milestone 8

Generates balanced, speaker-diverse batches for metric learning.
Supports:
1. Supervised Contrastive Loss (N-way K-shot balanced batches)
2. Pairwise Contrastive Loss (Positive pairs from distinct speakers, hard negative pairs)
"""

import os
import sys
import numpy as np
import pandas as pd
import soundfile as sf

# Repo root from this file location (portable; was a hardcoded Windows path).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)
from src.features.mfcc import MFCCFeatureExtractor
from src.data.augment import AudioAugmenter

class MetricLearningBatchGenerator:
    def __init__(self, manifest_path, feature_extractor=None, augmenter=None, max_samples_per_class=None, seed=42):
        self.manifest_path = manifest_path
        self.rng = np.random.default_rng(seed)
        self.feature_extractor = feature_extractor or MFCCFeatureExtractor(sample_rate=16000, n_mfcc=13, n_mels=40)
        self.augmenter = augmenter

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        self.df = pd.read_csv(manifest_path)
        if "duration_seconds" in self.df.columns:
            self.df = self.df[self.df["duration_seconds"] > 0.1].reset_index(drop=True)

        self.classes = sorted(self.df["label"].unique().tolist())
        self.class_to_id = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.id_to_class = {idx: cls_name for cls_name, idx in self.class_to_id.items()}

        # Group data by class and speaker
        self.class_samples = {}
        # Fix: a class with zero usable records after the duration_seconds > 0.1 filter
        # previously crashed deep inside sample_supcon_batch() (rng.integers(0, len(...))
        # on an empty list) with an opaque stack trace during training, not at load time.
        # Also flag single-recording classes: sample_supcon_batch()'s fallback will
        # replicate that one recording verbatim (when augment=False, the feature cache
        # returns byte-identical copies), giving low-diversity "positive" pairs -
        # harmless to SupCon's math, but a real signal that more custom-keyword
        # recordings should be collected.
        for cls_name in self.classes:
            sub = self.df[self.df["label"] == cls_name]
            if max_samples_per_class:
                sub = sub.head(max_samples_per_class)
            self.class_samples[cls_name] = sub.to_dict("records")

        empty_classes = [c for c, recs in self.class_samples.items() if len(recs) == 0]
        if empty_classes:
            raise ValueError(
                f"Manifest '{manifest_path}' has {len(empty_classes)} class(es) with zero "
                f"usable records after filtering: {empty_classes}. Fix the manifest or "
                f"remove these classes before training."
            )
        singleton_classes = [c for c, recs in self.class_samples.items() if len(recs) == 1]
        if singleton_classes:
            print(
                f"[WARN] {len(singleton_classes)} class(es) have only 1 recording: "
                f"{singleton_classes}. Batches will replicate that single recording to "
                f"fill samples_per_class - consider collecting more samples for these."
            )
        self.classes = [c for c in self.classes if c not in empty_classes]

        # In-memory feature cache for fast training on CPU
        self._cache = {}

    def _load_and_extract(self, sample_record, augment=False):
        fpath = sample_record["filepath"]
        cache_key = (fpath, False)

        if not augment and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            audio, sr = sf.read(fpath, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # Fix: was channel-0-only downmix, silently dropped other channels
            if len(audio) < 16000:
                audio = np.pad(audio, (0, 16000 - len(audio)), mode="constant")
            elif len(audio) > 16000:
                audio = audio[:16000]

            if augment and self.augmenter is not None:
                audio = self.augmenter.augment(audio)

            feat = self.feature_extractor.extract(audio)  # Shape: (98, 13)

            if not augment:
                self._cache[cache_key] = feat
            return feat
        except Exception:
            feat = np.zeros((98, 13), dtype=np.float32)
            return feat

    def preload_cache(self, samples_per_class=100, verbose=True):
        """Preloads features into memory to eliminate disk I/O during training."""
        if verbose:
            print(f"Preloading up to {samples_per_class} features per class into RAM cache...")
        count = 0
        for cls_name in self.classes:
            records = self.class_samples[cls_name][:samples_per_class]
            for r in records:
                self._load_and_extract(r, augment=False)
                count += 1
        if verbose:
            mb_size = (count * 98 * 13 * 4) / (1024 * 1024)
            print(f"Preloaded {count} audio features into memory ({mb_size:.2f} MB).")

    def sample_supcon_batch(self, num_classes=16, samples_per_class=4, augment=False):
        """
        Generates an N-way K-shot balanced batch for Supervised Contrastive Loss.
        Total batch size = num_classes * samples_per_class.
        Samples from different speakers per class whenever possible.
        """
        num_c = min(num_classes, len(self.classes))
        selected_classes = self.rng.choice(self.classes, size=num_c, replace=False)
        features_list = []
        labels_list = []

        for cls_name in selected_classes:
            records = self.class_samples[cls_name]
            cls_id = self.class_to_id[cls_name]

            # Group by speaker
            speakers = {}
            for r in records:
                spk = r.get("speaker_id", "unknown")
                speakers.setdefault(spk, []).append(r)

            chosen_records = []
            spk_keys = list(speakers.keys())
            self.rng.shuffle(spk_keys)

            for spk in spk_keys:
                r_idx = self.rng.integers(0, len(speakers[spk]))
                chosen_records.append(speakers[spk][r_idx])
                if len(chosen_records) == samples_per_class:
                    break

            # Fallback if not enough distinct speakers
            if len(chosen_records) < samples_per_class:
                needed = samples_per_class - len(chosen_records)
                fallback_idx = self.rng.choice(len(records), size=needed, replace=True)
                for idx in fallback_idx:
                    chosen_records.append(records[idx])

            for r in chosen_records:
                feat = self._load_and_extract(r, augment=augment)
                features_list.append(feat)
                labels_list.append(cls_id)

        X = np.array(features_list, dtype=np.float32)[:, :, :, np.newaxis]
        y = np.array(labels_list, dtype=np.int32)
        return X, y

    def sample_pairwise_batch(self, batch_size=64, augment=False):
        """
        Generates positive and negative pairs for standard Contrastive Loss.
        Half batch are positive pairs (same word, distinct speakers).
        Half batch are negative pairs (different words).
        """
        half_batch = batch_size // 2
        feats_a = []
        feats_b = []
        pair_labels = []

        # Positive pairs (y=1)
        for _ in range(half_batch):
            cls_name = self.rng.choice(self.classes)
            records = self.class_samples[cls_name]
            if len(records) >= 2:
                idx1, idx2 = self.rng.choice(len(records), size=2, replace=False)
            else:
                idx1 = idx2 = 0
            fa = self._load_and_extract(records[idx1], augment=augment)
            fb = self._load_and_extract(records[idx2], augment=augment)
            feats_a.append(fa)
            feats_b.append(fb)
            pair_labels.append(1.0)

        # Negative pairs (y=0)
        for _ in range(half_batch):
            cls1, cls2 = self.rng.choice(self.classes, size=2, replace=False)
            r1 = self.rng.choice(self.class_samples[cls1])
            r2 = self.rng.choice(self.class_samples[cls2])
            fa = self._load_and_extract(r1, augment=augment)
            fb = self._load_and_extract(r2, augment=augment)
            feats_a.append(fa)
            feats_b.append(fb)
            pair_labels.append(0.0)

        Xa = np.array(feats_a, dtype=np.float32)[:, :, :, np.newaxis]
        Xb = np.array(feats_b, dtype=np.float32)[:, :, :, np.newaxis]
        y = np.array(pair_labels, dtype=np.float32)
        return Xa, Xb, y
