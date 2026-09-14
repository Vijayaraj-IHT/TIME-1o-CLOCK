"""
Speaker-Independent Dataset Splitting Logic
SIH Problem Statement 26172
"""

import os
import random
import numpy as np

def assign_speech_commands_split(filepath, val_set, test_set):
    """
    Assigns split based on official Speech Commands validation_list and testing_list.
    """
    norm_path = filepath.replace("\\", "/")
    # Match against trailing category/filename.wav
    parts = norm_path.split("/")
    if len(parts) >= 2:
        rel = f"{parts[-2]}/{parts[-1]}"
        if rel in val_set:
            return "validation"
        if rel in test_set:
            return "test"
    return "train"

def partition_speakers_deterministic(speaker_ids, train_ratio=0.80, val_ratio=0.10, seed=42):
    """
    Partitions a list of unique speaker IDs into train, validation, and test sets.
    Ensures zero speaker leakage across splits.
    """
    rng = random.Random(seed)
    unique_speakers = sorted(list(set(speaker_ids)))
    rng.shuffle(unique_speakers)

    n_total = len(unique_speakers)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_speakers = set(unique_speakers[:n_train])
    val_speakers = set(unique_speakers[n_train:n_train + n_val])
    test_speakers = set(unique_speakers[n_train + n_val:])

    return train_speakers, val_speakers, test_speakers
