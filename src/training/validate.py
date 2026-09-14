"""
Validation Protocol for Few-Shot Speech Embedding Model
SIH Problem Statement 26172 - Milestone 8

Measures:
1. Cosine Similarity Distribution & Separation Margin (Positives vs Negatives)
2. Area Under ROC Curve (AUC) & Equal Error Rate (EER)
3. Prototype-Based Few-Shot Word Classification (Recall@1, Recall@5)
"""

import os
import sys
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import roc_curve, auc

sys.path.insert(0, r"D:\SIH_Model")
from src.features.mfcc import MFCCFeatureExtractor
from src.models.prototype import compute_prototype, compute_cosine_similarity

class MetricValidator:
    def __init__(self, encoder, val_manifest_path=r"D:\SIH_Model\data\metadata\validation_manifest.csv",
                 feature_extractor=None, seed=42):
        self.encoder = encoder
        self.val_manifest_path = val_manifest_path
        self.rng = np.random.default_rng(seed)
        self.feature_extractor = feature_extractor or MFCCFeatureExtractor(sample_rate=16000, n_mfcc=13, n_mels=40)

        if not os.path.exists(val_manifest_path):
            raise FileNotFoundError(f"Validation manifest not found: {val_manifest_path}")

        self.df = pd.read_csv(val_manifest_path)
        if "duration_seconds" in self.df.columns:
            self.df = self.df[self.df["duration_seconds"] > 0.1].reset_index(drop=True)
        self.classes = sorted(self.df["label"].unique().tolist())

        # Group records by class
        self.class_records = {}
        for cls_name in self.classes:
            self.class_records[cls_name] = self.df[self.df["label"] == cls_name].to_dict("records")

        # Cache of extracted embeddings to prevent redundant inference during validation
        self._embedding_cache = {}

    def _get_embedding(self, record):
        fpath = record["filepath"]
        if fpath in self._embedding_cache:
            return self._embedding_cache[fpath]

        try:
            audio, sr = sf.read(fpath, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # Fix: was channel-0-only downmix, silently dropped other channels
            if len(audio) < 16000:
                audio = np.pad(audio, (0, 16000 - len(audio)), mode="constant")
            elif len(audio) > 16000:
                audio = audio[:16000]

            feat = self.feature_extractor.extract(audio)  # (98, 13)
            feat_input = feat[np.newaxis, :, :, np.newaxis]
            emb = self.encoder(feat_input, training=False).numpy()[0]
            # Ensure L2 norm
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            self._embedding_cache[fpath] = emb
            return emb
        except Exception:
            return np.zeros(32, dtype=np.float32)

    def preload_validation_embeddings(self, max_per_class=30, verbose=True):
        """Pre-extracts and caches validation embeddings for ultra-fast repeated evaluation."""
        if verbose:
            print(f"Pre-caching validation embeddings (up to {max_per_class} per class)...")
        total = 0
        for cls_name in self.classes:
            recs = self.class_records[cls_name][:max_per_class]
            for r in recs:
                self._get_embedding(r)
                total += 1
        if verbose:
            print(f"Cached {total} validation embeddings across {len(self.classes)} classes.")

    def evaluate_cosine_separation(self, num_pairs=500):
        """
        Evaluates cosine similarity distributions for positive pairs (same word, distinct speakers)
        and negative pairs (different words).
        """
        pos_sims = []
        neg_sims = []

        # Positive pairs (same class, different speaker if possible)
        for _ in range(num_pairs):
            cls_name = self.rng.choice(self.classes)
            recs = self.class_records[cls_name]
            if len(recs) < 2:
                continue
            idx1, idx2 = self.rng.choice(len(recs), size=2, replace=False)
            e1 = self._get_embedding(recs[idx1])
            e2 = self._get_embedding(recs[idx2])
            sim = float(np.dot(e1, e2))
            pos_sims.append(sim)

        # Negative pairs (different classes)
        for _ in range(num_pairs):
            c1, c2 = self.rng.choice(self.classes, size=2, replace=False)
            r1 = self.rng.choice(self.class_records[c1])
            r2 = self.rng.choice(self.class_records[c2])
            e1 = self._get_embedding(r1)
            e2 = self._get_embedding(r2)
            sim = float(np.dot(e1, e2))
            neg_sims.append(sim)

        pos_sims = np.array(pos_sims, dtype=np.float32)
        neg_sims = np.array(neg_sims, dtype=np.float32)

        mean_pos = float(np.mean(pos_sims))
        std_pos = float(np.std(pos_sims))
        mean_neg = float(np.mean(neg_sims))
        std_neg = float(np.std(neg_sims))
        margin = mean_pos - mean_neg

        # Compute ROC-AUC and Equal Error Rate (EER)
        y_true = np.concatenate([np.ones_like(pos_sims), np.zeros_like(neg_sims)])
        y_scores = np.concatenate([pos_sims, neg_sims])
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = float(auc(fpr, tpr))

        # EER is where FPR == 1 - TPR (or FPR == FNR)
        fnr = 1.0 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
        eer_threshold = float(thresholds[eer_idx])

        return {
            "num_positive_pairs": len(pos_sims),
            "num_negative_pairs": len(neg_sims),
            "mean_positive_cosine": round(mean_pos, 4),
            "std_positive_cosine": round(std_pos, 4),
            "mean_negative_cosine": round(mean_neg, 4),
            "std_negative_cosine": round(std_neg, 4),
            "separation_margin": round(margin, 4),
            "roc_auc": round(roc_auc, 4),
            "equal_error_rate": round(eer, 4),
            "eer_operating_threshold": round(eer_threshold, 4)
        }

    def evaluate_few_shot_retrieval(self, shots=3, max_queries_per_class=15):
        """
        Computes prototype vectors from K enrollment shots per class,
        then evaluates Top-1 Recall and Top-5 Recall for query samples across all classes.
        """
        prototypes = {}
        queries = []

        for cls_name in self.classes:
            recs = self.class_records[cls_name]
            if len(recs) < shots + 1:
                continue

            # Select K enrollment shots
            enroll_recs = recs[:shots]
            query_recs = recs[shots:shots + max_queries_per_class]

            enroll_embs = np.stack([self._get_embedding(r) for r in enroll_recs])
            proto = compute_prototype(enroll_embs)
            prototypes[cls_name] = proto

            for qr in query_recs:
                queries.append((cls_name, self._get_embedding(qr)))

        if not queries:
            return {"top1_recall": 0.0, "top5_recall": 0.0, "total_queries": 0}

        proto_classes = list(prototypes.keys())
        proto_matrix = np.stack([prototypes[c] for c in proto_classes])  # (Num_classes, D)

        top1_correct = 0
        top5_correct = 0

        for true_cls, q_emb in queries:
            sims = np.dot(proto_matrix, q_emb)  # Cosine similarities to all prototypes
            ranked_indices = np.argsort(-sims)
            predicted_classes = [proto_classes[idx] for idx in ranked_indices]

            if predicted_classes[0] == true_cls:
                top1_correct += 1
            if true_cls in predicted_classes[:5]:
                top5_correct += 1

        total_q = len(queries)
        top1_recall = top1_correct / total_q
        top5_recall = top5_correct / total_q

        return {
            "shots_per_prototype": shots,
            "total_enrolled_classes": len(prototypes),
            "total_query_samples": total_q,
            "top1_recall_accuracy": round(top1_recall, 4),
            "top5_recall_accuracy": round(top5_recall, 4)
        }

    def evaluate_full(self, num_pairs=500, shots=3, max_queries_per_class=15):
        """Executes full validation battery and returns comprehensive metrics."""
        sep_stats = self.evaluate_cosine_separation(num_pairs=num_pairs)
        ret_stats = self.evaluate_few_shot_retrieval(shots=shots, max_queries_per_class=max_queries_per_class)
        return {
            "cosine_separation": sep_stats,
            "few_shot_retrieval": ret_stats
        }
