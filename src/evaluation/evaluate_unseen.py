"""
Unseen Keyword Evaluation Engine ('ZORA' Few-Shot Test)
SIH Problem Statement 26172 - Milestone 9

Quantifies runtime few-shot keyword spotting performance on completely unseen words:
1. Tests enrollment shot counts: K in {1, 2, 3, 5}
2. Evaluates cosine similarity separation against diverse negative speech and noise
3. Computes ROC-AUC, Equal Error Rate (EER), and TPR at fixed low FPR operating points
4. Demonstrates variance reduction through prototype averaging
"""

import os
import sys
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import roc_curve, auc

sys.path.insert(0, r"D:\SIH_Model")
from src.features.mfcc import MFCCFeatureExtractor
from src.models.prototype import compute_cosine_similarity
from src.enrollment.enroll import KeywordEnrollmentManager

class UnseenKeywordEvaluator:
    def __init__(self, encoder, feature_extractor=None, seed=42):
        self.encoder = encoder
        self.rng = np.random.default_rng(seed)
        self.feature_extractor = feature_extractor or MFCCFeatureExtractor(sample_rate=16000, n_mfcc=13, n_mels=40)
        self.enrollment_manager = KeywordEnrollmentManager(encoder, self.feature_extractor)
        self._embedding_cache = {}

    def get_embedding(self, filepath: str) -> np.ndarray:
        if filepath in self._embedding_cache:
            return self._embedding_cache[filepath]
        emb = self.enrollment_manager.extract_embedding_from_file(filepath)
        self._embedding_cache[filepath] = emb
        return emb

    def evaluate_shots_comparison(self, target_files: list, impostor_files: list,
                                  shot_counts=(1, 2, 3, 5), num_trials=10, keyword_name="ZORA") -> dict:
        """
        Conducts comparative few-shot evaluation across varying enrollment shot counts.
        """
        if len(target_files) < max(shot_counts) + 2:
            raise ValueError(f"Need at least {max(shot_counts) + 2} target samples, got {len(target_files)}")

        # Pre-cache impostor embeddings
        impostor_embs = np.stack([self.get_embedding(f) for f in impostor_files])

        results_by_shot = {}

        for k in shot_counts:
            trial_pos_sims = []
            trial_neg_sims = []
            trial_margins = []
            trial_aucs = []
            trial_eers = []
            trial_eer_thresholds = []
            trial_tpr_at_1pct_fpr = []
            trial_tpr_at_5pct_fpr = []

            for trial in range(num_trials):
                # Randomly choose K enrollment shots
                indices = np.arange(len(target_files))
                self.rng.shuffle(indices)
                enroll_idx = indices[:k]
                query_idx = indices[k:]

                enroll_paths = [target_files[i] for i in enroll_idx]
                query_paths = [target_files[i] for i in query_idx]

                enroll_info = self.enrollment_manager.enroll(enroll_paths, keyword_name=keyword_name)
                proto = enroll_info["prototype"]

                query_embs = np.stack([self.get_embedding(f) for f in query_paths])

                # Compute cosine similarities
                pos_sims = np.dot(query_embs, proto)
                neg_sims = np.dot(impostor_embs, proto)

                mean_pos = float(np.mean(pos_sims))
                mean_neg = float(np.mean(neg_sims))
                margin = mean_pos - mean_neg

                # Compute ROC & EER
                y_true = np.concatenate([np.ones_like(pos_sims), np.zeros_like(neg_sims)])
                y_scores = np.concatenate([pos_sims, neg_sims])
                fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                roc_auc = float(auc(fpr, tpr))

                fnr = 1.0 - tpr
                eer_idx = np.nanargmin(np.abs(fpr - fnr))
                eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
                eer_thresh = float(thresholds[eer_idx])

                # TPR at fixed FPR
                idx_1pct = np.where(fpr <= 0.01)[0]
                tpr_1pct = float(tpr[idx_1pct[-1]]) if len(idx_1pct) > 0 else 0.0

                idx_5pct = np.where(fpr <= 0.05)[0]
                tpr_5pct = float(tpr[idx_5pct[-1]]) if len(idx_5pct) > 0 else 0.0

                trial_pos_sims.extend(pos_sims.tolist())
                trial_neg_sims.extend(neg_sims.tolist())
                trial_margins.append(margin)
                trial_aucs.append(roc_auc)
                trial_eers.append(eer)
                trial_eer_thresholds.append(eer_thresh)
                trial_tpr_at_1pct_fpr.append(tpr_1pct)
                trial_tpr_at_5pct_fpr.append(tpr_5pct)

            results_by_shot[f"{k}_shot"] = {
                "k_shots": k,
                "num_trials": num_trials,
                "mean_positive_sim": round(float(np.mean(trial_pos_sims)), 4),
                "std_positive_sim": round(float(np.std(trial_pos_sims)), 4),
                "mean_negative_sim": round(float(np.mean(trial_neg_sims)), 4),
                "std_negative_sim": round(float(np.std(trial_neg_sims)), 4),
                "separation_margin": round(float(np.mean(trial_margins)), 4),
                "roc_auc": round(float(np.mean(trial_aucs)), 4),
                "equal_error_rate": round(float(np.mean(trial_eers)), 4),
                "eer_threshold": round(float(np.mean(trial_eer_thresholds)), 4),
                "tpr_at_5pct_fpr": round(float(np.mean(trial_tpr_at_5pct_fpr)), 4),
                "tpr_at_1pct_fpr": round(float(np.mean(trial_tpr_at_1pct_fpr)), 4)
            }

        return {
            "keyword_name": keyword_name.upper(),
            "total_target_samples": len(target_files),
            "total_impostor_samples": len(impostor_files),
            "shot_evaluations": results_by_shot
        }
