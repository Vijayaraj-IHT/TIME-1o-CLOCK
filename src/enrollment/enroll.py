"""
Runtime Few-Shot Keyword Enrollment Manager
SIH Problem Statement 26172 - Milestone 9

Simulates edge device enrollment:
1. Takes K spoken samples of an unseen keyword (e.g., 'ZORA')
2. Extracts MFCC features (98, 13)
3. Computes L2-normalized 32-D embeddings via frozen encoder
4. Computes prototype vector centroid: p = L2_norm(mean(e))
5. Exports C/C++ firmware header for ESP32-S3 deployment
"""

import os
import sys
from typing import Optional
import numpy as np
import soundfile as sf

# Repo root from this file location (portable; was a hardcoded Windows path).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)
from src.features.mfcc import MFCCFeatureExtractor
from src.models.prototype import compute_prototype, compute_cosine_similarity, compute_garbage_prototype

class KeywordEnrollmentManager:
    # Fix: enroll() computed intra_similarity_min but never acted on it. A judge/user
    # speaking the keyword inconsistently (volume, mispronunciation, wrong word by
    # mistake) produced a low-quality prototype that was still exported to firmware
    # with no warning - failure only surfaced live, during the actual demo.
    MIN_ACCEPTABLE_INTRA_SIMILARITY = 0.5

    def __init__(self, encoder, feature_extractor=None):
        self.encoder = encoder
        self.feature_extractor = feature_extractor or MFCCFeatureExtractor(sample_rate=16000, n_mfcc=13, n_mels=40)

    def extract_embedding_from_audio(self, audio: np.ndarray) -> np.ndarray:
        """Extracts L2-normalized 32-D embedding from 1D audio array."""
        if len(audio) < 16000:
            audio = np.pad(audio, (0, 16000 - len(audio)), mode="constant")
        elif len(audio) > 16000:
            audio = audio[:16000]

        feat = self.feature_extractor.extract(audio)  # (98, 13)
        feat_batch = feat[np.newaxis, :, :, np.newaxis]
        emb = self.encoder(feat_batch, training=False).numpy()[0]
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.astype(np.float32)

    def extract_embedding_from_file(self, filepath: str) -> np.ndarray:
        """Loads WAV file and extracts L2-normalized embedding."""
        audio, sr = sf.read(filepath, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Fix: was channel-0-only downmix, silently dropped other channels
        return self.extract_embedding_from_audio(audio)

    def enroll(self, audio_inputs, keyword_name="ZORA", negative_inputs=None) -> dict:
        """
        Enrolls target keyword using K spoken utterances.
        audio_inputs: List of filepaths or list of numpy audio arrays.
        """
        embeddings = []
        for inp in audio_inputs:
            if isinstance(inp, str):
                emb = self.extract_embedding_from_file(inp)
            elif isinstance(inp, np.ndarray):
                emb = self.extract_embedding_from_audio(inp)
            else:
                raise ValueError("Expected str filepath or np.ndarray audio")
            embeddings.append(emb)

        embeddings = np.array(embeddings, dtype=np.float32)
        k_shots = len(embeddings)

        # Compute centroid prototype
        prototype = compute_prototype(embeddings)

        # Compute intra-enrollment pairwise similarities
        if k_shots >= 2:
            sim_matrix = np.dot(embeddings, embeddings.T)
            # Upper triangle off-diagonal elements
            triu_idx = np.triu_indices(k_shots, k=1)
            pair_sims = sim_matrix[triu_idx]
            intra_mean = float(np.mean(pair_sims))
            intra_min = float(np.min(pair_sims))
            intra_std = float(np.std(pair_sims))
        else:
            intra_mean = 1.0
            intra_min = 1.0
            intra_std = 0.0

        # Compute distance of individual shots to centroid
        shot_to_proto_sims = [compute_cosine_similarity(prototype, emb) for emb in embeddings]

        is_valid = intra_min >= self.MIN_ACCEPTABLE_INTRA_SIMILARITY
        if not is_valid:
            print(
                f"[WARN] Enrollment quality gate FAILED for '{keyword_name.upper()}': "
                f"intra_similarity_min={intra_min:.4f} < required {self.MIN_ACCEPTABLE_INTRA_SIMILARITY}. "
                f"The K enrollment utterances are too inconsistent (volume/pronunciation/wrong word) - "
                f"re-record before exporting this prototype to firmware."
            )

        # Phase-1 Change 2: optional garbage prototype from negatives
        garbage_prototype = None
        garbage_n = 0
        if negative_inputs:
            neg_embs = []
            for inp in negative_inputs:
                if isinstance(inp, str):
                    neg_embs.append(self.extract_embedding_from_file(inp))
                elif isinstance(inp, np.ndarray):
                    neg_embs.append(self.extract_embedding_from_audio(inp))
                else:
                    raise ValueError("Expected str filepath or np.ndarray audio")
            garbage_prototype = compute_garbage_prototype(np.array(neg_embs, dtype=np.float32))
            garbage_n = len(neg_embs)

        return {
            "keyword_name": keyword_name.upper(),
            "k_shots": k_shots,
            "prototype": prototype,
            "embeddings": embeddings,
            "garbage_prototype": garbage_prototype,
            "garbage_n": garbage_n,
            "intra_similarity_mean": round(intra_mean, 4),
            "intra_similarity_min": round(intra_min, 4),
            "intra_similarity_std": round(intra_std, 4),
            "shot_to_prototype_sims": [round(float(s), 4) for s in shot_to_proto_sims],
            "is_valid": is_valid,  # Fix: callers (and export_prototype_c_header) must check this
        }

    def export_prototype_c_header(self, prototype: np.ndarray, keyword_name="ZORA",
                                  output_filepath=os.path.join(_REPO_ROOT, "src", "deployment", "esp32", "keyword_prototype.h"),
                                  is_valid: Optional[bool] = None, force: bool = False,
                                  garbage: Optional[np.ndarray] = None):
        """
        Exports the prototype array to a C header file for ESP32-S3 firmware.

        Fix: previously exported unconditionally regardless of enrollment quality.
        Pass the `is_valid` flag from enroll()'s return dict; if False, this now
        refuses to write firmware for a prototype known to be low-quality unless
        force=True is explicitly passed.
        """
        if is_valid is False and not force:
            raise ValueError(
                f"Refusing to export firmware header for '{keyword_name.upper()}': "
                f"enrollment failed the quality gate (intra_similarity_min below "
                f"{self.MIN_ACCEPTABLE_INTRA_SIMILARITY}). Re-enroll, or pass force=True "
                f"to override at your own risk."
            )
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        dim = len(prototype)
        proto_vals = ", ".join([f"{v:.7f}f" for v in prototype])
        if garbage is None:
            garbage_vals = ", ".join(["0.0000000f"] * dim)
        else:
            g = np.asarray(garbage, dtype=np.float32).flatten()
            assert len(g) == dim, "garbage embedding dim mismatch"
            garbage_vals = ", ".join([f"{v:.7f}f" for v in g])

        header_content = f"""/*
 * Auto-generated Keyword Prototype Header for ESP32-S3
 * SIH Problem Statement 26172
 * Keyword: {keyword_name.upper()}
 * Embedding Dimension: {dim}
 * Hypersphere Normalized: True (||p||_2 = 1.0)
 */

#ifndef KEYWORD_PROTOTYPE_H_
#define KEYWORD_PROTOTYPE_H_

#define ENROLLED_KEYWORD_NAME "{keyword_name.upper()}"
#define EMBEDDING_DIMENSION {dim}

// L2-normalized prototype vector centroid
static const float ENROLLED_KEYWORD_PROTOTYPE[{dim}] = {{
    {proto_vals}
}};

// Phase-1 Change 2: garbage (non-keyword) prototype + veto margin.
// All-zeros = veto disabled (dot = 0, veto only fires below margin anyway).
#define GARBAGE_MARGIN 0.05f
static const float GARBAGE_PROTOTYPE[{dim}] = {{
    {garbage_vals}
}};

#endif // KEYWORD_PROTOTYPE_H_
"""
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(header_content)
        return output_filepath
