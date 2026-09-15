"""
Prototype Vector Computation & Metric Comparison
SIH Problem Statement 26172
"""

import numpy as np

def compute_prototype(embeddings: np.ndarray) -> np.ndarray:
    """
    Computes a single prototype vector from K enrollment embeddings.
    p = L2_normalize(mean(embeddings))
    """
    if embeddings.ndim == 1:
        embeddings = embeddings[np.newaxis, :]

    mean_vec = np.mean(embeddings, axis=0, keepdims=True)
    norm = np.linalg.norm(mean_vec, axis=1, keepdims=True)
    if norm[0, 0] == 0:
        # Fix: previously returned the raw zero vector silently. Every future
        # compute_cosine_similarity(prototype, query) = dot(0, q) = 0 forever -
        # the enrolled keyword becomes permanently, silently undetectable, and the
        # caller (enroll.py) had no signal that anything went wrong. This almost
        # always means the encoder produced degenerate/antipodal embeddings for the
        # enrollment utterances (bad audio, wrong file, or an encoder bug) - it must
        # be surfaced, not swallowed.
        raise ValueError(
            "compute_prototype: mean of enrollment embeddings has zero norm - "
            "enrollment is invalid. Re-record the keyword and verify the encoder "
            "is producing sane (non-degenerate) embeddings before retrying."
        )
    prototype = (mean_vec / norm)[0]
    return prototype.astype(np.float32)

def compute_cosine_similarity(prototype: np.ndarray, query: np.ndarray) -> float:
    """
    Computes cosine similarity between an L2-normalized prototype and query.
    For L2-normalized vectors, cosine similarity is identical to the dot product.
    """
    # Ensure 1D vectors
    p = prototype.flatten()
    q = query.flatten()
    sim = float(np.dot(p, q))
    return max(min(sim, 1.0), -1.0)


# --- Phase-1 Change 2: garbage / background prototype (D-ProtoNets + Sphinx garbage loop) ---

GARBAGE_MARGIN_DEFAULT = 0.05


def compute_garbage_prototype(neg_embeddings: np.ndarray, min_count: int = 3) -> np.ndarray:
    """
    Builds the 'non-keyword' prototype from background/confuser/silence embeddings.
    Same math as compute_prototype (L2-normalized mean) but requires >= min_count
    samples: a garbage model from 1-2 clips is unstable and vetoes erratically.
    """
    neg_embeddings = np.asarray(neg_embeddings, dtype=np.float32)
    if neg_embeddings.ndim == 1:
        neg_embeddings = neg_embeddings[np.newaxis, :]
    if neg_embeddings.shape[0] < min_count:
        raise ValueError(
            f"compute_garbage_prototype: need >= {min_count} negative embeddings, "
            f"got {neg_embeddings.shape[0]}. Collect more background/confuser clips."
        )
    return compute_prototype(neg_embeddings)


def apply_garbage_veto(s_kw: float, s_gb: float, margin: float = GARBAGE_MARGIN_DEFAULT):
    """
    Open-set rejection rule. Returns (score, vetoed).
    Veto fires when the keyword fails to beat garbage by `margin` (strict <).
    Vetoed score is -1.0 (below any tau_low) so smoothing decays immediately.
    Callers pass garbage=None through as 'no veto' (legacy behavior).
    """
    s_kw = float(s_kw)
    s_gb = float(s_gb)
    if (s_kw - s_gb) < float(margin):
        return -1.0, True
    return s_kw, False
