"""K-shot template building (Track B): length-normalized centroid.

Deliberately simple (DTW Barycenter Averaging is future work): resample each
enrollment clip's MFCC to the median frame count with linear interpolation,
then take the per-frame mean. Deterministic, O(K*T*D), trivially portable.
"""
from dataclasses import dataclass, field

import numpy as np

from src.dtw.dtw_core import dtw_distance


def resample_frames(mfcc: np.ndarray, target_len: int) -> np.ndarray:
    """Linear-interpolate an (N, D) frame matrix to (target_len, D)."""
    mfcc = np.asarray(mfcc, dtype=np.float64)
    n = mfcc.shape[0]
    if n == target_len:
        return mfcc.copy()
    if n == 1:
        return np.repeat(mfcc, target_len, axis=0)
    xp = np.linspace(0.0, 1.0, n)
    x = np.linspace(0.0, 1.0, target_len)
    out = np.empty((target_len, mfcc.shape[1]))
    for d in range(mfcc.shape[1]):
        out[:, d] = np.interp(x, xp, mfcc[:, d])
    return out


@dataclass
class Template:
    name: str
    centroid: np.ndarray          # (T, D) mean frames
    frame_count: int              # T (median enrollment length)
    n_clips: int
    intra_dists: list = field(default_factory=list)  # dtw(clip, centroid)/step


def build_template(name: str, clip_mfccs: list, band=None) -> Template:
    """Build a centroid template from K enrollment MFCC matrices."""
    assert len(clip_mfccs) >= 1, "need >=1 enrollment clip"
    t = int(np.median([m.shape[0] for m in clip_mfccs]))
    normed = [resample_frames(m, t) for m in clip_mfccs]
    centroid = np.mean(np.stack(normed), axis=0)
    if band is None:
        band = max(8, t // 10)
    intra = []
    for m in normed:
        tot = dtw_distance(m, centroid, band=band)
        intra.append(tot / (2 * t))
    return Template(name=name, centroid=centroid, frame_count=t,
                    n_clips=len(clip_mfccs), intra_dists=intra)
