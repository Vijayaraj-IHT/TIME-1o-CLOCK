"""Core DTW distances (Track B). Numpy-only, no numba dependency.

Frame metric is SQUARED Euclidean on MFCC vectors (standard for DTW-KWS);
all distances below are total path costs in that unit. Callers normalize by
path steps when comparing across different lengths.
"""
import numpy as np

INF = float("inf")


def dtw_distance(a: np.ndarray, b: np.ndarray, band=None) -> float:
    """Classic DTW total cost with a Sakoe-Chiba band.

    a: (N, D), b: (M, D). band: max |i-j| in frames (None = unconstrained).
    Returns INF when no in-band path exists (|N-M| > band) -- this doubles as
    a duration gate: wildly wrong-length inputs cannot match.
    Cost model (also the ESP32 budget): N rows x (2*band+1) cells.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n, m = a.shape[0], b.shape[0]
    if band is None:
        band = max(n, m)
    if abs(n - m) > band:
        return INF
    prev = np.full(m + 1, INF)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = np.full(m + 1, INF)
        j0 = max(1, i - band)
        j1 = min(m, i + band)
        # frame distances for this row, vectorized over the band
        d = ((a[i - 1] - b[j0 - 1:j1]) ** 2).sum(axis=1)
        for k, j in enumerate(range(j0, j1 + 1)):
            best_prev = prev[j]
            if prev[j - 1] < best_prev:
                best_prev = prev[j - 1]
            if cur[j - 1] < best_prev:
                best_prev = cur[j - 1]
            cur[j] = d[k] + best_prev
        prev = cur
    return float(prev[m])


def dtw_envelope(template: np.ndarray, band: int):
    """Per-dim upper/lower envelope of `template` over a +/-band window.

    Returns (upper, lower), each (T, D). Used by lb_keogh.
    """
    t = np.asarray(template, dtype=np.float64)
    n, d = t.shape
    upper = np.empty_like(t)
    lower = np.empty_like(t)
    for i in range(n):
        lo, hi = max(0, i - band), min(n, i + band + 1)
        upper[i] = t[lo:hi].max(axis=0)
        lower[i] = t[lo:hi].min(axis=0)
    return upper, lower


def lb_keogh(query: np.ndarray, upper: np.ndarray, lower: np.ndarray) -> float:
    """LB_Keogh lower bound: sum of squared excursions outside the envelope.

    query: (N, D); upper/lower: (N, D) envelopes (same length -- resample the
    query to template length first, as the spotter/template code does).
    Guaranteed <= the (unconstrained) DTW distance: use it to skip full DTW
    when the bound alone exceeds the threshold (ESP32 cascade).
    """
    q = np.asarray(query, dtype=np.float64)
    acc = 0.0
    for i in range(q.shape[0]):
        hi = q[i] > upper[i]
        acc += float(((q[i][hi] - upper[i][hi]) ** 2).sum())
        lo = q[i] < lower[i]
        acc += float(((lower[i][lo] - q[i][lo]) ** 2).sum())
    return acc
