"""Track-B DTW keyword-spotting prototype (classical alternative to the neural path).

Same MFCC-13 frontend as the neural pipeline; different matcher: template DTW
instead of embedding cosine. See docs/TRACK_B_DTW_DESIGN.md.
"""
from src.dtw.dtw_core import dtw_distance, lb_keogh, dtw_envelope
from src.dtw.templates import build_template, resample_frames, Template
from src.dtw.spot import DTWSpotter

__all__ = ["dtw_distance", "lb_keogh", "dtw_envelope",
           "build_template", "resample_frames", "Template", "DTWSpotter"]
