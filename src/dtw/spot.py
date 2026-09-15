"""Streaming subsequence-DTW spotter (Track B): open begin + open end.

Maintains one cumulative-cost column (+ step counts) over the template; each
incoming 10ms MFCC frame advances the column in O(T) time and memory. A
trigger fires when the full-template path cost per step drops below threshold;
columns clear on trigger (mirrors the neural path's Change-3b history clear)
plus a cooldown, so one keyword yields exactly one trigger.

No Sakoe-Chiba band here (subsequence DTW has no diagonal anchor); at T~98
the full column is ~3k flops/frame -- negligible on ESP32. RAM: 2 column
sets x (T+1) x (4B cost + 4B steps + 4B frames) ~= 2.4 KB on device (int32
counters: the consumed-frames count grows between triggers, int16 would wrap
over hours of silence).

Hardening (Sep-14): a third column counts consumed STREAM frames along the
best path; triggers require consumed >= min_duration_frames (default T//8),
which blocks pathological warps (whole template matched to 1-5 frames).
Measured on the zora pilot: genuine matches consume 31 frames (T=98), so
T//8=12 sits 2.6x below genuine and 2.4x+ above degenerate -- tolerant of
shorter words too (a T//3 default sat exactly ON genuine=31: zero margin).
"""
import numpy as np

from src.dtw.templates import Template

INF = float("inf")


class DTWSpotter:
    def __init__(self, template: Template, threshold: float,
                 cooldown_frames: int = 150, min_duration_frames=None):
        self.template = template
        self.threshold = float(threshold)
        self.cooldown_frames = int(cooldown_frames)
        if min_duration_frames is None:
            min_duration_frames = template.frame_count // 8
        self.min_duration_frames = int(min_duration_frames)
        self.dim = template.centroid.shape[1]
        self.n_triggers = 0
        self._reset_columns()
        self._cooldown_left = 0

    def _reset_columns(self):
        t = self.template.frame_count
        self._cost = np.full(t + 1, INF)
        self._cost[0] = 0.0      # open begin: empty template matches anywhere
        self._steps = np.zeros(t + 1, dtype=np.int64)
        self._frames = np.zeros(t + 1, dtype=np.int64)  # stream frames consumed

    def reset(self):
        self._reset_columns()
        self._cooldown_left = 0

    def update(self, frame: np.ndarray, timestamp_ms: float) -> dict:
        """Advance one MFCC frame; returns trigger report dict."""
        f = np.asarray(frame, dtype=np.float64).flatten()
        assert f.shape[0] == self.dim
        tpl = self.template.centroid
        t = self.template.frame_count
        new_cost = np.empty(t + 1)
        new_steps = np.empty(t + 1, dtype=np.int64)
        new_cost[0] = 0.0  # open begin, every frame
        new_steps[0] = 0
        new_frames = np.empty(t + 1, dtype=np.int64)
        new_frames[0] = 0
        for i in range(1, t + 1):
            d = float(((tpl[i - 1] - f) ** 2).sum())
            # (i,j-1)->(i,j): stream advances; (i-1,j)->(i,j): stream stays;
            # (i-1,j-1)->(i,j): both advance. (An earlier comment had the
            # first two swapped; the min is symmetric so costs were right,
            # but the frames column needs the true mapping.)
            ci, cj, cm = self._cost[i], new_cost[i - 1], self._cost[i - 1]
            if ci <= cj and ci <= cm:
                new_cost[i] = d + ci
                new_steps[i] = self._steps[i] + 1
                new_frames[i] = self._frames[i] + 1
            elif cj <= cm:
                new_cost[i] = d + cj
                new_steps[i] = new_steps[i - 1] + 1
                new_frames[i] = new_frames[i - 1]
            else:
                new_cost[i] = d + cm
                new_steps[i] = self._steps[i - 1] + 1
                new_frames[i] = self._frames[i - 1] + 1
        self._cost, self._steps, self._frames = new_cost, new_steps, new_frames

        total, steps = new_cost[t], int(new_steps[t])
        consumed = int(new_frames[t])
        norm = (total / steps) if steps > 0 and total < INF else INF
        triggered = False
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
        elif norm < self.threshold and consumed >= self.min_duration_frames:
            triggered = True
            self.n_triggers += 1
            self._cooldown_left = self.cooldown_frames
            self._reset_columns()  # same-keyword tail must not re-trigger
        return {"triggered": triggered,
                "timestamp_ms": float(timestamp_ms),
                "cost_total": (float(total) if total < INF else None),
                "cost_per_step": (float(norm) if norm < INF else None),
                "frames_consumed": consumed,
                "cooldown_left": self._cooldown_left}


class DTWSpotterBank:
    """One spotter per enrollment variant (speaking rate, accent...).

    Pilot lesson: a rate+1 keyword costs 130x the intra-template spread
    against a rate+0 centroid (TTS rate changes the spectrum, not just the
    timing -- warping cannot absorb it). The bank keeps one centroid per
    variant and ORs the triggers with a SHARED cooldown + column reset, so
    one keyword still yields exactly one trigger. Cost scales xK (K variants);
    on-device, an LB_Keogh pre-filter can skip cold variants per frame.
    """

    def __init__(self, templates: dict, threshold: float,
                 cooldown_frames: int = 150, **spotter_kw):
        assert len(templates) >= 1, "bank needs >= 1 template"
        dims = {t.centroid.shape[1] for t in templates.values()}
        assert len(dims) == 1, f"mixed MFCC dims in bank: {dims}"
        self.members = {name: DTWSpotter(tpl, threshold, cooldown_frames,
                                         **spotter_kw)
                        for name, tpl in templates.items()}
        self.threshold = float(threshold)
        self.cooldown_frames = int(cooldown_frames)
        self.n_triggers = 0

    def reset(self):
        for sp in self.members.values():
            sp.reset()

    def update(self, frame, timestamp_ms: float) -> dict:
        best = None
        for name, sp in self.members.items():
            r = sp.update(frame, timestamp_ms)
            if r["triggered"] and (best is None
                                   or r["cost_per_step"] < best["cost_per_step"]):
                best = {"template": name, **r}
        if best is not None:
            self.n_triggers += 1
            # Shared cooldown: a trigger by ANY member blinds ALL members,
            # and clears their columns (their DP state references pre-trigger
            # audio that must not echo into a second trigger).
            for sp in self.members.values():
                sp._cooldown_left = self.cooldown_frames
                sp._reset_columns()
            best["n_triggers"] = self.n_triggers
            return best
        costs = {n: sp._cost[sp.template.frame_count] for n, sp in self.members.items()}
        return {"triggered": False, "timestamp_ms": float(timestamp_ms),
                "cost_total": {n: (float(c) if c < INF else None)
                               for n, c in costs.items()}}
