"""Synthetic benchmark for body-scale estimators.

An estimator is any function  f(L) -> s  where
    L : (T,) observed torso length per frame, NaN where the frame is missing
    s : scalar (one constant scale) or (T,) array (a time-varying scale)

and the aim is to recover the dancer's true scale. The scenarios inject the things
that make torso length unreliable: noise, bending (which only ever SHORTENS the
torso), long held bends, tracking dropouts, and genuine scale drift (e.g. a dancer
walking toward the camera in image space).

Use it on your own estimator:

    from scale_benchmark import evaluate, print_report
    print_report({"mine": my_estimator})

Run `python scale_benchmark.py` to see the built-in baselines.
"""

from __future__ import annotations

import warnings

import numpy as np

TRUE_SCALE = 0.5          # metres, a plausible torso length
SCENARIOS = ("noise_only", "short_bends", "long_bend", "mostly_bent", "dropouts", "scale_drift")


# ------------------------------------------------------------------ data
def _trapezoid(T: int, start: int, length: int, depth: float, ramp: int = 5) -> np.ndarray:
    t = np.arange(T, dtype=float)
    up = np.clip((t - start) / ramp, 0, 1)
    down = np.clip((start + length - t) / ramp, 0, 1)
    return depth * np.minimum(up, down)


def make_scenario(name: str, T: int = 300, seed: int = 0, noise: float = 0.02):
    """Return (L_observed, s_true), both shape (T,). `noise` is relative (2% by default)."""
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario {name!r}; choose from {SCENARIOS}")
    rng = np.random.default_rng(seed)
    s_true = np.full(T, TRUE_SCALE)
    dip = np.zeros(T)
    missing = np.zeros(T, dtype=bool)

    if name == "short_bends":                       # four 10-frame bends, torso 25% shorter
        for start in (40, 100, 170, 240):
            dip += _trapezoid(T, start, 10, 0.25, ramp=3)
    elif name == "long_bend":                       # one 2-second (60-frame) bow, 30% shorter
        dip += _trapezoid(T, 100, 60, 0.30)
    elif name == "mostly_bent":                     # bent about two thirds of the time, 20% shorter
        for start in (20, 90, 160, 230):
            dip += _trapezoid(T, start, 50, 0.20)
    elif name == "dropouts":                        # 15% random misses plus a 30-frame outage
        missing[rng.random(T) < 0.15] = True
        missing[120:150] = True
    elif name == "scale_drift":                     # true scale really changes by 30%
        s_true = np.linspace(TRUE_SCALE, TRUE_SCALE * 1.3, T)

    observed = s_true * (1 - dip) * (1 + noise * rng.normal(size=T))
    observed[missing] = np.nan
    return observed, s_true


# --------------------------------------------------------------- baselines
def fill_gaps(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs; hold the first/last valid value at the edges."""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if not ok.any():
        return x
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok])


def ffill(x: np.ndarray) -> np.ndarray:
    """Hold the last valid value forward. Leading NaNs stay NaN (nothing to hold yet).

    This is the only gap-filling that never looks at future frames.
    """
    out = np.asarray(x, dtype=float).copy()
    last = np.nan
    for i in range(len(out)):
        if np.isfinite(out[i]):
            last = out[i]
        else:
            out[i] = last
    return out


def per_frame(L):
    """No smoothing: each frame is scaled by its own torso length."""
    return fill_gaps(L)


def global_percentile(q: float):
    """One constant per recording: the q-th percentile of valid torso lengths."""
    return lambda L: float(np.nanpercentile(L, q))


def rolling_percentile(window: int, q: float, causal: bool = False, min_count: int = 5):
    """Time-varying scale: q-th percentile over a sliding window.

    causal=True uses only past frames (needed for live feedback) and fills gaps by
    holding the last estimate, so the first frames can be NaN (no data yet).
    Otherwise the window is centred and gaps are linearly interpolated (fine offline,
    e.g. for the reference video).
    """
    def estimator(L):
        L = np.asarray(L, dtype=float)
        T, half = len(L), window // 2
        out = np.full(T, np.nan)
        for t in range(T):
            lo, hi = (max(0, t - window + 1), t + 1) if causal else (max(0, t - half), min(T, t + half + 1))
            w = L[lo:hi]
            w = w[np.isfinite(w)]
            if w.size >= min_count:
                out[t] = np.percentile(w, q)
        return ffill(out) if causal else fill_gaps(out)
    return estimator


BASELINES = {
    "per_frame": per_frame,
    "global_median": global_percentile(50),
    "global_p75": global_percentile(75),
    "rolling_median_31": rolling_percentile(31, 50),
    "rolling_p75_31": rolling_percentile(31, 75),
    "rolling_p75_91": rolling_percentile(91, 75),
    "rolling_p75_91_causal": rolling_percentile(91, 75, causal=True),
}


# -------------------------------------------------------------- evaluation
def evaluate(estimator, scenarios=SCENARIOS, T: int = 300, seeds=range(5)) -> dict:
    """Score an estimator. Per scenario, averaged over seeds:

    mean_rel_err   mean |s_est - s_true| / s_true          (accuracy)
    max_rel_err    worst frame, same measure               (worst case)
    max_step       largest frame-to-frame change of s_est / s_true
                   (a jump here is injected, fake joint motion after dividing)
    nan_frac       fraction of frames with no usable estimate
    """
    results = {}
    for name in scenarios:
        rows = []
        for seed in seeds:
            observed, s_true = make_scenario(name, T=T, seed=seed)
            est = np.asarray(estimator(observed.copy()), dtype=float)
            if est.ndim == 0:
                est = np.full(T, float(est))
            if est.shape != (T,):
                raise ValueError(f"Estimator must return a scalar or shape ({T},), got {est.shape}")
            rel = np.abs(est - s_true) / s_true
            finite = np.isfinite(est)
            rows.append((
                np.nanmean(rel) if finite.any() else np.nan,
                np.nanmax(rel) if finite.any() else np.nan,
                np.nanmax(np.abs(np.diff(est)) / s_true[1:]) if finite.sum() > 1 else np.nan,
                1.0 - finite.mean(),
            ))
        with warnings.catch_warnings():              # an estimator that returns only NaN is a valid case
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_rel, max_rel, max_step, nan_frac = np.nanmean(np.array(rows, dtype=float), axis=0)
        results[name] = {"mean_rel_err": float(mean_rel), "max_rel_err": float(max_rel),
                         "max_step": float(max_step), "nan_frac": float(nan_frac)}
    return results


def print_report(estimators: dict, metric: str = "mean_rel_err", **kwargs) -> None:
    results = {name: evaluate(fn, **kwargs) for name, fn in estimators.items()}
    scenarios = list(next(iter(results.values())))
    width = max(len(n) for n in estimators) + 2
    print(f"{metric} (%), lower is better")
    print(" " * width + "".join(f"{s:>13}" for s in scenarios))
    for name, res in results.items():
        print(f"{name:<{width}}" + "".join(f"{100 * res[s][metric]:>13.1f}" for s in scenarios))


if __name__ == "__main__":
    for metric in ("mean_rel_err", "max_rel_err", "max_step"):
        print_report(BASELINES, metric=metric)
        print()