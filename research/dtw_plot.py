"""Plots for DTW alignments. matplotlib is imported here only, so pose_dtw stays pure NumPy."""

from __future__ import annotations

import numpy as np

from pose_dtw import StretchReport, warp_curve


def plot_alignments(rows: list[dict], fps: float = 30.0, suptitle: str | None = None):
    """One row of three panels per alignment; returns the matplotlib Figure.

    Each row is a dict with keys: title, cost (n_ref, n_query), result (DTWResult),
    report (StretchReport, optional), band (Sakoe-Chiba radius used, optional).

    Panels
      1. Cost matrix (dark = similar) with the path in red. Diagonal-ish = same timing.
         Long horizontal or vertical stretches of the path = one video held still while the
         other plays. Dashed lines mark the band, when there is one. Vertical or horizontal
         stripes of the cost = frames with tracking dropouts (no usable joints, constant cost).
      2. Timing offset: where the matched query time sits relative to expected, per reference
         second. Expected = the straight line (0,0) -> (n-1,m-1) when the path covers both
         recordings end to end, otherwise (subsequence) 1:1 from the path's start. Positive =
         learner behind. Flat = steady timing, steps = skips or pauses.
      3. Local rate: query frames consumed per reference frame over a sliding window. 1 = same
         speed. Shaded = allowed range; red dots = windows outside it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 3, figsize=(16, 4.6 * len(rows)), squeeze=False,
                             gridspec_kw={"width_ratios": [1.15, 1, 1]})
    for axs, row in zip(axes, rows):
        cost, res = row["cost"], row["result"]
        rep: StretchReport | None = row.get("report")
        band = row.get("band")
        n, m = cost.shape
        path = res.path
        ax = axs[0]
        ax.imshow(cost, origin="lower", aspect="auto", cmap="cividis",
                  vmin=0, vmax=float(np.percentile(cost, 95)),
                  extent=(-0.5 / fps, (m - 0.5) / fps, -0.5 / fps, (n - 0.5) / fps))
        slope = (m - 1) / (n - 1) if n > 1 else 0.0
        if band is not None:
            i = np.array([0, n - 1])
            ax.plot((i * slope - band) / fps, i / fps, "w--", lw=0.9)
            ax.plot((i * slope + band) / fps, i / fps, "w--", lw=0.9)
        ax.plot(path[:, 1] / fps, path[:, 0] / fps, color="tab:red", lw=1.6)
        ax.set_xlim(-0.5 / fps, (m - 0.5) / fps)
        ax.set_ylim(-0.5 / fps, (n - 0.5) / fps)
        ax.set_xlabel("learner time (s)")
        ax.set_ylabel("reference time (s)")
        head = f"{row['title']}\nnormalized cost {res.normalized_cost:.3f}"
        if rep is not None:
            head += (f"  |  longest run H{rep.longest_horizontal_run} V{rep.longest_vertical_run} frames"
                     f"  |  {'EXCESSIVE STRETCH' if rep.excessive else 'stretch OK'}")
        ax.set_title(head, fontsize=10, loc="left", color="tab:red" if rep is not None and rep.excessive else "black")

        qm = warp_curve(path, n)
        idx = np.arange(n)
        full_span = path[0, 1] == 0 and path[-1, 1] == m - 1
        expected = idx * slope if full_span else qm[0] + idx
        ax = axs[1]
        ax.plot(idx / fps, (qm - expected) / fps, color="tab:blue", lw=1.4)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel("reference time (s)")
        ax.set_ylabel("learner behind (+) / ahead (-)  (s)")
        ax.set_title("timing offset vs expected", fontsize=10, loc="left")
        ax.grid(alpha=0.3)

        ax = axs[2]
        if rep is not None and n > rep.window:
            w = rep.window
            rates = (qm[w:] - qm[:-w]) / w
            x = (np.arange(len(rates)) + w / 2) / fps
            lo, hi = rep.rate_range
            ax.axhspan(lo, hi, color="tab:green", alpha=0.12)
            ax.plot(x, rates, color="tab:blue", lw=1.4)
            out = (rates < lo) | (rates > hi)
            ax.plot(x[out], rates[out], "o", color="tab:red", ms=3)
            ax.set_ylim(min(-0.1, rates.min() - 0.1), max(hi + 0.3, rates.max() + 0.1))
            ax.set_title(f"local rate over {w / fps:g} s windows ({np.mean(out):.0%} outside range)",
                         fontsize=10, loc="left")
        else:
            ax.set_title("local rate (path too short for window)", fontsize=10, loc="left")
        ax.axhline(1.0, color="gray", lw=0.8)
        ax.set_xlabel("reference time (s)")
        ax.set_ylabel("learner frames per reference frame")
        ax.grid(alpha=0.3)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    return fig