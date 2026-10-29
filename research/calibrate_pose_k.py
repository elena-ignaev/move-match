"""Calibrate k of S_pose = 100 * exp(-k * mean_pose_distance) from example performances.

Two stages, so the distances can be inspected BEFORE any k is chosen:

    python research/calibrate_pose_k.py generate  --out research/pose_calibration_data.csv
        simulate graded performances of several dances, score them, save + print the distances
    python research/calibrate_pose_k.py calibrate --csv research/pose_calibration_data.csv --out-dir research
        choose k from the labelled distances, print the sweep, save figures

For real data, skip `generate` and give `calibrate` your own CSV with columns
    dance, level, mean_pose_distance
(level = a quality label you assigned, e.g. repeat / good / ok / poor). See --help for the level
names it expects; they are options, nothing is hard coded to the synthetic ones.

The synthetic quality levels are ASSUMPTIONS about what bad dancing looks like, not measurements.
They make the procedure runnable and show how the distances behave; they cannot say what a human
would call "good". Replace them with real labelled takes before trusting k.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # project root: pose_utils, landmark_store

import argparse
import csv
from pathlib import Path

import numpy as np

from pose_score import calibrate_k, distance_to_score, score_performance
from synthetic_dance import (FPS, Choreography, add_execution_error, add_tracking_artifacts,
                             scale_limbs, tau_from_knots, to_normalized)

DURATION = 16                                # seconds of choreography
SEG = [0, 4, 8, 12, 16]                      # tempo changes at these choreography times

# name, seed, motion amplitude, holds. Different seeds and amplitudes make different "dances".
DANCES = {
    "A": (0, 0.50, ((8.0, 4.0),)),
    "B": (1, 0.35, ((6.0, 3.0),)),
    "C": (2, 0.70, ()),
    "D": (3, 0.90, ((10.0, 3.0),)),
}

# level: (execution error sigma in meters, movement amplitude, limb scale, what it stands for)
# All levels also get tempo variation (+-15 % per 4 s), idle lead-in/out, tracking noise 1 cm and
# 1 % per-joint dropouts. Two extra negative controls are handled separately: "unrelated", "still".
LEVELS = {
    "repeat":     (0.00, 1.00, 1.00, "same dancer repeats it faithfully"),
    "other_body": (0.02, 1.00, 1.10, "skilled dancer, different limb proportions, 2 cm execution error"),
    "decent":     (0.04, 0.90, 1.00, "4 cm execution error, moves 10 % smaller"),
    "mediocre":   (0.08, 0.75, 1.00, "8 cm execution error, moves 25 % smaller"),
    "poor":       (0.15, 0.50, 1.00, "15 cm execution error, moves half size"),
    "unrelated":  (0.00, 1.00, 1.00, "performs a different dance"),
    "still":      (0.00, 1.00, 1.00, "stands still"),
}
TAKES = 5


def simulate(dance: str, level: str, take: int):
    seed, motion, holds = DANCES[dance]
    rng = np.random.default_rng(1000 * (list(DANCES).index(dance) + 1) + 10 * list(LEVELS).index(level) + take)
    sigma, amp, limb, _ = LEVELS[level]

    # reference: clean take of the choreography
    ref_choreo = Choreography(seed=seed, motion=motion, holds=holds)
    ref_xyz, ref_vis = add_tracking_artifacts(ref_choreo.pose(np.arange(DURATION * FPS) / FPS),
                                              noise=0.005, dropout=0.0, seed=1)
    ref = to_normalized(ref_xyz, ref_vis)

    # learner timeline: random tempo per segment, idle lead-in and tail
    idle = rng.uniform(0.3, 0.8)
    durations = np.diff(SEG) * rng.uniform(0.9, 1.15, len(SEG) - 1)
    learner_knots = np.concatenate([[0.0], idle + np.concatenate([[0.0], np.cumsum(durations)]), [idle + durations.sum() + 0.5]])
    tau = tau_from_knots([0] + SEG + [SEG[-1]], learner_knots)
    if level == "still":
        tau = np.zeros_like(tau)
    choreo = Choreography(seed=seed + 100, motion=motion, holds=holds) if level == "unrelated" else ref_choreo
    limb_factor = limb if level != "other_body" else (1.1 if take % 2 == 0 else 0.9)

    xyz = scale_limbs(choreo.pose(tau, amp), limb_factor)
    xyz = add_execution_error(xyz, sigma, seed=int(rng.integers(1 << 30)))
    xyz, vis = add_tracking_artifacts(xyz, noise=0.01, dropout=0.01, seed=int(rng.integers(1 << 30)))
    return ref, to_normalized(xyz, vis), vis


def generate(out: Path) -> list[dict]:
    rows = []
    for dance in DANCES:
        for level in LEVELS:
            for take in range(TAKES):
                ref, q, vis = simulate(dance, level, take)
                r = score_performance(ref, q, None, vis, k=1.0, fps=FPS, band_s=3.0, max_run=8)
                s = r.summary
                rows.append(dict(dance=dance, level=level, take=take,
                                 mean_pose_distance=r.mean_pose_distance, path_mean_distance=r.path_mean_distance,
                                 p90_distance=r.p90_distance, tracking_coverage=r.tracking_coverage,
                                 path_length=s.path_length, stretch_fraction=s.stretch_fraction,
                                 longest_h=s.stretch.longest_horizontal_run, longest_v=s.stretch.longest_vertical_run,
                                 flagged=int(s.stretch.excessive), band_edge=s.band_edge_fraction,
                                 n_warnings=len(r.warnings)))
        print(f"  dance {dance} done")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {len(rows)} performances to {out}")
    return rows


def load(path: Path) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["mean_pose_distance"] = float(r["mean_pose_distance"])
    return rows


def distance_table(rows: list[dict]) -> None:
    dances = sorted({r["dance"] for r in rows})
    levels = list(dict.fromkeys(r["level"] for r in rows))
    print(f"\nmean_pose_distance (torso lengths): median [min-max] over takes")
    print(f"{'level':12s}" + "".join(f"{'dance ' + d:>22s}" for d in dances) + f"{'all dances':>22s}")
    for lv in levels:
        cells = []
        for d in dances:
            v = np.array([r["mean_pose_distance"] for r in rows if r["level"] == lv and r["dance"] == d])
            cells.append(f"{np.median(v):.3f} [{v.min():.3f}-{v.max():.3f}]")
        v = np.array([r["mean_pose_distance"] for r in rows if r["level"] == lv])
        cells.append(f"{np.median(v):.3f} [{v.min():.3f}-{v.max():.3f}]")
        print(f"{lv:12s}" + "".join(f"{c:>22s}" for c in cells))


def group(rows, key=lambda r: True):
    out: dict[str, list[float]] = {}
    for r in rows:
        if key(r):
            out.setdefault(r["level"], []).append(r["mean_pose_distance"])
    return {k: np.array(v) for k, v in out.items()}


def calibrate(csv_path: Path, order: list[str], floor: str, bad: str, controls: list[str], out_dir: Path,
              floor_min: float, bad_max: float, min_gap: float) -> None:
    rows = load(csv_path)
    distance_table(rows)
    graded = group(rows, lambda r: r["level"] in order)
    cal = calibrate_k(graded, order, floor, bad, floor_min, bad_max, min_gap)
    print(f"\nrequirements: '{floor}' >= {floor_min:g}, '{bad}' <= {bad_max:g}, adjacent levels >= {min_gap:g} points apart")
    print(f"pooled over all dances: feasible={cal.feasible}  k in [{cal.k_min:.3f}, {cal.k_max:.3f}]  chosen k={cal.k:.3f}")
    for n in cal.notes:
        print("  note:", n)

    # does one k serve every dance?
    dances = sorted({r["dance"] for r in rows})
    lo, hi, n_feasible = 0.0, np.inf, 0
    print("\nper-dance feasible k range (same requirements, that dance only):")
    for d in dances:
        c = calibrate_k(group(rows, lambda r, d=d: r["dance"] == d and r["level"] in order), order, floor, bad,
                        floor_min, bad_max, min_gap)
        print(f"  dance {d}: " + (f"[{c.k_min:.3f}, {c.k_max:.3f}]" if c.feasible else "infeasible"))
        if c.feasible:
            n_feasible += 1
            lo, hi = max(lo, c.k_min), min(hi, c.k_max)
    if n_feasible == 0:
        print("  no dance is feasible on its own: the requirements conflict with this score shape, "
              "so there is nothing to intersect")
    elif n_feasible < len(dances) or lo > hi:
        print(f"  {n_feasible}/{len(dances)} dances feasible; intersection EMPTY -> dances need their own k")
    else:
        print(f"  intersection: [{lo:.3f}, {hi:.3f}] -> one global k works")

    print(f"\nscores at k = {cal.k:.3f} (median [min-max] of individual takes):")
    all_levels = order + controls
    for lv in all_levels:
        v = np.array([r["mean_pose_distance"] for r in rows if r["level"] == lv])
        s = distance_to_score(v, cal.k)
        print(f"  {lv:12s} {np.median(s):5.1f}  [{s.min():5.1f}-{s.max():5.1f}]")
    plot(rows, order, controls, cal, out_dir)


def plot(rows, order, controls, cal, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = order + controls
    dances = sorted({r["dance"] for r in rows})
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    # 1: raw distances per level, coloured by dance
    for di, d in enumerate(dances):
        for li, lv in enumerate(levels):
            v = [r["mean_pose_distance"] for r in rows if r["level"] == lv and r["dance"] == d]
            ax[0].plot(li + (di - 1.5) * 0.12 + np.zeros(len(v)), v, "o", color=f"C{di}", ms=4, alpha=0.8,
                       label=f"dance {d}" if li == 0 else None)
    ax[0].set_xticks(range(len(levels)), levels, rotation=30)
    ax[0].set_ylabel("mean_pose_distance (torso lengths)")
    ax[0].set_title("1. distances before any k", loc="left")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    # 2: mapping curve with medians
    grid = np.linspace(0, max(r["mean_pose_distance"] for r in rows) * 1.05, 300)
    ax[1].plot(grid, distance_to_score(grid, cal.k), "k-", lw=1.5, label=f"k = {cal.k:.2f}")
    for lv in levels:
        v = np.array([r["mean_pose_distance"] for r in rows if r["level"] == lv])
        ax[1].plot(v, distance_to_score(v, cal.k), "o", ms=4, alpha=0.6, label=lv)
    ax[1].set_xlabel("mean_pose_distance")
    ax[1].set_ylabel("pose score")
    ax[1].set_title("2. chosen mapping with every performance", loc="left")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    # 3: sweep
    ks = np.geomspace(0.1, 20, 300)
    for lv in order:
        ax[2].plot(ks, 100 * np.exp(-ks * cal.medians[lv]), label=lv)
    if cal.feasible:
        ax[2].axvspan(cal.k_min, cal.k_max, color="tab:green", alpha=0.15, label="feasible k")
    ax[2].axvline(cal.k, color="k", ls="--")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("k")
    ax[2].set_ylabel("median score of level")
    ax[2].set_title("3. k sweep (medians) and feasible range", loc="left")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    out = out_dir / "pose_k_calibration.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--out", type=Path, default=Path("pose_calibration_data.csv"))
    c = sub.add_parser("calibrate")
    c.add_argument("--csv", type=Path, default=Path("pose_calibration_data.csv"))
    c.add_argument("--order", nargs="+", default=["other_body", "decent", "mediocre", "poor"],
                   help="graded levels, best first")
    c.add_argument("--floor", default="other_body", help="level that must keep a high score")
    c.add_argument("--bad", default="poor", help="level that must get a low score")
    c.add_argument("--controls", nargs="*", default=["repeat", "unrelated", "still"],
                   help="levels reported but not used to fit k")
    c.add_argument("--floor-min", type=float, default=90.0)
    c.add_argument("--bad-max", type=float, default=30.0)
    c.add_argument("--min-gap", type=float, default=8.0)
    c.add_argument("--out-dir", type=Path, default=Path("."))
    a = ap.parse_args()
    if a.cmd == "generate":
        rows = generate(a.out)
        distance_table(rows)
    else:
        calibrate(a.csv, a.order, a.floor, a.bad, a.controls, a.out_dir, a.floor_min, a.bad_max, a.min_gap)