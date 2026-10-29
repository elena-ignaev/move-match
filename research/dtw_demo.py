"""Run constrained DTW on synthetic dances with a known warp, print metrics, save figures.

    python research/dtw_demo.py [output_dir]

Scenarios (all learners are simulated with a known timeline, so alignment error is measurable):
  1  complete performance, tempo drifting, a 4 s hold the learner shortens to 2.5 s
       -> unconstrained vs Sakoe-Chiba band vs band + slope constraint
  2  learner skips 4 s of the choreography
       -> unconstrained path squashes the skipped part; constraints cannot follow (cost shows it)
  3  one 8 s segment performed inside a longer recording, and a recording that starts 6 s early
       -> subsequence DTW vs a fixed-start band
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # project root: pose_utils, landmark_store

import sys
import time
from pathlib import Path

import numpy as np

from dtw_plot import plot_alignments
from pose_dtw import band_edge_fraction, band_radius_from_seconds, check_stretching, dtw, pairwise_pose_cost, subsequence_dtw
from synthetic_dance import (FPS, Choreography, make_performance, mean_path_error, tau_from_knots,
                             to_normalized)

N_REF = 600                      # 20 s reference
BAND_S = 2.0                     # timing tolerance for the band
MAX_RUN = 8                      # slope constraint, frames (~0.27 s)


def build():
    choreo = Choreography(seed=0, motion=0.5, holds=((8.0, 4.0),))
    ref_xyz, ref_vis = make_performance(choreo, np.arange(N_REF) / FPS, noise=0.005, dropout=0.0, seed=10)
    ref = to_normalized(ref_xyz, ref_vis)

    # 1: full performance. Knots map choreography time -> learner time; hold 8-12 s lasts 2.5 s.
    tau1 = tau_from_knots([0, 0, 5, 8, 12, 16, 20, 20], [0, 0.5, 5.6, 9.0, 11.5, 15.9, 20.6, 21.1])
    # 2: learner skips choreography 3 to 7 s
    tau2 = np.concatenate([np.zeros(15), np.arange(0, 3, 1 / FPS), np.arange(7, 20, 1 / FPS), np.full(15, 20.0)])
    # 3b: recording starts 6 s early (idle), then the same performance as scenario 1
    tau3 = np.concatenate([np.zeros(6 * FPS), tau1])
    scen = {}
    for name, tau, seed in (("full", tau1, 11), ("skip", tau2, 12), ("early", tau3, 13)):
        xyz, vis = make_performance(choreo, tau, noise=0.03, dropout=0.03, seed=seed)
        scen[name] = (to_normalized(xyz, vis), tau * FPS)
    return ref, scen


def summarize(label, res, rep, tau_frames, ref_offset=0.0, secs=None, edge=None):
    err = mean_path_error(res.path, tau_frames, ref_offset)
    t = "" if secs is None else f"  {secs:5.2f}s"
    e = "" if edge is None else f"  band-edge {edge:4.0%}"
    print(f"  {label:34s} cost {res.normalized_cost:.3f}  path err {err:6.2f} f  "
          f"runs H{rep.longest_horizontal_run:3d} V{rep.longest_vertical_run:3d}  "
          f"{'FLAGGED' if rep.excessive else 'ok     '}{e}{t}")


def run(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ref, scen = build()
    radius = band_radius_from_seconds(BAND_S, FPS)

    def go(cost, label, rows, tau_frames, ref_offset=0.0, sub=False, **kw):
        t0 = time.time()
        res = subsequence_dtw(cost, **kw) if sub else dtw(cost, **kw)
        secs = time.time() - t0
        rep = check_stretching(res.path, n_ref=cost.shape[0], fps=FPS)
        edge = None if kw.get("band") is None else band_edge_fraction(res.path, *cost.shape, kw["band"])
        summarize(label, res, rep, tau_frames, ref_offset, secs, edge)
        rows.append(dict(title=label, cost=cost, result=res, report=rep, band=kw.get("band")))

    # ---- 1: complete performance
    q, tf = scen["full"]
    cost, _ = pairwise_pose_cost(ref, q)
    print("Scenario 1: complete performance (hold shortened, tempo drift)")
    rows = []
    go(cost, "unconstrained", rows, tf)
    go(cost, f"Sakoe-Chiba band {BAND_S:g} s", rows, tf, band=radius)
    go(cost, f"band + max_run {MAX_RUN}", rows, tf, band=radius, max_run=MAX_RUN)
    plot_alignments(rows, FPS, "Scenario 1: complete performance with a held pose").savefig(
        out_dir / "dtw_1_full_performance.png", dpi=110)

    # ---- 2: skipped segment
    q, tf = scen["skip"]
    cost2, _ = pairwise_pose_cost(ref, q)
    print("Scenario 2: learner skips 4 s")
    rows = []
    go(cost2, "unconstrained", rows, tf)
    go(cost2, f"band {BAND_S:g} s + max_run {MAX_RUN}", rows, tf, band=radius, max_run=MAX_RUN)
    plot_alignments(rows, FPS, "Scenario 2: learner skips 4 s of the dance").savefig(
        out_dir / "dtw_2_skipped_segment.png", dpi=110)

    # ---- 3: subsequence
    rows = []
    print("Scenario 3a: reference = 8 s segment (6 to 14 s) inside a longer recording")
    q, tf = scen["full"]
    s0, s1 = 6 * FPS, 14 * FPS
    cost3, _ = pairwise_pose_cost(ref[s0:s1], q)
    go(cost3, "subsequence DTW (segment)", rows, tf, ref_offset=s0, sub=True, max_run=MAX_RUN)
    true_start, true_end = int(np.argmin(np.abs(tf - s0))), int(np.argmin(np.abs(tf - s1)))
    print("    flags:", rows[-1]["report"].flags)
    r = rows[-1]["result"]
    print(f"    recovered query frames {r.query_start}..{r.query_end}, truth {true_start}..{true_end}")
    print("Scenario 3b: full dance, recording starts 6 s early")
    q, tf = scen["early"]
    cost4, _ = pairwise_pose_cost(ref, q)
    go(cost4, "subsequence DTW (early start)", rows, tf, sub=True, max_run=MAX_RUN)
    go(cost4, f"fixed-start band {BAND_S:g} s (wrong tool)", rows, tf, band=radius, max_run=MAX_RUN)
    r = rows[1]["result"]
    first_move = int(np.flatnonzero(tf > 0)[0]) - 1      # idle frames match equally well, so truth = last idle frame
    print(f"    subsequence started at query frame {r.query_start}, performance begins at {first_move}")
    plot_alignments(rows, FPS, "Scenario 3: subsequence DTW").savefig(out_dir / "dtw_3_subsequence.png", dpi=110)


if __name__ == "__main__":
    run(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("."))