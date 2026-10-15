"""Compare MoveMatch's feedback with a dance teacher's (or experienced dancer's) judgment.

1. Record attempts of a section (correct, deliberately wrong, deliberately late, partly out of frame).
2. Export the moments MoveMatch judged:
       python evaluation/compare_with_teacher.py export --out evaluation/teacher_review.csv
3. Give the CSV to the teacher together with the replay pages. Without looking at MoveMatch's
   columns (hide them), they fill in teacher_status (close / minor / review / cannot_tell),
   teacher_timing (early / on_time / late / cannot_tell) and notes.
4. Report agreement and every disagreement:
       python evaluation/compare_with_teacher.py report --csv evaluation/teacher_review.csv \
              --out docs/evaluation_teacher_results.md

The point is to find where the app and the teacher disagree, not to produce a single agreement number.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIELDS = ["attempt_id", "section", "speed", "moment", "reference_time_s", "learner_time_s",
          "app_status", "app_issues", "app_timing", "app_offset_s",
          "teacher_status", "teacher_timing", "notes"]


def export(out: Path) -> None:
    from app import create_app
    from models import Attempt
    app = create_app()
    rows = []
    with app.app_context():
        for attempt in Attempt.query.order_by(Attempt.id).all():
            ev = attempt.evaluation.details_json if attempt.evaluation else None
            if not ev or ev.get("withheld"):
                continue
            timing = ev.get("timing", {})
            for m in ev.get("moments", []):
                rows.append({
                    "attempt_id": attempt.id, "section": f"{attempt.section_start_s}-{attempt.section_end_s}",
                    "speed": attempt.speed, "moment": m["label"], "reference_time_s": m["reference_time_s"],
                    "learner_time_s": m.get("learner_time_s"), "app_status": m["status"],
                    "app_issues": " | ".join(m.get("issues", [])), "app_timing": timing.get("direction") or "unmeasured",
                    "app_offset_s": timing.get("offset_s"), "teacher_status": "", "teacher_timing": "", "notes": "",
                })
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} moments to {out}")


def _flagged(status: str) -> bool | None:
    status = (status or "").strip().lower()
    if status in ("review",):
        return True
    if status in ("close", "minor"):
        return False
    return None                                    # cannot_tell / not_visible / not_covered / blank


def report(csv_path: Path, out: Path | None) -> None:
    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("teacher_status") or "").strip()]
    if not rows:
        sys.exit("No teacher judgments in the CSV yet (teacher_status is empty everywhere).")
    pairs = Counter()
    disagreements = []
    for r in rows:
        app, teacher = _flagged(r["app_status"]), _flagged(r["teacher_status"])
        if app is None or teacher is None:
            pairs["not comparable"] += 1
            continue
        key = {(True, True): "both flag", (False, False): "both fine",
               (True, False): "app flags, teacher does not", (False, True): "teacher flags, app does not"}[(app, teacher)]
        pairs[key] += 1
        if app != teacher:
            disagreements.append((key, r))
    timing_rows = {}
    for r in rows:                                 # one timing judgment per attempt
        timing_rows.setdefault(r["attempt_id"], r)
    timing_agree = sum(1 for r in timing_rows.values()
                       if r["teacher_timing"].strip() and r["teacher_timing"].strip() == r["app_timing"].strip())
    lines = ["# MoveMatch feedback vs. teacher judgment", "",
             f"{len(rows)} judged moments from {len(timing_rows)} attempts (`{csv_path}`).", "",
             "| Outcome | Moments |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in pairs.most_common()]
    lines += ["", f"Timing direction agreed on {timing_agree} of {len(timing_rows)} attempts.", "",
              "## Every disagreement", ""]
    for key, r in disagreements:
        lines.append(f"* Attempt {r['attempt_id']} at {r['moment']} ({key}). App: {r['app_status']} "
                     f"({r['app_issues'] or 'no issue'}). Teacher: {r['teacher_status']}. Notes: {r['notes'] or '–'}")
    if not disagreements:
        lines.append("None.")
    text = "\n".join(lines) + "\n"
    if out:
        out.write_text(text)
        print(f"Report written to {out}")
    else:
        print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", type=Path, default=ROOT / "evaluation" / "teacher_review.csv")
    r = sub.add_parser("report")
    r.add_argument("--csv", type=Path, required=True)
    r.add_argument("--out", type=Path)
    args = parser.parse_args()
    export(args.out) if args.cmd == "export" else report(args.csv, args.out)


if __name__ == "__main__":
    main()
