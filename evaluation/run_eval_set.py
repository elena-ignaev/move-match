"""Run the feedback method on the synthetic evaluation set and report agreements AND disagreements.

    python evaluation/run_eval_set.py                 # prints a table
    python evaluation/run_eval_set.py --report docs/evaluation_synthetic_results.md

Each case states what correct feedback must say (cases.py, `expect`). A case "passes" only if every
expectation holds. Failures are listed with the feedback that was actually produced: they are the
point of the exercise, not something to hide.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cases import scenario_cases  # noqa: E402

import feedback as fb  # noqa: E402


def check(case, result) -> list[str]:
    """Expectation failures for one case (empty list = pass)."""
    e, problems = case.expect, []
    if e.get("withheld") is not None and result["withheld"] != e["withheld"]:
        problems.append(f"withheld={result['withheld']}, expected {e['withheld']}")
    if result["withheld"]:
        reasons = " ".join(result.get("withheld_reasons", [])).lower()
        if "reason_contains" in e and e["reason_contains"] not in reasons:
            problems.append(f"reason does not mention '{e['reason_contains']}'")
        return problems
    timing = result.get("timing", {})
    posture = result.get("posture", {})
    moments = result.get("moments", [])
    if "timing" in e and timing.get("direction") != e["timing"]:
        problems.append(f"timing {timing.get('direction')} ({timing.get('offset_s')}), expected {e['timing']}")
    if "offset" in e and (timing.get("offset_s") is None or abs(timing["offset_s"] - e["offset"]) > 0.07):
        problems.append(f"offset {timing.get('offset_s')} s, expected {e['offset']} ± 0.07 s")
    if "max_review" in e and posture.get("review", 0) > e["max_review"]:
        flagged = [f"{m['label']}: {'; '.join(m['issues'])}" for m in moments if m["status"] == "review"]
        problems.append(f"{posture.get('review')} moments flagged, expected at most {e['max_review']} ({' | '.join(flagged)})")
    if "min_review" in e and posture.get("review", 0) < e["min_review"]:
        problems.append(f"{posture.get('review', 0)} moments flagged, expected at least {e['min_review']}")
    if "review_joint" in e:
        lo, hi = e.get("review_window", (0, 1e9))
        hits = [m for m in moments if lo <= m["reference_time_s"] <= hi and any(
            j["joint"] == e["review_joint"] and j["status"] in ("review", "minor") for j in m["joints"])]
        if not hits:
            problems.append(f"{e['review_joint']} never flagged between {lo} and {hi} s")
        outside = [m["label"] for m in moments if not (lo - 0.3 <= m["reference_time_s"] <= hi + 0.3) and any(
            j["joint"] == e["review_joint"] and j["status"] == "review" for j in m["joints"])]
        if outside:
            problems.append(f"{e['review_joint']} also flagged outside the error window at {outside}")
    if "excluded_contains" in e and e["excluded_contains"] not in result.get("excluded_joints", []):
        problems.append(f"{e['excluded_contains']} was not excluded")
    if "not_excluded" in e and e["not_excluded"] in result.get("excluded_joints", []):
        problems.append(f"{e['not_excluded']} was excluded although it was visible")
    if "tracking_message" in e and not any(e["tracking_message"] in m for m in result["tracking"]["messages"]):
        problems.append(f"no tracking message mentioning '{e['tracking_message']}'")
    if "facing_review_after" in e:
        after = [m for m in moments if m["reference_time_s"] >= e["facing_review_after"]]
        if not after or not any("turned" in " ".join(m["issues"]) for m in after):
            problems.append("missed turn was not flagged")
    return problems


def run() -> list[dict]:
    rows = []
    for case in scenario_cases():
        result = fb.evaluate_attempt(case.ref, case.learner, case.mapped, section=case.section,
                                     speed=case.speed, follow_mode=case.follow_mode)
        problems = check(case, result)
        timing = result.get("timing", {})
        posture = result.get("posture", {})
        rows.append({
            "case": case.name, "description": case.description, "pass": not problems, "problems": problems,
            "known_limitation": bool(case.expect.get("known_limitation")),
            "headline": result.get("headline", ""),
            "offset": timing.get("offset_s"), "direction": timing.get("direction"),
            "review": posture.get("review"), "close": posture.get("close"), "moments": posture.get("moments_total"),
            "median": posture.get("median_abs_diff_deg"), "withheld": result["withheld"],
            "score": (result.get("score") or {}).get("value"),
        })
    return rows


def to_markdown(rows) -> str:
    passed = sum(r["pass"] for r in rows)
    lines = [
        "# Synthetic evaluation set: results",
        "",
        f"Generated by `python evaluation/run_eval_set.py --report ...` on {date.today().isoformat()} "
        f"(feedback algorithm {fb.ALGORITHM_VERSION}). **{passed} of {len(rows)} cases behave as expected.**",
        "",
        "These are synthetic landmark sequences with known ground truth (see `evaluation/cases.py`). They test "
        "that the method reacts to the right things; they say nothing yet about agreement with a dance teacher "
        "on real recordings (see `docs/evaluation.md`).",
        "",
        "| Case | What was changed | Feedback headline | Score | Timing | Moments review/close/total | Median diff | Result |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        timing = "withheld" if r["withheld"] else f"{r['direction']} ({r['offset']:+.2f} s)" if r["offset"] is not None else str(r["direction"])
        moments = "–" if r["withheld"] else f"{r['review']}/{r['close']}/{r['moments']}"
        median = "–" if r["median"] is None else f"{r['median']:.0f}°"
        result = ("pass" if r["pass"] else ("known limitation: " if r["known_limitation"] else "**FAIL**: ")
                  + "; ".join(r["problems"]))
        score = "–" if r["score"] is None else str(r["score"])
        lines.append(f"| `{r['case']}` | {r['description']} | {r['headline']} | {score} | {timing} | {moments} | {median} | {result} |")
    failures = [r for r in rows if not r["pass"]]
    lines += ["", "## Disagreements with the expected feedback", ""]
    if failures:
        for r in failures:
            tag = " (known limitation of joint-angle comparison)" if r["known_limitation"] else ""
            lines.append(f"* `{r['case']}`{tag}: " + "; ".join(r["problems"]))
    else:
        lines.append("None on this synthetic set. That is expected, because the cases were designed with the method "
                     "in mind; see docs/evaluation.md for the known limitations that synthetic data cannot reveal.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, help="write a Markdown report to this path")
    args = parser.parse_args()
    rows = run()
    width = max(len(r["case"]) for r in rows)
    for r in rows:
        status = "PASS" if r["pass"] else "MISS" if r["known_limitation"] else "FAIL"
        score = "  –" if r["score"] is None else f"{r['score']:>3}"
        print(f"{status}  {r['case']:<{width}}  {score}  {r['headline']}")
        for p in r["problems"]:
            print(f"      - {p}")
    print(f"\n{sum(r['pass'] for r in rows)}/{len(rows)} cases behave as expected")
    if args.report:
        args.report.write_text(to_markdown(rows))
        print(f"Report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
