# Milestones

Status as of 2026-10-14. Milestones 1–3 are done; 4 was replaced by explainable feedback; 5 is done; 6 is planned.

# Milestone 1 — Landmark extraction
Extract and cache timestamped MediaPipe landmarks for the reference and learner videos. Validate frame ordering, confidence values, and coordinate conventions.

# Milestone 2 — Pose comparison
Implement normalization, joint confidence weighting, and per-frame pose distance. Test on identical and deliberately different poses.

# Milestone 3 — DTW alignment
Implement and test constrained DTW. Visualize the alignment path and check for excessive temporal stretching.

# Milestone 4 — Scoring (replaced)
Convert distances to calibrated pose scores, calculate timing and coverage, and combine the metrics.

*Outcome:* calibration showed a single score could not meet basic requirements (research/calibrate_pose_k.py). Replaced by explainable feedback: tracking, timing offset, key-moment joint angles (feedback.py).

# Milestone 5 — Flask integration
Save attempts, queue evaluation, expose status/results endpoints, and persist historical scores.

# Milestone 6 — Live feedback
Reuse pose preprocessing and local distance functions for rolling feedback; compare live results with post-session evaluation.