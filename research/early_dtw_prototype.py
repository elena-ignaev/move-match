"""Earliest DTW prototype (superseded by pose_dtw.py). Kept for the project history only."""


import numpy as np

# Sakoe-Chiba DTW. Assume two sequences are already sampled at comparable rates and that their starts are approximately aligned
def pose_distance(reference_pose, learner_pose):
    """
    Mean Euclidean distance across corresponding joints.

    Both inputs have shape (J, 3).
    Assumes coordinates have already been normalized.
    """
    joint_distances = np.linalg.norm(
        reference_pose - learner_pose,
        axis=1
    )
    return float(np.mean(joint_distances))


def dtw(reference, learner, window_ratio=0.15):
    """
    Constrained DTW for sequences of shape (N, J, 3)
    and (M, J, 3).

    Returns:
        mean_cost: average local cost along the path
        path: list of (reference_index, learner_index)

    Assumptions:
        - Inputs have been normalized consistently.
        - Sequence starts are approximately aligned.
        - A monotonic alignment is appropriate.
    """
    n = len(reference)
    m = len(learner)

    if n == 0 or m == 0:
        raise ValueError("Pose sequences cannot be empty.")

    if n == 1 or m == 1:
        # This simple version is intended for sequences
        # containing multiple frames.
        raise ValueError("Sequences need at least two frames.")

    # The window must allow the path to reach both endpoints.
    window = max(
        abs(n - m),
        int(max(n, m) * window_ratio)
    )

    # Accumulated costs, with an extra row and column.
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    # Track the predecessor for backtracking.
    parent = np.full((n, m), -1, dtype=np.int8)

    for i in range(n):
        # Convert the reference index into the learner
        # index range allowed by the warping band.
        center = i * (m - 1) / max(1, n - 1)

        j_start = max(0, int(np.floor(center - window)))
        j_end = min(m, int(np.ceil(center + window)) + 1)

        for j in range(j_start, j_end):
            local = pose_distance(reference[i], learner[j])

            candidates = (
                cost[i, j],       # diagonal predecessor
                cost[i, j + 1],   # reference advances
                cost[i + 1, j],   # learner advances
            )

            best = int(np.argmin(candidates))
            cost[i + 1, j + 1] = local + candidates[best]
            parent[i, j] = best

    if not np.isfinite(cost[n, m]):
        raise ValueError(
            "No valid DTW path. Increase the warping window."
        )

    # Backtrack from the final aligned frame pair.
    i, j = n - 1, m - 1
    path = []

    while True:
        path.append((i, j))

        if i == 0 and j == 0:
            break

        direction = parent[i, j]

        if direction == 0:
            i -= 1
            j -= 1
        elif direction == 1:
            i -= 1
        elif direction == 2:
            j -= 1
        else:
            raise RuntimeError("Invalid DTW backtracking path.")

    path.reverse()

    mean_cost = cost[n, m] / len(path)

    return float(mean_cost), path
