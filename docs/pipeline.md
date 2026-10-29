> **Historical planning notes.** Written before implementation. What was actually built differs in places, especially the score: the weighted 0–100 score was rebuilt from interpretable measurements (see technical_decisions.md, 6a). See `docs/technical_decisions.md` for the current design.

1. Overall architecture
Reference dance (Annotated video + MediaPipe landmark sequence)
--> Learner performance (Recorded video/Live camera frames --> MediaPipe landmarks)
--> Preprocessing (Validate/Normalize coordinates/Filter low-confidence landmarks)
--> Temporal alignment (DTW aligns corresponding movements)
--> Scoring (Pose similarity/Timing/Movement coverage/Confidence)
--> Evaluation resutl (Overall scores/Feedback/Results history)

2. Define metrics of evaluation
* Pose similarity (70%)
* Timing and synchronization (20%)
* Movement coverage (10%)
Note: Differentiate between full-performance scoring and specific coaching feedback

3. Create a reusable landmark format from reference video

4. Normalize the body pose
* Centering: Set the hip-center as the origin
* Scaling: Fundamentally, scale on torso length. To deal with leaning, smoothed body-scale estimate over several frames is used
* Joint angles as secondary representation:
    * Left and right elbow angles
    * Left and right knee angles
    * Shoulder-to-arm angles
    * Hip angles
    * Trunk inclination

5. Handle low-confidence landmarks
* Exclude/Down weight unreliable joints
* Avoid filling large missing sections with interpolated poses and treating them as genuine observations
* Flag frames where too few joints are reliable
* Reject or request another recording if most of the learner's body is consistently out of view
* Report confidence separately from similarity (Low confidence is not low score)

6. Constrained DTW

7. Convert DTW into meaningful scores
* Pose score
* Timing score
* Movement coverage
* Overall score: Once each component is stable, combine them:
S_overall = 0.7*S_pose + 0.2*S_timing + 0.1*S_coverage

8. Support different practice speeds correctly
When the practice session starts, record a session timeline with events such as:
* Reference playback started
* Playback paused
* Playback resumed
* Reference seeked
* Playback rate changed
* Recording started and stopped
--> Use these events to reconstruct which reference time corresponds to each learner frame

Separate two concepts:
* Pose similarity: Did the learner perform the right movement? --> DTW is useful here
* Execution timing: Did the learner perform it at the intended speed and time? --> Timing metrics are needed here

In live pose tracking, capture monotomic timestamp for every frame. --> To associate the learner's pose with the correct reference position even if camera frames arrive irregularly.

9. Integrate live pose-tracking
Do not wait until the user finishes the dance to compute every metric. Reuse the same pose normalization and distance functions, but maintain a rolling evaluation state
Mode A: Live coaching
1. Obtain current monotomic timestamp
2. Run MediaPipe pose landmarker
3. Normalize the learner's landmarks.
4. Determine the current reference timestamp from the playback timeline.
5. Retrieve or interpolate the reference pose at that timestamp
6. Calculate a local pose mismatch
7. Update rolling feedback over a short window
--> Gives user immediate feedback
Do not run full DTW over the entire dance on every camera frame. Instead, use direct timestamp correspondence and rolling metrics for immediate feedback

Mode B: Post-performance evaluation (Currently in this direction)
1. Extract or reuse the learner's pose sequence
2. Normalize the sequence
3. Align it to the reference using constrained DTW
4. Calculate pose, timing, and coverage scores
5. Save the final results
--> Simpler to debug and easier to compare results across different attempts

10. Hybrid approach: 
* Direct timestamp comparison for real-time coaching
* Constrained DTW for the final pose-similarity score
* A separate timing score that is not erased by temporal alignment
* Explicit coverage checks for missing sections

11. Store evaluation results and intermediate artifacts
* Extend the evaluation records to represent a complete attempt, not just a number (i think this is already done? - check again)
* No need to store every intermediate results in SQLite. Instead, store large arrays, detailed per-frame results, and alignment paths in files under `storage/evaluations`, then keep their filenames in the database
* For reproducibility, retain a metadata alongside each evaluation:
    * MediaPipe model/version and extraction settings
    * Landmark normalization method
    * DTW window and distance function
    * Score weights and calibration parameters
    * Reference and learning sampling rates
    * Algorithm version
    --> Helps explain why two attempts receive different scores and avoid silently changing the meaning of old results when the algorithm is improved

12. Implement Flask evaluation workflow
1. The learner finishes ghost mode
2. The frontend uploads the recording
3. Flask creates an Evaluation record with `status=recorded`
4. A background job extracts landmarks and compute scores
5. The job updates the record to `completed` or `failed`
6. The frontend polls the evaluation status
7. The results page retrieves and displays the saved scores

Following routes:
```
POST /videos/<video_id>/attempts
POST /attempts/<attempt_id>/evaluate
GET  /attempts/<attempt_id>/status
GET  /attempts/<attempt_id>/results
GET  /videos/<video_id>/evaluations
```

On first prototype, evaluation can be invoked synchronously on short recordings. But design the processing function so that it can later run in a background worker without depending on a Flask request object

Design the function as such: 
```
def evaluate_recording(
    reference_landmarks,
    learner_landmarks,
    reference_timestamps,
    learner_timestamps,
    session_timeline,
):
    # Normalize both sequences.
    # Calculate frame-level distances.
    # Align sequences using constrained DTW.
    # Calculate pose, timing, and coverage scores.
    # Return structured results.
    ...
```
It should not read HTTP requests, render templates, or write directly to the database
--> Make it straightforward to test with synthetic data and reuse for offline recordings or future live-session analysis