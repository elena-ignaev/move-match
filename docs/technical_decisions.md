# Technical decisions

Why MoveMatch works the way it does, including what was tried and changed. Each section names the
code that implements it.

## 1. A pretrained pose detector, not a trained model

MoveMatch uses Google's **MediaPipe Pose Landmarker (lite)**, a pretrained model that returns 33 body
landmarks per frame, in two forms: normalized image coordinates (where the joint is in the picture) and
"world" coordinates (metres, centred on the hips, with an estimated depth). MoveMatch does not train or
fine-tune any model. I tried HybrIK (a 3-D body-mesh model) first, but could not install its dependencies on
macOS, and it is far heavier than this problem needs.

The original work in this project is everything after detection: normalization, synchronization, the
comparison method, handling of missing data, the evaluation set, and the reliability work.

## 2. Normalization: make different bodies comparable (`pose_utils.py`, `pose_pipeline.py`)

A learner and a reference dancer differ in height, limb length, distance from the camera and position in
the room. Comparing raw coordinates would mostly measure those differences.

* **Translation:** every pose is re-centred on the hip midpoint.
* **Scale:** divided by torso length (hip midpoint to shoulder midpoint). Torso length shrinks whenever the
  dancer bends, so dividing each frame by its own torso length would inject fake motion. Instead one smooth
  body scale is estimated per recording: a rolling *high* percentile (75th) of torso lengths over 3 seconds,
  because bending only ever shortens the torso. `research/scale_benchmark.py` compares this with simpler
  estimators on synthetic recordings containing bends, noise and dropouts.
* **Orientation is deliberately not normalized.** Rotating every pose to face the camera would remove camera
  angle differences, but it would also erase turns, which are part of the choreography. MoveMatch keeps
  facing direction and checks it separately (section 6).

The feedback method itself compares **joint angles**, which are already independent of position, scale and
body size. The normalized coordinates are still used by the research DTW scorer and the developer tools.

## 3. Synchronization: which reference moment was the learner copying? (`sync_timeline.py`, `static/js/ghost_mode.js`)

The prototype assumed learner frame *N* matches reference frame *N*. That is wrong as soon as anything is
irregular:

* practice runs at 0.25–0.75× speed, so 1 s of recording covers 0.25–0.75 s of the reference;
* the reference can buffer for half a second while the learner keeps dancing;
* the recorder starts a moment before the music;
* browser recordings have a **variable frame rate**, so frame *N* is not at *N* / fps seconds.

**What the browser records.** While recording, every reference frame shown on screen is logged as
`[session time, reference time]` on one monotonic clock (`performance.now()`), using
`requestVideoFrameCallback`'s `expectedDisplayTime` (when that frame actually reaches the screen). The
MediaRecorder's own `start` event time, play/pause/buffering events and the playback rate are logged too.
When frame callbacks stop (background tab), `timeupdate` fills in about 4 times per second.

**What the server does.** A learner frame at recording time *t* happened at session time
`recorder_start + t`. It is mapped to a reference time by interpolating between logged samples of the
*same continuous playing stretch*. The log is cut wherever there is a gap (stall), a backwards jump (seek
or loop) or a frozen reference. Learner frames that fall in a cut, before playback or after it stopped get
**no** reference time and are never compared. Attempts saved by the prototype only have play/pause events;
for those, each playing segment is reconstructed at constant rate.

**Constant frame rate everywhere (`media.py`).** Every reference upload is transcoded once to H.264 at a
constant frame rate (≤ 30 fps), with its original audio, and pose detection runs on *that* file, the one the
browser plays. Then "frame index / fps" in the landmark file equals the browser's `currentTime` exactly.
Learner recordings are transcoded the same way, which also makes MediaRecorder WebM files seekable for
replay. Variable frame rate is detected by measuring timestamp spacing, because header fields can claim a
constant rate for files that are not (Matroska does).

*Evidence:* the prototype's real upload was VFR (average 29.98 vs declared 30 fps). On real pose data, a take
offset by 0.4 s was measured at 0.401 s, and a take at 0.5× speed was measured on time (−0.03 s).

## 4. One playback timeline, and the music (`ghost_mode.js`, `media.py`)

The prototype played two `<video>` elements (reference, plus a semi-transparent copy as the "ghost") that
only started together and could drift after buffering. Both were muted, and the annotated video had no audio
track because the conversion used `-an`. Music is central to dance practice.

Now there is **one** reference `<video>` with its music. The ghost skeleton is drawn on a transparent
`<canvas>` from cached landmarks at that video's `currentTime` on every animation frame, so it cannot drift.
It has an opacity control, and the reference's background no longer covers the camera image. Slowed playback
keeps the pitch (`preservesPitch`). The skeleton is fitted into the full-body guide box using the dancer's
bounding box over the selected section, so travel across the floor is preserved.

In replay, the learner's recording is the master clock. The reference follows it through the same mapping,
correcting small drift by nudging `playbackRate` and large drift by seeking (measured: drift converges from
70 ms to under 10 ms).

## 5. Mirror mode: one explicit convention (`pose_pipeline.py`, `models.Attempt`)

Previously only the camera preview was flipped (CSS), the reference was not, and the recording was raw, so
what the learner saw, what was saved and what was compared all used different conventions. Two separate
decisions are now stored with every attempt:

* `view_mirrored`: a **display** preference. When on, the camera preview *and* the reference are flipped, so
  the dancer's right hand appears on the right of the screen, like a studio mirror. Copying the same side of
  the screen then moves the correct side of the body, and the ghost overlay lines up with the learner. Replay
  uses the same setting.
* `follow_mode`: a **dance** decision: "same side as the dancer" (default) or "mirror image".
* `capture_mirrored`: a fact about the data. Recordings are never flipped, so it is always `False`.

The learner's landmarks are flipped for comparison only when exactly one of `capture_mirrored` and
"follow mode is mirror image" is true (`pose_pipeline.mirror_needed`). Feedback always names the
learner's **own** body side: in mirror-image mode the reference's right elbow is "your left elbow".

## 6. Feedback: narrow and explainable, not a percentage (`feedback.py`)

An earlier plan combined pose, timing and coverage into one 0–100 score (`research/pose_score.py`, kept as
research). Its own calibration showed the problem: on synthetic dancers no constant could make good dancers
score ≥ 90, poor ones ≤ 30, *and* neighbouring skill levels differ by 8 points. Above all, there is no evidence
that "73%" would mean the same thing to a learner as to a teacher. So each attempt gets three separate
answers:

1. **Could we see you?** For each body region (torso, each arm, each leg), the share of frames in which its
   joints were detected, confident, *and inside the camera frame*. If nobody was found or the torso was hidden,
   feedback is **withheld** with a specific reason ("your legs were below the bottom edge of the frame in 70%
   of the recording: step back or tilt the camera down"). If only one limb was hidden, only that limb's
   joints are left out.
2. **Were you early or late?** The best constant delay between −0.6 s and +1.0 s at which the learner's joint
   angles match the reference. Within ±0.15 s counts as on time. If the mismatch-vs-delay curve has no clear
   minimum, timing is reported as "could not be measured" instead of guessed. A windowed version detects
   drift ("on time at 0:02, 0.5 s late by 0:08"). The curve is shown on the feedback page.
3. **Did the shape match at the moments that matter?** Key moments are chosen from the **reference alone**:
   instants where its joints are momentarily still (hits and holds), spread through the section. Every
   attempt at the same section is therefore judged at the same moments, which makes moment-by-moment
   comparison between attempts possible. At each moment, elbow, shoulder, hip and knee angles are compared
   *after removing the measured delay*, allowing ±0.2 s of music for jitter. A joint within 20° is "close";
   more than 35° is "worth reviewing". Each flag is phrased as a direction ("your right elbow was straighter
   by about 40°").

**Turns.** Joint angles cannot see facing direction, so facing is estimated separately at each moment and a
difference above 60° is flagged ("the reference is turned about 90° here; you were facing the camera").

**Iteration driven by the evaluation set.** The first version flagged correct dancing whenever simulated
depth error drifted by about 6 cm ("right knee 47° more bent"). One camera only *estimates* depth. Since 1.1,
a joint is flagged only when the 3-D angle **and** the 2-D picture angle both differ, in the same direction.
A 3-D-only difference is shown as a small difference with a note that it may be a depth error. The cost:
mistakes that happen purely in depth are less likely to be flagged. A second fix came from the end-to-end
test: a recording that stopped early was being compared at moments it never reached, so such moments are now
"not recorded" instead of judged.

## 7. Missing landmarks

* Joints below 0.5 confidence, outside the frame, or with invalid confidence values (NaN, negative) count as
  missing, never as a mistake.
* Missing data lowers the *amount* of feedback, never the result: excluded joints are listed, and withheld
  feedback says why.
* Short gaps are not filled with interpolated poses that then count as observations.

## 8. Limitations of a single camera

* **Depth is guessed.** MediaPipe's world z is an estimate. Reaching forward vs. sideways changes no joint
  angle and little in the picture, so it is a known miss (`arm_forward_not_sideways` in the evaluation set).
* **Facing away.** When the dancer turns their back, MediaPipe can swap left and right labels. Facing estimates
  and left/right feedback are unreliable during back-facing parts.
* **Occlusion.** An arm behind the body is invisible. MoveMatch leaves it out rather than guess.
* **Symmetric choreography hides mirror mistakes:** in a symmetric pose, dancing the mirror image looks
  correct. On the real test clip a mirrored take was flagged at only 1 of 4 moments.
* **Scope.** One dancer, a fixed camera, the whole body visible, short sections (4–10 s recommended, clips ≤ 90 s).

## 9. Reliability (`jobs.py`, `media.py`, `app.py`)

* Uploads only save and inspect the file; transcoding and pose detection run in a background job with a
  progress bar. Interrupted jobs resume on restart, and failed jobs can be retried from the page.
* Uploads are validated by their **contents** (ffprobe: is there a video stream, how long, how large), not
  their extension. Size, duration (≤ 90 s) and resolution (≤ 1920 px) are limited. Rejected files are deleted
  immediately, and failed jobs remove partial outputs. The raw browser recording is deleted once its replay
  copy exists. The silent intermediate annotated video no longer exists.
* Recording controls are a state machine (idle → preparing → ready → countdown → starting → recording →
  saving → saved / save failed). Buttons are disabled *before* any asynchronous work, so double clicks cannot
  overlap. Any failure stops the recorder and camera and offers a recovery action.
* The recording stays in the page until the server confirms it. Uploads carry a client-generated id, so a
  retry after a lost response never creates a duplicate. "Download recording" is always available.
* Media loads have timeouts and error handlers with a clear recovery action.
* Recording uploads always get JSON errors, including for oversized files.
* The database schema is owned by Alembic alone. The first migration used to *alter* a table that only
  `create_all()` had made; a real baseline now creates the tables, and the old revision id was kept, so
  existing databases can be stamped and upgraded. Migrations run with SQLite foreign-key enforcement
  temporarily off (required for SQLite's table-rebuild migrations once rows exist; a test covers this).
* Feedback is stored per **attempt** (one evaluation row per attempt), with the algorithm version and
  settings, so old results stay interpretable after the method changes.

## 10. Protecting recordings (`security.py`, `config.py`)

Local mode is single-user. Public-demo mode adds:

* anonymous per-browser ownership (a random id in the signed session cookie); other visitors' items return 404;
* prepared sample videos readable by everyone but not deletable; visitor uploads off by default;
* CSRF protection on every form and upload (Flask-WTF), per-visitor upload rate limits, automatic deletion
  after 24 hours, and delete buttons for videos and attempts;
* a refusal to start without a strong `SECRET_KEY` or with debug on; secure, HTTP-only, SameSite cookies;
  a strict Content-Security-Policy (no inline scripts);
* detailed processing errors only in the server log; visitors see a safe message;
* a privacy page explaining what is uploaded, where it is stored and when it is deleted.

Not covered (would be needed for a real public service): accounts, a shared rate-limit store for multiple
processes, malware scanning, and an operator process for data requests.
