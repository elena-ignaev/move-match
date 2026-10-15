# User testing log

**Status: no sessions with other learners recorded yet.** This file is the plan and the place to log
findings. Only real observations belong here.

## Plan

* **Who:** 4–6 recreational dancers preparing for a performance (e.g. members of a school dance group),
  including at least one beginner and one person who has never seen the app.
* **Task (no help from the observer):** "Learn the 6-second section starting at 0:10 well enough to record a
  take, then improve it." That covers the whole journey: choose → practise → record → replay → feedback →
  retry.
* **Observe:** where they hesitate, what they click first, whether they understand mirror view and "dance
  side", whether they can say in their own words what the feedback told them to change, and whether the
  second take improved on what was flagged.
* **Ask afterwards:** What was confusing? Which note was the most useful? Did any note feel wrong? Would you
  trust it without watching the replay? What would make you use it again?
* **Measure:** time to the first recording; number of retries; how many flagged moments improved in the
  retake (shown in "Retry and compare"); anything that broke.
* **Consent:** recordings stay on the test computer and are deleted after the session unless the person agrees
  in writing to keep them (needed for the teacher comparison in `docs/evaluation.md`).

## Design questions to test

| Question | Current design | What would change it |
|---|---|---|
| Is "Mirror view" understood? | On by default; explained under the controls | people raising the wrong arm in recordings |
| Is a 5 s countdown enough to get into position? | 5 s default, 3/10 s options | people missing the start |
| Is "worth reviewing" too harsh or too soft? | flags > 35° differences confirmed in 2-D | disagreement with the teacher comparison |
| Is the timing curve understood? | small chart under the timing message | people ignoring or misreading it |
| Do people trust "feedback withheld"? | explains the tracking problem and how to fix it | people retrying without changing the camera |

## Session log

<!-- Copy for each session:
### Session N — date, participant (anonymous label), dance experience
* What happened:
* Confusing:
* Useful:
* Changed afterwards (commit/PR):
-->

(no sessions yet)

## Changes made from evidence so far

These came from the automated evaluation set and from testing the app myself, not from other users:

* False "knee bent" flags from depth error → flags now need 3-D and 2-D agreement (`docs/evaluation.md`).
* The camera-timeout message said "video took too long to load" → it now names the camera and offers "Try again".
* Section loops overshot the end when the browser tab was in the background → `timeupdate` now enforces the end.
* Moments after an early stop were compared anyway → shown as "not recorded".
