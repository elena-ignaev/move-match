// Step 3: practise with a ghost and record an attempt.
//
// Design notes
// * ONE playback timeline: a single reference <video> (with its music) drives everything. The ghost
//   skeleton is drawn on a transparent canvas from cached landmarks at reference.currentTime, so it
//   cannot drift from the music the way a second <video> element could after buffering.
// * Explicit states. Every button handler switches state synchronously BEFORE any await, so a
//   double click cannot start two recordings, and every failure path returns to a consistent state.
// * Synchronization: while recording, every presented reference frame is logged as
//   [session_ms, reference_time] on the performance.now() clock, together with the MediaRecorder
//   start time and playback events. The server maps each recorded frame to the reference time that
//   was on screen (sync_timeline.py), which handles slow speeds, buffering stalls and late starts.
// * The recording is the raw camera image (never flipped). "Mirror view" is a display preference
//   applied identically to the camera preview, the reference and the replay.
import { $, SkeletonTrack, TimeoutError, clamp, contentRect, csrfToken, drawGuide, drawSkeleton, fitCanvas,
    fmtTime, ghostMap, guideRect, seekTo, waitForMedia, withTimeout } from "./common.js";

const shell = document.querySelector(".ghost-shell");
const camera = $("cameraPreview");
const reference = $("referenceVideo");
const canvas = $("ghostCanvas");
const ui = {
    form: $("ghostSettings"), fieldset: $("settingsFieldset"), start: $("sectionStart"), end: $("sectionEnd"),
    mirror: $("mirrorView"), follow: $("followMode"), opacity: $("ghostOpacity"), countdownSel: $("countdownSeconds"),
    loop: $("loopPractice"), prepare: $("prepareCamera"), practice: $("practiceGhost"), record: $("recordGhost"),
    stop: $("stopGhost"), status: $("ghostStatus"), errorBox: $("errorBox"), errorText: $("errorText"),
    errorAction: $("errorAction"), countdown: $("countdown"), rec: $("recBadge"), cameraWrap: $("cameraWrap"),
    referenceWrap: $("referenceWrap"), volume: $("musicVolume"), savePanel: $("savePanel"), saveStatus: $("saveStatus"),
    saveProgress: $("saveProgress"), review: $("reviewLink"), retry: $("retryUpload"), download: $("downloadLink"),
    again: $("recordAgain"), discard: $("discardRecording"),
};

const S = {
    IDLE: "idle", PREPARING: "preparing", READY: "ready", PRACTICING: "practicing", COUNTDOWN: "countdown",
    STARTING: "starting", RECORDING: "recording", SAVING: "saving", SAVED: "saved", SAVE_FAILED: "save_failed",
    ERROR: "error",
};
const RECORDER_TYPES = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm", "video/mp4"];
const maxBytes = Number(shell.dataset.maxMb) * 1024 * 1024;

let state = S.IDLE;
let stream = null;
let recorder = null;
let chunks = [];
let discardRecording = false;
let track = null;
let runToken = 0;              // incremented to cancel an in-flight countdown/start
let sessionOrigin = 0;
let sync = null;
let pending = null;            // {blob, uploadId, settings, url} kept until the server confirms
let audioCtx = null;
let sampling = false;

// ---------------------------------------------------------------- state
function setState(next, message) {
    state = next;
    const busy = [S.PREPARING, S.COUNTDOWN, S.STARTING, S.RECORDING, S.SAVING, S.PRACTICING].includes(next);
    ui.fieldset.disabled = busy;
    ui.prepare.disabled = ![S.IDLE, S.ERROR, S.SAVED, S.SAVE_FAILED].includes(next);
    ui.prepare.textContent = stream ? "Restart camera" : "Turn on camera";
    ui.practice.disabled = next !== S.READY;
    ui.record.disabled = next !== S.READY;
    ui.stop.disabled = ![S.PRACTICING, S.COUNTDOWN, S.STARTING, S.RECORDING].includes(next);
    ui.rec.hidden = next !== S.RECORDING;
    shell.dataset.state = next;
    if (message) ui.status.textContent = message;
    if (next !== S.ERROR) hideError();
}

function showError(message, action) {
    ui.errorText.textContent = message;
    ui.errorBox.hidden = false;
    ui.errorAction.hidden = !action;
    if (action) {
        ui.errorAction.textContent = action.label;
        ui.errorAction.onclick = action.run;
    }
}
function hideError() { ui.errorBox.hidden = true; }

function fail(message, action = { label: "Turn on camera again", run: () => prepareCamera() }) {
    setState(S.ERROR, "Stopped.");
    showError(message, action);
}

// ---------------------------------------------------------------- settings
function settings() {
    const d = Number.isFinite(reference.duration) ? reference.duration : Number(shell.dataset.duration);
    const start = clamp(Number(ui.start.value) || 0, 0, Math.max(0, d - 0.5));
    let end = clamp(Number(ui.end.value) || d, 0, d);
    if (end - start < 0.5) end = Math.min(d, start + 0.5);
    const checked = document.querySelector("input[name='ghostSpeed']:checked");
    return {
        start, end, speed: checked ? Number(checked.value) : 0.5,
        viewMirrored: ui.mirror.checked, followMode: ui.follow.value,
    };
}

function applyMirror() {
    const s = settings();
    ui.cameraWrap.classList.toggle("mirrored", s.viewMirrored);
    // The reference is flipped like the camera, plus once more if the learner dances the mirror image.
    ui.referenceWrap.classList.toggle("mirrored", s.viewMirrored !== (s.followMode === "mirror"));
}

function loadQuerySettings() {
    const q = new URLSearchParams(window.location.search);
    if (q.has("start")) ui.start.value = q.get("start");
    if (q.has("end")) ui.end.value = q.get("end");
    if (q.has("speed")) {
        const wanted = Number(q.get("speed"));
        const radio = [...document.querySelectorAll("input[name='ghostSpeed']")].find((r) => Number(r.value) === wanted);
        if (radio) radio.checked = true;
    }
    if (q.has("mirror")) ui.mirror.checked = q.get("mirror") === "1";
    if (q.get("follow") === "mirror" || q.get("follow") === "same_side") ui.follow.value = q.get("follow");
}

// ---------------------------------------------------------------- camera
function cameraErrorMessage(error) {
    switch (error && error.name) {
        case "NotAllowedError":
        case "SecurityError":
            return "Camera permission was denied. Allow camera access in your browser's address bar, then try again.";
        case "NotFoundError":
        case "OverconstrainedError":
            return "No camera was found. Connect a camera and try again.";
        case "NotReadableError":
            return "The camera is in use by another app (video call?). Close it and try again.";
        default:
            return (error && error.message) || "The camera could not be started.";
    }
}

function stopCamera() {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null;
    camera.srcObject = null;
}

async function prepareCamera() {
    if (![S.IDLE, S.ERROR, S.SAVED, S.SAVE_FAILED].includes(state)) return;
    if (state === S.SAVE_FAILED && !window.confirm("Your last recording has not been saved. Start the camera anyway? (You can still retry or download it.)")) return;
    setState(S.PREPARING, "Requesting camera access…");
    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("This browser cannot use the camera here. Use a recent Chrome, Edge, Firefox or Safari on http://localhost or HTTPS.");
        }
        if (!window.MediaRecorder) throw new Error("This browser cannot record video (MediaRecorder is missing).");
        stopCamera();
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
            audio: false,
        });
        stream.getVideoTracks().forEach((t) => t.addEventListener("ended", onCameraLost));
        camera.srcObject = stream;
        await waitForMedia(camera, { timeout: 10000,
            timeoutMessage: "The camera was opened but sent no picture. Close other apps using it, then try again." });
        await camera.play();
        await prepareReference();
        if (state !== S.PREPARING) return;
        setState(S.READY, "Ready. Step back until your whole body is inside the dashed box, then practise or record.");
    } catch (error) {
        stopCamera();
        // getUserMedia rejects with a DOMException (NotAllowedError, ...); media problems are plain Errors.
        const cameraProblem = (error && error.name && error.name !== "Error" && !(error instanceof TimeoutError))
            || (error && /camera/i.test(error.message || ""));
        if (cameraProblem) fail(error.name && error.name !== "Error" ? cameraErrorMessage(error) : error.message,
            { label: "Try again", run: () => prepareCamera() });
        else fail((error && error.message) || "Setup failed.", { label: "Reload page", run: () => window.location.reload() });
    }
}

function onCameraLost() {
    if (state === S.RECORDING || state === S.STARTING || state === S.COUNTDOWN) {
        abortRecording("The camera stopped (disconnected or taken by another app). The recording was discarded.");
    } else if (state !== S.SAVING && state !== S.SAVED && state !== S.SAVE_FAILED) {
        stopCamera();
        fail("The camera stopped. Turn it on again to continue.");
    }
}

async function prepareReference() {
    await waitForMedia(reference, { readyState: 1, timeout: 20000 });
    if (!track) {
        try {
            track = await SkeletonTrack.load(shell.dataset.landmarksUrl);
        } catch (_) {
            track = null;          // the ghost is optional; recording still works
            ui.status.textContent = "The ghost skeleton could not be loaded; you can still record.";
        }
    }
    const s = settings();
    await seekTo(reference, s.start);
    await waitForMedia(reference, { readyState: 3, event: "canplay", timeout: 20000 });
}

// ---------------------------------------------------------------- drawing (single timeline)
function draw() {
    const { ctx, width, height } = fitCanvas(canvas);
    ctx.clearRect(0, 0, width, height);
    if (stream && camera.videoWidth) {
        const rect = contentRect(camera, width, height);
        const guide = guideRect(rect);
        const showGuide = [S.READY, S.PREPARING, S.COUNTDOWN, S.STARTING].includes(state);
        drawGuide(ctx, guide, { alpha: showGuide ? 0.9 : 0.25, label: showGuide });
        if (track) {
            const s = settings();
            const box = track.box(s.start, s.end);
            const aspect = (track.width || 16) / (track.height || 9);
            const flip = s.viewMirrored !== (s.followMode === "mirror");
            drawSkeleton(ctx, track.frameAt(reference.currentTime), track.connections,
                ghostMap(guide, box, aspect, flip), { alpha: Number(ui.opacity.value), width: 5 });
        }
    }
    if (state === S.PRACTICING || state === S.RECORDING) checkSectionEnd();
    requestAnimationFrame(draw);
}

function checkSectionEnd() {
    const s = settings();
    if (reference.currentTime < s.end - 0.5 / (Number(shell.dataset.fps) || 30)) return;
    if (state === S.PRACTICING) {
        if (ui.loop.checked) {
            reference.currentTime = s.start;
        } else {
            reference.pause();
            setState(S.READY, "Section finished. Practise again or record an attempt.");
        }
    } else if (state === S.RECORDING) {
        finishRecording("section_end");
    }
}

// ---------------------------------------------------------------- practice (no recording)
async function practice() {
    if (state !== S.READY) return;
    setState(S.PRACTICING, "Practising with the ghost (nothing is recorded). Press Stop when you are done.");
    const s = settings();
    try {
        reference.playbackRate = s.speed;
        await seekTo(reference, s.start);
        await withTimeout(reference.play(), 8000, "The reference did not start playing.");
    } catch (error) {
        reference.pause();
        if (state === S.PRACTICING) setState(S.READY, `Playback failed: ${error.message}`);
    }
}

// ---------------------------------------------------------------- recording
function beep(freq, ms) {
    try {
        audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.frequency.value = freq;
        gain.gain.value = 0.08;
        osc.connect(gain).connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + ms / 1000);
    } catch (_) { /* no audio: the visual countdown is enough */ }
}

function now() { return performance.now() - sessionOrigin; }

function logEvent(type) {
    if (!sync) return;
    sync.events.push({ type, session_time_ms: now(), reference_time: reference.currentTime, playback_rate: reference.playbackRate });
}

let lastFrameSample = 0;

function startSampling() {
    sampling = true;
    lastFrameSample = 0;
    if ("requestVideoFrameCallback" in reference) {
        const onFrame = (_now, meta) => {
            if (!sampling) return;
            // expectedDisplayTime is when this frame reaches the screen: what the learner actually saw.
            if (!reference.paused) {
                sync.samples.push([meta.expectedDisplayTime - sessionOrigin, meta.mediaTime]);
                lastFrameSample = performance.now();
            }
            reference.requestVideoFrameCallback(onFrame);
        };
        reference.requestVideoFrameCallback(onFrame);
    }
}

// Fallback sampler. Frame callbacks stop when the tab is in the background (and some browsers lack
// them); 'timeupdate' still fires about 4x per second, close enough to keep the timeline continuous.
reference.addEventListener("timeupdate", () => {
    if (!sampling || reference.paused || reference.seeking) return;
    if (performance.now() - lastFrameSample > 300) sync.samples.push([now(), reference.currentTime]);
});

async function record() {
    if (state !== S.READY) return;
    setState(S.COUNTDOWN, "Get into position…");   // synchronous: buttons are disabled before any await
    const token = ++runToken;
    const s = settings();
    pending = null;
    try {
        reference.pause();
        await seekTo(reference, s.start);
        const seconds = Number(ui.countdownSel.value) || 5;
        ui.countdown.hidden = false;
        for (let n = seconds; n > 0; n--) {
            if (token !== runToken) return;
            ui.countdown.textContent = String(n);
            beep(n <= 3 ? 660 : 440, 120);
            await new Promise((r) => setTimeout(r, 1000));
        }
        if (token !== runToken) return;
        ui.countdown.textContent = "Go!";
        beep(990, 250);
        setState(S.STARTING, "Starting…");

        const mimeType = RECORDER_TYPES.find((t) => MediaRecorder.isTypeSupported(t)) || "";
        chunks = [];
        discardRecording = false;
        sessionOrigin = performance.now();
        sync = { recorder_start_ms: null, samples: [], events: [] };
        recorder = new MediaRecorder(stream, { ...(mimeType ? { mimeType } : {}), videoBitsPerSecond: 2_500_000 });
        recorder.addEventListener("dataavailable", (e) => { if (e.data && e.data.size) chunks.push(e.data); });
        recorder.addEventListener("stop", onRecorderStop);
        recorder.addEventListener("error", (e) => abortRecording(`Recording failed: ${(e.error && e.error.message) || "unknown error"}.`));
        await withTimeout(new Promise((resolve) => {
            recorder.addEventListener("start", () => { sync.recorder_start_ms = now(); resolve(); }, { once: true });
            recorder.start(1000);
        }), 5000, "The recorder did not start.");
        if (token !== runToken) return;

        reference.playbackRate = s.speed;
        startSampling();
        logEvent("play_requested");
        await withTimeout(reference.play(), 8000, "The reference video did not start playing.");
        if (token !== runToken) return;
        logEvent("reference_play");
        ui.countdown.hidden = true;
        setState(S.RECORDING, `Recording ${fmtTime(s.start)}–${fmtTime(s.end)} at ${s.speed}×. It stops by itself at the end of the section.`);
    } catch (error) {
        if (token === runToken) abortRecording(`Could not start recording: ${error.message}`);
    }
}

/** Failure while starting or recording: stop the recorder (discarding data), playback and camera. */
function abortRecording(message) {
    runToken++;
    sampling = false;
    ui.countdown.hidden = true;
    reference.pause();
    if (recorder && recorder.state !== "inactive") {
        discardRecording = true;
        try { recorder.stop(); } catch (_) { /* already stopped */ }
    }
    recorder = null;
    chunks = [];
    stopCamera();
    fail(message);
}

function cancelCountdown() {
    runToken++;
    ui.countdown.hidden = true;
    reference.pause();
    setState(S.READY, "Countdown cancelled.");
}

function finishRecording(reason) {
    if (state !== S.RECORDING) return;
    logEvent(reason);
    reference.pause();
    sampling = false;
    setState(S.SAVING, "Finishing the recording…");
    if (recorder && recorder.state !== "inactive") recorder.stop();
}

function onRecorderStop() {
    if (discardRecording) { discardRecording = false; return; }
    logEvent("recording_stop");
    const type = (recorder && recorder.mimeType) || "video/webm";
    const blob = new Blob(chunks, { type });
    chunks = [];
    recorder = null;
    stopCamera();                    // camera light off while saving
    if (!blob.size) {
        fail("The recording was empty. Turn the camera on and try again.");
        return;
    }
    const s = settings();
    if (pending && pending.url) URL.revokeObjectURL(pending.url);
    pending = {
        blob, url: URL.createObjectURL(blob), settings: s, sync,
        uploadId: (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`),
    };
    ui.download.href = pending.url;
    ui.download.download = `movematch-attempt-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.${type.includes("mp4") ? "mp4" : "webm"}`;
    upload();
}

function upload() {
    if (!pending) return;
    setState(S.SAVING, "Saving your recording…");
    ui.savePanel.hidden = false;
    ui.review.hidden = ui.retry.hidden = ui.again.hidden = ui.discard.hidden = true;
    ui.download.hidden = false;
    ui.saveProgress.hidden = false;
    ui.saveProgress.value = 0;
    if (pending.blob.size > maxBytes) {
        onUploadFailed(`The recording is ${(pending.blob.size / 1048576).toFixed(0)} MB, above the ${shell.dataset.maxMb} MB limit. Record a shorter section, or download it.`, false);
        return;
    }
    const s = pending.settings;
    const form = new FormData();
    form.append("recording", pending.blob, `attempt.${pending.blob.type.includes("mp4") ? "mp4" : "webm"}`);
    form.append("upload_id", pending.uploadId);
    form.append("speed", String(s.speed));
    form.append("section_start", s.start.toFixed(3));
    form.append("section_end", s.end.toFixed(3));
    form.append("view_mirrored", s.viewMirrored ? "1" : "0");
    form.append("follow_mode", s.followMode);
    form.append("sync", JSON.stringify(pending.sync));
    form.append("user_agent", navigator.userAgent.slice(0, 120));

    const xhr = new XMLHttpRequest();
    xhr.open("POST", shell.dataset.attemptUrl);
    xhr.setRequestHeader("X-CSRFToken", csrfToken());
    xhr.setRequestHeader("Accept", "application/json");
    xhr.timeout = 5 * 60 * 1000;
    xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
            ui.saveProgress.value = e.loaded / e.total;
            ui.saveStatus.textContent = `Uploading… ${Math.round((100 * e.loaded) / e.total)}%`;
        }
    };
    xhr.onload = () => {
        let body = {};
        try { body = JSON.parse(xhr.responseText || "{}"); } catch (_) { /* non-JSON error page */ }
        if (xhr.status === 200 || xhr.status === 201) {
            onUploaded(body);
        } else {
            const retryable = xhr.status >= 500 || xhr.status === 429 || body.code === "csrf";
            onUploadFailed(body.error || `The server rejected the upload (${xhr.status}).`, retryable);
        }
    };
    xhr.onerror = () => onUploadFailed("Network error while uploading. Your recording is still here.", true);
    xhr.ontimeout = () => onUploadFailed("The upload timed out. Your recording is still here.", true);
    xhr.send(form);
}

function onUploaded(body) {
    ui.saveProgress.hidden = true;
    ui.saveStatus.textContent = "Saved. The server is analysing it now; you can already replay it.";
    ui.review.href = body.page_url;
    ui.review.hidden = false;
    ui.again.hidden = false;
    ui.retry.hidden = true;
    ui.discard.hidden = true;
    pending = null;                  // the server has it; the object URL stays valid for "Download"
    setState(S.SAVED, "Saved. Open the replay, or record another take.");
}

function onUploadFailed(message, retryable) {
    ui.saveProgress.hidden = true;
    ui.saveStatus.textContent = message;
    ui.retry.hidden = !retryable;
    ui.discard.hidden = false;
    ui.download.hidden = false;
    setState(S.SAVE_FAILED, "Not saved yet. Retry, or download the recording to keep it.");
}

function discard() {
    if (!window.confirm("Discard this recording? It has not been saved.")) return;
    if (ui.download.href.startsWith("blob:")) URL.revokeObjectURL(ui.download.href);
    pending = null;
    ui.savePanel.hidden = true;
    setState(S.IDLE, "Recording discarded. Turn on the camera to try again.");
}

function stop() {
    if (state === S.PRACTICING) {
        reference.pause();
        setState(S.READY, "Stopped.");
    } else if (state === S.COUNTDOWN) {
        cancelCountdown();
    } else if (state === S.STARTING) {
        abortRecording("Recording cancelled.");
    } else if (state === S.RECORDING) {
        finishRecording("manual_stop");
    }
}

// ---------------------------------------------------------------- wiring
ui.form.addEventListener("submit", (e) => e.preventDefault());
ui.prepare.addEventListener("click", prepareCamera);
ui.practice.addEventListener("click", practice);
ui.record.addEventListener("click", record);
ui.stop.addEventListener("click", stop);
ui.retry.addEventListener("click", upload);
ui.discard.addEventListener("click", discard);
ui.again.addEventListener("click", () => { ui.savePanel.hidden = true; prepareCamera(); });
[ui.mirror, ui.follow].forEach((el) => el.addEventListener("change", applyMirror));
[ui.start, ui.end].forEach((el) => el.addEventListener("change", () => {
    if (state === S.READY) seekTo(reference, settings().start).catch(() => {});
}));
ui.volume.addEventListener("input", () => { reference.volume = Number(ui.volume.value); });

reference.preservesPitch = true;
reference.volume = Number(ui.volume.value);
["pause", "waiting", "playing", "seeking", "seeked", "ratechange", "ended", "stalled"].forEach((name) => {
    reference.addEventListener(name, () => { if (state === S.RECORDING || state === S.STARTING) logEvent(`reference_${name}`); });
});
reference.addEventListener("ended", () => { if (state === S.RECORDING) finishRecording("reference_ended"); });
// requestAnimationFrame stops in background tabs; timeupdate keeps ending the section on time.
reference.addEventListener("timeupdate", () => { if (state === S.PRACTICING || state === S.RECORDING) checkSectionEnd(); });
reference.addEventListener("error", () => {
    const message = "The reference video failed to load (network or file problem).";
    if (state === S.RECORDING || state === S.STARTING || state === S.COUNTDOWN) abortRecording(message);
    else fail(message, { label: "Reload page", run: () => window.location.reload() });
}, true);

window.addEventListener("beforeunload", (event) => {
    if ([S.RECORDING, S.SAVING, S.SAVE_FAILED, S.COUNTDOWN, S.STARTING].includes(state)) {
        event.preventDefault();
        event.returnValue = "";
    }
});
window.addEventListener("pagehide", stopCamera);

loadQuerySettings();
applyMirror();
setState(S.IDLE);
waitForMedia(reference, { timeout: 20000 }).then(() => {
    if (Number(ui.end.value) <= 0 || Number(ui.end.value) > reference.duration) ui.end.value = reference.duration.toFixed(2);
}).catch((error) => fail(error.message, { label: "Reload page", run: () => window.location.reload() }));
requestAnimationFrame(draw);
