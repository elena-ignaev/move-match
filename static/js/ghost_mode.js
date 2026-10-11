const ghostShell = document.querySelector(".ghost-shell");
const prepareButton = document.getElementById("prepareCamera");
const startButton = document.getElementById("startGhost");
const stopButton = document.getElementById("stopGhost");
const statusText = document.getElementById("ghostStatus");
const cameraPreview = document.getElementById("cameraPreview");
const referenceVideo = document.getElementById("referenceVideo");
const referenceOverlay = document.getElementById("referenceOverlay");
const overlayToggle = document.getElementById("overlayToggle");

let cameraStream = null;
let recorder = null;
let chunks = [];
let selectedMimeType = "";
let recordingStartedAt = null;
let sessionOriginMs = 0;
let syncEvents = [];

const recorderTypes = [
    "video/webm;codecs=vp9,opus",
    "video/webm;codecs=vp8,opus",
    "video/webm",
    "video/mp4"
];

function setStatus(message) {
    statusText.textContent = message;
}

function sessionTimeMs() {
    return sessionOriginMs ? Math.round(performance.now() - sessionOriginMs) : 0;
}

function recordSyncEvent(type) {
    syncEvents.push({
        type,
        session_time_ms: sessionTimeMs(),
        reference_time: referenceVideo.currentTime,
        playback_rate: referenceVideo.playbackRate,
        wall_time: new Date().toISOString()
    });
}

function selectedSpeed() {
    const checked = document.querySelector("input[name='ghostSpeed']:checked");
    return checked ? Number(checked.value) : 0.25;
}

function preferredMimeType() {
    if (!window.MediaRecorder) return "";

    return recorderTypes.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

function setControls(state) {
    const isIdle = state === "idle";
    const isReady = state === "ready";
    const isRecording = state === "recording";

    prepareButton.disabled = isRecording;
    startButton.disabled = !isReady;
    stopButton.disabled = !isRecording;
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach((track) => track.stop());
        cameraStream = null;
    }
    cameraPreview.srcObject = null;
}

function stopPlayback() {
    referenceVideo.pause();
    referenceOverlay.pause();
}

function waitForEvent(video, eventName, readyState = 0) {
    return new Promise((resolve) => {
        if (video.readyState >= readyState) {
            resolve();
        } else {
            video.addEventListener(eventName, resolve, { once: true });
        }
    });
}

async function alignReferenceVideos() {
    const speed = selectedSpeed();
    referenceVideo.playbackRate = speed;
    referenceOverlay.playbackRate = speed;

    await Promise.all([
        waitForEvent(referenceVideo, "loadedmetadata", 1),
        waitForEvent(referenceOverlay, "loadedmetadata", 1)
    ]);

    referenceVideo.currentTime = 0;
    referenceOverlay.currentTime = 0;

    await Promise.all([
        waitForEvent(referenceVideo, "canplay", 2),
        waitForEvent(referenceOverlay, "canplay", 2)
    ]);
}

async function prepareCamera() {
    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("This browser does not support camera capture.");
        }
        if (!window.MediaRecorder) {
            throw new Error("This browser does not support MediaRecorder.");
        }

        stopCamera();
        setControls("idle");
        setStatus("Requesting camera permission...");

        selectedMimeType = preferredMimeType();
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false
        });
        cameraPreview.srcObject = cameraStream;
        await cameraPreview.play();
        await alignReferenceVideos();

        setControls("ready");
        setStatus("Camera ready. Start recording when you are ready.");
    } catch (error) {
        stopCamera();
        setControls("idle");
        setStatus(error.message || "Camera setup failed.");
    }
}

function uploadAttempt(blob) {
    const formData = new FormData();
    const speed = selectedSpeed();
    const extension = blob.type.includes("mp4") ? "mp4" : "webm";

    formData.append("recording", blob, `ghost-attempt.${extension}`);
    formData.append("speed", String(speed));
    formData.append("reference_started_at", recordingStartedAt || "");
    formData.append("reference_stopped_at", new Date().toISOString());
    formData.append("reference_time", String(referenceVideo.currentTime));
    formData.append("sync_events", JSON.stringify(syncEvents));

    return fetch(ghostShell.dataset.attemptUrl, {
        method: "POST",
        body: formData
    });
}

async function finishRecording() {
    stopPlayback();
    stopCamera();
    setControls("idle");
    recordSyncEvent("recording_stop");

    const blob = new Blob(chunks, { type: selectedMimeType || "video/webm" });
    chunks = [];

    if (!blob.size) {
        setStatus("Recording was empty. Prepare the camera and try again.");
        return;
    }

    try {
        setStatus("Saving recording...");
        const response = await uploadAttempt(blob);
        const payload = await response.json();

        if (!response.ok) {
            throw new Error(payload.error || "Recording upload failed.");
        }

        setStatus(`Recording saved as attempt ${payload.attempt_id}. Status: ${payload.status}.`);
    } catch (error) {
        setStatus(error.message || "Recording upload failed.");
    }
}

async function startGhostMode() {
    try {
        if (!cameraStream) {
            await prepareCamera();
        }
        if (!cameraStream) return;

        await alignReferenceVideos();
        chunks = [];
        sessionOriginMs = performance.now();
        syncEvents = [];
        recordSyncEvent("recording_start");
        recorder = selectedMimeType
            ? new MediaRecorder(cameraStream, { mimeType: selectedMimeType })
            : new MediaRecorder(cameraStream);

        recorder.addEventListener("dataavailable", (event) => {
            if (event.data && event.data.size > 0) {
                chunks.push(event.data);
            }
        });
        recorder.addEventListener("stop", finishRecording, { once: true });
        recorder.addEventListener("error", () => {
            setStatus("Recording failed. Stop and try again.");
        });

        recordingStartedAt = new Date().toISOString();
        recorder.start(250);

        await Promise.all([
            referenceVideo.play(),
            referenceOverlay.play()
        ]);
        recordSyncEvent("reference_play");

        setControls("recording");
        setStatus("Recording. Follow the reference until it ends, or stop manually.");
    } catch (error) {
        stopPlayback();
        setControls(cameraStream ? "ready" : "idle");
        setStatus(error.message || "Could not start ghost mode.");
    }
}

function stopGhostMode() {
    if (recorder && recorder.state !== "inactive") {
        recordSyncEvent("manual_stop");
        recorder.stop();
    } else {
        stopPlayback();
        stopCamera();
        setControls("idle");
    }
}

function syncOverlayVisibility() {
    referenceOverlay.classList.toggle("is-hidden", !overlayToggle.checked);
}

prepareButton.addEventListener("click", prepareCamera);
startButton.addEventListener("click", startGhostMode);
stopButton.addEventListener("click", stopGhostMode);
overlayToggle.addEventListener("change", syncOverlayVisibility);

referenceVideo.addEventListener("ended", () => {
    if (recorder && recorder.state === "recording") {
        recordSyncEvent("reference_ended");
        stopGhostMode();
    }
});

referenceVideo.addEventListener("pause", () => {
    if (recorder && recorder.state === "recording") {
        recordSyncEvent("reference_pause");
    }
});

referenceVideo.addEventListener("waiting", () => {
    if (recorder && recorder.state === "recording") {
        recordSyncEvent("reference_waiting");
    }
});

referenceVideo.addEventListener("playing", () => {
    if (recorder && recorder.state === "recording") {
        recordSyncEvent("reference_playing");
    }
});

window.addEventListener("beforeunload", () => {
    if (recorder && recorder.state === "recording") {
        recorder.stop();
    }
    stopCamera();
});

setControls("idle");
syncOverlayVisibility();
