// Step 2: practise a section. Slow playback with music, frame stepping, A/B section loop,
// mirror view and a skeleton overlay drawn from the cached landmarks on a transparent canvas.
import { $, SkeletonTrack, clamp, contentRect, drawSkeleton, fitCanvas, fmtTime, waitForMedia } from "./common.js";

const shell = document.querySelector(".practice-shell");
const video = $("danceVideo");
const canvas = $("skeletonCanvas");
const fps = Number(shell.dataset.fps) || 30;
const frameDuration = 1 / fps;
const storageKey = `movematch:section:${shell.dataset.videoId}`;

const ui = {
    play: $("playPause"), prev: $("prevFrame"), next: $("nextFrame"), speed: $("speed"),
    timeline: $("timeline"), info: $("frameInfo"), start: $("sectionStart"), end: $("sectionEnd"),
    setStart: $("setStart"), setEnd: $("setEnd"), loop: $("loopToggle"), clear: $("clearSection"),
    mirror: $("mirrorToggle"), skeleton: $("skeletonToggle"), opacity: $("skeletonOpacity"),
    record: $("recordLink"), bar: $("sectionBar"), error: $("mediaError"), wrap: $("mediaWrap"),
};

let track = null;
const duration = () => (Number.isFinite(video.duration) && video.duration > 0 ? video.duration : Number(shell.dataset.duration) || 0);

function readStored() {
    try { return JSON.parse(localStorage.getItem(storageKey) || "null"); } catch (_) { return null; }
}
function store() {
    try {
        localStorage.setItem(storageKey, JSON.stringify({
            start: Number(ui.start.value), end: Number(ui.end.value), speed: ui.speed.value,
            mirror: ui.mirror.checked,
        }));
    } catch (_) { /* storage unavailable: fine */ }
}

function section() {
    const d = duration();
    let a = clamp(Number(ui.start.value) || 0, 0, d);
    let b = clamp(Number(ui.end.value) || d, 0, d);
    if (b - a < 0.5) b = Math.min(d, a + 0.5);
    return [a, b];
}

function updateSectionUi() {
    const [a, b] = section();
    const d = duration() || 1;
    ui.bar.style.left = `${(a / d) * 100}%`;
    ui.bar.style.width = `${((b - a) / d) * 100}%`;
    const params = new URLSearchParams({
        start: a.toFixed(2), end: b.toFixed(2), speed: ui.speed.value, mirror: ui.mirror.checked ? "1" : "0",
    });
    ui.record.href = `${shell.dataset.ghostUrl}?${params}`;
    store();
}

function updateInfo() {
    const frame = Math.round(video.currentTime * fps);
    ui.info.textContent = `${fmtTime(video.currentTime)} · frame ${frame}`;
    const d = duration();
    if (d) ui.timeline.value = Math.round((video.currentTime / d) * 1000);
}

function seekFrame(direction) {
    if (video.readyState < 1) return;
    video.pause();
    // Land in the middle of the target frame so rounding never shows the neighbour.
    const current = Math.round(video.currentTime * fps);
    const target = clamp(current + direction, 0, Math.floor(duration() * fps) - 1);
    video.currentTime = (target + 0.5) * frameDuration;
}

async function togglePlay() {
    if (!video.paused) { video.pause(); return; }
    const [a, b] = section();
    if (ui.loop.checked && (video.currentTime < a || video.currentTime >= b - 0.02)) video.currentTime = a;
    try {
        await video.play();
    } catch (error) {
        showError(error.message || "Playback could not start.");
    }
}

function showError(message) {
    ui.error.textContent = message;
    ui.error.hidden = false;
}

function enforceLoop() {
    if (!video.paused && ui.loop.checked) {
        const [a, b] = section();
        if (video.currentTime >= b) video.currentTime = a;
    }
}

// One animation loop: precise section looping and drawing. Browsers pause requestAnimationFrame in
// background tabs, so 'timeupdate' (about 4x/s, never paused) also enforces the loop.
function frameLoop() {
    enforceLoop();
    const { ctx, width, height } = fitCanvas(canvas);
    ctx.clearRect(0, 0, width, height);
    if (track && ui.skeleton.checked) {
        const r = contentRect(video, width, height);
        drawSkeleton(ctx, track.frameAt(video.currentTime), track.connections,
            (x, y) => [r.x + x * r.w, r.y + y * r.h], { alpha: Number(ui.opacity.value) });
    }
    requestAnimationFrame(frameLoop);
}

function applyMirror() {
    // The wrapper holds both the video and the canvas, so the skeleton flips with the picture.
    ui.wrap.classList.toggle("mirrored", ui.mirror.checked);
    updateSectionUi();
}

// ---------------------------------------------------------------- wiring
const saved = readStored();
if (saved) {
    if (Number.isFinite(saved.start)) ui.start.value = saved.start;
    if (Number.isFinite(saved.end) && saved.end > 0) ui.end.value = saved.end;
    if (saved.speed) ui.speed.value = saved.speed;
    if (typeof saved.mirror === "boolean") ui.mirror.checked = saved.mirror;
}
video.preservesPitch = true;        // slow music keeps its pitch
video.playbackRate = Number(ui.speed.value);

ui.prev.addEventListener("click", () => seekFrame(-1));
ui.next.addEventListener("click", () => seekFrame(1));
ui.play.addEventListener("click", togglePlay);
ui.speed.addEventListener("change", () => { video.playbackRate = Number(ui.speed.value); updateSectionUi(); });
ui.timeline.addEventListener("input", () => {
    const d = duration();
    if (d) video.currentTime = (Number(ui.timeline.value) / 1000) * d;
});
ui.setStart.addEventListener("click", () => { ui.start.value = video.currentTime.toFixed(2); updateSectionUi(); });
ui.setEnd.addEventListener("click", () => { ui.end.value = video.currentTime.toFixed(2); updateSectionUi(); });
ui.clear.addEventListener("click", () => { ui.start.value = 0; ui.end.value = duration().toFixed(2); updateSectionUi(); });
[ui.start, ui.end].forEach((el) => el.addEventListener("change", updateSectionUi));
ui.mirror.addEventListener("change", applyMirror);

video.addEventListener("play", () => { ui.play.textContent = "Pause"; });
video.addEventListener("pause", () => { ui.play.textContent = "Play"; });
video.addEventListener("loadedmetadata", () => {
    if (!Number(ui.end.value) || Number(ui.end.value) > duration()) ui.end.value = duration().toFixed(2);
    updateSectionUi();
    updateInfo();
});
video.addEventListener("seeked", updateInfo);
video.addEventListener("timeupdate", () => { enforceLoop(); updateInfo(); });
video.addEventListener("error", () => showError("The video could not be loaded. Reload the page, or reprocess the video."), true);

document.addEventListener("keydown", (event) => {
    if (event.target.matches("input, textarea, select") || event.target.isContentEditable) return;
    const keys = {
        ArrowLeft: () => seekFrame(-1), ArrowRight: () => seekFrame(1), Space: togglePlay,
        BracketLeft: () => ui.setStart.click(), BracketRight: () => ui.setEnd.click(),
        KeyL: () => { ui.loop.checked = !ui.loop.checked; }, KeyM: () => { ui.mirror.checked = !ui.mirror.checked; applyMirror(); },
    };
    if (keys[event.code]) { event.preventDefault(); keys[event.code](); }
});

applyMirror();
waitForMedia(video, { timeout: 20000 }).catch((error) => showError(error.message));
SkeletonTrack.load(shell.dataset.landmarksUrl)
    .then((t) => { track = t; })
    .catch(() => { ui.skeleton.checked = false; ui.skeleton.disabled = true; });
requestAnimationFrame(frameLoop);
