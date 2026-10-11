const video = document.getElementById("danceVideo");
const prevButton = document.getElementById("prevFrame");
const nextButton = document.getElementById("nextFrame");
const playPauseButton = document.getElementById("playPause");
const speedSelect = document.getElementById("speed");
const timeline = document.getElementById("timeline");
const frameInfo = document.getElementById("frameInfo");

// Replace with the actual FPS from your processed video
// if you know it from the OpenCV pipeline.
const FPS = {{ video.fps | default(30, true) }};
const FRAME_DURATION = 1 / FPS;

function updateInfo() {
const frame = Math.round(video.currentTime * FPS);

frameInfo.textContent =
    `Frame: ${frame} | Time: ${video.currentTime.toFixed(3)} s`;

if (Number.isFinite(video.duration) && video.duration > 0) {
    timeline.value = Math.round(
        video.currentTime / video.duration * 1000
    );
}
}

function seekByFrame(direction) {
if (video.readyState < 1) return;

video.pause();

const lastTime = Number.isFinite(video.duration)
    ? Math.max(0, video.duration - FRAME_DURATION)
    : video.currentTime + direction * FRAME_DURATION;

const target = Math.max(
    0,
    Math.min(
        lastTime,
        video.currentTime + direction * FRAME_DURATION
    )
);

video.currentTime = target;
}

prevButton.addEventListener("click", () => seekByFrame(-1));
nextButton.addEventListener("click", () => seekByFrame(1));

playPauseButton.addEventListener("click", () => {
if (video.paused) {
    video.play().catch(console.error);
} else {
    video.pause();
}
});

video.addEventListener("play", () => {
playPauseButton.textContent = "Pause";
});

video.addEventListener("pause", () => {
playPauseButton.textContent = "Play";
});

speedSelect.addEventListener("change", () => {
video.playbackRate = Number(speedSelect.value);
});

timeline.addEventListener("input", () => {
if (Number.isFinite(video.duration)) {
    video.currentTime =
        Number(timeline.value) / 1000 * video.duration;
}
});

video.addEventListener("loadedmetadata", updateInfo);
video.addEventListener("seeked", updateInfo);
video.addEventListener("timeupdate", updateInfo);

document.addEventListener("keydown", (event) => {
// Don't hijack keyboard controls while editing a form field.
if (
    event.target.matches("input, textarea, select") ||
    event.target.isContentEditable
) {
    return;
}

if (event.code === "ArrowLeft") {
    event.preventDefault();
    seekByFrame(-1);
} else if (event.code === "ArrowRight") {
    event.preventDefault();
    seekByFrame(1);
} else if (event.code === "Space") {
    event.preventDefault();

    if (video.paused) {
        video.play().catch(console.error);
    } else {
        video.pause();
    }
}
});