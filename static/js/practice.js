const video = document.getElementById("danceVideo");
const prevButton = document.getElementById("prevFrame");
const nextButton = document.getElementById("nextFrame");
const playPauseButton = document.getElementById("playPause");
const speedSelect = document.getElementById("speed");
const timeline = document.getElementById("timeline");
const frameInfo = document.getElementById("frameInfo");

if (video && prevButton && nextButton && playPauseButton && speedSelect && timeline && frameInfo) {
    const fps = Number(video.dataset.fps) || 30;
    const frameDuration = 1 / fps;

    function updateInfo() {
        const frame = Math.round(video.currentTime * fps);
        frameInfo.textContent = `Frame: ${frame} | Time: ${video.currentTime.toFixed(3)} s`;

        if (Number.isFinite(video.duration) && video.duration > 0) {
            timeline.value = Math.round(video.currentTime / video.duration * 1000);
        }
    }

    function seekByFrame(direction) {
        if (video.readyState < 1) return;

        video.pause();

        const lastTime = Number.isFinite(video.duration)
            ? Math.max(0, video.duration - frameDuration)
            : video.currentTime + direction * frameDuration;

        video.currentTime = Math.max(
            0,
            Math.min(lastTime, video.currentTime + direction * frameDuration)
        );
    }

    function togglePlayback() {
        if (video.paused) {
            video.play().catch(console.error);
        } else {
            video.pause();
        }
    }

    prevButton.addEventListener("click", () => seekByFrame(-1));
    nextButton.addEventListener("click", () => seekByFrame(1));
    playPauseButton.addEventListener("click", togglePlayback);

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
            video.currentTime = Number(timeline.value) / 1000 * video.duration;
        }
    });

    video.addEventListener("loadedmetadata", updateInfo);
    video.addEventListener("seeked", updateInfo);
    video.addEventListener("timeupdate", updateInfo);

    document.addEventListener("keydown", (event) => {
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
            togglePlayback();
        }
    });
}
