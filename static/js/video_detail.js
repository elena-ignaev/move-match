// Video overview: processing progress polling, reference playback with a skeleton overlay.
import { $, SkeletonTrack, contentRect, drawSkeleton, fitCanvas, poll, wireConfirmForms } from "./common.js";

wireConfirmForms();

const panel = $("processingPanel");
if (panel) {
    const bar = $("processingProgress");
    const message = $("processingMessage");
    poll(panel.dataset.statusUrl, (data) => {
        bar.value = data.progress;
        message.textContent = data.message || data.status;
        if (data.status === "completed" || data.status === "failed") window.location.reload();
    }, (data) => data.status === "completed" || data.status === "failed");
}

const video = $("referenceVideo");
const canvas = $("skeletonCanvas");
if (video && canvas) {
    const stage = video.closest(".stage");
    const toggle = $("skeletonToggle");
    const opacity = $("skeletonOpacity");
    let track = null;
    SkeletonTrack.load(stage.dataset.landmarksUrl).then((t) => { track = t; }).catch(() => { toggle.disabled = true; });

    const draw = () => {
        const { ctx, width, height } = fitCanvas(canvas);
        ctx.clearRect(0, 0, width, height);
        if (track && toggle.checked) {
            const r = contentRect(video, width, height);
            drawSkeleton(ctx, track.frameAt(video.currentTime), track.connections,
                (x, y) => [r.x + x * r.w, r.y + y * r.h], { alpha: Number(opacity.value), width: 3 });
        }
        requestAnimationFrame(draw);
    };
    requestAnimationFrame(draw);
}
