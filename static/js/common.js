// Shared helpers for the MoveMatch pages (plain ES module, no build step).

export const $ = (id) => document.getElementById(id);

export function csrfToken() {
    const meta = document.querySelector("meta[name='csrf-token']");
    return meta ? meta.content : "";
}

export function fmtTime(seconds) {
    if (!Number.isFinite(seconds)) return "–";
    const s = Math.max(0, seconds);
    const m = Math.floor(s / 60);
    return `${m}:${(s - m * 60).toFixed(1).padStart(4, "0")}`;
}

export function clamp(x, lo, hi) {
    return Math.min(hi, Math.max(lo, x));
}

export class TimeoutError extends Error {}

export function withTimeout(promise, ms, message) {
    let timer;
    return Promise.race([
        promise,
        new Promise((_, reject) => {
            timer = setTimeout(() => reject(new TimeoutError(message || "Timed out")), ms);
        }),
    ]).finally(() => clearTimeout(timer));
}

export async function fetchJSON(url, { timeout = 15000, ...options } = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
        const response = await fetch(url, {
            ...options,
            signal: controller.signal,
            headers: { Accept: "application/json", "X-CSRFToken": csrfToken(), ...(options.headers || {}) },
        });
        const body = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
        return body;
    } catch (error) {
        if (error.name === "AbortError") throw new TimeoutError("The server did not answer in time.");
        throw error;
    } finally {
        clearTimeout(timer);
    }
}

/** Resolve when the media element reaches readyState (or fires `event`); reject on error or timeout. */
export function waitForMedia(video, { readyState = 1, event = "loadedmetadata", timeout = 15000,
    timeoutMessage = "The video took too long to load. Check your connection and reload the page." } = {}) {
    if (video.error) return Promise.reject(mediaError(video));
    if (video.readyState >= readyState) return Promise.resolve();
    return new Promise((resolve, reject) => {
        const cleanup = () => {
            clearTimeout(timer);
            video.removeEventListener(event, onReady);
            video.removeEventListener("error", onError, true);
        };
        const onReady = () => { cleanup(); resolve(); };
        const onError = () => { cleanup(); reject(mediaError(video)); };
        const timer = setTimeout(() => {
            cleanup();
            reject(new TimeoutError(timeoutMessage));
        }, timeout);
        video.addEventListener(event, onReady);
        video.addEventListener("error", onError, true);   // capture: <source> errors do not bubble
    });
}

export function mediaError(video) {
    const code = video.error ? video.error.code : 0;
    const reasons = {
        1: "Loading was aborted.",
        2: "A network error stopped the video from loading.",
        3: "The video could not be decoded.",
        4: "The video file is missing or in a format this browser cannot play.",
    };
    return new Error(`${reasons[code] || "The video could not be loaded."} Reload the page; if it keeps happening, reprocess the video.`);
}

/** Seek and wait for 'seeked' (with timeout). */
export async function seekTo(video, time, timeout = 10000) {
    await waitForMedia(video, { readyState: 1, timeout });
    if (Math.abs(video.currentTime - time) < 0.001) return;
    const done = new Promise((resolve) => video.addEventListener("seeked", resolve, { once: true }));
    video.currentTime = time;
    await withTimeout(done, timeout, "Seeking in the video took too long.");
}

// ------------------------------------------------------------------ skeleton data
const DRAW_MIN_VIS = 0.35;

export class SkeletonTrack {
    constructor(data) {
        this.times = data.timestamps_ms.map((t) => t / 1000);
        this.frames = data.frames;
        this.connections = data.connections;
        this.width = data.width;
        this.height = data.height;
        this.fps = data.fps || 30;
    }

    static async load(url) {
        return new SkeletonTrack(await fetchJSON(url, { timeout: 30000 }));
    }

    /** Landmarks of the frame shown at time t (seconds), or null. */
    frameAt(t) {
        const times = this.times;
        if (!times.length || t < times[0] - 0.5 / this.fps || t > times[times.length - 1] + 1 / this.fps) return null;
        let lo = 0, hi = times.length - 1;
        while (lo < hi) {                       // last frame with time <= t
            const mid = (lo + hi + 1) >> 1;
            if (times[mid] <= t + 1e-4) lo = mid; else hi = mid - 1;
        }
        return this.frames[lo];
    }

    /** Bounding box (normalized image coords) of the dancer over [a, b] seconds, for ghost placement. */
    box(a = 0, b = Infinity) {
        let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
        this.times.forEach((t, i) => {
            const f = this.frames[i];
            if (!f || t < a || t > b) return;
            for (const [x, y, v] of f) {
                if (v < DRAW_MIN_VIS) continue;
                x0 = Math.min(x0, x); x1 = Math.max(x1, x);
                y0 = Math.min(y0, y); y1 = Math.max(y1, y);
            }
        });
        if (!Number.isFinite(x0)) return { x0: 0.3, y0: 0.05, x1: 0.7, y1: 0.95 };
        return { x0, y0, x1, y1 };
    }
}

/** Size a canvas's backing store to its CSS box (sharp on high-DPI screens). Returns the 2D context. */
export function fitCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width: rect.width, height: rect.height };
}

/** Rectangle where a video's picture is drawn inside its element box (object-fit: contain). */
export function contentRect(video, boxWidth, boxHeight) {
    const vw = video.videoWidth || 16, vh = video.videoHeight || 9;
    const scale = Math.min(boxWidth / vw, boxHeight / vh);
    const w = vw * scale, h = vh * scale;
    return { x: (boxWidth - w) / 2, y: (boxHeight - h) / 2, w, h };
}

/**
 * Draw one frame of landmarks. `map(x, y)` converts normalized image coords to canvas pixels.
 */
export function drawSkeleton(ctx, frame, connections, map, { color = "#3ce0a0", joint = "#ffd60a", alpha = 0.8, width = 4 } = {}) {
    if (!frame || alpha <= 0) return;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.lineCap = "round";
    ctx.lineWidth = width;
    ctx.strokeStyle = color;
    for (const [a, b] of connections) {
        const p = frame[a], q = frame[b];
        if (!p || !q || p[2] < DRAW_MIN_VIS || q[2] < DRAW_MIN_VIS || a < 11 || b < 11) continue;   // skip face mesh
        const [x1, y1] = map(p[0], p[1]);
        const [x2, y2] = map(q[0], q[1]);
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    }
    ctx.fillStyle = joint;
    for (let i = 11; i < frame.length; i++) {
        const [x, y, v] = frame[i];
        if (v < DRAW_MIN_VIS) continue;
        const [px, py] = map(x, y);
        ctx.beginPath(); ctx.arc(px, py, width * 0.9, 0, Math.PI * 2); ctx.fill();
    }
    const nose = frame[0];
    if (nose && nose[2] >= DRAW_MIN_VIS && frame[11] && frame[12]) {     // simple head circle
        const [nx, ny] = map(nose[0], nose[1]);
        const [lx, ly] = map(frame[11][0], frame[11][1]);
        const [rx, ry] = map(frame[12][0], frame[12][1]);
        const r = Math.max(6, Math.hypot(lx - rx, ly - ry) * 0.35);
        ctx.beginPath(); ctx.arc(nx, ny, r, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.restore();
}

/** The full-body guide: the region the learner should stay inside (fractions of the camera picture). */
export function guideRect(rect) {
    const h = rect.h * 0.92;
    const w = Math.min(rect.w * 0.7, h * 0.9);
    return { x: rect.x + (rect.w - w) / 2, y: rect.y + rect.h * 0.04, w, h };
}

/**
 * Map for drawing the reference skeleton inside the guide: the dancer's bounding box over the section
 * is fitted into the guide (feet at the bottom), keeping proportions and travel across the floor.
 */
export function ghostMap(guide, box, refAspect, flip) {
    const bw = Math.max(1e-3, (box.x1 - box.x0) * refAspect);   // box in "height units"
    const bh = Math.max(1e-3, box.y1 - box.y0);
    const scale = Math.min(guide.w / bw, guide.h / bh);
    const drawnW = bw * scale, drawnH = bh * scale;
    const ox = guide.x + (guide.w - drawnW) / 2;
    const oy = guide.y + (guide.h - drawnH);
    return (x, y) => {
        const fx = flip ? box.x1 - (x - box.x0) : x;
        return [ox + (fx - box.x0) * refAspect * scale, oy + (y - box.y0) * scale];
    };
}

export function drawGuide(ctx, guide, { alpha = 0.9, label = true } = {}) {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.setLineDash([10, 8]);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#ffffff";
    ctx.strokeRect(guide.x, guide.y, guide.w, guide.h);
    ctx.setLineDash([]);
    if (label) {
        ctx.fillStyle = "rgba(0,0,0,0.55)";
        ctx.fillRect(guide.x, guide.y, guide.w, 26);
        ctx.fillStyle = "#fff";
        ctx.font = "600 13px system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Head and feet inside the box", guide.x + guide.w / 2, guide.y + 17);
    }
    ctx.restore();
}

/** Ask before submitting forms marked data-confirm (delete buttons). */
export function wireConfirmForms() {
    document.querySelectorAll("form[data-confirm]").forEach((form) => {
        form.addEventListener("submit", (event) => {
            if (!window.confirm(form.dataset.confirm)) event.preventDefault();
        });
    });
}

/** Poll a JSON endpoint until `done(data)` is true. Backs off on errors. */
export function poll(url, onData, done, interval = 1500) {
    let stopped = false;
    let delay = interval;
    const tick = async () => {
        if (stopped) return;
        try {
            const data = await fetchJSON(url);
            delay = interval;
            onData(data);
            if (done(data)) return;
        } catch (_) {
            delay = Math.min(delay * 2, 15000);
        }
        setTimeout(tick, delay);
    };
    tick();
    return () => { stopped = true; };
}
