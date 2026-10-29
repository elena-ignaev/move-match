// Steps 4-6: replay an attempt next to the reference, show feedback, compare with the previous take.
//
// The attempt video is the master clock. For its current time the server-computed replay map gives
// the reference time that was on screen during recording; the reference video (and the ghost
// skeleton) follow it. Small drift is corrected by nudging playbackRate, large drift by seeking.
import { $, SkeletonTrack, clamp, contentRect, drawSkeleton, fitCanvas, fmtTime, ghostMap, guideRect, poll,
    waitForMedia, wireConfirmForms } from "./common.js";

wireConfirmForms();

const panel = document.querySelector(".replay-panel");
const attemptVideo = $("attemptVideo");
const reference = $("referenceVideo");
const canvas = $("ghostCanvas");
const ui = {
    stage: $("replayStage"), play: $("replayPlay"), time: $("replayTime"), timeline: $("replayTimeline"),
    speed: $("replaySpeed"), offset: $("applyOffset"), opacity: $("ghostOpacity"), error: $("replayError"),
    syncNote: $("syncNote"), processing: $("processingBox"), progress: $("attemptProgress"),
    body: $("feedbackBody"), comparison: $("comparisonBody"), status: $("attemptStatus"),
};

let data = JSON.parse($("attemptData").textContent);
let track = null;
let map = data.replay_map || [];

// ---------------------------------------------------------------- time mapping
function refAt(t) {
    if (!map.length) return null;
    let lo = 0, hi = map.length - 1;
    if (t <= map[0][0]) return map[0][1];
    if (t >= map[hi][0]) return map[hi][1];
    while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (map[mid][0] <= t) lo = mid; else hi = mid;
    }
    const [t0, r0] = map[lo], [t1, r1] = map[hi];
    if (r0 == null && r1 == null) return null;
    if (r0 == null) return t1 - t <= 0.06 ? r1 : null;
    if (r1 == null) return t - t0 <= 0.06 ? r0 : null;
    if (r1 < r0 || r1 - r0 > 1.0) return t - t0 < t1 - t ? r0 : r1;     // a seek inside the step
    return r0 + ((t - t0) / (t1 - t0)) * (r1 - r0);
}

function delay() {
    const timing = data.feedback && data.feedback.timing;
    return ui.offset.checked && timing && timing.confident ? timing.offset_s : 0;
}

function referenceTarget() {
    return refAt(attemptVideo.currentTime - delay());
}

// ---------------------------------------------------------------- sync loop
function syncLoop() {
    const target = referenceTarget();
    if (target == null) {
        if (!reference.paused) reference.pause();
    } else if (reference.readyState >= 1) {
        const drift = reference.currentTime - target;
        if (attemptVideo.paused) {
            if (!reference.paused) reference.pause();
            if (Math.abs(drift) > 0.04 && !reference.seeking) reference.currentTime = target;
        } else {
            const base = data.speed * attemptVideo.playbackRate;
            if (Math.abs(drift) > 0.3 && !reference.seeking) {
                reference.currentTime = target;
            } else {
                reference.playbackRate = clamp(base * (1 - drift * 0.8), base * 0.5, base * 1.5);
            }
            if (reference.paused && !reference.seeking) reference.play().catch(() => {});
        }
    }
    drawGhost(target);
    ui.time.textContent = `${fmtTime(attemptVideo.currentTime)} · reference ${target == null ? "–" : fmtTime(target)}`;
    if (Number.isFinite(attemptVideo.duration) && attemptVideo.duration > 0 && !scrubbing) {
        ui.timeline.value = Math.round((attemptVideo.currentTime / attemptVideo.duration) * 1000);
    }
    requestAnimationFrame(syncLoop);
}

function drawGhost(target) {
    const { ctx, width, height } = fitCanvas(canvas);
    ctx.clearRect(0, 0, width, height);
    if (!track || target == null || !ui.stage.classList.contains("view-overlay")) return;
    const rect = contentRect(attemptVideo, width, height);
    const [a, b] = data.section;
    const box = track.box(a || 0, b || Infinity);
    const aspect = (track.width || 16) / (track.height || 9);
    // Same placement and flip rule as during recording, so the replay shows what the learner saw.
    const flip = data.view_mirrored !== (data.follow_mode === "mirror");
    drawSkeleton(ctx, track.frameAt(target), track.connections, ghostMap(guideRect(rect), box, aspect, flip),
        { alpha: Number(ui.opacity.value), width: 5 });
}

let scrubbing = false;

async function togglePlay() {
    if (!attemptVideo.paused) { attemptVideo.pause(); return; }
    try {
        await attemptVideo.play();
    } catch (error) {
        showError(`Playback could not start: ${error.message}`);
    }
}

function jumpTo(learnerTime) {
    attemptVideo.pause();
    attemptVideo.currentTime = Math.max(0, learnerTime);
    attemptVideo.closest(".replay-panel").scrollIntoView({ behavior: "smooth", block: "start" });
}

function showError(message) {
    ui.error.textContent = message;
    ui.error.hidden = false;
}

// ---------------------------------------------------------------- feedback rendering
function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
        if (k === "class") node.className = v;
        else if (k === "text") node.textContent = v;
        else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
        else node.setAttribute(k, v);
    }
    children.flat().forEach((c) => c != null && node.append(c));
    return node;
}

const STATUS_LABEL = { close: "Close", minor: "Small difference", review: "Worth reviewing", not_visible: "Not visible", not_covered: "Not recorded" };

function renderFeedback() {
    const fb = data.feedback;
    ui.body.replaceChildren();
    ui.status.textContent = data.status;
    ui.status.className = `status status-${data.status}`;
    if (data.status === "failed") {
        ui.body.append(el("p", { class: "message error", text: data.error || "Processing failed." }));
        return;
    }
    if (!fb) return;

    ui.body.append(el("p", { class: "headline", text: fb.headline || "" }));
    const scoreCard = renderScore(data.score);
    if (scoreCard) ui.body.append(scoreCard);

    if (fb.withheld) {
        ui.body.append(el("div", { class: "card withheld" },
            el("h3", { text: "Why there is no feedback" }),
            el("ul", {}, (fb.withheld_reasons || []).map((r) => el("li", { text: r }))),
            el("p", { class: "hint", text: "This is not a low score: MoveMatch only comments on what it could see. Fix the issue and try again." })));
        return;
    }

    const grid = el("div", { class: "feedback-grid" });
    // Timing card
    const timing = fb.timing || {};
    const timingCard = el("div", { class: "card" }, el("h3", { text: "Timing" }), el("p", { text: timing.message || "–" }));
    if (timing.drift_message) timingCard.append(el("p", { text: timing.drift_message }));
    if (timing.curve && timing.curve.length) timingCard.append(costCurve(timing.curve, timing.confident ? timing.offset_s : null));
    grid.append(timingCard);

    // Tracking card
    const tracking = fb.tracking || {};
    const regions = tracking.regions || {};
    const trackingCard = el("div", { class: "card" }, el("h3", { text: "Tracking" }),
        el("p", { text: `Share of the recording in which each part was clearly visible: upper body ${pct(regions.torso)}, ` +
            `left arm ${pct(regions.left_arm)}, right arm ${pct(regions.right_arm)}, left leg ${pct(regions.left_leg)}, right leg ${pct(regions.right_leg)}.` }));
    (tracking.messages || []).forEach((m) => trackingCard.append(el("p", { class: "warn", text: m })));
    if (fb.partial_note) trackingCard.append(el("p", { class: "warn", text: fb.partial_note }));
    grid.append(trackingCard);

    // Posture summary
    const posture = fb.posture || {};
    const postureCard = el("div", { class: "card" }, el("h3", { text: "Shape at key moments" }),
        el("p", { text: `${posture.close || 0} close, ${posture.minor || 0} small differences, ${posture.review || 0} worth reviewing` +
            (posture.moments_not_visible ? `, ${posture.moments_not_visible} not visible` : "") + "." }));
    if (posture.median_abs_diff_deg != null) {
        postureCard.append(el("p", { text: `Typical joint-angle difference: ${posture.median_abs_diff_deg.toFixed(0)}° (median over compared joints).` }));
    }
    if (posture.recurring) postureCard.append(el("p", { text: posture.recurring }));
    grid.append(postureCard);
    ui.body.append(grid);

    // Moments
    const list = el("ol", { class: "moments" });
    (fb.moments || []).forEach((m) => {
        const item = el("li", { class: `moment moment-${m.status}` },
            el("div", { class: "moment-head" },
                el("span", { class: "moment-time", text: m.label }),
                el("span", { class: `status status-${m.status}`, text: STATUS_LABEL[m.status] || m.status }),
                m.learner_time_s != null
                    ? el("button", { type: "button", class: "secondary-button compact-button", text: "Show in replay",
                        onclick: () => jumpTo(m.learner_time_s) })
                    : null));
        if (m.issues && m.issues.length) item.append(el("ul", { class: "issues" }, m.issues.map((t) => el("li", { text: t }))));
        else if (m.status === "close") item.append(el("p", { class: "hint", text: "Every compared joint within 20° of the reference." }));
        const joints = (m.joints || []).filter((j) => j.diff_deg != null);
        if (joints.length) {
            item.append(el("details", {}, el("summary", { text: "Joint angles" }),
                el("table", { class: "data-table small" },
                    el("thead", {}, el("tr", {}, ["Joint", "Reference", "You", "Difference"].map((h) => el("th", { text: h })))),
                    el("tbody", {}, joints.map((j) => el("tr", { class: `joint-${j.status}` },
                        el("td", { text: `your ${j.your_joint.replace("_", " ")}` }),
                        el("td", { text: `${j.reference_deg}°` }), el("td", { text: `${j.yours_deg}°` }),
                        el("td", { text: `${j.diff_deg > 0 ? "+" : ""}${j.diff_deg}°` })))))));
        }
        list.append(item);
    });
    if ((fb.moments || []).length) {
        ui.body.append(el("h3", { text: "Moments to check" }), list);
    }
}

const COMPONENT_LABEL = { posture: "Posture at key moments", timing: "Timing", completion: "Completion" };

/** The provisional score with its breakdown: every number on it can be traced to a card below. */
function renderScore(score) {
    if (!score || score.value == null) return null;
    const comps = score.components || {};
    const rows = ["posture", "timing", "completion"].filter((k) => comps[k]).map((key) => {
        const c = comps[key];
        const fill = el("span", { class: "score-bar-fill" });
        fill.style.width = `${c.value}%`;          // CSSOM, not a style attribute: allowed by the CSP
        return el("div", { class: "score-row" },
            el("span", { class: "score-row-label", text: `${COMPONENT_LABEL[key] || key} (${Math.round(c.effective_weight * 100)}%)` }),
            el("span", { class: "score-bar", role: "img", "aria-label": `${c.value} out of 100` }, fill),
            el("span", { class: "score-row-value", text: String(c.value) }));
    });
    return el("div", { class: "card score-card" },
        el("div", { class: "score-main" },
            el("span", { class: "score-value", text: String(score.value) }),
            el("span", { class: "score-out-of", text: "/ 100" }),
            el("span", { class: "score-band", text: score.band || "" })),
        el("div", { class: "score-rows" }, rows),
        ...(score.notes || []).map((n) => el("p", { class: "hint", text: n })),
        el("p", { class: "hint", text: "Provisional score: it summarizes the measurements below and has not yet been checked against a teacher's judgment. Use it to track your own progress on this section, not to compare different dancers." }));
}

function pct(x) { return x == null ? "–" : `${Math.round(x * 100)}%`; }

/** Tiny SVG of "mismatch vs. assumed delay": the dip shows the measured delay and how clear it is. */
function costCurve(curve, best) {
    const pts = curve.filter(([, c]) => c != null);
    if (pts.length < 3) return null;
    const W = 260, H = 70, P = 6;
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys) || 1;
    const sx = (x) => P + ((x - x0) / (x1 - x0 || 1)) * (W - 2 * P);
    const sy = (y) => H - P - ((y - y0) / (y1 - y0 || 1)) * (H - 2 * P);
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H + 14}`);
    svg.setAttribute("class", "cost-curve");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Average joint-angle mismatch for each assumed delay; the lowest point is your measured delay.");
    const line = document.createElementNS(ns, "polyline");
    line.setAttribute("points", pts.map(([x, y]) => `${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join(" "));
    svg.append(line);
    const zero = document.createElementNS(ns, "line");
    zero.setAttribute("x1", sx(0)); zero.setAttribute("x2", sx(0)); zero.setAttribute("y1", P); zero.setAttribute("y2", H - P);
    zero.setAttribute("class", "zero");
    svg.append(zero);
    if (best != null) {
        const dot = document.createElementNS(ns, "circle");
        const nearest = pts.reduce((a, p) => (Math.abs(p[0] - best) < Math.abs(a[0] - best) ? p : a));
        dot.setAttribute("cx", sx(nearest[0])); dot.setAttribute("cy", sy(nearest[1])); dot.setAttribute("r", 4);
        svg.append(dot);
    }
    [[x0, "start"], [0, "middle"], [x1, "end"]].forEach(([x, anchor]) => {
        const t = document.createElementNS(ns, "text");
        t.setAttribute("x", sx(x)); t.setAttribute("y", H + 11); t.setAttribute("text-anchor", anchor);
        t.textContent = x === 0 ? "0 s" : `${x > 0 ? "+" : ""}${x.toFixed(1)} s`;
        svg.append(t);
    });
    return el("figure", { class: "curve-figure" }, svg,
        el("figcaption", { text: "Mismatch for each assumed delay (lower is better). Left = early, right = late." }));
}

function renderComparison() {
    ui.comparison.replaceChildren();
    const rows = data.comparison || [];
    if (!data.previous_attempt) {
        ui.comparison.append(el("p", { class: "hint", text: "No earlier attempt at exactly this section with feedback yet." }));
        return;
    }
    const better = rows.filter((r) => r.change === "better").length;
    const worse = rows.filter((r) => r.change === "worse").length;
    const before = data.previous_attempt.score, now = data.score && data.score.value;
    const scoreChange = before != null && now != null
        ? ` Score ${before} → ${now} (${now - before >= 0 ? "+" : ""}${now - before}).` : "";
    const intro = el("p", {}, `Compared with `, el("a", { href: data.previous_attempt.url, text: `attempt #${data.previous_attempt.id}` }),
        `: ${better} moment(s) improved, ${worse} got worse, ${rows.length - better - worse} unchanged.${scoreChange}`);
    ui.comparison.append(intro);
    if (rows.length) {
        ui.comparison.append(el("div", { class: "table-wrap" }, el("table", { class: "data-table" },
            el("thead", {}, el("tr", {}, ["Moment", "Before", "Now", "Largest difference"].map((h) => el("th", { text: h })))),
            el("tbody", {}, rows.map((r) => el("tr", { class: `change-${r.change}` },
                el("td", { text: r.label }), el("td", { text: STATUS_LABEL[r.before] || r.before }),
                el("td", { text: STATUS_LABEL[r.now] || r.now }),
                el("td", { text: `${r.before_max_diff_deg ?? "–"}° → ${r.now_max_diff_deg ?? "–"}°` })))))));
    }
}

function update(newData) {
    data = newData;
    map = data.replay_map || map;
    ui.processing.hidden = !["uploaded", "processing"].includes(data.status);
    ui.progress.value = data.progress || 0;
    const timing = data.feedback && data.feedback.timing;
    ui.offset.disabled = !(timing && timing.confident);
    ui.syncNote.textContent = map.length
        ? "The reference follows the exact moment it was showing while you recorded (including slow speed and any buffering pauses)."
        : "No synchronization data for this attempt: the reference cannot be lined up.";
    renderFeedback();
    renderComparison();
}

// ---------------------------------------------------------------- wiring
ui.play.addEventListener("click", togglePlay);
attemptVideo.addEventListener("play", () => { ui.play.textContent = "Pause"; });
attemptVideo.addEventListener("pause", () => { ui.play.textContent = "Play"; reference.pause(); });
attemptVideo.addEventListener("ended", () => reference.pause());
ui.speed.addEventListener("change", () => { attemptVideo.playbackRate = Number(ui.speed.value); });
ui.timeline.addEventListener("pointerdown", () => { scrubbing = true; });
ui.timeline.addEventListener("pointerup", () => { scrubbing = false; });
ui.timeline.addEventListener("input", () => {
    if (Number.isFinite(attemptVideo.duration)) attemptVideo.currentTime = (Number(ui.timeline.value) / 1000) * attemptVideo.duration;
});
document.querySelectorAll("input[name='replayView']").forEach((radio) => radio.addEventListener("change", () => {
    ui.stage.classList.toggle("view-side", radio.value === "side" && radio.checked);
    ui.stage.classList.toggle("view-overlay", radio.value === "overlay" && radio.checked);
}));
attemptVideo.addEventListener("error", () => showError("Your recording could not be played. If it is still processing, reload in a moment."), true);
reference.addEventListener("error", () => showError("The reference video could not be loaded."), true);
reference.preservesPitch = true;

update(data);
waitForMedia(attemptVideo, { timeout: 20000 }).catch((e) => showError(e.message));
SkeletonTrack.load(panel.dataset.landmarksUrl).then((t) => { track = t; }).catch(() => {});
requestAnimationFrame(syncLoop);

if (["uploaded", "processing"].includes(data.status)) {
    const wasRaw = !data.recording_ready;
    poll(panel.dataset.apiUrl, (next) => {
        const becameReady = wasRaw && next.recording_ready;
        update(next);
        if (becameReady && attemptVideo.paused) {           // switch to the seekable replay file
            const t = attemptVideo.currentTime;
            attemptVideo.src = `${next.recording_url}?v=replay`;
            attemptVideo.addEventListener("loadedmetadata", () => { attemptVideo.currentTime = t; }, { once: true });
        }
    }, (next) => !["uploaded", "processing"].includes(next.status));
}
