// Reference upload: check size and length before sending, show upload progress, follow the redirect.
import { $, csrfToken } from "./common.js";

const form = $("uploadForm");
if (form) {
    const fileInput = $("videoFile");
    const button = $("uploadButton");
    const status = $("uploadStatus");
    const bar = $("uploadProgress");
    const maxMb = Number(form.dataset.maxMb);
    const maxSeconds = Number(form.dataset.maxSeconds);

    const durationOf = (file) => new Promise((resolve) => {
        const probe = document.createElement("video");
        const url = URL.createObjectURL(file);
        const finish = (value) => { URL.revokeObjectURL(url); resolve(value); };
        probe.preload = "metadata";
        probe.onloadedmetadata = () => finish(probe.duration);
        probe.onerror = () => finish(NaN);        // let the server decide (e.g. AVI the browser can't read)
        setTimeout(() => finish(NaN), 5000);
        probe.src = url;
    });

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const file = fileInput.files[0];
        if (!file) return;
        button.disabled = true;
        if (file.size > maxMb * 1024 * 1024) {
            status.textContent = `This file is ${(file.size / 1048576).toFixed(0)} MB; the limit is ${maxMb} MB.`;
            button.disabled = false;
            return;
        }
        const duration = await durationOf(file);
        if (Number.isFinite(duration) && duration > maxSeconds + 0.5) {
            status.textContent = `This video is ${duration.toFixed(0)} s long; trim it to at most ${maxSeconds.toFixed(0)} s (the part you want to learn).`;
            button.disabled = false;
            return;
        }
        const xhr = new XMLHttpRequest();
        xhr.open("POST", form.action);
        xhr.setRequestHeader("X-CSRFToken", csrfToken());
        bar.hidden = false;
        status.textContent = "Uploading…";
        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) {
                bar.value = e.loaded / e.total;
                status.textContent = `Uploading… ${Math.round(100 * e.loaded / e.total)}%`;
            }
        };
        xhr.onload = () => { window.location.assign(xhr.responseURL || form.action); };
        xhr.onerror = () => {
            status.textContent = "The upload failed (network error). Check your connection and try again.";
            button.disabled = false;
            bar.hidden = true;
        };
        xhr.send(new FormData(form));
    });
}
