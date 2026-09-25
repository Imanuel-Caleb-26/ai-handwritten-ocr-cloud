"use strict";

// ─── State ────────────────────────────────────────────────────────────────────
let currentId = null;

// ─── Elements ─────────────────────────────────────────────────────────────────
const dropZone      = document.getElementById("drop-zone");
const fileInput     = document.getElementById("file-input");
const browseBtn     = document.getElementById("browse-btn");
const resultSection = document.getElementById("result-section");
const imagePreview  = document.getElementById("image-preview");
const outputText    = document.getElementById("output-text");
const ocrStatus     = document.getElementById("ocr-status");
const btnDocx       = document.getElementById("btn-docx");
const btnPdf        = document.getElementById("btn-pdf");
const btnCopy       = document.getElementById("btn-copy");
const btnReset      = document.getElementById("btn-reset");
const progressWrap  = document.getElementById("progress-wrap");
const progressBar   = document.getElementById("progress-bar");

// ─── Upload & drag-drop ───────────────────────────────────────────────────────
browseBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") fileInput.click(); });
fileInput.addEventListener("change", () => fileInput.files[0] && handleFile(fileInput.files[0]));

dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) handleFile(f);
});

// ─── Progress helpers ─────────────────────────────────────────────────────────
let progressTimer = null;
function startProgress() {
    progressBar.style.width = "0%";
    progressWrap.style.display = "block";
    let pct = 0;
    progressTimer = setInterval(() => {
        // Slowly fill to 90% while waiting for server
        if (pct < 90) { pct += Math.random() * 3; progressBar.style.width = Math.min(pct, 90) + "%"; }
    }, 400);
}
function completeProgress() {
    clearInterval(progressTimer);
    progressBar.style.width = "100%";
    setTimeout(() => { progressWrap.style.display = "none"; }, 600);
}

// ─── Main handler ─────────────────────────────────────────────────────────────
async function handleFile(file) {
    // Preview
    const reader = new FileReader();
    reader.onload = (e) => { imagePreview.src = e.target.result; };
    reader.readAsDataURL(file);

    // Show result panel
    dropZone.classList.add("hidden");
    resultSection.classList.remove("hidden");
    outputText.value = "";
    ocrStatus.textContent = "Processing…";
    ocrStatus.className = "status";
    btnDocx.disabled = true;
    btnPdf.disabled  = true;
    btnCopy.disabled = true;
    currentId = null;

    startProgress();

    const formData = new FormData();
    formData.append("image", file);

    try {
        const res  = await fetch("/upload", { method: "POST", body: formData });
        const data = await res.json();

        completeProgress();

        if (!res.ok) {
            ocrStatus.textContent = "Error";
            outputText.value = "Error: " + (data.error || "Unknown error");
            return;
        }

        currentId = data.id;
        outputText.value = data.text || "";
        ocrStatus.textContent = "Done ✔";
        ocrStatus.className = "status done";
        btnDocx.disabled = false;
        btnPdf.disabled  = false;
        btnCopy.disabled = false;

    } catch (err) {
        completeProgress();
        ocrStatus.textContent = "Failed";
        outputText.value = "Network error: " + err.message;
    }
}

// ─── Downloads ────────────────────────────────────────────────────────────────
async function syncText() {
    if (!currentId) return;
    try {
        await fetch("/update_text", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ id: currentId, text: outputText.value })
        });
    } catch (e) {
        console.error("Failed to sync text", e);
    }
}

btnDocx.addEventListener("click", async () => {
    await syncText();
    if (currentId) window.location = `/download/docx/${currentId}`;
});
btnPdf.addEventListener("click", async () => {
    await syncText();
    if (currentId) window.location = `/download/pdf/${currentId}`;
});

// ─── Copy to clipboard ────────────────────────────────────────────────────────
btnCopy.addEventListener("click", async () => {
    if (!outputText.value) return;
    try {
        await navigator.clipboard.writeText(outputText.value);
        btnCopy.title = "Copied!";
        btnCopy.style.color = "var(--green)";
        setTimeout(() => {
            btnCopy.title = "Copy to clipboard";
            btnCopy.style.color = "";
        }, 2000);
    } catch {
        outputText.select();
        document.execCommand("copy");
    }
});

// ─── Reset ────────────────────────────────────────────────────────────────────
btnReset.addEventListener("click", () => {
    currentId = null;
    fileInput.value = "";
    outputText.value = "";
    imagePreview.src = "";
    ocrStatus.textContent = "Processing…";
    ocrStatus.className = "status";
    btnDocx.disabled = true;
    btnPdf.disabled  = true;
    btnCopy.disabled = true;
    resultSection.classList.add("hidden");
    dropZone.classList.remove("hidden");
});
