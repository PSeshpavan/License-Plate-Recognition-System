let liveStreamActive = false;
let cameraStream = null;
let inferLoopTimer = null;
let inflight = false; // avoid overlapping /infer_frame calls

function showProcessingMessage() {
  const fileInput = document.getElementById("file-input");
  const uploadBtn = document.getElementById("upload-btn");
  const placeholderContent = document.getElementById("placeholder-content");
  const processingContent = document.getElementById("processing-content");
  const processingText = document.getElementById("processing-text");

  if (!fileInput || !uploadBtn) return;
  if (fileInput.files.length === 0) return;

  placeholderContent.style.display = "none";
  processingContent.style.display = "block";

  const fileType = fileInput.files[0].type;
  if (fileType.startsWith("image/")) {
    processingText.textContent = "Please wait, processing your image…";
  } else if (fileType.startsWith("video/")) {
    processingText.textContent = "Please wait, processing your video…";
  } else {
    processingText.textContent = "Processing your file…";
  }

  uploadBtn.disabled = true;
  uploadBtn.textContent = "Processing…";
}

function resetFormState() {
  const uploadBtn = document.getElementById("upload-btn");
  const placeholderContent = document.getElementById("placeholder-content");
  const processingContent = document.getElementById("processing-content");
  if (uploadBtn) {
    uploadBtn.disabled = false;
    uploadBtn.textContent = "Upload & Detect";
  }
  if (placeholderContent) placeholderContent.style.display = "flex";
  if (processingContent) processingContent.style.display = "none";
}

document.addEventListener("DOMContentLoaded", () => {
  resetFormState();
  if (liveStreamActive) stopLiveDetection();
});
window.addEventListener("beforeunload", () => {
  if (liveStreamActive) stopLiveDetection();
});

function startLiveDetection() {
  const liveContainer = document.getElementById("live-detection-container");
  const startBtn = document.getElementById("start-live-btn");
  const stopBtn = document.getElementById("stop-live-btn");
  const cam = document.getElementById("cam");
  const img = document.getElementById("live-stream");
  if (!liveContainer || !cam || !img) return;

  navigator.mediaDevices.getUserMedia({ video: { width: 640 }, audio: false })
    .then(stream => {
      cameraStream = stream;
      cam.srcObject = stream;
      liveContainer.classList.add("live-detection-active");
      startBtn.disabled = true;
      stopBtn.disabled = false;
      liveStreamActive = true;

      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d", { willReadFrequently: true });

      const tick = async () => {
        if (!liveStreamActive || !cam.videoWidth || inflight) return;
        inflight = true;
        try {
          const w = cam.videoWidth, h = cam.videoHeight;
          canvas.width = w; canvas.height = h;
          ctx.drawImage(cam, 0, 0, w, h);
          const dataUrl = canvas.toDataURL("image/jpeg", 0.7);

          const res = await fetch("/infer_frame", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: dataUrl })
          });

          if (!res.ok) throw new Error("Inference error");
          const blob = await res.blob();
          const objURL = URL.createObjectURL(blob);
          img.onload = () => URL.revokeObjectURL(objURL);
          img.src = objURL;
        } catch (err) {
          console.error(err);
          stopLiveDetection();
          alert("Live detection stopped: " + err.message);
        } finally {
          inflight = false;
        }
      };

      // modest FPS for CPU hosts
      inferLoopTimer = setInterval(tick, 150);
    })
    .catch(err => {
      console.error("Camera error:", err);
      alert("Camera access denied or unsupported");
    });
}

function stopLiveDetection() {
  const liveContainer = document.getElementById("live-detection-container");
  const startBtn = document.getElementById("start-live-btn");
  const stopBtn = document.getElementById("stop-live-btn");
  const img = document.getElementById("live-stream");
  if (!liveContainer) return;

  if (inferLoopTimer) clearInterval(inferLoopTimer);
  inferLoopTimer = null;
  liveStreamActive = false;

  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
  if (img) img.src = "";

  liveContainer.classList.remove("live-detection-active");
  startBtn.disabled = false;
  stopBtn.disabled = true;
}

/* Manual camera capture (single-frame upload to "/") */
function startCamera() {
  const video = document.getElementById("live-video");
  const container = document.getElementById("live-video-container");
  if (!video || !navigator.mediaDevices) return;

  navigator.mediaDevices
    .getUserMedia({ video: { facingMode: "environment" }, audio: false })
    .then((stream) => {
      cameraStream = stream;
      video.srcObject = stream;
      container.style.display = "block";
    })
    .catch((err) => {
      console.error("Camera error:", err);
      alert("Camera access denied or unsupported");
    });
}

function stopCamera() {
  const container = document.getElementById("live-video-container");
  if (cameraStream) {
    cameraStream.getTracks().forEach((t) => t.stop());
    cameraStream = null;
  }
  container.style.display = "none";
}

function captureFrame() {
  const video = document.getElementById("live-video");
  if (!video) return;
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);

  canvas.toBlob(
    (blob) => {
      const form = new FormData();
      form.append("file", blob, "capture.jpg");
      fetch("/", { method: "POST", body: form })
        .then((r) => r.text())
        .then((html) => (document.body.innerHTML = html))
        .catch((err) => {
          console.error("Upload error:", err);
          alert("Failed to process captured frame");
        })
        .finally(() => {
          stopCamera();
        });
    },
    "image/jpeg",
    0.8
  );
}

/* Video helpers */
function videoError() { alert("Video playback error."); }
function videoLoadStart() { console.log("Video loading…"); }
function videoCanPlay() { console.log("Video ready to play"); }
