let cameraStream = null;
let liveDetectionActive = false;


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
  if (liveDetectionActive) stopLiveDetection();
});
window.addEventListener("beforeunload", () => {
  if (liveDetectionActive) stopLiveDetection();
});

function startLiveDetection() {
  const liveContainer = document.getElementById("live-detection-container");
  const liveStream = document.getElementById("live-stream");
  const startBtn = document.getElementById("start-live-btn");
  const stopBtn = document.getElementById("stop-live-btn");
  if (!liveContainer || !liveStream) return;

  liveContainer.classList.add("live-detection-active");
  liveStream.src = "/webcam_feed";
  startBtn.disabled = true;
  stopBtn.disabled = false;
  liveDetectionActive = true;

  liveStream.onerror = () => {
    showError(
      "Failed to connect to webcam feed. Make sure your camera is available."
    );
    stopLiveDetection();
  };
}

function stopLiveDetection() {
  const liveContainer = document.getElementById("live-detection-container");
  const liveStream = document.getElementById("live-stream");
  const startBtn = document.getElementById("start-live-btn");
  const stopBtn = document.getElementById("stop-live-btn");
  if (!liveContainer || !liveStream) return;

  liveContainer.classList.remove("live-detection-active");
  liveStream.src = "";
  startBtn.disabled = false;
  stopBtn.disabled = true;
  liveDetectionActive = false;
}

function startCamera() {
  const video = document.getElementById("live-video");
  const container = document.getElementById("live-video-container");
  if (!video || !navigator.mediaDevices) return;

  navigator.mediaDevices
    .getUserMedia({ video: { facingMode: "environment" } })
    .then((stream) => {
      cameraStream = stream;
      video.srcObject = stream;
      container.style.display = "block";
    })
    .catch((err) => {
      console.error("Camera error:", err);
      showError("Camera access denied or unsupported");
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
          showError("Failed to process captured frame");
        })
        .finally(() => {
          stopCamera();
        });
    },
    "image/jpeg",
    0.8
  );
}

function videoError(v) {
  alert("Video playback error.");
}
function videoLoadStart() {
  console.log("Video loading…");
}
function videoCanPlay() {
  console.log("Video ready to play");
}
