import argparse
import os
import time

from PIL import Image
import cv2
import numpy as np
import torch
from flask import (
    Flask, render_template, request,
    send_from_directory, url_for, Response
)
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import easyocr

# ─── OCR CONFIG ─────────────────────────────────────────────────────────────────────

SKIP_FRAMES    = 2   
TARGET_WIDTH   = 640  
MIN_CONFIDENCE = 0.5  
# ─── APP SETUP ─────────────────────────────────────────────────────────────────

app = Flask(__name__)

# ─── MODEL & OCR INIT ────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")

model = YOLO("./weights/best-ul-9c.pt").to(DEVICE)
if DEVICE == "cuda":
    model.model.half()   # half precision speeds it up ~2×
print("[INFO] YOLO loaded.")

reader = easyocr.Reader(["en"], gpu=(DEVICE == "cuda"))
print("[INFO] EasyOCR loaded.")

# ─── FRAME PROCESSING ─────────────────────────────────────────────────────────────

def process_frame(frame: np.ndarray) -> np.ndarray:
    """Run YOLO → crop → preprocess → EasyOCR → draw boxes & text."""
    results = model(frame)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        if conf < MIN_CONFIDENCE:
            continue

        # 1) Crop & upscale
        plate = frame[y1:y2, x1:x2]
        plate = cv2.resize(plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # 2) Grayscale & denoise
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # 3) Back to RGB for EasyOCR
        plate_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # 4) OCR with allowlist
        ocr_results = reader.readtext(
            plate_rgb,
            detail=1,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        # 5) Pick best result
        if ocr_results:
            text, _ = max(
                ((res[1].replace(" ", "").upper(), res[2]) for res in ocr_results),
                key=lambda x: x[1]
            )
        else:
            text = "NO-PLATE"

        # 6) Draw box + label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        tw, th = cv2.getTextSize(text,
                                 cv2.FONT_HERSHEY_SIMPLEX,
                                 0.6, 2)[0]
        tx, ty = x1, y2 + th + 10
        if ty + th > frame.shape[0]:
            ty = y1 - 10
        cv2.rectangle(frame,
                      (tx-2, ty-th-2),
                      (tx+tw+2, ty+2),
                      (0,0,0), -1)
        cv2.putText(frame, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255,255,255), 2, lineType=cv2.LINE_AA)

    return frame

# ─── ROUTE: HOME & UPLOAD ────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # 1) Save upload
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")
        f = request.files["file"]
        if not f.filename:
            return render_template("index.html", error="No file selected")

        UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        in_path = os.path.join(UPLOAD_DIR, secure_filename(f.filename))
        f.save(in_path)

        ext = f.filename.rsplit(".", 1)[1].lower()

        # ── IMAGE CASE ────────────────────────────────────────────────────────────
        if ext in ("jpg","jpeg","png"):
            img = cv2.imread(in_path)
            if img is None:
                return render_template("index.html", error="Failed to load image")

            out = process_frame(img)
            out_dir = os.path.join("runs","detect", f"exp_{int(time.time())}")
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, secure_filename(f.filename))
            cv2.imwrite(out_file, out)

            image_url = url_for("serve_image",
                                subfolder=os.path.basename(out_dir),
                                filename=os.path.basename(out_file))
            return render_template("index.html",
                                   content_type="image",
                                   image_path=image_url)

        # ── VIDEO CASE ────────────────────────────────────────────────────────────
        elif ext == "mp4":
            cap = cv2.VideoCapture(in_path)
            if not cap.isOpened():
                return render_template("index.html", error="Failed to open video")

            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

            scale = TARGET_WIDTH / float(w)
            th    = int(h * scale)

            PROC_DIR    = os.path.join("static","processed_videos")
            os.makedirs(PROC_DIR, exist_ok=True)
            ts           = int(time.time())
            output_name  = f"processed_video_{ts}.mp4"
            output_path  = os.path.join(PROC_DIR, output_name)

            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(output_path, fourcc, fps,
                                     (TARGET_WIDTH, th))
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, fps,
                                         (TARGET_WIDTH, th))

            idx, last = 0, None
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                idx += 1

                small = cv2.resize(frame,
                                   (TARGET_WIDTH, th),
                                   interpolation=cv2.INTER_AREA)
                if idx % SKIP_FRAMES == 0 or last is None:
                    last = process_frame(small)
                writer.write(last)

            cap.release()
            writer.release()

            video_url = url_for("static",
                                filename=f"processed_videos/{output_name}")
            return render_template("index.html",
                                   content_type="video",
                                   video_path=video_url)

        else:
            return render_template("index.html",
                                   error="Unsupported file format")

    return render_template("index.html")

# ─── ROUTE: SERVE IMAGE ─────────────────────────────────────────────────────────

@app.route("/runs/detect/<subfolder>/<filename>")
def serve_image(subfolder, filename):
    return send_from_directory(
        os.path.join("runs","detect",subfolder),
        filename
    )

# ─── ROUTE: LIVE WEBCAM FEED ────────────────────────────────────────────────────

@app.route("/webcam_feed")
def webcam_feed():
    cap = cv2.VideoCapture(0)
    last = None
    idx = 0

    def gen():
        nonlocal last, idx
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1

            # downscale
            h, w = frame.shape[:2]
            scale = TARGET_WIDTH / float(w)
            th = int(h * scale)
            small = cv2.resize(frame, (TARGET_WIDTH, th),
                               interpolation=cv2.INTER_AREA)

            # skip logic
            if idx % SKIP_FRAMES == 0 or last is None:
                last = process_frame(small)

            # encode JPEG
            ret2, buf = cv2.imencode('.jpg', last)
            if not ret2:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'
                   + buf.tobytes() + b'\r\n')

    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ─── MAIN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()

    os.makedirs("uploads", exist_ok=True)
    os.makedirs(os.path.join("static","processed_videos"), exist_ok=True)
    os.makedirs(os.path.join("runs","detect"), exist_ok=True)

    app.run(host="0.0.0.0", port=args.port, debug=False)
