import argparse
import os
import time
import tempfile
from pathlib import Path
from functools import lru_cache

from PIL import Image  # noqa: F401 (still required by YOLO internals)
import cv2
import numpy as np
import torch
from flask import (
    Flask, render_template, request,
    url_for, Response
)
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import easyocr

# ─── INTERNAL UTILS ────────────────────────────────────────────────────────────
from mongo.db_utils import (
    put_file, get_bytes, get_gridout, mongo_connection, vacate_if_low_space
)

# ─── OCR CONFIG ───────────────────────────────────────────────────────────────
SKIP_FRAMES    = 2     # process every nth frame from video / webcam
TARGET_WIDTH   = 640   # resize width fed into the detector
MIN_CONFIDENCE = 0.5   # YOLO score threshold

# ─── APP SETUP ────────────────────────────────────────────────────────────────
app = Flask(__name__)
mongo_connection()

# ─── MODEL & OCR INIT (lazy) ──────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")

WEIGHT_PATH = Path(__file__).parent / "weights" / "best-ul-11l.pt"

@lru_cache(maxsize=1)
def get_model():
    print("[INFO] Loading YOLO weights …")
    model = YOLO(str(WEIGHT_PATH)).to(DEVICE)
    if DEVICE == "cuda":
        model.model.half()
    print("[INFO] YOLO loaded.")
    return model

reader = easyocr.Reader(["en"], gpu=(DEVICE == "cuda"))
print("[INFO] EasyOCR loaded.")

# ─── FRAME PROCESSING ─────────────────────────────────────────────────────────

def process_frame(frame: np.ndarray) -> np.ndarray:
    """Run YOLO → crop → preprocess → EasyOCR → draw boxes & text."""
    model = get_model()
    results = model(frame)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        if conf < MIN_CONFIDENCE:
            continue
        plate = frame[y1:y2, x1:x2]
        plate = cv2.resize(plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        plate_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        ocr_results = reader.readtext(
            plate_rgb, detail=1,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
        if ocr_results:
            text, _ = max(((r[1].replace(" ", "").upper(), r[2]) for r in ocr_results), key=lambda x: x[1])
        else:
            text = "NO-PLATE"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        tx, ty = x1, y2 + th + 10
        if ty + th > frame.shape[0]:
            ty = y1 - 10
        cv2.rectangle(frame, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    return frame

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Purge old GridFS data if the DB is getting full
        vacate_if_low_space()

        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")
        f = request.files["file"]
        if not f.filename:
            return render_template("index.html", error="No file selected")

        filename = secure_filename(f.filename)
        ext = filename.rsplit(".", 1)[1].lower()

        # store ORIGINAL upload in GridFS
        original_id = put_file(f.read(), filename=filename, content_type=f.mimetype, kind="original")
        print(f"[INFO] File stored in GridFS with ID {original_id}")

        # ── IMAGE ─────────────────────────────────────────────────────────────
        if ext in ("jpg", "jpeg", "png"):
            img_np = cv2.imdecode(np.frombuffer(get_bytes(original_id), np.uint8), cv2.IMREAD_COLOR)
            if img_np is None:
                return render_template("index.html", error="Failed to load image")
            out = process_frame(img_np)
            ok, buf = cv2.imencode(".jpg", out)
            if not ok:
                return render_template("index.html", error="Encoding failed")
            processed_id = put_file(buf.tobytes(), filename="processed_" + filename, content_type="image/jpeg", source=original_id)
            return render_template("index.html", content_type="image", image_path=url_for("serve_file", file_id=processed_id))

        # ── VIDEO ─────────────────────────────────────────────────────────────
        elif ext == "mp4":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
                tmp_in.write(get_bytes(original_id))
                tmp_in_path = tmp_in.name
            cap = cv2.VideoCapture(tmp_in_path)
            if not cap.isOpened():
                os.unlink(tmp_in_path)
                return render_template("index.html", error="Failed to open video")
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
            scale = TARGET_WIDTH / float(w)
            th = int(h * scale)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
                output_path = tmp_out.name
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (TARGET_WIDTH, th))
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, fps, (TARGET_WIDTH, th))
            idx, last = 0, None
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                idx += 1
                small = cv2.resize(frame, (TARGET_WIDTH, th), interpolation=cv2.INTER_AREA)
                if idx % SKIP_FRAMES == 0 or last is None:
                    last = process_frame(small)
                writer.write(last)
            cap.release()
            writer.release()
            processed_bytes = open(output_path, "rb").read()
            processed_id = put_file(processed_bytes, filename="processed_" + filename, content_type="video/mp4", source=original_id)
            os.unlink(tmp_in_path)
            os.unlink(output_path)
            return render_template("index.html", content_type="video", video_path=url_for("serve_file", file_id=processed_id))

        else:
            return render_template("index.html", error="Unsupported file format")
    return render_template("index.html")


@app.route("/file/<file_id>")
def serve_file(file_id):
    try:
        fp = get_gridout(file_id)
        return Response(fp.read(), mimetype=fp.content_type, headers={"Content-Disposition": f"inline; filename={fp.filename}"})
    except Exception as exc:
        return Response(f"Error: {exc}", status=404)


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
            h, w = frame.shape[:2]
            scale = TARGET_WIDTH / float(w)
            th = int(h * scale)
            small = cv2.resize(frame, (TARGET_WIDTH, th), interpolation=cv2.INTER_AREA)
            if idx % SKIP_FRAMES == 0 or last is None:
                last = process_frame(small)
            ok, buf = cv2.imencode('.jpg', last)
            if not ok:
                continue
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes()
                   + b'\r\n')   
            
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')
        
        
@app.route("/healthz")
def healthz():
    return "OK", 200    
            
# ─── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", 5000))  # picks $PORT on Render, 5000 locally
    )
    args = parser.parse_args()

    # optional scratch dirs
    # os.makedirs("uploads", exist_ok=True)
    # os.makedirs(os.path.join("static", "processed_videos"), exist_ok=True)

    app.run(host="0.0.0.0", port=args.port, debug=False)
