import os
import gc
import base64
import tempfile
from pathlib import Path
from functools import lru_cache

from PIL import Image  # required by some YOLO internals
import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, url_for, Response
from werkzeug.utils import secure_filename

# ─── INTERNAL UTILS (yours) ───────────────────────────────────────────────────
from mongo.db_utils import (
    put_file, get_bytes, get_gridout, mongo_connection, vacate_if_low_space
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SKIP_FRAMES    = 2      # process every nth frame for video
TARGET_WIDTH   = 640    # resize width fed into detector
MIN_CONFIDENCE = 0.5    # YOLO score threshold
MAX_UPLOAD_MB  = 100    # hard cap for uploads (MB)

# ─── APP SETUP ────────────────────────────────────────────────────────────────
app = Flask(__name__)
mongo_connection()

# Force CPU on cloud; allow CUDA locally if present
DEVICE = "cuda" if (not os.environ.get("RENDER") and torch.cuda.is_available()) else "cpu"
print(f"[INFO] Using device: {DEVICE}")

# Path to your YOLOv11 weights (ensure this file is baked into the image)
WEIGHT_PATH = Path(__file__).parent / "weights" / "best-ul-11l.pt"


# ──── YOLOv9 'C3k2' compatibility shim ─────────────────────────────────────────
try:
    from ultralytics.nn.modules import block as _ultra_block
    if not hasattr(_ultra_block, "C3k2"):
        # Map C3k2 -> C2f so torch.load can unpickle v9 checkpoints
        class C3k2(_ultra_block.C2f):  # type: ignore
            pass
        _ultra_block.C3k2 = C3k2
        print("[INFO] Registered C3k2 shim (alias of C2f).")
except Exception as e:
    print("[WARN] Could not register C3k2 shim:", e)


# ─── LAZY LOADERS ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_model():
    """Lazy load YOLO model (YOLOv11 requires ultralytics>=8.3.x)."""
    from ultralytics import YOLO
    print("[INFO] Loading YOLO weights...")
    gc.collect()
    model = YOLO(str(WEIGHT_PATH))
    if DEVICE == "cuda":
        model.model.to("cuda")
        model.model.half()  # half precision only on CUDA
    print("[INFO] YOLO loaded.")
    gc.collect()
    return model

@lru_cache(maxsize=1)
def get_reader():
    """Lazy load EasyOCR reader."""
    print("[INFO] Loading EasyOCR...")
    gc.collect()
    import easyocr
    reader = easyocr.Reader(
    ["en"],
    gpu=(DEVICE == "cuda"),
    download_enabled=False,                     # don't try to fetch at runtime
    model_storage_directory=os.environ.get("EASYOCR_DIR", "/app/.easyocr")
)
    print("[INFO] EasyOCR loaded.")
    gc.collect()
    return reader

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def _resize_to_target(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w > TARGET_WIDTH:
        scale = TARGET_WIDTH / float(w)
        frame = cv2.resize(frame, (TARGET_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame

def _annotate(frame: np.ndarray, x1, y1, x2, y2, text: str):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
    tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    tx, ty = x1, y2 + th + 10
    if ty + th > frame.shape[0]:
        ty = max(th + 10, y1 - 10)
    cv2.rectangle(frame, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, lineType=cv2.LINE_AA)

def _jpeg_bytes(img: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()

# ─── CORE INFERENCE ───────────────────────────────────────────────────────────
def process_frame(frame: np.ndarray) -> np.ndarray:
    """
    Run YOLO → crop → preprocess → EasyOCR → draw boxes & text.
    Designed to be CPU-friendly for cloud hosts.
    """
    try:
        frame = _resize_to_target(frame)
        model  = get_model()
        reader = get_reader()

        results = model(frame)[0]
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < MIN_CONFIDENCE:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # clip to bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue

            plate = frame[y1:y2, x1:x2]
            if plate.size == 0:
                continue

            # light preprocessing for OCR
            plate = cv2.resize(plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            gray  = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            gray  = cv2.bilateralFilter(gray, 11, 17, 17)
            plate_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

            # OCR
            try:
                ocr_results = reader.readtext(
                    plate_rgb, detail=1,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                )
                if ocr_results:
                    text, _ = max(
                        ((r[1].replace(" ", "").upper(), r[2]) for r in ocr_results),
                        key=lambda x: x[1]
                    )
                else:
                    text = "NO-PLATE"
            except Exception:
                text = "OCR-ERROR"

            _annotate(frame, x1, y1, x2, y2, text)

        return frame

    except Exception as e:
        print(f"[ERROR] Frame processing failed: {e}")
        return frame
    finally:
        gc.collect()

# ─── REQUEST LIFECYCLE GC ─────────────────────────────────────────────────────
@app.before_request
def _before():
    gc.collect()

@app.after_request
def _after(resp):
    gc.collect()
    return resp

# ─── ROUTES: UI + UPLOADS ─────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        gc.collect()
        vacate_if_low_space()

        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")
        f = request.files["file"]
        if not f.filename:
            return render_template("index.html", error="No file selected")

        filename = secure_filename(f.filename)
        if "." not in filename:
            return render_template("index.html", error="Unsupported file format")
        ext = filename.rsplit(".", 1)[1].lower()

        data = f.read()
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            return render_template("index.html", error=f"File too large (max {MAX_UPLOAD_MB}MB)")

        original_id = put_file(data, filename=filename, content_type=f.mimetype, kind="original")
        print(f"[INFO] File stored in GridFS with ID {original_id}")

        # Images
        if ext in ("jpg", "jpeg", "png"):
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return render_template("index.html", error="Failed to read image")

            out = process_frame(img)
            processed_id = put_file(_jpeg_bytes(out), filename="processed_" + filename,
                                    content_type="image/jpeg", source=original_id)
            del img, out
            gc.collect()
            return render_template("index.html", content_type="image",
                                   image_path=url_for("serve_file", file_id=processed_id))

        # Video (mp4)
        elif ext == "mp4":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
                tmp_in.write(data)
                tmp_in_path = tmp_in.name

            cap = cv2.VideoCapture(tmp_in_path)
            if not cap.isOpened():
                os.unlink(tmp_in_path)
                return render_template("index.html", error="Failed to open video")

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
            scale = TARGET_WIDTH / float(w)
            out_h = int(h * scale)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
                output_path = tmp_out.name

            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (TARGET_WIDTH, out_h))
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, fps, (TARGET_WIDTH, out_h))

            idx, last = 0, None
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    idx += 1
                    small = cv2.resize(frame, (TARGET_WIDTH, out_h), interpolation=cv2.INTER_AREA)
                    if idx % SKIP_FRAMES == 0 or last is None:
                        last = process_frame(small)
                    writer.write(last)
            finally:
                cap.release()
                writer.release()

            processed_bytes = open(output_path, "rb").read()
            processed_id = put_file(processed_bytes, filename="processed_" + filename,
                                    content_type="video/mp4", source=original_id)

            try: os.unlink(tmp_in_path)
            except: pass
            try: os.unlink(output_path)
            except: pass
            gc.collect()

            return render_template("index.html", content_type="video",
                                   video_path=url_for("serve_file", file_id=processed_id))

        else:
            return render_template("index.html", error="Unsupported file format")

    return render_template("index.html")

@app.route("/file/<file_id>")
def serve_file(file_id):
    try:
        fp = get_gridout(file_id)
        return Response(fp.read(), mimetype=fp.content_type,
                        headers={"Content-Disposition": f"inline; filename={fp.filename}"})
    except Exception as exc:
        return Response(f"Error: {exc}", status=404)

# ─── NEW: Browser-pushed webcam frames → annotated JPEG back ──────────────────
@app.route("/infer_frame", methods=["POST"])
def infer_frame():
    """
    Accept a single frame from the browser and return an annotated JPEG.
    Supports:
      - multipart/form-data with 'frame' (Blob)
      - application/json with base64 data URL in 'image'
    Returns: image/jpeg bytes
    """
    try:
        img_bytes = None

        if "frame" in request.files:
            img_bytes = request.files["frame"].read()

        if img_bytes is None and request.is_json:
            data = request.get_json(silent=True) or {}
            b64 = data.get("image", "")
            if b64.startswith("data:"):
                b64 = b64.split(",", 1)[-1]
            if b64:
                img_bytes = base64.b64decode(b64)

        if not img_bytes:
            return Response("No frame provided", status=400)

        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return Response("Bad image", status=400)

        out = process_frame(img)
        jpg = _jpeg_bytes(out, quality=80)
        return Response(jpg, mimetype="image/jpeg")

    except Exception as e:
        print(f"[ERROR] /infer_frame failed: {e}")
        return Response(f"Error: {e}", status=500)
    finally:
        gc.collect()

# Optional: local-only MJPEG endpoint retained for dev boxes
@app.route("/webcam_feed")
def webcam_feed():
    if os.environ.get("RENDER"):
        return Response("Webcam is unavailable on server deployments.", status=501)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return Response("Failed to access local webcam.", status=500)

    last, idx = None, 0
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
            ok, buf = cv2.imencode(".jpg", last)
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/healthz")
def healthz():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
