import os
import gc
import time
import tempfile
from pathlib import Path
from functools import lru_cache

from PIL import Image
import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, url_for, Response
from werkzeug.utils import secure_filename

# Global variables for lazy loading
_model = None
_reader = None

# Internal utils
from mongo.db_utils import (
    put_file, get_bytes, get_gridout, mongo_connection, vacate_if_low_space
)

# OCR CONFIG
SKIP_FRAMES = 3  # Increased to reduce processing
TARGET_WIDTH = 480  # Reduced for memory
MIN_CONFIDENCE = 0.5

# APP SETUP
app = Flask(__name__)
mongo_connection()

# Force CPU usage for memory efficiency
DEVICE = "cpu"
print(f"[INFO] Using device: {DEVICE}")

WEIGHT_PATH = Path(__file__).parent / "weights" / "best-ul-11l.pt"

def get_model():
    """Lazy load YOLO model only when needed."""
    global _model
    if _model is None:
        print("[INFO] Loading YOLO weights...")
        try:
            from ultralytics import YOLO
            gc.collect()  # Clean memory before loading
            _model = YOLO(str(WEIGHT_PATH))
            print("[INFO] YOLO loaded successfully.")
            gc.collect()  # Clean memory after loading
        except Exception as e:
            print(f"[ERROR] Failed to load YOLO: {e}")
            raise
    return _model

def get_reader():
    """Lazy load EasyOCR reader only when needed."""
    global _reader
    if _reader is None:
        print("[INFO] Loading EasyOCR...")
        try:
            import easyocr
            gc.collect()  # Clean memory before loading
            _reader = easyocr.Reader(["en"], gpu=False, download_enabled=True)
            print("[INFO] EasyOCR loaded successfully.")
            gc.collect()  # Clean memory after loading
        except Exception as e:
            print(f"[ERROR] Failed to load EasyOCR: {e}")
            raise
    return _reader

def process_frame(frame: np.ndarray) -> np.ndarray:
    """Process frame with memory optimization."""
    try:
        # Resize frame early to save memory
        height, width = frame.shape[:2]
        if width > TARGET_WIDTH:
            scale = TARGET_WIDTH / width
            new_height = int(height * scale)
            frame = cv2.resize(frame, (TARGET_WIDTH, new_height))
        
        # Load models
        model = get_model()
        reader = get_reader()
        
        # Run YOLO detection
        results = model(frame)[0]
        
        for box in results.boxes:
            if box.conf[0] < MIN_CONFIDENCE:
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Extract and process license plate region
            plate = frame[y1:y2, x1:x2]
            if plate.size == 0:
                continue
            
            # Minimal preprocessing to save memory
            plate = cv2.resize(plate, None, fx=1.5, fy=1.5)
            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            plate_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            
            # OCR processing
            try:
                ocr_results = reader.readtext(
                    plate_rgb, detail=1,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                )
                text = max((r[1].replace(" ", "").upper() for r in ocr_results), 
                          key=len, default="NO-PLATE")
            except Exception:
                text = "OCR-ERROR"
            
            # Draw bounding box and text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 0), 2)
        
        return frame
        
    except Exception as e:
        print(f"[ERROR] Frame processing failed: {e}")
        return frame
    finally:
        gc.collect()

# Memory cleanup middleware
@app.before_request
def before_request():
    gc.collect()

@app.after_request  
def after_request(response):
    gc.collect()
    return response

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
        ext = filename.rsplit(".", 1)[1].lower()
        
        # Read and validate file
        file_data = f.read()
        if len(file_data) > 10 * 1024 * 1024:  # 10MB limit
            return render_template("index.html", error="File too large (max 10MB)")

        # Store original
        original_id = put_file(file_data, filename=filename, 
                              content_type=f.mimetype, kind="original")

        try:
            if ext in ("jpg", "jpeg", "png"):
                # Process image
                img_np = cv2.imdecode(np.frombuffer(file_data, np.uint8), 
                                    cv2.IMREAD_COLOR)
                if img_np is None:
                    return render_template("index.html", error="Invalid image")
                
                processed = process_frame(img_np)
                
                # Encode with compression
                ok, buf = cv2.imencode(".jpg", processed, 
                                     [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    return render_template("index.html", error="Encoding failed")
                
                processed_id = put_file(buf.tobytes(), 
                                      filename="processed_" + filename,
                                      content_type="image/jpeg", 
                                      source=original_id)
                
                # Cleanup
                del img_np, processed, buf
                gc.collect()
                
                return render_template("index.html", content_type="image",
                                     image_path=url_for("serve_file", 
                                                       file_id=processed_id))
            else:
                return render_template("index.html", 
                                     error="Only JPG/PNG images supported")
                
        except Exception as e:
            print(f"[ERROR] Processing failed: {e}")
            return render_template("index.html", 
                                 error=f"Processing failed: {str(e)}")
        finally:
            gc.collect()
    
    return render_template("index.html")

@app.route("/file/<file_id>")
def serve_file(file_id):
    try:
        fp = get_gridout(file_id)
        return Response(fp.read(), mimetype=fp.content_type,
                       headers={"Content-Disposition": f"inline; filename={fp.filename}"})
    except Exception as exc:
        return Response(f"Error: {exc}", status=404)

@app.route("/healthz")
def healthz():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    if not os.environ.get('RENDER'):
        app.run(host="0.0.0.0", port=port, debug=False)