FROM python:3.11-slim AS app

# --- system deps (minimal for OpenCV headless etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libgthread-2.0-0 \
    libfontconfig1 libgl1-mesa-dev libglib2.0-dev wget ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- env
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/hf \
    TORCH_HOME=/app/.cache/torch \
    YOLO_CONFIG_DIR=/app/.cache/ultralytics \
    # where we will bake EasyOCR models:
    EASYOCR_DIR=/app/.easyocr

# create caches + ocr dir (persisted in image)
RUN mkdir -p $HF_HOME $TORCH_HOME $YOLO_CONFIG_DIR $EASYOCR_DIR

# copy deps and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# copy app code (including your pre-trained YOLO weights under /app/weights)
COPY . .

# ---- PRELOAD MODELS AT BUILD TIME ----
# 1) Ensure YOLO can load your local weights (no net download needed)
#    This just imports the model once so Ultralytics writes its settings file.
RUN python - <<'PY'
from ultralytics import YOLO
from pathlib import Path
w = Path('weights') / 'best-ul-11l.pt'  # keep this name to match app.py
assert w.exists(), f"Missing weights at {w}"
# Load once on CPU (this does NOT download anything if weights are local)
_ = YOLO(str(w))
print("YOLO weights present and loadable.")
PY

# 2) Download EasyOCR detection/recognition models into /app/.easyocr
RUN python - <<'PY'
import easyocr, os
target = os.environ.get("EASYOCR_DIR", "/app/.easyocr")
reader = easyocr.Reader(['en'], gpu=False, download_enabled=True,
                        model_storage_directory=target)
print("EasyOCR models downloaded to:", target)
PY

# expose port for Render
EXPOSE 5000

# gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "300", "--max-requests", "50", "--max-requests-jitter", "10","--preload", "app:app"]
