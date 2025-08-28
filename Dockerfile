# ========= Base image =========
FROM python:3.11-slim

# ========= Environment =========
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Where EasyOCR models will live (baked at build time)
    EASYOCR_DIR=/app/.easyocr \
    # Ensure Ultralytics cache stays inside the image/container
    ULTRALYTICS_CACHE_DIR=/app/.cache/ultralytics

# ========= System deps (for OpenCV, ffmpeg encoding, etc.) =========
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# ========= Workdir =========
WORKDIR /app

# ========= Python deps =========
# Use CPU-only PyTorch wheels to avoid pulling heavy CUDA packages.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
    "torch==2.3.1" "torchvision==0.18.1" --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir \
    Flask==2.3.3 \
    gunicorn==21.2.0 \
    ultralytics==8.2.103 \
    easyocr==1.7.1 \
    numpy==1.26.4 \
    opencv-python-headless==4.9.0.80 \
    pillow==11.3.0 \
    pymongo==4.7.2 \
    python-dotenv==1.0.1

# ========= App code & static =========
# Make sure your repo contains:
# - app.py
# - templates/index.html
# - static/css/index.css
# - static/js/index.js
# - mongo/db_utils.py
# - weights/best-ul-11l.pt  (your trained checkpoint)
COPY . ./

# ========= Verify weights exist (but DO NOT load them at build-time) =========
RUN bash -lc 'test -f weights/best-ul-11l.pt || { echo "Missing weights/best-ul-11l.pt"; exit 1; }' \
 && echo "Found weights/best-ul-11l.pt"

# ========= Bake EasyOCR models into the image =========
RUN python - <<'PY'
import easyocr, os
# This will download the detection/recognition models into EASYOCR_DIR
easyocr.Reader(['en'], gpu=False, download_enabled=True,
               model_storage_directory=os.environ.get('EASYOCR_DIR','/app/.easyocr'))
print("EasyOCR models baked into image.")
PY

# ========= Network =========
EXPOSE 5000

# ========= Runtime =========
# --preload: import app early so import errors crash fast (YOLO loads lazily in your code)
# --timeout: allow long video jobs
# --max-requests* : recycle workers periodically to avoid memory bloat
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "600", "--max-requests", "100", "--max-requests-jitter", "20", "--preload", "app:app"]
