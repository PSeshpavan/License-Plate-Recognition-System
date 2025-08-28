FROM python:3.11-slim AS app

# Avoid debconf TTY warnings, reduce noise
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/tmp \
    YOLO_CONFIG_DIR=/tmp \
    EASYOCR_MODULE_PATH=/tmp \
    HF_HOME=/tmp \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Minimal system deps for headless OpenCV/Ultralytics/EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libfontconfig1 \
    libgl1 \
    wget \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Install CPU torch wheels first from the official index
#    (versions align well with Ultralytics 8.0.196 on py3.11)
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# 2) Then the rest of your deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip cache purge

# 3) Your application
COPY . .

# Create necessary directories
RUN mkdir -p weights /tmp

# (Optional) Pre-download EasyOCR models (kept as you had it)
# If this ever bloats your image, remove this step so it downloads at runtime.
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False, download_enabled=True)" || echo "EasyOCR pre-download failed, will download at runtime"

EXPOSE 5000

# Keep your gunicorn command; a couple of safe defaults already set via env
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "300", "--max-requests", "50", "--max-requests-jitter", "10", "--preload", "app:app"]
