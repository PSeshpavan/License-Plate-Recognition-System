FROM python:3.11-slim AS app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/tmp \
    YOLO_CONFIG_DIR=/tmp \
    EASYOCR_MODULE_PATH=/tmp \
    HF_HOME=/tmp \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# minimal runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libfontconfig1 libgl1 wget \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch CPU wheels from official index (compatible with py3.11)
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# Now the rest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip cache purge

# App
COPY . .
RUN mkdir -p weights /tmp

# (Optional) Pre-pull EasyOCR models
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False, download_enabled=True)" || echo "EasyOCR pre-download failed, will download at runtime"

EXPOSE 5000
CMD ["gunicorn","--bind","0.0.0.0:5000","--workers","1","--timeout","300","--max-requests","50","--max-requests-jitter","10","--preload","app:app"]
