FROM python:3.11-slim

# Install system dependencies (minimal set for headless OpenCV)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    libfontconfig1 \
    libgl1-mesa-dev \
    libglib2.0-dev \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Set environment variables for memory optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TORCH_HOME=/tmp
ENV YOLO_CONFIG_DIR=/tmp
ENV EASYOCR_MODULE_PATH=/tmp
ENV HF_HOME=/tmp

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge && \
    rm -rf /root/.cache/pip

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p weights /tmp

# Pre-download EasyOCR models to reduce startup memory spike
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False, download_enabled=True)" || echo "EasyOCR pre-download failed, will download at runtime"

# Expose port
EXPOSE 5000

# Run with gunicorn (optimized for memory)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "300", "--max-requests", "50", "--max-requests-jitter", "10", "--preload", "app:app"]