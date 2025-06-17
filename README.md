# License Plate Recognition System

A Python-based License Plate Recognition System that detects and recognizes license plates from images and videos using deep learning. The project is modular, supports experiment tracking, and is designed for easy extension and deployment.

## Features

- License plate detection and recognition using deep learning.
- Modular pipeline with components for data ingestion, model training, evaluation, and inference.
- Experiment tracking and reproducibility.
- Web interface for uploading and processing images/videos.
- Jupyter notebooks for experimentation and visualization.

## Project Structure

```

├── app.py                      # Main application entry point (Flask app)
├── setup.py                    # Package setup
├── requirements.txt            # Python dependencies
├── dvc.yaml                    # DVC pipeline configuration
├── config/
│   └── config.yaml             # Project configuration
├── data/
│   ├── annotated data/         # Labeled data for training/validation/testing
│   └── raw data/               # Raw, unprocessed data
├── notebooks/                  # Jupyter notebooks for experiments
├── src/
│   └── License_Plate_Recognition_System/
│       ├── components/         # Core ML components (data ingestion, training, etc.)
│       ├── config/             # Configuration management
│       ├── pipeline/           # Pipeline stages
│       └── utils/              # Utility functions
├── static/                     # Static files for web app (CSS, JS, processed videos)
├── templates/
│   └── index.html              # Web app HTML template
├── Uploads/                    # Uploaded images/videos for processing
├── weights/                    # Model weights
└── test.py                     # Test script
```

## Installation

1. **Clone the repository:**
   ```
   git clone https://github.com/PSeshpavan/License_Plate_Recognition_System.git
   cd License_Plate_Recognition_System
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **(Optional) Set up DVC for data versioning:**
   ```
   dvc pull
   ```

## Usage

### Run the Web Application

```sh
python app.py
```
- Open your browser and go to `http://localhost:5000` to use the web interface.

### Training & Evaluation

- Modify configuration in [`config/config.yaml`](config/config.yaml) and [`data/annotated data/data.yaml`](data/annotated%20data/data.yaml) as needed.
- Use the pipeline scripts in [`src/License_Plate_Recognition_System/pipeline/`](src/License_Plate_Recognition_System/pipeline/) to run different stages (data ingestion, training, evaluation).

### Notebooks

- Explore and run experiments in [`notebooks/License Plate Recognition System.ipynb`](notebooks/License%20Plate%20Recognition%20System.ipynb).

## Data

- Annotated data and splits are defined in [`data/annotated data/data.yaml`](data/annotated%20data/data.yaml).
- Place your raw and annotated data in the respective folders under `data/`.
