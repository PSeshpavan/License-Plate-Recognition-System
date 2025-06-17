# License Plate Recognition System

A Python-based License Plate Recognition (LPR) system leveraging deep learning for automatic detection and recognition of vehicle license plates from images, videos or live stream.

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Training](#training)
- [Evaluation](#evaluation)
- [Web Application](#web-application)
- [Notebooks](#notebooks)
- [Contributing](#contributing)
- [License](#license)

---

## Project Structure

```
.
├── app.py                      # Main Flask application
├── template.py                 # Project scaffolding script
├── setup.py                    # Package setup
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── data/                       # Dataset directory
│   ├── annotated data/         # Labeled data for training/validation/testing
│   └── raw data/               # Raw, unprocessed data
├── notebooks/                  # Jupyter notebooks for experiments
├── runs/                       # YOLO detection outputs
├── static/                     # Static files (CSS, JS, processed videos)
├── templates/                  # HTML templates for Flask
├── uploads/                    # Uploaded images/videos for inference
├── weights/                    # Trained model weights
└── ...
```

## Features

- Automatic detection and recognition of license plates in images, videos, live stream
- Deep learning-based detection using YOLOv9
- Web interface for uploading images/videos and viewing results
- Modular codebase for easy extension and experimentation

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/PSeshpavan/License_Plate_Recognition_System.git
   cd License_Plate_Recognition_System
   ```

2. **Create a virtual environment and activate it:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Running the Web Application

```sh
python app.py
```

- Open your browser and go to `http://localhost:5000`
- Upload an image or video to detect license plates

### Inference

- Uploaded images/videos will get saved in `uploads/` directory
- Results and processed files will appear in `static/processed_videos/`

## Data

- Annotated datasets are in `data/annotated data/` ([README.dataset.txt](data/annotated%20data/README.dataset.txt), [README.roboflow.txt](data/annotated%20data/README.roboflow.txt))
- Training, validation, and test splits are organized under `train/`, `valid/`, and `test/`
- Data configuration: [data.yaml](data/annotated%20data/data.yaml)

## Training

- Model training scripts and pipelines are located in `src/License_Plate_Recognition_System/pipeline/`
- Use the provided notebooks ([01_data_ingestion.ipynb](notebooks/01_data_ingestion.ipynb), [02_base_model.ipynb](notebooks/02_base_model.ipynb), etc.) for step-by-step training and evaluation

## Evaluation

- Evaluate model performance using the scripts in `src/License_Plate_Recognition_System/components/` and the evaluation notebook ([04_model_evaluation.ipynb](notebooks/04_model_evaluation.ipynb))

## Web Application

- The Flask app ([app.py](app.py)) serves the web interface
- HTML templates are in [templates/index.html](templates/index.html)
- Static assets (CSS/JS) are in [static/css/index.css](static/css/index.css) and [static/js/](static/js/)

## Notebooks

- [License Plate Recognition System.ipynb](notebooks/License%20Plate%20Recognition%20System.ipynb): End-to-end workflow
- [Ultralytics Yolov9.ipynb](notebooks/Ultralytics%20Yolov9.ipynb): YOLOv9 experiments using Yolov9

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.