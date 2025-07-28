# License Plate Recognition System

A Python-based License Plate Recognition (LPR) system leveraging deep learning for automatic detection and recognition of vehicle license plates from images, videos or live stream.

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
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
├── notebooks/                  # Jupyter notebooks for experiments
├── runs/                       # YOLO detection outputs
├── static/                     # Static files (CSS, JS, processed videos)
├── templates/                  # HTML templates for Flask
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