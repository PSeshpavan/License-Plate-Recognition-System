import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = 'License_Plate_Recognition_System'

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/base_model.py",
    f"src/{project_name}/components/dat_ingestion.py",
    f"src/{project_name}/components/model_evaluation_with_mlflow.py",
    f"src/{project_name}/components/model_training.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/stage_01.py",
    f"src/{project_name}/pipeline/stage_02.py",
    f"src/{project_name}/pipeline/stage_03.py",
    f"src/{project_name}/pipeline/stage_04.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    # "setup.py",
    # "requirements.txt",
    "notebooks/trails.ipynb",
    "notebooks/01_data_ingestion.ipynb",
    "notebooks/02_base_model.ipynb",
    "notebooks/03_model_trainer.ipynb",
    "notebooks/04_model_evaluation.ipynb",
    "templates/index.html",
    "test.py",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")
        
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} already exists. Please delete it if you want to override it.")
    