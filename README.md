# Skin Cancer Classification Project

## Description
This project implements a Convolutional Neural Network (CNN) for classifying skin cancer types using image data from the HAM10000 dataset. It includes scripts for training, retraining, generating classification reports, and running a prediction application.

## Project Structure
- **src/**: Contains the main Python scripts including `app.py` (main application), `ReTraining.py` (model retraining), `class_report.py` (classification reports), `skin_cancer_classification.py` (model training), and others.
- **models/**: Stores the trained model `skin_cancer_cnn_model.keras` and label mappings.
- **data/**: CSV files with metadata and processed image data.
- **figures/**: Generated plots like training history.
- **images/**: Dataset images organized in subfolders.
- **logs/**: Log files (excluded from Git via .gitignore).
- **requirements.txt**: List of Python dependencies.
- **.gitignore**: Specifies files/directories to ignore in Git (e.g., logs, data, images, models).

## Installation
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`

## Usage
- To run the main application: `python src/app.py`
- To retrain the model: `python src/ReTraining.py`
- To generate classification report: `python src/class_report.py`

## Dataset
The project uses the HAM10000 dataset for skin lesion images.

## Model
The CNN model is saved as `skin_cancer_cnn_model.keras` in the models directory.

For more details, refer to the source code in src/.