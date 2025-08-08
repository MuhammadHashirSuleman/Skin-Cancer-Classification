import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import resample
from tqdm import tqdm
import pickle

# Constants
IMAGE_SIZE = (128, 128)
NUM_CLASSES = 7
MAX_SAMPLES_PER_CLASS = 2000  # Same as original script

def load_and_preprocess_images(image_dirs, metadata):
    images, labels = [], []
    label_map = {label: idx for idx, label in enumerate(metadata['dx'].unique())}
    print(f"Processing {len(metadata)} images...")
    for image_dir in image_dirs:
        print(f"Loading from {image_dir}...")
        for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc=image_dir):
            img_path = os.path.join(image_dir, row['image_id'] + '.jpg')
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).resize(IMAGE_SIZE)
                    img_array = np.array(img) / 255.0
                    if img_array.shape == (128, 128, 3):
                        images.append(img_array)
                        labels.append(label_map[row['dx']])
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    print(f"Loaded {len(images)} images.")
    return np.array(images), np.array(labels), label_map

def balance_dataset(X, y):
    X_balanced, y_balanced = [], []
    print("Balancing dataset...")
    for label in tqdm(np.unique(y), desc="Oversampling"):
        X_label = X[y == label]
        y_label = y[y == label]
        n_samples = min(MAX_SAMPLES_PER_CLASS, len(X[y == np.argmax([len(X[y == l]) for l in np.unique(y)])]))
        X_resampled, y_resampled = resample(X_label, y_label, replace=True, n_samples=n_samples, random_state=42)
        X_balanced.append(X_resampled)
        y_balanced.append(y_resampled)
    return np.vstack(X_balanced), np.hstack(y_balanced)

if __name__ == '__main__':
    # 1. Load metadata
    metadata_path = "HAM10000_metadata.csv"
    if not os.path.exists(metadata_path):
        print("Metadata file missing. Ensure HAM10000_metadata.csv is in the directory.")
        exit(1)
    metadata = pd.read_csv(metadata_path)
    print("First 5 records:", metadata.head())

    # 2. Load and preprocess images
    image_dirs = ['HAM10000_images_part_1', 'HAM10000_images_part_2']
    X, y, label_map = load_and_preprocess_images(image_dirs, metadata)
    if len(X) == 0:
        print("No valid images found. Ensure image directories exist.")
        exit(1)

    # 3. Balance dataset
    X, y = balance_dataset(X, y)
    print(f"Balanced dataset to {len(X)} images.")

    # 4. Train/validation split (same as original script)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    y_train = y_train.reshape(-1)
    y_val = y_val.reshape(-1)

    # 5. Load saved model
    model_path = 'skin_cancer_cnn_model.keras'
    if not os.path.exists(model_path):
        print("Model file missing. Ensure skin_cancer_cnn_model.keras is in the directory.")
        exit(1)
    model = load_model(model_path)
    print("Model loaded successfully.")

    # 6. Load label map
    label_map_path = 'label_map.pkl'
    if not os.path.exists(label_map_path):
        print("Label map file missing. Regenerating label_map...")
        label_map = {label: idx for idx, label in enumerate(metadata['dx'].unique())}
        with open(label_map_path, 'wb') as f:
            pickle.dump(label_map, f)
    else:
        with open(label_map_path, 'rb') as f:
            label_map = pickle.load(f)
    print(f"Label map: {label_map}")

    # 7. Generate predictions
    print("Generating predictions on validation set...")
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # 8. Print classification report
    print("Classification Report (Focus on mel, bcc, akiec):")
    print(classification_report(y_val, y_pred_classes, target_names=label_map.keys()))