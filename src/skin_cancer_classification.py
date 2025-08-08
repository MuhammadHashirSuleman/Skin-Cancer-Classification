from dotenv import load_dotenv
load_dotenv()

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm
import pickle
import gc

# Constants
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
NUM_CLASSES = 7
EPOCHS = 50
MAX_SAMPLES_PER_CLASS = 2000  # To manage memory

def download_dataset():
    kaggle_username = os.environ.get('KAGGLE_USERNAME')
    kaggle_key = os.environ.get('KAGGLE_KEY')
    if not (kaggle_username and kaggle_key):
        print("Error: KAGGLE_USERNAME and KAGGLE_KEY not found in .env file.")
        sys.exit(1)
    api = KaggleApi()
    try:
        api.authenticate()
        print("Kaggle API authenticated successfully.")
    except Exception as e:
        print(f"Authentication error: {e}")
        sys.exit(1)
    if not (os.path.exists('HAM10000_images_part_1') and 
            os.path.exists('HAM10000_images_part_2') and 
            os.path.exists('HAM10000_metadata.csv')):
        print("Downloading HAM10000 dataset...")
        api.dataset_download_files('kmader/skin-cancer-mnist-ham10000', path='.', unzip=True)
    else:
        print("HAM10000 dataset already downloaded.")

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

def create_data_generators():
    train_gen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.3,
        shear_range=0.2,
        fill_mode='nearest'
    )
    val_gen = ImageDataGenerator()
    return train_gen, val_gen

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # 1. Download and load metadata
    download_dataset()
    if not os.path.exists("HAM10000_metadata.csv"):
        print("Metadata missing.")
        sys.exit(1)
    metadata = pd.read_csv("HAM10000_metadata.csv")
    print("First 5 records:", metadata.head())

    # 2. Load and balance dataset
    image_dirs = ['HAM10000_images_part_1', 'HAM10000_images_part_2']
    X, y, label_map = load_and_preprocess_images(image_dirs, metadata)
    if len(X) == 0:
        print("No valid images found.")
        sys.exit(1)
    X, y = balance_dataset(X, y)

    # 3. Train/Validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    y_train = y_train.reshape(-1)
    y_val = y_val.reshape(-1)

    # 4. Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Class Weights: {class_weights_dict}")

    # 5. Model setup
    modelEPA = build_model((128, 128, 3), NUM_CLASSES)
    modelEPA.summary()

    # 6. Generators
    train_datagen, val_datagen = create_data_generators()
    train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

    # 7. Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    # 8. Train
    history = modelEPA.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        class_weight=class_weights_dict,
        callbacks=[early_stopping, lr_scheduler]
    )

    # 9. Save model
    modelEPA.save('skin_cancer_cnn_model.keras')

    # 10. Training plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()

    plt.savefig('training_history.png')
    plt.close()

    # 11. Classification Report
    from sklearn.metrics import classification_report
    y_pred = modelEPA.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print("Classification Report (Focus on mel, bcc, akiec):")
    print(classification_report(y_val, y_pred_classes, target_names=label_map.keys()))

    # 12. Save label_map
    with open('label_map.pkl', 'wb') as f:
        pickle.dump(label_map, f)
