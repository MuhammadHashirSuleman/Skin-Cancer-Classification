from dotenv import load_dotenv
load_dotenv()

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

# Constants (same as original)
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
NUM_CLASSES = 7
EPOCHS = 20  # Additional epochs
MAX_SAMPLES_PER_CLASS = 2000

# Load and preprocess images (same as original)
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

# Balance dataset (same as original)
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
    # Load metadata
    metadata = pd.read_csv("HAM10000_metadata.csv")

    # Load and preprocess data
    image_dirs = ['HAM10000_images_part_1', 'HAM10000_images_part_2']
    X, y, label_map = load_and_preprocess_images(image_dirs, metadata)
    X, y = balance_dataset(X, y)

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    y_train = y_train.reshape(-1)
    y_val = y_val.reshape(-1)

    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Class Weights: {class_weights_dict}")

    # Load model
    model = tf.keras.models.load_model('skin_cancer_cnn_model.keras')

    # Recompile with lower learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Data generators
    train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.3,
        shear_range=0.2,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator()

    # Train
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        class_weight=class_weights_dict,
        callbacks=[early_stopping, lr_scheduler]
    )

    # Evaluate
    val_loss, val_accuracy = model.evaluate(val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE))
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    # Save model
    model.save('skin_cancer_cnn_model.keras')

    # Plot training history
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

    plt.savefig('retraining_history.png')
    plt.close()