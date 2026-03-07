"""
STYRS — Model Training Script
================================

Version : 2.0
Date    : February 2026
Purpose : Train a deep learning model for solar cell defect classification.

Description:
    This script trains a convolutional neural network to distinguish between
    "Good" and "Defective" solar cells using transfer learning.  The base
    architecture is Xception (pre-trained on ImageNet), with a custom
    classification head added on top.

    The training pipeline includes:
        - Data loading with real-time augmentation (rotation, flipping, shifting)
        - Transfer learning with frozen backbone + trainable head
        - Automatic learning rate reduction on validation loss plateau
        - Early stopping to prevent overfitting
        - Best model checkpointing based on validation accuracy
        - Post-training evaluation with confusion matrix and classification report
        - Optional Test-Time Augmentation (TTA) for improved predictions

Dataset Structure:
    data_dir/
    ├── train/
    │   ├── Defective/   (882 images)
    │   └── Good/        (1,237 images)
    └── test/
        ├── Defective/   (249 images)
        └── Good/        (286 images)

Usage:
    python train_model.py --data_dir ./solar_data
    python train_model.py --data_dir ./solar_data --epochs 30 --lr 0.0005
    python train_model.py --no_train --model_path best_model.keras
"""

# ─────────────────────────────────────────────
# LIBRARY IMPORTS
# ─────────────────────────────────────────────

import os
import argparse                                     # Command-line argument parsing
import numpy as np                                  # Array operations, random seeds
import pandas as pd                                 # (Available for results export)
import matplotlib.pyplot as plt                     # Training history plots
import seaborn as sns                               # Confusion matrix heatmap

# SSL workaround — some macOS environments have certificate verification issues
# when TensorFlow downloads pre-trained weights. This bypasses that check.
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass   # Older Python versions don't have this attribute
else:
    ssl._create_default_https_context = _create_unverified_https_context

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception         # Pre-trained backbone
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# Set random seeds for reproducibility — ensures the same results when re-training
np.random.seed(42)
tf.random.set_seed(42)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_data(data_dir, img_size=(299, 299), batch_size=32):
    """
    Load training and test datasets using Keras ImageDataGenerator.

    The training set uses real-time data augmentation to artificially increase
    the effective dataset size and improve generalisation.  The test set uses
    only rescaling (no augmentation) to ensure fair evaluation.

    Augmentation transforms applied to training images:
        - Random rotation (±20°)
        - Horizontal/vertical shifting (±20%)
        - Horizontal flipping
        - Nearest-neighbour fill for border pixels

    Args:
        data_dir (str): Root directory containing 'train/' and 'test/' subdirectories.
        img_size (tuple): Target image size (height, width). Default: (299, 299).
        batch_size (int): Number of images per training batch. Default: 32.

    Returns:
        tuple: (train_generator, validation_generator) — Keras data generators.

    Raises:
        FileNotFoundError: If the train or test directory doesn't exist.
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found at {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found at {test_dir}")

    # Training set: augmentation + rescaling to [0, 1]
    train_datagen = ImageDataGenerator(
        rescale=1./255,              # Normalise pixel values from [0, 255] to [0, 1]
        rotation_range=20,           # Random rotation up to ±20 degrees
        width_shift_range=0.2,       # Random horizontal shift up to ±20%
        height_shift_range=0.2,      # Random vertical shift up to ±20%
        horizontal_flip=True,        # Randomly flip images horizontally
        fill_mode='nearest'          # Fill border pixels by copying nearest value
    )

    # Test/validation set: ONLY rescaling (no augmentation, to get a fair evaluation)
    test_datagen = ImageDataGenerator(rescale=1./255)

    print(f"Loading training data from {train_dir}...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',    # One-hot encoded labels for softmax output
        shuffle=True                 # Shuffle training data each epoch
    )

    print(f"Loading test/validation data from {test_dir}...")
    validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False                # Don't shuffle test data (needed for correct evaluation)
    )

    return train_generator, validation_generator


# ─────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────

def build_model(input_shape=(299, 299, 3), num_classes=2):
    """
    Build a transfer learning model using Xception as the backbone.

    Architecture:
        Input (299×299×3)
        → Xception backbone (ImageNet weights, frozen)
        → Global Average Pooling 2D
        → Dense 256 (ReLU activation)
        → Dropout 0.5 (regularisation)
        → Dense 2 (Softmax activation)

    The Xception backbone is frozen (not trainable) so that only the
    custom classification head learns from our solar cell dataset.
    This is standard practice for transfer learning when the dataset
    is relatively small (~2,000 images).

    Args:
        input_shape (tuple): Shape of input images (H, W, C). Default: (299, 299, 3).
        num_classes (int): Number of output classes. Default: 2.

    Returns:
        tf.keras.Model: The compiled-ready model (not yet compiled).
    """
    # Load Xception pre-trained on ImageNet, without its top classification layer
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze all layers in the backbone — we only want to train the head
    base_model.trainable = False

    # Build the custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)           # Reduce spatial dims to a 1D vector
    x = Dense(256, activation='relu')(x)       # Fully connected layer for learning features
    x = Dropout(0.5)(x)                        # 50% dropout to prevent overfitting
    predictions = Dense(num_classes, activation='softmax')(x)  # Final classification layer

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_model(model, train_gen, val_gen, epochs=20, learning_rate=0.001, save_path='best_model.keras'):
    """
    Compile and train the model with callbacks.

    Training configuration:
        - Optimizer: Adam with the specified learning rate
        - Loss: Categorical cross-entropy (standard for multi-class classification)
        - Metric: Accuracy

    Callbacks:
        - ModelCheckpoint: Saves the model weights whenever validation accuracy improves
        - EarlyStopping: Stops training if validation loss doesn't improve for 10 epochs
        - ReduceLROnPlateau: Reduces learning rate by 5× if val_loss plateaus for 5 epochs

    Args:
        model (tf.keras.Model): The model to train.
        train_gen: Keras training data generator.
        val_gen: Keras validation data generator.
        epochs (int): Maximum number of training epochs. Default: 20.
        learning_rate (float): Initial learning rate for Adam. Default: 0.001.
        save_path (str): File path to save the best model. Default: 'best_model.keras'.

    Returns:
        tf.keras.callbacks.History: Training history with accuracy/loss per epoch.
    """
    # Compile the model with Adam optimizer and categorical cross-entropy loss
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Define training callbacks
    callbacks = [
        # Save the model whenever validation accuracy reaches a new high
        ModelCheckpoint(
            save_path, monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
        # Stop training early if validation loss doesn't improve for 10 epochs
        EarlyStopping(
            monitor='val_loss', patience=10,
            restore_best_weights=True, verbose=1
        ),
        # Reduce learning rate by factor of 5 if validation loss stalls for 5 epochs
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.2,
            patience=5, min_lr=1e-6, verbose=1
        )
    ]

    # Start training
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks
    )

    return history


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────

def plot_history(history, output_dir='.'):
    """
    Plot training and validation accuracy/loss curves.

    Creates a side-by-side figure with:
        - Left panel: Training vs Validation Accuracy
        - Right panel: Training vs Validation Loss

    These curves help diagnose overfitting (gap between train/val curves)
    and determine if more training epochs are needed.

    Args:
        history: Keras training history object.
        output_dir (str): Directory to save the plot image. Default: current directory.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path)
    print(f"Training history plot saved to {plot_path}")
    plt.close()


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(model, val_gen, class_names):
    """
    Evaluate the model on the validation set and print metrics.

    Outputs:
        - Classification report (precision, recall, F1-score per class)
        - Confusion matrix (as both text and a saved heatmap image)

    The confusion matrix heatmap is saved as 'confusion_matrix.png' and
    helps visualise which classes the model confuses most often.

    Args:
        model (tf.keras.Model): The trained model.
        val_gen: Keras validation data generator.
        class_names (list[str]): Human-readable class names.
    """
    print("Evaluating model on the test/validation set...")
    val_gen.reset()   # Reset generator to start from the beginning

    # Get predictions for all validation images
    Y_pred = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(Y_pred, axis=1)   # Convert softmax probabilities to class indices

    # Print classification report with precision, recall, and F1-score
    print('\nClassification Report:')
    print(classification_report(val_gen.classes, y_pred, target_names=class_names))

    # Generate and print the confusion matrix
    print('Confusion Matrix:')
    cm = confusion_matrix(val_gen.classes, y_pred)
    print(cm)

    # Save the confusion matrix as a visual heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")
    plt.close()


# ─────────────────────────────────────────────
# TEST-TIME AUGMENTATION (TTA)
# ─────────────────────────────────────────────

def predict_tta(model, test_dir, img_size=(299, 299), batch_size=32, tta_steps=5):
    """
    Perform Test-Time Augmentation (TTA) for improved prediction accuracy.

    TTA works by running inference multiple times on augmented versions
    of each test image, then averaging the predictions.  This reduces
    the model's sensitivity to the exact orientation/position of the
    solar cell in the image, often boosting accuracy by 1-3%.

    Args:
        model (tf.keras.Model): The trained model.
        test_dir (str): Path to the test data directory.
        img_size (tuple): Target image size. Default: (299, 299).
        batch_size (int): Batch size for prediction. Default: 32.
        tta_steps (int): Number of augmented prediction rounds. Default: 5.

    Returns:
        tuple: (averaged_predictions, true_labels, class_names)
    """
    print(f"Running Test-Time Augmentation ({tta_steps} steps)...")

    # Use the same augmentation transforms as training
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Collect predictions from multiple augmented passes
    preds_list = []
    for i in range(tta_steps):
        print(f"  TTA step {i+1}/{tta_steps}")
        test_generator.reset()
        preds = model.predict(test_generator, verbose=1)
        preds_list.append(preds)

    # Average all predictions — this smooths out augmentation-induced variance
    final_preds = np.mean(preds_list, axis=0)
    return final_preds, test_generator.classes, list(test_generator.class_indices.keys())


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def main():
    """
    Parse arguments and run the full training pipeline.

    Pipeline:
        1. Load dataset from disk
        2. Build the transfer learning model (or load a saved one)
        3. Train with augmentation and callbacks
        4. Plot training history curves
        5. Evaluate with classification report and confusion matrix
    """
    parser = argparse.ArgumentParser(
        description='STYRS — Train Solar Cell Classification Model'
    )
    parser.add_argument('--data_dir', type=str, default='./solarcell_2',
                        help='Path to dataset directory (must contain train/ and test/ subdirectories)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of images per training batch (default: 32)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Maximum number of training epochs (default: 20)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate for Adam optimizer (default: 0.001)')
    parser.add_argument('--no_train', action='store_true',
                        help='Skip training and load an existing model for evaluation only')
    parser.add_argument('--model_path', type=str, default='best_model.keras',
                        help='Path to save/load the model file (default: best_model.keras)')

    args = parser.parse_args()

    # Step 1: Load the dataset
    try:
        train_gen, val_gen = load_data(args.data_dir, batch_size=args.batch_size)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure your dataset is located at the specified path (default: ./solarcell_2)")
        return

    num_classes = train_gen.num_classes
    class_names = list(train_gen.class_indices.keys())
    print(f"Detected {num_classes} classes: {class_names}")

    # Step 2: Build or load the model
    if args.no_train:
        print(f"Loading pre-trained model from {args.model_path}...")
        try:
            model = tf.keras.models.load_model(args.model_path)
        except Exception as e:
            print(f"Failed to load model: {e}")
            return
    else:
        print("Building new model with Xception backbone...")
        model = build_model(num_classes=num_classes)
        model.summary()

        # Step 3: Train the model
        print("\nStarting training...\n")
        history = train_model(
            model, train_gen, val_gen,
            epochs=args.epochs,
            learning_rate=args.lr,
            save_path=args.model_path
        )
        plot_history(history)

    # Step 4: Evaluate the model
    evaluate_model(model, val_gen, class_names)

    # Step 5 (Optional): Test-Time Augmentation
    # Uncomment the lines below to run TTA on the test set:
    # final_preds, true_labels, _ = predict_tta(model, os.path.join(args.data_dir, 'test'))
    # tta_acc = np.mean(np.argmax(final_preds, axis=1) == true_labels)
    # print(f"TTA Accuracy: {tta_acc:.4f}")


if __name__ == '__main__':
    main()
