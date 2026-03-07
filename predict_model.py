"""
STYRS — Command-Line Prediction Tool
=====================================

Version : 2.0
Date    : February 2026
Purpose : Classify a single solar cell image from the command line.

Description:
    This script provides a quick way to test the trained model without
    starting the web server.  It loads the Keras model, preprocesses the
    input image, runs inference, and prints the predicted class, confidence,
    and raw probability values.

Usage:
    python predict_model.py --image_path path/to/image.jpg
    python predict_model.py --image_path photo.png --model_path custom_model.keras
    python predict_model.py --image_path img.jpg --classes Defective Good

Example Output:
    ------------------------------
    Prediction: Defective
    Confidence: 93.24%
    Raw Probabilities: [0.9324, 0.0676]
    ------------------------------
"""

# ─────────────────────────────────────────────
# LIBRARY IMPORTS
# ─────────────────────────────────────────────

import os
import argparse                                   # Command-line argument parsing
import numpy as np                                # Array manipulation
import tensorflow as tf                           # Deep learning framework
from tensorflow.keras.models import load_model    # Load saved .keras model
from tensorflow.keras.preprocessing import image  # Image → array conversion
import pandas as pd                               # (Available for batch prediction extensions)


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────

def load_trained_model(model_path):
    """
    Load a trained Keras model from the specified file path.

    Args:
        model_path (str): Path to the .keras model file.

    Returns:
        tf.keras.Model: The loaded model ready for inference.

    Raises:
        FileNotFoundError: If the model file doesn't exist at the given path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    return model


# ─────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_image(img_path, img_size=(299, 299)):
    """
    Load and preprocess an image file for model inference.

    Steps:
        1. Load the image and resize to the target dimensions
        2. Convert to a NumPy float32 array
        3. Add a batch dimension → shape (1, H, W, 3)
        4. Normalise pixel values from [0, 255] to [0.0, 1.0]

    Args:
        img_path (str): Path to the image file on disk.
        img_size (tuple): Target (height, width) for resizing. Default: (299, 299).

    Returns:
        np.ndarray: Preprocessed image tensor ready for model.predict().

    Raises:
        FileNotFoundError: If the image file doesn't exist.
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found at {img_path}")

    img = image.load_img(img_path, target_size=img_size)    # Load and resize
    img_array = image.img_to_array(img)                     # Convert to NumPy array
    img_array = np.expand_dims(img_array, axis=0)           # Add batch dimension
    img_array /= 255.0                                      # Normalise to [0, 1]
    return img_array


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────

def predict(model, img_path, class_names=None):
    """
    Run inference on a single image and return the result.

    The model outputs a softmax probability vector.  We select the class
    with the highest probability as the predicted label.

    Args:
        model (tf.keras.Model): The loaded classification model.
        img_path (str): Path to the image file.
        class_names (list[str], optional): Human-readable class labels.
            Default: None (falls back to class index).

    Returns:
        tuple: (predicted_label, confidence, raw_probabilities)
            - predicted_label (str): The predicted class name.
            - confidence (float): The max probability (0.0 to 1.0).
            - raw_probabilities (np.ndarray): Full softmax output vector.
    """
    processed_img = preprocess_image(img_path)
    prediction = model.predict(processed_img)

    # Find the class with the highest probability
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    # Map the index to a human-readable label
    if class_names:
        predicted_label = class_names[predicted_class_idx]
    else:
        predicted_label = str(predicted_class_idx)

    return predicted_label, confidence, prediction[0]


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def main():
    """
    Parse command-line arguments, load the model, and classify the image.
    """
    parser = argparse.ArgumentParser(
        description='STYRS — Predict Solar Cell Defect from Image'
    )
    parser.add_argument(
        '--image_path', type=str, required=True,
        help='Path to the solar cell image file (JPG, PNG, etc.)'
    )
    parser.add_argument(
        '--model_path', type=str, default='best_model.keras',
        help='Path to the trained Keras model (default: best_model.keras)'
    )
    parser.add_argument(
        '--classes', type=str, nargs='+', default=['Defective', 'Good'],
        help='Class names corresponding to model output indices (default: Defective Good)'
    )

    args = parser.parse_args()

    try:
        # Load model and run prediction
        model = load_trained_model(args.model_path)
        label, conf, probs = predict(model, args.image_path, args.classes)

        # Display results
        print("-" * 30)
        print(f"Prediction: {label}")
        print(f"Confidence: {conf:.2%}")
        print(f"Raw Probabilities: {probs}")
        print("-" * 30)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
