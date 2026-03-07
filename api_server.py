"""
STYRS — Flask REST API Server
==============================

Version : 2.0
Date    : February 2026
Purpose : Provides a REST API for the STYRS Android app and other clients
          to send solar cell images and receive defect classification results.

Description:
    This Flask server loads the trained EfficientNetB3 Keras model at startup
    and exposes three HTTP endpoints:

        GET  /health   — Check if the server is running and the model is loaded
        POST /predict  — Upload an image and get classification results
        GET  /classes  — List the available classification classes

    The Android app (SolarCellDetector) communicates with this server over
    the local network.  For production use, you would deploy this behind
    a WSGI server like Gunicorn.

Usage:
    python api_server.py

    The server will start on http://0.0.0.0:5001 by default.
"""

# ─────────────────────────────────────────────
# LIBRARY IMPORTS
# ─────────────────────────────────────────────

from flask import Flask, request, jsonify   # Web framework for REST API
from flask_cors import CORS                 # Cross-Origin Resource Sharing (for Android/browser clients)
import tensorflow as tf                     # Deep learning backend
from tensorflow.keras.models import load_model                    # Load saved .keras model
from tensorflow.keras.preprocessing import image as keras_image   # Image ↔ array utilities
import numpy as np                          # Numerical operations (argmax, normalisation)
from PIL import Image                       # Image opening and format conversion
import io                                   # In-memory byte streams for uploaded files
import os                                   # File path checking

# ─────────────────────────────────────────────
# FLASK APP SETUP
# ─────────────────────────────────────────────

app = Flask(__name__)
CORS(app)   # Allow requests from the Android app (different origin)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

MODEL_PATH = 'best_model.keras'             # Path to the trained Keras model file
CLASS_NAMES = ['Defective', 'Good']         # Must match the training directory order

# Global variables — the model is loaded once at startup and reused for all requests
model = None
IMG_SIZE = (299, 299)   # Default; will be overridden by the actual model input shape


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────

def load_trained_model():
    """
    Load the Keras model from disk and determine its expected input size.

    This function is called once when the server starts.  It sets the global
    `model` and `IMG_SIZE` variables so that all subsequent /predict requests
    can use them without re-loading the model.
    """
    global model, IMG_SIZE

    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)

        # Auto-detect the expected image dimensions from the model's input layer.
        # input_shape is typically (None, Height, Width, Channels).
        input_shape = model.input_shape
        IMG_SIZE = (input_shape[1], input_shape[2])
        print(f"Model loaded successfully! Expected input: {IMG_SIZE[0]}×{IMG_SIZE[1]}")
    else:
        print(f"WARNING: Model file not found at {MODEL_PATH}")


def preprocess_image(img):
    """
    Resize and normalise an image so it can be fed into the model.

    Steps:
        1. Resize to the model's expected dimensions (e.g. 300×300)
        2. Convert to a NumPy float32 array
        3. Add a batch dimension → shape becomes (1, H, W, 3)
        4. Scale pixel values from [0, 255] to [0.0, 1.0]

    Args:
        img (PIL.Image): The uploaded image in RGB mode.

    Returns:
        np.ndarray: Preprocessed tensor ready for model.predict().
    """
    img = img.resize(IMG_SIZE)                         # Resize to model input size
    img_array = keras_image.img_to_array(img)          # Convert PIL → NumPy array
    img_array = np.expand_dims(img_array, axis=0)      # Add batch dimension
    img_array /= 255.0                                 # Normalise to [0, 1]
    return img_array


# ─────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.

    The Android app calls this on startup to verify the server is reachable
    and the model is loaded.  Returns a simple JSON status.

    Example response:
        { "status": "healthy", "model_loaded": true }
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint — accepts an image file and returns classification.

    Expected request:
        POST /predict
        Content-Type: multipart/form-data
        Body: image file in the 'image' form field

    Response JSON:
        {
            "success": true,
            "predicted_class": "Defective" or "Good",
            "confidence": 0.923,        (float, 0-1)
            "probabilities": {
                "defective": 0.923,
                "good": 0.077
            }
        }
    """
    # Guard: make sure the model was loaded successfully at startup
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'success': False
        }), 500

    # Guard: make sure the client actually sent an image file
    if 'image' not in request.files:
        return jsonify({
            'error': 'No image file provided',
            'success': False
        }), 400

    try:
        # Read the uploaded file from the HTTP request into a PIL Image
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read()))

        # PNG images may have an alpha channel — convert to RGB for the model
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Run the prediction pipeline: preprocess → model.predict → parse results
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img, verbose=0)

        # Extract the predicted class and confidence from the softmax output
        predicted_class_idx = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))
        probabilities = prediction[0].tolist()

        return jsonify({
            'success': True,
            'predicted_class': CLASS_NAMES[predicted_class_idx],
            'confidence': confidence,
            'probabilities': {
                'defective': probabilities[0],
                'good': probabilities[1]
            }
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/classes', methods=['GET'])
def get_classes():
    """
    Return the list of classification classes the model was trained on.

    This is useful for clients that need to know what labels the model
    can output.
    """
    return jsonify({
        'classes': CLASS_NAMES
    })


# ─────────────────────────────────────────────
# SERVER ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == '__main__':
    # Load the model before starting the server so it's ready for requests
    load_trained_model()

    print("\n" + "=" * 50)
    print("  STYRS — Solar Cell Defect Detection API")
    print("=" * 50)
    print(f"  Server running at : http://0.0.0.0:5001")
    print(f"  Health check      : GET  http://localhost:5001/health")
    print(f"  Predict           : POST http://localhost:5001/predict")
    print(f"  Classes           : GET  http://localhost:5001/classes")
    print("=" * 50 + "\n")

    # debug=True enables auto-reload during development (disable in production)
    app.run(host='0.0.0.0', port=5001, debug=True)
