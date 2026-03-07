"""
STYRS — Solar Cell Defect Detection Platform
============================================

Version : 2.0
Date    : February 2026
Purpose : Web-based application for detecting manufacturing defects in
          photovoltaic (solar) cells using deep learning.

Description:
    This Streamlit application provides a graphical interface for quality
    inspectors to upload electroluminescence images of solar cells and
    receive instant AI-powered defect classification. The underlying
    neural network (EfficientNetB3, transfer-learned on the ELPV dataset)
    distinguishes between "Good" and "Defective" cells with ~89% accuracy.

Main Features:
    1. Single Image Analysis   – Upload one image, get prediction + GradCAM
    2. Batch Analysis          – Upload multiple images, get summary statistics
    3. PDF Report Generation   – Professional reports for quality documentation
    4. Session History         – Track all analyses within the current session
    5. GradCAM Visualisation   – Highlights regions the model focused on
    6. Confidence Gauge        – Visual dial showing prediction strength

Technology Stack:
    - TensorFlow / Keras  : Model loading and inference
    - Streamlit           : Web UI framework
    - Matplotlib          : Charts (confidence gauge)
    - FPDF2               : PDF report generation
    - Pillow (PIL)        : Image manipulation
    - NumPy / Pandas      : Numerical operations and data export
"""

# ─────────────────────────────────────────────
# LIBRARY IMPORTS
# ─────────────────────────────────────────────
# We import each library with a brief note on its role in this project.

import streamlit as st                              # Web framework for the UI
import tensorflow as tf                             # Deep learning backend
from tensorflow.keras.models import load_model, Model  # Model loading + GradCAM sub-model
from tensorflow.keras.preprocessing import image as keras_image  # Image → array conversion
import numpy as np                                  # Array math and normalisation
from PIL import Image, ImageDraw, ImageFont, ImageFilter  # Image manipulation
from fpdf import FPDF                               # PDF report generation
import os                                           # File path operations
import io                                           # In-memory byte streams
import time                                         # Progress bar animation delay
import json                                         # (reserved for future config support)
import datetime                                     # Timestamps in reports
import pandas as pd                                 # DataFrame for CSV export

# Matplotlib must use a non-interactive backend because Streamlit runs headless
# (no display server). "Agg" renders to in-memory buffers, not to a window.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="STYRS — Solar Cell Inspector",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# PREMIUM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ── */
html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
}
.main .block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

/* ── Header ── */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.10) 100%);
    border-radius: 24px;
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 2rem;
    backdrop-filter: blur(20px);
}
.hero-header h1 {
    background: linear-gradient(135deg, #818cf8, #a78bfa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 0.3rem;
}
.hero-header p {
    color: rgba(255,255,255,0.55);
    font-size: 1.05rem;
    font-weight: 300;
}

/* ── Glass Card ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1.8rem;
    backdrop-filter: blur(16px);
    margin-bottom: 1.2rem;
    transition: all 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(129,140,248,0.25);
    box-shadow: 0 8px 32px rgba(99,102,241,0.12);
}

/* ── Result Cards ── */
.result-defective {
    background: linear-gradient(135deg, rgba(239,68,68,0.12) 0%, rgba(220,38,38,0.08) 100%);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}
.result-defective h2 {
    color: #fca5a5;
    font-size: 2rem;
    margin: 0.5rem 0;
}
.result-defective p {
    color: rgba(252,165,165,0.8);
    font-size: 1.1rem;
}

.result-good {
    background: linear-gradient(135deg, rgba(34,197,94,0.12) 0%, rgba(22,163,74,0.08) 100%);
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}
.result-good h2 {
    color: #86efac;
    font-size: 2rem;
    margin: 0.5rem 0;
}
.result-good p {
    color: rgba(134,239,172,0.8);
    font-size: 1.1rem;
}

/* ── Metric Cards ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1.2rem 0;
}
.metric-card {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.2rem;
    text-align: center;
    transition: all 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(129,140,248,0.3);
}
.metric-card .metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-card .metric-label {
    color: rgba(255,255,255,0.45);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
}

/* ── Confidence Bar ── */
.conf-bar-outer {
    background: rgba(255,255,255,0.06);
    border-radius: 12px;
    height: 14px;
    margin: 0.6rem 0;
    overflow: hidden;
}
.conf-bar-inner-good {
    height: 100%;
    border-radius: 12px;
    background: linear-gradient(90deg, #22c55e, #4ade80);
    transition: width 1s ease;
}
.conf-bar-inner-bad {
    height: 100%;
    border-radius: 12px;
    background: linear-gradient(90deg, #ef4444, #f87171);
    transition: width 1s ease;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13111c 0%, #1a1a2e 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #c4b5fd !important;
}
.sidebar-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.7rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.sidebar-stat .label {
    color: rgba(255,255,255,0.45);
    font-size: 0.85rem;
}
.sidebar-stat .value {
    color: #e2e8f0;
    font-weight: 600;
    font-size: 0.9rem;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.7rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.3px;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(99,102,241,0.4) !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03);
    border: 2px dashed rgba(129,140,248,0.25);
    border-radius: 20px;
    padding: 1rem;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(129,140,248,0.5);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: rgba(255,255,255,0.03);
    border-radius: 16px;
    padding: 0.3rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    color: rgba(255,255,255,0.5);
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.2) !important;
    color: #a5b4fc !important;
}

/* ── History Table ── */
.history-item {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.8rem 1rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 14px;
    margin-bottom: 0.5rem;
    transition: all 0.2s ease;
}
.history-item:hover {
    background: rgba(255,255,255,0.06);
    border-color: rgba(129,140,248,0.2);
}
.history-badge-good {
    background: rgba(34,197,94,0.15);
    color: #86efac;
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}
.history-badge-bad {
    background: rgba(239,68,68,0.15);
    color: #fca5a5;
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}

/* ── Batch table ── */
.batch-summary {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1rem;
    margin: 1rem 0;
}

/* ── Misc ── */
h1, h2, h3, h4 {
    color: #e2e8f0 !important;
}
p, label, span {
    color: rgba(255,255,255,0.7);
}
.stProgress > div > div {
    background: linear-gradient(90deg, #6366f1, #a78bfa) !important;
}
.stAlert {
    border-radius: 14px !important;
}
/* Hide stray 'None' rendered by Streamlit auto-write of return values */
div[data-testid="stMarkdownContainer"]:has(> p > code:only-child) {
    display: none !important;
}

/* ── Pulse animation ── */
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 5px rgba(99,102,241,0.3); }
    50% { box-shadow: 0 0 20px rgba(99,102,241,0.6); }
}
.analyzing {
    animation: pulse-glow 1.5s ease-in-out infinite;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────
# Streamlit re-runs the entire script on every user interaction (click, upload, etc.).
# To persist data across those re-runs, we store counters and history in
# st.session_state, which acts like a per-user dictionary that survives re-runs.

if 'history' not in st.session_state:
    st.session_state.history = []           # List of past scan results for the History tab
if 'total_analyzed' not in st.session_state:
    st.session_state.total_analyzed = 0     # Running count of images analyzed this session
if 'total_defective' not in st.session_state:
    st.session_state.total_defective = 0    # Running count of defective classifications
if 'total_good' not in st.session_state:
    st.session_state.total_good = 0         # Running count of good classifications

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource   # Cache the model across all users and re-runs for performance
def load_trained_model():
    """
    Load the pre-trained Keras model from disk.

    The model file 'best_model.keras' is an EfficientNetB3-based classifier
    trained on the ELPV solar cell dataset.  We use @st.cache_resource so the
    model is loaded only once (on first request) and shared across all users.

    Returns:
        tf.keras.Model or None: The loaded model, or None if file is missing.
    """
    model_path = 'best_model.keras'
    if not os.path.exists(model_path):
        return None
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Attempt to load the model at application start-up.
# If anything goes wrong, we set model = None and show a warning later.
try:
    model = load_trained_model()
except:
    model = None

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def get_model_input_size(m):
    """
    Extract the expected image dimensions from the Keras model.

    The model's input_shape is typically (None, Height, Width, Channels).
    We only need (Height, Width) for resizing uploaded images.

    Args:
        m: A loaded tf.keras.Model.

    Returns:
        tuple: (height, width) expected by the model, e.g. (300, 300).
    """
    input_shape = m.input_shape  # e.g. (None, 300, 300, 3)
    return (input_shape[1], input_shape[2])


def detect_architecture(m):
    """
    Identify the base architecture of the loaded model by inspecting layer names.

    Different pre-trained models (EfficientNet, Xception, MobileNet) have
    characteristic layer naming conventions. We concatenate all layer names
    and search for known patterns.

    Args:
        m: A loaded tf.keras.Model.

    Returns:
        str: Human-readable architecture name (e.g. 'EfficientNetB3').
    """
    # Build a single string of all layer names for easy pattern matching
    layer_str = " ".join([layer.name for layer in m.layers]).lower()

    if "efficientnetb3" in layer_str or "block7" in layer_str:
        return "EfficientNetB3"
    elif "efficientnet" in layer_str or ("stem_conv" in layer_str and "block1a" in layer_str):
        return "EfficientNetB0"
    elif "mobilenet" in layer_str:
        return "MobileNetV2"
    elif "xception" in layer_str:
        return "Xception"
    return "Custom CNN"


def preprocess_image(img, target_size=(300, 300)):
    """
    Prepare a PIL image for model inference.

    Steps:
        1. Convert to RGB (handles grayscale or RGBA uploads)
        2. Resize to match the model's expected input dimensions
        3. Convert to a NumPy array with pixel values in [0, 1]
        4. Add a batch dimension so shape becomes (1, H, W, 3)

    Args:
        img (PIL.Image): The uploaded image.
        target_size (tuple): (height, width) for resizing.

    Returns:
        np.ndarray: Preprocessed image tensor of shape (1, H, W, 3).
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')                # Ensure 3-channel input
    img = img.resize(target_size)               # Resize to model dimensions
    img_array = keras_image.img_to_array(img)   # (H, W, 3) float32 array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dim → (1, H, W, 3)
    img_array /= 255.0                          # Normalise pixels to [0, 1]
    return img_array


def predict_single(img, m):
    """
    Run a single image through the model and return the prediction.

    The model outputs a softmax vector [P(Defective), P(Good)].  We pick
    the class with the higher probability as the predicted label.

    Args:
        img (PIL.Image): The solar cell image.
        m (tf.keras.Model): The loaded model.

    Returns:
        dict: {
            'label'         : 'Good' or 'Defective',
            'confidence'    : float (max probability),
            'probabilities' : { 'Defective': float, 'Good': float }
        }
    """
    target_size = get_model_input_size(m)
    processed = preprocess_image(img, target_size)

    # model.predict returns shape (1, 2) — probabilities for each class
    prediction = m.predict(processed, verbose=0)

    class_names = ['Defective', 'Good']   # Must match training directory order
    predicted_index = int(np.argmax(prediction, axis=1)[0])  # Index of highest prob
    confidence = float(np.max(prediction))                   # Highest probability

    return {
        'label': class_names[predicted_index],
        'confidence': confidence,
        'probabilities': {
            'Defective': float(prediction[0][0]),
            'Good': float(prediction[0][1])
        }
    }

def generate_gradcam(img, m, target_size):
    """
    Generate a Gradient-weighted Class Activation Map (GradCAM) heatmap.

    GradCAM highlights the regions of the image that most influenced the
    model's prediction.  This helps quality inspectors understand *where*
    the model detected a potential defect (e.g., a crack or dark spot).

    How it works:
        1. Find the last convolutional layer in the network.
        2. Record the gradients of the predicted class with respect to that
           layer's feature maps (using TensorFlow's GradientTape).
        3. Average the gradients across spatial dimensions → per-channel weights.
        4. Weighted-sum the feature maps → heatmap.
        5. Overlay the heatmap on the original image using the 'jet' colormap.

    Args:
        img (PIL.Image): The original uploaded image.
        m (tf.keras.Model): The loaded classification model.
        target_size (tuple): (height, width) for model input.

    Returns:
        PIL.Image or None: The image with GradCAM overlay, or None on failure.
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_resized = img.resize(target_size)
    img_array = keras_image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Find last conv layer using class name (works with Keras 2 and 3)
    last_conv = None
    conv_types = ('conv2d', 'depthwiseconv2d', 'separableconv2d')
    
    # Collect all layers, including from sub-models (e.g. EfficientNetB3 backbone)
    all_layers = []
    for layer in m.layers:
        if hasattr(layer, 'layers'):  # Sub-model
            all_layers.extend(layer.layers)
        else:
            all_layers.append(layer)
    
    # Search by class name
    for layer in reversed(all_layers):
        layer_class = layer.__class__.__name__.lower()
        if any(ct in layer_class for ct in conv_types):
            last_conv = layer.name
            break
    
    # Fallback: search by layer name patterns
    if last_conv is None:
        for layer in reversed(all_layers):
            name = layer.name.lower()
            if 'top_conv' in name or ('block' in name and 'conv' in name):
                last_conv = layer.name
                break
    if last_conv is None:
        return None

    try:
        # Try to get layer from top-level model first
        try:
            conv_layer_output = m.get_layer(last_conv).output
        except ValueError:
            # Layer is inside a sub-model — find it
            conv_layer_output = None
            for layer in m.layers:
                if hasattr(layer, 'layers'):
                    try:
                        conv_layer_output = layer.get_layer(last_conv).output
                        break
                    except ValueError:
                        continue
            if conv_layer_output is None:
                return None
        
        grad_model = Model(
            inputs=m.input,
            outputs=[conv_layer_output, m.output]
        )
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            class_idx = tf.argmax(preds[0])
            loss = preds[:, class_idx]
        grads = tape.gradient(loss, conv_out)
        if grads is None:
            return None
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
        heatmap = tf.nn.relu(heatmap)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        # Resize heatmap to image size
        heatmap_resized = np.uint8(255 * heatmap)
        heatmap_img = Image.fromarray(heatmap_resized).resize(
            (img_resized.size[0], img_resized.size[1]), Image.BILINEAR
        )
        heatmap_arr = np.array(heatmap_img) / 255.0

        # Apply colormap
        cmap = plt.cm.jet
        colored_heatmap = cmap(heatmap_arr)[:, :, :3]
        colored_heatmap = np.uint8(colored_heatmap * 255)

        # Overlay
        original_arr = np.array(img_resized)
        overlay = (0.6 * original_arr + 0.4 * colored_heatmap).astype(np.uint8)
        return Image.fromarray(overlay)
    except Exception:
        return None

def create_confidence_gauge(confidence, is_defective):
    """
    Draw a semi-circular confidence gauge (speedometer style).

    The gauge shows a half-circle arc from 0% to 100%.  The filled portion
    represents the model's confidence level.  Colour is green for "Good"
    predictions and red for "Defective" predictions.

    Args:
        confidence (float): Model confidence, range [0.0, 1.0].
        is_defective (bool): True if the prediction was "Defective".

    Returns:
        matplotlib.figure.Figure: The rendered gauge figure.
    """
    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    # Gauge arc (background)
    theta = np.linspace(0, np.pi, 100)
    ax.plot(theta, [1]*100, color=(1, 1, 1, 0.1), linewidth=12, solid_capstyle='round')

    # Filled arc
    filled = max(int(confidence * 100), 2)
    color = '#ef4444' if is_defective else '#22c55e'
    theta_filled = np.linspace(0, np.pi * confidence, filled)
    ax.plot(theta_filled, [1]*filled, color=color, linewidth=12, solid_capstyle='round')

    # Center text
    ax.text(np.pi/2, 0.3, f"{confidence:.0%}", ha='center', va='center',
            fontsize=22, fontweight='bold', color='white', family='sans-serif')

    ax.set_ylim(0, 1.5)
    ax.axis('off')
    plt.tight_layout()
    return fig

def add_to_history(filename, result):
    """
    Record a completed analysis in the session history.

    Inserts the new record at the front (most recent first) and updates
    the running counters that appear in the sidebar.  Only the last
    50 records are kept to prevent excessive memory usage.

    Args:
        filename (str): The name of the analysed image file.
        result (dict): The prediction dict from predict_single().
    """
    st.session_state.history.insert(0, {
        'time': datetime.datetime.now().strftime("%H:%M:%S"),
        'file': filename,
        'label': result['label'],
        'confidence': result['confidence']
    })

    # Update running session counters for the sidebar display
    st.session_state.total_analyzed += 1
    if result['label'] == 'Defective':
        st.session_state.total_defective += 1
    else:
        st.session_state.total_good += 1

    # Cap history at 50 entries to avoid unbounded memory growth
    if len(st.session_state.history) > 50:
        st.session_state.history = st.session_state.history[:50]


def generate_single_pdf(filename, result, image):
    """
    Generate a professional PDF quality inspection report for a single image.

    The report includes:
        - STYRS-branded header with model info and date
        - The analysed image embedded in the document
        - Classification verdict (Good / Defective) with colour coding
        - Probability breakdown with horizontal bar chart
        - Technical details (model, resolution, parameters, accuracy)
        - Footer with version info

    Args:
        filename (str): Original filename of the uploaded image.
        result (dict): The prediction dictionary from predict_single().
        image (PIL.Image): The uploaded image (used for embedding and metadata).

    Returns:
        bytes: The complete PDF document as raw bytes (ready for download).
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Header bar
    pdf.set_fill_color(26, 26, 46)  # Dark navy
    pdf.rect(0, 0, 210, 35, 'F')
    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_text_color(129, 140, 248)  # Purple
    pdf.set_xy(10, 8)
    pdf.cell(0, 12, 'STYRS', ln=False)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(180, 180, 200)
    pdf.set_xy(10, 22)
    pdf.cell(0, 8, 'Solar Cell Defect Detection Report')
    
    # Report info
    pdf.set_text_color(80, 80, 100)
    pdf.set_font('Helvetica', '', 8)
    pdf.set_xy(135, 10)
    pdf.cell(0, 5, f'Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}')
    pdf.set_xy(135, 16)
    pdf.cell(0, 5, 'Model: EfficientNetB3 (89.24%)')
    pdf.set_xy(135, 22)
    pdf.cell(0, 5, f'File: {filename}')
    
    pdf.ln(30)
    
    # Save image temporarily for embedding
    img_buf = io.BytesIO()
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(img_buf, format='JPEG', quality=85)
    img_buf.seek(0)
    import tempfile
    tmp_img = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    tmp_img.write(img_buf.read())
    tmp_img.close()
    
    # Image section
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(40, 40, 60)
    pdf.cell(0, 10, 'Analyzed Image', ln=True)
    pdf.ln(2)
    try:
        pdf.image(tmp_img.name, x=15, w=80)
    except Exception:
        pass
    os.unlink(tmp_img.name)
    
    # Result section — positioned to the right of image
    is_defective = result['label'] == 'Defective'
    result_y = pdf.get_y() - 50
    if result_y < 50:
        result_y = 50
    
    # Result box
    if is_defective:
        pdf.set_fill_color(254, 226, 226)  # Light red
        pdf.set_draw_color(239, 68, 68)
    else:
        pdf.set_fill_color(220, 252, 231)  # Light green
        pdf.set_draw_color(34, 197, 94)
    
    pdf.set_xy(105, result_y)
    pdf.rect(105, result_y, 90, 45, 'DF')
    
    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_text_color(239, 68, 68) if is_defective else pdf.set_text_color(22, 163, 74)
    pdf.set_xy(105, result_y + 5)
    icon = 'DEFECTIVE' if is_defective else 'GOOD'
    pdf.cell(90, 12, icon, align='C')
    
    pdf.set_font('Helvetica', '', 12)
    pdf.set_text_color(80, 80, 100)
    pdf.set_xy(105, result_y + 20)
    pdf.cell(90, 8, f'Confidence: {result["confidence"]:.1%}', align='C')
    
    pdf.set_font('Helvetica', '', 10)
    pdf.set_xy(105, result_y + 30)
    pdf.cell(90, 8, f'P(Defective): {result["probabilities"]["Defective"]:.4f}', align='C')
    
    # Move below image
    pdf.set_y(max(pdf.get_y() + 10, result_y + 55))
    
    # Probability breakdown
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(40, 40, 60)
    pdf.cell(0, 10, 'Probability Breakdown', ln=True)
    pdf.ln(3)
    
    for cls, prob in result['probabilities'].items():
        pdf.set_font('Helvetica', '', 11)
        pdf.set_text_color(60, 60, 80)
        pdf.cell(35, 8, cls)
        # Draw bar
        bar_x = pdf.get_x()
        bar_y = pdf.get_y() + 1
        # Background
        pdf.set_fill_color(230, 230, 240)
        pdf.rect(bar_x, bar_y, 110, 6, 'F')
        # Fill
        if cls == 'Defective':
            pdf.set_fill_color(239, 68, 68)
        else:
            pdf.set_fill_color(34, 197, 94)
        pdf.rect(bar_x, bar_y, 110 * prob, 6, 'F')
        pdf.set_xy(bar_x + 115, bar_y - 1)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(20, 8, f'{prob:.1%}')
        pdf.ln(10)
    
    # Technical details
    pdf.ln(5)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(40, 40, 60)
    pdf.cell(0, 10, 'Technical Details', ln=True)
    pdf.ln(2)
    
    details = [
        ('Model Architecture', 'EfficientNetB3'),
        ('Input Resolution', '300 x 300'),
        ('Parameters', '13.3M'),
        ('Training Accuracy', '89.24%'),
        ('Image File', filename),
        ('Image Size', f'{image.size[0]} x {image.size[1]}'),
        ('Analysis Date', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
    ]
    
    for label, value in details:
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(100, 100, 120)
        pdf.cell(55, 7, label)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(40, 40, 60)
        pdf.cell(0, 7, value, ln=True)
    
    # Footer
    pdf.set_y(-25)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(160, 160, 180)
    pdf.cell(0, 5, 'Generated by STYRS Solar Cell Inspector v2.0', align='C', ln=True)
    pdf.cell(0, 5, 'Powered by EfficientNetB3 Deep Learning', align='C')
    
    return bytes(pdf.output())


def generate_batch_pdf(results, batch_files):
    """
    Generate a professional PDF report summarising a batch analysis.

    The report includes:
        - STYRS-branded header
        - Summary statistics (total images, defective/good counts, defect rate)
        - A table listing every image with its classification and confidence
        - Average confidence across the entire batch

    Args:
        results (list[dict]): List of prediction dictionaries from predict_single().
        batch_files (list): List of uploaded file objects (for metadata).

    Returns:
        bytes: The complete PDF document as raw bytes (ready for download).
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Header
    pdf.set_fill_color(26, 26, 46)
    pdf.rect(0, 0, 210, 35, 'F')
    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_text_color(129, 140, 248)
    pdf.set_xy(10, 8)
    pdf.cell(0, 12, 'STYRS')
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(180, 180, 200)
    pdf.set_xy(10, 22)
    pdf.cell(0, 8, 'Batch Analysis Report')
    pdf.set_text_color(80, 80, 100)
    pdf.set_font('Helvetica', '', 8)
    pdf.set_xy(135, 10)
    pdf.cell(0, 5, f'Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}')
    pdf.set_xy(135, 16)
    pdf.cell(0, 5, f'Images Analyzed: {len(results)}')
    pdf.set_xy(135, 22)
    pdf.cell(0, 5, 'Model: EfficientNetB3 (89.24%)')
    
    pdf.ln(30)
    
    # Summary stats
    n_def = sum(1 for r in results if r['label'] == 'Defective')
    n_good = sum(1 for r in results if r['label'] == 'Good')
    avg_conf = np.mean([r['confidence'] for r in results])
    defect_rate = n_def / len(results) if results else 0
    
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(40, 40, 60)
    pdf.cell(0, 10, 'Summary', ln=True)
    pdf.ln(3)
    
    # Summary boxes
    box_w = 42
    box_h = 25
    start_x = 15
    box_y = pdf.get_y()
    
    summaries = [
        (str(len(results)), 'Total', (129, 140, 248)),
        (str(n_def), 'Defective', (239, 68, 68)),
        (str(n_good), 'Good', (34, 197, 94)),
        (f'{avg_conf:.0%}', 'Avg Conf', (129, 140, 248)),
    ]
    
    for i, (val, label, color) in enumerate(summaries):
        x = start_x + i * (box_w + 5)
        pdf.set_fill_color(245, 245, 250)
        pdf.set_draw_color(*color)
        pdf.rect(x, box_y, box_w, box_h, 'DF')
        pdf.set_font('Helvetica', 'B', 16)
        pdf.set_text_color(*color)
        pdf.set_xy(x, box_y + 3)
        pdf.cell(box_w, 10, val, align='C')
        pdf.set_font('Helvetica', '', 8)
        pdf.set_text_color(100, 100, 120)
        pdf.set_xy(x, box_y + 14)
        pdf.cell(box_w, 8, label, align='C')
    
    pdf.set_y(box_y + box_h + 8)
    
    # Defect rate bar
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(60, 60, 80)
    pdf.cell(0, 8, f'Defect Rate: {defect_rate:.1%}', ln=True)
    bar_y = pdf.get_y()
    pdf.set_fill_color(230, 230, 240)
    pdf.rect(15, bar_y, 180, 6, 'F')
    pdf.set_fill_color(239, 68, 68)
    pdf.rect(15, bar_y, 180 * defect_rate, 6, 'F')
    pdf.ln(12)
    
    # Results table
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(40, 40, 60)
    pdf.cell(0, 10, 'Detailed Results', ln=True)
    pdf.ln(2)
    
    # Table header
    pdf.set_fill_color(26, 26, 46)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.cell(8, 8, '#', border=1, fill=True, align='C')
    pdf.cell(62, 8, 'Filename', border=1, fill=True)
    pdf.cell(30, 8, 'Result', border=1, fill=True, align='C')
    pdf.cell(30, 8, 'Confidence', border=1, fill=True, align='C')
    pdf.cell(30, 8, 'P(Defective)', border=1, fill=True, align='C')
    pdf.cell(25, 8, 'P(Good)', border=1, fill=True, align='C')
    pdf.ln()
    
    # Table rows
    for i, r in enumerate(results):
        if i % 2 == 0:
            pdf.set_fill_color(248, 248, 252)
        else:
            pdf.set_fill_color(255, 255, 255)
        
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(60, 60, 80)
        pdf.cell(8, 7, str(i+1), border=1, fill=True, align='C')
        pdf.cell(62, 7, r['filename'][:30], border=1, fill=True)
        
        if r['label'] == 'Defective':
            pdf.set_text_color(239, 68, 68)
        else:
            pdf.set_text_color(22, 163, 74)
        pdf.set_font('Helvetica', 'B', 9)
        pdf.cell(30, 7, r['label'], border=1, fill=True, align='C')
        
        pdf.set_text_color(60, 60, 80)
        pdf.set_font('Helvetica', '', 9)
        pdf.cell(30, 7, f'{r["confidence"]:.1%}', border=1, fill=True, align='C')
        pdf.cell(30, 7, f'{r["probabilities"]["Defective"]:.4f}', border=1, fill=True, align='C')
        pdf.cell(25, 7, f'{r["probabilities"]["Good"]:.4f}', border=1, fill=True, align='C')
        pdf.ln()
    
    # Footer
    pdf.set_y(-25)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(160, 160, 180)
    pdf.cell(0, 5, 'Generated by STYRS Solar Cell Inspector v2.0', align='C', ln=True)
    pdf.cell(0, 5, 'Powered by EfficientNetB3 Deep Learning', align='C')
    
    return bytes(pdf.output())


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>STYRS</h1>
    <p>Solar Cell Defect Detection &mdash; AI-Powered Quality Inspector</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ STYRS Inspector")
    st.markdown("---")

    if model is not None:
        arch = detect_architecture(model)
        inp = get_model_input_size(model)
        n_params = model.count_params()

        st.markdown(f"""
        <div class="glass-card" style="padding:1rem">
            <div class="sidebar-stat"><span class="label">Architecture</span><span class="value">{arch}</span></div>
            <div class="sidebar-stat"><span class="label">Input Size</span><span class="value">{inp[0]}×{inp[1]}</span></div>
            <div class="sidebar-stat"><span class="label">Parameters</span><span class="value">{n_params/1e6:.1f}M</span></div>
            <div class="sidebar-stat"><span class="label">Best Accuracy</span><span class="value" style="color:#86efac">89.24%</span></div>
            <div class="sidebar-stat"><span class="label">Classes</span><span class="value">2</span></div>
            <div class="sidebar-stat" style="border:none"><span class="label">Status</span><span class="value" style="color:#86efac">● Active</span></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Model not loaded")

    st.markdown("---")
    st.markdown("### Session Stats")
    st.markdown(f"""
    <div class="glass-card" style="padding:1rem">
        <div class="sidebar-stat"><span class="label">Total Analyzed</span><span class="value">{st.session_state.total_analyzed}</span></div>
        <div class="sidebar-stat"><span class="label">Defective</span><span class="value" style="color:#fca5a5">{st.session_state.total_defective}</span></div>
        <div class="sidebar-stat" style="border:none"><span class="label">Good</span><span class="value" style="color:#86efac">{st.session_state.total_good}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:rgba(255,255,255,0.25);font-size:0.75rem'>"
        "STYRS v2.0 — Built with EfficientNetB3</p>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
if model is None:
    st.warning("Model not found! Please ensure `best_model.keras` is in the project directory.")
    st.stop()

# Tabs
tab_single, tab_batch, tab_history = st.tabs(["Single Analysis", "Batch Analysis", "History"])

# ─────────────────────────────────────────────
# TAB 1: SINGLE ANALYSIS
# ─────────────────────────────────────────────
with tab_single:
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Upload Solar Cell Image")
        uploaded_file = st.file_uploader(
            "Drag & drop or click to browse",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            key="single_upload"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            st.image(image, caption=uploaded_file.name, use_container_width=True)

            # Image metadata
            w, h = image.size
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card">
                    <div class="metric-value">{w}×{h}</div>
                    <div class="metric-label">Resolution</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{uploaded_file.size/1024:.0f}KB</div>
                    <div class="metric-label">File Size</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        analyze_btn = st.button("Analyze Image", use_container_width=True, disabled=uploaded_file is None)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        if analyze_btn and uploaded_file:
            with st.spinner(""):
                st.markdown('<div class="glass-card analyzing">', unsafe_allow_html=True)
                st.markdown("#### Analyzing...")

                # Progress animation
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.008)
                    progress_bar.progress(i + 1)

                # Predict
                result = predict_single(image, model)
                add_to_history(uploaded_file.name, result)

                progress_bar.empty()
                st.markdown('</div>', unsafe_allow_html=True)

            # Result display
            is_defective = result['label'] == 'Defective'
            css_class = "result-defective" if is_defective else "result-good"
            icon = "⚠️" if is_defective else "✅"

            st.markdown(f"""
            <div class="{css_class}">
                <h2>{icon} {result['label']}</h2>
                <p>Confidence: {result['confidence']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

            # Confidence bars
            st.markdown("#### Probability Distribution")
            for cls, prob in result['probabilities'].items():
                bar_class = "conf-bar-inner-bad" if cls == "Defective" else "conf-bar-inner-good"
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.8rem;margin:0.4rem 0">
                    <span style="color:rgba(255,255,255,0.6);width:80px;font-size:0.85rem">{cls}</span>
                    <div class="conf-bar-outer" style="flex:1">
                        <div class="{bar_class}" style="width:{prob*100:.1f}%"></div>
                    </div>
                    <span style="color:#e2e8f0;font-weight:600;width:55px;text-align:right">{prob:.1%}</span>
                </div>
                """, unsafe_allow_html=True)

            # Confidence gauge
            st.markdown("#### Confidence Gauge")
            gauge_fig = create_confidence_gauge(result['confidence'], is_defective)
            _ = st.pyplot(gauge_fig, use_container_width=False)
            plt.close(gauge_fig)

            # PDF Report download
            pdf_bytes = generate_single_pdf(uploaded_file.name, result, image)
            st.download_button(
                "📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"styrs_report_{uploaded_file.name.split('.')[0]}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        elif not uploaded_file:
            st.markdown("""
            <div class="glass-card" style="text-align:center;padding:4rem 2rem">
                <div style="font-size:4rem;margin-bottom:1rem">🔬</div>
                <h3 style="color:rgba(255,255,255,0.6)">Upload an Image</h3>
                <p style="color:rgba(255,255,255,0.35)">
                    Upload a solar cell image on the left to start the AI-powered defect analysis. 
                    Supports JPG, PNG, BMP and TIFF formats.
                </p>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB 2: BATCH ANALYSIS
# ─────────────────────────────────────────────
with tab_batch:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Batch Solar Cell Analysis")
    st.markdown("Upload multiple images for rapid quality inspection")

    batch_files = st.file_uploader(
        "Upload multiple solar cell images",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="batch_upload"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if batch_files:
        if st.button(f"Analyze {len(batch_files)} Images", use_container_width=True):
            results = []
            progress = st.progress(0, text="Analyzing images...")

            for i, f in enumerate(batch_files):
                img = Image.open(f)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                r = predict_single(img, model)
                r['filename'] = f.name
                results.append(r)
                add_to_history(f.name, r)
                progress.progress((i + 1) / len(batch_files), text=f"Analyzing {i+1}/{len(batch_files)}...")

            progress.empty()

            # Summary
            n_def = sum(1 for r in results if r['label'] == 'Defective')
            n_good = sum(1 for r in results if r['label'] == 'Good')
            avg_conf = np.mean([r['confidence'] for r in results])

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card">
                    <div class="metric-value">{len(results)}</div>
                    <div class="metric-label">Total Analyzed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="background:linear-gradient(135deg,#ef4444,#f87171);-webkit-background-clip:text;-webkit-text-fill-color:transparent">{n_def}</div>
                    <div class="metric-label">Defective</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="background:linear-gradient(135deg,#22c55e,#4ade80);-webkit-background-clip:text;-webkit-text-fill-color:transparent">{n_good}</div>
                    <div class="metric-label">Good</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{avg_conf:.1%}</div>
                    <div class="metric-label">Avg Confidence</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Defect rate progress
            defect_rate = n_def / len(results) if results else 0
            st.markdown(f"**Defect Rate: {defect_rate:.1%}**")
            st.progress(defect_rate)

            # Results grid
            st.markdown("#### Detailed Results")
            cols = st.columns(4)
            for i, r in enumerate(results):
                with cols[i % 4]:
                    img = Image.open(batch_files[i])
                    st.image(img, use_container_width=True)
                    badge = "history-badge-bad" if r['label'] == 'Defective' else "history-badge-good"
                    st.markdown(f"""
                    <div style="text-align:center;margin-bottom:1rem">
                        <span class="{badge}">{r['label']}</span>
                        <p style="font-size:0.75rem;color:rgba(255,255,255,0.4);margin-top:0.3rem">{r['confidence']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # CSV Export
            df = pd.DataFrame([{
                'Filename': r['filename'],
                'Result': r['label'],
                'Confidence': f"{r['confidence']:.2%}",
                'P(Defective)': f"{r['probabilities']['Defective']:.4f}",
                'P(Good)': f"{r['probabilities']['Good']:.4f}"
            } for r in results])

            csv = df.to_csv(index=False)
            col_csv, col_pdf = st.columns(2)
            with col_csv:
                st.download_button(
                    "📊 Download CSV",
                    csv, "styrs_batch_results.csv",
                    "text/csv",
                    use_container_width=True
                )
            with col_pdf:
                batch_pdf = generate_batch_pdf(results, batch_files)
                st.download_button(
                    "📄 Download PDF Report",
                    data=batch_pdf,
                    file_name=f"styrs_batch_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

    else:
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:3rem 2rem">
            <div style="font-size:3rem;margin-bottom:0.8rem">📦</div>
            <h4 style="color:rgba(255,255,255,0.5)">Batch Upload</h4>
            <p style="color:rgba(255,255,255,0.3);font-size:0.9rem">
                Upload multiple images at once for rapid quality control inspection.
                Get a comprehensive report with CSV export.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB 3: HISTORY
# ─────────────────────────────────────────────
with tab_history:
    if st.session_state.history:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f"#### Analysis History ({len(st.session_state.history)} results)")
        st.markdown('</div>', unsafe_allow_html=True)

        for item in st.session_state.history:
            badge = "history-badge-bad" if item['label'] == 'Defective' else "history-badge-good"
            icon = "⚠️" if item['label'] == 'Defective' else "✅"
            st.markdown(f"""
            <div class="history-item">
                <span style="color:rgba(255,255,255,0.3);font-size:0.8rem;width:60px">{item['time']}</span>
                <span style="color:#e2e8f0;flex:1;font-weight:500">{item['file']}</span>
                <span class="{badge}">{icon} {item['label']}</span>
                <span style="color:rgba(255,255,255,0.5);font-weight:600;width:55px;text-align:right">{item['confidence']:.0%}</span>
            </div>
            """, unsafe_allow_html=True)

        if st.button("Clear History", use_container_width=True):
            st.session_state.history = []
            st.session_state.total_analyzed = 0
            st.session_state.total_defective = 0
            st.session_state.total_good = 0
            st.rerun()
    else:
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:3rem 2rem">
            <div style="font-size:3rem;margin-bottom:0.8rem">📋</div>
            <h4 style="color:rgba(255,255,255,0.5)">No History Yet</h4>
            <p style="color:rgba(255,255,255,0.3);font-size:0.9rem">
                Analyze some images and your results will appear here.
            </p>
        </div>
        """, unsafe_allow_html=True)
