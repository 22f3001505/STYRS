"""
STYRS — Solar Cell Defect Detection Platform
Deployed on Hugging Face Spaces
"""

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image
from fpdf import FPDF
import tempfile
import os
import io
import time
import datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%); }
.main .block-container { padding-top: 2rem; max-width: 1200px; }

.hero-header {
    text-align: center; padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.10) 100%);
    border-radius: 24px; border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 2rem; backdrop-filter: blur(20px);
}
.hero-header h1 {
    background: linear-gradient(135deg, #818cf8, #a78bfa, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 2.6rem; font-weight: 800; letter-spacing: -1px; margin-bottom: 0.3rem;
}
.hero-header p { color: rgba(255,255,255,0.55); font-size: 1.05rem; font-weight: 300; }

.glass-card {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px; padding: 1.8rem; backdrop-filter: blur(16px);
    margin-bottom: 1.2rem; transition: all 0.3s ease;
}
.glass-card:hover { border-color: rgba(129,140,248,0.25); box-shadow: 0 8px 32px rgba(99,102,241,0.12); }

.result-defective {
    background: linear-gradient(135deg, rgba(239,68,68,0.12) 0%, rgba(220,38,38,0.08) 100%);
    border: 1px solid rgba(239,68,68,0.3); border-radius: 20px; padding: 2rem; text-align: center;
}
.result-defective h2 { color: #fca5a5; font-size: 2rem; margin: 0.5rem 0; }
.result-defective p { color: rgba(252,165,165,0.8); font-size: 1.1rem; }

.result-good {
    background: linear-gradient(135deg, rgba(34,197,94,0.12) 0%, rgba(22,163,74,0.08) 100%);
    border: 1px solid rgba(34,197,94,0.3); border-radius: 20px; padding: 2rem; text-align: center;
}
.result-good h2 { color: #86efac; font-size: 2rem; margin: 0.5rem 0; }
.result-good p { color: rgba(134,239,172,0.8); font-size: 1.1rem; }

.metric-row { display: flex; gap: 1rem; margin: 1.2rem 0; }
.metric-card {
    flex: 1; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.2rem; text-align: center; transition: all 0.2s ease;
}
.metric-card:hover { transform: translateY(-2px); border-color: rgba(129,140,248,0.3); }
.metric-card .metric-value {
    font-size: 1.8rem; font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-card .metric-label {
    color: rgba(255,255,255,0.45); font-size: 0.78rem;
    text-transform: uppercase; letter-spacing: 1px; margin-top: 0.3rem;
}

.conf-bar-outer { background: rgba(255,255,255,0.06); border-radius: 12px; height: 14px; margin: 0.6rem 0; overflow: hidden; }
.conf-bar-inner-good { height: 100%; border-radius: 12px; background: linear-gradient(90deg, #22c55e, #4ade80); transition: width 1s ease; }
.conf-bar-inner-bad { height: 100%; border-radius: 12px; background: linear-gradient(90deg, #ef4444, #f87171); transition: width 1s ease; }

[data-testid="stSidebar"] { background: linear-gradient(180deg, #13111c 0%, #1a1a2e 100%); border-right: 1px solid rgba(255,255,255,0.06); }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #c4b5fd !important; }
.sidebar-stat { display: flex; justify-content: space-between; align-items: center; padding: 0.7rem 0; border-bottom: 1px solid rgba(255,255,255,0.06); }
.sidebar-stat .label { color: rgba(255,255,255,0.45); font-size: 0.85rem; }
.sidebar-stat .value { color: #e2e8f0; font-weight: 600; font-size: 0.9rem; }

.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important; color: white !important;
    border: none !important; border-radius: 14px !important; padding: 0.7rem 2rem !important;
    font-weight: 600 !important; font-size: 1rem !important; font-family: 'Inter', sans-serif !important;
    transition: all 0.3s ease !important; box-shadow: 0 4px 15px rgba(99,102,241,0.25) !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 25px rgba(99,102,241,0.4) !important; }

[data-testid="stFileUploader"] { background: rgba(255,255,255,0.03); border: 2px dashed rgba(129,140,248,0.25); border-radius: 20px; padding: 1rem; }

.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; background: rgba(255,255,255,0.03); border-radius: 16px; padding: 0.3rem; }
.stTabs [data-baseweb="tab"] { border-radius: 12px; color: rgba(255,255,255,0.5); font-weight: 500; }
.stTabs [aria-selected="true"] { background: rgba(99,102,241,0.2) !important; color: #a5b4fc !important; }

.history-item {
    display: flex; align-items: center; gap: 1rem; padding: 0.8rem 1rem;
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05);
    border-radius: 14px; margin-bottom: 0.5rem; transition: all 0.2s ease;
}
.history-item:hover { background: rgba(255,255,255,0.06); border-color: rgba(129,140,248,0.2); }
.history-badge-good { background: rgba(34,197,94,0.15); color: #86efac; padding: 0.25rem 0.8rem; border-radius: 20px; font-size: 0.78rem; font-weight: 600; }
.history-badge-bad { background: rgba(239,68,68,0.15); color: #fca5a5; padding: 0.25rem 0.8rem; border-radius: 20px; font-size: 0.78rem; font-weight: 600; }

h1, h2, h3, h4 { color: #e2e8f0 !important; }
p, label, span { color: rgba(255,255,255,0.7); }
.stProgress > div > div { background: linear-gradient(90deg, #6366f1, #a78bfa) !important; }
/* Hide stray 'None' from Streamlit auto-write */
div[data-testid="stMarkdownContainer"]:has(> p > code:only-child) { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []
if 'total_analyzed' not in st.session_state:
    st.session_state.total_analyzed = 0
if 'total_defective' not in st.session_state:
    st.session_state.total_defective = 0
if 'total_good' not in st.session_state:
    st.session_state.total_good = 0

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource
def load_trained_model():
    model_path = 'best_model.keras'
    if not os.path.exists(model_path):
        return None
    try:
        return load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_trained_model()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_model_input_size(m):
    s = m.input_shape
    return (s[1], s[2])

def detect_architecture(m):
    layer_str = " ".join([l.name for l in m.layers]).lower()
    if "efficientnetb3" in layer_str or "block7" in layer_str:
        return "EfficientNetB3"
    elif "efficientnet" in layer_str:
        return "EfficientNetB0"
    return "Custom CNN"

def preprocess_image(img, target_size=(300, 300)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_single(img, m):
    target_size = get_model_input_size(m)
    processed = preprocess_image(img, target_size)
    prediction = m.predict(processed, verbose=0)
    class_names = ['Defective', 'Good']
    idx = int(np.argmax(prediction, axis=1)[0])
    confidence = float(np.max(prediction))
    return {
        'label': class_names[idx], 'confidence': confidence,
        'probabilities': {'Defective': float(prediction[0][0]), 'Good': float(prediction[0][1])}
    }

def create_confidence_gauge(confidence, is_defective):
    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    theta = np.linspace(0, np.pi, 100)
    ax.plot(theta, [1]*100, color=(1, 1, 1, 0.1), linewidth=12, solid_capstyle='round')
    filled = max(int(confidence * 100), 2)
    color = '#ef4444' if is_defective else '#22c55e'
    theta_filled = np.linspace(0, np.pi * confidence, filled)
    ax.plot(theta_filled, [1]*filled, color=color, linewidth=12, solid_capstyle='round')
    ax.text(np.pi/2, 0.3, f"{confidence:.0%}", ha='center', va='center',
            fontsize=22, fontweight='bold', color='white', family='sans-serif')
    ax.set_ylim(0, 1.5)
    ax.axis('off')
    plt.tight_layout()
    return fig

def add_to_history(filename, result):
    st.session_state.history.insert(0, {
        'time': datetime.datetime.now().strftime("%H:%M:%S"),
        'file': filename, 'label': result['label'], 'confidence': result['confidence']
    })
    st.session_state.total_analyzed += 1
    if result['label'] == 'Defective':
        st.session_state.total_defective += 1
    else:
        st.session_state.total_good += 1
    if len(st.session_state.history) > 50:
        st.session_state.history = st.session_state.history[:50]


def generate_single_pdf(filename, result, image):
    """Generate a professional PDF report for a single analysis"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_fill_color(26, 26, 46)
    pdf.rect(0, 0, 210, 35, 'F')
    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_text_color(129, 140, 248)
    pdf.set_xy(10, 8)
    pdf.cell(0, 12, 'STYRS', ln=False)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(180, 180, 200)
    pdf.set_xy(10, 22)
    pdf.cell(0, 8, 'Solar Cell Defect Detection Report')
    pdf.set_text_color(80, 80, 100)
    pdf.set_font('Helvetica', '', 8)
    pdf.set_xy(135, 10)
    pdf.cell(0, 5, f'Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}')
    pdf.set_xy(135, 16)
    pdf.cell(0, 5, 'Model: EfficientNetB3 (89.24%)')
    pdf.set_xy(135, 22)
    pdf.cell(0, 5, f'File: {filename}')
    pdf.ln(30)
    img_buf = io.BytesIO()
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(img_buf, format='JPEG', quality=85)
    img_buf.seek(0)
    tmp_img = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    tmp_img.write(img_buf.read())
    tmp_img.close()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(40, 40, 60)
    pdf.cell(0, 10, 'Analyzed Image', ln=True)
    pdf.ln(2)
    try:
        pdf.image(tmp_img.name, x=15, w=80)
    except Exception:
        pass
    os.unlink(tmp_img.name)
    is_defective = result['label'] == 'Defective'
    result_y = max(pdf.get_y() - 50, 50)
    if is_defective:
        pdf.set_fill_color(254, 226, 226)
        pdf.set_draw_color(239, 68, 68)
    else:
        pdf.set_fill_color(220, 252, 231)
        pdf.set_draw_color(34, 197, 94)
    pdf.set_xy(105, result_y)
    pdf.rect(105, result_y, 90, 45, 'DF')
    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_text_color(239, 68, 68) if is_defective else pdf.set_text_color(22, 163, 74)
    pdf.set_xy(105, result_y + 5)
    pdf.cell(90, 12, 'DEFECTIVE' if is_defective else 'GOOD', align='C')
    pdf.set_font('Helvetica', '', 12)
    pdf.set_text_color(80, 80, 100)
    pdf.set_xy(105, result_y + 20)
    pdf.cell(90, 8, f'Confidence: {result["confidence"]:.1%}', align='C')
    pdf.set_y(max(pdf.get_y() + 10, result_y + 55))
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(40, 40, 60)
    pdf.cell(0, 10, 'Probability Breakdown', ln=True)
    pdf.ln(3)
    for cls, prob in result['probabilities'].items():
        pdf.set_font('Helvetica', '', 11)
        pdf.set_text_color(60, 60, 80)
        pdf.cell(35, 8, cls)
        bar_x = pdf.get_x()
        bar_y = pdf.get_y() + 1
        pdf.set_fill_color(230, 230, 240)
        pdf.rect(bar_x, bar_y, 110, 6, 'F')
        pdf.set_fill_color(239, 68, 68) if cls == 'Defective' else pdf.set_fill_color(34, 197, 94)
        pdf.rect(bar_x, bar_y, 110 * prob, 6, 'F')
        pdf.set_xy(bar_x + 115, bar_y - 1)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(20, 8, f'{prob:.1%}')
        pdf.ln(10)
    pdf.ln(5)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(40, 40, 60)
    pdf.cell(0, 10, 'Technical Details', ln=True)
    pdf.ln(2)
    for label, value in [('Model', 'EfficientNetB3'), ('Input', '300x300'), ('Accuracy', '89.24%'), ('File', filename), ('Date', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))]:
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(100, 100, 120)
        pdf.cell(55, 7, label)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(40, 40, 60)
        pdf.cell(0, 7, value, ln=True)
    pdf.set_y(-25)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(160, 160, 180)
    pdf.cell(0, 5, 'Generated by STYRS Solar Cell Inspector v2.0', align='C')
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
            <div class="sidebar-stat"><span class="label">Input Size</span><span class="value">{inp[0]}x{inp[1]}</span></div>
            <div class="sidebar-stat"><span class="label">Parameters</span><span class="value">{n_params/1e6:.1f}M</span></div>
            <div class="sidebar-stat"><span class="label">Best Accuracy</span><span class="value" style="color:#86efac">89.24%</span></div>
            <div class="sidebar-stat" style="border:none"><span class="label">Status</span><span class="value" style="color:#86efac">Active</span></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Model not loaded")
    st.markdown("---")
    st.markdown("### Session Stats")
    st.markdown(f"""
    <div class="glass-card" style="padding:1rem">
        <div class="sidebar-stat"><span class="label">Analyzed</span><span class="value">{st.session_state.total_analyzed}</span></div>
        <div class="sidebar-stat"><span class="label">Defective</span><span class="value" style="color:#fca5a5">{st.session_state.total_defective}</span></div>
        <div class="sidebar-stat" style="border:none"><span class="label">Good</span><span class="value" style="color:#86efac">{st.session_state.total_good}</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='text-align:center;color:rgba(255,255,255,0.25);font-size:0.75rem'>STYRS v2.0</p>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if model is None:
    st.warning("Model not found! Ensure best_model.keras is in the project directory.")
    st.stop()

tab_single, tab_batch, tab_history = st.tabs(["Single Analysis", "Batch Analysis", "History"])

with tab_single:
    col_upload, col_result = st.columns([1, 1], gap="large")
    with col_upload:
        st.markdown("#### Upload Solar Cell Image")
        uploaded_file = st.file_uploader("Drag & drop or click to browse",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], key="single_upload")
        if uploaded_file:
            image = Image.open(uploaded_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            st.image(image, caption=uploaded_file.name, use_container_width=True)
            w, h = image.size
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card"><div class="metric-value">{w}x{h}</div><div class="metric-label">Resolution</div></div>
                <div class="metric-card"><div class="metric-value">{uploaded_file.size/1024:.0f}KB</div><div class="metric-label">File Size</div></div>
            </div>
            """, unsafe_allow_html=True)
        analyze_btn = st.button("Analyze Image", use_container_width=True, disabled=uploaded_file is None)

    with col_result:
        if analyze_btn and uploaded_file:
            with st.spinner("Analyzing..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.008)
                    progress_bar.progress(i + 1)
                result = predict_single(image, model)
                add_to_history(uploaded_file.name, result)
                progress_bar.empty()

            is_defective = result['label'] == 'Defective'
            css_class = "result-defective" if is_defective else "result-good"
            icon = "⚠️" if is_defective else "✅"
            st.markdown(f'<div class="{css_class}"><h2>{icon} {result["label"]}</h2><p>Confidence: {result["confidence"]:.1%}</p></div>', unsafe_allow_html=True)

            st.markdown("#### Probability Distribution")
            for cls, prob in result['probabilities'].items():
                bar_class = "conf-bar-inner-bad" if cls == "Defective" else "conf-bar-inner-good"
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.8rem;margin:0.4rem 0">
                    <span style="color:rgba(255,255,255,0.6);width:80px;font-size:0.85rem">{cls}</span>
                    <div class="conf-bar-outer" style="flex:1"><div class="{bar_class}" style="width:{prob*100:.1f}%"></div></div>
                    <span style="color:#e2e8f0;font-weight:600;width:55px;text-align:right">{prob:.1%}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### Confidence Gauge")
            gauge_fig = create_confidence_gauge(result['confidence'], is_defective)
            st.pyplot(gauge_fig, use_container_width=False)
            plt.close(gauge_fig)

            # PDF Report
            st.markdown('<div style="border-top:1px solid rgba(255,255,255,0.08);margin:1.5rem 0"></div>', unsafe_allow_html=True)
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
                <p style="color:rgba(255,255,255,0.35)">Upload a solar cell image to start AI-powered defect analysis.</p>
            </div>
            """, unsafe_allow_html=True)

with tab_batch:
    st.markdown("#### Batch Solar Cell Analysis")
    batch_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True, key="batch_upload")
    if batch_files and st.button(f"Analyze {len(batch_files)} Images", use_container_width=True):
        results = []
        progress = st.progress(0, text="Analyzing...")
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

        n_def = sum(1 for r in results if r['label'] == 'Defective')
        n_good = sum(1 for r in results if r['label'] == 'Good')
        avg_conf = np.mean([r['confidence'] for r in results])
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card"><div class="metric-value">{len(results)}</div><div class="metric-label">Total</div></div>
            <div class="metric-card"><div class="metric-value" style="background:linear-gradient(135deg,#ef4444,#f87171);-webkit-background-clip:text;-webkit-text-fill-color:transparent">{n_def}</div><div class="metric-label">Defective</div></div>
            <div class="metric-card"><div class="metric-value" style="background:linear-gradient(135deg,#22c55e,#4ade80);-webkit-background-clip:text;-webkit-text-fill-color:transparent">{n_good}</div><div class="metric-label">Good</div></div>
            <div class="metric-card"><div class="metric-value">{avg_conf:.1%}</div><div class="metric-label">Avg Confidence</div></div>
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(4)
        for i, r in enumerate(results):
            with cols[i % 4]:
                img = Image.open(batch_files[i])
                st.image(img, use_container_width=True)
                badge = "history-badge-bad" if r['label'] == 'Defective' else "history-badge-good"
                st.markdown(f'<div style="text-align:center;margin-bottom:1rem"><span class="{badge}">{r["label"]}</span><p style="font-size:0.75rem;color:rgba(255,255,255,0.4);margin-top:0.3rem">{r["confidence"]:.1%}</p></div>', unsafe_allow_html=True)

        df = pd.DataFrame([{'Filename': r['filename'], 'Result': r['label'], 'Confidence': f"{r['confidence']:.2%}"} for r in results])
        st.download_button("Download CSV", df.to_csv(index=False), "styrs_results.csv", "text/csv", use_container_width=True)

with tab_history:
    if st.session_state.history:
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
            <p style="color:rgba(255,255,255,0.3);font-size:0.9rem">Analyze images and results will appear here.</p>
        </div>
        """, unsafe_allow_html=True)
