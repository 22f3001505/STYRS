<p align="center">
  <img src="https://img.shields.io/badge/STYRS-Solar%20Cell%20Inspector-8B5CF6?style=for-the-badge&logo=lightning&logoColor=white" alt="STYRS Badge"/>
</p>

<h1 align="center">⚡ STYRS — Solar Cell Defect Detection</h1>

<p align="center">
  <strong>AI-Powered Quality Inspector for Solar Cell Manufacturing</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-89.24%25-22c55e?style=flat-square" alt="Accuracy"/>
  <img src="https://img.shields.io/badge/Model-EfficientNetB3-6366f1?style=flat-square" alt="Model"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.19-ff6f00?style=flat-square&logo=tensorflow" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Android-APK-34a853?style=flat-square&logo=android" alt="Android"/>
</p>

---

## 🎯 Overview

STYRS is a comprehensive solar cell defect detection platform that uses deep learning to identify manufacturing defects in photovoltaic cells. It combines a high-accuracy EfficientNetB3 model with a premium web interface and a native Android application.

### Key Features

- 🧠 **89.24% Accuracy** — EfficientNetB3 trained on 6 combined datasets
- 🎨 **Premium Web UI** — Dark theme with glassmorphism, confidence gauges, and animations
- 📱 **Android App** — Native mobile app for on-site quality inspection
- 📊 **Batch Analysis** — Process multiple images with CSV/PDF export
- 📋 **PDF Reports** — Professional inspection reports with STYRS branding
- 📈 **Real-time History** — Track all analysis results in session
- 🔥 **REST API** — Flask-based prediction endpoint for integrations

---

## 📁 Project Structure

```
STYRS/
├── app.py                    # Premium Streamlit web UI (main application)
├── api_server.py             # Flask REST API server
├── best_model.keras          # Trained EfficientNetB3 model (161 MB)
├── deploy_hf.py              # Hugging Face Spaces deployment script
├── requirements.txt          # Python dependencies
├── socell-v2.apk             # Android app (7.4 MB)
│
├── hf_deploy/                # Cloud deployment package
│   ├── app.py                # Streamlined app for HF Spaces
│   ├── requirements.txt      # HF-specific dependencies
│   └── README.md             # HF Space description
│
├── SolarCellDetector/        # Android source code (Kotlin)
│   ├── app/
│   │   ├── build.gradle      # App build config (AGP 9.0.1)
│   │   └── src/main/java/com/styrs/solarcell/
│   │       ├── ui/           # Activities and UI
│   │       ├── api/          # Retrofit API client
│   │       └── model/        # Data models
│   ├── build.gradle          # Root build config
│   └── gradle/wrapper/       # Gradle 9.3.1
│
├── train_ultra_colab.py      # Full training script (EfficientNetB3, 6 datasets)
├── train_optimized_colab.py  # Lightweight training script (EfficientNetB0)
├── download_and_train.py     # Local training script
└── mock_data.py              # Mock data generator for testing
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- TensorFlow 2.19+

### 1. Clone & Install

```bash
cd ~/Desktop/STYRS
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
python api_server.py
# → Running on http://localhost:5001
```

### 3. Launch the Web UI

```bash
streamlit run app.py
# → Running on http://localhost:8501
```

### 4. Open in Browser

Navigate to **[http://localhost:8501](http://localhost:8501)** and upload a solar cell image!

---

## 🧠 Model Architecture

| Property | Value |
|----------|-------|
| Architecture | EfficientNetB3 |
| Input Size | 300 × 300 × 3 |
| Parameters | 13.3M |
| Best Accuracy | 89.24% |
| Classes | 2 (Defective / Good) |
| Training | Google Colab T4 GPU |

### Training Datasets (6 sources)

1. **ELPV Dataset** — Electroluminescence images of solar cells
2. **Solar Panel Images** — Kaggle defect classification dataset
3. **Infrared Solar Modules** — Thermal imaging defect detection
4. **PV Multi-Defect** — Multi-class photovoltaic defects
5. **Dust Detection** — Dust/soiling impact dataset
6. **Synthetic Augmentation** — Generated with CutMix, MixUp, GridMask

### Training Strategy

- 5-phase progressive unfreezing
- Stochastic Weight Averaging (SWA)
- Advanced augmentation (CutMix, MixUp, GridMask, CoarseDropout)
- 10-pass Test-Time Augmentation (TTA)

---

## 🎨 Web Interface

The Streamlit app features a premium dark-mode UI:

### Single Analysis
- Upload any solar cell image (JPG, PNG, BMP, TIFF)
- AI-powered classification with confidence scores
- Semicircular confidence gauge visualization
- Probability distribution bars
- One-click PDF report download

### Batch Analysis
- Upload multiple images for bulk inspection
- Summary dashboard with defect rate visualization
- Results grid with thumbnails
- CSV and PDF export

### Analysis History
- Session-persistent result tracking
- Filterable history with badges
- One-click clear

---

## 📱 Android App

The native Android app (`socell-v2.apk`) provides mobile inspection:

| Property | Value |
|----------|-------|
| Size | 7.4 MB |
| Min SDK | 24 (Android 7.0) |
| Target SDK | 35 (Android 15) |
| Language | Kotlin |
| Build System | Gradle 9.3.1 + AGP 9.0.1 |

### Features
- Camera capture for real-time analysis
- Gallery image selection
- On-device result display with confidence
- Connected to REST API backend

### Install
```bash
adb install socell-v2.apk
```

---

## 🔌 REST API

### Health Check
```bash
curl http://localhost:5001/health
```
```json
{"status": "healthy", "model_loaded": true}
```

### Predict
```bash
curl -X POST -F "image=@solar_cell.jpg" http://localhost:5001/predict
```
```json
{
  "success": true,
  "predicted_class": "Defective",
  "confidence": 0.98,
  "probabilities": {
    "defective": 0.98,
    "good": 0.02
  }
}
```

---

## ☁️ Cloud Deployment

### Hugging Face Spaces (Free)

```bash
python deploy_hf.py
```

1. Get a token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Paste it when prompted
3. Your app will be live at `https://huggingface.co/spaces/YOUR_USERNAME/styrs-solar-inspector`

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | TensorFlow 2.19 + Keras 3 |
| Model | EfficientNetB3 (ImageNet pretrained) |
| Web UI | Streamlit 1.41 |
| API | Flask + Gunicorn |
| PDF Reports | FPDF2 |
| Android | Kotlin + Retrofit + AGP 9.0.1 |
| Build | Gradle 9.3.1 |
| Training | Google Colab (T4 GPU) |

---

## 📄 License

This project is proprietary. All rights reserved.

---

<p align="center">
  <strong>Built with ⚡ by STYRS</strong><br/>
  <em>AI-Powered Solar Cell Quality Inspection</em>
</p>
