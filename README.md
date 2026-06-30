# 🛡️ IoT Threat Intelligence Dashboard

An intelligent, web-based Intrusion Detection System (IDS) for detecting IoT botnet attacks using **Federated Learning**, **Autoencoders**, and **feature-level explainability**.

The system provides real-time threat analysis through an interactive Flask dashboard — upload IoT network traffic data and get attack predictions, threat severity, feature-level explanations, and model performance visualizations, all in one place.

---

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Deployment](#deployment)
- [Evaluating Models](#evaluating-models)
- [Model Evaluation](#model-evaluation)
- [Explainable AI](#explainable-ai)
- [Threat Score](#threat-score)
- [Dataset](#dataset)
- [Improvements Over the Base Paper](#improvements-over-the-base-paper)
- [Future Work](#future-work)
- [Citation](#citation)
- [Author](#author)
- [License](#license)

---

## Features

- 🔒 Federated Learning–based anomaly detection
- 🧠 Centralized vs. Federated Autoencoder comparison
- 📊 Feature-level explainability inspired by SHAP analysis
- 📈 Threat Score derived from reconstruction error
- 📉 Automatically generated confusion matrix
- 📊 Model performance comparison charts
- 🌙 Modern, responsive dark-themed dashboard
- 📂 CSV upload with real-time inference
- ⚡ Fast inference using pre-trained centralized and federated autoencoder models

---

## How It Works

From the dashboard, a user can:

1. Upload an IoT network traffic CSV file
2. Choose a detection model:
   - **Centralized Autoencoder** — trained on the full pooled dataset
   - **Federated Autoencoder** — trained via decentralized, privacy-preserving updates across simulated edge clients
3. Get a prediction — **BENIGN** or **ATTACK** — along with:
   - Reconstruction Error
   - Detection Threshold
   - Threat Score
   - Top contributing features (reconstruction-error–based explainability)
   - Confusion Matrix
   - Model performance comparison

**Pipeline:**

```text
CSV Dataset
      │
      ▼
Feature Preprocessing
      │
      ▼
Global Min-Max Scaling (shared preprocessing)
      │
      ▼
Autoencoder Reconstruction
      │
      ▼
Reconstruction Error
      │
      ▼
Threshold Comparison
      │
      ▼
Benign / Attack Prediction
      │
      ▼
Feature Contribution Analysis
      │
      ▼
Threat Intelligence Dashboard
```

A single Min-Max scaler, fitted on the benign training dataset, is reused consistently during training, evaluation, and live inference — ensuring reproducible preprocessing across the full pipeline.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Backend | Python, Flask |
| Deployment | Gunicorn, Render |
| Machine Learning | TensorFlow, Keras, Scikit-learn, SHAP |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Frontend | HTML5, CSS3, JavaScript |

---

## Project Structure

```text
Code/
│
├── app.py                     # Flask application
├── predict.py                 # Prediction pipeline
├── requirements.txt
├── INSTRUCTIONS.txt
│
├── dataset/                   # N-BaIoT datasets
│
├── models/
│   ├── autoencoder_model.h5
│   ├── federated_autoencoder.keras
│   ├── scaler.pkl
│   └── thresholds.json
│
├── scripts/
│   ├── autoencoder.py
│   ├── evaluate_models.py
│   ├── feature_explainer.py
│   ├── feature_knowledge.py
│   ├── shap_analysis.py
│   ├── preprocess.py
│   ├── data_loader.py
│   ├── fl_data_loader.py
│   ├── fl_client.py
│   ├── fl_client_base.py
│   ├── fl_server.py
│   ├── run_fl.py
│   ├── model.py
│   ├── find_threshold.py
│   ├── find_threshold_federated.py
│   └── suppress_logs.py
│
├── static/
│   ├── charts/
│   ├── css/
│   ├── images/
│   └── js/
│
├── templates/
│
└── uploads/
```

---

## Getting Started

**1. Create a virtual environment**

```bash
# Windows
py -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the Flask server**

```bash
py app.py        # Windows
python app.py     # macOS / Linux
```

**4. Open the dashboard**

```text
http://127.0.0.1:5000
```

---

## Deployment

This app is deployed on [Render](https://render.com). The steps below cover what's needed beyond local development.

**1. Add a production server**

Flask's built-in dev server isn't meant for production. Add Gunicorn to `requirements.txt`:

```text
gunicorn
```

**2. Add a `Procfile`** (or set this as Render's Start Command)

```text
web: gunicorn app:app
```

**3. Bind to Render's assigned port**

Render injects a `PORT` environment variable at runtime — the app cannot hardcode `5000`. If you're running via Gunicorn this is handled automatically; if `app.run()` is still used anywhere for local fallback, make sure it reads:

```python
import os
app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
```

**4. Push to GitHub and connect the repo on Render**

- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:app`

The trained models (`.h5` / `.keras`), `scaler.pkl`, and `thresholds.json` are loaded at runtime, so the deployed application uses the exact same preprocessing and decision thresholds as the offline evaluation pipeline.

**5. Known limitations on Render's free tier**

- **RAM (512MB cap):** TensorFlow + model loading can be memory-heavy — check actual usage before assuming free tier is sufficient.
- **Ephemeral disk:** the `uploads/` folder does not persist across restarts or redeploys; uploaded CSVs are wiped. This is expected behavior for the current single-session analysis flow, not a bug.
- **Cold starts:** the service spins down after inactivity on the free tier. The first request after idle can take 30+ seconds while it spins back up and TensorFlow re-initializes.

---

## Evaluating Models

To regenerate evaluation metrics and charts:

```bash
cd scripts
py evaluate_models.py
```

This generates:

- Centralized confusion matrix
- Federated confusion matrix
- Model comparison chart
- Metrics JSON
- Feature importance chart

---

## Model Evaluation

Both models are evaluated using:

- Accuracy
- Precision
- Recall
- Specificity (TNR)
- F1-Score
- Confusion Matrix

Evaluation charts are generated automatically from the trained models — no manual plotting required.

### Results

Evaluated on a balanced set of 99,096 samples (49,548 benign / 49,548 attack) from the N-BaIoT dataset:

| Metric | Centralized | Federated |
|---|---:|---:|
| Accuracy | 99.55% | 99.68% |
| Precision | 99.11% | 99.36% |
| Recall | 99.998% | 100.00% |
| Specificity | 99.10% | 99.36% |
| F1 Score | 99.55% | 99.68% |

The Federated model shows a small but consistent edge over Centralized — most notably in specificity (i.e., a lower false positive rate) — which is consistent with the qualitative finding reported in the base paper.

---

## Explainable AI

For every prediction, the dashboard explains *why* a sample was flagged using a feature contribution analysis derived from the autoencoder's per-feature reconstruction error — identifying which input features deviated most from learned normal behavior, then mapping each to a human-readable description (e.g., "abnormal communication behavior was detected between network ports").

A separate offline `shap_analysis.py` script is also included in the project for deeper SHAP (SHapley Additive Explanations) -based feature importance analysis on the trained models.

The dashboard surfaces the **top 5 contributing features** for every uploaded dataset.

---

## Threat Score

The Threat Score gives an intuitive 0–100% read on anomaly severity, while still preserving the underlying reconstruction error for transparency:

```text
Threat Score = min((Reconstruction Error / Detection Threshold) × 100, 100)
```

---

## Dataset

This project uses the **N-BaIoT** dataset for training and evaluation, which contains benign and botnet traffic collected from real IoT devices infected with the Mirai and BASHLITE (Gafgyt) malware.

**Attack classes:**

- Mirai — ACK, SYN, Scan, UDP, UDP-Plain
- Gafgyt (BASHLITE) — Combo, Junk, Scan, TCP, UDP

### Download

The dataset is **not included in this repository** due to its size (several GB). Download it separately from the original source:

- **N-BaIoT dataset:** https://www.kaggle.com/datasets/mkashifn/nbaiot-dataset

### Expected Folder Structure

After downloading, place the required CSV files inside `dataset/`:

```text
Code/
│
├── dataset/
│   ├── 1.benign.csv
│   ├── 1.mirai.ack.csv
│   :   ...
│   └── features.csv
│
├── models/
├── scripts/
├── app.py
└── ...
```

> **Note:** the `1.*.csv` prefix corresponds to device 1 in the N-BaIoT dataset, which is what `evaluate_models.py` and `predict.py` currently expect. The full dataset contains nine devices (`1.*` through `9.*`) — using a different device's files will require updating the filenames referenced in those scripts accordingly.

---

## Improvements Over the Base Paper

This implementation extends the original research with:

- An interactive Flask web application (the original work is research/code only)
- Real-time CSV upload and prediction
- Centralized vs. Federated model selection at inference time
- Feature-level explainability via reconstruction error contribution analysis
- Threat Score visualization
- Interactive model comparison charts
- Confusion matrix visualization
- A modern dashboard interface
- An automated evaluation pipeline
- A unified preprocessing pipeline with a shared Min-Max scaler for reproducible training, evaluation, and inference

---

## Future Work

- Per-flow prediction for batch CSV uploads (currently aggregate-level)
- Live packet capture instead of CSV-only analysis
- Live dashboard updates
- Support for additional IoT intrusion datasets
- Docker deployment
- Persistent storage for uploaded files (e.g. S3-compatible storage) beyond Render's ephemeral disk

---

## Citation

This project's methodology — Federated Averaging, the Autoencoder architecture, and N-BaIoT preprocessing — is based on:

> Olanrewaju-George, B., & Pranggono, B. (2025). *Federated learning-based intrusion detection system for the internet of things using unsupervised and supervised deep learning models.* Cyber Security and Applications, 3, 100068. https://doi.org/10.1016/j.csa.2024.100068

The original methodology has been extended here with a production-oriented web interface, an explainability dashboard, and an enhanced evaluation framework.

---

## Author

**Seshapriyan T**

M.Tech, Computer Science Engineering

SASTRA Deemed University

---

## License

This project is intended for academic and educational use.
