import suppress_logs

import os
import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import json

from tensorflow.keras.models import load_model

from feature_explainer import explain_feature


with open("models/thresholds.json", "r") as f:
    THRESHOLDS = json.load(f)

def predict_attack(
    csv_path,
    model_type
):

    if model_type == "Centralized":

        model = load_model(
            "models/autoencoder_model.h5",
            compile=False
        )

        threshold = THRESHOLDS["centralized"]

    else:

        model = load_model(
            "models/federated_autoencoder.keras",
            compile=False
        )

        threshold = THRESHOLDS["federated"]

    # Load CSV

    data = pd.read_csv(
        csv_path
    )

    # Normalize using the scaler fitted during training
    
    scaler = joblib.load("models/scaler.pkl")
    X = scaler.transform(data).astype(np.float32)

    # Reconstruction

    reconstructed = model.predict(
        X,
        verbose=0
    )

    # Reconstruction Error

    mse = np.mean(
        np.square(
            X - reconstructed
        ),
        axis=1
    )

    avg_error = float(
        np.mean(
            mse
        )
    )

    status = (
        "ATTACK"
        if avg_error > threshold
        else "BENIGN"
    )

    # Feature-wise error

    feature_errors = np.mean(

        np.square(
            X - reconstructed
        ),

        axis=0

    )

    top_features = (

        pd.Series(

            feature_errors,

            index=data.columns

        )

        .sort_values(
            ascending=False
        )

        .head(5)

    )

   # -----------------------------
    # Generate Feature Chart
    # -----------------------------

    os.makedirs(
        "static/charts",
        exist_ok=True
    )

    # Same palette as evaluation charts
    PALETTE = {
        "bg": "#0f131a",
        "surface": "#1b1f27",
        "border": "#353b46",
        "text": "#e6edf3",
        "muted": "#9aa4b2",
        "bar": "#2f80ed",
    }

    fig, ax = plt.subplots(figsize=(7, 4))

    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["surface"])

    ax.barh(
        top_features.index[::-1],
        top_features.values[::-1],
        color=PALETTE["bar"]
    )

    ax.set_title(
        "Top Feature Contributions",
        color=PALETTE["text"],
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    ax.set_xlabel(
        "Contribution Score",
        color=PALETTE["muted"],
        fontsize=10,
    )

    ax.tick_params(
        axis="x",
        colors=PALETTE["muted"]
    )

    ax.tick_params(
        axis="y",
        colors=PALETTE["text"]
    )

    for spine in ax.spines.values():
        spine.set_color(PALETTE["border"])

    ax.grid(
        axis="x",
        color=PALETTE["border"],
        alpha=0.35,
        linestyle="--",
    )

    ax.set_axisbelow(True)

    plt.tight_layout()

    chart_path = "static/charts/feature_importance.png"

    fig.savefig(
        chart_path,
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )

    plt.close(fig)

    # -----------------------------
    # XAI
    # -----------------------------

    xai_features = []

    for feature, score in top_features.items():

        xai_features.append({

            "name": feature,

            "friendly_name": explain_feature(feature),

            "description": explain_feature(feature),

            "score": float(score)

        })
    
    ratio = avg_error / threshold

    threat_score = min(ratio * 100, 100)

    return {

        "status": status,

        "error": round(avg_error,6),

        "threshold": round(threshold,6),

        "error_ratio": round(ratio, 3),
        
        "threat_score": round(threat_score, 1),

        "top_features": xai_features,

        "feature_chart":
            "charts/feature_importance.png"

    }