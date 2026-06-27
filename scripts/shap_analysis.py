import suppress_logs
import pandas as pd
import numpy as np
import shap

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

print("Loading model...")

model = load_model(
    "../models/autoencoder_model.h5",
    compile=False
)

print("Loading dataset...")

df = pd.read_csv(
    "../dataset/1.benign.csv"
)

# Use only first 1000 samples for SHAP
df = df.iloc[:1000]

scaler = MinMaxScaler()

X = scaler.fit_transform(df)

print("Creating background dataset...")

background = X[:100]

print("Creating SHAP explainer...")

def predict_fn(data):

    reconstructed = model.predict(
        data,
        verbose=0
    )

    mse = np.mean(
        np.square(data - reconstructed),
        axis=1
    )

    return mse

explainer = shap.KernelExplainer(
    predict_fn,
    background
)

print("Calculating SHAP values...")

sample_data = X[100:120]

shap_values = explainer.shap_values(
    sample_data
)

print("Generating summary plot...")

shap.summary_plot(
    shap_values,
    sample_data,
    feature_names=df.columns,
    show=False
)

import matplotlib.pyplot as plt

plt.tight_layout()

plt.savefig(
    "../results/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

print("SHAP plot saved as shap_summary.png in the results folder")