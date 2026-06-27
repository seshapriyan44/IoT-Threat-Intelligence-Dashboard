import suppress_logs
import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load model
model = load_model(
    "../models/autoencoder_model.h5",
    compile=False
)

# Load benign data
df = pd.read_csv(
    "../dataset/1.benign.csv"
)

# Scale
scaler = MinMaxScaler()
X = scaler.fit_transform(df)

# Predict
pred = model.predict(X, verbose=0)

# Reconstruction error
mse = np.mean(
    np.square(X - pred),
    axis=1
)

mean_error = np.mean(mse)
std_error = np.std(mse)

threshold = mean_error + (3 * std_error)

print("\nMean Error:", mean_error)
print("Std Error:", std_error)
print("Threshold:", threshold)