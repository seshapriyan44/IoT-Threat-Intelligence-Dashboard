import suppress_logs

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Load dataset
df = pd.read_csv(
    "../dataset/1.benign.csv"
)

# Scale data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df)

# Train/Test split
X_train, X_test = train_test_split(
    X_scaled,
    test_size=0.2,
    random_state=42
)

# AutoEncoder Architecture
input_dim = X_train.shape[1]

input_layer = Input(shape=(input_dim,))

# Encoder
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(16, activation='relu')(encoded)

# Decoder
decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)

autoencoder.compile(
    optimizer='adam',
    loss='mse'
)

autoencoder.summary()

history = autoencoder.fit(
    X_train,
    X_train,
    epochs=20,
    batch_size=256,
    validation_data=(X_test, X_test),
    verbose=1
)

autoencoder.save("../models/autoencoder_model.h5")

print("Model saved successfully!")