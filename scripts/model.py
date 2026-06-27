import suppress_logs

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def create_autoencoder(input_dim=115):

    input_layer = Input(shape=(input_dim,))

    encoded = Dense(64, activation="relu")(input_layer)
    encoded = Dense(32, activation="relu")(encoded)
    encoded = Dense(16, activation="relu")(encoded)

    decoded = Dense(32, activation="relu")(encoded)
    decoded = Dense(64, activation="relu")(decoded)
    decoded = Dense(input_dim, activation="sigmoid")(decoded)

    model = Model(inputs=input_layer, outputs=decoded)

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model