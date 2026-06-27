import pandas as pd
import joblib

def load_client_data(csv_file):

    df = pd.read_csv(csv_file)

    X = df.values

    scaler = joblib.load("../models/scaler.pkl")

    X_scaled = scaler.transform(X)

    return X_scaled