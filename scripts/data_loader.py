import pandas as pd

import joblib
from sklearn.model_selection import train_test_split

def load_device_data(device_id):

    file_path = rf"../dataset/{device_id}.benign.csv"

    df = pd.read_csv(file_path)

    scaler = joblib.load("../models/scaler.pkl")

    X = scaler.transform(df)

    X_train, X_test = train_test_split(
        X,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test