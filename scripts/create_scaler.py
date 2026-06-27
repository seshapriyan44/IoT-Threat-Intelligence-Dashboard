import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET = PROJECT_ROOT / "dataset" / "1.benign.csv"
OUTPUT = PROJECT_ROOT / "models" / "scaler.pkl"

print("Loading benign dataset...")
df = pd.read_csv(DATASET)

print(f"Rows: {len(df):,}")

scaler = MinMaxScaler()
scaler.fit(df.values)

joblib.dump(scaler, OUTPUT)

print(f"\nScaler saved successfully!")
print(f"Location: {OUTPUT}")