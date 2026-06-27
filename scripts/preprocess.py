import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load normal traffic
df = pd.read_csv(
    "../dataset/1.benign.csv"
)

print("Original Shape:", df.shape)

# Normalize features between 0 and 1
scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(df)

print("Scaled Shape:", X_scaled.shape)

# Split into train/test
X_train, X_test = train_test_split(
    X_scaled,
    test_size=0.2,
    random_state=42
)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)