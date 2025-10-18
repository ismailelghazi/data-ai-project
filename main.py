import pandas as pd
from model import train_model

# Load data
df = pd.read_csv("clean_data.csv")

# Separate features and target
X = df.drop(columns=["Delivery_Time_min"])
y = df["Delivery_Time_min"]

# Train models (RandomForest + SVR)
model, r2, mae = train_model(X, y)

print("\nTraining finished successfully!")
print(f"Best R2: {r2:.3f}")
print(f"Best MAE: {mae:.3f}")
