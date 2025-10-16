import pandas as pd
from model import train_model

# Charger les données
df = pd.read_csv("clean_data.csv")

X = df.drop(columns=["Delivery_Time_min"])
y = df["Delivery_Time_min"]

# Entraîner avec GridSearch
model, r2, mae = train_model(X, y)

print("\nEntraînement terminé avec succès !")
