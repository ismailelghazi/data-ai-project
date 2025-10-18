import pandas as pd
from model import train_model

def test_train_model():
    df = pd.read_csv("clean_data.csv")
    X = df.drop(columns=["Delivery_Time_min"])
    y = df["Delivery_Time_min"]

    model, r2, mae = train_model(X, y)

    assert r2 > 0, "R² too low"
    assert mae < 20, "MAE too high"
    assert model is not None, "No model returned"

if __name__ == "__main__":
    test_train_model()
