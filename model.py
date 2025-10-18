from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error

def train_model(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_mae = mean_absolute_error(y_test, rf_pred)

    print("Random Forest:")
    print("R2:", round(rf_r2, 3))
    print("MAE:", round(rf_mae, 3))

    # SVR
    svr = SVR()
    svr.fit(X_train, y_train)
    svr_pred = svr.predict(X_test)
    svr_r2 = r2_score(y_test, svr_pred)
    svr_mae = mean_absolute_error(y_test, svr_pred)

    print("\nSVR:")
    print("R2:", round(svr_r2, 3))
    print("MAE:", round(svr_mae, 3))

    # Choose best model
    if rf_r2 > svr_r2:
        print("\nBest model: Random Forest")
        return rf, rf_r2, rf_mae
    else:
        print("\nBest model: SVR")
        return svr, svr_r2, svr_mae
