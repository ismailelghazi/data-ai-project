from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error

def train_model(X, y):
    """
    Entraîne un modele RandomForest avec GridSearchCV
    et retourne le meilleur modele + ses scores.
    """

    # 1Séparer les données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Définir le modele de base
    model = RandomForestRegressor(random_state=42)

    # Définir la grille de paramètres à tester
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
    }

    # GridSearchCV (validation croisée)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',  # on évalue avec MAE
        cv=5,
        n_jobs=-1,
        verbose=1
    )


    #  Entraînement
    grid_search.fit(X_train, y_train)

    # Meilleur modèle trouvé
    best_model = grid_search.best_estimator_
    print("\n Meilleurs paramètres :", grid_search.best_params_)

    # Évaluer sur le jeu de test
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"R² : {r2:.3f}")
    print(f"MAE : {mae:.3f}")

    return best_model, r2, mae
