import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import  RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def load_data(path):
    df = pd.read_csv(path)

    df.drop(columns=['customerID', 'gender', 'MonthlyCharges', 'tenure'], inplace=True)

    df.drop_duplicates(inplace=True)

    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

    return df

def prepare_data(df, target='Churn'):
    df[target] = df[target].map({'Yes': 1, 'No': 0})
    X = df.drop(columns=[target])
    y = df[target]

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    preproc = ColumnTransformer([
       
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    X_train_t = preproc.fit_transform(X_train)
    X_test_t = preproc.transform(X_test)

    return X_train_t, X_test_t, y_train, y_test, preproc

def train_models(X_train, y_train):
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
        'SVM': SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    }

    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models

def evaluate_models(models, X_test, y_test):
    results = []

    print("\nModel Evaluation Results:")
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

        results.append({'Model': name, 'Accuracy': acc, 'Recall': rec, 'F1': f1, 'ROC-AUC': roc})

        print(f"\n{name}")
        print(f"Accuracy: {acc:.3f}")
        print(f"Recall:   {rec:.3f}")
        print(f"F1-score: {f1:.3f}")
        print(f"ROC-AUC:  {roc:.3f}")
        print("\nClassification Report:")
        # print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

        # cm = confusion_matrix(y_test, y_pred)
        # plt.figure(figsize=(4, 3))
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
        #             xticklabels=["No Churn", "Churn"],
        #             yticklabels=["No Churn", "Churn"])
        # plt.title(f"Confusion Matrix - {name}")
        # plt.xlabel("Predicted")
        # plt.ylabel("Actual")
        # plt.show()
    
    results_df = pd.DataFrame(results)
    # plt.figure(figsize=(6, 4))
    # sns.barplot(data=results_df, x='Model', y='F1')
    # plt.title("F1-score Comparison by Model")
    # plt.ylabel("F1-score")
    # plt.ylim(0, 1)
    # plt.show()

    return results_df

# def show_errors(model, X_test, y_test, preproc, df_original):
    try:
        X_test_df = pd.DataFrame(preproc.inverse_transform(X_test),
                                 columns=df_original.drop(columns=['Churn']).columns)
    except Exception:
        print("Inverse transform not supported for this preprocessor — skipping detailed view.")
        return

    y_pred = model.predict(X_test)
    errors = X_test_df.copy()
    errors['Actual'] = y_test.values
    errors['Predicted'] = y_pred
    errors = errors[errors['Actual'] != errors['Predicted']]

    print("\nExample of misclassified rows:")
    print(errors.head(10))
    print(f"\nTotal misclassified: {len(errors)} out of {len(y_test)}")

print("Loading data...")
df = load_data('Data.csv')
print("Preparing data...")
X_train, X_test, y_train, y_test, preproc = prepare_data(df)
print("Training models...")
models = train_models(X_train, y_train)
print("Evaluating models...")
results = evaluate_models(models, X_test, y_test)
# print("\nAnalyzing errors for best model (XGBoost)...")
# show_errors(models['XGBoost'], X_test, y_test, preproc, df)