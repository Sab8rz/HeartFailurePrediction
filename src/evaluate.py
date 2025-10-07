import pandas as pd
import joblib
import json
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate_model(model_name: str):
    # Загрузка тестовых данных
    test_df = pd.read_csv("data/processed/test_set.csv")
    X_test = test_df.drop("HeartDisease", axis=1)
    y_test = test_df["HeartDisease"]

    # Загрузка модели
    model = joblib.load(f"models/{model_name}.pkl")

    # Для логистической регрессии — применяем scaler
    if model_name == "logistic_regression":
        scaler = joblib.load("models/scaler.pkl")
        X_test = scaler.transform(X_test)

    # Предсказания
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Метрики
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    # Сохранение
    os.makedirs("reports", exist_ok=True)
    with open(f"reports/{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Метрики {model_name} сохранены")


def main():
    for model in ["logistic_regression", "random_forest", "xgboost"]:
        evaluate_model(model)


if __name__ == "__main__":
    main()
