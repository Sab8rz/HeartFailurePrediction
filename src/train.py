import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def main():
    df = pd.read_csv("data/processed/heart_processed.csv")
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs("data/processed", exist_ok=True)
    pd.concat([X_test, y_test], axis=1).to_csv("data/processed/test_set.csv", index=False)

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    print("Обучение logistic_regression...")
    scaler = StandardScaler()
    X_train_lr = scaler.fit_transform(X_train)
    lr = LogisticRegression(random_state=42, max_iter=2000)
    lr.fit(X_train_lr, y_train)

    cv_lr = cross_validate(lr, X_train_lr, y_train, cv=5, scoring=scoring, n_jobs=-1)
    lr_metrics = {metric: float(cv_lr[f"test_{metric}"].mean()) for metric in scoring}
    with open("reports/logistic_regression_metrics.json", "w") as f:
        json.dump(lr_metrics, f, indent=4)

    joblib.dump(lr, "models/logistic_regression.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("Обучение random_forest...")
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(X_train, y_train)

    cv_rf = cross_validate(rf, X_train, y_train, cv=5, scoring=scoring, n_jobs=-1)
    rf_metrics = {metric: float(cv_rf[f"test_{metric}"].mean()) for metric in scoring}
    with open("reports/random_forest_metrics.json", "w") as f:
        json.dump(rf_metrics, f, indent=4)

    joblib.dump(rf, "models/random_forest.pkl")

    print("Обучение xgboost...")
    xgb = XGBClassifier(random_state=42, eval_metric="logloss")
    xgb.fit(X_train, y_train)

    cv_xgb = cross_validate(xgb, X_train, y_train, cv=5, scoring=scoring, n_jobs=-1)
    xgb_metrics = {metric: float(cv_xgb[f"test_{metric}"].mean()) for metric in scoring}
    with open("reports/xgboost_metrics.json", "w") as f:
        json.dump(xgb_metrics, f, indent=4)

    joblib.dump(xgb, "models/xgboost.pkl")


if __name__ == "__main__":
    main()
