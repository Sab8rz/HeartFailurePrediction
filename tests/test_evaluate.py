import json
import os
import tempfile

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.evaluate import evaluate_model

from .test_train import df as df_set


def test_evaluate(df_set, monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        orig_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            os.makedirs(os.path.join("data", "processed"), exist_ok=True)
            os.makedirs(os.path.join("models"), exist_ok=True)
            os.makedirs(os.path.join("reports"), exist_ok=True)

            df_set_path = os.path.join("data", "processed", "heart_processed.csv")
            df_set.to_csv(df_set_path, index=False)

            df = pd.read_csv(df_set_path)
            X = df.drop("HeartDisease", axis=1)
            y = df["HeartDisease"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            test_set_path = os.path.join("data", "processed", "test_set.csv")
            pd.concat([X_test, y_test], axis=1).to_csv(test_set_path, index=False)

            scaler = StandardScaler()
            X_train_lr = scaler.fit_transform(X_train)
            lr = LogisticRegression(random_state=42, max_iter=2000)
            lr.fit(X_train_lr, y_train)
            joblib.dump(lr, "models/logistic_regression.pkl")
            joblib.dump(scaler, "models/scaler.pkl")

            evaluate_model("logistic_regression")

            report_path = os.path.join("reports", "logistic_regression_metrics.json")
            assert os.path.exists(report_path)

            with open(report_path, "r") as f:
                metrics = json.load(f)
                assert isinstance(metrics, dict)
                expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc"}
                assert expected_keys.issubset(metrics.keys())
        finally:
            os.chdir(orig_cwd)
