import json
import os
import tempfile

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


def test_evaluate():
    with tempfile.TemporaryDirectory() as temp_dir:
        os.makedirs(os.path.join(temp_dir, "data", "processed"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "reports"), exist_ok=True)

        test_set = pd.DataFrame(
            {
                "Age": [40, 50],
                "Sex": [1, 0],
                "ChestPainType_NAP": [0, 1],
                "RestingECG_Normal": [1, 0],
                "MaxHR": [172, 156],
                "ExerciseAngina": [0, 1],
                "Oldpeak": [0.0, 1.0],
                "ST_Slope_Up": [1, 0],
                "HeartDisease": [0, 1],
            }
        )
        test_set_path = os.path.join(
            temp_dir, "data", "processed", "test_set.csv"
        )
        test_set.to_csv(test_set_path, index=False)

        model = LogisticRegression()
        model.fit(
            test_set.drop("HeartDisease", axis=1), test_set["HeartDisease"]
        )
        model_path = os.path.join(
            temp_dir, "models", "logistic_regression.pkl"
        )
        joblib.dump(model, model_path)

        scaler = StandardScaler()
        scaler.fit(test_set.drop("HeartDisease", axis=1))
        scaler_path = os.path.join(temp_dir, "models", "scaler.pkl")
        joblib.dump(scaler, scaler_path)

        def evaluate_in_temp(model_name: str):
            test_df = pd.read_csv(test_set_path)
            X_test = test_df.drop("HeartDisease", axis=1)
            y_test = test_df["HeartDisease"]

            model = joblib.load(
                os.path.join(temp_dir, "models", f"{model_name}.pkl")
            )

            if model_name == "logistic_regression":
                scaler = joblib.load(
                    os.path.join(temp_dir, "models", "scaler.pkl")
                )
                X_test = scaler.transform(X_test)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred)),
                "recall": float(recall_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred)),
                "roc_auc": float(roc_auc_score(y_test, y_proba)),
            }

            report_path = os.path.join(
                temp_dir, "reports", f"{model_name}_metrics.json"
            )
            with open(report_path, "w") as f:
                json.dump(metrics, f, indent=4)

        evaluate_in_temp("logistic_regression")

        report_path = os.path.join(
            temp_dir, "reports", "logistic_regression_metrics.json"
        )
        assert os.path.exists(report_path)

        with open(report_path, "r") as f:
            metrics = json.load(f)
            assert isinstance(metrics, dict)
            assert "accuracy" in metrics
            assert "roc_auc" in metrics
