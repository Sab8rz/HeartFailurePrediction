import json
import os
import tempfile

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.evaluate import evaluate_model

from .test_train import df as test_set


def test_evaluate(test_set, monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        orig_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            os.makedirs(os.path.join("data", "processed"), exist_ok=True)
            os.makedirs(os.path.join("models"), exist_ok=True)
            os.makedirs(os.path.join("reports"), exist_ok=True)

            test_set_path = os.path.join("data", "processed", "test_set.csv")
            test_set.to_csv(test_set_path, index=False)

            model = LogisticRegression()
            model.fit(test_set.drop("HeartDisease", axis=1), test_set["HeartDisease"])
            model_path = os.path.join("models", "logistic_regression.pkl")
            joblib.dump(model, model_path)

            scaler = StandardScaler()
            scaler.fit(test_set.drop("HeartDisease", axis=1))
            scaler_path = os.path.join("models", "scaler.pkl")
            joblib.dump(scaler, scaler_path)

            evaluate_model("logistic_regression")

            report_path = os.path.join("reports", "logistic_regression_metrics.json")
            assert os.path.exists(report_path)

            with open(report_path, "r") as f:
                metrics = json.load(f)
                assert isinstance(metrics, dict)
                assert "accuracy" in metrics
                assert "roc_auc" in metrics
        finally:
            os.chdir(orig_cwd)
