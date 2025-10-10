import json
import os
import tempfile

import joblib
import pandas as pd
from pytest import fixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


@fixture
def df():
    df = pd.DataFrame(
        {
            "Age": [40, 50, 60, 30, 34, 21, 45, 55, 65, 25],
            "Sex": [1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
            "RestingBP": [140, 150, 130, 128, 234, 211, 135, 145, 160, 120],
            "Cholesterol": [289, 180, 304, 300, 200, 100, 250, 220, 280, 180],
            "FastingBS": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
            "MaxHR": [172, 156, 98, 100, 125, 123, 140, 150, 130, 160],
            "ExerciseAngina": [0, 1, 0, 0, 0, 1, 1, 0, 1, 0],
            "Oldpeak": [0.0, 1.0, 2.0, 3.0, 4.0, 2.6, 1.5, 0.5, 2.5, 3.0],
            "HeartDisease": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "ChestPainType_ATA": [1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
            "ChestPainType_NAP": [0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
            "ChestPainType_TA": [0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
            "RestingECG_Normal": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
            "RestingECG_ST": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
            "ST_Slope_Flat": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "ST_Slope_Up": [0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
        }
    )
    return df


def test_train(df):
    with tempfile.TemporaryDirectory() as temp_dir:
        orig_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            os.makedirs(os.path.join("data", "processed"), exist_ok=True)
            os.makedirs(os.path.join("models"), exist_ok=True)
            os.makedirs(os.path.join("reports"), exist_ok=True)

            input_path = os.path.join("data", "processed", "heart_processed.csv")
            df.to_csv(input_path, index=False)

            df = pd.read_csv(input_path)
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

            scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
            cv_lr = cross_validate(lr, X_train_lr, y_train, cv=3, scoring=scoring, n_jobs=1)
            lr_metrics = {metric: float(cv_lr[f"test_{metric}"].mean()) for metric in scoring}

            with open("reports/logistic_regression_metrics.json", "w") as f:
                json.dump(lr_metrics, f, indent=4)

            joblib.dump(lr, "models/logistic_regression.pkl")
            joblib.dump(scaler, "models/scaler.pkl")

            rf = RandomForestClassifier(random_state=42, n_estimators=100)
            rf.fit(X_train, y_train)

            cv_rf = cross_validate(rf, X_train, y_train, cv=3, scoring=scoring, n_jobs=1)
            rf_metrics = {metric: float(cv_rf[f"test_{metric}"].mean()) for metric in scoring}

            with open("reports/random_forest_metrics.json", "w") as f:
                json.dump(rf_metrics, f, indent=4)

            joblib.dump(rf, "models/random_forest.pkl")

            xgb = XGBClassifier(random_state=42, eval_metric="logloss")
            xgb.fit(X_train, y_train)

            cv_xgb = cross_validate(xgb, X_train, y_train, cv=3, scoring=scoring, n_jobs=1)
            xgb_metrics = {metric: float(cv_xgb[f"test_{metric}"].mean()) for metric in scoring}

            with open("reports/xgboost_metrics.json", "w") as f:
                json.dump(xgb_metrics, f, indent=4)

            joblib.dump(xgb, "models/xgboost.pkl")

            assert os.path.exists("models/logistic_regression.pkl")
            assert os.path.exists("models/random_forest.pkl")
            assert os.path.exists("models/xgboost.pkl")
            assert os.path.exists("models/scaler.pkl")
            assert os.path.exists("data/processed/test_set.csv")
            assert os.path.exists("reports/logistic_regression_metrics.json")
            assert os.path.exists("reports/random_forest_metrics.json")
            assert os.path.exists("reports/xgboost_metrics.json")

        finally:
            os.chdir(orig_cwd)
