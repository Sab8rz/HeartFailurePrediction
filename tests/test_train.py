import os
import tempfile

import joblib
import pandas as pd
from pytest import fixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


@fixture
def df():
    df = pd.DataFrame(
        {
            "Age": [40, 50, 60, 30, 34, 21],
            "Sex": [1, 0, 1, 1, 0, 0],
            "RestingBP": [140, 150, 130, 128, 234, 211],
            "Cholesterol": [289, 180, 304, 300, 200, 100],
            "FastingBS": [0, 1, 0, 1, 1, 0],
            "MaxHR": [172, 156, 98, 100, 125, 123],
            "ExerciseAngina": [0, 1, 0, 0, 0, 1],
            "Oldpeak": [0.0, 1.0, 2.0, 3.0, 4.0, 2.6],
            "HeartDisease": [0, 1, 0, 1, 0, 1],
            "ChestPainType_ATA": [1, 0, 1, 1, 1, 0],
            "ChestPainType_NAP": [0, 1, 0, 0, 0, 1],
            "ChestPainType_TA": [0, 0, 1, 1, 0, 0],
            "RestingECG_Normal": [1, 0, 1, 0, 1, 1],
            "RestingECG_ST": [1, 0, 1, 0, 1, 1],
            "ST_Slope_Flat": [1, 0, 1, 0, 1, 0],
            "ST_Slope_Up": [0, 1, 0, 0, 1, 0],
        }
    )
    return df


def test_train(df):
    with tempfile.TemporaryDirectory() as temp_dir:
        os.makedirs(os.path.join(temp_dir, "data", "processed"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "models"), exist_ok=True)

        input_path = os.path.join(temp_dir, "data", "processed", "heart_processed.csv")
        df.to_csv(input_path, index=False)

        df = pd.read_csv(input_path)
        X = df.drop("HeartDisease", axis=1)
        y = df["HeartDisease"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        test_set_path = os.path.join(temp_dir, "data", "processed", "test_set.csv")
        pd.concat([X_test, y_test], axis=1).to_csv(test_set_path, index=False)

        scaler = StandardScaler()
        X_train_lr = scaler.fit_transform(X_train)
        lr = LogisticRegression(random_state=42, max_iter=2000)
        lr.fit(X_train_lr, y_train)
        joblib.dump(lr, os.path.join(temp_dir, "models", "logistic_regression.pkl"))
        joblib.dump(scaler, os.path.join(temp_dir, "models", "scaler.pkl"))

        rf = RandomForestClassifier(random_state=42, n_estimators=100)
        rf.fit(X_train, y_train)
        joblib.dump(rf, os.path.join(temp_dir, "models", "random_forest.pkl"))

        xgb = XGBClassifier(random_state=42, eval_metric="logloss")
        xgb.fit(X_train, y_train)
        joblib.dump(xgb, os.path.join(temp_dir, "models", "xgboost.pkl"))

        assert os.path.exists(os.path.join(temp_dir, "models", "logistic_regression.pkl"))
        assert os.path.exists(os.path.join(temp_dir, "models", "random_forest.pkl"))
        assert os.path.exists(os.path.join(temp_dir, "models", "xgboost.pkl"))
        assert os.path.exists(os.path.join(temp_dir, "models", "scaler.pkl"))
        assert os.path.exists(os.path.join(temp_dir, "data", "processed", "test_set.csv"))
