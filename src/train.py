import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler


def main():
    df = pd.read_csv("data/processed/heart_processed.csv")
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pd.concat([X_test, y_test], axis=1).to_csv(
        "data/processed/test_set.csv", index=False
    )

    os.makedirs("models", exist_ok=True)

    scaler = StandardScaler()
    X_train_lr = scaler.fit_transform(X_train)
    lr = LogisticRegression(random_state=42, max_iter=2000)
    lr.fit(X_train_lr, y_train)
    joblib.dump(lr, "models/logistic_regression.pkl")
    joblib.dump(scaler, "models/scaler.pkl")  # сохраняем scaler
    print("Обучение logistic_regression...")

    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(X_train, y_train)
    joblib.dump(rf, "models/random_forest.pkl")
    print("Обучение random_forest...")

    xgb = XGBClassifier(random_state=42, eval_metric="logloss")
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, "models/xgboost.pkl")
    print("Обучение xgboost...")

    print("Все модели сохранены в папку 'models'")


if __name__ == "__main__":
    main()
