import pandas as pd
import joblib

model = joblib.load("models/logistic_regression.pkl")
scaler = joblib.load("models/scaler.pkl")
df = pd.read_csv("data/processed/heart_processed.csv")
feature_names = df.drop("HeartDisease", axis=1).columns

coefs = pd.Series(model.coef_[0], index=feature_names)
coefs_sorted = coefs.abs().sort_values(ascending=False)

print("Топ-5 самых влиятельных признаков (по модулю коэффициента):")
print(coefs_sorted.head())