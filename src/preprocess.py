import os

import pandas as pd


def load_and_preprocess(input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    df["Sex"] = df["Sex"].map({"M": 1, "F": 0})
    df["ExerciseAngina"] = df["ExerciseAngina"].map({"Y": 1, "N": 0})

    cat_cols = ["ChestPainType", "RestingECG", "ST_Slope"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    df["HeartDisease"] = df["HeartDisease"].astype(int)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    if (df["Cholesterol"] == 0).any():
        median_chol = df[df["Cholesterol"] > 0]["Cholesterol"].median()
        df["Cholesterol"] = df["Cholesterol"].replace(0, median_chol)

    if (df["RestingBP"] == 0).any():
        median_bp = df[df["RestingBP"] > 0]["RestingBP"].median()
        df["RestingBP"] = df["RestingBP"].replace(0, median_bp)

    return df


if __name__ == "__main__":
    load_and_preprocess(
        input_path="data/raw/heart.csv",
        output_path="data/processed/heart_processed.csv",
    )
