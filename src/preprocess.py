import pandas as pd
import os

def load_and_preprocess(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    print(f"Загружено {df.shape[0]} строк, {df.shape[1]} признаков.")

    df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})

    cat_cols = ['ChestPainType', 'RestingECG', 'ST_Slope']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    df['HeartDisease'] = df['HeartDisease'].astype(int)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Обработанный датасет сохранён: {output_path}")
    print(f"Финальное число признаков: {df.shape[1] - 1}")
    return df

if __name__ == "__main__":
    load_and_preprocess(
        input_path="data/raw/heart.csv",
        output_path="data/processed/heart_processed.csv"
    )