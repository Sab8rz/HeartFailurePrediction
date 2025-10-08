import os
import tempfile

import pandas as pd
from src.preprocess import load_and_preprocess


def test_preprocess():
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "heart.csv")
        output_path = os.path.join(temp_dir, "heart_processed.csv")

        df = pd.DataFrame(
            {
                "Age": [40, 50],
                "Sex": ["M", "F"],
                "ChestPainType": ["ATA", "NAP"],
                "RestingBP": [140, 160],
                "Cholesterol": [289, 180],
                "FastingBS": [0, 0],
                "RestingECG": ["Normal", "ST"],
                "MaxHR": [172, 156],
                "ExerciseAngina": ["N", "Y"],
                "Oldpeak": [0.0, 1.0],
                "ST_Slope": ["Up", "Flat"],
                "HeartDisease": [0, 1],
            }
        )
        df.to_csv(input_path, index=False)

        result_df = load_and_preprocess(input_path, output_path)

        assert "ChestPainType_NAP" in result_df.columns
        assert "RestingECG_ST" in result_df.columns
        assert "ST_Slope_Up" in result_df.columns

        assert set(result_df["Sex"].unique()) == {0, 1}
        assert set(result_df["ExerciseAngina"].unique()) == {0, 1}

        assert result_df["HeartDisease"].dtype == int

        assert os.path.exists(output_path)
