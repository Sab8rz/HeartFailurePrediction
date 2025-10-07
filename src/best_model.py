import joblib

joblib.dump(
    joblib.load("models/logistic_regression.pkl"), "models/best_model.pkl"
)
