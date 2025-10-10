import joblib

joblib.dump(joblib.load("models/random_forest.pkl"), "models/best_model.pkl")
