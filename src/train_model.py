# Part 1: The tools and settings

import argparse
import os

import joblib
import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
MODELS_DIR = "models"

RANDOM_STATE = 42

PARAMS = {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 3}

# Part 2: Three small helper functions

def load_data():
    data = load_breast_cancer()
    return data.data, data.target

def split_data(X,y):
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

def build_model(n_estimators, learning_rate, max_depth):
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
    )

# Part 3: The main recipe

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="Version name, e.g. 20261031120000_ab12cd3")
    version = parser.parse_args().version
    print(f"Training model version: {version}")

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    os.makedirs(DATA_DIR, exist_ok=True)
    joblib.dump((X_test, y_test), os.path.join(DATA_DIR, "test_set.joblib"))

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("breast-cancer-gradient-boosting")

    with mlflow.start_run(run_name=f"model_{version}"):
        mlflow.log_params({**PARAMS, "train_rows": len(X_train), "test_rows": len(X_test)})

        model = build_model(**PARAMS)
        model.fit(X_train, y_train)

        train_accuracy = model.score(X_train, y_train)
        mlflow.log_metric("train_accuracy", train_accuracy)


    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"model_{version}.joblib")
    joblib.dump(model, model_path)

    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Model saved to: {model_path}")

if __name__=="__main__":
    main()