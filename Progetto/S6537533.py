import argparse
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


def run_model(train_file, test_file):
    # Caricamento dati
    train_df = pd.read_csv(train_file)
    test_df  = pd.read_csv(test_file)

    target_col = "median_house_value"
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test  = test_df.drop(columns=[target_col], errors="ignore")

    # Pipeline: imputer → scaler → Random Forest
    # Parametri ottimali trovati con GridSearchCV in cross-validation:
    # max_depth=20, min_samples_leaf=1, n_estimators=150  (CV accuracy = 0.688)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            min_samples_leaf=1,
            random_state=42
        ))
    ])

    # Training sull'intero training set
    pipeline.fit(X_train, y_train)

    # Predizione sul test set
    predictions = pipeline.predict(X_test).astype(int)

    # Salvataggio risultati (una predizione per riga)
    with open("S6537533.txt", "w") as fh:
        for pred in predictions:
            fh.write(f"{pred}\n")

    print(f"Predizioni salvate: {len(predictions)} righe → S6537533.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test",  type=str, required=True)
    args = parser.parse_args()
    run_model(args.train, args.test)