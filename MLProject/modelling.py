import os
import argparse
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def run_autolog(test_size, n_estimators, max_depth, data_dir):
    uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(uri)

    data_path = os.path.join(data_dir, "heart_preprocessed.csv")
    df = pd.read_csv(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("Status", axis=1),
        df["Status"],
        test_size=test_size,
        random_state=42,
        stratify=df["Status"]
    )

    mlflow.autolog()
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    print("Model Basic (Autolog Only) Selesai.")
    print(f"Akurasi: {acc:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--n_estimators", type=int, default=100)
    p.add_argument("--max_depth", type=int, default=5)
    p.add_argument("--data_dir", type=str, default="heart_preprocessing")
    a = p.parse_args()
    run_autolog(a.test_size, a.n_estimators, a.max_depth, a.data_dir)
