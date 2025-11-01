
import argparse, os
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def run_autolog(test_size, n_estimators, max_depth, data_dir):
    mlflow.set_tracking_uri("file:./mlruns")  # aman di CI
    mlflow.set_experiment("Latihan Credit Scoring (Kriteria 3)")

    data = pd.read_csv(os.path.join(data_dir, "heart_preprocessed.csv"))
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("Status", axis=1), data["Status"],
        test_size=test_size, random_state=42
    )

    with mlflow.start_run():
        mlflow.autolog()
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        print("Model Basic (Autolog Only) Selesai.")
        print(f"Akurasi: {accuracy:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--n_estimators", type=int, default=100)
    p.add_argument("--max_depth", type=int, default=5)
    p.add_argument("--data_dir", type=str, default="heart_preprocessing")
    a = p.parse_args()
    run_autolog(a.test_size, a.n_estimators, a.max_depth, a.data_dir)
