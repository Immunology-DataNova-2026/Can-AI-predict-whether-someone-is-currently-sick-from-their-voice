import sys
from pathlib import Path
import pandas as pd, joblib

HERE = Path(__file__).resolve().parent
BUNDLE = joblib.load(HERE / "models" / "parkinsons_sakar_rf.pkl")


def predict(csv_path):
    df = pd.read_csv(csv_path)
    missing = [c for c in BUNDLE["features"] if c not in df.columns]
    if missing:
        sys.exit(f"CSV is missing {len(missing)} required feature columns, e.g. {missing[:3]}")
    X = df[BUNDLE["features"]].to_numpy(float)
    proba = BUNDLE["model"].predict_proba(X)[:, 1]
    return proba


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python predict.py <features.csv>")
    for i, p in enumerate(predict(sys.argv[1])):
        label = BUNDLE["classes"][int(p >= 0.5)]
        print(f"row {i}: P(parkinsons) = {p:.3f}  ->  {label.upper()}")
