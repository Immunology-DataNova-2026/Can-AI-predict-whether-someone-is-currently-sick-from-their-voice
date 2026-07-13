import sys
from pathlib import Path
import pandas as pd, joblib

HERE = Path(__file__).resolve().parent
BUNDLE = joblib.load(HERE / "models" / "parkinsons_uci_rf.pkl")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python predict.py <features.csv>")
    df = pd.read_csv(sys.argv[1])
    missing = [c for c in BUNDLE["features"] if c not in df.columns]
    if missing:
        sys.exit(f"CSV missing {len(missing)} required columns, e.g. {missing[:3]}")
    proba = BUNDLE["model"].predict_proba(df[BUNDLE["features"]].to_numpy(float))[:, 1]
    for i, p in enumerate(proba):
        print(f"row {i}: P(parkinsons) = {p:.3f}  ->  {BUNDLE['classes'][int(p >= 0.5)].upper()}")
