import sys
from pathlib import Path
import numpy as np, joblib, librosa, opensmile

HERE = Path(__file__).resolve().parent
BUNDLE = joblib.load(HERE / "models" / "svd_voice_disorder_svm.pkl")
_SMILE = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                         feature_level=opensmile.FeatureLevel.Functionals)


def predict(path):
    sig, _ = librosa.load(path, sr=BUNDLE["sample_rate"])
    sig = sig / (np.max(np.abs(sig)) + 1e-9)
    sig, _ = librosa.effects.trim(sig, top_db=30)
    feats = _SMILE.process_signal(sig, BUNDLE["sample_rate"])
    X = feats[BUNDLE["features"]].to_numpy(float)
    p = float(BUNDLE["model"].predict_proba(X)[0, 1])
    return p, BUNDLE["classes"][int(p >= 0.5)]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python predict.py <audio_file>")
    p, label = predict(sys.argv[1])
    print(f"P(pathological) = {p:.3f}  ->  {label.upper()}")
