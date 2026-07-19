import os, json, platform
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
from scipy.stats import mannwhitneyu

os.chdir(Path(__file__).resolve().parent.parent)
HERE = Path(__file__).resolve().parent
SEED = 42
CSV, READ = "external_datasets/Parkinsons-Sakar/pd_speech_features.csv", {'header': 1}
DROP, LABEL, GROUP = ["id", "gender", "class"], "class", "id"
POS, NEG, MODEL, AUDIO = "parkinsons", "healthy", "rf", False
MODEL_PKL = HERE / "models" / "parkinsons_sakar_rf.pkl"


def build_model():
    if MODEL == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(600, class_weight="balanced", random_state=SEED, n_jobs=-1))
    if MODEL == "svm":
        from sklearn.svm import SVC
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             SVC(kernel="rbf", C=10, probability=True, class_weight="balanced", random_state=SEED))
    from sklearn.linear_model import LogisticRegression
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5))


def main():
    print("==", "Parkinsons (Sakar 2018)", "==")
    df = pd.read_csv(CSV, **READ).drop_duplicates().reset_index(drop=True)
    cols = [c for c in df.columns if c not in DROP and pd.api.types.is_numeric_dtype(df[c])]
    df[cols] = df[cols].apply(lambda s: s.fillna(s.median()))
    cols = [c for c in cols if df[c].nunique() > 1]
    X = df[cols].to_numpy(float)
    y = df[LABEL].astype(int).to_numpy()

    if GROUP == "__name__":
        groups = df["name"].astype(str).str.extract(r"(S\d+)")[0].to_numpy()
    elif GROUP:
        groups = df[GROUP].astype(str).to_numpy()
    else:
        groups = None
    print(f"loaded {len(df)} rows, {len(cols)} features, class balance {dict(pd.Series(y).value_counts().sort_index())}")

    mi = pd.Series(mutual_info_classif(np.nan_to_num(X), y, random_state=SEED), index=cols).sort_values(ascending=False)
    print("top features by mutual information:", list(mi.head(5).index))

    n_sig = sum(mannwhitneyu(X[y == 0, j], X[y == 1, j]).pvalue < 0.05 for j in range(X.shape[1]))
    print(f"{n_sig}/{len(cols)} features differ significantly between classes (p<0.05)")

    cv = GroupKFold(5) if groups is not None else StratifiedKFold(5, shuffle=True, random_state=SEED)
    model = build_model()
    p = cross_val_predict(model, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    pred = (p >= 0.5).astype(int)
    print(f"{MODEL}: per-record AUC={roc_auc_score(y, p):.3f} bal_acc={balanced_accuracy_score(y, pred):.3f} "
          f"acc={accuracy_score(y, pred):.3f}")

    head_y, head_p = y, p
    if groups is not None:
        agg = pd.DataFrame({"g": groups, "y": y, "p": p}).groupby("g").agg(y=("y", "first"), p=("p", "mean"))
        head_y, head_p = agg["y"].to_numpy(), agg["p"].to_numpy()
        head_pred = (head_p >= 0.5).astype(int)
        print(f"per-speaker AUC={roc_auc_score(head_y, head_p):.3f} "
              f"bal_acc={balanced_accuracy_score(head_y, head_pred):.3f} acc={accuracy_score(head_y, head_pred):.3f}")

    model.fit(X, y)
    est = model.steps[-1][1]
    imp = getattr(est, "feature_importances_", None)
    if imp is None and hasattr(est, "coef_"):
        imp = np.abs(est.coef_[0])
    if imp is not None:
        print("most influential features:", [cols[i] for i in np.argsort(imp)[::-1][:5]])

    bundle = {"model": model, "features": cols, "classes": {0: NEG, 1: POS}}
    if AUDIO:
        bundle["sample_rate"] = 16000
    MODEL_PKL.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PKL)

    rng = np.random.default_rng(SEED)
    aucs = []
    for _ in range(2000):
        idx = rng.integers(0, len(head_y), len(head_y))
        if len(np.unique(head_y[idx])) == 2:
            aucs.append(roc_auc_score(head_y[idx], head_p[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    json.dump({"auc_mean": float(np.mean(aucs)), "auc_ci95": [float(lo), float(hi)], "n": int(len(y)),
               "seed": SEED, "python": platform.python_version(), "data": CSV},
              open(HERE / "metrics.json", "w"), indent=2)
    print(f"headline AUC {np.mean(aucs):.3f} 95% CI [{lo:.3f}, {hi:.3f}] -> saved {MODEL_PKL.name}")


if __name__ == "__main__":
    main()
