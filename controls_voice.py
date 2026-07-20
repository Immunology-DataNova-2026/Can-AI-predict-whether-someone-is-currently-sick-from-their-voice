from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
OUT = Path("reports")
OUT.mkdir(exist_ok=True)

def _family(col: str) -> str:
    lc = col.lower()
    if "jitter" in lc: return "jitter"
    if "shimmer" in lc: return "shimmer"
    if "harmonicity" in lc or "hnr" in lc or "nhr" in lc: return "HNR"
    if "mfcc" in lc: return "MFCC"
    if "tqwt" in lc: return "TQWT"
    return "other"


def _bootstrap_ci(hy, hp, n_iter=2000, seed=SEED):
    from sklearn.metrics import accuracy_score, roc_auc_score
    rng, aucs, accs = np.random.default_rng(seed), [], []
    for _ in range(n_iter):
        idx = rng.integers(0, len(hy), len(hy))
        if len(np.unique(hy[idx])) == 2:
            aucs.append(roc_auc_score(hy[idx], hp[idx]))
            accs.append(accuracy_score(hy[idx], (hp[idx] >= .5).astype(int)))
    alo, ahi = np.percentile(aucs, [2.5, 97.5])
    clo, chi = np.percentile(accs, [2.5, 97.5])
    return (roc_auc_score(hy, hp), alo, ahi,
            accuracy_score(hy, (hp >= .5).astype(int)), clo, chi)


def _speaker_probs(X, y, groups, model):
    from sklearn.model_selection import GroupKFold, cross_val_predict
    p = cross_val_predict(model, X, y, cv=GroupKFold(5), groups=groups,
                          method="predict_proba")[:, 1]
    agg = (pd.DataFrame({"g": groups, "y": y, "p": p})
             .groupby("g").agg(y=("y", "first"), p=("p", "mean")))
    return agg["y"].to_numpy(), agg["p"].to_numpy()


def run_voice():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    CSV = "external_datasets/Parkinsons-Sakar/pd_speech_features.csv"
    df = pd.read_csv(CSV, header=1).drop_duplicates().reset_index(drop=True)
    cols = [c for c in df.columns if c not in ("id", "gender", "class")
            and pd.api.types.is_numeric_dtype(df[c])]
    df[cols] = df[cols].apply(lambda s: s.fillna(s.median()))
    cols = [c for c in cols if df[c].nunique() > 1]
    X, y = df[cols].to_numpy(float), df["class"].astype(int).to_numpy()
    groups = df["id"].astype(str).to_numpy()

    model = make_pipeline(SimpleImputer(strategy="median"),
                          RandomForestClassifier(600, class_weight="balanced",
                                                 random_state=SEED, n_jobs=-1))
    hy, hp = _speaker_probs(X, y, groups, model)
    auc, alo, ahi, acc, clo, chi = _bootstrap_ci(hy, hp)

    lines = ["# Voice controls — Parkinson's (Sakar)", "",
             f"- speakers: {len(hy)}  ({(hy==0).sum()} healthy, {(hy==1).sum()} PD)",
             f"- per-speaker AUC: {auc:.3f}  95% CI [{alo:.3f}, {ahi:.3f}]",
             f"- per-speaker accuracy: {acc:.3f}  95% CI [{clo:.3f}, {chi:.3f}]", ""]

    mi = pd.Series(mutual_info_classif(np.nan_to_num(X), y, random_state=SEED),
                   index=cols).sort_values(ascending=False)
    model.fit(X, y)
    imp = pd.Series(model.steps[-1][1].feature_importances_, index=cols).sort_values(ascending=False)
    top25_names = mi.head(25).index
    feat = pd.DataFrame({"mutual_information": mi.loc[top25_names],
                        "rf_importance": imp.loc[top25_names]})
    feat["family"] = [_family(c) for c in feat.index]
    feat.to_csv(OUT / "feature_importance_sakar.csv")

    fam_counts = feat["family"].value_counts()
    fam_order = ["jitter", "shimmer", "HNR", "MFCC", "TQWT", "other"]
    lines += ["## Top 15 features by mutual information", ""]
    lines += [f"{i:>2}. {n:<28} MI={v:.4f}  RF={imp[n]:.4f}"
              for i, (n, v) in enumerate(mi.head(15).items(), 1)]
    lines += ["", "Saved: reports/feature_importance_sakar.csv", "",
              "## Top-25 features grouped by acoustic family", "",
              "| family | count in top 25 | share |", "|---|---|---|"]
    for f in fam_order:
        c = int(fam_counts.get(f, 0))
        lines.append(f"| {f} | {c} | {c/25:.0%} |")

    svd_csv = Path("external_datasets/SVD/svd_boost_features.csv")
    if svd_csv.exists():
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        sdf = pd.read_csv(svd_csv).drop_duplicates().reset_index(drop=True)
        scols = [c for c in sdf.columns if c not in ("speaker", "label")
                 and pd.api.types.is_numeric_dtype(sdf[c])]
        sdf[scols] = sdf[scols].apply(lambda s: s.fillna(s.median()))
        scols = [c for c in scols if sdf[c].nunique() > 1]
        sX = sdf[scols].to_numpy(float)
        sy = sdf["label"].astype(int).to_numpy()
        sgroups = sdf["speaker"].astype(str).to_numpy()
        smodel = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               SVC(kernel="rbf", C=10, probability=True,
                                   class_weight="balanced", random_state=SEED))
        shy, shp = _speaker_probs(sX, sy, sgroups, smodel)
        sauc, salo, sahi, sacc, sclo, schi = _bootstrap_ci(shy, shp)
        lines += ["", "# Voice controls — voice disorder (SVD)", "",
                  f"- speakers: {len(shy)}  ({(shy==0).sum()} healthy, {(shy==1).sum()} pathological)",
                  f"- per-speaker AUC: {sauc:.3f}  95% CI [{salo:.3f}, {sahi:.3f}]",
                  f"- per-speaker accuracy: {sacc:.3f}  95% CI [{sclo:.3f}, {schi:.3f}]"]
        svd_result = dict(auc=sauc, auc_ci=[salo, sahi], acc=sacc, acc_ci=[sclo, schi], n=len(shy))
    else:
        lines += ["", "# Voice controls — voice disorder (SVD)", "",
                  f"- SKIPPED: {svd_csv} not present."]
        svd_result = None

    (OUT / "controls_voice.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return dict(sakar=dict(auc=auc, auc_ci=[alo, ahi], acc=acc, acc_ci=[clo, chi], n=len(hy)),
                svd=svd_result, family_counts=fam_counts.to_dict())

if __name__ == "__main__":
    run_voice()
