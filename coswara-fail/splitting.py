\
\
\
\
\
\
\
\


import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
DATA_DIR = ROOT / "data"
FEATURES_PATH = DATA_DIR / "features_extracted.csv"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
EMBEDDINGS_HUBERT_PATH = DATA_DIR / "embeddings_hubert.npy"
SPLITS_PATH = DATA_DIR / "splits.csv"


def participant_level_split(
    clips_df: pd.DataFrame, train_size: float, val_size: float, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    participants_df = clips_df.groupby("participant_id", as_index=False)["label"].first()
    train_participants, remaining_participants = train_test_split(
        participants_df,
        test_size=(1 - train_size),
        random_state=seed,
        stratify=participants_df["label"],
    )
    val_ratio = val_size / (val_size + test_size)
    val_participants, test_participants = train_test_split(
        remaining_participants,
        test_size=(1 - val_ratio),
        random_state=seed,
        stratify=remaining_participants["label"],
    )
    train_clips = clips_df[clips_df["participant_id"].isin(train_participants["participant_id"])]
    val_clips = clips_df[clips_df["participant_id"].isin(val_participants["participant_id"])]
    test_clips = clips_df[clips_df["participant_id"].isin(test_participants["participant_id"])]
    return train_clips, val_clips, test_clips


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train-size", type=float, default=config.DEFAULT_TRAIN_SIZE)
    parser.add_argument("--val-size", type=float, default=config.DEFAULT_VAL_SIZE)
    parser.add_argument("--test-size", type=float, default=config.DEFAULT_TEST_SIZE)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    features_df = pd.read_csv(FEATURES_PATH)
    audio_type_dummies = pd.get_dummies(
        features_df["audio_type"], prefix="audio_type", dtype=float
    )
    features_df = pd.concat([features_df, audio_type_dummies], axis=1)

    train_df, val_df, test_df = participant_level_split(
        features_df, args.train_size, args.val_size, args.test_size, args.seed
    )

    split_assignment_rows = []
    for split_name, split_df in zip(
        config.SPLIT_SEQUENCE, [train_df, val_df, test_df], strict=True
    ):
        participant_labels = (
            split_df[["participant_id", "label", "is_sick"]].drop_duplicates().copy()
        )
        participant_labels["split"] = split_name
        split_assignment_rows.append(participant_labels)
    pd.concat(split_assignment_rows, ignore_index=True).to_csv(SPLITS_PATH, index=False)

    feature_columns = [c for c in features_df.columns if c not in config.NON_FEATURE_COLUMNS]
    train_features, val_features, test_features = (
        train_df[feature_columns],
        val_df[feature_columns],
        test_df[feature_columns],
    )

    scaler = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    train_features_scaled = scaler.fit_transform(train_features)
    val_features_scaled = scaler.transform(val_features)
    test_features_scaled = scaler.transform(test_features)
    joblib.dump(scaler, ARTIFACTS_DIR / "feature_scaler.joblib")

    bundle = {
        "X_train": train_features_scaled,
        "X_val": val_features_scaled,
        "X_test": test_features_scaled,
        "feature_cols": feature_columns,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }


    for embeddings_path, bundle_key in [
        (EMBEDDINGS_PATH, "emb"),
        (EMBEDDINGS_HUBERT_PATH, "emb_hubert"),
    ]:
        if embeddings_path.exists():
            embeddings = np.load(embeddings_path)
            bundle[f"{bundle_key}_train"] = embeddings[train_df.index.to_numpy()]
            bundle[f"{bundle_key}_val"] = embeddings[val_df.index.to_numpy()]
            bundle[f"{bundle_key}_test"] = embeddings[test_df.index.to_numpy()]
            print(f"{embeddings_path.name} attached to bundle as '{bundle_key}': {embeddings.shape}")
        else:
            print(f"no {embeddings_path.name} found - '{bundle_key}' model will be skipped")

    report_lines = [
        "# 06 Data Splitting Report",
        "",
        f"- Train rows: {len(train_df)}",
        f"- Validation rows: {len(val_df)}",
        f"- Test rows: {len(test_df)}",
        "- SMOTE applied per target: yes",
    ]
    for target in config.TARGETS:
        label_column = config.TARGET_LABEL_COLUMNS[target]
        train_labels = train_df[label_column]
        val_labels = val_df[label_column]
        test_labels = test_df[label_column]

        smote = SMOTE(random_state=args.seed)
        train_features_balanced, train_labels_balanced = smote.fit_resample(
            train_features_scaled, train_labels
        )

        bundle[f"y_train_{target}"] = train_labels.values
        bundle[f"y_val_{target}"] = val_labels.values
        bundle[f"y_test_{target}"] = test_labels.values
        bundle[f"X_train_bal_{target}"] = train_features_balanced
        bundle[f"y_train_bal_{target}"] = train_labels_balanced

        report_lines += [
            "",
            f"## Target: {target} ({label_column})",
            f"- Original train class counts: {train_labels.value_counts().to_dict()}",
            f"- Resampled train class counts: {pd.Series(train_labels_balanced).value_counts().to_dict()}",
        ]

    joblib.dump(bundle, ARTIFACTS_DIR / "split_data.joblib")
    (REPORTS_DIR / "06_data_splitting_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print(
        f"data split complete. train={len(train_df)} val={len(val_df)} test={len(test_df)}"
    )


if __name__ == "__main__":
    main()
