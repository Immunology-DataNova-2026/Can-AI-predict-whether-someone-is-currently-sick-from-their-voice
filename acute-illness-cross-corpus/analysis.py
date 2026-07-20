\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\


import argparse
import json
from pathlib import Path

import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, kruskal, shapiro
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FEATURES_PATH = DATA_DIR / "features_extracted.csv"
SPLIT_DATA_PATH = ARTIFACTS_DIR / "split_data.joblib"
RESULTS_PATH = ROOT / "results.json"
COSWARA_META = ROOT / config.DEFAULT_RAW_DATA_DIRNAME / "combined_data.csv"

sns.set_theme(style="whitegrid")


def _agg_by_participant(participant_ids, y_true, proba):
    unique_ids = pd.unique(participant_ids)
    aggregated_labels = pd.Series(y_true).groupby(participant_ids, sort=False).first().loc[unique_ids].to_numpy()
    aggregated_proba = pd.DataFrame(proba).groupby(participant_ids, sort=False).mean().loc[unique_ids].to_numpy()
    return aggregated_labels, aggregated_proba


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def _eda_class_distribution(class_counts: pd.Series) -> None:
    plt.figure(figsize=(8, 4))
    sns.barplot(x=class_counts.index.tolist(), y=class_counts.values.tolist())
    plt.title("Class Distribution")
    plt.xticks(rotation=30, ha="right")
    _savefig(FIGURES_DIR / "01_class_distribution.png")
    plt.figure(figsize=(7, 7))
    plt.pie(class_counts.values.tolist(), labels=class_counts.index.tolist(), autopct="%1.1f%%", startangle=90)
    plt.title("Class Balance")
    _savefig(FIGURES_DIR / "01_class_distribution_pie.png")


def _eda_waveforms_spectrograms(clips_df, class_counts, sample_rate, n_mels) -> None:
    top_classes = class_counts.index.tolist()[:2]
    if len(top_classes) < 2:
        return
    first_class_sample = clips_df[clips_df["label_name"] == top_classes[0]].head(1)
    second_class_sample = clips_df[clips_df["label_name"] == top_classes[1]].head(1)
    if first_class_sample.empty or second_class_sample.empty:
        return
    sample_paths = [
        Path(first_class_sample.iloc[0]["processed_audio_path"]),
        Path(second_class_sample.iloc[0]["processed_audio_path"]),
    ]
    class_labels = [top_classes[0], top_classes[1]]
    _, axes = plt.subplots(2, 1, figsize=(12, 6))
    for index, audio_path in enumerate(sample_paths):
        waveform, audio_sample_rate = librosa.load(audio_path, sr=sample_rate)
        librosa.display.waveshow(waveform, sr=audio_sample_rate, ax=axes[index])
        axes[index].set_title(f"{class_labels[index]} Waveform")
    _savefig(FIGURES_DIR / "02_sample_waveforms.png")
    figure, axes = plt.subplots(2, 1, figsize=(12, 7))
    for index, audio_path in enumerate(sample_paths):
        waveform, audio_sample_rate = librosa.load(audio_path, sr=sample_rate)
        mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=audio_sample_rate, n_mels=n_mels)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        spectrogram_image = librosa.display.specshow(mel_spectrogram_db, sr=audio_sample_rate, x_axis="time", y_axis="mel", ax=axes[index])
        axes[index].set_title(f"{class_labels[index]} Mel Spectrogram")
        figure.colorbar(spectrogram_image, ax=axes[index], format="%+2.0f dB")
    _savefig(FIGURES_DIR / "03_sample_spectrograms.png")


def _eda_quality_flags(clips_df, sample_rate) -> pd.DataFrame:
    quality_rows = []
    for _, row in clips_df.iterrows():
        audio_path = Path(row["processed_audio_path"])
        waveform, _ = librosa.load(audio_path, sr=sample_rate)
        max_abs_amplitude = np.max(np.abs(waveform))
        rms = np.sqrt(np.mean(waveform**2))
        flag = "clipped" if max_abs_amplitude >= config.CLIPPING_THRESHOLD else ("noisy" if rms < config.NOISE_RMS_THRESHOLD else "clean")
        quality_rows.append({"participant_id": row["participant_id"], "audio_path": str(audio_path), "quality_flag": flag})
    return pd.DataFrame(quality_rows)


def run_eda(args) -> None:
    index_df = pd.read_csv(ARTIFACTS_DIR / "preprocessed_index.csv")
    class_counts = index_df["label_name"].value_counts().sort_values(ascending=False)
    _eda_class_distribution(class_counts)
    _eda_waveforms_spectrograms(index_df, class_counts, args.sample_rate, args.n_mels)
    quality_df = _eda_quality_flags(index_df, args.sample_rate)
    quality_df.to_csv(ARTIFACTS_DIR / "quality_flags.csv", index=False)
    quality_flag_counts = quality_df["quality_flag"].value_counts() if not quality_df.empty else pd.Series(dtype=int)
    report = [
        "<html><body>", "<h1>02 EDA Report</h1>", f"<p>Usable recordings: {len(index_df)}</p>",
        "<h2>Class Counts</h2>", "<ul>", *[f"<li>{class_name}: {count}</li>" for class_name, count in class_counts.items()], "</ul>",
        "<h2>Quality Flag Counts</h2>", "<ul>", *[f"<li>{flag_name}: {flag_count}</li>" for flag_name, flag_count in quality_flag_counts.items()], "</ul>",
        "</body></html>",
    ]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "02_eda_report.html").write_text("\n".join(report), encoding="utf-8")
    print(f"eda complete. usable recordings={len(index_df)}")


def _eta_squared_from_f(f_stat: float, num_groups: int, num_samples: int) -> float:
    if np.isnan(f_stat) or num_samples <= num_groups:
        return np.nan
    return float((f_stat * (num_groups - 1)) / (f_stat * (num_groups - 1) + (num_samples - num_groups)))


def _stat_run_tests(features_df, feature_columns, class_labels) -> pd.DataFrame:
    results = []
    for column in feature_columns:
        groups = [features_df[features_df["label"] == class_label][column].dropna().values for class_label in class_labels]
        groups = [group for group in groups if len(group) >= config.MIN_GROUP_SAMPLE_SIZE]
        if len(groups) < 2:
            continue
        normal_flags = [shapiro(group[: min(config.SHAPIRO_MAX_N, len(group))]).pvalue > config.SIGNIFICANCE_ALPHA for group in groups]
        if all(normal_flags):
            statistic, p_value = f_oneway(*groups)
            test_name = "anova"
            effect_size = _eta_squared_from_f(float(statistic), num_groups=len(groups), num_samples=sum(len(group) for group in groups))
        else:
            statistic, p_value = kruskal(*groups)
            test_name, effect_size = "kruskal", np.nan
        row = {"feature": column, "test_used": test_name, "p_value": p_value, "statistic": statistic, "eta_squared": effect_size}
        for class_label in class_labels:
            class_values = features_df[features_df["label"] == class_label][column].dropna().values
            row[f"mean_class_{class_label}"] = float(np.mean(class_values)) if len(class_values) else np.nan
        results.append(row)
    return pd.DataFrame(results).sort_values("p_value")


def run_stattests(args) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    features_df = pd.read_csv(FEATURES_PATH)
    feature_columns = [column for column in features_df.columns if column not in config.NON_FEATURE_COLUMNS]
    class_labels = sorted(features_df["label"].unique().tolist())
    results_df = _stat_run_tests(features_df, feature_columns, class_labels)
    results_df.to_csv(DATA_DIR / "statistical_tests.csv", index=False)

    plt.figure(figsize=(12, 10))
    sns.heatmap(features_df[feature_columns].corr(method="spearman"), cmap="coolwarm", center=0, cbar=False)
    plt.title("Feature Correlation Heatmap (Spearman)")
    _savefig(FIGURES_DIR / "correlation_heatmap.png")

    top_features = results_df.head(config.TOP_DISCRIMINATIVE_FEATURES)["feature"].tolist()
    if top_features:
        melted_df = features_df[["label"] + top_features].melt(id_vars=["label"], var_name="feature", value_name="value")
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=melted_df, x="feature", y="value", hue="label", split=False)
        plt.xticks(rotation=45, ha="right")
        plt.title("Top Discriminative Features")
        _savefig(FIGURES_DIR / "05_statistical_tests_top_features.png")

    report = [
        "# 03 Feature Report", "",
        f"- Total features tested: {len(results_df)}",
        f"- Significant (p < {config.SIGNIFICANCE_ALPHA}): {int((results_df['p_value'] < config.SIGNIFICANCE_ALPHA).sum()) if not results_df.empty else 0}",
        "", "Top features:",
    ]
    for _, row in results_df.head(config.TOP_DISCRIMINATIVE_FEATURES).iterrows():
        report.append(f"- {row['feature']}: p={row['p_value']:.4g}, test={row['test_used']}")
    (REPORTS_DIR / "03_feature_report.md").write_text("\n".join(report), encoding="utf-8")
    print("statistical analysis complete")


def _sensitivity_sweep(rf, train_features, train_labels) -> pd.DataFrame:
    base_params = rf.get_params()
    records = []
    for param_name, param_values in config.SENSITIVITY_PARAM_GRID.items():
        for param_value in param_values:
            model = RandomForestClassifier(**{**base_params, param_name: param_value})
            scores = cross_val_score(model, train_features, train_labels, cv=config.SENSITIVITY_CV_SPLITS, scoring=config.RF_SCORING)
            records.append({"param": param_name, "value": str(param_value), "score_mean": float(np.mean(scores))})
    return pd.DataFrame(records)


def _validate_target(target: str, seed: int) -> dict | None:
    rf_path = MODELS_DIR / f"model_random_forest_{target}.pkl"
    if not rf_path.exists():
        print(f"skipping {target}: no trained random forest found")
        return None
    bundle = joblib.load(SPLIT_DATA_PATH)
    train_features, train_labels = bundle[f"X_train_bal_{target}"], bundle[f"y_train_bal_{target}"]
    calibrated_model = joblib.load(rf_path)
    rf = calibrated_model.calibrated_classifiers_[0].estimator if hasattr(calibrated_model, "calibrated_classifiers_") else calibrated_model
    cv_splitter = StratifiedKFold(n_splits=config.RF_CV_SPLITS, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(rf, train_features, train_labels, cv=cv_splitter, scoring=config.RF_SCORING)

    sensitivity_df = _sensitivity_sweep(rf, train_features, train_labels)
    sensitivity_df.to_csv(ARTIFACTS_DIR / f"rf_hyperparameter_sensitivity_{target}.csv", index=False)
    plt.figure(figsize=(10, 4))
    for param_name in sensitivity_df["param"].unique():
        param_subset = sensitivity_df[sensitivity_df["param"] == param_name]
        plt.plot(param_subset["value"], param_subset["score_mean"], marker="o", label=param_name)
    plt.title(f"Random Forest Hyperparameter Sensitivity - {target}")
    plt.xlabel("Value")
    plt.ylabel(f"Validation {config.RF_SCORING}")
    plt.legend()
    _savefig(FIGURES_DIR / f"rf_hyperparameter_sensitivity_{target}.png")
    return {"rf_cv_score_mean": float(np.mean(cv_scores)), "rf_cv_score_std": float(np.std(cv_scores))}


def run_validate(args) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    report = {target: _validate_target(target, args.seed) for target in config.TARGETS}
    report = {target: target_result for target, target_result in report.items() if target_result is not None}
    (ARTIFACTS_DIR / "validation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# 05 Reproducibility Report", "", f"- Random seed: {args.seed}", f"- Scoring: {config.RF_SCORING}"]
    for target, target_result in report.items():
        lines += ["", f"## {target}", f"- RF CV score mean: {target_result['rf_cv_score_mean']:.4f}", f"- RF CV score std: {target_result['rf_cv_score_std']:.4f}"]
    (REPORTS_DIR / "05_reproducibility.md").write_text("\n".join(lines), encoding="utf-8")
    print("validation complete")


def _err_metadata() -> pd.DataFrame:
    metadata_df = pd.read_csv(COSWARA_META).rename(columns={"id": "participant_id", "a": "age", "g": "gender"})
    metadata_df["record_date"] = pd.to_datetime(metadata_df["record_date"], errors="coerce")
    metadata_df["record_year_month"] = metadata_df["record_date"].dt.to_period("M").astype(str)
    return metadata_df[["participant_id", "age", "gender", "record_year_month"]]


def _age_band(age) -> str:
    if pd.isna(age):
        return "unknown"
    if age < 30:
        return "<30"
    if age < 45:
        return "30-44"
    if age < 60:
        return "45-59"
    return "60+"


def _accuracy_by(predictions_df, group_col) -> pd.DataFrame:
    return predictions_df.groupby(group_col).agg(n=("correct", "size"), accuracy=("correct", "mean")).sort_values("accuracy").reset_index()


def _analyze_errors(target, metadata_df, class_names) -> list[str]:
    pred_path = DATA_DIR / f"predictions_test_set_{target}.csv"
    if not pred_path.exists():
        return [f"## {target}", f"- no predictions file at {pred_path}", ""]
    predictions_df = pd.read_csv(pred_path).merge(metadata_df, on="participant_id", how="left")
    predictions_df["correct"] = (predictions_df["ensemble_pred"] == predictions_df["true_label"]).astype(int)
    predictions_df["age_band"] = predictions_df["age"].apply(_age_band)
    predictions_df["true_class"] = predictions_df["true_label"].map(class_names.get(target, {}))
    lines = [f"## {target} (overall ensemble accuracy: {predictions_df['correct'].mean():.3f})", "", "### Accuracy by true class"]
    accuracy_by_class = _accuracy_by(predictions_df.assign(cls=predictions_df["true_class"].fillna(predictions_df["true_label"])), "cls")
    lines += ["| class | n | accuracy |", "|---|---|---|"]
    lines += [f"| {row['cls']} | {int(row['n'])} | {row['accuracy']:.3f} |" for _, row in accuracy_by_class.iterrows()]
    lines.append("")
    for group_col, title in [("age_band", "age band"), ("gender", "gender"), ("record_year_month", "recording month (wave)")]:
        subset_df = _accuracy_by(predictions_df, group_col)
        lines += [f"### Accuracy by {title}", f"| {group_col} | n | accuracy |", "|---|---|---|"]
        lines += [f"| {row[group_col]} | {int(row['n'])} | {row['accuracy']:.3f} |" for _, row in subset_df.iterrows()]
        lines.append("")
    error_rows = predictions_df[predictions_df["correct"] == 0]
    lines += [
        "### Confidence",
        f"- Mean confidence on correct: {predictions_df[predictions_df['correct']==1]['ensemble_conf'].mean():.3f}",
        f"- Mean confidence on errors: {error_rows['ensemble_conf'].mean():.3f}",
        f"- High-confidence errors (conf>0.8): {(error_rows['ensemble_conf']>0.8).sum()} of {len(error_rows)} errors",
        "",
    ]
    error_columns = ["participant_id", "true_label", "true_class", "ensemble_pred", "ensemble_conf", "age", "gender", "record_year_month"]
    (error_rows[error_columns] if "true_class" in error_rows.columns else error_rows).to_csv(DATA_DIR / f"misclassified_{target}.csv", index=False)
    lines += [f"- Misclassified participants written to `misclassified_{target}.csv` ({len(error_rows)} rows)", ""]
    return lines


def run_errors(args) -> None:
    metadata_df = _err_metadata()
    class_names = {"binary": dict(enumerate(config.BINARY_CLASS_NAMES))}
    lines = ["# 13 Error Analysis", ""]
    for target in config.TARGETS:
        lines += _analyze_errors(target, metadata_df, class_names)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "13_error_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print("error analysis complete -> reports/13_error_analysis.md")


def _conf_metadata() -> pd.DataFrame:
    metadata_df = pd.read_csv(COSWARA_META).rename(columns={"id": "participant_id", "a": "age", "g": "gender"})
    metadata_df["record_date"] = pd.to_datetime(metadata_df["record_date"], errors="coerce")
    metadata_df["date_ordinal"] = (metadata_df["record_date"] - metadata_df["record_date"].min()).dt.days
    metadata_df["gender_code"] = metadata_df["gender"].map({"male": 0, "female": 1, "other": 2})
    return metadata_df[["participant_id", "age", "gender_code", "date_ordinal"]]


def _conf_participant_frame(split_df, metadata_df) -> pd.DataFrame:
    participant_df = split_df.groupby("participant_id").agg(is_sick=("is_sick", "first"), label=("label", "first")).reset_index()
    merged_df = participant_df.merge(metadata_df, on="participant_id", how="left")
    for column in ["age", "gender_code", "date_ordinal"]:
        merged_df[column] = merged_df[column].fillna(merged_df[column].median())
    return merged_df


def _conf_scores(train_participants, test_participants, target_col, n_classes):
    feature_columns = ["age", "gender_code", "date_ordinal"]
    model = HistGradientBoostingClassifier(random_state=config.RANDOM_SEED)
    model.fit(train_participants[feature_columns], train_participants[target_col])
    proba = model.predict_proba(test_participants[feature_columns])
    accuracy = accuracy_score(test_participants[target_col], model.predict(test_participants[feature_columns]))
    if n_classes == 2:
        auc = roc_auc_score(test_participants[target_col], proba[:, 1])
    else:
        auc = roc_auc_score(test_participants[target_col], proba, multi_class="ovr", average="macro")
    return accuracy, auc


def _conf_univariate_auc(test_participants, feature, target_col="is_sick") -> float:
    try:
        return roc_auc_score(test_participants[target_col], test_participants[feature])
    except ValueError:
        return float("nan")


def run_confound(args) -> None:
    bundle = joblib.load(SPLIT_DATA_PATH)
    metadata_df = _conf_metadata()
    train_participants = _conf_participant_frame(bundle["train_df"], metadata_df)
    test_participants = _conf_participant_frame(bundle["test_df"], metadata_df)
    voice_metrics_by_target = {}
    if RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        for target in config.TARGETS:
            ensemble_metrics = results.get(target, {}).get("ensemble", {})
            voice_metrics_by_target[target] = {"accuracy": ensemble_metrics.get("accuracy"), "auc": ensemble_metrics.get("auc"), "baseline": results.get(target, {}).get("baseline_accuracy")}
    lines = [
        "# 12 Confound Audit", "",
        "Metadata-only models (age + gender + recording date), trained on the same",
        "participant-disjoint split as the voice models. If these rival the voice",
        "ensemble, the headline accuracy is partly explained by *who/when* was",
        "recorded rather than *how their voice sounds*.", "",
        "| Target | Baseline | Metadata-only acc | Metadata-only AUC | Voice ensemble acc | Voice ensemble AUC |",
        "|---|---|---|---|---|---|",
    ]
    for target in config.TARGETS:
        target_col = config.TARGET_LABEL_COLUMNS[target]
        n_classes = int(test_participants[target_col].max()) + 1
        accuracy, auc = _conf_scores(train_participants, test_participants, target_col, n_classes)
        voice_metrics = voice_metrics_by_target.get(target, {})
        lines.append(
            f"| {target} | {voice_metrics.get('baseline', float('nan')):.3f} | {accuracy:.3f} | {auc:.3f} | "
            f"{voice_metrics.get('accuracy', float('nan')):.3f} | {voice_metrics.get('auc', float('nan')):.3f} |"
        )
    lines += [
        "", "## Univariate association with is_sick (test set)",
        f"- Age alone (AUC): {_conf_univariate_auc(test_participants, 'age'):.3f}",
        f"- Recording date alone (AUC): {_conf_univariate_auc(test_participants, 'date_ordinal'):.3f}",
        f"- Gender alone (AUC): {_conf_univariate_auc(test_participants, 'gender_code'):.3f}",
        "", "## How to read this",
        "- Metadata-only AUC near 0.5 and accuracy near baseline => weak confounds, likely real signal.",
        "- Metadata-only approaching the voice ensemble => a chunk is recruitment/demographic confound.",
    ]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "12_confound_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\nconfound audit complete -> reports/12_confound_audit.md")


def _cv_score(y_true, proba, n_classes):
    pred = np.argmax(proba, axis=1)
    accuracy = accuracy_score(y_true, pred)
    f1 = f1_score(y_true, pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(y_true, proba[:, 1]) if n_classes == 2 else roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    return accuracy, f1, auc


def _cv_run(features_df, target, n_splits, n_repeats):
    label_col = config.TARGET_LABEL_COLUMNS[target]
    full_df = pd.concat([features_df, pd.get_dummies(features_df["audio_type"], prefix="audio_type", dtype=float)], axis=1)
    feature_columns = [column for column in full_df.columns if column not in config.NON_FEATURE_COLUMNS]
    participant_df = full_df.groupby("participant_id", as_index=False)[label_col].first()
    scores = {"accuracy": [], "f1": [], "auc": []}
    for repeat in range(n_repeats):
        stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED + repeat)
        for train_indices, test_indices in stratified_kfold.split(participant_df["participant_id"], participant_df[label_col]):
            train_participant_ids = set(participant_df["participant_id"].iloc[train_indices])
            test_participant_ids = set(participant_df["participant_id"].iloc[test_indices])
            train_rows = full_df[full_df["participant_id"].isin(train_participant_ids)]
            test_rows = full_df[full_df["participant_id"].isin(test_participant_ids)]
            n_classes = int(full_df[label_col].max()) + 1
            model = make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(),
                HistGradientBoostingClassifier(random_state=config.RANDOM_SEED, class_weight="balanced" if n_classes > 2 else None),
            )
            model.fit(train_rows[feature_columns], train_rows[label_col])
            aggregated_labels, aggregated_proba = _agg_by_participant(test_rows["participant_id"].to_numpy(), test_rows[label_col].to_numpy(), model.predict_proba(test_rows[feature_columns]))
            accuracy, f1, auc = _cv_score(aggregated_labels, aggregated_proba, n_classes)
            scores["accuracy"].append(accuracy)
            scores["f1"].append(f1)
            scores["auc"].append(auc)
    return scores


def _fmt(values):
    values_array = np.array(values, dtype=float)
    return f"{values_array.mean():.3f} +/- {values_array.std():.3f}  (95% CI {np.percentile(values_array, 2.5):.3f}-{np.percentile(values_array, 97.5):.3f})"


def run_crossval(args) -> None:
    features_df = pd.read_csv(FEATURES_PATH)
    lines = [
        "# 14 Repeated Cross-Validation (tabular GBM, participant-level)", "",
        f"- {args.splits}-fold x {args.repeats} repeats = {args.splits * args.repeats} folds",
        "- Participant-disjoint; per-participant aggregation; GBM base model.", "",
    ]
    for target in config.TARGETS:
        scores = _cv_run(features_df, target, args.splits, args.repeats)
        lines += [f"## {target}", f"- Accuracy: {_fmt(scores['accuracy'])}", f"- Macro F1: {_fmt(scores['f1'])}", f"- AUC:      {_fmt(scores['auc'])}", ""]
        print(f"{target}: accuracy {_fmt(scores['accuracy'])}")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "14_cross_validation.md").write_text("\n".join(lines), encoding="utf-8")
    print("cross-validation complete -> reports/14_cross_validation.md")


def _date_dated_features() -> pd.DataFrame:
    features_df = pd.read_csv(FEATURES_PATH)
    metadata_df = pd.read_csv(COSWARA_META).rename(columns={"id": "participant_id"})
    metadata_df["record_date"] = pd.to_datetime(metadata_df["record_date"], errors="coerce")
    return features_df.merge(metadata_df[["participant_id", "record_date"]], on="participant_id", how="left").dropna(subset=["record_date"]).copy()


def _date_assign_blocks(features_df, n_blocks) -> pd.DataFrame:
    participant_df = features_df.groupby("participant_id")["record_date"].first().reset_index().sort_values("record_date")
    participant_df["block"] = pd.qcut(participant_df["record_date"].rank(method="first"), n_blocks, labels=False)
    return features_df.merge(participant_df[["participant_id", "block"]], on="participant_id", how="left")


def _date_one_granularity(features_df, target, n_blocks) -> list[dict]:
    label_col = config.TARGET_LABEL_COLUMNS[target]
    blocked_df = _date_assign_blocks(features_df, n_blocks)
    full_df = pd.concat([blocked_df, pd.get_dummies(blocked_df["audio_type"], prefix="audio_type", dtype=float)], axis=1)
    non_feature_columns = config.NON_FEATURE_COLUMNS | {"record_date", "block"}
    feature_columns = [column for column in full_df.columns if column not in non_feature_columns]
    block_rows = []
    for block in sorted(full_df["block"].dropna().unique()):
        train_df = full_df[full_df["block"] != block]
        test_df = full_df[full_df["block"] == block]
        if train_df[label_col].nunique() < 2 or test_df.empty:
            continue
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), HistGradientBoostingClassifier(random_state=config.RANDOM_SEED))
        model.fit(train_df[feature_columns], train_df[label_col])
        proba = model.predict_proba(test_df[feature_columns])
        positive_class_col = list(model.classes_).index(1) if 1 in model.classes_ else None
        aggregated_labels, aggregated_proba = _agg_by_participant(test_df["participant_id"].to_numpy(), test_df[label_col].to_numpy(), proba)
        pred = np.array([model.classes_[i] for i in np.argmax(aggregated_proba, axis=1)])
        accuracy = accuracy_score(aggregated_labels, pred)
        balanced_acc = balanced_accuracy_score(aggregated_labels, pred)
        f1 = f1_score(aggregated_labels, pred, average="macro", zero_division=0)
        auc = np.nan
        if positive_class_col is not None and len(np.unique(aggregated_labels)) == 2:
            try:
                auc = roc_auc_score(aggregated_labels, aggregated_proba[:, positive_class_col])
            except ValueError:
                pass
        majority_class = pd.Series(train_df[label_col]).mode()[0]
        block_rows.append({
            "n_blocks": n_blocks, "block": int(block),
            "date_range": f"{test_df['record_date'].min().date()} to {test_df['record_date'].max().date()}",
            "n_participants": len(aggregated_labels), "block_baseline_acc": float(np.mean(aggregated_labels == majority_class)),
            "accuracy": accuracy, "balanced_accuracy": balanced_acc, "f1_macro": f1, "auc": auc,
        })
    return block_rows


def _ci(values):
    return (float("nan"), float("nan")) if len(values) < 2 else (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))


def run_datesplit(args) -> None:
    features_df = _date_dated_features()
    target = "binary"
    all_block_rows = []
    for n_blocks in args.block_counts:
        block_rows = _date_one_granularity(features_df, target, n_blocks)
        all_block_rows += block_rows
        for row in block_rows:
            print(f"[n_blocks={n_blocks}] block {row['block']} ({row['date_range']}, n={row['n_participants']}): "
                  f"baseline={row['block_baseline_acc']:.3f} acc={row['accuracy']:.3f} bal_acc={row['balanced_accuracy']:.3f} auc={row['auc']:.3f}")
    result_df = pd.DataFrame(all_block_rows)
    auc_values = result_df["auc"].dropna().to_numpy()
    balanced_acc_values = result_df["balanced_accuracy"].to_numpy()
    acc_values = result_df["accuracy"].to_numpy()
    baseline_values = result_df["block_baseline_acc"].to_numpy()
    t_statistic, p_value = (stats.ttest_1samp(auc_values, 0.5) if len(auc_values) >= 2 and np.std(auc_values) > 0 else (float("nan"), float("nan")))
    per_config_mean_auc = result_df.groupby("n_blocks")["auc"].mean().dropna()
    t_statistic_config, p_value_config = (stats.ttest_1samp(per_config_mean_auc.to_numpy(), 0.5) if len(per_config_mean_auc) >= 2 and per_config_mean_auc.std() > 0 else (float("nan"), float("nan")))

    lines = [
        "# 15 Date-Stratified (Repeated Temporal Holdout) Evaluation - Binary Only", "",
        f"Pooled across block counts {args.block_counts} ({len(result_df)} block-holdout estimates). Each block:",
        "train on all other blocks, test on a chronological block never seen in training.", "",
        "## Pooled results",
        f"- Mean accuracy: {acc_values.mean():.3f} +/- {acc_values.std():.3f}  (95% CI {_ci(acc_values)[0]:.3f}-{_ci(acc_values)[1]:.3f})",
        f"- Mean block-own-baseline accuracy: {baseline_values.mean():.3f} +/- {baseline_values.std():.3f}",
        f"- Mean balanced accuracy: {balanced_acc_values.mean():.3f} +/- {balanced_acc_values.std():.3f}  (95% CI {_ci(balanced_acc_values)[0]:.3f}-{_ci(balanced_acc_values)[1]:.3f})  [0.5 = chance]",
        f"- Mean AUC: {auc_values.mean():.3f} +/- {auc_values.std():.3f}  (95% CI {_ci(auc_values)[0]:.3f}-{_ci(auc_values)[1]:.3f})  [0.5 = chance]",
        f"- t-test on pooled block AUCs vs chance: t={t_statistic:.3f}, p={p_value:.4f} (**not independent samples**)",
        f"- t-test on the {len(per_config_mean_auc)} per-configuration mean AUCs vs chance (more defensible): t={t_statistic_config:.3f}, p={p_value_config:.4f}",
        f"- Blocks beating their own baseline accuracy: {int((result_df['accuracy'] > result_df['block_baseline_acc']).sum())}/{len(result_df)}",
        "", "## Per-configuration summary", "",
        "| n_blocks | mean accuracy | mean balanced accuracy | mean AUC |", "|---|---|---|---|",
    ]
    for n_blocks in args.block_counts:
        subset_df = result_df[result_df["n_blocks"] == n_blocks]
        lines.append(f"| {n_blocks} | {subset_df['accuracy'].mean():.3f} | {subset_df['balanced_accuracy'].mean():.3f} | {subset_df['auc'].mean():.3f} |")
    lines += [
        "", "## How to read this",
        "- Pooled mean AUC CI excluding 0.5 (and p<0.05) => real signal beyond the date shortcut.",
        "- CI straddling 0.5 => earlier random-split accuracy was inflated by the recording-date confound.",
    ]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "15_date_stratified_evaluation.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nPooled: acc={acc_values.mean():.3f} bal_acc={balanced_acc_values.mean():.3f} auc={auc_values.mean():.3f} t-test p={p_value:.4f}")
    print("date-stratified evaluation complete -> reports/15_date_stratified_evaluation.md")


_XD_FEATURE_PATHS = {
    "coswara": (DATA_DIR / "features_extracted.csv", DATA_DIR / "embeddings.npy"),
    "coughvid": (DATA_DIR / "coughvid_features_extracted.csv", DATA_DIR / "coughvid_embeddings.npy"),
    "sounddr": (DATA_DIR / "sounddr_features_extracted.csv", DATA_DIR / "sounddr_embeddings.npy"),
}
_XD_EXTRA_NONFEATURE = {
    "source_dataset", "raw_audio_path", "record_date", "block", "recording_datetime",
    "duration_sec", "file_size_bytes", "age", "gender", "label_name",
}


def _xd_feature_columns(df: pd.DataFrame) -> list[str]:
    excl = config.NON_FEATURE_COLUMNS | _XD_EXTRA_NONFEATURE
    return [c for c in df.columns if c not in excl and pd.api.types.is_numeric_dtype(df[c])]


def _xd_load(name: str) -> dict:
    csv_path, emb_path = _XD_FEATURE_PATHS[name]
    if not csv_path.exists() or not emb_path.exists():
        raise FileNotFoundError(
            f"{name}: missing {csv_path.name} or {emb_path.name} - build it first "
            f"(preprocessing.py {name} / features.py embed)."
        )
    df = pd.read_csv(csv_path).reset_index(drop=True)
    emb = np.load(emb_path)
    if len(df) != len(emb):
        raise ValueError(f"{name}: feature rows ({len(df)}) != embedding rows ({len(emb)})")
    mask = (df["audio_type"] == "cough").to_numpy()
    df_cough = df[mask].reset_index(drop=True)
    emb_cough = emb[mask]
    return {
        "name": name,
        "df": df_cough,
        "emb": emb_cough,
        "y": df_cough["is_sick"].astype(int).to_numpy(),
        "pids": df_cough["participant_id"].astype(str).to_numpy(),
    }


def _xd_agg_by_participant(pids: np.ndarray, y: np.ndarray, p_sick: np.ndarray):
    agg = (
        pd.DataFrame({"pid": pids, "y": y, "p": p_sick})
        .groupby("pid")
        .agg(y=("y", "first"), p=("p", "mean"))
        .reset_index()
    )
    return agg["y"].to_numpy(), agg["p"].to_numpy()


def _xd_metrics(y_true: np.ndarray, p_sick: np.ndarray) -> dict:
    pred = (p_sick >= 0.5).astype(int)
    both_classes = len(np.unique(y_true)) == 2
    return {
        "n": int(len(y_true)),
        "baseline_acc": float(max(np.mean(y_true), 1 - np.mean(y_true))),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "recall_sick": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(y_true, p_sick)) if both_classes else float("nan"),
    }


def _xd_zscore_own(matrix: np.ndarray) -> np.ndarray:
    mu = np.nanmean(matrix, axis=0)
    sd = np.nanstd(matrix, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (matrix - mu) / sd


def _xd_train_eval(train_sets: list[dict], test_set: dict, feat_cols: list[str],
                   standardize_per_dataset: bool = False) -> dict:
    if standardize_per_dataset:
        feat_tr_blocks = [_xd_zscore_own(s["df"][feat_cols].to_numpy(dtype=float)) for s in train_sets]
        emb_tr_blocks = [_xd_zscore_own(s["emb"]) for s in train_sets]
        X_feat_te = _xd_zscore_own(test_set["df"][feat_cols].to_numpy(dtype=float))
        X_emb_te = _xd_zscore_own(test_set["emb"])
    else:
        feat_tr_blocks = [s["df"][feat_cols].to_numpy(dtype=float) for s in train_sets]
        emb_tr_blocks = [s["emb"] for s in train_sets]
        X_feat_te = test_set["df"][feat_cols].to_numpy(dtype=float)
        X_emb_te = test_set["emb"]
    X_feat_tr = np.vstack(feat_tr_blocks)
    X_emb_tr = np.vstack(emb_tr_blocks)
    y_tr = np.concatenate([s["y"] for s in train_sets])


    valid_cols = [
        j for j in range(X_feat_tr.shape[1])
        if np.unique(X_feat_tr[:, j][~np.isnan(X_feat_tr[:, j])]).size >= 2
    ]
    X_feat_tr = X_feat_tr[:, valid_cols]
    X_feat_te = X_feat_te[:, valid_cols]


    gbm = HistGradientBoostingClassifier(
        class_weight="balanced", random_state=config.RANDOM_SEED, max_iter=400
    ).fit(X_feat_tr, y_tr)
    emb_clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0),
    ).fit(X_emb_tr, y_tr)
    p_gbm = gbm.predict_proba(X_feat_te)[:, 1]
    p_emb = emb_clf.predict_proba(X_emb_te)[:, 1]
    p_ens = (p_gbm + p_emb) / 2.0

    out = {}
    for label, p in [("gbm_features", p_gbm), ("embedding", p_emb), ("ensemble", p_ens)]:
        y_agg, p_agg = _xd_agg_by_participant(test_set["pids"], test_set["y"], p)
        out[label] = _xd_metrics(y_agg, p_agg)
    return out


def _xd_fmt(name: str, m: dict) -> str:
    return (f"  {name:<16} auc={m['auc']:.3f}  bal_acc={m['balanced_accuracy']:.3f}  "
            f"acc={m['accuracy']:.3f}  recall_sick={m['recall_sick']:.3f}  "
            f"f1={m['f1_macro']:.3f}  (baseline_acc={m['baseline_acc']:.3f}, n={m['n']})")


def run_crossdataset(args) -> None:
    train_set = _xd_load(args.train)
    test_set = _xd_load(args.test)
    feat_cols = _xd_feature_columns(train_set["df"])
    standardize = getattr(args, "standardize", False)
    print(f"cross-dataset: train={args.train} (n={len(train_set['y'])} clips)  "
          f"test={args.test} (n={len(test_set['y'])} clips)  features={len(feat_cols)}  "
          f"per_dataset_standardize={standardize}")
    results = _xd_train_eval([train_set], test_set, feat_cols, standardize_per_dataset=standardize)

    lines = [
        f"# 16 Cross-Dataset Evaluation - train {args.train} -> test {args.test}",
        "",
        "Zero-shot: model never saw a single clip from the test dataset. Harmonized",
        "label = currently symptomatic vs healthy-asymptomatic. Cough clips only.",
        "Metrics are participant-aggregated.",
        "",
        f"- Train: **{args.train}** ({len(train_set['y'])} cough clips)",
        f"- Test:  **{args.test}** ({len(test_set['y'])} cough clips)",
        f"- Shared numeric features: {len(feat_cols)}; embedding dim: {test_set['emb'].shape[1]}",
        "",
        "| model | AUC | balanced acc | accuracy | recall(sick) | f1-macro |",
        "|---|---|---|---|---|---|",
    ]
    for label in ("gbm_features", "embedding", "ensemble"):
        m = results[label]
        lines.append(f"| {label} | {m['auc']:.3f} | {m['balanced_accuracy']:.3f} | "
                     f"{m['accuracy']:.3f} | {m['recall_sick']:.3f} | {m['f1_macro']:.3f} |")
    lines += ["", f"Test-set baseline (majority-class) accuracy: {results['ensemble']['baseline_acc']:.3f}."]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"16_cross_dataset_{args.train}_to_{args.test}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")

    for label in ("gbm_features", "embedding", "ensemble"):
        print(_xd_fmt(label, results[label]))
    print(f"cross-dataset evaluation complete -> {out_path.relative_to(ROOT)}")


def run_leaveoneout(args) -> None:
    datasets = args.datasets
    loaded = {name: _xd_load(name) for name in datasets}
    feat_cols = _xd_feature_columns(loaded[datasets[0]]["df"])
    standardize = getattr(args, "standardize", False)

    lines = [
        "# 17 Leave-One-Dataset-Out Evaluation",
        "",
        "Each dataset is held out once as the test set while the model trains on the",
        "pooled remaining datasets. Harmonized symptomatic-vs-healthy label, cough",
        "clips only, participant-aggregated metrics. This is the strongest",
        "generalization test in the project.",
        "",
        "| held-out test | train on | AUC | balanced acc | accuracy | recall(sick) | baseline acc | n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    print(f"leave-one-dataset-out over {datasets} (features={len(feat_cols)})")
    auc_values = []
    for test_name in datasets:
        train_names = [d for d in datasets if d != test_name]
        train_sets = [loaded[d] for d in train_names]
        results = _xd_train_eval(train_sets, loaded[test_name], feat_cols, standardize_per_dataset=standardize)
        m = results["ensemble"]
        auc_values.append(m["auc"])
        lines.append(f"| **{test_name}** | {'+'.join(train_names)} | {m['auc']:.3f} | "
                     f"{m['balanced_accuracy']:.3f} | {m['accuracy']:.3f} | {m['recall_sick']:.3f} | "
                     f"{m['baseline_acc']:.3f} | {m['n']} |")
        print(f"  held-out {test_name:<9} (train={'+'.join(train_names)}):")
        for label in ("gbm_features", "embedding", "ensemble"):
            print(_xd_fmt(label, results[label]))
    mean_auc = float(np.nanmean(auc_values)) if auc_values else float("nan")
    lines += ["", f"Mean held-out ensemble AUC across datasets: **{mean_auc:.3f}**."]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "17_leave_one_dataset_out.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"mean held-out ensemble AUC = {mean_auc:.3f}")
    print("leave-one-dataset-out evaluation complete -> reports/17_leave_one_dataset_out.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_eda = subparsers.add_parser("eda", help="exploratory data analysis")
    p_eda.add_argument("--sample-rate", type=int, default=config.DEFAULT_SAMPLE_RATE)
    p_eda.add_argument("--n-mels", type=int, default=config.DEFAULT_N_MELS)
    p_eda.set_defaults(func=run_eda)

    subparsers.add_parser("stattests", help="univariate feature tests + heatmap").set_defaults(func=run_stattests)

    p_val = subparsers.add_parser("validate", help="RF cross-val + sensitivity sweep")
    p_val.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p_val.set_defaults(func=run_validate)

    subparsers.add_parser("errors", help="error breakdown by subgroup").set_defaults(func=run_errors)
    subparsers.add_parser("confound", help="metadata-only confound audit").set_defaults(func=run_confound)

    p_cv = subparsers.add_parser("crossval", help="repeated participant-level CV")
    p_cv.add_argument("--splits", type=int, default=5)
    p_cv.add_argument("--repeats", type=int, default=3)
    p_cv.set_defaults(func=run_crossval)

    p_ds = subparsers.add_parser("datesplit", help="temporal holdout (confound-robust)")
    p_ds.add_argument("--block-counts", type=int, nargs="+", default=[6, 8, 10, 12, 15])
    p_ds.set_defaults(func=run_datesplit)

    p_xd = subparsers.add_parser("crossdataset", help="train on one dataset, test on another (zero-shot)")
    p_xd.add_argument("--train", required=True, choices=list(_XD_FEATURE_PATHS))
    p_xd.add_argument("--test", required=True, choices=list(_XD_FEATURE_PATHS))
    p_xd.add_argument("--standardize", action="store_true", help="per-dataset z-score before pooling")
    p_xd.set_defaults(func=run_crossdataset)

    p_lodo = subparsers.add_parser("leaveoneout", help="leave-one-dataset-out over pooled datasets")
    p_lodo.add_argument("--datasets", nargs="+", default=list(_XD_FEATURE_PATHS),
                        choices=list(_XD_FEATURE_PATHS))
    p_lodo.add_argument("--standardize", action="store_true", help="per-dataset z-score before pooling")
    p_lodo.set_defaults(func=run_leaveoneout)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
