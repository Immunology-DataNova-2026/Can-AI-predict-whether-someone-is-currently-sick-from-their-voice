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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

import config
from training import (
    CNNClassifier,
    SpectrogramDataset,
    Wav2Vec2Classifier,
    WaveformDataset,
)

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
FIGURES_DIR = ROOT / "figures"
REPORTS_DIR = ROOT / "reports"
RESULTS_PATH = ROOT / "results.json"
MODELS_DIR = ROOT / "models"

sns.set_theme(style="ticks")
DATA_DIR = ROOT / "data"
SPLIT_DATA_PATH = ARTIFACTS_DIR / "split_data.joblib"


def _load_tabular_model(model_name: str, target: str):
    path = MODELS_DIR / f"model_{model_name}_{target}.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


def _remap_proba(model, features, n_classes: int) -> np.ndarray:
    proba = model.predict_proba(features)
    model_classes = list(model.classes_)
    if len(model_classes) == n_classes and model_classes == list(range(n_classes)):
        return proba
    widened_proba = np.zeros((features.shape[0], n_classes))
    for column_index, class_label in enumerate(model_classes):
        widened_proba[:, int(class_label)] = proba[:, column_index]
    return widened_proba


def _tabular_proba(model_name: str, target: str, features, clips_df, n_classes: int):
    if config.PER_AUDIO_TYPE_TABULAR:
        audio_types = clips_df["audio_type"].to_numpy()
        proba = np.zeros((features.shape[0], n_classes))
        found_any = False
        for audio_type in np.unique(audio_types):
            path = MODELS_DIR / f"model_{model_name}_{target}_{audio_type}.pkl"
            if not path.exists():
                continue
            found_any = True
            type_mask = audio_types == audio_type
            proba[type_mask] = _remap_proba(joblib.load(path), features[type_mask], n_classes)
        if found_any:
            return proba
    model = _load_tabular_model(model_name, target)
    return None if model is None else _remap_proba(model, features, n_classes)


def _load_cnn_model(target: str, device: torch.device):
    checkpoint_path = MODELS_DIR / f"model_cnn_{target}.pth"
    if not checkpoint_path.exists():
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cnn = CNNClassifier(num_classes=int(checkpoint["num_classes"])).to(device)
    cnn.load_state_dict(checkpoint["state_dict"])
    cnn.eval()
    return cnn


def _load_finetune_model(target: str, device: torch.device):
    checkpoint_path = MODELS_DIR / f"model_finetune_{target}.pth"
    if not checkpoint_path.exists():
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = Wav2Vec2Classifier(
        int(checkpoint["num_classes"]),
        int(checkpoint["unfreeze_layers"]),
        float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _finetune_proba(
    model, target: str, clips_df: pd.DataFrame, device: torch.device
) -> np.ndarray:
    label_col = config.TARGET_LABEL_COLUMNS[target]
    loader = DataLoader(
        WaveformDataset(clips_df[["processed_audio_path", label_col]], label_col=label_col),
        batch_size=config.FINETUNE_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )
    probabilities = []
    with torch.no_grad():
        for batch_inputs, _ in loader:
            logits = model(batch_inputs.to(device))
            probabilities.extend(torch.softmax(logits, dim=1).cpu().numpy().tolist())
    return np.array(probabilities)


def _cnn_proba(
    cnn, target: str, clips_df: pd.DataFrame, batch_size: int, device: torch.device
) -> np.ndarray:
    label_col = config.TARGET_LABEL_COLUMNS[target]
    loader = DataLoader(
        SpectrogramDataset(
            clips_df[["spectrogram_path", label_col]], label_col=label_col, augment=False
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    probabilities = []
    with torch.no_grad():
        for batch_inputs, _ in loader:
            logits = cnn(batch_inputs.to(device))
            probabilities.extend(torch.softmax(logits, dim=1).cpu().numpy().tolist())
    return np.array(probabilities)


def _auc(y_true: np.ndarray, prob: np.ndarray, n_classes: int) -> float:
    if n_classes == 2:
        return float(roc_auc_score(y_true, prob[:, 1]))
    return float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro"))


def _auprc(y_true: np.ndarray, prob: np.ndarray, n_classes: int) -> float:
    if n_classes == 2:
        return float(average_precision_score(y_true, prob[:, 1]))
    y_true_binarized = label_binarize(y_true, classes=list(range(n_classes)))
    return float(average_precision_score(y_true_binarized, prob, average="macro"))


def _bootstrap_ci(
    y_true: np.ndarray, prob: np.ndarray, n_classes: int, metric_fn
) -> tuple[float, float]:
    rng = np.random.default_rng(config.RANDOM_SEED)
    metric_values = []
    n_samples = len(y_true)
    for _ in range(config.BOOTSTRAP_ITERATIONS):
        sample_indices = rng.integers(0, n_samples, n_samples)
        resampled_y, resampled_prob = y_true[sample_indices], prob[sample_indices]
        if len(np.unique(resampled_y)) < 2:
            continue
        try:
            metric_values.append(metric_fn(resampled_y, resampled_prob, n_classes))
        except ValueError:
            continue
    if not metric_values:
        return np.nan, np.nan
    return float(np.percentile(metric_values, 2.5)), float(np.percentile(metric_values, 97.5))


def _bootstrap_accuracy_vs_baseline(
    y_true: np.ndarray, pred: np.ndarray, baseline_accuracy: float
) -> tuple[tuple[float, float], float]:
    rng = np.random.default_rng(config.RANDOM_SEED)
    n_samples = len(y_true)
    accuracies = []
    for _ in range(config.BOOTSTRAP_ITERATIONS):
        sample_indices = rng.integers(0, n_samples, n_samples)
        accuracies.append(float(np.mean(pred[sample_indices] == y_true[sample_indices])))
    accuracies = np.array(accuracies)
    confidence_interval = (float(np.percentile(accuracies, 2.5)), float(np.percentile(accuracies, 97.5)))
    p_below_baseline = float(np.mean(accuracies <= baseline_accuracy))
    return confidence_interval, p_below_baseline


def _collect_metrics(
    y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray, n_classes: int
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(
            precision_score(y_true, pred, average="macro", zero_division=0)
        ),
        "recall": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "auc": _auc(y_true, prob, n_classes),
        "auprc": _auprc(y_true, prob, n_classes),
    }


def _tune_binary_threshold(y_val, val_prob_class1, metric: str) -> float:
    best_threshold, best_score = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        pred = (val_prob_class1 >= threshold).astype(int)
        score = (
            f1_score(y_val, pred, average="macro", zero_division=0)
            if metric == "f1"
            else balanced_accuracy_score(y_val, pred)
        )
        if score > best_score:
            best_score, best_threshold = score, float(threshold)
    return best_threshold


def _class_names(target: str, n_classes: int) -> list[str]:
    if target == "binary":
        return config.BINARY_CLASS_NAMES
    return [str(i) for i in range(n_classes)]


def _plot_roc_curves(
    target: str, y_true: np.ndarray, probs: dict[str, np.ndarray], n_classes: int
) -> None:
    plt.figure(figsize=(7, 6))
    if n_classes == 2:
        for model_name, prob in probs.items():
            false_positive_rate, true_positive_rate, _ = roc_curve(y_true, prob[:, 1])
            plt.plot(
                false_positive_rate,
                true_positive_rate,
                label=f"{model_name} (AUC={_auc(y_true, prob, n_classes):.3f})",
            )
    else:
        y_true_binarized = label_binarize(y_true, classes=list(range(n_classes)))
        for model_name, prob in probs.items():
            false_positive_rate, true_positive_rate, _ = roc_curve(y_true_binarized.ravel(), prob.ravel())
            plt.plot(
                false_positive_rate,
                true_positive_rate,
                label=f"{model_name} (micro AUC={_auc(y_true, prob, n_classes):.3f})",
            )
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - {target}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"06_roc_curves_{target}.png", dpi=300)
    plt.close()


def _plot_pr_curves(
    target: str, y_true: np.ndarray, probs: dict[str, np.ndarray], n_classes: int
) -> None:
    plt.figure(figsize=(7, 6))
    if n_classes == 2:
        prevalence = float(np.mean(y_true))
        for model_name, prob in probs.items():
            precision, recall, _ = precision_recall_curve(y_true, prob[:, 1])
            plt.plot(
                recall,
                precision,
                label=f"{model_name} (AUPRC={_auprc(y_true, prob, n_classes):.3f})",
            )
        plt.axhline(
            prevalence,
            color="k",
            linestyle="--",
            alpha=0.5,
            label=f"baseline prevalence ({prevalence:.3f})",
        )
    else:
        y_true_binarized = label_binarize(y_true, classes=list(range(n_classes)))
        for model_name, prob in probs.items():
            precision, recall, _ = precision_recall_curve(y_true_binarized.ravel(), prob.ravel())
            plt.plot(
                recall,
                precision,
                label=f"{model_name} (micro AUPRC={_auprc(y_true, prob, n_classes):.3f})",
            )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curves - {target}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"11_pr_curves_{target}.png", dpi=300)
    plt.close()


def _plot_confusion_matrices(
    target: str,
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    class_names: list[str],
) -> None:
    num_models = len(preds)
    _, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 4.5))
    axes = np.atleast_1d(axes)
    for ax, (model_name, pred) in zip(axes, preds.items(), strict=True):
        confusion = confusion_matrix(y_true, pred, normalize="true")
        ConfusionMatrixDisplay(confusion, display_labels=class_names).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format=".2f"
        )
        ax.set_title(model_name)
    plt.suptitle(f"Confusion Matrices (row-normalized) - {target}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"07_confusion_matrices_{target}.png", dpi=300)
    plt.close()


def _plot_calibration(
    target: str, y_true: np.ndarray, probs: dict[str, np.ndarray], n_classes: int
) -> None:
    plt.figure(figsize=(6, 5))
    for model_name, prob in probs.items():
        confidence = np.max(prob, axis=1)
        pred = np.argmax(prob, axis=1)
        correct = (pred == y_true).astype(int)
        observed_accuracy, predicted_confidence = calibration_curve(
            correct, confidence, n_bins=config.CALIBRATION_BINS, strategy="quantile"
        )
        plt.plot(predicted_confidence, observed_accuracy, marker="o", label=model_name)
    plt.plot([0, 1], [0, 1], "--", color="gray", label="perfectly calibrated")
    plt.xlabel("Predicted confidence")
    plt.ylabel("Observed accuracy")
    plt.title(f"Calibration - {target}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"09_calibration_curves_{target}.png", dpi=300)
    plt.close()


def _aggregate_by_participant(
    participant_ids: np.ndarray, y_true: np.ndarray, probs: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    unique_ids = pd.unique(participant_ids)
    labels_by_id = pd.Series(y_true).groupby(participant_ids, sort=False).first()
    aggregated_y_true = labels_by_id.loc[unique_ids].values
    aggregated_probs = {}
    for model_name, prob in probs.items():
        averaged_prob_df = pd.DataFrame(prob).groupby(participant_ids, sort=False).mean()
        aggregated_probs[model_name] = averaged_prob_df.loc[unique_ids].values
    return unique_ids, aggregated_y_true, aggregated_probs


def _stack_meta_features(probs: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([probs[model_name] for model_name in sorted(probs)], axis=1)


def _build_meta_learner():
    if getattr(config, "STACK_META_LEARNER", "logistic") == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return make_pipeline(
            StandardScaler(),
            HistGradientBoostingClassifier(
                random_state=config.RANDOM_SEED,
                max_leaf_nodes=15,
                max_iter=200,
                class_weight="balanced",
            ),
        )
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced"),
    )


def _participant_meta_extras(
    participant_ids: np.ndarray, clip_probs: dict[str, np.ndarray]
) -> np.ndarray:
    per_clip_mean = np.mean([clip_probs[model_name] for model_name in sorted(clip_probs)], axis=0)
    unique_ids = pd.unique(participant_ids)
    clip_mean_df = pd.DataFrame(per_clip_mean)
    grouped_by_participant = clip_mean_df.groupby(participant_ids, sort=False)
    clip_counts = grouped_by_participant.size().loc[unique_ids].to_numpy().reshape(-1, 1)
    clip_stds = grouped_by_participant.std().fillna(0.0).loc[unique_ids].to_numpy()
    mean_disagreement = clip_stds.mean(axis=1, keepdims=True)
    return np.hstack([np.log1p(clip_counts), mean_disagreement])


def evaluate_target(target: str, batch_size: int, device: torch.device) -> dict:
    bundle = joblib.load(SPLIT_DATA_PATH)
    test_features = bundle["X_test"]
    test_labels = np.array(bundle[f"y_test_{target}"])
    train_labels = np.array(bundle[f"y_train_{target}"])
    test_df = bundle["test_df"].reset_index(drop=True)
    n_classes = int(np.max(test_labels)) + 1
    class_names = _class_names(target, n_classes)

    embedding_model = _load_tabular_model("embedding", target)
    hubert_embedding_model = _load_tabular_model("embedding_hubert", target)
    cnn_model = _load_cnn_model(target, device)
    finetune_model = _load_finetune_model(target, device)

    allowed_models = set(
        getattr(config, "TARGET_ENSEMBLE", {}).get(target, config.ENSEMBLE_MODELS)
    )

    def assemble(features, embeddings, hubert_embeddings, clips_df) -> dict[str, np.ndarray]:
        model_probs: dict[str, np.ndarray] = {}
        if "random_forest" in allowed_models:
            rf_proba = _tabular_proba("random_forest", target, features, clips_df, n_classes)
            if rf_proba is not None:
                model_probs["random_forest"] = rf_proba
        if "gradient_boosting" in allowed_models:
            gbm_proba = _tabular_proba("gradient_boosting", target, features, clips_df, n_classes)
            if gbm_proba is not None:
                model_probs["gradient_boosting"] = gbm_proba
        if "embedding" in allowed_models and embedding_model is not None and embeddings is not None:
            model_probs["embedding"] = _remap_proba(embedding_model, embeddings, n_classes)
        if (
            "embedding_hubert" in allowed_models
            and hubert_embedding_model is not None
            and hubert_embeddings is not None
        ):
            model_probs["embedding_hubert"] = _remap_proba(hubert_embedding_model, hubert_embeddings, n_classes)
        if "cnn" in allowed_models and cnn_model is not None:
            model_probs["cnn"] = _cnn_proba(cnn_model, target, clips_df, batch_size, device)
        if "finetune" in allowed_models and finetune_model is not None:
            model_probs["finetune"] = _finetune_proba(finetune_model, target, clips_df, device)
        return model_probs

    probs = assemble(test_features, bundle.get("emb_test"), bundle.get("emb_hubert_test"), test_df)

    if not probs:
        print(f"skipping {target}: no trained models found")
        return {}

    test_participant_ids = test_df["participant_id"].values
    test_meta_extras = _participant_meta_extras(test_participant_ids, probs)
    participant_ids, test_labels, probs = _aggregate_by_participant(
        test_participant_ids, test_labels, probs
    )


    val_df = bundle["val_df"].reset_index(drop=True)
    val_features = bundle["X_val"]
    val_labels_raw = np.array(bundle[f"y_val_{target}"])

    val_probs = assemble(val_features, bundle.get("emb_val"), bundle.get("emb_hubert_val"), val_df)

    val_participant_ids = val_df["participant_id"].values
    val_meta_extras = _participant_meta_extras(val_participant_ids, val_probs)
    _, val_labels, val_probs = _aggregate_by_participant(
        val_participant_ids, val_labels_raw, val_probs
    )

    val_ensemble = None
    if len(val_probs) < 2 or len(np.unique(val_labels)) < n_classes:
        probs["ensemble"] = np.mean(list(probs.values()), axis=0)
        if val_probs:
            val_ensemble = np.mean(list(val_probs.values()), axis=0)
    else:
        meta_val_features = np.hstack([_stack_meta_features(val_probs), val_meta_extras])
        meta_test_features = np.hstack([_stack_meta_features(probs), test_meta_extras])
        meta_model = _build_meta_learner()
        meta_model.fit(meta_val_features, val_labels)

        def _widen(raw_proba):
            widened = np.zeros((raw_proba.shape[0], n_classes))
            for column_index, class_label in enumerate(meta_model.classes_):
                widened[:, int(class_label)] = raw_proba[:, column_index]
            return widened

        probs["ensemble"] = _widen(meta_model.predict_proba(meta_test_features))
        val_ensemble = _widen(meta_model.predict_proba(meta_val_features))

    preds = {model_name: np.argmax(prob, axis=1) for model_name, prob in probs.items()}


    binary_threshold = 0.5
    if (
        n_classes == 2
        and getattr(config, "TUNE_BINARY_THRESHOLD", False)
        and val_ensemble is not None
    ):
        binary_threshold = _tune_binary_threshold(
            val_labels, val_ensemble[:, 1], config.BINARY_THRESHOLD_METRIC
        )
        preds["ensemble"] = (probs["ensemble"][:, 1] >= binary_threshold).astype(int)

    results = {
        model_name: {
            **_collect_metrics(test_labels, probs[model_name], preds[model_name], n_classes),
            "auc_ci_95": list(_bootstrap_ci(test_labels, probs[model_name], n_classes, _auc)),
        }
        for model_name in probs
    }

    majority_class = int(pd.Series(train_labels).mode()[0])
    baseline_accuracy = float(np.mean(test_labels == majority_class))
    ensemble_accuracy_ci, p_below_baseline = _bootstrap_accuracy_vs_baseline(
        test_labels, preds["ensemble"], baseline_accuracy
    )
    results["baseline_accuracy"] = baseline_accuracy
    results["ensemble_accuracy_ci_95"] = list(ensemble_accuracy_ci)
    results["bootstrap_p_ensemble_at_or_below_baseline"] = p_below_baseline
    if n_classes == 2:
        results["baseline_prevalence"] = float(np.mean(test_labels))


        results["binary_threshold"] = binary_threshold

    predictions = pd.DataFrame(
        {"participant_id": participant_ids, "true_label": test_labels}
    )
    for model_name in probs:
        predictions[f"{model_name}_pred"] = preds[model_name]
        predictions[f"{model_name}_conf"] = np.max(probs[model_name], axis=1)
    predictions.to_csv(DATA_DIR / f"predictions_test_set_{target}.csv", index=False)

    _plot_roc_curves(target, test_labels, probs, n_classes)
    _plot_pr_curves(target, test_labels, probs, n_classes)
    _plot_confusion_matrices(target, test_labels, preds, class_names)
    _plot_calibration(target, test_labels, probs, n_classes)

    print(
        f"{target}: baseline_accuracy={baseline_accuracy:.4f} ensemble_accuracy={results['ensemble']['accuracy']:.4f} "
        f"ensemble_auc={results['ensemble']['auc']:.4f} p(ensemble<=baseline)={p_below_baseline:.4f}"
    )
    return results


def run_evaluate(args) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = {
        target: evaluate_target(target, args.batch_size, device) for target in config.TARGETS
    }
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print("evaluation complete")


def _model_results(target_results: dict) -> dict:
    return {key: value for key, value in target_results.items() if isinstance(value, dict)}


def plot_feature_importance(target: str) -> None:
    path = ARTIFACTS_DIR / f"rf_feature_importance_{target}.csv"
    if not path.exists():
        return
    importance_df = pd.read_csv(path).head(config.TOP_RF_FEATURES_TO_PLOT)
    plt.figure(figsize=(8, 8))
    sns.barplot(data=importance_df, x="importance", y="feature", orient="h")
    plt.title(f"Top {config.TOP_RF_FEATURES_TO_PLOT} Random Forest Features - {target}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"04_feature_importance_{target}.png", dpi=300)
    plt.close()


def plot_cnn_training_curves(target: str) -> None:
    path = ARTIFACTS_DIR / f"cnn_training_history_{target}.csv"
    if not path.exists():
        return
    history_df = pd.read_csv(path)
    _, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(history_df["epoch"], history_df["train_loss"], label="train")
    axes[0, 0].plot(history_df["epoch"], history_df["val_loss"], label="val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 1].plot(history_df["epoch"], history_df["train_acc"], label="train")
    axes[0, 1].plot(history_df["epoch"], history_df["val_acc"], label="val")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].legend()
    axes[1, 0].plot(history_df["epoch"], history_df["val_loss"], color="tab:orange")
    axes[1, 0].set_title("Validation Loss")
    axes[1, 1].plot(history_df["epoch"], history_df["val_acc"], color="tab:green")
    axes[1, 1].set_title("Validation Accuracy")
    plt.suptitle(f"CNN Training Curves - {target}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"08_cnn_training_curves_{target}.png", dpi=300)
    plt.close()


def plot_model_comparison(target: str, target_results: dict) -> None:
    model_results = _model_results(target_results)
    if not model_results:
        return
    comparison_rows = [
        {"model": model_name, "metric": metric, "value": metrics[metric]}
        for model_name, metrics in model_results.items()
        for metric in config.METRICS
        if metric in metrics
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=comparison_df, x="metric", y="value", hue="model")
    baseline_accuracy = target_results.get("baseline_accuracy")
    if baseline_accuracy is not None:
        plt.axhline(baseline_accuracy, color="black", linestyle="--", linewidth=1.5,
                    label=f"majority-class baseline ({baseline_accuracy:.2f})")
    plt.ylim(0, 1)
    plt.title(f"Model Comparison vs. Baseline - {target}")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"model_comparison_{target}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_summary_infographic(results: dict) -> None:
    binary_results = results.get("binary", {})
    ensemble_binary = binary_results.get("ensemble", {})
    _, ax = plt.subplots(figsize=(11, 7))
    ax.axis("off")
    lines = [
        "DATANOVA 2026: Voice-Based Illness Detection Summary",
        "",
        "Dataset: Coswara",
        "",
        "Research question - is this person currently sick? (binary)",
        f"  Ensemble accuracy: {ensemble_binary.get('accuracy', float('nan')):.3f}  "
        f"(majority-class baseline: {binary_results.get('baseline_accuracy', float('nan')):.3f})",
        f"  Ensemble AUC: {ensemble_binary.get('auc', float('nan')):.3f}   AUPRC: {ensemble_binary.get('auprc', float('nan')):.3f}",
        f"  P(ensemble accuracy <= baseline) under bootstrap: {binary_results.get('bootstrap_p_ensemble_at_or_below_baseline', float('nan')):.4f}",
        "",
        "Recommendation: screening aid at best, not a diagnostic tool.",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=13, family="monospace")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "10_summary_infographic.png", dpi=300)
    plt.close()


def write_summaries(results: dict) -> None:
    binary_results = results.get("binary", {})
    ensemble_binary = binary_results.get("ensemble", {})
    p_value = binary_results.get("bootstrap_p_ensemble_at_or_below_baseline")
    beats_baseline = p_value is not None and p_value < config.SIGNIFICANCE_ALPHA

    summary = [
        "# RESULTS",
        "",
        "## Research Question",
        "Can AI predict whether someone is currently sick from their voice?",
        "",
        "## Answer",
        (
            f"Yes, better than naive guessing: the ensemble model correctly classifies sick vs. "
            f"not-sick {ensemble_binary.get('accuracy', float('nan')):.1%} of the time on held-out "
            f"participants, against a {binary_results.get('baseline_accuracy', float('nan')):.1%} majority-class "
            f"baseline. Under 2000 bootstrap resamples of the test set, the ensemble's accuracy stayed at "
            f"or below that baseline in only {p_value:.2%} of resamples."
            if beats_baseline
            else "The model did not convincingly beat the majority-class baseline on this run - see "
            "results.json for the exact numbers before drawing conclusions."
        ),
        "",
        "## Result: Sick vs. Not Sick (Binary)",
        f"- Ensemble accuracy: {ensemble_binary.get('accuracy', float('nan')):.3f} (baseline: {binary_results.get('baseline_accuracy', float('nan')):.3f})",
        f"- Ensemble AUC: {ensemble_binary.get('auc', float('nan')):.3f}",
        f"- Ensemble AUPRC: {ensemble_binary.get('auprc', float('nan')):.3f} (baseline prevalence: {binary_results.get('baseline_prevalence', float('nan')):.3f})",
        f"- Bootstrap P(ensemble accuracy <= baseline): {p_value if p_value is not None else float('nan'):.4f}",
        "",
        "## Per-Model Breakdown",
    ]
    for model_name, metrics in _model_results(binary_results).items():
        summary.append(f"- {model_name}: accuracy={metrics['accuracy']:.3f} auc={metrics['auc']:.3f} f1={metrics['f1']:.3f}")
    (ROOT / "RESULTS.md").write_text("\n".join(summary), encoding="utf-8")

    model_report = [
        "# 04 Modeling Report",
        "",
        "## Ensemble (random forest + gradient boosting + CNN + embedding, stacked)",
        f"- Binary AUC: {ensemble_binary.get('auc', float('nan')):.3f}",
    ]
    (REPORTS_DIR / "04_modeling_report.md").write_text("\n".join(model_report), encoding="utf-8")


def run_visualize(args) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    for target in config.TARGETS:
        plot_feature_importance(target)
        plot_cnn_training_curves(target)
        plot_model_comparison(target, results.get(target, {}))
    plot_summary_infographic(results)
    write_summaries(results)
    print("visualization and communication assets generated")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_ev = sub.add_parser("evaluate", help="score the ensemble on the test set")
    p_ev.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    p_ev.set_defaults(func=run_evaluate)
    sub.add_parser("visualize", help="render figures and RESULTS.md from results.json").set_defaults(func=run_visualize)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
