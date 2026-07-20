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
\


import argparse
import math
import os
import time
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import config

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ROOT / "models"
SPLIT_DATA_PATH = ARTIFACTS_DIR / "split_data.joblib"


class ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(inputs) if self.shortcut is not None else inputs
        output = self.relu(self.bn1(self.conv1(inputs)))
        output = self.bn2(self.conv2(output))
        return self.relu(output + identity)


class CNNClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = ResidualBlock(32, 64)
        self.stage2 = ResidualBlock(64, 128)
        self.stage3 = ResidualBlock(128, 256)
        self.stage4 = ResidualBlock(256, 256)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(64, num_classes),
        )

    def forward(self, spectrogram_batch: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(spectrogram_batch)
        hidden = self.stage1(hidden)
        hidden = self.stage2(hidden)
        hidden = self.stage3(hidden)
        hidden = self.stage4(hidden)
        return self.classifier(hidden)


class Wav2Vec2Classifier(nn.Module):

    def __init__(self, num_classes: int, unfreeze_layers: int, dropout: float):
        super().__init__()
        bundle = getattr(torchaudio.pipelines, config.EMBEDDING_MODEL)
        self.backbone = bundle.get_model()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        transformer_layers = self.backbone.encoder.transformer.layers
        for layer in transformer_layers[len(transformer_layers) - unfreeze_layers :]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        self.head = nn.Sequential(
            nn.Linear(config.EMBEDDING_DIM, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.backbone.extract_features(waveforms)
        pooled = hidden_states[-1].mean(dim=1)
        return self.head(pooled)


class SpectrogramDataset(Dataset):

    def __init__(self, clips_df: pd.DataFrame, label_col: str = "is_sick", augment: bool = False):
        self.clips_df = clips_df.reset_index(drop=True)
        self.label_col = label_col
        self.augment = augment

    def __len__(self) -> int:
        return len(self.clips_df)

    def _augment_spec(self, spectrogram: np.ndarray) -> np.ndarray:
        if not self.augment:
            return spectrogram
        augmented = spectrogram.copy()
        time_shift = np.random.randint(config.AUG_TIME_SHIFT_MIN, config.AUG_TIME_SHIFT_MAX + 1)
        augmented = np.roll(augmented, shift=time_shift, axis=1)
        freq_mask_start = np.random.randint(0, config.AUG_FREQ_MASK_START_MAX + 1)
        freq_mask_width = np.random.randint(config.AUG_FREQ_MASK_WIDTH_MIN, config.AUG_FREQ_MASK_WIDTH_MAX + 1)
        augmented[freq_mask_start : min(augmented.shape[0], freq_mask_start + freq_mask_width), :] = 0
        time_mask_start = np.random.randint(0, config.AUG_TIME_MASK_START_MAX + 1)
        time_mask_width = np.random.randint(config.AUG_TIME_MASK_WIDTH_MIN, config.AUG_TIME_MASK_WIDTH_MAX + 1)
        augmented[:, time_mask_start : min(augmented.shape[1], time_mask_start + time_mask_width)] = 0
        augmented += np.random.normal(0, config.AUG_NOISE_STD, size=augmented.shape).astype(np.float32)
        return augmented

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.clips_df.iloc[idx]
        spectrogram = np.load(row["spectrogram_path"]).astype(np.float32)
        spectrogram = self._augment_spec(spectrogram)
        spectrogram_min, spectrogram_max = np.min(spectrogram), np.max(spectrogram)
        spectrogram = (spectrogram - spectrogram_min) / (spectrogram_max - spectrogram_min + config.SPEC_NORM_EPS)
        spectrogram_tensor = torch.tensor(spectrogram[None, :, :], dtype=torch.float32)
        label_tensor = torch.tensor(int(row[self.label_col]), dtype=torch.long)
        return spectrogram_tensor, label_tensor


class WaveformDataset(Dataset):

    def __init__(self, clips_df: pd.DataFrame, label_col: str, augment: bool = False):
        self.clips_df = clips_df.reset_index(drop=True)
        self.label_col = label_col
        self.augment = augment
        self.sample_rate = config.EMBEDDING_SAMPLE_RATE
        self.max_samples = config.FINETUNE_MAX_SAMPLES

    def __len__(self) -> int:
        return len(self.clips_df)

    def _fix_len(self, waveform: np.ndarray) -> np.ndarray:
        if len(waveform) >= self.max_samples:
            return waveform[: self.max_samples]
        padded = np.zeros(self.max_samples, dtype=np.float32)
        padded[: len(waveform)] = waveform
        return padded

    def __getitem__(self, idx: int):
        row = self.clips_df.iloc[idx]
        waveform, _ = librosa.load(row["processed_audio_path"], sr=self.sample_rate, mono=True)
        if waveform.size == 0:
            waveform = np.zeros(self.sample_rate, dtype=np.float32)
        if self.augment:
            waveform = waveform + np.random.normal(0, 0.005, size=waveform.shape).astype(np.float32)
        waveform = self._fix_len(waveform.astype(np.float32))
        return torch.from_numpy(waveform), torch.tensor(int(row[self.label_col]), dtype=torch.long)


class FocalLoss(nn.Module):

    def __init__(self, gamma: float, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=1)
        cross_entropy = F.nll_loss(log_probs, target, weight=self.weight, reduction="none")
        prob_of_true_class = torch.exp(log_probs).gather(1, target.unsqueeze(1)).squeeze(1)
        return (((1 - prob_of_true_class) ** self.gamma) * cross_entropy).mean()


def _tabular_estimator_and_grid(model_name: str, seed: int, balancing: str):
    if model_name == "random_forest":
        param_grid = dict(config.RF_PARAM_DISTRIBUTIONS)
        if balancing == "class_weight":
            param_grid = {**param_grid, "class_weight": ["balanced", "balanced_subsample"]}
        return RandomForestClassifier(random_state=seed, n_jobs=-1), param_grid, config.RF_CV_SPLITS, config.RF_RANDOM_SEARCH_ITER
    if model_name == "gradient_boosting":
        class_weight = "balanced" if balancing == "class_weight" else None
        return (
            HistGradientBoostingClassifier(random_state=seed, class_weight=class_weight),
            config.GBM_PARAM_DISTRIBUTIONS, config.GBM_CV_SPLITS, config.GBM_RANDOM_SEARCH_ITER,
        )
    raise ValueError(f"unknown tabular model: {model_name}")


def train_tabular_model(model_name: str, target: str, seed: int, balancing: str) -> dict:
    bundle = joblib.load(SPLIT_DATA_PATH)
    if balancing == "smote":
        train_features, train_labels = bundle[f"X_train_bal_{target}"], bundle[f"y_train_bal_{target}"]
    else:
        train_features, train_labels = bundle["X_train"], bundle[f"y_train_{target}"]

    estimator, param_distributions, cv_splits, search_iterations = _tabular_estimator_and_grid(model_name, seed, balancing)
    cv_splitter = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        estimator, param_distributions=param_distributions, n_iter=search_iterations, scoring=config.RF_SCORING,
        cv=cv_splitter, n_jobs=-1, random_state=seed, verbose=1,
    )
    start_time = time.time()
    search.fit(train_features, train_labels)
    print(f"{model_name} ({target}, {balancing}): cv_{config.RF_SCORING}={search.best_score_:.4f} train_time={time.time()-start_time:.1f}s")

    calibrated_model = CalibratedClassifierCV(search.best_estimator_, method=config.CALIBRATION_METHOD, cv=config.CALIBRATION_CV_FOLDS)
    calibrated_model.fit(train_features, train_labels)
    joblib.dump(calibrated_model, MODELS_DIR / f"model_{model_name}_{target}.pkl")

    if model_name == "random_forest":
        pd.DataFrame(
            {"feature": bundle["feature_cols"], "importance": search.best_estimator_.feature_importances_}
        ).sort_values("importance", ascending=False).to_csv(
            ARTIFACTS_DIR / f"rf_feature_importance_{target}.csv", index=False
        )
    return {"model": model_name, "target": target, "cv_score": float(search.best_score_)}


def train_embedding_model(target: str, seed: int, emb_key: str = "emb", model_name: str = "embedding") -> dict | None:
    bundle = joblib.load(SPLIT_DATA_PATH)
    if f"{emb_key}_train" not in bundle:
        print(f"skipping {model_name} model ({target}): no '{emb_key}' in split bundle")
        return None
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=5000, C=1.0, random_state=seed),
    )
    start_time = time.time()
    model.fit(bundle[f"{emb_key}_train"], bundle[f"y_train_{target}"])
    print(f"{model_name} ({target}): train_time={time.time() - start_time:.1f}s")
    joblib.dump(model, MODELS_DIR / f"model_{model_name}_{target}.pkl")
    return {"model": model_name, "target": target}


def _maybe_smote(features, labels, balancing: str, min_class_count: int, seed: int):
    if balancing == "smote" and min_class_count >= 2:
        k_neighbors = int(min(5, min_class_count - 1))
        try:
            return SMOTE(random_state=seed, k_neighbors=k_neighbors).fit_resample(features, labels)
        except ValueError:
            return features, labels
    return features, labels


def train_tabular_per_type(model_name: str, target: str, seed: int, balancing: str):
    bundle = joblib.load(SPLIT_DATA_PATH)
    train_df = bundle["train_df"].reset_index(drop=True)
    all_train_features = np.asarray(bundle["X_train"])
    all_train_labels = np.asarray(bundle[f"y_train_{target}"])
    audio_types = train_df["audio_type"].to_numpy()
    use_balanced_rf = getattr(config, "USE_BALANCED_RF", False)

    for audio_type in sorted(set(audio_types)):
        type_mask = audio_types == audio_type
        type_features, type_labels = all_train_features[type_mask], all_train_labels[type_mask]
        class_counts = np.bincount(type_labels)
        min_class_count = class_counts[class_counts > 0].min() if class_counts.size else 0
        class_weight = "balanced" if balancing == "class_weight" else None

        if model_name == "random_forest" and use_balanced_rf:
            balanced_features, balanced_labels = type_features, type_labels
            estimator = BalancedRandomForestClassifier(
                n_estimators=600, min_samples_leaf=2, sampling_strategy="all",
                replacement=True, bootstrap=False, random_state=seed, n_jobs=-1,
            )
        elif model_name == "random_forest":
            balanced_features, balanced_labels = _maybe_smote(type_features, type_labels, balancing, min_class_count, seed)
            estimator = RandomForestClassifier(
                n_estimators=600, min_samples_leaf=2, class_weight=class_weight, random_state=seed, n_jobs=-1,
            )
        else:
            balanced_features, balanced_labels = _maybe_smote(type_features, type_labels, balancing, min_class_count, seed)
            estimator = HistGradientBoostingClassifier(random_state=seed, class_weight=class_weight)

        balanced_label_counts = np.bincount(balanced_labels)
        calibration_cv = int(min(3, balanced_label_counts[balanced_label_counts > 0].min())) if balanced_label_counts.size else 2
        start_time = time.time()
        model = (
            CalibratedClassifierCV(estimator, method=config.CALIBRATION_METHOD, cv=calibration_cv)
            if calibration_cv >= 2 else estimator
        )
        model.fit(balanced_features, balanced_labels)
        joblib.dump(model, MODELS_DIR / f"model_{model_name}_{target}_{audio_type}.pkl")
        print(
            f"{model_name} ({target}/{audio_type}, {balancing}): n={len(balanced_labels)} "
            f"classes={len(balanced_label_counts[balanced_label_counts>0])} train_time={time.time() - start_time:.1f}s"
        )
    return {"model": model_name, "target": target, "per_type": True}


def _run_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train() if is_training else model.eval()
    total_loss, sample_count, correct_count = 0.0, 0, 0
    for batch_inputs, batch_labels in loader:
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
        if is_training:
            optimizer.zero_grad()
        logits = model(batch_inputs)
        loss = criterion(logits, batch_labels)
        if is_training:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * len(batch_labels)
        correct_count += int((torch.argmax(logits, dim=1) == batch_labels).sum().item())
        sample_count += len(batch_labels)
    return total_loss / max(sample_count, 1), correct_count / max(sample_count, 1)


def _warmup_cosine_lr(epoch: int, epochs: int) -> float:
    if epoch < config.CNN_LR_WARMUP_EPOCHS:
        return (epoch + 1) / config.CNN_LR_WARMUP_EPOCHS
    progress = (epoch - config.CNN_LR_WARMUP_EPOCHS) / max(1, epochs - config.CNN_LR_WARMUP_EPOCHS)
    return 0.5 * (1 + math.cos(math.pi * progress))


def train_cnn(
    target: str, epochs: int, batch_size: int, learning_rate: float, weight_decay: float,
    early_stopping_patience: int, seed: int, dropout: float = config.DEFAULT_CNN_DROPOUT,
    label_smoothing: float = config.DEFAULT_CNN_LABEL_SMOOTHING,
    model_path: Path | None = None, tag: str = "",
) -> dict:
    label_col = config.TARGET_LABEL_COLUMNS[target]
    bundle = joblib.load(SPLIT_DATA_PATH)
    train_df = bundle["train_df"][["spectrogram_path", label_col]].dropna()
    val_df = bundle["val_df"][["spectrogram_path", label_col]].dropna()
    num_classes = int(np.max(bundle[f"y_train_{target}"])) + 1
    if model_path is None:
        model_path = MODELS_DIR / f"model_cnn_{target}.pth"

    train_labels = train_df[label_col].values
    use_sampler = getattr(config, "CNN_USE_BALANCED_SAMPLER", False)
    if use_sampler:
        class_counts = np.bincount(train_labels, minlength=num_classes)
        sample_weights = 1.0 / np.maximum(class_counts[train_labels], 1)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(
            SpectrogramDataset(train_df, label_col=label_col, augment=True),
            batch_size=batch_size, sampler=sampler, num_workers=2,
        )
    else:
        train_loader = DataLoader(
            SpectrogramDataset(train_df, label_col=label_col, augment=True),
            batch_size=batch_size, shuffle=True, num_workers=2,
        )
    val_loader = DataLoader(
        SpectrogramDataset(val_df, label_col=label_col, augment=False),
        batch_size=batch_size, shuffle=False, num_workers=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNClassifier(num_classes=num_classes, dropout=dropout).to(device)

    if use_sampler:
        class_weight_tensor = None
    else:
        class_weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=train_labels)
        if num_classes > 2:
            class_weights = np.sqrt(class_weights)
        class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    if getattr(config, "CNN_USE_FOCAL_LOSS", False):
        criterion = FocalLoss(gamma=config.CNN_FOCAL_GAMMA, weight=class_weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weight_tensor, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: _warmup_cosine_lr(epoch, epochs))

    log_prefix = f"cnn ({target}{'/' + tag if tag else ''})"
    best_val_loss, epochs_without_improvement, best_epoch, history = float("inf"), 0, -1, []
    for epoch in range(epochs):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, None, device)
        scheduler.step()
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "train_acc": train_acc, "val_acc": val_acc})
        print(
            f"{log_prefix} epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        if val_loss < best_val_loss:
            best_val_loss, epochs_without_improvement, best_epoch = val_loss, 0, epoch + 1
            torch.save(
                {"state_dict": model.state_dict(), "num_classes": num_classes, "target": target,
                 "label_col": label_col, "dropout": dropout},
                model_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"{log_prefix} early stopping at epoch {epoch + 1}")
                break

    pd.DataFrame(history).to_csv(ARTIFACTS_DIR / f"cnn_training_history_{target}.csv", index=False)
    return {"target": target, "best_epoch": best_epoch, "best_val_loss": best_val_loss, "model_path": str(model_path)}


def search_cnn(target: str, epochs: int, batch_size: int, early_stopping_patience: int, seed: int, n_trials: int) -> dict:
    rng = np.random.default_rng(seed)
    search_space = config.CNN_SEARCH_SPACE
    final_path = MODELS_DIR / f"model_cnn_{target}.pth"
    best_trial = {"best_val_loss": float("inf")}
    for trial in range(n_trials):
        trial_params = {name: rng.choice(values).item() for name, values in search_space.items()}
        print(f"cnn search ({target}) trial {trial + 1}/{n_trials}: {trial_params}")
        trial_path = MODELS_DIR / f"model_cnn_{target}_trial{trial}.pth"
        result = train_cnn(
            target, epochs, batch_size, trial_params["learning_rate"], trial_params["weight_decay"],
            early_stopping_patience, seed, dropout=trial_params["dropout"],
            label_smoothing=trial_params["label_smoothing"], model_path=trial_path, tag=f"t{trial}",
        )
        if result["best_val_loss"] < best_trial["best_val_loss"]:
            best_trial = {**result, "params": trial_params, "trial_path": str(trial_path)}

    for trial in range(n_trials):
        trial_path = MODELS_DIR / f"model_cnn_{target}_trial{trial}.pth"
        if str(trial_path) == best_trial.get("trial_path"):
            os.replace(trial_path, final_path)
        elif trial_path.exists():
            trial_path.unlink()
    print(f"cnn search ({target}) best: val_loss={best_trial['best_val_loss']:.4f} params={best_trial.get('params')}")
    return best_trial


def _run_epoch_ft(model, loader, criterion, optimizer, device, scaler):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()
    total_loss, sample_count, correct_count = 0.0, 0, 0
    for batch_inputs, batch_labels in loader:
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
        if is_training:
            optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler is not None):
            logits = model(batch_inputs)
            loss = criterion(logits, batch_labels)
        if is_training:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * len(batch_labels)
        correct_count += int((torch.argmax(logits, dim=1) == batch_labels).sum().item())
        sample_count += len(batch_labels)
    return total_loss / max(sample_count, 1), correct_count / max(sample_count, 1)


def finetune_target(target: str, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    label_col = config.TARGET_LABEL_COLUMNS[target]
    bundle = joblib.load(SPLIT_DATA_PATH)
    train_df = bundle["train_df"][["processed_audio_path", label_col]].dropna()
    val_df = bundle["val_df"][["processed_audio_path", label_col]].dropna()
    num_classes = int(np.max(bundle[f"y_train_{target}"])) + 1

    train_loader = DataLoader(WaveformDataset(train_df, label_col, augment=True), batch_size=config.FINETUNE_BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(WaveformDataset(val_df, label_col, augment=False), batch_size=config.FINETUNE_BATCH_SIZE, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"finetune ({target}) device={device} classes={num_classes}")
    model = Wav2Vec2Classifier(num_classes, config.FINETUNE_UNFREEZE_LAYERS, config.FINETUNE_HEAD_DROPOUT).to(device)

    class_weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=train_df[label_col].values)
    if num_classes > 2:
        class_weights = np.sqrt(class_weights)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    trainable_backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": trainable_backbone_params, "lr": config.FINETUNE_BACKBONE_LR},
         {"params": model.head.parameters(), "lr": config.FINETUNE_HEAD_LR}],
        weight_decay=config.FINETUNE_WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None

    model_path = MODELS_DIR / f"model_finetune_{target}.pth"
    best_val_loss, epochs_without_improvement, best_epoch = float("inf"), 0, -1
    for epoch in range(config.FINETUNE_EPOCHS):
        train_loss, train_acc = _run_epoch_ft(model, train_loader, criterion, optimizer, device, scaler)
        with torch.no_grad():
            val_loss, val_acc = _run_epoch_ft(model, val_loader, criterion, None, device, None)
        print(f"finetune ({target}) epoch {epoch + 1}/{config.FINETUNE_EPOCHS}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
        if val_loss < best_val_loss:
            best_val_loss, epochs_without_improvement, best_epoch = val_loss, 0, epoch + 1
            torch.save(
                {"state_dict": model.state_dict(), "num_classes": num_classes, "target": target,
                 "label_col": label_col, "unfreeze_layers": config.FINETUNE_UNFREEZE_LAYERS,
                 "dropout": config.FINETUNE_HEAD_DROPOUT},
                model_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.FINETUNE_EARLY_STOPPING_PATIENCE:
                print(f"finetune ({target}) early stopping at epoch {epoch + 1}")
                break
    return {"target": target, "best_epoch": best_epoch, "best_val_loss": best_val_loss}


def run_train(args) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    targets = config.TARGETS if args.target == "both" else [args.target]


    all_models = sorted({m for t in config.TARGETS for m in config.TARGET_ENSEMBLE.get(t, [])})
    models = (
        all_models
        if args.model == "all"
        else {"rf": ["random_forest"], "gbm": ["gradient_boosting"], "cnn": ["cnn"], "emb": ["embedding"]}[args.model]
    )
    for target in targets:
        balancing = args.balancing or config.TARGET_BALANCING[target]
        train_tabular = train_tabular_per_type if config.PER_AUDIO_TYPE_TABULAR else train_tabular_model
        if "random_forest" in models:
            train_tabular("random_forest", target, args.seed, balancing)
        if "gradient_boosting" in models:
            train_tabular("gradient_boosting", target, args.seed, balancing)
        if "embedding" in models:
            train_embedding_model(target, args.seed, "emb", "embedding")
        if "cnn" in models:
            if args.cnn_search:
                search_cnn(target, config.CNN_SEARCH_EPOCHS, args.batch_size, args.early_stopping_patience, args.seed, args.cnn_trials)
            else:
                train_cnn(
                    target, args.epochs, args.batch_size, args.lr, args.weight_decay,
                    args.early_stopping_patience, args.seed, dropout=args.dropout, label_smoothing=args.label_smoothing,
                )


def run_finetune(args) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    targets = config.TARGETS if args.target == "both" else [args.target]
    for target in targets:
        finetune_target(target, args.seed)
    print("fine-tuning complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_tr = sub.add_parser("train", help="train ensemble base models (RF/GBM/embedding/CNN)")
    p_tr.add_argument("--target", choices=[*config.TARGETS, "both"], default="both")
    p_tr.add_argument("--model", choices=["rf", "gbm", "cnn", "emb", "all"], default="all")
    p_tr.add_argument("--balancing", choices=["smote", "class_weight"], default=None,
                      help="override the per-target balancing in config.TARGET_BALANCING")
    p_tr.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    p_tr.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    p_tr.add_argument("--lr", type=float, default=config.DEFAULT_LEARNING_RATE)
    p_tr.add_argument("--weight-decay", type=float, default=config.DEFAULT_WEIGHT_DECAY)
    p_tr.add_argument("--early-stopping-patience", type=int, default=config.DEFAULT_EARLY_STOPPING_PATIENCE)
    p_tr.add_argument("--dropout", type=float, default=config.DEFAULT_CNN_DROPOUT)
    p_tr.add_argument("--label-smoothing", type=float, default=config.DEFAULT_CNN_LABEL_SMOOTHING)
    p_tr.add_argument("--cnn-search", action="store_true", help="randomized CNN hyperparameter search")
    p_tr.add_argument("--cnn-trials", type=int, default=config.CNN_SEARCH_TRIALS)
    p_tr.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p_tr.set_defaults(func=run_train)

    p_ft = sub.add_parser("finetune", help="fine-tune wav2vec2 end-to-end")
    p_ft.add_argument("--target", choices=[*config.TARGETS, "both"], default="both")
    p_ft.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p_ft.set_defaults(func=run_finetune)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
