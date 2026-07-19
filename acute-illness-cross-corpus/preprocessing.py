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
\


import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

os.environ["PATH"] = str(Path(__file__).resolve().parent / "bin") + os.pathsep + os.environ.get("PATH", "")

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

import config

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
PROCESSED_DIR = ROOT / "processed_data"
DATA_DIR = ROOT / "data"
DATASET_INDEX_PATH = DATA_DIR / "dataset_index.csv"

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_RESAMPLERS: dict[tuple[int, int], torchaudio.transforms.Resample] = {}


def _resample_gpu(waveform: np.ndarray, original_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    resampler = _RESAMPLERS.get((original_sample_rate, target_sample_rate))
    if resampler is None:
        resampler = torchaudio.transforms.Resample(original_sample_rate, target_sample_rate).to(_DEVICE)
        _RESAMPLERS[(original_sample_rate, target_sample_rate)] = resampler
    with torch.no_grad():
        waveform_tensor = torch.from_numpy(waveform).float().to(_DEVICE)
        resampled_tensor = resampler(waveform_tensor)
    return resampled_tensor.cpu().numpy()


def _peak_normalize(waveform: np.ndarray) -> np.ndarray:
    peak_amplitude = np.max(np.abs(waveform))
    if peak_amplitude <= 0:
        return waveform
    return waveform / peak_amplitude * config.PEAK_NORMALIZE_TARGET


def _highest_energy_window(waveform: np.ndarray, target_samples: int) -> np.ndarray:
    """Slides a target_samples window across the waveform and returns the one with
    the most energy, instead of blindly grabbing the center. A blind center-crop on
    a much longer recording (e.g. Sound-Dr's ~23s sessions vs. our 5s target) risks
    landing in a silent gap between coughs or slicing straight through one -
    losing the actual signal, not just shifting it. Energy is a cheap, reliable
    proxy for "where the cough/breath/speech actually is" vs. background noise."""
    total_samples = len(waveform)
    hop_size = max(1, target_samples // 20)
    window_starts = list(range(0, total_samples - target_samples + 1, hop_size))
    if not window_starts:
        window_starts = [0]
    cumulative_energy = np.concatenate([[0.0], np.cumsum(waveform.astype(np.float64) ** 2)])
    window_energies = [
        cumulative_energy[start + target_samples] - cumulative_energy[start]
        for start in window_starts
    ]
    best_start = window_starts[int(np.argmax(window_energies))]
    return waveform[best_start : best_start + target_samples]


def _fix_duration(waveform: np.ndarray, target_samples: int) -> np.ndarray:
    if len(waveform) == target_samples:
        return waveform
    if len(waveform) < target_samples:
        padded_waveform = np.zeros(target_samples, dtype=np.float32)
        padded_waveform[: len(waveform)] = waveform
        return padded_waveform
    return _highest_energy_window(waveform, target_samples)


def preprocess_audio_file(
    path: Path, sample_rate: int, target_samples: int
) -> tuple[np.ndarray, dict[str, float]]:
    waveform, original_sample_rate = librosa.load(path, sr=None, mono=True)
    original_duration = len(waveform) / original_sample_rate if original_sample_rate else 0.0
    if original_sample_rate != sample_rate:
        waveform = _resample_gpu(waveform, original_sample_rate, sample_rate)
    trimmed_waveform, _ = librosa.effects.trim(waveform, top_db=config.TRIM_TOP_DB)
    normalized_waveform = _peak_normalize(trimmed_waveform.astype(np.float32))
    fixed_waveform = _fix_duration(normalized_waveform, target_samples)
    return fixed_waveform, {
        "original_sr": original_sample_rate,
        "standard_sr": sample_rate,
        "original_duration_sec": original_duration,
        "trimmed_duration_sec": len(trimmed_waveform) / sample_rate,
        "standard_duration_sec": len(fixed_waveform) / sample_rate,
        "is_all_zero": int(np.allclose(fixed_waveform, 0.0)),
        "has_nan": int(np.isnan(fixed_waveform).any()),
    }


def _load_any_metadata(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        try:
            raw_frame = pd.read_json(path)
            if isinstance(raw_frame, pd.Series):
                return pd.DataFrame([raw_frame.to_dict()])
            return raw_frame
        except ValueError:
            with path.open("r", encoding="utf-8") as file_handle:
                parsed_json = json.load(file_handle)
            return (
                pd.DataFrame([parsed_json])
                if isinstance(parsed_json, dict)
                else pd.DataFrame(parsed_json)
            )
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.DataFrame()


def _find_all_files(data_dir: Path) -> tuple[list[Path], list[Path]]:
    audio_paths, metadata_paths = [], []
    for path in data_dir.rglob("*"):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        if suffix in config.VALID_AUDIO_EXTENSIONS:
            audio_paths.append(path)
        elif suffix in config.METADATA_EXTENSIONS:
            metadata_paths.append(path)
    return audio_paths, metadata_paths


def _participant_id_from_path(path: Path) -> str | None:
    path_parts = [part for part in path.parts if part]
    for part in path_parts[::-1]:
        if len(part) >= config.MIN_PARTICIPANT_ID_LEN and any(char.isdigit() for char in part):
            return part
    return path.parent.name if path.parent else None


def _audio_type_from_filename(path: Path) -> str:
    filename = path.stem.lower()
    if "breath" in filename:
        return "breathing"
    if "cough" in filename:
        return "cough"
    if "count" in filename:
        return "counting"
    if "vowel" in filename or "ah" in filename or "ee" in filename:
        return "vowel"
    return "other"


def _normalize_colnames(metadata_df: pd.DataFrame) -> pd.DataFrame:
    metadata_df = metadata_df.copy()
    metadata_df.columns = [
        str(col).strip().lower().replace(" ", "_") for col in metadata_df.columns
    ]
    return metadata_df


def _resolve_bool(value: object) -> bool | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in config.TRUE_VALUES:
        return True
    if text in config.FALSE_VALUES:
        return False
    return None


def _infer_label(row: pd.Series) -> tuple[str | None, str | None]:
    """Harmonized 'currently sick' label (Option 1), matching the external
    datasets: symptomatic -> sick, healthy-and-asymptomatic -> not sick, and
    everything else (asymptomatic non-healthy: asymp positives, exposed,
    recovered, under-validation) -> excluded as ambiguous."""
    raw_status = str(row.get("covid_status", "")).strip().lower()
    has_symptom_signal = any(
        _resolve_bool(row.get(symptom)) is True for symptom in config.SICK_SYMPTOMS
    )
    if has_symptom_signal:
        return "infected", None
    if raw_status == config.HEALTHY_STATUS:
        return "not_infected", None
    return None, config.EXCLUSION_REASON_AMBIGUOUS_STATUS


def load_master_dataframe(data_dir: Path) -> pd.DataFrame:
    if not data_dir.exists():
        raise FileNotFoundError(
            f"raw data path not found: {data_dir}. place Coswara data there first."
        )
    audio_paths, metadata_paths = _find_all_files(data_dir)
    print(f"discovered {len(audio_paths)} audio files and {len(metadata_paths)} metadata files")

    metadata_frames = []
    for metadata_path in metadata_paths:
        try:
            metadata_frame = _load_any_metadata(metadata_path)
            if not metadata_frame.empty:
                metadata_frame["metadata_path"] = str(metadata_path)
                metadata_frames.append(metadata_frame)
        except Exception as error:
            print(f"warning: failed reading metadata file {metadata_path}: {error}")
    metadata_df = pd.concat(metadata_frames, ignore_index=True) if metadata_frames else pd.DataFrame()
    metadata_df = _normalize_colnames(metadata_df) if not metadata_df.empty else metadata_df

    audio_records = [
        {
            "participant_id": _participant_id_from_path(audio_path),
            "audio_path": str(audio_path),
            "audio_type": _audio_type_from_filename(audio_path),
            "recording_datetime": datetime.fromtimestamp(audio_path.stat().st_mtime),
            "file_size_bytes": audio_path.stat().st_size,
        }
        for audio_path in audio_paths
    ]
    audio_df = pd.DataFrame(audio_records)
    if audio_df.empty:
        raise RuntimeError("no audio files found")

    if not metadata_df.empty:
        if "participant_id" not in metadata_df.columns:
            id_candidate_columns = [col for col in metadata_df.columns if "id" in col]
            if id_candidate_columns:
                metadata_df["participant_id"] = metadata_df[id_candidate_columns[0]].astype(str)
        keep_columns = [col for col in config.METADATA_KEEP_COLUMNS if col in metadata_df.columns]
        metadata_df = metadata_df[keep_columns].drop_duplicates(subset=["participant_id"], keep="first")
        master_df = audio_df.merge(metadata_df, on="participant_id", how="left")
    else:
        master_df = audio_df.copy()

    if "covid_status" not in master_df.columns and "test_status" in master_df.columns:
        master_df["covid_status"] = master_df["test_status"]
    return master_df


def create_labels_and_clean(
    master_df: pd.DataFrame, audio_types: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    labeled_df = master_df.copy()
    labels_and_reasons = [_infer_label(row) for _, row in labeled_df.iterrows()]
    labeled_df["label_name"] = [label for label, _ in labels_and_reasons]
    labeled_df["exclusion_reason"] = [reason for _, reason in labels_and_reasons]

    for index, row in labeled_df.iterrows():
        if pd.isna(row.get("participant_id")):
            labeled_df.at[index, "exclusion_reason"] = config.EXCLUSION_REASON_MISSING_PARTICIPANT
        elif pd.isna(row.get("audio_path")):
            labeled_df.at[index, "exclusion_reason"] = config.EXCLUSION_REASON_MISSING_AUDIO

    excluded_df = labeled_df[labeled_df["exclusion_reason"].notna()].copy()
    clean_df = labeled_df[labeled_df["exclusion_reason"].isna()].copy()
    clean_df = clean_df[clean_df["audio_type"].isin(audio_types)].copy()

    class_names = sorted(clean_df["label_name"].dropna().unique().tolist())
    class_map = {name: class_index for class_index, name in enumerate(class_names)}
    clean_df["label"] = clean_df["label_name"].map(class_map)
    return clean_df, excluded_df, class_map


def _is_sick(label_name: str) -> int | None:
    if label_name in config.SICK_STATUS_LABELS:
        return 1
    if label_name in config.NOT_SICK_STATUS_LABELS:
        return 0
    return None


def generate_dataset_index(clean_df: pd.DataFrame) -> pd.DataFrame:
    index_rows = []
    for _, row in clean_df.iterrows():
        audio_path = Path(row["audio_path"])
        if not audio_path.exists():
            continue
        is_sick = _is_sick(row["label_name"])
        if is_sick is None:
            continue
        try:
            duration = librosa.get_duration(path=str(audio_path))
        except Exception:
            duration = None
        index_rows.append(
            {
                "participant_id": row["participant_id"],
                "label": int(row["label"]),
                "label_name": row["label_name"],
                "is_sick": is_sick,
                "audio_path": str(audio_path),
                "audio_type": row["audio_type"],
                "file_size_bytes": audio_path.stat().st_size,
                "duration_sec": duration,
                "age": row.get("age"),
                "gender": row.get("gender"),
            }
        )
    return pd.DataFrame(index_rows)


def write_data_report(master_df, excluded_df, dataset_index_df, class_map, raw_data_dir) -> None:
    participant_count = master_df["participant_id"].nunique(dropna=True)
    files_per_participant = master_df.groupby("participant_id")["audio_path"].count()
    class_counts = (
        dataset_index_df["label_name"].value_counts(dropna=False).to_dict()
        if not dataset_index_df.empty else {}
    )
    sick_counts = (
        dataset_index_df["is_sick"].value_counts(dropna=False).to_dict()
        if not dataset_index_df.empty else {}
    )
    exclusion_counts = Counter(
        excluded_df["exclusion_reason"].fillna(config.EXCLUSION_REASON_UNKNOWN).tolist()
    )
    report_lines = [
        "# 01 Data Report",
        "",
        f"- Total participants discovered: {participant_count}",
        f"- Total audio files discovered: {len(master_df)}",
        f"- Date range: {master_df['recording_datetime'].min()} to {master_df['recording_datetime'].max()}",
        f"- Average files per participant: {files_per_participant.mean() if len(files_per_participant) else 0:.2f}",
        "",
        "## Category Distribution",
    ]
    report_lines += [f"- {class_name}: {count}" for class_name, count in class_counts.items()]
    report_lines += [
        "",
        "## Sick vs. Not-Sick (Binary, Research Question)",
        f"- sick ({', '.join(sorted(config.SICK_STATUS_LABELS))}): {sick_counts.get(1, 0)}",
        f"- not sick ({', '.join(sorted(config.NOT_SICK_STATUS_LABELS))}): {sick_counts.get(0, 0)}",
    ]
    report_lines += ["", "## Label Mapping"] + [
        f"- {class_index}: {class_name}" for class_name, class_index in class_map.items()
    ]
    report_lines += ["", "## Exclusions"] + [
        f"- {reason}: {count}" for reason, count in exclusion_counts.items()
    ]
    report_lines += [
        "",
        "## Data Provenance",
        f"- Source: {config.PROVENANCE_SOURCE}",
        f"- Local dataset path: `{raw_data_dir}`",
        f"- Report generated at: {datetime.now(datetime.now().astimezone().tzinfo).isoformat()}",
        "",
        "## Known Limitations",
        "- Self-reported symptoms can introduce label noise.",
        "- Demographics and geography may be imbalanced.",
        "- Recordings can vary by device/environment quality.",
    ]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "01_data_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def run_index(args) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    master_df = load_master_dataframe(Path(args.raw_data_dir))
    clean_df, excluded_df, class_map = create_labels_and_clean(master_df, args.audio_types)
    dataset_index_df = generate_dataset_index(clean_df)

    dataset_index_df.to_csv(DATASET_INDEX_PATH, index=False)
    excluded_df[["participant_id", "audio_path", "exclusion_reason"]].to_csv(
        ARTIFACTS_DIR / "excluded_participants.csv", index=False
    )
    master_df.to_csv(ARTIFACTS_DIR / "master_dataframe_raw.csv", index=False)
    (ARTIFACTS_DIR / "class_map.json").write_text(json.dumps(class_map, indent=2), encoding="utf-8")
    write_data_report(master_df, excluded_df, dataset_index_df, class_map, Path(args.raw_data_dir))
    print(
        f"data loading complete. participants={master_df['participant_id'].nunique(dropna=True)} "
        f"files={len(master_df)} usable={len(dataset_index_df)} classes={len(class_map)}"
    )


def _process_row(row: dict, sample_rate: int, target_samples: int, processed_audio_dir: str):
    input_path = Path(row["audio_path"])
    if not input_path.exists():
        return {"audio_path": str(input_path), "status": "missing_file"}, None
    try:
        fixed_waveform, stats = preprocess_audio_file(input_path, sample_rate, target_samples)
        if stats["is_all_zero"] or stats["has_nan"]:
            return {"audio_path": str(input_path), "status": "failed_qc", **stats}, None
        output_path = Path(processed_audio_dir) / f"{row['participant_id']}_{input_path.stem}.wav"
        sf.write(output_path, fixed_waveform, sample_rate)
        return (
            {"audio_path": str(input_path), "status": "ok", **stats},
            {**row, "processed_audio_path": str(output_path), **stats},
        )
    except Exception as error:
        return {"audio_path": str(input_path), "status": f"error:{error}"}, None


def run_coswara(args) -> None:
    target_samples = args.sample_rate * args.target_duration_sec
    print(f"resampling device: {_DEVICE}, workers: {args.workers}")
    index_df = pd.read_csv(DATASET_INDEX_PATH)
    processed_audio_dir = PROCESSED_DIR / "audio"
    processed_audio_dir.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    log_entries, processed_rows = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(_process_row, row, args.sample_rate, target_samples, str(processed_audio_dir))
            for row in index_df.to_dict("records")
        ]
        for completed_count, future in enumerate(as_completed(futures), 1):
            log_entry, row_entry = future.result()
            log_entries.append(log_entry)
            if row_entry is not None:
                processed_rows.append(row_entry)
            if completed_count % 500 == 0 or completed_count == len(futures):
                print(f"progress: {completed_count}/{len(futures)}")

    processed_df = pd.DataFrame(processed_rows)
    processed_df.to_csv(ARTIFACTS_DIR / "preprocessed_index.csv", index=False)
    pd.DataFrame(log_entries).to_csv(ARTIFACTS_DIR / "preprocessing_log.csv", index=False)
    (REPORTS_DIR / "02_preprocessing_report.md").write_text(
        "\n".join(
            [
                "# 02 Preprocessing Summary",
                "",
                f"- Files attempted: {len(index_df)}",
                f"- Files successfully preprocessed: {len(processed_df)}",
                f"- Files discarded: {len(index_df) - len(processed_df)}",
                f"- Target sample rate: {args.sample_rate}",
                f"- Target duration (seconds): {args.target_duration_sec}",
            ]
        ),
        encoding="utf-8",
    )
    print(f"preprocessing complete. usable={len(processed_df)}/{len(index_df)}")


def _process_external_one(row: dict, out_audio_dir: str, out_spec_dir: str) -> dict | None:
    from features import extract_features_and_spectrogram

    sample_rate = config.DEFAULT_SAMPLE_RATE
    target_samples = sample_rate * config.DEFAULT_TARGET_DURATION_SEC
    try:
        fixed_waveform, stats = preprocess_audio_file(row["raw_audio_path"], sample_rate, target_samples)
        if stats["is_all_zero"] or stats["has_nan"]:
            return None
        output_wav_path = Path(out_audio_dir) / f"{row['participant_id']}.wav"
        sf.write(output_wav_path, fixed_waveform, sample_rate)
        features, spectrogram = extract_features_and_spectrogram(
            output_wav_path, sample_rate, config.DEFAULT_N_MELS, config.DEFAULT_N_FFT,
            config.DEFAULT_HOP_LENGTH, config.DEFAULT_TARGET_SPEC_FRAMES, "cough",
        )
        spectrogram_path = Path(out_spec_dir) / f"{row['participant_id']}.npy"
        np.save(spectrogram_path, spectrogram)
        return {
            "participant_id": row["participant_id"],
            "is_sick": int(row["is_sick"]),
            "audio_type": "cough",
            "processed_audio_path": str(output_wav_path),
            "spectrogram_path": str(spectrogram_path),
            "source_dataset": row["source_dataset"],
            **features,
        }
    except Exception as error:
        print(f"warning: failed for {row['participant_id']}: {error}")
        return None


def run_external(dataset_name: str, rows_df: pd.DataFrame, workers: int | None = None) -> pd.DataFrame:
    out_audio_dir = PROCESSED_DIR / f"{dataset_name}_audio"
    out_spec_dir = PROCESSED_DIR / f"{dataset_name}_spectrograms"
    out_audio_dir.mkdir(parents=True, exist_ok=True)
    out_spec_dir.mkdir(parents=True, exist_ok=True)

    results = []
    workers = workers or (os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_process_external_one, row, str(out_audio_dir), str(out_spec_dir))
            for row in rows_df.to_dict("records")
        ]
        for completed_count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result is not None:
                results.append(result)
            if completed_count % 500 == 0 or completed_count == len(futures):
                print(f"[{dataset_name}] progress: {completed_count}/{len(futures)}")

    features_df = pd.DataFrame(results)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(DATA_DIR / f"{dataset_name}_features_extracted.csv", index=False)
    print(f"[{dataset_name}] features done: {features_df.shape}")

    from features import extract_dataset_embeddings

    extract_dataset_embeddings(dataset_name, features_df)
    return features_df


def _coughvid_rows() -> pd.DataFrame:
    coughvid_dir = ROOT / "external_datasets" / "COUGHVID" / "extracted" / "coughvid_20211012"
    metadata_df = pd.read_csv(coughvid_dir / "metadata_compiled.csv").dropna(subset=["status"]).copy()
    metadata_df["is_sick"] = metadata_df["status"].map({"healthy": 0, "symptomatic": 1, "COVID-19": 1})
    metadata_df = metadata_df.dropna(subset=["is_sick"])
    rows, missing_count = [], 0
    for _, row in metadata_df.iterrows():
        audio_path = next(
            (coughvid_dir / f"{row['uuid']}{ext}" for ext in (".wav", ".webm", ".ogg")
             if (coughvid_dir / f"{row['uuid']}{ext}").exists()),
            None,
        )
        if audio_path is None:
            missing_count += 1
            continue
        rows.append({"participant_id": row["uuid"], "raw_audio_path": str(audio_path),
                     "is_sick": int(row["is_sick"]), "source_dataset": "coughvid"})
    if missing_count:
        print(f"warning: {missing_count} COUGHVID rows had no audio file, skipped")
    return pd.DataFrame(rows)


def _sounddr_rows() -> pd.DataFrame:
    sounddr_dir = ROOT / "external_datasets" / "Sound-Dr" / "sounddr_data" / "extracted"


    metadata_df = pd.read_csv(sounddr_dir.parent / "data.csv").dropna(subset=["label_symptom"]).copy()
    rows, missing_count = [], 0
    for row_index, row in metadata_df.iterrows():

        audio_path = sounddr_dir / str(row["file_path"]).replace(":", "_")
        if not audio_path.exists():
            missing_count += 1
            continue
        rows.append({"participant_id": f"sounddr_{row_index}", "raw_audio_path": str(audio_path),
                     "is_sick": int(row["label_symptom"]), "source_dataset": "sounddr"})
    if missing_count:
        print(f"warning: {missing_count} Sound-Dr rows had no audio file, skipped")
    return pd.DataFrame(rows)


def _balanced_subsample(rows_df: pd.DataFrame, max_clips: int, seed: int) -> pd.DataFrame:
    """Cap the number of clips at max_clips, drawn class-balanced on is_sick so the
    cross-dataset test set stays representative. Returns rows_df unchanged if it is
    already at or under the cap (or if max_clips <= 0, meaning 'no cap')."""
    if max_clips <= 0 or len(rows_df) <= max_clips:
        return rows_df
    per_class = max_clips // 2
    sampled = []
    for label, group in rows_df.groupby("is_sick"):
        take = min(per_class, len(group))
        sampled.append(group.sample(n=take, random_state=seed))
    result = pd.concat(sampled).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(f"subsampled to {len(result)} clips (balanced): "
          f"sick={int(result['is_sick'].sum())} healthy={int((result['is_sick'] == 0).sum())}")
    return result


def run_coughvid(args) -> None:
    rows = _coughvid_rows()
    print(f"coughvid rows: {len(rows)}  (sick={rows['is_sick'].sum()})")
    rows = _balanced_subsample(rows, getattr(args, "max_clips", 0), config.RANDOM_SEED)
    run_external("coughvid", rows)


def run_sounddr(args) -> None:
    rows = _sounddr_rows()
    print(f"sounddr rows: {len(rows)}  (sick={rows['is_sick'].sum()})")
    run_external("sounddr", rows)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="scan raw Coswara -> dataset_index.csv")
    p_index.add_argument("--raw-data-dir", type=str, default=str(ROOT / config.DEFAULT_RAW_DATA_DIRNAME))
    p_index.add_argument("--audio-types", type=str, nargs="+", default=config.DEFAULT_AUDIO_TYPES)
    p_index.set_defaults(func=run_index)

    p_cos = sub.add_parser("coswara", help="preprocess Coswara clips")
    p_cos.add_argument("--sample-rate", type=int, default=config.DEFAULT_SAMPLE_RATE)
    p_cos.add_argument("--target-duration-sec", type=int, default=config.DEFAULT_TARGET_DURATION_SEC)
    p_cos.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p_cos.set_defaults(func=run_coswara)

    p_cvid = sub.add_parser("coughvid", help="preprocess+features+embed COUGHVID")
    p_cvid.add_argument("--max-clips", type=int, default=6000,
                        help="cap clips (class-balanced) to speed up; 0 = no cap / use all")
    p_cvid.set_defaults(func=run_coughvid)
    sub.add_parser("sounddr", help="preprocess+features+embed Sound-Dr").set_defaults(func=run_sounddr)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
