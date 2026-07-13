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
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from scipy.signal import hilbert

import config

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
PROCESSED_DIR = ROOT / "processed_data"
DATA_DIR = ROOT / "data"
FEATURES_PATH = DATA_DIR / "features_extracted.csv"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"


def _stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.nanmean(values)),
        f"{prefix}_std": float(np.nanstd(values)),
        f"{prefix}_min": float(np.nanmin(values)),
        f"{prefix}_max": float(np.nanmax(values)),
    }


def _safe_voiced_f0(waveform: np.ndarray) -> np.ndarray:
    pitch_hz, _, _ = librosa.pyin(
        waveform,
        fmin=librosa.note_to_hz(config.F0_MIN_NOTE),
        fmax=librosa.note_to_hz(config.F0_MAX_NOTE),
    )
    return pitch_hz[~np.isnan(pitch_hz)] if pitch_hz is not None else np.array([])


def _fix_spec_frames(spectrogram_db: np.ndarray, target_frames: int) -> np.ndarray:
    current_frames = spectrogram_db.shape[1]
    if current_frames == target_frames:
        return spectrogram_db
    if current_frames < target_frames:
        padding = np.zeros(
            (spectrogram_db.shape[0], target_frames - current_frames), dtype=np.float32
        )
        return np.concatenate([spectrogram_db, padding], axis=1)
    return spectrogram_db[:, :target_frames]


def extract_features_and_spectrogram(
    path: Path,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    target_spec_frames: int,
    audio_type: str,
) -> tuple[dict[str, float], np.ndarray]:
    """Loads the audio once and shares a single STFT/mel-spectrogram across every
    feature that needs one (mfcc, spectral stats, chroma, onset, the CNN
    spectrogram itself) instead of each librosa call silently recomputing its
    own FFT from scratch. Pitch tracking (pyin) only runs for vowel clips -
    cough/breathing aren't pitched, so running it there mostly measured noise.
    """
    waveform, sample_rate = librosa.load(path, sr=sample_rate, mono=True)
    features: dict[str, float] = {}

    stft_complex = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
    stft_magnitude = np.abs(stft_complex)
    stft_power = stft_magnitude**2

    mel_power = librosa.feature.melspectrogram(
        S=stft_power, sr=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )


    mel_db = librosa.power_to_db(mel_power)
    mel_db_for_cnn = librosa.power_to_db(mel_power, ref=np.max).astype(np.float32)

    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=config.MFCC_COUNT)
    for coefficient_index in range(mfcc.shape[0]):
        features.update(_stats(f"mfcc_{coefficient_index + 1}", mfcc[coefficient_index]))

    pitch_hz = _safe_voiced_f0(waveform) if audio_type == "vowel" else np.array([])
    features.update(_stats("pitch", pitch_hz))
    jitter = (
        np.nanmean(np.abs(np.diff(pitch_hz))) / np.nanmean(pitch_hz)
        if pitch_hz.size > 1 and np.nanmean(pitch_hz) > 0
        else np.nan
    )
    features["jitter"] = float(jitter)

    amplitude_envelope = np.abs(hilbert(waveform))
    shimmer = (
        np.mean(np.abs(np.diff(amplitude_envelope))) / np.mean(amplitude_envelope)
        if amplitude_envelope.size > 1 and np.mean(amplitude_envelope) > 0
        else np.nan
    )
    features["shimmer"] = float(shimmer)

    features.update(
        _stats(
            "spec_centroid",
            librosa.feature.spectral_centroid(S=stft_magnitude, sr=sample_rate).ravel(),
        )
    )
    features.update(
        _stats(
            "spec_bandwidth",
            librosa.feature.spectral_bandwidth(S=stft_magnitude, sr=sample_rate).ravel(),
        )
    )
    features.update(
        _stats(
            "spec_rolloff",
            librosa.feature.spectral_rolloff(S=stft_magnitude, sr=sample_rate).ravel(),
        )
    )
    features.update(_stats("zcr", librosa.feature.zero_crossing_rate(waveform).ravel()))

    rms = librosa.feature.rms(y=waveform).ravel()
    features.update(_stats("rms", rms))

    short_time_energy = np.array(
        [
            np.sum(np.square(waveform[frame_start : frame_start + config.STE_FRAME_SIZE]))
            for frame_start in range(
                0, max(1, len(waveform) - config.STE_FRAME_SIZE), config.STE_HOP_SIZE
            )
            if frame_start + config.STE_FRAME_SIZE <= len(waveform)
        ]
    )
    features["ste_mean"] = (
        float(np.mean(short_time_energy)) if short_time_energy.size else np.nan
    )
    features["ste_std"] = (
        float(np.std(short_time_energy)) if short_time_energy.size else np.nan
    )

    chroma = librosa.feature.chroma_stft(S=stft_power, sr=sample_rate)
    for pitch_class_index in range(chroma.shape[0]):
        features[f"chroma_{pitch_class_index + 1}_mean"] = float(
            np.mean(chroma[pitch_class_index])
        )
        features[f"chroma_{pitch_class_index + 1}_std"] = float(
            np.std(chroma[pitch_class_index])
        )

    onset_strength = librosa.onset.onset_strength(S=mel_db, sr=sample_rate)
    features["onset_mean"] = (
        float(np.mean(onset_strength)) if onset_strength.size else np.nan
    )
    features["onset_std"] = (
        float(np.std(onset_strength)) if onset_strength.size else np.nan
    )
    features["onset_peak"] = (
        float(np.max(onset_strength)) if onset_strength.size else np.nan
    )

    spectral_flux = (
        np.sqrt(np.sum(np.diff(stft_magnitude, axis=1) ** 2, axis=0))
        if stft_magnitude.shape[1] > 1
        else np.array([])
    )
    features["spectral_flux_mean"] = (
        float(np.mean(spectral_flux)) if spectral_flux.size else np.nan
    )
    features["spectral_flux_std"] = (
        float(np.std(spectral_flux)) if spectral_flux.size else np.nan
    )

    spectrogram_db = _fix_spec_frames(mel_db_for_cnn, target_spec_frames)
    if np.isnan(spectrogram_db).any():
        raise ValueError(f"NaN values in spectrogram: {path}")

    return features, spectrogram_db


def _process_row(
    row: dict,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    target_spec_frames: int,
    spec_dir: str,
) -> dict | None:
    path = Path(row["processed_audio_path"])
    try:
        features, spectrogram = extract_features_and_spectrogram(
            path,
            sample_rate,
            n_mels,
            n_fft,
            hop_length,
            target_spec_frames,
            row["audio_type"],
        )
        spectrogram_path = Path(spec_dir) / f"{row['participant_id']}_{path.stem}.npy"
        np.save(spectrogram_path, spectrogram)
        return {
            "participant_id": row["participant_id"],
            "label": int(row["label"]),
            "is_sick": int(row["is_sick"]),
            "audio_type": row["audio_type"],
            "processed_audio_path": str(path),
            "spectrogram_path": str(spectrogram_path),
            **features,
        }
    except Exception as error:
        print(f"warning: feature extraction failed for {path}: {error}")
        return None


def run_extract(args) -> None:
    index_df = pd.read_csv(ARTIFACTS_DIR / "preprocessed_index.csv")
    spec_dir = PROCESSED_DIR / "spectrograms"
    spec_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    feature_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _process_row, row, args.sample_rate, args.n_mels, args.n_fft,
                args.hop_length, args.target_spec_frames, str(spec_dir),
            )
            for row in index_df.to_dict("records")
        ]
        for completed_count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result is not None:
                feature_rows.append(result)
            if completed_count % 250 == 0 or completed_count == len(futures):
                print(f"progress: {completed_count}/{len(futures)}")

    features_df = pd.DataFrame(feature_rows)
    features_df.to_csv(FEATURES_PATH, index=False)
    features_df.describe(include="all").transpose().to_csv(ARTIFACTS_DIR / "feature_summary.csv")
    print(f"feature extraction complete. rows={features_df.shape[0]} cols={features_df.shape[1]}")


def _embed_paths(paths: list[str], model_key: str, batch_size: int) -> np.ndarray:
    """Mean-pooled hidden states of a frozen pretrained audio model, one vector
    per clip. Masks padding so shorter clips aren't diluted by zeros."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = getattr(torchaudio.pipelines, model_key)
    model_sample_rate = bundle.sample_rate
    model = bundle.get_model().to(device).eval()
    embeddings = np.zeros((len(paths), config.EMBEDDING_DIM), dtype=np.float32)

    for batch_start in range(0, len(paths), batch_size):
        waveforms = []
        for audio_path in paths[batch_start : batch_start + batch_size]:
            waveform, _ = librosa.load(audio_path, sr=model_sample_rate, mono=True)
            if waveform.size == 0:
                waveform = np.zeros(model_sample_rate, dtype=np.float32)
            waveforms.append(waveform.astype(np.float32))
        waveform_lengths = torch.tensor([len(w) for w in waveforms], device=device)
        max_length = int(waveform_lengths.max())
        padded_waveforms = torch.zeros(len(waveforms), max_length, dtype=torch.float32)
        for row_index, waveform in enumerate(waveforms):
            padded_waveforms[row_index, : len(waveform)] = torch.from_numpy(waveform)
        padded_waveforms = padded_waveforms.to(device)
        with torch.inference_mode():
            hidden_states, output_lengths = model.extract_features(
                padded_waveforms, lengths=waveform_lengths
            )
            last_hidden = hidden_states[-1]
            for row_index in range(last_hidden.shape[0]):
                valid_frames = (
                    int(output_lengths[row_index])
                    if output_lengths is not None
                    else last_hidden.shape[1]
                )
                valid_frames = max(1, min(valid_frames, last_hidden.shape[1]))
                embeddings[batch_start + row_index] = (
                    last_hidden[row_index, :valid_frames].mean(dim=0).cpu().numpy()
                )
        if (batch_start // batch_size) % 25 == 0 or batch_start + batch_size >= len(paths):
            print(f"progress: {min(batch_start + batch_size, len(paths))}/{len(paths)}")
    return embeddings


def extract_dataset_embeddings(dataset_name: str, features_df: pd.DataFrame) -> np.ndarray:
    """Embeddings for an external dataset feature table (called by preprocessing.py)."""
    embeddings = _embed_paths(
        features_df["processed_audio_path"].tolist(),
        config.EMBEDDING_MODEL,
        config.EMBEDDING_BATCH_SIZE,
    )
    output_path = DATA_DIR / f"{dataset_name}_embeddings.npy"
    np.save(output_path, embeddings)
    print(f"[{dataset_name}] embeddings saved: {embeddings.shape} -> {output_path}")
    return embeddings


def run_embed(args) -> None:
    output_path = Path(args.output)
    print(f"embedding model={args.model} device={'cuda' if torch.cuda.is_available() else 'cpu'} -> {output_path}")
    features_df = pd.read_csv(FEATURES_PATH)
    embeddings = _embed_paths(features_df["processed_audio_path"].tolist(), args.model, args.batch_size)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    print(f"embeddings saved: {embeddings.shape} -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ext = sub.add_parser("extract", help="hand-crafted features + spectrograms")
    p_ext.add_argument("--sample-rate", type=int, default=config.DEFAULT_SAMPLE_RATE)
    p_ext.add_argument("--n-mels", type=int, default=config.DEFAULT_N_MELS)
    p_ext.add_argument("--n-fft", type=int, default=config.DEFAULT_N_FFT)
    p_ext.add_argument("--hop-length", type=int, default=config.DEFAULT_HOP_LENGTH)
    p_ext.add_argument("--target-spec-frames", type=int, default=config.DEFAULT_TARGET_SPEC_FRAMES)
    p_ext.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p_ext.set_defaults(func=run_extract)

    p_emb = sub.add_parser("embed", help="frozen pretrained embeddings")
    p_emb.add_argument("--batch-size", type=int, default=config.EMBEDDING_BATCH_SIZE)
    p_emb.add_argument("--model", type=str, default=config.EMBEDDING_MODEL)
    p_emb.add_argument("--output", type=str, default=str(EMBEDDINGS_PATH))
    p_emb.set_defaults(func=run_embed)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
