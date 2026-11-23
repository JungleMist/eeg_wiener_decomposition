#!/usr/bin/env python3
"""Train patient-level EEGNet models on Script 22 TUEP component epochs."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import yaml

from eeg_bg.config.settings import load_config
from eeg_bg.io.dataset import active_dataset_name
from eeg_bg.ml.cnn_model import EEGNet
from eeg_bg.ml.erp_eegnet import (
    classification_metrics,
    select_balanced_accuracy_threshold,
)


SCHEMA_VERSION = 1
SCRIPT22_SCHEMA_VERSION = 1
COMBINED_CONDITION = "specific_coherent"
CONDITIONS = ("raw", "specific", "coherent", COMBINED_CONDITION)
SUPPORTED_MODES = ("frequency", "phasegated", "zerophase")
SPLIT_MAP = {"train": "train", "val": "validation", "test": "test"}
CONDITION_SEED_OFFSETS = {
    "raw": 1, "specific": 2, "coherent": 3, COMBINED_CONDITION: 4,
}
CONDITION_ARTIFACTS = (
    "best_model.pt", "best_params.json", "history.csv", "val_metrics.json",
    "test_metrics.json", "val_predictions.csv", "test_predictions.csv",
    "val_epoch_predictions.csv", "test_epoch_predictions.csv",
)


class CacheMismatch(ValueError):
    """Raised when an EEGNet result cache is incomplete or incompatible."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", choices=SUPPORTED_MODES)
    parser.add_argument("--condition", choices=(*CONDITIONS, "all"), default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--random-state", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(_jsonable(payload), indent=2, allow_nan=False), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_yaml(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(_jsonable(payload), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(data: Any, key: str) -> Any:
    value = data[key]
    return value.item() if np.asarray(value).ndim == 0 else value


def _combined_channel_names(ch_names: tuple[str, ...]) -> tuple[str, ...]:
    return (
        *(f"specific::{channel}" for channel in ch_names),
        *(f"coherent::{channel}" for channel in ch_names),
    )


def _condition_n_channels(condition: str, base_channels: int) -> int:
    return 2 * base_channels if condition == COMBINED_CONDITION else base_channels


def _inspect_record(path: Path, cfg: dict, mode: str) -> dict[str, Any]:
    required = {
        "raw", "specific", "coherent", COMBINED_CONDITION,
        "specific_coherent_ch_names", "epoch_start_samples", "epoch_start_sec",
        "epoch_segment_indices", "label", "class_name", "split", "patient_id",
        "recording_id", "evaluation_id", "subject_id", "record_cache_id",
        "dataset_name", "ch_names", "sfreq", "epoch_samples", "n_epochs", "source_mode",
        "fingerprint", "schema_version",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"Missing Script 22 keys in {path}: {missing}")
        record = {
            "path": path,
            "label": int(_scalar(data, "label")),
            "class_name": str(_scalar(data, "class_name")),
            "source_split": str(_scalar(data, "split")),
            "patient_id": str(_scalar(data, "patient_id")),
            "recording_id": str(_scalar(data, "recording_id")),
            "evaluation_id": str(_scalar(data, "evaluation_id")),
            "subject_id": str(_scalar(data, "subject_id")),
            "record_cache_id": str(_scalar(data, "record_cache_id")),
            "dataset_name": str(_scalar(data, "dataset_name")),
            "ch_names": tuple(str(value) for value in data["ch_names"]),
            "specific_coherent_ch_names": tuple(
                str(value) for value in data["specific_coherent_ch_names"]
            ),
            "sfreq": float(_scalar(data, "sfreq")),
            "epoch_samples": int(_scalar(data, "epoch_samples")),
            "n_epochs": int(_scalar(data, "n_epochs")),
            "source_mode": str(_scalar(data, "source_mode")),
            "source_fingerprint": str(_scalar(data, "fingerprint")),
            "source_schema": int(_scalar(data, "schema_version")),
        }
        shapes = {condition: tuple(data[condition].shape) for condition in CONDITIONS}
        dtypes = {condition: data[condition].dtype for condition in CONDITIONS}
        position_shapes = {
            key: tuple(data[key].shape)
            for key in ("epoch_start_samples", "epoch_start_sec", "epoch_segment_indices")
        }
    if record["source_schema"] != SCRIPT22_SCHEMA_VERSION:
        raise ValueError(f"Unsupported Script 22 schema in {path}")
    if record["dataset_name"] != "tuep":
        raise ValueError(f"Script 22 cache is not TUEP: {path}")
    if record["source_split"] not in SPLIT_MAP:
        raise ValueError(f"Invalid split in {path}: {record['source_split']!r}")
    record["split"] = SPLIT_MAP[record["source_split"]]
    expected_class = {0: "epilepsy", 1: "control"}.get(record["label"])
    if record["class_name"] != expected_class:
        raise ValueError(f"Invalid TUEP label mapping in {path}")
    if record["subject_id"] != record["evaluation_id"]:
        raise ValueError(f"subject_id/evaluation_id mismatch in {path}")
    if record["record_cache_id"] != path.stem:
        raise ValueError(f"record_cache_id does not match filename in {path}")
    if record["evaluation_id"] != path.parent.name:
        raise ValueError(f"evaluation_id does not match parent directory in {path}")
    if record["source_mode"] != mode:
        raise ValueError(f"Script 22 mode mismatch in {path}")
    expected_channels = tuple(str(value) for value in cfg["channels"]["standard_19"])
    expected_sfreq = float(cfg["preprocessing"]["target_sfreq"])
    expected_samples = int(expected_sfreq * float(cfg["preprocessing"]["epoch_length_sec"]))
    if len(expected_channels) != 19 or len(set(expected_channels)) != 19:
        raise ValueError("channels.standard_19 must contain 19 unique channels")
    if record["ch_names"] != expected_channels:
        raise ValueError(f"Channel order mismatch in {path}")
    if record["specific_coherent_ch_names"] != _combined_channel_names(expected_channels):
        raise ValueError(f"Combined channel order mismatch in {path}")
    if not np.isclose(record["sfreq"], expected_sfreq):
        raise ValueError(f"Sampling-rate mismatch in {path}")
    if record["epoch_samples"] != expected_samples or record["n_epochs"] < 1:
        raise ValueError(f"Invalid epoch dimensions in {path}")
    for condition in CONDITIONS:
        expected_shape = (
            record["n_epochs"],
            _condition_n_channels(condition, len(expected_channels)),
            expected_samples,
        )
        if shapes[condition] != expected_shape or dtypes[condition] != np.float32:
            raise ValueError(f"{condition} shape/dtype mismatch in {path}")
    if any(shape != (record["n_epochs"],) for shape in position_shapes.values()):
        raise ValueError(f"Epoch position metadata mismatch in {path}")
    return record


def discover_records(input_root: Path, cfg: dict, mode: str) -> list[dict[str, Any]]:
    manifest_path = input_root / "manifest.csv"
    resolved_path = input_root / "config_resolved.yaml"
    if not manifest_path.is_file() or not resolved_path.is_file():
        raise FileNotFoundError(f"Script 22 manifest/config not found in {input_root}")
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    required = {"evaluation_id", "record_cache_id", "output_path", "status"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Invalid Script 22 manifest: {manifest_path}")
    unexpected = sorted(set(manifest["status"]) - {"processed", "cached", "skipped"})
    if unexpected:
        raise ValueError(f"Script 22 manifest has failed/unknown statuses: {unexpected}")
    selected = manifest[manifest["status"].isin({"processed", "cached"})]
    if selected.duplicated(["evaluation_id", "record_cache_id"]).any():
        raise ValueError("Script 22 manifest has duplicate recording identities")
    paths = [Path(value) for value in selected["output_path"]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"Script 22 manifest references missing caches: {missing[:5]}")
    if not paths:
        raise FileNotFoundError(f"No Script 22 epoch caches found in {input_root}")
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if active_dataset_name(resolved) != "tuep":
        raise ValueError("Script 22 resolved config is not dataset.active: tuep")
    records = [_inspect_record(path, cfg, mode) for path in paths]
    frame = pd.DataFrame(records)
    if frame.duplicated(["evaluation_id", "record_cache_id"]).any():
        raise ValueError("Duplicate Script 22 recording caches")
    if frame.groupby("patient_id")["split"].nunique().max() > 1:
        raise ValueError("TUEP patient leakage across splits")
    for evaluation_id, group in frame.groupby("evaluation_id"):
        if group["patient_id"].nunique() != 1 or group["label"].nunique() != 1:
            raise ValueError(f"Inconsistent TUEP evaluation unit: {evaluation_id}")
    patient_frame = frame.drop_duplicates("evaluation_id")
    for split in ("train", "validation", "test"):
        labels = set(patient_frame.loc[patient_frame["split"] == split, "label"])
        if labels != {0, 1}:
            raise ValueError(f"TUEP {split} split must contain both classes")
    return sorted(records, key=lambda item: (item["evaluation_id"], item["record_cache_id"]))


def group_patients(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    patients: list[dict[str, Any]] = []
    for evaluation_id, group in pd.DataFrame(records).groupby("evaluation_id", sort=True):
        rows = [records[int(index)] for index in group.index]
        first = rows[0]
        patients.append({
            "evaluation_id": str(evaluation_id), "patient_id": first["patient_id"],
            "label": int(first["label"]), "class_name": first["class_name"],
            "split": first["split"], "records": rows,
            "n_records": len(rows), "n_epochs": sum(row["n_epochs"] for row in rows),
            "ch_names": first["ch_names"], "epoch_samples": first["epoch_samples"],
        })
    return patients


def trial_channel_zscore(array: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    mean = values.mean(axis=-1, keepdims=True)
    std = values.std(axis=-1, keepdims=True)
    normalized = (values - mean) / np.maximum(std, np.float32(eps))
    return np.nan_to_num(normalized).astype(np.float32, copy=False)


def _load_patient(patient: dict[str, Any], condition: str) -> dict[str, Any]:
    epochs: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    starts_sec: list[np.ndarray] = []
    segment_indices: list[np.ndarray] = []
    recording_ids: list[np.ndarray] = []
    record_cache_ids: list[np.ndarray] = []
    for record in patient["records"]:
        with np.load(record["path"], allow_pickle=False) as data:
            if condition == COMBINED_CONDITION and condition not in data:
                values = np.concatenate((data["specific"], data["coherent"]), axis=1)
            else:
                values = np.asarray(data[condition], dtype=np.float32)
            epochs.append(values)
            starts.append(np.asarray(data["epoch_start_samples"], dtype=np.int64))
            starts_sec.append(np.asarray(data["epoch_start_sec"], dtype=np.float64))
            segment_indices.append(np.asarray(data["epoch_segment_indices"], dtype=np.int64))
            recording_ids.append(np.repeat(record["recording_id"], len(values)))
            record_cache_ids.append(np.repeat(record["record_cache_id"], len(values)))
    values = np.concatenate(epochs, axis=0)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite {condition} epochs for {patient['evaluation_id']}")
    return {
        "epochs": trial_channel_zscore(values),
        "epoch_start_samples": np.concatenate(starts),
        "epoch_start_sec": np.concatenate(starts_sec),
        "epoch_segment_indices": np.concatenate(segment_indices),
        "recording_ids": np.concatenate(recording_ids),
        "record_cache_ids": np.concatenate(record_cache_ids),
        **{key: patient[key] for key in (
            "evaluation_id", "patient_id", "label", "class_name", "split",
            "n_records", "n_epochs",
        )},
    }


class PatientDataset(Dataset):
    """Load all Script 22 recordings for one TUEP patient per item."""

    def __init__(self, patients: list[dict[str, Any]], condition: str) -> None:
        self.patients = list(patients)
        self.condition = condition

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return _load_patient(self.patients[index], self.condition)


def resolve_device(requested: str) -> str:
    requested = str(requested).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        backend = getattr(torch.backends, "mps", None)
        return "mps" if backend is not None and backend.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("EEGNet device 'cuda' requested but CUDA is unavailable")
    if requested == "mps":
        backend = getattr(torch.backends, "mps", None)
        if backend is None or not backend.is_available():
            raise RuntimeError("EEGNet device 'mps' requested but MPS is unavailable")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    return requested


def patient_class_weights(patients: Iterable[dict[str, Any]]) -> dict[int, float]:
    labels = np.asarray([int(patient["label"]) for patient in patients], dtype=int)
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Training patients must contain both TUEP classes")
    return {0: 1.0, 1: float(counts[0] / counts[1])}


def _make_loader(
    patients: list[dict[str, Any]], condition: str, workers: int,
    *, shuffle: bool, seed: int,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": PatientDataset(patients, condition), "batch_size": None,
        "shuffle": shuffle, "num_workers": workers,
        "generator": torch.Generator().manual_seed(seed),
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 1
    return DataLoader(**kwargs)


def _patient_groups(loader: DataLoader, size: int):
    group: list[dict[str, Any]] = []
    for item in loader:
        group.append(item)
        if len(group) == size:
            yield group
            group = []
    if group:
        yield group


def _as_int(value: Any) -> int:
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def _make_model(n_channels: int, n_times: int, model_cfg: dict) -> EEGNet:
    return EEGNet(
        n_channels=n_channels, n_times=n_times, F1=int(model_cfg["F1"]),
        D=int(model_cfg["D"]), dropout=float(model_cfg["dropout"]),
    )


def _train_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
    criterion: nn.Module, device: torch.device, class_weights: dict[int, float],
    epoch_batch_size: int, patients_per_step: int,
) -> float:
    model.train()
    weighted_patient_losses: list[float] = []
    for group in _patient_groups(loader, patients_per_step):
        optimizer.zero_grad(set_to_none=True)
        for item in group:
            epochs = item["epochs"]
            if not isinstance(epochs, torch.Tensor):
                epochs = torch.as_tensor(epochs, dtype=torch.float32)
            label = _as_int(item["label"])
            n_epochs = len(epochs)
            class_weight = float(class_weights[label])
            patient_loss = 0.0
            for start in range(0, n_epochs, epoch_batch_size):
                tensors = epochs[start:start + epoch_batch_size].to(device)
                labels = torch.full(
                    (len(tensors),), float(label), dtype=torch.float32, device=device
                )
                losses = criterion(model(tensors.unsqueeze(1)).squeeze(1), labels)
                (losses.sum() / n_epochs * class_weight / len(group)).backward()
                patient_loss += float(losses.detach().sum().cpu()) / n_epochs
            weighted_patient_losses.append(patient_loss * class_weight)
        optimizer.step()
    return float(np.mean(weighted_patient_losses))


def _predict_patients(
    model: nn.Module, patients: list[dict[str, Any]], condition: str,
    device: torch.device, epoch_batch_size: int, workers: int, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    loader = _make_loader(patients, condition, workers, shuffle=False, seed=seed)
    patient_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for item in loader:
            epochs = item["epochs"]
            if not isinstance(epochs, torch.Tensor):
                epochs = torch.as_tensor(epochs, dtype=torch.float32)
            chunks = []
            for start in range(0, len(epochs), epoch_batch_size):
                batch = epochs[start:start + epoch_batch_size].to(device)
                chunks.append(model(batch.unsqueeze(1)).squeeze(1).cpu().numpy())
            probabilities = np.concatenate(chunks)
            label = _as_int(item["label"])
            evaluation_id = str(item["evaluation_id"])
            patient_id = str(item["patient_id"])
            split = str(item["split"])
            recording_ids = np.asarray(item["recording_ids"]).astype(str)
            record_cache_ids = np.asarray(item["record_cache_ids"]).astype(str)
            starts = np.asarray(item["epoch_start_samples"], dtype=np.int64)
            starts_sec = np.asarray(item["epoch_start_sec"], dtype=np.float64)
            segments = np.asarray(item["epoch_segment_indices"], dtype=np.int64)
            per_record_indices: dict[str, int] = {}
            for probability, recording_id, cache_id, sample, sec, segment in zip(
                probabilities, recording_ids, record_cache_ids, starts, starts_sec, segments
            ):
                epoch_index = per_record_indices.get(cache_id, 0)
                per_record_indices[cache_id] = epoch_index + 1
                epoch_rows.append({
                    "condition": condition, "split": split,
                    "evaluation_id": evaluation_id, "patient_id": patient_id,
                    "recording_id": recording_id, "record_cache_id": cache_id,
                    "epoch_index": epoch_index, "segment_index": int(segment),
                    "epoch_start_sample": int(sample), "epoch_start_sec": float(sec),
                    "true_label": label, "pred_proba": float(probability),
                })
            patient_rows.append({
                "condition": condition, "split": split,
                "evaluation_id": evaluation_id, "subject_id": evaluation_id,
                "patient_id": patient_id, "recording_id": "",
                "n_records": _as_int(item["n_records"]),
                "n_epochs": len(probabilities), "true_label": label,
                "pred_proba": float(probabilities.mean()),
            })
    return pd.DataFrame(patient_rows), pd.DataFrame(epoch_rows)


def _condition_fingerprint(
    patients: list[dict[str, Any]], condition: str, model_cfg: dict,
    random_state: int, resolved_device: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION, "condition": condition,
        "inputs": [
            {
                "evaluation_id": patient["evaluation_id"], "split": patient["split"],
                "label": patient["label"],
                "records": [
                    {
                        "record_cache_id": record["record_cache_id"],
                        "fingerprint": record["source_fingerprint"],
                        "n_epochs": record["n_epochs"],
                    }
                    for record in patient["records"]
                ],
            }
            for patient in patients
        ],
        "model": _jsonable(model_cfg), "random_state": random_state,
        "resolved_device": resolved_device, "normalization": "epoch_channel_zscore",
        "loss": "patient_mean_weighted_bce",
        "checkpoint_selection": "validation_patient_auprc",
        "threshold_selection": "validation_patient_balanced_accuracy",
        "patient_aggregation": "mean_epoch_probability_across_recordings",
        "positive_label": "1=control",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_cache_marker(condition_dir: Path, fingerprint: str) -> None:
    artifacts = {
        name: {"size": (condition_dir / name).stat().st_size,
               "sha256": _sha256_file(condition_dir / name)}
        for name in CONDITION_ARTIFACTS
    }
    _atomic_json(
        {"schema_version": SCHEMA_VERSION, "fingerprint": fingerprint,
         "artifacts": artifacts},
        condition_dir / "cache.json",
    )


def _condition_cache_matches(condition_dir: Path, fingerprint: str) -> bool:
    if not condition_dir.exists() or not any(condition_dir.iterdir()):
        return False
    marker_path = condition_dir / "cache.json"
    if not marker_path.is_file():
        raise CacheMismatch(f"Incomplete EEGNet result cache: {condition_dir}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        int(marker.get("schema_version", -1)) != SCHEMA_VERSION
        or str(marker.get("fingerprint", "")) != fingerprint
    ):
        raise CacheMismatch(f"EEGNet cache mismatch: {condition_dir}; re-run with --force")
    for name in CONDITION_ARTIFACTS:
        path = condition_dir / name
        entry = marker.get("artifacts", {}).get(name, {})
        if (
            not path.is_file() or path.stat().st_size != int(entry.get("size", -1))
            or _sha256_file(path) != str(entry.get("sha256", ""))
        ):
            raise CacheMismatch(f"Incomplete/corrupt EEGNet artifact: {path}")
    return True


def _set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _release_condition_memory(device: str) -> None:
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def train_condition(
    condition: str, patients: list[dict[str, Any]], model_cfg: dict,
    condition_dir: Path, random_state: int, resolved_device: str,
    workers: int, fingerprint: str,
    *, fixed_threshold: float | None = None,
) -> dict[str, Any]:
    splits = {
        split: [patient for patient in patients if patient["split"] == split]
        for split in ("train", "validation", "test")
    }
    class_weights = patient_class_weights(splits["train"])
    _set_random_seed(random_state)
    device = torch.device(resolved_device)
    n_channels = _condition_n_channels(condition, len(patients[0]["ch_names"]))
    n_times = int(patients[0]["epoch_samples"])
    model = _make_model(n_channels, n_times, model_cfg).to(device)
    criterion = nn.BCELoss(reduction="none")
    optimizer = Adam(
        model.parameters(), lr=float(model_cfg["lr"]),
        weight_decay=float(model_cfg["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=float(model_cfg["lr_factor"]),
        patience=int(model_cfg["lr_patience"]),
    )
    batch_size = int(model_cfg["batch_size"])
    patients_per_step = int(model_cfg.get("records_per_step", 4))
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    train_loader = _make_loader(
        splits["train"], condition, workers, shuffle=True, seed=random_state
    )
    for epoch in range(1, int(model_cfg["max_epochs"]) + 1):
        train_loss = _train_epoch(
            model, train_loader, optimizer, criterion, device, class_weights,
            batch_size, patients_per_step,
        )
        val_predictions, _ = _predict_patients(
            model, splits["validation"], condition, device,
            batch_size, workers, random_state,
        )
        score = float(average_precision_score(
            val_predictions["true_label"], val_predictions["pred_proba"]
        ))
        scheduler.step(score)
        history.append({
            "epoch": epoch, "train_patient_weighted_loss": train_loss,
            "validation_patient_auprc": score,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })
        if np.isfinite(score) and score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(model_cfg["patience"]):
                break
    if best_state is None:
        raise RuntimeError("No finite validation patient AUPRC was produced")
    model.load_state_dict(best_state)
    val_predictions, val_epoch_predictions = _predict_patients(
        model, splits["validation"], condition, device,
        batch_size, workers, random_state,
    )
    threshold = fixed_threshold
    if threshold is None:
        threshold = select_balanced_accuracy_threshold(
            val_predictions["true_label"].to_numpy(),
            val_predictions["pred_proba"].to_numpy(),
        )
    test_predictions, test_epoch_predictions = _predict_patients(
        model, splits["test"], condition, device,
        batch_size, workers, random_state,
    )
    val_metrics = classification_metrics(
        val_predictions["true_label"], val_predictions["pred_proba"], threshold
    )
    test_metrics = classification_metrics(
        test_predictions["true_label"], test_predictions["pred_proba"], threshold
    )
    for frame in (
        val_predictions, test_predictions, val_epoch_predictions, test_epoch_predictions
    ):
        frame["predicted_label"] = (frame["pred_proba"] >= threshold).astype(np.int8)
    condition_dir.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save({
        "state_dict": best_state, "condition": condition,
        "fingerprint": fingerprint, "n_channels": n_channels,
        "n_times": n_times, "positive_label": "1=control",
    }, condition_dir / "best_model.pt")
    _atomic_csv(pd.DataFrame(history), condition_dir / "history.csv")
    _atomic_json(val_metrics, condition_dir / "val_metrics.json")
    _atomic_json(test_metrics, condition_dir / "test_metrics.json")
    _atomic_csv(val_predictions, condition_dir / "val_predictions.csv")
    _atomic_csv(test_predictions, condition_dir / "test_predictions.csv")
    _atomic_csv(val_epoch_predictions, condition_dir / "val_epoch_predictions.csv")
    _atomic_csv(test_epoch_predictions, condition_dir / "test_epoch_predictions.csv")
    best_params = {
        "schema_version": SCHEMA_VERSION, "fingerprint": fingerprint,
        "condition": condition, "random_state": random_state,
        "resolved_device": resolved_device, "n_channels": n_channels,
        "n_times": n_times, "F1": int(model_cfg["F1"]), "D": int(model_cfg["D"]),
        "dropout": float(model_cfg["dropout"]), "lr": float(model_cfg["lr"]),
        "weight_decay": float(model_cfg["weight_decay"]), "batch_size": batch_size,
        "records_per_step": patients_per_step,
        "evaluation_units_per_step": patients_per_step,
        "class_weights": class_weights, "best_epoch": best_epoch,
        "best_validation_patient_auprc": best_score, "threshold": threshold,
        "checkpoint_selection": "validation_patient_auprc",
        "threshold_selection": (
            "validation_patient_balanced_accuracy" if fixed_threshold is None else "fixed"
        ),
        "normalization": "epoch_channel_zscore", "loss": "patient_mean_weighted_bce",
        "patient_aggregation": "mean_epoch_probability_across_recordings",
        "positive_label": "1=control",
    }
    _atomic_json(best_params, condition_dir / "best_params.json")
    _write_cache_marker(condition_dir, fingerprint)
    return _load_condition_result(condition_dir)


def _load_condition_result(condition_dir: Path) -> dict[str, Any]:
    params = json.loads((condition_dir / "best_params.json").read_text(encoding="utf-8"))
    return {
        "condition": params["condition"], "best_params": params,
        "validation_metrics": json.loads(
            (condition_dir / "val_metrics.json").read_text(encoding="utf-8")
        ),
        "test_metrics": json.loads(
            (condition_dir / "test_metrics.json").read_text(encoding="utf-8")
        ),
        "validation_predictions": pd.read_csv(condition_dir / "val_predictions.csv"),
        "test_predictions": pd.read_csv(condition_dir / "test_predictions.csv"),
        "validation_epoch_predictions": pd.read_csv(
            condition_dir / "val_epoch_predictions.csv"
        ),
        "test_epoch_predictions": pd.read_csv(condition_dir / "test_epoch_predictions.csv"),
    }


def _split_manifest(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([{key: record[key] for key in (
        "evaluation_id", "patient_id", "recording_id", "record_cache_id",
        "split", "source_split", "label", "class_name", "n_epochs",
        "source_fingerprint", "path",
    )} for record in records])


def _condition_summary(result: dict[str, Any], patients: list[dict[str, Any]]) -> dict:
    params = result["best_params"]
    counts = pd.DataFrame(patients).groupby("split").size().to_dict()
    row: dict[str, Any] = {
        "condition": result["condition"],
        "n_train_patients": int(counts.get("train", 0)),
        "n_validation_patients": int(counts.get("validation", 0)),
        "n_test_patients": int(counts.get("test", 0)),
        "best_epoch": int(params["best_epoch"]),
        "best_validation_patient_auprc": float(params["best_validation_patient_auprc"]),
        "validation_threshold": float(params["threshold"]),
    }
    for prefix, metrics in (
        ("validation", result["validation_metrics"]),
        ("test", result["test_metrics"]),
    ):
        row.update({f"{prefix}_{key}": value for key, value in metrics.items()
                    if key != "confusion_matrix"})
        row[f"{prefix}_confusion_matrix"] = json.dumps(metrics["confusion_matrix"])
    return row


def run(
    config_path: str | Path = "configs/default.yaml", *, input_dir: Path | None = None,
    output_dir: Path | None = None, mode: str | None = None,
    condition: str = "all", device: str | None = None, workers: int | None = None,
    random_state: int | None = None, force: bool = False,
) -> Path:
    cfg = deepcopy(load_config(Path(config_path)))
    if active_dataset_name(cfg) != "tuep":
        raise ValueError("Script 23 requires dataset.active: tuep")
    try:
        model_cfg = deepcopy(cfg["ml"]["tuep_component_eegnet"])
    except KeyError as exc:
        raise KeyError("Missing ml.tuep_component_eegnet config") from exc
    effective_mode = str(mode or cfg["wiener"].get("mode", "frequency"))
    if effective_mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported Wiener mode: {effective_mode!r}")
    if condition not in (*CONDITIONS, "all"):
        raise ValueError(f"Unknown condition: {condition!r}")
    cfg["wiener"]["mode"] = effective_mode
    effective_workers = int(
        workers if workers is not None else model_cfg.get("num_workers", 0)
    )
    if effective_workers < 0:
        raise ValueError("--workers must be >= 0")
    base_seed = int(
        random_state if random_state is not None else model_cfg.get("random_state", 42)
    )
    requested_device = str(device or model_cfg.get("device", "auto"))
    resolved_device = resolve_device(requested_device)
    model_cfg.update({
        "device": requested_device, "num_workers": effective_workers,
        "random_state": base_seed,
    })
    input_root = (
        input_dir.expanduser().resolve() if input_dir is not None else
        Path(cfg["paths"]["cache_dir"])
        / f"tuep_continuous_wiener_{effective_mode}" / "epochs"
    )
    output_root = (
        output_dir.expanduser().resolve() if output_dir is not None else
        Path(cfg["paths"]["results_dir"])
        / f"tuep_component_eegnet_{effective_mode}"
    )
    records = discover_records(input_root, cfg, effective_mode)
    patients = group_patients(records)
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_csv(_split_manifest(records), output_root / "split_manifest.csv")
    fingerprints = {
        item: _condition_fingerprint(
            patients, item, model_cfg, base_seed + CONDITION_SEED_OFFSETS[item],
            resolved_device,
        )
        for item in CONDITIONS
    }
    selected = CONDITIONS if condition == "all" else (condition,)
    for index, item in enumerate(selected, start=1):
        condition_dir = output_root / "conditions" / item
        if not force and _condition_cache_matches(condition_dir, fingerprints[item]):
            print(f"[EEGNet {index}/{len(selected)}] {item}: cached")
            continue
        if force:
            (condition_dir / "cache.json").unlink(missing_ok=True)
        print(f"[EEGNet {index}/{len(selected)}] {item}: training")
        try:
            train_condition(
                item, patients, model_cfg, condition_dir,
                base_seed + CONDITION_SEED_OFFSETS[item], resolved_device,
                effective_workers, fingerprints[item],
            )
        finally:
            _release_condition_memory(resolved_device)
    completed = []
    for item in CONDITIONS:
        condition_dir = output_root / "conditions" / item
        try:
            if _condition_cache_matches(condition_dir, fingerprints[item]):
                completed.append(_load_condition_result(condition_dir))
        except CacheMismatch:
            if item in selected:
                raise
    metrics = pd.DataFrame([_condition_summary(result, patients) for result in completed])
    _atomic_csv(metrics, output_root / "condition_metrics.csv")
    _atomic_csv(pd.concat([
        frame for result in completed for frame in
        (result["validation_predictions"], result["test_predictions"])
    ], ignore_index=True), output_root / "predictions.csv")
    _atomic_csv(pd.concat([
        frame for result in completed for frame in
        (result["validation_epoch_predictions"], result["test_epoch_predictions"])
    ], ignore_index=True), output_root / "epoch_predictions.csv")
    resolved = deepcopy(cfg)
    resolved["ml"]["tuep_component_eegnet"] = {
        **model_cfg, "input_dir": str(input_root), "output_dir": str(output_root),
        "mode": effective_mode, "requested_device": requested_device,
        "resolved_device": resolved_device, "selected_conditions": list(selected),
        "checkpoint_selection": "validation_patient_auprc",
        "threshold_selection": "validation_patient_balanced_accuracy",
        "loss": "patient_mean_weighted_bce",
        "patient_aggregation": "mean_epoch_probability_across_recordings",
        "specific_coherent_layout": "channel_axis_specific_then_coherent",
        "positive_label": "1=control",
    }
    _atomic_yaml(resolved, output_root / "config_resolved.yaml")
    patient_frame = pd.DataFrame(patients)
    _atomic_json({
        "schema_version": SCHEMA_VERSION, "input_dir": str(input_root),
        "output_dir": str(output_root), "mode": effective_mode,
        "requested_conditions": list(selected),
        "completed_conditions": [result["condition"] for result in completed],
        "requested_device": requested_device, "resolved_device": resolved_device,
        "random_state": base_seed,
        "patient_counts": {
            split: int((patient_frame["split"] == split).sum())
            for split in ("train", "validation", "test")
        },
        "patient_disjoint": True, "positive_label": "1=control",
        "checkpoint_selection": "validation_patient_auprc",
        "threshold_selection": "validation_patient_balanced_accuracy",
        "final_test_evaluation": "single_frozen_evaluation_per_training_run",
        "normalization": "epoch_channel_zscore", "loss": "patient_mean_weighted_bce",
        "patient_aggregation": "mean_epoch_probability_across_recordings",
        "specific_coherent_layout": "channel_axis_specific_then_coherent",
        "condition_fingerprints": fingerprints,
        "comparison_summary": metrics.to_dict(orient="records"),
    }, output_root / "run_summary.json")
    print(f"TUEP component EEGNet training complete: {output_root}")
    return output_root


def main() -> None:
    args = _parse_args()
    run(
        args.config, input_dir=args.input_dir, output_dir=args.output_dir,
        mode=args.mode, condition=args.condition, device=args.device,
        workers=args.workers, random_state=args.random_state, force=args.force,
    )


if __name__ == "__main__":
    main()
