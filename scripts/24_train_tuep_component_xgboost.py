"""Train patient-level TUEP XGBoost models from Script 22 component epochs."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import yaml

from eeg_bg.config.settings import load_config
from eeg_bg.features.extraction import FeatureDataset
from eeg_bg.features.profiles import PROFILES
from eeg_bg.io.dataset import active_dataset_name
from eeg_bg.ml.xgb_pipeline import (
    train_xgboost,
    evaluation_level_predict,
    evaluate_subject_level,
    find_optimal_threshold,
)
from eeg_bg.ml.shap_analysis import (
    compute_shap_values,
    aggregate_shap_by_band,
    aggregate_shap_by_channel,
    plot_shap_summary,
)


SCRIPT22_SCHEMA_VERSION = 1
COMBINED_CONDITION = "specific_coherent"
BASE_CONDITIONS = ("raw", "specific", "coherent")
CONDITIONS = (*BASE_CONDITIONS, COMBINED_CONDITION)


def _release_condition_memory() -> None:
    gc.collect()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("workers must be a positive integer")
    return parsed


def _scalar(data: Any, key: str) -> Any:
    value = data[key]
    return value.item() if np.asarray(value).ndim == 0 else value


def _combined_channel_names(ch_names: list[str]) -> list[str]:
    return [
        *(f"specific::{channel}" for channel in ch_names),
        *(f"coherent::{channel}" for channel in ch_names),
    ]


def _condition_feature_names(profile_name: str, condition: str) -> list[str]:
    names = list(PROFILES[profile_name].names)
    if condition != COMBINED_CONDITION:
        return names
    return [
        *(f"specific::{name}" for name in names),
        *(f"coherent::{name}" for name in names),
    ]


def _condition_feature_dim(profile_name: str, condition: str) -> int:
    return PROFILES[profile_name].dim * (2 if condition == COMBINED_CONDITION else 1)


def _aggregate_feature_names(names: list[str]) -> list[str]:
    return [name.split("::", 1)[-1] for name in names]


def _save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _source_inventory(input_root: Path, cfg: dict) -> tuple[list[Path], str, dict]:
    manifest_path = input_root / "manifest.csv"
    resolved_path = input_root / "config_resolved.yaml"
    if not manifest_path.is_file() or not resolved_path.is_file():
        raise FileNotFoundError(f"Script 22 manifest/config not found in {input_root}")
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    required_columns = {
        "evaluation_id", "record_cache_id", "output_path", "status",
    }
    if not required_columns.issubset(manifest.columns):
        raise ValueError(f"Invalid Script 22 manifest: {manifest_path}")
    unexpected = sorted(set(manifest["status"]) - {"processed", "cached", "skipped"})
    if unexpected:
        raise ValueError(f"Script 22 manifest has failed/unknown statuses: {unexpected}")
    selected = manifest[manifest["status"].isin({"processed", "cached"})]
    if selected.duplicated(["evaluation_id", "record_cache_id"]).any():
        raise ValueError("Script 22 manifest has duplicate recording identities")
    selected = selected.sort_values(["evaluation_id", "record_cache_id"])
    files = [Path(value) for value in selected["output_path"]]
    if not files:
        raise FileNotFoundError(f"No Script 22 epoch caches found in {input_root}")
    missing_files = [str(path) for path in files if not path.is_file()]
    if missing_files:
        raise ValueError(f"Script 22 manifest references missing files: {missing_files[:5]}")
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if active_dataset_name(resolved) != "tuep":
        raise ValueError("Script 22 resolved config is not dataset.active: tuep")
    expected_channels = list(cfg["channels"]["standard_19"])
    expected_sfreq = float(cfg["preprocessing"]["target_sfreq"])
    requested_mode = str(cfg["wiener"].get("mode", "frequency"))
    records: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "raw", "specific", "coherent", COMBINED_CONDITION,
                "ch_names", "specific_coherent_ch_names", "sfreq", "split",
                "label", "class_name", "patient_id", "recording_id",
                "record_cache_id", "evaluation_id", "fingerprint",
                "dataset_name", "source_mode", "schema_version", "n_epochs",
            }
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError(f"Script 22 cache {path} missing keys: {missing}")
            schema = int(_scalar(data, "schema_version"))
            if schema != SCRIPT22_SCHEMA_VERSION:
                raise ValueError(f"Unsupported Script 22 schema in {path}")
            if str(_scalar(data, "dataset_name")) != "tuep":
                raise ValueError(f"Script 22 cache is not TUEP: {path}")
            channels = [str(value) for value in data["ch_names"]]
            if channels != expected_channels:
                raise ValueError(f"Script 22 cache {path} has invalid channel order")
            if [str(value) for value in data["specific_coherent_ch_names"]] != (
                _combined_channel_names(expected_channels)
            ):
                raise ValueError(f"Script 22 cache {path} has invalid combined layout")
            if not np.isclose(float(_scalar(data, "sfreq")), expected_sfreq):
                raise ValueError(f"Script 22 cache {path} has invalid sampling rate")
            label = int(_scalar(data, "label"))
            class_name = str(_scalar(data, "class_name"))
            if (label, class_name) not in ((0, "epilepsy"), (1, "control")):
                raise ValueError(f"Invalid TUEP label mapping in {path}")
            split = str(_scalar(data, "split"))
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Invalid split {split!r} in {path}")
            source_mode = str(_scalar(data, "source_mode"))
            if source_mode != requested_mode:
                raise ValueError(f"Wiener mode mismatch in {path}")
            shapes = {key: tuple(data[key].shape) for key in CONDITIONS}
            base_shape = shapes["raw"]
            if len(base_shape) != 3 or base_shape[1] != len(expected_channels):
                raise ValueError(f"Invalid epoch shape in {path}: {base_shape}")
            if any(shapes[key] != base_shape for key in BASE_CONDITIONS):
                raise ValueError(f"Component shape mismatch in {path}")
            if shapes[COMBINED_CONDITION] != (
                base_shape[0], 2 * len(expected_channels), base_shape[2]
            ):
                raise ValueError(f"Combined epoch shape mismatch in {path}")
            if any(data[key].dtype != np.float32 for key in CONDITIONS):
                raise ValueError(f"Script 22 cache {path} has non-float32 arrays")
            record = {
                "path": str(path), "fingerprint": str(_scalar(data, "fingerprint")),
                "source_mode": source_mode,
                "evaluation_id": str(_scalar(data, "evaluation_id")),
                "patient_id": str(_scalar(data, "patient_id")),
                "recording_id": str(_scalar(data, "recording_id")),
                "record_cache_id": str(_scalar(data, "record_cache_id")),
                "split": split, "label": label, "schema_version": schema,
                "n_epochs": int(_scalar(data, "n_epochs")),
            }
            if record["record_cache_id"] != path.stem:
                raise ValueError(f"record_cache_id does not match filename in {path}")
            if record["evaluation_id"] != path.parent.name:
                raise ValueError(f"evaluation_id does not match parent directory in {path}")
            if record["n_epochs"] != base_shape[0] or record["n_epochs"] < 1:
                raise ValueError(f"n_epochs does not match arrays in {path}")
            records.append(record)
            identities.append({key: record[key] for key in (
                "evaluation_id", "patient_id", "recording_id", "record_cache_id",
                "split", "label",
            )})
    frame = pd.DataFrame(identities)
    if frame.groupby("patient_id")["split"].nunique().max() > 1:
        raise ValueError("TUEP patient leakage across splits")
    for evaluation_id, group in frame.groupby("evaluation_id"):
        if group["patient_id"].nunique() != 1 or group["label"].nunique() != 1:
            raise ValueError(f"Inconsistent TUEP evaluation unit: {evaluation_id}")
    patient_frame = frame.drop_duplicates("evaluation_id")
    for split in ("train", "val", "test"):
        if set(patient_frame.loc[patient_frame["split"] == split, "label"]) != {0, 1}:
            raise ValueError(f"TUEP {split} split must contain both classes")
    records.sort(key=lambda row: (row["evaluation_id"], row["record_cache_id"]))
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    fingerprint = hashlib.sha256(encoded).hexdigest()
    return files, fingerprint, {
        "input_dir": str(input_root), "n_files": len(files),
        "n_patients": int(frame["evaluation_id"].nunique()),
        "source_fingerprint": fingerprint, "source_modes": [requested_mode],
    }


def _extract_file(args: tuple) -> tuple:
    (
        index, path_str, condition, split, sfreq, nperseg, freq_band,
        profile_name, conn_nperseg,
    ) = args
    profile = PROFILES[profile_name]
    with np.load(path_str, allow_pickle=False) as data:
        if str(_scalar(data, "split")) != split:
            return index, [], [], [], [], [], []
        epochs = np.asarray(data[condition])
        ch_names = [str(value) for value in data["ch_names"]]
        label = int(_scalar(data, "label"))
        evaluation_id = str(_scalar(data, "evaluation_id"))
        patient_id = str(_scalar(data, "patient_id"))
        recording_id = str(_scalar(data, "recording_id"))
        if condition == COMBINED_CONDITION:
            n_channels = len(ch_names)
            rows = [np.concatenate([
                profile.extract_fn(
                    epoch[:n_channels], ch_names, sfreq, nperseg,
                    freq_band, conn_nperseg,
                ),
                profile.extract_fn(
                    epoch[n_channels:], ch_names, sfreq, nperseg,
                    freq_band, conn_nperseg,
                ),
            ]) for epoch in epochs]
        else:
            rows = [profile.extract_fn(
                epoch, ch_names, sfreq, nperseg, freq_band, conn_nperseg
            ) for epoch in epochs]
    expected_dim = _condition_feature_dim(profile_name, condition)
    if any(len(row) != expected_dim for row in rows):
        raise ValueError(f"Feature profile {profile_name} returned wrong dimension")
    n_rows = len(rows)
    return (
        index, rows, [label] * n_rows, [evaluation_id] * n_rows,
        [patient_id] * n_rows, [recording_id] * n_rows, ["tuep"] * n_rows,
    )


def _build_features(
    files: list[Path], condition: str, split: str, cfg: dict,
    profile_name: str, max_workers: int | None,
) -> FeatureDataset:
    sfreq = float(cfg["preprocessing"]["target_sfreq"])
    nperseg = int(cfg["wiener"]["nperseg"])
    freq_band = tuple(cfg["wiener"]["freq_band"])
    conn_nperseg = int(
        cfg.get("ml", {}).get("features", {}).get("connectivity", {})
        .get("nperseg", nperseg)
    )
    args = [
        (index, str(path), condition, split, sfreq, nperseg, freq_band,
         profile_name, conn_nperseg)
        for index, path in enumerate(files)
    ]
    results: dict[int, tuple] = {}
    if max_workers == 1:
        for item in tqdm(args, desc=f"Features [{profile_name}/{condition}/{split}]", leave=False):
            result = _extract_file(item)
            results[result[0]] = result[1:]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_extract_file, item) for item in args]
            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc=f"Features [{profile_name}/{condition}/{split}]", leave=False,
            ):
                result = future.result()
                results[result[0]] = result[1:]
    rows, labels, evaluation_ids, patient_ids, recording_ids, dataset_names = (
        [] for _ in range(6)
    )
    for index in sorted(results):
        for target, source in zip(
            (rows, labels, evaluation_ids, patient_ids, recording_ids, dataset_names),
            results[index],
        ):
            target.extend(source)
    if not rows:
        return FeatureDataset(
            np.empty((0, _condition_feature_dim(profile_name, condition))),
            np.empty(0, dtype=np.int64), [], [], [], [],
        )
    return FeatureDataset(
        np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64),
        evaluation_ids, patient_ids, recording_ids, dataset_names,
    )


def _cache_path(root: Path, profile: str, condition: str, split: str) -> Path:
    return root / "features" / profile / f"{condition}_{split}.npz"


def _load_or_extract(
    files: list[Path], input_root: Path, source_fingerprint: str,
    condition: str, split: str, cfg: dict, profile_name: str,
    force: bool, max_workers: int | None,
) -> FeatureDataset:
    feature_path = _cache_path(input_root, profile_name, condition, split)
    feature_names = _condition_feature_names(profile_name, condition)
    schema_payload = {
        "profile": profile_name, "profile_hash": PROFILES[profile_name].hash,
        "sfreq": float(cfg["preprocessing"]["target_sfreq"]),
        "nperseg": int(cfg["wiener"]["nperseg"]),
        "freq_band": list(cfg["wiener"]["freq_band"]),
        "dataset_name": "tuep", "source_fingerprint": source_fingerprint,
        "condition": condition, "feature_dim": len(feature_names),
        "feature_names": feature_names,
        "specific_coherent_layout": (
            "channel_axis_specific_then_coherent"
            if condition == COMBINED_CONDITION else None
        ),
    }
    schema_hash = hashlib.sha256(
        json.dumps(schema_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    if feature_path.exists() and not force:
        with np.load(feature_path, allow_pickle=True) as data:
            if str(_scalar(data, "schema_hash")) != schema_hash:
                raise ValueError(f"Feature cache mismatch at {feature_path}; use --force")
            if data["X"].ndim != 2 or data["X"].shape[1] != len(feature_names):
                raise ValueError(f"Feature cache dimension mismatch at {feature_path}")
            return FeatureDataset(
                data["X"].astype(np.float64), data["y"].astype(np.int64),
                [str(value) for value in data["evaluation_ids"]],
                [str(value) for value in data["patient_ids"]],
                [str(value) for value in data["recording_ids"]],
                [str(value) for value in data["dataset_names"]],
            )
    dataset = _build_features(files, condition, split, cfg, profile_name, max_workers)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        feature_path, X=dataset.X, y=dataset.y,
        evaluation_ids=np.asarray(dataset.evaluation_ids, dtype="U"),
        patient_ids=np.asarray(dataset.patient_ids, dtype="U"),
        recording_ids=np.asarray(dataset.recording_ids, dtype="U"),
        dataset_names=np.asarray(dataset.dataset_names, dtype="U"),
        schema_hash=np.asarray(schema_hash),
        schema_payload=np.asarray(json.dumps(schema_payload)),
    )
    return dataset


def _split_stats(dataset: FeatureDataset) -> dict:
    return {
        "n_epochs": int(len(dataset.X)),
        "n_evaluations": int(len(set(dataset.evaluation_ids))),
        "n_patients": int(len(set(dataset.patient_ids))),
        "n_epochs_epilepsy": int(np.sum(dataset.y == 0)),
        "n_epochs_control": int(np.sum(dataset.y == 1)),
        "n_patients_epilepsy": int(len({value for value, label in zip(
            dataset.evaluation_ids, dataset.y) if label == 0})),
        "n_patients_control": int(len({value for value, label in zip(
            dataset.evaluation_ids, dataset.y) if label == 1})),
    }


def _assert_no_patient_overlap(datasets: dict[str, FeatureDataset]) -> None:
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(datasets[left].patient_ids) & set(datasets[right].patient_ids)
        if overlap:
            raise ValueError(f"Patient leakage between {left} and {right}: {sorted(overlap)[:5]}")


def _metrics(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    metrics = evaluate_subject_level(frame, threshold)
    labels = frame["true_label"].to_numpy()
    probabilities = frame["pred_proba"].to_numpy()
    predictions = (probabilities >= threshold).astype(int)
    metrics.update({
        "auprc": float(average_precision_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "positive_class": "control", "aggregation_unit": "patient",
    })
    return metrics


def run_condition(
    condition: str, cfg: dict, files: list[Path], input_root: Path,
    source_fingerprint: str, out_root: Path, profile_name: str, force: bool,
    max_workers: int | None, source_meta: dict,
) -> dict[str, Any]:
    datasets = {
        split: _load_or_extract(
            files, input_root, source_fingerprint, condition, split,
            cfg, profile_name, force, max_workers,
        )
        for split in ("train", "val", "test")
    }
    if not len(datasets["train"].X) or not len(datasets["val"].X):
        raise ValueError(f"Condition {condition!r} requires train and val data")
    _assert_no_patient_overlap(datasets)
    if set(datasets["train"].y) != {0, 1}:
        raise ValueError("Training split must contain both TUEP classes")
    out_dir = out_root / profile_name / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_names = _condition_feature_names(profile_name, condition)
    metadata = {
        **source_meta, "condition": condition, "feature_set": profile_name,
        "feature_dim": len(feature_names), "aggregation_unit": "patient",
        "specific_coherent_layout": (
            "channel_axis_specific_then_coherent"
            if condition == COMBINED_CONDITION else None
        ),
    }
    meta_path = out_dir / "run_metadata.json"
    if meta_path.exists() and not force:
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        if previous.get("source_fingerprint") != source_fingerprint:
            raise ValueError(f"Existing results at {out_dir} use different input")
    _save_json({
        **metadata,
        "data_stats": {key: _split_stats(value) for key, value in datasets.items()},
    }, meta_path)
    _save_json({
        **{key: _split_stats(value) for key, value in datasets.items()},
        "feature_set": profile_name, "dataset_name": "tuep",
        "aggregation_unit": "patient",
    }, out_dir / "data_stats.json")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(datasets["train"].X)
    X_val = scaler.transform(datasets["val"].X)
    X_test = scaler.transform(datasets["test"].X)
    joblib.dump(scaler, out_dir / "scaler.joblib")
    model = train_xgboost(
        X_train, datasets["train"].y, X_val, datasets["val"].y,
        cfg, groups=datasets["train"].patient_ids,
    )
    joblib.dump(model, out_dir / "model.joblib")
    _save_json(model.get_params(), out_dir / "best_params.json")
    val_df = evaluation_level_predict(
        model, X_val, datasets["val"].y, datasets["val"].evaluation_ids,
        datasets["val"].patient_ids, datasets["val"].recording_ids,
        datasets["val"].dataset_names,
    )
    threshold = find_optimal_threshold(val_df)
    val_df["predicted_label"] = (val_df["pred_proba"] >= threshold).astype(int)
    val_df.to_csv(out_dir / "val_predictions.csv", index=False)
    val_metrics = _metrics(val_df, threshold)
    _save_json(val_metrics, out_dir / "val_metrics.json")
    test_df = evaluation_level_predict(
        model, X_test, datasets["test"].y, datasets["test"].evaluation_ids,
        datasets["test"].patient_ids, datasets["test"].recording_ids,
        datasets["test"].dataset_names,
    )
    test_df["predicted_label"] = (test_df["pred_proba"] >= threshold).astype(int)
    test_df.to_csv(out_dir / "test_predictions.csv", index=False)
    test_metrics = _metrics(test_df, threshold)
    _save_json(test_metrics, out_dir / "test_metrics.json")
    aggregate_names = _aggregate_feature_names(feature_names)
    shap_values = compute_shap_values(model, X_test, feature_names)
    np.save(out_dir / "shap_values_test.npy", shap_values)
    _save_json(
        aggregate_shap_by_band(shap_values, aggregate_names),
        out_dir / "shap_by_band.json",
    )
    _save_json(
        aggregate_shap_by_channel(shap_values, aggregate_names),
        out_dir / "shap_by_channel.json",
    )
    plot_shap_summary(
        shap_values, X_test, feature_names,
        title=f"SHAP Summary — {condition} [{profile_name}] (test set)",
        output_path=out_dir / "shap_summary.png",
        max_display=int(cfg["ml"]["shap"]["max_display"]),
        dpi=int(cfg["ml"]["shap"]["dpi"]),
    )
    return {
        "condition": condition, "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "data_stats": {key: _split_stats(value) for key, value in datasets.items()},
    }


def main(
    config_path: str = "configs/default.yaml", condition: str = "all",
    force: bool = False, feature_set: str = "base211",
    max_workers: int | None = None,
) -> None:
    cfg = load_config(config_path)
    if active_dataset_name(cfg) != "tuep":
        raise ValueError("Script 24 requires dataset.active: tuep")
    mode = str(cfg["wiener"].get("mode", "frequency"))
    input_root = (
        Path(cfg["paths"]["cache_dir"])
        / f"tuep_continuous_wiener_{mode}" / "epochs"
    )
    files, source_fingerprint, source_meta = _source_inventory(input_root, cfg)
    out_root = Path(cfg["paths"]["results_dir"]) / "tuep_component_xgboost"
    out_root.mkdir(parents=True, exist_ok=True)
    source_meta = {**source_meta, "requested_mode": mode}
    resolved = dict(cfg)
    resolved["tuep_component_xgboost"] = {
        "input_dir": str(input_root), "output_dir": str(out_root),
        "requested_mode": mode, "source_fingerprint": source_fingerprint,
        "conditions": list(CONDITIONS), "aggregation_unit": "patient",
        "threshold_selection": "validation_patient_macro_f1",
        "specific_coherent_layout": "channel_axis_specific_then_coherent",
        "positive_label": "1=control",
    }
    (out_root / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    selected = CONDITIONS if condition == "all" else (condition,)
    results = []
    for item in selected:
        try:
            results.append(run_condition(
                item, cfg, files, input_root, source_fingerprint, out_root,
                feature_set, force, max_workers, source_meta,
            ))
        finally:
            _release_condition_memory()
    rows = []
    for result in results:
        row = {"condition": result["condition"]}
        for prefix in ("val", "test"):
            row.update({
                f"{prefix}_{key}": value
                for key, value in result[f"{prefix}_metrics"].items()
            })
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        out_root / feature_set / "comparison_summary.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--condition", choices=[*CONDITIONS, "all"], default="all")
    parser.add_argument("--feature-set", choices=list(PROFILES), default="base211")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=_positive_int, default=None)
    arguments = parser.parse_args()
    main(
        arguments.config, arguments.condition, arguments.force,
        arguments.feature_set, arguments.workers,
    )
