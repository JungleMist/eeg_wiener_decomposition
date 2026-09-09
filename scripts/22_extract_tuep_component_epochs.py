"""Extract paired TUEP raw/specific/coherent component epochs."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from eeg_bg.config.settings import load_config
from eeg_bg.io.dataset import active_dataset_config, active_dataset_name


CACHE_SCHEMA_VERSION = 1
SCRIPT21_SCHEMA_VERSION = 1
SUPPORTED_MODES = ("frequency", "phasegated", "zerophase")
COMPONENTS = ("specific", "coherent")
COMBINED_CONDITION = "specific_coherent"
IDENTITY_FIELDS = (
    "dataset_name", "patient_id", "recording_id", "evaluation_id",
    "class_name", "label", "split", "source_partition",
)


class CacheMismatch(ValueError):
    """Raised when a component epoch cache is incompatible."""


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
    return value


def _atomic_npz(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_yaml(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(_jsonable(payload), sort_keys=False), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _scalar(data: Any, key: str) -> Any:
    value = data[key]
    return value.item() if np.asarray(value).ndim == 0 else value


def _combined_channel_names(ch_names: list[str]) -> list[str]:
    return [
        *(f"specific::{channel}" for channel in ch_names),
        *(f"coherent::{channel}" for channel in ch_names),
    ]


def discover_inputs(input_root: Path) -> list[dict[str, Path | str]]:
    manifest_path = input_root / "manifest.csv"
    config_path = input_root / "config_resolved.yaml"
    if not manifest_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Script 21 manifest/config not found in {input_root}")
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    required = {
        "evaluation_id", "record_cache_id", "specific_path", "coherent_path",
        "metadata_path", "status",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Script 21 manifest missing columns: {missing}")
    unexpected = sorted(set(manifest["status"]) - {"processed", "cached", "skipped"})
    if unexpected:
        raise ValueError(f"Script 21 manifest has failed/unknown statuses: {unexpected}")
    selected = manifest[manifest["status"].isin({"processed", "cached"})]
    if selected.empty:
        raise FileNotFoundError(f"No completed Script 21 caches found in {input_root}")
    if selected.duplicated(["evaluation_id", "record_cache_id"]).any():
        raise ValueError("Script 21 manifest has duplicate recording cache identities")
    return [
        {
            "evaluation_id": str(row.evaluation_id),
            "record_cache_id": str(row.record_cache_id),
            "specific": Path(row.specific_path),
            "coherent": Path(row.coherent_path),
            "metadata": Path(row.metadata_path),
        }
        for row in selected.itertuples()
    ]


def _expected_class_mapping(cfg: dict) -> dict[int, str]:
    result: dict[int, str] = {}
    for name, value in active_dataset_config(cfg)["classes"].items():
        default = 0 if name == "epilepsy" else 1
        label = int(value.get("label", default)) if isinstance(value, dict) else default
        result[label] = str(name)
    return result


def _read_header(paths: dict[str, Path | str], cfg: dict) -> dict[str, Any]:
    component_paths = {key: Path(paths[key]) for key in (*COMPONENTS, "metadata")}
    missing_files = [str(path) for path in component_paths.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError("Incomplete Script 21 cache: " + ", ".join(missing_files))
    metadata = json.loads(component_paths["metadata"].read_text(encoding="utf-8"))
    if int(metadata.get("schema_version", -1)) != SCRIPT21_SCHEMA_VERSION:
        raise ValueError("Unsupported Script 21 schema")
    required = {
        "sequence", "component", "ch_names", "sfreq", "fingerprint",
        "schema_version", "record_cache_id",
        "segment_source_start_samples", "segment_source_stop_samples",
        "segment_cache_start_samples", "segment_cache_stop_samples",
        *IDENTITY_FIELDS,
    }
    headers: dict[str, dict[str, Any]] = {}
    for component in COMPONENTS:
        with np.load(component_paths[component], allow_pickle=False) as data:
            absent = sorted(required - set(data.files))
            if absent:
                raise ValueError(f"Missing keys in {component_paths[component]}: {absent}")
            headers[component] = {
                **{key: _scalar(data, key) for key in IDENTITY_FIELDS},
                "component": str(_scalar(data, "component")),
                "record_cache_id": str(_scalar(data, "record_cache_id")),
                "ch_names": [str(item) for item in data["ch_names"]],
                "sfreq": float(_scalar(data, "sfreq")),
                "fingerprint": str(_scalar(data, "fingerprint")),
                "schema_version": int(_scalar(data, "schema_version")),
                "segment_source_start_samples": np.asarray(
                    data["segment_source_start_samples"], dtype=np.int64
                ),
                "segment_source_stop_samples": np.asarray(
                    data["segment_source_stop_samples"], dtype=np.int64
                ),
                "segment_cache_start_samples": np.asarray(
                    data["segment_cache_start_samples"], dtype=np.int64
                ),
                "segment_cache_stop_samples": np.asarray(
                    data["segment_cache_stop_samples"], dtype=np.int64
                ),
                "sequence_shape": tuple(data["sequence"].shape),
            }
    specific = headers["specific"]
    coherent = headers["coherent"]
    for component, header in headers.items():
        if header["component"] != component:
            raise ValueError(f"Component mismatch in {component_paths[component]}")
        if header["schema_version"] != SCRIPT21_SCHEMA_VERSION:
            raise ValueError("Unsupported Script 21 component schema")
    scalar_fields = (*IDENTITY_FIELDS, "record_cache_id", "fingerprint", "ch_names")
    for field in scalar_fields:
        if specific[field] != coherent[field]:
            raise ValueError(f"Component identity mismatch for {field}")
    if not np.isclose(specific["sfreq"], coherent["sfreq"]):
        raise ValueError("Component sampling-rate mismatch")
    for field in (
        "segment_source_start_samples", "segment_source_stop_samples",
        "segment_cache_start_samples", "segment_cache_stop_samples",
    ):
        if not np.array_equal(specific[field], coherent[field]):
            raise ValueError(f"Component segment mapping mismatch for {field}")
    if specific["sequence_shape"] != coherent["sequence_shape"]:
        raise ValueError("Component sequence shape mismatch")
    if str(metadata.get("fingerprint", "")) != specific["fingerprint"]:
        raise ValueError("Component/metadata fingerprint mismatch")
    if str(metadata.get("record_cache_id", "")) != specific["record_cache_id"]:
        raise ValueError("Component/metadata record_cache_id mismatch")
    if metadata.get("ch_names") != specific["ch_names"]:
        raise ValueError("Component/metadata channel mismatch")
    if not np.isclose(float(metadata.get("sfreq", np.nan)), specific["sfreq"]):
        raise ValueError("Component/metadata sampling-rate mismatch")
    if int(metadata.get("n_times", -1)) != specific["sequence_shape"][1]:
        raise ValueError("Component/metadata sequence-length mismatch")
    if str(paths["evaluation_id"]) != str(specific["evaluation_id"]):
        raise ValueError("Manifest evaluation_id mismatch")
    if str(paths["record_cache_id"]) != specific["record_cache_id"]:
        raise ValueError("Manifest record_cache_id mismatch")
    if specific["dataset_name"] != "tuep":
        raise ValueError("Script 21 cache is not TUEP")
    expected_classes = _expected_class_mapping(cfg)
    label = int(specific["label"])
    if expected_classes.get(label) != str(specific["class_name"]):
        raise ValueError(
            f"Invalid TUEP label mapping: {label}={specific['class_name']!r}"
        )
    if str(specific["split"]) not in {"train", "val", "test"}:
        raise ValueError(f"Invalid TUEP split: {specific['split']!r}")
    standard = list(cfg["channels"]["standard_19"])
    if len(standard) != 19 or len(set(standard)) != 19:
        raise ValueError("channels.standard_19 must contain 19 unique channels")
    expected_present = [channel for channel in standard if channel in specific["ch_names"]]
    if specific["ch_names"] != expected_present:
        raise ValueError("TUEP component channel order is not a standard_19 subset")
    if not np.isclose(
        specific["sfreq"], float(cfg["preprocessing"]["target_sfreq"])
    ):
        raise ValueError("Unexpected component sampling rate")
    starts = specific["segment_cache_start_samples"]
    stops = specific["segment_cache_stop_samples"]
    source_starts = specific["segment_source_start_samples"]
    source_stops = specific["segment_source_stop_samples"]
    lengths = {len(starts), len(stops), len(source_starts), len(source_stops)}
    if lengths != {int(metadata.get("n_segments", -1))} or not len(starts):
        raise ValueError("Invalid segment count")
    if starts[0] != 0 or np.any(stops <= starts) or np.any(starts[1:] != stops[:-1]):
        raise ValueError("Invalid contiguous cache segment mapping")
    if np.any(source_stops <= source_starts):
        raise ValueError("Invalid source segment mapping")
    if not np.array_equal(stops - starts, source_stops - source_starts):
        raise ValueError("Source/cache segment lengths differ")
    if int(stops[-1]) != specific["sequence_shape"][1]:
        raise ValueError("Segment mapping does not cover component sequence")
    if specific["sequence_shape"][0] != len(specific["ch_names"]):
        raise ValueError("Component channel count mismatch")
    source_mode = str(metadata.get("mode", ""))
    if source_mode not in SUPPORTED_MODES:
        raise ValueError(f"Invalid Script 21 mode: {source_mode!r}")
    requested_mode = str(cfg["wiener"].get("mode", "frequency"))
    if source_mode != requested_mode:
        raise ValueError(
            f"Script 21 mode mismatch: expected {requested_mode!r}, "
            f"found {source_mode!r}"
        )
    return {
        **specific, "label": label, "split": str(specific["split"]),
        "source_mode": source_mode,
        "specific_path": component_paths["specific"],
        "coherent_path": component_paths["coherent"],
        "metadata_path": component_paths["metadata"],
    }


def _epoch_fingerprint(header: dict[str, Any], cfg: dict) -> str:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_fingerprint": header["fingerprint"],
        "epoch_length_sec": float(cfg["preprocessing"]["epoch_length_sec"]),
        "artifact_threshold_uv": float(cfg["preprocessing"]["artifact_threshold_uv"]),
        "rejection_policy": "raw_shared_mask",
        "standard_19": cfg["channels"]["standard_19"],
        "missing_channel_policy": "zero_fill_standard_19",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _output_cache_matches(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "raw", "specific", "coherent", COMBINED_CONDITION,
                "ch_names", "specific_coherent_ch_names", "fingerprint",
                "schema_version", "n_candidate_epochs", "n_rejected_epochs",
                "n_epochs", "epoch_samples", "sfreq", "epoch_segment_indices",
                "epoch_start_samples", "epoch_start_sec", "record_cache_id",
            }
            missing = sorted(required - set(data.files))
            if missing:
                raise CacheMismatch(f"Incomplete epoch cache {path}: {missing}")
            if int(_scalar(data, "schema_version")) != CACHE_SCHEMA_VERSION:
                raise CacheMismatch(f"Epoch cache schema mismatch: {path}")
            if str(_scalar(data, "fingerprint")) != fingerprint:
                raise CacheMismatch(f"Epoch cache mismatch: {path}; re-run with --force")
            n_epochs = int(_scalar(data, "n_epochs"))
            epoch_samples = int(_scalar(data, "epoch_samples"))
            ch_names = [str(value) for value in data["ch_names"]]
            base_shape = (n_epochs, len(ch_names), epoch_samples)
            for condition in ("raw", "specific", "coherent"):
                if data[condition].shape != base_shape or data[condition].dtype != np.float32:
                    raise CacheMismatch(f"Invalid {condition} array in {path}")
            if data[COMBINED_CONDITION].shape != (
                n_epochs, 2 * len(ch_names), epoch_samples
            ) or data[COMBINED_CONDITION].dtype != np.float32:
                raise CacheMismatch(f"Invalid combined array in {path}")
            for key, dtype in (
                ("epoch_start_samples", np.int64),
                ("epoch_start_sec", np.float64),
                ("epoch_segment_indices", np.int64),
            ):
                if data[key].shape != (n_epochs,) or data[key].dtype != dtype:
                    raise CacheMismatch(f"Invalid {key} metadata in {path}")
            return {
                key: int(_scalar(data, key))
                for key in ("n_candidate_epochs", "n_rejected_epochs", "n_epochs", "epoch_samples")
            }
    except CacheMismatch:
        raise
    except Exception as exc:
        raise CacheMismatch(f"Unreadable epoch cache {path}: {exc}") from exc


def _load_sequence(path: Path, expected_component: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if str(_scalar(data, "component")) != expected_component:
            raise ValueError(f"Component mismatch in {path}")
        sequence = np.asarray(data["sequence"], dtype=np.float32)
    if sequence.ndim != 2 or not np.isfinite(sequence).all():
        raise ValueError(f"Invalid component sequence in {path}")
    return sequence


def _expand_channels(
    epochs: np.ndarray, present: list[str], standard: list[str],
) -> np.ndarray:
    expanded = np.zeros(
        (epochs.shape[0], len(standard), epochs.shape[2]), dtype=np.float32
    )
    positions = {channel: index for index, channel in enumerate(standard)}
    for source_index, channel in enumerate(present):
        expanded[:, positions[channel], :] = epochs[:, source_index, :]
    return expanded


def _extract_epochs(
    specific: np.ndarray, coherent: np.ndarray, header: dict[str, Any], cfg: dict,
) -> dict[str, Any]:
    if specific.shape != coherent.shape:
        raise ValueError("Specific/coherent sequence shape mismatch")
    raw = specific + coherent
    sfreq = float(header["sfreq"])
    epoch_samples = int(float(cfg["preprocessing"]["epoch_length_sec"]) * sfreq)
    threshold = float(cfg["preprocessing"]["artifact_threshold_uv"])
    accepted_raw: list[np.ndarray] = []
    accepted_specific: list[np.ndarray] = []
    accepted_coherent: list[np.ndarray] = []
    absolute_starts: list[int] = []
    segment_indices: list[int] = []
    n_candidates = 0
    for segment_index, (source_start, cache_start, cache_stop) in enumerate(zip(
        header["segment_source_start_samples"],
        header["segment_cache_start_samples"],
        header["segment_cache_stop_samples"],
    )):
        length = int(cache_stop - cache_start)
        for local_start in range(0, length - epoch_samples + 1, epoch_samples):
            n_candidates += 1
            start = int(cache_start + local_start)
            stop = start + epoch_samples
            raw_epoch = raw[:, start:stop]
            if np.max(np.abs(raw_epoch)) > threshold:
                continue
            accepted_raw.append(raw_epoch)
            accepted_specific.append(specific[:, start:stop])
            accepted_coherent.append(coherent[:, start:stop])
            absolute_starts.append(int(source_start + local_start))
            segment_indices.append(segment_index)
    present = list(header["ch_names"])
    standard = list(cfg["channels"]["standard_19"])
    present_shape = (0, len(present), epoch_samples)
    raw_present = (
        np.stack(accepted_raw).astype(np.float32, copy=False)
        if accepted_raw else np.empty(present_shape, dtype=np.float32)
    )
    specific_present = (
        np.stack(accepted_specific).astype(np.float32, copy=False)
        if accepted_specific else np.empty(present_shape, dtype=np.float32)
    )
    coherent_present = (
        np.stack(accepted_coherent).astype(np.float32, copy=False)
        if accepted_coherent else np.empty(present_shape, dtype=np.float32)
    )
    raw_epochs = _expand_channels(raw_present, present, standard)
    specific_epochs = _expand_channels(specific_present, present, standard)
    coherent_epochs = _expand_channels(coherent_present, present, standard)
    np.testing.assert_allclose(
        specific_epochs + coherent_epochs, raw_epochs, rtol=1e-6, atol=1e-5
    )
    combined = np.concatenate([specific_epochs, coherent_epochs], axis=1)
    starts = np.asarray(absolute_starts, dtype=np.int64)
    return {
        "raw": raw_epochs, "specific": specific_epochs,
        "coherent": coherent_epochs, COMBINED_CONDITION: combined,
        "epoch_start_samples": starts,
        "epoch_start_sec": starts.astype(np.float64) / sfreq,
        "epoch_segment_indices": np.asarray(segment_indices, dtype=np.int64),
        "epoch_samples": epoch_samples, "n_candidate_epochs": n_candidates,
        "n_rejected_epochs": n_candidates - len(accepted_raw),
    }


def _process_one(args: tuple) -> dict[str, Any]:
    paths, output_root_str, cfg, force = args
    evaluation_id = str(paths["evaluation_id"])
    record_cache_id = str(paths["record_cache_id"])
    output_path = Path(output_root_str) / evaluation_id / f"{record_cache_id}.npz"
    manifest = {
        "evaluation_id": evaluation_id, "record_cache_id": record_cache_id,
        "specific_path": str(paths["specific"]),
        "coherent_path": str(paths["coherent"]),
        "metadata_path": str(paths["metadata"]), "output_path": str(output_path),
    }
    header: dict[str, Any] = {}
    try:
        header = _read_header(paths, cfg)
        fingerprint = _epoch_fingerprint(header, cfg)
        if not force:
            cached = _output_cache_matches(output_path, fingerprint)
            if cached is not None:
                return {
                    **manifest, **{key: header[key] for key in IDENTITY_FIELDS},
                    **cached, "status": "cached", "cache_hit": True,
                    "fingerprint": fingerprint, "source_mode": header["source_mode"],
                    "sfreq": float(header["sfreq"]), "error": "",
                }
        extracted = _extract_epochs(
            _load_sequence(header["specific_path"], "specific"),
            _load_sequence(header["coherent_path"], "coherent"),
            header, cfg,
        )
        n_epochs = int(extracted["raw"].shape[0])
        if n_epochs == 0:
            output_path.unlink(missing_ok=True)
            return {
                **manifest, **{key: header[key] for key in IDENTITY_FIELDS},
                "status": "skipped", "cache_hit": False,
                "fingerprint": fingerprint,
                "n_candidate_epochs": extracted["n_candidate_epochs"],
                "n_rejected_epochs": extracted["n_rejected_epochs"],
                "n_epochs": 0, "epoch_samples": extracted["epoch_samples"],
                "sfreq": float(header["sfreq"]), "source_mode": header["source_mode"],
                "error": "no valid epochs",
            }
        standard = list(cfg["channels"]["standard_19"])
        missing_channels = [ch for ch in standard if ch not in header["ch_names"]]
        _atomic_npz(
            output_path,
            raw=extracted["raw"], specific=extracted["specific"],
            coherent=extracted["coherent"],
            specific_coherent=extracted[COMBINED_CONDITION],
            epoch_start_samples=extracted["epoch_start_samples"],
            epoch_start_sec=extracted["epoch_start_sec"],
            epoch_segment_indices=extracted["epoch_segment_indices"],
            label=np.asarray(int(header["label"]), dtype=np.int8),
            class_name=np.asarray(str(header["class_name"])),
            split=np.asarray(str(header["split"])),
            dataset_name=np.asarray("tuep"),
            patient_id=np.asarray(str(header["patient_id"])),
            recording_id=np.asarray(str(header["recording_id"])),
            evaluation_id=np.asarray(str(header["evaluation_id"])),
            subject_id=np.asarray(str(header["evaluation_id"])),
            source_partition=np.asarray(str(header["source_partition"])),
            record_cache_id=np.asarray(record_cache_id),
            ch_names=np.asarray(standard, dtype="U"),
            source_ch_names=np.asarray(header["ch_names"], dtype="U"),
            missing_channels=np.asarray(missing_channels, dtype="U"),
            specific_coherent_ch_names=np.asarray(
                _combined_channel_names(standard), dtype="U"
            ),
            sfreq=np.asarray(float(header["sfreq"])),
            epoch_length_sec=np.asarray(float(cfg["preprocessing"]["epoch_length_sec"])),
            epoch_samples=np.asarray(extracted["epoch_samples"]),
            artifact_threshold_uv=np.asarray(
                float(cfg["preprocessing"]["artifact_threshold_uv"])
            ),
            rejection_policy=np.asarray("raw_shared_mask"),
            missing_channel_policy=np.asarray("zero_fill_standard_19"),
            n_candidate_epochs=np.asarray(extracted["n_candidate_epochs"]),
            n_rejected_epochs=np.asarray(extracted["n_rejected_epochs"]),
            n_epochs=np.asarray(n_epochs),
            source_fingerprint=np.asarray(header["fingerprint"]),
            source_mode=np.asarray(header["source_mode"]),
            fingerprint=np.asarray(fingerprint),
            schema_version=np.asarray(CACHE_SCHEMA_VERSION),
        )
        return {
            **manifest, **{key: header[key] for key in IDENTITY_FIELDS},
            "status": "processed", "cache_hit": False,
            "fingerprint": fingerprint,
            "n_candidate_epochs": extracted["n_candidate_epochs"],
            "n_rejected_epochs": extracted["n_rejected_epochs"],
            "n_epochs": n_epochs, "epoch_samples": extracted["epoch_samples"],
            "sfreq": float(header["sfreq"]), "source_mode": header["source_mode"],
            "error": "",
        }
    except Exception as exc:
        identity = {key: header.get(key, "") for key in IDENTITY_FIELDS}
        return {
            **manifest, **identity, "status": "failed", "cache_hit": False,
            "fingerprint": "", "n_candidate_epochs": np.nan,
            "n_rejected_epochs": np.nan, "n_epochs": np.nan,
            "epoch_samples": np.nan, "sfreq": np.nan,
            "source_mode": header.get("source_mode", ""),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _run_jobs(jobs: list[tuple], workers: int) -> list[dict[str, Any]]:
    if workers == 1:
        return [_process_one(job) for job in tqdm(jobs, desc="TUEP paired epochs")]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=min(workers, len(jobs)), mp_context=get_context("spawn")
    ) as executor:
        futures = [executor.submit(_process_one, job) for job in jobs]
        with tqdm(total=len(futures), desc="TUEP paired epochs") as progress:
            for future in as_completed(futures):
                results.append(future.result())
                progress.update(1)
    return results


def main(
    config_path: str = "configs/default.yaml", *, input_dir: str | None = None,
    output_dir: str | None = None, mode: str | None = None, workers: int = 1,
    force: bool = False,
) -> pd.DataFrame:
    if workers < 1:
        raise ValueError("--workers must be >= 1")
    cfg = deepcopy(load_config(config_path))
    if active_dataset_name(cfg) != "tuep":
        raise ValueError("Script 22 requires dataset.active: tuep")
    effective_mode = str(mode or cfg["wiener"].get("mode", "frequency"))
    if effective_mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported Wiener mode: {effective_mode!r}")
    cfg["wiener"]["mode"] = effective_mode
    input_root = (
        Path(input_dir).expanduser().resolve() if input_dir else
        Path(cfg["paths"]["cache_dir"]) / f"tuep_continuous_wiener_{effective_mode}"
    )
    output_root = (
        Path(output_dir).expanduser().resolve() if output_dir else input_root / "epochs"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    inputs = discover_inputs(input_root)
    rows = _run_jobs([(paths, str(output_root), cfg, force) for paths in inputs], workers)
    rows.sort(key=lambda row: (str(row["evaluation_id"]), str(row["record_cache_id"])))
    manifest = pd.DataFrame(rows)
    _atomic_dataframe(manifest, output_root / "manifest.csv")
    failures = manifest[manifest["status"] == "failed"]
    if not failures.empty:
        examples = "; ".join(
            f"{row.recording_id}: {row.error}" for row in failures.head(3).itertuples()
        )
        raise RuntimeError(
            f"{len(failures)} TUEP epoch caches failed; see {output_root / 'manifest.csv'}. "
            f"First errors: {examples}"
        )
    resolved = deepcopy(cfg)
    resolved["tuep_component_epochs"] = {
        "input_dir": str(input_root), "output_dir": str(output_root),
        "requested_mode": effective_mode,
        "source_modes": sorted(set(manifest["source_mode"].astype(str))),
        "rejection_policy": "raw_shared_mask",
        "missing_channel_policy": "zero_fill_standard_19",
        "conditions": ["raw", "specific", "coherent", COMBINED_CONDITION],
        "specific_coherent_layout": "channel_axis_specific_then_coherent",
    }
    _atomic_yaml(resolved, output_root / "config_resolved.yaml")
    counts = manifest["status"].value_counts().to_dict()
    print(
        "Done. "
        f"processed={counts.get('processed', 0)}, cached={counts.get('cached', 0)}, "
        f"skipped={counts.get('skipped', 0)}, failed={counts.get('failed', 0)}."
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--mode", choices=SUPPORTED_MODES)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    main(
        arguments.config, input_dir=arguments.input_dir,
        output_dir=arguments.output_dir, mode=arguments.mode,
        workers=arguments.workers, force=arguments.force,
    )
