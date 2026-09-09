"""Cache continuous TUEP background ECMAD specific/coherent sequences."""
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
from eeg_bg.io.cache import make_cache_key, make_wiener_cache_fingerprint
from eeg_bg.io.dataset import (
    active_dataset_name,
    assign_dataset_splits,
    build_recording_index,
    get_recording_intervals,
)
from eeg_bg.io.edf_reader import load_edf
from eeg_bg.preprocessing.continuous import wiener_continuous_raw


CACHE_SCHEMA_VERSION = 1
SUPPORTED_MODES = ("frequency", "phasegated", "zerophase")
COMPONENTS = ("specific", "coherent")
IDENTITY_FIELDS = (
    "dataset_name", "patient_id", "recording_id", "evaluation_id",
    "class_name", "label", "split", "source_partition",
)


class CacheFingerprintMismatch(ValueError):
    """Raised when an existing continuous cache is incompatible."""


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


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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


def _cache_paths(
    output_root: Path, evaluation_id: str, record_cache_id: str,
) -> dict[str, Path]:
    paths = {
        component: output_root / component / evaluation_id / f"{record_cache_id}.npz"
        for component in COMPONENTS
    }
    paths["metadata"] = (
        output_root / "metadata" / evaluation_id / f"{record_cache_id}.json"
    )
    return paths


def _file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path.resolve()), "exists": False}
    stat = path.stat()
    return {
        "path": str(path.resolve()), "exists": True,
        "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
    }


def _recording_fingerprint(
    row: dict, cfg: dict, mode: str, ch_names: list[str], sfreq: float,
    segments: list[tuple[int, int]],
) -> str:
    edf_path = Path(row["edf_path"])
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source": _file_identity(edf_path),
        "annotation": _file_identity(edf_path.with_suffix(".csv_bi")),
        "identity": {key: row[key] for key in IDENTITY_FIELDS},
        "segments": segments,
        "ch_names": ch_names,
        "sfreq": float(sfreq),
        "preprocessing": cfg["preprocessing"],
        "standard_19": cfg["channels"]["standard_19"],
        "split": cfg["split"],
        "wiener_mode": mode,
        "wiener_config_fingerprint": make_wiener_cache_fingerprint(cfg, mode),
    }
    encoded = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _scalar(data: Any, key: str) -> Any:
    value = data[key]
    return value.item() if np.asarray(value).ndim == 0 else value


def _complete_cache_matches(paths: dict[str, Path], fingerprint: str) -> bool:
    if not all(paths[key].is_file() for key in (*COMPONENTS, "metadata")):
        return False
    try:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if int(metadata.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
            raise CacheFingerprintMismatch("Continuous cache schema mismatch")
        cached_fingerprints = [str(metadata.get("fingerprint", ""))]
        expected_shape = (
            int(metadata.get("n_channels", -1)), int(metadata.get("n_times", -1))
        )
        for component in COMPONENTS:
            with np.load(paths[component], allow_pickle=False) as data:
                required = {
                    "sequence", "component", "ch_names", "sfreq", "fingerprint",
                    "schema_version", "record_cache_id",
                    "segment_source_start_samples", "segment_source_stop_samples",
                    "segment_cache_start_samples", "segment_cache_stop_samples",
                    *IDENTITY_FIELDS,
                }
                missing = sorted(required - set(data.files))
                if missing:
                    raise CacheFingerprintMismatch(
                        f"Incomplete {component} cache: missing={missing}"
                    )
                if str(_scalar(data, "component")) != component:
                    raise CacheFingerprintMismatch("Component name mismatch")
                if int(_scalar(data, "schema_version")) != CACHE_SCHEMA_VERSION:
                    raise CacheFingerprintMismatch("Continuous cache schema mismatch")
                if data["sequence"].shape != expected_shape:
                    raise CacheFingerprintMismatch("Continuous cache shape mismatch")
                if data["sequence"].dtype != np.float32:
                    raise CacheFingerprintMismatch("Continuous cache dtype mismatch")
                if [str(value) for value in data["ch_names"]] != metadata.get(
                    "ch_names"
                ):
                    raise CacheFingerprintMismatch("Continuous cache channel mismatch")
                cached_fingerprints.append(str(_scalar(data, "fingerprint")))
    except CacheFingerprintMismatch:
        raise
    except Exception as exc:
        raise CacheFingerprintMismatch(f"Unreadable continuous cache: {exc}") from exc
    if any(value != fingerprint for value in cached_fingerprints):
        raise CacheFingerprintMismatch(
            "Continuous Wiener cache configuration/source mismatch; re-run with --force"
        )
    return True


def _identity_payload(row: dict) -> dict[str, np.ndarray]:
    return {key: np.asarray(row[key]) for key in IDENTITY_FIELDS}


def _active_channels(diagnostics: dict) -> set[str]:
    active: set[str] = set()
    for window in diagnostics.get("window_diagnostics", []):
        active.update(str(channel) for channel in window.get("channel_sources", {}))
    return active


def _compact_diagnostics(diagnostics: dict) -> dict:
    return {
        key: _jsonable(value)
        for key, value in diagnostics.items()
        if key != "window_diagnostics"
    }


def _eligible_sample_segments(
    intervals: list[tuple[float, float]], sfreq: float, n_times: int,
    epoch_samples: int,
) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    for start_sec, stop_sec in intervals:
        start = max(0, min(n_times, int(start_sec * sfreq)))
        stop = max(start, min(n_times, int(stop_sec * sfreq)))
        if stop - start >= epoch_samples:
            segments.append((start, stop))
    return segments


def _manifest_base(
    row: dict, record_cache_id: str, paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        **{key: row[key] for key in IDENTITY_FIELDS},
        "record_cache_id": record_cache_id,
        "edf_path": row["edf_path"],
        "specific_path": str(paths["specific"]),
        "coherent_path": str(paths["coherent"]),
        "metadata_path": str(paths["metadata"]),
    }


def _process_one(args: tuple) -> dict[str, Any]:
    row, cfg, output_root_str, mode, force = args
    output_root = Path(output_root_str)
    edf_path = Path(row["edf_path"])
    record_cache_id = make_cache_key(edf_path, 0.0, cfg)
    paths = _cache_paths(output_root, str(row["evaluation_id"]), record_cache_id)
    manifest = _manifest_base(row, record_cache_id, paths)
    try:
        data_uv, ch_names, sfreq = load_edf(edf_path, cfg)
        expected = list(cfg["channels"]["standard_19"])
        if len(expected) != len(set(expected)):
            raise ValueError("channels.standard_19 must contain unique channels")
        if not ch_names or any(channel not in expected for channel in ch_names):
            raise ValueError(f"Unexpected TUEP channel order: {ch_names}")
        duration = data_uv.shape[1] / sfreq
        intervals = get_recording_intervals(row, cfg, duration)
        epoch_samples = int(float(cfg["preprocessing"]["epoch_length_sec"]) * sfreq)
        segments = _eligible_sample_segments(
            intervals, sfreq, data_uv.shape[1], epoch_samples
        )
        fingerprint = _recording_fingerprint(
            row, cfg, mode, ch_names, sfreq, segments
        )
        if not segments:
            for path in paths.values():
                path.unlink(missing_ok=True)
            return {
                **manifest, "status": "skipped", "cache_hit": False,
                "fingerprint": fingerprint, "n_channels": len(ch_names),
                "n_segments": 0, "n_times": 0, "sfreq": float(sfreq),
                "duration_sec": 0.0, "error": "no eligible background segment",
            }
        if not force and _complete_cache_matches(paths, fingerprint):
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            return {
                **manifest, "status": "cached", "cache_hit": True,
                "fingerprint": fingerprint, "n_channels": metadata["n_channels"],
                "n_segments": metadata["n_segments"], "n_times": metadata["n_times"],
                "sfreq": metadata["sfreq"], "duration_sec": metadata["duration_sec"],
                "error": "",
            }

        import mne

        specific_parts: list[np.ndarray] = []
        coherent_parts: list[np.ndarray] = []
        source_parts: list[np.ndarray] = []
        cache_starts: list[int] = []
        cache_stops: list[int] = []
        segment_metadata: list[dict[str, Any]] = []
        cache_cursor = 0
        active_union: set[str] = set()
        for segment_index, (source_start, source_stop) in enumerate(segments):
            source_uv = np.asarray(
                data_uv[:, source_start:source_stop], dtype=np.float64
            )
            info = mne.create_info(
                ch_names=ch_names, sfreq=float(sfreq),
                ch_types=["eeg"] * len(ch_names),
            )
            raw = mne.io.RawArray(source_uv * 1e-6, info, verbose=False)
            specific_raw, diagnostics = wiener_continuous_raw(
                raw, cfg,
                subject_id=f"{row['evaluation_id']}:{record_cache_id}:{segment_index}",
            )
            if (
                specific_raw.ch_names != ch_names
                or specific_raw.n_times != source_uv.shape[1]
                or not np.isclose(specific_raw.info["sfreq"], sfreq)
            ):
                raise ValueError("Continuous Wiener output is not aligned with input")
            specific_uv = specific_raw.get_data() * 1e6
            active = _active_channels(diagnostics)
            active_union.update(active)
            inactive_indices = [
                index for index, channel in enumerate(ch_names)
                if channel not in active
            ]
            coherent_uv = source_uv - specific_uv
            if inactive_indices:
                specific_uv[inactive_indices] = source_uv[inactive_indices]
                coherent_uv[inactive_indices] = 0.0
            if not np.isfinite(specific_uv).all() or not np.isfinite(coherent_uv).all():
                raise ValueError("Continuous Wiener output contains non-finite values")
            source_parts.append(source_uv)
            specific_parts.append(specific_uv)
            coherent_parts.append(coherent_uv)
            cache_starts.append(cache_cursor)
            cache_cursor += source_uv.shape[1]
            cache_stops.append(cache_cursor)
            segment_metadata.append({
                "segment_index": segment_index,
                "source_start_sample": source_start,
                "source_stop_sample": source_stop,
                "cache_start_sample": cache_starts[-1],
                "cache_stop_sample": cache_stops[-1],
                "duration_sec": (source_stop - source_start) / sfreq,
                "active_channels": sorted(active),
                "diagnostics": _compact_diagnostics(diagnostics),
            })

        source64 = np.concatenate(source_parts, axis=1)
        specific64 = np.concatenate(specific_parts, axis=1)
        coherent64 = np.concatenate(coherent_parts, axis=1)
        conservation_error = float(
            np.max(np.abs(source64 - specific64 - coherent64))
        )
        source32 = source64.astype(np.float32)
        specific32 = specific64.astype(np.float32)
        coherent32 = source32 - specific32
        float32_error = float(np.max(np.abs(source32 - specific32 - coherent32)))

        segment_arrays = {
            "segment_source_start_samples": np.asarray(
                [item[0] for item in segments], dtype=np.int64
            ),
            "segment_source_stop_samples": np.asarray(
                [item[1] for item in segments], dtype=np.int64
            ),
            "segment_cache_start_samples": np.asarray(cache_starts, dtype=np.int64),
            "segment_cache_stop_samples": np.asarray(cache_stops, dtype=np.int64),
        }
        common_payload = {
            "ch_names": np.asarray(ch_names, dtype="U"),
            "sfreq": np.asarray(float(sfreq)),
            "fingerprint": np.asarray(fingerprint),
            "schema_version": np.asarray(CACHE_SCHEMA_VERSION),
            "record_cache_id": np.asarray(record_cache_id),
            **segment_arrays,
            **_identity_payload(row),
        }
        paths["metadata"].unlink(missing_ok=True)
        _atomic_npz(
            paths["specific"], sequence=specific32,
            component=np.asarray("specific"), **common_payload,
        )
        _atomic_npz(
            paths["coherent"], sequence=coherent32,
            component=np.asarray("coherent"), **common_payload,
        )
        metadata = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "wiener_config_fingerprint": make_wiener_cache_fingerprint(cfg, mode),
            "mode": mode,
            "record_cache_id": record_cache_id,
            "source_path": str(edf_path.resolve()),
            "annotation": _file_identity(edf_path.with_suffix(".csv_bi")),
            "source_duration_sec": float(duration),
            "duration_sec": float(source32.shape[1] / sfreq),
            "n_channels": len(ch_names), "n_segments": len(segments),
            "n_times": int(source32.shape[1]), "sfreq": float(sfreq),
            "ch_names": ch_names,
            "missing_channels": [channel for channel in expected if channel not in ch_names],
            "active_channels": sorted(active_union),
            "max_abs_conservation_error_uv": conservation_error,
            "max_abs_float32_conservation_error_uv": float32_error,
            "segments": segment_metadata,
            "identity": {key: _jsonable(row[key]) for key in IDENTITY_FIELDS},
        }
        _atomic_json(metadata, paths["metadata"])
        return {
            **manifest, "status": "processed", "cache_hit": False,
            "fingerprint": fingerprint, "n_channels": len(ch_names),
            "n_segments": len(segments), "n_times": int(source32.shape[1]),
            "sfreq": float(sfreq), "duration_sec": float(source32.shape[1] / sfreq),
            "error": "",
        }
    except Exception as exc:
        return {
            **manifest, "status": "failed", "cache_hit": False,
            "fingerprint": "", "n_channels": np.nan, "n_segments": np.nan,
            "n_times": np.nan, "sfreq": np.nan, "duration_sec": np.nan,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _init_worker() -> None:
    import mne
    mne.set_log_level("ERROR")


def _run_jobs(jobs: list[tuple], workers: int) -> list[dict[str, Any]]:
    if workers == 1:
        return [_process_one(job) for job in tqdm(jobs, desc="TUEP continuous Wiener")]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=min(workers, len(jobs)), mp_context=get_context("spawn"),
        initializer=_init_worker,
    ) as executor:
        futures = [executor.submit(_process_one, job) for job in jobs]
        with tqdm(total=len(futures), desc="TUEP continuous Wiener") as progress:
            for future in as_completed(futures):
                results.append(future.result())
                progress.update(1)
    return results


def main(
    config_path: str = "configs/default.yaml", *, data_dir: str | None = None,
    cache_dir: str | None = None, mode: str | None = None, workers: int = 1,
    force: bool = False,
) -> pd.DataFrame:
    if workers < 1:
        raise ValueError("--workers must be >= 1")
    cfg = deepcopy(load_config(config_path))
    if data_dir is not None:
        cfg["paths"]["data_root"] = str(Path(data_dir).expanduser().resolve())
    if cache_dir is not None:
        cfg["paths"]["cache_dir"] = str(Path(cache_dir).expanduser().resolve())
    if active_dataset_name(cfg) != "tuep":
        raise ValueError("Script 21 requires dataset.active: tuep")
    effective_mode = str(mode or cfg["wiener"].get("mode", "frequency"))
    if effective_mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported Wiener mode: {effective_mode!r}")
    cfg["wiener"]["mode"] = effective_mode
    output_root = (
        Path(cfg["paths"]["cache_dir"])
        / f"tuep_continuous_wiener_{effective_mode}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    index = assign_dataset_splits(build_recording_index(cfg), cfg)
    if index.empty:
        raise RuntimeError(f"No TUEP EDF recordings found below {cfg['paths']['data_root']}")
    jobs = [
        (row.to_dict(), cfg, str(output_root), effective_mode, force)
        for _, row in index.iterrows()
    ]
    rows = _run_jobs(jobs, workers)
    rows.sort(key=lambda row: (str(row["evaluation_id"]), str(row["record_cache_id"])))
    manifest = pd.DataFrame(rows)
    _atomic_dataframe(manifest, output_root / "manifest.csv")
    counts = manifest["status"].value_counts().to_dict()
    print(
        "Done. "
        f"processed={counts.get('processed', 0)}, cached={counts.get('cached', 0)}, "
        f"skipped={counts.get('skipped', 0)}, failed={counts.get('failed', 0)}."
    )
    failures = manifest[manifest["status"] == "failed"]
    if not failures.empty:
        examples = "; ".join(
            f"{row.recording_id}: {row.error}" for row in failures.head(3).itertuples()
        )
        raise RuntimeError(
            f"{len(failures)} TUEP recordings failed; see {output_root / 'manifest.csv'}. "
            f"First errors: {examples}"
        )
    _atomic_yaml(cfg, output_root / "config_resolved.yaml")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument("--cache-dir")
    parser.add_argument("--mode", choices=SUPPORTED_MODES)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    main(
        arguments.config, data_dir=arguments.data_dir,
        cache_dir=arguments.cache_dir, mode=arguments.mode,
        workers=arguments.workers, force=arguments.force,
    )
