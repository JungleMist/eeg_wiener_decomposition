"""Atomic, fingerprinted artifacts. Pickle-free array caches."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

import numpy as np

from .types import PreparedRecording

SCHEMA_VERSION = 1


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def code_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "auto_ecmad").rglob("*.py"))
    paths += [root / relative for relative in (
        "decomposition/wiener.py", "decomposition/wiener_phasegated.py",
        "decomposition/phase_gate.py", "preprocessing/continuous.py",
        "ml/cnn_model.py", "ml/erp_eegnet.py", "config/settings.py",
    )]
    return fingerprint({str(p.relative_to(root)): sha256_file(p) for p in sorted(paths)})


def atomic_write(path: Path, writer: Callable) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    payload = json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False).encode()
    atomic_write(path, lambda handle: handle.write(payload))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_csv(path: Path, frame) -> None:
    atomic_write(path, lambda handle: handle.write(frame.to_csv(index=False).encode()))


def save_recording(path: Path, record: PreparedRecording) -> None:
    record.validate()
    # Do not deepcopy entire continuous recordings merely to serialize metadata.
    values = vars(record).copy()
    metadata = {k: values.pop(k) for k in ("subject_id", "recording_id", "sfreq", "channel_names", "diagnostics")}
    metadata["labeled"] = record.y is not None
    if record.y is None:
        values.pop("y")
    values["metadata"] = np.asarray(json.dumps(jsonable(metadata)))
    atomic_write(path, lambda handle: np.savez_compressed(handle, **values))


def load_recording(path: Path) -> PreparedRecording:
    with np.load(path, allow_pickle=False) as cache:
        meta = json.loads(str(cache["metadata"].item()))
        labeled = meta.pop("labeled")
        meta["channel_names"] = tuple(meta["channel_names"])
        arrays = {k: cache[k].copy() for k in ("data_uv", "samples", "offsets", "baseline_mask", "trial_ids")}
        result = PreparedRecording(**meta, **arrays, y=cache["y"].copy() if labeled else None)
    result.validate()
    return result
