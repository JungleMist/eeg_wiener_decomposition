"""Tests for Script 21's interval-safe continuous TUEP cache."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT = Path("scripts/21_cache_tuep_continuous_wiener.py")
CHANNELS = [
    "FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4", "T3", "T4",
    "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz",
]


def _load_script():
    spec = importlib.util.spec_from_file_location("script21_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cfg(tmp_path: Path) -> dict:
    return {
        "paths": {"cache_dir": str(tmp_path), "data_root": str(tmp_path)},
        "dataset": {
            "active": "tuep",
            "tuep": {
                "reference_scheme": "ar", "montage_dir": "01_tcp_ar",
                "classes": {
                    "epilepsy": {"folder": "00_epilepsy", "label": 0},
                    "control": {"folder": "01_no_epilepsy", "label": 1},
                },
            },
        },
        "split": {"train": 0.7, "val": 0.1, "test": 0.2, "random_seed": 42},
        "preprocessing": {
            "target_sfreq": 1.0, "bandpass": [0.1, 0.4],
            "epoch_length_sec": 20.0, "artifact_threshold_uv": 200.0,
            "seizure_buffer_sec": 30.0,
        },
        "channels": {
            "standard_19": CHANNELS, "channel_groups": [["FP1", "FP2"]],
            "passthrough": CHANNELS[2:],
        },
        "wiener": {
            "mode": "frequency", "nperseg": 20, "coherence_threshold": 0.15,
            "coherent_gate_enabled": True, "coherent_gate_threshold_uv": 100.0,
            "filter_magnitude_threshold": 50.0,
            "overlap_policy": "coherence_weighted",
            "phase_gate_threshold_rad": 0.05, "freq_band": [0.1, 0.4],
            "protected_band_hz": None,
        },
    }


def _row(source: Path) -> dict:
    return {
        "dataset_name": "tuep", "patient_id": "aaaaa001",
        "recording_id": "aaaaa001_s001_t000", "evaluation_id": "00_aaaaa001",
        "class_name": "epilepsy", "label": 0, "split": "train",
        "source_partition": "", "edf_path": str(source),
    }


def test_eligible_segments_use_script01_integer_boundaries():
    module = _load_script()
    assert module._eligible_sample_segments(
        [(0.9, 20.9), (30.0, 49.9), (50.0, 80.0)], 1.0, 75, 20
    ) == [(0, 20), (50, 75)]


def test_processes_each_background_interval_independently(tmp_path, monkeypatch):
    module = _load_script()
    cfg = _cfg(tmp_path)
    source = tmp_path / "record.edf"
    source.write_bytes(b"edf")
    data = np.vstack([
        np.arange(80, dtype=float) + index for index in range(len(CHANNELS) - 1)
    ])
    calls = []

    monkeypatch.setattr(
        module, "load_edf", lambda path, config: (data.copy(), CHANNELS[:-1], 1.0)
    )
    monkeypatch.setattr(
        module, "get_recording_intervals",
        lambda row, config, duration: [(0.0, 25.0), (40.0, 70.0)],
    )

    def fake_wiener(raw, config, subject_id="recording"):
        calls.append((subject_id, raw.n_times))
        specific = raw.copy()
        specific._data[0] *= 0.5
        return specific, {"window_diagnostics": [{"channel_sources": {"FP1": []}}]}

    monkeypatch.setattr(module, "wiener_continuous_raw", fake_wiener)
    result = module._process_one((_row(source), cfg, str(tmp_path / "out"), "frequency", False))

    assert result["status"] == "processed"
    assert [length for _, length in calls] == [25, 30]
    paths = module._cache_paths(tmp_path / "out", "00_aaaaa001", result["record_cache_id"])
    with np.load(paths["specific"], allow_pickle=False) as specific, np.load(
        paths["coherent"], allow_pickle=False
    ) as coherent:
        np.testing.assert_array_equal(specific["segment_source_start_samples"], [0, 40])
        np.testing.assert_array_equal(specific["segment_cache_start_samples"], [0, 25])
        assert specific["sequence"].shape == (18, 55)
        np.testing.assert_allclose(specific["sequence"] + coherent["sequence"],
                                   np.concatenate([data[:, :25], data[:, 40:70]], axis=1))


def test_fingerprint_changes_when_annotation_changes(tmp_path):
    module = _load_script()
    cfg = _cfg(tmp_path)
    source = tmp_path / "record.edf"
    source.write_bytes(b"edf")
    sidecar = source.with_suffix(".csv_bi")
    sidecar.write_text("first", encoding="utf-8")
    first = module._recording_fingerprint(
        _row(source), cfg, "frequency", CHANNELS, 1.0, [(0, 20)]
    )
    sidecar.write_text("second-longer", encoding="utf-8")
    second = module._recording_fingerprint(
        _row(source), cfg, "frequency", CHANNELS, 1.0, [(0, 20)]
    )
    assert first != second
