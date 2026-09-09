"""Tests for Script 22 TUEP component epoch extraction."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from eeg_bg.config.settings import load_config


SCRIPT = Path("scripts/22_extract_tuep_component_epochs.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("script22_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cfg() -> dict:
    return load_config("configs/default.yaml")


def _write_script21_cache(root: Path) -> dict[str, Path | str]:
    cfg = _cfg()
    present = cfg["channels"]["standard_19"][:-1]
    sfreq = 125.0
    epoch_samples = 2500
    specific = np.full((len(present), epoch_samples * 2), 10.0, dtype=np.float32)
    coherent = np.zeros_like(specific)
    specific[:, epoch_samples:] = 250.0
    evaluation_id = "00_aaaaa001"
    cache_id = "record-cache"
    paths = {
        "evaluation_id": evaluation_id, "record_cache_id": cache_id,
        "specific": root / "specific" / evaluation_id / f"{cache_id}.npz",
        "coherent": root / "coherent" / evaluation_id / f"{cache_id}.npz",
        "metadata": root / "metadata" / evaluation_id / f"{cache_id}.json",
    }
    common = {
        "ch_names": np.asarray(present), "sfreq": np.asarray(sfreq),
        "fingerprint": np.asarray("source-fingerprint"),
        "schema_version": np.asarray(1), "record_cache_id": np.asarray(cache_id),
        "segment_source_start_samples": np.asarray([0, 10000], dtype=np.int64),
        "segment_source_stop_samples": np.asarray([2500, 12500], dtype=np.int64),
        "segment_cache_start_samples": np.asarray([0, 2500], dtype=np.int64),
        "segment_cache_stop_samples": np.asarray([2500, 5000], dtype=np.int64),
        "dataset_name": np.asarray("tuep"), "patient_id": np.asarray("aaaaa001"),
        "recording_id": np.asarray("aaaaa001_s001_t000"),
        "evaluation_id": np.asarray(evaluation_id),
        "class_name": np.asarray("epilepsy"), "label": np.asarray(0),
        "split": np.asarray("train"), "source_partition": np.asarray(""),
    }
    for component, sequence in (("specific", specific), ("coherent", coherent)):
        path = Path(paths[component])
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, sequence=sequence, component=np.asarray(component), **common)
    metadata_path = Path(paths["metadata"])
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps({
        "schema_version": 1, "fingerprint": "source-fingerprint",
        "mode": "frequency", "record_cache_id": cache_id,
        "n_segments": 2, "n_times": 5000, "sfreq": sfreq,
        "ch_names": present,
    }), encoding="utf-8")
    return paths


def test_epochs_do_not_cross_background_gaps_and_zero_fill(tmp_path):
    module = _load_script()
    cfg = _cfg()
    paths = _write_script21_cache(tmp_path / "source")
    output = tmp_path / "epochs"
    result = module._process_one((paths, str(output), cfg, False))

    assert result["status"] == "processed"
    assert result["n_candidate_epochs"] == 2
    assert result["n_rejected_epochs"] == 1
    path = output / "00_aaaaa001" / "record-cache.npz"
    with np.load(path, allow_pickle=False) as data:
        assert data["raw"].shape == (1, 19, 2500)
        assert data["specific_coherent"].shape == (1, 38, 2500)
        np.testing.assert_array_equal(data["epoch_start_samples"], [0])
        np.testing.assert_array_equal(data["epoch_segment_indices"], [0])
        np.testing.assert_array_equal(data["raw"][:, -1], 0.0)
        assert list(data["missing_channels"]) == [cfg["channels"]["standard_19"][-1]]
        np.testing.assert_allclose(data["specific"] + data["coherent"], data["raw"])


def test_second_segment_uses_absolute_source_time(tmp_path):
    module = _load_script()
    cfg = _cfg()
    paths = _write_script21_cache(tmp_path / "source")
    header = module._read_header(paths, cfg)
    specific = module._load_sequence(Path(paths["specific"]), "specific")
    coherent = module._load_sequence(Path(paths["coherent"]), "coherent")
    specific[:, :2500] = 250.0
    specific[:, 2500:] = 10.0
    extracted = module._extract_epochs(specific, coherent, header, cfg)
    np.testing.assert_array_equal(extracted["epoch_start_samples"], [10000])
    np.testing.assert_array_equal(extracted["epoch_segment_indices"], [1])


def test_rejects_requested_mode_mismatch(tmp_path):
    module = _load_script()
    paths = _write_script21_cache(tmp_path / "source")
    metadata_path = Path(paths["metadata"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["mode"] = "phasegated"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="mode mismatch"):
        module._read_header(paths, _cfg())
