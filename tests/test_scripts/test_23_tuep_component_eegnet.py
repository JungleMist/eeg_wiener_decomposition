"""Focused patient-level tests for Script 23."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

from eeg_bg.config.settings import load_config


SCRIPT = Path("scripts/23_train_tuep_component_eegnet.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("script23_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_record(path: Path, n_epochs: int, recording_id: str) -> dict:
    values = np.random.default_rng(n_epochs).normal(size=(n_epochs, 19, 64)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, raw=values, specific=values, coherent=np.zeros_like(values),
        specific_coherent=np.concatenate([values, np.zeros_like(values)], axis=1),
        epoch_start_samples=np.arange(n_epochs, dtype=np.int64) * 64,
        epoch_start_sec=np.arange(n_epochs, dtype=np.float64),
        epoch_segment_indices=np.zeros(n_epochs, dtype=np.int64),
    )
    return {
        "path": path, "recording_id": recording_id,
        "record_cache_id": path.stem, "n_epochs": n_epochs,
    }


def test_default_training_numbers_match_script19_config():
    cfg = load_config("configs/default.yaml")["ml"]["tuep_component_eegnet"]
    tuab = load_config("configs/tuab.yaml")["ml"]["tuab_component_eegnet"]
    numeric_keys = {
        "random_state", "num_workers", "F1", "D", "dropout", "lr",
        "weight_decay", "batch_size", "records_per_step", "max_epochs",
        "patience", "lr_patience", "lr_factor",
    }
    assert {key: cfg[key] for key in numeric_keys} == {
        key: tuab[key] for key in numeric_keys
    }


def test_patient_class_weight_ignores_record_and_epoch_counts():
    module = _load_script()
    patients = [
        {"label": 0, "n_records": 9, "n_epochs": 1000},
        {"label": 0, "n_records": 1, "n_epochs": 1},
        {"label": 1, "n_records": 20, "n_epochs": 5000},
    ]
    assert module.patient_class_weights(patients) == {0: 1.0, 1: 2.0}


def test_prediction_aggregates_multiple_recordings_once_per_patient(tmp_path):
    module = _load_script()
    records = [
        _write_record(tmp_path / "00_p1" / "r1.npz", 1, "recording-1"),
        _write_record(tmp_path / "00_p1" / "r2.npz", 2, "recording-2"),
    ]
    patient = {
        "evaluation_id": "00_p1", "patient_id": "p1", "label": 0,
        "class_name": "epilepsy", "split": "validation", "records": records,
        "n_records": 2, "n_epochs": 3, "ch_names": tuple(str(i) for i in range(19)),
        "epoch_samples": 64,
    }

    class ConstantModel(torch.nn.Module):
        def forward(self, values):
            return torch.full((len(values), 1), 0.25, device=values.device)

    patient_frame, epoch_frame = module._predict_patients(
        ConstantModel(), [patient], "raw", torch.device("cpu"), 2, 0, 1
    )
    assert len(patient_frame) == 1
    assert patient_frame.loc[0, "n_records"] == 2
    assert patient_frame.loc[0, "n_epochs"] == 3
    assert patient_frame.loc[0, "pred_proba"] == 0.25
    assert set(epoch_frame["recording_id"]) == {"recording-1", "recording-2"}
