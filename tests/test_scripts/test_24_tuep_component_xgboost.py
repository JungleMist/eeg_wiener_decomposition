"""Focused adapter tests for Script 24."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


SCRIPT = Path("scripts/24_train_tuep_component_xgboost.py")
CHANNELS = [
    "FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4", "T3", "T4",
    "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz",
]


def _load_script():
    spec = importlib.util.spec_from_file_location("script24_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_cache(
    path: Path, split: str = "train", label: int = 0,
    patient_id: str = "p1", evaluation_id: str = "00_p1",
) -> None:
    rng = np.random.default_rng(4)
    arrays = {
        key: rng.normal(size=(2, 19, 128)).astype(np.float32)
        for key in ("raw", "specific", "coherent")
    }
    arrays["specific_coherent"] = np.concatenate(
        [arrays["specific"], arrays["coherent"]], axis=1
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, **arrays, ch_names=np.asarray(CHANNELS),
        specific_coherent_ch_names=np.asarray([
            *(f"specific::{channel}" for channel in CHANNELS),
            *(f"coherent::{channel}" for channel in CHANNELS),
        ]),
        sfreq=np.asarray(125.0), split=np.asarray(split), label=np.asarray(label),
        class_name=np.asarray("epilepsy" if label == 0 else "control"),
        patient_id=np.asarray(patient_id), recording_id=np.asarray(f"{patient_id}-recording"),
        record_cache_id=np.asarray(path.stem), evaluation_id=np.asarray(evaluation_id),
        dataset_name=np.asarray("tuep"),
        fingerprint=np.asarray("source-1"), source_mode=np.asarray("frequency"),
        schema_version=np.asarray(1), n_epochs=np.asarray(2),
    )


def test_extract_file_builds_single_and_combined_feature_dimensions(tmp_path):
    module = _load_script()
    path = tmp_path / "00_p1" / "record-cache.npz"
    _write_cache(path)
    tail = ("train", 125.0, 64, (0.5, 40.0), "base211", 64)
    outputs = {
        condition: module._extract_file((0, str(path), condition, *tail))
        for condition in module.CONDITIONS
    }
    assert all(outputs[key][1][0].shape == (211,) for key in module.BASE_CONDITIONS)
    assert outputs["specific_coherent"][1][0].shape == (422,)
    np.testing.assert_allclose(
        outputs["specific_coherent"][1][0],
        np.concatenate([outputs["specific"][1][0], outputs["coherent"][1][0]]),
    )
    assert outputs["raw"][3] == ["00_p1", "00_p1"]
    assert outputs["raw"][6] == ["tuep", "tuep"]


def test_metrics_are_patient_level_and_include_auprc():
    module = _load_script()
    frame = pd.DataFrame({
        "true_label": [0, 0, 1, 1],
        "pred_proba": [0.1, 0.3, 0.7, 0.9],
    })
    metrics = module._metrics(frame, 0.5)
    assert metrics["aggregation_unit"] == "patient"
    assert metrics["positive_class"] == "control"
    assert metrics["auprc"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0


def test_combined_profile_names_are_namespaced():
    module = _load_script()
    names = module._condition_feature_names("base211_conn80", "specific_coherent")
    assert len(names) == 582
    assert all(name.startswith("specific::") for name in names[:291])
    assert all(name.startswith("coherent::") for name in names[291:])


def test_source_inventory_uses_manifest_and_validates_patient_splits(tmp_path):
    module = _load_script()
    rows = []
    for split in ("train", "val", "test"):
        for label in (0, 1):
            patient_id = f"{split}-{label}"
            evaluation_id = f"{label:02d}_{patient_id}"
            path = tmp_path / evaluation_id / f"cache-{split}-{label}.npz"
            _write_cache(path, split, label, patient_id, evaluation_id)
            rows.append({
                "evaluation_id": evaluation_id,
                "record_cache_id": path.stem,
                "output_path": str(path), "status": "processed",
            })
    pd.DataFrame(rows).to_csv(tmp_path / "manifest.csv", index=False)
    (tmp_path / "config_resolved.yaml").write_text(
        yaml.safe_dump({"dataset": {"active": "tuep"}}), encoding="utf-8"
    )
    orphan = tmp_path / "features" / "orphan.npz"
    orphan.parent.mkdir(parents=True)
    np.savez(orphan, ignored=np.asarray(1))
    cfg = {
        "channels": {"standard_19": CHANNELS},
        "preprocessing": {"target_sfreq": 125.0},
        "wiener": {"mode": "frequency"},
    }
    files, fingerprint, metadata = module._source_inventory(tmp_path, cfg)
    assert len(files) == 6
    assert orphan not in files
    assert len(fingerprint) == 64
    assert metadata["n_patients"] == 6
