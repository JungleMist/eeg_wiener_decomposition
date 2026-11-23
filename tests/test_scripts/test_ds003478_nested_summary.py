"""Check aggregation of sealed nested CV artifacts without training."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import summarize_ds003478_nested_cv as summary


def test_eegnet_shap_weights_subjects_and_models_equally(tmp_path):
    paths = []
    ids, labels, counts = ("a", "b"), (0, 1), (1, 2)
    names = ["specific:FP1", "coherent:FP1"]
    for subject, label, count in zip(ids, labels, counts, strict=True):
        target = tmp_path / subject
        result = target / "result.json"
        paths.append(result)
        sealed = {}
        for model in range(5):
            relative = Path(f"inner_{model}") / "outer"
            path = target / relative
            path.mkdir(parents=True)
            # The two subjects have different numbers of windows. Reversed
            # signs across time samples would cancel in a signed average.
            magnitude = model if subject == "a" else 2 * model + 2
            values = np.tile([[[magnitude, -magnitude], [2, -2]]], (count, 1, 1))
            array_file, index_file = "shap_test.npz", "shap_test_epochs.csv"
            np.savez(path / array_file, values=values, names=names)
            pd.DataFrame({
                "participant_id": subject, "label": label, "recording_id": subject,
                "epoch_start_sample": np.arange(count),
            }).to_csv(path / index_file, index=False)
            for filename in (array_file, index_file):
                sealed[str(relative / filename)] = "already checked by the caller"
        (target / "result_complete.json").write_text(json.dumps(sealed))
    cfg = {"channels": {"standard_19": ["FP1"]},
           "preprocessing": {"target_sfreq": 2, "epoch_length_sec": 1}}
    rows = pd.DataFrame({"participant_id": ids, "true_label": labels})
    keys = ["phase_0.02_specific", "phase_0.02_coherent"]
    importance = summary.summarize_shap(paths, rows, "eegnet", keys, cfg, dict(zip(ids, counts)))
    assert importance.name.tolist() == names
    assert importance.mean_abs_shap.tolist() == [4.0, 2.0]
    assert importance.n_epoch_model_pairs.tolist() == [15, 15]
    groups = summary.condition_shap_summary(importance, "eegnet", "phase_0.02_specific_coherent")
    assert groups.component.tolist() == ["all", "specific", "coherent"]
    assert groups.mean_abs_shap.tolist() == [3.0, 4.0, 2.0]

    # All five models must contribute to each participant's result.
    (tmp_path / "a" / "result_complete.json").write_text("{}")
    with pytest.raises(RuntimeError, match="Missing sealed SHAP outputs"):
        summary.summarize_shap(paths, rows, "eegnet", keys, cfg, dict(zip(ids, counts)))


def test_cli_writes_26_shap_tables_and_replaces_combined_raw(tmp_path, monkeypatch):
    cache, output = tmp_path / "cache", tmp_path / "output"
    cache.mkdir()
    pd.DataFrame({
        "participant_id": ["a", "b"], "label": [0, 1],
        "n_epochs": [1, 1], "split": ["train", "test"],
    }).to_csv(cache / "recordings.csv", index=False)
    cfg = summary.load_config(summary.p.ROOT / "configs/ECMAD.yaml")
    conditions = list(summary.p.condition_specs(cfg))
    methods = ("eegnet", "xgboost")
    for method in methods:
        for condition, _, _ in conditions:
            for index, subject in enumerate(("a", "b")):
                path = output / method / f"outer_{index:03d}" / condition
                path.mkdir(parents=True)
                (path / "result.json").write_text(json.dumps({
                    "participant_id": subject, "true_label": index,
                    "pred_proba": 0.1 + 0.8 * index, "predicted_label": index,
                }))

    def sample_importance(paths, rows, method, keys, cfg, epoch_counts):
        components = [key.split("_")[-1] for key in keys]
        return pd.DataFrame({
            "component": np.repeat(components, 2),
            "name": ["first", "second"] * len(keys),
            "mean_abs_shap": [1.0, 3.0] * len(keys),
            "n_subjects": len(rows),
            "n_epoch_model_pairs": 10,
            "unit": "probability" if method == "eegnet" else "log_odds",
        })

    monkeypatch.setattr(summary, "complete", lambda *_: True)
    monkeypatch.setattr(summary, "summarize_shap", sample_importance)
    monkeypatch.setattr(sys, "argv", [
        "summarize", "--cache", str(cache), "--output", str(output),
        "--config", str(summary.p.ROOT / "configs/ECMAD.yaml"),
    ])
    tables = output / "shap_summary"
    tables.mkdir()
    (tables / "raw.csv").write_text("method,condition\neegnet,raw\nxgboost,raw\n")
    summary.main()

    assert {p.name for p in tables.glob("*.csv")} == {
        f"{method}_{condition}.csv" for method in methods
        for condition, _, _ in conditions
    }
    assert len(list(tables.glob("*.csv"))) == 26
    for method, unit in zip(methods, ["probability", "log_odds"], strict=True):
        raw = pd.read_csv(tables / f"{method}_raw.csv")
        assert raw.method.tolist() == [method]
        assert raw.unit.tolist() == [unit]
        assert raw.mean_abs_shap.tolist() == [2.0]
    assert len(pd.read_csv(output / "comparison.csv")) == 26
    assert len(list(output.glob("*_oof.csv"))) == 26
