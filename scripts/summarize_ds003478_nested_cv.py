"""Summarize subject-level nested CV predictions and held-out SHAP values."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

import ECMAD as p
from eeg_bg.config.settings import load_config
from eeg_bg.features.profiles import PROFILES
from run_ds003478_nested_cv import XGBOOST_PROTOCOL, complete


def summarize_shap(paths, rows, method, keys, cfg, epoch_counts):
    """Reduce time and epochs, then weight the five models and subjects equally."""
    eegnet = method == "eegnet"
    if eegnet:
        base_names = cfg["channels"]["standard_19"]
        prep = cfg["preprocessing"]
        n_times = round(prep["target_sfreq"] * prep["epoch_length_sec"])
        array_file, index_file = "shap_test.npz", "shap_test_epochs.csv"
    else:
        base_names = PROFILES[cfg["ds003478"]["feature_profile"]].names
        array_file, index_file = "outer_shap.npz", "outer_shap_epochs.csv"

    components = [key.split("_")[-1] for key in keys]
    names = base_names if len(keys) == 1 else [
        f"{component}:{name}" for component in components for name in base_names
    ]
    totals = np.zeros(len(names), dtype=np.float64)
    n_epoch_model_pairs = 0
    for result_path, row in zip(paths, rows.itertuples(), strict=True):
        directory = result_path.parent
        n_epochs = epoch_counts[row.participant_id]
        n_explained = min(2, n_epochs) if eegnet else n_epochs
        sealed = json.loads((directory / "result_complete.json").read_text())
        model_importance = []
        for inner in range(5):
            relative = Path(f"inner_{inner}") / "outer"
            if any(str(relative / name) not in sealed for name in (array_file, index_file)):
                raise RuntimeError(f"Missing sealed SHAP outputs: {directory / relative}")
            path = directory / relative
            epochs = pd.read_csv(path / index_file)
            epoch_ids = ["recording_id", "epoch_start_sample"]
            if (len(epochs) != n_explained or not epochs.participant_id.eq(row.participant_id).all()
                    or not epochs.label.eq(row.true_label).all()
                    or epochs[epoch_ids].isna().any().any()
                    or epochs.duplicated(epoch_ids).any()):
                raise RuntimeError(f"Invalid held-out SHAP epoch index: {path / index_file}")
            shape = (len(epochs), len(names), n_times) if eegnet else (len(epochs), len(names))
            with np.load(path / array_file, allow_pickle=False) as artifact:
                values = artifact["values"]
                if values.shape != shape or not np.isfinite(values).all():
                    raise RuntimeError(f"Invalid SHAP array: {path / array_file}, expected {shape}")
                if eegnet and artifact["names"].tolist() != names:
                    raise RuntimeError(f"SHAP channel order mismatch: {path / array_file}")
                # Average magnitudes, so SHAP signs cannot cancel between epochs or models.
                axes = (0, 2) if eegnet else 0
                model_importance.append(np.abs(values).mean(axis=axes, dtype=np.float64))
            n_epoch_model_pairs += len(epochs)
        totals += np.mean(model_importance, axis=0)

    return pd.DataFrame({
        "component": np.repeat(components, len(base_names)),
        "name": names,
        "mean_abs_shap": totals / len(rows),
        "n_subjects": len(rows),
        "n_epoch_model_pairs": n_epoch_model_pairs,
        "unit": "probability" if eegnet else "log_odds",
    })


def condition_shap_summary(importance, method, condition):
    """Average channels/features within the condition and its component banks."""
    groups = list(importance.groupby("component", sort=False))
    if len(groups) > 1:
        groups.insert(0, ("all", importance))
    return pd.DataFrame([
        {
            "method": method,
            "condition": condition,
            "component": component,
            "mean_abs_shap": values.mean_abs_shap.mean(),
            "n_items": len(values),
            "n_subjects": values.n_subjects.iloc[0],
            "n_epoch_model_pairs": values.n_epoch_model_pairs.iloc[0],
            "unit": values.unit.iloc[0],
        }
        for component, values in groups
    ])


def write_csv(path, frame):
    p.atomic_write(path, frame.to_csv(index=False).encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/ECMAD.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    records = p.read_records(args.cache)
    subjects = records[["participant_id", "label"]].drop_duplicates().set_index("participant_id")
    if not subjects.index.is_unique:
        raise RuntimeError("Conflicting labels in the cache")
    expected = subjects.label.to_dict()
    epoch_counts = records.groupby("participant_id").n_epochs.sum().to_dict()
    for manifest in (args.output / "xgboost").glob("outer_*/manifest.json"):
        stored = json.loads(manifest.read_text())
        if stored.get("xgboost_protocol") != XGBOOST_PROTOCOL:
            raise RuntimeError(f"Incompatible XGBoost protocol: {manifest}")
        if stored["config"]["ds003478"]["feature_profile"] != cfg["ds003478"]["feature_profile"]:
            raise RuntimeError(f"XGBoost feature profile mismatch: {manifest}")

    comparison, oof_tables, shap_tables = [], {}, {}
    for method in ("eegnet", "xgboost"):
        for condition, keys, _ in p.condition_specs(cfg):
            paths = sorted((args.output / method).glob(f"outer_*/{condition}/result.json"))
            if any(not complete(f.parent, "result_complete.json") for f in paths):
                raise RuntimeError(f"Unverified or damaged result: {method}/{condition}")
            rows = pd.DataFrame([json.loads(f.read_text()) for f in paths])
            if (rows.empty or set(rows.participant_id) != set(expected)
                    or rows.participant_id.duplicated().any()):
                raise RuntimeError(f"Incomplete results: {method}/{condition}")
            if not rows.participant_id.map(expected).eq(rows.true_label).all():
                raise RuntimeError(f"Conflicting subject labels: {method}/{condition}")
            importance = summarize_shap(paths, rows, method, keys, cfg, epoch_counts)
            shap_tables[f"{method}_{condition}.csv"] = condition_shap_summary(
                importance, method, condition,
            )
            oof_tables[f"{method}_{condition}_oof.csv"] = rows
            comparison.append(dict(
                method=method, condition=condition,
                auroc=roc_auc_score(rows.true_label, rows.pred_proba),
                auprc=average_precision_score(rows.true_label, rows.pred_proba),
                balanced_accuracy=balanced_accuracy_score(rows.true_label, rows.predicted_label),
                f1=f1_score(rows.true_label, rows.predicted_label, zero_division=0),
            ))

    # Validate the full run before replacing any previously summarized files.
    for name, rows in oof_tables.items():
        write_csv(args.output / name, rows)
    write_csv(args.output / "comparison.csv", pd.DataFrame(comparison))
    shap_dir = args.output / "shap_summary"
    shap_dir.mkdir(exist_ok=True)
    for name, frame in shap_tables.items():
        write_csv(shap_dir / name, frame)
    # Both model-specific raw tables are now written; remove the old combined output.
    (shap_dir / "raw.csv").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
