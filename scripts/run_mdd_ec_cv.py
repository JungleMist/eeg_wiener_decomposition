#!/usr/bin/env python3
"""Compare paired MDD_EC conditions using a locked test holdout and 10 development folds."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
from importlib import import_module
from importlib.metadata import version
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from sklearn.model_selection import StratifiedKFold, train_test_split

import ECMAD as pipeline
from eeg_bg.config.settings import load_config
from eeg_bg.features.connectivity import (
    build_connectivity_names, connectivity_features_for_layout,
)
from eeg_bg.features.extraction import build_feature_names, extract_epoch_features_for_layout
from run_ds003478_nested_cv import complete, save, seal, train_xgb_fold
from summarize_ds003478_nested_cv import condition_shap_summary

CHANNELS = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC3", "T7", "C3", "Cz",
            "CP3", "FC4", "T8", "C4", "CPz", "CP4", "P3", "P4", "Pz", "O1", "O2")
PAIRS = (("Fp1", "Fp2"), ("F3", "F4"), ("F7", "F8"), ("FC3", "FC4"), ("C3", "C4"),
         ("T7", "T8"), ("CP3", "CP4"), ("P3", "P4"), ("O1", "O2"))
FEATURES = build_feature_names(CHANNELS, PAIRS) + build_connectivity_names(PAIRS)
PHASES = ("g0.02", "g0.05", "g0.1", "g1")
COMPONENTS = ("specific", "coherent")
CONDITIONS = {"raw": ("raw",)}
for phase in PHASES:
    for component in COMPONENTS:
        key = f"{phase}_{component}"
        CONDITIONS[key] = (key,)
    CONDITIONS[f"{phase}_combo"] = tuple(f"{phase}_{component}" for component in COMPONENTS)
KEYS = tuple(keys[0] for keys in CONDITIONS.values() if len(keys) == 1)
METHODS = ("eegnet", "xgboost")
SEED = 42
FOLDS = 10
TEST_FRACTION = 0.2
DURATION = 120
SOURCE_RATE = 250
MODEL_RATE = 125
RECORDING = re.compile(r"(sub-[^_]+)_(ses-[^_]+)_task-restEC_eeg")


def discover(root):
    """Pair every raw and component EDF by recording identity within its cohort."""
    records, sources = [], {}
    for cohort, label in (("Control", 0), ("MDD", 1)):
        directories = {name: [p for p in (root / cohort).iterdir()
                              if p.is_dir() and p.name.casefold() == name]
                       for name in ("raw", *PHASES)}
        for name, matches in directories.items():
            if len(matches) != 1:
                raise ValueError(f"Expected one {name} directory in {root / cohort}")
        groups = {}
        for key in KEYS:
            folder, _, component = key.partition("_")
            suffix = f"_{component}.edf" if component else ".edf"
            groups[key] = {p.name.removesuffix(suffix): p for p in
                           directories[folder][0].glob(f"*{suffix}")}
        raw = groups["raw"]
        if not raw:
            raise ValueError(f"No raw EDFs in {directories['raw'][0]}")
        for key, files in groups.items():
            if set(files) != set(raw):
                missing, extra = sorted(set(raw) - set(files)), sorted(set(files) - set(raw))
                raise ValueError(f"{cohort}/{key}: missing={missing[:5]}, extra={extra[:5]}")
        for recording_id in sorted(raw):
            match = RECORDING.fullmatch(recording_id)
            if not match:
                raise ValueError(f"Unexpected EDF name: {raw[recording_id]}")
            participant_id, session = match.groups()
            if recording_id in sources:
                raise ValueError(f"Duplicate recording: {recording_id}")
            sources[recording_id] = {key: files[recording_id] for key, files in groups.items()}
            records.append({"participant_id": participant_id, "session": session,
                            "recording_id": recording_id, "cohort": cohort, "label": label,
                            "n_epochs": 1})
    records = pd.DataFrame(records).sort_values(["participant_id", "session"])
    if records.groupby("participant_id").label.nunique().gt(1).any():
        raise ValueError("A participant is present in both cohorts")
    return records, sources


def split_subjects(records):
    participants = records[["participant_id", "label"]].drop_duplicates().reset_index(drop=True)
    development, test = train_test_split(
        participants, test_size=TEST_FRACTION, random_state=SEED, stratify=participants.label,
    )
    development = development.sort_values("participant_id").reset_index(drop=True)
    if development.label.value_counts().min() < FOLDS or test.label.nunique() != 2:
        raise ValueError("Need both holdout classes and at least 10 development participants per class")
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    folds = []
    for train_idx, val_idx in cv.split(development, development.label):
        folds.append({
            **{key: development.iloc[indices].participant_id.tolist()
               for key, indices in (("train", train_idx), ("validation", val_idx))},
            "test": sorted(test.participant_id),
        })
    return folds


def source_manifest(root, sources):
    rows = []
    for recording_id, group in sources.items():
        for key, path in group.items():
            stat = path.stat()
            rows.append((recording_id, key, str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns))
    return pd.DataFrame(rows, columns=["recording_id", "bank", "file", "size", "mtime_ns"])


def open_edf(path, key):
    import mne

    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    try:
        actual = {}
        for name in raw.ch_names:
            actual.setdefault(name.casefold(), []).append(name)
        if any(len(actual.get(name.casefold(), [])) != 1 for name in CHANNELS):
            raise ValueError(f"EDF requires all 22 channels exactly once: {path}")
        channels = [actual[name.casefold()][0] for name in CHANNELS]
        expected = DURATION * SOURCE_RATE
        allowed = (expected, expected + SOURCE_RATE) if key == "raw" else (expected,)
        if raw.info["sfreq"] != SOURCE_RATE or raw.n_times not in allowed:
            raise ValueError(f"Incompatible EDF rate or length: {path} "
                             f"({raw.info['sfreq']} Hz, {raw.n_times} samples)")
        return raw, channels
    except Exception:
        raw.close()
        raise


def prepare_record(job):
    sources, epoch_path, feature_path, cfg, check_only = job
    if not check_only and all(path is None or path.is_file() for path in (epoch_path, feature_path)):
        return
    epochs, features = {"starts": [0]}, {"starts": [0]}
    for key, path in sources.items():
        raw, channels = open_edf(path, key)
        with raw:
            if check_only:
                continue
            signal = raw.get_data(picks=channels, stop=DURATION * SOURCE_RATE) * 1e6
        signal = resample_poly(signal, MODEL_RATE, SOURCE_RATE, axis=-1).astype(np.float32)
        if signal.shape != (len(CHANNELS), DURATION * MODEL_RATE) or not np.isfinite(signal).all():
            raise ValueError(f"Invalid EEG data in {path}")
        if epoch_path is not None:
            epochs[key] = signal[None]
        if feature_path is not None:
            base = extract_epoch_features_for_layout(
                signal, CHANNELS, MODEL_RATE, CHANNELS, PAIRS,
                nperseg=cfg["wiener"]["nperseg"], freq_band=tuple(cfg["wiener"]["freq_band"]),
            )
            connectivity = connectivity_features_for_layout(
                signal, CHANNELS, MODEL_RATE, CHANNELS, PAIRS,
                nperseg=cfg["ml"]["features"]["connectivity"]["nperseg"],
            )
            features[key] = np.concatenate([base, connectivity])[None].astype(np.float32)
            if not np.isfinite(features[key]).all():
                raise ValueError(f"Invalid XGBoost features in {path}")
    for path, payload in ((epoch_path, epochs), (feature_path, features)):
        if path is not None:
            pipeline.write_npz(path, payload)


def prepare(output, sources, methods, cfg, workers, *, check_only=False):
    directories = [output / name if method in methods and not check_only else None
                   for name, method in (("epochs", "eegnet"), ("features", "xgboost"))]
    for directory in directories:
        if directory is not None:
            directory.mkdir(exist_ok=True)
    jobs = [
        (group, *(directory / f"{recording_id}.npz" if directory else None
                  for directory in directories), cfg, check_only)
        for recording_id, group in sources.items()
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, _ in enumerate(pool.map(prepare_record, jobs), start=1):
            if index % 50 == 0 or index == len(jobs):
                print(f"{'EDF header checks' if check_only else 'Prepared recordings'} {index}/{len(jobs)}",
                      flush=True)


def item_names(method, keys):
    base = CHANNELS if method == "eegnet" else FEATURES
    if len(keys) == 1:
        return list(base)
    return [f"{bank}:{item}" for bank in COMPONENTS for item in base]


def threshold_for(method, predictions):
    if method == "eegnet":
        from eeg_bg.ml.erp_eegnet import select_balanced_accuracy_threshold

        return select_balanced_accuracy_threshold(predictions.true_label, predictions.pred_proba)
    from eeg_bg.ml.xgb_pipeline import find_optimal_threshold

    return find_optimal_threshold(predictions)


def fit_xgboost(output, target, records, keys, workers, device):
    import joblib

    from eeg_bg.ml.shap_analysis import compute_shap_values

    matrix_args = (output, records, "features", keys)
    train_x, train_rows = pipeline.matrix_for_split(*matrix_args, "train")
    val_x, val_rows = pipeline.matrix_for_split(*matrix_args, "validation")
    model, scaler, validation = train_xgb_fold(
        target, train_x, train_rows, val_x, val_rows, workers, device=device, need_model=True,
    )
    threshold = threshold_for("xgboost", validation)
    test_x, test_rows = pipeline.matrix_for_split(*matrix_args, "test")
    test_x = scaler.transform(test_x)
    test = pipeline.subject_predictions(test_rows, model.predict_proba(test_x)[:, 1])
    names = item_names("xgboost", keys)
    values = compute_shap_values(model.get_booster(), test_x, names)
    if values.shape != test_x.shape or not np.isfinite(values).all():
        raise ValueError("Invalid XGBoost SHAP output")
    pipeline.write_npz(target / "shap_test.npz", {"values": values, "names": names})
    test_rows.to_csv(target / "shap_test_epochs.csv", index=False)
    model.save_model(target / "model.ubj")
    joblib.dump(scaler, target / "scaler.joblib")
    return pipeline.evaluate(target, validation, test, threshold=threshold)


def run_training(args, cfg, records, folds, methods, conditions, indices):
    for method in methods:
        dependency = "torch" if method == "eegnet" else "xgboost"
        device = args.device
        if method == "eegnet":
            import torch

            resolve = import_module("23_train_tuep_component_eegnet").resolve_device
            device = torch.device(resolve(device))
        elif device == "auto":
            device = "cpu"
        elif device == "mps":
            raise ValueError("XGBoost requires --device auto, cpu, or cuda")
        identity = {"device": str(device), "workers": args.workers,
                    dependency: version(dependency), "shap": version("shap")}
        for index in indices:
            assignments = {participant: split for split, ids in folds[index].items()
                           for participant in ids}
            fold_records = records.assign(split=records.participant_id.map(assignments))
            for condition in conditions:
                keys = CONDITIONS[condition]
                target = args.output / method / f"fold_{index + 1:02d}" / condition
                target.mkdir(parents=True, exist_ok=True)
                manifest = target / "training_config.json"
                if manifest.exists() and json.loads(manifest.read_text()) != identity:
                    raise ValueError(f"Training environment changed for {target}. Use a new output directory.")
                save(manifest, identity)
                progress = target / "last_training.pt"
                if complete(target):
                    progress.unlink(missing_ok=True)
                    continue
                print(f"{method} fold {index + 1}/{FOLDS} {condition}", flush=True)
                names = ["metrics.json", "validation_predictions.csv", "test_predictions.csv",
                         "shap_test.npz", "shap_test_epochs.csv"]
                if method == "eegnet":
                    metrics = pipeline.fit_eegnet(
                        args.output, target, fold_records, keys, cfg, device, args.workers,
                        positive_class="1=MDD",
                    )
                    names.extend(["best_model.pt", "history.csv", "shap_background_epochs.csv",
                                  "shap_channel_importance.csv", "shap_summary.png"])
                else:
                    metrics = fit_xgboost(args.output, target, fold_records, keys, args.workers, device)
                    names.extend(["model.ubj", "scaler.joblib"])
                save(target / "metrics.json", metrics)
                seal(target, names)
                progress.unlink(missing_ok=True)


def collect_fold(target, method, condition, records, split):
    if not complete(target):
        raise ValueError(f"Missing or incomplete fold: {target}")
    validation, test = (
        pd.read_csv(target / f"{name}_predictions.csv") for name in ("validation", "test")
    )
    for name, frame in (("validation", validation), ("test", test)):
        expected = records.loc[records.participant_id.isin(split[name])]
        subjects = expected.groupby("participant_id").label.agg(["first", "size"]).sort_index()
        scored = frame.set_index("participant_id").sort_index()
        if (not subjects.index.equals(scored.index)
                or not np.array_equal(subjects["first"], scored.true_label)
                or not np.array_equal(subjects["size"], scored.n_epochs)
                or not scored.pred_proba.between(0, 1).all()):
            raise ValueError(f"{name} predictions do not match the split: {target}")
    epochs = pd.read_csv(target / "shap_test_epochs.csv")
    columns = ["recording_id", "participant_id", "label"]
    reference = records.loc[records.participant_id.isin(split["test"]), columns]
    if (not epochs[columns].sort_values("recording_id").reset_index(drop=True).equals(
            reference.sort_values("recording_id").reset_index(drop=True))
            or not epochs.epoch_start_sample.eq(0).all()):
        raise ValueError(f"SHAP epoch inventory does not match the holdout: {target}")
    names = item_names(method, CONDITIONS[condition])
    shape = (len(epochs), len(names))
    if method == "eegnet":
        shape += (DURATION * MODEL_RATE,)
    with np.load(target / "shap_test.npz") as archive:
        values = archive["values"]
        if values.shape != shape or list(archive["names"]) != names or not np.isfinite(values).all():
            raise ValueError(f"Invalid SHAP values or feature order: {target}")
    magnitude = np.abs(values)
    if method == "eegnet":
        magnitude = magnitude.mean(axis=-1)
    importance = np.average(magnitude, axis=0, weights=pipeline.equal_subject_weights(epochs))
    metrics = json.loads((target / "metrics.json").read_text())
    return validation, test, importance, len(epochs), metrics


def summarize(output, records, folds, methods, conditions):
    from eeg_bg.ml.erp_eegnet import classification_metrics

    summary, per_fold, shap_summary, shap_items = [], [], [], []
    directory = output / "summary"
    if tuple(methods) != METHODS or tuple(conditions) != tuple(CONDITIONS):
        method_name = "all" if tuple(methods) == METHODS else methods[0]
        condition_name = "all" if tuple(conditions) == tuple(CONDITIONS) else conditions[0]
        directory /= f"{method_name}_{condition_name}"
    directory.mkdir(parents=True, exist_ok=True)
    for method in methods:
        for condition in conditions:
            validation, test, importance, epoch_counts = [], [], [], []
            for index, split in enumerate(folds, start=1):
                target = output / method / f"fold_{index:02d}" / condition
                val, held, values, epoch_count, metrics = collect_fold(
                    target, method, condition, records, split,
                )
                validation.append(val.assign(fold=index))
                test.append(held)
                importance.append(values)
                epoch_counts.append(epoch_count)
                for role in ("validation", "test"):
                    per_fold.append({"method": method, "condition": condition, "fold": index,
                                     "split": role, **metrics[role]})
            oof = pd.concat(validation, ignore_index=True).sort_values("participant_id")
            held = pd.concat(test, ignore_index=True).groupby("participant_id", as_index=False).agg(
                true_label=("true_label", "first"), pred_proba=("pred_proba", "mean"),
                n_epochs=("n_epochs", "first"),
            )
            threshold = threshold_for(method, oof)
            held["predicted_label"] = (held.pred_proba >= threshold).astype(int)
            summary.append({
                "method": method, "condition": condition, "n_subjects": len(held),
                **classification_metrics(held.true_label, held.pred_proba, threshold),
            })
            prefix = f"{method}_{condition}"
            oof.to_csv(directory / f"{prefix}_validation_oof.csv", index=False)
            held.to_csv(directory / f"{prefix}_holdout.csv", index=False)
            names = item_names(method, CONDITIONS[condition])
            default_component = condition.rpartition("_")[2]
            importance_table = pd.DataFrame({
                "item": names,
                "component": [name.split(":", 1)[0] if ":" in name else default_component
                              for name in names],
                "mean_abs_shap": np.mean(importance, axis=0),
                "sd_fold_mean_abs_shap": np.std(importance, axis=0, ddof=1),
                "n_subjects": len(held),
                "n_epoch_model_pairs": sum(epoch_counts),
                "unit": "probability" if method == "eegnet" else "log-odds",
            }).sort_values("mean_abs_shap", ascending=False)
            importance_table.to_csv(directory / f"{prefix}_shap_items.csv", index=False)
            shap_summary.append(condition_shap_summary(importance_table, method, condition))
            shap_items.append(importance_table.assign(method=method, condition=condition))
    comparison = pd.DataFrame(summary)
    items = pd.concat(shap_items, ignore_index=True)
    items["base_item"] = items.item.str.rsplit(":", n=1).str[-1]
    if "raw" in conditions:
        for metric in ("auroc", "auprc", "f1", "balanced_accuracy", "accuracy"):
            baseline = comparison.loc[comparison.condition.eq("raw")].set_index("method")[metric]
            comparison[f"delta_{metric}_vs_raw"] = comparison[metric] - comparison.method.map(baseline)
        baseline = items.loc[items.condition.eq("raw"), ["method", "base_item", "mean_abs_shap"]]
        items = items.merge(
            baseline.rename(columns={"mean_abs_shap": "raw_mean_abs_shap"}),
            on=["method", "base_item"], validate="many_to_one",
        )
        items["delta_mean_abs_shap_vs_raw"] = items.mean_abs_shap - items.raw_mean_abs_shap
    comparison.to_csv(directory / "holdout_comparison.csv", index=False)
    pd.DataFrame(per_fold).to_csv(directory / "fold_metrics.csv", index=False)
    pd.concat(shap_summary, ignore_index=True).to_csv(directory / "shap_comparison.csv", index=False)
    items.to_csv(directory / "shap_item_comparison.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("~/Data/MDD_EC"))
    parser.add_argument("--output", type=Path, default=Path("results/mdd_ec_cv"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto",
                        help="auto uses the available EEGNet accelerator and CPU for XGBoost")
    parser.add_argument("--method", choices=("all", *METHODS), default="all")
    parser.add_argument("--condition", choices=("all", *CONDITIONS), default="all")
    parser.add_argument("--stage", choices=("all", "check", "prepare", "train", "summarize"),
                        default="all")
    parser.add_argument("--fold", type=int, choices=range(1, FOLDS + 1),
                        help="Run a single development fold with --stage train")
    parser.add_argument("--max-epochs", type=int, help="Override the EEGNet training budget")
    parser.add_argument("--shap-samples", type=int, help="Override EEGNet gradient SHAP samples per epoch")
    args = parser.parse_args()
    if any(value is not None and value < 1
           for value in (args.workers, args.max_epochs, args.shap_samples)):
        parser.error("--workers, --max-epochs and --shap-samples must be positive")
    if args.fold is not None and args.stage != "train":
        parser.error("--fold requires --stage train")
    if args.device == "mps" and args.method != "eegnet" and args.stage in ("all", "train"):
        parser.error("--device mps requires --method eegnet")
    args.output = args.output.expanduser().resolve()
    root = args.data_dir.expanduser().resolve()
    records, sources = discover(root)
    folds = split_subjects(records)
    participants = records.drop_duplicates("participant_id")
    counts = participants.cohort.value_counts().to_dict()
    print(f"{len(participants)} participants {counts}, {len(records)} paired epochs", flush=True)
    print(f"Holdout: {len(folds[0]['test'])} participants. "
          f"Development: {len(participants) - len(folds[0]['test'])} participants.", flush=True)

    cfg = load_config(pipeline.ROOT / "configs/ECMAD.yaml")
    cfg["channels"]["standard_19"] = list(CHANNELS)
    cfg["preprocessing"]["target_sfreq"] = MODEL_RATE
    cfg["ds003478"]["eegnet_shap_max_epochs_per_subject"] = None
    cfg["ds003478"]["eegnet_shap_storage_dtype"] = "float32"
    if args.shap_samples is not None:
        cfg["ds003478"]["eegnet_shap_samples"] = args.shap_samples
    if args.max_epochs is not None:
        cfg["ml"]["tuep_component_eegnet"]["max_epochs"] = args.max_epochs
    if args.stage == "check":
        prepare(args.output, sources, (), cfg, args.workers, check_only=True)
        return

    methods = METHODS if args.method == "all" else (args.method,)
    conditions = tuple(CONDITIONS) if args.condition == "all" else (args.condition,)
    indices = range(FOLDS) if args.fold is None else (args.fold - 1,)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = source_manifest(root, sources)
        manifest_csv = manifest.to_csv(index=False).encode()
        identity = {
            "version": 1,
            "source_sha256": hashlib.sha256(manifest_csv).hexdigest(),
            "channels": list(CHANNELS), "symmetric_pairs": [list(pair) for pair in PAIRS],
            "features": FEATURES, "conditions": {key: list(values) for key, values in CONDITIONS.items()},
            "epoch_seconds": DURATION, "edf_sfreq": SOURCE_RATE, "model_sfreq": MODEL_RATE,
            "seed": SEED, "test_fraction": TEST_FRACTION, "folds": folds,
            "labels": {"Control": 0, "MDD": 1},
            "settings": {key: cfg[key] for key in ("ds003478", "wiener", "ml")},
            "code_sha256": {
                str(path.relative_to(pipeline.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (Path(__file__).resolve(), Path(pipeline.__file__).resolve(),
                             pipeline.ROOT / "eeg_bg/features/connectivity.py")
            },
        }
        identity_path = args.output / "run_config.json"
        identity = json.loads(json.dumps(identity))
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("Input, split, code or training settings changed. Use a new --output directory.")
        save(identity_path, identity)
        pipeline.atomic_write(args.output / "source_files.csv", manifest_csv)
        records.to_csv(args.output / "recordings.csv", index=False)
        labels = participants.set_index("participant_id").label
        split_rows = [
            {"fold": index, "participant_id": participant, "label": int(labels[participant]), "split": role}
            for index, fold in enumerate(folds, start=1)
            for role, identifiers in fold.items()
            for participant in identifiers
        ]
        pd.DataFrame(split_rows).to_csv(args.output / "splits.csv", index=False)
        if args.stage in ("all", "prepare"):
            prepare(args.output, sources, methods, cfg, args.workers)
        if args.stage in ("all", "train"):
            for method in methods:
                cache = args.output / ("epochs" if method == "eegnet" else "features")
                if not all((cache / f"{name}.npz").is_file() for name in sources):
                    raise ValueError(f"Missing {method} cache. Run --stage prepare first.")
            run_training(args, cfg, records, folds, methods, conditions, indices)
        if args.stage in ("all", "summarize"):
            summarize(args.output, records, folds, methods, conditions)


if __name__ == "__main__":
    main()
