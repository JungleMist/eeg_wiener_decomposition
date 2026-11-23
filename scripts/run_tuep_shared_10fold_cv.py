#!/usr/bin/env python3
"""Run raw and five ECMAD phases with shared patient-level TUEP 10-fold CV."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib import import_module
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import signal

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from eeg_bg.config.settings import load_config
from eeg_bg.io.dataset import build_recording_index
from tuep_shared_cv_data import (
    COMBO, COMPONENTS, N_FOLDS, PHASES, PROFILE, SEED, WINDOW_KEY, continuous,
    extract_features, feature_matrix, prepare_epochs, run_record_jobs, split_patients,
)


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("eegnet", "xgboost")
STAGES = {"raw": ("raw",), **{phase: (*COMPONENTS, COMBO) for phase in PHASES}}
ID_COLUMNS = {col: str for col in (*WINDOW_KEY[:2], "patient_id", "recording_id")}
FOLD_FILES = ("test_predictions.csv", "test_epoch_predictions.csv",
              "shap.npz", "shap_background.csv", "model.json")


def read_csv(path):
    return pd.read_csv(path, dtype=ID_COLUMNS, keep_default_na=False)


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_sizes(path, files):
    sizes = {name: (path / name).stat().st_size for name in files}
    if not all(sizes.values()):
        raise ValueError(f"Empty artifact in {path}")
    return sizes


def completed(path, files):
    marker = path / "complete.json"
    if not marker.is_file():
        return False
    try:
        return read_json(marker) == artifact_sizes(path, files)
    except (FileNotFoundError, ValueError):
        return False


def dataset_identity(records):
    entries = []
    for record in records:
        edf = Path(record["edf_path"])
        entries.append({
            "record": record,
            "edf": continuous._file_identity(edf),
            "annotation": continuous._file_identity(edf.with_suffix(".csv_bi")),
        })
    return entries


def epoch_metrics_rows(windows, probabilities):
    result = windows.rename(columns={"label": "true_label"}).copy()
    result["pred_proba"] = probabilities
    return result


def patient_predictions(epochs):
    result = epochs.groupby("evaluation_id", sort=True, as_index=False).agg(
        patient_id=("patient_id", "first"), true_label=("true_label", "first"),
        pred_proba=("pred_proba", "mean"), n_epochs=("pred_proba", "size"),
    )
    result["predicted_label"] = (result["pred_proba"] >= 0.5).astype(int)
    return result


def verify_predictions(path, expected):
    epochs = read_csv(path / "test_epoch_predictions.csv")
    patients = read_csv(path / "test_predictions.csv")
    columns = [*WINDOW_KEY, "patient_id", "recording_id", "true_label", "segment_index"]
    pd.testing.assert_frame_equal(
        epochs[columns].reset_index(drop=True),
        expected.rename(columns={"label": "true_label"})[columns].reset_index(drop=True),
        check_dtype=False,
    )
    probability = epochs["pred_proba"].to_numpy()
    if not np.all(np.isfinite(probability) & (probability >= 0) & (probability <= 1)):
        raise ValueError(f"Invalid probabilities in {path}")
    pd.testing.assert_frame_equal(patients, patient_predictions(epochs), check_dtype=False)
    return epochs


def background_rows(windows, patients, fold, method):
    held = patients["outer_fold"] == fold
    validation = patients[f"eegnet_val_{fold}"]
    eligible = patients[~held & ~validation] if method == "eegnet" else patients[~held]
    if len(eligible) > 100:
        sampler = StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=SEED + fold)
        _, selected = next(sampler.split(eligible, eligible["label"]))
        eligible = eligible.iloc[selected]
    rng = np.random.default_rng(SEED + fold)
    groups = windows[windows["evaluation_id"].isin(eligible["evaluation_id"])].groupby(
        "evaluation_id", sort=True,
    )
    return np.array([rng.choice(frame.index) for _, frame in groups])


def eegnet_patients(windows, patients, waveforms, fold, cfg):
    eegnet = import_module("23_train_tuep_component_eegnet")
    indexed = patients.set_index("evaluation_id")
    records = []
    sfreq = cfg["preprocessing"]["target_sfreq"]
    for record_id, frame in windows.groupby("record_cache_id", sort=False):
        row = frame.iloc[0]
        patient = indexed.loc[row["evaluation_id"]]
        if patient["outer_fold"] == fold:
            split = "test"
        else:
            split = "validation" if patient[f"eegnet_val_{fold}"] else "train"
        records.append({
            **row[["evaluation_id", "patient_id", "label", "class_name", "recording_id",
                   "record_cache_id"]].to_dict(),
            "path": waveforms / f"{record_id}.npz",
            "split": split,
            "n_epochs": len(frame),
            "ch_names": tuple(cfg["channels"]["standard_19"]),
            "epoch_samples": int(sfreq * cfg["preprocessing"]["epoch_length_sec"]),
        })
    return eegnet.group_patients(records)


def load_eeg_windows(frame, condition, waveforms):
    eegnet = import_module("23_train_tuep_component_eegnet")
    banks = COMPONENTS if condition == COMBO else (condition,)
    for record_id, group in frame.groupby("record_cache_id", sort=False):
        with np.load(waveforms / f"{record_id}.npz", allow_pickle=False) as record:
            starts = record["epoch_start_samples"]
            rows = np.searchsorted(starts, group["epoch_start_sample"])
            np.testing.assert_array_equal(starts[rows], group["epoch_start_sample"])
            values = np.concatenate([record[bank][rows] for bank in banks], axis=1)
        yield eegnet.trial_channel_zscore(values)


def explain_eegnet(model, background, test, waveforms, condition, device, cfg):
    import shap
    import torch

    model.eval()
    bg = np.concatenate(list(load_eeg_windows(background, condition, waveforms)))
    tensor = torch.from_numpy(bg).unsqueeze(1).to(device)
    explainer = shap.DeepExplainer(model, tensor)
    baseline = float(np.asarray(explainer.expected_value).item())
    chunks = []
    max_error = 0.0
    for values in load_eeg_windows(test, condition, waveforms):
        for batch in torch.from_numpy(values).split(8):
            batch = batch.unsqueeze(1).to(device)
            explained = explainer.shap_values(batch, check_additivity=True)
            values_shap = np.asarray(explained).reshape(tuple(batch.shape))[:, 0]
            with torch.no_grad():
                prediction = model(batch).cpu().numpy().ravel()
            error = float(np.max(np.abs(baseline + values_shap.sum(axis=(1, 2)) - prediction)))
            if not np.isfinite(error) or error > 0.01:
                raise ValueError(f"DeepSHAP additivity failed (max error {error})")
            max_error = max(error, max_error)
            chunks.append(np.abs(values_shap).mean(axis=-1).astype(np.float32))
    names = named_banks(cfg["channels"]["standard_19"], condition)
    return np.concatenate(chunks), names, baseline, max_error


def named_banks(names, condition):
    return [f"{bank}::{name}" for bank in COMPONENTS for name in names] if condition == COMBO else list(names)


def run_eegnet(path, cfg, condition, windows, plan, waveforms, held, bg, fold, device, workers, fingerprint):
    import torch

    eegnet = import_module("23_train_tuep_component_eegnet")
    model_cfg = cfg["ml"]["tuep_component_eegnet"]
    torch.set_num_threads(workers)
    patients = eegnet_patients(windows, plan, waveforms, fold, cfg)
    training = path / "training"
    try:
        trained = eegnet._condition_cache_matches(training, fingerprint)
    except (eegnet.CacheMismatch, ValueError):
        trained = False
    if not trained:
        eegnet.train_condition(
            condition, patients, model_cfg, training, int(model_cfg["random_state"]),
            device, int(model_cfg["num_workers"]), fingerprint, fixed_threshold=0.5,
        )
    checkpoint = torch.load(training / "best_model.pt", map_location="cpu", weights_only=True)
    model = eegnet._make_model(checkpoint["n_channels"], checkpoint["n_times"], model_cfg)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    predictions = read_csv(training / "test_epoch_predictions.csv")
    probabilities = held.merge(
        predictions[[*WINDOW_KEY, "pred_proba"]], on=WINDOW_KEY, how="left",
        validate="one_to_one",
    )["pred_proba"].to_numpy()
    if len(predictions) != len(held) or not np.isfinite(probabilities).all():
        raise ValueError("EEGNet prediction coverage does not match the held-out windows")
    shap_output = explain_eegnet(model, bg, held, waveforms, condition, device, cfg)
    model_meta = {
        "parameters": model_cfg, "device": device, "model": "training/best_model.pt",
        "best_epoch": read_json(training / "best_params.json")["best_epoch"],
        "threshold": 0.5, "shap_method": "DeepSHAP", "shap_output": "control_probability",
        "shap_reduction": "mean_abs_over_time",
    }
    return probabilities, shap_output, model_meta


def run_xgboost(path, matrix, windows, held, bg, condition, device, workers):
    import joblib
    import shap
    train = ~windows["evaluation_id"].isin(held["evaluation_id"])
    scaler = StandardScaler().fit(matrix[train])
    x_train = scaler.transform(matrix[train])
    x_test = scaler.transform(matrix[held.index])
    model = XGBClassifier(random_state=SEED, device=device, n_jobs=workers)
    model.fit(x_train, windows.loc[train, "label"])
    # CPU prediction/explanation consumes the same trees and avoids a GPU/CPU device mismatch.
    model.get_booster().set_param({"device": "cpu"})
    probabilities = model.predict_proba(x_test)[:, 1]
    model.save_model(path / "xgboost.ubj")
    joblib.dump(scaler, path / "scaler.joblib")
    explainer = shap.TreeExplainer(
        model, data=scaler.transform(matrix[bg.index]),
        feature_perturbation="interventional", model_output="raw",
    )
    baseline = float(np.asarray(explainer.expected_value).item())
    chunks = []
    max_error = 0.0
    for start in range(0, len(x_test), 256):
        batch = x_test[start:start + 256]
        attribution = np.asarray(explainer.shap_values(batch, check_additivity=True))
        margin = model.predict(batch, output_margin=True)
        error = float(np.max(np.abs(baseline + attribution.sum(axis=1) - margin)))
        if attribution.shape != batch.shape or not np.isfinite(error) or error > 0.01:
            raise ValueError(f"TreeSHAP additivity failed (max error {error})")
        max_error = max(error, max_error)
        chunks.append(attribution.astype(np.float32))
    names = named_banks(PROFILE.names, condition)
    model_meta = {
        "parameters": {k: v for k, v in model.get_params().items() if k != "missing"},
        "training_device": device, "rounds": model.get_booster().num_boosted_rounds(),
        "model": "xgboost.ubj", "scaler": "scaler.joblib", "threshold": 0.5,
        "shap_method": "TreeSHAP/interventional", "shap_output": "control_log_odds",
    }
    return probabilities, (np.concatenate(chunks), names, baseline, max_error), model_meta


def fold_path(output, stage, method, condition, fold):
    return output / stage / method / condition / f"fold_{fold:02d}"


def fold_files(method):
    if method == "eegnet":
        eegnet = import_module("23_train_tuep_component_eegnet")
        return (*FOLD_FILES, *(f"training/{name}" for name in eegnet.CONDITION_ARTIFACTS),
                "training/cache.json")
    return (*FOLD_FILES, "xgboost.ubj", "scaler.joblib")


def validate_fold(path, held, background, method, condition, cfg):
    verify_predictions(path, held)
    pd.testing.assert_frame_equal(
        read_csv(path / "shap_background.csv")[WINDOW_KEY],
        background[WINDOW_KEY].reset_index(drop=True), check_dtype=False,
    )
    names = named_banks(
        cfg["channels"]["standard_19"] if method == "eegnet" else PROFILE.names, condition,
    )
    with np.load(path / "shap.npz", allow_pickle=False) as data:
        if (data["values"].shape != (len(held), len(names))
            or not np.array_equal(data["names"], names)
            or not np.isfinite(data["values"]).all()
            or not np.isfinite(data["baseline"]).item()
            or not 0 <= data["max_additivity_error"].item() <= 0.01):
            raise ValueError(f"Invalid SHAP artifact in {path}")


def run_fold(path, cfg, method, condition, windows, plan, waveforms, matrix,
             fold, device, workers, fingerprint):
    held = windows[windows["evaluation_id"].isin(plan.loc[plan["outer_fold"] == fold, "evaluation_id"])]
    bg = windows.loc[background_rows(windows, plan, fold, method)]
    files = fold_files(method)
    if completed(path, files):
        validate_fold(path, held, bg, method, condition, cfg)
        return False
    path.mkdir(parents=True, exist_ok=True)
    if method == "eegnet":
        result = run_eegnet(path, cfg, condition, windows, plan, waveforms, held, bg,
                            fold, device, workers, fingerprint)
    else:
        result = run_xgboost(path, matrix, windows, held, bg, condition, device, workers)
    probability, (values, names, baseline, error), metadata = result
    epochs = epoch_metrics_rows(held, probability)
    eegnet = import_module("23_train_tuep_component_eegnet")
    eegnet._atomic_csv(epochs, path / "test_epoch_predictions.csv")
    eegnet._atomic_csv(patient_predictions(epochs), path / "test_predictions.csv")
    eegnet._atomic_csv(bg, path / "shap_background.csv")
    continuous._atomic_npz(path / "shap.npz", values=values, names=np.asarray(names),
                           baseline=baseline, max_additivity_error=error)
    continuous._atomic_json(metadata, path / "model.json")
    validate_fold(path, held, bg, method, condition, cfg)
    continuous._atomic_json(artifact_sizes(path, files), path / "complete.json")
    return True


def clear_phase_cache(path):
    if path.is_symlink():
        raise ValueError(f"Phase cache path may not be a symlink: {path}")
    if path.exists():
        shutil.rmtree(path)


def progress(output, status, **update):
    status.update(update, updated_at=datetime.now(timezone.utc).isoformat())
    continuous._atomic_json(status, output / "status.json")
    details = " ".join(f"{key}={value}" for key, value in update.items())
    print(f"{status['updated_at']} {details}", flush=True)


def model_folds(stage):
    for method in METHODS:
        for condition in STAGES[stage]:
            for fold in range(1, N_FOLDS + 1):
                yield method, condition, fold


def resolve_device(device):
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device


def load_run(args, output, cache, device):
    configs = {
        name: load_config(ROOT / "configs" / config)
        for name, config in {"raw": PHASES["0.01"], **PHASES}.items()
    }
    raw = configs["raw"]
    for stage, cfg in configs.items():
        cfg["paths"]["data_root"] = str(args.data_dir)
        if any(cfg[key] != raw[key] for key in ("dataset", "preprocessing")):
            raise ValueError(f"Shared TUEP preparation differs for {stage}")
        if cfg["channels"]["standard_19"] != raw["channels"]["standard_19"]:
            raise ValueError(f"Channel order differs for {stage}")
    index = build_recording_index(raw)
    if index.empty:
        raise ValueError(f"No TUEP EDFs discovered in {args.data_dir}")
    discovered = index["patient_id"].nunique()
    if discovered < args.expected_patients:
        raise ValueError(f"Found only {discovered} patients before epoch rejection, "
                         f"expected {args.expected_patients}; check --data-dir")
    index = index.sort_values("edf_path").reset_index(drop=True)
    index["record_cache_id"] = index["edf_path"].map(
        lambda path: hashlib.sha256(str(Path(path).relative_to(args.data_dir)).encode()).hexdigest()[:24]
    )
    if not index["record_cache_id"].is_unique:
        raise ValueError("Duplicate recording cache identity")
    records = index.to_dict("records")
    run_meta = {
        "cache": str(cache), "device": device, "workers": args.workers,
        "expected_patients": args.expected_patients, "profile": PROFILE.hash,
        "folds": N_FOLDS, "seed": SEED, "configs": configs,
        "versions": {package: version(package) for package in
                     ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "torch", "shap", "mne")},
        "inputs": digest(dataset_identity(records)),
    }
    source_files = [
        Path(__file__), ROOT / "scripts/tuep_shared_cv_data.py",
        ROOT / "scripts/23_train_tuep_component_eegnet.py",
        ROOT / "scripts/21_cache_tuep_continuous_wiener.py",
        ROOT / "scripts/22_extract_tuep_component_epochs.py",
        *sorted((ROOT / "eeg_bg").rglob("*.py")),
    ]
    run_meta["code"] = digest({str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in source_files})
    fingerprint = digest(run_meta)
    manifest = output / "run.json"
    if manifest.is_file() and read_json(manifest)["fingerprint"] != fingerprint:
        raise ValueError("Run inputs, code or settings changed. Use the original settings or a new output directory.")
    continuous._atomic_json({**run_meta, "fingerprint": fingerprint}, manifest)
    audit = index.assign(annotation_available=index["edf_path"].map(
        lambda path: Path(path).with_suffix(".csv_bi").is_file()
    ))
    continuous._atomic_dataframe(audit, output / "recordings.csv")
    return configs, records, fingerprint


def run(args, output, cache, status):
    from tuep_shared_cv_report import summarize

    device = resolve_device(args.device)
    configs, records, fingerprint = load_run(args, output, cache, device)
    cache.mkdir(parents=True, exist_ok=True)
    owner = cache / "owner.json"
    if owner.is_file():
        if read_json(owner) != {"fingerprint": fingerprint}:
            raise ValueError(f"Temporary cache belongs to a different experiment: {cache}")
    elif any(cache.iterdir()):
        raise ValueError(f"Temporary cache has no matching run manifest: {cache}")
    else:
        continuous._atomic_json({"fingerprint": fingerprint}, owner)
    windows_path = output / "windows.csv"
    folds_path = output / "patient_folds.csv"
    windows = read_csv(windows_path) if windows_path.is_file() else None
    plan = None if windows is None else split_patients(windows, args.expected_patients)
    if folds_path.is_file():
        pd.testing.assert_frame_equal(read_csv(folds_path), plan, check_dtype=False)
    total_done = 0
    newly_done = 0
    for stage, conditions in STAGES.items():
        cfg = configs[stage]
        stage_output = output / stage
        stage_cache = cache / stage
        if stage_cache.is_symlink():
            raise ValueError(f"Phase cache path may not be a symlink: {stage_cache}")
        expected = list(model_folds(stage))
        already_done = {
            (method, condition, fold)
            for method, condition, fold in expected
            if completed(fold_path(output, stage, method, condition, fold), fold_files(method))
        }
        if len(already_done) == len(expected):
            if plan is None:
                raise ValueError("Missing shared windows for a completed phase")
            progress(output, status, phase=stage, step="cache_cleanup")
            for method, condition, fold in expected:
                held = windows[windows["evaluation_id"].isin(
                    plan.loc[plan["outer_fold"] == fold, "evaluation_id"])]
                bg = windows.loc[background_rows(windows, plan, fold, method)]
                validate_fold(fold_path(output, stage, method, condition, fold),
                              held, bg, method, condition, cfg)
            clear_phase_cache(stage_cache)
            total_done += len(expected)
            progress(output, status, completed_folds=total_done)
            continue

        stage_output.mkdir(parents=True, exist_ok=True)
        continuous._atomic_yaml(cfg, stage_output / "config.yaml")
        waveforms = stage_cache / "epochs"
        features = stage_cache / "features"
        progress(output, status, phase=stage, step="preprocessing_and_epochs", method=None, condition=None, fold=None)
        current = prepare_epochs(records, cfg, stage, waveforms, windows, args.workers)
        if windows is None:
            windows = current
            plan = split_patients(windows, args.expected_patients)
            continuous._atomic_dataframe(windows, windows_path)
        continuous._atomic_dataframe(plan, folds_path)
        progress(output, status, step="features")
        features.mkdir(parents=True, exist_ok=True)
        banks = ("raw",) if stage == "raw" else COMPONENTS
        jobs = [(key, cfg, banks, waveforms, features) for key in windows["record_cache_id"].unique()]
        run_record_jobs(extract_features, jobs, args.workers, "Feature generation")
        for method in METHODS:
            for condition in conditions:
                pending = [fold for fold in range(1, N_FOLDS + 1)
                           if (method, condition, fold) not in already_done]
                matrix = (feature_matrix(windows, features, condition)
                          if method == "xgboost" and pending else None)
                for fold in range(1, N_FOLDS + 1):
                    path = fold_path(output, stage, method, condition, fold)
                    progress(output, status, step="train_and_shap", method=method,
                             condition=condition, fold=fold)
                    with threadpool_limits(limits=args.workers):
                        new = run_fold(
                            path, cfg, method, condition, windows, plan, waveforms, matrix,
                            fold, device, args.workers,
                            digest([fingerprint, stage, method, condition, fold]),
                        )
                    newly_done += new
                    total_done += 1
                    progress(output, status, completed_folds=total_done)
                    if (new and args.max_model_folds == newly_done
                        and total_done < status["total_folds"]):
                        if (method, condition, fold) == expected[-1]:
                            progress(output, status, step="cache_cleanup")
                            clear_phase_cache(stage_cache)
                        progress(output, status, state="paused", step="pilot_limit")
                        return
                del matrix
            if method == "eegnet":
                import_module("23_train_tuep_component_eegnet")._release_condition_memory(device)
        progress(output, status, step="cache_cleanup", method=None, condition=None, fold=None)
        clear_phase_cache(stage_cache)

    progress(output, status, step="summary", phase=None, method=None, condition=None, fold=None)
    summarize(output, windows, plan, STAGES, METHODS, N_FOLDS)
    owner.unlink()
    cache.rmdir()
    progress(output, status, state="completed", step="done")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="run directory; reuse it to resume an identical run")
    parser.add_argument("--data-dir", type=Path, help="TUEP v3.1.0 root")
    parser.add_argument("--cache-dir", type=Path, help="parent for isolated temporary caches")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expected-patients", type=int, default=178,
                        help="expected eligible patient count; 178 also checks the historical class counts")
    parser.add_argument("--max-model-folds", type=int,
                        help="pause after this many newly finished model folds (pilot and resume)")
    args = parser.parse_args()
    if args.workers < 1 or args.expected_patients < 2 * N_FOLDS or (
        args.max_model_folds is not None and args.max_model_folds < 1
    ):
        parser.error("Require positive workers/pilot limit and at least 20 expected patients")
    defaults = load_config(ROOT / "configs" / PHASES["0.01"])["paths"]
    args.data_dir = (args.data_dir or Path(defaults["data_root"])).expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    cache_parent = (args.cache_dir or Path(defaults["cache_dir"])).expanduser().resolve()
    cache = cache_parent / f"tuep-10fold-{digest(str(output))[:24]}"
    if not args.data_dir.is_dir():
        parser.error(f"TUEP data directory does not exist: {args.data_dir}")
    if output.is_relative_to(cache) or args.data_dir.is_relative_to(cache) or cache.is_symlink():
        parser.error("The run and the TUEP dataset must be outside the temporary cache subtree")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"Another process is running in {output}")

        def terminate(signum, frame):
            raise KeyboardInterrupt(f"Signal {signum}")

        signal.signal(signal.SIGTERM, terminate)
        status = {"pid": os.getpid(), "cache_dir": str(cache), "completed_folds": 0,
                  "total_folds": len(METHODS) * N_FOLDS * sum(map(len, STAGES.values()))}
        try:
            progress(output, status, state="running", step="initialize")
            run(args, output, cache, status)
        except KeyboardInterrupt:
            progress(output, status, state="interrupted")
            return 130
        except Exception as error:
            progress(output, status, state="failed", error=f"{type(error).__name__}: {error}")
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
