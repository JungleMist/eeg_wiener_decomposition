#!/usr/bin/env python3
"""Fixed-subject ds003478 ECMAD, EEGNet and XGBoost comparison."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from eeg_bg.config.settings import load_config

ROOT = Path(__file__).resolve().parents[1]
BIDS_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
COMPONENTS = ("specific", "coherent", "specific_coherent")


def atomic_write(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(data)
    temp.replace(path)


def phase_key(phase):
    return f"phase_{phase:g}"


def subject_manifest(data_root, settings):
    subjects = pd.read_csv(data_root / "participants.tsv", sep="\t")
    if subjects["participant_id"].duplicated().any():
        raise ValueError("Duplicate participant IDs")
    bdi = subjects["BDI"]
    high = bdi.ge(settings["bdi_depressing_min"])
    low = bdi.le(settings["bdi_healthy_max"])
    if (high & low).any():
        raise ValueError("The two BDI ranges must be disjoint")
    subjects["label"] = pd.Series(pd.NA, index=subjects.index, dtype="Int64")
    subjects.loc[high, "label"] = 1
    subjects.loc[low, "label"] = 0
    subjects["class_name"] = subjects["label"].map({1: "depressing", 0: "healthy"})
    subjects["split"] = "excluded"
    subjects["exclusion_reason"] = np.select(
        [bdi.isna(), ~(high | low)], ["missing_BDI", "BDI_between_thresholds"], default="",
    )
    eligible = subjects.loc[high | low].sort_values("participant_id")
    seed = settings["random_state"]
    development, test = train_test_split(
        eligible, stratify=eligible["label"], random_state=seed,
        test_size=settings["test_fraction"],
    )
    train, val = train_test_split(
        development, stratify=development["label"], random_state=seed,
        test_size=settings["validation_fraction_within_development"],
    )
    for split, group in (("train", train), ("validation", val), ("test", test)):
        subjects.loc[group.index, "split"] = split
    return subjects


def input_identity(path):
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


def prepare_record(args):
    import mne

    from eeg_bg.preprocessing.continuous import wiener_continuous_raw

    source, destination, cfg = args
    if destination.exists():
        with np.load(destination, allow_pickle=False) as cache:
            return source.stem, len(cache["starts"])
    prep = cfg["preprocessing"]
    eeg = mne.io.read_raw_eeglab(source, preload=True, verbose="ERROR")
    names = {name.upper(): name for name in eeg.ch_names}
    channels = cfg["channels"]["standard_19"]
    source_names = [names[BIDS_ALIASES.get(name.upper(), name.upper())] for name in channels]
    raw = eeg.pick(source_names).reorder_channels(source_names)
    raw.rename_channels(dict(zip(raw.ch_names, channels)))
    raw.set_eeg_reference(ref_channels=channels, projection=False, verbose="ERROR")
    raw.filter(*prep["bandpass"], method="iir", verbose="ERROR")
    sfreq = prep["target_sfreq"]
    raw.resample(sfreq, verbose="ERROR")
    signal_uv = raw.get_data() * 1e6
    if not np.isfinite(signal_uv).all():
        raise ValueError(f"Nonfinite EEG in {source}")
    size = round(sfreq * prep["epoch_length_sec"])
    starts = np.arange(0, raw.n_times - size + 1, size)
    windows = starts[:, None] + np.arange(size)
    raw_epochs = signal_uv[:, windows].transpose(1, 0, 2)
    keep = np.max(np.abs(raw_epochs), axis=(1, 2)) <= prep["artifact_threshold_uv"]
    retained = starts[keep]
    raw_epochs = raw_epochs[keep]
    payload = {"starts": retained, "raw": raw_epochs.astype(np.float32)}
    # FFT and overlap-add span the input block. Removing rejected windows
    # before fitting prevents their transients from leaking into clean epochs.
    breaks = np.flatnonzero(np.diff(retained) != size) + 1
    blocks = []
    for block in np.split(retained, breaks):
        if len(block):
            block_data = signal_uv[:, block[0]:block[-1] + size] * 1e-6
            blocks.append(mne.io.RawArray(block_data, raw.info.copy(), verbose="ERROR"))
    for phase in cfg["ds003478"]["phase_thresholds_rad"]:
        phase_cfg = deepcopy(cfg)
        phase_cfg["wiener"]["phase_gate_threshold_rad"] = phase
        parts = []
        for block in blocks:
            specific, _ = wiener_continuous_raw(block, phase_cfg, subject_id=source.stem)
            parts.append(specific.get_data().reshape(len(channels), -1, size).transpose(1, 0, 2))
        specific_uv = np.concatenate(parts) * 1e6 if parts else np.empty_like(raw_epochs)
        for condition, values in (("specific", specific_uv), ("coherent", raw_epochs - specific_uv)):
            epochs = values.astype(np.float32)
            if not np.isfinite(epochs).all():
                raise ValueError(f"Nonfinite {condition} at {phase} rad in {source}")
            payload[f"{phase_key(phase)}_{condition}"] = epochs
    write_npz(destination, payload)
    return source.stem, int(keep.sum())


def prepare(data_root, output, cfg, workers):
    subjects = subject_manifest(data_root, cfg["ds003478"])
    sources = sorted(data_root.glob("sub-*/eeg/*_task-Rest_run-*_eeg.set"))
    if {p.parents[1].name for p in sources} != set(subjects["participant_id"]):
        raise ValueError("EEG recordings and participants.tsv disagree")
    frozen = {
        "config": cfg, "source_files": [
            input_identity(file)
            for source in sources for file in (source, source.with_suffix(".fdt"))
        ],
        "labels_sha256": hashlib.sha256((data_root / "participants.tsv").read_bytes()).hexdigest(),
        "channel_aliases": BIDS_ALIASES, "reference": "average_of_selected_channels",
        "window_policy": "overlap_add_within_contiguous_raw_accepted_epochs",
    }
    # Compare a stable JSON representation of paths and effective settings.
    frozen_bytes = json.dumps(frozen, indent=2, sort_keys=True, default=str).encode()
    frozen_path = output / "run_config.json"
    if frozen_path.exists() and frozen_path.read_bytes() != frozen_bytes:
        raise ValueError(f"Input or configuration changed. Use a new output directory: {output}")
    atomic_write(frozen_path, frozen_bytes)
    atomic_write(output / "subject_split.csv", subjects.to_csv(index=False).encode())
    cache = output / "epochs"
    cache.mkdir(exist_ok=True)
    jobs = [(source, cache / (source.stem + ".npz"), cfg) for source in sources]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (recording, count) in enumerate(pool.map(prepare_record, jobs), start=1):
            rows.append({"recording_id": recording, "participant_id": recording.split("_")[0],
                         "n_epochs": count})
            if i % 20 == 0 or i == len(jobs):
                print(f"ECMAD {i}/{len(jobs)} recordings", flush=True)
    records = pd.DataFrame(rows).merge(
        subjects[["participant_id", "split", "label", "class_name", "exclusion_reason"]],
        on="participant_id", validate="many_to_one",
    )
    counts = records.groupby("participant_id")["n_epochs"].sum()
    missing = subjects.loc[
        subjects["split"].ne("excluded") & subjects["participant_id"].map(counts).eq(0),
        "participant_id",
    ]
    if not missing.empty:
        raise ValueError(f"No retained EEG for {missing.tolist()}")
    atomic_write(output / "recordings.csv", records.to_csv(index=False).encode())


def condition_specs(cfg):
    yield "raw", ("raw",), None
    for phase in cfg["ds003478"]["phase_thresholds_rad"]:
        keys = tuple(f"{phase_key(phase)}_{condition}" for condition in COMPONENTS[:2])
        yield keys[0], keys[:1], phase
        yield keys[1], keys[1:], phase
        yield f"{phase_key(phase)}_{COMPONENTS[2]}", keys, phase


def read_records(output):
    records = pd.read_csv(output / "recordings.csv")
    return records.loc[(records["split"] != "excluded") & (records["n_epochs"] > 0)]


def matrix_for_split(output, records, directory, keys, split, normalize=None):
    chunks = []
    rows = []
    for record in records.loc[records["split"].eq(split)].itertuples():
        with np.load(output / directory / f"{record.recording_id}.npz") as data:
            values = data[keys[0]] if len(keys) == 1 else np.concatenate(
                [data[key] for key in keys], axis=1,
            )
            if normalize is not None:
                values = normalize(values)
            if len(values) != record.n_epochs:
                raise ValueError(f"Epoch count mismatch for {record.recording_id}")
            chunks.append(values)
            rows.append(pd.DataFrame({
                "participant_id": record.participant_id,
                "label": int(record.label),
                "recording_id": record.recording_id,
                "epoch_start_sample": data["starts"],
            }))
    return np.concatenate(chunks), pd.concat(rows, ignore_index=True)


def equal_subject_weights(rows):
    counts = rows.groupby("participant_id")["label"].transform("size")
    return 1.0 / counts.to_numpy(dtype=np.float64)


def subject_predictions(rows, scores):
    return rows.assign(pred_proba=scores).groupby("participant_id", as_index=False).agg(
        true_label=("label", "first"), pred_proba=("pred_proba", "mean"),
        n_epochs=("label", "size"),
    )


def evaluate(path, validation, test, *, threshold=None):
    from eeg_bg.ml.erp_eegnet import (
        classification_metrics, select_balanced_accuracy_threshold,
    )

    if threshold is None:
        threshold = select_balanced_accuracy_threshold(
            validation["true_label"].to_numpy(), validation["pred_proba"].to_numpy(),
        )
    metrics = {}
    for split, frame in (("validation", validation), ("test", test)):
        frame["predicted_label"] = (frame["pred_proba"] >= threshold).astype(int)
        frame.to_csv(path / f"{split}_predictions.csv", index=False)
        metrics[split] = classification_metrics(frame.true_label, frame.pred_proba, threshold)
    return metrics


def write_npz(path, payload):
    temp = path.with_suffix(".tmp.npz")
    np.savez(temp, **payload)
    temp.replace(path)


def write_torch(path, payload):
    import torch

    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def extract_record_features(args):
    from eeg_bg.features.profiles import PROFILES

    source, target, cfg, keys = args
    if target.exists():
        return
    profile = PROFILES[cfg["ds003478"]["feature_profile"]]
    sfreq = cfg["preprocessing"]["target_sfreq"]
    nperseg = cfg["wiener"]["nperseg"]
    freq_band = tuple(cfg["wiener"]["freq_band"])
    connectivity_nperseg = cfg["ml"]["features"]["connectivity"]["nperseg"]
    with np.load(source) as data:
        features = {
            key: np.asarray([
                profile.extract_fn(epoch, cfg["channels"]["standard_19"], sfreq, nperseg,
                                   freq_band, connectivity_nperseg)
                for epoch in data[key]
            ], dtype=np.float32)
            for key in keys
        }
        write_npz(target, {"starts": data["starts"], **features})


def prepare_features(output, records, cfg, specs, workers):
    target = output / "features"
    target.mkdir(exist_ok=True)
    keys = sorted({key for _, keys, _ in specs for key in keys})
    jobs = [(output / "epochs" / filename, target / filename, cfg, keys)
            for filename in (f"{name}.npz" for name in records["recording_id"])]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(extract_record_features, jobs), start=1):
            if i % 20 == 0 or i == len(jobs):
                print(f"XGBoost features {i}/{len(jobs)} recordings", flush=True)


def validation_ap(frame):
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(frame["true_label"], frame["pred_proba"]))


def fit_xgboost(output, path, records, keys, cfg, workers):
    import matplotlib
    matplotlib.use("Agg")
    import joblib
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    from eeg_bg.features.profiles import PROFILES
    from eeg_bg.ml.shap_analysis import compute_shap_values, plot_shap_summary
    from eeg_bg.ml.xgb_pipeline import find_optimal_threshold, train_xgboost

    train_x, train_rows = matrix_for_split(output, records, "features", keys, "train")
    val_x, val_rows = matrix_for_split(output, records, "features", keys, "validation")
    model_path = path / "model.ubj"
    scaler_path = path / "scaler.joblib"
    selection_path = path / "training_parameters.json"
    if model_path.exists() and selection_path.exists():
        model = XGBClassifier()
        model.load_model(model_path)
        scaler = joblib.load(scaler_path)
        val_x = scaler.transform(val_x)
        selection = json.loads(selection_path.read_text())
    else:
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        val_x = scaler.transform(val_x)
        training_cfg = deepcopy(cfg)
        training_cfg["ml"]["xgboost"]["fixed"].update(device="cpu", n_jobs=1)
        print(f"XGBoost {path.name}: TUEP grouped grid search", flush=True)
        model = train_xgboost(
            train_x, train_rows["label"].to_numpy(), val_x, val_rows["label"].to_numpy(),
            training_cfg, groups=train_rows["participant_id"].to_numpy(), grid_n_jobs=workers,
        )
        selection = {
            "best_iteration": model.best_iteration,
            "best_validation_epoch_auroc": model.best_score,
            "parameters": model.get_xgb_params(),
        }
        model.save_model(path / "model.tmp.ubj")
        joblib.dump(scaler, scaler_path)
        atomic_write(selection_path, json.dumps(selection, indent=2, allow_nan=False).encode())
        (path / "model.tmp.ubj").replace(model_path)
    test_x, test_rows = matrix_for_split(output, records, "features", keys, "test")
    test_x = scaler.transform(test_x)
    validation = subject_predictions(val_rows, model.predict_proba(val_x)[:, 1])
    test = subject_predictions(test_rows, model.predict_proba(test_x)[:, 1])
    metrics = evaluate(path, validation, test, threshold=find_optimal_threshold(validation))
    metrics["selection"] = selection
    # Prediction uses the stopping point. Use the same tree prefix for SHAP.
    stopped_booster = model.get_booster()[:model.best_iteration + 1]
    base_names = PROFILES[cfg["ds003478"]["feature_profile"]].names
    names = base_names if len(keys) == 1 else [
        f"{component}:{name}" for component in COMPONENTS[:2] for name in base_names
    ]
    values = compute_shap_values(stopped_booster, test_x, names)
    if values.shape != test_x.shape or not np.isfinite(values).all():
        raise ValueError("Invalid XGBoost SHAP output")
    write_npz(path / "shap_test.npz", {"values": values, "names": names, "features": test_x})
    test_rows.to_csv(path / "shap_test_epochs.csv", index=False)
    importance = pd.DataFrame({
        "feature": names,
        "mean_abs_shap": np.average(np.abs(values), axis=0,
                                    weights=equal_subject_weights(test_rows)),
    })
    importance.sort_values("mean_abs_shap", ascending=False).to_csv(
        path / "shap_feature_importance.csv", index=False,
    )
    plot_shap_summary(values, test_x, names, title=f"{path.name} test TreeSHAP",
                      output_path=path / "shap_summary.png")
    return metrics


def normalize_epochs(values):
    avg = values.mean(axis=-1, keepdims=True)
    sd = values.std(axis=-1, keepdims=True)
    return (values - avg) / np.maximum(sd, 1e-6)


def sample_each_subject(rows, rng, limit):
    return np.concatenate([
        rng.choice(indices, size=min(limit, len(indices)), replace=False)
        for indices in rows.groupby("participant_id").indices.values()
    ])


def eegnet_shap(model, train_x, train_rows, test_x, test_rows, path, keys, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    import torch

    settings = cfg["ds003478"]
    rng = np.random.default_rng(settings["random_state"])
    candidates = sample_each_subject(train_rows, rng, 1)
    bg = rng.choice(candidates, min(len(candidates), settings["eegnet_shap_background_subjects"]),
                    replace=False)
    explained = sample_each_subject(test_rows, rng, 2)
    model.eval()
    device = next(model.parameters()).device
    explainer = shap.GradientExplainer(
        model, train_x[bg].unsqueeze(1).to(device), batch_size=16,
    )
    values = []
    for start in range(0, len(explained), 4):
        selection = explained[start:start + 4]
        batch = explainer.shap_values(test_x[selection].unsqueeze(1).to(device),
                                      nsamples=settings["eegnet_shap_samples"],
                                      rseed=settings["random_state"] + start)
        values.append(np.asarray(batch)[:, 0, :, :, 0])
    values = np.concatenate(values)
    if values.shape != tuple(test_x[explained].shape) or not np.isfinite(values).all():
        raise ValueError("Invalid EEGNet SHAP output")
    channels = cfg["channels"]["standard_19"]
    names = channels if len(keys) == 1 else [
        f"{component}:{name}" for component in COMPONENTS[:2] for name in channels
    ]
    write_npz(path / "shap_test.npz", {"values": values, "names": names})
    explained_rows = test_rows.iloc[explained]
    explained_rows.to_csv(path / "shap_test_epochs.csv", index=False)
    train_rows.iloc[bg].to_csv(path / "shap_background_epochs.csv", index=False)
    importance = np.average(np.abs(values).mean(axis=-1), axis=0,
                            weights=equal_subject_weights(explained_rows))
    ranking = pd.DataFrame({"channel": names, "mean_abs_shap": importance}).sort_values(
        "mean_abs_shap", ascending=False,
    )
    ranking.to_csv(path / "shap_channel_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    top = ranking.head(20).iloc[::-1]
    ax.barh(top["channel"], top["mean_abs_shap"])
    ax.set_xlabel("Subject-weighted mean absolute SHAP (probability)")
    fig.tight_layout()
    fig.savefig(path / "shap_summary.png", dpi=150)
    plt.close(fig)


def fit_eegnet(output, path, records, keys, cfg, device, workers, *, compute_shap=True):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from importlib import import_module
    from eeg_bg.ml.cnn_model import EEGNet

    tuep = import_module("23_train_tuep_component_eegnet")

    model_cfg = cfg["ml"]["tuep_component_eegnet"]
    condition = COMPONENTS[2] if len(keys) == 2 else keys[0].split("_")[-1]
    seed = model_cfg["random_state"] + tuep.CONDITION_SEED_OFFSETS[condition]
    tuep._set_random_seed(seed)
    torch.set_num_threads(workers)
    matrix_args = (output, records, "epochs", keys)

    def load(split):
        data, rows = matrix_for_split(*matrix_args, split, normalize=normalize_epochs)
        return torch.from_numpy(data), rows

    def predict(values, rows):
        model.eval()
        with torch.no_grad():
            probabilities = torch.cat([
                model(batch.to(device).unsqueeze(1)).flatten().cpu()
                for batch in values.split(batch_size)
            ]).numpy()
        return subject_predictions(rows, probabilities)

    train_x, train_rows = load("train")
    val_x, val_rows = load("validation")
    batch_size = model_cfg["batch_size"]
    n_channels, n_times = train_x.shape[1:]
    architecture = {"n_channels": n_channels, "n_times": n_times,
                    **{key: model_cfg[key] for key in ("F1", "D", "dropout")}}
    model = EEGNet(**architecture).to(device)
    if device.type == "cpu":
        model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=model_cfg["lr"], weight_decay=model_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=model_cfg["lr_patience"], factor=model_cfg["lr_factor"],
    )
    criterion = nn.BCELoss(reduction="none")
    patients = [
        {"epochs": train_x[indices], "label": int(train_rows["label"].iloc[indices[0]])}
        for indices in train_rows.groupby("participant_id").indices.values()
    ]
    class_weights = tuep.patient_class_weights(patients)
    train_rng = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        patients, batch_size=None, shuffle=True, num_workers=model_cfg["num_workers"],
        generator=train_rng,
    )
    best_score, best_epoch, best_state = -np.inf, 0, None
    history = []
    progress_path = path / "last_training.pt"
    backend = getattr(torch, device.type) if device.type != "cpu" else None
    if progress_path.exists():
        cached = torch.load(progress_path, map_location="cpu", weights_only=True)
        model.load_state_dict(cached["model"])
        optimizer.load_state_dict(cached["optimizer"])
        scheduler.load_state_dict(cached["scheduler"])
        train_rng.set_state(cached["train_rng"])
        torch.set_rng_state(cached["cpu_rng"])
        if backend is not None and cached["device"] == str(device):
            backend.set_rng_state(cached["device_rng"])
        history = cached["history"]
        best_score, best_epoch, best_state = cached["best"]
    stale_epochs = len(history) - best_epoch
    for epoch in range(len(history) + 1, model_cfg["max_epochs"] + 1):
        if stale_epochs >= model_cfg["patience"]:
            break
        loss = tuep._train_epoch(
            model, loader, optimizer, criterion, device, class_weights,
            batch_size, model_cfg["records_per_step"],
        )
        score = validation_ap(predict(val_x, val_rows))
        scheduler.step(score)
        history.append({"epoch": epoch, "training_loss": loss,
                        "validation_auprc": score, "lr": optimizer.param_groups[0]["lr"]})
        if epoch == 1 or epoch % 5 == 0:
            print(f"EEGNet {path.name} epoch {epoch}: val AP {score:.3f}", flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {key: tensor.detach().cpu().clone()
                          for key, tensor in model.state_dict().items()}
        stale_epochs = epoch - best_epoch
        write_torch(progress_path, {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "train_rng": train_rng.get_state(),
            "cpu_rng": torch.get_rng_state(),
            "device_rng": backend.get_rng_state() if backend is not None else None,
            "device": str(device), "history": history, "best": (best_score, best_epoch, best_state),
        })
    model.load_state_dict(best_state)
    write_torch(path / "best_model.pt", {
        "architecture": architecture, "state_dict": best_state,
        "positive_class": "1=depressing (BDI ≥ 20)",
    })
    pd.DataFrame(history).to_csv(path / "history.csv", index=False)
    validation = predict(val_x, val_rows)
    del val_x
    test_x, test_rows = load("test")
    metrics = evaluate(path, validation, predict(test_x, test_rows))
    metrics["selection"] = {"epoch": best_epoch}
    metrics["device"] = str(device)
    if compute_shap:
        eegnet_shap(model, train_x, train_rows, test_x, test_rows, path, keys, cfg)
    return metrics


def train_models(method, output, cfg, workers, device_arg):
    import torch

    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    records = read_records(output)
    specs = list(condition_specs(cfg))
    if method == "xgboost":
        prepare_features(output, records, cfg, specs, workers)
    for condition, keys, phase in specs:
        path = output / method / condition
        path.mkdir(parents=True, exist_ok=True)
        metrics_path = path / "metrics.json"
        if metrics_path.exists():
            continue
        print(f"Training {method} {condition}", flush=True)
        if method == "eegnet":
            metrics = fit_eegnet(output, path, records, keys, cfg,
                                 torch.device(device_arg), workers)
        else:
            metrics = fit_xgboost(output, path, records, keys, cfg, workers)
        metrics["phase_rad"] = phase
        atomic_write(metrics_path, json.dumps(metrics, indent=2, allow_nan=False).encode())
        print(f"{method} {condition} complete, test AUROC {metrics['test']['auroc']:.3f}",
              flush=True)


def paired_auroc(y, predictions, reference, seed):
    pos, neg = y == 1, y == 0

    def pair_scores(probabilities):
        difference = probabilities[pos, None] - probabilities[None, neg]
        return (difference > 0) + 0.5 * (difference == 0)

    delta = pair_scores(predictions) - pair_scores(reference)
    rng = np.random.default_rng(seed)
    weights = [rng.multinomial(n, np.full(n, 1 / n), size=4000) / n
               for n in (pos.sum(), neg.sum())]
    samples = np.einsum("bi,ij,bj->b", weights[0], delta, weights[1])
    return delta.mean(), *np.quantile(samples, [0.025, 0.975])


def compare(output, cfg):
    lines = []
    contributions = []
    split = pd.read_csv(output / "subject_split.csv")
    cohort = split.loc[split["split"].eq("test")].set_index("participant_id")["label"].astype(int)
    for method in ("eegnet", "xgboost"):
        base = pd.read_csv(output / method / "raw" / "test_predictions.csv").set_index("participant_id")
        for condition, _, phase in condition_specs(cfg):
            path = output / method / condition
            result = json.loads((path / "metrics.json").read_text())
            frame = pd.read_csv(path / "test_predictions.csv").set_index("participant_id")
            if not frame["true_label"].sort_index().equals(cohort.sort_index()):
                raise ValueError(f"Test cohort mismatch: {path}")
            delta, low, high = paired_auroc(
                frame["true_label"].to_numpy(), frame["pred_proba"].to_numpy(),
                base.loc[frame.index, "pred_proba"].to_numpy(), cfg["ds003478"]["random_state"],
            )
            lines.append({
                "method": method, "condition": condition, "phase_rad": phase,
                **{f"{part}_{key}": value for part in ("validation", "test")
                   for key, value in result[part].items() if key != "confusion_matrix"},
                "test_auroc_delta_vs_raw": delta, "delta_ci95_low": low, "delta_ci95_high": high,
            })
            if condition.endswith("_specific_coherent"):
                field = "channel" if method == "eegnet" else "feature"
                ranking = pd.read_csv(path / f"shap_{field}_importance.csv")
                bank = ranking[field].str.split(":", n=1).str[0]
                mass = ranking.groupby(bank)["mean_abs_shap"].sum()
                total = mass.sum()
                for component, value in mass.items():
                    contributions.append({
                        "method": method, "phase_rad": phase, "component": component,
                        "sum_mean_abs_shap": value,
                        "fraction_within_combined_model": value / total if total else np.nan,
                    })
    summary = pd.DataFrame(lines)
    atomic_write(output / "comparison.csv", summary.to_csv(index=False).encode())
    atomic_write(output / "comparison_shap_components.csv",
                 pd.DataFrame(contributions).to_csv(index=False).encode())
    write_report(output, summary, split, pd.DataFrame(contributions))


def write_report(output, summary, split, contributions):
    counts = split.groupby(["split", "class_name"]).size().unstack(fill_value=0)
    sections = [
        "# ds003478 ECMAD 分类比较",
        "",
        "| 分区 | depressing | healthy |",
        "|---|---:|---:|",
        *(f"| {name} | {row['depressing']} | {row['healthy']} |"
          for name, row in counts.iterrows()),
        "",
        "标签依据 BDI ≥ 20 和 BDI ≤ 13。分类结果衡量对 BDI 分组的区分能力。",
        "以各受试者的平均 epoch 概率计算指标。ΔAUROC 相对同一模型架构的 raw，"
        "区间采用 4,000 次按类别分层的配对受试者 bootstrap，未经多重比较校正。",
        "",
    ]
    for method, rows in summary.groupby("method", sort=False):
        selected = rows.loc[rows["validation_auprc"].idxmax()]
        sections.extend([
            f"## {method}",
            "",
            f"按验证集 AUPRC 选出的条件为 `{selected.condition}`。",
            "",
            "| 条件 | 验证 AUPRC | 测试 AUROC | 测试 AUPRC | 平衡准确率 | F1 | ΔAUROC [95% CI] |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *(f"| {r.condition} | {r.validation_auprc:.3f} | {r.test_auroc:.3f} | "
              f"{r.test_auprc:.3f} | {r.test_balanced_accuracy:.3f} | {r.test_f1:.3f} | "
              f"{r.test_auroc_delta_vs_raw:+.3f} [{r.delta_ci95_low:+.3f}, "
              f"{r.delta_ci95_high:+.3f}] |" for r in rows.itertuples()),
            "",
        ])
    sections.extend([
        "## 拼接组的 SHAP 贡献",
        "",
        "| 模型 | 相位阈值（rad） | 分量 | 模型内绝对贡献占比 |",
        "|---|---:|---|---:|",
        *(f"| {r.method} | {r.phase_rad:g} | {r.component} | "
          f"{r.fraction_within_combined_model:.1%} |" for r in contributions.itertuples()),
        "",
        "贡献以受试者等权的平均绝对 SHAP 为基础。EEGNet 计算训练背景下的 "
        "expected gradients，XGBoost 计算 TreeSHAP。占比只在各自拼接模型中解释。",
        "",
    ])
    atomic_write(output / "comparison.md", "\n".join(sections).encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "xgboost", "eegnet", "compare", "all"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ECMAD.yaml")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    cfg = load_config(args.config)
    phases = cfg["ds003478"]["phase_thresholds_rad"]
    if len(set(phases)) != 4 or not all(0 <= p <= np.pi for p in phases):
        parser.error("Expected four distinct phase thresholds in [0, pi]")
    channels = cfg["channels"]["standard_19"]
    groups = cfg["channels"]["channel_groups"]
    if set().union(*map(set, groups)) != set(channels):
        parser.error("ECMAD groups must cover the selected channels with exact spelling")
    data_root = (args.data_dir or Path(cfg["paths"]["data_root"])).expanduser().resolve()
    output = (args.output_dir or Path(cfg["paths"]["results_dir"])).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    stages = ("prepare", "eegnet", "xgboost", "compare") if args.stage == "all" else (args.stage,)
    for stage in stages:
        with (output / f".{stage}.lock").open("a") as lock, threadpool_limits(1):
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"Stage {stage} is already running in {output}") from error
            if stage == "prepare":
                prepare(data_root, output, cfg, args.workers)
                continue
            frozen = json.loads((output / "run_config.json").read_text())
            if frozen["config"] != cfg:
                raise ValueError(f"Configuration differs from the prepared data in {output}")
            if stage == "compare":
                compare(output, cfg)
            else:
                train_models(stage, output, cfg, args.workers, args.device)


if __name__ == "__main__":
    main()
