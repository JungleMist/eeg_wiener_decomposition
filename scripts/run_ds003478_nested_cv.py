#!/usr/bin/env python3
"""Subject LOOCV with inner stratified five-fold selection and fold ensembles."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
import ECMAD as pipeline
from eeg_bg.config.settings import load_config


def save(path, value):
    pipeline.atomic_write(path, json.dumps(value, indent=2, allow_nan=False).encode())



def seal(directory, names, marker="complete.json"):
    hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
              for name in names}
    save(directory / marker, hashes)


def complete(directory, marker="complete.json"):
    path = directory / marker
    if not path.exists():
        return False
    try:
        hashes = json.loads(path.read_text())
        return bool(hashes) and all(
            (directory / name).is_file()
            and hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
            for name, digest in hashes.items()
        )
    except (ValueError, OSError):
        return False


def train_xgb_fold(path, train_x, train_rows, val_x, val_rows, fixed, params, rounds):
    import joblib
    from xgboost import XGBClassifier

    path.mkdir(parents=True, exist_ok=True)
    if complete(path):
        model = XGBClassifier()
        model.load_model(path / "model.ubj")
        return model, joblib.load(path / "scaler.joblib"), pd.read_csv(path / "validation.csv")
    scaler = StandardScaler().fit(train_x)
    model = XGBClassifier(**fixed, **params, n_estimators=500,
                          early_stopping_rounds=rounds)
    model.fit(scaler.transform(train_x), train_rows.label,
              eval_set=[(scaler.transform(val_x), val_rows.label)], verbose=False)
    predictions = pipeline.subject_predictions(
        val_rows, model.predict_proba(scaler.transform(val_x))[:, 1])
    model.save_model(path / "model.tmp.ubj")
    (path / "model.tmp.ubj").replace(path / "model.ubj")
    joblib.dump(scaler, path / "scaler.tmp")
    (path / "scaler.tmp").replace(path / "scaler.joblib")
    pipeline.atomic_write(path / "validation.csv", predictions.to_csv(index=False).encode())
    seal(path, ["model.ubj", "scaler.joblib", "validation.csv"])
    return model, scaler, predictions

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/ECMAD.yaml"))
    parser.add_argument("--method", choices=["eegnet", "xgboost"], required=True)
    parser.add_argument("--outer-index", type=int, required=True)
    parser.add_argument("--condition", default="all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This entry point requires CUDA")
    cfg = load_config(args.config)
    frozen = json.loads((args.cache / "run_config.json").read_text())
    for key in ("channels", "preprocessing", "wiener", "ds003478"):
        if frozen["config"][key] != cfg[key]:
            raise ValueError(f"Cache configuration differs in {key}")
    records = pipeline.read_records(args.cache)
    subjects = records[["participant_id", "label"]].drop_duplicates().sort_values("participant_id")
    if subjects.participant_id.duplicated().any():
        raise ValueError("Conflicting subject labels")
    if not 0 <= args.outer_index < len(subjects):
        parser.error("--outer-index is outside the eligible cohort")
    held = subjects.iloc[args.outer_index].participant_id
    development = subjects[subjects.participant_id != held].reset_index(drop=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg["ds003478"]["random_state"])
    folds = []
    for train, val in cv.split(development, development.label):
        folds.append({"train": development.iloc[train].participant_id.tolist(),
                      "validation": development.iloc[val].participant_id.tolist()})
    out = args.output / args.method / f"outer_{args.outer_index:03d}"
    out.mkdir(parents=True, exist_ok=True)
    identity = {"held_out": held, "folds": folds, "config": cfg,
                "cache_manifest": hashlib.sha256((args.cache / "recordings.csv").read_bytes()).hexdigest(),
                "implementation": {
                    name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                    for name in ("run_ds003478_nested_cv.py", "ECMAD.py",
                                 "23_train_tuep_component_eegnet.py")},
                "inputs": [
                    [str(file.relative_to(args.cache)), file.stat().st_size, file.stat().st_mtime_ns]
                    for directory in ("epochs", "features")
                    for file in sorted((args.cache / directory).glob("*.npz"))
                ]}
    with (out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = out / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != identity:
            raise ValueError("Existing fold has a different configuration or implementation")
        save(manifest, identity)
        for condition, keys, _ in pipeline.condition_specs(cfg):
            if args.condition not in ("all", condition):
                continue
            target = out / condition
            target.mkdir(exist_ok=True)
            if complete(target, "result_complete.json"):
                continue
            run_condition(args, cfg, records, folds, held, keys, target)


def run_condition(args, cfg, records, folds, held, keys, target):
    import torch
    import joblib
    from xgboost import XGBClassifier
    from eeg_bg.ml.cnn_model import EEGNet
    from eeg_bg.ml.erp_eegnet import select_balanced_accuracy_threshold
    from eeg_bg.ml.xgb_pipeline import find_optimal_threshold
    from eeg_bg.ml.shap_analysis import compute_shap_values

    method = args.method
    directory = "epochs" if method == "eegnet" else "features"
    normalize = pipeline.normalize_epochs if method == "eegnet" else None
    inner = []
    # Materialize development data only. Outer EEG is loaded after selection.
    for fold in folds:
        rows = records[records.participant_id != held].copy()
        rows["split"] = np.where(rows.participant_id.isin(fold["train"]), "train", "validation")
        inner.append((
            pipeline.matrix_for_split(args.cache, rows, directory, keys, "train", normalize),
            pipeline.matrix_for_split(args.cache, rows, directory, keys, "validation", normalize),
        ))
    selected_models, oof = [], []
    if method == "xgboost":
        grid = dict(cfg["ml"]["xgboost"]["param_grid"])
        grid.pop("n_estimators", None)
        best_score = -np.inf
        for candidate, params in enumerate(ParameterGrid(grid)):
            models, predictions = [], []
            for fold_index, ((train_x, train_rows), (val_x, val_rows)) in enumerate(inner):
                fixed = dict(cfg["ml"]["xgboost"]["fixed"])
                counts = np.bincount(train_rows.label.astype(int))
                fixed.update(device="cuda", tree_method="hist", n_jobs=args.workers,
                             scale_pos_weight=float(counts[0] / counts[1]))
                path = target / "search" / f"candidate_{candidate:03d}" / f"fold_{fold_index}"
                model, scaler, prediction = train_xgb_fold(
                    path, train_x, train_rows, val_x, val_rows, fixed, params,
                    cfg["ml"]["xgboost"]["early_stopping_rounds"],
                )
                predictions.append(prediction)
                models.append((model, scaler))
            score = float(np.mean([pipeline.validation_ap(frame) for frame in predictions]))
            print(target.name, "candidate", candidate, "inner AP", score, flush=True)
            if score > best_score:
                best_score, selected_models, oof = score, models, predictions
                save(target / "selection.json", {"params": params, "mean_inner_auprc": score})
        threshold = find_optimal_threshold(pd.concat(oof, ignore_index=True))
    else:
        # Reuse TUEP training, including subject-equal loss and checkpoint selection.
        # Its evaluation call receives another copy of inner validation only.
        for index, (fold, data) in enumerate(zip(folds, inner)):
            path = target / f"inner_{index}"
            path.mkdir(exist_ok=True)
            rows = records[records.participant_id != held].copy()
            rows["split"] = np.where(rows.participant_id.isin(fold["train"]), "train", "validation")
            duplicate = rows[rows.split == "validation"].copy()
            duplicate["split"] = "test"
            rows = pd.concat([rows, duplicate], ignore_index=True)
            if not complete(path, "training_complete.json"):
                pipeline.fit_eegnet(args.cache, path, rows, keys, cfg,
                                    torch.device("cuda"), args.workers, compute_shap=False)
                seal(path, ["best_model.pt", "validation_predictions.csv"],
                     "training_complete.json")
            checkpoint = torch.load(path / "best_model.pt", map_location="cpu", weights_only=True)
            model = EEGNet(**checkpoint["architecture"])
            model.load_state_dict(checkpoint["state_dict"])
            selected_models.append((model.cuda().eval(), None))
            oof.append(pd.read_csv(path / "validation_predictions.csv"))
        frame = pd.concat(oof, ignore_index=True)
        threshold = select_balanced_accuracy_threshold(frame.true_label.to_numpy(), frame.pred_proba.to_numpy())
    frame = pd.concat(oof, ignore_index=True)
    frame.to_csv(target / "inner_oof_predictions.csv", index=False)
    save(target / "frozen_selection.json", {"threshold": float(threshold), "ensemble_size": 5})
    outer = records[records.participant_id == held].copy()
    outer["split"] = "test"
    test_x, test_rows = pipeline.matrix_for_split(args.cache, outer, directory, keys, "test", normalize)
    scores = []
    for index, ((model, scaler), data) in enumerate(zip(selected_models, inner)):
        path = target / f"inner_{index}"
        path.mkdir(exist_ok=True)
        explanation = path / "outer"
        explanation.mkdir(exist_ok=True)
        if complete(explanation):
            scores.append(np.load(explanation / "probabilities.npy", allow_pickle=False))
            continue
        if method == "xgboost":
            x = scaler.transform(test_x)
            scores.append(model.predict_proba(x)[:, 1])
            model.save_model(path / "model.ubj")
            joblib.dump(scaler, path / "scaler.joblib")
            values = compute_shap_values(model.get_booster()[:model.best_iteration + 1], x, [])
            pipeline.write_npz(explanation / "outer_shap.npz", {"values": values})
            test_rows.to_csv(explanation / "outer_shap_epochs.csv", index=False)
        else:
            with torch.no_grad():
                scores.append(torch.cat([
                    model(batch.cuda().unsqueeze(1)).flatten().cpu()
                    for batch in torch.from_numpy(test_x).split(cfg["ml"]["tuep_component_eegnet"]["batch_size"])
                ]).numpy())
            (train_x, train_rows), _ = data
            pipeline.eegnet_shap(model, torch.from_numpy(train_x), train_rows,
                                torch.from_numpy(test_x), test_rows, explanation, keys, cfg)
        with (explanation / "probabilities.tmp").open("wb") as handle:
            np.save(handle, scores[-1], allow_pickle=False)
        (explanation / "probabilities.tmp").replace(explanation / "probabilities.npy")
        names = ["probabilities.npy"]
        names += (["outer_shap.npz", "outer_shap_epochs.csv"] if method == "xgboost"
                  else ["shap_test.npz", "shap_test_epochs.csv", "shap_background_epochs.csv",
                        "shap_channel_importance.csv", "shap_summary.png"])
        seal(explanation, names)
    probability = float(np.mean(scores))
    save(target / "result.json", {"participant_id": held, "true_label": int(outer.label.iloc[0]),
                                 "pred_proba": probability, "threshold": float(threshold),
                                 "predicted_label": int(probability >= threshold)})
    names = ["result.json", "inner_oof_predictions.csv", "frozen_selection.json"]
    for index in range(5):
        directory = target / f"inner_{index}" / "outer"
        hashes = json.loads((directory / "complete.json").read_text())
        names.extend(str((directory / name).relative_to(target)) for name in hashes)
    seal(target, names, "result_complete.json")
    print(target, "complete", flush=True)


if __name__ == "__main__":
    main()
