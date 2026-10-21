"""EEGNet backend with subject-equal BCE and strictly separate inference."""
from __future__ import annotations

import random
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from eeg_bg.ml.cnn_model import EEGNet
from eeg_bg.ml.erp_eegnet import classification_metrics, select_balanced_accuracy_threshold
from .artifacts import atomic_write, write_csv
from .decomposition import normalize_trials
from .types import FitResult, TrialBatch

logger = logging.getLogger(__name__)


def subject_weights(subject_ids: np.ndarray) -> np.ndarray:
    """Weights whose trial mean is the mean of the within-subject means."""
    _, inverse, counts = np.unique(subject_ids, return_inverse=True, return_counts=True)
    return len(subject_ids) / (len(counts) * counts[inverse])


def class_ratio(batch: TrialBatch) -> float:
    batch.validate(labeled=True)
    counts = np.bincount(batch.y.astype(int), minlength=2)
    if np.any(counts == 0):
        raise ValueError("Fit and selection partitions must contain both classes")
    return float(counts[0] / counts[1])


def validate_probabilities(probabilities: np.ndarray, size: int) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (size,) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Invalid model probabilities")
    return values


def subject_equal_bce(batch: TrialBatch, probabilities: np.ndarray, positive_weight: float) -> float:
    batch.validate(labeled=True)
    probabilities = validate_probabilities(probabilities, len(batch.X))
    # Match torch BCELoss's finite -100 logarithm clamp, including exact 0/1.
    with np.errstate(divide="ignore"):
        positive = np.maximum(np.log(probabilities), -100)
        negative = np.maximum(np.log1p(-probabilities), -100)
    loss = -(batch.y * positive_weight * positive + (1 - batch.y) * negative)
    return float(np.mean(subject_weights(batch.subject_ids) * loss))


def prediction_frame(batch: TrialBatch, probabilities: np.ndarray, threshold: float | None = None) -> pd.DataFrame:
    probabilities = validate_probabilities(probabilities, len(batch.X))
    frame = pd.DataFrame({"subject_id": batch.subject_ids, "trial_id": batch.trial_ids,
                          "pred_proba": probabilities})
    if batch.y is not None:
        frame["true_label"] = batch.y
    if threshold is not None:
        frame["predicted_label"] = (probabilities >= threshold).astype(np.int8)
    return frame


def evaluate_predictions(batch: TrialBatch, probabilities: np.ndarray, threshold: float) -> dict:
    frame = prediction_frame(batch, probabilities, threshold)
    batch.validate(labeled=True)
    return {
        "trial": classification_metrics(batch.y, probabilities, threshold),
        "subjects": {str(subject): classification_metrics(group.true_label.to_numpy(),
                    group.pred_proba.to_numpy(), threshold)
                     for subject, group in frame.groupby("subject_id", sort=True)},
    }


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return device


class EEGNetTrainer:
    def __init__(self, cfg: dict):
        self.cfg = dict(cfg)
        self.device = resolve_device(str(cfg.get("device", "auto")))
        for key in ("batch_size", "max_epochs", "patience", "lr_patience", "native_threads"):
            if int(cfg.get(key, 1)) < 1:
                raise ValueError(f"{key} must be positive")

    def _model(self, n_channels, n_times, model_cfg=None):
        options = self.cfg if model_cfg is None else model_cfg
        return EEGNet(n_channels=n_channels, n_times=n_times,
                      F1=int(options.get("F1", 8)), D=int(options.get("D", 2)),
                      dropout=float(options.get("dropout", 0.25)))

    def _predict_model(self, model, batch):
        batch = normalize_trials(batch)
        values = torch.from_numpy(batch.X).unsqueeze(1)
        model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(values), int(self.cfg.get("batch_size", 64))):
                predictions.append(model(values[start:start + int(self.cfg.get("batch_size", 64))].to(self.device))
                                   .squeeze(1).cpu().numpy())
        return validate_probabilities(np.concatenate(predictions), len(values))

    def fit(self, train: TrialBatch, validation: TrialBatch, out_dir: Path,
            *, selection_metric: str, seed: int) -> FitResult:
        if selection_metric not in ("bce", "auprc"):
            raise ValueError("Selection metric must be bce or auprc")
        train.validate(labeled=True)
        validation.validate(labeled=True)
        if set(train.subject_ids) & set(validation.subject_ids):
            raise ValueError("Subject leakage between fit and selection")
        if train.channel_names != validation.channel_names or train.X.shape[1:] != validation.X.shape[1:]:
            raise ValueError("Fit/selection input shapes differ")
        positive_weight = class_ratio(train)
        class_ratio(validation)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(int(self.cfg.get("native_threads", 1)))
        if self.device.type == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        model = self._model(*train.X.shape[1:]).to(self.device)
        normalized = normalize_trials(train)
        weights = subject_weights(train.subject_ids) * np.where(train.y == 1, positive_weight, 1)
        loader = DataLoader(TensorDataset(torch.from_numpy(normalized.X).unsqueeze(1),
                            torch.tensor(train.y, dtype=torch.float32),
                            torch.tensor(weights, dtype=torch.float32)),
                            batch_size=int(self.cfg.get("batch_size", 64)), shuffle=True,
                            num_workers=int(self.cfg.get("num_workers", 0)),
                            generator=torch.Generator().manual_seed(seed))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.cfg.get("lr", 0.001)),
                                     weight_decay=float(self.cfg.get("weight_decay", 0.0001)))
        minimize = selection_metric == "bce"
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min" if minimize else "max",
            patience=int(self.cfg.get("lr_patience", 10)), factor=float(self.cfg.get("lr_factor", 0.5)))
        criterion = nn.BCELoss(reduction="none")
        best_score = float("inf") if minimize else -float("inf")
        best_epoch, stalled, best_state = 0, 0, None
        history = []
        out_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, int(self.cfg.get("max_epochs", 200)) + 1):
            model.train()
            total_loss = 0.0
            for tensors, labels, batch_weights in loader:
                optimizer.zero_grad(set_to_none=True)
                predictions = model(tensors.to(self.device)).squeeze(1)
                loss = (criterion(predictions, labels.to(self.device)) * batch_weights.to(self.device)).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(labels)
            probabilities = self._predict_model(model, validation)
            bce = subject_equal_bce(validation, probabilities, positive_weight)
            auprc = float(average_precision_score(validation.y, probabilities))
            score = bce if minimize else auprc
            if not np.isfinite(score):
                raise ValueError("Non-finite selection score")
            scheduler.step(score)
            logger.info("%s epoch %d: selection BCE %.6f, AUPRC %.6f", out_dir.name, epoch, bce, auprc)
            history.append({"epoch": epoch, "train_subject_weighted_bce": total_loss / len(train.X),
                            "validation_subject_weighted_bce": bce, "validation_auprc": auprc,
                            "lr": optimizer.param_groups[0]["lr"]})
            write_csv(out_dir / "history.csv", pd.DataFrame(history))
            improved = score < best_score if minimize else score > best_score
            if improved:
                best_score, best_epoch, stalled = score, epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stalled += 1
                if stalled >= int(self.cfg.get("patience", 20)):
                    break
        if best_state is None:
            raise RuntimeError("No finite checkpoint was selected")
        model.load_state_dict(best_state)
        probabilities = self._predict_model(model, validation)
        checkpoint = out_dir / "best_model.pt"
        atomic_write(checkpoint, lambda handle: torch.save({
            "state_dict": best_state, "n_channels": train.X.shape[1], "n_times": train.X.shape[2],
            "channel_names": list(train.channel_names), "model_config": self.cfg,
            "best_epoch": best_epoch, "selection_metric": selection_metric,
            "positive_weight": positive_weight, "seed": seed,
        }, handle))
        write_csv(out_dir / "val_predictions.csv", prediction_frame(validation, probabilities))
        return FitResult(checkpoint, best_epoch, best_score, probabilities,
                         {"positive_weight": positive_weight, "selection_metric": selection_metric, "seed": seed})

    def predict(self, checkpoint: Path, batch: TrialBatch) -> np.ndarray:
        batch.validate()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if tuple(saved["channel_names"]) != batch.channel_names or (
            saved["n_channels"], saved["n_times"]
        ) != batch.X.shape[1:]:
            raise ValueError("Inference channel order or epoch shape differs from checkpoint")
        model = self._model(saved["n_channels"], saved["n_times"], saved["model_config"]).to(self.device)
        model.load_state_dict(saved["state_dict"])
        return self._predict_model(model, batch)


__all__ = ["EEGNetTrainer", "subject_equal_bce", "subject_weights", "class_ratio",
           "evaluate_predictions", "prediction_frame", "select_balanced_accuracy_threshold"]
