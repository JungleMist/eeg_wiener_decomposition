"""Array contracts and replaceable backends for automated ECMAD experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class IneligibleRecording(ValueError):
    """Expected raw eligibility exclusion, distinct from invalid numerical data."""


@dataclass(frozen=True)
class RecordingSource:
    subject_id: str
    recording_id: str
    files: tuple[Path, ...]


@dataclass
class PreparedRecording:
    """Continuous microvolt data and a fixed, raw-derived trial selection.

    samples are zero-based response locations in data_uv. offsets and
    baseline_mask describe the common epoch grid, including both endpoints.
    Labels are optional for inference; no denoised branch may change samples.
    """

    subject_id: str
    recording_id: str
    data_uv: np.ndarray
    sfreq: float
    channel_names: tuple[str, ...]
    samples: np.ndarray
    offsets: np.ndarray
    baseline_mask: np.ndarray
    trial_ids: np.ndarray
    y: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.data_uv.ndim != 2 or self.data_uv.shape[0] != len(self.channel_names):
            raise ValueError("Continuous data must be (channels, samples)")
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("Channel names must be unique")
        if not np.isfinite(self.sfreq) or self.sfreq <= 0 or not np.isfinite(self.data_uv).all():
            raise ValueError("Non-finite EEG data or invalid sampling rate")
        if self.offsets.ndim != 1 or not len(self.offsets) or not np.issubdtype(self.offsets.dtype, np.integer):
            raise ValueError("Epoch offsets must be a nonempty vector")
        if self.baseline_mask.dtype != bool or self.baseline_mask.shape != self.offsets.shape or not self.baseline_mask.any():
            raise ValueError("Baseline mask must select samples on the epoch grid")
        if self.samples.ndim != 1 or not np.issubdtype(self.samples.dtype, np.integer) or not len(self.samples) or len(self.samples) != len(self.trial_ids):
            raise ValueError("A recording must have aligned, nonempty trials")
        if len(np.unique(self.trial_ids)) != len(self.trial_ids):
            raise ValueError("Trial IDs must be unique")
        if np.any(self.samples + self.offsets.min() < 0) or np.any(
            self.samples + self.offsets.max() >= self.data_uv.shape[1]
        ):
            raise ValueError("Selected epochs exceed recording bounds")
        if self.y is not None and (self.y.shape != self.samples.shape or not np.isin(self.y, [0, 1]).all()):
            raise ValueError("Labels must be aligned binary values")


@dataclass
class TrialBatch:
    X: np.ndarray
    subject_ids: np.ndarray
    trial_ids: np.ndarray
    channel_names: tuple[str, ...]
    y: np.ndarray | None = None

    def validate(self, labeled: bool = False) -> None:
        if self.X.ndim != 3 or self.X.shape[1] != len(self.channel_names):
            raise ValueError("Trials must be (trials, channels, times)")
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("Trial channel names must be unique")
        if not len(self.X) or not np.isfinite(self.X).all():
            raise ValueError("Empty or non-finite trial data")
        if len(self.subject_ids) != len(self.X) or len(self.trial_ids) != len(self.X):
            raise ValueError("Trial metadata is not aligned")
        if len(np.unique(self.trial_ids)) != len(self.trial_ids):
            raise ValueError("Duplicate trial IDs")
        if labeled and self.y is None:
            raise ValueError("Training/evaluation requires labels")
        if self.y is not None and (self.y.shape != (len(self.X),) or not np.isin(self.y, [0, 1]).all()):
            raise ValueError("Labels must be aligned binary values")


@dataclass
class PairResult:
    coherent: np.ndarray
    specific: np.ndarray
    diagnostics: dict[str, Any]


@dataclass
class FitResult:
    checkpoint: Path
    best_epoch: int
    selection_score: float
    validation_probabilities: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetAdapter(Protocol):
    def discover(self) -> list[RecordingSource]: ...
    def prepare(self, source: RecordingSource, *, inference: bool = False) -> PreparedRecording: ...
    def epoch(self, recording: PreparedRecording, signals_uv: np.ndarray,
              channel_names: tuple[str, ...]) -> TrialBatch: ...


class PairDecomposer(Protocol):
    def decompose(self, target: np.ndarray, reference: np.ndarray, sfreq: float) -> PairResult: ...


class ModelTrainer(Protocol):
    def fit(self, train: TrialBatch, validation: TrialBatch, out_dir: Path,
            *, selection_metric: str, seed: int) -> FitResult: ...
    def predict(self, checkpoint: Path, batch: TrialBatch) -> np.ndarray: ...
