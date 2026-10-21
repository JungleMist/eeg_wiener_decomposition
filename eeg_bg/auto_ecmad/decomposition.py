"""Directed pair decomposition using the established ECMAD implementation."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy

import numpy as np

from eeg_bg.preprocessing.continuous import wiener_continuous_raw
from eeg_bg.decomposition.phase_gate import validate_phase_gate_threshold
from .types import PairResult, PreparedRecording, TrialBatch


def normalize_trials(batch: TrialBatch) -> TrialBatch:
    batch.validate()
    values = np.asarray(batch.X, dtype=np.float32)
    centered = values - values.mean(axis=-1, keepdims=True)
    values = centered / np.maximum(values.std(axis=-1, keepdims=True), 1e-6)
    return TrialBatch(values, batch.subject_ids, batch.trial_ids, batch.channel_names, batch.y)


def epoch_signals(record: PreparedRecording, signals_uv: np.ndarray,
                  channel_names: tuple[str, ...]) -> TrialBatch:
    """Apply the raw selection and a linear baseline operation to any branch."""
    record.validate()
    signals_uv = np.asarray(signals_uv)
    if signals_uv.shape != (len(channel_names), record.data_uv.shape[1]):
        raise ValueError("Branch must have the same continuous time axis as raw")
    indices = record.samples[:, None] + record.offsets[None, :]
    X = signals_uv[:, indices].transpose(1, 0, 2).copy()
    X -= X[:, :, record.baseline_mask].mean(axis=-1, keepdims=True)
    batch = TrialBatch(X.astype(np.float32), np.repeat(record.subject_id, len(X)),
                       record.trial_ids.copy(), channel_names,
                       None if record.y is None else record.y.copy())
    batch.validate()
    return batch


def concatenate_batches(batches: list[TrialBatch]) -> TrialBatch:
    if not batches:
        raise ValueError("No usable recordings in partition")
    for batch in batches:
        batch.validate()
        if batch.channel_names != batches[0].channel_names or batch.X.shape[2] != batches[0].X.shape[2]:
            raise ValueError("Incompatible trial channel order or time grid")
    labeled = [batch.y is not None for batch in batches]
    if any(labeled) and not all(labeled):
        raise ValueError("Cannot mix labeled and unlabeled recordings")
    result = TrialBatch(np.concatenate([b.X for b in batches]),
                        np.concatenate([b.subject_ids for b in batches]),
                        np.concatenate([b.trial_ids for b in batches]), batches[0].channel_names,
                        np.concatenate([b.y for b in batches]) if all(labeled) else None)
    result.validate()
    return result


class ECMADPairDecomposer:
    """Only one pair enters the fusion core, so no reference fusion occurs."""

    def __init__(self, cfg: dict):
        self.cfg = deepcopy(cfg)
        if cfg["wiener"]["mode"] != "phasegated":
            raise ValueError("Auto ECMAD currently requires phasegated mode")
        validate_phase_gate_threshold(cfg["wiener"]["phase_gate_threshold_rad"])
        if int(cfg["wiener"]["nperseg"]) < 2 or not np.isfinite(cfg["preprocessing"]["epoch_length_sec"]) or cfg["preprocessing"]["epoch_length_sec"] <= 0:
            raise ValueError("nperseg must be >=2 and window duration must be finite and positive")

    def decompose(self, target: np.ndarray, reference: np.ndarray, sfreq: float) -> PairResult:
        import mne

        pair = np.asarray([target, reference], dtype=np.float64)
        if pair.ndim != 2 or pair.shape[1] < 2 or not np.isfinite(pair).all():
            raise ValueError("Pair needs finite, equal-length signals with at least two samples")
        if not np.isfinite(sfreq) or sfreq <= 0:
            raise ValueError("Sampling rate must be finite and positive")
        if np.std(pair[0]) == 0 or np.std(pair[1]) == 0:
            return PairResult(np.zeros_like(pair[0]), pair[0].copy(), {
                "windows": 0, "target_status_counts": {"constant_signal": 1},
                "phase_gate_pass_fraction_mean": 0., "reconstruction_max_error_uv": 0.,
            })
        cfg = deepcopy(self.cfg)
        segment = min(int(cfg["wiener"]["nperseg"]), pair.shape[1],
                      max(2, int(round(cfg["preprocessing"]["epoch_length_sec"] * sfreq))))
        frequencies = np.fft.rfftfreq(segment, 1 / sfreq)
        low, high = cfg["wiener"]["freq_band"]
        if not np.any((frequencies >= low) & (frequencies <= high)):
            return PairResult(np.zeros_like(pair[0]), pair[0].copy(), {
                "windows": 0, "target_status_counts": {"short_recording_no_band_bins": 1},
                "phase_gate_pass_fraction_mean": 0., "reconstruction_max_error_uv": 0.,
            })
        cfg["channels"]["channel_groups"] = [["target", "reference"]]
        raw = mne.io.RawArray(pair * 1e-6, mne.create_info(
            ["target", "reference"], sfreq, ch_types="eeg"), verbose=False)
        specific_raw, diagnostics = wiener_continuous_raw(raw, cfg)
        specific = specific_raw.get_data()[0] * 1e6
        coherent = pair[0] - specific
        if not np.isfinite(specific).all() or not np.isfinite(coherent).all():
            raise ValueError("Non-finite ECMAD components")
        statuses = Counter()
        phase_fractions = []
        for window in diagnostics["window_diagnostics"]:
            for idx, key in enumerate(window["candidate_keys"]):
                if key.endswith("::target"):
                    statuses[str(window["candidate_status"][idx])] += 1
                    phase_fractions.append(window["phase_gate_pass_fraction"][idx])
        return PairResult(coherent, specific, {
            "windows": diagnostics["windows"], "target_status_counts": dict(statuses),
            "phase_gate_pass_fraction_mean": float(np.mean(phase_fractions)),
            "reconstruction_max_error_uv": float(np.max(np.abs(coherent + specific - pair[0]))),
        })
