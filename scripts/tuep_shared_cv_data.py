"""Recording-level preparation for shared patient-level EEG cross-validation."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from importlib import import_module
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from threadpoolctl import threadpool_limits

from eeg_bg.features.profiles import PROFILES
from eeg_bg.io.dataset import get_recording_intervals
from eeg_bg.io.edf_reader import load_edf
from eeg_bg.preprocessing.continuous import wiener_continuous_raw


continuous = import_module("21_cache_tuep_continuous_wiener")
epochs_io = import_module("22_extract_tuep_component_epochs")
PHASES = {
    "0.01": "tuep_phase001.yaml", "0.05": "tuep_phase005.yaml",
    "0.1": "tuep_phase01.yaml", "0.5": "tuep_phase05.yaml", "1": "tuep_phase1.yaml",
}
COMPONENTS = ("specific", "coherent")
COMBO = "specific_coherent"
PROFILE = PROFILES["base211_conn80"]
WINDOW_KEY = ["evaluation_id", "record_cache_id", "epoch_start_sample"]
SEED = 42
N_FOLDS = 10


def prepare_record(job):
    row, cfg, stage, frozen, root = job
    path = root / f"{row['record_cache_id']}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            return data["epoch_start_samples"], data["epoch_segment_indices"]

    # Preprocess and cut one recording at a time to keep continuous waveforms off disk.
    source, channels, sfreq = load_edf(Path(row["edf_path"]), cfg)
    size = int(sfreq * cfg["preprocessing"]["epoch_length_sec"])
    intervals = get_recording_intervals(row, cfg, source.shape[1] / sfreq)
    segments = continuous._eligible_sample_segments(intervals, sfreq, source.shape[1], size)
    threshold = float(cfg["preprocessing"]["artifact_threshold_uv"])
    starts, segment_ids, raw = [], [], []
    specific, coherent = [], []
    for segment, (first, last) in enumerate(segments):
        candidates = np.arange(first, last - size + 1, size)
        if stage == "raw":
            for i in candidates:
                if not np.isfinite(source[:, i:i + size]).all():
                    raise ValueError(f"Nonfinite raw EEG in {row['edf_path']}")
            chosen = [i for i in candidates if np.max(np.abs(source[:, i:i + size])) <= threshold]
        else:
            expected = frozen.get(segment, ())
            chosen = candidates[np.isin(candidates, expected)].tolist()
            if chosen != list(expected):
                raise ValueError(f"Frozen epoch is outside its background segment: {row['edf_path']}")
        if not chosen:
            continue
        starts.extend(chosen)
        segment_ids.extend([segment] * len(chosen))
        if stage == "raw":
            raw.extend(source[:, i:i + size] for i in chosen)
            continue
        import mne

        signal = source[:, first:last]
        info = mne.create_info(channels, sfreq, "eeg")
        recorded = mne.io.RawArray(signal * 1e-6, info, verbose=False)
        result, diagnostics = wiener_continuous_raw(
            recorded, cfg, subject_id=f"{row['record_cache_id']}:{segment}",
        )
        local = result.get_data() * 1e6
        active = continuous._active_channels(diagnostics)
        for index, name in enumerate(channels):
            if name not in active:
                local[index] = signal[index]
        local = local.astype(np.float32)
        residual = signal.astype(np.float32) - local
        specific.extend(local[:, i-first:i-first+size] for i in chosen)
        coherent.extend(residual[:, i-first:i-first+size] for i in chosen)

    starts = np.asarray(starts, dtype=np.int64)
    segment_ids = np.asarray(segment_ids, dtype=np.int64)
    if stage != "raw" and len(starts) != sum(map(len, frozen.values())):
        raise ValueError(f"Frozen windows missing from {row['edf_path']}")
    banks = {"raw": raw} if stage == "raw" else dict(zip(COMPONENTS, (specific, coherent)))
    standard = cfg["channels"]["standard_19"]
    blocks = {}
    for name, bank in banks.items():
        array = np.asarray(bank, dtype=np.float32).reshape(len(starts), len(channels), size)
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite {name} EEG in {row['edf_path']}")
        blocks[name] = epochs_io._expand_channels(array, channels, standard)
    continuous._atomic_npz(
        path, **blocks, epoch_start_samples=starts, epoch_start_sec=starts / sfreq,
        epoch_segment_indices=segment_ids,
    )
    return starts, segment_ids


def extract_features(job):
    record_id, cfg, names, waveform_root, feature_root = job
    output = feature_root / f"{record_id}.npz"
    if output.is_file():
        return output
    channels = cfg["channels"]["standard_19"]
    sfreq = cfg["preprocessing"]["target_sfreq"]
    wiener = cfg["wiener"]
    conn = cfg["ml"]["features"]["connectivity"]["nperseg"]
    values = {}
    with np.load(waveform_root / output.name, allow_pickle=False) as data:
        for name in names:
            values[name] = np.asarray([
                PROFILE.extract_fn(epoch, channels, sfreq, wiener["nperseg"],
                                   tuple(wiener["freq_band"]), conn)
                for epoch in data[name]
            ], dtype=np.float32)
            if values[name].shape != (len(data["epoch_start_samples"]), PROFILE.dim):
                raise ValueError(f"Unexpected feature shape for {record_id}/{name}")
            if not np.isfinite(values[name]).all():
                raise ValueError(f"Nonfinite features for {record_id}/{name}")
        continuous._atomic_npz(output, **values, epoch_start_samples=data["epoch_start_samples"])
    return output


def run_record_jobs(function, jobs, workers, label):
    print(f"{label}: {len(jobs)} recordings", flush=True)
    with threadpool_limits(limits=1):
        if workers == 1:
            results = map(function, jobs)
            return collect_jobs(results, len(jobs), label)
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("spawn"),
            initializer=continuous._init_worker,
        ) as pool:
            return collect_jobs(pool.map(function, jobs), len(jobs), label)


def collect_jobs(results, total, label):
    collected = []
    for index, row in enumerate(results, 1):
        collected.append(row)
        if index % 50 == 0 or index == total:
            print(f"{label}: {index}/{total}", flush=True)
    return collected


def prepare_epochs(records, cfg, stage, root, windows, workers):
    root.mkdir(parents=True, exist_ok=True)
    frozen = {}
    if windows is not None:
        for record_id, group in windows.groupby("record_cache_id", sort=False):
            frozen[record_id] = group.groupby("segment_index")["epoch_start_sample"].apply(list).to_dict()
        records = [row for row in records if row["record_cache_id"] in frozen]
    jobs = [(row, cfg, stage, frozen.get(row["record_cache_id"]), root) for row in records]
    results = run_record_jobs(prepare_record, jobs, workers, "Preprocessing / epoch generation")
    frames = []
    for row, (starts, segments) in zip(records, results, strict=True):
        if len(starts):
            frames.append(pd.DataFrame({
                **{key: row[key] for key in ("evaluation_id", "patient_id", "recording_id",
                                            "record_cache_id", "class_name", "label")},
                "epoch_start_sample": starts, "segment_index": segments,
            }))
    if not frames:
        raise ValueError("No eligible EEG epochs")
    current = pd.concat(frames, ignore_index=True)
    if current.duplicated(WINDOW_KEY).any():
        raise ValueError("Duplicate window identity")
    if windows is not None:
        pd.testing.assert_frame_equal(current, windows, check_dtype=False)
    return current


def split_patients(windows, expected_patients, *, dataset="tuep"):
    patients = windows[["evaluation_id", "patient_id", "label"]].drop_duplicates()
    patients = patients.sort_values("evaluation_id").reset_index(drop=True)
    if not patients["patient_id"].is_unique:
        raise ValueError("Each patient must have one evaluation ID and one class label")
    counts = patients["label"].value_counts().to_dict()
    if (expected_patients is not None and len(patients) != expected_patients) or (
        dataset == "tuep" and expected_patients == 178 and counts != {0: 96, 1: 82}
    ):
        raise ValueError(f"Expected {expected_patients} patients; found {len(patients)}: {counts}")
    if set(counts) != {0, 1} or min(counts.values()) < N_FOLDS:
        raise ValueError("Each class needs at least ten eligible patients")
    outer = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (dev, held) in enumerate(outer.split(patients, patients["label"]), 1):
        # Round up and leave room for both classes even during a small pilot.
        size = max(2, int(np.ceil(0.1 * len(dev))))
        inner = StratifiedShuffleSplit(n_splits=1, test_size=size, random_state=SEED + fold)
        _, validation = next(inner.split(dev, patients.iloc[dev]["label"]))
        patients.loc[held, "outer_fold"] = fold
        patients[f"eegnet_val_{fold}"] = False
        patients.loc[dev[validation], f"eegnet_val_{fold}"] = True
        labels = patients["label"]
        if (labels.iloc[dev[validation]].nunique() != 2 or
            labels.iloc[np.setdiff1d(dev, dev[validation])].nunique() != 2):
            raise ValueError(f"EEGNet fold {fold} requires both classes in training and validation")
    patients["outer_fold"] = patients["outer_fold"].astype(int)
    return patients


def feature_matrix(windows, feature_root, condition):
    names = COMPONENTS if condition == COMBO else (condition,)
    blocks = []
    for record_id, frame in windows.groupby("record_cache_id", sort=False):
        with np.load(feature_root / f"{record_id}.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(data["epoch_start_samples"], frame["epoch_start_sample"])
            block = np.concatenate([data[name] for name in names], axis=1)
            if block.shape != (len(frame), PROFILE.dim * len(names)) or not np.isfinite(block).all():
                raise ValueError(f"Invalid {condition} feature cache for {record_id}")
            blocks.append(block)
    return np.concatenate(blocks)
