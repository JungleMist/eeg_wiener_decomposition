"""ERP-CORE ERN adapter; raw alone determines event and rejection selection."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..decomposition import epoch_signals
from ..types import IneligibleRecording, PreparedRecording, RecordingSource, TrialBatch

ERP_CHANNELS = (
    "FP1", "F3", "F7", "FC3", "C3", "C5", "P3", "P7", "P9", "PO7",
    "PO3", "O1", "Oz", "Pz", "CPz", "FP2", "Fz", "F4", "F8", "FC4",
    "FCz", "Cz", "C4", "C6", "P4", "P8", "P10", "PO8", "PO4", "O2",
)
STIMULI = {"11": "left", "12": "right", "21": "left", "22": "right"}
RESPONSES = {
    "111": ("left", True), "112": ("left", False),
    "121": ("left", True), "122": ("left", False),
    "211": ("right", False), "212": ("right", True),
    "221": ("right", False), "222": ("right", True),
}


def response_events(events, event_id, sfreq, max_lag_sec=1.5):
    """Match existing numeric/semantic Flankers event semantics, without requiring y."""
    descriptions = {value: key for key, value in event_id.items()}
    stimulus = None
    rows = []
    for sample, _, code in events:
        description = descriptions.get(int(code), "")
        token = description.rsplit("/", 1)[-1]
        side = STIMULI.get(token) if description.isdigit() or description.startswith("stimulus/") else None
        if description.startswith("stimulus/"):
            if "target_left" in description:
                side = "left"
            elif "target_right" in description:
                side = "right"
        if side is not None:
            stimulus = (int(sample), side)
            continue
        response = RESPONSES.get(token) if description.isdigit() or description.startswith("response/") else None
        if description in ("response/left", "response/right"):
            response = (token, None)
        if response is None or stimulus is None:
            continue
        lag = (int(sample) - stimulus[0]) / sfreq
        if 0 <= lag <= max_lag_sec:
            correct = response[1] if response[1] is not None else response[0] == stimulus[1]
            rows.append((int(sample), int(not correct)))
            stimulus = None
    if not rows:
        raise ValueError("No valid ERP-CORE stimulus-response pairs")
    return np.asarray(rows, dtype=np.int64)


class ERPCoreAdapter:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        options = cfg["auto_ecmad"]["adapter"]
        self.root = Path(options["data_dir"]).expanduser().resolve()
        self.channels = tuple(options.get("channels", ERP_CHANNELS))
        if len(self.channels) < 2 or len(set(self.channels)) != len(self.channels):
            raise ValueError("At least two unique model channels are required")

    def discover(self) -> list[RecordingSource]:
        selected = self.cfg["auto_ecmad"]["adapter"].get("subjects")
        if selected and len(set(selected)) != len(selected):
            raise ValueError("Duplicate requested subjects")
        sources = []
        for path in sorted(self.root.glob("sub-*/eeg/*_task-ERN_eeg.set")):
            subject = path.parent.parent.name
            if selected and subject not in selected:
                continue
            files = (path, path.with_suffix(".fdt")) if path.with_suffix(".fdt").exists() else (path,)
            sources.append(RecordingSource(subject, path.stem, files))
        found = {s.subject_id for s in sources}
        if selected and set(selected) - found:
            raise ValueError(f"Missing requested subjects: {sorted(set(selected) - found)}")
        if not sources:
            raise ValueError(f"No ERP-CORE ERN recordings under {self.root}")
        return sources

    def prepare(self, source: RecordingSource, *, inference=False) -> PreparedRecording:
        import mne

        raw = mne.io.read_raw_eeglab(source.files[0], preload=True, verbose=False)
        eog = {ch: "eog" for ch in raw.ch_names if "EOG" in ch.upper()}
        if eog:
            raw.set_channel_types(eog, verbose=False)
        raw.pick("eeg")
        raw.set_montage("standard_1005", match_case=False, on_missing="ignore", verbose=False)
        line = float(self.cfg["erp_core"]["line_freq"])
        if 0 < line < raw.info["sfreq"] / 2:
            raw.notch_filter(line, verbose=False)
        low, high = self.cfg["preprocessing"]["bandpass"]
        raw.filter(low, high, method="fir", verbose=False)
        sfreq = float(self.cfg["preprocessing"]["target_sfreq"])
        if not np.isclose(raw.info["sfreq"], sfreq):
            raw.resample(sfreq, verbose=False)
        # Rejection uses every EEG channel, even when model channels are subsetted.
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        try:
            rows = response_events(events, event_id, sfreq,
                                   float(self.cfg["erp_core"]["response_pairing_window_sec"]))
        except ValueError as exc:
            raise IneligibleRecording(str(exc)) from exc
        spec = self.cfg["erp_core"]["ern"]
        response_array = np.column_stack((rows[:, 0], np.zeros(len(rows), int), np.ones(len(rows), int)))
        probe = mne.Epochs(raw, response_array, event_id={"response": 1},
                           tmin=spec["tmin"], tmax=spec["tmax"], baseline=None,
                           reject={"eeg": float(self.cfg["preprocessing"]["artifact_threshold_uv"]) * 1e-6},
                           preload=True, verbose=False)
        selection = probe.selection
        labels = rows[selection, 1].astype(np.int8)
        if not len(labels) or (not inference and len(np.unique(labels)) != 2):
            raise IneligibleRecording("Raw rejection left no usable correct/incorrect ERN pair")
        missing = set(self.channels) - set(raw.ch_names)
        if missing:
            raise IneligibleRecording(f"Missing EEG channels: {sorted(missing)}")
        samples = rows[selection, 0] - raw.first_samp
        offsets = np.rint(probe.times * sfreq).astype(np.int64)
        baseline = (probe.times >= spec["baseline"][0] - 1e-10) & (probe.times <= spec["baseline"][1] + 1e-10)
        raw.pick(list(self.channels)).reorder_channels(list(self.channels))
        rejected = [{"event_index": idx, "sample": int(rows[idx, 0]), "reasons": list(reason)}
                    for idx, reason in enumerate(probe.drop_log) if reason]
        result = PreparedRecording(
            source.subject_id, source.recording_id, raw.get_data() * 1e6, sfreq, self.channels,
            samples, offsets, baseline,
            np.asarray([f"{source.subject_id}/{source.recording_id}/{int(s)}" for s in samples]),
            None if inference else labels,
            {"input_trials": len(rows), "selected_trials": len(samples), "rejected_trials": rejected},
        )
        result.validate()
        return result

    def epoch(self, recording: PreparedRecording, signals_uv: np.ndarray,
              channel_names: tuple[str, ...]) -> TrialBatch:
        return epoch_signals(recording, signals_uv, channel_names)
