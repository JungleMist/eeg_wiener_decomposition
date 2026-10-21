"""Behavioral isolation, recovery, numerical and backend-extension checks."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from eeg_bg.auto_ecmad import AutoECMADPipeline
from eeg_bg.auto_ecmad.adapters.erp_core import ERPCoreAdapter, response_events
from eeg_bg.auto_ecmad.artifacts import load_recording, read_json, save_recording, write_json
from eeg_bg.auto_ecmad.decomposition import ECMADPairDecomposer, epoch_signals
from eeg_bg.auto_ecmad.pipeline import split_subjects
from eeg_bg.auto_ecmad.search import select_reference
from eeg_bg.auto_ecmad.training import (EEGNetTrainer, class_ratio, subject_equal_bce,
                                        subject_weights)
from eeg_bg.auto_ecmad.types import FitResult, PairResult, PreparedRecording, RecordingSource, TrialBatch
from eeg_bg.config.settings import load_config
from eeg_bg.ml.cnn_model import EEGNet


class SyntheticAdapter:
    def __init__(self, root, n_subjects=5):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.sources = []
        for idx in range(n_subjects):
            path = self.root / f"subject{idx}.txt"
            if not path.exists():
                path.write_text(str(idx))
            self.sources.append(RecordingSource(f"s{idx}", f"r{idx}", (path,)))

    def discover(self):
        return self.sources

    def prepare(self, source, *, inference=False):
        idx = int(source.subject_id[1:])
        data = np.random.default_rng(idx).normal(size=(3, 1600))
        samples = np.array([256, 512, 768, 1024])
        return PreparedRecording(source.subject_id, source.recording_id, data, 125.,
                                 ("A", "B", "C"), samples, np.arange(-32, 96),
                                 np.arange(128) < 32,
                                 np.array([f"{source.subject_id}/{s}" for s in samples]),
                                 None if inference else np.array([0, 0, 1, 1], np.int8))

    def epoch(self, record, signals, names):
        return epoch_signals(record, signals, names)


class SimpleDecomposer:
    def decompose(self, target, reference, sfreq):
        coherent = .2 * reference
        return PairResult(coherent, target - coherent, {"test_backend": True})


class FakeTrainer:
    """An alternate backend: no Torch state or ERP format dependencies."""
    def __init__(self):
        self.calls = []

    def fit(self, train, validation, out_dir, *, selection_metric, seed):
        self.calls.append((train, validation, selection_metric, seed))
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = out_dir / "model.json"
        write_json(checkpoint, {"channels": train.channel_names, "times": train.X.shape[-1]})
        p = np.where(validation.y == 1, .8, .2)
        score = subject_equal_bce(validation, p, class_ratio(train)) if selection_metric == "bce" else 1.
        return FitResult(checkpoint, 1, score, p)

    def predict(self, checkpoint, batch):
        saved = read_json(checkpoint)
        assert saved["channels"] == list(batch.channel_names)
        return np.full(len(batch.X), .6)


@pytest.fixture
def cfg(tmp_path):
    config = load_config("configs/erp_core_auto_ecmad_eegnet.yaml")
    config["paths"]["results_dir"] = str(tmp_path / "run")
    config["auto_ecmad"]["training"].update(device="cpu", F1=2, D=1, dropout=.1,
                                              batch_size=4, max_epochs=2, patience=1)
    config["preprocessing"].update(target_sfreq=125, epoch_length_sec=4)
    config["wiener"].update(nperseg=125)
    return config


def pipeline(cfg, tmp_path, trainer=None, decomposer=None):
    return AutoECMADPipeline(cfg, adapter=SyntheticAdapter(tmp_path / "sources"),
                             decomposer=decomposer or SimpleDecomposer(), trainer=trainer or FakeTrainer())


def test_loss_is_subject_equal_and_uses_fit_class_counts():
    batch = TrialBatch(np.ones((4, 2, 128), np.float32), np.array(["a", "b", "b", "b"]),
                       np.array(["0", "1", "2", "3"]), ("coherent", "specific"), np.array([1, 0, 0, 0]))
    assert class_ratio(batch) == 3
    np.testing.assert_allclose(subject_weights(batch.subject_ids), [2, 2/3, 2/3, 2/3])
    p = np.array([.8, .1, .2, .3])
    expected = (-3 * np.log(.8) + np.mean(-np.log(1-p[1:]))) / 2
    assert subject_equal_bce(batch, p, 3) == pytest.approx(expected)
    assert np.isfinite(subject_equal_bce(batch, np.array([0., 1., 0., 0.]), 3))
    with pytest.raises(ValueError, match="Invalid model probabilities"):
        subject_equal_bce(batch, np.full(4, np.nan), 3)


def test_subject_split_and_tie_rule():
    partitions = split_subjects([f"s{i}" for i in range(39)], {}, 42)
    assert {k: len(v) for k, v in partitions.items()} == {
        "inner_fit": 19, "inner_selection": 5, "validation": 7, "test": 8}
    assert partitions == split_subjects([f"s{i}" for i in range(39)], {}, 42)
    rows = [{"reference": "B", "selection_loss": .2 + 5e-7},
            {"reference": "C", "selection_loss": .2}]
    assert select_reference(rows, ("A", "B", "C"))["reference"] == "B"
    with pytest.raises(ValueError):
        split_subjects(["a", "b", "c", "d"], {}, 42)


def test_record_cache_roundtrip_and_epoch_linearity(cfg, tmp_path):
    adapter = SyntheticAdapter(tmp_path / "sources")
    record = adapter.prepare(adapter.discover()[0])
    path = tmp_path / "cache.npz"
    save_recording(path, record)
    restored = load_recording(path)
    np.testing.assert_array_equal(record.data_uv, restored.data_uv)
    result = SimpleDecomposer().decompose(record.data_uv[0], record.data_uv[1], record.sfreq)
    pair = epoch_signals(record, np.stack([result.coherent, result.specific]), ("coherent", "specific"))
    raw = epoch_signals(record, record.data_uv[:1], ("A",))
    np.testing.assert_allclose(pair.X.sum(axis=1), raw.X[:, 0], atol=1e-6)


def test_two_channel_search_isolation_and_final_order(cfg, tmp_path, monkeypatch):
    import eeg_bg.auto_ecmad.pipeline as pipeline_module

    trainer = FakeTrainer()
    experiment = pipeline(cfg, tmp_path, trainer)
    state = experiment.prepare()
    accessed = []
    original = pipeline_module.load_recording

    def observe(path):
        record = original(path)
        accessed.append(record.subject_id)
        return record

    monkeypatch.setattr(pipeline_module, "load_recording", observe)
    mapping = experiment.search()
    assert len(trainer.calls) == 6
    allowed = set(state["partitions"]["inner_fit"] + state["partitions"]["inner_selection"])
    assert set(accessed) <= allowed
    for train, validation, metric, seed in trainer.calls:
        assert train.X.shape[1] == validation.X.shape[1] == 2
        assert train.channel_names == ("coherent", "specific")
        assert not set(train.subject_ids) & set(validation.subject_ids)
        assert metric == "bce" and seed == 42
    assert mapping["references"] == {"A": "B", "B": "A", "C": "A"}
    accessed.clear()
    experiment.fit()
    assert not set(accessed) & set(state["partitions"]["test"])
    assert len(trainer.calls) == 8
    selected_train = trainer.calls[6][0]
    assert selected_train.X.shape[1] == 6
    assert selected_train.channel_names == ("A::coherent", "A::specific", "B::coherent",
                                            "B::specific", "C::coherent", "C::specific")
    assert set(selected_train.subject_ids) == allowed
    experiment.evaluate()
    assert set(state["partitions"]["test"]) <= set(accessed)
    assert (experiment.output_dir / "summary.json").is_file()


def test_completed_resume_and_corruption_detection(cfg, tmp_path):
    trainer = FakeTrainer()
    experiment = pipeline(cfg, tmp_path, trainer)
    experiment.run()
    resumed_trainer = FakeTrainer()
    resumed = pipeline(cfg, tmp_path, resumed_trainer)
    resumed.run()
    assert not resumed_trainer.calls
    checkpoint = resumed.output_dir / "candidates/t000_r001/model.json"
    checkpoint.write_text("corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        resumed.search()


def test_interrupted_candidate_restarts_without_retraining_completed(cfg, tmp_path):
    class InterruptingTrainer(FakeTrainer):
        def fit(self, *args, **kwargs):
            if len(self.calls) == 2:
                raise RuntimeError("interrupted")
            return super().fit(*args, **kwargs)

    trainer = InterruptingTrainer()
    experiment = pipeline(cfg, tmp_path, trainer)
    experiment.prepare()
    with pytest.raises(RuntimeError, match="interrupted"):
        experiment.search()
    assert not (experiment.output_dir / "selected_mapping.json").exists()
    trainer.calls.clear()  # Same backend class and settings, retry the interrupted search.
    trainer.fit = lambda *a, **kw: FakeTrainer.fit(trainer, *a, **kw)
    experiment.search()
    assert len(trainer.calls) == 4


def test_config_source_and_cache_invalidation(cfg, tmp_path):
    experiment = pipeline(cfg, tmp_path)
    state = experiment.prepare()
    changed = deepcopy(cfg)
    changed["wiener"]["phase_gate_threshold_rad"] = .5
    with pytest.raises(ValueError, match="changed"):
        pipeline(changed, tmp_path).search()
    first = experiment.output_dir / state["records"][0]["cache"]
    first.write_bytes(b"corrupt cache")
    with pytest.raises(ValueError, match="cache changed"):
        experiment._batch([state["records"][0]["subject_id"]])
    experiment.adapter.sources[0].files[0].write_text("changed source")
    with pytest.raises(ValueError, match="Source changed"):
        pipeline(cfg, tmp_path).search()


def test_frozen_bundle_reloads_without_labels(cfg, tmp_path):
    experiment = pipeline(cfg, tmp_path)
    experiment.prepare()
    experiment.search()
    with pytest.raises(FileNotFoundError):
        experiment.evaluate()
    experiment.fit()
    adapter = SyntheticAdapter(tmp_path / "new_sources")
    inference = AutoECMADPipeline.from_bundle(experiment.output_dir / "bundle", adapter=adapter,
                                              decomposer=SimpleDecomposer(), trainer=FakeTrainer(),
                                              output_dir=tmp_path / "predictions")
    records = [adapter.prepare(source, inference=True) for source in adapter.discover()]
    frame = inference.predict(records)
    assert len(frame) == 20 and "true_label" not in frame
    assert set(frame.pred_proba) == {.6}
    experiment.cfg["wiener"]["phase_gate_threshold_rad"] = .5
    with pytest.raises(ValueError, match="preprocessing differs"):
        experiment.predict(records)
    (experiment.output_dir / "bundle/model.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="bundle"):
        AutoECMADPipeline.from_bundle(experiment.output_dir / "bundle")


def test_changed_frozen_threshold_fails_before_loading_test(cfg, tmp_path, monkeypatch):
    experiment = pipeline(cfg, tmp_path)
    experiment.prepare()
    experiment.search()
    experiment.fit()
    path = experiment.output_dir / "models/selected/frozen_model.json"
    frozen = read_json(path)
    frozen["threshold"] += .01
    write_json(path, frozen)
    monkeypatch.setattr(experiment, "_batch", lambda *a, **kw: pytest.fail("Held-out EEG must not be loaded"))
    with pytest.raises(ValueError, match="Frozen model"):
        experiment.evaluate()


def test_adapter_cannot_change_raw_trial_selection(cfg, tmp_path):
    class InvalidAdapter(SyntheticAdapter):
        def epoch(self, *args):
            batch = super().epoch(*args)
            return TrialBatch(batch.X[1:], batch.subject_ids[1:], batch.trial_ids[1:],
                               batch.channel_names, batch.y[1:])

    experiment = AutoECMADPipeline(cfg, adapter=InvalidAdapter(tmp_path / "sources"),
                                   decomposer=SimpleDecomposer(), trainer=FakeTrainer())
    experiment.prepare()
    with pytest.raises(ValueError, match="shared trial selection"):
        experiment.search()


@pytest.mark.parametrize("channels", [2, 60])
def test_eegnet_forward_backward(channels):
    torch.set_num_threads(1)
    model = EEGNet(n_channels=channels, n_times=128, F1=2, D=1)
    loss = torch.nn.BCELoss()(model(torch.randn(2, 1, channels, 128)).squeeze(1), torch.tensor([0., 1.]))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_actual_eegnet_initialization_checkpoint_and_full_synthetic_flow(cfg, tmp_path):
    trainer = EEGNetTrainer(cfg["auto_ecmad"]["training"])
    experiment = pipeline(cfg, tmp_path, trainer, ECMADPairDecomposer(cfg))
    experiment.run()
    state = experiment._load_state()
    train, _ = experiment._batch(state["partitions"]["inner_fit"], pair=("A", "B"))
    validation, _ = experiment._batch(state["partitions"]["inner_selection"], pair=("A", "B"))
    a = trainer.fit(train, validation, tmp_path / "a", selection_metric="bce", seed=42)
    b = trainer.fit(train, validation, tmp_path / "b", selection_metric="bce", seed=42)
    np.testing.assert_array_equal(a.validation_probabilities, b.validation_probabilities)
    history = pd.read_csv(tmp_path / "a/history.csv")
    assert a.best_epoch == int(history.loc[history.validation_subject_weighted_bce.idxmin(), "epoch"])
    np.testing.assert_allclose(trainer.predict(a.checkpoint, validation), a.validation_probabilities)
    for candidate in (experiment.output_dir / "candidates").iterdir():
        assert torch.load(candidate / "best_model.pt", weights_only=True)["n_channels"] == 2
    assert torch.load(experiment.output_dir / "models/selected/best_model.pt", weights_only=True)["n_channels"] == 6
    inference = AutoECMADPipeline.from_bundle(experiment.output_dir / "bundle", adapter=experiment.adapter)
    records = [experiment.adapter.prepare(source, inference=True) for source in experiment.adapter.discover()]
    assert len(inference.predict(records)) == 20


@pytest.mark.parametrize("length", [2, 60, 1000])
def test_decomposition_reconstruction_and_zero_signal(cfg, length):
    decomposer = ECMADPairDecomposer(cfg)
    rng = np.random.default_rng(4)
    raw = rng.normal(size=length)
    reference = .5 * raw + rng.normal(size=length)
    result = decomposer.decompose(raw, reference, 125)
    np.testing.assert_allclose(result.coherent + result.specific, raw, atol=1e-12)
    zero = decomposer.decompose(np.zeros(length), np.zeros(length), 125)
    np.testing.assert_array_equal(zero.coherent, 0)
    np.testing.assert_array_equal(zero.specific, 0)
    with pytest.raises(ValueError, match="finite"):
        decomposer.decompose(np.full(length, np.nan), reference, 125)


def test_phase_gate_and_numerical_protection(cfg):
    rng = np.random.default_rng(10)
    reference = rng.normal(size=4000)
    permissive = deepcopy(cfg)
    permissive["wiener"]["phase_gate_threshold_rad"] = np.pi
    gated = ECMADPairDecomposer(cfg).decompose(-reference, reference, 125)
    accepted = ECMADPairDecomposer(permissive).decompose(-reference, reference, 125)
    assert np.std(gated.coherent) < .01 * np.std(accepted.coherent)
    protected = deepcopy(cfg)
    protected["wiener"]["filter_magnitude_threshold"] = .001
    fallback = ECMADPairDecomposer(protected).decompose(reference, reference, 125)
    np.testing.assert_allclose(fallback.coherent, 0, atol=1e-12)
    assert set(fallback.diagnostics["target_status_counts"]) != {"0"}
    reverse = ECMADPairDecomposer(cfg).decompose(reference, reference * 2, 125)
    forward = ECMADPairDecomposer(cfg).decompose(reference * 2, reference, 125)
    assert not np.allclose(reverse.coherent, forward.coherent)


def test_numeric_and_semantic_erp_events_do_not_require_both_labels():
    events = np.array([[10, 0, 11], [30, 0, 111], [50, 0, 12], [70, 0, 212]])
    rows = response_events(events, {str(c): c for c in (11, 111, 12, 212)}, 100)
    assert rows.tolist() == [[30, 0], [70, 0]]
    semantic = response_events(np.array([[10, 0, 1], [30, 0, 2]]),
                               {"stimulus/compatible/target_left": 1, "response/right": 2}, 100)
    assert semantic.tolist() == [[30, 1]]


def test_raw_erp_rejection_mask_and_unlabeled_adapter(cfg, tmp_path, monkeypatch):
    import mne

    adapter = ERPCoreAdapter(cfg)
    adapter.channels = ("FP1", "F3", "C3")
    sfreq = 250
    values = np.random.default_rng(1).normal(scale=1e-6, size=(3, 5000))
    raw = mne.io.RawArray(values, mne.create_info(list(adapter.channels), sfreq, "eeg"), verbose=False)
    raw.set_annotations(mne.Annotations([2., 2.3, 5., 5.3, 8., 8.3, 11., 11.3], [0.] * 8,
                                       ["11", "111", "11", "112", "11", "111", "11", "112"]))
    raw._data[:, 1200:1400] += .001  # Reject the second response from raw.
    monkeypatch.setattr(mne.io, "read_raw_eeglab", lambda *a, **kw: raw.copy())
    source = RecordingSource("sub-test", "record", (tmp_path / "record.set",))
    prepared = adapter.prepare(source)
    unlabeled = adapter.prepare(source, inference=True)
    assert prepared.diagnostics["rejected_trials"]
    np.testing.assert_array_equal(prepared.trial_ids, unlabeled.trial_ids)
    assert unlabeled.y is None
