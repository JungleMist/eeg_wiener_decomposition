"""Validate the deferred analysis and training-to-analysis supervision chain."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eeg_bg.auto_ecmad import AutoECMADPipeline
from eeg_bg.auto_ecmad.artifacts import read_json, write_json
from eeg_bg.auto_ecmad.training import EEGNetTrainer
from eeg_bg.config.settings import load_config
from test_pipeline import SyntheticAdapter, SimpleDecomposer


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def analyzer():
    return load_script("auto_ecmad_analyzer", "scripts/analyze_auto_ecmad_results.py")


def test_paired_bootstrap_preserves_pairing_and_subject_units(analyzer):
    raw = pd.DataFrame({"subject_id": np.repeat(["a", "b", "c"], 2),
                        "true_label": [0, 1] * 3, "pred_proba": [.9, .1] * 3})
    selected = raw.copy()
    selected["pred_proba"] = 1 - raw.pred_proba
    first = analyzer.paired_intervals(raw, selected, {"raw": .5, "selected": .5}, 20, 42)
    second = analyzer.paired_intervals(raw, selected, {"raw": .5, "selected": .5}, 20, 42)
    pd.testing.assert_frame_equal(first, second)
    auroc = first.set_index("metric").loc["auroc"]
    assert auroc["delta"] == auroc["lower_95"] == auroc["upper_95"] == 1
    assert auroc["test_subjects"] == 3
    assert analyzer.paired_intervals(raw.iloc[:2], selected.iloc[:2], {"raw": .5, "selected": .5}, 20, 42).empty


def test_analysis_reads_completed_artifacts_and_rejects_incomplete_or_inconsistent_run(analyzer, tmp_path):
    cfg = load_config("configs/erp_core_auto_ecmad_eegnet.yaml")
    cfg["paths"]["results_dir"] = str(tmp_path / "run")
    cfg["auto_ecmad"]["random_state"] = 7
    cfg["auto_ecmad"]["training"].update(device="cpu", max_epochs=1, batch_size=4, F1=2, D=1)
    adapter = SyntheticAdapter(tmp_path / "sources")
    run = AutoECMADPipeline(cfg, adapter=adapter, decomposer=SimpleDecomposer(),
                            trainer=EEGNetTrainer(cfg["auto_ecmad"]["training"]))
    run.run()
    evidence = analyzer.analyze(run.output_dir, tmp_path / "analysis", repetitions=10)
    assert evidence["candidate_count"] == 6
    assert evidence["eligible_subjects"] == 5
    assert evidence["parameter_counts"]["selected"] > evidence["parameter_counts"]["raw"]
    document = (tmp_path / "analysis/report.html").read_text()
    assert "data:image/png;base64," in document and "不确定性" in document
    assert "种子 7" in document
    assert "测试被试少于两名" in (tmp_path / "analysis/report.md").read_text()
    assert len(list((tmp_path / "analysis/figures").glob("*.pdf"))) == 6
    score_path = run.output_dir / "candidate_scores.csv"
    original = score_path.read_bytes()
    pd.read_csv(score_path).iloc[1:].to_csv(score_path, index=False)
    with pytest.raises(ValueError, match="incomplete"):
        analyzer.audit(run.output_dir)
    score_path.write_bytes(original)
    summary_path = run.output_dir / "summary.json"
    summary = read_json(summary_path)
    summary["results"]["selected"]["test"]["trial"]["auprc"] += .01
    write_json(summary_path, summary)
    with pytest.raises(ValueError, match="Summary metrics disagree"):
        analyzer.audit(run.output_dir)


@pytest.mark.parametrize("return_codes,expected_phases,expected_state", [
    ([0, 0], ["training", "analysis"], "completed"),
    ([1], ["training"], "failed"),
    ([0, 2], ["training", "analysis"], "failed"),
])
def test_supervisor_only_analyzes_successful_training(tmp_path, monkeypatch, return_codes, expected_phases, expected_state):
    module = load_script("auto_ecmad_job", "scripts/run_auto_ecmad_training_job.py")
    calls = []

    class Process:
        def __init__(self, rc):
            self.pid = 123
            self.rc = rc

        def wait(self):
            return self.rc

    def popen(command, **kwargs):
        phase = "analysis" if "analyze_auto_ecmad_results.py" in command[2] else "training"
        calls.append(phase)
        return Process(return_codes[len(calls) - 1])

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(module.sys, "argv", ["job", "--output-dir", str(tmp_path / "job"), "--device", "mps"])
    result = module.main()
    status = json.loads((tmp_path / "job/job_status.json").read_text())
    assert calls == expected_phases
    assert status["state"] == expected_state
    assert (result == 0) == (expected_state == "completed")
