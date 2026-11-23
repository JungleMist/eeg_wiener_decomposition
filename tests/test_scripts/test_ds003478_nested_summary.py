"""Check aggregation of sealed nested CV artifacts without training."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import summarize_ds003478_nested_cv as summary


def test_eegnet_shap_weights_subjects_and_models_equally(tmp_path):
    paths = []
    ids, labels, counts = ("a", "b"), (0, 1), (1, 2)
    names = ["specific:FP1", "coherent:FP1"]
    for subject, label, count in zip(ids, labels, counts, strict=True):
        target = tmp_path / subject
        result = target / "result.json"
        paths.append(result)
        sealed = {}
        for model in range(5):
            relative = Path(f"inner_{model}") / "outer"
            path = target / relative
            path.mkdir(parents=True)
            # The two subjects have different numbers of windows. Reversed
            # signs across time samples would cancel in a signed average.
            magnitude = model if subject == "a" else 2 * model + 2
            values = np.tile([[[magnitude, -magnitude], [2, -2]]], (count, 1, 1))
            array_file, index_file = "shap_test.npz", "shap_test_epochs.csv"
            np.savez(path / array_file, values=values, names=names)
            pd.DataFrame({
                "participant_id": subject, "label": label, "recording_id": subject,
                "epoch_start_sample": np.arange(count),
            }).to_csv(path / index_file, index=False)
            for filename in (array_file, index_file):
                sealed[str(relative / filename)] = "already checked by the caller"
        (target / "result_complete.json").write_text(json.dumps(sealed))
    cfg = {"channels": {"standard_19": ["FP1"]},
           "preprocessing": {"target_sfreq": 2, "epoch_length_sec": 1}}
    rows = pd.DataFrame({"participant_id": ids, "true_label": labels})
    keys = ["phase_0.02_specific", "phase_0.02_coherent"]
    importance = summary.summarize_shap(paths, rows, "eegnet", keys, cfg, dict(zip(ids, counts)))
    assert importance.name.tolist() == names
    assert importance.mean_abs_shap.tolist() == [4.0, 2.0]
    assert importance.n_epoch_model_pairs.tolist() == [15, 15]
    groups = summary.group_importance(importance, ["component"])
    assert groups.share_of_condition.tolist() == pytest.approx([2 / 3, 1 / 3])

    # All five models must contribute to each participant's result.
    (tmp_path / "a" / "result_complete.json").write_text("{}")
    with pytest.raises(RuntimeError, match="Missing sealed SHAP outputs"):
        summary.summarize_shap(paths, rows, "eegnet", keys, cfg, dict(zip(ids, counts)))


@pytest.mark.parametrize(("name", "band"), [
    ("specific:FP1_delta_power", "delta"), ("coh_FP1_FP2_theta", "theta"),
    ("asym_F3_F4_beta", "beta"), ("coherent:FP1_hjorth_complexity", "non_band"),
])
def test_feature_band_preserves_bank_and_feature_schema(name, band):
    assert summary.feature_band(name) == band
