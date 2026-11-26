"""Check SHAP aggregation warns-free alignment and patient-equal weighting."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import summarize_tuep_shared_shap as summary


def test_grouping_splits_component_and_block():
    assert summary.grouping("specific::coh_FP1_FP2_alpha", "specific_coherent") == (
        "specific", "coh_FP1_FP2_alpha", "coherence")
    assert summary.grouping("coherent::plv_O1_O2_beta", "specific_coherent") == (
        "coherent", "plv_O1_O2_beta", "plv")
    assert summary.grouping("FP1_alpha_power", "raw") == ("raw", "FP1_alpha_power", "base211")


def test_fold_table_weights_patients_not_windows():
    # Patient B has three windows and no attribution; a window-level mean would
    # halve the importance of both features to 0.5.
    data = {
        "names": ["a_power", "b_power"], "n_windows": 4, "n_patients": 2,
        "counts": np.array([1.0, 3.0]),
        "patient_abs": np.array([[2.0, 2.0], [0.0, 0.0]]),
        "patient_signed": np.array([[2.0, -2.0], [0.0, 0.0]]),
        "window_abs": np.array([0.5, 0.5]),
    }
    frame = summary.fold_table(data, {"method": "xgboost", "phase": "0.1",
                                      "condition": "specific"}, 3)
    assert frame["mean_abs_shap"].tolist() == [1.0, 1.0]
    assert frame["mean_shap"].tolist() == [1.0, -1.0]
    assert frame["window_weighted_mean_abs_shap"].tolist() == [0.5, 0.5]
    assert frame["positive_fraction"].tolist() == [0.5, 0.0]
    assert frame["share"].sum() == 1.0
    assert frame["outer_fold"].tolist() == [3, 3]
    assert frame["component"].tolist() == ["specific", "specific"]
    assert frame["block"].tolist() == ["base211", "base211"]


def test_cross_phase_consensus_counts_phases_per_family():
    rows = []
    for phase, ranks in {"0.01": (1, 5), "0.05": (2, 9), "0.1": (3, 30)}.items():
        for feature, rank in zip(("FP1_alpha_power", "T5_theta_power"), ranks, strict=True):
            rows.append({"method": "xgboost", "phase": phase, "condition": "specific",
                         "component": "specific", "block": "base211", "feature": feature,
                         "mean_abs_shap": 1.0 / rank, "mean_shap": 0.0, "share": 0.5,
                         "rank": rank})
    table = summary.cross_phase_consensus(pd.DataFrame(rows))
    leader = table[table["feature"] == "FP1_alpha_power"].iloc[0]
    assert leader["family"] == "specific"
    assert leader["n_phases"] == 3
    assert leader["phases_in_top10"] == 3
    assert leader["mean_rank"] == 2.0
    assert leader["best_phase"] == "0.01"
    assert np.isnan(leader["rank_0.5"])  # a phase absent from the family stays missing
