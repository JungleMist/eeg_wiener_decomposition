"""Resume contract tests without EEG training or CUDA."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("nested_resume", SCRIPTS / "run_ds003478_nested_cv.py")
nested = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nested)


def test_marker_detects_interruption_and_corruption(tmp_path):
    file = tmp_path / "model"
    file.write_bytes(b"partial")
    assert not nested.complete(tmp_path)
    nested.seal(tmp_path, ["model"])
    assert nested.complete(tmp_path)
    file.write_bytes(b"damaged")
    assert not nested.complete(tmp_path)
    (tmp_path / "complete.json").write_text("{")
    assert not nested.complete(tmp_path)


def test_completed_xgb_fold_skips_fit_and_incomplete_fold_restarts(tmp_path):
    class Model:
        fits = 0
        def __init__(self, **kwargs):
            pass
        def fit(self, *args, **kwargs):
            Model.fits += 1
        def predict_proba(self, x):
            return np.tile([0.7, 0.3], (len(x), 1))
        def save_model(self, path):
            path.write_text("model")
        def load_model(self, path):
            assert path.read_text() == "model"

    x = np.arange(12).reshape(4, 3)
    rows = pd.DataFrame({"participant_id": ["a", "a", "b", "b"], "label": [0, 0, 1, 1]})
    with patch.dict(sys.modules, {"xgboost": SimpleNamespace(XGBClassifier=Model)}):
        first = nested.train_xgb_fold(tmp_path, x, rows, x, rows, {}, {}, 30)[2]
        second = nested.train_xgb_fold(tmp_path, x, rows, x, rows, {}, {}, 30)[2]
        pd.testing.assert_frame_equal(first, second)
        assert Model.fits == 1
        (tmp_path / "complete.json").unlink()
        nested.train_xgb_fold(tmp_path, x, rows, x, rows, {}, {}, 30)
        assert Model.fits == 2
        assert {f.name for f in tmp_path.iterdir()} == {"complete.json", "validation.csv"}
        (tmp_path / "validation.csv").write_text("corrupt")
        nested.train_xgb_fold(tmp_path, x, rows, x, rows, {}, {}, 30)
        assert Model.fits == 3

def test_legacy_weight_cleanup_preserves_validation(tmp_path):
    path = tmp_path / "search/candidate_001/fold_0"
    path.mkdir(parents=True)
    for name in ["validation.csv", "model.ubj", "scaler.joblib"]:
        (path / name).write_text(name)
    nested.seal(path, ["validation.csv", "model.ubj", "scaler.joblib"])
    nested.migrate_search(tmp_path)
    assert nested.complete(path)
    assert {f.name for f in path.iterdir()} == {"complete.json", "validation.csv"}
