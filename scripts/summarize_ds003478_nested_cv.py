import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score
import ECMAD as p
from run_ds003478_nested_cv import XGBOOST_PROTOCOL, complete
from eeg_bg.config.settings import load_config

a = argparse.ArgumentParser()
a.add_argument('--cache', type=Path, required=True)
a.add_argument('--output', type=Path, required=True)
args = a.parse_args()
expected = set(p.read_records(args.cache).participant_id)
for manifest in (args.output / 'xgboost').glob('outer_*/manifest.json'):
    if json.loads(manifest.read_text()).get('xgboost_protocol') != XGBOOST_PROTOCOL:
        raise RuntimeError(f'Incompatible XGBoost protocol: {manifest}')
summary = []
for method in ('eegnet', 'xgboost'):
    for condition, _, _ in p.condition_specs(load_config('configs/ECMAD.yaml')):
        paths = list((args.output / method).glob(f'outer_*/{condition}/result.json'))
        if any(not complete(f.parent, 'result_complete.json') for f in paths):
            raise RuntimeError(f'Unverified or damaged result: {method}/{condition}')
        rows = pd.DataFrame([json.loads(f.read_text()) for f in paths])
        if rows.empty or set(rows.participant_id) != expected or rows.participant_id.duplicated().any():
            raise RuntimeError(f'Incomplete results: {method}/{condition}')
        rows.to_csv(args.output / f'{method}_{condition}_oof.csv', index=False)
        summary.append(dict(method=method, condition=condition,
            auroc=roc_auc_score(rows.true_label, rows.pred_proba),
            auprc=average_precision_score(rows.true_label, rows.pred_proba),
            balanced_accuracy=balanced_accuracy_score(rows.true_label, rows.predicted_label),
            f1=f1_score(rows.true_label, rows.predicted_label, zero_division=0)))
pd.DataFrame(summary).to_csv(args.output / 'comparison.csv', index=False)
