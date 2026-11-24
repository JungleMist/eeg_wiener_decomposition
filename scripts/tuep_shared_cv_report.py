"""Patient-level metrics, paired comparisons and SHAP summaries for EEG OOF."""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.metrics import balanced_accuracy_score, f1_score

from tuep_shared_cv_data import SEED, WINDOW_KEY, continuous


BOOTSTRAPS = 10_000


def delong(y, left, right):
    """Paired DeLong variance using positive and negative placement values."""
    pair_a = left[y == 1, None] - left[None, y == 0]
    pair_b = right[y == 1, None] - right[None, y == 0]
    contrast = (np.sign(pair_a) - np.sign(pair_b)) * 0.5
    delta = contrast.mean()
    variance = (
        contrast.mean(axis=1).var(ddof=1) / contrast.shape[0]
        + contrast.mean(axis=0).var(ddof=1) / contrast.shape[1]
    )
    se = np.sqrt(variance)
    if not se:
        return (1.0, 0.0, 0.0) if delta == 0 else (np.nan, np.nan, np.nan)
    interval = norm.ppf(0.975) * se
    return float(2 * norm.sf(abs(delta) / se)), delta - interval, delta + interval


def bootstrap_weights(y):
    rng = np.random.default_rng(SEED)
    weights = np.zeros((BOOTSTRAPS, len(y)), dtype=np.int16)
    for label in (0, 1):
        slots = np.flatnonzero(y == label)
        weights[:, slots] = rng.multinomial(len(slots), np.full(len(slots), 1 / len(slots)),
                                            size=BOOTSTRAPS)
    return weights


def rank_metrics(y, scores, weights):
    """AUROC and AP with multiplicities, using common tie groups."""
    order = np.argsort(-scores, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(scores[order])), len(y) - 1]
    weights = weights[:, order]
    tp = np.cumsum(weights * y[order], axis=1)[:, ends]
    all_seen = np.cumsum(weights, axis=1)[:, ends]
    fp = all_seen - tp
    new_tp = np.diff(tp, axis=1, prepend=0)
    new_fp = np.diff(fp, axis=1, prepend=0)
    auc = (new_tp * (fp[:, -1:] - fp + new_fp / 2)).sum(axis=1)
    auc = auc / (tp[:, -1] * fp[:, -1])
    precision = np.divide(tp, all_seen, out=np.zeros_like(tp, dtype=float), where=all_seen > 0)
    ap = (new_tp * precision).sum(axis=1) / tp[:, -1]
    return np.stack((auc, ap), axis=1)


def wilcoxon_exact(differences):
    differences = np.round(differences, 12)
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 1.0
    ranks = rankdata(np.abs(nonzero))
    observed = abs(ranks @ np.sign(nonzero))
    possibilities = np.abs(np.asarray(list(product((-1, 1), repeat=len(nonzero)))) @ ranks)
    return float(np.mean(possibilities >= observed))


def holm(pvalues):
    values = np.asarray(pvalues, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid])]
    # Keep the size of the prespecified family even if a statistic degenerates.
    adjusted = np.full(len(values), np.nan)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        values[order] * (len(values) - np.arange(len(order)))
    ))
    return adjusted


def confidence(distribution):
    return np.quantile(distribution, (0.025, 0.975), axis=0)


def read_fold(path, windows, fold):
    predictions = pd.read_csv(path / "test_predictions.csv", dtype={"evaluation_id": str})
    keys = pd.read_csv(path / "test_epoch_predictions.csv",
                       dtype={"evaluation_id": str, "record_cache_id": str})
    expected = windows[windows["evaluation_id"].isin(predictions["evaluation_id"])]
    pd.testing.assert_frame_equal(
        keys[WINDOW_KEY].reset_index(drop=True), expected[WINDOW_KEY].reset_index(drop=True),
        check_dtype=False,
    )
    with np.load(path / "shap.npz", allow_pickle=False) as shap:
        values = np.abs(shap["values"])
        names = shap["names"]
    if values.shape != (len(keys), len(names)) or not np.isfinite(values).all():
        raise ValueError(f"Invalid SHAP summary source in {path}")
    frame = pd.DataFrame(values, columns=names)
    frame["evaluation_id"] = keys["evaluation_id"]
    importance = frame.groupby("evaluation_id", sort=True).mean()
    predictions["outer_fold"] = fold
    return predictions, importance


def grouping(name, condition, method):
    component, sep, feature = name.partition("::")
    if not sep:
        component, feature = condition, name
    if method == "eegnet":
        return component, "channel"
    if feature.startswith("coh_"):
        return component, "coherence"
    if feature.startswith("plv_"):
        return component, "plv"
    return component, "base211"


def summarize(output, windows, patients, stages, methods, folds, *, dataset="tuep"):
    ids = patients["evaluation_id"].to_numpy()
    y = patients["label"].to_numpy()
    fold_ids = patients["outer_fold"].to_numpy()
    weights = bootstrap_weights(y)
    predictions, importance, metrics, by_fold, comparisons = [], [], [], [], []
    scores, bootstrap, point, fold_aucs = {}, {}, {}, {}
    for stage, conditions in stages.items():
        for method in methods:
            for condition in conditions:
                key = (method, stage, condition)
                paths = (output / stage / method / condition / f"fold_{fold:02d}"
                         for fold in range(1, folds + 1))
                fold_pairs = [read_fold(path, windows, fold) for fold, path in enumerate(paths, 1)]
                frame = pd.concat([pair[0] for pair in fold_pairs]).set_index("evaluation_id")
                frame = frame.loc[ids].reset_index()
                np.testing.assert_array_equal(frame["evaluation_id"], ids)
                np.testing.assert_array_equal(frame["true_label"], y)
                np.testing.assert_array_equal(frame["outer_fold"], fold_ids)
                scores[key] = frame["pred_proba"].to_numpy()
                bootstrap[key] = rank_metrics(y, scores[key], weights)
                point[key] = rank_metrics(y, scores[key], np.ones((1, len(y)), dtype=int))[0]
                scopes = {"method": method, "phase": stage, "condition": condition}
                lo, hi = confidence(bootstrap[key])
                metrics.append({
                    **scopes, "patients": len(y), "auroc": point[key][0],
                    "auroc_low": lo[0], "auroc_high": hi[0], "auprc": point[key][1],
                    "auprc_low": lo[1], "auprc_high": hi[1], "auprc_baseline": y.mean(),
                    "balanced_accuracy": balanced_accuracy_score(y, frame["predicted_label"]),
                    "macro_f1": f1_score(y, frame["predicted_label"], average="macro"),
                })
                fold_aucs[key] = []
                for fold in range(1, folds + 1):
                    mask = fold_ids == fold
                    count = int(mask.sum())
                    auc, ap = rank_metrics(y[mask], scores[key][mask],
                                           np.ones((1, count), dtype=int))[0]
                    fold_aucs[key].append(auc)
                    by_fold.append({**scopes, "outer_fold": fold, "patients": count,
                                    "auroc": auc, "auprc": ap})
                predictions.append(frame.assign(**scopes))
                attributions = pd.concat([pair[1] for pair in fold_pairs])
                np.testing.assert_array_equal(np.sort(attributions.index), np.sort(ids))
                if attributions.isna().any().any():
                    raise ValueError(f"SHAP item names differ between folds for {key}")
                for name, value in attributions.mean(axis=0).items():
                    component, block = grouping(name, condition, method)
                    importance.append({**scopes, "item": name, "component": component,
                                       "block": block, "mean_abs_shap": value})

    for method, stage, condition in scores:
        if stage == "raw":
            continue
        key = (method, stage, condition)
        references = [(method, "raw", "raw")]
        if condition == "specific_coherent":
            references.extend((method, stage, name) for name in ("specific", "coherent"))
        for ref in references:
            delta = point[key] - point[ref]
            distribution = bootstrap[key] - bootstrap[ref]
            lo, hi = confidence(distribution)
            p, dlo, dhi = delong(y, scores[key], scores[ref])
            p_ap = (1 + np.sum(np.abs(distribution[:, 1] - delta[1]) >= abs(delta[1]))) / (
                len(distribution) + 1
            )
            fold_deltas = np.array(fold_aucs[key]) - fold_aucs[ref]
            comparisons.append({
                "method": method, "phase": stage, "condition": condition,
                "reference_phase": ref[1], "reference_condition": ref[2],
                "delta_auroc": delta[0], "auroc_bootstrap_low": lo[0], "auroc_bootstrap_high": hi[0],
                "auroc_delong_low": dlo, "auroc_delong_high": dhi, "auroc_delong_p": p,
                "delta_auprc": delta[1], "auprc_low": lo[1], "auprc_high": hi[1],
                "auprc_bootstrap_p": p_ap, "wilcoxon_auroc_p": wilcoxon_exact(fold_deltas),
                **{f"fold_{i:02d}_delta_auroc": value for i, value in enumerate(fold_deltas, 1)},
            })

    paired = pd.DataFrame(comparisons)
    primary = paired["reference_phase"] == "raw"
    for column in ("auroc_delong_p", "auprc_bootstrap_p"):
        paired.loc[primary, f"{column}_holm"] = holm(paired.loc[primary, column])
    shap = pd.DataFrame(importance)
    scopes = ["method", "phase", "condition"]
    total = shap.groupby(scopes)["mean_abs_shap"].transform("sum")
    shap["share"] = shap["mean_abs_shap"].div(total.where(total != 0))
    blocks = shap.groupby([*scopes, "component", "block"], as_index=False)[
        ["mean_abs_shap", "share"]
    ].sum(min_count=1)

    target = output / "summary"
    target.mkdir(exist_ok=True)
    for filename, table in {
        "oof_patient_predictions.csv": pd.concat(predictions, ignore_index=True),
        "metrics.csv": pd.DataFrame(metrics), "fold_metrics.csv": pd.DataFrame(by_fold),
        "paired_comparisons.csv": paired, "shap_importance.csv": shap, "shap_blocks.csv": blocks,
    }.items():
        continuous._atomic_dataframe(table, target / filename)
    positive_class = "normal" if dataset == "tuab" else "control"
    lines = [
        f"# {dataset.upper()} ECMAD shared 10-fold evaluation", "",
        f"{len(y)} patients, {len(windows)} matched windows. Positive class: {positive_class}.", "",
    ]
    if dataset == "tuab":
        excluded = pd.read_csv(output / "recordings.csv")["exclusion_reason"].notna().sum()
        lines += [
            "Official train and eval recordings were pooled for patient-level cross-validation.",
            f"Patients with conflicting recording labels were excluded ({excluded} recordings). "
            "Estimates apply to patients with a consistent recording label.", "",
        ]
    lines += [
        "| Model | Phase | Input | OOF AUROC (95% CI) | OOF AP (95% CI) | Balanced accuracy |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            "| {method} | {phase} | {condition} | {auroc:.3f} [{auroc_low:.3f}, {auroc_high:.3f}] "
            "| {auprc:.3f} [{auprc_low:.3f}, {auprc_high:.3f}] | {balanced_accuracy:.3f} |".format(**row)
        )
    lines += [
        "", f"AP baseline: {y.mean():.3f}. These metrics describe downstream classification.",
        "Intervals use 10,000 stratified paired patient resamples of the fixed OOF predictions.",
        "DeLong and bootstrap on these predictions do not capture variation from splitting or refitting.",
        "Raw contrasts form one Holm family per endpoint across the included models. "
        "Fold Wilcoxon and within-phase contrasts are exploratory.",
        "SHAP importance gives each patient equal weight.",
    ]
    if "eegnet" in methods:
        lines.append(f"EEGNet explains {positive_class} probability.")
    if "xgboost" in methods:
        lines.append(f"XGBoost explains {positive_class} log-odds.")
    lines.append("")
    temporary = target / ".report.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(target / "report.md")
