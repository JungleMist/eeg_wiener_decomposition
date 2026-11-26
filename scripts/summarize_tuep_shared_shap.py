"""Aggregate held-out SHAP values across the TUEP shared 10-fold XGBoost run.

Reads each fold's ``shap.npz`` plus its epoch index, aligns features by name and
windows by ``evaluation_id``, and reports per-fold and per-config summaries.
Patients are weighted equally, mirroring ``tuep_shared_cv_report``: a patient's
attribution is the mean over its own windows inside the one fold that held it
out, so folds with many windows cannot dominate the global mean.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

METHOD = "xgboost"
FOLD_COUNT = 10
PHASES = ("raw", "0.01", "0.05", "0.1", "0.5", "1")
SCALES = ("raw", "specific", "coherent", "specific_coherent")
BLOCK_ORDER = ("base211", "coherence", "plv")


def grouping(name, condition):
    """Split a SHAP item into its component bank and feature block."""
    component, sep, feature = name.partition("::")
    if not sep:
        component, feature = condition, name
    if feature.startswith("coh_"):
        return component, feature, "coherence"
    if feature.startswith("plv_"):
        return component, feature, "plv"
    return component, feature, "base211"


def discover(root):
    """Return ``(phase, condition, [fold directories])`` for completed configs."""
    found = []
    for phase in PHASES:
        base = root / phase / METHOD
        conditions = [phase] if phase == "raw" else [s for s in SCALES if s != "raw"]
        for condition in conditions:
            folds = [base / condition / f"fold_{fold:02d}" for fold in range(1, FOLD_COUNT + 1)]
            missing = [str(path) for path in folds if not (path / "shap.npz").exists()]
            if missing:
                raise FileNotFoundError(f"Incomplete config {phase}/{condition}: {missing}")
            found.append((phase, condition, folds))
    if not found:
        raise FileNotFoundError(f"No SHAP artifacts under {root}")
    return found


def fold_table(fold_data, scopes, fold):
    """Per-fold detail, patient-weighted with a window-weighted diagnostic."""
    names = fold_data["names"]
    frame = pd.DataFrame({
        **scopes, "outer_fold": fold, "n_patients": fold_data["n_patients"],
        "n_windows": fold_data["n_windows"],
        "mean_abs_shap": fold_data["patient_abs"].mean(axis=0),
        "mean_shap": fold_data["patient_signed"].mean(axis=0),
        "positive_fraction": (fold_data["patient_signed"] > 0).mean(axis=0),
        "window_weighted_mean_abs_shap": fold_data["window_abs"],
    })
    groups = [grouping(name, scopes["condition"]) for name in names]
    frame.insert(3, "feature", [group[1] for group in groups])
    frame.insert(3, "component", [group[0] for group in groups])
    frame.insert(5, "block", [group[2] for group in groups])
    frame.insert(3, "shap_name", names)
    total = frame["mean_abs_shap"].sum()
    frame["share"] = frame["mean_abs_shap"] / total if total else np.nan
    frame["rank"] = frame["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
    return frame[["method", "phase", "condition", "component", "block", "shap_name", "feature",
                  "outer_fold", "n_patients", "n_windows", "mean_abs_shap", "mean_shap",
                  "positive_fraction", "share", "rank", "window_weighted_mean_abs_shap"]]


def read_folder(path, phase, condition, fold, expected_fold):
    """Read a fold, retaining window-level means used for diagnostics."""
    index = pd.read_csv(path / "test_epoch_predictions.csv",
                        dtype={"evaluation_id": str, "record_cache_id": str})
    with np.load(path / "shap.npz", allow_pickle=False) as artifact:
        values = artifact["values"].astype(np.float64, copy=False)
        names = [str(name) for name in artifact["names"]]
        additivity = float(artifact["max_additivity_error"])
    keys = ["evaluation_id", "record_cache_id", "epoch_start_sample"]
    if index[keys].duplicated().any():
        raise ValueError(f"Duplicated window keys in {path}")
    if values.shape != (len(index), len(names)):
        raise ValueError(f"SHAP shape {values.shape} does not match {len(index)} windows in {path}")
    if not np.isfinite(values).all() or not np.isfinite(additivity):
        raise ValueError(f"Non-finite SHAP values in {path}")
    if not index["evaluation_id"].map(expected_fold).eq(fold).all():
        raise ValueError(f"Fold {fold} contains windows held out by another fold in {path}")

    patients, codes = np.unique(index["evaluation_id"].to_numpy(), return_inverse=True)
    indicator = sparse.csr_matrix(
        (np.ones(len(codes)), (codes, np.arange(len(codes)))),
        shape=(len(patients), len(codes)),
    )
    counts = np.bincount(codes, minlength=len(patients)).astype(np.float64)
    return {
        "names": names, "n_windows": len(index), "n_patients": len(patients),
        "patients": patients, "counts": counts,
        "patient_abs": np.asarray(indicator @ np.abs(values)) / counts[:, None],
        "patient_signed": np.asarray(indicator @ values) / counts[:, None],
        "window_abs": np.abs(values).mean(axis=0),
        "window_signed": values.mean(axis=0),
        "additivity": additivity,
    }


def summarize(root, output, top_n):
    folds_meta = pd.read_csv(root / "patient_folds.csv", dtype={"evaluation_id": str})
    expected_fold = dict(zip(folds_meta["evaluation_id"], folds_meta["outer_fold"], strict=True))

    detail, globals_, blocks, manifest = [], [], [], {"configs": [], "fold_additivity_max": 0.0}
    index_rows = []
    for phase, condition, paths in discover(root):
        scopes = {"method": METHOD, "phase": phase, "condition": condition}
        names = None
        patient_abs, patient_signed, patient_counts = [], [], []
        patient_order, fold_frames = [], []
        for fold, path in enumerate(paths, 1):
            data = read_folder(path, phase, condition, fold, expected_fold)
            if names is None:
                names = data["names"]
            elif data["names"] != names:
                raise ValueError(f"Feature order differs across folds for {phase}/{condition}: {path}")
            fold_frames.append(fold_table(data, scopes, fold))
            index_rows.append(pd.DataFrame({
                **scopes, "outer_fold": fold, "evaluation_id": data["patients"],
                "n_windows": data["counts"].astype(int), "n_features": len(data["names"]),
            }))
            patient_abs.append(data["patient_abs"])
            patient_signed.append(data["patient_signed"])
            patient_counts.append(data["counts"])
            patient_order.append(data["patients"])
            manifest["fold_additivity_max"] = max(manifest["fold_additivity_max"], data["additivity"])
            del data

        names = np.asarray(names, dtype=object)
        component = np.array([grouping(name, condition)[0] for name in names], dtype=object)
        feature = np.array([grouping(name, condition)[1] for name in names], dtype=object)
        block = np.array([grouping(name, condition)[2] for name in names], dtype=object)

        patients = np.concatenate(patient_order)
        if len(np.unique(patients)) != len(patients):
            raise ValueError(f"Patients are explained by more than one fold for {phase}/{condition}")
        rows_abs = np.concatenate(patient_abs)
        rows_signed = np.concatenate(patient_signed)
        counts = np.concatenate(patient_counts)

        per_fold = pd.concat(fold_frames, ignore_index=True)
        detail.append(per_fold)
        fold_stats = per_fold.groupby("shap_name").agg(
            fold_mean_abs_sd=("mean_abs_shap", "std"),
            fold_rank_mean=("rank", "mean"),
            fold_rank_best=("rank", "min"),
            fold_rank_worst=("rank", "max"),
        )
        top10 = per_fold[per_fold["rank"] <= 10].groupby("shap_name").size()
        top20 = per_fold[per_fold["rank"] <= 20].groupby("shap_name").size()
        window_weighted = per_fold.groupby("shap_name")[
            "window_weighted_mean_abs_shap"].mean().reindex(names)

        spread = rows_signed.std(axis=0, ddof=1)
        table = pd.DataFrame({
            **scopes, "component": component, "block": block, "feature": feature,
            "n_patients": len(patients), "n_windows": int(counts.sum()),
            "mean_abs_shap": rows_abs.mean(axis=0),
            "mean_shap": rows_signed.mean(axis=0),
            "patient_signed_sd": spread,
            "direction_ratio": np.divide(rows_signed.mean(axis=0), spread,
                                         out=np.zeros(len(names)), where=spread > 0),
            "positive_fraction": (rows_signed > 0).mean(axis=0),
            "share": rows_abs.mean(axis=0) / rows_abs.mean(axis=0).sum(),
            "fold_mean_abs_sd": np.nan, "fold_rank_mean": np.nan,
            "fold_rank_best": np.nan, "fold_rank_worst": np.nan,
            "folds_in_top10": 0, "folds_in_top20": 0,
            "window_weighted_mean_abs_shap": window_weighted.to_numpy(),
            "mean_windows_per_patient": float(counts.mean()),
        })
        table.index = pd.Index(names, name="shap_name")
        table.update(fold_stats)
        table["folds_in_top10"] = top10.reindex(table.index).fillna(0).astype(int)
        table["folds_in_top20"] = top20.reindex(table.index).fillna(0).astype(int)
        table = table.reset_index()
        table["rank"] = table["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
        table = table.sort_values("rank", kind="stable")
        globals_.append(table)

        block_table = table.groupby(["method", "phase", "condition", "component", "block"],
                                    as_index=False).agg(
            mean_abs_shap=("mean_abs_shap", "sum"), share=("share", "sum"),
            n_features=("feature", "size"),
        )
        blocks.append(block_table)

        manifest["configs"].append({
            **scopes, "folds": len(paths), "features": int(len(names)),
            "windows": int(counts.sum()), "patients": int(len(patients)),
            "patients_per_fold": per_fold.groupby("outer_fold")["n_patients"].first().to_dict(),
            "windows_per_fold": per_fold.groupby("outer_fold")["n_windows"].first().to_dict(),
            "components": sorted(set(component)), "blocks": sorted(set(block)),
        })

    detail = pd.concat(detail, ignore_index=True)
    global_table = pd.concat(globals_, ignore_index=True)
    block_table = pd.concat(blocks, ignore_index=True)
    block_table["block"] = pd.Categorical(block_table["block"], BLOCK_ORDER, ordered=True)
    phase_order = {phase: index for index, phase in enumerate(PHASES)}
    condition_order = {name: index for index, name in enumerate(SCALES)}
    block_table = block_table.assign(
        _phase=block_table["phase"].map(phase_order),
        _condition=block_table["condition"].map(condition_order),
    ).sort_values(["_phase", "_condition", "component", "block"]).drop(
        columns=["_phase", "_condition"]).astype({"block": str})

    top = global_table[global_table["rank"] <= top_n].copy()
    top = top.assign(
        _phase=top["phase"].map(phase_order), _condition=top["condition"].map(condition_order)
    ).sort_values(["_phase", "_condition", "rank"]).drop(
        columns=["_phase", "_condition"]).reset_index(drop=True)
    top["direction"] = np.where(top["mean_shap"] > 0, "raises control log-odds",
                                "lowers control log-odds")

    consensus = cross_phase_consensus(global_table)
    manifest["feature_alignment"] = "SHAP names identical in content and order across all 10 folds"
    manifest["window_alignment"] = "windows mapped by evaluation_id to the shared patient folds"
    manifest["additivity_tolerance"] = 0.01
    manifest["weighting"] = "patient-equal: per-patient window means, then mean over patients"

    output.mkdir(parents=True, exist_ok=True)
    patient_index = pd.concat(index_rows, ignore_index=True)
    if patient_index.duplicated(["phase", "condition", "evaluation_id"]).any():
        raise ValueError("A patient was explained by more than one fold in the same config")
    detail.to_csv(output / "fold_feature_shap.csv", index=False)
    patient_index.to_csv(output / "fold_patient_index.csv", index=False)
    global_table.to_csv(output / "global_feature_shap.csv", index=False)
    top.to_csv(output / "top_features_by_config.csv", index=False)
    block_table.to_csv(output / "global_block_shap.csv", index=False)
    consensus.to_csv(output / "cross_phase_consensus.csv", index=False)
    (output / "alignment_manifest.json").write_text(json.dumps(manifest, indent=2))
    return detail, global_table, top, block_table, consensus, manifest


def cross_phase_consensus(global_table):
    """Rank features consistently important across the five ECMAD phases."""
    family = global_table["condition"].where(
        global_table["condition"] != "specific_coherent",
        "specific_coherent:" + global_table["component"])
    working = global_table.assign(family=family)
    rows = []
    for name, values in working.groupby("family", sort=False):
        if values["phase"].nunique() < 2:
            continue
        pivoted = values.pivot_table(index="feature", columns="phase", values="rank")
        counts = values[values["rank"] <= 10].groupby("feature")["phase"].nunique()
        top20 = values[values["rank"] <= 20].groupby("feature")["phase"].nunique()
        for feature, ranks in pivoted.iterrows():
            rows.append({
                "family": name, "feature": feature,
                "n_phases": int(ranks.notna().sum()),
                "phases_in_top10": int(counts.get(feature, 0)),
                "phases_in_top20": int(top20.get(feature, 0)),
                "mean_rank": float(np.nanmean(ranks.to_numpy(dtype=float))),
                "best_phase": ranks.idxmin(), "best_rank": int(ranks.min()),
                **{f"rank_{phase}": ranks.get(phase, np.nan) for phase in PHASES},
            })
    frame = pd.DataFrame(rows)
    return frame.sort_values(["family", "mean_rank"]).reset_index(drop=True)


def write_report(output, detail, global_table, top, block_table, consensus, manifest):
    lines = ["# TUEP shared 10-fold XGBoost — held-out SHAP summary", ""]
    total_windows = manifest["configs"][0]["windows"]
    total_patients = manifest["configs"][0]["patients"]
    lines += [
        f"{len(manifest['configs'])} configs × {FOLD_COUNT} folds. Every config reuses the same "
        f"{total_windows:,} held-out windows from {total_patients} patients, each patient held out "
        f"once. SHAP explains control log-odds (positive class: control).",
        "",
        "## Method",
        "",
        "- Each fold's `shap.npz` is reduced to per-patient means over that patient's own windows.",
        "- A patient appears as test exactly once, so the global mean weights every patient equally.",
        "- `mean_abs_shap` is importance; `mean_shap` keeps the sign and gives the direction.",
        "- `positive_fraction` is the share of patients whose signed attribution is positive, and "
        "`direction_ratio` is `mean_shap / patient_signed_sd`; a large mean |SHAP| with a near-zero "
        "ratio means the sign flips across patients rather than a shared direction.",
        "- `share` is the feature's fraction of that config's total mean |SHAP|.",
        "",
        "## Configurations",
        "",
        "| Phase | Input | Features | Windows | Patients | Top feature | mean \\|SHAP\\| | "
        "share | patients positive |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for config in manifest["configs"]:
        best = global_table[(global_table["phase"] == config["phase"])
                            & (global_table["condition"] == config["condition"])].iloc[0]
        lines.append(
            f"| {config['phase']} | {config['condition']} | {config['features']} | "
            f"{config['windows']:,} | {config['patients']} | {best['feature']} | "
            f"{best['mean_abs_shap']:.4f} | {best['share']:.1%} | "
            f"{best['positive_fraction']:.2f} |")

    lines += ["", f"## Top {int(top['rank'].max())} features per config", ""]
    for (phase, condition), values in top.groupby(["phase", "condition"], sort=False):
        lines += [f"### {phase} / {condition}", "",
                  "| # | Feature | Component | Block | mean \\|SHAP\\| | share | mean SHAP | "
                  "mean/SD | patients positive | folds top-10 |",
                  "|---:|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for row in values.itertuples():
            lines.append(
                f"| {row.rank} | {row.feature} | {row.component} | {row.block} | "
                f"{row.mean_abs_shap:.4f} | {row.share:.3%} | {row.mean_shap:+.4f} | "
                f"{row.direction_ratio:+.2f} | {row.positive_fraction:.2f} | "
                f"{row.folds_in_top10}/{FOLD_COUNT} |")
        lines.append("")

    lines += ["## Block shares", "",
              "| Phase | Input | Component | base211 | coherence | plv |",
              "|---|---|---|---:|---:|---:|"]
    pivot = block_table.pivot_table(
        index=["phase", "condition", "component"], columns="block", values="share",
        aggfunc="sum", fill_value=0.0)
    for (phase, condition, component), row in pivot.iterrows():
        lines.append(f"| {phase} | {condition} | {component} | "
                     + " | ".join(f"{row.get(block, 0.0):.1%}" for block in BLOCK_ORDER) + " |")

    lines += ["", "## Features with no attribution", "",
              "A feature with exactly zero mean |SHAP| was never used by any tree in any fold.", "",
              "| Phase | Input | Zero-SHAP features | Of total | Zero in coherence block |",
              "|---|---|---:|---:|---:|"]
    for (phase, condition), values in global_table.groupby(["phase", "condition"], sort=False):
        zeros = int((values["mean_abs_shap"] == 0).sum())
        coherence = values[values["block"] == "coherence"]
        lines.append(f"| {phase} | {condition} | {zeros} | {len(values)} | "
                     f"{int((coherence['mean_abs_shap'] == 0).sum())}/{len(coherence)} |")
    coherence_only = global_table[global_table["block"] == "coherence"].groupby(
        ["phase", "condition"])["mean_abs_shap"].max()
    silent = coherence_only[coherence_only == 0]
    if len(silent):
        lines += ["", "The `coherence` block never contributed a split in "
                  + ", ".join(f"`{phase}/{condition}`" for phase, condition in silent.index)
                  + ". A block that is exactly zero in every fold is consistent with a constant or "
                  "degenerate feature block, so check feature variance in those inputs before "
                  "interpreting the coherent-component models."]

    lines += ["", "## Direction stability", "",
              "How many of each config's top-20 features have a direction that holds across "
              "patients, measured by |mean SHAP| / patient SD.", "",
              "| Phase | Input | < 0.25 | 0.25–0.50 | ≥ 0.50 | Strongest feature | ratio |",
              "|---|---|---:|---:|---:|---|---:|"]
    for (phase, condition), values in top.groupby(["phase", "condition"], sort=False):
        magnitude = values["direction_ratio"].abs()
        strongest = values.loc[magnitude.idxmax()]
        lines.append(
            f"| {phase} | {condition} | {int((magnitude < 0.25).sum())} | "
            f"{int(((magnitude >= 0.25) & (magnitude < 0.5)).sum())} | "
            f"{int((magnitude >= 0.5).sum())} | {strongest['feature']} | "
            f"{strongest['direction_ratio']:+.2f} |")

    lines += ["", "## Cross-phase stability", "",
              "Features ranked in the top 10 by the most phases within one input family.", ""]
    leaders = (consensus.sort_values(["family", "phases_in_top10", "mean_rank"],
                                     ascending=[True, False, True])
               .groupby("family", sort=False).head(5))
    lines += ["| Family | Feature | Phases in top-10 | Mean rank | Best phase |",
              "|---|---|---:|---:|---|"]
    for row in leaders.itertuples():
        lines.append(f"| {row.family} | {row.feature} | {row.phases_in_top10}/5 | "
                     f"{row.mean_rank:.1f} | {row.best_phase} |")

    lines += ["", "## Checks", "",
              f"- Feature names identical and identically ordered across the 10 folds of each config.",
              f"- Windows mapped by `evaluation_id`; each patient is held out exactly once.",
              f"- Maximum TreeSHAP additivity error over {len(manifest['configs']) * FOLD_COUNT} "
              f"folds: {manifest['fold_additivity_max']:.2e} (tolerance 1e-2).",
              f"- Per-fold detail: {len(detail):,} rows in `fold_feature_shap.csv`.",
              "- Sample alignment: `fold_patient_index.csv` lists each patient's fold and window "
              "count per config.",
              "",
              "Direction is the mean signed SHAP relative to the interventional background, not a "
              "marginal effect on the raw feature scale. Report `positive_fraction` alongside it."]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1] / "results" / "tuep_shared_10fold_xgboost"
    parser.add_argument("--results-dir", type=Path, default=root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()
    output = args.output_dir or args.results_dir / "shap_fold_summary"
    tables = summarize(args.results_dir, output, args.top_n)
    write_report(output, *tables)
    detail, global_table, top, blocks, consensus, manifest = tables
    print(f"wrote {len(detail):,} fold rows and {len(global_table):,} global rows to {output}")
    for config in manifest["configs"]:
        best = global_table[(global_table["phase"] == config["phase"])
                            & (global_table["condition"] == config["condition"])].head(3)
        print(f"{config['phase']:>4} / {config['condition']:<17} "
              + "; ".join(f"{row.feature} ({row.mean_abs_shap:.4f})" for row in best.itertuples()))


if __name__ == "__main__":
    main()
