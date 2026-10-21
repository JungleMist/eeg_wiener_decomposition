#!/usr/bin/env python3
"""Audit and analyze a completed auto_ecmad run without fitting any models."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime
import html
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)

from eeg_bg.auto_ecmad.artifacts import fingerprint, read_json, sha256_file, write_csv, write_json

METRICS = ("auroc", "auprc", "f1", "balanced_accuracy", "accuracy")


def scores(y, p, threshold):
    predicted = p >= threshold
    return {"auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
            "auprc": float(average_precision_score(y, p)),
            "f1": float(f1_score(y, predicted, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "accuracy": float(np.mean(y == predicted)),
            "brier": float(brier_score_loss(y, p))}


def audit(root):
    state = read_json(root / "prepared.json")
    mapping = read_json(root / "selected_mapping.json")
    summary = read_json(root / "summary.json")
    channels = state["channels"]
    candidates = pd.read_csv(root / "candidate_scores.csv")
    expected = {(t, r) for t in channels for r in channels if r != t}
    if len(candidates) != len(expected) or set(zip(candidates.target, candidates.reference)) != expected:
        raise ValueError("Candidate search is incomplete or contains duplicates")
    if not np.isfinite(candidates.selection_loss).all():
        raise ValueError("Non-finite candidate selection loss")
    payload = {k: v for k, v in mapping.items() if k != "fingerprint"}
    if mapping["run_fingerprint"] != state["fingerprint"] or fingerprint(payload) != mapping["fingerprint"]:
        raise ValueError("Mapping provenance mismatch")
    tolerance = state["identity"]["config"]["auto_ecmad"]["search"]["tie_tolerance"]
    histories, candidate_rows = {}, []
    for row in candidates.to_dict("records"):
        directory = root / "candidates" / row["candidate_id"]
        marker = read_json(directory / "complete.json")
        if any(sha256_file(directory / name) != digest for name, digest in marker["files"].items()):
            raise ValueError(f"Candidate artifact checksum mismatch: {directory}")
        history = pd.read_csv(directory / "history.csv")
        fit = read_json(directory / "fit_result.json")
        best = history.loc[history.epoch == fit["best_epoch"]].iloc[0]
        if not np.isclose(best.validation_subject_weighted_bce, row["selection_loss"], atol=1e-7):
            raise ValueError("Candidate score and selected epoch differ")
        candidate_rows.append({**row, "epochs_run": len(history),
                               "best_train_bce": best.train_subject_weighted_bce,
                               "best_validation_auprc": best.validation_auprc})
    for target in channels:
        group = candidates.loc[candidates.target == target]
        eligible = group.loc[group.selection_loss <= group.selection_loss.min() + tolerance]
        expected_reference = min(eligible.reference, key=channels.index)
        if mapping["references"][target] != expected_reference:
            raise ValueError("Frozen reference differs from candidate ranking")
    manifest = pd.read_csv(root / "trial_manifest.csv")
    if manifest.trial_id.duplicated().any():
        raise ValueError("Duplicate trial identity")
    partitions = state["partitions"]
    ids = [s for values in partitions.values() for s in values]
    if len(ids) != len(set(ids)):
        raise ValueError("Subject overlap between partitions")
    predictions, frozen, checkpoints = {}, {}, {}
    for condition in ("raw", "selected"):
        directory = root / "models" / condition
        frozen[condition] = read_json(directory / "frozen_model.json")
        model = frozen[condition]
        if model["mapping_fingerprint"] != mapping["fingerprint"] or model["freeze_fingerprint"] != fingerprint(
            {key: value for key, value in model.items() if key != "freeze_fingerprint"}
        ) or sha256_file(directory / model["checkpoint"]) != model["checkpoint_sha256"]:
            raise ValueError("Frozen model provenance mismatch")
        checkpoints[condition] = directory / model["checkpoint"]
        histories[condition] = pd.read_csv(directory / "history.csv")
        for split in ("validation", "test"):
            frame = pd.read_csv(directory / f"{split}_predictions.csv")
            expected_trials = manifest.loc[manifest.split == split].set_index("trial_id").sort_index()
            if frame.trial_id.duplicated().any():
                raise ValueError("Duplicate prediction identity")
            frame = frame.set_index("trial_id").sort_index()
            if not frame.index.equals(expected_trials.index) or not np.array_equal(
                frame.true_label, expected_trials.true_label
            ) or not np.array_equal(frame.subject_id, expected_trials.subject_id):
                raise ValueError("Prediction trial/split/label mismatch")
            p = frame.pred_proba.to_numpy()
            if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise ValueError("Invalid predictions")
            observed = scores(frame.true_label.to_numpy(), p, model["threshold"])
            declared = summary["results"][condition][split]["trial"]
            if any(not np.isclose(observed[m], declared[m], atol=1e-8, equal_nan=True) for m in METRICS):
                raise ValueError("Summary metrics disagree with saved predictions")
            predictions[condition, split] = frame
    if not predictions["raw", "test"].index.equals(predictions["selected", "test"].index):
        raise ValueError("Raw and selected test trials do not match")
    return state, mapping, summary, manifest, pd.DataFrame(candidate_rows), predictions, frozen, histories, checkpoints


def paired_intervals(raw, selected, thresholds, repetitions, seed):
    subjects = np.asarray(sorted(raw.subject_id.unique()))
    columns = [f"delta_{metric}" for metric in METRICS]
    if len(subjects) < 2:
        return pd.DataFrame(columns=["metric", "delta", "lower_95", "upper_95", "test_subjects"])
    indices = {subject: np.flatnonzero(raw.subject_id.to_numpy() == subject) for subject in subjects}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        chosen = rng.choice(subjects, size=len(subjects), replace=True)
        idx = np.concatenate([indices[s] for s in chosen])
        a = scores(raw.true_label.to_numpy()[idx], raw.pred_proba.to_numpy()[idx], thresholds["raw"])
        b = scores(selected.true_label.to_numpy()[idx], selected.pred_proba.to_numpy()[idx], thresholds["selected"])
        values.append([b[m] - a[m] for m in METRICS])
    draws = pd.DataFrame(values, columns=columns)
    a = scores(raw.true_label.to_numpy(), raw.pred_proba.to_numpy(), thresholds["raw"])
    b = scores(selected.true_label.to_numpy(), selected.pred_proba.to_numpy(), thresholds["selected"])
    return pd.DataFrame([{"metric": m, "delta": b[m] - a[m],
                          "lower_95": draws[f"delta_{m}"].quantile(.025),
                          "upper_95": draws[f"delta_{m}"].quantile(.975),
                          "test_subjects": len(subjects)} for m in METRICS])


def figures(directory, channels, candidates, selected, predictions, frozen, histories):
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        paths.append(directory / f"{name}.png")

    matrix = candidates.pivot(index="target", columns="reference", values="selection_loss").reindex(index=channels, columns=channels)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(np.ma.masked_invalid(matrix.to_numpy()), cmap="viridis")
    ax.set_xticks(range(len(channels)), channels, rotation=90)
    ax.set_yticks(range(len(channels)), channels)
    ax.set(xlabel="Reference channel", ylabel="Target channel", title="Internal selection BCE (lower is better)")
    for target, reference in selected.items():
        ax.scatter(channels.index(reference), channels.index(target), marker="s", facecolors="none", edgecolors="white", s=90, linewidths=1.2)
    fig.colorbar(im, ax=ax, label="Subject-equal class-weighted BCE")
    save(fig, "candidate_loss_matrix")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    counts = Counter(selected.values())
    names = sorted(counts, key=lambda x: (-counts[x], channels.index(x)))
    axes[0].bar(names, [counts[n] for n in names], color="#315f84")
    axes[0].tick_params(axis="x", rotation=60)
    axes[0].set(title="Selected reference frequency", ylabel="Targets")
    gaps = []
    for channel in channels:
        ordered = candidates.loc[candidates.target == channel, "selection_loss"].sort_values().to_numpy()
        gaps.append(ordered[1] - ordered[0] if len(ordered) > 1 else np.nan)
    axes[1].bar(channels, gaps, color="#b9743a")
    axes[1].tick_params(axis="x", rotation=90)
    axes[1].set(title="Best vs second-best selection loss", ylabel="BCE margin")
    save(fig, "reference_selection")

    colors = {"raw": "#6b7280", "selected": "#136f63"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    prevalence = predictions["raw", "test"].true_label.mean()
    for condition in colors:
        frame = predictions[condition, "test"]
        y, p = frame.true_label, frame.pred_proba
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, color=colors[condition], label=f"{condition}: {roc_auc_score(y, p):.3f}")
        axes[1].plot(recall, precision, color=colors[condition], label=f"{condition}: AP {average_precision_score(y, p):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="#bbbbbb")
    axes[1].axhline(prevalence, linestyle="--", color="#bbbbbb", label=f"Prevalence {prevalence:.3f}")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Held-out ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Held-out precision-recall")
    for ax in axes:
        ax.legend()
    save(fig, "test_roc_pr")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for ax, condition in zip(axes, colors):
        frame = predictions[condition, "test"]
        matrix = confusion_matrix(frame.true_label, frame.pred_proba >= frozen[condition]["threshold"], labels=[0, 1])
        ax.imshow(matrix / matrix.sum(axis=1, keepdims=True), vmin=0, vmax=1, cmap="Blues")
        for i in range(2):
            for j in range(2):
                value = matrix[i, j]
                ax.text(j, i, f"{value}\n{value / matrix[i].sum():.1%}", ha="center", va="center", color="white" if value / matrix[i].sum() > .5 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Correct", "Error"], yticklabels=["Correct", "Error"],
               xlabel="Predicted", ylabel="Observed", title=f"{condition} / threshold {frozen[condition]['threshold']:.3f}")
    save(fig, "test_confusion_matrices")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for condition, history in histories.items():
        axes[0].plot(history.epoch, history.train_subject_weighted_bce, color=colors[condition], label=f"{condition} fit")
        axes[0].plot(history.epoch, history.validation_subject_weighted_bce, "--", color=colors[condition], label=f"{condition} validation")
        axes[1].plot(history.epoch, history.validation_auprc, color=colors[condition], label=condition)
        best_epoch = frozen[condition]["best_epoch"]
        value = history.loc[history.epoch == best_epoch, "validation_auprc"].iloc[0]
        axes[1].scatter([best_epoch], [value], color=colors[condition], s=50)
    axes[0].set(xlabel="Epoch", ylabel="Weighted BCE", title="Stage-two fit and validation loss")
    axes[1].set(xlabel="Epoch", ylabel="AUPRC", title="External validation checkpoint selection")
    for ax in axes:
        ax.legend()
    save(fig, "training_curves")

    fig, ax = plt.subplots(figsize=(5.8, 4.7))
    for condition in colors:
        frame = predictions[condition, "test"]
        fraction, mean = calibration_curve(frame.true_label, frame.pred_proba, n_bins=8, strategy="quantile")
        ax.plot(mean, fraction, "o-", color=colors[condition], label=f"{condition}: Brier {brier_score_loss(frame.true_label, frame.pred_proba):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#bbbbbb")
    ax.set(xlabel="Mean predicted error probability", ylabel="Observed error fraction", title="Test calibration (class-weighted training)")
    ax.legend()
    save(fig, "test_calibration")
    return paths


def markdown_table(frame):
    headers = list(frame.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in row) + " |")
    return "\n".join(lines)


def analyze(root, output, repetitions=1000, seed=42):
    state, mapping, summary, manifest, candidates, predictions, frozen, histories, checkpoints = audit(root)
    output.mkdir(parents=True, exist_ok=True)
    split_rows = []
    for split, frame in manifest.groupby("split", sort=False):
        split_rows.append({"split": split, "subjects": frame.subject_id.nunique(), "trials": len(frame),
                           "correct": int((frame.true_label == 0).sum()), "error": int(frame.true_label.sum()),
                           "error_prevalence": frame.true_label.mean()})
    split_table = pd.DataFrame(split_rows)
    metric_rows, subject_rows = [], []
    for (condition, split), frame in predictions.items():
        values = scores(frame.true_label.to_numpy(), frame.pred_proba.to_numpy(), frozen[condition]["threshold"])
        metric_rows.append({"condition": condition, "split": split, **values,
                            "threshold": frozen[condition]["threshold"], "best_epoch": frozen[condition]["best_epoch"]})
        for subject, group in frame.groupby("subject_id"):
            subject_rows.append({"condition": condition, "split": split, "subject_id": subject,
                                 "trials": len(group), "errors": int(group.true_label.sum()),
                                 **scores(group.true_label.to_numpy(), group.pred_proba.to_numpy(), frozen[condition]["threshold"])})
    metrics, subject_metrics = pd.DataFrame(metric_rows), pd.DataFrame(subject_rows)
    intervals = paired_intervals(predictions["raw", "test"], predictions["selected", "test"],
                                 {c: f["threshold"] for c, f in frozen.items()}, repetitions, seed)
    selection_table = pd.DataFrame(mapping["selected"])
    raw_test, selected_test = predictions["raw", "test"], predictions["selected", "test"]
    transitions = raw_test[["subject_id", "true_label", "pred_proba"]].rename(columns={"pred_proba": "raw_proba"})
    transitions["selected_proba"] = selected_test.pred_proba
    transitions["raw_correct"] = (transitions.raw_proba >= frozen["raw"]["threshold"]) == transitions.true_label
    transitions["selected_correct"] = (transitions.selected_proba >= frozen["selected"]["threshold"]) == transitions.true_label
    fixes = int((~transitions.raw_correct & transitions.selected_correct).sum())
    regressions = int((transitions.raw_correct & ~transitions.selected_correct).sum())
    for name, table in (("split_counts", split_table), ("model_metrics", metrics), ("subject_metrics", subject_metrics),
                        ("candidate_details", candidates), ("selected_references", selection_table),
                        ("paired_cluster_bootstrap", intervals)):
        write_csv(output / f"{name}.csv", table)
    write_csv(output / "test_prediction_transitions.csv", transitions.reset_index())
    images = figures(output / "figures", state["channels"], candidates, mapping["references"], predictions, frozen, histories)
    # Parameter counts concern the actual saved models, rather than channel count alone.
    import torch
    from eeg_bg.ml.cnn_model import EEGNet
    parameters = {}
    for condition, path in checkpoints.items():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        cfg = saved["model_config"]
        model = EEGNet(n_channels=saved["n_channels"], n_times=saved["n_times"],
                       F1=int(cfg["F1"]), D=int(cfg["D"]), dropout=float(cfg["dropout"]))
        parameters[condition] = sum(p.numel() for p in model.parameters())
    test_metrics = metrics.loc[metrics.split == "test"].set_index("condition")
    deltas = {name: float(test_metrics.loc["selected", name] - test_metrics.loc["raw", name]) for name in METRICS}
    counts = Counter(mapping["references"].values())
    capped = int((candidates.epochs_run >= state["identity"]["config"]["auto_ecmad"]["training"]["max_epochs"]).sum())
    excluded = read_json(root / "excluded_recordings.json")
    source_subjects = len({row["path"].split("/eeg/")[0] for row in state["sources"]})
    evidence = {"run_fingerprint": state["fingerprint"], "mapping_fingerprint": mapping["fingerprint"],
                "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "candidate_count": len(candidates), "eligible_subjects": manifest.subject_id.nunique(),
                "total_trials": len(manifest), "test_deltas_selected_minus_raw": deltas,
                "parameter_counts": parameters, "corrected_test_trials": fixes,
                "regressed_test_trials": regressions, "reference_frequency": dict(counts),
                "bootstrap_repetitions": repetitions, "bootstrap_seed": seed,
                "candidate_epoch_summary": candidates.epochs_run.describe().to_dict(),
                "candidates_reaching_epoch_cap": capped,
                "sources": {"prepared": str(root / "prepared.json"), "mapping": str(root / "selected_mapping.json"),
                            "summary": str(root / "summary.json"), "trial_manifest": str(root / "trial_manifest.csv")}}
    write_json(output / "analysis.json", evidence)
    device = state["identity"]["config"]["auto_ecmad"]["training"]["device"]
    training_seed = state["identity"]["config"]["auto_ecmad"].get("random_state", 42)
    direction = "提高" if deltas["auprc"] > 0 else "降低" if deltas["auprc"] < 0 else "相同"
    lead = (f"测试集 selected 相对 Raw 的 AUPRC {direction} {abs(deltas['auprc']):.6f}，"
            f"AUROC 变化 {deltas['auroc']:+.6f}，F1 变化 {deltas['f1']:+.6f}。"
            "这些是冻结模型在同一被试留出测试集上的比较。")
    provenance = (f"已核对 {len(candidates)} 个有方向候选及全部保存的模型校验和、参考排名、标签和 trial 对齐。"
                  f"输入为 {len(state['channels'])} 通道，阶段一每个候选仅输入 coherent/specific 两路；"
                  f"阶段二为 {2*len(state['channels'])} 通道。设备配置 {device}，种子 {training_seed}，"
                  f"run fingerprint: {state['fingerprint']}。")
    data_note = (f"本地样本发现记录来自 {source_subjects} 个被试，最终纳入 {manifest.subject_id.nunique()} 名、"
                 f"{len(manifest)} 个试次。未通过 Raw 资格筛选的记录："
                 + ("；".join(f"{row['subject_id']}: {row['reason']}" for row in excluded) or "无")
                 + "。这不是对 ERP-CORE 完整发布样本的声明；1=错误反应，0=正确反应。")
    search_note = (f"每目标穷举全部参考后独立选优。候选实际训练 epochs 中位数为 {candidates.epochs_run.median():.1f}，"
                   f"范围 {candidates.epochs_run.min()}–{candidates.epochs_run.max()}，"
                   f"达到上限的候选有 {capped} 个。被选作参考最多的通道为 "
                   + "、".join(f"{channel} ({count})" for channel, count in counts.most_common(5))
                   + "。参考频率及最优/次优损失差描述当前划分的选择，不代表跨种子稳定性或物理伪迹来源。")
    uncertainty = (f"以测试被试为成对重采样单元，固定模型和验证集阈值，进行 {repetitions} 次 cluster bootstrap。"
                   "区间反映当前测试样本的不确定性，不包括重新划分、重新训练或重新选择参考造成的变化；"
                   "测试被试较少时区间本身也不稳定。" if len(intervals) else
                   "测试被试少于两名，未计算有意义的被试重采样区间；当前仅可作流程验证。")
    limits = (f"selected/Raw 参数量分别为 {parameters['selected']}/{parameters['raw']}；输入通道数和表示均有变化，"
              "不能将差异全部归因于去噪。两路分量在归一化前之和等于 Raw，分类提升也可能来自表示重构。"
              "第一阶段独立选择不保证组合联合最优；大量候选可能适应内部留出集。"
              "本次只有单种子、单划分，分类指标不证明生理成分被保留或伪迹被正确去除。"
              "类别加权 BCE 还会影响概率校准，应结合 Brier 和校准图解释预测概率。")
    errors = f"按各自冻结阈值，selected 修正了 {fixes} 个 Raw 的错误，同时新增 {regressions} 个错误。阈值均来自外部验证，不在测试集重新优化。"
    sections = [("结论", lead, None), ("来源与完整性", provenance, None), ("样本与划分", data_note, split_table),
                ("模型表现", "同时列出外部验证与最终测试；AUPRC 应结合对应错误反应比例解读。", metrics),
                ("不确定性", uncertainty, intervals), ("参考选择", search_note, selection_table),
                ("测试错误变化", errors, None), ("解释边界", limits, None)]
    markdown = ["# 自动 ECMAD 完整训练结果分析", ""]
    html_sections = []
    for title, text, table in sections:
        markdown += [f"## {title}", "", text, ""]
        table_html = ""
        if table is not None and len(table):
            markdown += [markdown_table(table), ""]
            table_html = table.to_html(index=False, float_format=lambda x: f"{x:.6f}", escape=True, classes="data")
        html_sections.append(f"<section><h2>{html.escape(title)}</h2><p>{html.escape(text)}</p><div class='scroll'>{table_html}</div></section>")
    markdown += ["## 图表", ""]
    for path in images:
        markdown += [f"![{path.stem}](figures/{path.name})", ""]
    (output / "report.md").write_text("\n".join(markdown), encoding="utf-8")
    embedded = "".join(f"<figure><img alt='{p.stem}' src='data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}'><figcaption>{p.stem}</figcaption></figure>" for p in images)
    document = ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
                "<title>自动 ECMAD 训练结果分析</title><style>body{margin:0;background:#f5f4ef;color:#203037;font:16px/1.7 system-ui,sans-serif}"
                "main{max-width:1150px;margin:auto;padding:40px 24px}h1{font-size:32px}h2{font-size:23px}section,figure{background:white;padding:24px;margin:24px 0;border:1px solid #dde4df;border-radius:8px}"
                ".scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;font-size:13px;width:100%}th,td{padding:9px 12px;text-align:right;border-bottom:1px solid #e6ece7}th{background:#f0f5f2}"
                "img{width:100%;height:auto}figcaption{color:#64746d;font-size:13px}p{overflow-wrap:anywhere}</style><main><h1>自动 ECMAD 完整训练结果分析</h1>"
                + "".join(html_sections) + embedded + "</main></html>")
    (output / "report.html").write_text(document, encoding="utf-8")
    print(json.dumps({"report": str(output / "report.md"), "html": str(output / "report.html"),
                      "test_deltas_selected_minus_raw": deltas}, ensure_ascii=False, indent=2))
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_repetitions < 1:
        parser.error("Bootstrap repetitions must be positive")
    root = args.results_dir.expanduser().resolve()
    analyze(root, args.output_dir or root / "analysis", args.bootstrap_repetitions, args.seed)


if __name__ == "__main__":
    main()
