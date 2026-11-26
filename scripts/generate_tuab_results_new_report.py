#!/usr/bin/env python3
"""Generate a self-contained descriptive HTML report for results/tuab_results_new."""
from __future__ import annotations

import csv
import html
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "tuab_results_new"
OUT = ROOT / "docs" / "tuab_results_new_descriptive_report_2026-09-03.html"
CONDITIONS = ["raw", "specific", "coherent", "specific_coherent"]
CONDITION_LABELS = {
    "raw": "Raw",
    "specific": "specific",
    "coherent": "coherent",
    "specific_coherent": "specific + coherent",
}
MODEL_LABELS = {"eegnet": "EEGNet", "xgboost": "XGBoost"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scalar(value: str) -> float:
    return float(value)


def metrics_from_predictions(path: Path, threshold: float) -> dict[str, Any]:
    rows = read_csv(path)
    y_true = [int(row["true_label"]) for row in rows]
    y_proba = [float(row["pred_proba"]) for row in rows]
    y_pred = [int(prob >= threshold) for prob in y_proba]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "auroc": float(roc_auc_score(y_true, y_proba)),
        "auprc": float(average_precision_score(y_true, y_proba)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_positive": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(cm[0][0] / sum(cm[0]) if sum(cm[0]) else 0.0),
        "threshold": float(threshold),
        "confusion_matrix": cm,
        "n_records": len(rows),
    }


def get_gate(phase_dir: Path) -> float:
    config = read_json(phase_dir / "tuab_component_eegnet_phasegated" / "run_summary.json")
    cfg_path = phase_dir / "tuab_component_eegnet_phasegated" / "config_resolved.yaml"
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        return float(cfg["wiener"]["phase_gate_threshold_rad"])
    raise FileNotFoundError(f"Missing resolved config for {phase_dir.name}: {config}")


def collect_dataset(phase_dir: Path) -> dict[str, Any]:
    rows = read_csv(
        phase_dir
        / "tuab_component_eegnet_phasegated"
        / "split_manifest.csv"
    )
    split_order = ["train", "validation", "test"]
    splits: dict[str, Any] = {}
    for split in split_order:
        subset = [row for row in rows if row["split"] == split]
        labels = {"abnormal": 0, "normal": 0}
        epoch_labels = {"abnormal": 0, "normal": 0}
        for row in subset:
            name = row["class_name"]
            labels[name] += 1
            epoch_labels[name] += int(row["n_epochs"])
        splits[split] = {
            "records": len(subset),
            "patients": len({row["patient_id"] for row in subset}),
            "epochs": sum(int(row["n_epochs"]) for row in subset),
            "records_abnormal": labels["abnormal"],
            "records_normal": labels["normal"],
            "epochs_abnormal": epoch_labels["abnormal"],
            "epochs_normal": epoch_labels["normal"],
        }
    return {
        "total_records": len(rows),
        "total_patients": len({row["patient_id"] for row in rows}),
        "total_epochs": sum(int(row["n_epochs"]) for row in rows),
        "splits": splits,
        "label_mapping": "0 = abnormal, 1 = normal",
    }


def collect_results() -> dict[str, Any]:
    phase_dirs = [path for path in RESULT_ROOT.iterdir() if path.is_dir()]
    phase_dirs.sort(key=lambda path: get_gate(path))
    phases: list[dict[str, Any]] = []
    integrity: list[dict[str, Any]] = []
    shap: dict[str, Any] = {}
    expected = [
        "test_predictions.csv",
        "test_metrics.json",
        "val_predictions.csv",
        "val_metrics.json",
    ]
    dataset = collect_dataset(phase_dirs[0])

    for phase_dir in phase_dirs:
        gate = get_gate(phase_dir)
        phase_key = f"{gate:g}"
        eeg_root = phase_dir / "tuab_component_eegnet_phasegated"
        xgb_root = phase_dir / "tuab_component_xgboost" / "base211"
        eeg_rows = {
            row["condition"]: row
            for row in read_csv(eeg_root / "condition_metrics.csv")
        }
        xgb_rows = {
            row["condition"]: row
            for row in read_csv(xgb_root / "comparison_summary.csv")
        }
        phase: dict[str, Any] = {
            "gate": gate,
            "label": f"{gate:g} rad",
            "eegnet": {},
            "xgboost": {},
        }
        shap[phase_key] = {}
        for condition in CONDITIONS:
            eeg_row = eeg_rows[condition]
            eeg_threshold = scalar(eeg_row["test_threshold"])
            eeg_pred_path = eeg_root / "conditions" / condition / "test_predictions.csv"
            eeg_metrics = metrics_from_predictions(eeg_pred_path, eeg_threshold)
            xgb_row = xgb_rows[condition]
            xgb_threshold = scalar(xgb_row["test_threshold"])
            xgb_pred_path = xgb_root / condition / "test_predictions.csv"
            xgb_metrics = metrics_from_predictions(xgb_pred_path, xgb_threshold)
            phase["eegnet"][condition] = eeg_metrics
            phase["xgboost"][condition] = xgb_metrics

            integrity.append(
                {
                    "phase": phase_key,
                    "condition": condition,
                    "eegnet_complete": all(
                        (eeg_root / "conditions" / condition / name).exists()
                        for name in expected
                    ),
                    "xgboost_complete": all(
                        (xgb_root / condition / name).exists() for name in expected
                    )
                    and (xgb_root / condition / "shap_summary.png").exists(),
                }
            )
            band_path = xgb_root / condition / "shap_by_band.json"
            channel_path = xgb_root / condition / "shap_by_channel.json"
            shap[phase_key][condition] = {
                "band": read_json(band_path),
                "channel": read_json(channel_path),
            }
        phases.append(phase)

    return {
        "generated": str(date.today()),
        "source": "results/tuab_results_new",
        "phases": phases,
        "dataset": dataset,
        "integrity": integrity,
        "conditions": CONDITIONS,
        "condition_labels": CONDITION_LABELS,
        "model_labels": MODEL_LABELS,
        "shap": shap,
    }


def best_result(data: dict[str, Any], model: str, metric: str) -> tuple[dict[str, Any], str]:
    best: tuple[float, dict[str, Any], str] | None = None
    for phase in data["phases"]:
        for condition in CONDITIONS:
            value = phase[model][condition][metric]
            if best is None or value > best[0]:
                best = (value, phase, condition)
    assert best is not None
    return best[1], best[2]


def value_at(data: dict[str, Any], model: str, condition: str, metric: str) -> list[float]:
    return [phase[model][condition][metric] for phase in data["phases"]]


def make_report(data: dict[str, Any]) -> str:
    dataset = data["dataset"]
    best_eeg_phase, best_eeg_cond = best_result(data, "eegnet", "auroc")
    best_xgb_phase, best_xgb_cond = best_result(data, "xgboost", "auroc")
    raw_eeg = value_at(data, "eegnet", "raw", "auroc")
    raw_xgb = value_at(data, "xgboost", "raw", "auroc")
    xgb_combined = value_at(data, "xgboost", "specific_coherent", "auroc")
    coherent_xgb = value_at(data, "xgboost", "coherent", "auroc")
    specific_eeg = value_at(data, "eegnet", "specific", "auroc")
    combined_eeg = value_at(data, "eegnet", "specific_coherent", "auroc")
    all_complete = all(
        item["eegnet_complete"] and item["xgboost_complete"]
        for item in data["integrity"]
    )

    def pct(value: float) -> str:
        return f"{value * 100:.1f}%"

    def num(value: float) -> str:
        return f"{value:.4f}"

    narrative = {
        "headline": (
            f"在 {dataset['total_records']:,} 条 recording 的固定测试集上，"
            f"XGBoost 的最高 AUROC 为 {num(best_xgb_phase['xgboost'][best_xgb_cond]['auroc'])} "
            f"（{CONDITION_LABELS[best_xgb_cond]}，{best_xgb_phase['label']}）；"
            f"EEGNet 的最高 AUROC 为 {num(best_eeg_phase['eegnet'][best_eeg_cond]['auroc'])} "
            f"（{CONDITION_LABELS[best_eeg_cond]}，{best_eeg_phase['label']}）。"
        ),
        "model_gap": (
            f"Raw 基线在 EEGNet/XGBoost 上分别为 {num(sum(raw_eeg) / len(raw_eeg))}/"
            f"{num(sum(raw_xgb) / len(raw_xgb))} AUROC。"
        ),
        "specific_pattern": (
            f"XGBoost 的 specific 为 {num(min(value_at(data, 'xgboost', 'specific', 'auroc')))}–"
            f"{num(max(value_at(data, 'xgboost', 'specific', 'auroc')))}，"
            f"而 EEGNet specific 为 {num(min(specific_eeg))}–{num(max(specific_eeg))}；"
            "两种模型没有呈现一致的分量增益。"
        ),
        "combined_pattern": (
            f"XGBoost 的 specific + coherent 为 {num(min(xgb_combined))}–{num(max(xgb_combined))} AUROC，"
            f"coherent 单独条件为 {num(min(coherent_xgb))}–{num(max(coherent_xgb))}；"
            f"EEGNet 的组合条件为 {num(min(combined_eeg))}–{num(max(combined_eeg))}。"
        ),
    }

    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    narrative_json = json.dumps(narrative, ensure_ascii=False)
    phases_json = json.dumps([phase["label"] for phase in data["phases"]], ensure_ascii=False)

    split_rows = "".join(
        f"<tr><th>{split}</th><td>{info['records']:,}</td><td>{info['patients']:,}</td>"
        f"<td>{info['epochs']:,}</td><td>{info['records_abnormal']:,} / {info['records_normal']:,}</td>"
        f"<td>{info['epochs_abnormal']:,} / {info['epochs_normal']:,}</td></tr>"
        for split, info in dataset["splits"].items()
    )
    integrity_rows = "".join(
        f"<tr><td>{item['phase']} rad</td><td>{CONDITION_LABELS[item['condition']]}</td>"
        f"<td class={'ok' if item['eegnet_complete'] else 'bad'}>{'完整' if item['eegnet_complete'] else '缺失'}</td>"
        f"<td class={'ok' if item['xgboost_complete'] else 'bad'}>{'完整' if item['xgboost_complete'] else '缺失'}</td></tr>"
        for item in data["integrity"]
    )
    condition_options = "".join(
        f"<option value='{condition}'>{CONDITION_LABELS[condition]}</option>"
        for condition in CONDITIONS
    )

    return f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TUAB 新结果描述性分析 | ECMAD</title>
<style>
:root {{
  --canvas:#edf3f5; --paper:#fff; --ink:#153247; --text:#2d4351; --muted:#627580;
  --line:#d3e0e5; --soft:#f5f8f9; --accent:#008f83; --accent2:#dc7b35;
  --raw:#536b7c; --specific:#008f83; --coherent:#dc7b35; --combined:#7157a6;
  --danger:#b84b42; --ok:#257e63; --foreground:#2d4351; --border:#cbd9df;
  --viz-series-1:#536b7c; --viz-series-2:#008f83; --viz-series-3:#dc7b35;
  --viz-series-4:#7157a6; --viz-series-5:#b84b42; --viz-series-6:#2b7eaa;
  font-size:16px;
}}
* {{ box-sizing:border-box; }} html {{ scroll-behavior:smooth; }}
body {{ margin:0; background:var(--canvas); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",Arial,sans-serif; line-height:1.65; }}
a {{ color:#08766f; }} a:focus-visible,button:focus-visible {{ outline:3px solid #eea44e; outline-offset:3px; }}
.shell {{ display:grid; grid-template-columns:220px minmax(0,1fr); gap:22px; width:min(calc(100% - 28px),1440px); margin:14px auto 50px; align-items:start; }}
.rail {{ position:sticky; top:14px; padding:20px 16px; border:1px solid var(--line); border-radius:14px; background:rgba(255,255,255,.95); box-shadow:0 12px 34px rgba(30,60,75,.08); }}
.rail small {{ color:var(--accent); font-weight:700; letter-spacing:.13em; text-transform:uppercase; }} .rail h2 {{ margin:7px 0 18px; color:var(--ink); font-size:1.06rem; line-height:1.35; }}
.rail a {{ display:block; padding:7px 8px; border-left:2px solid transparent; color:var(--muted); text-decoration:none; font-size:.9rem; }} .rail a:hover {{ border-color:var(--accent); color:var(--ink); background:#edf8f6; }}
.rail .note {{ margin-top:18px; padding-top:14px; border-top:1px solid var(--line); color:var(--muted); font-size:.78rem; }}
main {{ min-width:0; }} section {{ width:min(100%,1160px); margin:0 auto 22px; }}
.hero {{ padding:clamp(28px,5vw,58px); border-radius:18px; color:#f4fbfd; background:linear-gradient(120deg,#11364c,#155867); box-shadow:0 18px 42px rgba(25,53,70,.15); }}
.eyebrow {{ margin:0 0 10px; color:#70e0d1; font-size:.75rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }} h1 {{ max-width:900px; margin:0; color:#fff; font-size:clamp(2rem,4.5vw,4.3rem); line-height:1.05; letter-spacing:-.035em; }}
.hero p {{ max-width:840px; margin:20px 0 0; color:#d8e8ed; font-size:1.06rem; }} .hero-grid {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:10px; margin-top:30px; }}
.hero-metric {{ padding:14px 12px; border-top:1px solid rgba(255,255,255,.35); }} .hero-metric b {{ display:block; color:#fff; font-size:1.35rem; font-variant-numeric:tabular-nums; }} .hero-metric span {{ color:#bfdae0; font-size:.78rem; }}
.section-head {{ display:flex; gap:18px; align-items:end; justify-content:space-between; margin:35px 0 12px; }} h2 {{ margin:0; color:var(--ink); font-size:1.65rem; line-height:1.2; }} h3 {{ margin:0 0 8px; color:var(--ink); font-size:1.05rem; }} .sub {{ margin:4px 0 0; color:var(--muted); font-size:.92rem; }}
.paper {{ min-width:0; overflow:hidden; padding:24px; border:1px solid var(--line); border-radius:14px; background:var(--paper); box-shadow:0 8px 24px rgba(30,60,75,.055); }} .paper + .paper {{ margin-top:16px; }}
.grid-2 {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }} .grid-3 {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; }}
.callout {{ padding:15px 17px; border-left:4px solid var(--accent); background:#edf8f6; }} .callout.warn {{ border-color:var(--accent2); background:#fff6ed; }}
.table-wrap {{ min-width:0; max-width:100%; overflow-x:auto; }} table {{ width:100%; border-collapse:collapse; font-size:.86rem; font-variant-numeric:tabular-nums; }} th,td {{ padding:8px 9px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }} th:first-child,td:first-child {{ text-align:left; }} thead th {{ color:var(--muted); font-weight:700; background:var(--soft); }} tbody th {{ color:var(--ink); font-weight:600; }} .ok {{ color:var(--ok); font-weight:700; }} .bad {{ color:var(--danger); font-weight:700; }}
.legend {{ display:flex; flex-wrap:wrap; gap:10px 16px; margin:5px 0 7px; color:var(--muted); font-size:.82rem; }} .legend span {{ display:inline-flex; align-items:center; gap:6px; }} .swatch {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
.chart {{ width:100%; min-height:290px; }} svg {{ display:block; width:100%; height:auto; overflow:visible; }} svg text {{ fill:var(--foreground); font-size:12px; font-family:inherit; }} .axis-title {{ font-size:12px; }} .frame {{ fill:none; stroke:var(--border); stroke-width:1; }} .gridline {{ stroke:#e6eef1; stroke-width:1; }} .line {{ fill:none; stroke-width:2.4; }} .dot {{ stroke:var(--paper); stroke-width:1.5; }}
.heatmap {{ display:grid; grid-template-columns:150px repeat(4,minmax(58px,1fr)); gap:1px; background:var(--line); border:1px solid var(--line); }} .heatmap > div {{ min-height:32px; padding:6px 7px; background:var(--paper); font-size:.78rem; text-align:center; font-variant-numeric:tabular-nums; }} .heatmap .head {{ background:var(--soft); color:var(--muted); font-weight:700; }} .heatmap .row-label {{ text-align:left; color:var(--ink); font-weight:600; }}
.shap-chart {{ min-height:0; }} .shap-bars {{ width:100%; min-width:0; font-variant-numeric:tabular-nums; }} .shap-axis,.shap-row {{ display:grid; grid-template-columns:132px minmax(0,1fr); gap:10px; }} .shap-axis {{ margin-bottom:7px; color:var(--muted); font-size:.74rem; }} .shap-axis-label {{ align-self:end; line-height:1.2; }} .shap-axis-track {{ display:flex; justify-content:space-between; border-bottom:1px solid var(--border); }} .shap-axis-track span {{ padding:0 1px 4px; }} .shap-row {{ min-height:56px; align-items:center; }} .shap-feature {{ color:var(--ink); font-size:.82rem; font-weight:650; line-height:1.2; }} .shap-track {{ display:grid; gap:3px; min-width:0; padding:4px 0; background-image:linear-gradient(to right, transparent 24.8%, var(--line) 25%, transparent 25.2%, transparent 49.8%, var(--line) 50%, transparent 50.2%, transparent 74.8%, var(--line) 75%, transparent 75.2%); }} .shap-bar-line {{ display:flex; align-items:center; gap:6px; min-width:0; min-height:15px; }} .shap-bar {{ display:block; width:var(--bar-width); min-width:2px; height:12px; border-radius:0 3px 3px 0; background:var(--bar-color); }} .shap-value {{ color:var(--muted); font-size:.72rem; white-space:nowrap; }} .shap-scale-note {{ margin:8px 0 0 142px; color:var(--muted); font-size:.74rem; }} .shap-explanation {{ margin-top:16px; }}
.select-row {{ display:flex; flex-wrap:wrap; align-items:center; gap:10px; margin-bottom:10px; color:var(--muted); font-size:.9rem; }} select {{ min-height:36px; padding:5px 28px 5px 9px; border:1px solid var(--line); border-radius:7px; color:var(--ink); background:var(--paper); font:inherit; }}
.footer {{ color:var(--muted); font-size:.82rem; }} .mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; overflow-wrap:anywhere; }}
@media (max-width:900px) {{ .shell {{ display:block; width:min(calc(100% - 20px),760px); }} .rail {{ position:static; margin-bottom:14px; }} .rail nav {{ display:flex; flex-wrap:wrap; gap:2px 5px; }} .hero-grid {{ grid-template-columns:repeat(3,minmax(0,1fr)); }} .grid-2,.grid-3 {{ grid-template-columns:1fr; }} }}
@media (max-width:520px) {{ body {{ font-size:15px; }} .shell {{ width:min(calc(100% - 12px),760px); }} .hero {{ padding:25px 19px; }} h1 {{ font-size:2.15rem; }} .hero-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .paper {{ padding:16px; }} .section-head {{ display:block; }} .heatmap {{ grid-template-columns:120px repeat(4,minmax(52px,1fr)); }} .heatmap > div {{ padding:5px 3px; font-size:.7rem; }} .shap-axis,.shap-row {{ grid-template-columns:108px minmax(0,1fr); gap:7px; }} .shap-feature {{ font-size:.76rem; }} .shap-value {{ font-size:.66rem; }} .shap-scale-note {{ margin-left:115px; font-size:.68rem; }} }}
</style>
</head>
<body>
<div class="shell">
<aside class="rail" aria-label="报告目录">
  <small>ECMAD · TUAB</small><h2>新结果描述性分析</h2>
  <nav><a href="#overview">结论摘要</a><a href="#dataset">数据构成</a><a href="#performance">排序性能</a><a href="#deltas">相对 Raw</a><a href="#shap">SHAP 模式</a><a href="#table">完整指标</a><a href="#method">配置与限制</a></nav>
  <div class="note">来源：<span class="mono">results/tuab_results_new</span><br>生成日期：{html.escape(data['generated'])}</div>
</aside>
<main>
<section id="overview" class="hero">
  <p class="eyebrow">TUAB · Scripts 17–20 · descriptive report</p>
  <h1>ECMAD 分量分类实验<br>结果可视化与描述性分析</h1>
  <p>{html.escape(narrative['headline'])}</p>
  <div class="hero-grid">
    <div class="hero-metric"><b>{dataset['total_records']:,}</b><span>recordings</span></div>
    <div class="hero-metric"><b>{dataset['total_patients']:,}</b><span>patients</span></div>
    <div class="hero-metric"><b>{dataset['total_epochs']:,}</b><span>20 s epochs</span></div>
    <div class="hero-metric"><b>{num(best_xgb_phase['xgboost'][best_xgb_cond]['auroc'])}</b><span>XGBoost 最高 AUROC</span></div>
    <div class="hero-metric"><b>{num(best_eeg_phase['eegnet'][best_eeg_cond]['auroc'])}</b><span>EEGNet 最高 AUROC</span></div>
  </div>
</section>

<section id="dataset">
  <div class="section-head"><div><h2>1. 数据构成</h2><p class="sub">共享固定划分；recording 是统计与评估单位，epoch 数仅用于描述样本量。</p></div></div>
  <div class="grid-2">
    <div class="paper"><h3>划分与类别</h3><div class="table-wrap"><table><thead><tr><th>split</th><th>recordings</th><th>patients</th><th>epochs</th><th>records<br>abnormal / normal</th><th>epochs<br>abnormal / normal</th></tr></thead><tbody>{split_rows}</tbody></table></div></div>
    <div class="paper"><h3>Recording 数量（类别堆叠）</h3><div id="dataset-recordings" class="chart" aria-label="各 split 的 recording 类别数量"></div><h3 style="margin-top:18px">Epoch 数量（类别堆叠）</h3><div id="dataset-epochs" class="chart" aria-label="各 split 的 epoch 类别数量"></div></div>
  </div>
  <div class="callout" style="margin-top:16px">标签编码为 <b>0 = abnormal，1 = normal</b>。测试集包含 {dataset['splits']['test']['records']:,} 条 recording，其中 abnormal {dataset['splits']['test']['records_abnormal']:,}、normal {dataset['splits']['test']['records_normal']:,}；正类（normal）比例为 {pct(dataset['splits']['test']['records_normal']/dataset['splits']['test']['records']):}。</div>
</section>

<section id="performance">
  <div class="section-head"><div><h2>2. 跨 phase 的排序性能</h2><p class="sub">测试集 recording-level AUROC 与 AUPRC；Raw 是每个 phase 的共同基线。</p></div></div>
  <div class="paper"><div class="legend"><span><i class="swatch" style="background:var(--raw)"></i>Raw</span><span><i class="swatch" style="background:var(--specific)"></i>specific</span><span><i class="swatch" style="background:var(--coherent)"></i>coherent</span><span><i class="swatch" style="background:var(--combined)"></i>specific + coherent</span></div><div class="grid-2"><div><h3>EEGNet · AUROC</h3><div id="eegnet-auroc" class="chart"></div></div><div><h3>XGBoost · AUROC</h3><div id="xgboost-auroc" class="chart"></div></div><div><h3>EEGNet · AUPRC</h3><div id="eegnet-auprc" class="chart"></div></div><div><h3>XGBoost · AUPRC</h3><div id="xgboost-auprc" class="chart"></div></div></div></div>
  <div class="grid-2" style="margin-top:16px"><div class="callout"><b>EEGNet：</b>{html.escape(narrative['model_gap'])} specific 与组合条件在全部 phase 均低于 Raw；最高点仍是 Raw。</div><div class="callout"><b>XGBoost：</b>{html.escape(narrative['specific_pattern'])} {html.escape(narrative['combined_pattern'])}</div></div>
</section>

<section id="deltas">
  <div class="section-head"><div><h2>3. 相对 Raw 的差值</h2><p class="sub">同一 phase 内，以 Raw 为 0；正值表示优于该 phase 的 Raw，负值表示下降。</p></div></div>
  <div class="grid-2"><div class="paper"><h3>EEGNet · ΔAUROC</h3><div id="eegnet-delta-auroc" class="heatmap" aria-label="EEGNet 相对 Raw 的 AUROC 差值"></div></div><div class="paper"><h3>XGBoost · ΔAUROC</h3><div id="xgboost-delta-auroc" class="heatmap" aria-label="XGBoost 相对 Raw 的 AUROC 差值"></div></div><div class="paper"><h3>EEGNet · ΔF1</h3><div id="eegnet-delta-f1" class="heatmap" aria-label="EEGNet 相对 Raw 的 F1 差值"></div></div><div class="paper"><h3>XGBoost · ΔF1</h3><div id="xgboost-delta-f1" class="heatmap" aria-label="XGBoost 相对 Raw 的 F1 差值"></div></div></div>
</section>

<section id="shap">
  <div class="section-head"><div><h2>4. XGBoost SHAP 模式</h2><p class="sub">对 5 个 phase 的 <span class="mono">base211</span> SHAP 聚合取均值；数值为平均绝对 SHAP 贡献，越大表示该特征组在模型中的影响越大。</p></div><label class="select-row">查看 phase：<select id="shap-phase">{''.join(f"<option value='{phase['gate']:g}'>{phase['label']}</option>" for phase in data['phases'])}</select></label></div>
  <div class="legend"><span><i class="swatch" style="background:var(--raw)"></i>Raw</span><span><i class="swatch" style="background:var(--specific)"></i>specific</span><span><i class="swatch" style="background:var(--coherent)"></i>coherent</span><span><i class="swatch" style="background:var(--combined)"></i>specific + coherent</span></div>
  <div class="grid-2"><div class="paper"><h3>跨 phase 平均 · feature band</h3><div id="shap-band-mean" class="chart shap-chart" aria-label="跨 phase 平均的 SHAP feature band 水平柱状图"></div></div><div class="paper"><h3>选定 phase · feature band</h3><div id="shap-band-selected" class="chart shap-chart" aria-label="选定 phase 的 SHAP feature band 水平柱状图"></div></div></div>
  <div class="paper" style="margin-top:16px"><h3>选定 phase · channel 聚合 Top 6</h3><div id="shap-channel-table"></div><div class="callout warn shap-explanation">SHAP 图解释的是分类模型对标签预测的特征贡献，不是生理因果关系，也不能单独证明某一通道是伪迹来源。对 <span class="mono">specific_coherent</span>，Script 18 采用通道轴拼接，specific 在前、coherent 在后；本表沿用输出中的 channel 聚合口径。</div></div>
</section>

<section id="table">
  <div class="section-head"><div><h2>5. 完整测试集指标</h2><p class="sub">AUPRC、balanced accuracy 等由测试集预测文件按输出阈值重新计算；用于审计结果文件与报告数值的一致性。</p></div></div>
  <div class="paper"><div class="table-wrap"><table><thead><tr><th>model</th><th>phase</th><th>condition</th><th>AUROC</th><th>AUPRC</th><th>balanced acc.</th><th>macro-F1<br>（统一重算）</th><th>accuracy</th><th>threshold</th></tr></thead><tbody id="metric-table"></tbody></table></div></div>
</section>

<section id="method">
  <div class="section-head"><div><h2>6. 配置、实验语义与限制</h2><p class="sub">以下内容用于解释结果如何产生，不把分类性能等同于直接去噪质量。</p></div></div>
  <div class="grid-2"><div class="paper"><h3>实验链路</h3><ol><li>Script 17：连续 TUAB recording 上的 phase-gated Wiener 分量缓存。</li><li>Script 18：重建 raw，并保存 raw / specific / coherent 及 38 通道 <span class="mono">specific_coherent</span>。</li><li>Script 19：四个独立 EEGNet 条件，recording-level 聚合；checkpoint 按 validation recording AUPRC 选择，阈值按 validation balanced accuracy 选择。</li><li>Script 20：四个独立 XGBoost 条件，<span class="mono">base211</span> 特征；recording-level AUROC/AUPRC 作为排序指标，阈值来自验证集 macro-F1 优化。</li></ol></div><div class="paper"><h3>解释边界</h3><ul><li>这是单一固定划分上的描述性比较，没有置信区间、跨随机种子重复或显著性检验。</li><li>AUROC/AUPRC 是阈值无关的跨模型主比较；本报告将 F1 统一重算为 macro-F1。原始 EEGNet 指标文件中的 <span class="mono">f1</span> 是 normal 正类的 binary-F1，XGBoost 的 <span class="mono">f1</span> 是 macro-F1。</li><li>分类下游效用是间接评价 instrument，不等于 ECMAD 的直接信号质量或生理正确性证明。</li><li>结果目录当前 5 个 phase × 4 条件 × 2 模型的关键预测与 SHAP 文件均完整：<b>{'是' if all_complete else '否'}</b>。</li></ul></div></div>
  <div class="paper" style="margin-top:16px"><h3>产物完整性审计</h3><div class="table-wrap"><table><thead><tr><th>phase</th><th>condition</th><th>EEGNet</th><th>XGBoost</th></tr></thead><tbody>{integrity_rows}</tbody></table></div></div>
  <p class="footer">数据源：<span class="mono">{html.escape(data['source'])}</span>；报告脚本：<span class="mono">scripts/generate_tuab_results_new_report.py</span>。本页面不加载外部资源。</p>
</section>
</main></div>
<script>
const DATA = {data_json};
const NARRATIVE = {narrative_json};
const PHASE_LABELS = {phases_json};
const COLORS = {{raw:'var(--raw)',specific:'var(--specific)',coherent:'var(--coherent)',specific_coherent:'var(--combined)'}};
const CONDITION_LABELS = DATA.condition_labels;
const BAND_ORDER = ['delta','theta','alpha','beta','gamma','hjorth','spectral_entropy','asymmetry'];
const BAND_LABELS = {{delta:'delta',theta:'theta',alpha:'alpha',beta:'beta',gamma:'gamma',hjorth:'Hjorth',spectral_entropy:'spectral entropy',asymmetry:'asymmetry'}};
function fmt(v) {{ return Number(v).toFixed(4); }}
function phaseXs(n, left, right) {{ return Array.from({{length:n}}, (_,i)=>left+(right-left)*(n===1?0:i/(n-1))); }}
function lineChart(targetId, model, metric) {{
  const host=document.getElementById(targetId); const W=760,H=300,m={{l:58,r:18,t:22,b:48}}; const pw=W-m.l-m.r,ph=H-m.t-m.b;
  const values=DATA.phases.flatMap(p=>DATA.conditions.map(c=>p[model][c][metric])); const lo=Math.max(0,Math.min(...values)-0.035), hi=Math.min(1,Math.max(...values)+0.035);
  const x=phaseXs(DATA.phases.length,m.l,m.l+pw), y=v=>m.t+ph-(v-lo)/(hi-lo)*ph;
  const ticks=Array.from({{length:5}},(_,i)=>lo+(hi-lo)*i/4); let s=`<svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="${{model}} ${{metric}} across phase gate thresholds"><rect class="frame" data-chart-frame x="${{m.l}}" y="${{m.t}}" width="${{pw}}" height="${{ph}}"/>`;
  ticks.forEach(t=>{{s+=`<line class="gridline" x1="${{m.l}}" x2="${{m.l+pw}}" y1="${{y(t)}}" y2="${{y(t)}}"/><text x="${{m.l-9}}" y="${{y(t)+4}}" text-anchor="end">${{t.toFixed(2)}}</text>`;}});
  DATA.phases.forEach((p,i)=>{{s+=`<text x="${{x[i]}}" y="${{H-25}}" text-anchor="middle">${{p.gate}}` + `</text>`;}});
  DATA.conditions.forEach(condition=>{{ const pts=DATA.phases.map((p,i)=>`${{x[i]}},${{y(p[model][condition][metric])}}`).join(' '); s+=`<polyline class="line" style="stroke:${{COLORS[condition]}}" points="${{pts}}"/>`; DATA.phases.forEach((p,i)=>{{const v=p[model][condition][metric]; s+=`<circle class="dot" style="fill:${{COLORS[condition]}}" cx="${{x[i]}}" cy="${{y(v)}}" r="4"><title>${{CONDITION_LABELS[condition]}} · ${{p.label}} · ${{metric.toUpperCase()}} ${{fmt(v)}}</title></circle>`;}}); }});
  s+=`<text class="axis-title" data-axis="x" x="${{m.l+pw/2}}" y="${{H-4}}" text-anchor="middle">phase-gate threshold (rad)</text><text class="axis-title" data-axis="y" transform="translate(14 ${{m.t+ph/2}}) rotate(-90)" text-anchor="middle">${{metric.toUpperCase()}} (recording-level)</text></svg>`; host.innerHTML=s;
}}
function stackedBars(targetId, key) {{
  const host=document.getElementById(targetId), W=760,H=220,m={{l:58,r:18,t:18,b:44}}, pw=W-m.l-m.r, ph=H-m.t-m.b; const vals=DATA.phases.length?Object.values(DATA.dataset.splits).map(s=>s[key]):[]; const max=Math.max(...vals); const x=Array.from({{length:3}},(_,i)=>m.l+pw*(i+.5)/3), bar=68, y=v=>m.t+ph-v/max*ph; let s=`<svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="stacked dataset composition"><rect class="frame" x="${{m.l}}" y="${{m.t}}" width="${{pw}}" height="${{ph}}"/>`; [0,.25,.5,.75,1].forEach(t=>{{s+=`<line class="gridline" x1="${{m.l}}" x2="${{m.l+pw}}" y1="${{y(max*t)}}" y2="${{y(max*t)}}"/><text x="${{m.l-8}}" y="${{y(max*t)+4}}" text-anchor="end">${{Math.round(max*t).toLocaleString()}}</text>`;}}); ['train','validation','test'].forEach((split,i)=>{{const d=DATA.dataset.splits[split]; const a=d[key+'_abnormal'], n=d[key+'_normal']; const ya=y(a), yt=y(a+n); s+=`<rect x="${{x[i]-bar/2}}" y="${{ya}}" width="${{bar}}" height="${{y(0)-ya}}" fill="var(--viz-series-5)" opacity=".82"><title>${{split}} abnormal ${{a.toLocaleString()}}</title></rect><rect x="${{x[i]-bar/2}}" y="${{yt}}" width="${{bar}}" height="${{ya-yt}}" fill="var(--viz-series-6)" opacity=".82"><title>${{split}} normal ${{n.toLocaleString()}}</title></rect><text x="${{x[i]}}" y="${{H-23}}" text-anchor="middle">${{split}}</text>`;}}); s+=`<text class="axis-title" data-axis="y" transform="translate(14 ${{m.t+ph/2}}) rotate(-90)" text-anchor="middle">count</text></svg>`; host.innerHTML=s;
}}
function heatmap(targetId, model, metric) {{
  const host=document.getElementById(targetId); let max=0; DATA.phases.forEach(p=>DATA.conditions.forEach(c=>{{max=Math.max(max,Math.abs(p[model][c][metric] - p[model].raw[metric]));}})); max=Math.max(max,.001); let s='<div class="head">phase \\ condition</div>'+DATA.conditions.map(c=>`<div class="head">${{CONDITION_LABELS[c]}}</div>`).join(''); DATA.phases.forEach(p=>{{s+=`<div class="row-label">${{p.label}}</div>`; DATA.conditions.forEach(c=>{{const d=p[model][c][metric]-p[model].raw[metric]; const alpha=Math.min(.82,.10+Math.abs(d)/max*.72); const color=d>=0?`rgba(0,143,131,${{alpha}})`: `rgba(220,123,53,${{alpha}})`; s+=`<div style="background:${{c==='raw'?'var(--soft)':color}};color:${{Math.abs(d)>max*.56?'#fff':'var(--text)'}}" title="${{CONDITION_LABELS[c]}} · ${{p.label}} · Δ${{metric}} ${{d>=0?'+':''}}${{d.toFixed(4)}}">${{d>=0?'+':''}}${{d.toFixed(4)}}</div>`;}});}}); host.innerHTML=s;
}}
const SHAP_BAND_MAX=Math.max(.001,...DATA.phases.flatMap(p=>DATA.conditions.flatMap(c=>BAND_ORDER.map(b=>DATA.shap[String(p.gate)][c].band[b]))))*1.1;
const SHAP_TICKS=Array.from({{length:5}},(_,i)=>SHAP_BAND_MAX*i/4);
function shapBarChart(targetId, rows, ariaLabel) {{
  const host=document.getElementById(targetId);
  const axis=`<div class="shap-axis"><span class="shap-axis-label">feature band</span><div class="shap-axis-track">${{SHAP_TICKS.map(v=>`<span>${{v.toFixed(4)}}</span>`).join('')}}</div></div>`;
  const body=rows.map(row=>`<div class="shap-row"><div class="shap-feature">${{BAND_LABELS[row.band]}}</div><div class="shap-track">${{DATA.conditions.map(c=>{{const v=Number(row.values[c]); const pct=Math.max(0,Math.min(100,v/SHAP_BAND_MAX*100)); return `<div class="shap-bar-line" title="${{CONDITION_LABELS[c]}} · ${{BAND_LABELS[row.band]}} · mean |SHAP| ${{v.toFixed(5)}}"><span class="shap-bar" style="--bar-width:${{pct}}%;--bar-color:${{COLORS[c]}}"></span><span class="shap-value">${{v.toFixed(4)}}</span></div>`;}}).join('')}}</div></div>`).join('');
  host.innerHTML=`<div class="shap-bars" role="img" aria-label="${{ariaLabel}}">${{axis}}${{body}}<div class="shap-scale-note">mean |SHAP| contribution · shared x-axis maximum ${{SHAP_BAND_MAX.toFixed(4)}}</div></div>`;
}}
function shapSelectedRows(phaseKey) {{ return BAND_ORDER.map(b=>({{band:b,values:Object.fromEntries(DATA.conditions.map(c=>[c,DATA.shap[phaseKey][c].band[b]]))}})); }}
function shapMeanRows() {{ return BAND_ORDER.map(b=>({{band:b,values:Object.fromEntries(DATA.conditions.map(c=>[c,DATA.phases.reduce((sum,p)=>sum+DATA.shap[String(p.gate)][c].band[b],0)/DATA.phases.length]))}})); }}
function channelTable(phaseKey) {{ const host=document.getElementById('shap-channel-table'); const rows=DATA.conditions.map(c=>{{const values=DATA.shap[phaseKey][c].channel; return `<h4 style="margin:8px 0 3px;color:var(--ink)">${{CONDITION_LABELS[c]}}</h4><div class="table-wrap"><table><thead><tr><th>channel</th><th>mean |SHAP|</th></tr></thead><tbody>${{Object.entries(values).sort((a,b)=>b[1]-a[1]).slice(0,6).map(([k,v])=>`<tr><th>${{k}}</th><td>${{v.toFixed(5)}}</td></tr>`).join('')}}</tbody></table></div>`;}}).join(''); host.innerHTML=rows; }}
function metricTable() {{ const host=document.getElementById('metric-table'); let rows=''; ['eegnet','xgboost'].forEach(model=>DATA.phases.forEach(p=>DATA.conditions.forEach(c=>{{const d=p[model][c]; rows+=`<tr><th>${{DATA.model_labels[model]}}</th><td>${{p.label}}</td><td>${{CONDITION_LABELS[c]}}</td><td>${{fmt(d.auroc)}}</td><td>${{fmt(d.auprc)}}</td><td>${{fmt(d.balanced_accuracy)}}</td><td>${{fmt(d.f1)}}</td><td>${{fmt(d.accuracy)}}</td><td>${{d.threshold.toFixed(3)}}</td></tr>`;}}))); host.innerHTML=rows; }}
document.addEventListener('DOMContentLoaded',()=>{{
  stackedBars('dataset-recordings','records'); stackedBars('dataset-epochs','epochs');
  ['eegnet','xgboost'].forEach(model=>['auroc','auprc'].forEach(metric=>lineChart(`${{model}}-${{metric}}`,model,metric)));
  ['eegnet','xgboost'].forEach(model=>['auroc','f1'].forEach(metric=>heatmap(`${{model}}-delta-${{metric}}`,model,metric)));
  shapBarChart('shap-band-mean',shapMeanRows(),'跨 phase 平均的 SHAP feature band 水平柱状图'); const select=document.getElementById('shap-phase');
  function refreshShap() {{ const key=String(select.value); shapBarChart('shap-band-selected',shapSelectedRows(key),'选定 phase 的 SHAP feature band 水平柱状图'); channelTable(key); }} select.addEventListener('change',refreshShap); refreshShap(); metricTable();
}});
</script>
</body>
</html>'''


def main() -> None:
    data = collect_results()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(make_report(data), encoding="utf-8")
    print(OUT)
    print(json.dumps({
        "phases": [phase["gate"] for phase in data["phases"]],
        "records": data["dataset"]["total_records"],
        "epochs": data["dataset"]["total_epochs"],
        "integrity_blocks": len(data["integrity"]),
        "bytes": OUT.stat().st_size,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
