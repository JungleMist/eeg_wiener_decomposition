#!/usr/bin/env python3
"""Generate a descriptive HTML report for the TUEP phase-gated component runs."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import cm, colors


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
FIG = DOCS / "figures"
REPORT = DOCS / "tuep_component_eegnet_xgboost_phase_descriptive_report_2026-09-18.html"
PHASE_DIRS = [
    ("0.01", RESULTS / "results_tuep_phase001"),
    ("0.05", RESULTS / "results_tuep_phase005"),
    ("0.1", RESULTS / "results_tuep_phase01"),
    ("0.5", RESULTS / "results_tuep_phase05"),
    ("1", RESULTS / "results_tuep_phase1"),
]
CONDITIONS = ["raw", "specific", "coherent", "specific_coherent"]
COLORS = {"raw": "#53657a", "specific": "#008c82", "coherent": "#d97a2b", "specific_coherent": "#a25aa8"}
LABELS = {"raw": "Raw", "specific": "Specific", "coherent": "Coherent", "specific_coherent": "Specific + coherent"}


def load_data():
    rows = []
    shap_rows = []
    phase_info = []
    for phase, directory in PHASE_DIRS:
        ep = directory / "tuep_component_eegnet_phasegated"
        xg = directory / "tuep_component_xgboost" / "base211"
        cfg = yaml.safe_load((ep / "config_resolved.yaml").read_text())
        phase_info.append({
            "phase": phase,
            "rad": cfg["wiener"]["phase_gate_threshold_rad"],
            "eegnet_file": (ep / "condition_metrics.csv").exists(),
            "xgb_file": (xg / "comparison_summary.csv").exists(),
        })
        if (ep / "condition_metrics.csv").exists():
            frame = pd.read_csv(ep / "condition_metrics.csv")
            frame["model"] = "EEGNet"
            frame["phase"] = phase
            rows.append(frame)
        if (xg / "comparison_summary.csv").exists():
            frame = pd.read_csv(xg / "comparison_summary.csv")
            frame["model"] = "XGBoost"
            frame["phase"] = phase
            rows.append(frame)
        for condition in CONDITIONS:
            path = xg / condition / "shap_by_band.json"
            if path.exists():
                values = json.loads(path.read_text())
                for band, value in values.items():
                    shap_rows.append({"phase": phase, "condition": condition, "band": band, "mean_abs_shap": value})
    metrics = pd.concat(rows, ignore_index=True)
    shap = pd.DataFrame(shap_rows)
    return metrics, shap, phase_info


def save_figures(metrics: pd.DataFrame, shap: pd.DataFrame):
    FIG.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    phase_order = [p for p, _ in PHASE_DIRS]

    # Data composition / coverage.
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    counts = metrics.groupby(["phase", "model"]).size().unstack(fill_value=0).reindex([p for p, _ in PHASE_DIRS])
    counts.plot(kind="bar", ax=ax, color=["#008c82", "#d97a2b"], width=0.72)
    ax.set_title("TUEP phase sweep: reported condition coverage")
    ax.set_xlabel("Phase-gate threshold (rad)")
    ax.set_ylabel("Number of condition rows")
    ax.legend(title="Model", frameon=False)
    ax.set_ylim(0, 5)
    fig.tight_layout()
    fig.savefig(FIG / "tuep_component_phase_coverage.png", dpi=180)
    plt.close(fig)

    # Phase × condition × AUROC 3D comparison.
    # The two panels keep model identity separate while sharing one color scale.
    fig = plt.figure(figsize=(14, 6.8))
    axes = [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
    phase_x = {phase: i for i, phase in enumerate(phase_order)}
    condition_y = {condition: i for i, condition in enumerate(CONDITIONS)}
    norm = colors.Normalize(vmin=float(metrics["test_auroc"].min()), vmax=float(metrics["test_auroc"].max()))
    cmap = plt.get_cmap("YlGnBu")
    for ax, model in zip(axes, ["EEGNet", "XGBoost"]):
        sub = metrics[metrics.model == model]
        for _, row in sub.iterrows():
            height = float(row.test_auroc)
            ax.bar3d(
                phase_x[row.phase] - 0.23,
                condition_y[row.condition] - 0.23,
                0,
                0.46,
                0.46,
                height,
                color=cmap(norm(height)),
                edgecolor="#4d6474",
                linewidth=0.25,
                shade=True,
                alpha=0.95,
            )
        ax.set_title(model, pad=14)
        ax.set_xlabel("Phase (rad)", labelpad=9)
        ax.set_ylabel("Condition", labelpad=10)
        ax.set_zlabel("Test AUROC", labelpad=9)
        ax.set_xticks(range(len(phase_order)))
        ax.set_xticklabels(phase_order)
        ax.set_yticks(range(len(CONDITIONS)))
        ax.set_yticklabels([LABELS[c] for c in CONDITIONS])
        ax.set_zlim(0, 0.85)
        ax.view_init(elev=24, azim=-58)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    # Reserve a dedicated axis for the color legend so it cannot overlap the
    # XGBoost panel or its tick labels.
    colorbar_axis = fig.add_axes([0.90, 0.22, 0.018, 0.56])
    fig.colorbar(sm, cax=colorbar_axis, label="Test AUROC")
    fig.suptitle("Phase × condition comparison of patient-level test AUROC", y=0.98)
    fig.subplots_adjust(left=0.03, right=0.87, bottom=0.08, top=0.88, wspace=0.08)
    fig.savefig(FIG / "tuep_component_phase_condition_auroc_3d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Phase curves.
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5), sharey=True)
    phase_x = {phase: i for i, phase in enumerate(phase_order)}
    for ax, model in zip(axes, ["EEGNet", "XGBoost"]):
        sub = metrics[metrics.model == model]
        for condition in CONDITIONS:
            x = sub[sub.condition == condition].copy()
            x["phase_x"] = x.phase.map(phase_x)
            x = x.sort_values("phase_x")
            if not x.empty:
                ax.plot(x.phase_x, x.test_auroc, marker="o", linewidth=2, label=LABELS[condition], color=COLORS[condition])
        ax.axhline(0.5, color="#8d98a4", linewidth=1, linestyle="--")
        ax.set_title(model)
        ax.set_xlabel("Phase-gate threshold (categorical, rad)")
        ax.set_xticks(range(len(phase_order)), phase_order)
        ax.set_ylim(0.3, 0.85)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Patient-level test AUROC")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Test AUROC across phase-gate thresholds", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "tuep_component_phase_auroc_lines.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Comparable four-condition ranking at 0.05.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, model in zip(axes, ["EEGNet", "XGBoost"]):
        sub = metrics[(metrics.model == model) & (metrics.phase == "0.05")].copy()
        # The concatenated EEGNet/XGBoost frame contains model-specific columns.
        # Dropping NA values across the full row would remove every condition
        # because each model has valid values only for its own metric schema.
        sub = sub.set_index("condition").reindex(CONDITIONS)[["test_auroc"]].reset_index()
        y = np.arange(len(sub))
        ax.barh(y, sub.test_auroc, color=[COLORS[c] for c in sub.condition], alpha=0.9)
        ax.set_yticks(y, [LABELS[c] for c in sub.condition])
        ax.invert_yaxis()
        ax.set_xlim(0.35, 0.82)
        ax.set_xlabel("Test AUROC")
        ax.set_title(f"{model} · 0.05 rad")
        for yi, value in zip(y, sub.test_auroc):
            ax.text(value + 0.008, yi, f"{value:.3f}", va="center", fontsize=9)
    axes[0].set_ylabel("Condition")
    fig.suptitle("Comparable four-condition snapshot", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "tuep_component_ranking_005.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # XGBoost SHAP heatmap, all available phases/conditions.
    order = ["delta", "theta", "alpha", "beta", "gamma", "hjorth", "spectral_entropy", "asymmetry"]
    shp = shap[shap.band.isin(order)].copy()
    shp["phase_condition"] = shp.phase + " · " + shp.condition.map(LABELS)
    cols = []
    for phase, _ in PHASE_DIRS:
        for condition in CONDITIONS:
            if ((shp.phase == phase) & (shp.condition == condition)).any():
                cols.append(f"{phase} · {LABELS[condition]}")
    matrix = shp.pivot(index="band", columns="phase_condition", values="mean_abs_shap").reindex(order).reindex(columns=cols)
    fig, ax = plt.subplots(figsize=(13, 5.6))
    im = ax.imshow(matrix.fillna(0).values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(np.arange(matrix.shape[0]), matrix.index)
    ax.set_title("XGBoost mean |SHAP| by feature block")
    ax.set_xlabel("Phase · condition")
    ax.set_ylabel("Feature block")
    fig.colorbar(im, ax=ax, label="Mean absolute SHAP")
    fig.tight_layout()
    fig.savefig(FIG / "tuep_component_xgboost_shap_heatmap.png", dpi=180)
    plt.close(fig)


def fmt(value, digits=3):
    return "—" if pd.isna(value) else f"{value:.{digits}f}"


def metric_table(metrics, model, phase, cols=("test_auroc", "test_auprc", "test_f1", "test_balanced_accuracy", "test_accuracy")):
    sub = metrics[(metrics.model == model) & (metrics.phase == phase)].set_index("condition").reindex(CONDITIONS)
    labels = {"test_auroc": "AUROC", "test_auprc": "AUPRC", "test_f1": "F1", "test_balanced_accuracy": "Balanced acc.", "test_accuracy": "Accuracy"}
    head = "<tr><th>Condition</th>" + "".join(f"<th>{labels[c]}</th>" for c in cols) + "</tr>"
    body = "".join("<tr><th>" + LABELS[c] + "</th>" + "".join(f"<td>{fmt(row[col])}</td>" if c in sub.index and not pd.isna(row[col]) else "<td>—</td>" for col in cols) + "</tr>" for c, row in sub.iterrows())
    return "<table><thead>" + head + "</thead><tbody>" + body + "</tbody></table>"


def build_report(metrics: pd.DataFrame, shap: pd.DataFrame, phase_info):
    counts = metrics.groupby(["phase", "model"]).size().unstack(fill_value=0)
    best_xgb = metrics[metrics.model == "XGBoost"].sort_values("test_auroc", ascending=False).iloc[0]
    best_eeg = metrics[metrics.model == "EEGNet"].sort_values("test_auroc", ascending=False).iloc[0]
    comp = metrics[metrics.phase == "0.05"]
    xgb_specific = comp[(comp.model == "XGBoost") & (comp.condition == "specific")].iloc[0]
    xgb_raw = comp[(comp.model == "XGBoost") & (comp.condition == "raw")].iloc[0]
    all_phases = ", ".join(p for p, _ in PHASE_DIRS)
    phase_rows = "".join(
        f"<tr><th>{p} rad</th><td>{int(counts.loc[p, 'EEGNet']) if 'EEGNet' in counts.columns and p in counts.index else 0}</td><td>{int(counts.loc[p, 'XGBoost']) if 'XGBoost' in counts.columns and p in counts.index else 0}</td><td>{'四条件完整' if (counts.loc[p].min() if p in counts.index else 0) == 4 else '需核查'}</td></tr>"
        for p, _ in PHASE_DIRS
    )
    shap_focus = shap[(shap.phase == "0.05") & (shap.condition == "specific")].sort_values("mean_abs_shap", ascending=False).head(4)
    shap_text = "、".join(f"{row.band} ({row.mean_abs_shap:.4f})" for _, row in shap_focus.iterrows())

    css = """
    :root{--canvas:#eaf0f4;--paper:#fff;--ink:#142c43;--text:#283b4d;--muted:#5a6a78;--line:#d2dde5;--soft:#f5f8fa;--teal:#008c82;--orange:#d97a2b;--purple:#a25aa8;--blue:#53657a;--warn:#fff4e8;--shadow:0 14px 40px rgba(25,48,69,.10);--radius:16px;--content:1180px}*{box-sizing:border-box}body{margin:0;background:var(--canvas);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",Arial,sans-serif;font-size:16px;line-height:1.7}a{color:#0b6e78}.shell{display:grid;grid-template-columns:238px minmax(0,1fr);width:min(calc(100% - 32px),1440px);margin:16px auto 56px;gap:22px;align-items:start}.rail{position:sticky;top:16px;padding:24px 20px;border:1px solid var(--line);border-radius:var(--radius);background:rgba(255,255,255,.94);box-shadow:var(--shadow)}.rail h2{font-size:1.05rem;line-height:1.3;margin:0 0 18px;color:var(--ink)}.rail a{display:block;padding:7px 10px;color:var(--muted);text-decoration:none;font-size:.9rem}.rail a:hover{color:var(--ink);background:#eef7f6}.rail p{border-top:1px solid var(--line);margin:20px 0 0;padding-top:16px;color:var(--muted);font-size:.78rem}main{min-width:0}.hero,section,footer{width:min(100%,var(--content));margin-inline:auto}.hero{padding:clamp(30px,5vw,62px);border-radius:20px;color:#f6fbff;background:linear-gradient(115deg,rgba(8,38,61,.98),rgba(18,62,83,.94));box-shadow:var(--shadow)}.eyebrow{margin:0 0 12px;color:#6ce2d6;font-size:.78rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase}.hero h1{max-width:900px;margin:0;color:#fff;font-size:clamp(2rem,4.7vw,4.1rem);line-height:1.05;letter-spacing:-.035em}.deck{max-width:820px;margin:22px 0 0;color:#d6e4ed;font-size:1.1rem}.tags{display:flex;flex-wrap:wrap;gap:8px;margin-top:24px}.tags span{padding:6px 10px;border:1px solid rgba(255,255,255,.19);border-radius:999px;color:#e6f0f5;background:rgba(255,255,255,.07);font-size:.78rem;font-weight:700}section{margin-top:22px;padding:clamp(24px,4vw,42px);border-radius:var(--radius);background:var(--paper);box-shadow:var(--shadow)}h2{margin:0 0 15px;color:var(--ink);font-size:clamp(1.45rem,2.5vw,2.2rem);line-height:1.2}h3{margin:24px 0 8px;color:var(--ink);font-size:1.08rem}p{margin:10px 0}.lead{font-size:1.08rem}.callout{padding:15px 18px;border-left:4px solid var(--orange);background:var(--warn);border-radius:8px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin-top:18px}.figure{margin:20px 0 0}.figure img{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:10px;background:#fff}.figure figcaption{margin-top:8px;color:var(--muted);font-size:.85rem}.table-wrap{overflow-x:auto;margin:16px 0}.table-wrap table{width:100%;border-collapse:collapse;font-size:.9rem}.table-wrap th,.table-wrap td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}.table-wrap th:first-child,.table-wrap td:first-child{text-align:left}.table-wrap thead th{color:var(--ink);background:var(--soft)}.table-wrap tbody th{font-weight:650;color:var(--ink)}.small{color:var(--muted);font-size:.88rem}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.9em;overflow-wrap:anywhere}.cols{columns:2;column-gap:30px}.cols li{break-inside:avoid;margin-bottom:7px}details{margin-top:14px;border-top:1px solid var(--line)}summary{cursor:pointer;padding:12px 0;color:var(--ink);font-weight:700}footer{padding:22px 4px;color:var(--muted);font-size:.86rem}@media(max-width:900px){.shell{display:block;width:min(calc(100% - 20px),760px)}.rail{position:static;margin-bottom:14px}.grid{grid-template-columns:1fr}.cols{columns:1}}"""
    body = f"""<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>TUEP ECMAD 分量分类实验：相位门控描述性报告</title><style>{css}</style></head><body><div class=\"shell\"><aside class=\"rail\"><h2>TUEP 相位门控<br>描述性报告</h2><nav><a href=\"#summary\">结论速览</a><a href=\"#scope\">范围与口径</a><a href=\"#phase\">相位扫描</a><a href=\"#models\">模型对照</a><a href=\"#shap\">SHAP 模式</a><a href=\"#limits\">限制与后续</a><a href=\"#appendix\">附录指标</a></nav><p>生成日期：2026-09-18<br>数据根目录：<span class=\"mono\">results/results_tuep_phase*</span></p></aside><main><header class=\"hero\"><p class=\"eyebrow\">ECMAD · TUEP · PATIENT-LEVEL EVALUATION</p><h1>分量分类结果的<br>相位门控描述性分析</h1><p class=\"deck\">对五个 phase-gated 阈值下的 TUEP 患者级 EEGNet 与 XGBoost 产物进行统一汇总，重点观察 specific / coherent 分量及其组合对下游分类效用的影响。</p><div class=\"tags\"><span>5 个相位阈值：{all_phases} rad</span><span>正类：control（标签 1）</span><span>聚合单位：patient</span><span>feature profile：base211</span></div></header>
    <section id=\"summary\"><h2>结论速览</h2><p class=\"lead\">高阈值阶段已补齐四个条件，五个阈值现在都支持完整的 Raw / specific / coherent / specific_coherent 比较。当前 XGBoost 的最佳测试 AUROC 为 <strong>0.794</strong>，出现在 0.5 和 1 rad 的 specific_coherent；但在同一阈值下，specific 在 0.5 rad 达到 AUROC 0.776、在 1 rad 达到 0.782，说明提升并不只来自组合条件。EEGNet 的最高测试 AUROC 为 0.650，出现在 0.05 rad 的 specific。</p><div class=\"callout\"><strong>更新后的解释边界：</strong>新增高阈值条件后，phase × condition 的覆盖已经完整，因而可以进行描述性的全因素横向比较；但这仍是单次 split、37 位测试患者的结果，不能据此声称稳定优越性或生理因果关系。</div><div class=\"grid\"><div><h3>最佳 XGBoost 条目</h3><p><strong>{best_xgb.phase} rad · {LABELS[best_xgb.condition]}</strong><br>AUROC {best_xgb.test_auroc:.3f} · AUPRC {best_xgb.test_auprc:.3f}<br>F1 {best_xgb.test_f1:.3f} · balanced accuracy {best_xgb.test_balanced_accuracy:.3f}</p></div><div><h3>最佳 EEGNet 条目</h3><p><strong>{best_eeg.phase} rad · {LABELS[best_eeg.condition]}</strong><br>AUROC {best_eeg.test_auroc:.3f} · AUPRC {best_eeg.test_auprc:.3f}<br>F1 {best_eeg.test_f1:.3f} · balanced accuracy {best_eeg.test_balanced_accuracy:.3f}</p></div></div><figure class=\"figure\"><img src=\"figures/tuep_component_phase_condition_auroc_3d.png\" alt=\"EEGNet和XGBoost中五个phase与四个condition对应测试AUROC的三维柱状图\"><figcaption>图 0｜phase × condition × test AUROC：柱高和颜色深浅均编码 AUROC，两个模型使用同一颜色刻度。</figcaption></figure></section>
    <section id=\"scope\"><h2>数据范围与分析口径</h2><p>本报告读取五个结果目录中的 <span class=\"mono\">config_resolved.yaml</span>、EEGNet <span class=\"mono\">condition_metrics.csv</span>、XGBoost <span class=\"mono\">comparison_summary.csv</span>、各条件的 <span class=\"mono\">test_metrics.json</span> / <span class=\"mono\">data_stats.json</span> 与 SHAP 汇总。所有完整结果的患者划分为 train 124、validation 17、test 37，且 <span class=\"mono\">patient_disjoint=true</span>；正类是 control。</p><figure class=\"figure\"><img src=\"figures/tuep_component_phase_coverage.png\" alt=\"五个相位阈值下 EEGNet 和 XGBoost 的条件行数\"><figcaption>图 1｜结果覆盖：五个阈值均包含 Raw、specific、coherent、specific_coherent 四条件。</figcaption></figure><div class=\"table-wrap\"><table><thead><tr><th>phase threshold</th><th>EEGNet rows</th><th>XGBoost rows</th><th>coverage</th></tr></thead><tbody>{phase_rows}</tbody></table></div><details><summary>模型与评估规则</summary><ul class=\"cols\"><li>EEGNet：患者均衡加权 BCE；按 validation patient AUPRC 选择 checkpoint。</li><li>EEGNet：按 validation patient balanced accuracy 冻结阈值。</li><li>XGBoost：患者级聚合；报告中保留结果文件的测试阈值。</li><li>所有 test 指标都是冻结规则后的单次最终测试评估。</li><li>specific_coherent：通道轴先 specific、后 coherent。</li><li>base211：211 维特征，SHAP 按 feature block 汇总。</li></ul></details></section>
    <section id=\"phase\"><h2>相位阈值扫描</h2><p>相位门控阈值越大，允许通过的相位差范围越宽。新增结果后，所有 phase × condition 组合均有实测数据，曲线不再需要对缺失条件做解释性留白。</p><figure class=\"figure\"><img src=\"figures/tuep_component_phase_auroc_lines.png\" alt=\"EEGNet和XGBoost在五个相位阈值下的测试AUROC曲线\"><figcaption>图 2｜患者级测试 AUROC；虚线为 0.5 的排序基准，五条条件曲线均由实际结果绘制。</figcaption></figure><p>XGBoost 的 specific_coherent 在 0.5 和 1 rad 均为 0.794；但 specific 在 0.5/1 rad 分别为 0.776/0.782，且 coherent 在 1 rad 达到 0.694，表明高阈值下各分量的排序性能整体改善。EEGNet 则没有同样的单调趋势：specific 在 0.05 rad 达到峰值 0.650，specific_coherent 在 0.1 rad 降至 0.441 后回升。</p></section>
    <section id=\"models\"><h2>模型与条件对照</h2><h3>0.05 rad：四条件横向比较</h3><figure class=\"figure\"><img src=\"figures/tuep_component_ranking_005.png\" alt=\"0.05 rad下EEGNet与XGBoost四条件测试AUROC横向比较\"><figcaption>图 3｜0.05 rad 的四条件快照。</figcaption></figure><p>在 XGBoost 中，specific 相比 Raw 的测试 AUROC 提高 <strong>{xgb_specific.test_auroc - xgb_raw.test_auroc:+.3f}</strong>，balanced accuracy 提高 <strong>{xgb_specific.test_balanced_accuracy - xgb_raw.test_balanced_accuracy:+.3f}</strong>；specific_coherent 也保持较高水平（AUROC 0.759）。coherent 单独使用较弱（AUROC 0.574）。EEGNet 的 0.05 rad 结果不重复这一排序：specific 为 0.650，而 specific_coherent 为 0.526。</p><h3>高阈值 specific_coherent：组合条件与单分量</h3><div class=\"grid\"><div><h3>EEGNet · 0.5 / 1 rad</h3>{metric_table(metrics,'EEGNet','0.5')} {metric_table(metrics,'EEGNet','1')}</div><div><h3>XGBoost · 0.5 / 1 rad</h3>{metric_table(metrics,'XGBoost','0.5')} {metric_table(metrics,'XGBoost','1')}</div></div></section>
<section id=\"shap\"><h2>特征贡献模式：XGBoost</h2><p>SHAP 汇总显示，base211 的贡献主要集中在频带功率、Hjorth 与 spectral entropy / asymmetry；connectivity、wavelet、complexity、temporal 在该 profile 中为 0，不应被解读为已参与本次模型。</p><figure class=\"figure\"><img src=\"figures/tuep_component_xgboost_shap_heatmap.png\" alt=\"不同相位和条件下XGBoost feature block的平均绝对SHAP热图\"><figcaption>图 4｜XGBoost mean absolute SHAP；颜色越深表示相对贡献越大，数值来自各结果目录的 shap_by_band.json。</figcaption></figure><p>在 0.05 rad 的 specific 条件中，贡献较高的 block 为 {shap_text}。这只是模型解释层面的关联，不等同于这些 block 对生理信号的独立因果作用。</p></section>
<section id=\"limits\"><h2>限制、风险与下一步</h2><ul><li><strong>已解决的覆盖问题：</strong>五个阈值现在都包含四个条件，报告可以进行完整的 phase × condition 描述比较。</li><li><strong>样本量：</strong>测试集只有 37 位患者；患者级指标可受少数患者影响，尤其是 F1、balanced accuracy 和冻结阈值下的混淆矩阵。</li><li><strong>模型差异：</strong>EEGNet 与 XGBoost 的输入与优化路径不同，不能将它们的绝对数值直接视为同一模型的重复验证。</li><li><strong>实验目标：</strong>分类效用是下游评价，不替代 ERP-CORE 的直接信号保真/去噪指标，也不能据此断言 specific_coherent 在生理意义上“最干净”。</li></ul><h3>建议的确认路径</h3><ol><li>对 XGBoost 0.5/1 rad 的候选结果做重复随机种子或 bootstrap 患者级置信区间。</li><li>检查高阈值下分量性能提升是否在独立 split 或外部数据上复现。</li><li>将高阈值候选与 ERP-CORE 直接去噪指标联合判断，而不是只按 AUROC 选阈值。</li></ol></section>
<section id=\"appendix\"><h2>完整测试指标</h2><details open><summary>EEGNet：各相位实际存在的条件</summary><div class=\"table-wrap\">{"".join(f"<h3>{p} rad</h3>{metric_table(metrics,'EEGNet',p)}" for p,_ in PHASE_DIRS)}</div></details><details><summary>XGBoost：各相位实际存在的条件</summary><div class=\"table-wrap\">{"".join(f"<h3>{p} rad</h3>{metric_table(metrics,'XGBoost',p)}" for p,_ in PHASE_DIRS)}</div></details><details><summary>来源与复核</summary><p>报告由 <span class=\"mono\">scripts/generate_tuep_phase_descriptive_report.py</span> 读取结果目录生成。原始产物包括各 phase 的 <span class=\"mono\">config_resolved.yaml</span>、<span class=\"mono\">run_summary.json</span>、<span class=\"mono\">split_manifest.csv</span>、模型指标 CSV/JSON 与 SHAP JSON。报告没有重新训练模型，也没有对缺失条件做插值或填充。</p></details></section><footer>生成文件：<span class=\"mono\">docs/tuep_component_eegnet_xgboost_phase_descriptive_report_2026-09-18.html</span> · 本报告为结果描述性汇总，不替代正式统计检验。</footer></main></div></body></html>"""
    REPORT.write_text(body, encoding="utf-8")


def main():
    metrics, shap, phase_info = load_data()
    save_figures(metrics, shap)
    build_report(metrics, shap, phase_info)
    print(REPORT)
    print(metrics.groupby(["model", "phase"]).size().to_string())


if __name__ == "__main__":
    main()
