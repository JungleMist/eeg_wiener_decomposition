"""Associate subject-level nested-CV predictions with depression scale scores."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


N_BOOTSTRAP = 10_000
SEED = 478
GROUP_NAMES = {0: "Low BDI", 1: "High BDI"}
GROUP_COLORS = {0: "#2478a2", 1: "#bf503b"}


def load_inputs(results, participants):
    metadata = pd.read_csv(participants, sep="\t")
    metadata = metadata[["participant_id", "BDI", "HamD", "age", "sex"]]
    thresholds = pd.read_csv(results / "comparison.csv")
    if thresholds.empty or thresholds.duplicated(["method", "condition"]).any():
        raise ValueError("comparison.csv must contain distinct model conditions")
    frames = []
    reference = None
    for row in thresholds.itertuples(index=False):
        path = results / f"{row.method}_{row.condition}_oof.csv"
        frame = pd.read_csv(path).merge(
            metadata, on="participant_id", how="left", validate="one_to_one",
        ).sort_values("participant_id").reset_index(drop=True)
        if frame.empty or frame.BDI.isna().any():
            raise ValueError(f"Missing BDI or empty OOF input in {path}")
        expected = np.select([frame.BDI <= 13, frame.BDI >= 20], [0, 1], default=-1)
        if np.any(expected == -1) or not np.array_equal(expected, frame.true_label):
            raise ValueError(f"BDI and OOF labels disagree in {path}")
        if not frame.pred_proba.between(0, 1).all():
            raise ValueError(f"Invalid OOF probabilities in {path}")
        identity = frame[["participant_id", "true_label"]]
        if reference is None:
            reference = identity
        if not identity.equals(reference):
            raise ValueError(f"OOF cohort differs in {path}")
        auc = roc_auc_score(frame.true_label, frame.pred_proba)
        if not np.isclose(auc, row.auroc, atol=1e-9, rtol=0):
            raise ValueError(f"Recomputed AUROC disagrees with comparison.csv in {path}")
        frames.append(frame.assign(method=row.method, condition=row.condition, auroc=auc))
    return frames


def bootstrap_indices(labels, rng):
    draws = [
        rng.choice(positions, size=(N_BOOTSTRAP, len(positions)))
        for label in np.unique(labels)
        if len(positions := np.flatnonzero(labels == label))
    ]
    return np.column_stack(draws)


def correlations(x, y, draws):
    sampled_x, sampled_y = x[draws], y[draws]
    intervals = {
        "spearman": stats.pearsonr(
            stats.rankdata(sampled_x, axis=1), stats.rankdata(sampled_y, axis=1), axis=1,
        ).statistic,
        "pearson": stats.pearsonr(sampled_x, sampled_y, axis=1).statistic,
    }
    for name, statistic in (("spearman", stats.spearmanr), ("pearson", stats.pearsonr)):
        estimate, p = statistic(x, y)
        values = intervals[name]
        lower, upper = np.nanpercentile(values, [2.5, 97.5])
        yield dict(
            correlation=name, n=len(x), estimate=estimate, ci_low=lower, ci_high=upper,
            p=p, bootstrap_valid=np.isfinite(values).sum(),
        )


def fit_ols(x, y):
    y = np.asarray(y)
    coefficient, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank != x.shape[1] or len(y) <= rank:
        raise ValueError("Regression design is rank deficient or lacks residual degrees of freedom")
    inverse = np.linalg.pinv(x)
    fitted = x @ coefficient
    residuals = y - fitted
    leverage = np.einsum("ij,ji->i", x, inverse)
    scaled = inverse * (residuals / (1 - leverage))
    covariance = scaled @ scaled.T
    r_squared = 1 - np.sum(residuals**2) / np.sum((y - y.mean())**2)
    return coefficient, covariance, len(y) - rank, r_squared


def regressions(frame):
    complete = frame.dropna(subset=["age", "sex"])
    covariates = [np.ones(len(complete)), complete.pred_proba * 10,
                  complete.age - complete.age.mean(), complete.sex.eq(2).astype(int)]
    bdi = complete.BDI.to_numpy()
    for model, terms in (("age_sex", covariates),
                         ("age_sex_bdi_group", [*covariates, complete.true_label])):
        coefficient, covariance, df, r_squared = fit_ols(np.column_stack(terms), bdi)
        beta = coefficient[1]
        se = np.sqrt(covariance[1, 1])
        margin = stats.t.ppf(0.975, df) * se
        yield dict(
            model=model, n=len(complete), beta_per_0_1=beta, ci_low=beta - margin,
            ci_high=beta + margin, p=2 * stats.t.sf(abs(beta / se), df), r_squared=r_squared,
        )


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("Non-finite p value in multiple-testing family")
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def scatter_plots(frames, output):
    for method in sorted({frame.method.iat[0] for frame in frames}):
        selected = [frame for frame in frames if frame.method.iat[0] == method]
        fig, axes = plt.subplots(
            (len(selected) + 3) // 4, 4, figsize=(14, 10), sharex=True, sharey=True,
            layout="constrained",
        )
        for ax, frame in zip(axes.flat, selected):
            for label, group in frame.groupby("true_label"):
                ax.scatter(group.pred_proba, group.BDI, s=11, alpha=0.6,
                           label=GROUP_NAMES[label], color=GROUP_COLORS[label], linewidth=0)
            x = frame.pred_proba.to_numpy()
            grid = np.linspace(x.min(), x.max(), 100)
            design = np.column_stack([np.ones(len(x)), x])
            grid_design = np.column_stack([np.ones(len(grid)), grid])
            coefficient, covariance, df, _ = fit_ols(design, frame.BDI)
            fitted = grid_design @ coefficient
            error = np.sqrt(np.einsum("ij,jk,ik->i", grid_design, covariance, grid_design))
            margin = stats.t.ppf(0.975, df) * error
            ax.plot(grid, fitted, color="#333333", linewidth=1.4)
            ax.fill_between(grid, fitted - margin, fitted + margin, color="#333333", alpha=0.14)
            condition = frame.condition.iat[0].replace("_", " ")
            rho = stats.spearmanr(x, frame.BDI).statistic
            ax.set_title(f"{condition}\nSpearman ρ = {rho:+.2f}", fontsize=9)
            ax.set(xlim=(-0.015, 1.015), ylim=(-1, 33), xticks=[0, 0.25, 0.5, 0.75, 1])
            ax.tick_params(labelbottom=True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.2)
        for ax in list(axes.flat)[len(selected):]:
            ax.set_axis_off()
        axes.flat[0].legend(frameon=False, fontsize=8, loc="upper left")
        fig.suptitle(f"{method.upper()} | Subject-level out-of-fold scores and BDI", fontsize=15)
        fig.supxlabel("Out-of-fold model probability for high BDI")
        fig.supylabel("BDI total")
        fig.savefig(output / f"{method}_bdi_scatter.png", dpi=170)
        plt.close(fig)


def write_report(output, data, checks, corr, reg):
    primary = corr.query("scale == 'BDI' and correlation == 'spearman'")
    pooled = primary.query("group == 'all'")
    within = primary.query("group != 'all'")
    adjusted = reg.query("model == 'age_sex_bdi_group'")
    hamd = corr.query("scale == 'HamD' and correlation == 'spearman'")
    selected = primary.pivot(index=["method", "condition"], columns="group", values="estimate")
    selected = checks.set_index(["method", "condition"])[["auroc"]].join(selected)
    selected = selected.join(adjusted.set_index(["method", "condition"])[["beta_per_0_1", "p_holm"]])
    selected = selected.rename(columns={
        "auroc": "AUROC", "all": "总体 ρ", "low_bdi": "低 BDI ρ", "high_bdi": "高 BDI ρ",
        "beta_per_0_1": "控制分组后 β", "p_holm": "β 的 Holm p",
    })
    cohort = data.drop_duplicates("participant_id")
    n_low, n_high = (cohort.true_label.eq(label).sum() for label in (0, 1))
    n_age_missing = cohort.age.isna().sum()
    n_sex_missing = cohort.sex.isna().sum()
    best = pooled.loc[pooled.estimate.idxmax()]
    smallest_pooled = pooled.loc[pooled.p.idxmin()]
    smallest_adjusted = adjusted.loc[adjusted.p.idxmin()]
    min_within = within.loc[within.p.idxmin()]
    best_hamd = hamd.loc[hamd.p.idxmin()]
    report = f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ds003478 v1.1.0 OOF 概率与抑郁量表</title>
<style>
body {{ font-family: system-ui, sans-serif; color: #1f2933; max-width: 1100px; margin: 36px auto; padding: 0 18px; line-height: 1.65; }}
a {{ color: #185f85; }} h2 {{ margin-top: 1.8em; }} table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: 5px 10px; border-bottom: 1px solid #dce2e6; text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
th {{ background: #eaf1f4; }} .table {{ overflow-x: auto; }} img {{ max-width: 100%; }}
</style></head>
<body>
<h1>ds003478 v1.1.0 的 OOF 概率与抑郁量表</h1>
<p>在 {len(checks)} 个模型条件中，总体 BDI Spearman 相关经 Holm 校正后有 {(pooled.p_holm < .05).sum()} 项达到 0.05。
组内相关有 {(within.p_holm < .05).sum()}/{len(within)} 项，控制年龄、性别和 BDI 分组后的回归有
{(adjusted.p_holm < .05).sum()}/{len(adjusted)} 项。总体相关与组内严重程度梯度回答不同问题，分类标签由同一 BDI 分数切分，解读时优先检查组内和控制分组后的结果。</p>
<ul>
<li>最大总体 ρ 出现在 {best.method} / {best.condition}，ρ = {best.estimate:+.3f}，95% CI [{best.ci_low:+.3f}, {best.ci_high:+.3f}]，Holm p = {best.p_holm:.4g}。</li>
<li>总体最小原始 p 出现在 {smallest_pooled.method} / {smallest_pooled.condition}，ρ = {smallest_pooled.estimate:+.3f}，95% CI [{smallest_pooled.ci_low:+.3f}, {smallest_pooled.ci_high:+.3f}]，Holm p = {smallest_pooled.p_holm:.4g}。</li>
<li>组内最小原始 p 出现在 {min_within.method} / {min_within.condition} / {min_within.group}，ρ = {min_within.estimate:+.3f}，95% CI [{min_within.ci_low:+.3f}, {min_within.ci_high:+.3f}]，原始 p = {min_within.p:.4g}，Holm p = {min_within.p_holm:.4g}。</li>
<li>控制分组的回归中，最小原始 p 出现在 {smallest_adjusted.method} / {smallest_adjusted.condition}，β = {smallest_adjusted.beta_per_0_1:+.3f} BDI 分 / 0.1 概率，95% CI [{smallest_adjusted.ci_low:+.3f}, {smallest_adjusted.ci_high:+.3f}]，Holm p = {smallest_adjusted.p_holm:.4g}。</li>
<li>方向校验显示 {checks.mean_difference.gt(0).sum()}/{len(checks)} 个条件的高 BDI 组平均概率更高，
{checks.auroc.gt(.5).sum()}/{len(checks)} 个条件的 AUROC &gt; 0.5。平均概率差和 AUROC 衡量不同性质。</li>
<li>探索性 HamD 分析限定在同时有 HamD 的 {int(best_hamd.n)} 名高 BDI 受试者。最小原始 p 为 {best_hamd.p:.4g}（{best_hamd.method} / {best_hamd.condition}，ρ = {best_hamd.estimate:+.3f}，Holm p = {best_hamd.p_holm:.4g}）。
HamD 的访谈可用性与招募过程相关，样本存在选择偏差。</li>
</ul>
<h2>样本和统计方法</h2>
<p>将外层留一交叉验证的受试者预测与数据集 participants.tsv 按 participant_id 一对一合并。每个条件纳入 {len(cohort)} 人，低 BDI（≤13）{n_low} 人，高 BDI（≥20）{n_high} 人。
各概率已经在实验中对该受试者的 EEG 窗和五个内层模型取均值。标签 1 代表高 BDI 症状组。临床访谈只覆盖部分人，高 BDI 概率的解释限于此二分类模型分数。</p>
<p>对总体、低 BDI、高 BDI 分别计算 Spearman 秩相关和 Pearson 相关。两侧 p 值用常规检验，95% CI 取
{N_BOOTSTRAP:,} 次受试者配对自助抽样的百分位数，总体按 BDI 分组分层抽样。
HamD 采用同样的相关方法，限高 BDI 完整观测者。回归以 BDI 为因变量，分别调整年龄与性别，以及年龄、性别与 BDI 分组。
β 表示高 BDI 概率每增加 0.1 对应的 BDI 分差。采用 HC3 稳健标准误和 t 区间，另报告完整模型 R²。
完整案例回归中年龄缺失 {n_age_missing} 人、性别缺失 {n_sex_missing} 人。</p>
<p>Holm 校正分别用于 BDI Spearman 的 {len(primary)} 项、BDI Pearson 的 {len(primary)} 项、两种回归合计的 {len(reg)} 项，以及 HamD 各相关类型的 {len(hamd)} 项。
区间为逐项区间。推断条件于已生成的 OOF 概率，外层训练集高度重叠，区间未包含重训变异。
低高分切点移除了中间段，组内量表范围较窄。分析是探索性的，无法验证临床校准或外部泛化。</p>
<p>数据来源 <a href="https://doi.org/10.18112/openneuro.ds003478.v1.1.0">OpenNeuro ds003478 v1.1.0</a></p>
<h2>各模型条件</h2>
<p><a href="correlations.csv">完整相关统计</a> · <a href="regressions.csv">完整回归统计</a> ·
<a href="probability_checks.csv">方向校验</a> · <a href="subject_scores.csv">受试者配对表</a></p>
<div class="table">{selected.reset_index().to_html(index=False, float_format=lambda x: f"{x:.4f}")}</div>
<h2>受试者散点图</h2>
<p>颜色区分低、高 BDI。直线为总体线性拟合，阴影为基于 HC3 标准误的逐点 95% 均值置信带。</p>
{"".join(f'<img src="{method}_bdi_scatter.png" alt="{method} OOF 概率与 BDI 的 13 组散点图">' for method in checks.method.unique())}
</body></html>
"""
    (output / "report.html").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("participants", type=Path)
    args = parser.parse_args()
    frames = load_inputs(args.results, args.participants)
    rng = np.random.default_rng(SEED)
    reference = frames[0]
    specs = [
        ("BDI", "all", np.ones(len(reference), dtype=bool)),
        ("BDI", "low_bdi", reference.true_label.eq(0)),
        ("BDI", "high_bdi", reference.true_label.eq(1)),
        ("HamD", "high_bdi", reference.true_label.eq(1) & reference.HamD.notna()),
    ]
    samples = [(scale, group, mask, bootstrap_indices(reference.loc[mask, "true_label"], rng))
               for scale, group, mask in specs]
    corr_rows, reg_rows, check_rows = [], [], []
    for frame in frames:
        identity = dict(method=frame.method.iat[0], condition=frame.condition.iat[0])
        means = frame.groupby("true_label").pred_proba.mean()
        check_rows.append(dict(**identity, auroc=frame.auroc.iat[0], n=len(frame),
                               low_bdi_mean=means[0], high_bdi_mean=means[1],
                               mean_difference=means[1] - means[0]))
        for scale, group, mask, draws in samples:
            for result in correlations(
                frame.loc[mask, "pred_proba"].to_numpy(), frame.loc[mask, scale].to_numpy(), draws,
            ):
                corr_rows.append(dict(**identity, scale=scale, group=group, **result))
        reg_rows.extend(dict(**identity, **result) for result in regressions(frame))
    corr = pd.DataFrame(corr_rows)
    corr["p_holm"] = corr.groupby(["scale", "correlation"]).p.transform(holm)
    reg = pd.DataFrame(reg_rows)
    reg["p_holm"] = holm(reg.p)
    checks = pd.DataFrame(check_rows)
    data = pd.concat(frames, ignore_index=True)
    output = args.results / "bdi_association"
    output.mkdir(exist_ok=True)
    data[["method", "condition", "participant_id", "pred_proba", "true_label", "BDI", "HamD",
          "age", "sex"]].to_csv(output / "subject_scores.csv", index=False)
    corr.to_csv(output / "correlations.csv", index=False)
    reg.to_csv(output / "regressions.csv", index=False)
    checks.to_csv(output / "probability_checks.csv", index=False)
    scatter_plots(frames, output)
    write_report(output, data, checks, corr, reg)
    print(output / "report.html")


if __name__ == "__main__":
    main()
