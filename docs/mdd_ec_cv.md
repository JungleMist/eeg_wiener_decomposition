# MDD_EC 下游分类与 SHAP 比较

在项目根目录运行，默认读取 `~/Data/MDD_EC`，可用 `--data-dir` 覆盖。

```bash
conda run --no-capture-output -n eeg_pipeline python scripts/run_mdd_ec_cv.py \
  --data-dir "$HOME/Data/MDD_EC" --output results/mdd_ec_cv --workers 4
```

默认运行两种模型及全部 13 个条件。只检查 EDF 配对及文件头时，传入 `--stage check`。也可以依次使用 `--stage prepare`、`--stage train` 和 `--stage summarize`，保持数据路径、输出目录和训练设置一致。训练阶段可用 `--fold 1` 至 `--fold 10`、`--method` 和 `--condition` 限定运行范围。重复命令会跳过完整结果，EEGNet 从最近保存的训练轮次恢复。`--device auto` 为 EEGNet 选择可用的加速器，XGBoost 使用 CPU。CUDA 服务器可显式指定 `--device cuda`。

## 数据与划分

- 从 `MDD` 和 `Control` 目录分别读取正负类。每个片段 EDF 对应一个 epoch，文件名中的 `sub-*` 定义受试者。同一受试者的全部 session 始终同组。
- 按 EDF 文件名配对 Raw 和四个相位的 Specific、Coherent。对每个 EDF 严格按指定顺序提取 22 个通道。各组取开头 120 秒，从 250 Hz 降采样至 125 Hz。Combo 先拼 Specific 再拼 Coherent。EEGNet 输入 44 通道，XGBoost 分别提取两组特征后拼接。
- 按受试者分层保留 20% 固定测试集。其余受试者做分层 10 折，每折使用九折拟合、一折验证。全部条件和模型读取共同的 `splits.csv`。每个条件独立训练十个模型，并对同一留出集预测。每个片段先独立预测，受试者概率取片段平均。
- EEGNet 使用现有 TUEP 超参数和受试者等权、类别加权 BCE，以验证受试者 AUPRC 选择检查点。XGBoost 沿用近期 nested CV 的库默认训练参数与训练数据拟合的 StandardScaler。22 通道特征沿用已有计算定义，包括 198 个单通道特征、45 个功率不对称特征和 90 个相干性及 PLV 特征，每组 333 维。Combo 为 666 维。
- 单折阈值从各自验证集选取，EEGNet 优化平衡准确率，XGBoost 优化宏 F1。最终留出集结果使用十个模型的平均概率，并采用开发集折外预测选出的阈值。十次留出评估涉及同一组受试者，折间离散度描述模型拟合波动。

## 产物

- 根目录保存 `run_config.json`、`source_files.csv`、`recordings.csv` 和 `splits.csv`。
- `<method>/fold_XX/<condition>/` 保存检查点或模型、验证及测试预测、评估指标、留出片段 SHAP 数组和索引。EEGNet 用各折训练受试者作背景，归因单位为概率。XGBoost 使用 TreeSHAP，单位为 log-odds。
- `summary/holdout_comparison.csv` 保存 26 组集成结果及各指标相对 Raw 的差值。`fold_metrics.csv` 保存每折验证集和留出集的指标。每个模型及条件另有开发折外预测、留出预测和逐特征或逐通道 SHAP 表。`shap_comparison.csv` 汇总成分组，保留 Combo 的整体及两个成分，并标明单位。先对 EEGNet 归因沿时间取绝对值均值，再对每位留出受试者的片段、受试者和折依次等权平均。
- 完整实验包含每种模型各 130 次拟合。EEGNet 的逐时间点归因会占用数十 GB 磁盘。只汇总部分条件或部分模型时，结果写入 `summary/` 下的独立子目录。
- `shap_item_comparison.csv` 按同一模型和同名通道或特征对齐各条件与 Raw，保留 Combo 各成分的归因差值。
