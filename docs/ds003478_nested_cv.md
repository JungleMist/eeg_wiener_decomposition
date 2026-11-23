# ds003478 nested CV

外层按受试者 LOOCV，106 折；内层 StratifiedKFold，k=5，seed=42。原固定三个分区合并，16 名排除受试者继续排除。两次记录跟随受试者。预处理、13 条件、特征与 EEGNet 的 TUEP 参数继承 configs/ECMAD.yaml。嵌套 XGBoost 使用库默认超参数。

采用五模型集成。EEGNet 保留各内层验证 AP 最优检查点。XGBoost 每个内层折训练一个 XGBClassifier，仅指定运算设备和 CPU 线程数。训练参数由安装的 XGBoost 版本决定。没有网格搜索、早停或手动类别权重。内层验证用于产生折外预测并冻结阈值。Scaler 仅拟合各内层训练数据。

阈值用内层折外预测，EEGNet 最大化平衡准确率，XGBoost 最大化宏 F1。外层数据在选择完成后加载。五模型平均 epoch 概率再按受试者平均。所有外层预测汇总后计算 AUROC/AUPRC/F1/平衡准确率；单人折无法单独计算 AUROC。

SHAP 按五个内层模型分别保存到条件目录的 inner_N/outer。EEGNet 背景只来自其训练受试者，外层最多两个 EEG 窗，64 个梯度采样，产物为 shap_test.npz。XGBoost 的 outer_shap.npz 对应完整 booster 的 log-odds。两类解释量纲不同。

## 服务器命令

项目根目录，Python 3.11 eeg_pipeline 环境，安装项目及支持 CUDA 的 PyTorch/XGBoost。将路径改为服务器实际路径。

```bash
conda activate eeg_pipeline
pip install -e .
export CUDA_VISIBLE_DEVICES=0
export MPLBACKEND=Agg
python scripts/ECMAD.py prepare --data-dir /data/ds003478-1.1.0 --output-dir /data/ecmad_cache --workers 8
python scripts/prepare_ds003478_nested_features.py --cache /data/ecmad_cache --workers 8
python scripts/run_ds003478_nested_cv.py --cache /data/ecmad_cache --output /data/nested --method eegnet --outer-index 0
python scripts/run_ds003478_nested_cv.py --cache /data/ecmad_cache --output /data/nested --method xgboost --outer-index 0
python scripts/summarize_ds003478_nested_cv.py --cache /data/ecmad_cache --output /data/nested
```

XGBoost 默认使用 CUDA，也可显式传入 `--device cpu`。外层清单记录实际设备和 XGBoost 版本，换设备或版本应使用独立结果目录。EEGNet 仍要求 CUDA。

可以直接复制正式 epochs/features/recordings.csv/run_config.json 缓存到服务器。outer-index 范围 0–105，按 participant_id 排序。使用 Slurm array=0-105，每任务一块 GPU，传入 --outer-index "$SLURM_ARRAY_TASK_ID"。两个架构分别提交。--condition 可限定单个条件。每个方法的同一外层折有独占锁，完成条件跳过，EEGNet 可恢复轮次，XGBoost 按内层折恢复，当前未完成的单次拟合重跑。配置或脚本指纹不同需新目录。

预算为 EEGNet 和 XGBoost 各 6,890 次基础拟合，另需 SHAP 和中断后的必要重拟合。建议先在服务器跑一折测量资源成本。

本地只做语法检查，未执行新训练和 CUDA 验证。


## 断点恢复规则

重复原命令并保持 --cache、--output、--method、--outer-index 和配置一致，即可恢复。无需额外 resume 开关。Slurm 超时后重新提交对应 array task 即可。一个外层折有文件锁，进程退出后由操作系统释放。

XGBoost 在 inner_N 下仅落盘验证预测和校验标记，训练时临时保留模型于内存。重启时复用验证预测，并仅对缺少有效外层解释且未持有模型的内层折重新拟合。旧网格搜索的外层清单不能用于默认参数协议，已有旧 XGBoost 结果时，使用同一缓存和新的 --output 目录。

EEGNet 的 last_training.pt 和 best_model.pt 作为临时断点，支持逐 epoch 恢复。对应外层预测与 SHAP 校验完成后自动删除。完成的实验保留预测、指标、SHAP、划分和验证记录。更新服务器脚本前应先结束旧训练进程，再用原命令和原结果目录恢复。

外层解释保存到 inner_N/outer，包含该模型的预测和 SHAP。完成标记只在全部文件写完后生成。重启跳过完整模型的解释，中途退出只重做当前模型的解释。EEGNet 外层最多两个 epoch，所以无需进一步拆分 SHAP 批次。结果 result_complete.json 校验最终结果和所有外层解释；汇总入口拒绝未完成或损坏的结果。

缓存文件路径、大小、mtime 和依赖脚本指纹写入外层清单。修改配置、脚本或缓存后恢复会拒绝混用。迁移服务器时保留缓存文件时间戳，例如 rsync -a，并使用同一有效配置。CUDA 算子可能有非确定性，断点机制保证状态恢复，不承诺跨硬件逐位相同。

本地运行了模拟中断、完成跳过和文件损坏检测单元测试，未启动实际 EEG 训练。CUDA 端到端恢复仍需服务器验证。

## 跨折 SHAP 汇总

上面的汇总命令在完整实验目录中运行即可，不需要 GPU、模型权重或重新训练。它读取每个外层受试者的五个 `inner_N/outer/` 归因数组和窗口索引，并验证输出哈希、受试者和标签、窗数、归因形状及 EEGNet 通道顺序。已有的 `*_oof.csv` 和 `comparison.csv` 仍会输出，另增加：

- `shap_importance.csv`：各方法和条件的通道（EEGNet）或特征（XGBoost）排名。
- `shap_components.csv`：各信号成分的归因总量及组内占比。单成分组只有一个成分；拼接组分别列出 specific 与 coherent。
- `shap_xgboost_bands.csv`：按成分和频带分组的特征归因。带功率、不对称和连接特征按名称标记的频带纳入，其余列为 `non_band`。

先在每个模型内对窗（EEGNet 另对时间点）平均绝对 SHAP，再等权平均该受试者的五个模型，最后等权平均所有受试者。`share_of_condition` 是该组各项平均绝对 SHAP 的和占同条件总量的比例，总量为零时留空。两种模型分别使用概率和 log-odds 量纲。只传回原分类汇总 CSV 时，需在服务器原目录运行新版汇总，或将原始归因文件及其结果清单传回后再运行。
