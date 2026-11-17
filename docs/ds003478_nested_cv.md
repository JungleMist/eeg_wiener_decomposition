# ds003478 nested CV

外层按受试者 LOOCV，106 折；内层 StratifiedKFold，k=5，seed=42。原固定三个分区合并，16 名排除受试者继续排除。两次记录跟随受试者。其余预处理、13 条件、特征及 TUEP 参数继承 configs/ECMAD.yaml。

采用五模型集成。EEGNet 保留各内层验证 AUPRC 最优检查点。XGBoost 在原网格每个候选上训练五个模型，500 轮上限、30 轮早停，按平均内层受试者 AUPRC 选择候选，保留五个模型。内层同时承担早停和超参数选择，外层独立测试评估整个选择流程。Scaler 仅拟合各内层训练数据。

阈值用内层折外预测，EEGNet 最大化平衡准确率，XGBoost 最大化宏 F1。外层数据在选择完成后加载。五模型平均 epoch 概率再按受试者平均。所有外层预测汇总后计算 AUROC/AUPRC/F1/平衡准确率；单人折无法单独计算 AUROC。

SHAP 每个内层模型分别保存。EEGNet 背景只来自其训练受试者，外层最多两个 epoch，64 个梯度采样。外层 SHAP 位于 inner_*/shap_test.npz；复用训练函数还会生成内层验证解释，完成标记之前会生成外层解释。XGBoost 的 outer_shap.npz 对应早停树前缀的 log-odds。两类解释量纲不同。

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

可以直接复制正式 epochs/features/recordings.csv/run_config.json 缓存到服务器。outer-index 范围 0–105，按 participant_id 排序。使用 Slurm array=0-105，每任务一块 GPU，传入 --outer-index "$SLURM_ARRAY_TASK_ID"。两个架构分别提交。--condition 可限定单个条件。每个方法的同一外层折有独占锁，完成条件跳过，EEGNet 可恢复轮次，XGBoost 按候选参数和内层折恢复，当前未完成的单次拟合重跑。配置或脚本指纹不同需新目录。

预算为 EEGNet 6,890 次拟合。XGBoost 原 256 组合网格约 1,763,840 次拟合，另需 SHAP。建议先在服务器跑一折测量资源成本。

本地只做语法检查，未执行新训练和 CUDA 验证。


## 断点恢复规则

重复原命令并保持 --cache、--output、--method、--outer-index 和配置一致，即可恢复。无需额外 resume 开关。Slurm 超时后重新提交对应 array task 即可。一个外层折有文件锁，进程退出后由操作系统释放。

XGBoost 在 search/candidate_NNN/fold_N 下保存模型、scaler、验证预测，全部写完后才写带 SHA256 的 complete.json。恢复会检查哈希，复用完整折，重做缺失或损坏折。搜索仍按原顺序比较分数，确保平分时的选择规则一致。单次拟合中途退出时重跑该次拟合，上限 500 轮。保留全部候选折模型会增加磁盘占用，请在服务器首折完成后估算完整实验磁盘需求。

EEGNet 每轮原子写入 last_training.pt，包括模型、优化器、scheduler、最佳检查点、历史、训练采样器和 CPU/GPU 随机状态。中途退出最多重跑当前未保存轮次。内层完成后写 training_complete.json，验证哈希后跳过该内层训练。损坏的 torch 检查点加载会报错，需移走相应 last_training.pt 后重跑该内层折。

外层解释保存到 inner_N/outer，包含该模型的预测和 SHAP。完成标记只在全部文件写完后生成。重启跳过完整模型的解释，中途退出只重做当前模型的解释。EEGNet 外层最多两个 epoch，所以无需进一步拆分 SHAP 批次。结果 result_complete.json 校验最终结果和所有外层解释；汇总入口拒绝未完成或损坏的结果。

缓存文件路径、大小、mtime 和依赖脚本指纹写入外层清单。修改配置、脚本或缓存后恢复会拒绝混用。迁移服务器时保留缓存文件时间戳，例如 rsync -a，并使用同一有效配置。CUDA 算子可能有非确定性，断点机制保证状态恢复，不承诺跨硬件逐位相同。

本地运行了模拟中断、完成跳过和文件损坏检测单元测试，未启动实际 EEG 训练。CUDA 端到端恢复仍需服务器验证。
