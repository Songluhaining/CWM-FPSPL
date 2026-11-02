# Codex Phase 02 — 可复现的 CLAP/PULP 与监督表格基线

先阅读 `outputs/audit.md`, `DATA_CONTRACTS.md`, `EVALUATION_PROTOCOL.md`, `STAGES.md` 和 phase 01 的实际文件。目标是明确后续世界模型的性能下界，不是现在构建新模型。

任务：
1. 建立 baseline adapter，优先直接调用已确认能运行的第三方 CLAP/PULP 方法，保留原版源码与依赖在 `third_party/` 下只读。任何兼容性修改放在 adapter 或独立 patch 目录并记录 diff，不无说明升级 PULP 原始算法。
2. 复现 CLAP 有监督 FP/TP 预测与 PULP PU Bagging 的数据协议；保留论文原评测协议，同时新增严格 version split 和 LOSO 协议（如果原论文不是此协议，分别命名，不能将数字混为同一对比）。
3. 使用 train-fold 内统计构建的 PULP 50D 特征（如果实际数据可用），训练 Logistic Regression、linear SVM、可选 LightGBM/XGBoost 作为监督基线。标签仅为原本 PASS 的 FP vs TP；原本失败的 F 是证据库样本，不作为 FP/TP accuracy 的第三类混合计数。
4. 标准化输出每个 PASS 产品的 `sample_id,split_id,y_true,y_score,y_pred,system_id,bug_version_id,origin` 到 **仅 evaluator 可读的** `outputs/metrics/`；不把 eval 标签传入模型调用。输出 FP precision/recall/F1、PR-AUC、balanced accuracy、ROC-AUC 等，并说明未能计算的项目。
5. 增加禁止 oracle bug metadata、label/origin leakage 的测试；报告 natural/simulated 两个子集，不把构造方式特征用作预测输入。

禁止使用公开论文中的报告数字作为本次脚本实验输出；公开数字仅能放在 comparison reference 表中。任何复现失败都需要给出明确异常与路径。

产出：`src/cwm_fp/baselines/`, `configs/baselines.yaml`, CLI、test、结果 schema、`outputs/baselines_reproduction.md`，更新 phase_status。不要提前实现 GNN/RL。
