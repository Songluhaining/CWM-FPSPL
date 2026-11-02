# Codex Phase 05 — 局部程序因果/结构世界模型（核心）

先阅读根目录 `PROJECT_SPEC.md`, `DATA_CONTRACTS.md`, `EVALUATION_PROTOCOL.md`，并检查 phase 03 的实际能力等级和 phase 04 的真实 baseline。请先写一页设计决策：具体可用的状态、事件、控制/数据依赖和有效目标是否足以支持拟用损失。世界模型不等于将 GNN 重新命名。

主要科学目标：条件化学习 `p_theta(z_{t+1}, branch_{t+1}, output_{t+1} | z_t, action_t, feature_config, local_dependency_graph)`，使模型可跨配置/缺陷迁移，而不是记忆 bug_version ID。

实现：
1. `src/cwm_fp/models/world_model.py`: 关系图编码器（可复用阶段 04）、局部动态事件编码器、时间递推单元（GRU/Transformer）、用于 next-state/branch/output/next-event 的预测头。给出每个头的 mask/target 数据规范。
2. 若数据 T2 有实际运行值：按类型正确处理数值、bool、null、对象摘要、未知值；训练一步预测和有限多步 rollout。对外部复杂对象不要求重建全部字节，只预测可观测且有定义的状态摘要；不能填补缺失值为真实值。
3. 若只有 T1（真实结构图但没有动态值）：降级为**结构/事件转移预测模型**，预测下一条语句、依赖边/分支候选和输出可达性，不得宣称预测了数值程序状态；把 `L_value=disabled` 和原因明确打印进 config、metrics。
4. 若只有 T0：阻止运行世界模型，提示必须先获取 T1/T2，绝不以静态切片或覆盖图冒充真实动态因果图。
5. 训练损失包括 `L_next_value`（T2）、`L_branch`（有真实 branch）、`L_next_event`、`L_output`（有 oracle/observed output）、`L_relation`。在可用的实际 targets 上训练，不从 FP 标签中提取真实 bug 位置当输入。
6. 训练只使用 TRAIN 版本（可包括 F 的真实失败轨迹与 PASS 的真实通过轨迹）；VAL/TST 的轨迹不能参与世界模型参数学习或标准化。实验中另做 teacher-forcing 与 rollout（1/5/10 步）性能、branch F1、output accuracy、transition NLL/MAE、calibration、OOD cross-version 测试。
7. 实现置信度估计，例如 ensemble 或校准，记录失败场景（长期累积误差、罕见分支、未见特征组合）。本阶段只学习实际观察到的机制，**不直接将 CDG/DDG 当已被证实的因果 SCM**。
8. 采用小图过拟合、typed value、循环、mask、顺序打乱 sanity tests，比较 graph-only encoder 与 added transition loss 的差异，保存真实 checkpoint 和 seed。

产出：训练/评估 CLI、config、checkpoint 结构、JSON metrics、明确的数据 T1/T2 限制、`outputs/world_model_eval.md` 及 phase_status。不要本阶段实现策略优化或声称干预因果效应。
