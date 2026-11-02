# Codex Phase 06 — 世界模型与有监督 FP/TP 识别联合训练

请阅读 `PROJECT_SPEC.md`, `EVALUATION_PROTOCOL.md`，以及 phase 04/05 的实际训练结果与数据契约。目标：验证世界模型的辅助状态转换目标是否对未见缺陷/系统的假通过识别有**真实增益**。

实现：
1. 在经训练的局部状态/结构世界模型上添加 product-level 分类头：test graph embeddings → temporal/attention pooling → product-level representation → sigmoid FP risk。分类训练仅对 `observed_result=PASS` 的 TRAIN 样本采用 FP=1、TP=0 标签。
2. Failing products 可作为共享失败行为 prototype、对比学习 anchor 或世界模型转移任务训练样本；必须在消融时独立去除这种利用，避免 F 被错误当作 FP 标签。
3. 联合损失 `L_total = λ_F L_FP + Σ λ_i L_world_i`。先训练仅分类头、仅预训练/冻结世界模型、端到端联合训练三个版本；超参数全部使用 VAL 选择，固定数据划分与相近训练预算进行公平比较。
4. 类别不均衡：提供 class weights 或 balanced sampler。指标 FP precision/recall/F1、PR-AUC、Brier/ECE（概率校准）以及 per-system、per-bug-count、natural/simulated 分层；置信区间按 buggy-version 分组抽样。
5. 解释输出：高风险产品的风险来源要能分解为可疑缺陷模式重合、测试感知不足、传播链/断言观测的证据，但不得将未经干预验证的注意力热图称作 causal effect。
6. 完整消融：Tabular supervised, graph-only, graph+world-model, graph+world-model+aux-F; CDG-only, DDG-only, both; value-aware vs structural-only; no test-quality. 仅报告真实可执行的组合，不拿无法运行的组合凑表。
7. 测试危险捷径：去掉 test-suite length / coverage-quality shortcuts，看自然 FP 的泛化是否仍成立；训练和 eval 的 bug version 不重叠。保存真实预测文件、配置和 checkpoint。

产出：`src/cwm_fp/models/joint_detector.py`, `train_joint.py`, `evaluate.py`, `configs/joint.yaml`, `outputs/joint_comparison.md`, tests, phase_status。停止。
