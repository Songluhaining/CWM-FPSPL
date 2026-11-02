# Codex One-shot ORIENTATION Prompt

我正在研究“Generalizable False-Passing Product Detection in Software Product Lines via Causal World Models”。本目录是项目入口。请按如下规则工作：

1. 阅读根目录 `AGENTS.md`, `PROJECT_SPEC.md`, `DATA_PREPARATION.md`, `DATA_CONTRACTS.md`, `EVALUATION_PROTOCOL.md`, `INSTRUMENTATION_PLAN.md`, `STAGES.md`，两篇 `references/*.pdf`（使用可用 PDF 阅读工具），以及 `prompts/00_audit.md`。
2. 我的目标是有监督学习驱动的**可跨缺陷/跨系统泛化的结构化因果世界模型**。核心输入为真实的 LDDG（静态控制依赖、动态数据依赖、SPL 特征配置、测试事件与值）。首先实现严格分组的 supervised baselines 和图分类器，再做状态/事件转换学习，验证可用后才做真实反事实干预与模型驱动 RL。
3. 先进行 Phase 00 的**只读审计**，不要一口气实现所有阶段。准确识别 CLAP/PULP 数据、823-version 子集、`variant_labels.csv`、官方 V2 基准、instrumenter 的真实数据能力。不把覆盖率冒充动态 DDG，不把模型模拟当真实因果效应，不让模型利用真实故障位置、被移除失败测试或 TEST 的 FP/TP 标签。
4. 用实际文件/命令/日志证明完成了什么。如果没有原始基准或 instrumenter，就明确写 `blocked`，但可针对合成样本完成数据契约验证；绝不虚构真实实验结果。
5. Phase 00 结束时只给出 `outputs/audit.md`, `outputs/phase_status.md` 和 phase 01 的具体输入清单，不擅自开始下一阶段。
