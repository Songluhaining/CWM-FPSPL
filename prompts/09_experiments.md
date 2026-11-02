# Codex Phase 09 — 完整论文级评估与可复现打包

阅读所有协议、phase 状态与实际 checkpoint/log。不要主动补造缺失数据，不要使用先前论文的指标填本项目数据。

请完成：
1. 生成以版本为组的 within-system 训练/测试结果，以及六折 Leave-One-System-Out。检查每折训练/验证/测试的唯一 group 完全不重叠，并检测预训练特征、统计量、可疑度是否泄漏。
2. 分析 FP precision/recall/F1、PR-AUC、balanced accuracy、ROC-AUC、ECE，macro/micro，按系统和 1/2/3 bug 划分。按 `origin=natural|simulated|unknown` 划分，注明 origin 是否真实可核对。
3. 与 CLAP supervised、PULP weak-supervision、supervised tabular features、graph-only、world-model joint、causal intervention 以及 (若有) model-based RL 公平比较；论文结果仅作 reported reference，和实际复现运行要分栏。
4. 世界模型评估：一步/多步预测、数值状态 MAE/relative error、分支 F1、输出预测、跨版本/跨系统性能，intervention effects 必须来自实际 rerun。展示失败案例，尤其是不可执行缺陷、值不可观测、长距离传播、配置信息不足。
5. 完整消融：CDG、DDG、配置边、value nodes、world-model auxiliary loss、intervention loss、label fraction、masking、测试质量特征，预设与消融方向一致的同预算实验。
6. 统计检验按 buggy version 作分组 bootstrap，多个随机 seed；报告置信区间、时间、峰值显存、插桩开销、图抽取失败率、测试等价验证结果。不可只以 product 数为独立样本数计算显著性。
7. 生成 `outputs/paper_tables/`, `outputs/figures/`，一个 `reproduce.sh`（只在用户显式运行才触发大实验）和 `outputs/reproducibility.md`（硬件、JDK/Python、Git commit、依赖锁定、随机种子、已知故障）；提供论文 Results/Threats of Validity 段落的 American English LaTeX 草稿，但不得虚构结果。
8. 给出总结：哪些研究假设得到数据支持、哪些未得到支持、哪些因没有干预观测而无法证明因果识别、是否有足够证据把方法称为 causal world model。

最后执行已有 test/linters，`git diff --check`，记录所有实际运行命令和完整输出路径。结束本阶段，不要做未授权部署。
