# Codex Phase 04 — 严格监督式 LDDG 图分类基线

阅读 `AGENTS.md`, `PROJECT_SPEC.md`, `DATA_CONTRACTS.md`, `EVALUATION_PROTOCOL.md`，并读取 phase 02/03 的真实结果。

目标：先检验动态依赖图本身是否优于 PULP/CLAP，不要先加入复杂因果或 RL 模型。

实现：
1. 将每测试图编码为 PyTorch Geometric HeteroData 或明确的自定义张量，区分 feature/statement/value/output 节点及不同边类型；per-test pooling → per-product pooling。较大图采用有限窗口、局部子图或采样，记录丢失的边/事件比例。
2. 先实现 coverage-only MLP、type-agnostic GNN、关系 GCN（R-GCN），如资源允许再实现 HGT。禁止把 system_id、bug_version_id、product_id 原始字符串直接当模型分类特征；可用显式符号/语义类型与受约束的特征配置状态。
3. 只用 TRAIN PASS 产品标签 FP=1、TP=0 训练二分类头；失败产品 F 仅作为失败证据图或辅助任务输入。特征归一化等只基于 train，val 早停，测试只评估一次。
4. 控制参量规模，报告显存、训练时间、epoch、类别权重和 PR-AUC；在同样 split 上对比表格基线。
5. 实现归因可视化（受关注语句/依赖边），但不要称 attention weights 为 causal effect；实验消融 coverage-only / CDG-only / DDG-only / both。
6. 通过合成 graph smoke test 跑通前向/反向与小规模过拟合检查，再跑真实数据。训练时间必须有边界；不可伪造运行结果。

产出：`src/cwm_fp/models/graph_encoder.py`、`graph_classifier.py`、`train_graph.py`、config、测试、输出与误差分析。更新 `outputs/phase_status.md`，终止本阶段。
