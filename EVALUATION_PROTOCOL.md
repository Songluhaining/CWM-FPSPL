# Non-Leaking Evaluation Protocol

## Current four-system design (not yet executed)

The active first comparison is limited to BankAccountTP, Email, Elevator and ExamDB single-fault data. See `outputs/phaseDESIGN1_review_packet.md`, section 7. It specifies a new version-grouped protocol and four whole-system holdout folds; old frozen splits are not rewritten. Match CLAP/PULP benchmark labels to the actual suite view rather than designing new labels.

Report strict inductive and target-version-context baseline regimes separately. Original CLAP/PULP feature construction may use observable failing products in the target version; silently removing that information and filling features with zeros is not an original-method reproduction. Target FP/TP truth never becomes a feature or fitting input. Freeze the method across outer folds before seeing target scores, and disclose prior BankAccountTP/Email development exposure.

New comparative experiments have not run. Historical sections below retain their original scope and measurements.

## 阶段02实际执行范围

2026-10-31已完成本地两系统109个单缺陷版本的`strict_train_reference_v1`：一套按版本分组、两套完整系统留出；每套固定划分上执行三个训练随机种子。词典、可疑度、失败参考池及归一化仅使用训练集合；验证选择C后冻结预测，再由评价进程连接标签。一次最终评价限制按运行目录执行。标签资格仍为本地未独立核验的代理标签。

论文原始CLAP五折／六系统留出及PULP版本内／跨系统汇总协议分别保留为未完成，不能将当前适配结果冒称论文复现。PULP保留本地原源码5%抽样及训练准确率加权，论文等量抽样／未抽中轮次聚合差异明确记录。缺失四系统、双／三缺陷、模拟隔离套件的结果不可用。

基线同时报告平均精确率与梯形PR-AUC。跨系统五十维缺参考时成为常量，此时梯形面积会因端点插值较高，应结合平均精确率、ROC-AUC与缺证据记录解释。真实运行详情见`outputs/baselines_reproduction.md`。下述图模型、状态转换、干预和策略对照均留待对应阶段，并未因阶段02完成而自动启动。

## Partitions
- Primary: grouped by `(system_id, bug_version_id)`; all variants/tests/traces from one version in the SAME partition. For within-system, seeded train/val/test about 70/15/15 or equivalent after group balancing; no test-set tuning.
- Strong transfer: leave-one-system-out (LOSO), 6 runs; hold out test system with no supervised or self-supervised pretraining access unless a *separate and explicit* transductive experiment.
- Keep explicit `split_id` persisted; train-only normalization, vocabulary, embeddings, graph-statistics and thresholds, validation-only early stopping; final test only once per agreed experimental run.
- If PULP/CLAP original results use another split protocol, report both original-paper comparable setup and strict version/LOSO setup. Never compare incomparable protocols as a single head-to-head win.

## Metrics
FP precision, FP recall, FP F1, PR-AUC, ROC-AUC, balanced accuracy, accuracy; confusion matrix, calibrated Brier/ECE where relevant. Metrics calculated among observed PASS products only. Report bootstrap confidence intervals clustered by `bug_version_id`; report each of six systems and micro/macro average. Breakdown: natural vs simulated FP, 1-/2-/3-bug versions, test coverage strata, graph availability tier.

## Baselines
- CLAP supervised (source-original settings when possible) and a tuned supervised tabular model on PULP 50D features.
- PULP PU Bagging under original protocol and strict-split adaptation, with disclosure.
- Structural controls: coverage-only, static control graph, dynamic data graph, both, type-agnostic GNN, relational GNN/HGT supervised graph classifier.
- For world model: joint supervised classifier with and without next-step/branch/output training; with and without true intervention pairs.
- For active selection: random, coverage-gain, entropy/uncertainty, world-model greedy, model-based RL under same legal action set and cost budget.

## Intervention evaluation
Test-world model on held-out REAL before/after reruns, not only simulated counterfactual loss. Compare predicted and observed branch changes, state transitions, output changes; include validity rate and calibration/uncertainty. Do not count a model's `predicted failure` as actual fault revealed. If test oracle is unavailable, report change/coverage only, not failure discovery.

## FP label leakage and constructed FPs
- All original tests removed to construct simulated FPs belong in a sealed `oracle/` location that the training/inference process cannot read or select. Do not use directly as newly discovered failures in an active policy experiment.
- Label `origin` never enters classifier/world-model features; only post-hoc stratified evaluation.
- Include trivial baseline using test suite size/coverage to diagnose shortcuts; optionally evaluate with those features removed.

## Reproducibility
Persist manifest SHA256, source git hashes, split seed, config, conda/pip freeze, JDK version, machine and actual execution time. Checkpoint selection based on validation only. Fail loudly if splits overlap or a forbidden field appears in a model input schema. Run multiple seeds and variance estimates.
