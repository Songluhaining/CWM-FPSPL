# Non-Leaking Evaluation Protocol

## Existing-model matched holdout evaluation (EVAL1, 2026-11-02)

The user authorized evaluating the existing runnable graph classifier against baselines. `matched_04R1_holdout_v1` retains all 28 products and the original 17/5/6 split. Nine already selected neural checkpoints and two frozen statistical models are restored without fitting or reselection. Their old validation outputs must reproduce within a predeclared absolute tolerance of 1e-6; parameters must remain unchanged.

Four supervised tabular controls use the same 17 PASS labels and 5 validation / 6 test products, with BASE4 version-context features and its fixed C grid. Same labeled cohort does not mean identical context: CLAP/PULP features may use observable failures in target versions, and their TRAIN scaler includes 91 original F references. Exact17 source PU cannot sample a negative at its unchanged 5% ratio; a separately declared comparison uses 215 unlabeled PASS and 91 F from the same nine TRAIN versions, never held-out versions. Do not merge this information difference into an architecture-only claim.

All 26 model/seed prediction groups are sealed before TEST labels are joined by the evaluator. Prior Bank/Email development and BASE4 overlap are disclosed. Six test products from three versions, with only one FP, are a limited pilot and not a four-system or causal-world-model evaluation. Current R-GCN test F1 is 0.6667 versus 1.0000 for CLAP and PULP50-SVM on that cohort. No post-test threshold adjustment, new fit, or checkpoint reselection is allowed within this run. Details and actual evidence: `outputs/phaseEVAL1_review_packet.md`.

## Four-system baseline execution (BASE4, 2026-11-02)

The tabular baseline portion has now executed on all 178 single-fault versions and 4206 products, without base-feature or single-class exclusion. This does not mean the unified graph/world model comparison has executed. Frozen artifacts and actual outcomes are documented in `outputs/phaseBASE4_review_packet.md`.

The named regime `version_context_v1` allows each target version's observed original test outcomes, coverage and static slices to construct its own reference context. It never uses target FP/TP truth. Version identities do not share reference state. Spectrum statistics use only original failing products within that version. Classifier/scaler fitting uses TRAIN only, regularization selection uses VAL only, and held-out predictions are sealed before TEST evaluation. It is not a strict no-target-context inductive experiment. New version-grouped splits use seed 20261102 and proportions 70/15/15; whole-system holdouts split source versions 80/20. Historical splits remain unchanged.

Five fixed method families and three seeds (20261102/03/04) yield 75 selected evaluations. Three tunable tabular methods use C in {0.1, 1, 10}, with ties selecting the first ascending C on validation FP F1; source CLAP uses fixed linear SVM C=1. Source PULP retains its 30-estimator, 5%-sampled-negative, accuracy-weighted implementation. This source algorithm differs from the paper's stated balanced-sampling/out-of-bag description; do not label it as an exact paper reproduction.

A separate `source_pool_transductive_v1` run fits original PULP F/U learning to the whole observable pool, with per-version feature normalization, without FP/TP fitting targets. Its results are separate from the holdout ranking. Original FAIL products remain F; only the 2756 original PASS products are FP/TP targets. No failing tests are removed. Undefined single-class areas are null. Three-seed variation and 200-replicate version-cluster confidence intervals are reported; seeds do not represent independent dataset splits. No thresholds or hyperparameters are adjusted after test scores are opened.

## Four-system design context

The active first comparison is limited to BankAccountTP, Email, Elevator and ExamDB single-fault data. See `outputs/phaseDESIGN1_review_packet.md`, section 7. It specifies a new version-grouped protocol and four whole-system holdout folds; old frozen splits are not rewritten. Match CLAP/PULP benchmark labels to the actual suite view rather than designing new labels.

Report strict inductive and target-version-context baseline regimes separately. Original CLAP/PULP feature construction may use observable failing products in the target version; silently removing that information and filling features with zeros is not an original-method reproduction. Target FP/TP truth never becomes a feature or fitting input. Freeze the method across outer folds before seeing target scores, and disclose prior BankAccountTP/Email development exposure.

The unified graph/world-model comparison has not run. BASE4 tabular experiments and the separate EVAL1 existing-classifier pilot are described above. Historical sections below retain their original scope and measurements.

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
