# CWM-FP — Scientific and Engineering Specification

## Current design scope — 2026-11-02

The active design is [the four-system single-fault model and algorithm specification](outputs/phaseDESIGN1_review_packet.md), with machine-readable design choices in `configs/model_design_v1.json`. This is a design revision, not a new training result. BankAccountTP, Email, Elevator and ExamDB are in scope; preprocessing runs locally and training runs in the existing user container. Historical two-system data, partitions and results remain frozen.

The first unified model uses an observable dependency-graph prefix representation, not mandatory recurrent memory. Product classification shares an encoder with evidence-masked branch and local-result auxiliary heads. Prediction errors are not treated as defect scores. Reuse the matched CLAP/PULP benchmark labels; do not require a new labeling research project. Native baseline context and strict inductive information regimes must be reported separately. No causal-identification or intervention guarantee is inherited from a graph encoder or a dynamics paper.

The architecture below describes the broader research roadmap. The linked current design limits its first implementation; policy, counterfactual rollout and full-state losses remain inactive.

## Research question
Given a Java software product line (SPL) with multiple buggy versions, product configurations, original test suites, execution traces and TRAIN-only ground-truth labels, can a **transferable structured world model** improve detection of false-passing (FP) products over CLAP, PULP and purely discriminative graph models? Can model-guided valid interventions improve diagnosis under a bounded test budget?

## Labels and supervision
- `F`: at least one ORIGINAL test failed (observed); use as failure-evidence anchor.
- `FP`: original suite entirely passes, but the product contains a fault relevant to the studied bug(s) that could be exposed by a sufficiently effective test.
- `TP`: original suite entirely passes and lacks such an executable relevant fault, per baseline dataset oracle.
- FP/TP ground truth is derived offline from benchmark fault knowledge. This is allowed as an ordinary supervised target on TRAIN only; it must not be an input feature or an online decision oracle.
- Distinguish naturally occurring FPs and FPs constructed by removing originally failing tests. Removed tests MUST be inaccessible to the model and active-test agent during training/decision/evaluation except in a clearly marked, post hoc ground-truth audit that is never used as feedback.

## Staged hypotheses, not established results
H1: Dynamic control/data-dependence structure provides information beyond CLAP/PULP hand-crafted features.
H2: Conditional state-transition prediction improves transfer to unseen buggy versions compared with supervised graph classification.
H3: Physically rerun, legal program interventions improve counterfactual fidelity and explain masked error propagation.
H4: World-model-guided active test selection can outperform equal-budget greedy/uncertainty/coverage policies.

## Graph/data tiers
T0 COVERAGE: only config, tests, coverage XML, statement spectra. Eligible: CLAP/PULP baseline, supervised tabular/coverage classifier. NOT DDG/world model.
T1 GRAPH: per-test static control-dependence + actual dynamic def-use edges with order, mapping to consistent statements. Eligible: graph classifier, structural/latent graph transition model. Do not claim numerical program-state transition prediction.
T2 VALUE: T1 plus typed values and execution events, test inputs and observed outputs/assertions. Eligible: supervised or self-supervised state-transition world model.
T3 INTERVENTION: T2 plus executable legal interventions and matched before/after observed traces, credible oracle. Eligible: empirically grounded causal-effect training and model-based RL.

## Architecture
1. Read-only benchmark adapters: source/config/test/labels/coverage/CLAP/PULP outputs.
2. Java instrumentation (reuse existing instrumenter when available) to capture branch events, statement events, dynamic def-use, event ordering, typed value summaries, test and output events; static control-dependence graph produced separately and joined with dynamic execution at canonical statement IDs.
3. Per-test LDDG and per-product graph collection. Typed nodes: `feature`, `statement`, `value`, `assertion`, `output`. Typed edges: `feature_controls`, `control_dep`, `data_dep`, `produces`, `consumes`, `observes`, `temporal`. Include edge provenance.
4. Baselines: CLAP supervised, PULP PU with original 50D features, supervised feature classifier, R-GCN/HGT graph classifier. Baseline implementations reproducible and leakage-free.
5. Encoder: hierarchical typed graph encoder; segment-by-test, pool to product; causal/temporal edge masks; canonical schema and missing-feature masking.
6. World model: context encoder and transition `p_theta(z_{t+1}, branch_{t+1}, output_{t+1} | z_t, a_t, config, G_t)`. Teacher-forced real events; measured next-step, branch and output targets when available; uncertainty/calibration; learning mechanisms is a hypothesis, not automatic causal identification.
7. Supervised product head: score FP versus TP ONLY among observed PASS products. F products can contribute to world-model training and failure-pattern learning; do not present F as a third class in an FP-vs-TP-only detection metric.
8. Counterfactual module: legal test-input / branch-condition / state-value interventions where execution harness supports them. Pair with real reruns; otherwise label as simulation only.
9. Optional policy: world-model-guided selection from `run_valid_test`, `request_trace_subgraph`, `intervene_value` (when instrumenter supports it), `stop`. Reward verified newly failed tests, calibrated information gain minus real cost, invalid action penalty; train only after model accuracy/reliability checks.

## Losses
`L = lambda_state * L_state + lambda_branch * L_branch + lambda_output * L_output + lambda_relation * L_relation + lambda_label * L_FP + lambda_cf * L_intervention`
Every term may be enabled only if corresponding targets/evidence exist. FP classifier label training uses only train partition. Distinguish observed transitions (ordinary supervised/self-supervised traces) from explicit interventional causal loss. Condition on configuration and input, not raw product names/bug IDs.

## Research validity risks
- Many products share one buggy source version; split first by VERSION to prevent leakage. Cross-system evaluation must hold out entire system.
- Artificial FP from removing tests can produce a shortcut via suite size/coverage. Report `natural` and `simulated` subsets separately.
- Good next-state prediction alone does not prove causal identification or FP-classification improvement.
- If faults were not executed by any original test, that absence is not present as a value trace; obtain static reachable-code and feature-mapping context instead.
- Valid `do()` intervention requires an actual state/test manipulation + rerun; a graph edge deletion in a neural network is only a model ablation.

## Minimal success ladder
A. Dataset audit + non-leaking manifests, labels, version splits.
B. Baseline reproducibility on 823-version eligible subset (or honestly report partial subset).
C. True LDDGs with provenance and tested graph parser.
D. Supervised graph classifier outperforms/characterizes simple baseline.
E. State-transition model beats non-world-model graph encoder under controlled splits and verified targets.
F. Valid counterfactual reruns demonstrate predictive accuracy; optional RL yields equal-budget gains.
