# CWM-FP Repository Guidance for Codex

Read `PROJECT_SPEC.md`, `DATA_CONTRACTS.md`, `EVALUATION_PROTOCOL.md`, `INSTRUMENTATION_PLAN.md`, and `STAGES.md` before modifying code. Follow the current phase prompt in `prompts/`. The repository studies **supervised causal/structural world models for false-passing product detection in Java software product lines**. Do not silently change the scientific task.

## Invariants

1. `F`: observed failing product; `FP`: product with a relevant executable fault but all original tests pass; `TP`: original tests pass and no relevant executable fault. Use source-study definitions where ambiguities arise, document choices. Do not equate `FP` with any passing test that merely executed a suspicious statement.
2. CLAP/PULP oracle labels may supervise the FP/TP detection head **only on train partition**, and may be read for validation/test metrics by the evaluator. Never read true fault locations, mutant/patch metadata, removed failing tests, oracle labels, or label provenance as inference features.
3. Product records that share `(system_id, bug_version_id)` stay in one partition. Cross-system test holds out an entire system. Pretraining, normalization, calibration and model selection obey the same split boundaries, except for explicitly declared label-free test-time adaptation experiments.
4. Coverage is NOT a dynamic data-dependence graph. Static slicing is NOT a verified causal graph. Declare trace evidence tiers; never fabricate values, def-use edges, intervention observations, or baseline scores.
5. Record edge origin (static/dynamic), statement location mapping confidence, config constraints, events and trace provenance. If values or valid interventions are absent, disable corresponding world-model losses or intervention claims rather than creating synthetic real measurements.
6. Predicted counterfactual is NOT a verified failure. Only a real test with a credible oracle can prove a newly observed failure. Respect valid feature configurations and executable inputs.
7. Keep original SPL/CLAP/PULP trees read-only under `third_party/` and source datasets read-only under `data/raw/`. Place generated normalized data under `data/processed/`; do not edit original benchmark artifacts.
8. Never run massive test suites or mutate source datasets without an explicit bounded command and output path. Prefer small smoke tests; preserve reproducibility (seeds, versions, configs, commands, logs).
9. Maintain a plain-language `outputs/phase_status.md` after each phase with completed work, exact commands, tests, artifacts, caveats and the next phase. If data unavailable, implement schema + synthetic smoke tests and report blockage without claiming empirical results.
10. User wants publishable research: include fair baselines, uncertainty estimates, natural vs simulated FP analysis, leave-one-system-out evaluation and ablation.

## Implementation preferences

- Python 3.11 project tooling; Java 8 compatibility for the legacy benchmark and instrumenter; inspect live environment before selecting dependency versions. PyTorch + PyG are suggested, not assumed installed.
- Code under `src/cwm_fp/`, tests under `tests/`, commands via `python -m cwm_fp.<module>` or CLI entrypoints. Typed data models, clear exceptions, deterministic sample IDs.
- Each implementation phase: read-only audit -> minimal plan -> implement -> execute bounded tests -> report actual evidence. Do not replace a working baseline with an untested refactor.
- For claims about literature, identify whether grounded in provided PDFs, public resources, or an experimental hypothesis.

## 阶段总结与文档整理

- 从03B-4开始，每阶段必须在完成实际工作后交付`outputs/phase<编号>_review_packet.md`、同名JSON、`phase<编号>_product_status.csv`和`phase<编号>_graph_status.csv`。中文交接包应可独立阅读，区分工程实现、人工验证、真实数据验证和研究有效性；包含前后指标、全部拒绝原因、三种冻结划分、T1/T2/T3、时间前缀风险、实际命令/退出码/资源及最多三个后续建议。机器文件保留完整身份与证据路径，未知测量为null并解释；只经授权训练入口统计训练标签，不读取验证/测试标签进行研发决策。不得用空白交接包替代阶段实施。已有`summary.md`可作为该完整交接包的索引，避免重复正文。

- 每个实现阶段结束后，交付一份可独立阅读的中文阶段总结，并在最终回复中给出文件链接。统一使用`outputs/phase<阶段编号>_summary.md`，例如`outputs/phase03B1_summary.md`；已经有同用途总结时更新或复用，不重复新建。
- 总结说明：本阶段目标和范围、实际修改、真实验证与关键前后指标、当前进展、未完成事项和下一步条件。区分代码实现、单元检查、真实采集和正式研究实验；所有数字注明实际来源，不把历史测量写成本次重新执行。
- 继续维护`outputs/phase_status.md`作为当前状态入口；阶段提示明确要求的详细报告和机器证据仍按要求交付，最终回复优先链接阶段总结，避免让用户自行拼接多份文档。
- 每次整理时检查临时、空白、被替代且无独立用途的重复文档。删除前核对源码、测试、报告链接和冻结指纹引用；不得把历史记录、旧契约、正式结果或必要证据备份仅因过期而判为无用。不为清理目录修改冻结证据或降低校验要求。没有符合条件的文件时如实说明；清理范围或删除清单写入当次总结，不额外生成重复的清理报告。
