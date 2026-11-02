# Codex Phase 03 — 从实际执行证据构造 LDDG

阅读 `PROJECT_SPEC.md`, `DATA_CONTRACTS.md`, `outputs/audit.md`, `DATA_PREPARATION.md`, `INSTRUMENTATION_PLAN.md`。重点：静态 CDG + 实际动态 DDG + SPL 特征控制约束。图用于后续训练，因此可追溯性优先于算法花哨。

必须先输出数据能力判断：T0 only, T1 GRAPH, T2 VALUE or T3 INTERVENTION。用户已有插桩器时先复用和适配，不要无端改写生产者。

分步：
1. 从每个 Java 组合产品映射静态控制依赖；与合法 feature model / FeatureHouse `roles.meta` 的 feature-to-statement 映射集成。静态 CFG 不等于控制依赖图；保留映射不确定性。
2. 从插桩器获取每个测试的有序事件。为每个动态定义赋唯一值版本，对真实 reads 找到运行时 reaching-definition，形成 `data_dep`。跨线程需显式 happens-before，若无法确定则记录不确定或禁用。
3. 不同 buggy version / variant 的语句统一为 original feature source `(feature,class,method,line,ordinal)`；保存 composed `(class,line)` ↔ original 的映射及无法匹配计数。循环和重复方法调用不能丢事件顺序。
4. 建立 schema 中异构节点 (`feature`, `statement`, `value`, `assertion`, `output`) 与边 (`feature_controls`, `control_dep`, `data_dep`, `produces`, `consumes`, `observes`, `temporal`)；标明静态/动态证据来源。
5. 测试用例：赋值链 x→y→z；分支真假；循环多次写；不同 feature 的变体；断言输出；缺失值；特别测试仅覆盖 XML 不得产生伪装的真实 DDG。
6. Java 8 插桩必须验证 original vs instrumented 的测试 pass/fail 一致性（允许性能开销，但不能修改结果），对于编译器优化掉的源码行应显式标为 non-executable/unmapped，而非补一条虚构边。
7. 限制运行量：先一个 BankAccountTP 小样本，设置单测试超时、事件数与文件大小上限；给出失败日志并支持恢复。

若没有运行值，只产生 T1 graph，不在报告中宣称 state/value world model 已具备训练数据；绝不从 coverage 推断实际 def-use。输出 `graphs.jsonl`, `events.jsonl`（如果可用）、source map、extractor version、quality report、最小测试命令、`outputs/phase_status.md`。停止。
