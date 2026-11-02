# Codex Phase 07 — 合法干预与因果世界模型反事实验证

先读 `DATA_CONTRACTS.md` 中的 `interventions.jsonl`、`PROJECT_SPEC.md` 的 T3 门槛，检查 runner 是否实际存在。明确：**相关性结构/attention 不是已识别因果效应；模型模拟失败不是软件真实失败。**

设计和实现：
1. `interventions/runner.py` 仅通过可执行且可复现的操作进行干预：合法测试输入改动、合法产品配置变化（作为独立产品评估，不混为同一产品对照）、有类型检查与可达性验证的状态值替换。对已通过测试系统优先使用测试输入改动与变形测试；状态替换必须验证类型/边界且说明这是人工干预而非天然软件行为。
2. 强制程序行为一致性控制：相同 bug version、相同原始程序、不修改缺陷本身；不同配置变化需注意改变源码组合导致潜在混杂，不能直接当作只改变一个状态变量的 `do`。
3. 保存 before/after 的真实事件、DDG、branch、assertion/output，和 `interventions.jsonl`。区分 runner fail、超时、配置非法、真正的测试断言失败。
4. 候选选择可用 FAIL 产品可疑传播路径、未观察到的分支条件、断言影响链；不得通过 `X.mutant.log` 直接选 true bug line，也不得访问人工构造 FP 所移除的失败测试内容。
5. 使用 TRAIN 真实干预对训练（若有足够样本）`L_cf`；在完全独立 held-out interventions 上比较世界模型预测的 affected nodes、next-state delta、branch flips、output changes 与实际执行结果。提供 negative control（理论无影响路径），并报告有效干预比例与失配案例。
6. 若没有 runner / oracle：只实现接口和 mock/synthetic test，输出 `T3 unavailable` 与所需文件/Java API，停止，不将纯神经 graph ablation 叫成 counterfactual proof。

验收：至少一个真实的、受控、重复可执行的干预实验和输出校验（没有数据则明确未满足）；`outputs/counterfactual_report.md` 含 action、cost、effect、validity、真实日志；更新 phase_status。不得提前实现 RL。
