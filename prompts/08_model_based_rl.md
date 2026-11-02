# Codex Phase 08 — 可选模型驱动 RL（前置条件：T3 + 可靠世界模型）

在开始任何 RL 代码前，先审计 Phase 05/07：是否有真实干预执行器、oracle、合规候选动作与 held-out counterfactual fidelity？若缺少则写 `outputs/rl_not_ready.md` 并结束，禁止以自模拟奖励创造论文结果。

目标：在固定测试执行预算内，最大化原先 PASS 产品隐藏故障的可验证发现率与最终 FP 判别质量。POMDP：隐藏的程序缺陷状态不可直接观察，模型 belief 包含图编码、FP 风险、预测失败分布、测试覆盖与模型不确定度。

实现：
1. `env.py`: 状态 `b_t=(graph_embedding, prior_tests, fp_risk, uncertainty, remaining_budget, legal_action_mask)`；动作从实际可执行的合法候选集选择，例如 `run_valid_test`, `select_dependency_subgraph`, `perform_valid_value_intervention`, `stop`。不同动作真实成本由 runner 统计。
2. 环境动作接口只接受配置、合法 input 或真实干预，严格封锁隐藏 removed failing tests、oracle bug line、TEST labels。未知的结果必须等待 runner 或模拟预测，不能由真实 label 决定。
3. 奖励：`+alpha*verified_new_failure + beta*calibrated_information_gain - lambda*measured_cost - mu*invalid_action`；真实失败须有可信测试 oracle。world-model hallucination 给予 conservative uncertainty penalty。停止动作考虑预算与校准终态识别性能。
4. 先做 MPC/贪心 world-model action selection、random、coverage gain、uncertainty sampling 作为 baselines；只有多步选择相对贪心方案存在收益时再考虑 PPO/Actor-Critic。所有策略拥有相同 action space、预算和候选生成器。
5. World model 可用于规划/rollout，但必须定期与真实执行比对，用真实训练轨迹纠正模型偏差；不能在 eval TEST 标签上训练策略或重复调整超参。
6. 结果给出不同预算 K 下的 verified discovery@K、FP F1@K、cost、illegal action rate、time、std/CI；分开统计 original natural vs simulated FP。对模拟 FP，严禁直接拿训练时移除的失败测试做候选动作。

测试：toy POMDP 环境的最优策略对照；真实少量 product 执行日志可追踪；行动成本=真实调用数。交付 `src/cwm_fp/policy/`, `outputs/active_policy_report.md`、config、tests、phase_status。只报告真正完成的实验。
