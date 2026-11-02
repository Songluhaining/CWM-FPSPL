# Codex Phase 01 — 标准化数据契约、监督标签隔离和数据集划分

阅读 `AGENTS.md`, `DATA_CONTRACTS.md`, `EVALUATION_PROTOCOL.md`, `outputs/audit.md`。优先复用 CLAP/PULP 真实标签来源和 SPL benchmark mapping，禁止以猜测代替源数据。

在 `src/cwm_fp/data/` 实现：
1. `benchmark_adapter.py`: 只读解析真实基准目录，输出标准化 `products.csv`, `tests.csv` 和路径/数据来源审计报告。若基准存在 version 名/系统别名，维护显式 alias 表。声明所有样本的唯一键 `(system_id,bug_version_id,product_id)`，不靠目录排序推断关系。
2. `labels.py`: 将 PULP `variant_labels.csv` 或已核对的 CLAP 标注导入独立 `data/processed/labels/`；必须标注 label source；在 model/inference 数据加载器中默认不返回 label、origin、oracle bug line、removed failed tests。未知的自然/模拟 origin 保留 `unknown`。
3. `splits.py`: 同一 `(system_id,bug_version_id)` 全部产品和测试严格同组；提供固定种子 version-wise train/val/test 和 Leave-One-System-Out 共 6 折；split 文件可重复生成且持久化。只允许 TRAIN 数据计算归一化、词典与可疑度先验，VAL 做早停，不得通过 TEST 信息构建学习特征。
4. 用 `scripts/validate_contract.py` 增强实际数据校验；输出按系统统计的 F/FP/TP、版本数、自然/模拟/未知 origin、缺失覆盖/测试，以及每个 split 的数量，列出与论文统计值差异及理由。
5. 提供小规模 smoke 模式（1 个 BankAccountTP buggy version + 几个产品）并加入 pytest/unittest 断言无 version overlap、无重复 ID、标签与原始结果一致。

禁止：训练模型；自行将 test suite 数量下降作为模拟 FP 判定；把 `X.mutant.log` 作为输入特征；偷偷读取 TEST 标签来选择特征或超参。若真实数据不足，先跑本包 `data/examples` 的合成 schema 测试，并报告真实数据待补充。

产出：明确 CLI、`data/processed/manifests/*.csv`、split CSV、分组和标签审计 JSON、测试。执行最小命令展示真实结果并在 `outputs/phase_status.md` 记录。停止，不进入阶段 02。
