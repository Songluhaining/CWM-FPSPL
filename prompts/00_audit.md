# Codex Phase 00 — 只读审计：原始 SPL/CLAP/PULP 数据与世界模型可行性

你是研究工程师，研究软件产品线假通过产品（FP）识别。请先阅读根目录 `AGENTS.md`, `PROJECT_SPEC.md`, `DATA_PREPARATION.md`, `DATA_CONTRACTS.md`, `EVALUATION_PROTOCOL.md`, `STAGES.md`，以及 `references/` 的两篇论文。严格区分论文原始方法与本项目提出的假设。

**本阶段不要训练任何模型，不要批量跑 Java 测试，不要更改 third_party 或 data/raw 中任何文件。**

任务：
1. 只读扫描第三方 CLAP/PULP 仓库。给出 label 生成、特征提取、训练、评估和 FL 下游调用的实际入口及文件格式，尤其是 `PULP.py`, `consistent_testing_manager/LabelData.py`, `fp_detection/`, `spectrum_manager/`。若本地仓库缺失，明确记录未下载状态，不要凭论文猜测现存路径。
2. 扫描 SPL 基准 V2 的实际文件结构：系统、bug 版本、产品、测试、`coverage/*.xml`、`config.report.csv.done`、feature model、组合源代码、bug 元信息，并核对 CLAP/PULP 823 版本子集的选取办法。统计每项实际数量，注明缺失/歧义。
3. 搜索用户已经有的 Java 插桩和 LDDG 工程（只有本地可见工程才可审计）。判断是否有事件顺序、动态 def-use、值、控制分支、断言输出、合法干预执行接口；列出实际字段和缺失字段。**coverage 并不等于 DDG。**
4. 检查运行环境：Python/JDK、Git、CPU、CUDA/GPU、PyTorch/PyG 是否可用（仅检测，不全局 pip 安装）。旧 PULP/CLAP 保持独立复现环境。
5. 对应 `PROJECT_SPEC.md` 的 T0/T1/T2/T3 数据级别，给出当前能真实支持哪一级。提出最小实现路径：已有代码可复用的位置、新增 Java 插桩必要工作以及量化工作量的不确定因素。
6. 给出明确的标签泄漏风险清单：oracle 缺陷行、移除的失败测试、模拟 FP 来源、同一 buggy version 的多产品、静态特征文件中可能隐藏的 ground truth 信息。

产出：写 `outputs/audit.md`，包含：已发现路径/数量/字段表、调用入口/命令、能力矩阵、数据缺口、潜在泄漏点、phase 01 所需具体文件；写或更新 `outputs/phase_status.md`。保留所有实际 shell 检查命令及运行结果，不伪造数据。只允许创建/更新上述审计文档，不改任何代码。结束时仅报告可靠的证据和阻塞项，不自动开始下一阶段。
