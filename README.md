# CWM-FP: Codex 实施包（研究设计 + 分阶段提示词 + 数据检查脚本）

本项目包含研究设计、数据契约、分阶段提示词与已完成阶段的实现。不要将小规模合成示例当成真实实验数据；实际数据范围及复现资格以各阶段报告为准。

当前已完成BankAccountTP真实图分类先导试验、有限结构转换试训及服务器GPU工程验收。已有完整图分类队列为28产品、16缺陷版本、463测试图；Email局部观测仍有缺口。已有代码与训练不等于跨系统有效性或因果机制已获证明。

当前工作是[四系统单缺陷统一模型与算法设计](outputs/phaseDESIGN1_review_packet.md)，包括理论依据、图与状态输入、辅助转换的作用及可证伪对照。范围为BankAccountTP、Email、Elevator、ExamDB；本机预处理、服务器容器训练。设计尚未变成新的统一模型实现或四系统实验结果。机器设计见 `configs/model_design_v1.json`；版本规则见 [版本管理](VERSION_CONTROL.md)。

本地历史报告和冻结数据保留在既有 `outputs/` 与数据目录，默认不发布到公开Git仓库。此仓库初始导入包含现有源码与测试，但不携带标签、原始基准、运行日志、环境和检查点；不能把克隆仓库本身当作已准备完整运行环境。

## 0. 研究方向
先实现监督式图模型与条件化状态转换世界模型，并验证跨缺陷/跨系统泛化；仅在有真实干预执行器和可靠世界模型的情况下加入模型驱动强化学习。用户提供的两篇论文已复制到 `references/`。

## 1. 建议目录
```
CWM_FP_Codex_Starter/
  AGENTS.md                       # Codex 每个阶段遵循
  PROJECT_SPEC.md                 # 研究目标、架构与定义
  DATA_PREPARATION.md             # 原始文件与环境准备
  DATA_CONTRACTS.md               # 导出图、事件、标签格式
  INSTRUMENTATION_PLAN.md         # Java 插桩与真实 DDG 对接细则
  EVALUATION_PROTOCOL.md          # 严格实验协议和基线
  STAGES.md                       # 10 个阶段及验收条件
  prompts/00_...09_...md          # 每阶段单独发给 Codex
  scripts/                        # 数据契约及拆分辅助脚本
  data/examples/                  # 仅供 smoke 测试的合成数据
  references/                     # 用户提供的两篇 PDF
  third_party/{CLAP,PULP}/        # 已登记指纹的只读Python源码快照
  data/raw/spl_benchmark_v2/      # 用户自行下载并解压基准
  data/processed/                 # Codex 后续生成
  src/cwm_fp/                     # Codex 后续实现
  outputs/                        # Codex 后续输出审计/结果
```

## 2. 在含本包的根目录安装源代码
```
mkdir -p third_party data/raw
git clone https://github.com/ttrangnguyen/CLAP.git third_party/CLAP
git clone https://github.com/dsjads/PULP.git third_party/PULP
```
基准数据：访问 https://tuanngokien.github.io/splc2021/ 下载 V2。由于实际下载路径与解压结构可能不同，先在 `configs/project.example.yaml` 中修改，再让 Codex 运行 00 阶段做只读审计。

## 3. 快速验证本包的合成数据契约
```
python scripts/validate_contract.py --products data/examples/products.example.csv --tests data/examples/tests.example.csv --graphs data/examples/graphs.example.jsonl
python scripts/make_splits.py --products data/examples/products.example.csv --out /tmp/cwm_splits.csv --strategy grouped --seed 42
python -m unittest discover -s tests -v
```
此处 synthetic 数据只有两个系统/三个 buggy versions，仅验证解析与防泄漏条件，不用于科研训练。

## 4. Codex 使用方法
- 在当前项目根目录启动 Codex，让它自动读取 `AGENTS.md`（官方支持的仓库级指引）。
- 按顺序将 `prompts/00_audit.md` 至 `prompts/09_experiments.md` 的内容发给 Codex，**每次一个阶段**。
- 可以在对话中只写：`请阅读 prompts/00_audit.md 并完成本阶段，不要开始下一阶段。`
- 每阶段要求 Codex 在 `outputs/phase_status.md` 记录实际运行的命令、文件、验收结果和失败原因。对未获取的数据不应生成假实验结果。
- 每阶段结束后交付一份独立中文总结，统一保存为`outputs/phase<阶段编号>_summary.md`，并提供文件链接。同步检查无用临时文档；有审计、复现或冻结校验用途的历史记录保留。

## 5. 环境建议
Python 3.11 + PyTorch/PyTorch Geometric（根据用户 GPU/CUDA 版本决定具体 wheel）+ pandas、scikit-learn、networkx、pyarrow、pytest；历史基准工具按 Java 8 兼容性隔离。避免对 PULP/CLAP Python 3.7/旧 Java 插件硬升级造成复现破坏。

## 6. 研究最重要的边界
- 真实监督训练只在训练版本中使用 FP/TP 标签；测试版本只在 final evaluator 读取答案。
- 动态依赖必须来自实际 def-use/插桩轨迹，不能只靠 coverage 推断。
- `FP` 不等于所有覆盖可疑代码且测试通过的产品。
- 干预前后必须有真实执行对，模型预测不能当成真实因果效果。
- 人工构造 FP 使用被删除的失败测试作隐藏答案时，不能让主动策略访问该测试，否则存在评价捷径。
