# 阶段01：清单、标签隔离与划分

在项目根目录使用 Python 3.11；本阶段只使用标准库，无需安装模型依赖。外部路径在 `configs/data_sources.local.json`，其来源是阶段00审计。银行账户与邮件以外的数据未纳入主清单。

## 一次执行本阶段验收

```powershell
python -X utf8 -B scripts/run_phase01_checks.py
```

此命令先跑单元测试，再读取银行账户1版本3产品，之后以最多109版本、3454产品为边界扫描两系统，生成清单、划分、隔离标签和校验报告。遇到更多数据会报错，不会悄悄截取。它只清点已有执行产物，不启动Java测试、特征计算或训练。准确参数、耗时、退出码、输入与代码指纹保存在 `outputs/phase01_run.json`；各命令输出保存在相应日志。

## 分步执行

首先让Python能够找到项目代码：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
```

银行账户小样本使用明确的版本 `_MultipleBugs_.NOB_1.ID_1`、产品 `model_m_ca4_0001` 至 `0003`；不是从目录排序猜测关联。

```powershell
python -X utf8 -B -m cwm_fp.data.benchmark_adapter --sources configs/data_sources.local.json --out data/processed/manifests/smoke --smoke --max-versions 1 --max-products 3
python -X utf8 -B -m cwm_fp.data.labels --products data/processed/manifests/smoke/products.csv --sources configs/data_sources.local.json --out data/processed/labels/smoke
python -X utf8 -B scripts/validate_contract.py --products data/processed/manifests/smoke/products.csv --tests data/processed/manifests/smoke/tests.csv --labels data/processed/labels/smoke/labels.csv --check-paths --out outputs/phase01_smoke_validation.json
```

完整的本地两系统数据：

```powershell
python -X utf8 -B -m cwm_fp.data.benchmark_adapter --sources configs/data_sources.local.json --out data/processed/manifests --max-versions 109 --max-products 3454
python -X utf8 -B -m cwm_fp.data.splits --products data/processed/manifests/products.csv --tests data/processed/manifests/tests.csv --out data/processed/splits --seed 20261030
python -X utf8 -B -m cwm_fp.data.labels --products data/processed/manifests/products.csv --sources configs/data_sources.local.json --out data/processed/labels/full --splits data/processed/splits/splits.csv data/processed/splits/loso_BankAccountTP.csv data/processed/splits/loso_Email.csv
python -X utf8 -B scripts/validate_contract.py --products data/processed/manifests/products.csv --tests data/processed/manifests/tests.csv --labels data/processed/labels/full/labels.csv --splits data/processed/splits/splits.csv data/processed/splits/loso_BankAccountTP.csv data/processed/splits/loso_Email.csv --check-paths --out outputs/phase01_validation.json
python -X utf8 -B -m unittest discover -s tests -v
```

只有一个版本的小样本不能划分为训练、验证、测试三个集合。主清单有三套可用划分：按版本分组、留出银行账户、留出邮件。另四系统在 `split_audit.json` 记为缺失，不生成空表。六系统划分能力用合成样本测试，不代表补齐真实数据。

## 读取接口与禁止项

仅使用 `cwm_fp.data.inputs.load_inputs` 获取配置与原始观测。它默认不返回标签、来源、版本名、路径、覆盖率原始属性或隐藏测试。传入 `splits_path` 和 `partition` 可只取一个集合；样本散列只作连接，不作为特征。

标签读取必须使用 `cwm_fp.data.labels.load_targets`，传入划分专属标签表、产品清单、划分表，并声明 `partition="train", purpose="train"`；该接口拒绝用全量审计标签表训练。验证／测试指标只能显式声明 `purpose="evaluate"`。预处理拟合前必须用 `require_training_keys` 检查所有贡献样本属于训练集合；本阶段没有计算这些统计量。

`labels/full/labels.csv` 及各划分标签文件含来源元数据，属于封存审计资料。模型输入读取器不会打开它们；标签读取器只返回目标标签。文件目录分离不是操作系统权限控制，后续进程不得绕过这些接口访问原始缺陷信息。

600个来源转换型假通过在本次原始套件视图中保留为失败产品。转换实验视图尚未启用；其隐藏失败测试不能用来宣称发现新失败。929个原始通过的代理假通过依据已匹配报告与非转换标记记为自然来源，但缺陷真实性仍待核验。

## 产物

| 位置 | 内容 |
|---|---|
| `data/processed/manifests/{products,tests}.csv` | 真实产品和已存在测试执行产物清单 |
| `data/processed/manifests/manifest_audit.json` | 原始路径、系统别名、报告散列、清单指纹和证据限制 |
| `data/processed/labels/full/` | 全量审计标签、标签审计和每套划分分别封存的训练／验证／测试标签 |
| `data/processed/splits/` | 按版本与两系统留出的划分、六系统计划及分组审计 |
| `data/processed/*/smoke/` | 明确选择的小样本及不可分组说明 |
| `outputs/phase01_validation.json` | 每系统、每集合的数量，来源分层、缺失信息与论文差异 |
| `outputs/phase01_run.json`、对应日志 | 实际命令、环境、耗时和验证结果 |

按现有 `.gitignore` 规则，生成数据和日志不自动纳入版本管理；它们已经实际写入本机，不能据此假设已上传。`data/examples/` 仅为原有格式测试恢复的合成示例。阶段01结束后停止，阶段02须由后续指令启动。
