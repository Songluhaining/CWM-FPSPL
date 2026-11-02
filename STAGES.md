# Milestones and Evidence Gates

## 当前设计修订 DESIGN1（2026-11-02）

用户授权先统一理论、模型及算法，再在BankAccountTP、Email、Elevator、ExamDB单缺陷数据上比较。本轮设计见`outputs/phaseDESIGN1_review_packet.md`。不默认添加循环记忆；将局部转换作为共享编码器的可检验辅助监督，产品分类为主目标。标签复用基准，原生基线上下文与严格归纳评价分列。本机预处理、服务器容器训练。

本轮没有实施阶段06联合训练，没有新Java采集或基线重跑，不修改以下历史阶段结果。下一实施任务以设计接口、有限验收及四系统新协议为边界，保留旧冻结数据。

| Phase | Objective | Deliverable / acceptance criterion |
|---|---|---|
| 00 | Repo/env/data audit | `outputs/audit.md` with measured paths, versions, tier and blockers; no fabricated data |
| 01 | Manifest + schema + strict splits | canonical `products.csv`, `tests.csv`, `splits.csv` and schema validation; version/LOSO separation |
| 02 | CLAP/PULP/supervised tabular baseline | results on a declared subset with exact reproducible command; no test label leaks |
| 03 | LDDG and instrumentation audit | per-test graph with true static CDG + observed dynamic DDG, event provenance, unit/integration tests |
| 04 | Graph-only supervised FP classifier | R-GCN/HGT baseline, result under same version split |
| 05 | Structured world model | teacher-forced next-value/branch/output metrics on held-out traces (when T2), graph transition when T1 |
| 06 | Joint supervised training | world model + classification head; ablate world-model losses; model/metric artifacts |
| 07 | Counterfactual verification | legal before/after rerun pairs and held-out intervention prediction; if no runner, marked unavailable |
| 08 | Optional active policy | matched budget random/greedy/MPC/RL, measured cost and verified failure rate |
| 09 | Full experiments/figures | LOSO, natural/simulated FP, 1/2/3 bug, statistical tests, logs and limitations |

Start with phase 00 prompt only. Phase 08 is optional; do not implement before phase 05-07 demonstrate useful predictions.
# 阶段03B-4补充状态（2026-11-01）

有限框架效果建模与真实验证已完成；范围为已有银行批次16个拒绝测试、原124图和旧8图回归。新批次联合合格图124→138，新增4个原始通过完整产品正式准入，保留旧1个。2张交易图继续拒绝；训练类别与跨系统分区不足，默认读取采样及时间前缀构造仍需后续工作。本阶段未拟合模型或启动后续阶段。

统一研究交接包：`outputs/phase03B4_review_packet.md`及同名JSON、产品/测试图状态CSV。具体后续阶段尚待用户指令。

## 阶段04P补充状态（2026-11-01）

仅完成五个既有合格产品、75图的无损读取、相对路径数据包、不同根目录复核、固定参数CPU前向和时间前缀契约设计。50项针对性及相关回归通过；75图全部节点/边/事件与冻结源一致。新增语义合格产品为0，默认16测试造成的7项输入损失已消除。未运行Java、Linux/GPU、模型拟合或新采集。

数据管线在限定范围内通过，正式训练仍被冻结划分与类别/跨系统样本不足阻塞；真实前缀生产器、充分状态观测、合法干预对和GPU环境验收仍未完成。完整交接：`outputs/phase04P_review_packet.md`及JSON、产品/图状态CSV。下一阶段只提出建议，等待用户授权。

## 阶段04D-1补充状态（2026-11-01）

已实际执行BankAccountTP冻结分组协议子集的标签盲队列：新增12个独立版本、36个原始通过产品、713对未插桩/插桩测试，全部完成且结果一致。713图身份完整，697图语义合格；13个产品因16张转账测试图未认证而拒绝，新增23个完整合格产品。全部失败证据保留，未替换产品、修改划分或扩展Java语义。

新旧合并数据包为28产品、16版本、463图，全部节点/边/事件/登记测试无损，旧75图逐项一致。47项数据侧回归及19,827项文件指纹核对通过。冻结包后仅经授权训练入口统计：训练17产品/9版本，其中代理FP10产品/6版本、代理TP7产品/4版本；验证5产品/4版本，测试6产品/3版本。预注册小规模数据数量目标达标，但并非正式泛化证据。

当前首要任务为独立训练入口与反向传播验收；服务器访问和环境尚未验收，转账配置在完整合格队列中缺失，真实安全时间前缀、充分数值状态和干预对仍未具备。本阶段未执行模型前向、拟合、分类评价、Linux/GPU运行或后续阶段。独立中文交接：`outputs/phase04D1_review_packet.md`及JSON、`phase04D1_cohort_status.csv`、`phase04D1_graph_status.csv`。生成数据位于`D:/dataset/whn/cwm_fp_generated/data/processed/lddg/phase04D1_bank_v1/`。

## 阶段04T-1补充状态（2026-11-01）

已在Windows CPU完成真实数据包训练入口、反向传播、优化器更新、整产品/逐测试梯度对照及保存恢复验收。仅使用既有分组协议的17训练产品/9版本/283图，授权代理标签FP10、TP7；验证5产品/4版本/85图、测试6产品/3版本/95图仅无标签内容核验。没有读取其目标或生成预测，没有改变数据包、划分、标签或Java语义。

实际执行236次参数更新：4次梯度对照、40次人工拟合、100次真实4产品微型拟合、2次恢复对照、90次全训练区10轮试跑。微型损失0.695353344→0.680341393，全训练区损失0.683606516→0.679590274；每轮全部17产品/283图无损参与。24测试及6测试产品两种计算路径在事先固定容差内一致；恢复后输出、下一步梯度/参数差为0。61项针对性与相关回归通过。此处是实际有限训练的工程证据，没有正式泛化成绩。

包总量仍为28产品/16版本/463图，新增语义合格产品为0。可准备固定协议的小规模模型比较与独立评价，但代理标签、转账子空间覆盖缺口、无Email完整合格队列等限制继续保留。真实时间前缀生产器、充分数值状态、合法干预对仍缺失；未训练世界模型或强化学习。Linux启动脚本已准备，服务器/GPU未验证，未重试此前被拒绝的X:/codes访问。

完整中文交接仅`outputs/phase04T1_review_packet.md`，机器交接为同名JSON、`phase04T1_training_trace.csv`及`phase04T1_gradient_checks.json`。独立运行及精确执行源码位于`D:/dataset/whn/cwm_fp_generated/data/processed/graph_classifier/phase04T1_cpu_v1/`。本阶段结束，不自动执行后续阶段。

## 阶段04R-1补充状态（2026-11-01）

固定训练／验证先导比较已完成。仍使用17训练产品/9版本/283图；协议冻结后由独立评价入口读取5验证产品/4版本的目标，实际FP2、TP3。测试6产品/3版本的标签及预测继续封存。没有修改数据、标签、划分或图语义。

常数与统计逻辑回归各一次，新增无边节点和仅合并边关系对照，复用既有R-GCN及两级均值。三种神经模型各3种子×200轮，加原4产品500步诊断，共16700神经更新；逻辑回归16求解迭代。微型诊断损失0.69535→0.00220，证明原100步不足以判断不能拟合。验证选择后R-GCN损失0.3018—0.3119，关系合并0.3442—0.3588，节点对照0.5476—0.5802，统计逻辑回归0.4511，常数0.7446。

全部非恒定方法在5个验证产品上的AP/ROC均为1，不能将饱和的小样本排序结果视为正式泛化；关系模型的附加收益仅在当前概率损失/Brier上观察到，且存在按关系归一化差异。72项人工/相关回归通过；冻结前一次导入语法错误保留记录，修复后才冻结和执行，没有看分数后追加训练。

真实运行仍是Windows CPU，主训练保守模型窗口约207.843秒，图进程峰值驻留内存约1.140GiB。Linux/GPU未执行，没有新Java或世界模型训练。当前首要建议为独立建设真实时间前缀生产器；代理标签、Transaction/Email覆盖与小样本不确定性继续保留。

唯一完整中文交接为`outputs/phase04R1_review_packet.md`，另有协议JSON、机器交接JSON、指标CSV、学习曲线CSV及预测诊断JSON。独立运行目录`D:/dataset/whn/cwm_fp_generated/data/processed/graph_classifier/phase04R1_cpu_v1/`保留全部候选预测、检查点、目标访问和源码摘要。本阶段结束，等待下一步指令。



## 阶段05A补充状态（2026-11-01）

真实物理观测前缀生产器及独立目标已完成。只使用原17训练产品／9版本／283轨迹和5验证产品／4版本／85轨迹，核对真实事件13597＋3921，共生成17518个前缀；下一事件有效目标17150个，分支目标523个，局部直接观测算术配对547组。测试区只核对清单／既有读取器文件元数据，不展开轨迹；所有产品分类标签均未读取。

31769个物理位置的截断重放及158845次后缀扰动比较通过，独立核对了数值、帧、异常和依赖原证据；1104次公开前缀读取复核通过，146项针对性／相关回归通过。没有新Java、模型拟合、服务器或GPU执行。

普通静态唯一后继约覆盖79.5%的有效下一事件目标，30次例外均为已认证空字段异常。存在116组训练／验证共有的完全相同前缀输入。可另行冻结有限的观测性结构预测试训，但完整数值状态、真实干预、跨系统有效性及世界模型训练仍未具备。原28产品分类准入和冻结划分保持不变。

唯一完整中文交接为`outputs/phase05A_review_packet.md`，另有协议JSON、机器交接JSON、轨迹状态CSV及隔离检查JSON。独立数据位于`D:/dataset/whn/cwm_fp_generated/data/processed/prefix/phase05A_v1/`。完成后停止，不自动执行05B。

## 阶段05B补充状态（2026-11-01）

已按本阶段授权实际实施有限观测性结构转换预测。复用05A训练17产品/9版本/283轨迹/13597前缀及验证5产品/4版本/85轨迹/3921前缀，不读取产品分类目标、不展开测试轨迹、不重跑Java。固定协议后，三个模型各3种子×20轮完整训练，正式12780次更新，真实工程验收3次。实际设备为Windows CPU，未使用Linux/GPU。

验证弃权A版本宏平均损失（三种子均值）完整图0.340581、移除依赖图0.390928、顺序模型0.218933；B为0.657350、0.663473、0.596503，训练先验0.667943。共同未见319个前缀中有300个有效A、80个弃权A和17个B；非平凡A/B仅3验证版本。共同未见B完整图0.753657，差于先验0.719994和顺序模型0.593462。仅观察到相对消融的局部改善，未证明依赖结构对执行顺序有稳定优势，不能单独归因动态数据依赖或声称因果机制成立。

115项相关回归、真实梯度/更新/恢复检查、1104次公开读取、17518个缓存摘要比对，以及训练3552/验证1027关键位置的模型层后缀隔离通过。单张图及完整产品准入未改变；没有新增合格产品。实际状态是工程试训完成、研究证据部分具备。下一步只建议修正局部状态转换任务和可观测表示，不自动实施。

唯一完整中文交接为`outputs/phase05B_review_packet.md`，附协议、机器交接、指标、曲线和捷径诊断，以及产品/轨迹状态CSV。独立运行目录`D:/dataset/whn/cwm_fp_generated/data/processed/transitions/phase05B_cpu_v1/`保留候选预测、检查点、缓存、命令、失败日志、旧契约和模型隔离证据。参数化Linux工程入口未运行，状态NOT_VERIFIED。本阶段结束，等待新指令。

## 阶段05C-1补充状态（2026-11-02）

已完整取得管理员指南正文，真实SSH连接指定服务器并复用`wanghaining_ubuntu_ssh`容器；X盘、宿主机、容器路径通过同一非空文件内容及摘要核验。没有在宿主机安装环境。Python3.11.13、PyTorch2.6.0+cu118及NumPy2.4.6安装在容器个人挂载目录的独立环境；实际单卡RTX3090Ti完成三个旧模型的前向、损失、反向、更新、旧检查点/优化器恢复和真实批次隔离。各模型仅一轮完整训练区遍历，总226次优化器更新，未形成新的正式泛化排名。顺序模型固定批次约1.83倍加速，两个小图模型约0.48/0.39倍，实际更慢。图模型输出非逐位一致的首次失败及网络、路径失败均保留。

冻结17训练产品/9版本/283轨迹及5验证产品/4版本/85轨迹保持不变，不读FP/TP、不展开测试轨迹、不新增Java采集。新增执行前操作数诊断1070行；418/523个分支以直接操作数核验，均一致。另105个分支的返回复制证据单列为语义推导，未计作直接测量。547组算术输入全部直接可见，472组有限整数运算核验一致，75组i2l保留未认证。全部17518事件的静态身份关联通过；旧图未加边，两跳查询仍无法到达对应操作数值节点。

最终49项不同相关测试通过；初次6项读取测试因Windows路径失败，移植后通过。旧数据与8710个导出文件结束核验不变。完整中文交接为`outputs/phase05C1_review_packet.md`，另有机器交接、服务器验收、语义检查及局部状态/产品/轨迹CSV。证据位于`D:/dataset/whn/cwm_fp_generated/data/processed/local_state/phase05C1_v1/`，服务器对应`/workspace/codes/wanghaining_cwm_fp/phase05C1_v1/`，报告同步在已核实X盘目录。结构和局部数值证据部分具备，未证明跨系统泛化、下游分类收益或因果干预。首要后续建议为Email有限机制数据接入与可观测性验收，需另行授权；本阶段结束并停止。

## 阶段05C-2补充状态（2026-11-02）

实际复用服务器现有容器和Python环境，在CPU完成Email原分组训练区版本8、2产品、24条冻结轨迹的局部数据生产。24对原始/插桩记录一致，完整读取3325条指令（产品2194、测试1131），旧解析器独立核对数量/操作码/顺序一致。Windows来源路径采用受摘要约束的解析；静态字段读取完成标记的兼容修复不产生虚构值或初始定义，既有静态身份逐项不变。首轮299条不完整解析及所有记录保留。

可信前缀194事件，其中产品87；20条轨迹有首个阻碍（字段未观测定义15、数组/类型类3、库调用2），4条未检出前缀缺口但没有可用产品分支/算术查询。全部46产品分支及21个iadd均在可信边界之外，局部转换正式准入0/67；覆盖内一致率未定义。3112动态身份连接可信，213未认证；39条产品静态身份缺口未解决。未恢复任何完整分类产品。

53项不同专项/回归测试通过；两版合计103次测试执行通过。1678位置截断重放、8390次后缀扰动通过；3325条输入在缺少目标文件的新根目录可读，与Bank既有1070条诊断合计4395行字段/掩码/整数格式核验通过。没有新Java、神经模型、GPU计算、标签访问、Email验证/测试轨迹展开或干预。输入格式兼容不等于跨系统泛化。

唯一完整中文总结为`outputs/phase05C2_review_packet.md`，附机器交接、协议、核验和逐轨迹/查询/产品/图状态CSV。服务器根`/workspace/codes/wanghaining_cwm_fp/phase05C2_v1/`，本地证据`D:/dataset/whn/cwm_fp_generated/data/processed/local_state/phase05C2_v1/`。首要后续建议是另行授权最小必要初始字段/堆状态观测与认证；当前不启动双系统训练或后续阶段。

## 阶段05C-3A补充状态（2026-11-02）

已在同一服务器容器完成Email原版本8、2产品、24条冻结轨迹的字段证据分层及Java运行前置检查。全部3325行严格输入和24条首次拒绝边界与05C2一致。15条首个字段缺口均为B类：9次clientCounter静态读取只有完成标记，6次实例读取只有Obj:0对象散列；均不足以认证当前值。A类为0，条件式局部转换生产模式未启用，严格模式仍0/67，新增局部转换与完整产品均0。

另外保存全部294次字段读取，其中86次带可解析整数/布尔回调；278次位于更早语义缺口之后，不能用于恢复整条轨迹。1次原本已有写入来源的观测单列，不误记为缺失。16项新增人工记录测试与47项旧回归通过，共63项不同测试，含初版共78次执行。15组真实读取边界、150次物理后缀扰动、75次当前输入后缀扰动及6次删除真实回调检查通过。首次审计的元组/列表表示误比较失败及两次成功审计均保留。

原代理、依赖JAR、产品/测试类、JUnit入口和构建源码已按历史摘要核验并迁移，原静态导出辅助源码从既有冻结备份恢复。容器未找到Linux Java/java编译器，故最多两条代表测试的Java复现未执行，代理重建未执行。另已定位测试准备硬编码java.io.tmpdir=/tmp与根分区无余量的兼容性待验收，未修改测试、挂载、权限或宿主机。没有初态补采、GPU或模型训练。

完整中文交接为`outputs/phase05C3A_review_packet.md`，用户真实前置项单列`phase05C3A_user_actions.md`；附协议、机器交接、字段CSV、观测检查、运行前置JSON及产品/图状态。独立服务器目录`/workspace/codes/wanghaining_cwm_fp/phase05C3A_v1`，本地证据`D:/dataset/whn/cwm_fp_generated/data/processed/local_state/phase05C3A_v1`。下一步先补齐已核验Linux Java8并验收生命周期路径，再另行授权最小字段读值观测。本阶段结束，不自动实施。
