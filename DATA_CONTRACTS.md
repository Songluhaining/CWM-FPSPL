# Data Contracts — Canonical Interfaces (v0.12)

## 当前模型同队列测试评价增补（EVAL1，2026-11-02）

用户授权评价现有模型。新增 `models/matched_evaluation.py`，单独冻结 `matched_04R1_holdout_v1`，沿用04R1全部28产品、原划分、已选择检查点及完整图包，不改变04T1/04R1旧入口的测试禁读规则。新入口允许在全部预测封存后由评价器读取这6个测试产品的标签；不是重新选择检查点或训练神经网络的许可。

图推理须验证包外清单摘要、历史检查点和契约、旧验证预测复现（绝对容差1e-6）、参数不变及完整产品预测身份。表格监督基线只使用同17通过产品目标，复用BASE4特征；91个同训练版本的原始失败产品作为既有上下文/缩放参考。原PULP对17个通过产品按5%抽样为零，另行声明同9训练版本215个无标签通过产品的扩展池，不能称为完全相同输入条件。

新增分区标签副本只含连接所需身份、套件与目标；每个导出进程仅可读取相应原分区及自己的导出分区，不能读取其他分区。预测先封存、测试标签后连接；旧原始标签和旧结果不可覆盖。当前成绩仅验证既有图分类器，统一分类/转换模型仍未实现。

## 四系统全量传统基线增补（BASE4，2026-11-02）

新增独立基线产物 `data/processed/baselines/four_systems_v1/`，历史图、标签和划分不变。全部178个单缺陷版本、4206产品及244772条原始测试覆盖记录进入清单；基础特征缺陷和单类版本不得过滤。原始失败产品保留为可观察参考，不成为FP/TP二分类目标。封存标签使用原PULP特征与聚合源码位置成员规则，原始失败不转换，标签与来源字段不进入特征。

`version_context_v1` 为明确声明的“同一版本可观察测试上下文”：每版本独立构建覆盖与静态切片参考，排序统计仅使用该版本原始失败产品的测试。包括目标版本的可观察上下文，但不读取其FP/TP目标。分类器与外层缩放器仍仅在训练分区拟合。`source_pool_transductive_v1` 单列全量无标签池PULP源码算法，不宣称留出泛化。

本次使用 `source_count_only` 覆盖规则，仅原XML的 `count>0` 计入执行集合，与CLAP/PULP源读取规则一致。只有分支计数的记录另计诊断；实际 `count>0` 的位置缺失映射继续报错，不猜测补齐。旧 `all_counts` 默认语义保留。

静态切片缓存绑定真实源码、分析种子集合、切片JAR与Java运行依赖。逐产品观测检查点绑定产品/测试记录及数据摘要，恢复时重新核验覆盖、切片与源码内容。静态切片仍是近似静态分析，不是动态定义使用证据。结果CSV保留完整身份和评价口径；标签与缺陷来源仅在独立导出器及评价器使用，不能回流拟合。详细结果与限制见 `outputs/phaseBASE4_review_packet.md`。

## 阶段05A增补：物理观测前缀（2026-11-01）

新增独立格式`physical_prefix_dataset_v1`，不重解释既有完整图或旧`prefix_observation_contract_v1`。生产入口为`cwm_fp.models.prefix_dataset`，输入重建为`prefix_reader.load_prefix`；任务字段和目标先冻结在`outputs/phase05A_protocol.json`。真实数据位于`D:/dataset/whn/cwm_fp_generated/data/processed/prefix/phase05A_v1/`。历史数据和旧源码备份保留。

- 使用物理日志行号及行内子记录序号：指令记录先于执行，栈回调后于执行，即使在同一行也分开。预测位置是当前产品／测试指令的执行前记录之后、回调之前。`occurred_at`只说明已经观察到指令执行前边界；定义的精确发生时刻未知时为null并给出发生区间。`observed_at`和`evidence_available_at`不能用定义事件编号替代。
- 输入生产器只接受独立静态材料和已经到达的记录，不接收最终图。复用旧动态解释器重放已观测前缀，暂未认证语义的定义／使用被屏蔽；普通指令的完成需回调或实际后续记录，调用效果需真实返回，精确空字段异常引用需实际处理器。调用待定期间不宣称堆状态保持不变，当前完整堆状态始终未知。未支持的Java语义不因此升级。
- 静态上下文来自原编译表中的全部产品指令和普通静态控制依赖，静态已执行列固定0；不能从最终执行的测试节点集合构造静态输入。测试指令身份仅在其原记录到达时连接；类／方法／位置／完整轨迹摘要只作审计，不成为学习身份。源码行身份仍不等于语法语句级认证。
- 模型输入保留22列固定类别／布尔语义，配置两列继续屏蔽。另有真实数值及独立观测掩码、当前事件和已出现事件的局部连接索引、待定调用局部索引及当前入度。缺值占位0不代表观测值；历史版本观测不等于当前字段状态。没有最终结果、断言／输出节点、最终覆盖率、最终长度或后验属性。当前版本没有加入特征归属边，也不认证完整动态控制者关系；普通静态控制依赖不能被称为异常控制图或已发生因果边。
- 目标与输入分别落盘。A为下一条实际可观测产品／测试指令的固定语义类别；物理框架区间、回调和返回仍保存，跨调用投影不冒称单步虚拟机转换。B仅在当前条件分支执行前、两候选普通后继及真实同帧后继均能核对时启用。根返回只有执行前记录，末尾A关闭，不由缺失后缀生成正常结束目标。局部算术配对只保留所有输入先前已有直接数值回调、结果随后真实回调的情形，不启用完整数值状态训练。
- 共享静态材料／观测流、位置索引、独立目标及源指纹分别保存；公开读取器需包外可信清单摘要，并复核重放张量摘要，完全不打开目标文件。内部破坏性副本检查不以整文件摘要拒绝代替后缀隔离证明，真实读取仍保留完整性核验。
- 本阶段仅训练17产品／283图和验证5产品／85图，继承原按版本分组；测试区仅核对清单身份。所有产品级分类目标均禁止读取，不改标签、原图准入或划分。不训练模型；前缀／结构目标合格不等于完整T2、合法T3或世界模型有效性成立。

## 阶段03B-3精确空引用字段异常补充（2026-11-01）

旧契约原样归档为`outputs/phase03B2_data_contracts.md`。新增控制证据版本`null_field_exception_v1`显式启用；默认调用仍生成并核验`normal_path_v1`，不重解释历史图。本阶段新产物在`D:/dataset/whn/cwm_fp_generated/data/processed/lddg/phase03B3_bank_v1/`，原批次保持只读。

- 唯一新增动态语义是：实际`aconst_null`及其可信局部／字段／调用传播形成确定空引用，在真实`getfield/putfield`事件上触发异常，下一条记录实际到达有序原异常表指定的精确`NullPointerException`处理器。未知引用和对象散列0均不能代替空值定义证据。
- 原异常表完整、所有相关帧非同步且非本地方法、之前没有数据流缺口、实际处理器入口匹配才可认证。跨帧展开必须对应真实活动调用，已正常返回的调用不能被称作异常中止。清空处理器操作数栈，保留处理帧局部变量，抛弃真实展开的子帧；失败字段读写不产生正常读取值或堆写入。
- 异常引用值是由已发生指令及真实处理器证明的符号引用，不是额外执行事件或标量测量。其`observed=false`，不补零；最初引用的`evidence_available_at_event`为实际处理器事件。重复发生仍有不同事件、值版本和调用实例。修复处理器所属帧会改变相应规范事件摘要，旧摘要和原始轨迹保留。
- 新证明保存原指令表、字段成员、方法标志、原始事件字段及数据流摘要。消费者重放空值来源、帧展开、值版本和定义使用端点，校验摘要及图中边／值与重放一致。摘要不是防恶意篡改签名，外部原始日志、类文件及来源指纹仍必须保全。
- `control_path_verified`表示在`normal-and-exact-null-field-catch`范围内的路径核验；`normal_path_verified`只在没有异常转移的合格正常路径为真。`certified_null_field_transitions`与证明中的`certified_exception_transitions`只计此子集。通用异常支持字段`exception_control_supported=false`继续保留。当前独立数据流、身份、完整套件、结果一致性门槛全部仍适用。
- 静态`control_dep`仍为普通控制流后支配关系，不冒称完整异常控制依赖图；认证异常转移单独位于控制证明中。动态定义使用来自实际事件及虚拟机确定语义。未知库调用、类型转换、空接收者方法调用、数组、其他异常、通用捕获／清理及同步展开仍按证据拒绝。正常返回不能证明库调用无副作用。
- 控制路径未通过时继续抑制动态依赖。路径通过但存在其他数据流缺口时，可以保存已支持部分的依赖供审计，整图仍拒绝，不能因此进入产品训练集合。
- 阶段04保留22列表示、完整原始套件核对、原始失败产品排除及冻结划分规则。现有每产品16测试、单图512节点／128事件／4096边为编码采样默认值；128 MiB是整个图输入文件上限。正式准入与无采样读取分别验收，不丢测试换取资格。
- 后续世界模型不能将整图核验结果、已知返回、异常处理器、后续边、断言或最终输出放入更早时间输入。需要专门的可见时间前缀构造；本阶段尚未实现该数据集，数值状态、干预及强化学习损失继续关闭。重放旧日志所得值节点增加不等于新增数值观测。

## 阶段03B-2跨版本小批次补充（2026-11-01，图格式与准入不变）

旧契约归档为`outputs/phase03B1_data_contracts.md`。批次入口为`cwm_fp.lddg.batch`，验收入口为`cwm_fp.lddg.batch_audit`；复用既有运行、源码身份、动态解释和阶段04正式读取器，不升级其语义支持范围。

- `configs/phase03B2_collection.json`与独立输出目录的`plan.json`必须字节一致，并通过冻结摘要验证。先按系统与原分组划分分层，再用固定种子和散列排序选版本、产品；只使用原始测试通过／失败状态，不使用假通过／真通过目标。不因采集后的图质量改变选定名单。
- 生成目录位于`D:/dataset/whn/cwm_fp_generated/data/processed/lddg/phase03B2_v1/`，与两套原始数据目录分离。`p000`等短目录只用于隔离；所有连接均使用完整系统、版本、产品、测试键。同名产品、局部事件／值／帧编号不得跨版本或跨测试混用。历史03B-1数据只读复用。
- 每产品记录全部冻结登记测试，包括失败、对照不一致、语义不支持及提取失败记录。分别报告“全部测试已执行”“全部执行与原记录一致”“全部测试图已生成”“图身份／依赖合格”“完整通过产品正式准入”；这些状态不得互相替代。
- 严格执行一致性包含测试失败类型及消息。即使通过／失败结论相同，异常类不同仍拒绝，不能把模拟异常类与原生类自动视为等价。原始失败产品始终不进入假通过／真通过二分类目标。
- 持久预算累计控制动作时间并预留进程数量；中断中的预留保守计费。冻结上限为900个新测试进程、450对、2700秒控制动作、4GiB／30000文件，D盘至少保留4GiB。单次与单产品限制见配置和运行记录。监测为轮询，非系统硬配额；人工审阅耗时不冒称子进程执行时间。
- 执行缓存绑定真实源码、类文件、Java环境、代理、依赖与策略；图缓存核对封存的输入与输出指纹。未封存或失败尝试保留，不隐式覆盖重跑。每测试落盘；半次执行仍可能不完整，不能宣称任意中断后无损恢复。
- 正式准入仍调用既有`load_corpus`，单输入128MiB上限不变。22列表示只编码既有语义类别，不把路径、产品／版本、类／方法名、行号或标签转为学习特征。默认采样损失与图准入分别报告；无采样接口检查不训练模型，也不表示全部原始数值已进入22列。
- 训练类别可用性仅在名单及准入封存后，以每个协议的独立进程读取该协议训练标签文件，输出聚合计数。禁止读取同协议验证／测试目标或用其他协议训练标签反向筛选该协议测试样本。历史代理标签的证据资格保留，不升级为独立验证的真值。
- 原始未压缩日志、事件、值版本及缺失标记保留。控制路径未认证时沿用既有保守抑制动态依赖规则；不能将抑制后没有值节点解释成完整状态。测试完成结果不等于程序业务输出，断言调用不等于捕获期望／实际值；本批次不生成状态转换训练集或干预样本。

## 阶段03B-1完整套件采集补充（2026-11-01，图格式不变）

本节补充运行和产品集合约束，不改变`program_identity_v1`、`normal_path_v1`、22列表示或阶段04准入。旧契约归档于`outputs/phase03A3_data_contracts.md`。本次独立目录为`data/processed/lddg/phase03B1_bank0001_v1/`。

- 完整集合必须严格等于冻结公开清单中同一`(system_id, bug_version_id, product_id, suite_id)`的登记测试集合；历史套件完整性未独立证明时，不扩大“完整”的含义。任一图身份、语义、完成记录或执行一致性不合格，整个产品继续拒绝。
- 单次完成缓存必须绑定原源码、实际编译类、代理、测试运行器、依赖、Java运行文件、跟踪类范围和资源策略。校验已封存产物的集合及逐文件散列；差异或未封存执行拒绝自动覆盖。旧产物需另行核验冻结指纹与兼容证据，缺失的旧资源测量不能补造。
- 新编译产物与旧类逐字节一致时，允许复用旧静态表和源码身份依据；必须保留新旧类比较记录及实际执行路径。两个同名类或相同行号本身不是一致证据。
- `collection.json`按测试完成后原子更新并保留先前完成记录；`collection_history.jsonl`区分新执行、已有记录复用和主动检查点停止。未封存的半次执行不自动重用、不删除；人工指定新尝试目录前保持停止。
- `configs/phase03B1_collection.json`声明固定一个产品、30秒单次测试、512MiB Java堆、1GiB直接子进程驻留内存、2万事件、32MiB单次输出目录、128文件，以及整个新目录512MiB／2048文件、最少1GiB磁盘余量。运行前预留一次输出空间，运行中和退出后检查。50毫秒轮询可能短暂超限，不能表述为操作系统硬配额或完整进程树内存上限。
- `dynamic_instances.json`是审计附表：动态值、分配对象和调用帧只能在完整产品键与测试身份命名空间内解释；局部的`v:0`、`allocation:0`、帧0在不同测试中不相同。静态节点仅按真实编译类和指令证据对齐。以上坐标不得成为数值身份特征。
- `time_prefix_audit.json`记录完整测试图中的未来信息风险，不是已实现的时间前缀数据集。动态定义使用边要等使用发生后才可见；分支结果要等真实后继，调用完成要等返回标记，标量回调要按指令后时点处理。整次测试的执行标记、后续值、断言和最终结果不能作为较早时间的输入。部分标量不自动升级为状态转换级别，插桩等价对照不升级为干预级别。


## 阶段03A-3正常路径证明与异常保守拒绝（2026-11-01）

本节优先于历史章节中“执行含异常处理器的方法即拒绝”的笼统规则。新数据只写入`data/processed/lddg/phase03A3_exception_v1/`，旧03A-2契约归档为`outputs/phase03A2_data_contracts.md`。原身份版本`program_identity_v1`及22列语义表示不变，新增控制证据版本`normal_path_v1`，提取器版本`phase03A3-0.4`。

- 原始类文件导出完整异常表：代码长度、表项数量、真实表顺序、左闭右开的保护范围、处理器入口及捕获类型；`*`表示类文件中的类型0，即捕获所有异常。旧`HANDLER`记录保留，新`EXCEPTIONTABLE/HANDLERDETAIL`提供完整元数据。没有表项也必须显式记录数量0，不能把缺证据当成没有处理器。
- 图新增`control_flow_evidence`，含原类散列、普通后继与调用目标、实际事件及入口／正常返回标记、线程和独立调用帧、运行完成一致性、测试结果及轨迹散列、证据摘要和核验结果。身份坐标、测试结果、异常类型及摘要只用于审计／准入，不加入数值学习特征。摘要不是防恶意改写的签名。
- `normal_path_verified`要求逐条普通后继及调用／返回关系一致、异常表完整、事件连续、单线程、入口及帧恢复可核对、执行完成且插桩前后结果一致。方法存在处理器不再单独拒绝，但实际处理器入口、显式抛出、缺调用完成、异常帧切换或无法确定的情况继续拒绝。普通`astore`不是捕获证据。
- 唯一允许的未正常返回外部终止边界是完整测试最后一个JUnit断言调用，根测试帧以外无活动帧，真实完成结果为失败且唯一失败类型为`java.lang.AssertionError`。该边界不生成异常对象或异常传播依赖，不认证测试框架内部异常语义。未知调用失败、其他异常类型及不完整执行不得通过。
- `control_semantics_scope=normal-path-only`明确只认证当前正常路径；`exception_control_supported=false`和`certified_exception_transitions=0`必须保留。`dependency_semantics_valid`还独立要求既有数据流支持范围、无字节码不一致、无栈／调用缺口及无已执行控制流缺口，不能仅由路径或身份完整推导。
- A表示含处理器的方法本次只走已核验正常路径；B表示受保护位置到处理器的非正常转移已定位但异常语义未支持；C表示证据不完整或恢复算法未支持。后两类不准入。`observed_handler_pcs`只表示真实事件位于处理器入口；不单独证明异常对象和展开。`events_checked`与`events_recorded`分开，早停不声称核验全部后缀。
- 实际普通跳转、进入调用、返回调用方作为`normal_transitions`审计关系保存；静态可能异常路径只在异常表中记录，不伪造动态异常边。原学习图控制边继续标为静态正常流后支配来源，不冒称完整异常控制依赖或特定动态控制者实例。
- 新格式未通过控制核验时，`dynamic_dependency_emission=withheld_unverified_control_path`：保留原始事件、规范身份、真实回调与缺口，但不输出整张图未经认证的动态值节点、产生／使用及数据依赖边，规范事件的`def_ids/use_ids`为空。这不是删图使其通过；整图仍严格拒绝。正常路径保持原动态实例和值版本。
- 消费者重新计算路径结论并核对事件身份、摘要、结果及质量汇总；不能仅相信通过标志。新版证据／版本不完整拒绝；旧产物仍走原门槛，含处理器的旧图不会自动升级。预览也不能放宽异常或未知身份限制。字段格式见`src/cwm_fp/lddg/control_flow.schema.json`及图格式。
- 产品级规则不变：全部登记原始测试完成、每图均合格、产品原始通过且遵守既有划分。单张测试图可编码不等于产品可训练。状态、配置约束、完整套件和干预证据仍分别验收；不启用缺少真实数据的状态／干预损失。

本次六图节点／事件／依赖保持不变，限定正常路径组合准入由4/6变为6/6，完整异常认证仍0/6，完整可训练产品仍0。详见`outputs/phase03A3_exception_validation.md`，不是对一般Java异常的认证。

## 阶段03A-2程序身份与分层准入（2026-11-01）

本节优先于下文历史规则。新产物只写入`data/processed/lddg/phase03A2_identity_v1/`，旧图、事件、清单、划分和实验结果不回写。完整设计见`outputs/phase03A2_identity_design.md`，实际验证见`outputs/phase03A2_validation.md`。

- 新身份版本为`program_identity_v1`。程序指令分为有实际源码依据（`SOURCE_BACKED`）、经源码语法树与真实字节码共同确认的编译器生成元素（`COMPILER_GENERATED`）、证据不足（`UNRESOLVED`）。只有前两类通过证据一致性检查才具有可接收身份。缺行号、缺源码成员或存在合成标志均不能单独证明生成身份。
- 静态指令的`instruction_id`由编译类文件散列、声明类、方法名、完整描述符及字节码偏移确定；`locator`保留这些审计坐标。来源证据还绑定源码、解析结果、编译产物清单散列和确认规则。默认构造器限定为已有验证范围内的普通类、无显式构造器、无参且调用`Object.<init>()V`、无异常处理器的线性实现；显式字段初始化保留源码身份，不笼统把整个构造器都标为生成代码。
- 每张图新增`instruction_catalog`。动态节点及事件记录保存`static_instruction_id`与`canonical_event_id`；后者同时绑定完整测试键、原始轨迹散列、事件序号、线程和调用帧。原节点编号、依赖边和执行顺序继续保留。产品动态节点连接原静态节点，测试动态节点连接编译指令目录；不为目录条目虚构运行事件。
- `source_line_id`与原`canonical_source_line_id`仍只表示行级证据；`source_statement_verified=false`，不认证行内语句序号。生成指令的源码行身份为空，保留真实字节码定位，不投射到相邻语句。覆盖率结构锚点不进入指令目录，也不产生动态事件。
- `source_mapping_coverage`是可信源码支持的静态产品指令数／本图静态产品指令总数；`generated_identity_coverage`是可信生成指令数／同一分母，并非生成候选自身的通过率。`unresolved_identity_count`单列其余指令。`dynamic_event_identity_coverage`覆盖本图全部已记录产品及测试事件；另报`dynamic_product_identity_coverage`。跨图统计静态指令时按完整产品键去重，不能把同产品两个测试的静态图重复累加。
- `program_identity_complete`要求全部产品静态节点和已记录动态事件具有可校验身份、目录连接及唯一事件身份。消费者逐记录重算指标，拒绝与汇总标志矛盾的产物。`dependency_semantics_valid`单独表明原有单线程、支持操作和正常控制流范围的结构准入，绝不由身份完整推导。`suite_completeness`单独记录登记原始套件应有／已采测试数；读入产品时再与公开测试清单核对。这仍不证明历史测试套件绝对完整。
- 新版测试图准入同时要求可信程序身份和原有依赖语义、执行完成及结果一致性检查；不再要求生成元素存在显式源码行。旧`canonical_mapping_complete`继续保留原源码定位含义，新六图仍均为false。旧格式继续走原映射门槛；不得删掉新版本字段而回退到旧检查。新版未知身份连预览也不放行。`structure_qualified`独立统计结构维度，正式准入始终使用组合检查。
- 产品训练准入仍要求原始套件通过、所有登记测试已采集、每张图均通过准入，且符合既有划分与训练条件。一张测试图可编码不等于产品可训练；不能过滤异常测试后补称产品合格。本次真实产品准入仍为0。
- 新数值列版本`semantic_identity_v2`为22列：原19列加三类来源的固定类别位。原`semantic_v1`的19列可显式选择以匹配旧权重；禁止静默混批或把旧权重当作新版本。新增静态指令操作码来自真实字节码；无类名、方法名、绝对行号、字节码偏移、散列、产品／版本编号、测试结果、标签及标签来源嵌入。来源的三类固定类别不同于这些唯一身份。
- 原图的时间边和独立执行实例保留；现有阶段04消融仍排除时间邻接边，编码审计元数据另保留事件顺序。实际观测值、类型及缺失位经独立元数据接口保留，本阶段不加入分类数值列，不启用状态损失。运行状态不完整、异常控制、完整采集及配置约束继续分别验收。

旧契约归档为`outputs/phase03A_data_contracts.md`；字段格式见`src/cwm_fp/lddg/program_identity.schema.json`及扩展后的图／事件格式。

## 阶段03A源码映射修订（2026-10-31）

本次只重验原先六个测试；新产物在`data/processed/lddg/phase03A_mapping_v1/`，历史`phase03_bank_v2/`保持不变。此节优先于下文阶段03历史映射描述，不改变标签、划分或模型输入白名单。

- 源码证据来自Java 8语法树的真实成员范围、显式特征映射注释、完整方法描述符、方法体逐标记对齐，以及字段声明标记与字节码限定字段写入的核对。证据绑定源文件散列；不由缺陷位置、标签或执行是否命中决定映射。
- `canonical_source_line_id`以`line|`开头，表示经过核对的原始特征源码行。为兼容既有读者，`canonical_statement_id`暂为同一行标识的别名；**它不是已验证语句身份**。`source_statement_verified=false`、`statement_ordinal=null`、`statement_ordinal_verified=false`；不再用尾部序号0冒充语句编号。
- `composed_source_line_id`与`composed_source_line_verified`单独记录组合产品源码行。隐式默认构造器仍保留真实字节码偏移、组合源码行及`implicit_default_constructor`分类；没有显式源码成员时，特征源码规范身份保留空值，原因是`UNKNOWN_no_explicit_source_member_for_default_constructor`。
- 静态节点仍由完整类、方法、描述符、字节码偏移识别；动态节点由完整测试键和事件序号识别。相同行号／相同规范行标识不能合并不同指令或动态事件。编译器生成的返回／异常清理只认证其行级定位，不能由此认证异常控制语义。
- 覆盖率中没有独立字节码行号的声明／`try`关键字位置记为`non_instruction_source_anchor`。它不表示整个方法不可执行，也不表示编译器优化；未知情况保留`unmapped_or_non_executable`，不制造事件。
- `dynamic_product_mapping_complete`只衡量已观察产品事件。`canonical_mapping_complete`同时要求静态产品指令和动态产品事件的规范行定位完整。旧提取器仅检查动态部分；此次补上静态部分，阶段04的准入检查保持开启。未执行本身不构成映射失败，缺少规范源码成员的静态指令仍如实计为未知。
- 6张图的格式与引用一致性通过不等于训练许可；正常控制流结构支持、异常控制、源码语句身份、数值状态、配置约束及完整原始套件仍须分别验收。详细逐条证据见`outputs/phase03A_mapping_diagnosis.md`与`outputs/phase03A_mapping_validation.md`。

## 阶段04实际落地规则（2026-10-31）

本阶段实现监督图分类及合成验收；真实图仍不足以训练或独立评价。最终生成目录为`data/processed/graph_classifier/phase04_v2/`，`synthetic/`下所有分数均为人工图的工程检查，不能作为软件产品线实验结果。

- 自定义张量见`src/cwm_fp/models/graph_encoder.py`：节点矩阵19列，仅含5种节点类型、9种固定指令语义类别、是否执行／可用／动态事件及经过约束验证的配置布尔状态和有效位。系统／版本／产品／测试编号、源码行号和方法名、值内容、断言结果、标签及其来源均不进入矩阵。值节点仅保留结构角色，不启用运行值预测。
- 边索引为`[2,E]`整数矩阵，关系为`[E]`整数向量；测试汇聚后再对产品汇聚。原始边证据、位置及身份只保存在独立审计元数据中。来源字段须区分标签来源、边证据来源和已观察标量回调来源，后两者也不作分类特征。
- 正式准入须原始产品通过、已登记原始测试的图集合齐全、所有测试结构通过阶段03检查且规范映射完整。不得丢弃某个失败／不支持的测试后将产品重新称为通过。已有原始清单仍不能证明历史套件绝对完整。
- 支持结构但映射不全的图可显式生成`preview_only=true`的无标签检查张量；归一化和训练拒绝这些张量。缺约束验证时屏蔽配置状态，不构造合法配置干预。
- 上限为每产品16个测试、每测试512节点／128事件／4096边；按冻结顺序取窗口，记录测试、事件、边的损失比例。消融移除的边数另行记录。仅控制依赖时同时移除值节点和数据／生产／消费边；仅数据依赖时移除特征控制和控制依赖边；两类图均不加入时间邻接边。
- 覆盖率对照使用每测试原图的静态节点数、已执行静态节点数、命中比例、事件数，计数取对数；它不是阶段02的两维产品覆盖率表示。其均值和标准差仅由训练通过产品拟合，图模型的布尔／语义列使用固定编码。
- `train_graph.py`仅接收训练通过产品的FP／TP目标；独立评价进程读取验证目标返回损失用于早停。模型冻结后才生成测试预测。最终评价核对源码输入、划分、队列、模型和预测指纹，并先独占创建评价凭证，禁止在同一运行目录重评。
- 真实准入报告`readiness.json`和三份`*_blocked.json`记录读取标签前的阻断。本次三个划分的真实训练次数、测试评价次数均为0。分数不可用时不能以0填表；阶段02的全量表格结果不能充当相同图样本集合的成对对照。
- `synthetic/training.json`保存参数量、轮次、时间、类别权重、归一化贡献及验证选择；`test_evaluation.json`保存人工图指标和错误样本。归因的节点输入梯度及边消息权重梯度只是模型敏感度，不是因果效应，也没有修改或重跑真实程序。

完整命令、环境、真实缺口与测试记录见`outputs/graph_classifier.md`和`outputs/phase_status.md`；阶段03契约原文保存在`outputs/phase03_data_contracts.md`。

## 阶段03实际落地规则（2026-10-31）

以下只描述银行账户一个版本的小样本适配，不覆盖全量数据。数据存于`data/processed/lddg/phase03_bank_v2/`；合成Java验收另存`phase03_synthetic_v2/`。

- `graphs.jsonl`、`events.jsonl`均使用完整系统／版本／产品／测试键；源码位置与身份只作连接元数据。图格式、事件格式见`src/cwm_fp/lddg/graphs.schema.json`和`events.schema.json`。
- 图中的`tier=GRAPH`表示候选结构格式，**不是单独的训练许可**。须同时检查`quality.eligible_graph_structure`、`quality.verified_tier`和`provenance.learnable`；另检查`canonical_mapping_complete`及`constraints_verified`是否满足实验要求。异常控制、已执行控制流分析缺口、未支持的动态语义、位置不一致、结果不一致或不完整均阻止结构准入。
- 静态语句节点按字节码位置保留，动态语句节点按逐测试`event_index`独立保留；`thread_id`来自回调，`frame_id`来自实际调用序列。不能把重复行号合并后冒称完整执行顺序。
- 每次实际定义具有唯一值版本；栈复制保留引用对象身份但生成新版本。动态数据依赖保存定义事件、使用事件、值编号和读操作角色。对象字段键使用实际分配事件与限定字段身份，不依赖对象哈希。缺同步证据的多线程轨迹禁用动态依赖认证。
- `source_map.jsonl`记录组合类／方法／描述符／行／字节偏移、原始特征位置、规范身份和置信说明。当前身份末尾序号0仅为行粒度占位，`statement_ordinal_verified=false`。未映射为空；缺失字节码的位置记`unmapped_or_non_executable`，不补造运行事件。
- `roles.meta`与配置启用集合一致会记录通过；缺版本匹配的特征模型时仍保持`constraints_verified=false`，不能将可编译配置等同于已验证合法干预。
- 实际栈标量回调可标`value_observed=true`／节点`observed=true`；缺回调的存储、字符串内容等继续缺失。不使用旧因子图正确性概率、`obsvalue`或推断值。断言只记录已观察的调用／返回和完整测试结果；期望／实际值并不完整。
- `quality_report.json`记录能力范围和缺口，`extractor_version.json`保存实现指纹，`collection.json`连接原始／插桩执行。所有T2状态值和T3干预训练开关保持关闭。
- 原覆盖率XML单独输入不能产生动态定义使用图。模拟套件、标签和原缺陷位置不进入提取；历史生成测试不纳入本次原始测试集合。

## 阶段02实际落地规则（2026-10-31）

本节新增基线格式；阶段01清单及划分保持不变。图、动态事件、运行值与干预格式仍不是已完成的真实产物。

- 当前实验协议标识为`strict_train_reference_v1`，仅原始测试套件。CLAP六维、PULP规则重建五十维、覆盖率／测试数两维分开存入`data/processed/baselines/<运行名>/<划分名>/features.npz`；矩阵仅含固定数值列，不含标签、来源、系统／版本名称或缺陷元数据。
- `features_audit.json`保存样本顺序、训练贡献产品键、词典规模、训练可疑度范围、来源限制与特征／清单／划分散列。`scaler_audit.json`保存仅训练集合拟合的最小／最大值。原始失败产品可以贡献训练统计，但不能进入FP/TP指标分母。
- `observations.joblib`是本项目自己生成的无标签观测快照；不加载未知来源的序列化文件。其产品、测试、来源配置及内容散列见`observations_audit.json`。冻结复用必须显式配置，不能声称复用时重新核验全部原始XML。
- 冻结预测文件只含`sample_id,partition,model,seed,y_score,y_pred,score_kind`。仅评价端连接真值后写入`outputs/metrics/<运行名>/<划分名>/<种子>/<模型>.csv`，另加`split_id,y_true,system_id,bug_version_id,origin`。完整列契约见`src/cwm_fp/baselines/result.schema.json`。`split_id`为划分名加划分文件SHA256；身份列只用于结果连接。
- `y_true=1`表示原始通过的代理FP，0表示TP；`y_pred`仅0或1。支持向量分类器的`score_kind=margin`表示分类间隔，不能作概率解释；逻辑回归／集成的`probability_uncalibrated`表示未额外校准的概率。
- `*_metrics.json`分别保存训练／验证／测试指标、每系统和来源分层、按系统与缺陷版本分簇的重采样区间。单类曲线面积为空，空模拟层标为不可用；不得以0冒充缺失实验结果。训练内指标与泛化指标分开。
- 文件访问检查限制准备、训练、冻结预测进程读取封存标签与指标。训练仅放行当前折训练标签，并在换折时撤销。该限制属于Python程序角色约束，不是Windows账户权限隔离；模拟套件仍禁用。
- 结果声明为训练参考池适配，不能与原论文按当前版本提取属性的结果混名；留出系统无同系统先验时，五十维原始值全零并记录缺证据状态，不能解释为成功跨系统语义匹配。详见`outputs/baselines_reproduction.md`。

## 阶段01实际落地规则（2026-10-31）

以下规则限定阶段01生成的真实清单；后文图、事件与干预格式仍为未来阶段的契约，并非已有真实数据。原有合成示例保持0.1兼容格式。

- `data/processed/manifests/products.csv` 保留后文列名以兼容原校验器，但 `label`、`label_provenance` 必须为空，`origin` 必须固定为 `unknown` 占位；实际标签和来源只存在于独立 `data/processed/labels/`。这些占位不表示真实标签来源未知。
- 真实来源配置为 `configs/data_sources.local.json`。优先读取 `config.report.csv.done`，只有它不存在时才读 `config.report.csv`；两者并存但内容不同时报错。报告行按产品名连接实际产品目录，版本名保留完整缺陷数与编号，不按排序连接。系统别名见 `src/cwm_fp/data/common.py`。
- 新增 `sample_id` 为完整三元键的确定性散列，仅用于连接，不作为模型特征。`features_json` 是直接读取报告得到的布尔特征配置；`constraints_verified=false`，因为尚未证明特征模型与版本匹配。
- 新增 `suite_view=original` 与 `suite_id`，明确使用原始套件。后者对产品键、已发现测试名、结果目录和文件大小计算散列；它是执行产物清单身份，**不是测试源码或XML全文指纹，也不证明原套件完整**。
- `tests.csv` 按完整覆盖率文件名去掉 `.coverage.xml` 后生成 `test_id`。`result` 来自原有通过／失败目录，与配置报告交叉核对；`result_evidence=coverage_directory_not_rerun`。`trace_path` 为空；不虚构轨迹。每个测试只能通过完整产品键继承划分，不能独立随机分组。
- `source_report_sha256` 和审计中的 `manifest_sha256` 记录报告与规范清单指纹；路径、缺陷数、数据族、论文子集资格、套件身份与测试计数均是清单管理信息，不能直接传入模型。论文子集资格目前统一 `unknown`。

### 封存标签与原始套件

`labels.csv` 使用相同产品三元键，另含 `label,origin,label_source,label_source_path,label_source_sha256,source_label,source_transformed_from_f,source_origin,decision,label_provenance,suite_id,suite_view,quality`。标签适配器只导入已存在的标签表，不执行源项目的标签生成函数，不读变异日志或缺陷行。

原始报告失败的产品，主视图标签为 `F`。原来源 `FP` 且明确转换标记为1时，保存来源标签及 `source_origin=simulated`，主视图仍为 `F`、`origin=unknown`，决策记为 `restore_original_failure`。不得将失败报告改成通过。阶段01**没有建立可用的转换实验视图**；该视图必须另行隔离隐藏测试、源码、覆盖率和旧输出，并继续继承原缺陷版本组。

已审计的本地PULP代理来源中，同时具备“配置报告逐字节匹配、原始通过、来源FP、显式非转换标记”的条目记 `origin=natural`；含义只是在已审计原始套件下没有经过失败测试删除。它不表示真实生产缺陷，也不证明相关可执行缺陷已独立核实。其他来源即使套件规模相似也保持 `unknown`。来源 `TP` 同样是代理标签，不能宣称经过论文人工核查。

### 实际读取边界

`src/cwm_fp/data/inputs.py` 的 `load_inputs` 只返回 `ModelInput(sample_id,features,observed_result)`，其中只有配置与原始结果是可用观测；编号只用于连接。它不打开标签表或任何原始路径，不读取XML内容。禁止把整个清单行或原始XML属性字典传给模型；后续覆盖率适配必须另建字段白名单。

离线审计程序按每套划分导出 `data/processed/labels/full/<划分名>/train.csv,val.csv,test.csv`。`load_targets` 必须指定用途和划分；训练只能打开对应训练标签文件，拒绝验证／测试用途及全量标签表。评价程序可以显式读取验证／测试分区用于指标。完整标签表、标签来源和校验统计均属审计资料，不参与特征选择或参数选择。

`require_training_keys` 为归一化、词典、可疑度先验及预训练提供仅训练集合的贡献样本检查。本阶段只实现边界检查，没有拟合任何统计量或模型。验证集合供后续早停／选择，测试集合不参与学习。上述是程序接口隔离，**不是操作系统文件访问权限隔离**；后续执行进程仍需限制可见文件。

### 划分与不完整数据

固定种子默认 `20261030`。`splits.csv` 为按系统和完整缺陷版本分组的约70/15/15划分；`loso_<system>.csv` 为完整留出该系统的划分，其余版本按70:15分成训练／验证。种子、输入指纹、划分身份、每组计数保存在 `split_audit.json`。

六个系统的计划都写入审计；缺失系统记 `unavailable`，不写空的伪划分。只有一个版本的小样本也记划分不可用，不拆产品凑出三个集合。真实数据当前有两系统，不能称为完成六系统实验；六折逻辑仅在明确合成的数据上验证过。

The conventions here are proposed for new CWM-FP code, not claimed to be native CLAP/PULP file layouts. Adapters must inspect originals rather than assume column names. Synthetic examples are under `data/examples/`.

## IDs and splits
Canonical product key: `(system_id, bug_version_id, product_id)`; canonical test key additionally includes `test_id`. `bug_version_id` must identify exact buggy version (including bug count) within a system; preserve raw path in provenance. Normalize system aliases in an explicit mapping table, not by guessing.

### `products.csv` (one row per product, labels sealed for inference)
Required:
- `system_id`, `bug_version_id`, `product_id`: nonempty identifiers.
- `observed_result`: `PASS` or `FAIL`, from the ORIGINAL suite; `F` must be `FAIL`, `FP/TP` must be `PASS`.
- `label`: `F`, `FP`, `TP`, or empty for unknown; only evaluator and train classifier adapter may read.
- `origin`: `natural`, `simulated`, `unknown`; evaluation strata only, NEVER a model feature.
- `source_variant_path`, `feature_config_path`, `test_suite_path`: path/provenance (not features).
- `label_provenance`: e.g. `oracle_fault_location`, `unknown`; metadata only.
Recommended: `manifest_sha256`, `bug_count`, `dataset_family`, `eligible_for_paper_subset`, `label_source_path` (sealed).

### `tests.csv` (one row per original test execution)
Required: `system_id,bug_version_id,product_id,test_id,result,coverage_path,trace_path`. `result` is `PASS|FAIL|UNKNOWN`. Tests excluded to synthesize false-passing products must NOT appear as available action candidates.

### `splits.csv` (one row per product)
Required: `system_id,bug_version_id,product_id,split` where `split` is `train|val|test`. Persist group IDs and seeds separately in JSON experiment config. `split` must be constant across a `(system_id,bug_version_id)` group. For LOSO, all rows from the holdout system are test. Only train rows may update model parameters or preprocessing statistics; val rows can be used for selection.

## `graphs.jsonl` (one JSON object per original test execution)
Required:
```
{
  "system_id":"Demo","bug_version_id":"Demo-1BUG-1","product_id":"p001","test_id":"T001",
  "tier":"GRAPH",
  "nodes":[{"id":"f:Weight","type":"feature","attrs":{"enabled":true}},
           {"id":"s:Weight.update:8:0","type":"statement","attrs":{"canonical_statement_id":"Weight|pkg.Class|update|8|0"}},
           {"id":"v:1","type":"value","attrs":{"dtype":"int","observed":false}},
           {"id":"o:1","type":"output","attrs":{}}],
  "edges":[{"source":"f:Weight","target":"s:Weight.update:8:0","type":"feature_controls","evidence":"static"},
           {"source":"s:Weight.update:8:0","target":"v:1","type":"produces","evidence":"dynamic","event_index":1}],
  "provenance":{"producer":"synthetic_example","coverage_only":false}
}
```
`type` node enum: `feature|statement|value|assertion|output`; edge enum: `feature_controls|control_dep|data_dep|produces|consumes|observes|temporal`. `evidence` must say `static`, `dynamic`, `observed_intervention`, or `unknown` (unknown evidence not usable as causal data). Graph endpoints must exist, no duplicate node IDs, and dynamic edges should carry nonnegative event ordering when available. `tier`: `GRAPH` or `VALUE`, and `COVERAGE` is NOT a valid dynamic graph. For actual runtime values, `attrs.observed=true`, typed `attrs.value`/`attrs.value_summary` plus safe hash/truncation policy. Do not save secret raw values.

## `events.jsonl` (one JSON object per time-ordered event)
Required: `system_id,bug_version_id,product_id,test_id,event_index,event_type,statement_id`. `event_type` examples: `input`, `statement_enter`, `branch`, `assignment`, `assertion`, `output`, `return`. Optional `thread_id,def_id,use_ids,variable_type,value_observed,value_summary,branch_taken,assertion_passed`. Event indices are ordered per trace; inter-thread happens-before is optional and cannot be inferred from wall clock alone.

## `interventions.jsonl` (one matched BEFORE/AFTER real rerun)
Required: `intervention_id,system_id,bug_version_id,product_id,base_test_id,action_type,action_target,action_value,valid,execution_kind,before_trace_ref,after_trace_ref,oracle_available,observed_failure`.
- `execution_kind`: `REAL_RERUN|MODEL_SIMULATION`. Only `REAL_RERUN` can supervise identified intervention effects or count as a verified new failure.
- `valid`: true only when feature model/typing/runner constraints are checked.
- `oracle_available`: true only with a credible oracle. `observed_failure` cannot be claimed when false.
- If a simulated FP was constructed by withholding tests, its withheld failing tests MUST NOT be selected by the agent.

## Feature ownership boundaries
- Allowed MODEL input: actual feature vector under current configuration, static CDG, observed per-test dynamic def-use, state/value events where available, original tests and outputs, optional train-derived suspiciousness score (must be computed fold-local).
- Disallowed MODEL input: `label`, `label_provenance`, `origin`, true bug locations/patch information, synthetic FP creation details, per-system IDs used as memorization tokens, test-split IDs, hidden removed tests, overall benchmark truth-derived labeling fields.
- Allowed supervised target: `label` on TRAIN observed `PASS` products (FP=1, TP=0). F products for world-model/test-failure auxiliary tasks.
- Allowed evaluator-only: all test-label fields, ground truth fault location for post-hoc FL scoring.

## Graph and transition quality requirements
- CDG extraction: distinguish static CFG from CDG; an arbitrary `if` ancestor is not automatically a verified CDG edge. Record parser/tool method and confidence.
- DDG extraction: actual reaching runtime def-use edge between dynamic events; do not claim exact DDG using only may-use static analysis.
- Canonical statement locations: preserve original `(feature source, qualified method, original line)` and composed product `(class, method, line)` with explicit map and unmapped/ambiguous counts.
- Repeated execution events: distinct event IDs/SSA-like values; loops and recursion must not be collapsed without marking information loss.
- Partial observability: unknown values or unavailable outputs must be masked and corresponding loss disabled.
- Instrumentation correctness: insertion must not alter original semantics, test outcomes, or feature composition; compare original vs instrumented test result and (when feasible) coverage.
# 阶段03B-4增补：有限框架调用效果证据

本增补不修改历史标签、划分或已冻结图。新控制证据版本为`framework_effect_v1`，旧`normal_path_v1`和`null_field_exception_v1`继续按原版本核验。

仅对真实观测到的`EvoAssertions.verifyException(String,Throwable)`及其`assertThrownBy`栈顶类名相等后返回路径建立有限摘要。调用者必须具有已认证的空字段异常前缀、真实字符串常量定义和未逃逸的精确JVM空指针异常引用。JAR、运行库、虚拟机、插桩器及原始框架字节码均绑定已审计摘要；不满足任一前提则拒绝。

`framework_evidence`记录原始日志、日志摘要、运行环境绑定和框架字节码路径。消费者重新读取原始日志和绑定文件，验证实际30条内部指令、进入/返回顺序及动态参数来源，再重算效果、数据依赖和质量；不能单独相信`accepted`。`framework_effects`记录参数值版本、物理调用/返回位置、可能的异常私有缓存和新数组写入、仍未知的内部状态。无返回值不得生成数值定义；可能写入不得生成虚构的内部写事件。调用后异常引用保守标记为外部修改可能，之后依赖其堆状态的访问仍拒绝。

图中`events`仍为完整产品/测试事件投影；`physical_trace_events`另计框架内部真实指令。内部指令全部保留在未压缩日志中，作为调用效果证据，不冒充具有特征源码身份的产品节点。产品回调不得被投影删除；反射/回调/缺少正常返回等路径继续拒绝。`framework_internal_events`用于数量对账。这是明确的库调用抽象，不是完整JVM状态图；当前22列模型特征不读取这些身份、摘要字符串或结果字段。

有限效果只能在实际返回标记后使用；异常引用只能从真实处理器可见位置开始使用。当前完整图及事后认证不能直接作为时间前缀输入：下一阶段须单独实现并检验前缀构造，排除未来事件/边、未返回效果、最终断言与完整轨迹统计。数值状态损失和干预损失仍关闭。

实现与实际适用范围详见`outputs/phase03B4_effect_contract.md`及`outputs/phase03B4_review_packet.md`。修改前契约保存在`data/processed/lddg/phase03B4_framework_v1/producer_before/DATA_CONTRACTS.md`。

## 阶段04P增补：无损产品数据包与观测可见时刻

新格式`lossless_product_package_v1`以产品完整三元身份和测试四元身份关联。每个产品的全部登记测试必须与清单一致；单图合格不能替代完整套件。使用`semantic_identity_v2`的22列语义/布尔特征及`full`七类关系；旧19列和旧消融关系保留版本含义，不自动混用。名称、行号、绝对位置、产品/版本身份、监督标签不作为数值身份特征。

包包含相对路径清单、每测试无对象张量文件和无损压缩元数据。保留全部事件/节点/边/值版本、静态指令目录、框架效果、质量和采集端认证摘要。原绝对路径改为审计标记，本机回执保留反查；路径字符串摘要与内容摘要必须区别。包不含FP/TP目标，目标仍由独立授权入口提供。

导出须通过既有源码/JVM证据和完整套件准入。消费者必须使用包外可信清单摘要，验证分片摘要、格式/大小、身份关联、事件顺序、端点、定义使用顺序和完整套件。迁移端不重新执行Java，内容核验不得称为重新证明执行语义。未认证图继续拒绝；不能删除不合格测试制造完整产品。

默认产品读取与训练命令不再截取前16测试。明确指定的旧消融/采样配置仍保留审计，但正式入口在打开训练目标前拒绝任一分区的采样损失。容量超限必须拒绝，不能静默截断。当前包限制单分片/解压元数据16MiB、清单4MiB、单图4096节点/4096事件/65536边、总包512MiB；旧JSON行文件128MiB保护不取消。逐测试迭代适用于受限内存；完整物化的64MiB只限制张量估算，不能代替进程内存监控。

现有模型保持测试内均值、产品内测试等权均值。逐测试分批前向已在固定参数、评价且不求梯度模式认证；带梯度的分片训练尚未认证。恢复读取的后缀不是完整套件，必须与前缀合并后核对全部测试。

未来时间前缀契约`prefix_observation_contract_v1`要求独立生产者提供真实可见时点。动态事件、使用及返回不得晚于当前已完成事件；静态结构不得带事后执行标记；最终结果、断言结果、后验/全轨迹统计和标签不得进入前缀输入。有值字段须有实际观测掩码，缺失保持null。当前仅实现契约检查，真实前缀生产器未实现；不得从完整事后图制造已认证预测数据。

范围、实际结果与剩余限制见`outputs/phase04P_review_packet.md`。修改前契约保存在`data/processed/graph_classifier/phase04P_v1/producer_before/`。

## 阶段04T-1增补：训练区限定的数据包训练验收

`train_graph --mode package_smoke`是独立的训练工程验收模式，复用原图编码器、关系传播与产品分类头。仅接受`grouped_bank_pilot_v1`继承的`grouped/train`、22列`semantic_identity_v2`及全部7类关系。必须显式提供包外清单摘要、原产品/测试清单、冻结划分摘要和授权训练目标摘要；先检查完整产品、分区和数据内容，再通过原`load_targets`接口连接17个训练产品。原接口会校验整个grouped训练目标文件，但其余产品的标签及观测不参与拟合。验证/测试只允许无标签内容核验，不调用模型预测、归一化或评价。

目标编码为FP=1、TP=0。一个完整产品生成一个输出及损失，测试内节点均值、产品内全部测试等权均值保持原语义；批次内产品等权，不把测试当成独立监督样本。图特征为固定语义/布尔编码，不拟合图统计量；归一化贡献者仅登记当前训练产品。输入身份、方法/类名、绝对位置、目标来源与缺陷位置不得进入数值特征。

整产品路径用于本阶段训练。新增带梯度的逐测试分批路径按各批实际测试数加权，并在产品全部测试汇聚后计算损失和更新；24测试及6测试真实产品的输出、损失、逐参数梯度和一次更新已按预先固定绝对容差1e-6、相对容差1e-5验证。该认证限于当前模型、设备和数据；旧不求梯度的推理接口保持原约束。分批读取仍保留全部反向传播计算图，不保证常量激活内存。合理的局部零梯度不等于链路断开；验收需要编码器、关系传播及分类头整体具有实际梯度和更新证据。

工程模型保存为`engineering_train_only`，绑定清单、列版本、关系模式、协议、目标摘要、模型/优化器配置与种子，保存模型、优化器、随机状态和归一化记录。使用包外摘要检查保存文件，拒绝跨格式或队列静默恢复。每次验收写独立目录，已有计划不可覆盖。全部验收共用不超过400更新、1800秒模型执行上限；实际计划1740秒并另留60秒检查余量。不使用验证早停、测试标签或超参数搜索。程序级文件访问检查不能声称为操作系统绝对隔离。

阶段04T-1完成了真实训练区有限拟合；其损失下降仅为工程学习证据，不是泛化成绩。真实时间前缀生产器、完整数值状态和合法干预对仍缺失，相关世界模型训练保持关闭。证据见`outputs/phase04T1_review_packet.md`。修改前契约保存在`D:/dataset/whn/cwm_fp_generated/data/processed/graph_classifier/phase04T1_cpu_v1/producer_before/DATA_CONTRACTS.md`。

## 阶段04R-1增补：固定训练／验证先导比较

新增`train_graph --mode package_compare`与独立`graph_evaluator --mode package_validation`。旧04T1的`package_smoke`模式仍只允许训练目标。本阶段须先冻结`outputs/phase04R1_protocol.json`及包外摘要，验证源码、队列、公开清单/划分、全部原始测试、22列和7类关系。只用当前17训练产品拟合；独立评价端允许读取本协议5验证产品目标，原接口校验分区文件后仅保留这5产品。测试目标和其他协议目标禁止读取，测试图仅作无标签内容核验，不生成测试预测。

三种神经模型使用同22列、隐藏宽16、2层及测试/产品两级均值。新增`nodeonly`只保留节点自身变换；`untyped`保留全部节点列、边方向和重边，只合并关系，不能与会删除节点类别的旧`agnostic`混称。R-GCN按每种关系的入边均值求和，关系合并模型按全部入边求均值，归一化差异必须随对照结论披露。配置列保持原掩码，数值状态不进入此轮分类输入。

常数比例只由训练目标计算。统计逻辑回归使用冻结16列、训练区标准化和固定L2设置；它不是CLAP/PULP复现。三个神经方法各用20261101/02/03种子，Adam0.001、无类别权重、产品批次2、200轮；另沿用原4训练产品的500步诊断。全阶段上限20000优化迭代/3600秒，神经与统计迭代分别记账。所有产品每轮一次，保留末批，不截断套件。

只在10/50/100/200轮封存验证预测，固定训练完成后由独立评价端统一评分，按验证产品平均对数损失选择；改善不超过1e-6取较早轮次，所有种子均报告，阈值固定0.5。FP=1、TP=0，混淆矩阵行真实/列预测，顺序均为TP/FP。报告AP而非混称梯形PR面积；单类时AP/ROC为空且不作排序解释，显式二分类损失仍有效；精确率/召回率/F1除零取0并报告支持数。按版本统计损失/错误，种子重复不计为独立样本。

本次真实验证标签访问是授权评价的一部分，必须明确披露，不能继续声称未读任何保留标签。评价结果仅用于冻结的轮次选择和报告，不回流到梯度、表示、标准化或采样。受控目录及程序访问钩子不是操作系统绝对隔离。完整事后图分类结果不认证时间前缀、状态转换或因果效应。详见`outputs/phase04R1_review_packet.md`；修改前契约位于本阶段独立运行目录`producer_before/`。

## 阶段05B：结构转换预测输入与训练契约

本阶段沿用05A物理执行前缀，不改观察器、JVM语义或原始标签。新增结构转换格式`structural_transition_v1`及缓存`structural_transition_cache_v1`。输入对象`StructureInput`只包含22列语义/布尔矩阵、局部边端点、关系类型及已观测的当前事件局部索引；它不接受目标字典。配置两列保持零掩码，原始数值及其观测标记继续保留在05A中，本轮不进入模型。锚点必须是最后已观测动态事件，不能由后继目标选择。

实际关系仅有静态控制依赖、动态定义使用、值产生、值使用、时间邻接五种；没有配置归属、输出观察或新造动态控制关系。移除依赖的图模型同时移除前四种关系，包括经值节点恢复数据传递的间接路径；保留所有节点和时间邻接，归一化仍为实际可见入边的每关系均值。相同静态子图的共享计算必须满足静态/动态部分无跨边，且通过完整复制方式的输出、梯度等价验证；各动态轨迹不可连边。

顺序模型逐前缀读取已观测事件的22列，不从整条轨迹最终属性切片制造早期输入。图查询由锚点表示、动态事件均值、静态均值及存在标志组成。局部索引、哈希、真实名称和审计身份不能成为可学习数值特征。精确输入签名排除原始值和来源身份，只认张量相等；不得把签名不相同等同于程序语义新颖或模型函数不相同。

转换目标独立加载：A是下一个实际可观测产品/测试事件的九类语义；B仅为05A认证条件分支的跳转/顺序结果。无效目标为-1并关闭损失，非正常结束标签。静态覆盖与弃权由当前指令和正常静态后继决定，后验异常/框架间隔仅作评价分层。

损失为覆盖A、弃权A、B分别按位置→轨迹→产品→版本等权归一化后固定除以3。轨迹批次4、前缀微批64，每轨迹批次仅一次更新；固定全局权重乘每轮批次数构成批次估计，缺失监督部分贡献零而不改变分母，末批保留。3模型×3种子×20轮全部遍历训练轨迹；候选轮次5/10/20，固定验证弃权A和B版本宏平均归一化损失选择，1e-6内取更早候选。具体代码和设备以`outputs/phase05B_protocol.json`及独立回执为准。

本阶段所有产品FP/TP目标和测试轨迹禁止读取。训练目标仅拟合本协议训练区；验证转换目标可由缓存准备独立保存、由独立评价入口评分，正式拟合进程禁止打开验证目标缓存。重复前缀只做存储去重，不删除样本、不改变经验目标分布。已有05A验证统计公开，本轮不得声称验证任务从未查看。一步真实观测条件预测不等同自主多步展开、完整状态模拟、下游分类改善或干预因果证据。

## 阶段05C-1：执行前局部状态诊断与容器工程验收

新增独立诊断格式`pre_instruction_local_state_v1`，不改05A/05B输入、图关系、目标、冻结清单或旧模型。输入截取点为原解释器建立当前调用帧之后、当前指令消费操作数及定义结果之前。当前标量回调已经到达时拒绝此入口。操作数按栈底到栈顶保留value1/value2顺序，调用帧、值版本、物理时钟和静态身份只作证据连接。

来源严格区分`DIRECT_OBSERVED`、`STATIC_CONSTANT`、`SEMANTICALLY_DERIVED`和`UNKNOWN`。主符号核验仅接受前两者；有限复制链的推导值不能称为新的实测值。复制值的可见时刻不得早于实际完成证据。没有观测的堆槽、引用日志散列、当前指令未来回调及最终测试结果不得填入操作数。未知值使用空值及独立掩码，禁止补实测零。

逐位置数据将`pre_state_evidence`、`audit`、`learning_input`及后连接的`target`分开。可学习接口只有精确操作码、输入个数、类型、来源类别、可信可见掩码和允许的局部值；不包含类名、方法名、源码行、字节码偏移、产品/版本/帧/值唯一身份。该接口本阶段仅用于诊断，没有送入旧模型训练。精确输入冲突统计不得解释为总体预测上界或分类成绩。

主语义规则限于已声明的32位有符号整数条件比较及有限整数运算，空引用仅接受明确空常量；浮点、引用相等和宽数值转换保持未认证。`i2l`的实际操作名称可保留，但不意味着其语义已经通过本阶段认证。完整分母为523个分支和547组既有局部配对，禁止按可解释性删样本。

`event_of_static_instruction`只允许由已有规范身份、真实类文件摘要、完整方法描述符和字节码位置验证后写入独立侧表；不向旧图加边，不自动成为动态控制依赖。原22列图输入继续保持不变。

跨平台导出保留原文件和独立摘要，另建相对路径导出回执及代码快照；不得把Windows绝对来源路径改写后声称原摘要未变。容器验收标记`server_acceptance_only`，只用训练区做优化，不选择新验证检查点。GPU恢复区分状态张量逐位一致和输出在预先声明数值容差内一致；不得因并行浮点归约而声称图模型逐位输出复现。本阶段实际总更新226次，各旧模型仅一次完整遍历，无正式新排名。

## 阶段05C-2：独立局部转换准入与可迁移来源

新增`email_local_source_v1`与`local_transition_dataset_v1`，独立于完整产品分类资格和05A/05B冻结产物。当前生产范围固定为Email原分组训练区版本8的2产品/24既有轨迹；不得借另一留出协议展开验证/测试轨迹。开发访问单独登记，不改变原划分，不再把该版本称为未曝光系统证据。旧契约保存在本阶段`source_before/DATA_CONTRACTS.md`。

- 来源清单由包外SHA256绑定，所有复制文件逐项核验；实际类字节、源码、AST、静态表及执行状态之间的连接保留。原Windows路径只作审计，显式解析表指向新根目录。`program_identities`的可选`artifact_resolver`只改变读文件位置，不改变原证据字符串和身份摘要；两个产品的既有静态身份必须逐项相等，未知不提升。
- 分别记录`source_execution_verified`、`trusted_prefix_available`、`local_input_verified`、`local_target_verified`、`local_transition_admitted`与继承的`full_product_admission`。原始失败测试有合法完成记录且插桩结果一致时可供局部审计，但失败产品不成为FP/TP候选。完整运行器接受的退出码1不能单独被称为采集失败。
- 从原测试入口逐记录重放；最早未认证身份、帧、栈、字段定义、别名、控制或调用效果形成永久可信截止点。之后的直接标量回调不恢复信任；未观测初始堆保持未知。当前指令处理前捕获操作数，目标后连接，跨截止点的目标设空且关闭掩码。未来信息不能用于填输入。
- 本阶段局部适配器独立支持真实`getstatic`完成标记。原插桩器在178指令后输出`###RET@`而非标量回调；严格核对实际操作码和位置。只登记完成时刻，不产生字段数值/定义，不建立方法返回或无副作用证明。原观察器和旧冻结数据保持原格式含义。
- `inputs.jsonl`只含连接用查询键与学习字段：精确操作码、输入个数、类型、四类来源、可信数值及掩码。`targets.jsonl`另存目标值、可见时刻和准入位；`evidence.jsonl`保留完整身份、物理时钟、值版本、来源、帧和拒绝原因。公开输入读取器不打开目标文件；整数以64位容器无损保留已认证32位JVM整数，引用空值有独立标志，缺值占位0始终关闭观测掩码。原始身份/路径/类方法名/源码位置不进入可学习数值。
- 所有候选指令分母保留，产品代码与测试/框架分开。其他指令的审计记录不自动成为新增状态预测任务；沿用05C1有限整数和空常量比较规则，i2l、浮点、引用相等、数组/类型转换、一般库副作用和通用异常不升级。后缀的直接回调或正常CFG后继可以单独审计，但在不可信前缀之外不能成为已认证训练目标。
- 每条轨迹首/中/末与所有实际调用、返回、分支、处理器关键位置做独立截断和后缀扰动；输入摘要一致不能代替数值/语义认证。生产、独立来源核验和读取均需真实容器执行。本轮3325事件完整读取、4395行跨系统字段检查通过，但67个产品分支/算术候选的正式局部准入为0；一致率为空，不能写成100%。完整产品准入未改变，没有训练或干预。

## 阶段05C-3A：字段读取证据分层（独立审计，不改严格准入）

新增`field_read_evidence_protocol_v1`及`models/field_evidence.py`，只审计05C2固定训练区24条Email轨迹。旧契约、阶段状态保存在`D:/dataset/whn/cwm_fp_generated/data/processed/local_state/phase05C3A_v1/source_before/`。不重解释或覆盖05C2严格前缀数据。

- 五类证据：A为真实读取载荷、完成时刻、身份、帧和栈关联可认证，仅上游写入来源未知；B为读取完成但无可信数值／引用载荷；C为回调与身份、类型、帧或接收者的关联未认证；D为更早已存在未知语义；E保留其他情况，包括原本已具有写入来源的读取。分类不是完整产品准入标记。
- 独立保存`observed_value_valid`、`upstream_write_provenance_complete`、`current_heap_complete`、`local_transition_valid`、`full_product_admission`。已有写入来源的真实观测仍可保留观测有效；“不属于A类”不能自动解释为缺失值。
- 真实字段完成标记与值回调分别记录是否存在、关联可信性和物理时钟。178路径的`###RET@`不是数值回调；180引用回调的`Obj:`散列不是无碰撞引用身份，也不是独立空值类型标记。不得用散列0、相同散列、默认初始化、未来分支／加法／断言恢复数值、空值或堆别名。
- 字段声明及类型由原静态表和已核验类文件连接；实例接收者使用原分配事件和已有栈／局部变量／调用证据，不按日志散列合并。第一次栈消费的事件关联与真实值转换有效性分别保存，不能据此伪造此前写入或动态依赖边。
- 本阶段的整数／布尔载荷检查及可见时刻辅助函数只检查证据；不是新局部转换生产器。只有真实A类存在，才可另行激活已验证的独立观测条件模式；原严格模式的首次缺口永久拒绝规则保持。后续标量回调不能穿过未知身份、类型、库副作用、异常或多线程恢复整条轨迹。
- 当前15条首个字段缺口均B类：9条无读值回调，6条有对象散列但不能认证引用值。A类为0，独立转换模式未启用；新增转换和完整产品均0。覆盖内一致率为null。原67个产品分支／算术候选分母保留，未知或未执行指标不得填伪测量。
- Java新执行必须先通过原运行时、类、JAR、代理摘要、构建源码、JUnit生命周期、Linux路径与临时目录条件验收；源码闭包齐全不代表Java可执行。所有结果、失败回执、字节证据在新目录，宿主机及原测试／业务源码不修改。

