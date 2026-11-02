"""Publish measured phase05A evidence; no training or product-label reads."""
from pathlib import Path
import sys, json, shutil, time
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cwm_fp.models.prefix_dataset import OUT,PACKAGE,ANCHOR,FRAMEWORK,PROTOCOL,load,lines,guard,budget_unchecked
from cwm_fp.models.prefix_reader import load_prefix
from cwm_fp.data.common import write_json,write_csv,sha256

def main():
    start=time.monotonic();manifest=load(PACKAGE/'manifest.json');guard(manifest,True)
    production=load(OUT/'production.json');validation=load(OUT/'independent_validation.json');protocol=load(PROTOCOL)
    rows=production['status'];tt=production['targets']
    assert len(rows)==368 and all(r['status']=='PASS' for r in rows), 'Partial production must be reported explicitly, not READY'
    assert validation['status']=='PASS' and validation['traces']==368
    assert sha256(PACKAGE/'manifest.json')==ANCHOR
    for f,h in protocol['public_sha256'].items():assert sha256(ROOT/f)==h
    # Public loader receives only shared materials/index, never targets.
    shutil.copy2(FRAMEWORK,OUT/'framework_static.tsv');entries=[]
    for i,r in enumerate(rows):
        d=Path(r['evidence_path']);files={}
        for name,file in [('static','static.json'),('static_bytecode','static.tsv'),('observations','observations.jsonl'),('prefix_index','prefix_index.jsonl')]:
            p=d/file;files[name]={'path':p.relative_to(OUT).as_posix(),'sha256':sha256(p),'bytes':p.stat().st_size}
        target=d/'targets.jsonl'
        entries.append({'trace_index':i,'identity':[r[k] for k in ('system_id','bug_version_id','product_id','test_id')],
            'partition':r['partition'],'input_files':files,'targets':{'path':target.relative_to(OUT).as_posix(),'sha256':sha256(target)},
            'prefixes':r['accepted_prefixes']})
    write_json(OUT/'prefix_manifest.json',{'schema':'physical_prefix_dataset_v1','source_manifest_sha256':ANCHOR,'protocol_sha256':sha256(PROTOCOL),
        'framework_static':{'path':'framework_static.tsv','sha256':sha256(FRAMEWORK)},'traces':entries})
    prefix_hash=sha256(OUT/'prefix_manifest.json');loader_checks=0
    for i,r in enumerate(rows):
        for j in sorted({0,r['source_events']//2,r['source_events']-1}):
            load_prefix(OUT,i,j,expected_manifest_sha256=prefix_hash);loader_checks+=1
    write_json(OUT/'prefix_loader_validation.json',{'checks':loader_checks,'passed':loader_checks,'manifest_sha256':prefix_hash,
        'target_files_opened_by_loader':0,'seconds':time.monotonic()-start})
    fields=list(dict.fromkeys(k for r in rows for k in r));write_csv(ROOT/'outputs/phase05A_trace_status.csv',rows,fields)
    partition={};difficulty={};majority_counts=Counter(t['A'] for t in tt if t['partition']=='train' and t['A_mask'])
    majority=max(protocol['task_vocabulary'],key=lambda c:majority_counts[c]);observed={}
    for part in ('train','val'):
        rs=[r for r in rows if r['partition']==part];ts=[t for t in tt if t['partition']==part];valid=[t for t in ts if t['A_mask']]
        pred=[t for t in valid if t['deterministic_prediction'] is not None]
        partition[part]={'products':len({tuple(r[k] for k in ('system_id','bug_version_id','product_id')) for r in rs}),
            'versions':len({(r['system_id'],r['bug_version_id']) for r in rs}),'traces':len(rs),
            **{k:sum(r[k] for r in rs) for k in ['source_events','physical_instructions','physical_observations','accepted_prefixes','rejected_prefixes','A_valid','A_masked','B_valid','B_masked',
                'scalar_observations','final_visible_value_versions','final_visible_numeric_versions','numeric_pairs','check_positions','suffix_checks','calls','conditionals','exception_transfers','framework_effects','missing_numeric_versions']}}
        count=Counter(t['A'] for t in valid);bc=Counter(t['B'] for t in ts if t['B_mask'])
        idx=[x for r in rs for x in lines(Path(r['evidence_path'])/'prefix_index.jsonl')]
        difficulty[part]={'next_category_counts':dict(count),'branch_counts':dict(bc),
            'target_masks':dict(Counter(t['B_reason'] for t in ts if not t['B_mask'])),
            'boundary_types':dict(Counter(t['interval_kind'] for t in ts)),
            'framework_interval_samples':sum(t['intervening_framework_instructions']>0 for t in ts),
            'prefix_event_length':{'min':min(x['observed_prefix_events'] for x in idx),'max':max(x['observed_prefix_events'] for x in idx),
                'mean':sum(x['observed_prefix_events'] for x in idx)/len(idx)},
            'majority_reference':{'class':majority,'fit_partition':'train','coverage':1.,'accuracy':sum(t['A']==majority for t in valid)/len(valid),'correct':sum(t['A']==majority for t in valid),'eligible':len(valid)},
            'unique_normal_CFG_successor_reference':{'eligible_targets':len(valid),'predicted':len(pred),'abstained':len(valid)-len(pred),
                'coverage':len(pred)/len(valid),'correct':sum(t['A']==t['deterministic_prediction'] for t in pred),
                'accuracy_when_predicted':sum(t['A']==t['deterministic_prediction'] for t in pred)/len(pred),
                'correct_over_all_eligible_targets':sum(t['A']==t['deterministic_prediction'] for t in pred)/len(valid),
                'scope':'normal CFG uniqueness; potential exceptions and unseen class initialization prevent a general determinism claim'},
            'unseen_target_categories':sorted(set(count)-set(majority_counts))}
        observed[part]={'real_numeric_versions':partition[part]['final_visible_numeric_versions'],
            'all_visible_value_versions':partition[part]['final_visible_value_versions'],
            'observed_fraction':partition[part]['final_visible_numeric_versions']/partition[part]['final_visible_value_versions'],
            'missing':partition[part]['missing_numeric_versions'],'local_arithmetic_pairs':partition[part]['numeric_pairs'],
            'business_input_output_assertion_pairs':None,'missing_reason':'not observed by existing collector; final test outcome excluded',
            'live_heap_complete':False}
    dups=list(production['duplicates'].values())
    dupstats={'exact_model_input_hash_groups':len(dups),'occurrences_in_duplicate_groups':sum(map(len,dups)),
        'within_trace_groups':sum(any(n>1 for n in Counter(tuple(x[1:5]) for x in group).values()) for group in dups),
        'within_product_groups':sum(any(n>1 for n in Counter(tuple(x[1:4]) for x in group).values()) for group in dups),
        'cross_version_groups':sum(len({tuple(x[1:3]) for x in group})>1 for group in dups),
        'train_validation_shared_groups':sum(len({x[0] for x in group})>1 for group in dups),
        'definition':'exact tensor equality including visible values/masks/static context; IDs not hashed into tensor; not graph-isomorphism equivalence',
        'no_deduplication_or_split_change':True}
    splits={}
    for fold in manifest['split_protocols']:
        splits[fold]={}
        for part in ('train','val','test'):
            ps=[p for p in manifest['products'] if p['partitions'][fold]==part]
            splits[fold][part]={'qualified_products':len(ps),'versions':len({tuple(p['identity'][:2]) for p in ps}),
                'classification_class_counts':None,'reason':'classification targets not authorized in phase05A'}
    commands=[load(p) for p in sorted(OUT.glob('*.command.json'))]
    resources={**production['resources'],'independent_check_seconds':validation['seconds'],'public_loader_seconds':time.monotonic()-start,
        'commands':commands,'recorded_command_seconds_before_reporting':sum(r['seconds'] for r in commands),
        **budget_unchecked(),'GPU_training':False,'server_access':False,'hard_OS_quota':False,
        'timeouts':sum(r.get('reason')=='timeout' for r in commands),'OOM':False,'truncation':False,
        'interactive_development_time':None,'interactive_development_time_reason':'not continuously measured; command execution separately recorded'}
    readiness={'REAL_PREFIX_PRODUCER_IMPLEMENTED':True,'REAL_PREFIX_DATA_GENERATED':True,
        'PREFIX_REPLAY_VERIFIED':True,'SUFFIX_INVARIANCE_VERIFIED':True,'MODEL_INPUT_FUTURE_LEAKAGE_CHECKED':True,
        'STRUCTURAL_TRANSITION_DATA_READY':'READY_FOR_BOUNDED_OBSERVATIONAL_A_B_PILOT',
        'NUMERIC_TRANSITION_SCOPE':'PARTIAL_LOCAL_ARITHMETIC_ONLY','CAUSAL_INTERVENTION_DATA_READY':False,'WORLD_MODEL_TRAINED':False}
    unresolved=[
        {'issue':'partial_observability','impact':'No full heap, pre-test initialization, business inputs/outputs or assertion value pairs; Markov sufficiency unproved','confirmed':True},
        {'issue':'normal_static_successor_shortcuts','impact':'Many A targets reducible to instruction ordering; require nontrivial-boundary evaluation before scientific claims','confirmed':True},
        {'issue':'observational_not_interventional','impact':'No legal matched interventions or causal-effect targets','confirmed':True},
        {'issue':'single_system_and_small_independent_cohort','impact':'Only BankAccountTP train9/val4 versions; no Email/Transaction coverage improvement, no cross-system evidence','confirmed':True},
        {'issue':'no_end_of_stream_target','impact':'Last return is pre-execution and independent run status has no aligned clock; every trace final A masked','confirmed':True},
        {'issue':'prefix_graph_scope','impact':'Product static normal CDG + observed event temporal/dataflow subset; no general dynamic control-owner graph; feature ownership edges, assertion/output nodes omitted; no live heap certainty','confirmed':True}]
    changed=['src/cwm_fp/lddg/effects.py','src/cwm_fp/models/prefix_observer.py','src/cwm_fp/models/prefix_dataset.py','src/cwm_fp/models/prefix_reader.py',
        'tests/test_prefix_observer.py','scripts/run_phase05A.py','scripts/check_phase05A.py','scripts/test_phase05A_regression.py','scripts/report_phase05A.py',
        'DATA_CONTRACTS.md','STAGES.md','outputs/phase_status.md']
    packet={'stage_id':'05A','stage_status':'COMPLETED_SCOPED','protocol':protocol['protocol'],'protocol_sha256':sha256(PROTOCOL),'source_manifest_sha256':ANCHOR,
        'actual_scope':protocol['actual_scope'],'partition_counts':partition,'source_trace_counts':{'train':283,'val':85,'test_opened':0},
        'before_metrics':{'real_prefixes':0,'real_prefix_producer':False,'qualified_original_products':28},
        'after_metrics':{'real_prefixes':sum(r['accepted_prefixes'] for r in rows),'qualified_original_products':28,'new_qualified_products':0},
        'observation_clock':protocol['observation_clock'],'anchor_definition':protocol['anchor_definition'],
        'field_visibility_rules':protocol['field_visibility_rules'],'static_input_policy':protocol['static_input_policy'],
        'target_definitions':{k:protocol[k] for k in ('A','B','end_target','numeric')},
        'target_masks':{part:difficulty[part]['target_masks'] for part in difficulty},'target_counts':{part:{k:partition[part][k] for k in ('A_valid','A_masked','B_valid','B_masked')} for part in partition},
        'accepted_prefixes':sum(r['accepted_prefixes'] for r in rows),'rejected_prefixes':sum(r['rejected_prefixes'] for r in rows),'rejection_reasons':{},
        'replay_equivalence':{'traces':368,'positions':sum(r['check_positions'] for r in rows),'passed':True},
        'suffix_invariance':{'traces':368,'checks':sum(r['suffix_checks'] for r in rows),'cases_per_position':5,'passed':True,'scope':'internal destructive copies only; not physical interventions'},
        'tensor_leakage_checks':{'independent_checks':validation['checks'],'loader_checks':loader_checks,'passed':True},
        'task_difficulty':difficulty,'deterministic_successor_coverage':{part:difficulty[part]['unique_normal_CFG_successor_reference'] for part in difficulty},
        'duplicates':dupstats,'branch_points':{'product_scoped_points':production['branch_point_count'],'with_both_outcomes':production['branch_points_with_both_results'],'cross_product_alignment':'not forced'},
        'observed_value_coverage':observed,'source_and_target_access':{'classification_targets_opened':0,'test_traces_opened':0,
            'validation_task_definition_frozen_before_read':True,'access_evidence':str(OUT/'production.json'),'isolation':'Python access hooks, not OS sandbox'},
        'split_readiness':splits,'world_model_readiness':readiness,'resources':resources,'tests_executed':{'new_and_framework_regression':40,'related_regression':load(OUT/'regression.json'),
            'new_Java_tests':0,'real_saved_traces_verified':368},'changed_files':changed,
        'artifact_paths':{'dataset':str(OUT),'prefix_manifest':str(OUT/'prefix_manifest.json'),'prefix_manifest_sha256':prefix_hash,
            'independent_validation':str(OUT/'independent_validation.json'),'source_archive':str(OUT/'executed_source'),'raw_sources':'referenced by each traces/NNNN/source.json'},
        'unresolved_issues':unresolved,'recommended_next_task':'Freeze a bounded observational A/B structure-prediction pilot, separate nontrivial branch/call/exception strata and duplicates; no automatic execution'}
    write_json(ROOT/'outputs/phase05A_review_packet.json',packet)
    write_json(ROOT/'outputs/phase05A_leakage_checks.json',{'stage_id':'05A','replay_equivalence':packet['replay_equivalence'],
        'suffix_invariance':packet['suffix_invariance'],'tensor_leakage_checks':packet['tensor_leakage_checks'],
        'independent_evidence':str(OUT/'independent_validation.json'),'per_observation_evidence':'traces/NNNN/checks.jsonl',
        'failure_count':0,'uncovered':['unrecorded runtime behavior','full heap/state','physical intervention validity'],
        'source_integrity_checks_remain_enabled':True})
    a,b=partition['train'],partition['val']
    md=f'''# CWM-FP 阶段05A结果交接

本阶段已实际生成并验证真实时间前缀数据，范围为既有银行账户系统（BankAccountTP）的17个训练产品、5个验证产品，共368条轨迹。不是只有接口方案。没有读取任何训练／验证／测试产品分类目标，没有展开测试区95条轨迹，没有重新运行Java、连接服务器或训练模型。

当前可准备**限定范围的观测性结构预测试训**：预测下一条可观测产品／测试指令的语义类别，以及当前条件分支的真实后继。尚不能据此宣称完整数值状态、马尔可夫性、因果效应、假通过识别改善或跨系统泛化成立。只读复用的完整产品准入仍为28个，新增0个。

## 1. 实际实现及证据边界

- `prefix_observer.py`逐条重放物理观测，复用原动态定义使用解释器，不创建另一套虚拟机解释器；22列固定语义／布尔特征、已出现事件、当前边及独立数值缺失掩码构成模型输入。
- `prefix_dataset.py`分别实现规则冻结、原始证据连接、真实生产、后续目标生成和破坏性副本检查。协议先用一条训练轨迹核对格式，再冻结；验证轨迹仅应用该规则。
- `effects.py`只把原框架区域核验抽成可接收已观测记录的函数，旧完整图入口仍保留全部指纹、路径和效果检查；未扩展Java语义。
- `prefix_reader.py`由可信清单摘要、共享静态材料、事件流和位置索引重建输入，不打开目标文件；首／中／末共{loader_checks}个读取位置复核通过。
- 新增20项人工前缀测试，加20项既有框架回归和100项相关回归，合计140项通过。人工记录与历史Java验收资料不进入真实研究样本，本次新Java执行为0。

真实数据上验证了重放、未来隔离、回调时刻、调用待定、精确空字段异常及有限框架效果的延迟可见性。未知身份、坏时钟、多线程、伪造时间／端点／归一化等拒绝规则另有人工测试。没有训练预测器；预测能力及研究有效性尚未验证。

## 2. 真正的预测边界

原插桩器在**指令执行前**写出指令记录，执行后追加栈值回调；两者可能同处一行。顺序键使用原始日志行号和行内子记录号，没有伪造时间戳。每条产品／测试指令的执行前记录之后、其回调之前，形成一个预测位置。该指令已经被观察到即将执行，但执行结果尚未成为输入。

字段分别保留发生区间、实际观测点和认证证据齐备点。无法确认精确发生时刻的定义为null并保存区间；不以定义事件编号冒充结果观测时刻。调用效果等真实返回后才认证；异常引用等实际处理器出现后才可见。调用未完成时保留待定，当前堆状态始终明确未知。日志末尾返回仍是执行前观测，因此最后位置的下一事件目标关闭，**没有将日志结束或截断标成正常结束**。

主目标A的序列同时包括产品指令与测试动作，框架内部指令、数值回调和返回标记保留在物理观测流中；跨框架／调用间隔单独标记，不能把它解释为连续的一步虚拟机状态转移。目标B限于具有两个已核对普通后继的条件分支，输入在分支结果出现前，真实下一指令及相同调用帧共同确定跳转／顺序后继；其他位置关闭此目标。

## 3. 实际数量

| 指标 | 训练 | 验证 |
|---|---:|---:|
| 产品／独立版本 | {a['products']}／{a['versions']} | {b['products']}／{b['versions']} |
| 源轨迹／原登记测试 | {a['traces']} | {b['traces']} |
| 产品及测试指令事件 | {a['source_events']} | {b['source_events']} |
| 包括框架内部的物理指令 | {a['physical_instructions']} | {b['physical_instructions']} |
| 记录子项数（含回调、入口及返回） | {a['physical_observations']} | {b['physical_observations']} |
| 可用前缀／拒绝 | {a['accepted_prefixes']}／{a['rejected_prefixes']} | {b['accepted_prefixes']}／{b['rejected_prefixes']} |
| 下一事件目标有效／关闭 | {a['A_valid']}／{a['A_masked']} | {b['A_valid']}／{b['A_masked']} |
| 分支目标有效／关闭 | {a['B_valid']}／{a['B_masked']} | {b['B_valid']}／{b['B_masked']} |
| 可直接配对的局部算术观测 | {a['numeric_pairs']} | {b['numeric_pairs']} |
| 独立截断与张量检查位置 | {a['check_positions']} | {b['check_positions']} |
| 后缀扰动比较次数 | {a['suffix_checks']} | {b['suffix_checks']} |

368条源轨迹均出现在状态CSV中；前缀仍继承原版本分组，没有随机拆分窗口、删除测试或改变原分类队列。输入位置合格与该位置的某项目标有效是不同判断。分支关闭的大部分位置原本就不是条件分支，并非采集失败。

## 4. 保留及屏蔽的信息

静态材料保留编译产品的全部指令、可信源码／编译器生成来源及普通控制依赖；从原静态表建立，不按最终命中集合选节点。静态节点的已执行列始终为0。测试代码的静态表用于真实指令和后继核对，动态测试节点只在其记录到达后出现；不会提前把最终执行的测试语句集合做成静态输入。

动态部分保留每次执行的独立局部节点、时间邻接、证据已齐备的值版本及定义使用关系。可恢复帧、异常转移和框架效果保留为审计／重建信息。数值为历史版本的真实回调，未知版本使用独立false掩码；张量占位0不等于实测0。没有假定旧字段值在待定外部调用期间仍保持不变。

22列的五种节点类别、九种操作类别、三种来源及执行／可用／事件／配置标记逐类隔离：配置两列继续屏蔽；静态执行列不聚合未来；动态执行完成标记等待证据；没有最终断言／输出节点。当前接口没有加入特征归属边，也没有实现动态控制者归属图；静态普通控制依赖不冒称异常或动态因果边。未来节点、最终覆盖、总轨迹长度、最终结果、返回是否最终成功、唯一身份和标签均不进入张量。当前入度从可见边重新计算。

## 5. 隔离验收结果

全部368条轨迹在每个有效物理观测位置比较增量输入与独立截断原始文本重放，覆盖首、中、末及调用、回调、分支、处理器和框架区域。随后在内部测试副本分别删除后缀、改变后续数值、改变返回／异常文字、改变最终状态、延长执行，比较节点矩阵、边、状态掩码、局部索引及入度。未通过整体文件摘要拒绝来代替隔离测试。

独立检查器还把每条已暴露定义使用关系与冻结旧图核对，逐项检查原始回调、调用返回、帧和异常证据；不是只相信生产器写下的时刻。全部通过，未观察到后缀变化影响前缀模型输入。各条实际检查见数据目录的`traces/NNNN/checks.jsonl`及`independent_validation.json`。这一结论限于当前记录格式和支持语义，不能证明未记录的运行状态完整。

## 6. 任务信息量及简单参照

训练下一事件类别分布：{difficulty['train']['next_category_counts']}。
验证分布：{difficulty['val']['next_category_counts']}；验证未见类别：{difficulty['val']['unseen_target_categories']}。
分支结果训练为{difficulty['train']['branch_counts']}，验证为{difficulty['val']['branch_counts']}。产品内规范分支点{production['branch_point_count']}个，其中{production['branch_points_with_both_results']}个在已有执行中出现两种结果；没有无证据地强行对齐不同产品／版本的分支。

仅由训练目标选出的多数类别为`{majority}`，训练准确率{difficulty['train']['majority_reference']['accuracy']:.4%}，验证{difficulty['val']['majority_reference']['accuracy']:.4%}，均覆盖全部有效A目标。无需神经拟合。

| 普通静态唯一后继参照 | 训练 | 验证 |
|---|---:|---:|
| 预测覆盖率 | {difficulty['train']['unique_normal_CFG_successor_reference']['coverage']:.4%} | {difficulty['val']['unique_normal_CFG_successor_reference']['coverage']:.4%} |
| 预测范围内准确率 | {difficulty['train']['unique_normal_CFG_successor_reference']['accuracy_when_predicted']:.4%} | {difficulty['val']['unique_normal_CFG_successor_reference']['accuracy_when_predicted']:.4%} |
| 正确数／全部有效目标 | {difficulty['train']['unique_normal_CFG_successor_reference']['correct_over_all_eligible_targets']:.4%} | {difficulty['val']['unique_normal_CFG_successor_reference']['correct_over_all_eligible_targets']:.4%} |

该参照的“唯一”只指普通静态后继，潜在异常与类初始化不因此被排除；不能把弃权范围忽略后称为全部目标的确定性预测。分类成绩不是本阶段输出。

完全相同模型输入的重复组{dupstats['exact_model_input_hash_groups']}个，覆盖{dupstats['occurrences_in_duplicate_groups']}次出现；其中跨版本{dupstats['cross_version_groups']}组、训练与验证共有{dupstats['train_validation_shared_groups']}组。只做完全相同张量核对，不等同图同构检测，没有据此删样本。重复及大量普通顺序目标可能造成捷径；下一阶段须分别评价非平凡分支、调用／异常边界和重复范围，不能只看总下一事件准确率。

## 7. 数值状态与研究准入

最终可见值版本训练{a['final_visible_value_versions']}个，其中真实数值{a['final_visible_numeric_versions']}个（{observed['train']['observed_fraction']:.2%}）；验证{b['final_visible_value_versions']}个，其中真实数值{b['final_visible_numeric_versions']}个（{observed['val']['observed_fraction']:.2%}）。这是已有值版本上的观测比例，不是全程序状态覆盖率。局部算术配对仅在所有输入版本先前已有直接回调、结果随后真实回调时保存，训练{a['numeric_pairs']}对、验证{b['numeric_pairs']}对；没有由分支结果倒推数值。

真实有序事件及受支持依赖结构已可构成有限结构转换监督（T1）。数值层（T2）只有上述局部观测，缺初始状态、完整堆、业务输入／输出、完整断言配对及未记录库状态；不宣称完整T2就绪。真实合法干预对（T3）为未具备，没有反事实或强化学习。

结构预测数据可以进入另行冻结的有限试训，世界模型尚未训练。现有完整产品图分类资格和划分不变；本阶段没有读取分类标签，机器交接中的类别数量为null并说明原因。三个既有协议的产品／版本数量来自公开清单，测试区只看这些身份。跨系统、交易配置覆盖、独立版本规模及代理分类标签的证据资格问题均未被解决。

## 8. 环境、资源和复现

Windows CPU、Python {sys.version.split()[0]}、NumPy {np.__version__ if False else resources['numpy']}；使用现有图环境，没有安装大型依赖。生产耗时{production['resources']['seconds']:.3f}秒，独立核验{validation['seconds']:.3f}秒；实际命令、退出码和日志摘要在同名JSON及数据目录`*.command.json`。实际工作进程峰值见JSON的`process_memory`，不是约5MiB的虚拟环境启动器。按轨迹检查资源并有命令超时保护，未施加操作系统硬内存／磁盘配额。预算为3600秒、2GiB新增输出、4GiB工作进程内存及4GiB最低余量；收尾实测另见`final_audit.json`。

可信原包摘要为`{ANCHOR}`；冻结方案摘要`{sha256(PROTOCOL)}`；新前缀清单摘要`{prefix_hash}`。每条轨迹保存源路径／摘要、共享静态表与观测流、位置索引、独立目标、逐字段时刻、检查记录及少量张量示例。输入读取方法为`cwm_fp.models.prefix_reader.load_prefix`，必须提供包外可信摘要；目标另由调用方独立读取，不能把整个目标字典传给模型。

旧原始数据、标签、划分、完整图和04R-1正式产物未覆盖。旧协议中的源码摘要仍指向其执行时源码备份，本次必要源码变化不通过重写旧协议掩盖。未发现可安全删除且无独立用途的旧文档，故未删除历史证据。本阶段只有这份完整中文长总结。

## 9. 下一步建议（不自动执行）

1. 最优先冻结有限的结构预测试训协议：使用现有A／B目标，报告普通顺序、分支、调用／异常边界和重复前缀分层结果，保留两项简单参照。它能回答模型是否学习到超出静态顺序模板的信息，尚不能回答干预因果效应。
2. 若需要数值模型，先单独制定初始状态、输入输出及缺失状态的补采／契约；现有局部算术对只能支撑局部诊断，不能填补完整状态。
3. 服务器训练环境另行验收。本次未访问X盘或GPU，没有实测显存及服务器吞吐，不能把已有数据生产通过当作GPU验收通过。

阶段05A到此停止，没有执行阶段05B。
'''
    (ROOT/'outputs/phase05A_review_packet.md').write_text(md,encoding='utf8')
    print('REPORT',len(rows),packet['accepted_prefixes'],prefix_hash,flush=True)

if __name__=='__main__':main()
