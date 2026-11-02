"""Produce one Chinese handoff from completed, fixed-protocol measured evidence."""
from pathlib import Path
from collections import defaultdict,Counter
import json,sys,time,shutil,statistics
import numpy as np
from cwm_fp.data.common import ROOT,sha256,write_json,write_csv
from cwm_fp.models.transition_data import load,verify_source,read_targets,hierarchy_weights
from cwm_fp.models.transition_train import verify_protocol

out=Path(sys.argv[1]);source=Path(sys.argv[2]);start=time.monotonic()
p,psha=verify_protocol(out,'cpu');manifest,counts,source_artifacts=verify_source(source)
environment=load(out/'environment.json');ledger=load(out/'resource_ledger.json');training=load(out/'training_progress.json')
evaluation=load(out/'evaluation.json');isolation=load(out/'model_isolation_checks.json');regression=load(out/'regression_tests.json')
diag=load(ROOT/'outputs/phase05B_shortcut_diagnostics.json');metrics=load(out/'metrics.json');preflight=load(out/'preflight.json')
postfit=load(out/'postfit_measurements.json')
indexes={part:load(out/'cache'/f'index_{part}.json') for part in ('train','val')};targets={k:read_targets(out,v) for k,v in indexes.items()}
assert len(training['completed'])==9 and all(r['epochs']==20 and r['updates']==1420 for r in training['completed'])
assert ledger['updates']==12783 and regression['passed']==115 and isolation['passed']
# Recheck every frozen05A artifact and public split/manifest, not only self-report.
for rel,ref in source_artifacts.items():
    if sha256(source/rel)!=ref['sha256']:raise ValueError('Historical05A artifact changed: '+rel)
oldplan=load(ROOT/'outputs/phase05A_protocol.json')
for rel,digest in oldplan['public_sha256'].items():
    if sha256(ROOT/rel)!=digest:raise ValueError('Public manifest/split changed')
for rel,digest in oldplan['source_sha256'].items():
    if sha256(ROOT/rel)!=digest:raise ValueError('Historical producer changed')
names={'seq_gru':'顺序模型','no_dep':'移除依赖的图模型','rgcn':'完整依赖图模型','Prior':'训练先验','Category-Markov':'语义类别转移表',
 'Exact-rgcn':'完整输入精确查表','Exact-no_dep':'消融输入精确查表','Exact-seq_gru':'顺序输入精确查表'}
def select(model,task,subset='all',stratum='all',aggregation='version_macro',part='val'):
    return [r for r in metrics if r['model']==model and r['task']==task and r['subset']==subset and r['stratum']==stratum and r['aggregation']==aggregation and r['partition']==part]
def avg(model,task,field,subset='all',stratum='all',aggregation='version_macro'):
    vals=[r[field] for r in select(model,task,subset,stratum,aggregation) if r.get(field) is not None]
    return sum(vals)/len(vals) if vals else None
def fmt(value):return '缺失' if value is None else f'{value:.6f}'
def selected_numbers(r):
    c=next(c for c in r['candidates'] if c['epoch']==r['selected_epoch'])
    return [fmt(c['A_abstain_version_macro']['NLL']),fmt(c['B_version_macro']['NLL']),c['A_abstain_version_macro']['version_count'],c['B_version_macro']['version_count']]
def table(lines,headers):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in lines)+'\n'

comparisons={}
for subset in ('all','common_unseen'):
    comparisons[subset]={k:{'A_abstain_NLL':avg(k,'A','NLL',subset,'A_abstain'),'B_NLL':avg(k,'B','NLL',subset),
        'B_accuracy':avg(k,'B','accuracy',subset),'A_NLL':avg(k,'A','NLL',subset)} for k in names if select(k,'A',subset)}
def beats(subset,task,controls):
    key='A_abstain_NLL' if task=='A' else 'B_NLL';full=comparisons[subset]['rgcn'][key]
    return full is not None and all(comparisons[subset][c][key] is not None and full<comparisons[subset][c][key]-1e-6 for c in controls)
nontrivial=beats('all','A',['Prior','Category-Markov','no_dep']) or beats('all','B',['Prior','no_dep'])
unseen=beats('common_unseen','A',['no_dep','seq_gru']) or beats('common_unseen','B',['no_dep','seq_gru'])
statuses={'STRUCTURAL_PREDICTOR_IMPLEMENTED':True,'STRUCTURAL_PREDICTOR_TRAINED':True,'INPUT_FUTURE_ISOLATION_RECHECKED':True,
 'NONTRIVIAL_TRANSITION_GAIN_OBSERVED':nontrivial,'UNSEEN_PREFIX_GAIN_OBSERVED':unseen,'DYNAMIC_DEPENDENCY_CONTRIBUTION_SUPPORTED':False,
 'LINUX_GPU_VERIFIED':False,'DOWNSTREAM_FP_TP_GAIN_TESTED':False,'CAUSAL_INTERVENTION_VALIDATED':False}
status_reasons={'NONTRIVIAL_TRANSITION_GAIN_OBSERVED':'descriptive three-seed mean version NLL: Aabstain better than Prior,Markov,NoDep OR B better than Prior,NoDep; not statistical significance',
 'UNSEEN_PREFIX_GAIN_OBSERVED':'descriptive full-vs-NoDep-and-GRU lower mean version NLL on same common-unseen Aabstain OR B subset; both targets and supports separately reported',
 'DYNAMIC_DEPENDENCY_CONTRIBUTION_SUPPORTED':'single joint control/data/value-edge ablation cannot separate dynamic-data contribution from static-control contribution; inspect actual joint comparison, do not claim independent causal benefit'}
sizes={k:v['stats'] for k,v in indexes.items()}
snapshot_totals={}
for part,index in indexes.items():
    repeats=Counter(r['input_key'] for r in index['rows']);nodes=edges=0
    for key,n in repeats.items():
        with np.load(out/index['files'][key]['path'],allow_pickle=False) as z:nodes+=len(z['x'])*n;edges+=len(z['edge_type'])*n
    snapshot_totals[part]={'observed_product_test_events':len(index['rows']),'nodes_summed_over_prefix_snapshots':nodes,
        'edges_summed_over_prefix_snapshots':edges,'event_instances_summed_over_prefix_snapshots':sum(r['event_index']+1 for r in index['rows']),
        'interpretation':'one event per anchor; snapshot node/edge sums repeat static and historical dynamic contents, not independent program elements'}
product_rows=[];graph_rows=[]
for part,index in indexes.items():
    groups=defaultdict(list);product_traces=defaultdict(list)
    for r,t in zip(index['rows'],targets[part]):groups[r['trace_index']].append((r,t))
    for trace,items in groups.items():
        r=items[0][0];identity=r['identity'];product_traces[tuple(identity[:3])].append(trace)
        graph_rows.append(dict(zip(('system_id','bug_version_id','product_id','test_id'),identity))|{'partition':part,'trace_index':trace,
            'prefixes':len(items),'A_valid':sum(t['A']>=0 for _,t in items),'B_valid':sum(t['B']>=0 for _,t in items),
            'source_graph_admission':'inherited05A verified lineage; not re-executed','prefix_identity_and_timing':'PASS',
            'model_isolation':'PASS','rejection_reason':'','evidence':str(out/'model_isolation_checks.json'),'source_manifest':str(source/'prefix_manifest.json')})
    for identity,traces in product_traces.items():
        sample=[r for r in graph_rows if tuple(r[k] for k in ('system_id','bug_version_id','product_id'))==identity]
        product_rows.append(dict(zip(('system_id','bug_version_id','product_id'),identity))|{'partition':part,'source_original_suite_traces':len(traces),
            'all_listed_traces_retained':True,'prefixes':sum(r['prefixes'] for r in sample),'A_valid':sum(r['A_valid'] for r in sample),'B_valid':sum(r['B_valid'] for r in sample),
            'classification_admission':'historical unchanged; no new product admission in05B','classification_label_access':'PROHIBITED','transition_training':part=='train','transition_evaluation':part=='val',
            'evidence':str(out/'cache'/f'index_{part}.json'),'rejection_reason':''})
write_csv(ROOT/'outputs/phase05B_product_status.csv',product_rows,product_rows[0].keys());write_csv(ROOT/'outputs/phase05B_graph_status.csv',graph_rows,graph_rows[0].keys())
peak=max(c['memory']['peak_working_set_bytes'] or 0 for c in ledger['commands'])
resources={'environment':environment,'budget':p['budgets'],'worker_commands':ledger['commands'],'worker_seconds':ledger['seconds'],
 'regression_seconds':regression['seconds'],'measured_worker_plus_regression_seconds':ledger['seconds']+regression['seconds'],
 'timing_scope':'inside actual workers; includes cache/read/train/predict/eval/isolation, not human/code-authoring time or every interpreter startup. Failed startup and freeze/test command logs separately retained.',
 'peak_worker_RSS_bytes':peak,'peak_GPU_memory_bytes':None,'GPU_missing_reason':'CPU experiment; no GPU training or memory measurement',
 'full_task_wall_seconds':None,'full_task_wall_seconds_missing_reason':'No single timer spanning coding and all tool sessions; measured worker-command totals provided instead',
 'runtime_directory_bytes':sum(f.stat().st_size for f in out.rglob('*') if f.is_file()),'disk_free_bytes':shutil.disk_usage(out).free,
 'optimizer_updates_total':ledger['updates'],'formal_updates':12780,'engineering_updates':3,'timeouts':0,'OOM':0,'truncated_examples':0,'cache_invalidations':0,
 'timer_caveat':'Windows monotonic timer coarse; a preflight backward interval rounded to0, not evidence of zero computation; aggregate formal timings reported',
 'pre_freeze_failures':[{'command':'transition_run preflight (first attempt)','exit_code':1,'log':'outputs/phase05B_preflight.log','reason':'Windows default GBK decoding resource ledger; fixed explicit UTF8','seconds':None},
 {'command':'python -m unittest discover -s tests -p test_transition.py -v (first prefreeze expanded suite)','exit_code':1,'log':'outputs/phase05B_unit_prefreeze.log','reason':'evaluation module syntax; fixed before freeze','seconds':None},
 {'command':'python scripts/phase05B_freeze.py <same source> <same output> (first attempt)','exit_code':1,'log':'outputs/phase05B_freeze.log','reason':'manual independent audit hash one character mistyped; checked original actual receipt then corrected before freeze','seconds':None}],
 'failed_startup_seconds_missing_reason':'not independently timed inside worker; cannot report as zero; failures occurred before fit and are retained as logs',
 'budget_enforcement':'worker self checks between trajectory/micro batches; no OS hard quota; all measured totals within limits'}
issues=[{'name':'单系统小队列与表示歧义','confirmed':True,'impact':'当前9训练/4验证版本；22列粗语义和相同输入多个目标限制下一步预测；不能宣称跨系统泛化或程序固有噪声。','evidence':'outputs/phase05B_shortcut_diagnostics.json'},
 {'name':'联合消融不能单独归因动态数据依赖','confirmed':True,'impact':'同时移除静态控制、动态数据、产生和使用边；完整图比较只支持或否定依赖关系族的整体收益。','evidence':'outputs/phase05B_protocol.json'},
 {'name':'状态观测与动作信息不足','confirmed':True,'impact':'本轮不输入真实标量；无完整局部/堆状态、输入输出断言数值或合法干预执行对。一步类别预测不能等同完整状态转换。','evidence':'outputs/phase05A_review_packet.md'},
 {'name':'静态与动态图未连接','confirmed':True,'impact':'05A静态控制子图与动态实例分离；静态结构仅通过预定汇聚进入查询。没有新增动态控制者映射。','evidence':'src/cwm_fp/models/transition_model.py'},
 {'name':'新颖性判定为精确张量层面','confirmed':True,'impact':'图排列等价、未参与查询的孤立值节点等可能令不同签名产生相同函数表示；共同未见不等于语义新颖。','evidence':'src/cwm_fp/models/transition_model.py'},
 {'name':'服务器与下游迁移未验证','confirmed':True,'impact':'Linux GPU尚未运行；不能把结构预测收益等同FP/TP识别改善。','evidence':'scripts/phase05B_linux_acceptance.py'}]
recommendation={'task':'先修正局部状态转换的任务与可观测表示，在冻结边界内设计真实局部数值观测的最小对照',
 'reason':'完整图仅相对移除依赖的图模型有局部损失改善，顺序模型仍更强；共同未见分支仅17个且完整图不及训练先验。当前粗粒度结构输入存在相同输入多目标，分支依赖操作数；先明确可预测状态和缺失信息，再决定是否开展下游迁移。不得以更大模型掩盖任务歧义。',
 'scope':'仅为下一阶段建议，本阶段不执行、不扩采、不训练数值模型。','GPU':'可准备独立Linux验收；尚不能声称服务器或GPU正式实验已通过。'}
packet={'stage_id':'05B','stage_status':'COMPLETED_BOUNDED_PILOT','research_readiness':'PARTIAL','actual_scope':counts,'protocol_sha256':psha,
 'statuses':statuses,'status_reasons':status_reasons,'before_metrics':{'trained_structural_transition_models':0,'source_prefixes':sum(v['prefixes'] for v in counts.values())},
 'after_metrics':{'trained_runs':9,'completed_epochs':180,'formal_updates':12780,'prefixes_retained':sum(v['prefixes'] for v in counts.values()),'new_complete_products':0},
 'source_stats':sizes,'snapshot_totals':snapshot_totals,'comparisons':comparisons,'all_actual_metrics':metrics,'selection':evaluation['selection'],'parameters':p['parameter_reports'],
 'seeds':p['seeds'],'actual_active_parameter_measurements':postfit['active_parameters'],'loss_and_weights':{k:p[k] for k in ('loss','weights','weight_sums','trajectory_batch','prefix_microbatch')},
 'weight_hierarchy_actual_support':load(out/'weight_hierarchy_summary.json'),'actual_weights_path':str(out/'train_supervision_weights.npz'),
 'duplicate_diagnostics_path':'outputs/phase05B_shortcut_diagnostics.json','duplicate_summary':{k:{s:v for s,v in q.items() if s!='tasks'} for k,q in diag['views'].items()},
 'admitted_products':{'new':0,'historical_full_classification_cohort':28,'in_scope_transition_products':22},'recovered_graphs':{'new':0,'not_applicable_reason':'No graph semantic repair or new collection in05B'},
 'rejection_reasons':{'transition_prefixes_rejected':0,'invalid_A_end_positions':368,'B_masked_nonconditional_or_unproven':sum(t['B']<0 for ts in targets.values() for t in ts)},
 'split_readiness':{'grouped_bank_pilot_v1':{'train':counts['train'],'val':counts['val'],'test':{'traces':95,'expanded':False,'source':'known05A scope; not reread'}},
 'leave_BankAccountTP_out':{'status':'NOT_EVALUATED','reason':'outside authorized05B scope; no other protocol data read'},'leave_Email_out':{'status':'NOT_EVALUATED','reason':'outside authorized05B scope'},
 'FP_TP_class_counts':None,'FP_TP_missing_reason':'all product targets prohibited in05B'},
 'world_model_readiness':{'T1':'PARTIAL: measured observational one-step structure predictor in certified scope','T2':'BLOCKED for full numerical state; historical partial observations retained but not used here',
 'T3':'BLOCKED: no actual legal intervention pairs','autoregressive_rollout':False,'future_isolation':isolation['passed']},
 'tests_executed':{'regression':regression,'preflight':preflight,'model_isolation':{k:v for k,v in isolation.items() if k!='checks'},'new_Java_tests':0},
 'resources':resources,'target_access':{'product_FP_TP':0,'test_transitions':0,'train_transition_targets':True,'val_transition_targets':'independent evaluator only during fitted-model selection; cache preparation previously separates and hashes targets',
 'fit_process_val_target_reads':0,'source_target_read_logs':[str(out/(s+'_target_access.json')) for s in ('cache_train','cache_val','train','evaluate','isolation')]},
 'changed_files':[str(x.relative_to(ROOT)).replace('\\','/') for x in sorted((ROOT/'src/cwm_fp/models').glob('transition_*.py'))]+['tests/test_transition.py','scripts/phase05B_freeze.py','scripts/phase05B_regression.py','scripts/phase05B_report.py','scripts/phase05B_supplement.py','scripts/phase05B_linux_acceptance.py','scripts/phase05B_seal.py','DATA_CONTRACTS.md','STAGES.md','outputs/phase_status.md'],
 'artifact_paths':{'runtime':str(out),'review':'outputs/phase05B_review_packet.md','protocol':'outputs/phase05B_protocol.json','metrics':'outputs/phase05B_metrics.csv',
 'curves':'outputs/phase05B_learning_curves.csv','shortcuts':'outputs/phase05B_shortcut_diagnostics.json','products':'outputs/phase05B_product_status.csv','graphs':'outputs/phase05B_graph_status.csv',
 'branch_group_metrics':str(out/'branch_group_metrics.csv'),'model_checks':str(out/'model_isolation_checks.json')},
 'unresolved_issues':issues,'recommended_next_stage':recommendation,'historical_evidence_checked':len(source_artifacts),'cleanup':'No historical/independent evidence deleted; no redundant long summary created.'}
write_json(ROOT/'outputs/phase05B_review_packet.json',packet)
lines=['# 阶段05B结果交接：依赖结构引导的执行转换预测试训','',
 '本阶段已实际完成三个模型、三个固定种子、每组20轮的观测性一步转换预测试训。全部在Windows CPU执行；没有读取产品FP/TP目标、展开测试区轨迹、重跑Java、执行服务器训练或开展干预。工程实施完成，研究结论仍属有限先导证据，不能称为完整因果世界模型。','',
 '核心研究结果：完整图在静态参照弃权位置上较移除依赖的图模型有改善，但顺序模型更好；共同未见分支上没有观察到完整图的收益。因此尚未证明依赖结构带来了超越执行顺序和重复前缀记忆的稳定预测收益。','',
 '## 1. 范围、来源及执行完整性','',
 table([[part,c['products'],c['versions'],c['traces'],c['prefixes'],sizes[part]['A_valid'],sizes[part]['B_valid']] for part,c in counts.items()],['分区','产品','独立版本','轨迹','前缀','下一事件A','分支B']),
 '沿用按缺陷版本分组的冻结协议（grouped_bank_pilot_v1）。所有原前缀、轨迹及分区保留。训练A分为静态参照覆盖10585、弃权2729；验证覆盖3054、弃权782。无后续观测的368个A位置继续关闭；无效分支位置不当负例。历史28个完整分类准入产品维持原状，本次新增合格产品、修复测试图均为0。其他系统留出协议不在本轮授权范围，分类标签数量不统计。','',
 f'前缀清单、05A方案、原完整图清单经独立回执和项目输出交叉核对；收尾复核{len(source_artifacts)}个05A冻结文件以及公开清单、划分和旧生产代码，未改变。05B冻结方案摘要：`{psha}`。05A验证统计已公开，本轮不声称首次查看验证任务。','',
 '## 2. 实际实现与模型可见信息','',
 '新增代码复用原前缀观察器、解释器及关系图层，没有重写Java语义。输入只包括22列语义/布尔特征、当前可见边及已观测锚点。配置两列保持掩码；原始数值、类名、行号、版本/产品身份、目标字典、最终长度和最终测试结果不进入模型。数值仍保存在05A产物，没有删除或补零冒充观测。','',
 table([[{1:'静态控制依赖',2:'动态数据依赖',3:'值产生',4:'值使用',6:'时间邻接'}[k]+'（'+p['actual_relations'][str(k)]+'）',sizes['train']['relation_counts_over_prefix_snapshots'][str(k)],sizes['val']['relation_counts_over_prefix_snapshots'][str(k)]] for k in (1,2,3,4,6)],['实际关系','训练前缀内累计边次','验证前缀内累计边次']),
 table([[part,v['observed_product_test_events'],v['nodes_summed_over_prefix_snapshots'],v['edges_summed_over_prefix_snapshots']] for part,v in snapshot_totals.items()],['分区','实际观测事件/锚点数','前缀快照节点累计','前缀快照边累计']),
 '以上是各前缀快照的累计边次，包含历史关系在后续前缀中的重复可见，不能称为独立执行边。方向依次为静态控制者→受控位置、早期定义→后期使用、定义→值版本、值版本→使用、前一事件→后一事件。没有特征归属、输出观察或新造动态控制边。静态控制子图与动态实例仍分离，只通过既定上下文汇聚进入预测。','',
 table([[names[k],q['total'],q['structurally_active'],q['inactive_relation_coefficients']] for k,q in p['parameter_reports'].items()],['模型','总参数','关系掩码下活跃参数上限','未用关系系数']),
 '这些活跃上限只扣除消融/不存在关系的系数，不代表所有输入列都出现过或每步梯度均非零。另在第一个种子的已选固定参数上，对全部训练前缀分批反向传播、不更新参数，实测非零梯度参数元素的并集如下；这与整个正式训练历史中的活跃参数并集并不相同。','',
 table([[names[r['model']],r['seed'],r['epoch'],r['actual_nonzero_gradient_parameter_union']] for r in postfit['active_parameters']],['模型','固定种子','固定轮次','实测非零梯度元素并集']),
 '顺序模型：单向一层、隐藏32，按真实事件顺序读取当前前缀的22列，用最后有效事件表示。图模型：两层隐藏32，拼接锚点、可见动态事件均值、静态节点均值及三个存在标志，再输出9类A和单个B分数。移除依赖的图模型同时屏蔽控制、数据、产生、使用四类边，保留节点、时间邻接与自身变换。两图模型同种子初始化完全对齐；每关系入边均值公式相同。顺序模型没有完整静态图，故它不是严格等信息的拓扑消融。','',
 '同一微批中的完全相同、互不连通静态子图可以共享计算，动态部分始终独立；已通过真实及人工的输出/梯度等价检查。缓存按实际输入去重存储，训练样本和目标一条未删。图签名是精确张量相等，不是图同构或模型函数等价证明；孤立值节点、排列等因素仍可能使签名的新颖性高于模型有效表示的新颖性。','',
 '代码职责：`transition_model.py`定义白名单、三模型及损失；`transition_data.py`负责来源核验、缓存和四级权重；`transition_train.py`负责真实梯度检查与固定训练；`transition_evaluate.py`负责参照、选择、分层和歧义统计；`transition_checks.py`负责模型后缀/批次隔离；`transition_budget.py`和`transition_run.py`负责有界执行。对应回归为`tests/test_transition.py`。未实现新的数值世界模型、完整状态模拟、多步自主展开、FP/TP联合学习或干预。','',
 '## 3. 训练、候选选择与测试证据','',
 '三个监督部分各自按位置→轨迹→产品→版本等权归一化，再固定除以3。每批4条轨迹，64个前缀为计算微批并累计梯度；每个轨迹批次只更新一次，末批保留3条轨迹。每轮283条轨迹/13597个前缀全部遍历，71次更新。Adam学习率0.001、32位浮点，无额外类别权重或标签平滑。正式12780次更新，加真实工程验收3次，共12783次。','',
 '固定种子20261101、20261102、20261103；每组从头初始化。只在第5、10、20轮产生候选预测，由独立评价步骤按验证版本宏平均的弃权A和B归一化负对数损失各占一半选择；改善不足1e-6保留更早轮次。全部种子报告，不选最优种子。','',
 table([[names[r['model']],r['seed'],r['selected_epoch'],fmt(r['score']),*selected_numbers(r)] for r in evaluation['selection']],['模型','种子','选择轮次','固定选择分数','弃权A损失','B损失','A版本数','B版本数']),
 f'115项人工/相关回归全部通过。真实训练区预检的编码器和两个预测头都有非零梯度与参数更新，保存恢复输出完全一致。共享静态计算最大梯度绝对差不超过{max(r["shared_static_equivalence"]["max_gradient_absolute_difference"] for r in preflight["records"] if r["shared_static_equivalence"]):.3g}。缓存全部17518个前缀与原05A摘要一致，另有1104次公开读取器首/中/末核验。','',
 f'模型层检查：训练{isolation["counts"]["train"]["checked_prefixes"]}个位置、验证{isolation["counts"]["val"]["checked_prefixes"]}个位置，覆盖每条轨迹首/中/末与调用、条件分支、返回及认证异常边界。固定各模型第一个种子的已选参数，删除/替换未来后缀后当前原始分数不变；反转样本顺序及改变微批边界也在2e-6容差内一致。模型后缀最大分数差：{isolation["max_suffix_logit_error"]}。这些是当前有限真实队列上的检查，不是任意JVM语义的证明。','',
 '冻结前遇到三类工程错误：Windows默认编码读取资源账本失败、评价模块括号语法错误、手写独立摘要一字符错误。均保留失败日志，核对证据后在正式冻结/拟合前修复；没有根据验证分数改模型或追加训练。','',
 '## 4. 实际预测结果','',
 '下表为已选检查点的验证结果：神经模型取三个种子均值，损失越低越好。主表使用版本宏平均，先算每个版本内的事件均值，再对有目标的版本等权；事件微平均、固定九类宏F1、混淆矩阵和各个种子完整结果见指标CSV，不能用多个随机种子代替独立软件版本。','',
 'B混淆矩阵以0=顺序执行、1=跳转；其中真阳性等统计指分支跳转，不是产品FP/TP标签。阈值固定0.5，没有用验证结果调阈值。两种结果都出现的分支位置只作分区内事后统计，未作为输入或挑选样本的依据。','',
 table([[names[k],fmt(avg(k,'A','accuracy')),fmt(avg(k,'A','NLL')),fmt(avg(k,'A','NLL',stratum='A_abstain')),fmt(avg(k,'B','NLL')),fmt(avg(k,'B','Brier')),fmt(avg(k,'B','balanced_accuracy'))] for k in names],['方法','A准确率','A损失','弃权A损失','B损失','B均方概率误差','B平衡准确率']),
 f'明确比较：弃权A的版本宏平均损失，完整图{avg("rgcn","A","NLL",stratum="A_abstain"):.6f}，移除依赖图{avg("no_dep","A","NLL",stratum="A_abstain"):.6f}，顺序模型{avg("seq_gru","A","NLL",stratum="A_abstain"):.6f}。完整图优于消融图和非神经参照，但没有优于顺序模型。B损失完整图{avg("rgcn","B","NLL"):.6f}，训练先验{avg("Prior","B","NLL"):.6f}，消融图{avg("no_dep","B","NLL"):.6f}，顺序模型{avg("seq_gru","B","NLL"):.6f}；其相对先验/消融的改善很小，也弱于顺序模型。负对数损失越低，表示模型赋予真实结果的概率越高。','',
 '普通静态唯一后继参照只在其覆盖范围输出预测，额外使用准确静态后继查表信息，不能视作与神经模型等输入。训练覆盖10585/13314，正确10563；验证覆盖3054/3836，正确3046，覆盖内准确率99.7380%，正确数/全部A为79.4056%。30个错误保留，均是05A认证的实际空字段异常位置。神经指标没有删除这些位置。','',
 '关于依赖贡献：完整图与移除依赖图的实际差值应按弃权A、B及共同未见分别看，不能用普通顺序占多数的总体准确率取代。当前联合消融同时改变静态控制和动态数据关系，无法单独证明动态数据依赖的独立贡献，更不能称为因果验证。','',
 table([[k,v] for k,v in statuses.items()],['分开的完成/研究状态','实际判定']),
 '上述“观察到增益”只指预定比较下的三种子平均损失降低，不代表统计显著、所有种子均改善或跨系统成立。具体判定规则与全部种子指标在机器交接中；动态依赖独立贡献保持未确认。','',
 '## 5. 重复前缀、共同未见与歧义','',
 table([[names[k],q['train_unique_groups'],q['cross_partition_groups'],q['val_seen_prefixes'],q['val_unseen_prefixes'],q['tasks']['A']['conflicting_groups'],q['tasks']['A']['affected_targets'],q['tasks']['B']['conflicting_groups'],q['tasks']['B']['affected_targets']] for k,q in diag['views'].items()],['输入视图','训练唯一组','跨训练验证共有组','验证已见前缀','验证未见前缀','A冲突组','A受影响数','B冲突组','B受影响数']),
 f'三个模型共同未见的同一验证子集有{diag["common_unseen_prefixes"]}个前缀，A有效{diag["common_unseen_A_valid"]}、B有效{diag["common_unseen_B_valid"]}。不能拿三个不同的“各自未见”子集直接排名。','',
 table([[names[k],fmt(avg(k,'A','NLL','common_unseen','A_abstain')),fmt(avg(k,'B','NLL','common_unseen')),fmt(avg(k,'B','accuracy','common_unseen'))] for k in ('seq_gru','no_dep','rgcn')],['模型','共同未见弃权A损失','共同未见B损失','共同未见B准确率']),
 f'共同未见子集的弃权A只有80个目标、3个版本；完整图仍优于消融图但弱于顺序模型，说明并非全部收益都来自精确查表，却不能说明依赖结构优于顺序表示。共同未见B仅17个目标、3个版本，完整图损失{avg("rgcn","B","NLL","common_unseen"):.6f}，高于消融图{avg("no_dep","B","NLL","common_unseen"):.6f}和训练先验{avg("Prior","B","NLL","common_unseen"):.6f}，顺序模型为{avg("seq_gru","B","NLL","common_unseen"):.6f}。该子集很小，不能作稳定泛化或显著性结论。','',
 '精确查表仅由训练目标频次构建，未命中回退训练语义类别表；命中/未命中的覆盖、指标和经验冲突分布见诊断JSON及指标CSV。签名不再包含原始数值，因此不可照搬05A的116组共有输入。相同输入多目标保留，不删除、不改标签，也不添加唯一身份追求满分。经验多数上界和条件熵仅刻画本队列的表示歧义，不是程序固有随机性或总体误差下界。','',
 '## 6. 资源、复现及证据位置','',
 f'实际环境：{environment["platform"]}，Python3.11.4，PyTorch{environment["torch"]}，NumPy{environment["numpy"]}，单线程CPU。接口检测到CUDA可用，但本轮明确选择CPU，没有使用GPU训练，显存峰值未知；没有访问X盘或远端会话。真实工作进程最大驻留内存{peak/2**20:.2f}MiB。','',
 f'内部累计缓存/预检/读取/训练/预测/评价/隔离耗时{ledger["seconds"]:.3f}秒，另115项回归{regression["seconds"]:.3f}秒；不包括人工编码时间和所有解释器启动时间。训练主过程{training["seconds"]:.3f}秒，缓存训练{sizes["train"]["seconds"]:.3f}、验证{sizes["val"]["seconds"]:.3f}秒。输出目录当前{resources["runtime_directory_bytes"]/2**20:.2f}MiB，卷余量{resources["disk_free_bytes"]/2**30:.2f}GiB，最终封存大小另见收尾审计。无预算退出、OOM、数据截断或缓存失效。资源检查为进程内周期检查，并非操作系统硬配额。','',
 '主要命令均使用现有图环境解释器，设置`PYTHONPATH=src`后运行`python -m cwm_fp.models.transition_run`，依次执行`cache_train`、`preflight`、冻结脚本、`cache_val`、`train`、`evaluate`、`isolation`；参数均是下列实际源目录和独立输出目录。成功步骤退出码均0；完整参数、时间和退出码见机器交接的`resources.worker_commands`，冻结前失败日志仍在`outputs/phase05B_*.log`。','',
 f'- 输入：`{source}`。\n- 独立运行产物：`{out}`。\n- 逐位置预测及候选检查点：`runs/<模型>_<种子>/`，按缓存索引逐行对应。\n- 单产品/轨迹/版本B结果：`branch_group_metrics.csv`。\n- 模型隔离证据：`model_isolation_checks.json`。\n- 核心报告：`outputs/phase05B_review_packet.json`、`phase05B_metrics.csv`、`phase05B_learning_curves.csv`、`phase05B_shortcut_diagnostics.json`。',
 '', '参数化Linux入口为`PYTHONPATH=src python scripts/phase05B_linux_acceptance.py --source <05A目录> --cache-root <05B目录> --out <新的验收目录> --device cuda:0`。其状态为NOT_VERIFIED，尚未实际运行；只做独立环境/前后向一步验收，不能沿用CPU冻结方案直接宣称GPU正式比较。','',
 '## 7. 当前阻碍与唯一首要建议','',
 '\n'.join(f'- **{i["name"]}**：{i["impact"]}' for i in issues),
 '', '当前可用的是有限认证语义范围内的真实有序事件与依赖结构，以及读取真实历史的一步预测。没有自主多步展开，也没有完整下一状态恢复。部分历史真实数值存在但本轮未输入；完整数值世界模型所需状态、输入输出和干预证据仍不足。预测误差不能直接当作缺陷证据。','',
 '**唯一首要建议：'+recommendation['task']+'。** '+recommendation['reason']+' 本阶段只提出建议，不开始下一阶段。','',
 '文档整理：只维护这一份完整中文长报告；未发现可删除而不损失独立证据的历史文档，因此没有删除旧报告、冻结数据或失败日志。`outputs/phase_status.md`作为简短入口。阶段05B到此停止。','']
(ROOT/'outputs/phase05B_review_packet.md').write_text('\n'.join(lines),encoding='utf8')
write_json(out/'report_generation.json',{'seconds':time.monotonic()-start,'historical_files_verified':len(source_artifacts),'protocol_unchanged':True})
print('REPORT',statuses,'runtime',resources['runtime_directory_bytes'],'seconds',resources['worker_seconds'])
