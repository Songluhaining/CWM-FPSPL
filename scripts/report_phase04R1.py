"""One Chinese handoff from frozen results; no target reads or model execution."""
from __future__ import annotations
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cwm_fp.baselines.boundaries import install_io_guard
from cwm_fp.data.common import write_json,write_csv,sha256

OUT=Path(r'D:/dataset/whn/cwm_fp_generated/data/processed/graph_classifier/phase04R1_cpu_v1')
NAMES={'Prior':'训练比例常数（Prior）','Stats-LR':'统计逻辑回归（Stats-LR）','nodeonly':'无边节点模型（NodeOnly）',
       'untyped':'合并关系图模型（Untyped-GNN）','rgcn':'关系图模型（R-GCN）'}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def table(rows):return '\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in rows)
def fmt(x):return '不适用' if x is None else f'{x:.4f}'


def main(out):
    install_io_guard('04R1_report_no_target_files')
    protocol=read(ROOT/'outputs/phase04R1_protocol.json');receipt=read(out/'protocol.receipt.json')
    assert sha256(ROOT/'outputs/phase04R1_protocol.json')==receipt['sha256']==sha256(out/'protocol.json')
    for n,h in protocol['source_sha256'].items():assert sha256(ROOT/n)==h and sha256(out/'executed_source'/n)==h
    result=read(out/'training_result.json');val=read(out/'validation_results.json');diag=read(out/'prediction_diagnostics.json')
    prior=read(out/'prior.json');stats=read(out/'stats_lr_audit.json');inputs=read(out/'actual_model_inputs.json')
    curves=list(csv.DictReader((out/'learning_curves.csv').open(encoding='utf-8')))
    steps=list(csv.DictReader((out/'training_trace.csv').open(encoding='utf-8')))
    assert result['status']=='COMPLETE' and result['actual_updates']==len(steps)==16700
    assert len(curves)==1800 and all(r['products']=='17' and r['tests']=='283' and r['lossless']=='True' for r in curves)
    assert Counter(r['phase'] for r in steps)=={'micro500':500,**{f'{k}_{s}':1800 for k in protocol['neural_methods'] for s in protocol['seeds']}}
    for kind in protocol['neural_methods']:
        for seed in protocol['seeds']:
            assert [int(r['epoch']) for r in curves if r['method']==kind and int(r['seed'])==seed]==list(range(1,201))
    assert len(val['candidates'])==38 and len(val['selected'])==9
    accesses={name:read(out/(name+'_target_access.json')) for name in ('trainer','stats','validation')}
    for name,d in accesses.items():
        expected=protocol['paths']['val_labels' if name=='validation' else 'train_labels']
        assert set(d['reads'])=={expected} and not d['blocked']
    assert (out/'validation.claim').stat().st_mtime_ns> (out/'training_result.json').stat().st_mtime_ns
    for e in val['candidates']:
        for part in ('train','val'):assert sha256(out/e[part+'_file'])==e[part+'_sha256']
        if e['checkpoint']:assert sha256(out/e['checkpoint']['path'])==e['checkpoint']['sha256']
    selected=[c for c in val['candidates'] if c['seed'] is None]+val['selected']
    selected_ids={(x['method'],x['seed'],x['epoch']) for x in selected}
    metrics_rows=[]
    for entry in val['candidates']:
        for partition,m in [('train',entry['train_metrics']),('val',entry['metrics'])]:
            prob=m['probability_summary'];cm=m['confusion_matrix']
            metrics_rows.append({'method':entry['method'],'seed':entry['seed'],'epoch':entry['epoch'],'partition':partition,
                'selected':(entry['method'],entry['seed'],entry['epoch']) in selected_ids,'products':m['n'],'versions':m['versions'],
                'FP_products':m['FP'],'TP_products':m['TP'],'FP_versions':m['class_versions']['FP'],'TP_versions':m['class_versions']['TP'],
                **{k:m[k] for k in ('log_loss','brier','roc_auc','average_precision','fp_precision','fp_recall','fp_f1')},
                'tn':cm[0][0],'false_positive':cm[0][1],'false_negative':cm[1][0],'true_positive':cm[1][1],
                'predicted_FP':m['predicted_counts']['FP'],'predicted_TP':m['predicted_counts']['TP'],
                **{'probability_'+k:v for k,v in prob.items()},'evidence':str(out/'validation_results.json')})
    write_csv(ROOT/'outputs/phase04R1_metrics.csv',metrics_rows,tuple(metrics_rows[0]))
    shutil.copy2(out/'learning_curves.csv',ROOT/'outputs/phase04R1_learning_curves.csv')
    groups={}
    for k in protocol['neural_methods']:
        rs=[x for x in selected if x['method']==k];groups[k]={}
        for part in ('train','val'):
            ms=[r['train_metrics' if part=='train' else 'metrics'] for r in rs]
            groups[k][part]={key:{'mean':statistics.mean(m[key] for m in ms),'seed_sample_std':statistics.stdev(m[key] for m in ms),
                'min':min(m[key] for m in ms),'max':max(m[key] for m in ms)} for key in ('log_loss','brier','fp_f1','roc_auc','average_precision')}
    diagnostics={'scope':'TRAIN representation diagnostics only; validation metrics are separate, no test predictions',
        'frozen_thresholds':protocol['diagnostics'],'engineering_vs_prior_loss_delta':diag['04T1_engineering']['metrics']['log_loss']-prior['train_metrics']['log_loss'],
        'runs':diag,'active_input_columns':inputs['feature_columns_observed'],
        'interpretation':{'optimization':'Same4 product model reaches0.002201 at500 steps; prior100-step run did not establish inability to fit',
            'constant_output':'04T1 probabilities have narrow but nonzero range; population std exceeds the predeclared1e-4 near-constant threshold',
            'pooling':'node-to-test variance contracts; test-to-product variance need not decrease; not proof of rare-fault-signal loss',
            'relation_gain':'Selected R-GCN has lower validation log loss/Brier than merged-relation model in all3 seeds; normalization also differs'}}
    write_json(ROOT/'outputs/phase04R1_prediction_diagnostics.json',diagnostics)
    commands={n:read(out/(n+'.command.json')) for n in ('comparison_tests','comparison_tests_fixed','legacy_regression','entry_help_after','freeze','training','validation')}
    assert all(c['exit_code']==0 for n,c in commands.items() if n!='comparison_tests')
    unit=read(out/'regression/unit_tests.json');assert unit['passed']==58
    assert 'Ran 14 tests' in (out/'comparison_tests_fixed.log').read_text() and '\nOK' in (out/'comparison_tests_fixed.log').read_text()
    packet_files=['outputs/phase04R1_protocol.json','outputs/phase04R1_review_packet.md','outputs/phase04R1_review_packet.json',
        'outputs/phase04R1_metrics.csv','outputs/phase04R1_learning_curves.csv','outputs/phase04R1_prediction_diagnostics.json']
    changes=['src/cwm_fp/models/graph_encoder.py','src/cwm_fp/models/graph_classifier.py','src/cwm_fp/models/train_graph.py',
        'src/cwm_fp/models/graph_evaluator.py','src/cwm_fp/models/graph_metrics.py','src/cwm_fp/models/package_comparison.py',
        'tests/test_package_comparison.py','scripts/report_phase04R1.py','scripts/run_phase04R1_linux.sh',
        'configs/phase04R1_graph_requirements.txt','configs/phase04R1_stats_requirements.txt','DATA_CONTRACTS.md','STAGES.md','outputs/phase_status.md']+packet_files
    resources={'training_process_seconds':commands['training']['seconds'],'model_window_seconds':result['model_window_seconds'],
        'conservative_model_and_checks_seconds':result['model_window_seconds']+sum(c['seconds'] for n,c in commands.items() if n!='training'),
        'top_level_recorded_command_seconds':sum(c['seconds'] for c in commands.values()),'validation_seconds':commands['validation']['seconds'],
        'neural_process_memory':result['memory'],'stats_worker_seconds_including_import':stats['seconds'],'stats_worker_peak_memory':None,
        'stats_memory_reason':'No independent peak sampler for the statistics worker','gpu_peak_bytes':None,'gpu_reason':'No GPU execution',
        'timeout':False,'OOM':False,'NaN_Inf':False,'silent_graph_or_product_truncation':False,
        'peak_scope':'Actual graph worker OS lifetime peak; outer Windows venv launcher RSS is not model memory',
        'run_directory_bytes_at_report':sum(f.stat().st_size for f in out.rglob('*') if f.is_file()),'disk_free_bytes_at_report':shutil.disk_usage(out).free}
    issue_list=[
        {'name':'验证规模与选择偏差','status':'confirmed','impact':'只有5产品/4版本；本轮用同一验证集选择轮次并报告其成绩，可能乐观；种子重复不是独立抽样'},
        {'name':'关系类型对照同时改变归一化','status':'confirmed','impact':'合并关系由分关系邻居均值求和变为所有入边均值；不能把差异全部归于关系语义'},
        {'name':'代理标签尚未独立核验','status':'confirmed','impact':'全部监督目标仍为既有代理，不新增缺陷真值认证'},
        {'name':'配置及系统覆盖缺口','status':'confirmed','impact':'没有完整合格Transaction配置/套件及Email产品；不能宣称跨系统泛化'},
        {'name':'配置约束未认证','status':'confirmed','impact':'config_enabled/config_known两列保持屏蔽，不代表配置真实关闭'},
        {'name':'真实时间前缀生产器缺失','status':'confirmed','impact':'完整事后图含未来执行标记和最终观测，不能作为早期预测输入；世界模型尚不能直接开训'},
        {'name':'数值状态与干预证据不足','status':'confirmed','impact':'部分标量、输入/输出/断言观测不完整；无合法干预对，T2/T3未就绪'},
        {'name':'服务器未验收','status':'not_verified','impact':'没有可用Linux会话；GPU显存/驱动/吞吐未知。当前比较已在同一Windows CPU完成，统计与神经库使用两套既有解释器'}]
    next_task={'priority':1,'task':'实现并认证真实时间前缀生产器，作为结构世界模型的数据前置阶段',
        'evidence':['固定4产品可充分拟合；当前表示并非完全无可学习信号','3种子关系模型的验证损失低于所列对照，但只有5验证产品',
                    '现有完整事后图仍不能提供无未来泄漏的状态/事件预测输入'],
        'scope':'先定义真实可见的下一执行事件/指令类别、可认证分支后继等局部结构目标；只对有真实观测的数值保留掩码，禁止未来边和最终测试结果输入',
        'limits':'不因此宣称正式泛化或因果效应；代理标签、Transaction/Email覆盖仍需后续独立处理。此处仅建议，不实施。'}
    packet={'stage_id':'04R1','stage_status':'COMPLETE_FIXED_TRAIN_VALIDATION_PILOT','actual_scope':'Five fixed methods;9 neural runs;500-step TRAIN micro diagnosis; no test evaluation',
        'protocol':protocol['protocol'],'protocol_file':'outputs/phase04R1_protocol.json','protocol_sha256':receipt['sha256'],
        'cohort_manifest_sha256':protocol['manifest_sha256'],'cohort_counts':protocol['cohort_counts'],'source_sha256':protocol['source_sha256'],
        'label_access':accesses,'authorized_validation_access_performed':True,'test_labels_read':False,'test_predictions_generated':False,
        'training_labels':prior['train_counts'],'validation_labels':val['class_counts'],'validation_target_sha256':val['target_sha256'],
        'models':protocol['models'],'statistics_columns':protocol['statistics_columns'],'stats_lr_audit':stats,'prior':prior,
        'optimizer_budget':protocol['budget'],'actual_adam_updates':16700,'actual_LBFGS_iterations':sum(stats['iterations']),
        'conservative_total_optimizer_iterations':16700+sum(stats['iterations']),'neural_epochs_total':1800,
        'selected_results':selected,'all_candidate_results':val['candidates'],'seed_aggregates':groups,
        'checkpoint_selection':protocol['candidate_selection'],'prediction_diagnostics':'outputs/phase04R1_prediction_diagnostics.json',
        'environment':result['environment'],'stats_environment':stats['environment'],'resources':resources,
        'server':{'status':'NOT_VERIFIED','linux_executed':False,'gpu_executed':False,'X_codes_retried':False,
                  'previous_observation':read(Path(protocol['paths']['package']).parent/'server_access.json'),
                  'linux_script':'scripts/run_phase04R1_linux.sh','script_executed':False},
        'tests':{'passed':72,'regression':unit,'new_tests':14,'new_test_log':str(out/'comparison_tests_fixed.log'),
            'failed_pre_freeze_attempt':{'reason':'unterminated sample_id string literal; import failed before any real fitting or validation read',
                'evidence':str(out/'comparison_tests.log'),'corrected_before_protocol_freeze':True},'commands':commands,
            'new_Java_processes':0,'real_data_graphs':{'train':283,'val':85,'test_integrity_only':95}},
        'before_metrics':{'qualified_products':28,'graphs':463,'real_graph_training_verified':True,'validation_comparison_executed':False},
        'after_metrics':{'qualified_products':28,'graphs':463,'real_graph_training_verified':True,'validation_comparison_executed':True},
        'new_qualified_products':0,'recovered_graphs':0,'split_readiness':{'current_grouped':'PILOT_COMPLETED; no final-test/generalization certification',
            'cross_system':'BLOCKED; no complete Email products in current package','test_target_classes':None,'test_target_reason':'sealed; not read'},
        'world_model_readiness':{'T1':'PARTIAL; supported ordered graph events, no real certified temporal prefixes',
            'T2':'BLOCKED; insufficient observed numerical state/input/output','T3':'BLOCKED; no real interventions','world_model_trained':False},
        'changed_files':changes,'artifact_paths':{name:str(ROOT/name) for name in packet_files},
        'controlled_artifacts':{'root':str(out),'predictions_index':str(out/'prediction_index.json'),'validation_results':str(out/'validation_results.json'),
            'joined_validation_predictions':str(out/'controlled_validation'),'checkpoints':str(out/'checkpoints'),'target_access_logs':
            [str(out/(n+'_target_access.json')) for n in accesses]},
        'unresolved_issues':issue_list,'recommended_next_task':next_task,
        'claims':{'learning_signal':'Supported on current train subset with limited pilot validation evidence',
            'edge_connection_increment':'Observed validation benefit over NodeOnly under this fixed architecture/budget',
            'relation_increment':'Observed validation loss/Brier improvement in3 seeds, with normalization confound and4-version uncertainty',
            'causal_effect_identified':False,'beats_CLAP_PULP':None,'baseline_comparison_reason':'Different cohorts and information boundaries; not compared'}}
    write_json(ROOT/'outputs/phase04R1_review_packet.json',packet)
    scores=[]
    for e in selected:
        m=e['metrics'];scores.append([NAMES[e['method']],e['seed'] or '单次',e['epoch'] or '—',fmt(e['train_metrics']['log_loss']),fmt(m['log_loss']),fmt(m['brier']),
            fmt(m['roc_auc']),fmt(m['average_precision']),fmt(m['fp_precision']),fmt(m['fp_recall']),fmt(m['fp_f1'])])
    micro=diag['micro'];eng=diag['04T1_engineering'];ep=eng['metrics']['probability_summary']
    epoch_choices=[[NAMES[k],','.join(str(e['epoch']) for e in selected if e['method']==k),
        fmt(groups[k]['val']['log_loss']['mean']),fmt(groups[k]['val']['log_loss']['seed_sample_std']),
        fmt(groups[k]['val']['log_loss']['min'])+'—'+fmt(groups[k]['val']['log_loss']['max'])] for k in protocol['neural_methods']]
    rep=eng['representation'];rep_rows=[[n,v['rows'],v['dimension'],f"{v['mean_feature_variance']:.7g}",f"{v['mean_row_norm']:.7g}"] for n,v in rep.items() if n!='product_distances']
    version_rows=[]
    for e in selected:
        if e['method'] not in ('Stats-LR','rgcn'):continue
        for item in e['metrics']['by_version']:
            version_rows.append([NAMES[e['method']],e['seed'] or '单次',item['identity'][1],item['products'],fmt(item['log_loss']),item['errors']])
    counts_rows=[[part,d['products'],d['versions'],d['graphs'],d['nodes'],d['edges'],d['events']] for part,d in protocol['cohort_counts'].items()]
    rowsources=[[name,{'src/cwm_fp/models/graph_encoder.py':'新增节点自身变换和只合并关系类型两种最小对照；旧R-GCN计算保持不变',
        'src/cwm_fp/models/graph_classifier.py':'允许新对照复用既有分类头及两级汇聚',
        'src/cwm_fp/models/train_graph.py':'增加独立package_compare模式，保留04T1禁止保留目标的模式',
        'src/cwm_fp/models/graph_evaluator.py':'增加独立package_validation评价入口，只允许本协议验证目标',
        'src/cwm_fp/models/graph_metrics.py':'复用二分类指标，补明确AP、概率/类别/版本诊断',
        'src/cwm_fp/models/package_comparison.py':'冻结、训练、统计模型和评价调度，复用数据包/训练步骤/目标接口'}[name]] for name in changes[:6]]
    command_text='\n\n'.join('```text\n'+' '.join('"'+a+'"' if ' ' in a else a for a in c['argv'])+
        f"\n# exit={c['exit_code']}; seconds={c['seconds']}\n```" for name,c in commands.items())
    report=f'''# CWM-FP阶段04R-1结果交接

2026-11-01。本阶段已完成冻结队列上的学习能力诊断及五种固定训练／验证对照。**当前结构和节点语义具有可学习信号；关系图模型在这5个验证产品上的损失优于所列对照，但这只是小样本先导结果，不是最终测试成绩、跨系统泛化或因果有效性证明。**

本次实际执行：原4训练产品500步诊断，三种神经模型各3种子×200轮，常数和统计逻辑回归各一次。神经参数更新16700次，另逻辑回归求解16次迭代，保守合计16716，低于20000上限。独立评价端在协议冻结和全部训练完成之后读取了验证标签；测试标签始终封存，没有测试预测。实际位置为Windows CPU，未执行Linux/GPU。

## 一、冻结方案、数据及权限

协议`grouped_bank_pilot_v1`保持阶段01版本分组归属。冻结文件为`outputs/phase04R1_protocol.json`，摘要`{receipt['sha256']}`；输入包清单摘要`{protocol['manifest_sha256']}`，再次与独立采集回执核对。协议在验证目标首次读取之前封存，包含所有列、模型、种子、预算、指标、选择规则、源码摘要和设备。执行源码存于本阶段`executed_source/`，报告生成时再次逐文件核验一致。

| 分区 | 产品 | 独立版本 | 测试图 | 节点 | 边 | 执行事件 |
|---|---:|---:|---:|---:|---:|---:|
{table(counts_rows)}

总量28产品/16版本/463图保持不变，没有新增合格产品、恢复拒绝图或重新运行Java。全部登记测试、节点、事件、边保留，未使用旧16测试截断。训练每轮17产品/283图完整参与；测试6产品/95图只作既有无标签完整性核验，没有进入模型、表示诊断或统计拟合。

训练区10代理FP、7代理TP，对应6和4个版本（共享版本导致合计独立版本为9）；FP=1、TP=0。验证区实际2代理FP、3代理TP，对应2和3个版本（合计独立版本4）。本次验证**不是单类**，预先规定的单类处理只通过人工单元用例验证。测试类别未知，不读取。所有标签仍是既有未独立核实的代理。

神经训练端和统计训练端各只打开当前`grouped/train.csv`两次；评价端只打开`grouped/val.csv`两次（接口读取及摘要），访问日志分别保留。原目标接口在文件级校验完整分区文件，随后仅保留本队列17或5个产品的目标；其他产品目标未用于拟合、选择或指标。没有读取其他系统留出协议目标，也没有读取缺陷位置。

验证预测仅在10、50、100、200轮生成并封存，全部固定训练结束后由独立进程统一评价38候选（2基准＋36神经候选）。验证标签用于约定轮次选择和诊断，没有进入梯度、标准化、类别比例或样本选择。程序级路径限制不等于操作系统绝对隔离，受控目录也不宣称已经建立额外系统权限。

## 二、实际实现与五种固定方法

| 源码 | 实际修改 |
|---|---|
{table(rowsources)}

另新增针对性测试`tests/test_package_comparison.py`、本报告生成脚本、Linux同入口脚本和两份依赖版本文件；更新数据契约、阶段表和当前状态。数据包读取、完整套件准入、原始标签定义、冻结划分及JVM语义未改动。没有创建第二套图模型系统，04T1工程验收模式及其训练目标限制保持不变。

| 方法 | 输入及处理 | 实际可训练参数 |
|---|---|---:|
| 训练比例常数（Prior） | 仅由17训练目标计算p，不优化 | 1个统计比例 |
| 统计逻辑回归（Stats-LR） | 16列公开图统计，训练区标准化，L2、C=1、最多1000迭代 | 17（含截距） |
| 无边节点模型（NodeOnly） | 同22列；22→16投影＋2层自身线性/ReLU，无消息 | 1201 |
| 合并关系图模型（Untyped-GNN） | 同22列，所有有向边和重边保留，只合并关系类型 | 2229 |
| 关系图模型（R-GCN） | 同22列、7关系、2层、宽16，复用既有实现 | 2253 |

三种神经模型均使用原分类头16→16→1及“每测试节点均值、再产品内全部测试等权均值”。各产品对应一个监督损失，不把测试图当作独立样本。FP32、Adam学习率0.001、无类别权重、产品批次2、末批1保留，种子固定20261101/02/03。全部从预定种子重新初始化，不延续04T1或微型模型。

关系层公式为`ReLU(W_self*h + Σ_r mean_(入边关系r)(h_source*W_r))`，关系矩阵以2个基矩阵组合。合并关系模型只剩一个入边均值；因此**它同时改变了按关系归一化的方式**。不能把全部差异归因于关系名称或类型语义。无边对照没有未使用的关系参数。旧类型无关模式会抹去节点类别，本轮没有复用它冒充只合并边类型。

统计16列依次为：登记测试数；每测试节点、边、事件数均值；7种关系边数的每测试均值；5类节点在每测试中的比例再对测试等权平均。前11计数列统一log1p，其余5比例列保持原值。标准化只由17训练产品拟合；本次没有零方差列，预定零方差处理为尺度1。逻辑回归使用既有scikit-learn1.3.0的L2/lbfgs求解，16次迭代正常收敛，无警告，不搜索参数。这不是CLAP或PULP复现。

实际进入图模型的关系数量：

| 关系代码 | 训练边数 | 验证边数 |
|---|---:|---:|
{table([[k,inputs['relations']['train'][k],inputs['relations']['val'][k]] for k in inputs['relations']['train']])}

全部22列保留；19列有变化，`program_unresolved`为0表示合格节点无未解析身份，`config_enabled/config_known`两列因约束未认证保持屏蔽，不表示配置实际全部关闭。未增加原始数值、方法名、产品/版本编号或绝对位置特征。

## 三、常数基准与04T1工程模型复核

实际训练比例为10/17＝{prior['p_train']:.9f}，由本次训练目标计算；产品等权常数交叉熵为{prior['train_metrics']['log_loss']:.15f}。验证仍使用同一训练比例，没有使用2/5重新拟合常数。其验证损失0.7446。

通过原摘要和契约安全恢复04T1最终工程模型，只在17训练产品复核。训练交叉熵{eng['metrics']['log_loss']:.9f}，比常数基准高{diagnostics['engineering_vs_prior_loss_delta']:.9f}。概率最小/最大为{ep['min']:.9f}/{ep['max']:.9f}，均值{ep['mean']:.9f}、总体标准差{ep['std_population']:.9f}；阈值0.5下17个全部预测为FP。

| 真实训练类别 | 产品数 | 预测均值 | 标准差 | 最小—最大 |
|---|---:|---:|---:|---|
{table([[c,d['n'],f"{d['mean']:.9f}",f"{d['std_population']:.9f}",f"{d['min']:.9f}—{d['max']:.9f}"] for c,d in eng['metrics']['by_class_probabilities'].items()])}

这支持“输出变化很小且阈值预测仍为多数类”，但不是严格常数：概率标准差大于事先定义的1e-4近常数阈值，范围非零，训练排序指标也不是常数模型值。不能由平均损失接近常数基准就断言完全没有使用输入。

该模型的表示均为16维；136对产品表示的距离最小0.000246、均值0.014353、最大0.035420，均非近零阈值1e-6。表示差异较小，但不完全相同。

| 位置（代码名） | 行数 | 维度 | 各维总体方差的均值 | 行向量范数均值 |
|---|---:|---:|---:|---:|
{table(rep_rows)}

第二层节点到测试均值方差明显缩小，但测试均值到产品均值并未进一步缩小。不同统计的行数、对象及权重不同，这些描述量只能提示均值汇聚的平滑作用，不能认证稀有缺陷信号被删除，更不能据此读取缺陷位置或更改模型。

## 四、原4产品的500步学习能力诊断

沿用04T1固定4产品、4版本、87图，两类各2个，没有重新选样；同一小型R-GCN重新初始化，固定500步后停止。诊断模型没有参加验证比较。

| 优化步 | 训练损失 | 概率均值 | 概率标准差 | 概率最小—最大 | 产品表示方差均值 |
|---:|---:|---:|---:|---|---:|
{table([[x['step'],fmt(x['metrics']['log_loss']),fmt(x['metrics']['probability_summary']['mean']),fmt(x['metrics']['probability_summary']['std_population']),fmt(x['metrics']['probability_summary']['min'])+'—'+fmt(x['metrics']['probability_summary']['max']),f"{x['representation']['product_means']['mean_feature_variance']:.7g}"] for x in micro])}

同样样本在100步时损失约0.6803，500步下降至0.002201，产品表示最小距离升至约0.4037。这直接支持“先前有限100步存在优化预算不足”的解释，排除这4样本在当前实现下绝对不能拟合的说法。没有因此宣称全部17样本可完全分离，也未追加容量、步骤或种子。当前仍不能确证全训练残余损失来自标签噪声、输入不可区分或特定汇聚缺陷。

## 五、固定训练／验证结果

所有9神经运行均实际完成200轮，合计1800轮；每轮17产品/283图，无截断或静默跳过。只从冻结的四个候选轮次中选择验证产品平均对数损失最小者；改善不超过1e-6时选择更早轮次。未按F1调阈值，未选择最好的随机种子，未据验证结果追加训练。下表列出所有预定种子，数值保留四位只便于阅读，完整值在机器文件。

| 方法 | 种子 | 选定轮次 | 训练损失 | 验证损失 | 验证Brier | 验证ROC面积 | 验证AP | FP精确率 | FP召回率 | FP F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table(scores)}

AP为平均精确率，按分数并列组的召回增量×精确率累计；不是梯形精确率—召回率曲线面积。固定阈值0.5，精确率/召回率/F1除零取0并报告样本支持数；单类时ROC面积/AP为空、不作排序解释，二分类损失与Brier仍可计算。本轮验证两类均存在，所有非恒定方法在这5个样本上ROC面积/AP均为1，存在明显的指标饱和，不能据此声称真实部署完美分类。

验证混淆矩阵的行是真实[TP=0,FP=1]、列是预测[TP=0,FP=1]。常数为`[[0,3],[0,2]]`（预测FP5/TP0）；统计逻辑回归为`[[2,1],[0,2]]`（3/2）；节点对照三个种子均为`[[1,2],[0,2]]`（4/1）；合并关系及R-GCN各三个种子均为`[[3,0],[0,2]]`（2/3）。训练区的全部混淆矩阵、概率范围/均值/标准差和按类别分布也保留在JSON/CSV，没有省略不利的训练结果。

| 神经方法 | 三种子选定轮次 | 验证损失均值 | 种子样本标准差 | 范围 |
|---|---|---:|---:|---|
{table(epoch_choices)}

种子标准差仅描述优化随机性，同一5产品重复三次不是15个独立样本；这也不是抽样置信区间。验证只有4独立版本，而且同时用于轮次选择，因此结果存在选择乐观性，本阶段不作强显著性主张。不使用测试区进行复核或模型修改。

## 六、哪些结论得到支持

**训练学习信号：** R-GCN三次最终200轮训练损失约0.5278—0.5371，低于训练比例基准0.6775；选定轮次的均值0.5334。节点和统计对照同样学到信号，统计逻辑回归训练损失0.5020甚至更低。因此不能将一切可学习性归功于关系结构。

**验证表现：** 节点对照、统计对照均优于常数损失；合并关系模型相比无边模型进一步降低损失并减少固定阈值错误，提示图连接在当前先导条件下有可观察价值。R-GCN相对关系合并模型的验证损失在对应三个种子均降低，均值0.3080对0.3492，Brier也更低；这一增益发生在概率质量上，ROC/AP及固定阈值F1已经相同，不能再声称排序或错误数有进一步提升。

**关系类型增益的界限：** R-GCN与合并关系模型既有关系区分差异，也有归一化差异及少量参数差异。3个种子上的方向一致是当前队列的可观察结果，尚不能识别纯关系语义的独立效应，更不是因果效应。统计/节点对照解释了部分收益，但没有在本次验证损失上达到R-GCN水平。

**输出与表示：** 选定R-GCN训练概率标准差约0.254—0.262，已明显区别于04T1的0.000453；训练阈值预测为FP10/TP7，仍有训练错误，并非全部预测多数类。各R-GCN产品表示没有近零距离对。节点对照每个种子有1对近零表示，说明该对照存在表示压缩；本次没有定位该对是否标签冲突，不能据此编造根因。全部训练层级方差、距离和初始/候选表示摘要见诊断JSON。

按版本的验证损失与错误示例（完整五方法/三种子版本统计见机器交接；版本身份只作审计）：

| 方法 | 种子 | 缺陷版本 | 产品数 | 平均损失 | 错误数 |
|---|---|---|---:|---:|---:|
{table(version_rows)}

不能把这些先导分数与阶段02不同队列、不同训练参考池的CLAP/PULP分数相减后宣称超越；本次没有做这样的文献基线比较。当前数据也不足以证明跨系统泛化。

## 七、实际测试、资源、失败记录及服务器

14项新对照/指标/读取边界与资源检查通过，58项既有无损读取、图结构和04T1边界回归通过，共72项；没有执行会额外拟合旧模型或读取测试目标的回归。人工验证包括无边模型忽略边但保留22列、合并关系仅消除类型、方向/重边保留、固定参数量、统计等权、选择容差、AP并列及单类/除零处理、禁止测试预测和独立验证目标路径限制。真实图训练、完整汇聚和验证结果来自本次实际运行，人工测试不计作真实SPL分数。

冻结前曾有一次新增模块字符串漏引号，导致测试导入失败，尚未开始真实拟合或读取验证标签；修复后14项通过。原失败日志保留为`comparison_tests.log`，成功为`comparison_tests_fixed.log`。预检还确认图环境没有scikit-learn，没有擅自安装或升级，改用现有可验证的统计解释器并在冻结方案中声明。实际正式训练/评价均一次完成，未发生看分数后的代码修复或一致性重跑需求。

图环境：`{result['environment']['hostname']}`，`{result['environment']['os']}`，Python3.11.4，解释器`{result['environment']['executable']}`，PyTorch2.6.0+cu126、NumPy2.4.6，单CPU计算线程。统计环境在同一主机CPU，解释器`{stats['environment']['executable']}`，NumPy1.24.3、SciPy1.10.1、scikit-learn1.3.0；不是同一虚拟环境，软件差异如实记录。CUDA构建后缀不表示执行GPU。

训练主进程耗时{resources['training_process_seconds']:.3f}秒；首个模型至训练落盘的保守窗口{resources['model_window_seconds']:.3f}秒，包含诊断、统计子进程等待和中间写盘。连同冻结/单元检查等进程计入的保守模型与检查上界{resources['conservative_model_and_checks_seconds']:.3f}秒，低于3600秒；全部计划的神经更新16700，另L-BFGS16次迭代，共16716。验证评价进程{resources['validation_seconds']:.3f}秒。各神经200轮运行用时也记录在诊断JSON，不能从此推算GPU吞吐。

真实图工作进程操作系统生命周期峰值驻留内存{result['memory']['peak_working_set_bytes']}字节（约{result['memory']['peak_working_set_bytes']/1024**3:.3f}GiB）。外部监控看到的约5MiB只是Windows虚拟环境启动器，未用作模型内存；该监控不保证张量子进程硬内存配额。统计进程独立峰值、反向激活内存和GPU显存未测量，使用null及原因。没有非数值、无穷值、超时、内存耗尽或静默截断。

写报告时目录占{resources['run_directory_bytes_at_report']}字节，D盘剩余{resources['disk_free_bytes_at_report']}字节；收尾指标见`final_audit.json`。历史图包、标签、划分、正式结果保持冻结。没有删除被引用的旧报告或失败证据；本次仅一份完整中文交接，未发现可安全删除且无独立用途的重复长文档。

没有可用的明确Linux服务器会话，本次未访问或改动X:/codes、未扫描连接信息、未修改共享环境；历史访问拒绝不能当作本次服务器检查。Linux/GPU状态均为未验证。已提供Linux同一CPU对照启动脚本及两套分离的依赖记录，不能将脚本存在写成Linux已运行。目标机器需先核验环境并以真实路径生成新冻结协议，不能直接使用本机Windows路径；当前可移植冻结入口对解释器路径会解析符号链接，应使用已核对的环境实际解释器，避免解析到缺依赖的全局环境。GPU前向/反向/显存验收仍未执行，也未用CPU内存推算显存。

本阶段不需要迁移整个BRFL或重新执行Java。迁移清单保存在运行目录`deployment_manifest.json`：公开图包与源码、独立04T1工程检查点及回执、公开清单/冻结划分分别传递；训练和验证目标仅提供给对应角色，测试目标不迁移。尚未在Linux执行该启动脚本或完成Bash语法验收。

## 八、与世界模型的关系及唯一首要建议

本轮验证的是监督分类，不是世界模型：真实有序事件/支持范围内的依赖结构（T1）部分可用；完整实测状态、输入、输出和断言数值（T2）不足；合法干预前后执行对（T3）缺失。当前22列只表示类别和布尔状态，未把原始数值混入训练；没有世界模型、反事实或强化学习结果。

**下一步首要任务：实现并认证真实时间前缀生产器。** 依据是本轮已观察到有限学习信号，没有证据要求立即增加分类器容量；而世界模型仍缺无法替代的、按真实可见时点构造的输入。保留当前分类器及简单对照作为后续参考，先将真实可观测的下一执行事件/指令类别、可认证分支后继等局部结构作为候选预测目标。数值目标只在实际观测充分时启用并保留缺失掩码。

前缀必须排除未来事件、尚未发生的依赖边、未完成调用效果、最终断言和测试结果；静态结构与事后执行标记分别处理。完整事后图不能直接当作早期状态，模型预测值也不能冒充真实观测。此项成功只解除结构事件预测的数据前置阻碍，不直接证明因果机制。代理标签资格、交易配置/Email覆盖及独立版本规模仍是正式研究的后续限制，但本报告只推荐这一项首要任务，不自动实施。

## 九、命令与交付索引

核心六文件为`phase04R1_protocol.json`、`phase04R1_review_packet.md`、同名JSON、`phase04R1_metrics.csv`、`phase04R1_learning_curves.csv`、`phase04R1_prediction_diagnostics.json`，均位于`outputs/`。指标CSV含76条候选×分区记录，`selected`标识最终选中/固定基准共22条；学习曲线1800条，微型诊断的0/100/200/300/400/500步保存在诊断JSON。

受控运行目录：`{out.as_posix()}`。逐产品训练/验证预测、评价连接记录、36个检查点及回执均留在此处，公共报告只引用路径和摘要。训练预测索引、评价结果、三份目标访问日志、原始逐步损失和命令记录可独立核对，未复制测试标签。

以下实际命令的参数、退出码和耗时均来自`.command.json`；执行时`PYTHONPATH`指向项目`src`，测试另外加入`tests`。成功命令不是预期成绩；首条失败记录也保留：

{command_text}

阶段04R-1结束，没有执行测试集评价或下一阶段。
'''
    (ROOT/'outputs/phase04R1_review_packet.md').write_text(report,encoding='utf-8')
    migration={'status':'PREPARED_NOT_EXECUTED','device_in_supplied_comparison_script':'cpu','Linux_GPU_verified':False,
        'public':['src/cwm_fp/','scripts/run_phase04R1_linux.sh','configs/phase04R1_graph_requirements.txt','configs/phase04R1_stats_requirements.txt',
            protocol['paths']['package'],str(Path(protocol['paths']['package']).parent/'export_receipt.json')]+[protocol['paths'][k] for k in ('products','tests','splits')],
        'engineering_reference':[str(Path(protocol['paths']['old_run'])/n) for n in ('plan.json','micro_selection.json','engineering_train_only.pt','engineering_train_only.receipt.json','result.json')]+
            ['outputs/phase04T1_review_packet.md','outputs/phase04T1_review_packet.json'],
        'separate_role_targets':{'trainer':protocol['paths']['train_labels'],'evaluator':protocol['paths']['val_labels'],'test':'DO NOT COPY'},
        'freeze_entry':'python -m cwm_fp.models.package_comparison --action freeze --protocol-file NEW_PROTOCOL --run-dir NEW_OUT --old-run OLD_RUN --stats-python STATS_PY --package PACKAGE --products PRODUCTS --tests TESTS --splits SPLITS --train-labels TRAIN --val-labels VAL',
        'run_entry':'bash scripts/run_phase04R1_linux.sh GRAPH_PY NEW_PROTOCOL TRUSTED_PROTOCOL_SHA',
        'path_caveat':'Use actual environment interpreter; current freeze path.resolve follows symlinks. Verify environment before freezing. No remote execution here.'}
    write_json(out/'deployment_manifest.json',migration)
    # Final byte/format checks do not open any original target.
    package=Path(protocol['paths']['package']);manifest=read(package/'manifest.json');checks=1
    assert sha256(package/'manifest.json')==protocol['manifest_sha256']
    for product in manifest['products']:
        for g in product['tests']:
            for kind in ('tensor','metadata'):assert sha256(package/g[kind]['path'])==g[kind]['sha256'];checks+=1
    historical=[]
    for f in (out/'producer_before/outputs').glob('phase04T1*'):
        assert sha256(f)==sha256(ROOT/'outputs'/f.name);historical.append(f.name)
    for name in changes:
        src=ROOT/name;assert src.is_file(),name
        if src.suffix=='.py':ast.parse(src.read_text(encoding='utf-8'),filename=name)
        dst=out/'producer_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    report_json=read(ROOT/'outputs/phase04R1_review_packet.json')
    assert report_json['authorized_validation_access_performed'] and not report_json['test_labels_read']
    assert len(metrics_rows)==76 and sum(bool(x['selected']) for x in metrics_rows)==22
    audit={'status':'PASS','protocol_sha256':receipt['sha256'],'source_files':len(protocol['source_sha256']),
        'package_files_verified':checks,'history_unchanged':historical,'curves_rows':1800,'metric_rows':76,'selected_metric_rows':22,
        'adam_updates':16700,'stats_iterations':sum(stats['iterations']),'validation_after_training':True,
        'checkpoint_candidates':36,'test_labels_opened_by_report':False,'model_runs_by_report':0,
        'disk_free_bytes':shutil.disk_usage(out).free,'run_bytes_before_index':sum(f.stat().st_size for f in out.rglob('*') if f.is_file())}
    assert audit['run_bytes_before_index']<protocol['budget']['new_bytes']
    write_json(out/'final_audit.json',audit)
    index={str(f.relative_to(out)):{'sha256':sha256(f),'bytes':f.stat().st_size} for f in out.rglob('*') if f.is_file() and f.name not in ('artifact_index.json','report.log','report.command.json')}
    write_json(out/'artifact_index.json',{'files':index,'excluded':'self and current report command files'})
    print(json.dumps({'status':'PASS','metrics_rows':len(metrics_rows),'learning_rows':len(curves),'artifact_files':len(index)},ensure_ascii=False))


if __name__=='__main__':main(OUT)
