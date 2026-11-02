"""Build the phase04P handoff from actual, sealed five-product checks only."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/processed/graph_classifier/phase04P_v1';REPORT=ROOT/'outputs'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def write(name,data):
    p=REPORT/('phase04P_'+name)
    if isinstance(data,str):p.write_text(data,encoding='utf-8')
    else:p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def mib(x):return f'{x/1024**2:.3f}'
def csv_write(name,rows):
    with (REPORT/('phase04P_'+name)).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    plan=load(OUT/'plan.json');v=load(OUT/'verification.json');m=load(OUT/'package/manifest.json')
    integrity=load(OUT/'integrity_verification.json');receipt=load(OUT/'export_receipt.json');unit=load(OUT/'unit_tests.json')
    old=load(REPORT/'phase03B4_review_packet.json');env=load(OUT/'environment.json');migration=load(OUT/'relocation.json')
    assert integrity['manifest_sha256']==v['manifest_sha256']==migration['manifest_sha256']==sha(OUT/'package/manifest.json')
    assert unit['tests_run']==50 and unit['failures']==unit['errors']==0
    for p,h in plan['source_sha256'].items():assert sha(p)==h,p
    assert {tuple(p['identity']) for p in m['products']}=={(p['system_id'],p['bug_version_id'],p['product_id']) for p in old['admitted_products']}
    totals=Counter();relations=Counter()
    for p in v['products']:
        for k in ('nodes','edges','events'):totals[k]+=p[k]
    for g in v['graphs']:relations.update(g['relations'])
    package_bytes=sum(p.stat().st_size for p in (OUT/'package').rglob('*') if p.is_file())
    max_error=max(c['max_abs_error'] for p in v['products'] for c in p['comparisons'].values())
    parts={}
    for fold in m['split_protocols']:
        pcounts={s:sum(p['partitions'][fold]==s for p in m['products']) for s in ('train','val','test')}
        versions={s:len({tuple(p['identity'][:2]) for p in m['products'] if p['partitions'][fold]==s}) for s in pcounts}
        prior=old['split_readiness'][fold];assert pcounts==prior['products'] and versions==prior['independent_versions']
        audit=prior['authorized_train_audit'];assert audit['admission_sha256']==sha(ROOT/'data/processed/lddg/phase03B4_framework_v1/admission.json')
        parts[fold]={'status':'BLOCKED','products':pcounts,'independent_versions':versions,'reason':prior['reason'],
            'authorized_train_counts':{c:audit['train_target_counts'].get(c,0) for c in ('FP','TP')},
            'train_has_both_classes':audit['train_has_both_FP_TP'],
            'target_count_basis':'Reuse of phase03B4 authorized train-only audit; cohort, admission and split hashes unchanged. No targets read in04P.',
            'train_audit_evidence':'outputs/phase03B4_review_packet.json#/split_readiness/'+fold+'/authorized_train_audit',
            'validation_target_counts':None,'test_target_counts':None,'unknown_reason':'Validation/test targets are protected and were not opened.'}
    changed=['src/cwm_fp/models/'+x for x in ['graph_data.py','graph_encoder.py','graph_classifier.py','train_graph.py','graph_package.py','prefix_contract.py']]
    changed+=['tests/test_graph_package.py']+['scripts/'+x for x in ['verify_phase04P.py','test_phase04P.py','audit_phase04P_integrity.py','run_phase04P_checks.py','report_phase04P.py']]
    changed+=['DATA_CONTRACTS.md','STAGES.md','outputs/phase_status.md']
    statuses={
      'DATA_PIPELINE_READY':{'status':'READY','reason':'Five complete products /75 graphs losslessly loaded; source equality and fixed-parameter CPU equivalence verified. Scoped to bounded input and current semantic support.'},
      'GPU_ENVIRONMENT_READY':{'status':'NOT_VERIFIED','reason':'Linux deployment instructions prepared; no Linux/server/GPU execution.'},
      'SUPERVISED_TRAINING_READY':{'status':'BLOCKED','reason':'Frozen grouped training lacks TP; both LOSO protocols lack nonempty independent partitions.'},
      'WORLD_MODEL_T1_READY':{'status':'PARTIAL','reason':'Real ordered events and dependencies retained; prefix contract checked only on artificial records, no certified real prefix producer.'},
      'WORLD_MODEL_T2_READY':{'status':'BLOCKED','reason':'Partial scalar observations only; complete inputs/heap/outputs and visible-time state transition pairs absent.'},
      'COUNTERFACTUAL_READY':{'status':'BLOCKED','reason':'No real legal intervention pairs; instrumentation parity is not intervention.'}}
    resources={k:load(OUT/(k+'_self_resource.json')) for k in ['build','verify','stream_only','relocated']}
    commands=[load(OUT/(n+'.command.json')) for n in ['build_release','verify_release','stream_only_release','relocated_release','unit_final_v3','integrity']]
    stream=load(OUT/'stream_forward.json')
    resource={'environment':env,'budget':{k:val for k,val in plan.items() if k.startswith('max_') or k=='min_free_bytes'},
        'actual_python_processes':resources,'final_validation_commands':commands,'final_validation_command_seconds_sum':sum(c['seconds'] for c in commands),
        'total_stage_wall_seconds':None,'total_stage_wall_seconds_reason':'Interactive development and two interrupted monitoring attempts were not measured by one continuous timer.',
        'package_bytes':package_bytes,'two_package_copies_bytes':package_bytes*2,
        'generated_directory_bytes_at_report':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),
        'disk_free_bytes_at_report':shutil.disk_usage(OUT).free,'disk_before_bytes':plan['disk_before']['free'],
        'max_graph_nodes':max(g['loaded_nodes'] for g in v['graphs']),'max_graph_edges':max(g['loaded_edges'] for g in v['graphs']),
        'max_graph_events':max(g['loaded_events'] for g in v['graphs']),'max_product_tests':20,
        'max_shard_bytes':max(g[k]['bytes'] for p in m['products'] for g in p['tests'] for k in ('metadata','tensor')),
        'GPU_peak_bytes':None,'GPU_throughput':None,'GPU_unknown_reason':'No GPU execution or training was authorized/performed.',
        'actual_timeouts':0,'actual_OOM':0,'actual_data_truncations':0,'java_runs':0,'training_runs':0,
        'monitoring_caveat':'Outer runner measures the Windows venv launcher (~5MiB), not tensor process. Use actual_python_processes sampled RSS; import-time peaks before monitor starts are not separately captured.',
        'development_failures':[{'code':'MISSING_PSUTIL','exit_code':1,'evidence':rel(OUT/'verify.log'),'resolution':'Reuse already-installed base-environment psutil after importing torch; no install.'},
            {'code':'TEMP_STAT_RACE','attempts':2,'wrapper_exit_code':1,'child_test_exit_code':None,'reason':'Temporary unittest file/directory removed during rglob/stat. Wrapper had no final command record; these partial runs are not counted as passed.',
             'evidence':[rel(OUT/'unit_final.log'),rel(OUT/'unit_final_v2.log')],'resolution':'Phase-local os.walk/stat monitor ignores only disappeared files under this phase temp directory; frozen source hashes remain strict.'}]}
    rows=[]
    for p,pm in zip(v['products'],m['products']):
        k=p['identity'];r=Counter()
        for g in pm['tests']:r.update(g['relations'])
        row=dict(zip(('system_id','bug_version_id','product_id'),k))
        row.update(original_result='PASS',registered_tests=p['registered_tests'],loaded_tests=p['loaded_tests'],identity_qualified_tests=p['loaded_tests'],semantic_qualified_tests=p['loaded_tests'],suite_complete=True,
            formal_data_admission=True,formal_training_experiment_ready=False,original_nodes=p['nodes'],loaded_nodes=p['nodes'],original_edges=p['edges'],loaded_edges=p['edges'],original_events=p['events'],loaded_events=p['events'],
            control_dep_edges=r['control_dep'],data_dep_edges=r['data_dep'],test_loss_fraction=0,node_loss_fraction=0,edge_loss_fraction=0,event_loss_fraction=0,feature_columns=22,
            input_shard_bytes=p['input_shard_bytes'],tensor_batch_bytes=p['tensor_bytes'],**p['partitions'],rejection_reason_codes='',training_blocker_code='FROZEN_PROTOCOL_NOT_READY',
            evidence=rel(OUT/'verification.json'),manifest=rel(OUT/'package/manifest.json'))
        rows.append(row)
    csv_write('product_status.csv',rows)
    grows=[]
    for g in v['graphs']:
        row=dict(zip(('system_id','bug_version_id','product_id','test_id'),g['identity']))
        row.update(original_result='PASS',instrumented_result='PASS',result_basis='inherited_verified03B4; not rerun in04P',identity_qualified=True,dependency_semantics_qualified=True,read_qualified=True,
            original_nodes=g['original_nodes'],loaded_nodes=g['loaded_nodes'],original_edges=g['original_edges'],loaded_edges=g['loaded_edges'],original_events=g['original_events'],loaded_events=g['loaded_events'],
            control_dep_edges=g['relations']['control_dep'],data_dep_edges=g['relations']['data_dep'],node_loss_fraction=0,edge_loss_fraction=0,event_loss_fraction=0,feature_columns=22,
            rejection_reason_codes='',evidence=rel(OUT/'package'/g['evidence']['path']),evidence_sha256=g['evidence']['sha256'],source_comparison=rel(OUT/'integrity_verification.json'))
        grows.append(row)
    csv_write('graph_status.csv',grows)
    before={'scope':'Same five previously admitted products, not all03B4 graphs','registered_tests':75,'candidate_graphs':75,'identity_qualified_graphs':75,'dependency_semantics_qualified_graphs':75,'complete_products':5,'admitted_products':5,'versions':4,**dict(totals),
            'default_tests_loaded':68,'default_test_loss_count':7,'default_test_loss_fraction':7/75,'default_sampling_lossless_products':3,
            'actual_old_full_relation_tensor_nodes':None,'unknown_reason':'Old report measured sampling losses but its both ablation intentionally excluded temporal relations; no reconstruction of unmeasured old tensor totals.'}
    after={**before,'default_tests_loaded':75,'default_test_loss_count':0,'default_test_loss_fraction':0,'default_sampling_lossless_products':5,'loaded_nodes':totals['nodes'],'loaded_edges':totals['edges'],'loaded_events':totals['events'],
           'relations':dict(relations),'test_loss_fraction':0,'node_loss_fraction':0,'edge_loss_fraction':0,'event_loss_fraction':0,'feature_columns':22,'rejected_graphs':0,'new_admitted_products':0,'new_semantically_recovered_graphs':0}
    after.pop('actual_old_full_relation_tensor_nodes');after.pop('unknown_reason')
    issues=[
        {'code':'SPLIT_CLASS_COVERAGE','cause_confirmed':True,'affected':'all three protocols','evidence':'outputs/phase03B4_review_packet.json#/split_readiness','repair':'Future label-blind bounded cohort plan, retain frozen assignments','product_recovery_upper_bound':None,'unknown_reason':'No new collection authorized/planned in04P','world_model_impact':'Insufficient independent cross-version/system evaluation.'},
        {'code':'UNVERIFIED_LINUX_GPU','cause_confirmed':True,'affected':'all five products on target server','evidence':'outputs/phase04P_linux_portability.md','repair':'Run deployment and identical package checks after authorization','product_recovery_upper_bound':0,'world_model_impact':'Deployment evidence only, no new states.'},
        {'code':'PREFIX_STATE_VISIBILITY','cause_confirmed':True,'affected':'all75 graphs for predictive state learning','evidence':'outputs/phase04P_world_model_interface.md','repair':'Implement observer-supplied availability and independently validate real prefixes','product_recovery_upper_bound':0,'world_model_impact':'Whole-graph executed flags/outputs/posterior checks must not enter early inputs.'},
        {'code':'OUT_OF_SCOPE_SEMANTICS','cause_confirmed':True,'affected':'Historical remaining Transaction/Email cases; not reopened in04P','evidence':'outputs/phase03B4_review_packet.json#/unresolved_issues','repair':'Separate bounded semantic study; not inferred safe from current five products','product_recovery_upper_bound':None,'unknown_reason':'Current scope excludes these products; no new benefit estimate','world_model_impact':'Uncertified exceptions/arrays/library effects cannot supervise mechanisms.'}]
    rec=[{'priority':1,'task':'保持冻结划分，制定下一批标签盲的数据与语义补齐计划','reason':'训练缺TP及跨系统独立分区是主要研究阻碍；先审定范围和预算，不能靠这5产品拟合解决。'},
         {'priority':2,'task':'在Linux RTX3090Ti执行独立迁移、依赖和未训练前向验收','reason':'当前包与部署步骤已可准备；先核验驱动、包摘要、CPU/GPU输出与资源，不自动启动分类训练。'},
         {'priority':3,'task':'按观测可见时刻实现并验证真实时间前缀生成器','reason':'先获得无未来泄漏的T1前缀；数值状态和合法干预证据仍需后续补齐。'}]
    packet={'stage_id':'04P','stage_status':'COMPLETED_SCOPED_VALIDATION','actual_scope':{'system':'BankAccountTP','products':5,'versions':4,'registered_tests':75,'new_Java_runs':0,'new_versions':0,'models_fitted':0,'labels_read':False},
        'before_metrics':before,'after_metrics':after,'recovered_graphs':{'new_semantic_recoveries':0,'restored_default_test_inputs':7,'restored_lossless_products':2,'original16_note':'03B4 previously recovered14/16;04P reuses admitted75 and does not revisit rejected graphs.'},
        'admitted_products':[p['identity'] for p in m['products']],'rejection_reasons':[],'split_readiness':parts,'readiness':statuses,
        'world_model_readiness':{'T1':statuses['WORLD_MODEL_T1_READY'],'T2':statuses['WORLD_MODEL_T2_READY'],'T3':statuses['COUNTERFACTUAL_READY'],'observations':integrity['counts'],'real_prefix_dataset_available':False,'real_intervention_pairs':0},
        'tests_executed':{'regression':unit,'real_graphs_verified':75,'source_equal_graphs':75,'root_relocated_graphs':75,'Java_tests_executed':0,'model_forward_comparisons':30,'max_abs_error':max_error,'atol':1e-6,'rtol':1e-5,'frozen_parameters_unchanged':True},
        'resources':resource,'changed_files':changed,'artifact_paths':{n:rel(REPORT/('phase04P_'+n)) for n in ['review_packet.md','review_packet.json','product_status.csv','graph_status.csv','data_integrity.json','linux_portability.md','world_model_interface.md','resource_profile.md']},
        'package_path':rel(OUT/'package'),'manifest_sha256':receipt['manifest_sha256'],'unresolved_issues':issues,'recommended_next_stage':rec,
        'preservation':{'original_fingerprints_unchanged':True,'verified_frozen_files':len(plan['source_sha256']),'before_code_snapshot':rel(OUT/'producer_before'),'development_attempts_preserved':True}}
    write('review_packet.json',packet)
    write('data_integrity.json',{'stage_id':'04P','status':'PASS','manifest_sha256':receipt['manifest_sha256'],'checks':integrity,'products':v['products'],'graphs':v['graphs'],
        'relocation':migration,'node_feature_policy':'22 semantic/boolean whitelist only; full identities and values are audit metadata, never numerical IDs',
        'evidence_boundary':'Collector rechecked original source/JVM support; portable importer rechecks hashes, shape, identity, order and edges, not Java execution itself.'})
    ptable=table(['系统/版本/产品','登记→读取','节点原始=读取','边原始=读取','事件原始=读取','控制/数据依赖','四项损失'],[
        [r['system_id']+'/'+r['bug_version_id']+'/'+r['product_id'],f"{r['registered_tests']}→{r['loaded_tests']}",r['loaded_nodes'],r['loaded_edges'],r['loaded_events'],f"{r['control_dep_edges']}/{r['data_dep_edges']}",'均0%'] for r in rows])
    stable=table(['协议','训练产品/版本','验证产品/版本','测试产品/版本','授权训练类别（历史核验）','结论'],[
        [f,*[f"{d['products'][s]}/{d['independent_versions'][s]}" for s in ('train','val','test')],str(d['authorized_train_counts']),'BLOCKED'] for f,d in parts.items()])
    rtable=table(['测量进程','本体耗时/秒','采样峰值驻留内存/MiB'],[[k,f"{r['seconds']:.3f}",mib(r['peak_sampled_rss_bytes'])] for k,r in resources.items()])
    write('review_packet.md',f'''# CWM-FP 阶段04P结果交接报告

本阶段已完成限定范围的工程实现与真实图验收。**五个已准入产品、75张图全部无损读取，22维节点特征和全部7类关系可进入现有模型；正式监督训练仍阻塞。** 本阶段没有新增Java执行、模型拟合或分类成绩，也没有连接Linux服务器。

## A. 实际完成情况与证据层次

研究仍是利用真实执行依赖和监督目标识别假通过产品，最终研究因果世界模型。本阶段只处理BankAccountTP已有4个版本的5个原始通过产品。输入及划分的17个冻结文件摘要重新核对一致，见`data/processed/graph_classifier/phase04P_v1/plan.json`。

修改了`graph_encoder.py`：新增完整关系模式`full`及有容量上限的无损编码，旧消融模式保留。修改`graph_data.py`：默认不再截取16测试，超过承载范围明确拒绝。修改`graph_classifier.py`：把原来的表示计算提取为可复用方法，参数结构、层数和两级均值汇聚不变。修改`train_graph.py`：默认完整读取，所有分区在打开训练目标前检查损失，显式旧采样结果有损时拒绝。

新增`graph_package.py`：逐测试张量/压缩元数据、相对路径、外部可信摘要、完整套件关联、分片恢复和评价模式逐测试汇聚。新增`prefix_contract.py`：仅定义未来观测可见时刻契约及拒绝检查，不从事后图制造前缀。新增`test_graph_package.py`及5份本阶段验证/报告脚本。完整文件清单在同名JSON的`changed_files`。

真实研究图上验证：75图与原始张量逐值一致；节点、边、全部产品事件、静态目录、框架效果和质量记录逐项一致（绝对路径仅改为可追溯标记）；重新执行既有源码/JVM证据准入，未重跑Java。五产品分别验证3类未训练模型、1/3测试分批与整产品计算，共30次对照，最大绝对差{max_error:.10g}，容差绝对1e-6、相对1e-5；参数摘要不变。

人工或破坏性副本验证：单测试、超过16测试、混合大小、恢复、同名跨版本、损坏/缺失/摘要错误、坏质量、19/22列混用、不完整套件、越界路径、容量拒绝、默认命令行为、时间前缀未来信息拒绝。**50项测试通过，0失败、0错误、0跳过**，其中17项既有图模型非拟合回归及33项本阶段检查。坏文件仅为隔离副本。未验证Linux/GPU执行、带梯度的分片训练和真实前缀生成；未实现新的Java语义或状态模型。

## B. 五个产品的无损核对

{ptable}

共{totals['nodes']}节点、{totals['edges']}边、{totals['events']}条产品/测试动态事件，全部保留。七类边计数：{dict(relations)}。静态控制依赖是静态证据，不因此改称实际动态因果边；框架内部物理事件仍保留在原证据及调用效果记录中，不新增产品事件。

候选图75→75、身份合格75→75、依赖语义合格75→75、完整套件产品5→5、数据正式准入5→5；**新增语义合格产品为0**。旧默认16测试仅读取68项，现在75项，恢复7项默认输入；默认采样无损产品3→5。旧`both`关系消融有意排除时间相邻边，新`full`完整保留3511条该关系，不改写旧实验。原16张语义拒绝图在03B4已恢复14张；04P没有继续修复剩余2张，也不能把输入恢复记作语义恢复。

最大单图{resource['max_graph_nodes']}节点、{resource['max_graph_edges']}边、{resource['max_graph_events']}事件，最大产品20测试。每产品、每测试完整身份、检查结果和证据引用见两份状态CSV。

## C. 训练数据接口与边界

已经能够构建完整产品表示：每测试单独编码和均值汇聚，再按测试等权汇聚产品，禁止跨测试动态边。每张图22列固定语义/布尔特征，不编码类名、方法名、源码行、产品号、缺陷号或监督标签；这些真实身份仅保留在连接和审计元数据中。真实状态值及缺失标记单独保存，未作为分类器数值输入。

新默认读取无静默采样，包中节点/边/事件/测试均零损失。旧配置中的128MiB是整个JSON行文件的容量保护，仍保留；新入口每测试分片、单文件及解压后16MiB、节点/事件各4096、边65536、总包512MiB，超限失败。整产品物化另限制张量64MiB（不等于全部Python对象内存），大产品应逐测试读取。当前完整产品小于此上限。

分片前向仅在评价且不求梯度模式认证；后续训练可消费相同产品张量，但尚未验证梯度累积/检查点节省显存。旧训练命令尚未增加直接读取迁移包的命令行参数，可通过已有产品张量接口连接；04P不启动拟合。

## D. Linux迁移准备

标准包在`data/processed/graph_classifier/phase04P_v1/package/`，实际{package_bytes:,}字节（{mib(package_bytes)}MiB），含1份清单和75对张量/压缩元数据。元数据不损失内容。清单摘要：`{receipt['manifest_sha256']}`。包内绝对路径扫描为0，源路径在本机回执中保留反查；不把路径标记的散列冒充文件内容散列。

已实际复制到另一个根目录并禁止读取原始证据，75图全部通过。**实际操作系统仍为Windows，尚未在Linux运行。** 采集端重新验证了原源码、JVM与受限语义证据；迁移端独立核验可信摘要、字节内容、身份关联、张量/边对应、顺序及套件。迁移端没有重新执行Java，不能声称重新证明其执行语义。数据包不携带FP/TP目标；清单摘要须通过独立可信交接获得，散列不是数字签名。详见迁移说明。

## E. 模型准备结论与资源

{table(['检查项','状态'],[[k,d['status']] for k,d in statuses.items()])}

数据管线通过的范围是这75图及已认证支持语义。显卡环境未验收；正式监督分类缺样本条件；T1真实事件/结构可用于后续开发但尚无真实安全前缀入口；T2完整数值状态/输入输出不足；T3无合法干预执行对。不能把工程通过等同研究有效性。

{rtable}

内存是实际Python进程每10ms采样的驻留内存，包含已导入模型库，不是GPU显存，也不是操作系统精确全程峰值。外部监控的约5MiB只属于Windows虚拟环境启动器，未作为模型内存结论。最终六条验收命令耗时合计{resource['final_validation_command_seconds_sum']:.3f}秒；交互开发总耗时未连续测量。单元测试内部耗时{unit['seconds']:.3f}秒。没有超时、数据截断或OOM。

本阶段记录过一次缺少资源监控依赖、两次临时文件删除竞争造成的监控失败，已解决并保留记录；失败的部分运行不计作通过。新数据目录当前约{mib(resource['generated_directory_bytes_at_report'])}MiB（含保留的开发尝试和迁移副本），生成预算512MiB；报告时磁盘余量{mib(resource['disk_free_bytes_at_report'])}MiB，高于1GiB下限。不是复制整个原始轨迹库。

## F. 冻结划分与监督资格

{stable}

类别数量只复用03B4**授权训练目标入口**的已封存审计，并验证本次产品集合、准入及冻结划分摘要未变；04P未重新打开任何目标文件。验证/测试类别分布未知且不读取。各协议产品数和独立版本数由当前包重新计算，同版本没有拆分。数据正式准入5个不等于某一协议具备可开展实验的5个样本；三个协议均阻塞。

## G. 仍存问题与下一阶段建议

1. **最优先补齐冻结协议下的数据条件**：分组训练区只有1个FP、没有TP；留出银行无训练/验证，留出邮件无验证/测试。后续先审定标签盲的有界采集/语义计划，不按测试真值挑产品。收益上界尚未知，不能承诺新增多少合格产品。本阶段未重审其他交易路径或邮件语义。
2. 可准备Linux RTX3090Ti部署和相同数据包验收，核验驱动、依赖、清单、未训练输出及显存。当前没有GPU瓶颈的实测证据；不能据CPU内存保证训练显存或吞吐。即使显卡验收成功，仍不能越过样本分布阻碍开始正式分类训练。
3. 独立实现真实时间前缀生产器及可见时刻认证。保留的2858个值版本中仅1088个带真实值观察标记，其余缺失不能填零或预测后冒充真实值；这也不是全程序状态覆盖率。当前不足以训练完整数值状态世界模型或验证反事实。前缀契约只能检查生产者明确给出的时点，不能证明伪造时点真实。

本阶段到此停止，不执行04R、世界模型或后续实验。完整命令、指标、未知项说明见同名JSON和资源报告；原始输入及历史正式结果未覆盖。
''')
    write('resource_profile.md',f'''# 阶段04P实际资源记录

环境：{env}。复用既有Python环境，没有安装或升级依赖。未进行GPU训练、Java测试或Linux连接。

## 真实输入规模

{table(['产品','测试','压缩分片字节','批张量字节','单测试分批前向/秒'],[[p['identity'][1]+'/'+p['identity'][2],p['loaded_tests'],p['input_shard_bytes'],p['tensor_bytes'],f"{s['seconds']:.6f}"] for p,s in zip(v['products'],stream['products'])])}

批张量字节包括测试/产品归属索引，不包括Python元数据对象。最大单图{resource['max_graph_nodes']}节点、{resource['max_graph_edges']}边、{resource['max_graph_events']}事件；最大单个磁盘分片{resource['max_shard_bytes']:,}字节。

{rtable}

本体计时从模型库导入后开始；外部命令计时包含进程启动/导入。采样峰值包含实际Python模型进程，10ms间隔可能遗漏瞬时峰值，未测GPU。外部旧监控只看虚拟环境启动器，约5MiB不能代表运算内存。最终命令总和{resource['final_validation_command_seconds_sum']:.3f}秒；完整交互开发耗时未知。

## 实际命令与退出码

以下均以项目根为工作目录；完整参数数组在对应`.command.json`，不依赖此处缩写。

{table(['命令','退出码','秒'],[[' '.join(c['argv']),c['exit_code'],c['seconds']] for c in commands])}

`scripts/run_phase04P_checks.py`依次执行单元与原始内容核对；50项测试通过，75图内容核对通过。`verify_phase04P.py build`先检查冻结摘要和既有三协议准入；`verify`验证30次冻结参数对照；`stream_only`独立进程测量单图分批；`relocated`改变根目录并禁止原证据读取。

开发失败：`verify.log`保留缺少psutil导致的退出1；随后使用已安装的基础环境依赖。`unit_final.log`及`unit_final_v2.log`为两次监控竞争的部分输出，外层退出1且无完整命令报告；不能视为通过。修复仅在本阶段监控脚本中容忍本阶段临时目录已消失的文件；旧采集器源码未改。

## 磁盘及承载边界

标准包{package_bytes:,}字节（{mib(package_bytes)}MiB）；迁移后保留两份包最少{2*package_bytes:,}字节，不含代码/依赖/文件系统开销。建议为此包预留64MiB交接空间（容量预留，不是测量）；Python/显卡依赖空间另行检查。本次不复制整个原始Java库或日志。

预算：新增512MiB，磁盘至少余1GiB，每命令600秒，实际Python RSS1.5GiB，日志8MiB。每分片及解压元数据16MiB、清单4MiB、单图4096节点/4096事件/65536边；超限报错不裁剪。一次物化的64MiB只限制张量估算，不能作为元数据内存上限；大产品采用迭代接口并测量进程内存。

报告快照生成目录{resource['generated_directory_bytes_at_report']:,}字节；磁盘剩余{resource['disk_free_bytes_at_report']:,}字节。包含为保留开发记录而保留的两个尝试目录；正式包以当前清单摘要为准。未发生超时、OOM或真实数据截断。无法从本次CPU结果推算RTX3090Ti训练显存/吞吐。
''')
    write('linux_portability.md',f'''# 阶段04P Linux数据迁移说明

**已准备迁移包并完成Windows更换根目录读取；未运行Linux或GPU。** 不需要复制原始Java数据集以消费本包，但如需重新证明采集端语义，应另行取得对应原始证据与环境。

## 包结构与信任边界

`data/processed/graph_classifier/phase04P_v1/package/manifest.json`登记5产品、75测试、完整身份、原始套件登记证明、三种划分、特征版本`semantic_identity_v2`、格式版本`lossless_product_package_v1`、7类关系、提取器/核验代码摘要及来源摘要。

`graphs/000000.npz`等75个文件保存float32的22列特征/覆盖量、int64的边端点及关系；不启用任意对象反序列化。对应`.json.gz`无损保存全部节点/边记录、事件、调用帧/值版本、静态目录、框架效果、质量及采集端认证；读取时限制解压大小。身份、行号、名称均在审计字段，不作可学习数值身份。

源绝对路径替换为`external/<路径字符串散列>`审计标记；该标记不是内容摘要或可打开路径。本机`export_receipt.json`保存反查，不必迁往Linux。实际来源文件、日志及二进制内容摘要仍保留。路径扫描绝对路径数为0；不允许目录逃逸、盘符或跨测试共享分片。

可信清单SHA-256为：`{receipt['manifest_sha256']}`。必须从本交接报告或独立受控渠道传递；从可能被修改的包自行计算摘要后立刻当作可信输入，不能证明来源。散列不替代数字签名。

采集端重验了旧图严格准入，包括真实运行证据。迁移端验证内容、张量与记录一致性、事件/静态身份、边端点、先定义后使用和完整套件，不重放Java。模型输入是已认证观察的确定性编码，没有预测值补全。包内不含FP/TP目标，原始PASS结果只是测试证据；未来监督目标仍走独立授权接口。

## 推荐环境与待执行步骤

优先沿用Python3.11、PyTorch2.6.0，减少与本机已验证环境的差异；CPU验证使用了NumPy{env['numpy']}。本实现直接使用PyTorch，无需安装图学习扩展库。PyTorch官方历史版本页面提供2.6.0的Linux CUDA12.6安装源，推荐先在隔离环境固定版本并完成CPU复验，再检查显卡驱动兼容性。[官方安装说明](https://pytorch.org/get-started/previous-versions/)

CUDA12系列在驱动兼容性上存在最低版本及功能限制，须以服务器实际驱动与所选运行库组合验收，不能只因显卡是3090Ti就标记通过。[NVIDIA官方兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

以下是供下一阶段使用的命令，**本阶段未在服务器执行，也未安装这些依赖**：

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install numpy=={env['numpy']} psutil==5.9.0
# 将本项目src/和可信package/分别复制到目标机器；不需要标签或原始Java库。
export PYTHONPATH="$PWD/src"
```

可在任意目录直接调用已有接口（`package`替换为实际相对/绝对包根）：

```python
import torch
from cwm_fp.models.graph_package import ProductPackage, forward_stream
from cwm_fp.models.graph_classifier import GraphClassifier
package = ProductPackage('package', expected_manifest_sha256='{receipt['manifest_sha256']}')
torch.manual_seed(20261101)
torch.set_num_threads(1)
model = GraphClassifier('rgcn', hidden=16, layers=2).eval()
assert sum(len(p['registered_tests']) for p in package.products) == 75
for product in package.products:
    assert len(list(package.iter_tests(product))) == len(product['registered_tests'])
    score = forward_stream(model, package, product, micro_tests=1)
    assert torch.isfinite(score).all()
# 这里只是未训练前向；不输出分类精度、不读取目标、不反向传播。
```

先核对CPU结果和资源，再另行记录`nvidia-smi`、`torch.cuda.is_available()`、实际设备及驱动，进行同样的未训练显卡对照并测量峰值。带梯度训练和训练吞吐不在已验收范围。包{package_bytes:,}字节，两份包最少{2*package_bytes:,}字节，依赖空间另计。

Windows迁移验证证据：`data/processed/graph_classifier/phase04P_v1/relocation.json`，75图/{totals['events']}事件通过，原始证据读取被禁止。所有Linux执行/显存指标目前为未知，不能写0代表测量。
''')
    write('world_model_interface.md',f'''# 阶段04P时间前缀接口与状态观测边界

当前完整图用于执行结束后的产品分类，**没有可直接训练的真实时间前缀数据集**。本阶段只新增游标及契约检查，未建立状态预测或因果世界模型。

## 已保留的真实证据

75条测试轨迹共{totals['events']}个产品/测试事件，事件顺序、不同动态实例、静态指令关联、调用帧、定义/使用和值版本均与冻结原记录一致。框架调用效果保留物理调用/返回位置及支持范围。{integrity['counts']['value_nodes']}个值节点中{integrity['counts']['observed_value_nodes']}个标记实际已观测，{integrity['counts']['value_nodes']-integrity['counts']['observed_value_nodes']}个没有真实值；此比例只覆盖值节点，不代表完整程序状态覆盖率。69个断言节点不是69个完整断言输入输出记录，也不能据此推断其余测试没有断言。

原图的真实值与缺失标记被完整保存，不填零，不以模型预测冒充实测。没有新的数值回调、输入输出或状态快照采集；T2仍不足。没有合法干预前后对；插桩对照不是反事实干预。

## 预留接口

`PrefixCursor(test_identity, completed_event)`使用完整系统/版本/产品/测试四元身份，游标表示已经完成观测的事件，-1表示尚无动态事件。`validate_prefix(cursor, records)`验证记录由独立受信观测器提供，不自动裁切事后图。

静态记录`evidence_kind=static`仅包含执行前可知结构；不得携带全轨迹`executed`、实际分支或状态标志。动态记录`evidence_kind=observed`必须给出整数`available_after_event`且不晚于游标；事件、使用、返回和所依赖事件不能晚于记录可见时刻。负数、布尔伪整数、未知时刻、未来返回及模型推断来源均拒绝。

若有`value`字段，必须显式提供`observed`布尔掩码；缺失值保持null，真实观测不能标注null后冒充有效状态。未知值类型/不可观测引用须记录缺失原因，不能统一补成0。

所有最终结果、断言通过标志、后验、质量结论、完整覆盖统计和标签不得出现在前缀输入，递归检查嵌套字段。监督下一步真实值必须单独放在目标侧，不能混入当前输入。每条动态依赖最早在实际使用被观察后可用；异常引用只能在真实处理器可见时刻使用；调用效果至少在真实返回后才能生效。

## 尚未实现及不能声称的能力

目前只保留旧证据中**已存在**的时间索引/返回边界，未为缺少时点的状态、断言或外部调用猜测可见时间。旧`event_index`可能标记指令进入，数值回调可能发生在执行之后，不能机械把二者等同；需生产者明确事件边界。缺乏可见时点的数据先排除。

契约检查通过只能说明字段自洽，无法证明提供者没有伪造时间。尚缺真实观测器到契约的适配、状态版本与可见时点认证、前缀对真实执行的对照及跨线程先后关系。不能用完整图执行标记、后续依赖及终局结果切片伪装已验证前缀。

人工契约用例覆盖缺失值、未来事件/使用/返回、跨测试、静态字段污染、终局信息、伪造缺失值掩码等。真实75图仅验证保留证据，没有生成前缀训练样本。结论：T1结构证据部分具备，T2不足，T3不存在；目前不能启动数值状态或反事实模型训练。
''')
    write('summary.md','# 阶段04P完整总结\n\n独立完整的中文总结以[研究交接报告](phase04P_review_packet.md)为准；本文件仅保留统一入口，避免重复维护多份叙述。\n')
    (OUT/'report_metrics.json').write_text(json.dumps({'written_at_utc':datetime.now(timezone.utc).isoformat(),'package_bytes':package_bytes,'max_error':max_error,'tests':unit['tests_run'],'totals':dict(totals)},indent=2),encoding='utf-8')
    print('REPORTS WRITTEN',package_bytes,dict(totals),unit['tests_run'])

if __name__=='__main__':main()
