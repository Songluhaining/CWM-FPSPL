"""Generate the single Chinese handoff from completed container evidence."""
import argparse,csv,json,os,platform,resource,shutil,sys,time
from collections import Counter,defaultdict
from pathlib import Path
from cwm_fp.data.common import sha256,write_json

def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def lines(p):
    with Path(p).open(encoding='utf8') as f:return [json.loads(s) for s in f]
def csvout(p,rows):
    with p.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def size(root):return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    if not Path('/.dockerenv').exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Container CPU only')
    out=root/'outputs'
    if out.exists():raise ValueError('Do not overwrite handoff')
    out.mkdir();prod=root/'production_code_v2';verify=root/'verification_code_v3'
    checks=load(prod/'checks.json');independent=load(verify/'verification.json');comparison=load(verify/'bank_email_comparison.json')
    if not independent['passed']:raise ValueError('Independent verification failed')
    traces=load(prod/'trace_status.json');states=lines(prod/'evidence.jsonl');first=load(verify/'first_gap_evidence.json')
    isolation=load(prod/'isolation_checks.json');protocol=load(root/'inputs/protocol.json');manifest=load(root/'inputs/source_manifest.json')
    old=load(root/'production_code_v1/checks.json');oldtraces=load(root/'production_code_v1/trace_status.json')
    products=[];graphrows=[];blockers=defaultdict(list);nums=Counter();perproduct=defaultdict(list)
    for t in traces:perproduct[tuple(t['identity'][:3])].append(t)
    for p in protocol['selected']:
        ident=load(prod/(p['batch_product']+'_identity.json'));ts=perproduct[tuple(p['identity'])]
        coll=load(root/'inputs/email'/p['batch_product']/'collection.json')
        row={**dict(zip(('system_id','bug_version_id','product_id'),p['identity'])),'partition':'grouped/train',
            'registered_original_tests':p['registered_tests'],'historical_completed_tests':sum(r['status']['complete'] for r in coll['records']),
            'historical_pair_consistent_tests':sum(r['status']['parity'] for r in coll['records']),
            'selected_traces':len(ts),'source_verified_traces':sum(t['source_execution_verified'] for t in ts),
            'events':sum(t['physical_instruction_events'] for t in ts),'product_events':sum(t['product_events'] for t in ts),
            'trusted_prefix_events':sum(t['trusted_event_prefix'] for t in ts),'unresolved_product_static_instructions':ident['unresolved_product_instructions'],
            'new_local_transitions_admitted':sum(t['local_admitted'] for t in ts),'full_product_admission':False,
            'full_product_admission_changed':False,'rejection_codes':'UNRESOLVED_STATIC_IDENTITY|INCOMPLETE_SUPPORTED_DYNAMIC_SEMANTICS'+('|ORIGINAL_SUITE_FAIL' if p['observed_result']=='FAIL' else ''),
            'evidence':'production_code_v2/'+p['batch_product']+'_identity.json;production_code_v2/trace_status.json;inputs/email/'+p['batch_product']+'/collection.json'}
        products.append(row)
    for t in traces:
        g={**dict(zip(('system_id','bug_version_id','product_id','test_id'),t['identity'])),'partition':'grouped/train',
           'original_result':t['original_result'],'instrumented_result':t['instrumented_result'],'source_execution_verified':t['source_execution_verified'],
           'raw_events':t['physical_instruction_events'],'identity_verified_events':t['identity_verified_events'],'trusted_prefix_events':t['trusted_event_prefix'],
           'local_transitions_admitted':t['local_admitted'],'full_graph_identity_complete':False,'full_graph_semantics_revalidated':None,
           'full_graph_revalidation_reason':'本阶段只验收局部前缀，没有重建或重验完整图',
           'full_product_admission':False,'rejection_codes':t['trusted_until']['reason'] if t['trusted_until'] else 'NO_SUPPORTED_PRODUCT_BRANCH_ARITHMETIC_QUERY',
           'evidence':f"production_code_v2/traces/{t['trace_index']:03d}/status.json",'source_log':'inputs/'+t['source_log']}
        graphrows.append(g)
    for r in first:
        raw=r['raw_record'];key=(r['barrier']['reason'],raw['opcode'] if raw else None)
        blockers[key].append(r)
    code=Path(__file__).resolve().parents[1]
    changed=['src/cwm_fp/lddg/static.py','src/cwm_fp/models/local_dataset.py','tests/test_local_dataset.py',
        'scripts/phase05C2_export.py','scripts/phase05C2_snapshot.py','scripts/phase05C2_produce.py','scripts/phase05C2_verify.py',
        'scripts/phase05C2_regressions.py','scripts/phase05C2_run.sh','scripts/phase05C2_report.py','DATA_CONTRACTS.md','STAGES.md','outputs/phase_status.md']
    commands=[load(f) for f in sorted((root/'command_logs').glob('*.json'))]
    metrics={'versions':1,'products':len(products),'traces':len(traces),'original_suite_registered_tests':sum(p['registered_tests'] for p in protocol['selected']),
        'paired_execution_verified':sum(t['source_execution_verified'] for t in traces),'events':sum(t['physical_instruction_events'] for t in traces),
        'product_events':sum(t['product_events'] for t in traces),'test_context_events':sum(t['physical_instruction_events']-t['product_events'] for t in traces),
        'identity_verified_events':checks['identity_links']['verified'],'unresolved_event_identity':independent['unresolved_dynamic_events'],
        'trusted_prefix_events':sum(t['trusted_event_prefix'] for t in traces),'trusted_product_prefix_events':sum(r['scope']=='product' and r['trusted_prefix_available'] for r in states),
        'traces_without_observed_prefix_gap':sum(t['trusted_until'] is None for t in traces),'first_gap_trace_count':sum(t['trusted_until'] is not None for t in traces),
        'product_branch_candidates':checks['summary']['product']['branch']['denominator'],'product_arithmetic_candidates':checks['summary']['product']['arithmetic']['denominator'],
        'product_other_candidates':checks['summary']['product']['other']['denominator'],'local_transitions_admitted':sum(t['local_admitted'] for t in traces),
        'new_complete_classifier_products':0,'new_Java_runs':0,'neural_updates':0,'new_full_graphs':0,
        'graph_nodes':None,'graph_edges':None,'graph_count_reason':'本阶段生产局部状态侧表，不重建或重计完整图；原图未修改。'}
    unresolved=[]
    for (reason,op),rr in blockers.items():
        unresolved.append({'name':reason,'opcode':op,'confirmed_root':True,'affected_traces':len(rr),
            'affected_products':[list(x) for x in sorted({tuple(r['identity'][:3]) for r in rr})],
            'evidence':'verification_code_v3/first_gap_evidence.json','example':rr[0],
            'implemented_repair':False,'reliable_next_step':'补齐实际初始定义/值/时刻证据' if reason=='field_read_without_observed_definition' else '独立认证该语义；本阶段不扩展解释器',
            'complete_product_recovery_upper_bound':len({tuple(r['identity'][:3]) for r in rr}),
            'upper_bound_note':'仅解除这一阻碍的估计上界，其他身份、数组、库效果或原始失败仍会阻断；不是已恢复数量。',
            'state_world_model_impact':'越过此边界会把未知堆/控制/副作用当作已知，导致错误状态监督。'})
    unresolved.extend([
        {'name':'unresolved_program_identity','static_product_instructions':sum(p['unresolved_product_static_instructions'] for p in products),
         'dynamic_events':independent['unresolved_dynamic_events'],'evidence':'production_code_v2/p007_identity.json;production_code_v2/p008_identity.json;verification_code_v3/unresolved_event_identities.json',
         'repair_implemented':False,'note':'原始身份逐项保持，未知不能通过改路径被提升；是完整图准入的独立阻碍。'},
        {'name':'narrow_development_scope','versions':1,'traces':24,'evidence':'inputs/protocol.json;inputs/development_exposure.json',
         'note':'只暴露原grouped/train版本8；不能声称跨版本或跨系统泛化。验证/测试轨迹未展开。'},
        {'name':'no_verified_local_training_targets','count':metrics['local_transitions_admitted'],'evidence':'production_code_v2/targets.jsonl',
         'note':'物理后缀有值或有分支后继不等于可信前缀内的转换目标；不据此训练。'}])
    flags={'SERVER_ENVIRONMENT_REUSED':True,'EMAIL_SOURCE_EVIDENCE_VERIFIED':all(t['source_execution_verified'] for t in traces),
        'EMAIL_TRUSTED_PREFIXES_GENERATED':metrics['trusted_prefix_events']>0,'EMAIL_PRODUCT_LOCAL_STATES_GENERATED':metrics['product_events']>0,
        'LOCAL_TARGETS_VERIFIED':metrics['local_transitions_admitted']>0,'PREFIX_ISOLATION_CHECKED':bool(isolation) and all(r['passed'] for r in isolation),
        'CROSS_SYSTEM_INPUT_CONTRACT_COMPATIBLE':independent['cross_system_contract_rows_checked']==len(states)+comparison['bank_prior_rows'],
        'FULL_PRODUCT_ADMISSION_CHANGED':False,'NEURAL_MODEL_TRAINED':False,'CROSS_SYSTEM_GENERALIZATION_EVALUATED':False,'CAUSAL_INTERVENTION_VALIDATED':False}
    researchbytes=size(root)-size(root/'inputs')-(root/'inputs_v1.tar.gz').stat().st_size-size(root/'environment')
    resources={'experiment_commands_before_report':commands,'experiment_seconds_before_report':sum(r['seconds'] for r in commands),
        'peak_sampled_process_rss_bytes':max(r['sampled_peak_process_rss_bytes'] for r in commands),
        'research_bytes_before_reports':researchbytes,'input_bytes':manifest['bytes'],'compressed_input_archive_bytes':(root/'inputs_v1.tar.gz').stat().st_size,
        'free_bytes':shutil.disk_usage(root).free,'threads_limit':2,'gpu_used':False,'gpu_memory_bytes':None,'gpu_memory_reason':'本阶段禁用GPU，无训练或GPU验收',
        'new_environment_install_bytes':0,'environment_reused_from':str(root.parent/'phase05C1_v1/environment/venv'),
        'deployment_total_seconds':None,'deployment_total_seconds_reason':'传输与人工代码准备未统一计时，不能用实验秒数代替全部耗时。',
        'python_path':sys.executable,'python_version':sys.version,'platform':platform.platform(),'container_id':platform.node(),
        'container_name':'wanghaining_ubuntu_ssh','image':'wanghaining_ubuntu_env:20260304','host_environment_changed':False,
        'monitor':'0.5秒主进程内存/剩余磁盘轮询；非系统硬配额，不声称所有子进程瞬时峰值已测量。',
        'timeout':any(r['termination_reason'] for r in commands),'oom':False,'final_raw_trace_truncated':False,
        'first_attempt_format_rejections':sum(t['parse_error'] is not None for t in oldtraces)}
    artifacts={name:'outputs/'+name for name in ['phase05C2_protocol.json','phase05C2_review_packet.md','phase05C2_review_packet.json',
        'phase05C2_trace_status.csv','phase05C2_local_state_status.csv','phase05C2_checks.json','phase05C2_product_status.csv','phase05C2_graph_status.csv','phase05C2_summary.md']}
    packet={'stage_id':'05C2','stage_status':'COMPLETED_WITH_BLOCKED_LOCAL_TRAINING_ADMISSION','actual_scope':protocol['selected'],'flags':flags,
        'before_metrics':{'previous_Email_local_transition_dataset':None,'reason':'05C1只诊断Bank，Email本次是首次独立局部数据生产',
            'first_attempt_parsed_events':old['identity_links']['events'],'first_attempt_parse_rejected_traces':sum(t['parse_error'] is not None for t in oldtraces),
            'first_attempt_complete_denominator_verified':False,'existing_Email_classifier_admitted_products':0},
        'after_metrics':metrics,'recovered_graphs':{'new_full_graphs':0,'historical16':None,'reason':'本阶段不重建原图或重审银行16图'},
        'admitted_products':{'new':0,'selected_Email_full_classifier_products':0,'local_transition_samples':metrics['local_transitions_admitted']},
        'rejection_reasons':checks['first_trust_gaps'],'unresolved_issues':unresolved,
        'split_readiness':{'status':'BLOCKED','grouped':{'accessed_partition':'train','selected_products':2,'independent_versions':1,
             'Email_local_admitted_train':0,'val':None,'test':None,'reason':'本轮不展开Email验证/测试，也不读取任何FP/TP目标。'},
             'loso_BankAccountTP':{'status':'NOT_EVALUATED','train':None,'val':None,'test':None,'reason':'未按其他协议读取轨迹/标签或重审全量产品。'},
             'loso_Email':{'status':'NOT_EVALUATED','train':None,'val':None,'test':None,'reason':'版本8已作开发访问，不能再称其为盲测系统证据；原划分未改。'},
             'historical_Bank_grouped_classifier_queue':{'train_products':17,'train_versions':9,'val_products':5,'val_versions':4,'test_products':6,'test_versions':3,
                  'source':'05C1/05B既有交接，未重新读取标签或重验产品，不是本阶段新增成果'},
             'train_class_coverage':None,'train_class_coverage_reason':'本阶段禁止读取FP/TP目标，未统计类别。'},
        'world_model_readiness':{'T1':{'status':'PARTIAL','reason':'194条可信前缀事件，有限顺序/依赖可审计；整条Email轨迹未认证。'},
             'T2':{'status':'BLOCKED','reason':'67个产品分支/算术候选均无可信输入目标对；只有少量其他位置标量。'},
             'T3':{'status':'BLOCKED','real_intervention_pairs':0},'prefix_isolation':checks['checks'],
             'algorithm_effectiveness_evaluated':False},
        'tests_executed':{'unique_unit_and_regression_tests_passed':53,'new_tests_passed':15,'old_prefix_state_reader_tests_passed':38,
             'test_executions':103,'failed_test_executions':0,'real_Java_new_runs':0,'frozen_real_trace_replays':24,
             'independent_raw_trace_order_checks':24,'cross_system_input_contract_rows':independent['cross_system_contract_rows_checked'],
             'initial_format_attempt_retained':True},
        'resources':resources,'changed_files':changed,'artifact_paths':artifacts,'checks':checks,'independent_verification':independent,
        'cross_system_comparison':comparison,'recommended_next_stage':{'priority':1,'task':'最小必要初始字段/堆状态观测补充与独立认证',
             'reason':'缺少定义的字段读取是15/24条轨迹的首个阻碍，全部46分支和19/21算术位置位于这些边界之后。',
             'authorization_required':'下一阶段单独授权有界Java成对重跑/观测扩充；当前不执行。','gpu_training_ready':False}}
    write_json(out/'phase05C2_review_packet.json',packet)
    write_json(out/'phase05C2_checks.json',{'production':checks,'independent':independent,'comparison':comparison,'flags':flags})
    shutil.copyfile(root/'inputs/protocol.json',out/'phase05C2_protocol.json')
    for source_name,public_name in [('trace_status.csv','phase05C2_trace_status.csv'),('local_state_status.csv','phase05C2_local_state_status.csv')]:
        with (prod/source_name).open(encoding='utf8') as f:exported=list(csv.DictReader(f))
        for row in exported:
            row['evidence']='production_code_v2/'+row['evidence']
            for key in ('source_log','source_status'):
                if key in row:row[key]='inputs/'+row[key]
        csvout(out/public_name,exported)
    csvout(out/'phase05C2_product_status.csv',products);csvout(out/'phase05C2_graph_status.csv',graphrows)
    producttable='\n'.join(f"| {p['product_id']} | {p['registered_original_tests']} | {p['selected_traces']} | {p['events']} | {p['product_events']} | {p['trusted_prefix_events']} | {p['new_local_transitions_admitted']} |" for p in products)
    blocktable='\n'.join(f"| {reason} | {op} | {len(rr)} | {rr[0]['identity'][2]}/{rr[0]['identity'][3]} |" for (reason,op),rr in blockers.items())
    bcmp=comparison['bank'];ecmp=comparison['email'];overlap=comparison['product_branch_arithmetic_exact_input_overlap']
    text=f'''# CWM-FP 阶段05C-2：Email有限机制数据接入与局部状态可观测性验收

2026年11月2日。本阶段已完成真实服务器容器中的有界重放、局部状态生产、独立核验与交接。**工程接入完成；Email局部转换训练准入仍被阻塞。** 不是只提出计划，也没有因零准入而替换样本或越过未知语义。

固定范围是Email原按缺陷版本分组协议的训练区：版本`_MultipleBugs_.NOB_1.ID_8`，两个产品各12条既有轨迹，共24条。实际完整保留{metrics['events']}条指令，其中产品代码{metrics['product_events']}条、测试上下文{metrics['test_context_events']}条。产品代码有46个分支、21个算术候选，**可准入转换为0/67**。没有新增完整分类产品，没有运行Java、神经模型、优化器更新、GPU计算或真实干预。

## 1. 服务器与执行位置

真实SSH连接指定服务器的标准22端口，使用已有密钥并严格核验既有主机记录；未读取或保存密码，未更改SSH配置或权限。复用05C1保存的管理员指南正文及取得回执（`D:/dataset/whn/cwm_fp_generated/data/processed/local_state/phase05C1_v1/guide/receipt.json`）。没有声称本次重新登录语雀取得新版本；继续遵守个人容器内执行、宿主机不安装环境的约束。

指定容器`wanghaining_ubuntu_ssh`本次已在运行，镜像`wanghaining_ubuntu_env:20260304`，容器ID`81e96c2ae461e8a5acbe7820d5dd36eb0539a367ba51c31368c7446490395473`。未启动/重启/重建容器，未改挂载、驱动、Docker服务、权限或清理历史文件。复用Python3.11.13环境：`/workspace/codes/wanghaining_cwm_fp/phase05C1_v1/environment/venv/bin/python`。PyTorch2.6.0+cu118、NumPy2.4.6为05C1已锁定环境，本次未安装依赖。所有重放、统计、张量读取和测试均在容器内；Windows只做文件审阅、冻结文件选择、打包传输和报告副本。

| 位置 | 本次真实路径 |
|---|---|
| Windows映射目录 | `X:/codes/wanghaining_cwm_fp/phase05C2_v1` |
| 宿主机项目目录 | `/sdc/wanghaining/codes/wanghaining_cwm_fp/phase05C2_v1` |
| 容器项目目录 | `/workspace/codes/wanghaining_cwm_fp/phase05C2_v1` |
| 容器只读输入 | 上述目录下`inputs/` |
| 容器结果 | `production_code_v2/`、`verification_code_v3/`、`outputs/` |

三处同一个非空探针文件摘要均为`0af604f7cb1c21a7c02ae551390a46e51625ef4c5c5e462c0e6625fedb68e10e`。X盘实际为用户个人目录的远程文件映射。没有重试直接写权限；文件经已成功的SSH传输路径落盘，再通过X盘只读核对。环境临时目录和缓存仍位于挂载盘，未使用已满容器根分区安装环境。

## 2. 冻结选择、访问边界与来源核验

种子20261101，命名空间05C2，按完整版本、产品、测试身份的SHA256顺序选择；名单在打开局部状态结果前冻结。原批次只有一个符合原分组训练区的Email版本和两个产品，实际选24条，没有凑足72条、补选容易解释的位置或切换留出协议。产品0011原始套件失败，0014原始套件通过；这是可观测测试结果，不是假通过/真通过标签。一个被选测试原始失败，仍完整保留。

| 产品（同一版本8） | 原登记测试 | 本次选定 | 指令总数 | 产品指令 | 可信前缀事件 | 合格转换 |
|---|---:|---:|---:|---:|---:|---:|
{producttable}

24对原始/插桩完成结果和旧运行记录已重新核对，全部一致；没有把原始测试失败当作采集失败。两套原套件共108条登记测试，**24条局部重放不等于重新采集完整108测试**。历史采集完成性与完整图拒绝仍独立保留。没有读取FP/TP、缺陷位置、变异日志、其他协议标签或Email验证/测试轨迹。

输入包690文件、{manifest['bytes']:,}字节，压缩传输{resources['compressed_input_archive_bytes']:,}字节。只包含已选日志、对照/完成记录、类文件、源码与静态身份依据、公开划分和必要元数据；原Java运行库/代理的既有摘要在文件导出时复核并记录，未为此搬运完整原数据集、BRFL或Windows环境。来源清单摘要`{checks['source_manifest_sha256']}`；协议摘要`{checks['protocol_sha256']}`。全部原始文件保持不变，服务器结束核验通过。旧绝对路径作为审计字段保留，独立相对路径映射负责实际文件读取。

本次开发访问明确登记版本8。原批次的版本185和196在本次没有展开轨迹；未声称其历史上从未被任何阶段查看。未来不能把已开发访问的版本8重新宣传为未见系统盲测证据，原冻结划分没有修改。证据：`inputs/development_exposure.json`、`inputs/protocol.json`、`production_code_v2/source_access.json`。

## 3. 实际实现与验证边界

- `lddg/static.py`仅增加受来源清单约束的文件路径解析入口，并兼容Windows来源路径的文件名。原字节码定位、类摘要和身份规则不改。两个产品原静态身份逐项重建后与冻结记录完全一致；未知身份未被自动提升。
- `models/local_dataset.py`复用原解释器、执行前捕获和观察器，新增可迁移来源读取、独立局部准入、首个缺口永久拒绝、输入/目标分离和整数精度保留。真实类别、方法、行号、偏移、版本、产品与帧编号只在审计中连接，不进入可学习数值。
- 首轮真实生产发现静态字段读取的完成标记`###RET@opcode=178(getstatic)`被旧观察器当作无对应调用。原`TraceTransformer.java`及`GetStaticInst.java`证明这是插桩器既有格式。本次仅在新局部适配器识别它；不增加栈值、不构造初始字段定义、不当成方法返回、不认证类初始化副作用。原观察器和冻结Bank输入不被重解释。
- 新入口最初把退出码1一律归为执行不完整；已按原配对运行器明确接受完整失败测试的既有规则纠正，同时保留完成文件、异常消息、截断和插桩失败检查。没有放宽结果一致性。
- 新增阶段文件选择/快照/生产/独立核验/报告脚本及15项专项测试。复用旧监测器和38项前缀、局部状态、读取回归；没有重新开发通用JVM、数组、集合、反射或异常解释器。

原首轮解析记录为{old['identity_links']['events']}条，有{sum(t['parse_error'] is not None for t in oldtraces)}条轨迹遇到格式拒绝，不能作为完整数据分母。修正版完整读取3325条；旧解析器独立核对24条日志的全部指令数量、操作码和行序一致。修正读取的是**原来已有的指令**，没有产生新Java事件。首轮`production_code_v1/`及代码/日志全部保留。

真实Email验证覆盖来源迁移、完整读取、逐事件身份、可信前缀、永久拒绝和格式读取。人工记录测试另外验证了错误返回标记、重复读取、整数精度、空引用与零的区别、未知值掩码、未来目标、非法身份字段和后缀扰动。没有真实Email合格分支/算术对，故**没有声称这两个任务已经在Email上通过符号语义一致性认证**。

## 4. 全分母、来源类别与首个阻碍

| 作用域与任务 | 完整候选分母 | 完整可信输入 | 可信目标 | 合格转换 | 覆盖内一致率 |
|---|---:|---:|---:|---:|---|
| 产品条件分支 | 46 | 0 | 0 | 0 | 未定义：没有可核验对 |
| 产品局部算术 | 21 | 0 | 0 | 0 | 未定义：没有可核验对 |
| 产品其他指令 | 2127 | 0（完整标量操作数组） | 不启用此任务 | 0 | 不适用 |
| 测试代码算术/转换 | 36 | 1 | 0 | 0 | 未定义；含未认证i2l |
| 测试其他指令 | 1095 | 不作为产品机制覆盖 | 不启用此任务 | 0 | 不适用 |

产品分支精确操作码：ifeq25、ifne6、if_icmpge10、ifnull4、ifnonnull1；21个产品算术均为iadd。产品46个分支的56个操作数槽、21个算术的42个槽全部保持未知，因为在首个可信边界之后。产品其他位置有9个直接观测操作数槽，但每个对应完整输入仍有未知部分，不是新增分支/算术监督对。测试上下文另有1个直接标量、3个静态常量和1个有限复制推导槽，分别单列，不用这些常量或测试模板夸大业务状态覆盖。

总计{metrics['trusted_prefix_events']}条事件处于可信前缀，其中产品{metrics['trusted_product_prefix_events']}条。四条轨迹未检出首个缺口，但没有受支持的产品分支/算术查询；其余20条保留最早证据：

| 首个原因代码 | 实际操作码 | 影响轨迹 | 一个实际测试证据 |
|---|---:|---:|---|
{blocktable}

每条轨迹的实际起点、可信截止时钟、原指令字段和原因位于`verification_code_v3/first_gap_evidence.json`。`field_read_without_observed_definition`表示当前字段读取缺此前可信写入/初始定义，不能从后续结果倒推它；回调值也不能证明未知当前堆未被修改。合并的数组/类型原因必须结合上表实际操作码解释，不能一律称为数组。库调用真实返回不能证明无副作用。

全部3325事件中3112个静态身份连接可信，213个未认证。两个产品共有39条原静态身份缺口，本阶段没有修复；未执行的未知静态位置不单独排除其他已认证局部前缀，但一旦执行到未知元素即保守拒绝。身份侧表不是动态控制依赖，也不是因果图。完整产品身份与依赖门槛继续保留。

解决字段缺口理论上最多解除15条轨迹的**首个阻碍**，不能宣称获得15条完整合格轨迹或两个完整产品。后面仍有未知构造器身份、数组/类型操作、集合或库效果，且一个产品原始失败。本阶段实际恢复的完整产品数量明确为0。

## 5. 时间隔离、独立目标与读取验收

输入在当前指令消费操作数前捕获；保留精确32位整数、类型、值版本、来源及可见时刻。当前/未来回调、真实分支结果、最终测试结果或后验边不进入输入；复制结果必须等真实完成记录到达。未知值为空并有独立掩码，不是实测0。目标在输入全部捕获后独立连接，跨越可信边界的目标关闭，不生成默认后继标签。

已执行{checks['checks']['truncated_replays']}个位置的独立截断重放及{checks['checks']['suffix_mutations']}次后缀扰动，覆盖每条轨迹首/中/末位置和全部实际调用、返回、分支与处理器关键边界。比较不是用整文件摘要拒绝替代：删除后缀、修改未来值/返回/异常/终局、追加伪后缀都只用于隔离测试副本。全部通过。合格转换为0，因此“全部合格转换额外截断核验”没有样本，不能把空集合写成成功的数值转换认证。已有可见操作数时刻检查为{checks['checks']['operand_time_checks']}个槽；详见隔离明细，未知边界之后的检查主要验证拒绝和掩码不被后缀改变。

3325行实际输入经另一根目录读取，目标文件根本不存在仍可成功构造输入，证明输入读取器不打开目标。与冻结Bank1070行局部状态共同核对，共{independent['cross_system_contract_rows_checked']}行通过相同字段/精确整数/来源掩码接口。目标、审计身份、原始路径不能通过字段白名单进入张量。没有改变旧22列图输入或关系，没有重新拟合归一化，也没有启动模型。

## 6. 与Bank既有诊断的只读比较

Bank既有05C1范围为22产品、13版本、368训练/验证轨迹，独立诊断1070行；其中产品查询{bcmp['product']['rows']}、测试查询{bcmp['test']['rows']}。Email本次1版本、2产品、24轨迹，产品分支/算术67个，其他产品指令2127个。两者分母不同：Bank统计的是原冻结有效分支/算术配对，Email统计所选轨迹内全部候选，不能直接比较百分比后声称系统更难或算法退化。

Email引入了真实不同代码、字段初态、引用和库行为，但尚未提供可信的数值分支/算术训练对。相同局部字段签名中，Bank产品查询有{overlap['bank_unique']}种、Email67候选有{overlap['email_unique']}种，两系统共有{overlap['intersection']}种；Email有{overlap['email_rows_already_seen_in_bank']}行输入与Bank产品诊断完全相同。没有用产品编号增加“多样性”。原始位置不相同不等于机制新颖。

可信Email目标为空，因此可信输入多目标冲突无法评价。另独立保留真实后缀的未认证分支指示：28次跳转、18次顺序后继；67个候选共有6种局部掩码输入，其中4组对应多个后缀结果，涉及60个候选。它们不是已认证目标，也不是随机程序行为的证据；完整记录在`verification_code_v3/bank_email_comparison.json`及`unverified_suffix_observations.json`。两系统签名不重叠主要包含Email缺值掩码的影响，不能据此认定新机制已经被覆盖。精确操作码不能弥补缺失初始状态，不能把零核验率写成一致率100%。本轮没有跨系统神经泛化成绩或因果干预结论。

## 7. 测试、资源与可复现性

最终不同测试共53项通过：新专项15项、原局部状态12项、原前缀20项、原读取6项。两版合计103次测试执行均通过；初版的真实格式拒绝和误分类不因此消失，已独立保留。真实Java新增0次；本轮重放冻结24对轨迹，不是重新验证虚拟机执行等价性。完整来源690文件、代码快照和原始事件分母结束核验通过。

主要实际命令为`bash code_v1/scripts/phase05C2_run.sh tests code_v1`、`regressions code_v1`、`produce code_v1`，修正版对应`code_v2`三个动作，独立核验为`bash code_v3/scripts/phase05C2_run.sh verify code_v3`。完整展开后的Python路径、参数、进程号、退出码、秒数、内存及日志摘要在`command_logs/*.json`；当前已完成这些命令均退出0。首轮逻辑不完整不能被退出0掩盖。

报告生成前实验命令累计{resources['experiment_seconds_before_report']:.3f}秒；其后报告生成命令另有回执，不把此小计当全部人工部署耗时。修正版生产约471.919秒；主要耗时为独立重放和重复的静态结构核验，没有GPU加速要求。实测主进程峰值约{resources['peak_sampled_process_rss_bytes']/1024**2:.2f}MiB；研究结果/代码/日志在报告生成前约{researchbytes/1024**2:.2f}MiB，另有约{manifest['bytes']/1024**2:.2f}MiB迁移输入，环境新增安装0字节。剩余磁盘约{resources['free_bytes']/1024**3:.1f}GiB。GPU显存为未测量/未使用，不填伪训练结果。

未发生实验超时、OOM、原始日志截断或缓存失效。首轮解析早停是格式适配缺陷，已经与物理数据截断分开；原始日志始终完整。监测并非系统硬配额，未独立测所有短暂子进程峰值。代码快照、输入清单、固定协议、所有尝试及失败依据可复核；复现需新目录，不能覆盖冻结结果。

## 8. 研究准入与唯一优先建议

**Email局部监督转换训练：阻塞（BLOCKED）。跨系统输入格式：兼容；跨系统研究有效性：未评价。** 当前不能用这67个未知候选启动双系统状态训练，更不能据此宣布Email完整分类产品合格。既有Bank分组队列17训练/5验证/6测试产品仅继承历史交接，本阶段不重读其FP/TP标签、不修改类别或准入。另两套留出协议没有执行；其新统计是未测量，不用0冒充全项目统计。

结构层面（T1）已有194条有限可信前缀事件及审计关系，仍非完整Email依赖图；数值状态层面（T2）只有少量局部观测，产品分支/算术训练对为0；真实干预层面（T3）没有执行对。观测性依赖结构不等于学到因果机制。

**下一阶段首要任务：针对冻结训练队列，设计并独立验收最小必要的初始字段/堆状态观测。** 优先证明真实静态初始化/测试复位后的定义和值及其可见时刻，覆盖本轮15条轨迹的首个字段缺口；必须在新授权阶段用有限配对Java执行核验。还需单列其后的身份、数组、类型与库效果阻碍，不能通过一次标量回调恢复整段未知历史。GPU算力不是本次阻塞原因；目前不应扩大网络或继续调同一Bank验证集。

只有补充真实状态证据并形成足够可信Email机制样本后，才适合另行冻结两系统、同信息输入的局部预测对照。当前不执行新采集、训练、分类联合优化、反事实或强化学习。

## 9. 文件交接

唯一完整中文长总结为本文件。机器结论、全部24条轨迹、3325个查询、两个产品及逐图继承状态分别见同目录的`phase05C2_review_packet.json`、`phase05C2_trace_status.csv`、`phase05C2_local_state_status.csv`、`phase05C2_product_status.csv`和`phase05C2_graph_status.csv`；`phase05C2_checks.json`连接生产、独立核验及跨系统比较。公共CSV是单独导出视图，仅扩展证据相对路径，不回写生产文件。CSV证据引用相对于本阶段容器根目录；来源绝对路径与SHA在`inputs/source_manifest.json`。

服务器报告同时通过已核实的X盘目录可读，本地项目`E:/Myself/投稿/work7-cwm/outputs/`保存相同副本；交付回执另核对摘要。此前文档、冻结数据、首轮失败材料和原正式结果均未删除或覆盖；本阶段没有可安全删除的历史文档。总结索引仅链接本文件，不再维护第二份长总结。阶段到此结束，等待下一步指令。
'''
    (out/'phase05C2_review_packet.md').write_text(text,encoding='utf8')
    (out/'phase05C2_summary.md').write_text('# 阶段05C-2总结索引\n\n唯一完整中文总结：[阶段结果交接](phase05C2_review_packet.md)。机器证据和逐行状态表由该文档链接。\n',encoding='utf8')
    # Standard parsers validate every delivered JSON/CSV before receipt creation.
    for f in out.glob('*.json'):load(f)
    csvcounts={}
    for f in out.glob('*.csv'):
        with f.open(encoding='utf8') as stream:csvcounts[f.name]=len(list(csv.DictReader(stream)))
    write_json(root/'report_receipt.json',{'files':{f.name:sha256(f) for f in out.iterdir() if f.is_file()},'csv_rows':csvcounts,
        'source_manifest_unchanged':sha256(root/'inputs/source_manifest.json')==checks['source_manifest_sha256'],
        'protocol_unchanged':sha256(out/'phase05C2_protocol.json')==checks['protocol_sha256'],
        'report_script_sha256':sha256(Path(__file__)),'report_generation_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
    print(json.dumps({'metrics':metrics,'flags':flags,'resources':resources,'csv_rows':csvcounts},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
