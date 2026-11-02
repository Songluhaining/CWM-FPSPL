"""Build one handoff from frozen evidence; never load targets or run a model."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from collections import Counter
from datetime import datetime,timezone
import ast

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(r'D:/dataset/whn/cwm_fp_generated/data/processed/graph_classifier/phase04T1_cpu_v1')


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def finalize(out,packet):
    """Validate delivery, snapshot sources/documents and index their bytes."""
    changed=packet['changed_files']
    for name in changed:
        source=ROOT/name
        assert source.is_file(),name
        if source.suffix=='.py':ast.parse(source.read_text(encoding='utf-8'),filename=name)
        destination=out/'producer_snapshot'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
    # Include unchanged local validators/target readers needed to replay without Git.
    for source in (ROOT/'src/cwm_fp').rglob('*.py'):
        destination=out/'producer_snapshot'/source.relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
    actual=read(ROOT/'outputs/phase04T1_review_packet.json')
    required=('stage_id','protocol','cohort_manifest_sha256','training_products','training_versions','training_graphs',
        'authorized_label_source','target_file_access','model_config','optimizer_config','seed','device','gradient_checks',
        'parameter_update_checks','whole_vs_chunked_equivalence','representation_conflicts','actual_optimization_steps',
        'loss_summary','checkpoint_resume_checks','execution_environment','server_access','resources','changed_files','artifact_paths','unresolved_issues','recommended_next_stage')
    assert all(k in actual for k in required)
    assert len(actual['recommended_next_stage'])<=3
    assert actual['readiness_states']['HELDOUT_EVALUATION_PERFORMED'] is False
    assert actual['readiness_states']['WORLD_MODEL_TRAINED'] is False
    assert actual['readiness_states']['LINUX_GPU_VERIFIED'] is None
    for name in ('training_trace.csv','gradient_checks.json'):
        assert sha(out/name)==sha(ROOT/'outputs'/('phase04T1_'+name))
    for key,path in actual['artifact_paths'].items():
        if key=='final_audit':continue
        assert (Path(path) if Path(path).is_absolute() else ROOT/path).exists(),(key,path)
    # Explicitly prove the post-run change did not alter any training function.
    def functions(path):
        return {n.name:ast.dump(n) for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    before=functions(out/'executed_source/src/cwm_fp/models/package_training.py')
    after=functions(ROOT/'src/cwm_fp/models/package_training.py')
    assert all(before[n]==after[n] for n in before if n!='run_package_acceptance')
    a=ast.parse((out/'executed_source/src/cwm_fp/models/package_training.py').read_text(encoding='utf-8'))
    b=ast.parse((ROOT/'src/cwm_fp/models/package_training.py').read_text(encoding='utf-8'))
    old_main=next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name=='run_package_acceptance')
    new_main=next(n for n in b.body if isinstance(n,ast.FunctionDef) and n.name=='run_package_acceptance')
    for function in (old_main,new_main):
        for node in ast.walk(function):
            if isinstance(node,ast.Try):node.finalbody=[]
    assert ast.dump(old_main)==ast.dump(new_main)
    audit={'stage':'04T1','status':'PASS','json_required_fields':list(required),'changed_files_present':len(changed),
        'python_syntax':'PASS','public_trace_rows':236,'public_artifact_copies_equal':True,'artifact_references_exist':True,
        'package_and_public_hash_checks':packet['freeze_checks']['package_and_public_file_hashes_verified'],
        'post_training_change_scope_verified':'resource function plus finally resource collection only; all training operations unchanged',
        'heldout_labels_read_by_this_report':False,'new_model_updates_by_this_report':0,
        'saved_sources':'producer_snapshot/ and executed_source/; the latter matches frozen training plan',
        'disk_free_bytes':shutil.disk_usage(out).free,'directory_bytes_before_final_index':sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}
    assert audit['directory_bytes_before_final_index']<512*1024**2
    write(out/'final_audit.json',audit)
    files={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in out.rglob('*') if p.is_file()
        and p.name not in ('artifact_index.json','report.log','report.command.json')}
    write(out/'artifact_index.json',{'scope':'run directory excluding self and currently written report command files; public outputs copied to producer_snapshot',
        'files':files})


def main(out):
    r={name:read(out/(name+'.json')) for name in ('result','plan','environment','input_integrity','training_target_audit',
        'gradient_checks','representation_conflicts','feature_column_audit','synthetic','micro_fit','short_run','resume_checks',
        'target_access','self_resources','unit_tests','prior_plan_difference','dependency_versions','micro_selection')}
    args=r['plan']['args'];package=Path(args['package']);previous=ROOT/'outputs/phase04D1_review_packet.json';old=read(previous)
    manifest=read(package/'manifest.json');independent=read(package.parent/'export_receipt.json')
    manifest_sha=sha(package/'manifest.json')
    # Both a separately saved export receipt and the frozen report must carry the digest.
    assert manifest_sha==args['manifest_sha256']
    assert manifest_sha in json.dumps(independent) and manifest_sha in json.dumps(old)
    checked=1
    for p in manifest['products']:
        for t in p['tests']:
            for field in ('tensor','metadata'):
                f=package/t[field]['path'];assert sha(f)==t[field]['sha256'];checked+=1
    for field,expected in [('products',manifest['source_provenance']['frozen_manifests_sha256']['products']),
                           ('tests',manifest['source_provenance']['frozen_manifests_sha256']['tests']),('splits',args['split_sha256'])]:
        assert sha(Path(args[field]))==expected;checked+=1
    # Historical reports and original model are not updated to make this stage pass.
    history=[]
    for f in (out/'producer_before/outputs').glob('phase04D1*'):
        assert sha(f)==sha(ROOT/'outputs'/f.name);history.append(f.name)
    source_verified=[]
    for name,h in r['plan']['source_sha256'].items():
        assert sha(out/'executed_source'/name)==h;source_verified.append(name)
    rows=list(csv.DictReader((out/'training_trace.csv').open(encoding='utf-8')))
    assert len(rows)==r['result']['actual_updates']==236
    assert [int(x['step']) for x in rows]==list(range(1,237))
    assert all(x['device']=='cpu' and x['lossless']=='True' for x in rows)
    assert all(x['products']==17 and x['graphs']==283 and x['lossless'] for x in r['short_run']['history'])
    assert set(r['target_access']['reads'])=={args['train_labels']} and not r['target_access']['blocked_attempts']
    counts=dict(Counter(x['phase'] for x in rows));cmds=[read(out/(x+'.command.json')) for x in
        ('unit_command','acceptance','portability_tests','entry_help','target_guard_test')]
    assert all(c['exit_code']==0 and c['reason'] is None for c in cmds)
    for name in ('portability_tests','target_guard_test'):
        text=(out/(name+'.log')).read_text();assert '\nOK' in text
    full=r['short_run'];gradient=r['gradient_checks'];resume=r['resume_checks'];conflicts=r['representation_conflicts']
    params={name:{'all_present':all(c['gradient_checks'][name]['present'] for c in gradient['cases']),
                  'all_finite':all(c['gradient_checks'][name]['finite'] for c in gradient['cases']),
                  'nonzero_in_any_case':any(c['gradient_checks'][name]['nonzero'] for c in gradient['cases']),
                  'max_abs_change':max(c['parameter_update_checks'][name] for c in gradient['cases'])}
            for name in gradient['cases'][0]['gradient_checks']}
    unresolved=[
        {'code':'NO_HELDOUT_EFFECTIVENESS_EVALUATION','status':'confirmed','impact':'当前28产品；本阶段没有验证/测试预测或泛化成绩',
         'evidence':'target_access.json; result.json','recovery_product_upper_bound':None,'reason':'不是产品拒绝问题；下一阶段需预先确定比较及独立评价协议'},
        {'code':'PROXY_TARGETS_UNVERIFIED','status':'confirmed','impact':'当前监督目标仍是既有PULP代理标签，本阶段未独立核实',
         'evidence':'training_target_audit.json','recovery_product_upper_bound':None,'reason':'训练损失不能认证标签真值'},
        {'code':'TRANSACTION_UNSUPPORTED_SUBSET','status':'confirmed_scope_root_cause_partial','impact':'04D1保留的16张拒绝图、13个拒绝产品；合格队列没有Transaction配置/完整Transaction_ESTest',
         'evidence':'outputs/phase04D1_graph_status.csv; outputs/phase04D1_review_packet.md H/I',
         'recovery_product_upper_bound':13,'reason':'仅全部剩余语义经后续认证时的估计上界；本阶段恢复0产品，单修CHECKCAST收益未证实'},
        {'code':'CONFIGURATION_CONSTRAINTS_UNVERIFIED','status':'confirmed','impact':'17训练产品的config_enabled/config_known两列均屏蔽为0',
         'evidence':'feature_column_audit.json; src/cwm_fp/models/graph_encoder.py','recovery_product_upper_bound':None,
         'reason':'掩码不是配置均关闭的真实观测；目前分类主要使用结构和语义类型'},
        {'code':'NO_CERTIFIED_REAL_TIME_PREFIX','status':'confirmed','impact':'完整图含事后执行/输出信息，不能直接作为较早时间步的输入',
         'evidence':'src/cwm_fp/models/prefix_contract.py; outputs/phase04D1_review_packet.md',
         'recovery_product_upper_bound':None,'reason':'仅人工前缀契约检查；缺真实生产器，阻碍时间状态世界模型'},
        {'code':'INCOMPLETE_STATE_AND_NO_INTERVENTION','status':'confirmed','impact':'局部标量不等于完整状态；缺完整输入/输出/断言数值及合法干预对；当前22列不编码原始数值',
         'evidence':'outputs/phase04D1_review_packet.md; src/cwm_fp/models/graph_encoder.py',
         'recovery_product_upper_bound':None,'reason':'T2不足，T3缺失，不可用模型预测值充当实测'},
        {'code':'LINUX_GPU_NOT_VERIFIED','status':'not_executed','impact':'服务器环境和显存仍未知，不阻碍已完成Windows CPU验收',
         'evidence':str(package.parent/'server_access.json'),'recovery_product_upper_bound':None,
         'reason':'历史X:/codes访问拒绝，本次未重试；没有可用连接信息'},
        {'code':'ACTIVATION_MEMORY_NOT_SEPARATELY_MEASURED','status':'confirmed','impact':'分批读取保留全部计算图；扩大产品时可能增加训练内存',
         'evidence':'gradient_checks.json; self_resources.json','recovery_product_upper_bound':None,
         'reason':'只测到整个张量进程的操作系统生命周期峰值，不能声称常量内存或分片训练节省显存'}]
    next_steps=[
        {'priority':1,'task':'预先固定当前分组队列上的小规模模型比较与独立评价协议',
         'reason':'训练链路已通过；下一缺口是同队列基线、验证选择规则、按版本不确定性及封存测试评价',
         'evidence':'result.json; input_integrity.json','can_start_now':'可准备协议和入口对接；本阶段不执行验证或测试评价，也不自动使用工程模型作为最终模型'},
        {'priority':2,'task':'取得可用服务器会话后，在隔离环境完成Linux及单GPU有界复验',
         'reason':'已交付同入口脚本；设备、驱动、依赖及实际峰值显存未验证',
         'evidence':'scripts/run_phase04T1_linux.sh; dependency_versions.json','can_start_now':'可准备文件；尚不能声称已在服务器运行或已具备经过验收的GPU训练环境'},
        {'priority':3,'task':'为结构世界模型单独建立并核验真实时间前缀生产器',
         'reason':'当前已认证的完整轨迹需按真实可见时刻裁定状态和边，避免未来结果泄漏',
         'evidence':'src/cwm_fp/models/prefix_contract.py; outputs/phase04D1_review_packet.md',
         'can_start_now':'可开展独立数据工程；数值状态世界模型和反事实推理仍不具备充分实测条件'}]
    changed=['src/cwm_fp/models/train_graph.py','src/cwm_fp/models/graph_package.py','src/cwm_fp/models/package_training.py',
        'tests/test_package_training.py','tests/test_training_resources.py','scripts/test_phase04T1.py','scripts/report_phase04T1.py',
        'scripts/run_phase04T1_linux.sh','configs/phase04T1_requirements.txt','DATA_CONTRACTS.md','STAGES.md','outputs/phase_status.md',
        'outputs/phase04T1_review_packet.md','outputs/phase04T1_review_packet.json','outputs/phase04T1_training_trace.csv','outputs/phase04T1_gradient_checks.json']
    resources={**r['self_resources'],'model_window_seconds':r['result']['model_window_seconds'],
        'model_and_check_process_seconds_conservative':r['self_resources']['model_window_seconds']+sum(c['seconds'] for c in cmds if 'package_smoke' not in c['argv']),
        'recorded_test_and_acceptance_process_seconds':sum(c['seconds'] for c in cmds),
        'activation_memory_bytes':None,'activation_memory_reason':'No separate autograd activation measurement; all chunk graphs retained',
        'gpu_peak_bytes':None,'gpu_peak_reason':'No GPU execution','timeout':False,'OOM':False,'truncation':False,
        'nan_or_inf':False,'silent_product_loss':False,'cache_invalidation':False,
        'rss_monitor_caveat':'Outer command monitor observed only the small Windows venv launcher; authoritative memory is self_resources actual tensor worker OS peak, not hard OS enforcement',
        'disk_free_bytes_at_report':shutil.disk_usage(out).free,'run_directory_bytes_at_report':sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
        'initial_disk_free_bytes':None,'initial_disk_reason':'Not persisted as an independently frozen pre-run resource record',
        'time_scope':'Actual command durations exclude planning/report writing; model window conservatively includes I/O inside first-model-to-final-checkpoint interval'}
    states={'TRAINING_ENTRY_VERIFIED':True,'BACKWARD_PASS_VERIFIED':True,'OPTIMIZER_UPDATE_VERIFIED':True,
        'TRAIN_ONLY_SMOKE_COMPLETED':True,'CHECKPOINT_RESUME_VERIFIED':True,'LINUX_GPU_VERIFIED':None,
        'HELDOUT_EVALUATION_PERFORMED':False,'WORLD_MODEL_TRAINED':False}
    losses={n:{k:r[n][k] for k in ('initial_loss','final_loss')} for n in ('synthetic','micro_fit','short_run')}
    graph_metrics={'products':28,'versions':16,'registered_tests':463,'candidate_graphs':463,'identity_valid_graphs':463,
        'dependency_valid_graphs':463,'full_suite_products':28,'admitted_products':28,**old['after_metrics']['package_totals'],
        'control_dep':old['after_metrics']['packaged_control_dep'],'data_dep':old['after_metrics']['packaged_data_dep']}
    packet={'stage_id':'04T1','stage_status':'COMPLETE_LOCAL_TRAINING_ENGINEERING','actual_scope':'Windows CPU; frozen BankAccountTP17 TRAIN products only; no Java runs, no held-out evaluation, no new semantic certification',
        'generated_at':datetime.now(timezone.utc).isoformat(),'protocol':args['protocol'],'cohort_manifest_sha256':manifest_sha,
        'training_products':17,'training_versions':9,'training_graphs':283,'authorized_label_source':r['training_target_audit'],
        'target_file_access':r['target_access'],'model_config':r['plan']['model'],'optimizer_config':r['plan']['optimizer'],
        'seed':r['plan']['seed'],'device':'cpu','gradient_checks':gradient,'parameter_update_checks':params,
        'whole_vs_chunked_equivalence':{'passed':gradient['passed'],'atol':gradient['atol'],'rtol':gradient['rtol'],
            'cases':[{k:c[k] for k in ('case','tests','micro_tests','last_micro_tests','checks','passed')} for c in gradient['cases']]},
        'representation_conflicts':conflicts,'feature_columns':r['feature_column_audit'],
        'actual_optimization_steps':236,'optimization_step_breakdown':counts,'loss_summary':losses,
        'full_train_history':full['history'],'checkpoint_resume_checks':resume,'execution_environment':r['environment'],
        'dependency_versions':r['dependency_versions'],'server_access':{'current_status':'NOT_VERIFIED','retried_this_stage':False,
            'historical_observation':read(package.parent/'server_access.json'),'evidence':str(package.parent/'server_access.json'),
            'linux_script_status':'prepared; not executed under Linux; no Bash syntax check available locally'},
        'resources':resources,'changed_files':changed,'artifact_paths':{'run_dir':str(out),'package':str(package),
            'review_markdown':'outputs/phase04T1_review_packet.md','review_json':'outputs/phase04T1_review_packet.json',
            'training_trace':'outputs/phase04T1_training_trace.csv','gradient_checks':'outputs/phase04T1_gradient_checks.json',
            'final_checkpoint':str(out/'engineering_train_only.pt'),'checkpoint_receipt':str(out/'engineering_train_only.receipt.json'),
            'executed_source':str(out/'executed_source'),'final_audit':str(out/'final_audit.json')},
        'readiness_states':states,'readiness_reasons':{'LINUX_GPU_VERIFIED':'NOT_VERIFIED: no available authorized server session',
            'HELDOUT_EVALUATION_PERFORMED':'false by current task boundary','WORLD_MODEL_TRAINED':'false by current task boundary'},
        'before_metrics':{**graph_metrics,'real_graph_training_acceptance_verified':False},
        'after_metrics':{**graph_metrics,'real_graph_training_acceptance_verified':True},'recovered_graphs':0,'new_admitted_products':0,
        'recovered_graphs_reason':'No semantic repair or new collection this stage;04D1 rejected evidence remains frozen',
        'split_readiness':{'grouped_bank_pilot_v1':{'status':'READY_ENGINEERING_ONLY','partitions':r['input_integrity']['partitions'],
            'train_class_counts':r['training_target_audit']['counts'],'validation_test_classes':None,'reason':'Validation/test targets sealed; no effectiveness claim'},
            'other_protocols':{'status':'BLOCKED_FOR_COMPLETE_CROSS_SYSTEM_EXPERIMENT','unlabeled_counts_inherited':old['split_readiness'],
                'class_counts':None,'reason':'Historical public counts only; no other protocol target opened; Email admitted sample absent'}},
        'world_model_readiness':{'T1':{'status':'PARTIAL','reason':'Ordered events and supported dependencies available; real leakage-safe prefix producer absent'},
            'T2':{'status':'BLOCKED','reason':'Incomplete state/input/output/assertion values; classifier22 columns are semantic flags'},
            'T3':{'status':'BLOCKED','reason':'No real legal intervention pairs'},'prefix_status':'contract_only_not_real_producer'},
        'tests_executed':{'unit_passed':61,'unit_failures':0,'unit_errors':0,'original_regression':r['unit_tests'],
            'additional_tests':['TrainingResourceTests.test_actual_process_has_positive_peak','TrainingResourceTests.test_unknown_measurement_is_null_with_reason',
                'PackageTrainingTests.test_actual_guard_blocks_heldout_and_other_protocol'],'new_Java_test_processes':0,'commands':cmds},
        'prior_plan_difference':r['prior_plan_difference'],'unresolved_issues':unresolved,'recommended_next_stage':next_steps,
        'freeze_checks':{'package_and_public_file_hashes_verified':checked,'preserved_history':history,'executed_model_source_files':len(source_verified),
            'training_source_matches_final_except':'package_training.py post-run resource-only portability adjustment; executed_source preserves exact training bytes'},
        'diagnostic_failures_not_training':[
            'Read-only inspection attempted to summarize a non-existent CSV column section; actual column phase used on retry; no training rerun.',
            'Portability check launcher initially passed unsupported max_log_bytes; child was not started; corrected to existing runner max_bytes.',
            'git status confirmed this workspace is not a git repository; optional pyproject.toml was absent; no repo metadata or dependency file invented.']}
    write(ROOT/'outputs/phase04T1_review_packet.json',packet)
    shutil.copy2(out/'training_trace.csv',ROOT/'outputs/phase04T1_training_trace.csv')
    shutil.copy2(out/'gradient_checks.json',ROOT/'outputs/phase04T1_gradient_checks.json')
    table=lambda rows:'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)
    eqrows=[]
    for c in gradient['cases']:
        for name,title in [('logits','产品输出'),('loss','产品损失'),('gradients','逐参数梯度'),('update','更新后参数/缓冲区')]:
            v=c['checks'][name];eqrows.append([c['tests'],title,f"{v['max_absolute_difference']:.10g}",f"{v['max_relative_difference']:.10g}",'通过' if v['passed'] else '未通过'])
    gradrows=[[name,f"{gradient['cases'][0]['gradient_checks'][name]['norm']:.8g}",f"{gradient['cases'][1]['gradient_checks'][name]['norm']:.8g}",
               f"{value['max_abs_change']:.8g}"] for name,value in params.items()]
    lossrows=[[title,steps,f"{r[n]['initial_loss']:.9f}",f"{r[n]['final_loss']:.9f}"] for n,title,steps in
              [('synthetic','人工两产品','40'),('micro_fit','真实4产品/4版本/87测试图','100'),('short_run','全训练17产品/9版本/283测试图','90（10轮）')]]
    epochrows=[[x['epoch'],x['products'],x['graphs'],f"{x['mean_update_time_loss']:.9f}",f"{x['post_epoch_mean_loss']:.9f}"] for x in full['history']]
    sourcerows=[[p,{'src/cwm_fp/models/train_graph.py':'增加包验收命令分支，旧评价训练分支保留',
        'src/cwm_fp/models/graph_package.py':'增加保留梯度的全测试分批汇聚，旧推理接口保留',
        'src/cwm_fp/models/package_training.py':'复用既有模型/读取/目标接口的受限验收流程、预算及保存恢复',
        'tests/test_package_training.py':'目标、身份、完整性、表示冲突及读取隔离边界',
        'tests/test_training_resources.py':'真实Windows资源读取和未知测量为空值',
        'scripts/test_phase04T1.py':'限定回归入口，避开旧验证拟合/预测测试',
        'scripts/report_phase04T1.py':'只读实际证据生成本交接，无模型/目标读取',
        'scripts/run_phase04T1_linux.sh':'同一入口的参数化启动脚本；Linux未执行',
        'configs/phase04T1_requirements.txt':'记录最小依赖版本；未安装或升级共享环境'}[p]] for p in changed[:9]]
    commands='\n\n'.join('```text\n'+' '.join('"'+x+'"' if ' ' in x else x for x in c['argv'])+f"\n# exit={c['exit_code']}; seconds={c['seconds']}\n```" for c in cmds)
    report=f'''# CWM-FP阶段04T-1结果交接

2026-11-01。本阶段已完成本地训练工程验收：真实训练区的17产品/9缺陷版本/283测试图接入既有关系图卷积分类器（R-GCN），反向传播、参数更新、整产品与逐测试汇聚对照、保存恢复均通过。实际进行了有限拟合，不能继续称为“没有真实图训练”。全部验收236次更新，其中人工40次，真实图相关196次；没有验证/测试集评价和泛化成绩，没有Java重跑、数据扩采、语义修复或后续世界模型训练。

结论：当前受支持银行账户子集的训练链路已就绪，可准备固定协议的小规模模型比较；科学有效性尚未验证。Linux/GPU尚未验证，状态为未执行（NOT_VERIFIED），不影响本地核心任务完成。下列结果来自实际日志，工程实现、测试通过及研究有效性分别判断。

## 1. 数据与冻结边界

主协议`grouped_bank_pilot_v1`继承阶段01原按缺陷版本分组划分，没有重新划分。

| 分区 | 产品 | 独立版本 | 登记测试/测试图 | 执行事件 | 节点 | 边 | 本阶段用途 |
|---|---:|---:|---:|---:|---:|---:|---|
{table([[s,d['products'],d['versions'],d['graphs'],d['events'],d['nodes'],d['edges'], '训练及诊断' if s=='train' else '仅无标签内容核验'] for s in ('train','val','test') for d in [r['input_integrity']['partitions'][s]]])}

合计28产品/16版本/463图，106183节点、183787边、22086事件，控制依赖边24822条、动态数据依赖边21658条。修复前后这些数据指标完全不变；463张均继承已通过的身份和依赖语义认证，28产品均完整准入。新增合格图/产品为0，未恢复04D1保留的16张拒绝图和13个拒绝产品。本阶段的数据包内容核验不是重新运行Java或重新证明所有执行语义。

可信包清单摘要为`{manifest_sha}`，与包外`export_receipt.json`及旧04D1交接JSON交叉核对。收尾再次核验{checked}项包/公开清单文件摘要，旧交接文件保留不变。没有采样、丢测试、丢节点、丢事件或丢关系；本轮每个训练轮次均覆盖全部17产品/283图。外部证据目录与具体文件索引见末节。

训练目标通过原grouped训练目标接口读取，编码FP=1、TP=0，连接当前17产品，类别为10个代理FP（6版本）、7个代理TP（4版本）；类别版本数可交叠，总独立版本仍为9。原接口在文件级校验整个grouped训练文件，其余未入队产品标签随即丢弃，未用于拟合或类别统计。训练目标摘要`{args['train_label_sha256']}`。真实目标访问记录只有该训练文件的2次读取（摘要和接口读取），无验证/测试/其他协议目标读取。程序级访问检查不是操作系统绝对隔离。

对其他历史冻结协议仅引用旧无标签数量：留出BankAccountTP时训练0/验证0/测试28产品（0/0/16版本）；留出Email时训练25/验证3/测试0产品（15/1/0版本）。未读取这些协议的任何目标，不能用其训练池补充当前17产品。两个跨系统协议仍缺完整可用的跨系统评价队列。

关于04D1计划差异，一次性核对已找到真实用户附件第4.2、5节明确授权36新增产品、每版本3个、1000测试对、5400秒及3GiB上限；实际计划额外保留更严格的4GiB剩余空间下限。因此24/2属于早先建议，实际采用有明确来源的36/3任务要求。原附件副本和摘要保存在`prior_plan_difference.json`及`04D1_user_request.txt`，未修改旧计划或补造授权。

## 2. 实际修改、复用和验证范围

| 文件 | 实际作用 |
|---|---|
{table(sourcerows)}

另更新`DATA_CONTRACTS.md`、`STAGES.md`及简短`outputs/phase_status.md`，生成本阶段四个核心交付文件。旧`graph_classifier.py`、`graph_encoder.py`和所有Java程序/采集器均未改动；未建立新模型架构。旧训练模式会调用验证评价，本次在读取旧配置之前选择独立的包验收分支，避免调用该评价流程。

真实研究图上验证：包到模型输入、全部测试汇聚、梯度/更新、有限拟合与恢复。人工图/单元用例验证：错误目标连接、缺图、未认证图、跨分区读取、损坏摘要/格式、冲突样本保留、未知资源测量；人工验收目录仅有训练目标，验证/测试目标文件不存在，仍完成40次更新。没有运行新的人工Java用例。Linux启动脚本与GPU分支已提供但未在服务器验证；真实安全时间前缀、完整状态及干预功能未实现。

训练结束后仅把资源记录改为Python标准库的操作系统查询，去除了对本机其他Python安装目录的依赖；此改动由2项针对性测试验证，没有重跑训练或改变模型计算。精确训练源码12文件保存在`executed_source/`，其摘要逐一匹配冻结`plan.json`，最终源码另作快照，明确区分执行版本与后续资源记录调整。

## 3. 产品级训练语义与梯度

固定配置：22列语义/布尔表示、7类关系、2层/宽16的小型既有R-GCN、单输出、32位浮点数、种子20261101、Adam学习率0.001。采用直接接收未归一化输出的二分类损失（BCEWithLogitsLoss），FP=1、TP=0，没有重复概率转换，没有类别权重、验证早停或参数搜索。

先在每个测试图内汇聚节点，再等权汇聚一个产品的全部测试，最后一个产品对应一个输出和损失；批次内产品等权。全训练区批次大小2，末批1产品按实际数量计算；每轮每产品恰好一次。不存在跨测试依赖连接或将产品标签复制成测试级独立监督。图编码为固定语义字典，不拟合归一化统计；归一化审计贡献者仅当前17产品。

反向传播执行顺序为清梯度、完整产品前向、产品损失、反向传播、参数更新。前向钩子确认编码器输出保留求梯度路径；全部14组参数注册于优化器并具有有限梯度。下表给出24测试和6测试对照中的实际梯度范数与参数最大变化；个别未激活元素出现零梯度是合法现象，不能因此伪造更新或要求每个元素每步非零。

| 参数组（代码名称） | 24测试梯度范数 | 6测试梯度范数 | 一步最大绝对变化 |
|---|---:|---:|---:|
{table(gradrows)}

所有236更新记录均保留损失、梯度范数和参数变化，未出现非数值、无穷值、梯度断开或内存耗尽。这里的数值正常只覆盖实际有界运行，不代表所有未来输入都已验证。

## 4. 整产品与分批计算对照

固定选取训练区测试数最多且超过16的产品（24测试）及最小产品（6测试），身份排序破同数。每小批5测试，末批分别4与1，按实际测试数量加权；两条路径从同一模型与优化器初态开始。对照采用评价状态但保持梯度开启；随后三个拟合阶段均实际执行训练状态。

预先冻结绝对容差1e-6、相对容差1e-5，判据为逐元素`|a-b| <= atol + rtol*|b|`。相对差分母采用两侧绝对值较大者并下限1e-12，仅作诊断。近零梯度的最大相对差可超过1e-5，但绝对差远小于1e-6，仍符合事先判据，未事后放宽。

| 产品测试数 | 比较项 | 最大绝对差 | 最大相对差 | 结果 |
|---:|---|---:|---:|---|
{table(eqrows)}

两条带梯度路径均通过。实际微型拟合及短程试跑仍采用简单的整产品路径。分批路径保留全部反向传播计算图，未单独测得激活内存，不宣称常量训练内存或节省显存。

## 5. 表示冲突与真实学习行为

仅检查17训练产品。283测试图中240种规范张量输入；产品测试集合及按测试频率归一化后的集合均无完全相同输入的重复产品组，因此没有发现相同输入对应相反标签的产品。未发现同一图对象/张量存储误接多产品，未发现空图、所有关系丢失或全零输入。检测保持节点行顺序、排序边多重集合、忽略产品内测试顺序，不是一般图同构或程序语义等价证明；也不能排除关系图模型本身的表示能力限制。

22列中3列全0：未解析程序身份列表示合格图无此类节点；配置启用与配置已知两列由于配置约束尚未认证而按原掩码屏蔽，不能解释为所有配置均关闭。其余19列均有变化。原始数值状态不是本次22列的学习输入。训练区保留7类关系：控制依赖16824条、数据依赖13323条，其余关系计数见`representation_conflicts.json`。未为改善拟合删除冲突样本、加入产品编号或改变标签。

| 阶段 | 实际更新数 | 初始产品均值损失 | 最终产品均值损失 |
|---|---:|---:|---:|
{table(lossrows)}

微型4产品按冻结种子排序、训练标签两类各2个、优先不同版本选择；实际为4版本/87图。未要求100%训练准确率，也没有根据结果调整超参数。全训练区从同一预定规则重新初始化，没有使用微型拟合模型继续训练。损失下降有限，证明当前模型在预算内产生学习行为，不证明收敛、跨缺陷泛化或优于传统基线。

| 轮次 | 产品数 | 测试图数 | 更新时损失的产品加权均值 | 轮末固定模型产品均值损失 |
|---:|---:|---:|---:|---:|
{table(epochrows)}

所有轮次无损。两列损失分别反映轮内变化中的模型与轮末同一个模型，不能混作同一测量。逐更新曲线见`phase04T1_training_trace.csv`，初始和轮末完整记录见运行目录的三个拟合JSON。

更新总数构成为4次梯度对照＋40次人工拟合＋100次真实微型拟合＋2次恢复对照＋90次全训练短跑＝236，低于400上限。没有额外种子、额外调参或未记录模型更新。

## 6. 保存恢复与目标隔离

保存模型、Adam状态、Python/张量随机状态、归一化记录、配置、种子、步骤及数据清单摘要，使用`engineering_train_only`标记及独立回执。固定训练输入在恢复前后输出最大绝对/相对差均为0；连续训练与恢复后下一步的全部梯度、参数/缓冲区差均为0。错误队列摘要、旧列版本、缺关系模式、错误文件摘要4种恢复均拒绝；数据损坏/缺失拒绝沿用既有包检查并通过相关回归。

最终工程文件`engineering_train_only.pt`摘要为`{r['result']['checkpoint_sha256']}`。它不是经过验证选出的正式最终模型，不自动用于后续正式比较。当前恢复证明限于确定性同环境的下一步；没有验证跨设备逐字节恢复或任意训练调度的中断恢复。

读取隔离另用人工临时目录验证真实访问钩子：允许指定训练文件，拒绝验证、测试及其他协议训练路径。真实训练记录无越权尝试；没有输出任何验证/测试预测、准确率、F1或曲线面积，也不在公共报告列出逐产品目标。

## 7. 实际环境、测试、资源及部署准备

执行位置：Windows CPU。主机`{r['environment']['hostname']}`，系统`{r['environment']['os']}`；Python3.11.4，解释器`{r['environment']['executable']}`，PyTorch2.6.0+cu126、NumPy2.4.6，单CPU计算线程。安装包具有CUDA后缀不代表使用了GPU。当前目录为项目根目录，包路径见末节；没有升级依赖、修改驱动或共享环境。

58项初始针对性/相关回归通过，后补2项资源移植检查和1项目标钩子检查通过，共61项；命令帮助检查也成功。旧会调用验证预测/拟合的测试未执行，避免超出授权。没有新Java测试进程；真实研究图验收覆盖17训练产品/283图，另11产品/180图只做无标签核验。全部正式测试及训练命令退出0。

训练验收进程总耗时{cmds[1]['seconds']:.3f}秒，其中从首个模型构建到最终检查点的保守模型窗口{r['result']['model_window_seconds']:.3f}秒。连同相关检查进程计入的保守上界{resources['model_and_check_process_seconds_conservative']:.3f}秒，小于1800秒；实际计划模型窗口1740秒、余60秒留给检查。列出的测试/验收命令合计{resources['recorded_test_and_acceptance_process_seconds']:.3f}秒，不含阅读、开发和报告编写。

实际张量工作进程生命周期峰值驻留内存{r['self_resources']['peak_working_set_bytes']}字节（约{r['self_resources']['peak_working_set_bytes']/1024**3:.3f}GiB），来自工作进程内部操作系统记录。外部监控只看到Windows虚拟环境的小启动进程，不能用其约5MiB值冒充训练内存；它也不构成实际张量工作进程的硬内存配额。激活内存与GPU显存均未单独测量，JSON保留null及原因。无超时、截断、内存耗尽、失效缓存或静默跳过产品。

写报告时运行目录占{resources['run_directory_bytes_at_report']}字节，D盘剩余{resources['disk_free_bytes_at_report']}字节（约{resources['disk_free_bytes_at_report']/1024**3:.3f}GiB）；最终目录大小/余量另在`final_audit.json`记录。启动前磁盘读数没有独立冻结到资源文件，机器交接不以0代替该未知项。

X:/codes的最后真实记录来自04D1：当前进程访问被拒绝（WinError5）。本阶段遵循要求未重试、未改权限、未推断Linux路径；目前服务器可访问性仍未验证。没有服务器连接信息、Linux会话或GPU执行，不能以Windows结果冒充服务器结果。

交付`configs/phase04T1_requirements.txt`及`scripts/run_phase04T1_linux.sh`。使用Python3.11的独立环境，按目标设备核对后选择对应PyTorch发行包；依赖文件只记录本次已用版本，不自动安装。脚本接收解释器、数据包、包外摘要、公开产品/测试清单、冻结划分与摘要、独立训练目标及摘要、新输出目录、设备参数。图包与目标分开迁移，不迁移BRFL或执行Java，也不携带验证/测试目标。示例参数顺序：

```text
bash scripts/run_phase04T1_linux.sh PYTHON PACKAGE MANIFEST_SHA PRODUCTS TESTS SPLITS SPLIT_SHA TRAIN_CSV TRAIN_SHA NEW_OUTPUT [cpu|cuda]
```

此脚本尚未在Linux/Bash执行；本地未发现可用的已知Bash安装，未伪称语法或服务器验收通过。脚本仅供下一步显式执行，不会自动启动远端工作。

实际命令如下，完整工作目录、退出码、摘要和时间见同名`.command.json`。运行时设置`PYTHONPATH`指向项目`src`（测试时额外包含`tests`）；复现须另给新输出目录，不能覆盖本次计划。

{commands}

开发中的辅助检查曾出现错误列名`section`（实际为`phase`）、错误监控参数名（子进程未启动，已改用现有`max_bytes`）；工作区不是Git仓库，也无`pyproject.toml`。这些只读/启动前检查不属于训练失败，没有因此重复拟合。准确训练源码与实际失败说明均保留，不虚构版本控制提交或工程接口。

## 8. 当前准入、剩余问题和下一步

| 验收状态键 | 实际结论 |
|---|---|
{table([[k,'未执行，null＋原因' if v is None else '是' if v else '否'] for k,v in states.items()])}

当前分组协议可进入预先定义的小规模模型比较的准备工作，状态为工程就绪；不能立即宣布正式分类有效或跨系统实验就绪。标签仍是未独立核实的代理标签；交易配置及完整交易套件不在支持子集中，Email没有合格完整产品进入本包。应在同一队列上安排公平基线、冻结验证选择与测试评价流程，并按版本报告不确定性。当前工程检查点不可自动转为最终模型。

真实有序事件与依赖结构（T1）部分可用；控制边仍主要来自静态后支配分析及受支持路径核验，不是所有逐次执行控制者关系的认证。真实安全时间前缀生产器缺失，完整图含最终输出/执行标记，不能直接输入较早预测时间步。单元测试能拒绝人工越界前缀，不等于已经生产真实安全前缀。

数值状态与转换观测（T2）不足：只有部分局部标量，缺充分状态、完整输入/输出和断言期望/实测值；当前分类器22列也不是这些原始数值的编码。合法干预前后对（T3）缺失。世界模型、反事实和强化学习均未训练，不能把图分类链路通过解释为已经学到因果机制。

04D1的转账类型转换及后续对象/异常语义仍有认证缺口，继续影响16图/13产品；若全部问题日后经认证，最多恢复13产品只是估计上界，本阶段收益为0。配置约束认证、真实前缀、服务器运行和大规模激活内存也分别保留限制，不能互相替代。

下一步候选最多三项，按优先级排列：

1. 固定当前队列的小规模模型比较与独立评价协议，首要解除“有训练链路、无有效性证据”的缺口；先明确同队列基线、验证选择和封存测试评价，不自动使用任何保留标签。
2. 获得可用服务器会话后做Linux与单GPU有界复验，核对设备、驱动、依赖和峰值显存；可以准备迁移文件，尚不能声称服务器已验收。
3. 为结构世界模型单独建设真实时间前缀生产器，按可见时点隔离未来事件、后续边、返回效果和最终结果；数值状态世界模型及反事实仍需额外实测证据。

## 9. 交付与复核入口

核心四文件：`outputs/phase04T1_review_packet.md`（本文）、同名JSON、`outputs/phase04T1_training_trace.csv`（236更新）、`outputs/phase04T1_gradient_checks.json`（逐参数完整对照）。只维护这一份完整中文总结，`phase_status.md`仅作入口；不另复制同内容长报告，不删除仍被引用的历史证据。

冻结输入包：`{package.as_posix()}`。本次独立运行目录：`{out.as_posix()}`。

关键依据：`plan.json`、`input_integrity.json`、`training_target_audit.json`、`target_access.json`、`representation_conflicts.json`、`feature_column_audit.json`、`gradient_checks.json`、`synthetic.json`、`micro_fit.json`、`short_run.json`、`resume_checks.json`、两个工程检查点及回执、`environment.json`、`self_resources.json`、`unit_tests.json`和各命令日志。`producer_before/`保留修改前文档，`executed_source/`保留精确训练源码，`final_audit.json`与最终文件索引记录交付摘要。

阶段04T-1结束。没有启动下一阶段。
'''
    (ROOT/'outputs/phase04T1_review_packet.md').write_text(report,encoding='utf-8')
    finalize(out,packet)
    print(json.dumps({'status':'report_generated','updates':236,'unit_passed':61,'manifest_sha256':manifest_sha,'file_checks':checked}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,default=OUT)
    main(parser.parse_args().run_dir)
