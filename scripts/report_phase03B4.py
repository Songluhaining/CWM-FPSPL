"""Render phase03B4 handoff strictly from saved execution/admission evidence."""
from collections import Counter
from datetime import datetime,timezone
import csv
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cwm_fp.data.common import KEY,sha256,write_json
from cwm_fp.lddg.runner import tree_usage
from cwm_fp.lddg.suite import load,lines
from cwm_fp.lddg.effects import project_and_verify
from verify_phase03B4_framework import OUT,OLD,entries,FOLDS

REPORTS=ROOT/'outputs'
def dump(x):return json.dumps(x,ensure_ascii=False,sort_keys=True)
def csvout(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (REPORTS/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader()
        for r in rows:w.writerow({k:dump(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def md(name,text):(REPORTS/name).write_text(text,encoding='utf-8')
def table(headers,rows):return '|'+ '|'.join(headers)+'|\n|'+'|'.join('---' for _ in headers)+'|\n'+''.join('|'+ '|'.join(str(x) for x in r)+'|\n' for r in rows)

def preserve():
    seal=load(OUT/'baseline_seal.json');unchanged=0;archived=[]
    for path,h in seal.items():
        p=Path(path)
        if p.is_file() and sha256(p)==h:unchanged+=1;continue
        if not p.is_relative_to(ROOT):raise ValueError('Frozen external artifact changed: '+path)
        backup=OUT/'producer_before'/p.relative_to(ROOT)
        assert backup.is_file() and sha256(backup)==h,path
        archived.append({'current':path,'before_sha256':h,'archive':str(backup)})
    originals={}
    for _,work,p in entries():originals.update(p['source_sha256'])
    for path,h in originals.items():assert sha256(Path(path))==h,path
    result={'valid':True,'frozen_unchanged':unchanged,'mutable_files_preserved_before_change':archived,'readonly_source_files_verified':len(originals)}
    write_json(OUT/'preservation.json',result);return result

def main():
    rows=load(OUT/'comparison.json');new=[r for r in rows if r['batch_product']!='reuse03B1'];recovered=[r for r in new if r['qualified_after'] and not r['qualified_before']]
    admission=load(OUT/'admission.json');fixtures=load(OUT/'synthetic/verification.json');unit=load(OUT/'unit_test_results.json');plan=load(OUT/'plan.json')
    split={f:{'status':'BLOCKED','products':a['eligible_by_partition'],'independent_versions':a['versions_by_partition'],
              'authorized_train_audit':load(OUT/'readiness'/f'{f}.json')} for f,a in admission['aggregates'].items()}
    split['grouped']['reason']='训练区只有1个独立版本、1个FP，缺少TP；验证2版本、测试1版本规模极小。'
    split['loso_BankAccountTP']['reason']='当前合格图均来自银行，训练和验证区为空。'
    split['loso_Email']['reason']='授权训练区具有两类，但邮件留出测试区及验证区无合格产品。'
    before={'original_tests':140,'candidate_graphs':140,'identity_qualified_graphs':140,'control_qualified_graphs':138,
            'semantics_qualified_graphs':124,'collected_complete_products':7,'semantic_complete_products':0,'admitted_original_PASS_products':0,
            'rejected_graphs':16,'projected_execution_events':8200,'physical_execution_events':8200,'nodes':41100,'edges':73626,
            'control_edges':12689,'data_edges_including_partial':8002,'data_edges_in_qualified_graphs':7248,'value_versions':6428,'observed_scalar_value_nodes':2403}
    after={**before,'semantics_qualified_graphs':sum(r['qualified_after'] for r in new),'semantic_complete_products':5,
           'admitted_original_PASS_products':4,'rejected_graphs':sum(not r['qualified_after'] for r in new),'physical_execution_events':sum(r['physical_events_after'] for r in new),
           'data_edges_in_qualified_graphs':sum(r['edges_by_type'].get('data_dep',0) for r in new if r['qualified_after']),
           'new_framework_internal_events':480,'new_framework_scalar_constant_callbacks':16}
    assert all(sum(r[k] for r in new)==v for k,v in [('events_after',8200),('nodes_after',41100),('edges_after',73626),('values_after',6428),('observed_values_after',2403)])
    assert len(recovered)==14 and len(rows)==148
    graphs=[];callrows=[]
    for r in rows:
        key={k:r[k] for k in (*KEY,'test_id')};gaps=r['dynamic_gaps'];effects=r['framework_effects']
        graphs.append({**key,'batch_product':r['batch_product'],'original_test_result':r['original_test_result'],
            'instrumented_test_result':r['instrumented_test_result'],'candidate_exists':Path(r['graph_path']).is_file(),
            'complete_pair':r['complete_pair'],'parity':r['parity'],'identity_valid':r['identity_valid'],'control_valid':r['control_valid'],
            'data_dependency_valid':r['control_valid'] and not gaps,'external_effect_status':'CERTIFIED' if effects and all(e['accepted'] for e in effects) else 'REJECTED' if effects else 'NOT_APPLICABLE',
            'joint_qualified_before':r['qualified_before'],'joint_qualified_after':r['qualified_after'],
            'recovered':not r['qualified_before'] and r['qualified_after'],'rejection_codes':sorted(set(r['rejection_reasons']+[g['reason'] for g in gaps])),
            'first_gap_event_index':gaps[0]['event_index'] if gaps else None,'first_gap_log_line':None,
            'first_gap_opcode':None,'first_gap_bci':None,'dynamic_gaps':gaps,
            'projected_events':r['events_after'],'physical_events':r['physical_events_after'],'nodes':r['nodes_after'],'edges':r['edges_after'],
            'control_edges':r['edges_by_type'].get('control_dep',0),'data_edges':r['edges_by_type'].get('data_dep',0),
            'canonical_event_ids_changed':r['canonical_event_ids_changed'],
            'graph_evidence':r['graph_path'],'log_evidence':r['evidence_path'],'paired_status_evidence':r['status_path']})
        if effects:
            g=next(g for g in lines(Path(r['graph_path'])) if g['test_id']==r['test_id'])
            proof=g['control_flow_evidence']['proof'];raw,calls,full=project_and_verify(proof['framework_evidence'])
            if gaps:
                first=raw[gaps[0]['event_index']]
                graphs[-1].update(first_gap_log_line=first['raw_log_line'],first_gap_opcode=first['opcode'],first_gap_bci=first['bci'])
            for e in effects:
                event=raw[e['event_index']]
                callrows.append({**key,'batch_product':r['batch_product'],'caller':list(event['domain']),'bci':event['bci'],
                    'opcode':event['opcode'],'log_line':event['raw_log_line'],'log':r['evidence_path'],
                    'event_index':e['event_index'],'argument_value_ids':e['argument_value_ids'],'returned':event['call_returned'],
                    'return_log_line':event.get('return_log_line'),'framework_internal_events':e['internal_events'],
                    'observed_path':e['path'],'accepted':e['accepted'],'reason':e['reason']})
    csvout('phase03B4_graph_status.csv',graphs);write_json(OUT/'framework_calls.json',callrows)
    product_rows=[]
    for p in admission['products']:
        fold=p['folds']['grouped'];audit=fold['audit']['products'][0];lossless=p.get('lossless_interface');default=p.get('default_interface')
        product_rows.append({**dict(zip(KEY,p['identity'])),'batch_product':p['batch_product'],'original_product_result':p['original_result'],
            'registered_tests':p['registered_tests'],'collected_tests':audit['collected_tests'],'suite_complete':audit['suite_completeness']['complete'],
            'semantic_qualified_tests':p['semantic_graphs'],'suite_semantics_valid':p['semantic_graphs']==p['registered_tests'],
            'formal_classification_admission':fold['eligible'],'lossless_config_verified':lossless is not None,
            'default_lossless':all(default[k]==0 for k in ('lost_test_fraction','lost_event_fraction','lost_edge_fraction')) if default else None,
            'default_not_checked_reason':None if default else 'formal_admission_rejected',
            'default_lost_test_fraction':default['lost_test_fraction'] if default else None,
            'default_lost_event_fraction':default['lost_event_fraction'] if default else None,
            'default_lost_edge_fraction':default['lost_edge_fraction'] if default else None,
            **{f:part for f,part in p['partitions'].items()},'rejection_codes':audit['reasons'],
            'evidence':str(OUT/'admission.json'),'graph_evidence':str(OUT/p['batch_product']/'graphs.jsonl')})
    csvout('phase03B4_product_status.csv',product_rows)
    js=[]
    for category,base in [('benchmark',OUT/'runs'),('artificial',OUT/'synthetic/runs')]:
        for p in base.rglob('*.command.json'):js.append({'category':category,'evidence':str(p),**load(p)})
    write_json(OUT/'java_commands.json',js)
    preservation=preserve();usage=tree_usage(OUT)
    resources={'environment':load(OUT/'environment.json'),'output_directory':str(OUT),'output_snapshot':usage,'free_disk_bytes':shutil.disk_usage(ROOT).free,
        'java_test_processes':len(js),'java_test_process_seconds_sum':round(sum(x['seconds'] for x in js),3),
        'java_peak_sampled_rss_bytes':max(x['resource']['peak_sampled_rss_bytes'] or 0 for x in js),
        'memory_scope':'largest sampled direct Java test process, not Python/whole-machine peak',
        'python_peak_memory_bytes':None,'python_peak_memory_unknown_reason':'Main Python process memory was not measured throughout.',
        'entire_task_elapsed_seconds':None,'entire_task_elapsed_unknown_reason':'No continuous timer spans initial inspection, coding and all retries.',
        'elapsed_since_frozen_plan_seconds':round((datetime.now(timezone.utc)-datetime.fromisoformat(plan['created_utc'])).total_seconds(),3),
        'timeouts':sum(x['reason']=='timeout' for x in js),'truncations':sum(x['reason'] in ('file_size_limit','output_limit') for x in js),
        'memory_limit_failures':sum(x['reason']=='process_memory_limit' for x in js),'cache_invalidations':0,
        'GPU_training_executed':False,'commands_evidence':str(OUT/'java_commands.json')}
    actions=lines(OUT/'commands.jsonl')
    resources['successful_stage_commands']=actions
    resources['development_attempts']=load(OUT/'development_attempts.json')
    replay_seconds=next(r['seconds'] for r in reversed(actions) if r['argv'][-1]=='replay')
    admission_seconds=next(r['seconds'] for r in reversed(actions) if r['argv'][-1]=='admission')
    issues=[
        {'code':'TRANSACTION_CAST_NULL_INVOKE','name':'类型转换及空接收者调用异常未认证','root_confirmed':True,
         'affected_graphs':[ {k:r[k] for k in (*KEY,'test_id')} for r in new if not r['qualified_after']],
         'products':['p000','p003'],'evidence':str(REPORTS/'phase03B4_graph_status.csv'),
         'repair_status':'需另行设计并独立验证，当前不能复用字段异常规则直接放行','product_recovery_upper_bound':2,'realized_recovery':0,
         'world_model_impact':'不支持路径的帧展开及依赖不能作为正确状态转换监督。'},
        {'code':'DEFAULT_SAMPLING','name':'默认16测试采样丢失完整产品内容','root_confirmed':True,'products':['p001','p006'],
         'evidence':str(OUT/'admission.json'),'repair_status':'既有可配置限额已实际验证无损；批处理/分片入口尚未建设',
         'product_recovery_upper_bound':0,'default_lossless_products_restorable_upper_bound':2,
         'world_model_impact':'截断执行前缀会破坏状态历史，应保留全部登记测试及真实顺序。'},
        {'code':'SPLIT_CLASS_COVERAGE','name':'独立版本与合法训练类别/跨系统分区不足','root_confirmed':True,
         'evidence':str(OUT/'readiness'),'repair_status':'须由后续有界、标签盲的数据计划解决；不改变冻结划分',
         'product_recovery_upper_bound':None,'unknown_reason':'未计划或执行新采集，不能承诺恢复数量','world_model_impact':'限制跨版本和跨系统机制检验。'},
        {'code':'STATE_AND_PREFIX','name':'数值状态观测、时间前缀构造及一般库/数组语义不完整','root_confirmed':True,
         'evidence':str(REPORTS/'phase03B4_effect_contract.md'),'repair_status':'当前仅有限调用抽象；无完整状态快照或经过验证的前缀数据入口',
         'product_recovery_upper_bound':None,'unknown_reason':'不是本批次单一可计数产品阻碍','world_model_impact':'T2及预测输入正确性仍不足；不能把事后认证结果提前输入。'},
        {'code':'PORTABLE_EVIDENCE','name':'绝对路径与Windows二进制绑定尚未封装为可迁移证据包','root_confirmed':True,
         'evidence':str(OUT/'framework/binding.json'),'repair_status':'需保持原摘要的相对路径清单与消费者复核；更换采集JVM须重新认证',
         'product_recovery_upper_bound':0,'world_model_impact':'影响Linux上可复核地读取，不提供新状态观测。'}]
    wm={'T1':{'status':'PARTIAL','qualified_graphs':146,'complete_qualified_original_PASS_products':5,
              'reason':'真实产品/测试序列和限定依赖可用，框架内部有480条真实路径证据；不是完整JVM因果模型。'},
        'T2':{'status':'BLOCKED','product_scalar_value_nodes_new_batch':2403,'new_product_numeric_observations':0,
              'new_framework_constant_callbacks':16,'reason':'仅新增真实常量0回调与分支路径证据；无完整参数/堆/输入输出和数值转移配对。'},
        'T3':{'status':'BLOCKED','verified_intervention_pairs':0,'reason':'插桩等价配对不是干预。'},
        'prefix_safe_training_entry':False,'prefix_reason':'保留物理调用/返回和处理器可见时点；尚无验证过的按时刻裁切入口；事后效果/认证/终局结果不得前移。'}
    changed=['src/cwm_fp/lddg/trace.py','src/cwm_fp/lddg/control.py','src/cwm_fp/lddg/extract.py','src/cwm_fp/lddg/effects.py',
        'src/cwm_fp/lddg/control_flow.schema.json','src/cwm_fp/lddg/graphs.schema.json','tests/test_framework_effects.py',
        'tests/java/framework/FrameworkProgram.java','tests/java/framework/FrameworkFixtureTest.java',
        'scripts/verify_phase03B4_framework.py','scripts/test_phase03B4_regression.py','scripts/report_phase03B4.py',
        'AGENTS.md','DATA_CONTRACTS.md','STAGES.md','outputs/phase_status.md']
    report_names=['review_packet.md','review_packet.json','product_status.csv','graph_status.csv','framework_call_analysis.md',
                  'effect_contract.md','semantic_validation.md','training_readiness.md','summary.md']
    changed += ['outputs/phase03B4_'+n for n in report_names]
    recs=[{'priority':1,'task':'建设无损产品读取和可迁移证据包','benefit':'保留19/20测试产品全部内容，并使Linux消费者可验证原始证据','prerequisite':'保持所有摘要和质量门槛，禁止默认采样悄悄截断'},
          {'priority':2,'task':'另阶段认证交易测试的类型转换/空接收者调用语义','benefit':'最多再恢复2个原始通过产品，这是条件上界','prerequisite':'精确异常及帧规则、独立Java边界用例；不利用目标标签选修复'},
          {'priority':3,'task':'制定冻结协议下的两类和跨系统数据补足计划','benefit':'解除真实监督训练分区和类别覆盖阻碍','prerequisite':'独立授权、标签盲选择、有界资源；本阶段不自动执行'}]
    packet={'stage_id':'03B4','stage_status':'COMPLETE_WITH_LIMITATIONS','actual_scope':{'system':'BankAccountTP','new_versions':0,'selected_versions':3,'products':7,'real_rerun_pairs':16,'historical_regression_graphs':132,'artificial_pairs':13,'model_training_runs':0},
        'before_metrics':before,'after_metrics':after,'recovered_graphs':[{k:r[k] for k in (*KEY,'test_id')} for r in recovered],
        'admitted_products':[p for p in product_rows if p['formal_classification_admission']],
        'rejection_reasons':[r for r in graphs if not r['joint_qualified_after']],
        'retained_previous_product_count':1,'total_formal_products':5,'split_readiness':split,'training_readiness':'BLOCKED',
        'world_model_readiness':wm,'tests_executed':{'unit':unit,'real_pairs':16,'real_Java_processes':32,'real_tests_PASS_processes':32,
            'artificial_pairs':13,'artificial_Java_processes':26,'artificial_PASS_processes':24,'expected_FAIL_processes':2,
            'artificial_semantically_supported':6,'artificial_correctly_rejected':7,'pair_parity':29,'fixtures':str(OUT/'synthetic/verification.json')},
        'resources':resources,'changed_files':changed,'artifact_paths':{'evidence_directory':str(OUT),'summary':'outputs/phase03B4_review_packet.md',
            'machine_packet':'outputs/phase03B4_review_packet.json','products':'outputs/phase03B4_product_status.csv','graphs':'outputs/phase03B4_graph_status.csv',
            'calls':str(OUT/'framework_calls.json'),'admission':str(OUT/'admission.json'),'comparison':str(OUT/'comparison.json'),'preservation':str(OUT/'preservation.json'),
            'technical_reports':['outputs/phase03B4_'+n for n in report_names if n.endswith('.md')],
            'current_artifact_index':'outputs/phase03B4_artifacts.json'},
        'unresolved_issues':issues,'recommended_next_stage':recs,'preservation':preservation,'labels_policy':'Only per-protocol authorized train entry after sealed admission; never per-product target labels in public CSV.'}
    write_json(REPORTS/'phase03B4_review_packet.json',packet)
    # Narrative reports use the measured quantities above, with clear scopes.
    metrics=table(['指标（新批次140测试口径）','修复前','修复后'],[
        ['候选图／身份合格图','140／140','140／140'],['控制路径合格图',138,138],['全部依赖语义合格图',124,138],['拒绝图',16,2],
        ['已采全登记套件产品',7,7],['完整语义合格产品',0,5],['正式准入的原始通过产品',0,4],
        ['产品/测试事件',8200,8200],['含框架内部指令的物理事件',8200,8680],['图节点／边','41100／73626','41100／73626'],
        ['控制依赖边',12689,12689],['数据依赖边（含部分图）',8002,8002],['合格图中的数据依赖',7248,8002],
        ['值版本／带真实标量证据的值节点','6428／2403','6428／2403']])
    product_table=table(['产品代号','版本','产品编号','原始结果','登记测试','合格图前→后','完整套件语义','正式准入'],[
        [p['batch_product'],p['bug_version_id'],p['product_id'],p['original_product_result'],p['registered_tests'],
         str(sum(r['qualified_before'] for r in rows if r['batch_product']==p['batch_product']))+'→'+str(p['semantic_qualified_tests']),
         '通过' if p['suite_semantics_valid'] else '拒绝','通过' if p['formal_classification_admission'] else '拒绝'] for p in product_rows])
    splits_table=table(['冻结协议','训练：产品/版本','验证：产品/版本','测试：产品/版本','仅授权训练区类别','结论'],[
        [f,*[str(split[f]['products'][k])+'/'+str(split[f]['independent_versions'][k]) for k in ('train','val','test')],
         dump(split[f]['authorized_train_audit']['train_target_counts']),'阻塞（BLOCKED）'] for f in FOLDS])
    fixture_table=table(['人工用例','原始/插桩结果','物理事件','数据边','语义结论'],[
        [r['case'],r['original_result']['outcome']+'/'+r['instrumented_result']['outcome'],r['physical_events'],
         r['data_edges'] if r['data_edges'] is not None else '未生成：框架区域无法闭合','合格' if r['certified'] else '按预期拒绝'] for r in fixtures])
    code_table=table(['文件','实际改动'],[
        ['trace.py','可选物理行号/返回位置；新语义版本；有限效果前置条件；异常引用保守失效'],
        ['effects.py（新增）','实际日志/二进制复核、30指令路径摘要、产品回调保留、效果及可见时点'],
        ['control.py','消费者重新执行路径/参数/依赖校验，拒绝摘要、质量、事件数或版本伪造'],
        ['extract.py','有限调用证据接入、物理事件与产品/测试投影分别计数'],
        ['两份JSON格式契约','增加新版本及效果证据，保留旧版本'],
        ['一份Python测试、两份Java用例、三个阶段脚本','208项回归、13个人工配对、真实重放/准入/报告'],
        ['AGENTS.md、DATA_CONTRACTS.md、STAGES.md、phase_status.md','统一交接规范、有限效果/前缀边界及当前阶段状态']])
    md('phase03B4_review_packet.md',f'''# CWM-FP阶段03B-4：研究结果交接报告

日期：2026-11-01。本阶段已完成真实实施及验证；正式模型训练仍为**阻塞（BLOCKED）**。研究目标仍是从软件产品线真实执行构建结构化因果世界模型并监督识别假通过/真通过产品，目前没有真实图分类、世界模型或强化学习成绩。

## A. 实际完成情况

复核银行既有3缺陷版本、7产品、140测试，确认16张拒绝图共同调用异常核验函数`org.evosuite.runtime.EvoAssertions.verifyException(String,Throwable)`。已有日志只看见调用/返回，不能证明内部效果；实际JAR含反射/类加载分支，异常栈读取也可能写入异常私有缓存。

选择有限插桩并结合有版本约束的效果摘要：沿用原插桩器，仅把该框架类加入选定类列表；16个测试各执行未插桩/插桩对照，全部保持通过。每次实际走30条内部指令，栈顶类名相等后返回，未走反射分支。仅14次调用同时满足已认证的精确空字段异常前缀；2张交易图仍保守拒绝。

{code_table}

静态提取器、运行器、套件采集器、批次审计器以及阶段04图读取/编码代码均复用，没有改动BRFL、原始产品和测试。真实验证覆盖限定路径；对象/静态字段写后读取、嵌套、重复调用及拒绝边界另由13个人工Java用例验证，不属于真实科研样本。完整异常、一般数组/反射/外部副作用、数值世界模型和时间前缀入口仍未实现；没有把实现或通过检查写成研究有效性。

## B. 前后指标及产品级结果

{metrics}

原16图恢复14图（87.5%），新批次合格率124/140→138/140（88.57%→98.57%）。没有新增产品/测试指令、节点或依赖边；原先处于部分图的**754条数据依赖、886条控制依赖、634个值版本（95个带标量证据）**现在随14图进入可认证范围。新增480条真实框架内部指令保留在原始日志和效果证据中，不伪装为有特征源码映射的产品节点；其中16次常量0回调是真实新增观测，但不是新增产品状态值。

原124图及早期8图的节点、边、事件完全不变。16个重跑测试的921个产品/测试事件保持指令顺序、静态身份、帧及既有依赖；其运行身份摘要因新日志指纹而改变，这是新执行实例的必要变化。8份源码映射原样复制，未生成结构锚点运行事件。两张仍拒绝的图没有被删除。

{product_table}

新增完整语义合格套件5个，其中`p002`为原始失败产品，排除于假通过/真通过二分类；其余4个新增正式准入。加上旧产品后，8个已采全产品中6个语义合格、5个原始通过且正式准入（原始通过产品分母7，5/7；新批次为4/6）。合并148图后身份148、控制146、联合合格146；产品/测试事件8436，含新增框架指令8916，节点41966、边75118、数据依赖8220、带标量值节点2465。没有修改原始138通过/2失败测试结果。

## C. 未解决问题

1. `p000`和`p003`的`main.Transaction_ESTest.test1`仍有类型转换192、空接收者调用异常及连锁帧/操作数缺口，框架效果也因前缀不可信而拒绝。根因已确认；另行严谨修复最多恢复2个原始通过套件，**这是条件上界，实际收益0**。逐条原因和最早事件见图状态CSV、原始日志及效果证据。
2. 默认每产品16测试会使`p001`少3/19测试、`p006`少4/20测试；对应事件丢失约17.18%和24.35%，边丢失约15.79%和21.87%。既有可配置接口已在这5个完整产品上实际验证零测试/事件/边采样损失，不能把默认配置也说成无损。该批合格输入60,139,366字节，未触发128MiB文件门槛；未来扩批仍需分片与产品级聚合。编码中的关系消融按原设计排除时间邻接边，此项不冒充采样损失。
3. 三种冻结协议的类别和独立分区不足，见下表。不能改划分、删除测试或借验证/测试目标补足。
4. 当前效果模型只认证精确JVM空字段异常和指定二进制的已观测相等返回路径；异常缓存、虚拟机内部状态不完整。一般数组、对象别名外部逃逸、反射和其他异常未认证。它们会影响状态转换监督，不能当作无副作用。
5. 当前证据含绝对路径及Windows运行库绑定，Linux可迁移证据封装尚未实现。换采集JVM需要重新审计其效果，不能简单替换摘要放行。各问题的影响、证据和收益上界另见JSON的`unresolved_issues`。

## D. 是否满足真实图分类训练条件

{splits_table}

按缺陷版本分组训练仅1个FP、没有TP；验证虽2产品/2版本，测试2产品实际只1版本。留出银行无训练/验证；留出邮件的授权训练区5产品来自4版本、有FP2/TP3，但没有合格的邮件测试或验证产品。每个协议在独立进程中仅访问自身训练目标文件，准入与修复决策已经冻结；公共CSV不包含目标标签。不能用一个协议的训练标签反向为另一个协议调参。

22列图特征和产品聚合可按实测限额无损读取，未新增类名、方法名、绝对位置、产品编号或真值数值特征。数据工程已有实质进展，但现有冻结协议均不满足正式关系图神经网络实验条件；没有启动任何拟合。

## E. 因果世界模型数据条件

结构事件层（T1）为部分具备：新恢复14图，真实依赖和异常/调用证据可用，但不等于学到了因果机制。数值状态层（T2）仍阻塞：产品标量值节点仍2403，缺完整输入/输出、断言期望与实际值、堆状态和状态转换。框架栈长度、字符串完整内容和布尔返回没有直接标量回调，不能根据分支补造数值；缓存可能写入不能生成虚构写事件。合法干预层（T3）仍无真实执行对，插桩对照不算反事实。

效果证据显式保留物理调用、返回和日志行位置；它只能在返回标记后可见。异常引用以真实处理器时点可见。完整图中的后续依赖、最终断言/结果、整图统计和事后认证不能送入早期预测时间步。目前尚无经验证的时间前缀入口，因此不能直接以完整图开展状态预测损失；数值和干预训练开关继续关闭。

## F. 实验、资源与复现

真实Java：16对、32进程，全部通过且原始/插桩/历史结果一致。人工Java：13对、26进程，12对通过；“异常未发生”用例两端均按预期失败。人工语义6项合格、7项正确拒绝。208项不同单元/回归检查全部通过，0失败、0错误、0跳过，包含上一阶段188项及新增20项，模型拟合入口被显式禁止。

Windows，Python3.11.4，Java8u381，JSON校验库4.17.3、进程监测库5.9.0；模型接口环境为既有PyTorch2.6.0+cu126，未执行GPU训练。58个Java测试进程累计{resources['java_test_process_seconds_sum']}秒，采样最大直接进程驻留{resources['java_peak_sampled_rss_bytes']:,}字节。最终重放{replay_seconds:.3f}秒，产品准入{admission_seconds:.3f}秒，208项回归{unit['seconds']:.3f}秒；这些可能重叠，不能相加冒充全任务时长。初始审计至结束没有连续计时，全任务耗时和Python峰值为未测量。

报告快照新目录{usage['bytes']:,}字节、{usage['files']}文件，磁盘余量{resources['free_disk_bytes']:,}字节；预算为新目录1GiB、最低余量1GiB、单次30秒/2万事件/32MiB输出/512MiB堆/1GiB直接进程驻留，实际无超时、截断或内存上限失败。先行1对测试在正式16对流程中通过绑定缓存复用，没有重复执行。首次提取曾因格式契约尚未接纳新字段而退出1，补充版本化契约后通过；这不是Java采集失败。详见验证报告的开发失败记录。

全部新产物位于`data/processed/lddg/phase03B4_framework_v1/`。原始日志未压缩；每个配对目录有真实命令、退出码、资源采样与指纹。`comparison.json`、`admission.json`、`framework_calls.json`、`unit_tests.log`、`preservation.json`和机器交接包可复核主要结论。修改前源码/契约/状态归档于`producer_before/`；历史正式数据保持原指纹。当前目录不是Git仓库，使用内容摘要记录修改，不声称已有提交。

## G. 下一阶段建议（仅建议，不执行）

1. **优先建设无损产品读取及可迁移证据包**：已有5个正式产品足以验证工程接口，解除默认截断和Linux路径依赖；同时保留现有严格准入。
2. 独立修复两张交易图的类型转换/空调用异常，条件收益最多2个通过产品。当前训练区缺少的另一类不能靠猜标签选择修复。
3. 在新授权阶段制定标签盲、资源有界的数据补足计划，满足冻结协议两类与跨系统分区，再讨论真实GPU训练。

当前可准备服务器上的依赖与证据迁移方案，**尚不应启动正式GPU模型训练**。结构世界模型只具备部分数据基础；数值状态世界模型和反事实推理条件不足。本阶段结束，等待下一条研究指令。

文档整理：本次要求的技术报告各有独立用途，既有历史记录又被指纹引用；没有找到可安全删除的无用文档，未删除历史证据。`phase03B4_summary.md`只作交接包索引，不重复维护另一份长总结。
''')
    md('phase03B4_framework_call_analysis.md',f'''# 阶段03B-4：测试框架调用分析

依据实际JAR字节码、旧冻结日志及本次16对Java执行，不依据方法名称猜测纯函数。对应JAR为`third_party/existing_instrumenter/lib/evosuite-standalone-runtime-1.0.6.jar`；完整二进制摘要和运行库绑定见新目录`framework/binding.json`与每次`status.json`。

所有调用均为操作码184，目标`org.evosuite.runtime.EvoAssertions.verifyException`，描述符`(Ljava/lang/String;Ljava/lang/Throwable;)V`，调用者均为下表测试方法，描述符`()V`。参数先由真实字符串常量装载，再由处理器局部变量装载异常引用；14次精确异常来源已认证，交易2次前缀不可认证。字符串回调只有类型，没有完整内容；摘要不补造运行时字符串。

{table(['产品','测试','调用字节码位置','新日志调用/返回行','参数值版本','路径/效果认证'],[[r['batch_product'],r['test_id'],r['bci'],str(r['log_line'])+'/'+str(r['return_log_line']),','.join(r['argument_value_ids']),'相等返回／'+('合格' if r['accepted'] else '拒绝：'+str(r['reason']))] for r in callrows])}

完整系统、版本、产品、调用者及日志路径在`framework_calls.json`，产品代号映射见交接包。全部16次正常返回、无标量返回值；每次有30条真实内部指令。`verifyException`先检查两类框架异常，当前分支均未触发重抛，再调用`assertThrownBy`。后者调用`Throwable.getStackTrace()`、检查数组非空、读第0项、读取类名并调用`String.equals`，观察到相等分支早返回。它接收已经捕获的异常，不是在这条路径上执行被测操作，也不是核验异常类型的通用断言。

静态可能的其他路径包括：特殊框架异常重抛、空栈/空类名返回、不相等后的类加载、注解反射、父类检查和抛出断言错误。旧日志没有这些分支的内部指令，旧正常返回不足以判断路径。本次真实16例均未进入这些路径；人工错误来源和回调用例证明拒绝分支不能直接放行。

效果分解：调用者局部值只被读取，无非局部局部变量写入；已认证的产品对象字段和静态字段不被该闭合早返回路径修改；异常内部栈缓存可能写入，可能分配/写入新栈数组；虚拟机内部状态、缓存是否实际物化仍未知。没有观测到集合或产品回调，不等于所有未插桩调用均无副作用。精确JVM异常保证不分派到自定义`getStackTrace`覆盖方法；自定义异常人工用例确实执行了产品回调并改变静态字段，整图正确拒绝。未来对已交给框架的异常堆引用继续保守拒绝。

原插桩器按选定类工作，原集合不含该框架类，所以只有调用及返回标记。本阶段复用相同agent，只增加这一类；该类没有静态初始化器，实际仅执行两个必要方法。没有复用BRFL推断的因子图值或自动真值。只读参考文件包括`ByteCodeGraph.java`、`TraceChunk.java`、`ParseInfo.java`、`TraceTransformer.java`、`Interpreter.java`、`opcode/InvokeInst.java`及`SplGraphBuilder.java`。其调用/返回标记与参数栈能力被复用；未跟踪调用的近似因素构造和“存储即捕获”的宽松匹配不是本研究认证证据。

官方说明用于限定语义，最终证据仍为本机绑定二进制：异常栈接口允许栈信息不完整，因此不能凭抛出位置推测真实栈内容，见[Java8异常接口文档](https://docs.oracle.com/javase/8/docs/api/java/lang/Throwable.html#getStackTrace--)；空接收者字段访问的异常规则见[Java8指令规范](https://docs.oracle.com/javase/specs/jvms/se8/html/jvms-6.html#jvms-6.5.getfield)。[BRFL公开仓库](https://github.com/Songluhaining/BRFL)仅作来源参考，没有拉取覆盖本地工程。
''')
    md('phase03B4_effect_contract.md','''# 阶段03B-4：有限外部调用效果契约

选择方案：旧日志的静态摘要（方案A）不足以排除反射路径；有限真实插桩（方案B）复用现有agent、只添加一个类，成本16对真实测试；结合实际路径和二进制限定的摘要，未满足前提时执行保守拒绝（方案C）。恢复5套件/4通过产品原本只是条件上界，最终收益由执行与消费者核验确定。

语义版本`framework_effect_v1`，具体摘要`evosuite_1_0_6_observed_equal_path_v1`。目标、描述符、调用位置及每次参数值版本见`framework_calls.json`。固定指纹：JAR `231b939688bf69603374e8698ec2c2c4b6ca3aa030d1bcc7fa5073da4ee0a2eb`；EvoAssertions类 `dd9da3bb59b6cc8986d1b3cfe5c21de53e11d4c8e685d42e3521c6ebfdb3abc2`；运行库 `6eab9457e892290ffb7876da77855d082e4cdd3a548bbb83204624d4d6cb22b1`。虚拟机、启动器、agent也由实现固定摘要并复核实际文件，不接受任意版本。

必须同时满足：完整配对与历史一致；单线程；产品/测试的前缀无语义缺口；异常是从真实空字段规则和精确处理器得到的JVM空指针异常，尚未逃逸；参数0有真实字符串常量定义；框架方法入口、正常返回标记及全部原始字节码位置匹配；内部恰为已审计的30条相等返回路径；实际二进制、运行绑定与日志摘要全部一致。

读取集合：字符串参数、异常引用及其栈元信息。返回为void，没有返回值定义和数值观测。真实路径中的三个JDK调用正常返回；非当前路径可能重抛或断言失败，全部拒绝。当前闭合路径无产品回调、无产品对象/静态字段写入；Throwable字节码证明可能写自身stackTrace并写入新数组，故不是纯函数。摘要只记录“可能写”，不伪造实际写事件、数组元素值或缓存状态；异常引用在调用后保守失效。虚拟机内部栈元信息和类解析等状态保持未知，不参与数值状态训练。

产品/测试图保持完整序列；框架内部指令保留在未压缩原始日志和独立效果证据。投影只识别明确的框架类，产品回调事件不会被移除。没有把库数组指令认证成通用数组操作，没有为库指令伪造源码身份。无正常返回的框架区域、错误来源反射路径、自定义异常回调、重复使用已逃逸异常均拒绝。

消费者通过`control_evidence_errors`读取绑定原始日志和二进制，再执行路径校验与动态解释，重算参数定义、异常来源、效果、依赖端点和时序。摘要状态、质量布尔值、内部事件数或自重新计算摘要后的伪造投影均不能单独放行。旧版本未改变语义；20项新增回归包含伪造/缺失证据反向检查。格式契约扩展保留旧版本，历史图不被覆盖。

可见时点：方法效果仅在真实返回日志行后成立；`physical_call_event`与`physical_return_after_event`保留真实边界。异常引用以实际处理器为可见起点。当前完整图仅供离线图结构学习，不能直接转换为早期预测输入；尚未实现安全时间前缀构造。库路径的实际常量0回调是观测，不把分支成立推算出的栈长度/布尔值包装成直接数值测量。

适用范围没有通过更换名称自动扩展到其他方法、JAR或JVM。当前消费者需能访问绑定证据路径；Linux迁移需保持原摘要的路径清单/证据包，未来更换采集环境须独立认证。原图22列节点数值表示未改变，具体身份和效果文字仅用于审计。
''')
    md('phase03B4_semantic_validation.md',f'''# 阶段03B-4：实际验证记录

完整结果见交接包；本报告记录验证边界、命令和失败尝试。

{metrics}

148张候选图经过现有格式与质量检查。原124+8图的节点/边/事件完全一致；新增物理观察只在16个重跑测试中，原产品/测试顺序、标量回调（对象哈希不作身份）、静态身份、调用帧、空字段异常转移和全部依赖边不变。921个运行事件摘要因新日志改变，所有静态身份保持。回归消费者对14张恢复图接受，2张交易图拒绝；不存在删测试换取完整套件。

人工Java验收（全部完成并配对一致）：

{fixture_table}

`stateAfter`实际验证产品对象字段最大整数、静态字段最小整数在框架调用前写入后仍正确读取；`callback`中自定义异常栈方法真实写入静态状态91，测试通过但依赖语义拒绝；`reflection`同样保留真实反射回调并拒绝。重复调用产生3个不同异常和执行身份；同一异常反复交给框架因状态已失效而拒绝。错误来源用例的未正常返回区域保留原日志，提取明确拒绝，数据边数量为null，不填0冒充测量。

208项不同单元/回归通过，0失败/错误/跳过。其中旧188项逐名复用，新20项涵盖二进制/运行绑定、投影/返回/参数伪造、质量与内部计数伪造、依赖时序篡改、状态缺失、嵌套/重复/回调、未支持路径。日志`data/processed/lddg/phase03B4_framework_v1/unit_tests.log`，机器列表同目录`unit_test_results.json`。阶段04回归只作接口和未训练网络计算，拟合函数被禁止。

实际命令（项目根目录执行；未重新跑已封存的132个真实测试）：

```powershell
D:/SoftInstall/python.exe scripts/verify_phase03B4_framework.py prepare
D:/SoftInstall/python.exe scripts/verify_phase03B4_framework.py pilot
D:/SoftInstall/python.exe scripts/verify_phase03B4_framework.py binaries
D:/SoftInstall/python.exe scripts/verify_phase03B4_framework.py collect
D:/SoftInstall/python.exe scripts/verify_phase03B4_framework.py fixtures
E:/Myself/投稿/work7/FP-SPL/.venv/Scripts/python.exe -X utf8 scripts/verify_phase03B4_framework.py replay
E:/Myself/投稿/work7/FP-SPL/.venv/Scripts/python.exe -X utf8 scripts/test_phase03B4_regression.py
E:/Myself/投稿/work7/FP-SPL/.venv/Scripts/python.exe -X utf8 scripts/verify_phase03B4_framework.py admission
D:/SoftInstall/python.exe -X utf8 scripts/verify_phase03B4_framework.py labels --fold grouped
D:/SoftInstall/python.exe -X utf8 scripts/verify_phase03B4_framework.py labels --fold loso_BankAccountTP
D:/SoftInstall/python.exe -X utf8 scripts/verify_phase03B4_framework.py labels --fold loso_Email
D:/SoftInstall/python.exe -X utf8 scripts/report_phase03B4.py
```

阶段成功入口退出码0，所有Java原始参数、输出及退出码由运行器保存在各`*.command.json`。58个测试进程中56个退出0、2个退出1（人工“异常未发生”预期失败），不是采集故障。全部29对一致，无超时、截断、内存上限失败。先行测试在collect阶段缓存命中；缓存绑定源/类/库/agent/JVM/选定类/限额，未静默重用旧插桩范围。

开发尝试失败：第一次replay在新字段`framework_effects`尚未加入JSON格式时退出1，未完成输出；随后追加独立语义版本的格式分支并重放成功。该次完整异常输出仅在会话工具结果中，未保存本地，不能声称有完整本地日志或耗时测量。两个只读探针还出现PowerShell通配路径不匹配及默认字符编码错误，改为实际路径/UTF-8后完成，不影响Java结果。失败摘要另存`development_attempts.json`；未覆盖原始失败产品/测试记录。

当前工作区不是Git仓库。修改前文件归档、旧数据及只读源码校验见`preservation.json`；当前内容清单见`outputs/phase03B4_artifacts.json`。严格缓存目录内没有加入提取图文件，图/报告保存在独立目录，支持复核。
''')
    md('phase03B4_training_readiness.md',f'''# 阶段03B-4：训练准备结论

结论：**阻塞（BLOCKED）**，没有拟合模型。单图认证、完整套件认证、无损读取、合法分区及两类监督覆盖分开检查。

{product_table}

{splits_table}

恢复14图后，5个新套件语义完整；其中1个原始失败，不进入FP/TP目标。4个新通过产品+旧1个正式准入，共5个、4个独立版本、75张完整测试图。按版本分组训练只有1个FP、没有TP，不能训练二分类；留出银行没有训练/验证，留出邮件没有验证/测试。只有相应协议的训练文件在准入冻结后由授权入口读取，验证/测试标签未用于修复、选样或模型研发；公共产品/图CSV没有目标标签。

合格产品合并输入60,139,366字节，低于现有128MiB文件门槛。五产品在既有可配置接口上全部通过22列无采样读取，测试/事件/边丢失均0；默认配置只让旧产品及p004、p005三者无损，p001与p006各丢3/19和4/20测试。现有接口可以显式配置完整限额，不需关闭准入；下一阶段应防止入口默认采样，建立无损分片与产品聚合，不在本阶段改模型或降低质量门槛。

剩余语义：交易2图的类型转换/空接收者异常，未认证的一般外部副作用、数组、反射/回调及通用异常；后者仅在人工反例观察并保守拒绝。精确框架摘要不是通用纯函数。数值状态缺失和时间前缀裁切未解决，T1部分可用，T2/T3未满足训练要求。

Linux RTX3090Ti：本阶段未测服务器或显存，也没有GPU成绩。当前数据约束先于算力；需先补无损读取、证据迁移及合法分区/类别。代码消费者含Windows二进制摘要及绝对路径，不能直接宣布复制后即可训练。可准备迁移方案，不能启动正式分类或世界模型训练。
''')
    md('phase03B4_summary.md','# 阶段03B-4完成总结\n\n独立、完整中文总结统一维护在[阶段结果交接报告](phase03B4_review_packet.md)。机器指标见同名JSON，产品及测试图状态见`phase03B4_product_status.csv`和`phase03B4_graph_status.csv`。\n\n本阶段恢复14/16图，新增4个原始通过完整产品正式准入，连同旧产品共5个；208项回归通过。正式训练仍因类别/分区不足、默认采样及迁移准备而阻塞。未训练模型，已停止等待后续指令。\n')
    md('phase_status.md','''# 当前阶段状态：03B-4有限测试框架效果认证

2026-11-01。本阶段实际实现和验证完成，等待下一条研究指令。完整中文总结：[phase03B4_review_packet.md](phase03B4_review_packet.md)；机器指标、产品/图完整身份及拒绝理由见同名JSON和两份状态CSV。

银行3版本7产品140测试，16对真实配对重跑一致；14图恢复、2张交易图仍拒绝，联合合格124→138。新增5个完整语义套件中1个原始失败排除，4个原始通过正式准入，加旧1个共5个。原124图及旧8图保持；208项回归通过。人工13对中12对通过、1对预期失败，6种受支持、7种正确拒绝。

新版本明确区分库调用边界及物理事件：产品/测试事件8200不变，新增框架内部480真实指令；图节点41100/边73626不变，754条原部分图数据依赖进入认证范围。未知缓存/数值状态不填造，异常引用在调用后保守失效，消费者读取原日志及二进制重放验证。

正式训练仍阻塞：分组训练1个FP无TP；两种跨系统协议缺相应分区。五产品可配置无损读取已验证，默认16测试仍截断两个产品；128MiB文件限制本批未触发。时间前缀入口、完整状态、反事实及Linux证据迁移均未完成。没有模型拟合、GPU训练或后续阶段执行。

实际入口：`scripts/verify_phase03B4_framework.py`的prepare、pilot、binaries、collect、fixtures、replay、admission、labels；`scripts/test_phase03B4_regression.py`；`scripts/report_phase03B4.py`。完整命令/退出码/失败尝试见`phase03B4_semantic_validation.md`，实际Java命令见新证据目录`java_commands.json`，回归日志`unit_tests.log`。

新产物：`data/processed/lddg/phase03B4_framework_v1/`，仅项目内写入。修改前状态与源码位于该目录`producer_before/`；旧冻结数据和原始源码指纹复核通过。未删除历史证据。下一步建议先建设无损产品读取及可迁移证据包，不自动执行。
''')
    # Standard parser and full-key/count checks for public handoff files.
    parsed=load(REPORTS/'phase03B4_review_packet.json');assert parsed['total_formal_products']==5
    for name,count,keys in [('phase03B4_product_status.csv',8,KEY),('phase03B4_graph_status.csv',148,(*KEY,'test_id'))]:
        with (REPORTS/name).open(encoding='utf-8-sig',newline='') as f:rr=list(csv.DictReader(f))
        assert len(rr)==count and len({tuple(r[k] for k in keys) for r in rr})==count
        assert not any('label' in k or k in ('FP','TP') for k in rr[0])
    write_json(OUT/'handoff_validation.json',{'json_parse':True,'product_rows':8,'graph_rows':148,'unique_full_keys':True,'target_labels_in_public_CSV':False})
    print('Rendered validated handoff: 8 products, 148 graphs, 14 recovered, 5 total formal products',flush=True)

if __name__=='__main__':main()
