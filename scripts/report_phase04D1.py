"""One handoff, two public status tables; no target reads or model execution."""
import csv
from collections import Counter,defaultdict
from pathlib import Path
import json
import shutil
import statistics
import sys
import re
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cwm_fp.data.common import KEY,key,sha256,write_json,read_csv
from cwm_fp.lddg.cohort import OUT,OLD,PROTOCOL,read_plan,input_rows
from cwm_fp.lddg.suite import load,lines
from cwm_fp.lddg.runner import tree_usage
from cwm_fp.models.graph_encoder import evidence_errors
from cwm_fp.baselines.boundaries import install_io_guard

def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def dist(xs):return {'count':len(xs),'min':min(xs) if xs else None,'median':statistics.median(xs) if xs else None,'max':max(xs) if xs else None,'sum':sum(xs)}
def public_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def j(x):return json.dumps(x,ensure_ascii=False,sort_keys=True)

def main():
    install_io_guard('features');plan=read_plan();ps,ss,ts=input_rows();admission=load(OUT/'admission.json');package=load(OUT/'package/manifest.json')
    validation=load(OUT/'package_validation.json');targets=load(OUT/'readiness/grouped_train.json');seal=load(OUT/'cohort_seal.json')
    for p,h in seal['files_sha256'].items():assert sha256(Path(p))==h,p
    assert targets['cohort_seal_sha256']==sha256(OUT/'cohort_seal.json')
    adm={tuple(r['identity']):r for r in admission['products']};packed={tuple(p['identity']) for p in package['products']}
    graphrows=[];productrows=[];testclasses=defaultdict(Counter);node_totals=Counter();state_counts=Counter();opcode_gaps=defaultdict(list)
    specs=[(p,False) for p in plan['products']]
    for p in plan['reused_products']:
        specs.append(({**p,'index':None,'partition':ss[tuple(p['identity'])]['split'],'tests':[{'test_id':x,'result':'PASS'} for x in p['registered_tests']]},True))
    for p,reused in specs:
        k=tuple(p['identity']);entry=adm.get(k,{});work=OUT/p.get('directory','')
        state=load(work/'product_status.json') if not reused and (work/'product_status.json').exists() else {}
        records={r['test_id']:r for r in load(work/'collection.json')['records']} if not reused and (work/'collection.json').exists() else {}
        graphpath=Path(entry['graph_path']) if entry.get('graph_path') else work/'g/graphs.jsonl'
        gs={}
        if graphpath.exists():
            with graphpath.open(encoding='utf-8') as f:
                for line in f:
                    try:g=json.loads(line)
                    except json.JSONDecodeError:break
                    gs[g['test_id']]=g
        qs={r['test_id']:r for r in lines(work/'g/test_quality.jsonl')} if not reused and (work/'g/test_quality.jsonl').exists() else {}
        classlist=Counter();qualified=0;complete=0;consistent=0;collected=0;pr_reasons=set(entry.get('reasons',[]));sizes=Counter()
        for test in p['tests']:
            tid=test['test_id'];clazz=tid.rsplit('.',1)[0].rsplit('.',1)[-1];classlist[clazz]+=1
            r=records.get(tid);g=gs.get(tid);q=g['quality'] if g else {};sr=r['status'] if r else None
            done=bool(reused or sr and sr['complete']);parity=bool(reused or sr and sr['parity'] and sr['matches_historical_outcome']);collected+=int(reused or r is not None)
            complete+=done;consistent+=done and parity
            errors=evidence_errors(g) if g else ['graph_missing']
            if not reused and not state.get('extraction_complete'):errors+=['unsealed_or_failed_extraction']
            if not done:errors+=['execution_incomplete']
            if not parity:errors+=['execution_parity_or_historical_mismatch']
            if qs.get(tid,{}).get('error'):errors+=[qs[tid]['error']]
            pr_reasons.update(errors)
            ok=not errors;qualified+=ok
            gaps=q.get('unsupported_dynamic',[]);byid={n['id']:n for n in g['nodes']} if g else {}
            enriched=[]
            for gap in gaps:
                attrs=byid.get('e:'+str(gap.get('event_index')),{}).get('attrs',{})
                item={**gap,**{f:attrs.get(f) for f in ('opcode','composed_class','composed_method','descriptor','bytecode_offset')}}
                enriched.append(item)
                if gap['reason']=='unsupported_array_type_or_monitor_opcode':
                    opcode_gaps[str(attrs.get('opcode'))].append({'identity':list(k)+[tid],'evidence':str(graphpath),'first_unhandled_event':item})
            rr={**dict(zip(KEY,k)),'test_id':tid,'partition':p['partition'],'test_class':clazz,'reused':reused,
                'registered_result':'PASS','execution_record_present':bool(reused or r),'paired_complete':done,'parity_and_historical':parity,
                'original_result':'PASS' if reused else sr['runs']['original']['result']['outcome'] if sr else 'NOT_STARTED',
                'instrumented_result':'PASS' if reused else sr['runs']['instrumented']['result']['outcome'] if sr else 'NOT_STARTED',
                'graph_generated':bool(g),'identity_complete':q.get('program_identity_complete'),'dependency_semantics_valid':q.get('dependency_semantics_valid'),
                'qualified_graph':ok,'in_packaged_product':k in packed,'nodes':len(g['nodes']) if g else None,'edges':len(g['edges']) if g else None,
                'events':q.get('events'),'physical_events':q.get('physical_trace_events'),'observed_scalar_events':q.get('observed_scalar_events'),
                'rejection_reason_codes':j(sorted(set(errors))),'unsupported_dynamic':j(enriched),'first_dynamic_gap':j(enriched[0]) if enriched else '',
                'execution_evidence':str(Path(r['directory'])/'status.json') if r else str(OLD/'integrity_verification.json') if reused else '',
                'graph_evidence':str(graphpath),'quality_evidence':str(work/'g/test_quality.jsonl') if not reused else str(OLD/'verification.json')}
            graphrows.append(rr)
            cl=testclasses[('old' if reused else 'new',clazz)]
            cl.update(registered=1,collected=int(reused or r is not None),paired_complete=int(done),graphs=int(bool(g)),qualified=int(ok),in_packaged_products=int(k in packed))
            cl['rejected']+=int(not ok)
            if g:
                sizes.update(nodes=len(g['nodes']),edges=len(g['edges']),events=q['events'])
                if not reused:
                    node_totals.update(nodes=len(g['nodes']),edges=len(g['edges']),events=q['events'])
                    node_totals.update({t:sum(e['type']==t for e in g['edges']) for t in ('control_dep','data_dep','temporal')})
                    vals=[n for n in g['nodes'] if n['type']=='value'];state_counts.update(value_nodes=len(vals),observed_value_nodes=sum(n['attrs'].get('observed') is True for n in vals),observed_scalar_events=q.get('observed_scalar_events',0),physical_events=q.get('physical_trace_events',q['events']),
                        assertion_nodes=sum(n['type']=='assertion' for n in g['nodes']),complete_assertion_values=sum(n['type']=='assertion' and n['attrs'].get('expected_actual_values_complete') is True for n in g['nodes']),
                        original_test_result_nodes=sum(n['type']=='output' and n['attrs'].get('kind')=='original_test_result' for n in g['nodes']))
        registered=len(p['tests']);semantic=qualified==registered
        if entry.get('eligible'):assert semantic,(k,qualified,registered)
        extraction_reason=(state.get('extraction_execution') or {}).get('reason')
        if state.get('error'):pr_reasons.add(state['error'])
        if extraction_reason:pr_reasons.add(extraction_reason)
        if not reused and not state and not records:pr_reasons.add('not_started')
        capacity=[r['reason'] for r in validation['failures'] if r['identity']==list(k)]
        pr_reasons.update(capacity)
        productrows.append({**dict(zip(KEY,k)),'partition':p['partition'],'execution_index':p['index'],'reused':reused,'original_result':'PASS',
            'registered_tests':registered,'collected_tests':collected,'complete_pairs':complete,'consistent_pairs':consistent,'generated_graphs':len(gs),'qualified_graphs':qualified,
            'started':reused or bool(state or records),'complete_execution_suite':complete==registered,'complete_consistent_suite':consistent==registered,'complete_semantic_suite':semantic,
            'formal_data_admission':bool(entry.get('eligible')),'lossless_packaged':k in packed,'rejection_reason_codes':j(sorted(pr_reasons)),
            'resource_stop_reason':extraction_reason or (load(OUT/'collection_summary.json')['stop_reason'] if not state and not reused else ''),
            'cache_reused_tests':registered if reused else sum(r.get('cached',False) for r in records.values()),'test_class_counts':j(dict(classlist)),
            'suite_nodes':sizes['nodes'],'suite_edges':sizes['edges'],'suite_events':sizes['events'],'features_json':ps[k]['features_json'],
            'test_loss_fraction':0 if k in packed else None,'node_loss_fraction':0 if k in packed else None,'edge_loss_fraction':0 if k in packed else None,'event_loss_fraction':0 if k in packed else None,
            'evidence_path':str(OLD/'verification.json') if reused else str(work/'product_status.json'),'admission_evidence':str(OUT/'admission.json'),'package_evidence':str(OUT/'package/manifest.json') if k in packed else ''})
    new=[r for r in productrows if not r['reused']];old=[r for r in productrows if r['reused']]
    def funnel(rows,candidates):return {'candidate_products':len(candidates),'candidate_versions':len({tuple(r['identity'][:2]) for r in candidates}),
        'selected_products':len(rows),'selected_versions':len({(r['system_id'],r['bug_version_id']) for r in rows}),
        **{name:sum(r[field] for r in rows) for name,field in [('started_products','started'),('complete_execution_products','complete_execution_suite'),('complete_consistent_products','complete_consistent_suite'),('complete_semantic_products','complete_semantic_suite'),('admitted_products','formal_data_admission'),('packaged_products','lossless_packaged')]}}
    fun={s:funnel([r for r in new if r['partition']==s],[r for r in plan['candidate_pool'] if r['partition']==s]) for s in ('train','val','test')}
    perversion={v:funnel([r for r in new if r['bug_version_id']==v],[r for r in plan['candidate_pool'] if r['identity'][1]==v]) for v in sorted({r['bug_version_id'] for r in new})}
    ready={s:{'products':sum(r['lossless_packaged'] and r['partition']==s for r in productrows),
        'versions':len({r['bug_version_id'] for r in productrows if r['lossless_packaged'] and r['partition']==s})} for s in ('train','val','test')}
    goals={'train_products':ready['train']['products']>=12,'train_versions':ready['train']['versions']>=5,
        'FP_products':targets['counts'].get('FP',0)>=3,'TP_products':targets['counts'].get('TP',0)>=3,
        'FP_versions':targets['class_version_counts']['FP']>=2,'TP_versions':targets['class_version_counts']['TP']>=2,
        'val_products':ready['val']['products']>=4,'val_versions':ready['val']['versions']>=2,'test_products':ready['test']['products']>=4,'test_versions':ready['test']['versions']>=2,'zero_loss':validation['zero_loss']}
    small=all(goals.values());completed=all(r['complete_execution_suite'] for r in new)
    bias={}
    for category,rows in [('accepted',[r for r in new if r['formal_data_admission']]),('rejected',[r for r in new if r['started'] and not r['formal_data_admission']]),('not_started',[r for r in new if not r['started']])]:
        feature_counts=Counter()
        for r in rows:feature_counts.update({k:int(v) for k,v in json.loads(r['features_json']).items()})
        bias[category]={'products':len(rows),'suite_sizes':dist([r['registered_tests'] for r in rows]),'graph_nodes_per_product':dist([r['suite_nodes'] for r in rows]),
            'graph_edges_per_product':dist([r['suite_edges'] for r in rows]),'enabled_feature_counts':dict(feature_counts),'configuration_constraints_verified':False}
    reason_counts=Counter(reason for r in graphrows if not r['reused'] for reason in json.loads(r['rejection_reason_codes']))
    class_reasons={clazz:dict(Counter(reason for r in graphrows if not r['reused'] and r['test_class']==clazz for reason in json.loads(r['rejection_reason_codes']))) for _,clazz in testclasses}
    newg=[r for r in graphrows if not r['reused']]
    first_reason_counts=Counter(json.loads(r['first_dynamic_gap'])['reason'] for r in newg if r['first_dynamic_gap'])
    order=[int(m.group(1)) for m in re.finditer(r'^PRODUCT_DONE (\d+) ',(OUT/'collect.log').read_text(encoding='utf-8'),re.M)]
    expected_order=[p['index'] for p in plan['products']]
    pair_order=[(int(m.group(1)),m.group(2)) for m in re.finditer(r'^PAIR (\d+) \d+ (\S+) ',(OUT/'collect.log').read_text(encoding='utf-8'),re.M)]
    expected_pairs=[(p['index'],t['test_id']) for p in plan['products'] for t in p['tests']]
    sequence={'completed_products':order,'product_order_matches_plan_prefix':order==expected_order[:len(order)],
        'pair_log_records':len(pair_order),'pair_order_matches_plan_prefix':pair_order==expected_pairs[:len(pair_order)],
        'plan_replacement_count':0}
    assert sequence['product_order_matches_plan_prefix'] and sequence['pair_order_matches_plan_prefix'],sequence
    commands=[{'path':str(f),**load(f)} for f in sorted(OUT.rglob('*.command.json')) if 'producer_before' not in f.parts]
    testcmd=[c for c in commands if 'Phase03TestRunner' in c.get('argv',[])]
    launched=[c for c in testcmd if c.get('resource',{}).get('launched')]
    peak=max((c['resource'].get('peak_sampled_rss_bytes') or 0 for c in launched),default=0)
    resources={'budget':plan['config'],'collection_budget':load(OUT/'collection_budget.json'),'collection_controller':load(OUT/'collect.command.json'),
        'snapshot_utc':datetime.now(timezone.utc).isoformat(),
        'test_processes_launched':len(launched),'test_processes_registered':len(testcmd),'paired_test_attempts':len(testcmd)//2,'retries':0,
        'test_process_seconds_sum':sum(c['seconds'] for c in launched),'peak_sampled_Java_RSS_bytes':peak or None,
        'memory_scope':'Direct Java process sampled by existing runner,50ms. Not guaranteed instantaneous/system peak. Torch packaging launcher RSS is not model memory.',
        'Python_packaging_peak_RSS_bytes':None,'Python_memory_reason':'Use separately named OS process-lifetime peak working set from the actual tensor worker; do not use launcher RSS.',
        'tensor_worker_resources':{a:load(OUT/(a+'_self_resource.json')) for a in ('admission','package')},
        'generated_directory_usage':tree_usage(OUT),'disk_free_bytes':shutil.disk_usage(OUT).free,'package_bytes':tree_usage(OUT/'package')['bytes'],
        'resource_reasons':dict(Counter(c['reason'] for c in commands if c.get('reason'))),'GPU_runs':0,'training_runs':0,
        'all_commands_evidence':str(OUT/'command_index.json'),'server_access':load(OUT/'server_access.json')}
    resources['environment']=load(OUT/'environment.json')
    resources['total_agent_wall_seconds']=None
    resources['total_agent_wall_reason']='Only recorded command durations measured; planning, inspection and reporting wall time not independently timed.'
    write_json(OUT/'command_index.json',commands)
    old_manifest=load(OLD/'package/manifest.json');old_refs=[g for p in old_manifest['products'] for g in p['tests']]
    before={'admitted_products':len(old_manifest['products']),'versions':len({tuple(p['identity'][:2]) for p in old_manifest['products']}),'graphs':len(old_refs),
        **{f:sum(g[f] for g in old_refs) for f in ('events','nodes','edges')},
        'control_dep':sum(g['relations']['control_dep'] for g in old_refs),'data_dep':sum(g['relations']['data_dep'] for g in old_refs),
        'grouped':{s:{'products':sum(p['partitions']['grouped']==s for p in old_manifest['products']),'versions':len({tuple(p['identity'][:2]) for p in old_manifest['products'] if p['partitions']['grouped']==s})} for s in ('train','val','test')}}
    after={'new_selected_versions':len({r['bug_version_id'] for r in new}),'new_selected_products':len(new),'new_registered_tests':sum(r['registered_tests'] for r in new),
        'new_complete_pairs':sum(r['complete_pairs'] for r in new),'new_consistent_pairs':sum(r['consistent_pairs'] for r in new),'new_candidate_graphs':sum(r['generated_graphs'] for r in new),
        'new_qualified_graphs':sum(r['qualified_graphs'] for r in new),'new_admitted_products':sum(r['formal_data_admission'] for r in new),
        'new_identity_valid_graphs':sum(r['identity_complete'] is True for r in newg),'new_dependency_valid_graphs':sum(r['dependency_semantics_valid'] is True for r in newg),
        'new_admitted_versions':len({r['bug_version_id'] for r in new if r['formal_data_admission']}),'old_reused_products':5,'old_reverified_graphs':validation['old_graphs_reverified'],
        'total_packaged_products':len(packed),'total_packaged_versions':len({k[:2] for k in packed}),'total_packaged_graphs':len(validation['graphs']),
        'new_graph_totals':dict(node_totals),'package_totals':{f:sum(g[f] for g in validation['graphs']) for f in ('nodes','edges','events')},'zero_loss':validation['zero_loss'],
        'new_semantic_rejection_products':sum(r['complete_execution_suite'] and not r['complete_semantic_suite'] for r in new),'new_resource_incomplete_products':sum(not r['complete_execution_suite'] for r in new)}
    after['packaged_control_dep']=sum(g['relations']['control_dep'] for g in validation['graphs'])
    after['packaged_data_dep']=sum(g['relations']['data_dep'] for g in validation['graphs'])
    state_counts['missing_value_nodes']=state_counts['value_nodes']-state_counts['observed_value_nodes']
    absent_features=[f for f,n in bias['rejected']['enabled_feature_counts'].items() if n and not bias['accepted']['enabled_feature_counts'].get(f)]
    unsupported_classes=[cls for (scope,cls),n in testclasses.items() if scope=='new' and n['registered'] and not n['in_packaged_products']]
    root_cause={'first_dynamic_reason_graph_counts':dict(first_reason_counts),'unhandled_opcode_evidence':dict(opcode_gaps),
        'interpretation':'Opcode192 is a CHECKCAST type conversion, not evidence of an array operation. Later stack/heap/call/exception gaps may cascade; correcting one opcode is not proof all remaining semantics are supported.',
        'no_semantic_repair_performed':True,'potential_product_recovery':None,'unknown_reason':'No repair/validation permitted in04D1; do not infer recoverable suites from earliest gap alone.'}
    loso_counts={f:{s:{'products':sum(p['partitions'][f]==s for p in package['products']),'versions':len({tuple(p['identity'][:2]) for p in package['products'] if p['partitions'][f]==s})} for s in ('train','val','test')} for f in package['split_protocols'] if f!='grouped'}
    statuses={'COLLECTION_PLAN_EXECUTED':'COMPLETE' if completed else 'PARTIAL','LOSSLESS_DATA_READY':'READY' if len(packed)==sum(r['formal_data_admission'] for r in productrows) else 'PARTIAL',
        'SMALL_PILOT_DATA_READY':'PRELIMINARY_READY' if small else 'BLOCKED','TRAINING_INTERFACE_READY':'PARTIAL','BACKWARD_PASS_VERIFIED':'NOT_VERIFIED','LINUX_GPU_VERIFIED':'NOT_VERIFIED','GENERALIZATION_EVIDENCE_AVAILABLE':'NOT_AVAILABLE'}
    next_task='对当前冻结队列开展独立训练入口、目标隔离和反向传播验收' if small else '先审定一次新的标签盲补数计划，补齐当前未达标的数据条件'
    changed=['src/cwm_fp/lddg/batch.py','src/cwm_fp/lddg/cohort.py','tests/test_phase04D1_cohort.py','scripts/report_phase04D1.py','scripts/test_phase04D1.py','STAGES.md','outputs/phase_status.md']
    packet={'stage_id':'04D1','stage_status':'COMPLETE_WITH_LIMITATIONS' if completed else 'PARTIAL_RESOURCE_STOP','primary_protocol':PROTOCOL,
        'frozen_plan_sha256':sha256(OUT/'cohort_plan.json'),'selected_cohort':[{k:p[k] for k in ['identity','index','partition','registered_tests']} for p in plan['products']],
        'actual_scope':'Frozen BankAccountTP collection and lossless input preparation only; no model execution, semantic extension or server installation.',
        'execution_funnel':{'by_partition':fun,'by_version':perversion,'sequence_verification':sequence},'before_metrics':before,'after_metrics':after,'grouped_split_readiness':{'partitions':ready,'pre_registered_goals':goals,'status':statuses['SMALL_PILOT_DATA_READY']},
        'other_protocol_unlabeled_coverage':loso_counts,'other_protocol_label_reads':[],
        'other_protocol_readiness':{f:{'status':'BLOCKED','reason':'Only BankAccountTP is packaged; at least one required held-out-system partition is absent.',
            'class_coverage':None,'class_coverage_reason':'Other-protocol target access not authorized in04D1.'} for f in loso_counts},
        'split_readiness':{'grouped':ready,**loso_counts},
        'admitted_products':[p['identity'] for p in package['products']],
        'recovered_graphs':None,'recovered_graphs_reason':'No historical semantic repair in04D1; all75 previously qualified graphs retained. Earlier16 rejected-graph repair belongs to03B4.',
        'authorized_training_class_coverage':targets,'selection_bias_audit':{'label_blind':True,'replacements':0,'distribution':bias,'test_classes':{scope+':'+cls:dict(c) for (scope,cls),c in testclasses.items()},'test_class_rejection_reasons':class_reasons,'not_population_estimate':True,
            'enabled_features_absent_after_admission':absent_features,'test_classes_absent_from_complete_products':unsupported_classes},
        'rejection_root_cause_audit':root_cause,
        'lossless_package_status':{'status':statuses['LOSSLESS_DATA_READY'],'manifest':str(OUT/'package/manifest.json'),'manifest_sha256':validation['manifest_sha256'],'failures':validation['failures'],'old_graphs_reverified':75},
        'training_interface_status':{'status':'PARTIAL','reason':'Complete22-column/7-relation tensors verified. No direct portable-package training CLI or end-to-end target/evaluator integration verified.'},
        'backward_validation_status':{'status':'NOT_VERIFIED','reason':'No optimizer or backward pass authorized/executed.'},
        'world_model_readiness':{'T1':'PARTIAL','safe_real_prefix_producer':False,'T2':'BLOCKED','observations':dict(state_counts),'new_measurements':True,
            'measurement_scope':f"All {after['new_candidate_graphs']} newly collected candidate graphs, including rejected graphs. New actual scalar callbacks; derived missing value versions are not additional measurements. No complete JVM states or input/output pairs.",
            'control_edge_scope':'Static normal postdominance plus supported path checks; not per-execution dynamic controlling-instance edges.',
            'value_node_missing_fraction':state_counts['missing_value_nodes']/state_counts['value_nodes'] if state_counts['value_nodes'] else None,
            'assertion_expected_actual_missing_fraction':1-state_counts['complete_assertion_values']/state_counts['assertion_nodes'] if state_counts['assertion_nodes'] else None,
            'whole_state_missing_fraction':None,'whole_state_missing_reason':'No complete JVM-state denominator or general input/output vector observation.',
            'T3':'BLOCKED','real_intervention_pairs':0,'state_and_counterfactual_training_enabled':False},
        'tests_executed':{'regression':load(OUT/'unit_tests.json'),'real_java_pairs':after['new_complete_pairs'],'real_java_result_consistent_pairs':after['new_consistent_pairs'],'old_java_pairs_rerun':0},'resources':resources,'readiness_states':statuses,'changed_files':changed,'rejection_reasons':dict(reason_counts),
        'artifact_paths':{'review':str(ROOT/'outputs/phase04D1_review_packet.md'),'cohort_status':str(ROOT/'outputs/phase04D1_cohort_status.csv'),'graph_status':str(ROOT/'outputs/phase04D1_graph_status.csv'),
            'plan':str(OUT/'cohort_plan.json'),'package_manifest':str(OUT/'package/manifest.json'),'seal':str(OUT/'cohort_seal.json'),'preservation':str(OUT/'preservation.json')},
        'unresolved_issues':[{'code':'SEMANTIC_SELECTION','evidence':str(ROOT/'outputs/phase04D1_graph_status.csv'),'reason':'Every rejected graph first encounters unsupported CHECKCAST192; downstream heap/call/exception gaps remain uncertified. Transaction configurations absent from the qualified cohort.',
            'affected_products':[list(key(r)) for r in new if not r['complete_semantic_suite']],
            'confirmed_earliest_gap':True,'repair_validated':False,'single_opcode_repair_recovery':None,
            'recovery_upper_bound_if_all_gaps_verified':after['new_semantic_rejection_products'],'upper_bound_is_achieved':False,
            'world_model_impact':'No qualified complete Transaction suites; cannot claim its complete state transitions are covered.'},
            {'code':'PILOT_COVERAGE','unmet_goals':[k for k,v in goals.items() if not v],'reason':'No replacements after quality or labels'},
            {'code':'TRAIN_BACKWARD_SERVER','reason':'Input verified, target/evaluator integration and backward path not exercised; mapped server directory currently denied access'},
            {'code':'STATE_PREFIX_INTERVENTION','reason':'No certified real prefix producer or adequate full states/intervention pairs'}],
        'recommended_next_stage':[{'priority':1,'task':next_task,'condition':'必须明确保留当前语义选择范围和代理标签限制；不在本阶段执行'},
            {'priority':2,'task':'解决X:/codes访问并验收Linux与GPU环境','condition':'本地映射盘不等于服务器解释器；需独立确认运行位置、驱动与依赖'},
            {'priority':3,'task':'另立有界阶段认证转账测试的类型转换及后续异常/对象语义','condition':'针对本轮16张拒绝图逐项验证；单修一个操作码的实际恢复收益尚未知，不放宽质量门槛'}]}
    public_csv(ROOT/'outputs/phase04D1_cohort_status.csv',productrows);public_csv(ROOT/'outputs/phase04D1_graph_status.csv',graphrows)
    write_json(ROOT/'outputs/phase04D1_review_packet.json',packet)
    ftable=table(['分区','候选产品/版本','选中','开始','完整执行','完整语义合格','无损封装'],[[s,f"{r['candidate_products']}/{r['candidate_versions']}",r['selected_products'],r['started_products'],r['complete_execution_products'],r['complete_semantic_products'],r['packaged_products']] for s,r in fun.items()])
    st=table(['分区','合格产品','独立版本'],[[s,r['products'],r['versions']] for s,r in ready.items()])
    ct=table(['范围/测试类','登记','采集','配对完成','生成图','单图合格','合格率（按登记）','进入完整产品'],[[scope+'/'+cls,*[c[k] for k in ['registered','collected','paired_complete','graphs','qualified']],f"{c['qualified']/c['registered']:.2%}",c['in_packaged_products']] for (scope,cls),c in sorted(testclasses.items())])
    rt=table(['未支持的实际指令','涉及图数','涉及产品数'],[[op,len({tuple(x['identity']) for x in rs}),len({tuple(x['identity'][:3]) for x in rs})] for op,rs in sorted(opcode_gaps.items())])
    bias_table=table(['产品状态','产品数','套件测试数：最小/中位/最大','每产品节点：最小/中位/最大','每产品边：最小/中位/最大'],
        [[title,bias[cat]['products'],*[' / '.join(str(bias[cat][field][x]) for x in ('min','median','max')) for field in ('suite_sizes','graph_nodes_per_product','graph_edges_per_product')]] for cat,title in [('accepted','接收'),('rejected','拒绝')]])
    feature_names={'BankAccount':'银行账户','CreditWorthiness':'信用检查','DailyLimit':'每日限额','Interest':'利息','InterestEstimation':'利息估算','Lock':'锁定','Overdraft':'透支','Transaction':'转账'}
    feature_table=table(['启用特征（源码名称）','接收产品数','拒绝产品数'],[[name+'（'+f+'）',bias['accepted']['enabled_feature_counts'].get(f,0),bias['rejected']['enabled_feature_counts'].get(f,0)] for f,name in feature_names.items()])
    metric_table=table(['范围','节点','全部边','执行事件','控制依赖边','动态数据依赖边'],[
        ['旧合格包',before['nodes'],before['edges'],before['events'],before['control_dep'],before['data_dep']],
        ['本轮全部候选',*[after['new_graph_totals'][k] for k in ('nodes','edges','events','control_dep','data_dep')]],
        ['新旧合并合格包',*[after['package_totals'][k] for k in ('nodes','edges','events')],after['packaged_control_dep'],after['packaged_data_dep']]])
    status_names=['冻结采集计划','完整产品无损数据','小规模试验数据','训练入口','反向传播','Linux与显卡环境','泛化有效性证据']
    status_text={'COMPLETE':'已完成','READY':'已就绪','PRELIMINARY_READY':'初步就绪','PARTIAL':'部分具备，仍待验收','BLOCKED':'仍有阻碍','NOT_VERIFIED':'未验证','NOT_AVAILABLE':'尚无'}
    readiness_table=table(['准备项','状态'],[[name,status_text[value]] for name,value in zip(status_names,statuses.values())])
    txt=f'''# CWM-FP阶段04D-1研究交接报告

本阶段实际执行了冻结的银行账户跨缺陷采集队列；未训练任何模型。主协议为`grouped_bank_pilot_v1`，只是阶段01原grouped划分的BankAccountTP子集，未重新划分，未使用其他协议的目标。

项目研究如何通过程序依赖结构和运行观测识别“原始测试全部通过、但仍含相关可执行缺陷”的产品。阶段04P已经具备五个完整产品、75张真实图的无损读取能力；图分类模型已有工程实现，但此前训练区只有一个代理FP样本。当前工作是补充真实监督试验数据，不是训练或证明因果世界模型有效；最终研究仍需覆盖银行账户和邮件两个系统。

## A. 新增真实数据与复用

新增选中{after['new_selected_versions']}个独立版本、{after['new_selected_products']}个原始通过产品、{after['new_registered_tests']}个登记测试。实际完成{after['new_complete_pairs']}对，其中{after['new_consistent_pairs']}对与原始/插桩/历史结果一致；生成{after['new_candidate_graphs']}张候选图，{after['new_qualified_graphs']}张单图通过严格准入。新增完整合格产品{after['new_admitted_products']}个，来自{after['new_admitted_versions']}个版本。

身份认证合格{after['new_identity_valid_graphs']}图，动态依赖语义合格{after['new_dependency_valid_graphs']}图。下表中被拒绝图的节点和边只是保留的候选证据，不计为正式训练输入；三行统计范围不同，不能把候选总数与合格包相减称为读取损失。

{metric_table}

完整套件以阶段01冻结的原始登记清单为核对依据；配对一致按既有JUnit完成状态和测试结果判据核验，不代表已经逐值观测并证明未插桩程序的全部内部状态等价。

旧5产品/4版本/75图只读复用并全部重新核验，没有重跑旧Java。新旧合并包包含{after['total_packaged_products']}产品、{after['total_packaged_versions']}版本、{after['total_packaged_graphs']}测试图。单图合格与完整产品合格分别统计，不能相互替代。

本阶段只新增批次配置与调度适配：`cohort.py`调用原有批次准备、缓存、配对运行、提取和数据包接口；`batch.py`仅增加显式传入已认证的语义配置及框架证据参数。源码身份、异常、动态定义使用、框架调用认证和图模型架构没有扩展或放宽。新增检查与报告脚本详见JSON的`changed_files`。

封装与原图核对按产品顺序进行，跨产品仅保存文件定位信息，不把整个队列的原始图对象同时留在内存中；仍保留每个产品的全部测试，没有采样或截断。这是调度层的内存约束，不涉及重写图编码器。

实际修改清单：

- `src/cwm_fp/lddg/cohort.py`：新增冻结选择、资源预留、原有采集器调用、产品准入、封装及独立训练类别统计入口。
- `src/cwm_fp/lddg/batch.py`：让批次提取显式复用已有框架调用效果与语义配置，原默认调用保持兼容。
- `tests/test_phase04D1_cohort.py`：确定性选择、分区配额、完整产品预算和目标访问边界检查。
- `scripts/test_phase04D1.py`：复用数据侧回归、核对旧75图、冻结源码与历史文件；跳过模型前向检查。
- `scripts/report_phase04D1.py`：从实际证据生成一份中文交接、机器指标和两份状态表。
- `STAGES.md`、`outputs/phase_status.md`：更新阶段入口和后续条件。核心交接路径见本文末尾及同名JSON，不另建重复长报告。

真实研究数据验证了配对采集、现有支持范围内的图生成、完整套件准入及无损加载。选择规则、容量/损坏拒绝、标签读取边界与前缀契约的边界条件只由人工单元用例验证；本阶段没有运行新的人工Java程序。恢复入口和严格缓存绑定沿用旧实现，本轮没有主动制造中断或触发恢复，不能算作新完成的恢复压力试验。训练入口、反向传播、Linux及GPU均未在本阶段执行。

## B. 冻结顺序与资源边界

固定种子20261101，阶段命名空间04D1。按公开完整身份和冻结分区的散列顺序，优先新版本，每版本最多3个未采集原始通过产品。套件大小仅用于预算；没有按是否含Transaction或易通过语义检查进行选择。完整候选池、选中名单、执行顺序及来源指纹在`{OUT.as_posix()}/cohort_plan.json`，摘要`{sha256(OUT/'cohort_plan.json')}`。

计划24训练、6验证、6测试产品，713对测试，先轮转训练/验证/测试，队列耗尽后继续剩余训练项；没有计划外产品替换或看到标签后追加。执行停止原因：{load(OUT/'collection_summary.json')['stop_reason']}。该值为空表示未因预算提前停止。真实结果及全部未开始/拒绝项均保留在产品CSV。

逐产品完成顺序与逐测试执行日志分别核对冻结队列，结果：{j(sequence)}。

硬上限1000对/2000测试进程、5400秒采集控制运行、3GiB新目录；保留更严格的4GiB磁盘余量下限。单进程30秒、512MiB堆、1GiB采样驻留内存、2万事件、32MiB输出及128文件沿用旧规则，顺序执行。产品开始前预留完整套件的最坏执行时间/进程量及512MiB目录空间；超限拒绝，绝不靠截断完成产品。轮询监控不是操作系统精确配额。

## C. 从候选到准入的漏斗

{ftable}

分版本数量见JSON的`execution_funnel.by_version`。新产品中资源/执行未完整{after['new_resource_incomplete_products']}个，已完整执行但未通过完整语义套件{after['new_semantic_rejection_products']}个。具体原因不能以“语义不支持”一笔带过：逐图CSV给出认证错误、动态缺口、结果一致性和原日志位置；主要原因计数为{j(dict(reason_counts))}。原因计数可能重叠。

本批次完整产品准入率为{after['new_admitted_products']}/{len(new)}，不是全库合格率。未采集候选不能作为拒绝或成功样本。所有被拒绝测试保留，没有删测试凑完整产品。

本轮新增拒绝图{after['new_candidate_graphs']-after['new_qualified_graphs']}张，属于04D-1的新版本/产品批次，不能与03B-4曾处理的历史16张拒绝图混为同一组。本阶段没有重新修复或重跑那些历史拒绝图。

另有{after['new_qualified_graphs']-(after['total_packaged_graphs']-75)}张合格单图因同产品仍有拒绝测试而没有进入监督包，原文件完整保留。这是产品准入限制，不是封装阶段丢弃测试或截断图。

实际未支持的操作码分布如下，完整身份、指令位置和最早证据在图CSV及JSON的`rejection_root_cause_audit`。

{rt}

字节码192表示引用类型转换（CHECKCAST），不能因旧错误代码统称“数组/类型/监视器”就认定发生了数组操作。各图第一个动态缺口分布为{j(dict(first_reason_counts))}。后续操作数栈、对象身份、未跟踪调用及异常帧错误可能相互关联；只找到最早缺口不等于其他语义已获认证，也不能预言修复一个操作码后所有产品必然恢复。本阶段没有做此类修复。

这些是依赖分析器无法完成认证的原因，不能等同于Java程序实际发生栈错误或测试失败。真实测试执行结果、插桩一致性和图语义认证分别记录。

## D. 合并队列与冻结划分

{st}

前后正式数据产品数5→{len(packed)}，独立版本4→{after['total_packaged_versions']}，图数75→{after['total_packaged_graphs']}。同版本全部产品/测试继续继承原分区；旧5产品归属未变。本阶段不宣称跨系统验证，Email未新增采集或修复。

另外两种留出系统协议仅统计无标签覆盖：{j(loso_counts)}。它们均缺少至少一个必要分区，因此跨系统试验仍阻塞；没有为此读取它们的训练、验证或测试目标。

## E. 授权训练类别及准备目标

先冻结计划、产品准入、完整包和内容指纹，再由独立进程调用既有训练目标接口，只打开grouped的训练文件。训练类别计数：{j(targets['counts'])}；类别覆盖独立版本：{j(targets['class_version_counts'])}；仅单版本的类别：{j(targets['single_version_classes'])}。未读取验证、测试或其他协议目标，也未用其他协议已知类别推断当前分区。

标签仍是既有本地PULP缺陷知识代理，未独立核实真实FP/TP。原始PASS只表示原套件通过，不等于TP。训练聚合审计在`readiness/grouped_train.json`；公共CSV和模型包不包含FP/TP目标。

预注册小规模目标检查：{j(goals)}。结果为**{statuses['SMALL_PILOT_DATA_READY']}**。达到工程数量目标也只表示小规模试验数据初步就绪，不能视为论文统计充分、验证目标可用于选择、测试指标已可计算或训练链路已通过。

## F. 测试类型覆盖与质量选择范围

{ct}

每类单图合格率按“单图合格/登记”及“单图合格/生成图”分别解释；完整产品列显示该类实际进入训练输入的测试数。Transaction单图即使大多数合格，只要同产品仍有一个拒绝测试，该完整产品就不能准入。

接收/拒绝产品的套件和图规模：

{bias_table}

启用特征按产品计数，每个产品可同时启用多个特征：

{feature_table}

这些分布是当前有界计划经既有语义门槛筛选后的描述，不使用目标，也不作为分类特征。配置来自公开清单，配置约束尚未独立认证，22列中的配置状态按既有缺失规则屏蔽。特定测试/配置若在合格队列缺失，只能声明支持范围限制，不能宣称方法已覆盖该子空间。详细各类拒绝原因在机器交接包。

本轮被拒绝配置中存在、但接收配置中完全没有的启用特征：{j(absent_features)}；登记过但没有进入任何完整产品的测试类：{j(unsupported_classes)}。这些是本批实际覆盖缺口，不是全库因果归因。

## G. 无损封装与接口

新包位于`{OUT.as_posix()}/package/`，未覆盖phase04P_v1。清单摘要`{validation['manifest_sha256']}`。全部封装产品保留所有登记测试、22维特征、7类关系及独立事件；节点/边/事件/测试损失均0。包中合计{j(after['package_totals'])}，与原始图和事件逐项核对；旧75图张量与04P包逐值一致。

容量或编码拒绝：{j(validation['failures'])}。保留04P的单分片/解压、图规模、总包和旧128MiB单输入门槛，超限不采样。采集端语义核验与包端内容验证仍分开，包不证明重新执行Java。本阶段只验证读取，没有优化器、反向传播或分类指标。

新包采用已有相对路径分片和外部可信摘要机制；本次实测加载发生在Windows本地，未把04P旧包的不同根目录验证写成本次新包的Linux验收。实际迁移时需连同独立导出回执核验清单摘要，并在服务器重新进行读取检查。

## H. 实际资源、命令与复现

实际启动Java测试进程{len(launched)}个，配对尝试{len(testcmd)//2}对，自动重试0；旧75测试只复用。Java测试进程耗时之和{resources['test_process_seconds_sum']:.3f}秒；采集控制进程总耗时{resources['collection_controller']['seconds']:.3f}秒，含编译、静态导出、提取及控制开销。两者有包含关系，不相加冒充总耗时。

Java测试直接进程最大采样驻留内存{resources['peak_sampled_Java_RSS_bytes']:,}字节；50ms采样不是精确瞬时峰值。提取器进程资源和所有实际命令、退出码见`command_index.json`及各`.command.json`。准入、封装的真实Python进程工作集峰值分别为{resources['tensor_worker_resources']['admission']['OS_peak_working_set_bytes']:,}、{resources['tensor_worker_resources']['package']['OS_peak_working_set_bytes']:,}字节，由工作进程内部读取操作系统的生命周期峰值。它不等于系统总内存，也不使用约5MiB的虚拟环境启动器数字冒充张量进程内存。

生成目录快照{resources['generated_directory_usage']['bytes']:,}字节，新包{resources['package_bytes']:,}字节，D盘剩余{resources['disk_free_bytes']:,}字节。资源中止类别：{j(resources['resource_reasons'])}。原始轨迹不压缩，数据包元数据可无损压缩。缓存和恢复绑定源码、类文件、依赖、插桩范围、Java环境及语义代码摘要；失败或未封存尝试不自动覆盖。

实际入口：`python -m cwm_fp.lddg.cohort plan/collect/extract/admission/package/labels`，其中extract只由批次逐产品调用，labels仅授权grouped训练文件。实际完整参数和执行结果以`actions.jsonl`与命令索引为准，不把入口存在写成已完成。数据侧回归检查{load(OUT/'unit_tests.json')['tests_run']}项，成功{load(OUT/'unit_tests.json')['passed']}项，退出码{load(OUT/'unit_tests.json')['exit_code']}，耗时{load(OUT/'unit_tests.json')['seconds']:.3f}秒；完整名称与日志在`unit_tests.json`及`unit_tests.log`。冻结输入与选定源码的末次核验另见`preservation.json`。

实际环境：{resources['environment']['os']}、Python3.11.4、Java8u381，张量库PyTorch{resources['environment']['torch']}、数组库NumPy{resources['environment']['numpy']}。这是本地已安装环境的实测版本，CUDA构建信息不是服务器显卡或驱动验收结果；完整版本命令及输出在`environment.json`。本阶段没有GPU训练、分类分数或训练耗时估算；没有独立测量的完整交互任务墙钟时间为null，不把嵌套命令耗时简单相加。

## I. 首要剩余阻碍

{readiness_table}

本次实际结果对应的首要任务为：**{next_task}**。同时不能忽略本批语义选择范围、代理标签资格、独立版本数量和未完成跨系统评估。现有22列读取并不证明训练过程正确，也没有泛化成绩。

## J. 服务器及世界模型下一步条件

用户指定后续服务器工作目录为`X:/codes`。当前仅看到X盘，读取该目录返回WinError5拒绝访问；未写入、安装或连接服务器，未验收Linux或GPU。可准备迁移当前已封装数据，但应先解决访问和实际运行环境，不能把Windows映射盘上的本地执行当成服务器执行。

就本次数据条件而言，{'已满足预注册的小规模数量目标，可以进入独立的Linux环境与有界训练链路验收' if small else '尚未满足预注册的小规模数量目标；纯环境验收可单独准备，监督试验队列仍需下一阶段决策'}。这与正式跨缺陷泛化、跨系统泛化或因果有效性证明是不同结论。

最多三个后续候选：①{next_task}；②解决服务器目录访问并核验Linux、驱动与显卡环境；③另立有界阶段认证转账测试的类型转换及后续异常/对象语义。第一项直接处理当前小规模试验的首要前置条件，第二项是实际服务器执行条件，第三项针对本轮明确的配置覆盖偏差。全部剩余语义若获认证，最多涉及{after['new_semantic_rejection_products']}个产品的恢复；这只是上界，单修一个操作码的收益未知。均不在本阶段执行。

新增原始物理事件{state_counts['physical_events']:,}条，投影后的产品/测试事件{after['new_graph_totals']['events']:,}条；框架内部事件保留在原始日志和调用效果证据中。实际标量回调事件{state_counts['observed_scalar_events']:,}条；构建的值节点{state_counts['value_nodes']:,}个，其中实测值节点{state_counts['observed_value_nodes']:,}个，缺失{state_counts['missing_value_nodes']:,}个。标量回调数与值节点数统计对象不同，不能互换。

这些计数覆盖本轮全部{after['new_candidate_graphs']}张候选图，含未认证图，不等于全部都可作为状态训练输入；值版本数不等于实测数，未知值保持缺失。当前仅有局部有序依赖和部分标量，无真实安全前缀生产器、完整数值状态及合法干预对；状态预测和反事实训练继续关闭。不能把更多图解释成因果机制已被验证。

| 世界模型数据目标 | 当前结论 | 主要依据 |
| --- | --- | --- |
| 有序事件与依赖结构（T1） | 部分可用 | 合格图保留事件顺序和依赖；缺少真实安全前缀生产器，转账完整套件缺失 |
| 数值状态与转换观测（T2） | 尚不足 | 值节点缺失比例{state_counts['missing_value_nodes']/state_counts['value_nodes']:.2%}；{state_counts['assertion_nodes']-state_counts['complete_assertion_values']}个断言节点缺完整期望/实际值，未建立完整输入输出状态对 |
| 合法干预前后执行对（T3） | 缺失 | 本阶段没有干预或反事实配对执行 |

控制边仍是静态后支配依赖，并核验受支持的实际路径；不是逐次执行的控制者事件关系。输出节点中的原始测试通过结果是真实后验观测，却不是完整程序数值输出；断言是否通过与期望值/实际值是否完整也分别统计。完整图含未来执行标记和最终结果，不能直接作为早期预测输入；现有前缀契约能拒绝人工越界样例，真实前缀生产器尚未完成，因此不开展时间状态预测。

核心交接文件为`outputs/phase04D1_review_packet.md`、`outputs/phase04D1_review_packet.json`、`outputs/phase04D1_cohort_status.csv`和`outputs/phase04D1_graph_status.csv`。产品表共41条（36新增、5复用），测试图表共788条（713新增、75复用），保留被拒绝对象和证据路径，不含受保护目标。

本阶段未找到可安全删除且未被引用的冗余历史证据；未删除旧报告或日志。按本次核心交付要求，只维护本中文报告、机器交接包和两份状态CSV，不另复制同内容长报告。任务结束后停止，等待下一阶段指令。
'''
    (ROOT/'outputs/phase04D1_review_packet.md').write_text(txt,encoding='utf-8')
    print('REPORT',after,ready,targets['counts'],statuses)

if __name__=='__main__':main()
