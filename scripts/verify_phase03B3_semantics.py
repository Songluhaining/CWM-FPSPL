"""Bounded, Bank-only replay of the seven sealed 03B2 products. Never fit."""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cwm_fp.data.common import KEY,sha256,write_json
from cwm_fp.lddg.suite import load,lines,validate_saved
from cwm_fp.lddg.static import load_static,program_identities,mapping
from cwm_fp.lddg.trace import parse_trace,Dataflow,NULL_FIELD_PROFILE
from cwm_fp.lddg.extract import graph_for,validate_graph,identity_errors
from cwm_fp.lddg.control import control_evidence_errors

OLD=Path('D:/dataset/whn/cwm_fp_generated/data/processed/lddg/phase03B2_v1')
OUT=OLD.parent/'phase03B3_bank_v1'
REUSE=ROOT/'data/processed/lddg/phase03B1_bank0001_v1'
FOLDS=('grouped','loso_BankAccountTP','loso_Email')
CASES=('normalReadWrite','sameRead','sameWrite','callerRead','callerWrite','repeated','aliasedNull',
       'nullInvokeRejected','castRejected','broadCatchRejected','finallyRejected','externalEffectsRejected','divideRejected')
SUPPORTED=set(CASES[:7])
def dumps(x):return json.dumps(x,ensure_ascii=False,sort_keys=True)
def jsonlines(p,rows):p.write_text(''.join(dumps(r)+'\n' for r in rows),encoding='utf-8')
def split_path(fold):return ROOT/'data/processed/splits'/('splits.csv' if fold=='grouped' else fold+'.csv')
def budget():
    if shutil.disk_usage(OUT).free<4*1024**3:raise RuntimeError('Less than 4 GiB free')
    if sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())>1024**3:raise RuntimeError('Phase03B3 1 GiB output budget exceeded')
def entries(include_reuse=False):
    plan=load(ROOT/'configs/phase03B2_collection.json')
    for i,p in enumerate(plan['products'][:7]):
        assert p['identity'][0]=='BankAccountTP'
        yield f'p{i:03d}',OLD/f'p{i:03d}',p
    if include_reuse:
        yield 'reuse03B1',REUSE,{'identity':['BankAccountTP','_MultipleBugs_.NOB_1.ID_1','model_m_ca4_0001'],
            'observed_result':'PASS','tests':[r['test_id'] for r in load(REUSE/'collection.json')['records']],
            'partitions':{'grouped':'val','loso_BankAccountTP':'test','loso_Email':'train'}}
def loc(raw,i,path):
    if i is None or i<0 or i>=len(raw):return None
    r=raw[i]
    # Raw log bytecode line number, separately from Java source line and BCI.
    text=path.read_text(encoding='utf-8').splitlines();marker='lineinfo='+'#'.join(map(str,(*r['domain'],r['line'],r['bci'])))
    matches=[n+1 for n,s in enumerate(text) if s.strip().startswith('opcode=') and marker in s]
    prior=sum(e['domain']==r['domain'] and e['bci']==r['bci'] for e in raw[:i])
    return {'event_index':i,'domain':r['domain'],'bci':r['bci'],'opcode':r['opcode'],'source_line':r['line'],
            'raw_log':str(path),'raw_line':matches[prior] if prior<len(matches) else None}

def audit():
    from cwm_fp.models.graph_encoder import evidence_errors,encode_test
    rows=[];products=[]
    for name,work,p in entries():
        graphs=lines(work/'g/graphs.jsonl');records=load(work/'collection.json')['records'];by_test={r['test_id']:r for r in records}
        static=load_static(work/p['identity'][2]/'static.tsv');pr=[]
        for g in graphs:
            r=by_test[g['test_id']];status=validate_saved(r);log=Path(r['directory'])/'instrumented/run.log'
            raw=parse_trace(log,r['test_id']);errors=evidence_errors(g);q=g['quality']
            visits=[e for e in raw if any(h['handler']==e['bci'] for h in static['exception_tables'][e['domain']]['handlers'])]
            exceptional=[e for e in visits if e['event_index'] and (raw[e['event_index']-1]['domain']!=e['domain'] or
                e['bci'] not in static['methods'][e['domain']][raw[e['event_index']-1]['bci']]['successors'])]
            faults=[raw[e['event_index']-1] for e in exceptional]
            categories=[]
            if any(e['opcode'] in (180,181) for e in faults):categories.append('null_field_exception_transfer')
            if any(e['opcode'] in range(182,187) for e in faults):categories.append('null_receiver_invocation_exception')
            if any(e['opcode']==192 for e in raw):categories.append('checkcast_not_certified')
            helpers=[e for e in raw if e['fields'].get('callclass')=='org.evosuite.runtime.EvoAssertions']
            if helpers:categories.append('untraced_exception_assertion_library_effects')
            assert (not errors)==q['eligible_graph_structure']
            first=q['unsupported_dynamic'][0]['event_index'] if q['unsupported_dynamic'] else None
            cap=encode_test(g,'both').metadata if not errors else None
            row={**{k:g[k] for k in (*KEY,'test_id')},'batch_product':name,'original_product_result':p['observed_result'],
                'original_test_result':status['runs']['original']['result']['outcome'],
                'instrumented_test_result':status['runs']['instrumented']['result']['outcome'],
                'complete_pair':status['complete'],'parity':status['parity'],'historical_match':status['matches_historical_outcome'],
                'identity_complete':not identity_errors(g),'actual_exception_transition':bool(exceptional),
                'dynamic_def_use_certified':not errors,'frames_and_returns_certified':not errors,
                'array_opcodes':[e['event_index'] for e in raw if 46<=e['opcode']<=53 or 79<=e['opcode']<=86 or e['opcode'] in (188,189,190,197)],
                'library_effect_gap':bool(helpers),'static_initializer_events':[e['event_index'] for e in raw if e['domain'][1]=='<clinit>'],
                'object_alias_status':'precise_allocation_propagation' if not errors else 'null_vs_unknown_or_exception_reference_not_certified',
                'unsupported_opcodes':sorted({raw[x['event_index']]['opcode'] for x in q['unsupported_dynamic'] if x['event_index']>=0}),
                'root_categories':categories,'old_rejection_reasons':errors,'old_dynamic_gaps':q['unsupported_dynamic'],
                'old_control_errors':g['control_flow_evidence']['assessment']['errors'],
                'first_dynamic_gap_evidence':loc(raw,first,log),'actual_fault_evidence':[loc(raw,e['event_index'],log) for e in faults],
                'helper_evidence':[loc(raw,e['event_index'],log) for e in helpers],
                'handler_evidence':[loc(raw,e['event_index'],log) for e in exceptional],
                'events_before':len(raw),'nodes_before':len(g['nodes']),'edges_before':len(g['edges']),
                'values_before':sum(n['type']=='value' for n in g['nodes']),
                'edge_types_before':dict(Counter(e['type'] for e in g['edges'])),
                'registered_tests':len(p['tests']),'default_16_test_capacity_loss':len(p['tests'])>16,
                'graph_file_bytes_before':(work/'g/graphs.jsonl').stat().st_size,
                'file_128MiB_rejection':(work/'g/graphs.jsonl').stat().st_size>134217728,
                'default_graph_encoding_audit':cap,'qualified_before':not errors,
                'trace_sha256':sha256(log),'graph_file_sha256_before':sha256(work/'g/graphs.jsonl')}
            rows.append(row);pr.append(row)
        assert len(pr)==len(p['tests'])
        products.append({'batch_product':name,**p,'graphs':len(pr),'qualified_before':sum(r['qualified_before'] for r in pr),
            'remaining_categories':sorted({c for r in pr for c in r['root_categories']})})
    assert len(rows)==140 and sum(r['qualified_before'] for r in rows)==124
    write_json(OUT/'matrix_before.json',rows);write_json(OUT/'product_blockers_before.json',products)
    write_json(OUT/'repair_selection.json',{'labels_read':False,'matrix_sha256':sha256(OUT/'matrix_before.json'),
        'selected':'null_field_exception_transfer','affected_graphs':14,'affected_products':7,
        'single_category_full_suite_recovery_upper_bound':0,
        'conditional_recovery_if_separate_helper_effects_also_certified':{'semantic_complete':5,'original_PASS':4},
        'reason':'All one-category complete-product gains are zero; this is the necessary largest evidenced dynamic-control prerequisite. Library reflection/helper effects are not certified by normal return.'})
    print('Audited 140 paired traces, 124 qualified; 14 null-field + 2 invocation/cast paths; 16 separate library-effect gaps',flush=True)

def replay():
    from cwm_fp.models.graph_encoder import evidence_errors
    # Reuse the installed validator; do not change the frozen Torch environment.
    try:import jsonschema
    except ModuleNotFoundError:sys.path.append('D:/SoftInstall/Lib/site-packages')
    import jsonschema
    validators={n:jsonschema.Draft202012Validator(load(ROOT/'src/cwm_fp/lddg'/(n+'.schema.json'))) for n in ('graphs','events')}
    before=load(OUT/'matrix_before.json');by_key={tuple(r[k] for k in (*KEY,'test_id')):r for r in before};comparisons=[]
    for name,work,p in entries(True):
        budget();oldg=work/'graphs.jsonl' if name=='reuse03B1' else work/'g/graphs.jsonl'
        olde=work/'events.jsonl' if name=='reuse03B1' else work/'g/events.jsonl'
        graph_rows=lines(oldg);saved_events=lines(olde);records={r['test_id']:r for r in load(work/'collection.json')['records']}
        static=load_static(work/p['identity'][2]/'static.tsv');dest=OUT/name;dest.mkdir(exist_ok=True)
        results=[];all_events=[]
        for old in graph_rows:
            r=records[old['test_id']];status=validate_saved(r);raw=parse_trace(Path(r['directory'])/'instrumented/run.log',r['test_id'])
            mapped={(a['composed_class'],a['composed_method'],a['descriptor'],a['bytecode_offset']):
                {k:v for k,v in a.items() if k not in {'available','executed','opcode'}}
                for n in old['nodes'] if n['id'].startswith('s:') for a in [n['attrs']]}
            identities={tuple(v['locator'][k] for k in ('declaring_class','method','descriptor','bytecode_offset')):v for v in old['instruction_catalog']}
            for v in identities.values():assert sha256(Path(v['evidence']['class_file']))==v['locator']['class_sha256']
            features={n['id'][2:]:n['attrs']['enabled'] for n in old['nodes'] if n['type']=='feature'}
            lineage={k:r[k] for k in (*KEY,'test_id')}
            g,events=graph_for(raw,static,mapped,features,lineage,status,identities,semantics_profile=NULL_FIELD_PROFILE)
            g['quality']['suite_completeness']=old['quality']['suite_completeness']
            for k,v in old['provenance'].items():
                if k not in g['provenance']:g['provenance'][k]=v
            validate_graph(g,events);validators['graphs'].validate(g)
            for e in events:validators['events'].validate(e)
            assert not identity_errors(g)
            old_events=[e for e in saved_events if e['test_id']==r['test_id']]
            assert len(events)==len(old_events)==len(raw)
            for a,b in zip(old_events,events):
                for field in ('event_index','thread_id','composed_location','opcode','value_observed','value_summary','branch_taken','assertion_passed'):
                    assert a[field]==b[field],(name,r['test_id'],field)
            for typ in ('control_dep','temporal','feature_controls','observes'):
                assert [e for e in g['edges'] if e['type']==typ]==[e for e in old['edges'] if e['type']==typ]
            oldok=not evidence_errors(old);newok=not evidence_errors(g)
            if oldok:
                assert newok
                assert old['edges']==g['edges'] and old['nodes']==g['nodes'] and old_events==events
            assessment=g['control_flow_evidence']['assessment']
            rec={**lineage,'batch_product':name,'qualified_before':oldok,'qualified_after':newok,
                'raw_sequence_scalar_callbacks_and_control_temporal_edges_unchanged':True,
                'old_qualified_nodes_edges_events_exactly_unchanged':oldok,
                'event_ids_changed':sum(a.get('canonical_event_id')!=b.get('canonical_event_id') for a,b in zip(old_events,events)),
                'frame_ids_changed':sum(a['frame_id']!=b['frame_id'] for a,b in zip(old_events,events)),
                'events_after':len(events),'nodes_after':len(g['nodes']),'edges_after':len(g['edges']),
                'values_after':sum(n['type']=='value' for n in g['nodes']),
                'observed_values_after':sum(n['type']=='value' and n['attrs'].get('observed',False) for n in g['nodes']),
                'edge_types_after':dict(Counter(e['type'] for e in g['edges'])),
                'certified_null_field_transfers':assessment['certified_exception_transitions'],
                'transfers':assessment.get('certified_transfers',[]),'control_path_verified':assessment['accepted'],
                'new_dynamic_gaps':g['quality']['unsupported_dynamic'],'new_rejection_reasons':evidence_errors(g)}
            comparisons.append(rec)
            if name!='reuse03B1':by_key[tuple(r[k] for k in (*KEY,'test_id'))].update(rec)
            results.append(g);all_events.extend(events)
        jsonlines(dest/'graphs.jsonl',results);jsonlines(dest/'events.jsonl',all_events)
        source=work/'source_map.jsonl' if name=='reuse03B1' else work/'g/source_map.jsonl';shutil.copyfile(source,dest/'source_map.jsonl')
        write_json(dest/'quality_report.json',{'tests':[g['quality'] for g in results],'original_collection':str(work/'collection.json'),
            'original_graph_sha256':sha256(oldg),'new_real_test_executions':0})
        print(name,len(results),'qualified',sum(not evidence_errors(g) for g in results),'certified transfers',sum(g['quality']['certified_null_field_transitions'] for g in results),flush=True)
    write_json(OUT/'comparison.json',comparisons);write_json(OUT/'matrix_after.json',before)
    path=ROOT/'outputs/phase03B3_failure_matrix.csv'
    fields=list(dict.fromkeys(k for r in before for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in before:w.writerow({k:dumps(v) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()})

def fixtures():
    import verify_phase03A2_identity as helpers
    from cwm_fp.lddg.runner import checked,execute_pair
    import jsonschema
    helpers.OUT=OUT;c=load(REUSE/'collection.json')['config'];c={**c,'max_seconds_per_test':30,'max_events':20000}
    cp=helpers.tools_build(c);dest=OUT/'synthetic';dest.mkdir(exist_ok=True);variant=dest/'variants/fixture'
    producer=ROOT/'third_party/existing_instrumenter';bytecode=ROOT/'data/processed/lddg/phase03_bank_v2'
    for folder,name in [('src','NullFieldProgram.java'),('test','NullFieldFixtureTest.java')]:
        (variant/folder).mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/'tests/java/nullfield'/name,variant/folder/name)
    (variant/'roles.meta').write_text('{"features":[],"roles":{}}',encoding='utf-8');(variant/'coverage').mkdir(exist_ok=True)
    build=dest/'build'
    for part,folder in [('main','src'),('tests','test')]:
        (build/part).mkdir(parents=True,exist_ok=True)
        checked([Path(c['java_home'])/'bin/javac.exe','-g','-encoding','UTF-8','-cp',os.pathsep.join(map(str,[build/'main',producer/'lib/*'])),
            '-d',build/part,*sorted((variant/folder).glob('*.java'))],dest,dest/(part+'-build.log'))
    source=helpers.source_export(c,cp,list(variant.rglob('*.java')),dest);static=helpers.export_static(c,cp,list(build.rglob('*.class')),dest)
    binding=helpers.binding_for(source,list(build.rglob('*.class')),synthetic=True);write_json(dest/'compilation_binding.json',binding)
    _,mapped,_=mapping(variant,static,{},source);outcomes=[]
    for name in CASES:
        budget();identity='nullfieldfixture.NullFieldFixtureTest.'+name;stage=dest/name
        status=execute_pair(c,bytecode,producer,bytecode/'producer-classes',bytecode/'agent.jar',variant,identity,
            ['nullfieldfixture.NullFieldProgram'],build,stage,'PASS')
        assert all(status[k] for k in ('complete','parity','matches_historical_outcome')),(name,status)
        raw=parse_trace(stage/'instrumented/run.log',identity)
        ids=program_identities(variant,static,mapped,source,binding,[(*r['domain'],r['bci']) for r in raw])
        g,events=graph_for(raw,static,mapped,{},dict(system_id='synthetic',bug_version_id='null-field-fixture',product_id='fixture',test_id=identity),status,ids,semantics_profile=NULL_FIELD_PROFILE)
        write_json(stage/'graph.json',g);write_json(stage/'events.json',events)
        validate_graph(g,events);jsonschema.validate(g,load(ROOT/'src/cwm_fp/lddg/graphs.schema.json'))
        assert g['quality']['eligible_graph_structure']==(name in SUPPORTED),(name,g['quality'])
        if name in SUPPORTED:assert not control_evidence_errors(g) and not identity_errors(g)
        outcomes.append({'case':name,'expected_supported':name in SUPPORTED,'status':status,'quality':g['quality'],
            'transfers':g['control_flow_evidence']['assessment'].get('certified_transfers',[])})
        print('FIXTURE',name,'parity PASS, semantics',g['quality']['eligible_graph_structure'],flush=True)
    write_json(dest/'verification.json',{'cases':outcomes,'paired_tests':len(CASES),'Java_test_processes':2*len(CASES),
        'supported':len(SUPPORTED),'rejected':len(CASES)-len(SUPPORTED),'benchmark_experiments':False})

def admission():
    from cwm_fp.models.graph_data import load_corpus,tensor_products
    from cwm_fp.models.graph_encoder import Limits,collate,encode_test,evidence_errors
    products=[];admitted=[];interfaces=[]
    for name,work,p in entries(True):
        gpath=OUT/name/'graphs.jsonl';rows=lines(gpath);result={'batch_product':name,'identity':p['identity'],'original_result':p['observed_result'],'folds':{}}
        for fold in FOLDS:
            ps,ss,gs,eligible,a=load_corpus(ROOT/'data/processed/manifests/products.csv',ROOT/'data/processed/manifests/tests.csv',gpath,split_path(fold),holdout=fold[5:] if fold.startswith('loso_') else None)
            result['folds'][fold]={'eligible':bool(eligible),'audit':a}
            if eligible:
                default=tensor_products(ps,ss,eligible,'both',Limits()); b=collate(default)
                assert all(default[0].audit[k]==0 for k in ('lost_test_fraction','lost_event_fraction','lost_edge_fraction'))
                result['folds'][fold]['lossless_interface']={'shape':list(b.x.shape),'edge_shape':list(b.edge_index.shape),**default[0].audit}
                if fold=='grouped':admitted.append(gpath)
        for g in rows:
            if evidence_errors(g):continue
            limits=Limits(max_nodes=len(g['nodes']),max_events=g['quality']['events'],max_edges=len(g['edges']))
            full=encode_test(g,'both',limits);default=encode_test(g,'both')
            assert full.x.shape[1]==22 and full.metadata['lost_event_fraction']==full.metadata['lost_edge_fraction']==0
            interfaces.append({'batch_product':name,'test_id':g['test_id'],'lossless_shape':list(full.x.shape),'default':default.metadata})
        result['graph_file_bytes']=gpath.stat().st_size;products.append(result)
    combined=OUT/'admitted_graphs.jsonl'
    with combined.open('wb') as dest:
        for p in admitted:
            with p.open('rb') as f:shutil.copyfileobj(f,dest)
    aggregates={}
    for fold in FOLDS:
        *_,a=load_corpus(ROOT/'data/processed/manifests/products.csv',ROOT/'data/processed/manifests/tests.csv',combined,split_path(fold),holdout=fold[5:] if fold.startswith('loso_') else None)
        aggregates[fold]=a
    write_json(OUT/'admission.json',{'products':products,'aggregates':aggregates,'selection_sha256':sha256(OUT/'repair_selection.json'),'labels_read':False})
    write_json(OUT/'interfaces.json',{'qualified_graphs':interfaces,'models_fitted':0})
    print('Admission', {f:a['eligible_by_partition'] for f,a in aggregates.items()},flush=True)

def labels(fold):
    # A fresh process per protocol, only after immutable repair/admission results exist.
    from cwm_fp.data.labels import load_targets
    from cwm_fp.baselines.boundaries import install_io_guard
    a=load(OUT/'admission.json');rows=a['aggregates'][fold]['products'];train=[tuple(r['identity']) for r in rows if r['partition']=='train' and not r['reasons']]
    allowed=ROOT/'data/processed/labels/full'/fold/'train.csv'
    install_io_guard('readiness_train_only',[allowed])
    targets=load_targets(allowed,ROOT/'data/processed/manifests/products.csv',split_path(fold),partition='train',purpose='train') if train else {}
    counts=Counter(targets.get(k,'MISSING') for k in train);(OUT/'readiness').mkdir(exist_ok=True)
    write_json(OUT/'readiness'/f'{fold}.json',{'fold':fold,'train_eligible_products':len(train),'train_target_counts':dict(counts),
        'train_has_both_FP_TP':bool(counts['FP'] and counts['TP']),'label_files_opened':[str(allowed)] if train else [],
        'validation_test_label_files_opened':[],'admission_sha256':sha256(OUT/'admission.json'),'model_training_runs':0})
    print(fold,dict(counts),flush=True)

def preserve():
    seal=load(OUT/'baseline_seal.json')
    for name,h in seal['frozen'].items():assert sha256(Path(name))==h,name
    for v in seal['archive'].values():assert sha256(Path(v['archive']))==v['sha256']
    sources={}
    for _,_,p in entries():sources.update(p['source_sha256'])
    for name,h in sources.items():assert sha256(Path(name))==h,name
    write_json(OUT/'preservation.json',{'valid':True,'frozen_files':len(seal['frozen']),'archived_files':len(seal['archive']),
        'selected_readonly_sources':len(sources),'new_benchmark_executions':0})
    print('Preserved',len(seal['frozen']),len(seal['archive']),len(sources),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['audit','replay','fixtures','admission','labels','preserve'],required=True);parser.add_argument('--fold',choices=FOLDS)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    if args.stage=='labels':labels(args.fold)
    else:
        from cwm_fp.baselines.boundaries import install_io_guard
        install_io_guard('features');globals()[args.stage]()
    with (OUT/'commands.jsonl').open('a',encoding='utf-8') as f:f.write(dumps({'argv':sys.argv,'seconds':time.monotonic()-start,'success':True})+'\n')
