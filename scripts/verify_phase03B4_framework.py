"""Bounded phase03B4 evidence acquisition. No fitting or target-label access."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cwm_fp.data.common import KEY,sha256,write_json
from cwm_fp.lddg.suite import load,lines
from cwm_fp.lddg.runner import execute_pair,checked
from cwm_fp.lddg.trace import parse_trace
from verify_phase03B3_semantics import entries,split_path,FOLDS

OLD=Path('D:/dataset/whn/cwm_fp_generated/data/processed/lddg/phase03B3_bank_v1')
OUT=ROOT/'data/processed/lddg/phase03B4_framework_v1'
BYTECODE=ROOT/'data/processed/lddg/phase03_bank_v2'
PRODUCER=ROOT/'third_party/existing_instrumenter'
HELPER='org.evosuite.runtime.EvoAssertions'
JAR=PRODUCER/'lib/evosuite-standalone-runtime-1.0.6.jar'

def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'plan.json').exists():raise ValueError('Plan already frozen')
    rows=load(OLD/'matrix_after.json');bad=[r for r in rows if not r['qualified_after']]
    assert len(rows)==140 and len(bad)==16 and {r['system_id'] for r in bad}=={'BankAccountTP'}
    frozen={}
    for folder in (ROOT/'src',ROOT/'tests',ROOT/'scripts',ROOT/'outputs',ROOT/'configs'):
        for p in folder.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                frozen[str(p)]=sha256(p)
                if p.suffix in {'.py','.java','.json','.md','.csv'}:
                    d=OUT/'producer_before'/p.relative_to(ROOT);d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
    for p in ROOT.glob('*.md'):
        frozen[str(p)]=sha256(p);shutil.copyfile(p,OUT/'producer_before'/p.name)
    for base in (OLD,*(w for _,w,_ in entries(True))):
        for p in base.rglob('*'):
            if p.is_file():frozen[str(p)]=sha256(p)
    write_json(OUT/'baseline_seal.json',frozen)
    c=load(ROOT/'configs/phase03B2_collection.json')
    c={k:c[k] for k in ('java_home','java_heap_mb','max_events','max_output_bytes','max_seconds_per_test','max_files_per_run','max_rss_bytes')}
    c.update(strict_runtime_binding=True,resource_policy={'root':str(OUT),'max_total_bytes':1024**3,'max_total_files':10000,'min_free_bytes':1024**3})
    plan={'stage':'03B4','created_utc':datetime.now(timezone.utc).isoformat(),'config':c,
          'real_pairs_max':16,'artificial_pairs_max':16,'test_java_processes_max':64,'java_collection_seconds_max':1200,
          'disk_before':dict(zip(('total','used','free'),shutil.disk_usage(ROOT))),
          'selected':[{k:r[k] for k in (*KEY,'test_id','batch_product','original_test_result')} for r in bad],
          'selection_rule':'All 16 rejected BankAccountTP tests in sealed phase03B3; no target labels',
          'choice':'B: bounded EvoAssertions body observations + A: binary-pinned finite early-return effect contract; C for every unmet precondition',
          'A_alone_rejected':'Old call/return markers do not determine reflection path or exception-private cache effects',
          'conditional_upper_bound_only':{'new_semantic_suites':5,'new_original_PASS_suites':4}}
    write_json(OUT/'plan.json',plan)
    print(json.dumps(plan,ensure_ascii=False,indent=2))

def collect(limit):
    plan=load(OUT/'plan.json');c=plan['config'];selected=plan['selected'][:limit];start=time.monotonic()
    results=[]
    for name,work,p in entries():
        records={r['test_id']:r for r in load(work/'collection.json')['records']}
        for row in selected:
            if row['batch_product']!=name:continue
            elapsed=sum(load(q)['seconds'] for q in (OUT/'runs').rglob('*.command.json')) if (OUT/'runs').exists() else 0
            if elapsed>1200:raise RuntimeError('Frozen collection time budget exceeded')
            r=records[row['test_id']];old=load(Path(r['directory'])/'status.json')
            classes=old['inputs']['runtime_binding']['selected_classes']
            clazz=row['test_id'].rsplit('.',1)[0];classes=[s for s in classes if s!=clazz]
            stage=OUT/'runs'/name/Path(r['directory']).name
            status=execute_pair(c,BYTECODE,PRODUCER,BYTECODE/'producer-classes',BYTECODE/'agent.jar',Path(r['variant']),
                                r['test_id'],classes+[HELPER],work/'b',stage,row['original_test_result'])
            raw=parse_trace(stage/'instrumented/run.log',r['test_id'])
            result={**row,'directory':str(stage),'status':status,'raw_events':len(raw),
                    'helper_events':sum(e['domain'][0]==HELPER for e in raw)}
            results.append(result)
            print(name,row['test_id'],status['parity'],status['runs']['instrumented']['result'],len(raw),result['helper_events'],flush=True)
    write_json(OUT/('pilot.json' if limit<16 else 'collection.json'),{'records':results,'seconds':time.monotonic()-start})

def binaries():
    import verify_phase03A2_identity as h
    h.OUT=OUT;c=load(OUT/'plan.json')['config'];cp=h.tools_build(c)
    dest=OUT/'framework';dest.mkdir(exist_ok=True);files=[]
    with zipfile.ZipFile(JAR) as z:
        p=dest/'EvoAssertions.class';p.write_bytes(z.read('org/evosuite/runtime/EvoAssertions.class'));files.append(p)
    h.export_static(c,cp,files,dest)
    rt=Path(c['java_home'])/'jre/lib/rt.jar';names=['java/lang/Throwable','java/lang/Exception','java/lang/RuntimeException','java/lang/NullPointerException','java/lang/StackTraceElement','java/lang/String']
    with zipfile.ZipFile(rt) as z:
        for name in names:
            p=dest/(name.rsplit('/',1)[1]+'.class');p.write_bytes(z.read(name+'.class'));files.append(p)
    for name in ['org.evosuite.runtime.EvoAssertions',*[n.replace('/','.') for n in names]]:
        checked([Path(c['java_home'])/'bin/javap.exe','-classpath',JAR,'-c','-p','-s',name],dest,dest/(name.rsplit('.',1)[1]+'.javap.txt'))
    binding={'jar':{'path':str(JAR),'sha256':sha256(JAR)},'rt':{'path':str(rt),'sha256':sha256(rt)},
             'classes':{p.name:sha256(p) for p in files},'static_sha256':sha256(dest/'static.tsv')}
    write_json(dest/'binding.json',binding);print(json.dumps(binding,indent=2))

def replay():
    from cwm_fp.lddg.static import load_static
    from cwm_fp.lddg.extract import graph_for,validate_graph,identity_errors
    from cwm_fp.lddg.effects import make_evidence,project_and_verify
    from cwm_fp.lddg.trace import FRAMEWORK_PROFILE,NULL_FIELD_PROFILE
    from cwm_fp.models.graph_encoder import evidence_errors
    from cwm_fp.lddg.suite import validate_saved
    try:import jsonschema
    except ModuleNotFoundError:sys.path.append('D:/SoftInstall/Lib/site-packages');import jsonschema
    validators={n:jsonschema.Draft202012Validator(load(ROOT/'src/cwm_fp/lddg'/(n+'.schema.json'))) for n in ('graphs','events')}
    reruns={(r['batch_product'],r['test_id']):r for r in load(OUT/'collection.json')['records']};comparisons=[]
    for name,work,p in entries(True):
        oldgs=lines(OLD/name/'graphs.jsonl');olde=lines(OLD/name/'events.jsonl')
        records={r['test_id']:r for r in load(work/'collection.json')['records']}
        static=load_static(work/p['identity'][2]/'static.tsv');dest=OUT/name;dest.mkdir(exist_ok=True)
        graphs=[];events=[]
        for old in oldgs:
            r=records[old['test_id']];new=reruns.get((name,old['test_id']));status=validate_saved(r)
            evidence=None;raw=parse_trace(Path(r['directory'])/'instrumented/run.log',old['test_id'])
            oldraw=raw
            if new:
                status=load(Path(new['directory'])/'status.json')
                assert all(status[k] for k in ('complete','parity','matches_historical_outcome'))
                evidence=make_evidence(Path(new['directory'])/'instrumented/run.log',old['test_id'],status,OUT/'framework/static.tsv')
                raw,calls,full=project_and_verify(evidence)
                assert len(raw)==len(oldraw)
                for a,b in zip(oldraw,raw):
                    for k in ('event_index','domain','line','bci','opcode','entry','call_returned'):assert a[k]==b[k],(name,old['test_id'],k)
                    # Object hash callbacks are deliberately not identity evidence.
                    for k,v in a['fields'].items():
                        if k=='stack' and v.startswith('Obj:'):continue
                        assert b['fields'].get(k)==v,(name,old['test_id'],k)
            mapped={(a['composed_class'],a['composed_method'],a['descriptor'],a['bytecode_offset']):
                    {k:v for k,v in a.items() if k not in {'available','executed','opcode'}}
                    for n in old['nodes'] if n['id'].startswith('s:') for a in [n['attrs']]}
            ids={tuple(v['locator'][k] for k in ('declaring_class','method','descriptor','bytecode_offset')):v for v in old['instruction_catalog']}
            features={n['id'][2:]:n['attrs']['enabled'] for n in old['nodes'] if n['type']=='feature'}
            lineage={k:old[k] for k in (*KEY,'test_id')}
            g,ee=graph_for(raw,static,mapped,features,lineage,status,ids,semantics_profile=FRAMEWORK_PROFILE if new else NULL_FIELD_PROFILE,framework_evidence=evidence)
            g['quality']['suite_completeness']=old['quality']['suite_completeness']
            for k,v in old['provenance'].items():
                if k not in g['provenance']:g['provenance'][k]=v
            if new:g['provenance']['status_sha256']=sha256(Path(new['directory'])/'status.json')
            validate_graph(g,ee)
            issues=list(validators['graphs'].iter_errors(g))
            if issues:raise ValueError([(list(e.path),e.message[:300]) for e in issues])
            for e in ee:validators['events'].validate(e)
            assert not identity_errors(g)
            assert g['edges']==old['edges'],(name,old['test_id'],'edges changed')
            oe=[e for e in olde if e['test_id']==old['test_id']]
            if not new:assert g['nodes']==old['nodes'] and ee==oe
            else:
                assert [e['frame_id'] for e in ee]==[e['frame_id'] for e in oe]
                assert all(a['static_instruction_id']==b['static_instruction_id'] for a,b in zip(ee,oe))
            q=g['quality'];errors=evidence_errors(g)
            comparisons.append({**lineage,'batch_product':name,'original_product_result':p['observed_result'],
                'partitions':p['partitions'],'original_test_result':status['runs']['original']['result']['outcome'],
                'instrumented_test_result':status['runs']['instrumented']['result']['outcome'],
                'complete_pair':status['complete'],'parity':status['parity'],'historical_match':status['matches_historical_outcome'],
                'qualified_before':not evidence_errors(old),'qualified_after':not errors,'identity_valid':not identity_errors(g),
                'control_valid':q['control_path_verified'],'dynamic_gaps':q['unsupported_dynamic'],
                'framework_effects':g.get('framework_effects',[]),'rejection_reasons':errors,
                'events_before':len(oe),'events_after':len(ee),'physical_events_after':q.get('physical_trace_events',len(ee)),
                'nodes_before':len(old['nodes']),'nodes_after':len(g['nodes']),'edges_before':len(old['edges']),'edges_after':len(g['edges']),
                'edges_by_type':dict(Counter(e['type'] for e in g['edges'])),'values_after':sum(n['type']=='value' for n in g['nodes']),
                'observed_values_after':sum(n['type']=='value' and n['attrs'].get('observed',False) for n in g['nodes']),
                'control_edges_unchanged':True,'data_edges_unchanged':True,'frames_unchanged':True,
                'canonical_event_ids_changed':sum(a['canonical_event_id']!=b['canonical_event_id'] for a,b in zip(ee,oe)),
                'graph_path':str(dest/'graphs.jsonl'),'evidence_path':str(Path(new['directory'] if new else r['directory'])/'instrumented/run.log'),
                'status_path':str(Path(new['directory'] if new else r['directory'])/'status.json')})
            graphs.append(g);events.extend(ee)
        for filename,items in [('graphs',graphs),('events',events)]:
            with (dest/(filename+'.jsonl')).open('w',encoding='utf-8') as f:
                for item in items:f.write(json.dumps(item,ensure_ascii=False,sort_keys=True)+'\n')
        shutil.copyfile(OLD/name/'source_map.jsonl',dest/'source_map.jsonl')
        write_json(dest/'quality_report.json',{'tests':[g['quality'] for g in graphs],'old_graph_sha256':sha256(OLD/name/'graphs.jsonl')})
        print(name,len(graphs),sum(not evidence_errors(g) for g in graphs),flush=True)
    write_json(OUT/'comparison.json',comparisons)

def fixtures():
    import verify_phase03A2_identity as h
    from cwm_fp.lddg.static import mapping,program_identities
    from cwm_fp.lddg.effects import make_evidence,project_and_verify
    from cwm_fp.lddg.trace import FRAMEWORK_PROFILE
    from cwm_fp.lddg.extract import graph_for,validate_graph
    from cwm_fp.lddg.control import control_evidence_errors
    h.OUT=OUT;c=load(OUT/'plan.json')['config'];cp=h.tools_build(c)
    dest=OUT/'synthetic';dest.mkdir(exist_ok=True);variant=dest/'variants/fixture';build=dest/'build'
    for folder,name in [('src','FrameworkProgram.java'),('test','FrameworkFixtureTest.java')]:
        (variant/folder).mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/'tests/java/framework'/name,variant/folder/name)
    (variant/'roles.meta').write_text('{"features":[],"roles":{}}');(variant/'coverage').mkdir(exist_ok=True)
    for part,folder in [('main','src'),('tests','test')]:
        (build/part).mkdir(parents=True,exist_ok=True)
        checked([Path(c['java_home'])/'bin/javac.exe','-g','-encoding','UTF-8','-cp',os.pathsep.join(map(str,[build/'main',PRODUCER/'lib/*'])),
                 '-d',build/part,*sorted((variant/folder).glob('*.java'))],dest,dest/(part+'-build.log'))
    source=h.source_export(c,cp,list(variant.rglob('*.java')),dest);static=h.export_static(c,cp,list(build.rglob('*.class')),dest)
    binding=h.binding_for(source,list(build.rglob('*.class')),synthetic=True);write_json(dest/'compilation_binding.json',binding)
    _,mapped,_=mapping(variant,static,{},source)
    supported={'read','write','nested','stateAfter','repeated','noException'}
    cases=['read','write','nested','stateAfter','repeated','noException','unexpected','wrongOrigin','callback','reflection','arrayAfter','repeatedSameException','nullInvoke']
    outcomes=[]
    for name in cases:
        test='frameworkfixture.FrameworkFixtureTest.'+name;stage=dest/'runs'/name
        status=execute_pair(c,BYTECODE,PRODUCER,BYTECODE/'producer-classes',BYTECODE/'agent.jar',variant,test,
                            ['frameworkfixture.FrameworkProgram','frameworkfixture.FrameworkProgram$CustomNPE',HELPER],build,stage,'FAIL' if name=='noException' else 'PASS')
        assert all(status[k] for k in ('complete','parity','matches_historical_outcome')),(name,status)
        log=stage/'instrumented/run.log';full=parse_trace(log,test);evidence=make_evidence(log,test,status,OUT/'framework/static.tsv')
        row={'case':name,'expected_supported':name in supported,'original_result':status['runs']['original']['result'],
             'instrumented_result':status['runs']['instrumented']['result'],'status_path':str(stage/'status.json'),
             'complete':status['complete'],'parity':status['parity'],'physical_events':len(full),'data_edges':None,
             'certified':False,'errors':[]}
        try:raw,calls,full=project_and_verify(evidence)
        except ValueError as exc:row['errors']=[str(exc)]
        else:
            ids=program_identities(variant,static,mapped,source,binding,[(*r['domain'],r['bci']) for r in raw])
            g,events=graph_for(raw,static,mapped,{},dict(system_id='synthetic',bug_version_id='framework-effects',product_id='fixture',test_id=test),status,ids,
                               semantics_profile=FRAMEWORK_PROFILE,framework_evidence=evidence)
            validate_graph(g,events)
            write_json(dest/(name+'.graph.json'),g);write_json(dest/(name+'.events.json'),events)
            row.update(certified=g['quality']['eligible_graph_structure'],data_edges=sum(e['type']=='data_dep' for e in g['edges']),
                       errors=g['quality']['unsupported_dynamic'],effects=g.get('framework_effects'),quality=g['quality'])
            if row['certified']:assert not control_evidence_errors(g)
        outcomes.append(row);write_json(dest/'verification.json',outcomes)
        print('FIXTURE',name,row['parity'],row['certified'],row['errors'][:2],flush=True)
        assert row['certified']==(name in supported),(name,row['errors'])

def admission():
    from cwm_fp.models.graph_data import load_corpus,tensor_products
    from cwm_fp.models.graph_encoder import Limits,collate
    products=[];paths=[]
    for name,work,p in entries(True):
        path=OUT/name/'graphs.jsonl';graphs=lines(path);record={'batch_product':name,'identity':p['identity'],
            'original_result':p['observed_result'],'partitions':p['partitions'],'registered_tests':len(p['tests']),
            'semantic_graphs':sum(g['quality']['eligible_graph_structure'] for g in graphs),'folds':{},'graph_bytes':path.stat().st_size}
        for fold in FOLDS:
            ps,ss,gs,eligible,a=load_corpus(ROOT/'data/processed/manifests/products.csv',ROOT/'data/processed/manifests/tests.csv',path,split_path(fold),holdout=fold[5:] if fold.startswith('loso_') else None)
            record['folds'][fold]={'eligible':tuple(p['identity']) in eligible,'audit':a}
            if eligible and fold=='grouped':
                paths.append(path)
                default=tensor_products(ps,ss,eligible,'both',Limits());db=collate(default)
                limits=Limits(max_nodes=max(len(g['nodes']) for g in graphs),max_events=max(g['quality']['events'] for g in graphs),max_tests=len(graphs),max_edges=max(len(g['edges']) for g in graphs))
                full=tensor_products(ps,ss,eligible,'both',limits);fb=collate(full)
                assert all(full[0].audit[n]==0 for n in ('lost_test_fraction','lost_event_fraction','lost_edge_fraction'))
                assert fb.x.shape[1]==22 and not (fb.node_test[fb.edge_index[0]]!=fb.node_test[fb.edge_index[1]]).any()
                record['default_interface']={'shape':list(db.x.shape),**default[0].audit}
                record['lossless_interface']={'limits':vars(limits),'shape':list(fb.x.shape),'edge_shape':list(fb.edge_index.shape),**full[0].audit}
        products.append(record);print('ADMISSION',name,record['semantic_graphs'],record['folds']['grouped']['eligible'],flush=True)
    combined=OUT/'admitted_graphs.jsonl'
    assert sum(p.stat().st_size for p in paths)<=134217728
    with combined.open('wb') as dest:
        for p in paths:
            with p.open('rb') as src:shutil.copyfileobj(src,dest)
    aggregates={}
    for fold in FOLDS:
        *_,a=load_corpus(ROOT/'data/processed/manifests/products.csv',ROOT/'data/processed/manifests/tests.csv',combined,split_path(fold),holdout=fold[5:] if fold.startswith('loso_') else None)
        a['versions_by_partition']={part:len({tuple(r['identity'][:2]) for r in a['products'] if r['partition']==part and not r['reasons']}) for part in ('train','val','test')}
        aggregates[fold]=a
    write_json(OUT/'admission.json',{'products':products,'aggregates':aggregates,'plan_sha256':sha256(OUT/'plan.json'),
                                   'combined_bytes':combined.stat().st_size,'combined_sha256':sha256(combined),'labels_read':False,'models_fitted':0})
    print('SPLITS',{f:a['eligible_by_partition'] for f,a in aggregates.items()},flush=True)

def labels(fold):
    import verify_phase03B3_semantics as previous
    previous.OUT=OUT;previous.labels(fold)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','pilot','collect','binaries','replay','fixtures','admission','labels']);parser.add_argument('--fold',choices=FOLDS);a=parser.parse_args()
    start=time.monotonic();code=1
    try:
        if a.action=='labels':labels(a.fold)
        else:
            from cwm_fp.baselines.boundaries import install_io_guard
            install_io_guard('features')
            if a.action in ('pilot','collect'):collect(1 if a.action=='pilot' else 16)
            else:globals()[a.action]()
        code=0
    finally:
        with (OUT/'commands.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'argv':[sys.executable,*sys.argv],'seconds':time.monotonic()-start,'exit_code':code,'timestamp':datetime.now(timezone.utc).isoformat()})+'\n')
