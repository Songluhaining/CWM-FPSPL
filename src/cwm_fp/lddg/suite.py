"""Phase03B-1: a deliberately fixed, resumable one-product pilot. No fitting."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys

from cwm_fp.data.common import ROOT, KEY, read_csv, key, sha256, write_json
from .runner import command, completion, execute_pair, runtime_binding, tree_usage, budget_check

TARGET=('BankAccountTP','_MultipleBugs_.NOB_1.ID_1','model_m_ca4_0001')
OUT=ROOT/'data/processed/lddg/phase03B1_bank0001_v1'
OLD=ROOT/'data/processed/lddg/phase03A3_exception_v1'
IDENTITY=ROOT/'data/processed/lddg/phase03A2_identity_v1'
BYTECODE=ROOT/'data/processed/lddg/phase03_bank_v2'
PRODUCER=ROOT/'third_party/existing_instrumenter'
CONFIG=ROOT/'configs/phase03B1_collection.json'

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines()]
def atomic_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');write_json(temp,value);temp.replace(path)

def manifest():
    products=[p for p in read_csv(ROOT/'data/processed/manifests/products.csv',KEY) if key(p)==TARGET]
    tests=sorted((t for t in read_csv(ROOT/'data/processed/manifests/tests.csv',(*KEY,'test_id')) if key(t)==TARGET),key=lambda t:t['test_id'])
    if len(products)!=1:raise ValueError('Target product missing or duplicate')
    product=products[0]
    if product['suite_view']!='original' or product['observed_result']!='PASS':raise ValueError('Only the registered original PASS product')
    if len(tests)!=8 or len({t['test_id'] for t in tests})!=8:raise ValueError('Expected eight distinct frozen tests')
    if any(t['suite_id']!=product['suite_id'] or t['result']!='PASS' for t in tests):raise ValueError('Suite/result mismatch')
    return product,tests

def validate_saved(record):
    stage=Path(record['directory']);status=load(stage/'status.json')
    if status!=record['status']:raise ValueError('Collection/status mismatch')
    for name,h in status['inputs']['sources'].items():
        if sha256(Path(name))!=h:raise ValueError('Original source changed')
    for name,h in status['artifacts_sha256'].items():
        if sha256(stage/name)!=h:raise ValueError('Saved artifact changed: '+name)
    for mode in ('original','instrumented'):
        if completion(stage/mode/'result.tsv',record['test_id'])!=status['runs'][mode]['result']:raise ValueError('Completion mismatch')
    if not all(status[x] for x in ('complete','parity','matches_historical_outcome')):raise ValueError('Pair not qualified')
    return status

def guarded(c,args,cwd,log,seconds=180):
    rec=command(args,cwd,log,seconds,c['max_output_bytes'],True,
                max_files=c['resource_policy']['max_total_files'] if cwd==OUT else c['max_files_per_run'],
                budget=c['resource_policy'],max_rss_bytes=c['max_rss_bytes'])
    if rec['exit_code']!=0 or rec['reason']:raise RuntimeError('Bounded subprocess stopped: '+str(log))
    return rec

def prepare(c):
    product,tests=manifest();variant=Path(product['source_variant_path']);dest=OUT/TARGET[2];dest.mkdir(exist_ok=True)
    build=dest/'build';stamp=dest/'build_status.json';oldbuild=BYTECODE/TARGET[2]/'build'
    inputs={'source_sha256':{str(p):sha256(p) for folder in ('src','test') for p in sorted((variant/folder).rglob('*.java'))},
            'javac_sha256':sha256(Path(c['java_home'])/'bin/javac.exe'),
            'libs_sha256':{str(p):sha256(p) for p in sorted((PRODUCER/'lib').glob('*.jar'))}}
    if stamp.exists():
        state=load(stamp)
        if state['inputs']!=inputs:raise ValueError('Build inputs changed')
        actual={p.relative_to(build).as_posix():sha256(p) for p in build.rglob('*.class')}
        if actual!=state['class_sha256']:raise ValueError('Build cache corrupted')
    else:
        if build.exists() and any(build.rglob('*.class')):raise ValueError('Unsealed build; retain and declare a new attempt')
        before={'disks':{str(p):shutil.disk_usage(p)._asdict() for p in [ROOT,Path('D:/dataset/whn')]},
                'input_source_bytes':sum(Path(p).stat().st_size for p in inputs['source_sha256']),
                'previous_build':tree_usage(oldbuild),'generated_before':tree_usage(OUT),
                'policy':c,'registered_test_ids':[t['test_id'] for t in tests],'suite_id':product['suite_id'],
                'manifest_sha256':{n:sha256(ROOT/'data/processed/manifests'/n) for n in ('products.csv','tests.csv')}}
        write_json(OUT/'preflight.json',before)
        why,_=budget_check(c['resource_policy'],c['max_output_bytes'])
        if why:raise RuntimeError(why)
        runs={}
        for part,folder in [('main','src'),('tests','test')]:
            target=build/part;target.mkdir(parents=True,exist_ok=True)
            cp=os.pathsep.join(map(str,[build/'main',PRODUCER/'lib/*']))
            runs[part]=guarded(c,[Path(c['java_home'])/'bin/javac.exe','-J-Xmx512m','-g','-encoding','UTF-8',
                                  '-cp',cp,'-d',target,*sorted((variant/folder).rglob('*.java'))],dest,dest/(part+'-build.log'))
        actual={p.relative_to(build).as_posix():sha256(p) for p in build.rglob('*.class')}
        expected={p.relative_to(oldbuild).as_posix():sha256(p) for p in oldbuild.rglob('*.class')}
        if actual!=expected:raise ValueError('New classes differ; old static/AST binding cannot be reused')
        state={'inputs':inputs,'class_sha256':actual,'identical_to_frozen_classes':True,'runs':runs,
               'compile_scope':'shared once for all eight tests; not per-test recompilation'}
        atomic_json(stamp,state)
    # New runtime classes are byte-identical to the original classes named in the
    # source/bytecode identity proof. Preserve those identifiers and evidence paths.
    for source,target in [(OLD/TARGET[2]/'static.tsv',dest/'static.tsv'),
                          (IDENTITY/'source_evidence.json',OUT/'source_evidence.json'),
                          (IDENTITY/'compilation_binding.json',OUT/'compilation_binding.json')]:
        if target.exists() and sha256(source)!=sha256(target):raise ValueError('Frozen evidence changed')
        if not target.exists():shutil.copyfile(source,target)
    return product,tests,variant,build

def collect(c,stop_after=None):
    product,tests,variant,build=prepare(c)
    names=sorted(p.relative_to(build/'main').as_posix()[:-6].replace('/','.') for p in (build/'main').rglob('*.class'))
    old={r['test_id']:r for r in load(OLD/'collection.json')['records'] if key(r)==TARGET}
    if set(old)!={'main.Account_ESTest.test1','main.Account_ESTest.test4'}:raise ValueError('Unexpected old scope')
    records=[];new=0;cached=0
    previous=load(OUT/'collection.json')['records'] if (OUT/'collection.json').exists() else []
    previous={r['test_id']:r for r in previous}
    binding=runtime_binding(c,PRODUCER,BYTECODE/'producer-classes',BYTECODE/'agent.jar',build,names)
    if (OUT/'runtime_binding.json').exists() and load(OUT/'runtime_binding.json')!=binding:raise ValueError('Runtime binding changed')
    write_json(OUT/'runtime_binding.json',binding)
    # Authenticate old inputs against frozen historical fingerprints as well as
    # their completion seals; absence of old RSS samples is never filled in.
    frozen=load(OUT/'frozen_inputs_sha256.json')
    required=[BYTECODE/'agent.jar',*list((BYTECODE/'producer-classes').rglob('*.class')),*list((BYTECODE/TARGET[2]/'build').rglob('*.class'))]
    for p in required:
        rel=p.relative_to(ROOT).as_posix()
        if frozen.get(rel)!=sha256(p):raise ValueError('Missing/changed frozen runtime evidence: '+rel)
    for test in tests:
        name=test['test_id']
        if name in old:
            r=old[name];status=validate_saved(r)
            if status['inputs']['agent_sha256']!=sha256(BYTECODE/'agent.jar'):raise ValueError('Old agent changed')
            origin='reused_phase03A3'
        else:
            stage=OUT/TARGET[2]/name;hit=(stage/'status.json').exists()
            if stop_after is not None and new>=stop_after and not hit:continue
            status=execute_pair(c,OUT,PRODUCER,BYTECODE/'producer-classes',BYTECODE/'agent.jar',variant,name,names,build,stage,test['result'])
            new+=0 if hit else 1;cached+=int(hit)
            r={**dict(zip(KEY,TARGET)),'test_id':name,'variant':str(variant),'directory':str(stage),'status':status}
            origin='new_phase03B1'
        r={**r,'collection_origin':origin};records.append(r)
        checkpoint={**previous,**{x['test_id']:x for x in records}}
        atomic_json(OUT/'collection.json',{'config':c,'records':[checkpoint[n] for n in sorted(checkpoint)],
                    'registered_test_ids':[t['test_id'] for t in tests],'suite_complete':len(checkpoint)==8})
        print(name,origin,'parity='+str(status['parity']),flush=True)
        if not status['parity'] or not status['complete'] or not status['matches_historical_outcome']:
            raise RuntimeError('Pair not qualified; checkpoint retained, stop collection')
    # Every previously sealed test must survive a partial-resume invocation.
    if not set(previous).issubset({r['test_id'] for r in records}):raise ValueError('Resume lost a sealed record')
    run={'new_pairs':new,'new_java_test_executions':new*2,'cached_new_pairs':cached,'reused_old_pairs':2,
         'collected':len(records),'complete':len(records)==8,'controlled_checkpoint_stop':stop_after}
    with (OUT/'collection_history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(run)+'\n')
    print(json.dumps(run),flush=True)

def extract_stage(c):
    collection=load(OUT/'collection.json');_,tests=manifest()
    if {r['test_id'] for r in collection['records']}!={t['test_id'] for t in tests}:raise ValueError('Suite incomplete')
    for r in collection['records']:validate_saved(r)
    if (OUT/'graphs.jsonl').exists():raise ValueError('Graph output already sealed; use verify, not overwrite')
    guarded(c,[sys.executable,'-X','utf8','-m','cwm_fp.lddg.extract','--work-dir',OUT,
               '--source-evidence',OUT/'source_evidence.json','--identity-binding',OUT/'compilation_binding.json'],
               OUT,OUT/'extract.log')

def verify():
    from .extract import validate_graph
    from .static import load_static
    from .trace import parse_trace,Dataflow
    from .control import control_evidence_errors
    import jsonschema
    sys.path.insert(0,str(ROOT/'scripts'))
    from verify_phase03A_mapping import edge_checks
    gs=lines(OUT/'graphs.jsonl');events=lines(OUT/'events.jsonl');collection=load(OUT/'collection.json');_,tests=manifest()
    assert len(gs)==8 and {g['test_id'] for g in gs}=={t['test_id'] for t in tests} and all(key(g)==TARGET for g in gs)
    static=load_static(OUT/TARGET[2]/'static.tsv');oldgs={g['test_id']:g for g in lines(OLD/'graphs.jsonl') if key(g)==TARGET}
    oldevents=lines(OLD/'events.jsonl');rows=[];event_ids=set();static_ids={};dynamic_instances=[]
    schema_dir=Path(__file__).parent
    validators={n:jsonschema.Draft202012Validator(load(schema_dir/(n+'.schema.json'))) for n in ('graphs','events')}
    for g,r in zip(gs,collection['records']):
        assert g['test_id']==r['test_id'];validate_saved(r);validators['graphs'].validate(g)
        es=[e for e in events if e['test_id']==g['test_id']];validate_graph(g,es)
        for e in es:
            validators['events'].validate(e);assert e['canonical_event_id'] not in event_ids;event_ids.add(e['canonical_event_id'])
        raw=parse_trace(Path(r['directory'])/'instrumented/run.log',r['test_id']);flow=Dataflow(static).run(raw)
        checks=edge_checks(g,es,static,flow)
        checks['normal_path_semantics_certified']=checks.pop('whole_graph_semantics_certified')
        for ident in g['instruction_catalog']:
            if ident['scope']!='product':continue
            loc=json.dumps(ident['locator'],sort_keys=True)
            if loc in static_ids:assert static_ids[loc]==ident['instruction_id']
            static_ids[loc]=ident['instruction_id']
        if g['test_id'] in oldgs:
            old=oldgs[g['test_id']]
            for field in ('nodes','edges','instruction_catalog'):assert old[field]==g[field]
            assert es==[e for e in oldevents if key(e)==TARGET and e['test_id']==g['test_id']]
        # Explicit namespaces are audit keys, never numerical learning features.
        for v in flow.values:
            dynamic_instances.append({'test_id':r['test_id'],'value_id':v.id,'definition_event':v.definition,
                                      'object_reference':v.ref,'dynamic_key':[*TARGET,r['test_id'],v.id]})
        values=[n for n in g['nodes'] if n['type']=='value']
        branches=[e for e in flow.events if 153<=e['opcode']<=166 or e['opcode'] in (198,199)]
        rows.append({'test_id':g['test_id'],'origin':r['collection_origin'],'quality':g['quality'],
                     'nodes':len(g['nodes']),'edges':dict(Counter(e['type'] for e in g['edges'])),'edge_checks':checks,
                     'control_errors':control_evidence_errors(g),'values':len(values),
                     'observed_values':sum(n['attrs'].get('observed',False) for n in values),
                     'missing_values':sum(not n['attrs'].get('observed',False) for n in values),
                     'branch_events':len(branches),'branch_results_known':sum(e.get('branch_taken') is not None for e in branches),
                     'assertions':sum(n['type']=='assertion' for n in g['nodes']),
                     'frames':len({e['frame_id'] for e in flow.events}),
                     'object_references':len({v.ref for v in flow.values if v.ref is not None}),
                     'original_graph_unchanged':g['test_id'] in oldgs})
    maps=lines(OUT/'source_map.jsonl');anchors=[m for m in maps if not m['executable']]
    write_json(OUT/'verification.json',{'valid':True,'tests':rows,'events':len(events),'unique_event_ids':len(event_ids),
               'unique_product_instructions':len(static_ids),'source_anchors_not_events':len(anchors),
               'reused_graph_nodes_edges_events_unchanged':2,'model_training_runs':0,'labels_read':False})
    write_json(OUT/'dynamic_instances.json',{'scope':'full_product_key + test_id + local value/object/frame identity',
               'values':dynamic_instances,'no_cross_test_dynamic_merges':True})
    prefix=[]
    for g in gs:
        cut=g['quality']['events']//2
        prefix.append({'test_id':g['test_id'],'illustrative_cut_after_event':cut,
                       'future_dynamic_edges_in_full_graph':sum(e['evidence']=='dynamic' and e.get('event_index',-1)>cut for e in g['edges']),
                       'future_value_definitions_in_full_graph':sum(n['type']=='value' and n['attrs'].get('definition_event',-1)>cut for n in g['nodes']),
                       'full_graph_not_safe_as_prefix_input':True})
    write_json(OUT/'time_prefix_audit.json',{'implemented_prefix_dataset':False,'per_test':prefix,
        'required_future_masks':['events after cutoff','future def-use edges and values','full-test static executed flags',
        'branch result until successor is observed','call completion until RET marker','assertion outcome and test completion'],
        'partial_values_are_not_complete_state':True,'verified_tier':'T1','T2_enabled':False,'T3_enabled':False})
    print('Verified eight independent graphs, frozen pair equality, identities, edges and schemas',flush=True)

def admission():
    from cwm_fp.models.graph_data import load_corpus,tensor_products
    from cwm_fp.models.graph_encoder import Limits,collate
    config=load(ROOT/'configs/graph_classifier.json');audits={};encoded={}
    for fold in config['folds']:
        split=ROOT/config['split_dir']/('splits.csv' if fold=='grouped' else fold+'.csv')
        products,splits,_,eligible,audit=load_corpus(ROOT/config['products'],ROOT/config['tests'],OUT/'graphs.jsonl',split,
                                                   holdout=fold[5:] if fold.startswith('loso_') else None)
        audits[fold]=audit;items=tensor_products(products,splits,eligible,'both',Limits(**config['limits']))
        if items:
            batch=collate(items)
            assert len(items)==1 and len(items[0].tests)==8 and batch.x.shape[1]==22
            assert all(batch.node_test[a]==batch.node_test[b] for a,b in batch.edge_index.T.tolist())
            encoded[fold]={'shape':list(batch.x.shape),'edge_index_shape':list(batch.edge_index.shape),
                'tests':len(batch.test_product),'product_audit':items[0].audit,
                'per_test':[{'test_id':t.metadata['identity'][-1],'shape':list(t.x.shape),
                             'lost_event_fraction':t.metadata['lost_event_fraction'],'lost_edge_fraction':t.metadata['lost_edge_fraction']} for t in items[0].tests],
                'cross_test_edges':0,'feature_schema':'semantic_identity_v2'}
    write_json(OUT/'admission.json',audits)
    write_json(OUT/'encoding_check.json',{'folds':encoded,'labels_read':False,'model_training_runs':0})
    print({k:a['eligible_by_partition'] for k,a in audits.items()},flush=True)

def profile():
    rows=[]
    for r in load(OUT/'collection.json')['records']:
        item={'test_id':r['test_id'],'origin':r['collection_origin'],'run_directory':r['directory'],'modes':{}}
        for mode,rec in r['status']['runs'].items():
            p=Path(r['directory'])/mode
            item['modes'][mode]={'seconds':rec['execution']['seconds'],'run_log_bytes':(p/'run.log').stat().st_size,
                                'all_files':tree_usage(p),'resource':rec['execution'].get('resource'),
                                'reason':rec['execution']['reason'],'result':rec['result']}
        a,b=[item['modes'][m]['seconds'] for m in ('original','instrumented')]
        item.update(overhead_seconds=round(b-a,3),overhead_ratio=b/a);rows.append(item)
    oldbytes=sum(tree_usage(Path(r['directory']))['bytes'] for r in load(OUT/'collection.json')['records'] if r['collection_origin']=='reused_phase03A3')
    write_json(OUT/'resource_profile.json',{'per_test':rows,'build':load(OUT/TARGET[2]/'build_status.json'),
        'extraction':load(OUT/'extract.command.json'),'generated_directory_before_profile':tree_usage(OUT),
        'referenced_old_pair_bytes':oldbytes,'disk_now':shutil.disk_usage(OUT)._asdict(),
        'collection_history':lines(OUT/'collection_history.jsonl'),
        'memory_caveat':'Observed direct-child Windows peak working set up to last poll; not a guaranteed full lifetime or process-tree peak.'})
    print('Resource profile recorded',flush=True)

def preserve():
    counts={}
    for file,relative in [('frozen_inputs_sha256.json',True),('readonly_source_sha256.json',False)]:
        hashes=load(OUT/file)
        for name,h in hashes.items():
            if sha256(ROOT/name if relative else Path(name))!=h:raise ValueError('Preservation failure: '+name)
        counts[file]=len(hashes)
    for name,h in load(OUT/'runtime_binding.json')['artifacts_sha256'].items():
        if sha256(Path(name))!=h:raise ValueError('Runtime input changed')
    for r in load(OUT/'collection.json')['records']:validate_saved(r)
    write_json(OUT/'preservation.json',{'valid':True,'unchanged':counts,'runtime_and_original_inputs_unchanged':True})
    print(counts,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['collect','extract','verify','admission','profile','preserve'],required=True)
    p.add_argument('--stop-after-new-tests',type=int);a=p.parse_args();c=load(CONFIG)
    os.environ['PYTHONPATH']=str(ROOT/'src')
    if c['target']!=list(TARGET) or Path(c['resource_policy']['root']).resolve()!=OUT:raise ValueError('Fixed one-product scope')
    if a.stage!='preserve':
        from cwm_fp.baselines.boundaries import install_io_guard
        install_io_guard('features')
    if a.stage=='collect':collect(c,a.stop_after_new_tests)
    elif a.stage=='extract':extract_stage(c)
    else:globals()[a.stage]()

if __name__=='__main__':main()
