"""Bounded phase03A: only the frozen six tests; no label access or model import."""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cwm_fp.data.common import KEY, indexed, read_csv, sha256, write_json
from cwm_fp.lddg.runner import checked, completion, execute_pair
from cwm_fp.lddg.static import SourceEvidence, load_static, mapping
from cwm_fp.lddg.trace import Dataflow, parse_trace
from cwm_fp.lddg.extract import extract, sid, validate_graph

WORK=ROOT/'data/processed/lddg/phase03_bank_v2'
OUT=ROOT/'data/processed/lddg/phase03A_mapping_v1'
SELECTED=[('model_m_ca4_0001','main.Account_ESTest.test1'),('model_m_ca4_0001','main.Account_ESTest.test4'),
          ('model_m_ca4_0002','main.Account_ESTest.test00'),('model_m_ca4_0002','main.Transaction_ESTest.test5'),
          ('model_m_ca4_0003','main.Account_ESTest.test00'),('model_m_ca4_0003','main.Transaction_ESTest.test5')]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def ident(r):return tuple(r[k] for k in (*KEY,'test_id'))
def map_key(r):return tuple(r.get(k) for k in (*KEY,'composed_class','composed_method','descriptor','composed_line','bytecode_offset'))


def source_export(collection):
    java=Path(collection['config']['java_home']);build=OUT/'source-parser';build.mkdir(exist_ok=True)
    checked([java/'bin/javac.exe','-encoding','UTF-8','-cp',java/'lib/tools.jar','-d',build,
             ROOT/'src/cwm_fp/lddg/java/Phase03SourceExport.java'],OUT,OUT/'source-parser-build-final.log')
    sources=set()
    for r in collection['records']:
        v=Path(r['variant']);sources.update((v/'src').rglob('*.java'))
        for feature in load(v/'roles.meta')['features']:sources.update((v.parent.parent/'features'/feature).rglob('*.java'))
    checked([java/'bin/java.exe','-Xmx256m','-cp',os.pathsep.join(map(str,[build,java/'lib/tools.jar'])),
             'Phase03SourceExport',OUT/'source-ast.jsonl',*sorted(sources)],OUT,OUT/'source-parser-final.log')
    write_json(OUT/'source_evidence.json',{'producer':'JDK8_JavacTask_parse_only',
        'source_sha256':{str(p):sha256(p) for p in sorted(sources)},'records':lines(OUT/'source-ast.jsonl')})


def edge_checks(graph, events, static, flow):
    validate_graph(graph,events);nodes={n['id']:n for n in graph['nodes']}
    runtime={r['event_index']:r for r in flow.events};values={v.id:v for v in flow.values}
    expected={(v.id,v.definition,event,role) for val,event,role in flow.links
              for v in [values[val]] if v.definition is not None and v.definition<event}
    actual=set();control=0
    for e in graph['edges']:
        if e['type']=='data_dep':
            d,u=e['definition_event'],e['event_index'];v=values[e['value_id']]
            assert (e['source'],e['target'])==('e:'+str(d),'e:'+str(u))
            assert runtime[d]['thread_id']==runtime[u]['thread_id'] and d<u
            assert v.definition==d and v.certain
            assert v.id in runtime[d]['def_ids'] and v.id in runtime[u]['use_ids']
            assert runtime[d]['frame_id'] is not None and runtime[u]['frame_id'] is not None
            actual.add((v.id,d,u,e['role']))
        elif e['type']=='control_dep':
            a,b=nodes[e['source']]['attrs'],nodes[e['target']]['attrs']
            domain=tuple(a[k] for k in ('composed_class','composed_method','descriptor'))
            assert domain==tuple(b[k] for k in ('composed_class','composed_method','descriptor'))
            assert (a['bytecode_offset'],b['bytecode_offset']) in static['controls'][domain]
            assert e['evidence']=='static' and e['exception_control'] is False
            if b.get('executed') and 'event_index' in b:
                observed=runtime[b['event_index']]
                assert observed['domain']==domain and observed['bci']==b['bytecode_offset']
            control+=1
    assert actual==expected
    return {'control_endpoints_checked':control,'data_endpoints_and_order_checked':len(actual),
            'data_edges':sum(e['type']=='data_dep' for e in graph['edges']),
            'control_semantics':'static_normal_postdominance_not_dynamic_controller_instance',
            'whole_graph_semantics_certified':graph['quality']['eligible_graph_structure']}


def main(rerun=False):
    from cwm_fp.baselines.boundaries import install_io_guard
    install_io_guard('features')
    frozen=load(OUT/'frozen_inputs_sha256.json')
    for name,h in frozen.items():assert sha256(ROOT/name)==h, name
    collection=load(WORK/'collection.json');config=collection['config']
    assert [(r['product_id'],r['test_id']) for r in collection['records']]==SELECTED
    assert {r['system_id'] for r in collection['records']}=={'BankAccountTP'}
    assert {r['bug_version_id'] for r in collection['records']}=={'_MultipleBugs_.NOB_1.ID_1'}
    assert config['max_tests']==6 and config['max_seconds_per_test']==30 and config['java_heap_mb']==512
    before=lines(WORK/'source_map.jsonl');old_graphs={ident(g):g for g in lines(WORK/'graphs.jsonl')}
    assert Counter(r.get('mapping_confidence',r.get('status')) for r in before)=={
        'xml_roles_line_checked':381,'unmapped':80,'unmapped_or_non_executable':33}
    products=indexed(read_csv(ROOT/'data/processed/manifests/products.csv',KEY))
    old_by_product=defaultdict(list)
    for row in before:old_by_product[row['product_id']].append(row)
    source_export(collection);source=SourceEvidence(OUT/'source_evidence.json')
    copied=set();records=[];new_executions=0;checks=[];input_hashes={}
    for r in collection['records']:
        p=r['product_id'];v=Path(r['variant']);stage=Path(r['directory']);status=load(stage/'status.json')
        assert status==r['status'] and status['complete'] and status['parity'] and status['matches_historical_outcome']
        for name,h in status['inputs']['sources'].items():assert sha256(Path(name))==h;input_hashes[name]=h
        for name,h in status['artifacts_sha256'].items():assert sha256(stage/name)==h
        for mode in ('original','instrumented'):assert completion(stage/mode/'result.tsv',r['test_id'])==status['runs'][mode]['result']
        static=load_static(WORK/p/'static.tsv')
        if p not in copied:
            features=json.loads(products[tuple(r[k] for k in KEY)]['features_json'])
            original,_,_=mapping(v,static,features)
            expected=[{k:x for k,x in row.items() if k not in KEY} for row in old_by_product[p]]
            assert original==expected,'Legacy mapping does not reproduce frozen evidence'
            dest=OUT/p;dest.mkdir(exist_ok=True)
            java=Path(config['java_home'])/'bin/java.exe'
            checked([java,'-Xmx256m','-cp',os.pathsep.join(map(str,[WORK/'producer-classes',ROOT/'third_party/existing_instrumenter/lib/*'])),
                     'Phase03StaticExport',dest/'static.tsv',*sorted((WORK/p/'build').rglob('*.class'))],OUT,dest/'static-reexport.log')
            assert sha256(dest/'static.tsv')==sha256(WORK/p/'static.tsv')
            copied.add(p)
            for path in [v/'roles.meta',*sorted((v/'coverage').glob('spectrum_*_coverage.xml'))]:input_hashes[str(path)]=sha256(path)
        new_stage=OUT/'reruns'/p/r['test_id']
        if rerun:
            build=WORK/p/'build'
            names=[f.relative_to(build/'main').as_posix()[:-6].replace('/','.') for f in sorted((build/'main').rglob('*.class'))]
            was_cached=(new_stage/'status.json').exists()
            new_status=execute_pair(config,WORK,ROOT/'third_party/existing_instrumenter',WORK/'producer-classes',WORK/'agent.jar',
                                    v,r['test_id'],names,build,new_stage,status['inputs']['expected'])
            new_executions+=0 if was_cached else 2
        else:
            new_status=load(new_stage/'status.json')
        for name,h in new_status['artifacts_sha256'].items():assert sha256(new_stage/name)==h
        assert new_status['inputs']==status['inputs']
        assert new_status['complete'] and new_status['parity'] and new_status['matches_historical_outcome']
        for mode in ('original','instrumented'):
            assert completion(new_stage/mode/'result.tsv',r['test_id'])==status['runs'][mode]['result']
        raw=parse_trace(stage/'instrumented/run.log',r['test_id']);new=parse_trace(new_stage/'instrumented/run.log',r['test_id'])
        # Ignore object identityHashCode callback text, never used by the dataflow engine.
        location=lambda rs:[tuple(x[k] if k!='domain' else tuple(x[k]) for k in ('domain','line','bci','opcode','entry','call_returned')) for x in rs]
        assert location(raw)==location(new)
        flow,other=Dataflow(static).run(raw),Dataflow(static).run(new)
        assert flow.links==other.links and flow.gaps==other.gaps
        assert [(x.definition,x.ref,x.width,x.observed,x.certain) for x in flow.values]==[(x.definition,x.ref,x.width,x.observed,x.certain) for x in other.values]
        records.append({**r,'directory':str(new_stage),'status':new_status})
        checks.append({'product_id':p,'test_id':r['test_id'],'result':new_status['runs']['original']['result']['outcome'],
                       'parity':True,'frozen_event_order_and_dataflow_equal':True,
                       'test_JVM_seconds':sum(x['execution']['seconds'] for x in new_status['runs'].values())})
    write_json(OUT/'collection.json',{'records':records,'config':config,'frozen_input_collection':str(WORK/'collection.json')})
    extract(OUT,OUT,OUT/'source_evidence.json')
    import jsonschema
    schemas={name:jsonschema.Draft202012Validator(load(ROOT/'src/cwm_fp/lddg'/(name+'.schema.json'))) for name in ('graphs','events')}
    after=lines(OUT/'source_map.jsonl');graphs=lines(OUT/'graphs.jsonl');events=lines(OUT/'events.jsonl');grouped=defaultdict(list)
    for event in events:schemas['events'].validate(event);grouped[ident(event)].append(event)
    for g,r,check in zip(graphs,records,checks):
        schemas['graphs'].validate(g);static=load_static(OUT/r['product_id']/'static.tsv')
        raw=parse_trace(Path(r['directory'])/'instrumented/run.log',r['test_id']);flow=Dataflow(static).run(raw)
        check.update(edge_checks(g,grouped[ident(g)],static,flow))
        old=old_graphs[ident(g)]
        for kind in ('control_dep','data_dep','produces','consumes','temporal','observes'):
            assert [e for e in old['edges'] if e['type']==kind]==[e for e in g['edges'] if e['type']==kind]
        assert [n['id'] for n in old['nodes']]==[n['id'] for n in g['nodes']]
        assert all((x['line'],x['opcode'])==(static['methods'][x['domain']][x['bci']]['line'],static['methods'][x['domain']][x['bci']]['opcode']) for x in raw)
        check['quality']=g['quality'];check['before_quality']=old['quality']
        check['nonproduct_event_classes']=dict(Counter(x['domain'][0] for x in raw if x['domain'][0] not in source.product_classes(Path(r['variant']))))
        assert set(check['nonproduct_event_classes'])=={r['test_id'].rsplit('.',1)[0]}
    assert {map_key(r) for r in before}=={map_key(r) for r in after}
    changed={map_key(r):r for r in after};diagnosis=[]
    for old in before:
        new=changed[map_key(old)]
        if old.get('canonical_statement_id'):assert new['canonical_source_line_id']
        else:diagnosis.append({'before':old,'after':new})
    write_json(OUT/'record_diagnosis.json',{'records':diagnosis,'instructions':80,'coverage_positions':33})
    for name,h in frozen.items():assert sha256(ROOT/name)==h
    for name,h in input_hashes.items():assert sha256(Path(name))==h
    for name,h in load(OUT/'source_evidence.json')['source_sha256'].items():assert sha256(Path(name))==h
    verification={'valid':True,'scope':'phase03A_six_frozen_tests_only','new_test_JVM_executions_this_command':new_executions,
        'paired_real_test_runs_in_output':6,'model_training_runs':0,'frozen_files_unchanged':len(frozen),
        'source_and_test_java_files_unchanged':len({name for name in input_hashes if name.endswith('.java')}),
        'source_evidence_files':len(source.text),'all_schema_valid':True,'all_dependency_endpoints_checked':True,
        'per_test':checks,'before_map_counts':dict(Counter(r.get('mapping_confidence',r.get('status')) for r in before)),
        'after_map_counts':dict(Counter(r.get('mapping_confidence',r.get('status')) for r in after)),
        'old_80_root_causes':dict(Counter(x['after']['source_construct'] for x in diagnosis if x['before']['executable'])),
        'old_33_root_causes':dict(Counter(x['after']['reason'] for x in diagnosis if not x['before']['executable'])),
        'source_inputs_sha256':input_hashes}
    write_json(OUT/'verification.json',verification)
    history=OUT/'verification_runs.jsonl'
    with history.open('a',encoding='utf-8') as f:f.write(json.dumps({'new_test_JVM_executions':new_executions,'valid':True})+'\n')
    print(json.dumps({k:v for k,v in verification.items() if k not in {'per_test','source_inputs_sha256'}},ensure_ascii=False,indent=2))

def admission_only():
    """Existing phase04 gate, read-only cohort join; no encoder or model runs."""
    from cwm_fp.baselines.boundaries import install_io_guard
    from cwm_fp.models.graph_data import load_corpus
    install_io_guard('features')
    config=load(ROOT/'configs/graph_classifier.json');result={}
    for fold in config['folds']:
        split=ROOT/config['split_dir']/('splits.csv' if fold=='grouped' else fold+'.csv')
        _,_,_,eligible,audit=load_corpus(ROOT/config['products'],ROOT/config['tests'],OUT/'graphs.jsonl',split,
                                       holdout=fold[5:] if fold.startswith('loso_') else None)
        assert not eligible
        result[fold]=audit
        print(fold,audit['eligible_by_partition'],audit['products'])
    write_json(OUT/'admission.json',result)
    print('Model creation, training, predictions and label reads: 0')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--rerun',action='store_true',help='Run/reuse only the six explicitly frozen original tests')
    parser.add_argument('--admission-only',action='store_true',help='Read-only use of the existing phase04 gate; requires torch environment')
    args=parser.parse_args()
    if args.admission_only:admission_only()
    else:main(args.rerun)
