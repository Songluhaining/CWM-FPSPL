"""Six frozen real traces, eight bounded artificial exception cases; never train."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cwm_fp.data.common import KEY,sha256,write_json
from cwm_fp.lddg.runner import checked,completion,execute_pair
from cwm_fp.lddg.static import SourceEvidence,load_static,mapping,program_identities
from cwm_fp.lddg.trace import Dataflow,parse_trace
from cwm_fp.lddg.extract import extract,graph_for,validate_graph,identity_errors
from cwm_fp.lddg.control import control_evidence_errors
from verify_phase03A_mapping import SELECTED,edge_checks
import verify_phase03A2_identity as helpers

OLD=ROOT/'data/processed/lddg/phase03A2_identity_v1'
OUT=ROOT/'data/processed/lddg/phase03A3_exception_v1'
BYTECODE=ROOT/'data/processed/lddg/phase03_bank_v2'
helpers.OUT=OUT  # Reuse the existing exporters, always writing in the new directory.
CASES=('normalTryCatch','sameMethodThrow','callerCatch','normalFinally','exceptionalFinally','ordinaryStore','untracedException','implicitDivide')
SUPPORTED={'normalTryCatch','normalFinally','ordinaryStore'}
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]


def real(c,cp):
    collection=load(OLD/'collection.json');records=collection['records']
    assert [(r['product_id'],r['test_id']) for r in records]==SELECTED
    assert {r['system_id'] for r in records}=={'BankAccountTP'}
    assert {r['bug_version_id'] for r in records}=={'_MultipleBugs_.NOB_1.ID_1'}
    seen=set()
    for r in records:
        stage=Path(r['directory']);status=load(stage/'status.json');assert status==r['status']
        for name,h in status['inputs']['sources'].items():assert sha256(Path(name))==h
        for name,h in status['artifacts_sha256'].items():assert sha256(stage/name)==h
        assert status['complete'] and status['parity'] and status['matches_historical_outcome']
        for mode in ('original','instrumented'):assert completion(stage/mode/'result.tsv',r['test_id'])==status['runs'][mode]['result']
        if r['product_id'] in seen:continue
        seen.add(r['product_id']);dest=OUT/r['product_id'];dest.mkdir(exist_ok=True)
        helpers.export_static(c,cp,sorted((BYTECODE/r['product_id']/'build').rglob('*.class')),dest)
    write_json(OUT/'collection.json',collection|{'control_source_collection':str(OLD/'collection.json')})
    extract(OUT,OUT,OLD/'source_evidence.json',OLD/'compilation_binding.json')
    import jsonschema
    schemas={n:jsonschema.Draft202012Validator(load(ROOT/'src/cwm_fp/lddg'/(n+'.schema.json'))) for n in ('graphs','events')}
    before=lines(OLD/'graphs.jsonl');after=lines(OUT/'graphs.jsonl');events=lines(OUT/'events.jsonl')
    assert len(after)==6 and events==lines(OLD/'events.jsonl')
    assert lines(OUT/'source_map.jsonl')==lines(OLD/'source_map.jsonl')
    for e in events:schemas['events'].validate(e)
    checks=[];diagnoses=[]
    for old,g,r in zip(before,after,records):
        schemas['graphs'].validate(g)
        assert old['nodes']==g['nodes'] and old['edges']==g['edges'] and old['instruction_catalog']==g['instruction_catalog']
        assert not identity_errors(g) and not control_evidence_errors(g)
        assert g['quality']['eligible_graph_structure'] and not g['quality']['exception_control_supported']
        raw=parse_trace(Path(r['directory'])/'instrumented/run.log',r['test_id']);static=load_static(OUT/r['product_id']/'static.tsv')
        old_static=load_static(OLD/r['product_id']/'static.tsv')
        for field in ('methods','handlers','fields','allocations','controls','gaps','classes','method_flags','calls'):assert static[field]==old_static[field]
        flow=Dataflow(static).run(raw);oldflow=Dataflow(old_static).run(raw)
        assert flow.events==oldflow.events and flow.values==oldflow.values and flow.links==oldflow.links
        selected=[e for e in events if (e['product_id'],e['test_id'])==(r['product_id'],r['test_id'])]
        edges=edge_checks(g,selected,static,flow)
        edges['supported_path_semantics_certified']=edges.pop('whole_graph_semantics_certified')
        edges['certification_scope']='normal-path-only; static normal-flow projection; no exception propagation certification'
        checks.append({**{k:r[k] for k in (*KEY,'test_id')},'before_quality':old['quality'],'after_quality':g['quality'],
                       'control_assessment':g['control_flow_evidence']['assessment'],'edge_checks':edges,
                       'all_nodes_edges_events_value_versions_frames_and_results_unchanged':True,
                       'result':r['status']['runs']['instrumented']['result']})
        if 'Transaction' in r['test_id']:
            d=('main.Transaction','transfer','(Lmain/Account;Lmain/Account;I)Z')
            method=next(m for m in g['control_flow_evidence']['proof']['methods'] if m['domain']==list(d))
            direct=[e for e in flow.events if e['domain']==d]
            assert not any(e['bci'] in {h['handler'] for h in method['handlers']} for e in direct)
            assert g['control_flow_evidence']['assessment']['classification']=='A'
            assert direct[-1]['opcode'] in range(172,178)
            # Independent javap rendering of the original class, not instrumentation offsets.
            checked([Path(c['java_home'])/'bin/javap.exe','-c','-v','-p',static['classes'][d[0]]['class_file']],OUT,
                    OUT/r['product_id']/'Transaction.javap.txt')
            diagnoses.append({'product_id':r['product_id'],'test_id':r['test_id'],'classification':'A',
                'trace':str(Path(r['directory'])/'instrumented/run.log'),'trace_sha256':sha256(Path(r['directory'])/'instrumented/run.log'),
                'method':method,'transfer_events':[{'event':e['event_index'],'bci':e['bci'],'opcode':e['opcode'],'frame':e['frame_id'],'returned':e['call_returned']} for e in direct],
                'all_events':g['control_flow_evidence']['proof']['events'],'assessment':g['control_flow_evidence']['assessment'],
                'result':r['status']['runs']['instrumented']['result']})
    write_json(OUT/'exception_diagnosis.json',{'tests':diagnoses})
    write_json(OUT/'verification.json',{'valid':True,'real_traces_reprocessed':6,'new_real_test_executions':0,'events':len(events),
        'identity_complete_graphs':6,'normal_path_qualified_graphs':6,'exception_semantics_certified_graphs':0,
        'control_edges':sum(x['edge_checks']['control_endpoints_checked'] for x in checks),
        'data_edges':sum(x['edge_checks']['data_edges'] for x in checks),'per_test':checks,'model_training_runs':0})
    print('Six frozen graphs: identities/nodes/events/dependencies unchanged; 6 normal-path-only; no full exception certification')


def fixtures(c,cp):
    import jsonschema
    schemas={n:jsonschema.Draft202012Validator(load(ROOT/'src/cwm_fp/lddg'/(n+'.schema.json'))) for n in ('graphs','events')}
    dest=OUT/'synthetic';dest.mkdir(exist_ok=True);variant=dest/'variants/fixture'
    for folder in ('src','test'):(variant/folder).mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/'tests/java/exception/ExceptionProgram.java',variant/'src/ExceptionProgram.java')
    shutil.copyfile(ROOT/'tests/java/exception/ExceptionFixtureTest.java',variant/'test/ExceptionFixtureTest.java')
    (variant/'roles.meta').write_text(json.dumps({'features':[],'roles':{}}),encoding='utf-8');(variant/'coverage').mkdir(exist_ok=True)
    build=dest/'build';producer=ROOT/'third_party/existing_instrumenter'
    for part,folder in [('main','src'),('tests','test')]:
        (build/part).mkdir(parents=True,exist_ok=True)
        checked([Path(c['java_home'])/'bin/javac.exe','-g','-encoding','UTF-8','-cp',os.pathsep.join(map(str,[build/'main',producer/'lib/*'])),
                 '-d',build/part,*sorted((variant/folder).glob('*.java'))],dest,dest/(part+'-build.log'))
    source=helpers.source_export(c,cp,list(variant.rglob('*.java')),dest)
    static=helpers.export_static(c,cp,sorted(build.rglob('*.class')),dest)
    binding=helpers.binding_for(source,list(build.rglob('*.class')),synthetic=True);write_json(dest/'compilation_binding.json',binding)
    _,mapped,_=mapping(variant,static,{},source);outcomes=[];new_executions=0
    for name in CASES:
        identity='exceptionfixture.ExceptionFixtureTest.'+name;stage=dest/name;cached=(stage/'status.json').exists()
        status=execute_pair(c,BYTECODE,producer,BYTECODE/'producer-classes',BYTECODE/'agent.jar',variant,identity,
                            ['exceptionfixture.ExceptionProgram'],build,stage,'PASS')
        new_executions+=0 if cached else 2
        assert status['complete'] and status['parity'] and status['matches_historical_outcome'],name
        raw=parse_trace(stage/'instrumented/run.log',identity)
        ids=program_identities(variant,static,mapped,source,binding,[(*r['domain'],r['bci']) for r in raw])
        g,events=graph_for(raw,static,mapped,{},dict(system_id='synthetic',bug_version_id='exception-fixture',product_id='fixture',test_id=identity),status,ids)
        validate_graph(g,events)
        schemas['graphs'].validate(g)
        for event in events:schemas['events'].validate(event)
        assert g['quality']['eligible_graph_structure']==(name in SUPPORTED),(name,g['quality'])
        if name in SUPPORTED:assert not identity_errors(g) and not control_evidence_errors(g)
        else:
            assert control_evidence_errors(g)
            assert not any(e['type'] in {'data_dep','produces','consumes'} for e in g['edges'])
            assert all(not e['def_ids'] and not e['use_ids'] for e in events)
        write_json(stage/'graph.json',g);write_json(stage/'events.json',events)
        outcomes.append({'case':name,'synthetic':True,'parity':True,'result':status['runs']['instrumented']['result'],
            'events':len(events),'identity_errors':identity_errors(g),'quality':g['quality'],'control':g['control_flow_evidence']['assessment']})
        print(name,'PASS parity; semantics',g['quality']['eligible_graph_structure'],flush=True)
    history=dest/'verification_runs.jsonl'
    with history.open('a',encoding='utf-8') as f:f.write(json.dumps({'new_java_test_executions':new_executions,'valid':True})+'\n')
    write_json(dest/'verification.json',{'valid':True,'cases':outcomes,'new_java_test_executions_this_command':new_executions,
        'total_new_java_test_executions':sum(x['new_java_test_executions'] for x in lines(history)),
        'supported':len(SUPPORTED),'safely_rejected':len(CASES)-len(SUPPORTED),'benchmark_experiments':False})


def admission():
    from cwm_fp.models.graph_data import load_corpus
    from cwm_fp.models.graph_encoder import encode_test
    config=load(ROOT/'configs/graph_classifier.json');audits={};encodings=[]
    for fold in config['folds']:
        split=ROOT/config['split_dir']/('splits.csv' if fold=='grouped' else fold+'.csv')
        _,_,_,eligible,audit=load_corpus(ROOT/config['products'],ROOT/config['tests'],OUT/'graphs.jsonl',split,holdout=fold[5:] if fold.startswith('loso_') else None)
        assert not eligible and audit['identity_and_semantics_qualified']==6
        audits[fold]=audit
    for g in lines(OUT/'graphs.jsonl'):
        t=encode_test(g,'both');encodings.append({'product_id':g['product_id'],'test_id':g['test_id'],'shape':list(t.x.shape),'preview_only':t.metadata['preview_only'],
            'lost_event_fraction':t.metadata['lost_event_fraction'],'lost_edge_fraction':t.metadata['lost_edge_fraction']})
    for name in CASES:
        g=load(OUT/'synthetic'/name/'graph.json')
        if name in SUPPORTED:encode_test(g,'both')
        else:
            try:encode_test(g,'both')
            except ValueError:pass
            else:raise AssertionError('Unsupported exception graph admitted')
    write_json(OUT/'admission.json',audits);write_json(OUT/'encoding_check.json',{'real_test_encodings':encodings,'eligible_products':0,'labels_read':False,'model_training_runs':0})
    print('6 test graphs normal-path-only; 0 complete eligible products; 5 artificial exceptional cases refused')


def preserve():
    frozen=load(OUT/'frozen_inputs_sha256.json');readonly=load(OUT/'readonly_source_sha256.json')
    for name,h in frozen.items():assert sha256(ROOT/name)==h,name
    for name,h in readonly.items():assert sha256(Path(name))==h,name
    write_json(OUT/'preservation.json',{'valid':True,'frozen_files_unchanged':len(frozen),'readonly_source_files_unchanged':len(readonly)})
    print('Unchanged:',len(frozen),'frozen artifacts;',len(readonly),'read-only source/library files')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['real','fixtures','admission','preservation'],required=True);a=p.parse_args()
    if a.stage=='preservation':preserve()
    else:
        from cwm_fp.baselines.boundaries import install_io_guard
        install_io_guard('features');c=load(OLD/'collection.json')['config']
        if a.stage=='admission':admission()
        else:
            cp=helpers.tools_build(c)
            if a.stage=='real':real(c,cp)
            else:fixtures(c,cp)
