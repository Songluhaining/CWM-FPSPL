"""Frozen, label-blind multi-product collection; reuse existing program analysis.

No fitting. Products have separate directories even when their product IDs match.
Unsupported tests remain in the collection and prevent whole-product admission.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

from cwm_fp.data.common import ROOT, KEY, key, read_csv, sha256, write_json
from .runner import command, execute_pair, runtime_binding, tree_usage, budget_check
from .suite import atomic_json, load, lines, validate_saved

CONFIG=ROOT/'configs/phase03B2_collection.json'
BYTECODE=ROOT/'data/processed/lddg/phase03_bank_v2'
PRODUCER=ROOT/'third_party/existing_instrumenter'
VERSION='phase03B2-batch-0.2'

def read_plan():
    c=load(CONFIG);out=Path(c['out_dir']).resolve()
    if out!=Path('D:/dataset/whn/cwm_fp_generated/data/processed/lddg/phase03B2_v1').resolve():
        raise ValueError('Unexpected generated root')
    if sha256(CONFIG)!=sha256(out/'plan.json') or sha256(CONFIG)!=(out/'plan.sha256').read_text().strip():
        raise ValueError('Frozen plan changed')
    for name,h in c['manifest_split_sha256'].items():
        if sha256(ROOT/name)!=h:raise ValueError('Frozen manifest/split changed')
    keys=[tuple(p['identity']) for p in c['products']]
    if len(keys)!=len(set(keys)) or len({k[:2] for k in keys})>12:raise ValueError('Duplicate product or version budget')
    if sum(p['observed_result']=='PASS' for p in c['products'])>24:raise ValueError('PASS product budget')
    if sum(len(p['tests']) for p in c['products'])!=c['new_registered_tests']:raise ValueError('Test budget changed')
    public={key(p):p for p in read_csv(ROOT/'data/processed/manifests/products.csv',KEY)}
    actual={}
    for t in read_csv(ROOT/'data/processed/manifests/tests.csv',(*KEY,'test_id')):
        actual.setdefault(key(t),[]).append({'test_id':t['test_id'],'result':t['result']})
    for p in c['products']:
        k=tuple(p['identity']);row=public[k]
        if row['suite_view']!='original' or row['suite_id']!=p['suite_id'] or row['observed_result']!=p['observed_result']:
            raise ValueError('Product suite mismatch')
        if sorted(actual[k],key=lambda t:t['test_id'])!=p['tests']:raise ValueError('Incomplete selected original suite')
    return c,out

def product_dir(out,index):return out/f'p{index:03d}'

class CollectionBudget:
    """Persist actual wall time; charge unfinished reservations conservatively."""
    def __init__(self,c,out):
        self.c=c;self.out=out;self.path=out/'collection_budget.json'
        self.state=load(self.path) if self.path.exists() else {'seconds':0.,'reserved_test_processes':0,'interrupted_reservations':[],'pending':None}
        if self.state['pending']:
            self.state['seconds']+=self.state['pending']['seconds']
            self.state['interrupted_reservations'].append(self.state['pending']);self.state['pending']=None
            atomic_json(self.path,self.state)

    def run(self,name,seconds,processes,fn):
        if self.state['seconds']+seconds>self.c['collection_time_budget_seconds']:raise RuntimeError('batch_time_budget')
        if self.state['reserved_test_processes']+processes>self.c['new_java_test_process_limit']:raise RuntimeError('batch_process_budget')
        why,_=budget_check(self.c['resource_policy'],33554432)
        if why:raise RuntimeError(why)
        self.state['pending']={'name':name,'seconds':seconds,'test_processes':processes}
        self.state['reserved_test_processes']+=processes;atomic_json(self.path,self.state);start=time.monotonic()
        try:return fn()
        finally:
            self.state['seconds']+=time.monotonic()-start;self.state['pending']=None;atomic_json(self.path,self.state)

def local_config(c,work):
    # Eleven disjoint 256 MiB work areas plus small shared records fit the frozen
    # 4 GiB total. Local monitoring avoids repeatedly walking every old product.
    return {**c,'resource_policy':{'root':str(work),'max_total_bytes':268435456,
                                  'max_total_files':4096,'min_free_bytes':c['resource_policy']['min_free_bytes']}}

def checked(c,args,cwd,log,seconds=180,*,large=False):
    rec=command(args,cwd,log,seconds=seconds,max_bytes=268435456 if large else 33554432,
                monitor_files=True,max_files=4096,budget=c['resource_policy'],max_rss_bytes=2147483648 if large else c['max_rss_bytes'])
    if rec['exit_code']!=0 or rec['reason']:raise RuntimeError(f'Bounded command failed: {log}: {rec["reason"] or rec["exit_code"]}')
    return rec

def tools_prepare(c,out,budget):
    dest=out/'tools';dest.mkdir(exist_ok=True)
    source=[ROOT/'src/cwm_fp/lddg/java'/n for n in ('Phase03SourceExport.java','Phase03StaticExport.java')]
    inputs={str(p):sha256(p) for p in source};stamp=dest/'status.json'
    cp=os.pathsep.join(map(str,[dest,PRODUCER/'lib/*',Path(c['java_home'])/'lib/tools.jar']))
    if stamp.exists():
        saved=load(stamp)
        if saved['inputs']!=inputs:raise ValueError('Exporter source changed')
        for n,h in saved['classes'].items():
            if sha256(dest/n)!=h:raise ValueError('Exporter class changed')
        return cp
    budget.run('compile_existing_exporters',180,0,lambda:checked(local_config(c,dest),
        [Path(c['java_home'])/'bin/javac.exe','-J-Xmx512m','-encoding','UTF-8','-cp',cp,'-d',dest,*source],dest,dest/'build.log'))
    write_json(stamp,{'inputs':inputs,'classes':{p.name:sha256(p) for p in dest.glob('*.class')}})
    return cp

def prepare(c,p,work,cp,budget):
    variant=Path(p['source_variant_path']);build=work/'b';stamp=work/'prepare.json';lc=local_config(c,work)
    for n,h in p['source_sha256'].items():
        if sha256(Path(n))!=h:raise ValueError('Selected source changed')
    if sha256(variant/'roles.meta')!=p['roles_sha256']:raise ValueError('Configuration roles changed')
    compile_inputs={str(f):sha256(f) for f in [Path(c['java_home'])/'bin/javac.exe',*sorted((PRODUCER/'lib').glob('*.jar'))]}
    if stamp.exists():
        state=load(stamp)
        if state['status']!='complete':return state
        if state['compile_inputs']!=compile_inputs:raise ValueError('Compiler/dependency build cache changed')
        for n,h in state['artifacts'].items():
            if sha256(work/n)!=h:raise ValueError('Prepared input cache corrupted')
        return state
    if build.exists():raise ValueError('Unsealed preparation; preserve and declare a new attempt')
    state={'identity':p['identity'],'status':'preparing','step':None}
    try:
        for part,folder in [('main','src'),('tests','test')]:
            state['step']='compile_'+part;target=build/part;target.mkdir(parents=True,exist_ok=True)
            args=[Path(c['java_home'])/'bin/javac.exe','-J-Xmx512m','-g','-encoding','UTF-8','-cp',
                  os.pathsep.join(map(str,[build/'main',PRODUCER/'lib/*'])),'-d',target,*sorted((variant/folder).rglob('*.java'))]
            budget.run(str(p['identity'])+state['step'],180,0,lambda:checked(lc,args,work,work/(part+'-build.log')))
        state['step']='source_export'
        budget.run(str(p['identity'])+state['step'],180,0,lambda:checked(lc,[Path(c['java_home'])/'bin/java.exe','-Xmx256m',
                   '-cp',cp,'Phase03SourceExport',work/'source_ast.jsonl',*sorted(p['source_sha256'])],work,work/'source-export.log'))
        write_json(work/'source_evidence.json',{'producer':'JDK8_JavacTask_parse_only','source_sha256':p['source_sha256'],
                                              'records':lines(work/'source_ast.jsonl')})
        state['step']='static_export';dest=work/p['identity'][2];dest.mkdir(exist_ok=True)
        classes=sorted(build.rglob('*.class'))
        budget.run(str(p['identity'])+state['step'],180,0,lambda:checked(lc,[Path(c['java_home'])/'bin/java.exe','-Xmx256m',
                   '-cp',cp,'Phase03StaticExport',dest/'static.tsv',*classes],work,work/'static-export.log'))
        write_json(work/'compilation_binding.json',{'source_sha256':p['source_sha256'],
            'class_sha256':{str(f.resolve()):sha256(f) for f in classes},'javac_sha256':sha256(Path(c['java_home'])/'bin/javac.exe')})
        state.update(status='complete',compile_inputs=compile_inputs,artifacts={f.relative_to(work).as_posix():sha256(f) for f in
            [*classes,work/'source_evidence.json',work/'compilation_binding.json',dest/'static.tsv']})
    except Exception as e:state.update(status='failed',error=type(e).__name__+': '+str(e))
    atomic_json(stamp,state);return state

def collect(c,out,stop_after=None):
    budget=CollectionBudget(c,out);cp=tools_prepare(c,out,budget);new_pairs=0;cached=0
    results=load(out/'collection_summary.json')['products'] if (out/'collection_summary.json').exists() else []
    by={tuple(r['identity']):r for r in results}
    for i,p in enumerate(c['products']):
        work=product_dir(out,i);work.mkdir(exist_ok=True);lc=local_config(c,work)
        prep=prepare(c,p,work,cp,budget)
        result=by.get(tuple(p['identity']),{'identity':p['identity'],'directory':str(work),'registered_tests':len(p['tests'])})
        result['preparation']=prep['status']
        if prep['status']=='complete':
            build=work/'b';names=sorted(f.relative_to(build/'main').as_posix()[:-6].replace('/','.') for f in (build/'main').rglob('*.class'))
            collpath=work/'collection.json';records=load(collpath)['records'] if collpath.exists() else []
            records={r['test_id']:r for r in records}
            for j,t in enumerate(p['tests']):
                run=work/'r'/f't{j:03d}';status=run/'status.json'
                if status.exists() and not load(status)['complete']:
                    saved=load(status);cached+=1  # retain a failed attempt, never silently retry
                elif t['test_id'] in records and not status.exists():raise ValueError('Completed checkpoint lost its seal')
                else:
                    hit=status.exists()
                    if stop_after is not None and new_pairs>=stop_after and not hit:
                        atomic_json(out/'collection_summary.json',{'products':list(by.values()),'stopped_at_checkpoint':True})
                        print('CHECKPOINT',new_pairs,cached,flush=True);return
                    fn=lambda:execute_pair(lc,work,PRODUCER,BYTECODE/'producer-classes',BYTECODE/'agent.jar',Path(p['source_variant_path']),
                                           t['test_id'],names,build,run,t['result'])
                    saved=fn() if hit else budget.run('|'.join(p['identity'])+'|'+t['test_id'],60,2,fn)
                    new_pairs+=int(not hit);cached+=int(hit)
                records[t['test_id']]={**dict(zip(KEY,p['identity'])),'test_id':t['test_id'],'variant':p['source_variant_path'],
                                      'directory':str(run),'status':saved}
                atomic_json(collpath,{'config':lc,'records':[records[n] for n in sorted(records)]})
                print(i,t['test_id'],'complete',saved['complete'],'parity',saved['parity'],'historical',saved['matches_historical_outcome'],flush=True)
            result.update(collected_tests=len(records),complete_pairs=sum(r['status']['complete'] for r in records.values()),
                          consistent_pairs=sum(r['status']['complete'] and r['status']['parity'] and r['status']['matches_historical_outcome'] for r in records.values()))
        else:result.update(error=prep.get('error'),collected_tests=0,complete_pairs=0,consistent_pairs=0)
        by[tuple(p['identity'])]=result;atomic_json(out/'collection_summary.json',{'products':list(by.values()),'stopped_at_checkpoint':False})
        print('PRODUCT',i,result,flush=True)
    history=out/'collection_history.jsonl'
    with history.open('a',encoding='utf-8') as f:f.write(json.dumps({'new_pairs':new_pairs,'cached_records':cached,'new_test_process_reservations':new_pairs*2})+'\n')
    print('COLLECTION FINISHED',new_pairs,cached,flush=True)

def extract_product(c,out,index,*,semantics_profile=None,framework_static=None):
    """Per-product streaming adapter around existing mapping/graph_for/validators."""
    from .static import SourceEvidence,load_static,mapping,program_identities
    from .trace import parse_trace,Dataflow
    from .extract import graph_for,validate_graph,identity_errors
    from .control import control_evidence_errors
    import jsonschema
    sys.path.insert(0,str(ROOT/'scripts'))
    from verify_phase03A_mapping import edge_checks
    p=c['products'][index];work=product_dir(out,index);dest=work/'g';dest.mkdir(exist_ok=True)
    if (dest/'quality_report.json').exists():raise ValueError('Already sealed extraction; do not overwrite')
    collection=load(work/'collection.json');records=collection['records'];source=SourceEvidence(work/'source_evidence.json')
    binding=load(work/'compilation_binding.json');static=load_static(work/p['identity'][2]/'static.tsv')
    # Confirm the actual classes/environment that produced the cached executions,
    # not merely a path with the same product name.
    actual_inputs={}
    for r in records:
        for name,digest in r['status']['inputs']['runtime_binding']['artifacts_sha256'].items():
            if name in actual_inputs and actual_inputs[name]!=digest:
                raise ValueError('Execution environment changed within one product')
            actual_inputs[name]=digest
    for name,digest in binding['class_sha256'].items():
        if actual_inputs.get(name)!=digest:raise ValueError('Static/dynamic compiled class binding differs')
    for name,digest in actual_inputs.items():
        if sha256(Path(name))!=digest:raise ValueError('Executed class/environment artifact changed')
    for name,digest in load(work/'prepare.json')['artifacts'].items():
        if sha256(work/name)!=digest:raise ValueError('Prepared analysis input changed')
    rows,mapped,_=mapping(Path(p['source_variant_path']),static,p['features'],source)
    program_identities(Path(p['source_variant_path']),static,mapped,source,binding)
    (dest/'source_map.jsonl').write_text(''.join(json.dumps({**dict(zip(KEY,p['identity'])),**r},ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    schemas={n:jsonschema.Draft202012Validator(load(Path(__file__).parent/(n+'.schema.json'))) for n in ('graphs','events')}
    reports=[];graph_count=0
    with (dest/'graphs.jsonl').open('x',encoding='utf-8') as gf,(dest/'events.jsonl').open('x',encoding='utf-8') as ef, (dest/'test_quality.jsonl').open('x',encoding='utf-8') as qf:
        for r in records:
            summary={'test_id':r['test_id'],'graph_generated':False};lineage={k:r[k] for k in (*KEY,'test_id')}
            try:
                # Even nonqualified pairs retain actual output/trace evidence; no
                # invented completion or synthetic events on parsing failures.
                status=r['status'];run=Path(r['directory'])
                for n,h in status['artifacts_sha256'].items():
                    if sha256(run/n)!=h:raise ValueError('Execution artifact changed')
                raw=parse_trace(run/'instrumented/run.log',r['test_id'],c['max_events'],c['max_output_bytes'])
                framework_evidence=None;calls={}
                if framework_static is not None:
                    from .effects import make_evidence,project_and_verify
                    framework_evidence=make_evidence(run/'instrumented/run.log',r['test_id'],status,framework_static)
                    raw,calls,_=project_and_verify(framework_evidence)
                ids=program_identities(Path(p['source_variant_path']),static,mapped,source,binding,[(*e['domain'],e['bci']) for e in raw])
                graph,events=graph_for(raw,static,mapped,p['features'],lineage,status,ids,
                    semantics_profile=semantics_profile,framework_evidence=framework_evidence)
                graph['quality']['suite_completeness']={'complete':len(records)==len(p['tests']),'expected_tests':len(p['tests']),
                    'collected_tests':len(records),'basis':'registered_original_suite_not_absolute_historical_completeness'}
                graph['provenance'].update(trace_sha256=sha256(run/'instrumented/run.log'),status_sha256=sha256(run/'status.json'),
                    static_sha256=sha256(work/p['identity'][2]/'static.tsv'),roles_sha256=p['roles_sha256'],
                    agent_sha256=status['inputs']['agent_sha256'],source_evidence_sha256=source.digest)
                validate_graph(graph,events);schemas['graphs'].validate(graph)
                for e in events:schemas['events'].validate(e)
                checks=None
                if graph['quality']['eligible_graph_structure']:
                    checks=edge_checks(graph,events,static,Dataflow(static,semantics_profile=semantics_profile,framework_calls=calls).run(raw))
                    checks['declared_profile_supported_semantics' if semantics_profile else 'normal_path_supported_semantics']=checks.pop('whole_graph_semantics_certified')
                vals=[n for n in graph['nodes'] if n['type']=='value'];assertions=[n for n in graph['nodes'] if n['type']=='assertion']
                summary.update(graph_generated=True,quality=graph['quality'],identity_errors=identity_errors(graph),
                    control_errors=control_evidence_errors(graph),edge_checks=checks,nodes=len(graph['nodes']),edges=len(graph['edges']),
                    values=len(vals),observed_values=sum(n['attrs'].get('observed',False) for n in vals),
                    assertions=len(assertions),complete_assertion_values=sum(n['attrs'].get('expected_actual_values_complete',False) for n in assertions),
                    branch_events=sum(153<=e['opcode']<=166 or e['opcode'] in (198,199) for e in events),
                    known_branch_results=sum(e['branch_taken'] is not None for e in events))
                gf.write(json.dumps(graph,ensure_ascii=False)+'\n');ef.writelines(json.dumps(e,ensure_ascii=False)+'\n' for e in events)
                graph_count+=1
            except Exception as e:summary.update(error=type(e).__name__+': '+str(e))
            reports.append(summary);qf.write(json.dumps(summary,ensure_ascii=False)+'\n');qf.flush();gf.flush();ef.flush()
            print(index,r['test_id'],'GRAPH' if summary['graph_generated'] else summary['error'],flush=True)
    write_json(dest/'quality_report.json',{'identity':p['identity'],'tests':reports,'graphs':graph_count,
        'registered_tests':len(p['tests']),'collected_records':len(records),'batch_version':VERSION,
        'semantics_profile':semantics_profile,'framework_static_sha256':sha256(framework_static) if framework_static else None,
        'source_sha256':{str(f.relative_to(ROOT)):sha256(f) for f in Path(__file__).parent.glob('*.py')},
        'inputs_sha256':{n:sha256(work/n) for n in ('collection.json','prepare.json','source_evidence.json','compilation_binding.json')},
        'artifact_sha256':{n:sha256(dest/n) for n in ('graphs.jsonl','events.jsonl','source_map.jsonl')},'labels_read':False,'training_runs':0})

def extract_batch(c,out):
    budget=CollectionBudget(c,out);records=[]
    for i,p in enumerate(c['products']):
        work=product_dir(out,i);dest=work/'g'
        if not (work/'collection.json').exists():records.append({'index':i,'status':'no_execution_collection'});continue
        if (work/'extraction_attempt.json').exists() and load(work/'extraction_attempt.json')['status']!='complete':
            # A worker may seal its files just before the exit-time resource check
            # rejects the attempt. Its seal must not erase that rejection on resume.
            records.append(load(work/'extraction_attempt.json'));continue
        if (dest/'quality_report.json').exists():
            q=load(dest/'quality_report.json')
            for n,h in q['artifact_sha256'].items():
                if sha256(dest/n)!=h:raise ValueError('Graph cache changed')
            for n,h in q['inputs_sha256'].items():
                if sha256(work/n)!=h:raise ValueError('Graph cache input changed')
            for n,h in q['source_sha256'].items():
                if sha256(ROOT/n)!=h:raise ValueError('Graph cache implementation changed')
            records.append({'index':i,'status':'complete','cached':True});continue
        if (work/'extraction_attempt.json').exists():
            records.append(load(work/'extraction_attempt.json'));continue
        dest.mkdir(exist_ok=True)
        try:
            # Raw execution directories retain their 256 MiB per-product budget.
            # Normalized graphs have a separate 256 MiB bound under the frozen
            # global 4 GiB budget; formal phase04's 128 MiB input gate is unchanged.
            rec=budget.run('extract '+str(p['identity']),180,0,lambda:command(
                [sys.executable,'-X','utf8','-m','cwm_fp.lddg.batch','--stage','extract-product','--index',str(i)],dest,dest/'extract.log',
                seconds=180,max_bytes=268435456,monitor_files=True,max_files=20,budget=c['resource_policy'],max_rss_bytes=2147483648))
            record={'index':i,'status':'complete' if rec['exit_code']==0 and not rec['reason'] and (dest/'quality_report.json').exists() else 'failed','execution':rec}
        except Exception as e:record={'index':i,'status':'failed','error':type(e).__name__+': '+str(e)}
        write_json(work/'extraction_attempt.json',record);records.append(record)
        atomic_json(out/'extraction_summary.json',{'products':records});print('EXTRACTION',i,record['status'],flush=True)
    atomic_json(out/'extraction_summary.json',{'products':records})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['collect','extract','extract-product'],required=True)
    parser.add_argument('--index',type=int);parser.add_argument('--stop-after-new-tests',type=int);a=parser.parse_args()
    os.environ['PYTHONPATH']=str(ROOT/'src')
    from cwm_fp.baselines.boundaries import install_io_guard
    install_io_guard('features');c,out=read_plan()
    if a.stage=='collect':collect(c,out,a.stop_after_new_tests)
    elif a.stage=='extract':extract_batch(c,out)
    else:extract_product(c,out,a.index)

if __name__=='__main__':main()
