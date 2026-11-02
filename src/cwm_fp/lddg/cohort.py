"""04D1 orchestration only: frozen selection, existing paired runner and extractor.

No new JVM semantics, target-guided selection, model fitting or replacement.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict,deque
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from cwm_fp.data.common import ROOT,KEY,key,read_csv,sha256,write_json
from cwm_fp.data.splits import validate_split
from . import batch
from .suite import load,lines,atomic_json
from .runner import execute_pair,command,tree_usage,budget_check

OUT=Path('D:/dataset/whn/cwm_fp_generated/data/processed/lddg/phase04D1_bank_v1')
RAW=Path('D:/dataset/whn/4wise-BankAccountTP-1BUG-Full')
OLD=ROOT/'data/processed/graph_classifier/phase04P_v1'
FRAMEWORK=ROOT/'data/processed/lddg/phase03B4_framework_v1/framework/static.tsv'
SEED=20261101
PROTOCOL='grouped_bank_pilot_v1'
QUOTAS={'train':8,'val':2,'test':2}
PYTORCH='E:/Myself/投稿/work7/FP-SPL/.venv/Scripts/python.exe'

def rank(kind,*identity):return hashlib.sha256('|'.join(['04D1',str(SEED),kind,*identity]).encode()).hexdigest()
def split_path(f='grouped'):return ROOT/'data/processed/splits'/('splits.csv' if f=='grouped' else f+'.csv')
def input_rows():
    ps={key(p):p for p in read_csv(ROOT/'data/processed/manifests/products.csv',KEY)}
    ss={key(p):p for p in read_csv(split_path(),(*KEY,'split'))};validate_split(ps,ss)
    ts=defaultdict(list)
    for t in read_csv(ROOT/'data/processed/manifests/tests.csv',(*KEY,'test_id')):ts[key(t)].append(t)
    if any(p['label'] or p['label_provenance'] or p['origin']!='unknown' for p in ps.values()):raise ValueError('Public manifest contains targets')
    return ps,ss,ts

def select(candidates):
    """Only identity/partition/PASS/collection state and count are accepted inputs."""
    fields={'identity','partition','observed_result','previously_collected','registered_tests'}
    if any(set(p)!=fields for p in candidates):raise ValueError('Selection accepts only declared label-blind fields')
    by=defaultdict(list);collected_versions={tuple(p['identity'][:2]) for p in candidates if p['previously_collected']}
    for p in candidates:
        if p['observed_result']=='PASS' and not p['previously_collected']:by[tuple(p['identity'][:2])].append(p)
    selected={}
    for part,n in QUOTAS.items():
        versions=[v for v,ps in by.items() if ps[0]['partition']==part]
        # Unseen versions first, never semantic quality/test class/target labels.
        versions.sort(key=lambda v:(v in collected_versions,rank('version',part,*v),v))
        selected[part]=[]
        chosen=versions[:n]
        ordered={v:sorted(by[v],key=lambda p:(rank('product',*p['identity']),p['identity']))[:3] for v in chosen}
        for slot in range(3):
            selected[part].extend(ordered[v][slot] for v in chosen if len(ordered[v])>slot)
    queues={p:deque(v) for p,v in selected.items()};result=[]
    while any(queues.values()):
        for part in ('train','val','test'):
            if queues[part]:result.append(queues[part].popleft())
    return result

def prepare_plan():
    if (OUT/'cohort_plan.json').exists():raise ValueError('Plan already frozen; do not replace')
    ps,ss,ts=input_rows();seen=set();inventory={}
    prior_batch=load(ROOT/'configs/phase03B2_collection.json');prior_root=Path(prior_batch['out_dir']).resolve()
    for base in (ROOT/'data/processed/lddg',OUT.parent):
        for path in sorted(base.rglob('collection.json')):
            if path.is_relative_to(OUT) or any(x in path.parts for x in ('producer_before','producer_snapshot')):continue
            if path.resolve().is_relative_to(prior_root):
                part=path.resolve().relative_to(prior_root).parts[0]
                if part.startswith('p') and part[1:].isdigit() and prior_batch['products'][int(part[1:])]['identity'][0]!='BankAccountTP':continue
            doc=load(path);records=doc.get('records',[]) if isinstance(doc,dict) else doc
            relevant=[]
            for r in records:
                if all(k in r for k in KEY):
                    k=key(r)
                    if k in ps and k[0]=='BankAccountTP':seen.add(k);relevant.append(list(k))
            if relevant:inventory[str(path)]={'sha256':sha256(path),'identities':sorted(set(map(tuple,relevant)))}
    candidates=[]
    for k,p in sorted(ps.items()):
        if k[0]!='BankAccountTP' or p['observed_result']!='PASS':continue
        candidates.append({'identity':list(k),'partition':ss[k]['split'],'observed_result':'PASS','registered_tests':len(ts[k]),'previously_collected':k in seen})
    choices=select(candidates)
    if len(choices)>36 or len({tuple(p['identity'][:2]) for p in choices})>12:raise ValueError('Selection budget')
    old=load(OLD/'package/manifest.json');oldreceipt=load(OLD/'export_receipt.json')
    assert sha256(OLD/'package/manifest.json')==oldreceipt['manifest_sha256']
    products=[]
    folds={f:{key(p):p['split'] for p in read_csv(split_path(f),(*KEY,'split'))} for f in old['split_protocols']}
    for i,row in enumerate(choices):
        k=tuple(row['identity']);p=ps[k];variant=Path(p['source_variant_path']).resolve()
        if not variant.is_relative_to(RAW.resolve()):raise ValueError('Unexpected raw input')
        tests=sorted([{'test_id':t['test_id'],'result':t['result']} for t in ts[k]],key=lambda t:t['test_id'])
        if not tests or any(t['result']!='PASS' for t in tests) or len({t['test_id'] for t in tests})!=len(tests):raise ValueError('Original PASS suite mismatch')
        features=json.loads(p['features_json'])
        source=list((variant/'src').rglob('*.java'))+list((variant/'test').rglob('*.java'))
        for f,on in features.items():
            if on:source+=list((variant.parent.parent/'features'/f).rglob('*.java'))
        products.append({**row,'index':i,'directory':f'p{i:03d}','suite_id':p['suite_id'],'source_variant_path':str(variant),
            'features':features,'tests':tests,'roles_sha256':sha256(variant/'roles.meta'),
            'source_sha256':{str(f):sha256(f) for f in sorted(set(source))},'partitions':{f:s[k] for f,s in folds.items()}})
    c=load(ROOT/'configs/phase03B2_collection.json')
    c={k:c[k] for k in ('java_home','java_heap_mb','max_events','max_output_bytes','max_seconds_per_test','max_files_per_run','max_rss_bytes','strict_runtime_binding')}
    c.update(collection_time_budget_seconds=5400,new_java_test_process_limit=2000,new_pair_limit=1000,
             resource_policy={'root':str(OUT),'max_total_bytes':3*1024**3,'max_total_files':30000,'min_free_bytes':4*1024**3})
    frozen=[ROOT/'data/processed/manifests/products.csv',ROOT/'data/processed/manifests/tests.csv',*[split_path(f) for f in folds],
        ROOT/'outputs/phase04P_artifacts.json',ROOT/'outputs/phase03B4_artifacts.json',OLD/'package/manifest.json',OLD/'export_receipt.json',FRAMEWORK,
        ROOT/'data/processed/lddg/phase03B4_framework_v1/framework/binding.json']
    frozen+=[f for f in (ROOT/'src/cwm_fp/lddg').glob('*.py') if f.name not in ('cohort.py','batch.py')]+list((ROOT/'src/cwm_fp/models').glob('*.py'))
    plan={'stage_id':'04D1','primary_protocol':PROTOCOL,'seed':SEED,'created_utc':datetime.now(timezone.utc).isoformat(),
        'selection_rule':'SHA256(04D1|20261101|version|partition|system|version), unseen versions first; products SHA256(04D1|20261101|product|system|version|product). Three slots across ranked versions; partitions cycle train,val,test.',
        'selection_fields':['identity','partition','observed_result','previously_collected','registered_tests'],
        'version_quotas':QUOTAS,'candidate_pool':candidates,'collection_inventory':inventory,'products':products,
        'reused_products':[{'identity':p['identity'],'partitions':p['partitions'],'registered_tests':p['registered_tests']} for p in old['products']],
        'config':c,'new_registered_tests':sum(len(p['tests']) for p in products),'frozen_input_sha256':{str(p):sha256(p) for p in frozen},
        'preflight_disks':{str(p):shutil.disk_usage(p)._asdict() for p in (Path('D:/'),Path('E:/'))},
        'resource_policy_note':'Retain stricter previous4GiB free-disk floor; 3GiB new-directory ceiling; sequential, no retries by default. Reserve complete-product worst-case time/processes and512MiB before start.',
        'orchestration_at_freeze_sha256':{str(f):sha256(f) for f in [Path(__file__),Path(batch.__file__)]},
        'label_files_opened':[],'quality_filter_used':False,'replacement_allowed':False,'server_future_workdir':'X:/codes'}
    assert plan['new_registered_tests']<=1000
    atomic_json(OUT/'cohort_plan.json',plan);(OUT/'cohort_plan.sha256').write_text(sha256(OUT/'cohort_plan.json'),encoding='ascii')
    print('FROZEN',sha256(OUT/'cohort_plan.json'),'products',len(products),'tests',plan['new_registered_tests'],flush=True)
    for p in products:print(p['directory'],p['partition'],*p['identity'],len(p['tests']),flush=True)

def read_plan():
    p=load(OUT/'cohort_plan.json')
    if sha256(OUT/'cohort_plan.json')!=(OUT/'cohort_plan.sha256').read_text().strip():raise ValueError('Frozen selection changed')
    for name,h in p['frozen_input_sha256'].items():
        if sha256(Path(name))!=h:raise ValueError('Frozen input/analysis implementation changed: '+name)
    if [r['identity'] for r in select(p['candidate_pool'])]!=[r['identity'] for r in p['products']]:raise ValueError('Selection order changed')
    return p

def product_reservation(p):return 900+60*len(p['tests']),2*len(p['tests'])

def run_collection():
    plan=read_plan();c=plan['config'];budget=batch.CollectionBudget(c,OUT)
    cp=batch.tools_prepare(c,OUT,budget);start=time.monotonic();prior=budget.state['seconds'];stop=None
    for i,p in enumerate(plan['products']):
        work=batch.product_dir(OUT,i);work.mkdir(exist_ok=True);resultpath=work/'product_status.json'
        if resultpath.exists():print('REUSE_PRODUCT_CHECKPOINT',i,flush=True);continue
        reserve,processes=product_reservation(p)
        remaining=5400-max(budget.state['seconds'],prior+time.monotonic()-start)
        why,_=budget_check(c['resource_policy'],512*1024**2)
        if why or remaining<reserve or budget.state['reserved_test_processes']+processes>2000:
            stop=why or ('product_time_reservation' if remaining<reserve else 'product_process_reservation');break
        lc=batch.local_config(c,work)
        result={'identity':p['identity'],'index':i,'started':True,'extraction_complete':False,'resource_stop':None}
        prep=batch.prepare(c,p,work,cp,budget);result['preparation']=prep['status']
        if prep['status']=='complete':
            build=work/'b';names=sorted(f.relative_to(build/'main').as_posix()[:-6].replace('/','.') for f in (build/'main').rglob('*.class'))
            names+=['org.evosuite.runtime.EvoAssertions']
            records=load(work/'collection.json')['records'] if (work/'collection.json').exists() else []
            records={r['test_id']:r for r in records}
            for j,t in enumerate(p['tests']):
                run=work/'r'/f't{j:03d}';stamp=run/'status.json';cached=stamp.exists()
                if cached and not load(stamp)['complete']:status=load(stamp)
                else:
                    fn=lambda:execute_pair(lc,work,batch.PRODUCER,batch.BYTECODE/'producer-classes',batch.BYTECODE/'agent.jar',
                        Path(p['source_variant_path']),t['test_id'],names,build,run,t['result'])
                    status=fn() if cached else budget.run(p['directory']+'|'+t['test_id'],60,2,fn)
                records[t['test_id']]={**dict(zip(KEY,p['identity'])),'test_id':t['test_id'],'variant':p['source_variant_path'],
                    'directory':str(run),'status':status,'cached':cached}
                atomic_json(work/'collection.json',{'config':lc,'records':[records[k] for k in sorted(records)]})
                print('PAIR',i,j,t['test_id'],status['complete'],status['parity'],flush=True)
            result.update(collected_tests=len(records),complete_pairs=sum(r['status']['complete'] for r in records.values()),
                parity_pairs=sum(r['status']['complete'] and r['status']['parity'] and r['status']['matches_historical_outcome'] for r in records.values()))
            dest=work/'g';dest.mkdir(exist_ok=True)
            rec=budget.run('extract '+p['directory'],180,0,lambda:command([sys.executable,'-X','utf8','-m','cwm_fp.lddg.cohort','extract','--index',str(i)],
                dest,dest/'extract.log',seconds=180,max_bytes=256*1024**2,monitor_files=True,max_files=20,budget=c['resource_policy'],max_rss_bytes=2*1024**3))
            result['extraction_execution']=rec;result['extraction_complete']=rec['exit_code']==0 and not rec['reason'] and (dest/'quality_report.json').exists()
        else:result['error']=prep.get('error')
        atomic_json(resultpath,result);print('PRODUCT_DONE',i,result,flush=True)
    atomic_json(OUT/'collection_summary.json',{'products':[load(x) for x in sorted(OUT.glob('p*/product_status.json'))],
        'stop_reason':stop,'session_seconds':time.monotonic()-start,'budget':budget.state,'frozen_plan_sha256':sha256(OUT/'cohort_plan.json')})

def extract_one(index):
    from .trace import FRAMEWORK_PROFILE
    plan=read_plan();batch.extract_product({**plan['config'],'products':plan['products']},OUT,index,
        semantics_profile=FRAMEWORK_PROFILE,framework_static=FRAMEWORK)

def admission():
    from cwm_fp.models.graph_data import load_corpus
    plan=read_plan();records=[]
    for p in plan['products']:
        work=batch.product_dir(OUT,p['index']);row={'identity':p['identity'],'index':p['index'],'reused':False,'graph_path':str(work/'g/graphs.jsonl'),'event_path':str(work/'g/events.jsonl'),'eligible':False}
        state=load(work/'product_status.json') if (work/'product_status.json').exists() else {}
        if not state.get('extraction_complete'):row['reasons']=['unsealed_or_failed_extraction']
        else:
            try:
                *_,eligible,audit=load_corpus(ROOT/'data/processed/manifests/products.csv',ROOT/'data/processed/manifests/tests.csv',Path(row['graph_path']),split_path())
                rec=next((x for x in audit['products'] if x['identity']==p['identity']),None)
                row.update(eligible=tuple(p['identity']) in eligible,reasons=rec['reasons'] if rec else ['missing_graph_product'],audit=audit)
            except Exception as e:row.update(reasons=['consumer_error'],error=str(e))
        records.append(row);print('ADMITTED',p['index'],row['eligible'],row.get('reasons'),flush=True)
    for p in load(OLD/'plan.json')['products']:
        path=ROOT/'data/processed/lddg/phase03B4_framework_v1'/p['name']/'graphs.jsonl'
        *_,eligible,audit=load_corpus(ROOT/'data/processed/manifests/products.csv',ROOT/'data/processed/manifests/tests.csv',path,split_path())
        assert tuple(p['identity']) in eligible
        records.append({'identity':p['identity'],'reused':True,'graph_path':str(path),'event_path':str(path.with_name('events.jsonl')),'eligible':True,'reasons':[],'audit':audit})
    write_json(OUT/'admission.json',{'primary_protocol':PROTOCOL,'products':records,'plan_sha256':sha256(OUT/'cohort_plan.json'),'labels_read':False})

def package():
    import torch
    from cwm_fp.models.graph_encoder import encode_test_lossless
    from cwm_fp.models.graph_package import ProductPackage,write_test,write,portable,read_metadata
    plan=read_plan();a=load(OUT/'admission.json');original=ProductPackage(OLD/'package',expected_manifest_sha256=load(OLD/'export_receipt.json')['manifest_sha256'])
    dest=OUT/'package';dest.mkdir(exist_ok=False);(dest/'graphs').mkdir();locators={};products=[];sources={};count=0;failures=[]
    ps,ss,ts=input_rows()
    for row in a['products']:
        if not row['eligible']:continue
        k=tuple(row['identity']);gs=sorted(lines(row['graph_path']),key=lambda g:g['test_id']);ev=lines(row['event_path'])
        p={'identity':list(k),'sample_id':ps[k]['sample_id'],'suite_id':ps[k]['suite_id'],'observed_result':'PASS',
            'registered_tests':[g['test_id'] for g in gs],'partitions':{f:next(r for r in read_csv(split_path(f),KEY) if key(r)==k)['split'] for f in original.manifest['split_protocols']},
            'tests':[],'registration_proof':original.manifest['source_provenance']['frozen_manifests_sha256']}
        try:
            for g in gs:
                events=[e for e in ev if e['test_id']==g['test_id']]
                ref=write_test(dest,count,g,events,{'graph_file_sha256':sha256(Path(row['graph_path'])),'event_file_sha256':sha256(Path(row['event_path'])),
                    'frozen_plan_sha256':sha256(OUT/'cohort_plan.json')},locators)
                count+=1;p['tests'].append(ref)
        except Exception as e:
            failures.append({'identity':list(k),'reason':str(e),'status':'capacity_or_encoding_rejection; partial shards retained, product excluded'});continue
        # Keep locators, not every decoded graph of the entire cohort in RAM.
        products.append(p);sources[k]=row
    manifest={**original.manifest,'products':products,'primary_protocol':PROTOCOL,
        'source_provenance':{'frozen_manifests_sha256':original.manifest['source_provenance']['frozen_manifests_sha256'],
            'frozen_plan_sha256':sha256(OUT/'cohort_plan.json'),'producer_sha256':{str(f.relative_to(ROOT)):sha256(f) for base in ['lddg','models'] for f in (ROOT/'src/cwm_fp'/base).glob('*.py')}}}
    write(dest/'manifest.json',manifest);anchor=sha256(dest/'manifest.json');pack=ProductPackage(dest,expected_manifest_sha256=anchor);checks=[]
    old_by={tuple(p['identity']):p for p in original.products}
    for p in pack.products:
        row=sources[tuple(p['identity'])]
        gs=sorted(lines(row['graph_path']),key=lambda g:g['test_id']);ev=lines(row['event_path'])
        oldtests=list(original.iter_tests(old_by[tuple(p['identity'])])) if row['reused'] else None
        for i,(g,t,ref) in enumerate(zip(gs,pack.iter_tests(p),p['tests'])):
            expected=encode_test_lossless(g);meta=read_metadata(dest/ref['metadata']['path'],16*1024**2)
            for field in ('x','edge_index','edge_type','coverage'):
                assert torch.equal(getattr(expected,field),getattr(t,field))
                if oldtests:assert torch.equal(getattr(oldtests[i],field),getattr(t,field))
            assert meta['tensor_metadata']==portable(expected.metadata,{})
            assert meta['events']==portable([e for e in ev if e['test_id']==g['test_id']],{})
            assert meta['quality']==g['quality']
            assert meta['instruction_catalog']==portable(g.get('instruction_catalog',[]),{})
            assert meta['framework_effects']==portable(g.get('framework_effects',[]),{})
            if oldtests:
                oldref=old_by[tuple(p['identity'])]['tests'][i]
                oldmeta=read_metadata(OLD/'package'/oldref['metadata']['path'],16*1024**2)
                for field in ('tensor_metadata','events','quality','instruction_catalog','framework_effects'):
                    assert oldmeta[field]==meta[field],field
            checks.append({'identity':ref['identity'],'nodes':ref['nodes'],'edges':ref['edges'],'events':ref['events'],'relations':ref['relations'],
                'source_equal':True,'legacy_tensor_equal':True if oldtests else None,'reused':row['reused'],'metadata_path':ref['metadata']['path']})
    write_json(OUT/'package_validation.json',{'products':len(products),'graphs':checks,'old_products':len(old_by),'old_graphs_reverified':sum(r['reused'] for r in checks),
        'failures':failures,'manifest_sha256':anchor,'zero_loss':True,'models_fitted':0,'backward_executed':False,'labels_read':False})
    write_json(OUT/'export_receipt.json',{'manifest_sha256':anchor,'source_paths':{json.dumps(k):r for k,r in sources.items()},'locator_audit':locators})
    files=[OUT/'cohort_plan.json',OUT/'admission.json',OUT/'package_validation.json',dest/'manifest.json',*dest.glob('graphs/*')]
    write_json(OUT/'cohort_seal.json',{'stage':'04D1','products':[p['identity'] for p in products],'files_sha256':{str(f):sha256(f) for f in files},'labels_opened_before_seal':False})
    print('PACKAGED',len(products),len(checks),'old graphs',sum(r['reused'] for r in checks),flush=True)

def labels():
    from cwm_fp.data.labels import load_targets
    seal=load(OUT/'cohort_seal.json')
    for n,h in seal['files_sha256'].items():
        if sha256(Path(n))!=h:raise ValueError('Post-admission cohort changed before labels')
    ps,ss,_=input_rows();keys=[tuple(k) for k in seal['products'] if ss[tuple(k)]['split']=='train']
    allowed=ROOT/'data/processed/labels/full/grouped/train.csv'
    targets=load_targets(allowed,ROOT/'data/processed/manifests/products.csv',split_path(),partition='train',purpose='train')
    counts=Counter(targets.get(k,'MISSING') for k in keys)
    versions={label:len({k[:2] for k in keys if targets.get(k)==label}) for label in ('FP','TP')}
    write_json(OUT/'readiness/grouped_train.json',{'protocol':PROTOCOL,'counts':dict(counts),'class_version_counts':versions,
        'single_version_classes':[k for k,n in versions.items() if n==1],'train_products':len(keys),'train_versions':len({k[:2] for k in keys}),
        'cohort_seal_sha256':sha256(OUT/'cohort_seal.json'),'label_file_opened':str(allowed),'label_file_sha256':sha256(allowed),
        'validation_test_or_other_protocol_labels_opened':[],'qualification':'existing_local_PULP_fault_location_proxy; not independently verified truth',
        'plan_replacement_after_labels':False})
    print('AUTHORIZED_GROUPED_TRAIN',dict(counts),versions,flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','collect','extract','admission','package','labels']);parser.add_argument('--index',type=int);a=parser.parse_args()
    os.environ['PYTHONPATH']=str(ROOT/'src')
    from cwm_fp.baselines.boundaries import install_io_guard
    allowed=[ROOT/'data/processed/labels/full/grouped/train.csv'] if a.action=='labels' else []
    install_io_guard('readiness_train_only' if allowed else 'features',allowed)
    start=time.monotonic();code=1
    try:
        {'plan':prepare_plan,'collect':run_collection,'extract':lambda:extract_one(a.index),'admission':admission,'package':package,'labels':labels}[a.action]();code=0
    finally:
        if a.action in ('admission','package'):
            # Observe the actual tensor worker, not the Windows venv launcher.
            try:import psutil
            except ModuleNotFoundError:
                sys.path.append('D:/SoftInstall/Lib/site-packages');import psutil
            mem=psutil.Process().memory_info()
            write_json(OUT/(a.action+'_self_resource.json'),{'pid':os.getpid(),'scope':'actual Python tensor worker; OS process-lifetime peak working set',
                'OS_peak_working_set_bytes':getattr(mem,'peak_wset',None),'final_RSS_bytes':mem.rss,
                'seconds':time.monotonic()-start,'exit_code':code,'argv':[sys.executable,*sys.argv]})
        with (OUT/'actions.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'argv':[sys.executable,*sys.argv],'seconds':time.monotonic()-start,'exit_code':code,'utc':datetime.now(timezone.utc).isoformat()})+'\n')

if __name__=='__main__':main()
