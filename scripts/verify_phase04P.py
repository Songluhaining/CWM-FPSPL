"""Five admitted products only; bounded package export and untrained CPU forward."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
import time
import threading

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from cwm_fp.data.common import KEY,sha256,write_json
from cwm_fp.lddg.suite import load,lines
from cwm_fp.models.graph_data import load_corpus
from cwm_fp.models.graph_encoder import FEATURES,FEATURE_SCHEMA,RELATIONS,collate,encode_test_lossless
from cwm_fp.models.graph_package import PACKAGE_SCHEMA,ProductPackage,write_test,write,digest,forward_stream
from cwm_fp.models.graph_classifier import GraphClassifier

OUT=ROOT/'data/processed/graph_classifier/phase04P_v1'
OLD=ROOT/'data/processed/lddg/phase03B4_framework_v1'
FOLDS=('grouped','loso_BankAccountTP','loso_Email')
def split_path(f):return ROOT/'data/processed/splits'/('splits.csv' if f=='grouped' else f+'.csv')
def seal():
    plan=load(OUT/'plan.json')
    for p,h in plan['source_sha256'].items():assert sha256(Path(p))==h,p
    return plan

def build():
    plan=seal();dest=OUT/'package';dest.mkdir(exist_ok=False);locators={};records=[];number=0;checks=[]
    manifests={'products':ROOT/'data/processed/manifests/products.csv','tests':ROOT/'data/processed/manifests/tests.csv'}
    provenance={'frozen_manifests_sha256':{k:sha256(p) for k,p in manifests.items()},
                'split_sha256':{f:sha256(split_path(f)) for f in FOLDS},
                'verifier_sha256':{p.relative_to(ROOT).as_posix():sha256(p) for base in ['src/cwm_fp/lddg','src/cwm_fp/models'] for p in (ROOT/base).glob('*.py')}}
    for selected in plan['products']:
        work=OLD/selected['name'];path=work/'graphs.jsonl';k=tuple(selected['identity']);parts={}
        for fold in FOLDS:
            ps,ss,gs,eligible,audit=load_corpus(manifests['products'],manifests['tests'],path,split_path(fold),holdout=fold[5:] if fold.startswith('loso_') else None)
            assert set(eligible)=={k};parts[fold]=ss[k]['split']
        graphs=eligible[k];events=lines(work/'events.jsonl')
        row={'identity':list(k),'sample_id':ps[k]['sample_id'],'suite_id':ps[k]['suite_id'],'observed_result':'PASS','partitions':parts,
             'registered_tests':[g['test_id'] for g in graphs],'tests':[],'registration_proof':provenance['frozen_manifests_sha256']}
        assert len(graphs)==selected['tests']
        for g in graphs:
            ev=[e for e in events if e['test_id']==g['test_id']]
            ref=write_test(dest,number,g,ev,{'graph_file_sha256':sha256(path),'event_file_sha256':sha256(work/'events.jsonl'),**provenance},locators)
            row['tests'].append(ref);number+=1
            checks.append({**dict(zip(KEY,k)),'test_id':g['test_id'],'original_nodes':len(g['nodes']),'original_edges':len(g['edges']),
                           'original_events':len(ev),'package_reference':ref,'original_graph_path':str(path),'source_record_sha256':digest(g),
                           'identity_and_semantics_verified_at_export':True})
        records.append(row);print('EXPORTED',selected['name'],len(row['tests']),flush=True)
    manifest={'schema':PACKAGE_SCHEMA,'feature_schema':FEATURE_SCHEMA,'features':list(FEATURES),'relations':list(RELATIONS),'relation_policy':'full',
              'split_protocols':list(FOLDS),'products':records,'source_provenance':provenance,
              'evidence_boundary':'Original source/JVM semantics verified on collector. Import verifies anchored content/structure, not Java semantics.',
              'contains_supervised_targets':False,'model_input_kind':'deterministic_encoding_of_verified_observations; no predicted values'}
    write(dest/'manifest.json',manifest)
    write_json(OUT/'export_receipt.json',{'manifest_sha256':sha256(dest/'manifest.json'),'graphs':checks,'external_locator_audit_only':locators,
        'plan_sha256':sha256(OUT/'plan.json'),'source_validation':'all five products admitted through existing consumer for all three frozen protocols'})
    print('PACKAGE',number,'graphs',sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()),flush=True)

def verify(root=None,name='verification'):
    seal();root=Path(root) if root else OUT/'package';receipt=load(OUT/'export_receipt.json')
    package=ProductPackage(root,expected_manifest_sha256=receipt['manifest_sha256'])
    torch.manual_seed(20261101);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    models={kind:GraphClassifier(kind,hidden=16,layers=2).eval() for kind in ('coverage','agnostic','rgcn')}
    params={k:digest({n:v.tolist() for n,v in m.state_dict().items()}) for k,m in models.items()}
    results=[];graph_rows=[]
    start=time.monotonic()
    for p in package.products:
        product_start=time.monotonic();full=package.materialize(p,'grouped');batch=collate([full])
        assert not (batch.node_test[batch.edge_index[0]]!=batch.node_test[batch.edge_index[1]]).any()
        scores={}
        for kind,model in models.items():
            with torch.inference_mode():reference=model(batch)
            for micro in (1,3):
                t=time.monotonic();actual=forward_stream(model,package,p,micro_tests=micro)
                error=float((reference-actual).abs().max());torch.testing.assert_close(actual,reference,atol=1e-6,rtol=1e-5)
                scores[kind+'_'+str(micro)]={'max_abs_error':error,'seconds':time.monotonic()-t,'atol':1e-6,'rtol':1e-5}
        # Resume is a suffix; reconstruct complete sequence and compare exact tensors.
        half=len(full.tests)//2
        resumed=list(package.iter_tests(p))[:half]+list(package.iter_tests(p,start=half))
        for i,(a,b) in enumerate(zip(full.tests,resumed)):
            assert torch.equal(a.x,b.x) and torch.equal(a.edge_index,b.edge_index) and torch.equal(a.edge_type,b.edge_type)
            g=p['tests'][i];graph_rows.append({'identity':g['identity'],'loaded_nodes':len(a.x),'loaded_edges':a.edge_index.shape[1],
                'loaded_events':len(a.metadata['events']),'original_nodes':g['nodes'],'original_edges':g['edges'],'original_events':g['events'],
                'relations':g['relations'],'feature_columns':a.x.shape[1],'exact_tensor_resume':True,'qualified':True,
                'losses':{'nodes':0,'edges':0,'events':0},'evidence':g['metadata']})
        results.append({'identity':p['identity'],'partitions':p['partitions'],'registered_tests':len(p['registered_tests']),
            'loaded_tests':len(full.tests),'nodes':len(batch.x),'edges':len(batch.edge_type),'events':sum(g['events'] for g in p['tests']),
            'input_shard_bytes':sum(g[k]['bytes'] for g in p['tests'] for k in ('tensor','metadata')),
            'tensor_bytes':sum(t.numel()*t.element_size() for t in vars(batch).values() if isinstance(t,torch.Tensor)),
            'losses':{'tests':0,'nodes':0,'edges':0,'events':0},'comparisons':scores,'seconds':time.monotonic()-product_start})
        print('VERIFIED',p['identity'],len(full.tests),'max_error',max(x['max_abs_error'] for x in scores.values()),flush=True)
    assert params=={k:digest({n:v.tolist() for n,v in m.state_dict().items()}) for k,m in models.items()}
    write_json(OUT/(name+'.json'),{'products':results,'graphs':graph_rows,'seconds':time.monotonic()-start,'frozen_parameters_sha256':params,
        'seed':20261101,'mode':'eval; inference_mode; CPU only','models_fitted':0,'model_output_is_classification_metric':False,'manifest_sha256':receipt['manifest_sha256']})

def stream_only():
    receipt=load(OUT/'export_receipt.json');p=ProductPackage(OUT/'package',expected_manifest_sha256=receipt['manifest_sha256'])
    torch.manual_seed(20261101);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    model=GraphClassifier('rgcn',hidden=16,layers=2).eval();results=[]
    for product in p.products:
        t=time.monotonic();forward_stream(model,p,product);results.append({'identity':product['identity'],'seconds':time.monotonic()-t})
    write_json(OUT/'stream_forward.json',{'products':results,'micro_tests':1,'device':'cpu','models_fitted':0})

def relocated():
    dest=OUT/'relocated/package';dest.parent.mkdir(exist_ok=True)
    shutil.copytree(OUT/'package',dest,dirs_exist_ok=False)
    # Deny every evidence path outside the relocated package once code/imports loaded.
    # Torch imports are already complete; no JVM/source path can be consulted.
    def guard(event,args):
        if event!='open' or not args or isinstance(args[0],int):return
        path=Path(args[0]).resolve();mode=args[1]
        reading=not isinstance(mode,str) or not any(c in mode for c in ('w','a','x','+'))
        if reading and not (path.is_relative_to(dest.resolve()) or path==OUT/'export_receipt.json' or path.suffix in ('.py','.pyc','.pyd','.dll')):
            raise PermissionError('Relocation proof forbids reading external evidence: '+str(path))
    receipt=load(OUT/'export_receipt.json');sys.addaudithook(guard)
    p=ProductPackage(dest,expected_manifest_sha256=receipt['manifest_sha256']);count=0;events=0
    for prod in p.products:
        for test in p.iter_tests(prod):count+=1;events+=len(test.metadata['events'])
    write_json(OUT/'relocation.json',{'graphs':count,'events':events,'root_changed':True,'external_reads_forbidden':True,
                                    'actual_OS':'Windows','Linux_executed':False,'manifest_sha256':receipt['manifest_sha256']})
    print('RELOCATED',count,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['build','verify','stream_only','relocated']);args=a.parse_args()
    from cwm_fp.baselines.boundaries import install_io_guard
    install_io_guard('features')
    # Reuse the installed monitoring dependency without changing either environment.
    try:import psutil
    except ModuleNotFoundError:
        sys.path.append('D:/SoftInstall/Lib/site-packages');import psutil
    process=psutil.Process();done=threading.Event();samples=[];start=time.monotonic()
    def monitor():
        while not done.is_set():
            mem=process.memory_info();samples.append(mem.rss)
            if mem.rss>1610612736:os._exit(87)
            done.wait(.01)
    t=threading.Thread(target=monitor,daemon=True);t.start();code=1
    try:globals()[args.action]();code=0
    finally:
        done.set();t.join()
        write_json(OUT/(args.action+'_self_resource.json'),{'pid':os.getpid(),'scope':'actual Python tensor process; includes imported torch',
            'peak_sampled_rss_bytes':max(samples),'memory_samples':len(samples),'seconds':time.monotonic()-start,'exit_code':code,
            'argv':[sys.executable,*sys.argv],'torch_version':torch.__version__,'python':sys.version,'GPU_executed':False})
