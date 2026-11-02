"""Frozen 04R1 comparison using the existing package, model and evaluator APIs.

Module imports are torch-free so the existing CPU sklearn environment and the
independent validation evaluator can run without importing a training model.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from cwm_fp.data.common import ROOT,KEY,sha256,write_json,write_csv
from cwm_fp.data.labels import load_targets
from cwm_fp.baselines.boundaries import install_io_guard
from .graph_metrics import diagnostic_metrics

MANIFEST='71770793f4c7a063c084591f5ebf184d21368296895b3ebe9e3e4d77a7d41289'
SEEDS=(20261101,20261102,20261103)
KINDS=('nodeonly','untyped','rgcn')
CANDIDATES=(10,50,100,200)


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def environment():
    versions={}
    for name in ('torch','numpy','scipy','scikit-learn'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    return {'os':platform.platform(),'hostname':socket.gethostname(),'python':sys.version,'executable':sys.executable,
            'versions':versions,'device':'cpu','cwd':str(Path.cwd())}


def label_guard(role,allowed):
    allowed=Path(allowed).resolve();install_io_guard(role,[allowed]);reads=Counter();blocked=[]
    roots=[(ROOT/'data/processed/labels').resolve(),allowed.parent.parent]
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1],args[2]
        writing=isinstance(mode,str) and any(c in mode for c in 'wax+') or isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if not writing and any(p.is_relative_to(r) for r in roots):
            if p!=allowed:blocked.append(str(p));raise PermissionError(role+' target boundary: '+str(p))
            reads[str(p)]+=1
    sys.addaudithook(hook)
    return reads,blocked


def load_protocol(path,expected,*,sources=True):
    if path is None or expected is None or sha256(Path(path))!=expected:raise ValueError('Independent frozen protocol hash required')
    p=read(path)
    if (p['stage_id']!='04R1' or p['protocol']!='grouped_bank_pilot_v1' or p['manifest_sha256']!=MANIFEST
        or p['test_evaluation_allowed'] or p['seeds']!=list(SEEDS) or p['candidates']!=list(CANDIDATES)
        or p['neural_methods']!=list(KINDS)):raise ValueError('Unexpected frozen experiment')
    if sources:
        for name,h in p['source_sha256'].items():
            if sha256(ROOT/name)!=h:raise ValueError('Source changed after freeze: '+name)
    for name in ('products','tests','splits'):
        if sha256(Path(p['paths'][name]))!=p['input_sha256'][name]:raise ValueError('Public input changed')
    if sha256(Path(p['paths']['package'])/'manifest.json')!=MANIFEST:raise ValueError('Package changed')
    return p


def input_args(p):
    a=dict(p['legacy_input_args']);a.update({k:v for k,v in p['paths'].items() if k in a})
    for key in ('package','products','tests','splits','train_labels','out'):a[key]=Path(a[key])
    return SimpleNamespace(**a)


def stats_columns(relations,node_types):
    return ['log1p_registered_tests','log1p_mean_nodes','log1p_mean_edges','log1p_mean_events']+[
        'log1p_mean_relation_'+r for r in relations]+['mean_node_fraction_'+n for n in node_types]


def freeze(a):
    import torch
    from .graph_classifier import GraphClassifier
    from .graph_encoder import FEATURES,RELATIONS,NODE_TYPES
    from .package_training import load_train_inputs
    start=time.monotonic();out=a.run_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    if a.protocol_file.exists() or (out/'protocol.json').exists():raise ValueError('A frozen protocol already exists')
    old=read(a.old_run/'plan.json');paths={k:old['args'][k] for k in ('package','products','tests','splits','train_labels')}
    for k in paths:
        if getattr(a,k,None):paths[k]=str(getattr(a,k).resolve())
    val=a.val_labels or Path(paths['train_labels']).with_name('val.csv')
    # File name was located by directory inventory; never open labels during freeze.
    if val.name!='val.csv' or val.parent.name!='grouped' or not val.is_file():raise ValueError('Existing grouped validation target required')
    paths.update(val_labels=str(val.resolve()),out=str(out),old_run=str(a.old_run.resolve()),stats_python=str(a.stats_python.resolve()))
    packpath=Path(paths['package']);receipt=packpath.parent/'export_receipt.json'
    if sha256(packpath/'manifest.json')!=MANIFEST or MANIFEST not in json.dumps(read(receipt)):raise ValueError('Independent collector receipt mismatch')
    existing=SimpleNamespace(**old['args'])
    for k in ('package','products','tests','splits','train_labels','out'):setattr(existing,k,Path(paths[k]))
    pack,train,assignments,counts=load_train_inputs(existing)
    old_review=read(ROOT/'outputs/phase04T1_review_packet.json')
    cp=a.old_run/'engineering_train_only.pt';cp_receipt=read(a.old_run/'engineering_train_only.receipt.json')
    if sha256(cp)!=cp_receipt['checkpoint_sha256']:raise ValueError('Engineering checkpoint mismatch')
    if (cp_receipt['checkpoint_sha256']!=read(a.old_run/'result.json')['checkpoint_sha256'] or
        cp_receipt['checkpoint_sha256'] not in (ROOT/'outputs/phase04T1_review_packet.md').read_text(encoding='utf-8')):
        raise ValueError('Independent04T1 result mismatch')
    micro=read(a.old_run/'micro_selection.json')
    if len(micro['identities'])!=4 or not {tuple(k) for k in micro['identities']}.issubset({p.identity for p in train}):raise ValueError('Old micro cohort changed')
    model_defs={}
    for kind in KINDS:
        torch.manual_seed(SEEDS[0]);m=GraphClassifier(kind,16,2)
        model_defs[kind]={'kind':kind,'hidden':16,'layers':2,'features':list(FEATURES),'parameters':sum(v.numel() for v in m.parameters()),
            'input_projection':'Linear22->16 + ReLU','head':'Linear16->16 + ReLU + Linear16->1',
            'pooling':'mean nodes within each test; mean all registered tests within each product',
            'message':'none; self Linear16->16+ReLU per layer' if kind=='nodeonly' else
            'ReLU(W_self*h + sum_r mean_incoming_r(h_source*W_r)); W_r=sum_2bases coeff[r,b]*base[b]',
            'relation_count':0 if kind=='nodeonly' else 1 if kind=='untyped' else 7,
            'preserves_all22_node_features':True,'preserves_edge_direction_and_multiplicity':kind!='nodeonly'}
    p={'stage_id':'04R1','protocol':'grouped_bank_pilot_v1','manifest_sha256':MANIFEST,'paths':paths,
        'legacy_input_args':old['args'],'input_sha256':{k:sha256(Path(paths[k])) for k in ('products','tests','splits')},
        'independent_receipt_sha256':sha256(receipt),'train_target_sha256':old['args']['train_label_sha256'],
        'validation_target_sha256_before_read':None,'validation_hash_reason':'First target open occurs only in evaluator after this protocol is frozen',
        'cohort':[{k:r[k] for k in ('identity','sample_id','suite_id','partitions','registered_tests')} for r in pack.products],
        'cohort_counts':counts,'models':model_defs,'methods':['Prior','Stats-LR','NodeOnly','Untyped-GNN','R-GCN'],
        'neural_methods':list(KINDS),'seeds':list(SEEDS),'candidates':list(CANDIDATES),'epochs':200,'micro_steps':500,
        'batch_products':2,'dtype':'float32','optimizer':{'name':'Adam','learning_rate':.001,'class_weight':None},
        'initialization':'fresh GraphClassifier(seed) per model/seed; fixed same seed shuffle per epoch; no warm start',
        'micro_identities':micro['identities'],'micro_selection_sha256':sha256(a.old_run/'micro_selection.json'),
        'engineering_checkpoint_sha256':sha256(cp),'engineering_contract':cp_receipt['contract'],
        'statistics_columns':stats_columns(RELATIONS,NODE_TYPES),'statistics_count_columns':11,
        'statistics_formula':'log1p(product registered tests), log1p(mean per-test node/edge/event and7 relation counts); mean per-test node-category fractions5',
        'stats_lr':{'C':1.0,'max_iter':1000,'solver':'lbfgs','tol':1e-4,'penalty':'l2','fit_intercept':True,'class_weight':None,
                    'scaler':'TRAIN17 mean and population std; zero variance scale=1','random_state':SEEDS[0]},
        'candidate_selection':'minimum validation product mean log loss; improvement must exceed1e-6, otherwise earlier epoch; every seed reported',
        'validation_timing':'Only epochs10/50/100/200 generate sealed validation predictions; independent evaluation of all fixed candidates after training, no adaptive feedback',
        'metrics':{'threshold':.5,'label_encoding':{'FP':1,'TP':0},'probability_clip_log_loss':1e-12,
            'zero_denominator':'precision/recall/F1=0; explicitly report support','single_class':'ROC-AUC and AP=null; no ranking interpretation; binary log loss/Brier remain defined',
            'AP':'sum recall increment times tied-score precision; not trapezoidal PR area','confusion_axes':'rows true[TP0,FP1],columns predicted[TP0,FP1]',
            'versions':'per-version mean loss and error counts, no strong significance inference from4 validation versions'},
        'diagnostics':{'micro_at_steps':[0,100,200,300,400,500],'representation':'TRAIN only; projected/layer-node,test-mean,product-mean variances; all product-pair distances',
            'near_constant_probability_std_threshold':1e-4,'near_equal_product_pair_distance_threshold':1e-6,
            'pooling_warning':'different row populations; variance contraction alone does not prove rare signal dilution'},
        'budget':{'optimizer_updates':20000,'planned_neural_updates':16200,'planned_micro_updates':500,
            'lr_max_iterations_accounted':1000,'planned_max_all_optimizer_iterations':17700,
            'model_seconds_total':3600,'run_wall_seconds':3500,'checks_reserved_seconds':100,'new_bytes':512*1024**2,'minimum_free_bytes':1024**3},
        'environment':environment(),'stats_environment_preflight':'Separate existing sklearn interpreter, same host and CPU; no shared environment changes',
        'source_sha256':{str(f.relative_to(ROOT)):sha256(f) for f in (ROOT/'src/cwm_fp').rglob('*.py')},
        'target_permissions':{'trainer':'only frozen grouped/train.csv; only17 retained','evaluator':'only frozen grouped/val.csv; only5 retained',
            'test':'prohibited, no prediction or diagnostic','other_protocols':'prohibited','boundary_kind':'program-level, not OS isolation'},
        'test_evaluation_allowed':False,'device':'cpu','gpu_status':'NOT_VERIFIED','freeze_elapsed_seconds':time.monotonic()-start}
    a.protocol_file.parent.mkdir(parents=True,exist_ok=True)
    with a.protocol_file.open('x',encoding='utf-8') as f:json.dump(p,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    shutil.copy2(a.protocol_file,out/'protocol.json');h=sha256(a.protocol_file)
    write_json(out/'protocol.receipt.json',{'sha256':h,'file':str(a.protocol_file.resolve()),'validation_targets_opened':False})
    for name in p['source_sha256']:
        dest=out/'executed_source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    print(json.dumps({'protocol_sha256':h,'counts':counts,'parameters':{k:v['parameters'] for k,v in model_defs.items()}}))


def product_statistics(items):
    import numpy as np
    from .graph_encoder import RELATIONS
    rows=[]
    for p in items:
        counts=[];fractions=[]
        for t in p.tests:
            counts.append([len(t.x),len(t.edge_type),len(t.metadata['events'])]+[int((t.edge_type==i).sum()) for i in range(len(RELATIONS))])
            fractions.append(t.x[:,:5].mean(0).tolist())
        rows.append([math.log1p(len(p.tests))]+np.log1p(np.mean(counts,axis=0)).tolist()+np.mean(fractions,axis=0).tolist())
    return rows


def make_model(kind,seed,train,assignments):
    import torch
    from .graph_classifier import GraphClassifier
    torch.manual_seed(seed);random.seed(seed)
    model=GraphClassifier(kind,16,2);model.fit_normalization(train,assignments)
    return model


def predict_records(model,items):
    from .train_graph import predict
    if any(p.partition not in ('train','val') for p in items):raise ValueError('No test predictions authorized')
    return predict(model,items,batch_products=2)


def representation_summary(model,items):
    import torch
    from .graph_encoder import collate,mean_pool
    if any(p.partition!='train' for p in items):raise ValueError('Representation diagnostics are TRAIN only')
    model.eval();collected={};products=[];handles=[];captured={}
    handles.append(model.encoder.input.register_forward_hook(lambda m,i,o:captured.__setitem__('projected_nodes',torch.relu(o))))
    for i,layer in enumerate(model.encoder.layers):
        handles.append(layer.register_forward_hook(lambda m,args,o,i=i:captured.__setitem__('layer_'+str(i+1)+'_nodes',o)))
    try:
        with torch.no_grad():
            for product in items:
                batch=collate([product]);h=model.encode(batch);products.append(h)
                tests=mean_pool(captured['layer_2_nodes'],batch.node_test,len(batch.test_product))
                for name,value in {**captured,'test_means':tests}.items():collected.setdefault(name,[]).append(value)
        all_products=torch.cat(products);collected['product_means']=[all_products]
        summary={}
        for name,parts in collected.items():
            x=torch.cat(parts);summary[name]={'rows':len(x),'dimension':x.shape[1],'mean_feature_variance':float(x.var(0,unbiased=False).mean()),
                'mean_row_norm':float(x.norm(dim=1).mean())}
        distances=torch.pdist(all_products)
        summary['product_distances']={'pairs':len(distances),'min':float(distances.min()),'max':float(distances.max()),
            'mean':float(distances.mean()),'near_zero_le_1e_6':int((distances<=1e-6).sum())}
        return summary
    finally:
        for h in handles:h.remove()


def select_checkpoint(candidates):
    if not candidates:raise ValueError('No candidate to select')
    best=None
    for c in sorted(candidates,key=lambda c:c['epoch']):
        if not math.isfinite(c['metrics']['log_loss']):raise ValueError('Nonfinite validation loss')
        if best is None or c['metrics']['log_loss']<best['metrics']['log_loss']-1e-6:best=c
    return best


def stats_worker(path,expected):
    p=load_protocol(path,expected);out=Path(p['paths']['out']);a=input_args(p)
    reads,blocked=label_guard('04R1_stats_train',a.train_labels);start=time.monotonic()
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.exceptions import ConvergenceWarning
    import warnings
    try:
        if sha256(a.train_labels)!=p['train_target_sha256']:raise ValueError('Training target hash mismatch')
        all_targets=load_targets(a.train_labels,a.products,a.splits,partition='train',purpose='train')
        data=read(out/'stats_features.json');tr=data['train'];va=data['val']
        expected_keys={tuple(r['identity']) for r in p['cohort'] if r['partitions']['grouped']=='train'}
        if {tuple(r['identity']) for r in tr}!=expected_keys or len(tr)!=17:raise ValueError('Statistics training cohort mismatch')
        valkeys={tuple(r['identity']) for r in p['cohort'] if r['partitions']['grouped']=='val'}
        if {tuple(r['identity']) for r in va}!=valkeys or len(va)!=5:raise ValueError('Statistics validation cohort mismatch')
        x=np.asarray([r['features'] for r in tr]);v=np.asarray([r['features'] for r in va]);y=np.asarray([int(all_targets[tuple(r['identity'])]=='FP') for r in tr]);del all_targets
        if x.shape!=(17,16) or v.shape!=(5,16) or not np.isfinite(x).all() or not np.isfinite(v).all():raise ValueError('Invalid frozen statistics columns')
        scaler=StandardScaler();z=scaler.fit_transform(x);zv=scaler.transform(v)
        model=LogisticRegression(C=1,max_iter=1000,solver='lbfgs',tol=1e-4,penalty='l2',fit_intercept=True,class_weight=None,random_state=SEEDS[0])
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always');model.fit(z,y)
        np.savez(out/'stats_lr_parameters.npz',coef=model.coef_,intercept=model.intercept_,classes=model.classes_,mean=scaler.mean_,scale=scaler.scale_,var=scaler.var_)
        for part,rows,arr in [('train',tr,z),('val',va,zv)]:
            pred=model.predict_proba(arr)[:,list(model.classes_).index(1)]
            result=[{k:r[k] for k in ('sample_id','identity','partition')}|{'y_score':float(s),'y_pred':int(s>=.5),'score_kind':'probability_uncalibrated'} for r,s in zip(rows,pred)]
            write_json(out/('Stats-LR_'+part+'.json'),result)
        info={'environment':environment(),'converged':not any(issubclass(w.category,ConvergenceWarning) for w in ws),
            'warnings':[str(w.message) for w in ws],'iterations':model.n_iter_.tolist(),'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
            'zero_variance_columns':[p['statistics_columns'][i] for i in np.flatnonzero(scaler.var_==0)],
            'fit_product_keys':[r['identity'] for r in tr],'fit_products':17,'seconds':time.monotonic()-start,'parameters_sha256':sha256(out/'stats_lr_parameters.npz')}
        write_json(out/'stats_lr_audit.json',info)
    finally:write_json(out/'stats_target_access.json',{'role':'TRAIN only','reads':dict(reads),'blocked':blocked,'OS_isolation':False})
    print('STATS_LR_COMPLETE')


def evaluate_candidates(path,expected,index_path):
    p=load_protocol(path,expected);out=Path(p['paths']['out']);index_path=Path(index_path)
    if index_path.resolve()!= (out/'prediction_index.json').resolve():raise ValueError('Unknown prediction index')
    index=read(index_path)
    if index['protocol_sha256']!=expected:raise ValueError('Prediction protocol mismatch')
    from .graph_evaluator import score_rows
    val=Path(p['paths']['val_labels']);reads,blocked=label_guard('04R1_validation_only',val)
    with (out/'validation.claim').open('x',encoding='utf-8') as f:f.write(expected+'\n'+sha256(index_path)+'\n')
    try:
        wanted=[r for r in p['cohort'] if r['partitions']['grouped']=='val'];keys={tuple(r['identity']) for r in wanted}
        labels=load_targets(val,Path(p['paths']['products']),Path(p['paths']['splits']),partition='val',purpose='evaluate')
        targets={k:labels[k] for k in keys};del labels
        if any(x not in ('FP','TP') for x in targets.values()):raise ValueError('Validation target missing/nonbinary')
        truth={r['sample_id']:{'identity':r['identity'],'y_true':int(targets[tuple(r['identity'])]=='FP')} for r in wanted}
        results=[];seen=set()
        for entry in index['candidates']:
            method,seed,epoch=entry['method'],entry['seed'],entry['epoch'];ident=(method,seed,epoch)
            if ident in seen:raise ValueError('Repeated validation candidate')
            seen.add(ident)
            if not (method in ('Prior','Stats-LR') and seed is None and epoch==0 or method in KINDS and seed in SEEDS and epoch in CANDIDATES):raise ValueError('Unplanned validation candidate')
            file=out/entry['val_file']
            if not file.resolve().is_relative_to(out.resolve()) or sha256(file)!=entry['val_sha256']:raise ValueError('Validation prediction changed')
            rows=read(file);score_rows(rows,truth,partition='val',expected_ids=set(truth))
            metric=diagnostic_metrics(rows,targets);results.append({**entry,'metrics':metric})
            joined=[r|{'y_true':truth[r['sample_id']]['y_true']} for r in rows]
            write_json(out/'controlled_validation'/file.name,joined)
        selections=[]
        for kind in KINDS:
            for seed in SEEDS:
                candidates=[r for r in results if r['method']==kind and r['seed']==seed]
                if candidates:selections.append(select_checkpoint(candidates))
        write_json(out/'validation_results.json',{'protocol_sha256':expected,'index_sha256':sha256(index_path),
            'target_sha256':sha256(val),'candidates':results,'selected':selections,'target_products':5,'target_versions':4,
            'class_counts':dict(Counter(targets.values())),'test_evaluated':False})
    finally:write_json(out/'validation_target_access.json',{'role':'independent validation evaluator after protocol freeze','reads':dict(reads),'blocked':blocked,'OS_isolation':False})
    print(json.dumps({'validation_candidates':len(results),'selected':len(selections),'test_evaluated':False}))


def run_comparison(path,expected):
    p=load_protocol(path,expected);out=Path(p['paths']['out']);a=input_args(p)
    if (out/'run_claim.json').exists():raise ValueError('Existing run cannot be silently retrained')
    if shutil.disk_usage(out).free<p['budget']['minimum_free_bytes']:raise ValueError('Insufficient free space before training')
    write_json(out/'run_claim.json',{'protocol_sha256':expected,'started_utc':time.time()})
    reads,blocked=label_guard('04R1_trainer',a.train_labels)
    import torch
    from .package_training import (load_train_inputs,join_targets,ModelBudget,BudgetStop,training_step,optimizer_for,
        save_checkpoint,load_checkpoint,process_memory)
    from .graph_encoder import collate,RELATIONS
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    pack,train,assignments,integrity=load_train_inputs(a)
    validation=[pack.materialize(r,'grouped') for r in pack.products if r['partitions']['grouped']=='val']
    targets=join_targets(a,train);write_json(out/'input_integrity.json',integrity)
    relations={part:{r:sum(int((t.edge_type==i).sum()) for product in items for t in product.tests) for i,r in enumerate(RELATIONS)} for part,items in [('train',train),('val',validation)]}
    write_json(out/'actual_model_inputs.json',{'relations':relations,'feature_columns':22,'lossless':True,'test_model_calls':0,
        'feature_columns_observed':[{'column':i,'values':torch.unique(torch.cat([t.x[:,i] for product in train for t in product.tests])).tolist()} for i in range(22)]})
    budget=ModelBudget(out,p['budget']['optimizer_updates'],p['budget']['run_wall_seconds']);curves=[];candidates=[];diagnostics={};run_status='RUNNING'
    def append_candidate(method,seed,epoch,train_rows,val_rows,checkpoint=None):
        stem=f'{method}_{seed if seed is not None else "fixed"}_{epoch}'
        tf=stem+'_train.json';vf=stem+'_val.json';write_json(out/tf,train_rows);write_json(out/vf,val_rows)
        entry={'method':method,'seed':seed,'epoch':epoch,'train_file':tf,'val_file':vf,'train_sha256':sha256(out/tf),'val_sha256':sha256(out/vf),
               'train_metrics':diagnostic_metrics(train_rows,targets),'checkpoint':checkpoint}
        candidates.append(entry);write_json(out/'prediction_index.json',{'protocol_sha256':expected,'candidates':candidates})
    def diagnostic(model,items,labels):
        budget.check();rows=predict_records(model,items)
        return {'metrics':diagnostic_metrics(rows,labels),'representation':representation_summary(model,items)}
    try:
        proportion=sum(v=='FP' for v in targets.values())/len(targets)
        def constant(items):return [{'sample_id':x.sample_id,'identity':list(x.identity),'partition':x.partition,
            'y_score':proportion,'y_pred':int(proportion>=.5),'score_kind':'probability_uncalibrated'} for x in items]
        append_candidate('Prior',None,0,constant(train),constant(validation))
        write_json(out/'prior.json',{'p_train':proportion,'train_counts':dict(Counter(targets.values())),'train_metrics':candidates[-1]['train_metrics']})
        # Restore exactly the previous engineering model, for TRAIN diagnostics only.
        model=make_model('rgcn',SEEDS[0],train,assignments);opt=optimizer_for(model)
        load_checkpoint(Path(p['paths']['old_run'])/'engineering_train_only.pt',p['engineering_checkpoint_sha256'],model,opt,p['engineering_contract'])
        diagnostics['04T1_engineering']=diagnostic(model,train,targets)
        write_json(out/'engineering_train_predictions.json',predict_records(model,train))
        microkeys={tuple(k) for k in p['micro_identities']};micro=[x for x in train if x.identity in microkeys];mt={k:targets[k] for k in microkeys}
        if len(micro)!=4 or set(mt.values())!={'FP','TP'}:raise ValueError('Micro cohort changed')
        model=make_model('rgcn',SEEDS[0],train,assignments);opt=optimizer_for(model)
        diagnostics['micro']=[{'step':0,**diagnostic(model,micro,mt)}]
        for step in range(1,501):
            model.train();training_step(model,opt,micro,targets,budget,'micro500',step)
            if step%100==0:
                info={'step':step,**diagnostic(model,micro,mt)};diagnostics['micro'].append(info)
                print('MICRO',step,info['metrics']['log_loss'],flush=True)
        write_json(out/'prediction_diagnostics.json',diagnostics)
        data={}
        for part,items in [('train',train),('val',validation)]:
            features=product_statistics(items)
            data[part]=[{'identity':list(x.identity),'sample_id':x.sample_id,'partition':part,'features':f} for x,f in zip(items,features)]
        write_json(out/'stats_features.json',data);budget.check()
        command=[p['paths']['stats_python'],'-X','utf8','-m','cwm_fp.models.package_comparison','--action','stats','--protocol-file',str(path),'--protocol-sha256',expected]
        ts=time.monotonic()
        with (out/'stats_worker.log').open('w',encoding='utf-8') as log:proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=min(120,budget.max_seconds-(time.monotonic()-budget.start)))
        write_json(out/'stats_worker.command.json',{'argv':command,'exit_code':proc.returncode,'seconds':time.monotonic()-ts})
        if proc.returncode:raise RuntimeError('Statistics worker failed; inspect frozen log')
        append_candidate('Stats-LR',None,0,read(out/'Stats-LR_train.json'),read(out/'Stats-LR_val.json'))
        for kind in KINDS:
            for seed in SEEDS:
                budget.check();model=make_model(kind,seed,train,assignments);opt=optimizer_for(model);rng=random.Random(seed)
                runkey=f'{kind}_{seed}';diagnostics[runkey]={'initial':diagnostic(model,train,targets),'candidates':{}}
                runstart=time.monotonic()
                for epoch in range(1,201):
                    order=list(train);rng.shuffle(order);model.train();seen=[];gradient_norms=[];weighted=0.;changes=[]
                    for start in range(0,17,2):
                        sub=order[start:start+2];r=training_step(model,opt,sub,targets,budget,runkey,epoch)
                        seen.extend(x.identity for x in sub);weighted+=float(r['loss'])*len(sub)
                        gradient_norms.append(budget.rows[-1]['gradient_norm']);changes.append(budget.rows[-1]['parameter_max_abs_change'])
                    if len(seen)!=17 or set(seen)!=set(targets):raise ValueError('Silent train loss or duplicate product')
                    budget.check();train_rows=predict_records(model,train);metrics=diagnostic_metrics(train_rows,targets)
                    row={'method':kind,'seed':seed,'epoch':epoch,'global_step':budget.steps,'train_log_loss':metrics['log_loss'],
                        'update_time_mean_loss':weighted/17,'products':17,'tests':283,'lossless':True,
                        'mean_gradient_norm':sum(gradient_norms)/len(gradient_norms),'max_parameter_change':max(changes),
                        'device':'cpu','run_elapsed_seconds':time.monotonic()-runstart,'cumulative_seconds':time.monotonic()-budget.start}
                    curves.append(row);write_csv(out/'learning_curves.csv',curves,tuple(row))
                    if epoch in CANDIDATES:
                        if (shutil.disk_usage(out).free<p['budget']['minimum_free_bytes'] or
                            sum(f.stat().st_size for f in out.rglob('*') if f.is_file())>p['budget']['new_bytes']):raise BudgetStop('disk_budget')
                        checkpoint=out/'checkpoints'/f'{runkey}_{epoch}.pt';checkpoint.parent.mkdir(exist_ok=True)
                        contract={'kind':'04R1_fixed_train_validation','protocol_sha256':expected,'method':kind,'seed':seed,'epoch':epoch,'model':p['models'][kind]}
                        ch=save_checkpoint(checkpoint,model,opt,contract,budget.steps)
                        append_candidate(kind,seed,epoch,train_rows,predict_records(model,validation),{'path':str(checkpoint.relative_to(out)),'sha256':ch})
                        diagnostics[runkey]['candidates'][str(epoch)]={'metrics':metrics,'representation':representation_summary(model,train)}
                        write_json(out/'prediction_diagnostics.json',diagnostics)
                        print('CANDIDATE',kind,seed,epoch,'train_loss',metrics['log_loss'],'steps',budget.steps,flush=True)
                diagnostics[runkey]['seconds']=time.monotonic()-runstart
        run_status='COMPLETE'
    except BudgetStop as exc:run_status='PARTIAL_BUDGET';write_json(out/'failure.json',{'error':str(exc),'updates':budget.steps})
    except BaseException as exc:
        run_status='FAILED';write_json(out/'failure.json',{'type':type(exc).__name__,'error':str(exc),'updates':budget.steps});raise
    finally:
        write_json(out/'prediction_diagnostics.json',diagnostics)
        write_json(out/'trainer_target_access.json',{'role':'trainer TRAIN only','reads':dict(reads),'blocked':blocked,'OS_isolation':False})
        write_json(out/'training_result.json',{'status':run_status,'actual_updates':budget.steps,'epochs_completed':len(curves),'candidate_count':len(candidates),
            'model_window_seconds':time.monotonic()-budget.start,'environment':environment(),'memory':process_memory(),
            'GPU_peak_bytes':None,'GPU_reason':'CPU only; no Linux/GPU session','test_predictions':0,'test_targets_read':False})
    print('COMPARISON_TRAINING',run_status,flush=True)
    return 0 if run_status=='COMPLETE' else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('freeze','stats'),required=True);parser.add_argument('--protocol-file',type=Path,required=True)
    parser.add_argument('--protocol-sha256');parser.add_argument('--run-dir',type=Path);parser.add_argument('--old-run',type=Path)
    parser.add_argument('--stats-python',type=Path)
    for name in ('package','products','tests','splits','train-labels','val-labels'):parser.add_argument('--'+name,type=Path)
    a=parser.parse_args()
    if a.action=='freeze':freeze(a)
    else:stats_worker(a.protocol_file,a.protocol_sha256)


if __name__=='__main__':main()
