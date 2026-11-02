"""Train-only engineering acceptance of the existing R-GCN and pooling model.

No held-out predictions, evaluator callback, early stopping or new architecture.
All optimizer updates (including controls) share one bounded budget.
"""
from __future__ import annotations
import copy
import csv
from collections import Counter,defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import socket
import sys
import time
import torch
from torch.nn import functional as F
from cwm_fp.data.common import ROOT,KEY,key,read_csv,sha256,write_json,write_csv
from cwm_fp.data.labels import load_targets
from cwm_fp.baselines.boundaries import install_io_guard
from .graph_classifier import GraphClassifier
from .graph_encoder import FEATURE_SCHEMA,FEATURES,RELATIONS,ProductTensor,collate,encode_test_lossless,FORBIDDEN
from .graph_package import ProductPackage,forward_train_stream,digest

PROTOCOL='grouped_bank_pilot_v1'
ATOL=1e-6
RTOL=1e-5


def add_arguments(p):
    p.add_argument('--package',type=Path)
    p.add_argument('--manifest-sha256')
    p.add_argument('--protocol',default=PROTOCOL)
    p.add_argument('--partition',default='train')
    p.add_argument('--train-labels',type=Path)
    p.add_argument('--train-label-sha256')
    p.add_argument('--products',type=Path)
    p.add_argument('--tests',type=Path)
    p.add_argument('--splits',type=Path)
    p.add_argument('--split-sha256')
    p.add_argument('--feature-schema',default=FEATURE_SCHEMA)
    p.add_argument('--relation-mode',default='full')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--out',type=Path)
    p.add_argument('--max-updates',type=int,default=400)
    p.add_argument('--max-model-seconds',type=float,default=1800)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--micro-steps',type=int,default=100)
    p.add_argument('--synthetic-steps',type=int,default=40)
    p.add_argument('--batch-products',type=int,default=2)


def validate_options(a):
    if any(getattr(a,n,None) is None for n in ('package','manifest_sha256','train_labels','train_label_sha256','products','tests','splits','split_sha256','out')):
        raise ValueError('Package, independent hashes, public manifests, TRAIN targets and output are required')
    if a.protocol!=PROTOCOL or a.partition!='train' or a.fold!='grouped':raise ValueError('Only the frozen grouped TRAIN subset is authorized')
    if a.kind!='rgcn' or a.feature_schema!=FEATURE_SCHEMA or a.relation_mode!='full' or a.ablation!='full':
        raise ValueError('04T1 requires existing R-GCN,22 semantic columns and all7 relations')
    if a.train_labels.name!='train.csv' or a.train_labels.parent.name!='grouped':raise ValueError('Only grouped/train.csv may be opened')
    if not (0<a.max_updates<=400 and 0<a.max_model_seconds<=1800 and 1<=a.epochs<=10 and 1<=a.micro_steps<=100 and 1<=a.synthetic_steps<=40 and a.batch_products in (1,2)):
        raise ValueError('Engineering budget exceeded')
    if a.out.resolve().is_relative_to(a.package.resolve()) or a.package.resolve().is_relative_to(a.out.resolve()):raise ValueError('Outputs must be separate from frozen package')


def access_guard(label_path):
    label_path=label_path.resolve();install_io_guard('engineering_train_only',[label_path]);reads=Counter();rejections=[]
    sealed=(ROOT/'data/processed/labels').resolve();external_root=label_path.parent.parent
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1],args[2]
        writing=isinstance(mode,str) and any(c in mode for c in 'wax+') or isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if not writing and (path.is_relative_to(sealed) or path.is_relative_to(external_root)):
            if path!=label_path:rejections.append(str(path));raise PermissionError('TRAIN-only target boundary: '+str(path))
            reads[str(path)]+=1
    sys.addaudithook(hook)
    return reads,rejections


def reject_target_fields(value):
    if isinstance(value,dict):
        if FORBIDDEN.intersection(value):raise ValueError('Protected target or fault fields in package metadata')
        for v in value.values():reject_target_fields(v)
    elif isinstance(value,list):
        for v in value:reject_target_fields(v)


def load_train_inputs(a):
    from .train_graph import check_partitions
    pack=ProductPackage(a.package,expected_manifest_sha256=a.manifest_sha256)
    if pack.manifest.get('primary_protocol')!=a.protocol:raise ValueError('Wrong package protocol')
    sources=pack.manifest['source_provenance']['frozen_manifests_sha256']
    if sha256(a.products)!=sources['products'] or sha256(a.tests)!=sources['tests'] or sha256(a.splits)!=a.split_sha256:
        raise ValueError('Public manifest or frozen split mismatch')
    products={key(p):p for p in read_csv(a.products,KEY)};splits={key(p):p['split'] for p in read_csv(a.splits,(*KEY,'split'))}
    tests=defaultdict(list)
    for t in read_csv(a.tests,(*KEY,'test_id')):tests[key(t)].append(t['test_id'])
    train=[];totals={s:Counter() for s in ('train','val','test')};versions=defaultdict(set)
    for row in pack.products:
        k=tuple(row['identity']);part=row['partitions']['grouped']
        if k[0]!='BankAccountTP' or part!=splits[k] or sorted(tests[k])!=row['registered_tests'] or products[k]['suite_id']!=row['suite_id']:
            raise ValueError('Package differs from frozen product/suite/split')
        totals[part].update(products=1,graphs=len(row['tests']));versions[part].add(k[:2])
        item=pack.materialize(row,'grouped')
        for t in item.tests:
            reject_target_fields(t.metadata)
            if t.x.shape[1]!=22:raise ValueError('Feature width mismatch')
            totals[part].update(nodes=len(t.x),edges=len(t.edge_type),events=len(t.metadata['events']))
        if part=='train':train.append(item)
        # Held-out tensors are only verified and discarded, never passed to a model.
    assignment={tuple(p['identity']):p['partitions']['grouped'] for p in pack.products}
    check_partitions(train,assignment)
    expected={'train':(17,9,283),'val':(5,4,85),'test':(6,3,95)}
    for s,(np,nv,ng) in expected.items():
        if (totals[s]['products'],len(versions[s]),totals[s]['graphs'])!=(np,nv,ng):raise ValueError('Unexpected04D1 cohort')
    return pack,train,assignment,{s:{**totals[s],'versions':len(versions[s])} for s in totals}


def join_targets(a,train):
    if sha256(a.train_labels)!=a.train_label_sha256:raise ValueError('Authorized training target fingerprint changed')
    # Reuse the original strict partition validator; only selected keys survive.
    all_targets=load_targets(a.train_labels,a.products,a.splits,partition='train',purpose='train')
    targets={p.identity:all_targets.get(p.identity) for p in train};del all_targets
    check_training_batch(train,targets)
    if set(targets.values())!={'FP','TP'}:raise ValueError('Both TRAIN target classes required')
    return targets


def check_training_batch(products,targets):
    if not products or len({p.identity for p in products})!=len(products):raise ValueError('Empty/duplicate product batch')
    for p in products:
        if p.partition!='train' or p.observed_result!='PASS' or targets.get(p.identity) not in ('FP','TP'):
            raise ValueError('TRAIN-only FP/TP join required')
        if not p.tests or p.audit.get('preview_only') or any(p.audit.get(k,0) for k in ('lost_node_fraction','lost_edge_fraction','lost_event_fraction','lost_test_fraction')):
            raise ValueError('Lossy/empty/unqualified product')
        for t in p.tests:
            if tuple(t.metadata['identity'][:3])!=p.identity or t.metadata.get('preview_only') or not t.metadata.get('lossless'):
                raise ValueError('Wrong test identity or unqualified test')
        if p.audit.get('tests_before')!=len(p.tests):raise ValueError('Missing original tests')


def new_model(seed,device,train=None,assignments=None):
    torch.manual_seed(seed);random.seed(seed)
    m=GraphClassifier('rgcn',hidden=16,layers=2,feature_schema=FEATURE_SCHEMA)
    if train:m.fit_normalization(train,assignments)
    return m.to(device)


def optimizer_for(model):
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in model.parameters() if p.requires_grad}
    return opt


class BudgetStop(RuntimeError):pass


class ModelBudget:
    def __init__(self,out,updates,seconds):
        self.out=out;self.max_updates=updates;self.max_seconds=seconds;self.steps=0;self.start=time.monotonic();self.rows=[]
    def check(self,update=False):
        if time.monotonic()-self.start>=self.max_seconds:raise BudgetStop('model_time_budget')
        if update and self.steps>=self.max_updates:raise BudgetStop('optimizer_update_budget')
    def log_step(self,phase,epoch,loss,products,grad,delta,device):
        self.steps+=1
        row={'phase':phase,'epoch':epoch,'step':self.steps,'loss':float(loss.detach()),'products':len(products),'tests':sum(len(p.tests) for p in products),
            'nodes':sum(len(t.x) for p in products for t in p.tests),'edges':sum(len(t.edge_type) for p in products for t in p.tests),
            'gradient_norm':grad,'parameter_max_abs_change':delta,'device':str(device),'elapsed_seconds':time.monotonic()-self.start,'lossless':True}
        self.rows.append(row)
        with (self.out/'training_trace.csv').open('a',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(row))
            if self.steps==1:w.writeheader()
            w.writerow(row)
        write_json(self.out/'budget.json',{'updates':self.steps,'max_updates':self.max_updates,'elapsed_seconds':row['elapsed_seconds'],'max_seconds':self.max_seconds,
            'scope':'Conservative wall time from first model construction through all diagnostics,forward,backward,serialization; includes intervening I/O.'})


def gradient_report(model):
    report={}
    for name,p in model.named_parameters():
        g=p.grad
        report[name]={'registered':True,'requires_grad':p.requires_grad,'present':g is not None,'finite':bool(torch.isfinite(g).all()) if g is not None else None,
            'norm':float(g.detach().norm()) if g is not None else None,'nonzero':int(torch.count_nonzero(g)) if g is not None else None}
    if any(not r['present'] or not r['finite'] for r in report.values()):raise ValueError('Disconnected/nonfinite parameter gradient')
    # Zero is valid for an inactive relation or cancelling/converged batch.
    # Nonzero encoder/head evidence is checked across the acceptance cases.
    return report


def training_step(model,opt,products,targets,budget,phase,epoch=0,logits_fn=None):
    check_training_batch(products,targets);budget.check(update=True);opt.zero_grad(set_to_none=True)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    encoded=[]
    h=model.encoder.register_forward_hook(lambda m,i,o:encoded.append(bool(o.requires_grad and o.grad_fn is not None)))
    try:logits=logits_fn() if logits_fn else model(collate(products).to(next(model.parameters()).device))
    finally:h.remove()
    if not logits.requires_grad or not encoded or not all(encoded):raise ValueError('Encoder/autograd graph disconnected')
    y=logits.new_tensor([1. if targets[p.identity]=='FP' else 0. for p in products])
    loss=F.binary_cross_entropy_with_logits(logits,y,reduction='mean')
    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
    loss.backward();report=gradient_report(model)
    grads={n:p.grad.detach().clone() for n,p in model.named_parameters()}
    norm=math.sqrt(sum(r['norm']**2 for r in report.values()))
    budget.check(update=True);opt.step()
    delta={n:float((p.detach()-before[n]).abs().max()) for n,p in model.named_parameters()}
    if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Nonfinite parameter update')
    budget.log_step(phase,epoch,loss,products,norm,max(delta.values()),next(model.parameters()).device)
    return {'logits':logits.detach().clone(),'loss':loss.detach().clone(),'gradients':grads,'gradient_checks':report,'parameter_changes':delta,'encoder_graph_retained':all(encoded)}


def compare(a,b):
    d=(a.detach().cpu()-b.detach().cpu()).abs()
    denom=torch.maximum(a.detach().cpu().abs(),b.detach().cpu().abs()).clamp_min(1e-12)
    return {'passed':bool(torch.allclose(a.detach().cpu(),b.detach().cpu(),atol=ATOL,rtol=RTOL)),
            'max_absolute_difference':float(d.max()) if d.numel() else 0.,'max_relative_difference':float((d/denom).max()) if d.numel() else 0.}


def compare_maps(a,b):
    if set(a)!=set(b):raise ValueError('Parameter names differ')
    per={n:compare(a[n],b[n]) for n in a}
    return {'passed':all(x['passed'] for x in per.values()),'max_absolute_difference':max(x['max_absolute_difference'] for x in per.values()),
        'max_relative_difference':max(x['max_relative_difference'] for x in per.values()),'parameters':per}


@torch.no_grad()
def mean_training_loss(model,products,targets,budget):
    budget.check();check_training_batch(products,targets);model.eval();losses=[]
    for p in products:
        budget.check();z=model(collate([p]).to(next(model.parameters()).device));y=z.new_tensor([float(targets[p.identity]=='FP')])
        losses.append(float(F.binary_cross_entropy_with_logits(z,y)))
    return sum(losses)/len(losses)


def equivalence(pack,train,targets,a,budget):
    large=sorted([p for p in train if len(p.tests)>16],key=lambda p:(-len(p.tests),p.identity))[0]
    small=sorted(train,key=lambda p:(len(p.tests),p.identity))[0];results=[]
    for label,p in [('over16',large),('small',small)]:
        m=new_model(a.seed,a.device);m.eval();n=copy.deepcopy(m);o=optimizer_for(m);v=optimizer_for(n)
        row=next(r for r in pack.products if tuple(r['identity'])==p.identity)
        left=training_step(m,o,[p],targets,budget,'gradient_whole_'+label)
        right=training_step(n,v,[p],targets,budget,'gradient_chunked_'+label,
            logits_fn=lambda:forward_train_stream(n,pack,row,micro_tests=5))
        checks={'logits':compare(left['logits'],right['logits']),'loss':compare(left['loss'],right['loss']),
                'gradients':compare_maps(left['gradients'],right['gradients']),'update':compare_maps(m.state_dict(),n.state_dict())}
        results.append({'case':label,'tests':len(p.tests),'micro_tests':5,'last_micro_tests':len(p.tests)%5 or 5,'grad_enabled':True,
            'mode':'eval, gradients enabled; actual training mode tested separately','checks':checks,'gradient_checks':left['gradient_checks'],
            'parameter_update_checks':left['parameter_changes'],'passed':all(c['passed'] for c in checks.values())})
    nonzero={prefix:any(r['nonzero'] for case in results for name,r in case['gradient_checks'].items() if name.startswith(prefix)) for prefix in ('encoder.','head.')}
    return {'atol':ATOL,'rtol':RTOL,'cases':results,'passed':all(r['passed'] for r in results),'nonzero_module_evidence':nonzero,
        'formal_training_path':'whole_product_only','stream_activation_memory_constant':False}


def representation_audit(train,targets):
    by=defaultdict(list);by_distribution=defaultdict(list);graph_hashes=Counter();object_ids=set();storages=set();degenerate=0;relations=Counter();shapes=[]
    for p in train:
        hashes=[]
        for t in p.tests:
            if id(t) in object_ids or t.x.data_ptr() in storages:raise ValueError('Same graph object/storage reused across test owners')
            object_ids.add(id(t));storages.add(t.x.data_ptr())
            h=hashlib.sha256();h.update(json.dumps(list(t.x.shape)).encode());h.update(t.x.contiguous().numpy().tobytes())
            triples=sorted(zip(t.edge_index[0].tolist(),t.edge_index[1].tolist(),t.edge_type.tolist()))
            h.update(json.dumps(triples,separators=(',',':')).encode());h.update(FEATURE_SCHEMA.encode());hashes.append(h.hexdigest());graph_hashes[hashes[-1]]+=1
            relations.update(t.edge_type.tolist());shapes.append(len(t.x));degenerate+=int(not len(t.x) or not t.edge_type.numel() or not torch.any(t.x))
        by[digest(sorted(hashes))].append(p.identity)
        counts=Counter(hashes);divisor=math.gcd(*counts.values());by_distribution[digest(sorted((h,n//divisor) for h,n in counts.items()))].append(p.identity)
    def stats(groups):
        duplicates=[ks for ks in groups.values() if len(ks)>1]
        conflicts=[ks for ks in duplicates if len({targets[k] for k in ks})>1]
        return {'duplicate_groups':len(duplicates),'products_in_duplicate_groups':sum(map(len,duplicates)),'opposite_label_groups':len(conflicts),'products_in_opposite_label_groups':sum(map(len,conflicts))}
    return {'scope':'TRAIN17 only; exact ordered node-feature rows with sorted edge multisets; product test order ignored. Not general graph isomorphism.',
        'canonical_product_tensor':stats(by),'equal_mean_test_distributions':stats(by_distribution),'unique_graph_inputs':len(graph_hashes),'graph_inputs':sum(graph_hashes.values()),
        'degenerate_graphs':degenerate,'shared_object_or_storage_errors':0,'relations':{RELATIONS[i]:relations[i] for i in range(len(RELATIONS))},
        'min_nodes':min(shapes),'max_nodes':max(shapes),'input_identities_not_encoded':True,'removed_conflicting_samples':False}


def save_checkpoint(path,model,opt,contract,step):
    torch.save({'kind':'engineering_train_only','contract':contract,'model':model.state_dict(),'optimizer':opt.state_dict(),
        'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),'cuda_rng':torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None,
        'step':step,'normalization':model.normalization_audit},path)
    h=sha256(path);write_json(path.with_suffix('.receipt.json'),{'checkpoint_sha256':h,'contract':contract});return h


def load_checkpoint(path,expected_sha,model,opt,contract):
    if sha256(path)!=expected_sha:raise ValueError('Checkpoint hash mismatch')
    d=torch.load(path,map_location=next(model.parameters()).device,weights_only=True)
    if d.get('kind')!='engineering_train_only' or d['contract']!=contract:raise ValueError('Checkpoint/cohort/schema/config mismatch')
    model.load_state_dict(d['model']);opt.load_state_dict(d['optimizer']);model.normalization_audit=d['normalization']
    torch.set_rng_state(d['torch_rng'].cpu());random.setstate(d['python_rng'])
    if d['cuda_rng'] is not None:torch.cuda.set_rng_state_all(d['cuda_rng'])
    return d


def resume_check(model,opt,train,targets,contract,a,budget):
    p=train[0];path=a.out/'resume_before.pt';h=save_checkpoint(path,model,opt,contract,budget.steps)
    restored=new_model(a.seed,a.device);other=optimizer_for(restored);load_checkpoint(path,h,restored,other,contract)
    model.eval();restored.eval()
    with torch.no_grad():before=compare(model(collate([p]).to(a.device)),restored(collate([p]).to(a.device)))
    model.train();left=training_step(model,opt,[p],targets,budget,'resume_continuous')
    restored.train();right=training_step(restored,other,[p],targets,budget,'resume_reloaded')
    after=compare_maps(model.state_dict(),restored.state_dict())
    gradients=compare_maps(left['gradients'],right['gradients']);rejections={}
    for field,bad in [('manifest_sha256','0'*64),('feature_schema','semantic_v1'),('relations',['data_dep'])]:
        try:load_checkpoint(path,h,restored,other,{**contract,field:bad})
        except ValueError:rejections[field]=True
        else:rejections[field]=False
    try:load_checkpoint(path,'0'*64,restored,other,contract)
    except ValueError:rejections['wrong_digest']=True
    return {'output':before,'next_step_parameters':after,'next_step_gradients':gradients,'mismatch_rejections':rejections,
        'passed':before['passed'] and after['passed'] and gradients['passed'] and all(rejections.values()),'checkpoint_sha256':h}


def artificial_inputs(out):
    from .graph_smoke import synthetic_graph
    folder=out/'synthetic_isolation';(folder/'grouped').mkdir(parents=True)
    rows=[];splits=[];targets=[];items=[]
    for i,part in enumerate(('train','train','val','test')):
        k=('SyntheticOnly','v'+str(i),'p');row=dict(zip(KEY,k));rows.append({**row,'observed_result':'PASS','suite_id':'s'+str(i)})
        splits.append({**row,'split':part})
        if part=='train':
            targets.append({**row,'label':'FP' if i else 'TP','suite_id':'s'+str(i)})
            g=synthetic_graph(k,'synthetic_test',i,coverage_signal=True);t=encode_test_lossless(g)
            items.append(ProductTensor(k,str(i),'train','PASS',[t],{'tests_before':1,'preview_only':False}))
    write_csv(folder/'products.csv',rows,(*KEY,'observed_result','suite_id'));write_csv(folder/'splits.csv',splits,(*KEY,'split'))
    write_csv(folder/'grouped/train.csv',targets,(*KEY,'label','suite_id'))
    labels=load_targets(folder/'grouped/train.csv',folder/'products.csv',folder/'splits.csv',partition='train',purpose='train')
    assert not (folder/'grouped/val.csv').exists() and not (folder/'grouped/test.csv').exists()
    return items,labels


def fit_fixed(model,opt,items,targets,budget,phase,epochs,batch,seed):
    initial=mean_training_loss(model,items,targets,budget);rng=random.Random(seed);history=[]
    for epoch in range(1,epochs+1):
        order=list(items);rng.shuffle(order);model.train();seen=[];weighted=0.
        for start in range(0,len(order),batch):
            sub=order[start:start+batch];r=training_step(model,opt,sub,targets,budget,phase,epoch)
            weighted+=float(r['loss'])*len(sub);seen.extend(p.identity for p in sub)
        assert set(seen)=={p.identity for p in items} and len(seen)==len(items)
        post=mean_training_loss(model,items,targets,budget)
        history.append({'epoch':epoch,'products':len(seen),'graphs':sum(len(p.tests) for p in items),'mean_update_time_loss':weighted/len(items),'post_epoch_mean_loss':post,'lossless':True})
        if phase=='full_train' or epoch in (1,epochs):print(phase,epoch,'loss',post,'updates',budget.steps,flush=True)
    return {'initial_loss':initial,'final_loss':history[-1]['post_epoch_mean_loss'],'history':history,'model_train_mode_executed':True,'product_equal_weight':True}


def process_memory():
    """OS lifetime peak, not saved autograd activation memory; stdlib only."""
    try:
        if sys.platform=='win32':
            import ctypes
            from ctypes import wintypes
            class Counters(ctypes.Structure):
                _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[
                    (name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage',
                    'QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
            kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
            kernel.GetCurrentProcess.restype=wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
            counters=Counters();counters.cb=ctypes.sizeof(counters)
            if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):raise ctypes.WinError(ctypes.get_last_error())
            return {'peak_working_set_bytes':counters.PeakWorkingSetSize,'final_RSS_bytes':counters.WorkingSetSize,'memory_measurement_reason':None}
        import resource
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {'peak_working_set_bytes':int(peak if sys.platform=='darwin' else peak*1024),'final_RSS_bytes':None,
                'memory_measurement_reason':'OS peak only; final resident memory not measured on this platform'}
    except (ImportError,OSError,AttributeError) as exc:
        return {'peak_working_set_bytes':None,'final_RSS_bytes':None,'memory_measurement_reason':type(exc).__name__+': '+str(exc)}


def run_package_acceptance(a):
    validate_options(a);a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'plan.json').exists():raise ValueError('Existing attempt is immutable; do not silently retrain')
    reads,rejections=access_guard(a.train_labels);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if a.device=='cuda':
        if not torch.cuda.is_available():raise ValueError('Requested CUDA device unavailable')
        torch.cuda.reset_peak_memory_stats()
    environment={'os':platform.platform(),'hostname':socket.gethostname(),'python':sys.version,'executable':sys.executable,'torch':str(torch.__version__),
        'device':a.device,'cwd':str(Path.cwd()),'package':str(a.package),'threads':1,'actual_location':'Windows CPU' if os.name=='nt' and a.device=='cpu' else platform.system()+' '+a.device}
    write_json(a.out/'environment.json',environment)
    plan={'stage':'04T1','protocol':a.protocol,'model':{'kind':'rgcn','hidden':16,'layers':2,'feature_schema':FEATURE_SCHEMA,'relations':list(RELATIONS),'dtype':'float32'},
        'optimizer':{'name':'Adam','learning_rate':.001,'class_weights':None,'loss':'mean product BCEWithLogits;FP=1,TP=0'},
        'seed':a.seed,'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'atol':ATOL,'rtol':RTOL,
        'gradient_cases':'largest TRAIN suite above16, smallest TRAIN suite; full identity tie-break; chunk5',
        'micro_selection':'2 FP +2 TP; SHA256(04T1|seed|identity); prefer distinct versions across all4; TRAIN only',
        'budgets':{'updates':a.max_updates,'model_seconds':a.max_model_seconds,'artificial_updates':a.synthetic_steps,'micro_updates':a.micro_steps,'full_epochs':a.epochs},
        'normalization':'fixed semantic vocabulary; no fitted graph statistics; only TRAIN contributors registered',
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in Path(__file__).parent.glob('*.py')},'heldout_prediction_allowed':False}
    write_json(a.out/'plan.json',plan)
    pack,train,assignments,integrity=load_train_inputs(a);write_json(a.out/'input_integrity.json',{'partitions':integrity,'labels_read_before_integrity':False,'zero_loss':True})
    targets=join_targets(a,train);counts=Counter(targets.values());versions={c:len({k[:2] for k,v in targets.items() if v==c}) for c in ('FP','TP')}
    write_json(a.out/'training_target_audit.json',{'source':str(a.train_labels),'source_sha256':a.train_label_sha256,'retained_targets':len(targets),'counts':dict(counts),
        'class_versions':versions,'encoding':{'FP':1,'TP':0},'qualification':'existing unverified local PULP proxy',
        'loader_scope':'Original loader validates grouped/train.csv; only current17 keys retained, no other product observations or targets used for fitting.'})
    conflicts=representation_audit(train,targets);write_json(a.out/'representation_conflicts.json',conflicts)
    budget=ModelBudget(a.out,a.max_updates,a.max_model_seconds);result={'status':'RUNNING'}
    try:
        eq=equivalence(pack,train,targets,a,budget);write_json(a.out/'gradient_checks.json',eq)
        # A failed streaming comparison does not authorize it or block correct whole-product training.
        toy,toy_targets=artificial_inputs(a.out);model=new_model(a.seed,a.device);opt=optimizer_for(model)
        synthetic=fit_fixed(model,opt,toy,toy_targets,budget,'synthetic',a.synthetic_steps,2,a.seed)
        synthetic['heldout_target_files_absent']=True;write_json(a.out/'synthetic.json',synthetic)
        ranked=sorted(train,key=lambda p:hashlib.sha256(('04T1|'+str(a.seed)+'|'+'|'.join(p.identity)).encode()).hexdigest())
        micro=[];used=set()
        for label in ('FP','TP'):
            candidates=[p for p in ranked if targets[p.identity]==label]
            for _ in range(2):
                p=next((p for p in candidates if p.identity[:2] not in used),candidates[0]);micro.append(p);used.add(p.identity[:2]);candidates.remove(p)
        write_json(a.out/'micro_selection.json',{'identities':[list(p.identity) for p in sorted(micro,key=lambda p:p.identity)],'count':len(micro),'versions':len(used),'no_individual_targets':True})
        model=new_model(a.seed,a.device,train,assignments);opt=optimizer_for(model)
        miniature=fit_fixed(model,opt,micro,targets,budget,'real_micro',a.micro_steps,4,a.seed);write_json(a.out/'micro_fit.json',miniature)
        contract={'manifest_sha256':a.manifest_sha256,'feature_schema':FEATURE_SCHEMA,'relations':list(RELATIONS),'protocol':a.protocol,
            'target_sha256':a.train_label_sha256,'model':plan['model'],'optimizer':plan['optimizer'],'seed':a.seed}
        resume=resume_check(model,opt,train,targets,contract,a,budget);write_json(a.out/'resume_checks.json',resume)
        # Fresh declared initialization, not the micro-fit checkpoint.
        model=new_model(a.seed,a.device,train,assignments);opt=optimizer_for(model)
        short=fit_fixed(model,opt,train,targets,budget,'full_train',a.epochs,a.batch_products,a.seed);write_json(a.out/'short_run.json',short)
        checkpoint=save_checkpoint(a.out/'engineering_train_only.pt',model,opt,contract,budget.steps)
        result={'status':'COMPLETE','actual_updates':budget.steps,'model_window_seconds':time.monotonic()-budget.start,'training_entry_verified':True,
            'backward_verified':all(eq['nonzero_module_evidence'].values()),'optimizer_update_verified':any(r['parameter_max_abs_change']>0 for r in budget.rows),'train_only_smoke_completed':True,'checkpoint_resume_verified':resume['passed'],
            'whole_vs_chunked_verified':eq['passed'],'checkpoint_sha256':checkpoint,'artificial_loss_decreased':synthetic['final_loss']<synthetic['initial_loss'],
            'training_products':len(train),'training_graphs':sum(len(p.tests) for p in train),'training_versions':len({p.identity[:2] for p in train}),
            'heldout_evaluation_performed':False,'world_model_trained':False,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if a.device=='cuda' else None,
            'gpu_measurement_reason':None if a.device=='cuda' else 'CPU only; no GPU execution'}
    except BudgetStop as e:result={'status':'PARTIAL_BUDGET_STOP','reason':str(e),'actual_updates':budget.steps,'model_window_seconds':time.monotonic()-budget.start}
    except BaseException as e:
        result={'status':'FAILED','error_type':type(e).__name__,'reason':str(e),'actual_updates':budget.steps,'model_window_seconds':time.monotonic()-budget.start};raise
    finally:
        write_json(a.out/'result.json',result)
        write_json(a.out/'target_access.json',{'reads':dict(reads),'blocked_attempts':rejections,'only_current_protocol_train_allowed':True,'OS_security_boundary':False})
        write_json(a.out/'self_resources.json',{'pid':os.getpid(),'scope':'actual tensor worker; OS lifetime peak working set',
            **process_memory(),'model_window_seconds':time.monotonic()-budget.start})
    print('ENGINEERING_RESULT',json.dumps(result),flush=True)
    return 0 if result['status']=='COMPLETE' else 2
