"""Fresh fixed-seed training with trajectory-batch accumulated gradients."""
from __future__ import annotations
from collections import defaultdict
import copy,json,math,os,sys,time
from pathlib import Path
import numpy as np
import torch
from cwm_fp.data.common import ROOT,sha256,write_json,write_csv
from .transition_model import TransitionModel,KINDS,parameter_report,weighted_loss
from .transition_data import read_cache,read_targets,hierarchy_weights,load,PREFIX_SHA,PLAN05A_SHA,GRAPH_SHA,VOCAB

SEEDS=(20261101,20261102,20261103)

def dataset(out,part):
    start=time.monotonic();index,store=read_cache(out,part)
    return index,[store[r['input_key']] for r in index['rows']],time.monotonic()-start

def target_arrays(targets,device):return torch.tensor([[t['A'],t['B']] for t in targets],dtype=torch.float32,device=device)

def trajectory_batches(rows,rng):
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['trace_index']].append(i)
    ids=sorted(groups);rng.shuffle(ids)
    return [[i for k in ids[n:n+4] for i in groups[k]] for n in range(0,len(ids),4)]

def loss_batch(model,opt,inputs,ys,weights,ids,micro,scale,budget):
    opt.zero_grad(set_to_none=True);parts=np.zeros(3);forward=backward=0.
    for j in range(0,len(ids),micro):
        indices=ids[j:j+micro];tick=time.monotonic();pred=model([inputs[i] for i in indices]);loss,ps=weighted_loss(pred,ys[indices],weights[indices],scale)
        forward+=time.monotonic()-tick;tick=time.monotonic();loss.backward();backward+=time.monotonic()-tick;parts+=ps.cpu().numpy()
    opt.step();budget.update()
    return parts,forward,backward

def predict(model,inputs,micro,budget):
    model.eval();aa=[];bb=[]
    with torch.no_grad():
        for j in range(0,len(inputs),micro):
            budget();a,b=model(inputs[j:j+micro]);aa.append(torch.softmax(a,1).cpu().numpy());bb.append(torch.sigmoid(b).cpu().numpy())
    return np.concatenate(aa),np.concatenate(bb)

def preflight(a,budget):
    index,inputs,read_seconds=dataset(a.out,'train');targets=read_targets(a.out,index);weights=hierarchy_weights(index['rows'],targets)
    batches=trajectory_batches(index['rows'],np.random.default_rng(SEEDS[0]));ids=batches[0]
    y=target_arrays(targets,a.device);w=torch.as_tensor(weights,dtype=torch.float32,device=a.device);records=[]
    for kind in KINDS:
        budget();torch.manual_seed(SEEDS[0]);model=TransitionModel(kind).to(a.device);opt=torch.optim.Adam(model.parameters(),lr=.001)
        before={k:v.detach().clone() for k,v in model.state_dict().items()};tick=time.monotonic()
        parts,forward,backward=loss_batch(model,opt,inputs,y,w,ids,64,len(batches),budget)
        grad={k:float(p.grad.norm()) if p.grad is not None else None for k,p in model.named_parameters()}
        changed=[k for k,v in model.state_dict().items() if not torch.equal(v,before[k])]
        required=['gru.weight_ih_l0','gru.weight_hh_l0'] if kind=='seq_gru' else ['encoder.input.weight','encoder.layers.0.bases','encoder.layers.1.bases']
        required+=['head_a.weight','head_b.weight']
        if not all(grad[k] is not None and grad[k]>0 and k in changed for k in required):raise ValueError('Disconnected gradient/update')
        model.eval();probe=inputs[:3]
        with torch.no_grad():expected=model(probe)
        checkpoint=a.out/'engineering'/f'{kind}.pt';checkpoint.parent.mkdir(exist_ok=True)
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict()},checkpoint)
        restored=TransitionModel(kind).to(a.device);saved=torch.load(checkpoint,map_location=a.device,weights_only=True);restored.load_state_dict(saved['model']);restored.eval()
        with torch.no_grad():actual=restored(probe)
        assert all(torch.equal(x,z) for x,z in zip(expected,actual))
        sharing=None
        if kind!='seq_gru':
            comparison=[inputs[i] for i in ids[:8]];opt.zero_grad(set_to_none=True);z=model(comparison);sum(t.sum() for t in z).backward();g1={k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}
            opt.zero_grad(set_to_none=True);zz=model(comparison,share_static=False);sum(t.sum() for t in zz).backward()
            errors=[float((g1[k]-p.grad).abs().max()) for k,p in model.named_parameters() if p.grad is not None]
            for k,p in model.named_parameters():
                if p.grad is not None:torch.testing.assert_close(g1[k],p.grad,atol=2e-5,rtol=2e-5)
            for x,z in zip(z,zz):torch.testing.assert_close(x,z,atol=2e-6,rtol=2e-6)
            sharing={'max_gradient_absolute_difference':max(errors),'equivalent':True}
        records.append({'model':kind,'parameters':parameter_report(model),'train_traces_in_batch':len({index['rows'][i]['trace_index'] for i in ids}),
            'prefixes_in_batch':len(ids),'microbatch':64,'forward_seconds':forward,'backward_seconds':backward,'seconds':time.monotonic()-tick,
            'gradients':grad,'changed_parameters':changed,'save_restore_exact':True,'shared_static_equivalence':sharing})
    result={'read_seconds':read_seconds,'records':records,'weight_sums':weights.sum(0).tolist(),'updates':len(KINDS),'microbatch':64,'only_training_used':True}
    write_json(a.out/'preflight.json',result);print(json.dumps(result),flush=True)

def verify_protocol(out,device):
    path=ROOT/'outputs/phase05B_protocol.json';p=load(path);receipt=load(out/'protocol_receipt.json')
    if sha256(path)!=receipt['sha256'] or p['device']!=device:raise ValueError('Frozen protocol/device changed')
    for rel,digest in p['code_sha256'].items():
        if sha256(ROOT/rel)!=digest:raise ValueError('Frozen source changed: '+rel)
    if sha256(out/'cache/index_train.json')!=p['cache_train_index_sha256']:raise ValueError('Training cache changed')
    if (out/'cache_val_receipt.json').exists() and sha256(out/'cache/index_val.json')!=load(out/'cache_val_receipt.json')['index_sha256']:raise ValueError('Validation cache changed')
    return p,receipt['sha256']

def train(a,budget):
    protocol,plan_sha=verify_protocol(a.out,a.device)
    # The training process may access validation inputs, never validation targets.
    def heldout_guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and str(Path(os.fsdecode(args[0])).resolve()).lower()==str((a.out/'cache/targets_val.json').resolve()).lower():raise PermissionError('Evaluator only: validation targets')
    sys.addaudithook(heldout_guard)
    train_index,inputs,read_seconds=dataset(a.out,'train');val_index,val_inputs,val_seconds=dataset(a.out,'val')
    targets=read_targets(a.out,train_index);weights=hierarchy_weights(train_index['rows'],targets)
    y=target_arrays(targets,a.device);w=torch.as_tensor(weights,dtype=torch.float32,device=a.device)
    curves=[];completed=[];micro=protocol['prefix_microbatch'];start=time.monotonic()
    runs=a.out/'runs';runs.mkdir(exist_ok=True)
    try:
        for kind in KINDS:
            for seed in SEEDS:
                budget();run=runs/f'{kind}_{seed}'
                if run.exists():raise ValueError('Immutable training run already exists')
                run.mkdir();torch.manual_seed(seed);np_rng=np.random.default_rng(seed)
                model=TransitionModel(kind).to(a.device);opt=torch.optim.Adam(model.parameters(),lr=.001);updates=0
                for epoch in range(1,21):
                    model.train();tick=time.monotonic();parts=np.zeros(3);forward=backward=0.;batches=trajectory_batches(train_index['rows'],np_rng);visited=[]
                    for ids in batches:
                        budget();ps,fw,bw=loss_batch(model,opt,inputs,y,w,ids,micro,len(batches),budget);parts+=ps;forward+=fw;backward+=bw;updates+=1;visited.extend(ids)
                    if sorted(visited)!=list(range(len(inputs))):raise ValueError('Epoch omitted/repeated samples')
                    curve={'model':kind,'seed':seed,'epoch':epoch,'prefixes':len(visited),'traces':len({r['trace_index'] for r in train_index['rows']}),'updates':len(batches),
                        'weighted_A_covered':parts[0],'weighted_A_abstain':parts[1],'weighted_B':parts[2],'objective':sum(parts)/3,
                        'forward_seconds':forward,'backward_seconds':backward,'train_seconds':time.monotonic()-tick}
                    curves.append(curve);write_json(run/'learning_curves.json',[r for r in curves if r['model']==kind and r['seed']==seed])
                    print('EPOCH',kind,seed,epoch,'loss',round(sum(parts)/3,6),'seconds',round(time.monotonic()-tick,2),flush=True)
                    if epoch in (5,10,20):
                        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'updates':updates,'protocol_sha256':plan_sha,
                            'rng_torch':torch.get_rng_state(),'rng_numpy':np_rng.bit_generator.state,'model_kind':kind,'seed':seed},run/f'epoch{epoch:02d}.pt')
                        for partition,xx in (('train',inputs),('val',val_inputs)):
                            st=time.monotonic();pa,pb=predict(model,xx,micro,budget);np.savez_compressed(run/f'epoch{epoch:02d}_{partition}.npz',A=pa,B=pb)
                            curve[partition+'_prediction_seconds']=time.monotonic()-st
                completed.append({'model':kind,'seed':seed,'epochs':20,'updates':updates,'parameters':parameter_report(model)})
                write_json(a.out/'training_progress.json',{'completed':completed,'read_seconds':read_seconds+val_seconds,'seconds':time.monotonic()-start})
    except Exception:
        if 'model' in locals():torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch_incomplete':epoch,'updates':updates,'protocol_sha256':plan_sha},run/'interrupted.pt')
        raise
    finally:
        write_json(a.out/'learning_curves.json',curves)
        write_json(a.out/'training_progress.json',{'completed':completed,'read_seconds':read_seconds+val_seconds,'seconds':time.monotonic()-start,
            'no_validation_targets_read':True,'all_epochs_full_traversal':all(r['prefixes']==len(inputs) for r in curves)})
