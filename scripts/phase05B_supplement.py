"""Postfit measurement only: exact lookup hits and actual active gradients.

No checkpoint choice, optimizer update, feature change or new fitting.
"""
import sys,time,json,traceback,csv
from pathlib import Path
import numpy as np
import torch
from cwm_fp.data.common import ROOT,write_json,write_csv,sha256
from cwm_fp.models.transition_budget import Budget
from cwm_fp.models.transition_data import load,read_targets,access_guard,hierarchy_weights
from cwm_fp.models.transition_train import verify_protocol,dataset,target_arrays,SEEDS
from cwm_fp.models.transition_model import TransitionModel,KINDS,weighted_loss
from cwm_fp.models.transition_evaluate import fit_baselines,baseline_predict,grouped_metric,bin_metrics

out=Path(sys.argv[1]);source=Path(sys.argv[2]);torch.set_num_threads(1);budget=Budget(out);error=None;code=0
try:
    access_guard(source,{'train','val'});plan,psha=verify_protocol(out,'cpu')
    idx={p:load(out/'cache'/f'index_{p}.json') for p in ('train','val')};tt={p:read_targets(out,i) for p,i in idx.items()}
    fit=fit_baselines(idx['train']['rows'],tt['train']);diag=load(ROOT/'outputs/phase05B_shortcut_diagnostics.json');metrics=load(out/'metrics.json')
    diag['exact_lookup_target_hit_coverage']={};before_hash=sha256(out/'metrics.json');added=[]
    for kind in KINDS:
        diag['exact_lookup_target_hit_coverage'][kind]={}
        for part in ('train','val'):
            rows=idx[part]['rows'];pa,pb=baseline_predict(rows,fit,'Exact-'+kind);q={}
            for task in ('A','B'):
                valid=np.array([t[task]>=0 for t in tt[part]]);hit=np.array([r['signatures'][kind] in fit[3][kind][task] for r in rows])
                q[task]={'valid':int(valid.sum()),'hit':int((hit&valid).sum()),'miss':int((~hit&valid).sum()),'hit_coverage':float((hit&valid).sum()/valid.sum())}
                for subset,mask in (('lookup_hit',hit),('lookup_miss',~hit)):
                    micro,macro=grouped_metric(rows,tt[part],pa,pb,mask,task)
                    for agg,mm in (('event_micro',micro),('version_macro',macro)):
                        added.append({'model':'Exact-'+kind,'seed':None,'epoch':None,'partition':part,'subset':subset,'stratum':'all','task':task,'aggregation':agg,**mm})
            diag['exact_lookup_target_hit_coverage'][kind][part]=q
    # Inspect fixed first-seed selected parameters on every TRAIN prefix. Union
    # elementwise gradients over microbatches; never take an optimizer step.
    index,inputs,_=dataset(out,'train');exact_weights=hierarchy_weights(index['rows'],tt['train']);weights=torch.tensor(exact_weights,dtype=torch.float32);y=target_arrays(tt['train'],'cpu')
    np.savez_compressed(out/'train_supervision_weights.npz',weights=exact_weights)
    weight_summary={}
    for j,name in enumerate(('A_covered','A_abstain','B')):
        selected=[i for i in range(len(index['rows'])) if exact_weights[i,j]>0]
        weight_summary[name]={'positions':len(selected),'traces':len({index['rows'][i]['trace_index'] for i in selected}),
            'products':len({tuple(index['rows'][i]['identity'][:3]) for i in selected}),'versions':len({tuple(index['rows'][i]['identity'][:2]) for i in selected}),
            'sum':float(exact_weights[:,j].sum()),'row_mapping':'cache/index_train.json rows; float64 theoretical weights cast to float32 by fixed training'}
    write_json(out/'weight_hierarchy_summary.json',weight_summary)
    evaluation=load(out/'evaluation.json');active=[]
    for kind in KINDS:
        selected=next(r for r in evaluation['selection'] if r['model']==kind and r['seed']==SEEDS[0]);model=TransitionModel(kind)
        ckpt=torch.load(out/'runs'/f'{kind}_{SEEDS[0]}'/f"epoch{selected['selected_epoch']:02d}.pt",weights_only=True,map_location='cpu')
        model.load_state_dict(ckpt['model']);model.eval();union={n:torch.zeros_like(p,dtype=torch.bool) for n,p in model.named_parameters()}
        start=time.perf_counter()
        for j in range(0,len(inputs),64):
            budget();model.zero_grad(set_to_none=True);loss,_=weighted_loss(model(inputs[j:j+64]),y[j:j+64],weights[j:j+64]);loss.backward()
            for n,p in model.named_parameters():
                if p.grad is not None:union[n]|=p.grad!=0
        active.append({'model':kind,'seed':SEEDS[0],'epoch':selected['selected_epoch'],'all_train_prefixes':len(inputs),'optimizer_updates':0,
            'actual_nonzero_gradient_parameter_union':sum(int(v.sum()) for v in union.values()),'per_parameter':{k:int(v.sum()) for k,v in union.items()},
            'seconds':time.perf_counter()-start,'interpretation':'measured at this fixed checkpoint, not every optimizer step or a guaranteed structural parameter count'})
    metrics+=added;write_json(out/'metrics.json',metrics)
    write_csv(ROOT/'outputs/phase05B_metrics.csv',metrics,list(dict.fromkeys(k for r in metrics for k in r)))
    write_json(ROOT/'outputs/phase05B_shortcut_diagnostics.json',diag)
    with (out/'branch_group_metrics.csv').open(encoding='utf8',newline='') as f:branch_rows=list(csv.DictReader(f))
    configurations={tuple(r[k] for k in ('model','seed','epoch','partition')) for r in branch_rows};existing={tuple(r[k] for k in ('model','seed','epoch','partition','level','identity')) for r in branch_rows};empty_added=0
    for model,seed,epoch,part in sorted(configurations):
        for level,width in (('version',2),('product',3),('trajectory',4)):
            identities={tuple(r['identity'][:width]) for r in idx[part]['rows']}
            for identity in sorted(identities):
                key=(model,seed,epoch,part,level,json.dumps(identity,ensure_ascii=False))
                if key in existing:continue
                if any(t['B']>=0 for r,t in zip(idx[part]['rows'],tt[part]) if tuple(r['identity'][:width])==identity):raise ValueError('Missing nonempty branch group')
                branch_rows.append(dict(zip(('model','seed','epoch','partition','level','identity'),key))|bin_metrics([],np.array([])));empty_added+=1
    write_csv(out/'branch_group_metrics.csv',branch_rows,list(branch_rows[0]))
    write_json(out/'postfit_measurements.json',{'active_parameters':active,'metrics_before_sha256':before_hash,'added_lookup_rows':len(added),
        'empty_branch_groups_retained':empty_added,'no_parameter_update':True,'no_selection_change':True,'script_sha256':sha256(Path(__file__))})
    print('POSTFIT',active,flush=True)
except Exception as exc:
    code=1;error=str(exc);traceback.print_exc()
finally:budget.finish([sys.executable,*sys.argv],code,error)
raise SystemExit(code)
