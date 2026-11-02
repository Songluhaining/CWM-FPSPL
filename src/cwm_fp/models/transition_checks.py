"""Fixed-model suffix, batch and public-prefix equivalence checks."""
from collections import defaultdict
import time
import numpy as np
import torch
from cwm_fp.data.common import write_json
from cwm_fp.lddg.static import load_static
from .prefix_observer import ObservationReplay,PrefixBuilder
from .transition_data import load,lines,checked,verify_source,read_cache
from .transition_model import TransitionModel,from_snapshot,signature,KINDS
from .transition_train import SEEDS,verify_protocol,predict

def replay_to(builder,test_id,observations,clock):
    obs=ObservationReplay(test_id);obs.active=True;obs.entry=True
    for record in observations:
        if record['clock']>clock:break
        obs.push(record)
    return from_snapshot(builder.build(obs))

def isolation(a,budget):
    plan,plan_sha=verify_protocol(a.out,a.device);source,_,_=verify_source(a.source);start=time.monotonic()
    evaluation=load(a.out/'evaluation.json');models={}
    for k in KINDS:
        run=next(r for r in evaluation['selection'] if r['model']==k and r['seed']==SEEDS[0]);m=TransitionModel(k).to(a.device)
        checkpoint=torch.load(a.out/'runs'/f'{k}_{SEEDS[0]}'/f"epoch{run['selected_epoch']:02d}.pt",map_location=a.device,weights_only=True)
        if checkpoint['protocol_sha256']!=plan_sha:raise ValueError('Wrong frozen checkpoint')
        m.load_state_dict(checkpoint['model']);m.eval();models[k]=m
    fw=load_static(checked(a.source,source['framework_static']));checks=[];max_errors={k:0. for k in KINDS};counts={}
    for part in ('train','val'):
        index,store=read_cache(a.out,part);bytrace=defaultdict(list)
        for row in index['rows']:
            if row['critical']:bytrace[row['trace_index']].append(row)
        selected=[];original=[];mutated=[]
        for t in source['traces']:
            if t['partition']!=part:continue
            budget();refs=t['input_files'];static=load_static(checked(a.source,refs['static_bytecode']));bundle=load(checked(a.source,refs['static']))
            builder=PrefixBuilder(static,bundle,fw);observations=lines(checked(a.source,refs['observations']))
            for r in bytrace[t['trace_index']]:
                cut=r['clock'];prefix=[v for v in observations if v['clock']<=cut]
                # Original observations, then a genuinely changed suffix and length.
                # A malformed future instruction would fail if the reader consumed it.
                suffix=[{'clock':[cut[0]+9999,0],'kind':'instruction','text':'FORBIDDEN_FUTURE','cut':-1}]
                x=replay_to(builder,t['identity'][3],prefix,cut);z=replay_to(builder,t['identity'][3],prefix+suffix,cut)
                if signature(x,'rgcn')!=r['input_key'] or signature(z,'rgcn')!=r['input_key']:raise ValueError('Suffix/cache mismatch')
                original.append(x);mutated.append(z);selected.append(r)
                checks.append({'partition':part,'identity':t['identity'],'trace_index':t['trace_index'],'event_index':r['event_index'],
                    'clock':cut,'source_input_sha256':refs['observations']['sha256'],'suffix_deleted_and_replaced':True,'cache_equals_replay':True})
        for kind,model in models.items():
            cached=[store[r['input_key']] for r in selected];aa,bb=predict(model,cached,64,budget);za,zb=predict(model,mutated,64,budget)
            np.testing.assert_allclose(aa,za,atol=2e-6,rtol=2e-6);np.testing.assert_allclose(bb,zb,atol=2e-6,rtol=2e-6)
            # Reverse sample order and change batch boundaries: no padding or
            # shared-static/cross-trajectory contamination may change logits.
            ra,rb=predict(model,list(reversed(original)),31,budget);ra=ra[::-1];rb=rb[::-1]
            np.testing.assert_allclose(aa,ra,atol=2e-6,rtol=2e-6);np.testing.assert_allclose(bb,rb,atol=2e-6,rtol=2e-6)
            # Also check raw logits, not only probabilities.
            errors=[]
            with torch.no_grad():
                for j in range(0,len(original),64):
                    base=model(cached[j:j+64]);future=model(mutated[j:j+64]);independent=model(original[j:j+64])
                    for u,v,w in zip(base,future,independent):
                        torch.testing.assert_close(u,v,atol=2e-6,rtol=2e-6);torch.testing.assert_close(u,w,atol=2e-6,rtol=2e-6)
                        errors.append(float((u-v).abs().max()))
            max_errors[k]=max(max_errors[k],max(errors))
        counts[part]={'traces':len(bytrace),'checked_prefixes':len(selected),'models':len(models),'first_middle_last_all_traces':True}
    write_json(a.out/'model_isolation_checks.json',{'passed':True,'protocol_sha256':plan_sha,'counts':counts,'fixed_model_seeds':SEEDS[0],
        'max_suffix_logit_error':max_errors,'batch_probability_atol':2e-6,'target_dict_passed_to_model':False,'seconds':time.monotonic()-start,'checks':checks})
    print('ISOLATION',counts,'seconds',time.monotonic()-start,flush=True)
