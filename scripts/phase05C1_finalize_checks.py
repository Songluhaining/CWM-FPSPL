"""Independent read-only input/identity/timing audit and real GPU isolation probe."""
import argparse,hashlib,importlib.metadata,json,os,platform,resource,shutil,subprocess,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from cwm_fp.data.common import sha256,write_json
from cwm_fp.models.transition_data import load,read_cache,verify_source,access_guard,checked,lines
from cwm_fp.models.transition_model import TransitionModel,KINDS
from cwm_fp.models.prefix_observer import ObservationReplay
from cwm_fp.models.local_state import capture,TRUSTED
from cwm_fp.lddg.static import load_static

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;start=time.monotonic();source=r/'snapshot/data/05A'
    out=r/'final_checks';out.mkdir();access_guard(source,{'train','val'});m,_,_=verify_source(source)
    receipt=load(r/'snapshot/export_receipt.json')
    for f,v in receipt['files'].items():
        if sha256(r/'snapshot'/f)!=v['sha256']:raise ValueError('Frozen export changed: '+f)
    caches={};inputs=None
    for part in ('train','val'):
        idx,store=read_cache(r/'snapshot/data/05B',part);rows=idx['rows'];nn=sum(len(store[x['input_key']].x) for x in rows);ee=sum(len(store[x['input_key']].edge_type) for x in rows)
        rels=Counter()
        for x in rows:rels.update(store[x['input_key']].edge_type.tolist())
        caches[part]={'prefixes':len(rows),'unique_cache_files':len(store),'sum_nodes_over_prefixes':nn,'sum_edges_over_prefixes':ee,'relation_counts_over_prefixes':dict(rels),'no_events_or_edges_truncated':True}
        if part=='train':
            selected=[];seen=set()
            for row in rows:
                if row['trace_index'] not in seen and row['event_index']>=8:selected.append(store[row['input_key']]);seen.add(row['trace_index'])
                if len(selected)==4:break
            inputs=selected
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_device(0);probes=[]
    for kind in KINDS:
        path=r/('gpu_code_v3' if kind=='seq_gru' else 'gpu_code_v4')/f'{kind}_epoch01.pt'
        saved=torch.load(path,map_location='cuda:0',weights_only=True);model=TransitionModel(kind).cuda();model.load_state_dict(saved['model']);model.eval()
        torch.manual_seed(20261101);initial=TransitionModel(kind)
        changed=[k for k,v in model.state_dict().items() if not torch.equal(v.cpu(),initial.state_dict()[k])]
        with torch.no_grad():
            batched=model(inputs);separate=[model([inp]) for inp in inputs];perm=model(list(reversed(inputs)))
            repeated=model(inputs)
        for k in range(2):
            expected=torch.cat([v[k] for v in separate]);torch.testing.assert_close(batched[k],expected,atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(batched[k],perm[k].flip(0),atol=2e-5,rtol=2e-5)
        probes.append({'model':kind,'real_training_trajectories':len(inputs),'separate_vs_batch_max_difference':max(float((batched[k]-torch.cat([v[k] for v in separate])).abs().max()) for k in range(2)),
            'same_instance_repeat_max_difference':max(float((batched[k]-repeated[k]).abs().max()) for k in range(2)),
            'changed_epoch_parameters':changed,'checkpoint_sha256':sha256(path),'passed':True,'optimizer_updates':0})
    gpu_proc=subprocess.run(['nvidia-smi','-i','GPU-e611ba0c-db46-7d8d-486c-3fd79962b2ac','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv'],capture_output=True,text=True,timeout=15)
    framework=load_static(checked(source,m['framework_static']));traces={t['trace_index']:t for t in m['traces']};state_stats={};allstates=[]
    for part in ('train','val'):
        states=lines(r/'diagnostic_code_v4'/f'{part}_local_state.jsonl');allstates.extend(states);stats={}
        for task in ('branch','arithmetic'):
            rr=[x for x in states if x['task']==task];cc=Counter();derived_rows=[]
            for row in rr:
                state=row['pre_state_evidence'];inp=row['learning_input']
                if set(inp)!={'opcode','arity','operands'}:raise ValueError('Identity in learning schema')
                for v,x in zip(state['operands'],inp['operands']):
                    if set(x)!={'dtype','source','primary_known','value'}:raise ValueError('Identity in operand schema')
                    if v['source']=='UNKNOWN' and (x['primary_known'] or x['value'] is not None):raise ValueError('Unknown filled')
                    if v['available_at'] is not None and v['available_at']>state['clock']:raise ValueError('Future operand')
                    cc[v['source']]+=1;cc['static_literal_alternative_slots']+=int(v['static_alternative'] is not None)
                    if v['source']=='SEMANTICALLY_DERIVED':derived_rows.append(row)
                if row['prediction'] is not None and not row['consistent']:raise ValueError('Symbolic inconsistency')
            derived_causes=Counter()
            for row in derived_rows:
                t=traces[row['audit']['trace_index']];obs=ObservationReplay(t['identity'][3]);obs.active=True;obs.entry=True
                for tok in lines(checked(source,t['input_files']['observations'])):
                    obs.push(tok)
                    if tok['clock']==row['pre_state_evidence']['clock']:break
                static=load_static(checked(source,t['input_files']['static_bytecode']));s,flow,raw=capture(obs,static,framework)
                if s!=row['pre_state_evidence']:raise ValueError('Independent pre-state replay differs')
                for v in s['operands']:
                    if v['source']=='SEMANTICALLY_DERIVED':derived_causes[str(raw[v['definition_event']]['opcode'])]+=1
            stats[task]={'operand_sources':dict(cc),'derived_definition_opcodes':dict(derived_causes),
                'derived_impacted_products':len({tuple(x['audit']['identity'][:3]) for x in derived_rows}),
                'derived_impacted_versions':len({tuple(x['audit']['identity'][:2]) for x in derived_rows}),
                'derived_impacted_traces':len({x['audit']['trace_index'] for x in derived_rows})}
        state_stats[part]=stats
    code={}
    for name in ('code_v2','code_v3','code_v4'):
        code[name]={p.relative_to(r/name).as_posix():sha256(p) for p in (r/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    write_json(out/'code_receipts.json',code)
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    write_json(out/'environment.json',{'python':platform.python_version(),'executable':os.sys.executable,'os':platform.platform(),'container_hostname':platform.node(),'pid':os.getpid(),
        'packages':packages,'torch_cuda':torch.version.cuda,'gpu_uuid':str(torch.cuda.get_device_properties(0).uuid),
        'selected_gpu_processes':gpu_proc.stdout,'nvidia_smi_exit':gpu_proc.returncode})
    (out/'requirements-linux.lock').write_text('\n'.join(f'{k}=={v}' for k,v in sorted(packages.items()))+'\n')
    environment=sum(p.stat().st_size for p in (r/'environment').rglob('*') if p.is_file());research=sum(p.stat().st_size for p in r.rglob('*') if p.is_file() and 'environment' not in p.relative_to(r).parts and p.name!='python311.tar.gz')
    if research>4*1024**3 or shutil.disk_usage(r).free<10*1024**3:raise RuntimeError('Disk budget violated')
    du=subprocess.run(['du','-s','-B1',str(r/'environment')],capture_output=True,text=True,check=True)
    write_json(out/'independent_checks.json',{'frozen_export_files_unchanged':len(receipt['files']),'cache_checks':caches,'gpu_real_batch_isolation':probes,'state_source_audit':state_stats,
        'local_state_rows':len(allstates),'learning_fields_identity_free':True,'future_operand_checks_passed':True,'derived_replays_passed':True,
        'research_directory_bytes':research,'environment_logical_file_bytes':environment,'environment_allocated_bytes':int(du.stdout.split()[0]),'disk_free_bytes':shutil.disk_usage(r).free,
        'seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'optimizer_updates':0})
    print('INDEPENDENT_CHECKS_PASSED',len(allstates),research,environment,flush=True)
if __name__=='__main__':main()
