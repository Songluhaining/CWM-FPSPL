"""Hash-bound prefix tensor cache; observations, audit rows and targets separate."""
from __future__ import annotations
from collections import Counter,defaultdict
import json,os,sys,time,hashlib
from pathlib import Path
import numpy as np
from cwm_fp.data.common import ROOT,sha256,write_json
from cwm_fp.baselines.boundaries import install_io_guard
from cwm_fp.lddg.static import load_static
from .prefix_observer import ObservationReplay,PrefixBuilder
from .prefix_reader import load_prefix
from .transition_model import StructureInput,from_snapshot,signature,validate_input,KINDS

PREFIX_SHA='b863c9714eb68caaa54b49f86ad45ef391a81762d169760ef55771a68d7d4a7c'
PLAN05A_SHA='c16b5e92716f27c5d61e703b0e44c1cd10f9d08720104311868e0aa0aeb18aad'
GRAPH_SHA='71770793f4c7a063c084591f5ebf184d21368296895b3ebe9e3e4d77a7d41289'
VOCAB=('constant','load','store','arithmetic','branch','call','return','field','other')

def load(path):return json.loads(Path(path).read_text(encoding='utf8'))
def lines(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf8').splitlines() if s]

def access_guard(source,allowed_parts):
    install_io_guard('phase05B_no_product_labels');m=load(source/'prefix_manifest.json');access=[]
    protected={str((source/t['targets']['path']).resolve()):t['partition'] for t in m['traces']}
    disallowed={str((source/ref['path']).resolve()) for t in m['traces'] if t['partition'] not in allowed_parts for ref in t['input_files'].values()}
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=str(Path(os.fsdecode(args[0])).resolve());low=p.lower().replace('\\','/')
        if '/labels/' in low:raise PermissionError('FP/TP targets prohibited')
        if p in disallowed:raise PermissionError('Trajectory partition not authorized')
        if p in protected:
            if protected[p] not in allowed_parts:raise PermissionError('Transition target partition not authorized')
            access.append({'path':p,'partition':protected[p]})
    sys.addaudithook(hook);return access

def verify_source(source):
    if sha256(source/'prefix_manifest.json')!=PREFIX_SHA:raise ValueError('Untrusted05A manifest')
    if sha256(source/'final_audit.json')!='e450af903c40547ca22eea33d42dafb653f352f31fe115df3a54a905efd8da92':raise ValueError('Independent05A audit anchor mismatch')
    m=load(source/'prefix_manifest.json')
    if m['protocol_sha256']!=PLAN05A_SHA or m['source_manifest_sha256']!=GRAPH_SHA:raise ValueError('Source lineage mismatch')
    receipt=load(source/'protocol_receipt.json');audit=load(source/'final_audit.json')
    for name in ('phase05A_protocol.json','phase05A_review_packet.json'):
        if sha256(ROOT/'outputs'/name)!=audit['outputs'][name]:raise ValueError('Independent repository05A receipt mismatch')
    if receipt['sha256']!=PLAN05A_SHA or audit['outputs']['phase05A_protocol.json']!=PLAN05A_SHA:raise ValueError('Independent05A receipt mismatch')
    index=source/'artifact_index.json'
    if sha256(index)!=audit['artifact_index_sha256']:raise ValueError('Independent artifact index changed')
    evidence=load(index)
    if evidence['prefix_manifest.json']['sha256']!=PREFIX_SHA:raise ValueError('Independent prefix hash mismatch')
    groups={};counts={};seen=set()
    for part in ('train','val'):
        ts=[t for t in m['traces'] if t['partition']==part]
        counts[part]={'traces':len(ts),'products':len({tuple(t['identity'][:3]) for t in ts}),'versions':len({tuple(t['identity'][:2]) for t in ts}),'prefixes':sum(t['prefixes'] for t in ts)}
        for t in ts:
            if tuple(t['identity']) in seen:raise ValueError('Repeated source trace')
            seen.add(tuple(t['identity']))
            if groups.setdefault(tuple(t['identity'][:2]),part)!=part:raise ValueError('Version split overlap')
    if any(t['partition'] not in ('train','val') for t in m['traces']):raise ValueError('Test trajectories forbidden')
    return m,counts,evidence

def checked(source,ref):
    p=(source/ref['path']).resolve()
    if not p.is_relative_to(source.resolve()) or sha256(p)!=ref['sha256']:raise ValueError('Source input hash/path mismatch')
    return p

def cache_partition(source,out,part,budget,trace_limit=None):
    m,counts,evidence=verify_source(source);dest=out/'cache';(dest/'inputs').mkdir(parents=True,exist_ok=True)
    target_path=dest/f'targets_{part}.json';index_path=dest/f'index_{part}.json'
    if index_path.exists():raise ValueError('Cache already sealed; explicit reuse via saved index only')
    rows=[];target_rows=[];files={};relations=Counter();node_sizes=[];edge_sizes=[];verified=0;start=time.monotonic()
    framework=load_static(checked(source,m['framework_static']))
    # Post-hoc strata derived only from this partition's replay below.
    exceptional=set()
    selected=[t for t in m['traces'] if t['partition']==part]
    if trace_limit is not None:selected=selected[:trace_limit]
    for t in selected:
        budget();begin=len(rows);refs=t['input_files'];static=load_static(checked(source,refs['static_bytecode']));bundle=load(checked(source,refs['static']))
        observer=ObservationReplay(t['identity'][3]);observer.active=True;observer.entry=True
        builder=PrefixBuilder(static,bundle,framework);idx=lines(checked(source,refs['prefix_index']));byclock={tuple(r['clock']):r for r in idx}
        # Targets loaded separately, never passed to the observer or model.
        tr=lines(checked(source,t['targets']));byevent={r['event_index']:r for r in tr}
        if len(tr)!=len(idx) or len(idx)!=t['prefixes']:raise ValueError('Target/index completeness mismatch')
        with checked(source,refs['observations']).open(encoding='utf8') as f:
            for line in f:
                obs=json.loads(line);observer.push(obs);item=byclock.get(tuple(obs['clock']))
                if item is None:continue
                snap=builder.build(observer)
                if snap.fingerprint()!=item['input_sha256']:raise ValueError('Incremental prefix differs from05A')
                inp=from_snapshot(snap);sigs={k:signature(inp,k) for k in KINDS};name=sigs['rgcn']
                file=dest/'inputs'/(name+'.npz')
                if not file.exists():np.savez_compressed(file,x=inp.x,edge_index=inp.edge_index,edge_type=inp.edge_type,anchor=np.asarray(inp.anchor,np.int64))
                files[name]={'path':str(file.relative_to(out).as_posix()),'sha256':sha256(file)}
                target=byevent[item['event_index']]
                if target['anchor']!=item['clock'] or target['partition']!=part:raise ValueError('Wrong target ownership')
                category=int(np.argmax(inp.x[inp.anchor,5:14]));a=VOCAB.index(target['A']) if target['A_mask'] else -1;b=int(target['B']=='jump') if target['B_mask'] else -1
                audit={'sample_index':len(rows),'trace_index':t['trace_index'],'event_index':item['event_index'],'identity':t['identity'],
                    'partition':part,'input_key':name,'signatures':sigs,'category':category,'clock':item['clock'],
                    'critical':item['event_index'] in (0,len(idx)//2,len(idx)-1) or target['interval_kind'] in ('call','return','conditional') or (t['trace_index'],item['event_index']) in exceptional}
                rows.append(audit)
                target_rows.append({'A':a,'B':b,'covered':target['deterministic_prediction'] is not None,
                    'static_prediction':VOCAB.index(target['deterministic_prediction']) if target['deterministic_prediction'] else -1,
                    'boundary':target['interval_kind'],'exception':(t['trace_index'],item['event_index']) in exceptional,
                    'framework_interval':target['intervening_framework_instructions']>0,'branch_point':[t['identity'][:3],target['branch_audit_key']]})
                relations.update(inp.edge_type.tolist());node_sizes.append(len(inp.x));edge_sizes.append(len(inp.edge_type))
                if item['event_index'] in (0,len(idx)//2,len(idx)-1):
                    public=load_prefix(source,t['trace_index'],item['event_index'],expected_manifest_sha256=PREFIX_SHA)
                    assert signature(from_snapshot(public),'rgcn')==name;verified+=1
        transfers={r['source_event'] for r in snap.audit['exception_transfers']}
        for j in range(begin,len(rows)):
            if rows[j]['event_index'] in transfers:
                rows[j]['critical']=True;target_rows[j]['exception']=True
        if len(rows)%500<100:print('CACHE',part,'samples',len(rows),flush=True)
    stats={'partition':part,'source_counts':counts[part],'actual_traces':len(selected),'prefixes':len(rows),'A_valid':sum(t['A']>=0 for t in target_rows),'B_valid':sum(t['B']>=0 for t in target_rows),
        'unique_inputs':len(files),'public_reader_checks':verified,'node_distribution':{'min':min(node_sizes),'max':max(node_sizes),'mean':sum(node_sizes)/len(rows)},
        'edge_distribution':{'min':min(edge_sizes),'max':max(edge_sizes),'mean':sum(edge_sizes)/len(rows)},
        'relation_counts_over_prefix_snapshots':dict(relations),'seconds':time.monotonic()-start,'scalar_features':False}
    write_json(target_path,target_rows);write_json(index_path,{'schema':'structural_transition_cache_v1','source_manifest_sha256':PREFIX_SHA,'rows':rows,'files':files,'stats':stats,
        'targets':{'path':target_path.relative_to(out).as_posix(),'sha256':sha256(target_path)}})
    return load(index_path)

def read_cache(out,part):
    index=load(out/'cache'/f'index_{part}.json');inputs={}
    for key,ref in index['files'].items():
        p=out/ref['path']
        if sha256(p)!=ref['sha256']:raise ValueError('Corrupt tensor cache')
        with np.load(p,allow_pickle=False) as z:
            if set(z.files)!={'x','edge_index','edge_type','anchor'}:raise ValueError('Non-whitelist cache field')
            inp=StructureInput(z['x'].copy(),z['edge_index'].copy(),z['edge_type'].copy(),int(z['anchor']))
        validate_input(inp)
        if signature(inp,'rgcn')!=key:raise ValueError('Canonical cache differs')
        inputs[key]=inp
    return index,inputs

def read_targets(out,index):
    ref=index['targets'];path=out/ref['path']
    if sha256(path)!=ref['sha256']:raise ValueError('Transition target cache changed')
    ts=load(path)
    if len(ts)!=len(index['rows']):raise ValueError('Wrong target count')
    return ts

def hierarchy_weights(rows,targets):
    """position->trace->product->version equal weighting per supervised part."""
    weights=np.zeros((len(rows),3),np.float64)
    for part in range(3):
        group=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
        for i,(r,t) in enumerate(zip(rows,targets)):
            valid=t['B']>=0 if part==2 else t['A']>=0 and bool(t['covered'])==(part==0)
            if not valid:continue
            if r['partition']!='train':raise ValueError('Training weights may not fit held-out data')
            identity=r['identity'];group[tuple(identity[:2])][tuple(identity[:3])][r['trace_index']].append(i)
        if not group:raise ValueError('Empty training supervised component')
        for products in group.values():
            for traces in products.values():
                for positions in traces.values():weights[positions,part]=1/(len(group)*len(products)*len(traces)*len(positions))
    if not np.allclose(weights.sum(0),1,atol=1e-12):raise ValueError('Hierarchy weights not normalized')
    return weights
