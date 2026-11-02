"""Lossless, bounded, relative-path product packages; no Java certification bypass.

Export requires the original consumer's semantic checks. Import verifies an
externally anchored manifest, exact tensors/records and original-suite joins.
It carries a collector attestation, not a claim to replay Java on Linux.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import gzip
import json
from pathlib import Path, PurePosixPath
import re
import zipfile
import numpy as np
import torch
from cwm_fp.data.common import sha256
from .graph_encoder import (FEATURE_SCHEMA,FEATURES,RELATIONS,Limits,TestTensor,
                            ProductTensor,encode_test_lossless,collate)

PACKAGE_SCHEMA='lossless_product_package_v1'
ABSOLUTE=re.compile(r'^(?:[A-Za-z]:[\\/]|[/\\])')

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')),encoding='utf-8')
def read_metadata(path,max_bytes):
    if path.suffix=='.gz':
        with gzip.open(path,'rb') as f:data=f.read(max_bytes+1)
    else:
        with path.open('rb') as f:data=f.read(max_bytes+1)
    if len(data)>max_bytes:raise ValueError('Decoded metadata exceeds capacity')
    return json.loads(data.decode('utf-8'))
def portable(value,locators):
    if isinstance(value,str) and ABSOLUTE.match(value):
        key='external/'+hashlib.sha256(value.encode()).hexdigest();locators[key]=value;return key
    if isinstance(value,list):return [portable(v,locators) for v in value]
    if isinstance(value,dict):return {portable(k,locators):portable(v,locators) for k,v in value.items()}
    return value

@dataclass(frozen=True)
class ReadBudget:
    max_shard_bytes:int=16*1024**2
    max_manifest_bytes:int=4*1024**2
    max_nodes:int=4096
    max_events:int=4096
    max_edges:int=65536
    max_tests:int=4096
    max_total_bytes:int=512*1024**2

def safe_path(root,name):
    p=PurePosixPath(name)
    if not name or '\\' in name or ':' in name or p.is_absolute() or any(x in ('..','.') for x in p.parts):
        raise ValueError('Unsafe package-relative path')
    resolved=(root/Path(*p.parts)).resolve()
    if not resolved.is_relative_to(root.resolve()):raise ValueError('Package path escapes root')
    return resolved

def write_test(root,number,graph,events,source_receipt,locators,budget=ReadBudget()):
    """Called only after product-level suite admission; repeats per-graph gate."""
    if graph.get('provenance',{}).get('identity_schema')!='program_identity_v1':raise ValueError('Portable export requires certified program identity schema')
    tensor=encode_test_lossless(graph,capacity=Limits(budget.max_nodes,budget.max_events,1,budget.max_edges))
    identity=[graph[k] for k in ('system_id','bug_version_id','product_id','test_id')]
    if len(events)!=graph['quality']['events'] or [e['event_index'] for e in events]!=list(range(len(events))):
        raise ValueError('Incomplete ordered event records')
    if any([e[k] for k in ('system_id','bug_version_id','product_id','test_id')]!=identity for e in events):
        raise ValueError('Cross-test event ownership')
    npath=f'graphs/{number:06d}.npz';mpath=f'graphs/{number:06d}.json.gz'
    (root/'graphs').mkdir(exist_ok=True)
    np.savez(root/npath,x=tensor.x.numpy(),edge_index=tensor.edge_index.numpy(),edge_type=tensor.edge_type.numpy(),coverage=tensor.coverage.numpy())
    attrs=portable(tensor.metadata,locators)
    # Retained observations/identities are audit side channels, never numerical IDs.
    record={'schema':PACKAGE_SCHEMA,'identity':identity,'tensor_metadata':attrs,
        'events':portable(events,locators),'instruction_catalog':portable(graph.get('instruction_catalog',[]),locators),
        'quality':graph['quality'],'framework_effects':portable(graph.get('framework_effects',[]),locators),
        'collector_attestation':{'identity_verified':True,'dependencies_verified':True,
            'source_graph_sha256':digest(graph),'source_receipt':portable(source_receipt,locators),
            'extractor_version':graph['provenance']['extractor_version'],
            'control_schema':graph['provenance'].get('control_schema'),
            'trace_sha256':graph['provenance'].get('trace_sha256'),
            'semantic_scope':graph['quality'].get('control_semantics_scope'),
            'verification_location':'collector; not independently replayed by portable loader'},
        'visibility_contract':{'schema':'prefix_observation_contract_v1','actual_prefix_dataset':False,
            'unknown_call_return_visibility':'excluded until producer-supplied availability boundary',
            'final_outcomes_allowed_as_early_input':False}}
    data=json.dumps(record,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
    if len(data)>budget.max_shard_bytes:raise ValueError('Decoded metadata exceeds capacity')
    (root/mpath).write_bytes(gzip.compress(data,mtime=0))
    files={k:{'path':p,'sha256':sha256(root/p),'bytes':(root/p).stat().st_size} for k,p in [('tensor',npath),('metadata',mpath)]}
    if any(f['bytes']>budget.max_shard_bytes for f in files.values()):raise ValueError('Single shard exceeds capacity')
    return {'identity':identity,**files,'nodes':len(tensor.x),'edges':len(tensor.edge_type),'events':len(events),
            'relations':{r:sum(e['type']==r for e in graph['edges']) for r in RELATIONS}}

class ProductPackage:
    def __init__(self,root,*,expected_manifest_sha256,budget=ReadBudget()):
        self.root=Path(root);self.budget=budget;p=self.root/'manifest.json'
        if p.stat().st_size>budget.max_manifest_bytes or sha256(p)!=expected_manifest_sha256:
            raise ValueError('Package manifest differs from trusted collector receipt')
        self.manifest=json.loads(p.read_text(encoding='utf-8'));m=self.manifest
        if m['schema']!=PACKAGE_SCHEMA or m['feature_schema']!=FEATURE_SCHEMA or m['features']!=list(FEATURES) or m['relations']!=list(RELATIONS):
            raise ValueError('Unknown or mixed feature/package schema')
        if m['relation_policy']!='full':raise ValueError('Lossless package must retain all relations')
        self.products=m['products'];seen=set();groups={f:{} for f in m['split_protocols']};paths=set();total=0
        for p in self.products:
            key=tuple(p['identity'])
            if len(key)!=3 or key in seen or p['observed_result']!='PASS':raise ValueError('Duplicate/non-PASS product')
            seen.add(key)
            tests=p['tests'];expected=p['registered_tests']
            if not tests or len(tests)>budget.max_tests or expected!=sorted(set(expected)) or [g['identity'][3] for g in tests]!=expected:
                raise ValueError('Incomplete/duplicate/reordered original test suite')
            for fold,part in p['partitions'].items():
                if fold not in groups or part not in ('train','val','test'):raise ValueError('Unknown partition')
                if groups[fold].setdefault(key[:2],part)!=part:raise ValueError('Version split overlap')
                if fold.startswith('loso_') and key[0]==fold[5:] and part!='test':raise ValueError('Held-out system in training')
            if set(p['partitions'])!=set(groups):raise ValueError('Missing frozen partition')
            for g in tests:
                if tuple(g['identity'][:3])!=key:raise ValueError('Cross-product graph')
                if g['nodes']>budget.max_nodes or g['edges']>budget.max_edges or g['events']>budget.max_events:
                    raise ValueError('Single graph exceeds lossless capacity')
                for kind in ('tensor','metadata'):
                    ref=g[kind];path=safe_path(self.root,ref['path'])
                    if ref['path'] in paths:raise ValueError('Shared shard across test identities')
                    paths.add(ref['path']);total+=ref['bytes']
                    if ref['bytes']>budget.max_shard_bytes or path.stat().st_size!=ref['bytes']:raise ValueError('Missing/oversize shard')
        if total>budget.max_total_bytes:raise ValueError('Package exceeds configured total budget')
        # Absolute source locations are never operational package paths.

    def _read(self,ref):
        path=safe_path(self.root,ref['path'])
        if path.stat().st_size!=ref['bytes'] or sha256(path)!=ref['sha256']:raise ValueError('Corrupt shard or hash mismatch')
        return path

    def load_test(self,g):
        meta=read_metadata(self._read(g['metadata']),self.budget.max_shard_bytes)
        if meta['schema']!=PACKAGE_SCHEMA or meta['identity']!=g['identity']:raise ValueError('Mismatched test lineage')
        a=meta['collector_attestation'];q=meta['quality'];m=meta['tensor_metadata']
        if (a['identity_verified'] is not True or a['dependencies_verified'] is not True
            or q['eligible_graph_structure'] is not True or q['program_identity_complete'] is not True
            or q['dependency_semantics_valid'] is not True or q['unsupported_dynamic'] or not q['suite_completeness']['complete']):
            raise ValueError('Uncertified graph cannot enter portable training input')
        if m['feature_schema']!=FEATURE_SCHEMA or not m['lossless'] or m['ablation']!='full':raise ValueError('Non-lossless feature representation')
        path=self._read(g['tensor'])
        with zipfile.ZipFile(path) as z:
            if sum(i.file_size for i in z.infolist())>self.budget.max_shard_bytes:raise ValueError('Decompressed tensor exceeds capacity')
        with np.load(path,allow_pickle=False) as f:
            if set(f.files)!={'x','edge_index','edge_type','coverage'}:raise ValueError('Unknown tensor fields')
            x,ei,et,c=[f[k].copy() for k in ('x','edge_index','edge_type','coverage')]
        if x.dtype!=np.float32 or ei.dtype!=np.int64 or et.dtype!=np.int64 or c.dtype!=np.float32:raise ValueError('Tensor dtype mismatch')
        if x.shape!=(g['nodes'],22) or ei.shape!=(2,g['edges']) or et.shape!=(g['edges'],) or c.shape!=(4,):raise ValueError('Tensor shape mismatch')
        if not np.isfinite(x).all() or not np.isfinite(c).all() or not np.isin(x,[0,1]).all():raise ValueError('Invalid numerical features')
        nodes=m['node_records'];edges=m['edge_records'];events=meta['events']
        if (len(nodes),len(edges),len(events))!=(g['nodes'],g['edges'],g['events']):raise ValueError('Record count mismatch')
        positions={n['id']:i for i,n in enumerate(nodes)}
        if len(positions)!=len(nodes):raise ValueError('Duplicate nodes')
        for i,e in enumerate(edges):
            if e['source'] not in positions or e['target'] not in positions or e['type'] not in RELATIONS:raise ValueError('Unknown edge endpoint')
            if tuple(ei[:,i])!=(positions[e['source']],positions[e['target']]) or et[i]!=RELATIONS.index(e['type']):raise ValueError('Edge tensor/record mismatch')
            if e['type']=='data_dep' and not (0<=e['definition_event']<e['event_index']<g['events']):raise ValueError('Invalid dependency order')
        if [e['event_index'] for e in events]!=list(range(g['events'])):raise ValueError('Incomplete execution sequence')
        for e in events:
            if [e[k] for k in ('system_id','bug_version_id','product_id','test_id')]!=g['identity']:raise ValueError('Cross-test event')
            n=nodes[positions[e['statement_id']]]['attrs']
            for key in ('canonical_event_id','static_instruction_id','frame_id','event_index'):
                if e.get(key)!=n.get(key):raise ValueError('Event/instruction association mismatch')
        if len({e['canonical_event_id'] for e in events})!=len(events):raise ValueError('Merged dynamic event identities')
        if g['relations']!={r:sum(e['type']==r for e in edges) for r in RELATIONS}:raise ValueError('Relation count mismatch')
        m={**m,'events':events,'visibility_contract':meta['visibility_contract'],'collector_attestation':a,'suite_completeness':q['suite_completeness']}
        return TestTensor(torch.from_numpy(x),torch.from_numpy(ei),torch.from_numpy(et),torch.from_numpy(c),m)

    def iter_tests(self,product,*,start=0):
        """Resume by ordered test position; suffix alone is never a complete product."""
        if not 0<=start<=len(product['tests']):raise ValueError('Invalid resume position')
        if product not in self.products:raise ValueError('Product is not in the anchored manifest')
        for g in product['tests'][start:]:
            result=self.load_test(g)
            # The trusted collection attestation and the registered suite must agree.
            suite=result.metadata['suite_completeness']
            if suite['expected_tests']!=len(product['registered_tests']) or suite['collected_tests']!=len(product['tests']):
                raise ValueError('Suite completeness claim differs from anchored registration')
            yield result

    def materialize(self,product,fold,*,max_tensor_bytes=64*1024**2):
        # Full materialization is explicitly bounded. Streaming has no all-product list.
        estimate=sum(g['nodes']*22*4+g['edges']*24+16 for g in product['tests'])
        if estimate>max_tensor_bytes:raise ValueError('Product exceeds materialization capacity; use streaming')
        tests=list(self.iter_tests(product))
        return ProductTensor(tuple(product['identity']),product['sample_id'],product['partitions'][fold],'PASS',tests,
            {'tests_before':len(tests),'tests_after':len(tests),'lost_test_fraction':0.,'lost_node_fraction':0.,'lost_event_fraction':0.,'lost_edge_fraction':0.,'preview_only':False})

@torch.inference_mode()
def forward_stream(model,package,product,*,fold='grouped',micro_tests=1):
    """Exact existing test-mean/product-mean hierarchy; evaluation only, no fit.

    Training with gradient accumulation is deliberately not claimed here.
    """
    if model.training or not 1<=micro_tests<=32:raise ValueError('Frozen evaluation model and bounded microbatch required')
    count=0;total=None;pending=[]
    for test in package.iter_tests(product):
        pending.append(test)
        if len(pending)==micro_tests or count+len(pending)==len(product['tests']):
            p=ProductTensor(tuple(product['identity']),product['sample_id'],product['partitions'][fold],'PASS',pending,{})
            batch=collate([p]).to(next(model.parameters()).device)
            h=model.encode(batch)*len(pending);total=h if total is None else total+h
            count+=len(pending);pending=[]
    if pending or count!=len(product['registered_tests']):raise ValueError('Cannot aggregate an incomplete product')
    return model.head(total/count).squeeze(-1)


def forward_train_stream(model,package,product,*,fold='grouped',micro_tests=5):
    """Differentiable whole-product mean; retains all chunk activation graphs.

    This bounds disk reads, NOT activation memory. No parameter update is valid
    until the single complete-product loss has been formed and backpropagated.
    """
    if product['partitions'].get(fold)!='train' or not 1<=micro_tests<=32:
        raise ValueError('Gradient streaming requires a complete TRAIN product')
    total=None;count=0;pending=[]
    for test in package.iter_tests(product):
        pending.append(test)
        if len(pending)==micro_tests or count+len(pending)==len(product['tests']):
            p=ProductTensor(tuple(product['identity']),product['sample_id'],'train','PASS',pending,{})
            h=model.encode(collate([p]).to(next(model.parameters()).device))
            weighted=h*len(pending)  # The short final chunk must have its true weight.
            total=weighted if total is None else total+weighted
            count+=len(pending);pending=[]
    if pending or count!=len(product['registered_tests']) or count==0:
        raise ValueError('Cannot train on an incomplete product')
    return model.head(total/count).squeeze(-1)
