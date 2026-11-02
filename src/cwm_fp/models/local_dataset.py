"""Verified portable local-transition evidence, independent of product admission.

No labels, full graphs, predicted states or new JVM semantics are accepted.
The interpreter and pre-consumption hook are the existing phase05C1 code.
"""
from __future__ import annotations
import copy,csv,hashlib,json
from pathlib import Path
import numpy as np
from cwm_fp.data.common import sha256
from cwm_fp.lddg.static import load_static,program_identities,trusted_program_identity
from cwm_fp.lddg.runner import completion
from cwm_fp.lddg.trace import observed_scalar
from .prefix_observer import ObservationReplay,tokens,fields
from .local_state import capture,before_instruction,semantic_check,TRUSTED,BRANCH_NAMES,ARITH_NAMES

def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def norm(p):return str(p).replace('\\','/').rstrip('/')
def signature(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

class SourceBundle:
    def __init__(self,root,anchor):
        self.root=Path(root).resolve();path=self.root/'source_manifest.json'
        if sha256(path)!=anchor:raise ValueError('Source manifest anchor mismatch')
        self.manifest=load(path);self.access=[]
        if self.manifest['schema']!='email_local_source_v1':raise ValueError('Wrong source schema')
        self.files=self.manifest['files'];self.paths=self.manifest['path_map']
        for rel in self.files:self.checked(rel)
        self.protocol=load(self.checked('protocol.json'))
        if sha256(self.root/'protocol.json')!=self.manifest['protocol_sha256']:raise ValueError('Protocol changed')
        with self.checked('metadata/splits.csv').open(encoding='utf8') as f:
            self.splits={tuple(r[k] for k in ('system_id','bug_version_id','product_id')):r['split'] for r in csv.DictReader(f)}
        planned=[p['identity']+[t['test_id']] for p in self.protocol['selected'] for t in p['tests']]
        if planned!=[t['identity'] for t in self.manifest['traces']]:raise ValueError('Selected trace order changed')
        for t in self.manifest['traces']:
            if t['identity'][0]!='Email' or t['partition']!='train' or self.splits[tuple(t['identity'][:3])]!='train':raise ValueError('Non-training Email access')
    def checked(self,rel):
        p=(self.root/rel).resolve()
        if rel not in self.files or not p.is_relative_to(self.root) or sha256(p)!=self.files[rel]['sha256']:raise ValueError('Unverified input: '+rel)
        self.access.append(rel);return p
    def resolve(self,original):
        rel=self.paths.get(norm(original))
        if rel is None:raise ValueError('Unmapped original artifact: '+str(original))
        return self.checked(rel)
    def product(self,p):
        base='email/'+p['batch_product'];static=load_static(self.checked(base+'/'+p['identity'][2]+'/static.tsv'))
        binding=load(self.checked(base+'/compilation_binding.json'));evpath=self.checked(base+'/source_evidence.json');evidence=load(evpath)
        class Evidence:
            digest=sha256(evpath)
            def members(_,root):return [r for r in evidence['records'] if norm(r['path']).startswith(norm(root)+'/')]
        for name,h in {**binding['class_sha256'],**binding['source_sha256']}.items():
            if sha256(self.resolve(name))!=h:raise ValueError('Binding bytes changed')
        for name,h in evidence['source_sha256'].items():
            if sha256(self.resolve(name))!=h:raise ValueError('AST source changed')
        with self.checked(base+'/g/source_map.jsonl').open(encoding='utf8') as f:rows=[json.loads(s) for s in f]
        mapped={ (r['composed_class'],r['composed_method'],r['descriptor'],r['bytecode_offset']):copy.deepcopy(r)
                 for r in rows if r.get('bytecode_offset') is not None }
        old={k:copy.deepcopy(r['program_identity']) for k,r in mapped.items()}
        original=next(x for x in load(self.checked('metadata/source_plan.json'))['products'] if x['identity']==p['identity'])
        required=[(*d,b) for d,ii in static['methods'].items() for b in ii if '_ESTest' in d[0]]
        identities=program_identities(Path(original['source_variant_path']),static,mapped,Evidence(),binding,required,artifact_resolver=self.resolve)
        if any(identities[k]!=v for k,v in old.items()):raise ValueError('Portable reconstruction changed frozen product identity')
        return static,identities,binding,{'frozen_product_identities_equal':len(old),'test_identities':len(set(identities)-set(old)),
             'unresolved_product_instructions':sum(v['provenance_type']=='UNRESOLVED' for v in old.values())}
    def execution(self,t,binding):
        status=load(self.checked(t['status']));reasons=[]
        for rel,h in status['artifacts_sha256'].items():
            if sha256(self.checked(t['run']+'/'+rel))!=h:reasons.append('execution_artifact_hash_mismatch')
        for name,h in binding['class_sha256'].items():
            if status['inputs']['runtime_binding']['artifacts_sha256'].get(name)!=h:reasons.append('compiled_execution_class_mismatch')
        results={}
        for mode in ('original','instrumented'):
            run=status['runs'][mode];results[mode]=completion(self.checked(t['run']+'/'+mode+'/result.tsv'),t['identity'][3])
            if results[mode]!=run['result']:reasons.append('completion_status_disagrees')
            # The original paired runner explicitly accepts exit 1 for a
            # completed failing JUnit test; this is not a collection failure.
            if run['execution']['exit_code'] not in (0,1) or run['execution']['reason'] is not None:reasons.append('incomplete_execution')
            text=self.checked(t['run']+'/'+mode+'/run.log').read_text(encoding='utf8')
            if 'CWM_TRUNCATED' in text or '[Agent] bytecode transform failed' in text:reasons.append('truncation_or_transform_failure')
        if results['original']!=results['instrumented']:reasons.append('paired_execution_not_equal')
        if results['original']['outcome']!=t['expected_result']:reasons.append('historical_outcome_mismatch')
        if not all(status[k] for k in ('complete','parity','matches_historical_outcome')):reasons.append('sealed_execution_rejected')
        return status,results,sorted(set(reasons))

def task_for(op):
    return 'branch' if op in BRANCH_NAMES else 'arithmetic' if 96<=op<=152 and op!=132 else 'other'

def identity_check(r,static,identities):
    loc=(*r['domain'],r['bci']);p=identities.get(loc);inst=static['methods'].get(r['domain'],{}).get(r['bci'])
    ok=bool(p and inst and trusted_program_identity(p) and p['opcode']==inst['opcode']==r['opcode']
        and p['locator']['class_sha256']==static['classes'][r['domain'][0]]['class_sha256'])
    return ok,p,inst

def control_prefix_errors(flow,raw,current,static):
    """Only observed adjacency; pending calls do not require future returns.

    Existing bytecode CFGs and interpreter frame IDs are cross-checked. Actual
    handler entries stay conservative here; this does not add exception support.
    """
    errors=[];last={};entered=set();events=list(flow.events)+[current]
    for i,e in enumerate(events):
        d=e['domain'];info=static['methods'].get(d,{});inst=info.get(e['bci']);table=static['exception_tables'].get(d)
        if inst is None or table is None:return [{'event_index':i,'reason':'missing_static_CFG_or_exception_table'}]
        if '|'.join(d) in static.get('gaps',{}):return [{'event_index':i,'reason':'unsupported_static_control'}]
        if any(h['handler']==e['bci'] for h in table['handlers']):return [{'event_index':i,'reason':'handler_entry_outside_local_normal_prefix_scope'}]
        fid=e['frame_id']
        if e['entry']:
            if fid in entered or e['bci']!=min(info):return [{'event_index':i,'reason':'invalid_invocation_entry'}]
            if i:
                prev=events[i-1];target=static['calls'].get((*prev['domain'],prev['bci']))
                if not 182<=prev['opcode']<=186 or target!=d:return [{'event_index':i,'reason':'unmatched_observed_callee_entry'}]
            entered.add(fid)
        elif i==0:return [{'event_index':0,'reason':'missing_initial_method_entry'}]
        prev=last.get(fid)
        if prev:
            if prev['domain']!=d or e['bci'] not in info[prev['bci']]['successors']:return [{'event_index':i,'reason':'non_normal_CFG_transition'}]
            if 182<=prev['opcode']<=186 and not prev['call_returned']:return [{'event_index':i,'reason':'caller_resume_without_observed_return'}]
        if 182<=e['opcode']<=186:
            f=e['fields'];target=(f.get('callclass','').replace('/','.'),f.get('callname'),f.get('calltype'))
            if static['calls'].get((*d,e['bci']))!=target:return [{'event_index':i,'reason':'call_target_metadata_mismatch'}]
        last[fid]=e
    return errors

def unknown_state(r,reason):
    op=r['opcode'];arity=(2 if 159<=op<=166 else 1) if op in BRANCH_NAMES else max(0,int(r['fields'].get('popnum','0')))
    return {'opcode':op,'operation':BRANCH_NAMES.get(op,ARITH_NAMES.get(op,'i2l' if op==133 else 'unsupported_'+str(op))),
        'arity':arity,'operand_order':'bottom_to_top_value1_then_value2','clock':r['pre_clock'],'frame_id':None,
        'event_index':r['event_index'],'prior_gaps':[{'event_index':r['event_index'],'reason':reason}], 'current_instruction_executed':False,
        'operands':[{'value_id':None,'definition_event':None,'source':'UNKNOWN','dtype':None,'value':None,'available_at':None,
                    'reason':reason,'static_alternative':None,'derivation':[]} for _ in range(arity)]}

def learning_input(state):
    return {'opcode':state['opcode'],'arity':state['arity'],'operands':[
        {'dtype':v['dtype'],'source':v['source'],'primary_known':v['source'] in TRUSTED,'value':v['value'] if v['source'] in TRUSTED else None}
        for v in state['operands']]}

class LocalObservationReplay(ObservationReplay):
    """Opt-in format adapter for the producer's real getstatic RET marker.

    TraceTransformer/GetStaticInst insert this completion record instead of a
    scalar callback. It certifies completion only, never a field value, initial
    definition, call frame or absence of class-initializer effects. The original
    observer and all frozen Bank snapshots retain their prior format behavior.
    """
    def push(self,t):
        static_index=None
        if self.active and t['kind']=='return_marker':
            f=fields(t['text'][7:]);where=f.get('lineinfo');pending=self.pending.get(where,[])
            if pending and self.full[pending[-1]]['opcode']==178:
                static_index=pending[-1];r=self.full[static_index]
                if f.get('opcode')!='178(getstatic)' or where!='#'.join((*r['domain'],str(r['line']),str(r['bci']))):
                    raise ValueError('Static field completion marker mismatch')
        super().push(t)
        if self.active and t['kind']=='instruction' and self.full[-1]['opcode']==178:
            f=fields(t['text']);self.pending.setdefault(f['lineinfo'],[]).append(len(self.full)-1)
        if static_index is not None:
            r=self.full[static_index];r['call_returned']=False
            r['completion_marker_kind']='static_field_access_completed_no_value'

def prefix_probe(observer,static,framework,identities):
    r=observer.full[-1];ok,p,inst=identity_check(r,static,identities)
    if not ok:return unknown_state(r,'unresolved_or_mismatched_instruction_identity'),None,[{'event_index':len(observer.full)-1,'reason':'unresolved_or_mismatched_instruction_identity'}]
    state,flow,raw=capture(observer,static,framework)
    if state['arity']<0:
        # Legacy void-return metadata uses -1, not a negative operand count.
        state['arity']=0;state['operands']=[]
    current={'frame_id':state['frame_id'],**raw[-1]}
    gaps=[g for g in flow.gaps if g['reason']!='unterminated_frame']+control_prefix_errors(flow,raw,current,static)
    return state,flow,gaps

def replay_inputs(text,test_id,static,framework,identities,*,source_ok=True,stop_cut=None):
    """Sequential prefix production with a permanent first-gap latch."""
    observer=LocalObservationReplay(test_id);states=[];links=[];barrier=None;error=None;cuts={};critical=[];observations=[]
    if not source_ok:barrier={'clock':[0,0],'reason':'source_execution_not_verified','event_index':None}
    for tok in tokens(text):
        if stop_cut is not None and tok['cut']>stop_cut:break
        try:observer.push(tok)
        except (ValueError,KeyError,IndexError) as exc:
            error={'clock':tok['clock'],'reason':'physical_record_parse_error:'+str(exc),'event_index':len(observer.full)}
            if barrier is None:barrier=error
            break
        if observer.active:observations.append(tok)
        if tok['kind']!='instruction' or not observer.full:continue
        r=observer.full[-1];i=len(observer.full)-1;cuts[i]=tok['cut']
        ok,p,inst=identity_check(r,static,identities)
        scope=p.get('scope') if p else ('framework' if r['domain'][0].startswith(('org.','java.','junit.')) else 'unresolved')
        links.append({'event_index':i,'clock':r['pre_clock'],'scope':scope,'identity_verified':ok,
            'instruction_id':p.get('instruction_id') if p else None,'opcode':r['opcode'],
            'static_control_context_available':ok and r['domain'] in static['controls'],
            'relation':'event_of_static_instruction','dynamic_control_dependency':False})
        if r['opcode'] in BRANCH_NAMES or 172<=r['opcode']<=186 or r['opcode']==191 or any(h['handler']==r['bci'] for h in static.get('exception_tables',{}).get(r['domain'],{}).get('handlers',[])):
            critical.append(i)
        flow=None
        if barrier is None:
            try:
                state,flow,gaps=prefix_probe(observer,static,framework,identities)
                if gaps:
                    g=min(gaps,key=lambda x:x['event_index']);n=max(0,min(g['event_index'],len(observer.full)-1))
                    barrier={'clock':observer.full[n]['pre_clock'],'reason':g['reason'],'event_index':n,'detected_at':r['pre_clock'],
                             'opcode':observer.full[n]['opcode'],'domain':observer.full[n]['domain'],'bci':observer.full[n]['bci']}
            except (ValueError,KeyError,IndexError) as exc:
                barrier={'clock':r['pre_clock'],'reason':'prefix_replay_rejected:'+str(exc),'event_index':i,'opcode':r['opcode']}
                state=unknown_state(r,barrier['reason'])
        else:state=unknown_state(r,'after_trust_boundary:'+barrier['reason'])
        if barrier is not None and r['pre_clock']>=barrier['clock']:state=unknown_state(r,'after_trust_boundary:'+barrier['reason'])
        # Dependency references are historical local indices, never learnable IDs.
        evidence={'links':[list(x) for x in flow.links] if flow is not None else [],
                  'frames':[{'id':f.id,'domain':list(f.domain),'caller_event':f.caller_event} for f in flow.frames] if flow is not None else [],
                  'prefix_event_count':len(flow.events) if flow is not None else None}
        states.append({'state':state,'learning_input':learning_input(state),'scope':scope,'identity_verified':ok,
                       'cut':tok['cut'],'event_index':i,'evidence':evidence})
    if not observer.active or not observer.full:
        if barrier is None:barrier={'clock':[0,0],'event_index':None,'reason':'missing_original_root_marker_or_events'}
    return {'observer':observer,'states':states,'links':links,'barrier':barrier,'parse_error':error,'cuts':cuts,
            'critical':critical,'observations':observations}

def target_for(r,nxt,static,frame,nxtframe,boundary):
    """Separate target-side function. Never called during input capture."""
    op=r['opcode'];task=task_for(op);out={'value':None,'mask':False,'available_at':None,'reason':'not_branch_or_arithmetic'}
    if task=='branch':
        info=static['methods'].get(r['domain'],{}).get(r['bci'])
        if not nxt:out['reason']='no_real_successor'
        elif frame is None or frame!=nxtframe or nxt['domain']!=r['domain'] or nxt['entry']:out['reason']='successor_frame_unverified'
        elif not info or len(info['successors'])!=2 or nxt['bci'] not in info['successors']:out['reason']='successor_not_binary_CFG'
        else:out.update(value=nxt['bci']!=info['next'],mask=True,available_at=nxt['pre_clock'],reason=None)
    elif task=='arithmetic':
        ob=observed_scalar(r['fields'].get('stack'))
        if ob.get('observed') and ob.get('dtype') in {'I','Z','S','B'} and type(ob.get('value')) in {int,bool} and r['callback_clock']:
            out.update(value=int(ob['value']),mask=True,available_at=r['callback_clock'],reason=None)
        else:out['reason']='no_supported_direct_result_callback'
    if out['mask'] and (out['available_at']<=r['pre_clock'] or boundary is not None and out['available_at']>=boundary):
        out.update(value=None,mask=False,reason='target_crosses_trust_boundary_or_time')
    return out

def tensor_input(value):
    """Exact int64 representation with separate masks; no target or audit IDs."""
    if set(value)!={'opcode','arity','operands'}:raise ValueError('Forbidden learning input field')
    if value['arity']!=len(value['operands']):raise ValueError('Arity mismatch')
    sources=('DIRECT_OBSERVED','STATIC_CONSTANT','SEMANTICALLY_DERIVED','UNKNOWN');numbers=[];known=[];null=[];origins=[]
    for v in value['operands']:
        if set(v)!={'dtype','source','primary_known','value'}:raise ValueError('Forbidden operand feature')
        if v['source'] not in sources or v['primary_known']!=(v['source'] in TRUSTED):raise ValueError('Invalid source mask')
        k=v['primary_known'];x=v['value'];isnull=k and v['dtype']=='reference_null' and x is None
        if k and not isnull and (v['dtype']!='I' or type(x) is not int or not -(1<<31)<=x<(1<<31)):raise ValueError('Not precise supported JVM input')
        if not k and x is not None:raise ValueError('Unknown masquerades as observed value')
        numbers.append(x if k and not isnull else 0);known.append(k);null.append(isnull);origins.append(sources.index(v['source']))
    return {'opcode':np.asarray(value['opcode'],dtype=np.int64),'value':np.asarray(numbers,dtype=np.int64),
            'known':np.asarray(known,dtype=bool),'null':np.asarray(null,dtype=bool),'source':np.asarray(origins,dtype=np.int64)}

def read_input(root,query_id,*,expected_manifest_sha256):
    """Model-facing reader: validates only input/evidence refs, never targets."""
    root=Path(root).resolve();manifest=root/'manifest.json'
    if sha256(manifest)!=expected_manifest_sha256:raise ValueError('Local dataset anchor mismatch')
    m=load(manifest)
    if m['schema']!='local_transition_dataset_v1':raise ValueError('Unknown dataset')
    ref=m['inputs'];p=(root/ref['path']).resolve()
    if not p.is_relative_to(root) or sha256(p)!=ref['sha256']:raise ValueError('Corrupt input')
    with p.open(encoding='utf8') as f:
        for line in f:
            row=json.loads(line)
            if row['query_id']==query_id:return tensor_input(row['learning_input'])
    raise ValueError('Unknown query')
