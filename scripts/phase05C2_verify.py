"""Independent evidence audit and read-only Bank comparison; no model calls."""
import argparse,csv,json,os,platform,resource,sys,time
from collections import Counter,defaultdict
from pathlib import Path
from cwm_fp.data.common import sha256,write_json
from cwm_fp.models.local_dataset import SourceBundle,load,signature,tensor_input,task_for
from cwm_fp.models.prefix_observer import tokens,fields
from cwm_fp.lddg.trace import parse_trace,observed_scalar
from cwm_fp.lddg.static import trusted_program_identity,load_static

def lines(p):
    with p.open(encoding='utf8') as f:return [json.loads(s) for s in f]
def ambiguity(rows):
    groups=defaultdict(Counter)
    for r in rows:groups[signature(r['input'])][json.dumps(r['target'])]+=1
    conflict=[c for c in groups.values() if len(c)>1]
    return {'rows':len(rows),'groups':len(groups),'conflicting_groups':len(conflict),'targets_in_conflicts':sum(sum(c.values()) for c in conflict)}
def distribution(rows):
    result={}
    for scope in ('product','test','framework','unresolved'):
        rr=[r for r in rows if r['scope']==scope];result[scope]={'rows':len(rr),'tasks':dict(Counter(r['task'] for r in rr)),
            'opcodes':dict(Counter(str(r['input']['opcode']) for r in rr)),
            'sources':dict(Counter(v['source'] for r in rr for v in r['input']['operands'])),
            'dtypes':dict(Counter(str(v['dtype']) for r in rr for v in r['input']['operands'])),
            'primary_complete':sum(bool(r['input']['operands']) and all(v['primary_known'] for v in r['input']['operands']) for r in rr),
            'verified_pairs':sum(r['target'] is not None for r in rr)}
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--production',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    if not Path('/.dockerenv').exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Container CPU required')
    start=time.monotonic();out=a.root/a.out
    if out.exists():raise ValueError('Never overwrite verification')
    out.mkdir();src=SourceBundle(a.root/'inputs','23e96a1bfc51a218b7ba50a1840c27a14e1d6cad9c1ad36aa2a45dbd498b0b01')
    production=a.root/a.production;states=lines(production/'evidence.jsonl');inputs={r['query_id']:r['learning_input'] for r in lines(production/'inputs.jsonl')};targets={r['query_id']:r for r in lines(production/'targets.jsonl')}
    statuses=load(production/'trace_status.json');links=lines(production/'event_of_static_instruction.jsonl');bytrace=defaultdict(list)
    for r in states:bytrace[r['trace_index']].append(r)
    checked=[];allcounts=Counter();rawop=Counter();rootissues=[];early=[];unresolved=[];markers=[];posthoc=[]
    for index,t in enumerate(src.manifest['traces']):
        path=src.checked(t['run']+'/instrumented/run.log');text=path.read_text(encoding='utf8');active=False;inventory=[];completion_markers=[]
        for tok in tokens(text):
            if tok['text']=='###'+'::'.join(t['identity'][3].rsplit('.',1)):active=True;continue
            if not active:continue
            if tok['kind']=='instruction':
                f=fields(tok['text']);op=int(f['opcode'].split('(')[0]);loc=f['lineinfo'].split('#')
                inventory.append({'clock':tok['clock'],'opcode':op,'loc':loc});rawop[op]+=1
            if tok['kind']=='return_marker' and fields(tok['text'][7:]).get('opcode')=='178(getstatic)':completion_markers.append(tok['clock'])
        # A second, pre-existing parser establishes complete physical instruction
        # count/order independently of the new prefix adapter.
        legacy=parse_trace(path,t['identity'][3],physical_evidence=True)
        rows=bytrace[index]
        if len(legacy)!=len(inventory):raise AssertionError('Raw physical denominator disagreement')
        if len(rows)!=len(legacy):raise AssertionError('Production omitted physical instructions')
        if [(r['state']['opcode'],r['state']['clock'][0]) for r in rows]!=[(r['opcode'],r['raw_log_line']) for r in legacy]:raise AssertionError('Event order/location changed')
        if statuses[index]['parse_error'] is not None:raise AssertionError('Production not fully parsed')
        first=statuses[index]['trusted_until'];trusted=[r for r in rows if r['trusted_prefix_available']]
        if first:
            if any(r['state']['clock']>=first['clock'] for r in trusted):raise AssertionError('Trust resurrected after gap')
            firsti=first['event_index'];raw=legacy[firsti] if firsti is not None else None
            early.append({'identity':t['identity'],'barrier':first,'raw_record':raw,'evidence':t['run']+'/instrumented/run.log'})
        identityfile=load(production/(t['batch_product']+'_identity.json'));ids={tuple(p['locator'][k] for k in ('declaring_class','method','descriptor','bytecode_offset')):p for p in identityfile['identities']}
        static=load_static(src.checked('email/'+t['batch_product']+'/'+t['identity'][2]+'/static.tsv'))
        staticproduct={p['locator']['declaring_class'] for p in ids.values() if p['scope']=='product'}
        counters=Counter()
        for physical,r in zip(legacy,rows):
            p=ids.get((*physical['domain'],physical['bci']));scope='product' if physical['domain'][0] in staticproduct else 'test' if '_ESTest' in physical['domain'][0] else 'framework'
            if scope!=r['scope']:raise AssertionError('Scope classification mismatch')
            counters[scope+'_'+task_for(physical['opcode'])]+=1
            if not p or not trusted_program_identity(p):unresolved.append({'identity':t['identity'],'event_index':physical['event_index'],'opcode':physical['opcode'],
                'locator':[list(physical['domain']),physical['bci']],'reason':p.get('unresolved_reason') if p else 'missing_identity','evidence':t['run']+'/instrumented/run.log'})
            if inputs[r['query_id']]!=r.get('learning_input',inputs[r['query_id']]):raise AssertionError('Input join')
            tensor_input(inputs[r['query_id']])
            if r['state']['current_instruction_executed']:raise AssertionError('Post-state input')
            if r['local_transition_admitted'] and (not r['local_input_verified'] or not r['local_target_verified'] or not r['trusted_prefix_available']):raise AssertionError('Admission inconsistent')
            if scope=='product' and task_for(physical['opcode']) in ('branch','arithmetic'):
                observed=None;reason=None;kind=task_for(physical['opcode']);i=physical['event_index']
                if kind=='branch':
                    nxt=legacy[i+1] if i+1<len(legacy) else None;inst=static['methods'].get(physical['domain'],{}).get(physical['bci'])
                    if nxt and inst and nxt['domain']==physical['domain'] and not nxt['entry'] and len(inst['successors'])==2 and nxt['bci'] in inst['successors']:
                        observed=nxt['bci']!=inst['next']
                    else:reason='no_direct_same_method_binary_successor'
                else:
                    ob=observed_scalar(physical['fields'].get('stack'))
                    if ob.get('observed'):observed=ob['value']
                    else:reason='no_direct_result_callback'
                posthoc.append({'identity':t['identity'],'query_id':r['query_id'],'task':kind,'opcode':physical['opcode'],
                    'observed_suffix_value':observed,'reason':reason,'target_verified':r['local_target_verified'],
                    'not_an_admitted_training_target':not r['local_transition_admitted'],'input':inputs[r['query_id']],
                    'evidence':t['run']+'/instrumented/run.log','line':physical['raw_log_line']})
        allcounts.update(counters)
        checked.append({'identity':t['identity'],'raw_events':len(legacy),'produced_events':len(rows),'counts':dict(counters),
            'getstatic_completion_markers':len(completion_markers),'source_verified':statuses[index]['source_execution_verified'],'order_verified':True,
            'trusted_prefix_events':len(trusted),'evidence':t['run']+'/instrumented/run.log'})
        markers.extend({'identity':t['identity'],'clock':c,'evidence':t['run']+'/instrumented/run.log'} for c in completion_markers)
    write_json(out/'physical_denominator_checks.json',checked);write_json(out/'first_gap_evidence.json',early);write_json(out/'unresolved_event_identities.json',unresolved);write_json(out/'getstatic_completion_evidence.json',markers)
    write_json(out/'unverified_suffix_observations.json',posthoc)
    # Read-only comparison to prior diagnostic records; no original Bank trace,
    # target classification file or test partition is opened.
    bankroot=a.root.parent/'phase05C1_v1/diagnostic_code_v4';bank=[];bank_refs=[]
    for part in ('train','val'):
        pth=bankroot/(part+'_local_state.jsonl');bank_refs.append({'path':str(pth),'sha256':sha256(pth)})
        for r in lines(pth):
            tensor_input(r['learning_input'])
            bank.append({'input':r['learning_input'],'target':r['target'] if r['consistent'] is not None else None,
                'scope':'product' if r['audit']['static_node'] is not None else 'test','task':r['task'],'partition':part,'identity':r['audit']['identity']})
    email=[{'input':inputs[r['query_id']],'target':targets[r['query_id']]['value'] if targets[r['query_id']]['mask'] else None,
            'scope':r['scope'],'task':r['task'],'partition':'train','identity':r['identity']} for r in states]
    eb=[r for r in email if r['scope']=='product' and r['task'] in ('branch','arithmetic')];bb=[r for r in bank if r['scope']=='product']
    bs={signature(r['input']) for r in bb};es={signature(r['input']) for r in eb}
    comparable=[r for r in eb if r['target'] is not None]
    comparison={'bank_frozen_references':bank_refs,'bank_prior_rows':len(bank),'email_all_query_rows':len(email),
        'bank':distribution(bank),'email':distribution(email),'product_branch_arithmetic_exact_input_overlap':{
            'bank_unique':len(bs),'email_unique':len(es),'intersection':len(bs&es),'email_rows_already_seen_in_bank':sum(signature(r['input']) in bs for r in eb)},
        'bank_certified_product_pair_ambiguity':ambiguity([r for r in bb if r['target'] is not None]),
        'email_verified_target_ambiguity':ambiguity(comparable),
        'email_unverified_suffix_value_ambiguity':ambiguity([{'input':r['input'],'target':r['observed_suffix_value']} for r in posthoc if r['observed_suffix_value'] is not None]),
        'email_unverified_branch_successor_counts':dict(Counter(str(r['observed_suffix_value']) for r in posthoc if r['task']=='branch')),
        'email_unverified_suffix_warning':'Physical suffix observations only; prefix/frame semantics may be untrusted. Never used as model input or admitted target.',
        'email_ambiguity_limitation':'Targets outside trusted prefixes are masked; no claim of conflict removal from unknown targets.',
        'bank_counts':{'products':len({tuple(r['identity'][:3]) for r in bank}),'versions':len({tuple(r['identity'][:2]) for r in bank})},
        'neural_generalization_evaluated':False,'causal_intervention_validated':False}
    write_json(out/'bank_email_comparison.json',comparison)
    code=Path(__file__).resolve().parents[1];code_receipt=load(code/'code_receipt.json')
    for rel,h in code_receipt['files'].items():
        if sha256(code/rel)!=h:raise AssertionError('Executing snapshot changed')
    for rel,ref in src.files.items():
        if sha256(src.root/rel)!=ref['sha256']:raise AssertionError('Source changed')
    result={'passed':True,'trace_count':len(checked),'complete_physical_events':sum(r['raw_events'] for r in checked),'denominators':dict(allcounts),
        'opcodes':dict(rawop),'getstatic_completion_markers':len(markers),'unresolved_dynamic_events':len(unresolved),
        'source_files_unchanged':len(src.files),'executed_code_files_unchanged':len(code_receipt['files']),
        'cross_system_contract_rows_checked':len(bank)+len(email),'different_protocols_repartitioned':False,
        'bank_comparison':'bank_email_comparison.json','neural_training':False,'gpu_used':False,'seconds':time.monotonic()-start,
        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'python':sys.version,'container':platform.node()}
    write_json(out/'verification.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
