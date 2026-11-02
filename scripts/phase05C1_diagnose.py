"""Container-only, frozen train then validation local-state audit; no fitting."""
import argparse,csv,json,os,platform,resource,time
from collections import Counter,defaultdict
from pathlib import Path
from cwm_fp.data.common import ROOT,sha256,write_json
from cwm_fp.lddg.static import load_static,trusted_program_identity
from cwm_fp.lddg.effects import HELPER
from cwm_fp.models.prefix_observer import ObservationReplay,PrefixBuilder
from cwm_fp.models.transition_data import load,lines,checked,verify_source,access_guard
from cwm_fp.models.local_state import capture,semantic_check,TRUSTED,BRANCH_NAMES,ARITH_NAMES

def emit(f,row):f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')

def ambiguity(rows,level):
    groups=defaultdict(Counter)
    for r in rows:
        s=r['pre_state_evidence'];key=[r['structure_signature']]
        if level>=1:key.append(s['opcode'])
        if level>=2:key.append([(v['dtype'],v['value'],True) if v['source'] in TRUSTED else (None,None,False) for v in s['operands']])
        groups[json.dumps(key,sort_keys=True)][r['target']]+=1
    conflicts=[c for c in groups.values() if len(c)>1]
    return {'groups':len(groups),'conflicting_groups':len(conflicts),'targets_in_conflicting_groups':sum(sum(c.values()) for c in conflicts),
        'majority_correct_upper_bound_count':sum(max(c.values()) for c in groups.values()),'denominator':len(rows)}

def summarize(rows):
    result={}
    for task in ('branch','arithmetic'):
        rr=[r for r in rows if r['task']==task];counts=Counter();sources=Counter();missing=Counter();ops=defaultdict(Counter)
        for r in rr:
            s=r['pre_state_evidence'];vs=s['operands'];n=sum(v['source'] in TRUSTED for v in vs)
            counts['all_known' if n==len(vs) else 'all_unknown' if n==0 else 'partial_known']+=1
            counts['with_derived_all_known' if all(v['source']!='UNKNOWN' for v in vs) else 'with_derived_still_missing']+=1
            if any(v['source']=='SEMANTICALLY_DERIVED' for v in vs):counts['contains_derived_evidence']+=1
            if r['prediction'] is not None:counts['checked']+=1;counts['consistent' if r['consistent'] else 'inconsistent']+=1
            else:missing[r['abstention_reason']]+=1
            for v in vs:
                sources[v['source']]+=1
                if v['reason']:missing[v['reason']]+=1
            if all(v['source']=='DIRECT_OBSERVED' for v in vs):counts['direct_only_complete']+=1
            if all(v['static_alternative'] is not None for v in vs):counts['static_literal_only_complete']+=1
            if n==len(vs) and any(v['source']=='STATIC_CONSTANT' for v in vs):counts['static_added_complete']+=1
            ops[s['operation']][str(r['target'])]+=1
        result[task]={'denominator':len(rr),'coverage_counts':dict(counts),'operand_sources':dict(sources),'missing_reasons':dict(missing),'opcode_targets':dict(ops),
            'coverage':counts['checked']/len(rr) if rr else None,'covered_consistency':counts['consistent']/counts['checked'] if counts['checked'] else None}
        if task=='branch':result[task]['ambiguity']={name:ambiguity(rr,k) for k,name in enumerate(('structure','plus_opcode','plus_trusted_operands'))}
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if platform.system()!='Linux':raise ValueError('Run inside the verified container')
    if a.out.exists():raise ValueError('Use a new output directory')
    a.out.mkdir(parents=True);start=time.monotonic();access_guard(a.source,{'train','val'})
    m,counts,evidence=verify_source(a.source);framework=load_static(checked(a.source,m['framework_static']))
    # Refuse unnoticed changes to the original replay engine and prefix producer.
    old=load(ROOT/'outputs/phase05A_protocol.json')
    for f,h in old['source_sha256'].items():
        if sha256(ROOT/f)!=h:raise ValueError('Frozen replay source changed: '+f)
    protocol={'schema':'pre_instruction_local_state_v1','source_manifest_sha256':sha256(a.source/'prefix_manifest.json'),
        'code_sha256':{f:sha256(ROOT/f) for f in ['src/cwm_fp/models/local_state.py','scripts/phase05C1_diagnose.py']},
        'supported_branch':BRANCH_NAMES,'supported_arithmetic':ARITH_NAMES,'primary_sources':sorted(TRUSTED),
        'reference_policy':'null literal only; no object-hash equality','numeric_policy':'int32 only; no wide/float coercion',
        'constant_policy':'iconst and aconst_null opcode literals only; no guessing missing bytecode immediates',
        'selection':'all original valid conditional targets and original numeric_pairs; denominator never filtered',
        'identity_columns':'audit side table only','validation_opened_before_freeze':False}
    write_json(a.out/'protocol.json',protocol);frozen=sha256(a.out/'protocol.json')
    allrows=[];identity_counts=Counter();reach=Counter();graph_status=[]
    with (a.out/'event_of_static_instruction.jsonl').open('w',encoding='utf8') as links:
        for part in ('train','val'):
            if sha256(a.out/'protocol.json')!=frozen:raise ValueError('Support protocol changed')
            for f,h in protocol['code_sha256'].items():
                if sha256(ROOT/f)!=h:raise ValueError('Diagnostic code changed')
            index=load(a.cache_root/'cache'/f'index_{part}.json');sig={(r['trace_index'],r['event_index']):r['signatures']['rgcn'] for r in index['rows']}
            partrows=[]
            for trace in [t for t in m['traces'] if t['partition']==part]:
                if time.monotonic()-start>6000 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*1024**3:raise RuntimeError('Diagnostic budget')
                refs=trace['input_files'];static=load_static(checked(a.source,refs['static_bytecode']));bundle=load(checked(a.source,refs['static']))
                builder=PrefixBuilder(static,bundle,framework)
                if len(builder.identities)!=len(bundle['dynamic_identity_lookup']):raise ValueError('Ambiguous static identity')
                idx=lines(checked(a.source,refs['prefix_index']));byclock={tuple(r['clock']):r for r in idx}
                pairrel=f"traces/{trace['trace_index']:04d}/numeric_pairs.json"
                pairpath=checked(a.source,{'path':pairrel,'sha256':evidence[pairrel]['sha256']})
                pairs={r['event_index']:r for r in load(pairpath)};captured={}
                obs=ObservationReplay(trace['identity'][3]);obs.active=True;obs.entry=True;verified=0
                with checked(a.source,refs['observations']).open(encoding='utf8') as stream:
                    for line in stream:
                        tok=json.loads(line);obs.push(tok);item=byclock.get(tuple(tok['clock']))
                        if item is None:continue
                        r=obs.full[-1];i=item['event_index'];loc=(*r['domain'],r['bci']);identity=builder.identities.get(loc)
                        inst=static['methods'].get(r['domain'],{}).get(r['bci']);cl=static['classes'].get(r['domain'][0],{})
                        ok=bool(identity and inst and trusted_program_identity(identity) and identity['opcode']==inst['opcode']==r['opcode'] and identity['locator']['class_sha256']==cl.get('class_sha256'))
                        staticnode=builder.positions.get(loc);identity_counts[part+'_events']+=1;identity_counts[part+'_verified' if ok else part+'_unresolved']+=1
                        if staticnode is not None:identity_counts[part+'_in_product_static_input']+=1
                        emit(links,{'identity':trace['identity'],'partition':part,'trace_index':trace['trace_index'],'event_index':i,'clock':tok['clock'],
                            'instruction_id':identity.get('instruction_id') if identity else None,'verified':ok,'scope':identity.get('scope') if identity else None,
                            'class_sha256':cl.get('class_sha256'),'source_static_sha256':refs['static_bytecode']['sha256'],'product_static_node':staticnode,
                            'relation':'event_of_static_instruction','dynamic_control_dependency':False})
                        if not ok:raise ValueError('Static/dynamic identity mismatch')
                        verified+=1
                        if r['opcode'] not in BRANCH_NAMES and i not in pairs:continue
                        state,flow,raw=capture(obs,static,framework)
                        if state['event_index']!=i:raise ValueError('Projected event alignment')
                        snap=builder.build(obs)
                        if snap.fingerprint()!=item['input_sha256']:raise ValueError('Existing graph changed')
                        sn=snap.audit['static_nodes'];anchor=sn+i;reachable={anchor}
                        for _ in range(2):reachable|={int(s) for s,d in snap.model['edge_index'].T if int(d) in reachable}
                        nodes={v['value_id']:v['local_node'] for v in snap.audit['values']}
                        wanted=[v for v in state['operands'] if v['value_id'] is not None]
                        rv=[nodes.get(v['value_id']) in reachable for v in wanted]
                        rd=[v['definition_event'] is not None and sn+v['definition_event'] in reachable for v in wanted]
                        reach['queries']+=1;reach['operand_slots']+=len(wanted);reach['value_nodes_in_two_hops']+=sum(rv);reach['definition_events_in_two_hops']+=sum(rd)
                        reach['queries_all_values_in_two_hops']+=int(bool(rv) and all(rv));reach['queries_all_definitions_in_two_hops']+=int(bool(rd) and all(rd))
                        reach['static_context_directly_reachable']+=int(any(n<sn for n in reachable))
                        if staticnode is not None:reach['static_context_recoverable_via_audit_link']+=1
                        captured[i]={'pre_state_evidence':state,'structure_signature':sig[(trace['trace_index'],i)],
                            'learning_input':{'opcode':state['opcode'],'arity':state['arity'],'operands':[
                                {'dtype':v['dtype'],'source':v['source'],'primary_known':v['source'] in TRUSTED,'value':v['value'] if v['source'] in TRUSTED else None} for v in state['operands']]},
                            'audit':{'identity':trace['identity'],'trace_index':trace['trace_index'],'event_index':i,'instruction_id':identity['instruction_id'],
                                'static_node':staticnode,'source_observations':refs['observations'],'prefix_fingerprint':item['input_sha256'],
                                'two_hop_value_reachable':rv,'two_hop_definition_reachable':rd}}
                if verified!=trace['prefixes']:raise ValueError('Event omitted')
                # Targets are joined ONLY AFTER every prefix input was captured.
                targets=lines(checked(a.source,trace['targets']));validbranches={r['event_index']:r for r in targets if r['B_mask']}
                for task,selected in (('branch',validbranches),('arithmetic',pairs)):
                    for i,target in selected.items():
                        row=captured[i];s=row['pre_state_evidence'];pred,why=semantic_check(s)
                        if task=='branch':actual=target['B']=='jump'
                        else:
                            outs=target['result_values'];actual=outs[0]['value'] if len(outs)==1 else None
                            if set(v['value_id'] for v in s['operands'])!=set(v['value_id'] for v in target['input_values']):raise ValueError('Arithmetic reaching values differ from frozen evidence')
                            if not all(v['observed_at']>s['clock'] for v in outs):raise ValueError('Arithmetic result not later than pre-state')
                        partrows.append(dict(row,task=task,partition=part,target=actual,prediction=pred,abstention_reason=why,consistent=pred==actual if pred is not None else None))
                graph_status.append({'identity':trace['identity'],'partition':part,'events':verified,'identity_verified':verified,
                    'branch_rows':len(validbranches),'arithmetic_rows':len(pairs),'source_graph_modified':False,'evidence':refs['observations']['path']})
                if len(graph_status)%40==0:print('DIAGNOSTIC',part,len(graph_status),flush=True)
            write_json(a.out/f'{part}_summary.json',summarize(partrows))
            with (a.out/f'{part}_local_state.jsonl').open('w',encoding='utf8') as f:
                for r in partrows:emit(f,r)
            allrows.extend(partrows)
    if sum(r['task']=='branch' for r in allrows)!=523 or sum(r['task']=='arithmetic' for r in allrows)!=547:raise ValueError('Frozen denominator changed')
    write_json(a.out/'semantic_checks.json',{'protocol_sha256':frozen,'source_counts':counts,'train':summarize([r for r in allrows if r['partition']=='train']),
        'val':summarize([r for r in allrows if r['partition']=='val']),'identity':dict(identity_counts),'query_two_hops':dict(reach),
        'seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'source_graphs_changed':False,'static_dynamic_edges_added':False,'product_targets_read':False,'test_trajectories_read':False})
    write_json(a.out/'graph_status.json',graph_status)
    with (a.out/'local_state_status.csv').open('w',encoding='utf8',newline='') as f:
        cols=['system_id','bug_version_id','product_id','test_id','partition','trace_index','event_index','task','opcode','operation','arity','primary_known','sources','missing_reasons','prediction','target','consistent','evidence']
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader()
        for r in allrows:
            s=r['pre_state_evidence'];au=r['audit'];writer.writerow(dict(zip(cols[:4],au['identity']),partition=r['partition'],trace_index=au['trace_index'],event_index=au['event_index'],task=r['task'],opcode=s['opcode'],operation=s['operation'],arity=s['arity'],
                primary_known=sum(v['source'] in TRUSTED for v in s['operands']),sources='|'.join(v['source'] for v in s['operands']),missing_reasons='|'.join(v['reason'] or '' for v in s['operands']),
                prediction=r['prediction'],target=r['target'],consistent=r['consistent'],evidence=f"{r['partition']}_local_state.jsonl;trace={au['trace_index']};event={au['event_index']}"))
    print('COMPLETE',len(allrows),dict(identity_counts),flush=True)
if __name__=='__main__':main()
