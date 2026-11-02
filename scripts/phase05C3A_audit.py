"""Actual frozen-Email field evidence, prefix boundaries and no-upgrade audit."""
import argparse,copy,csv,json,os,resource,shutil,time
from collections import Counter
from pathlib import Path
from cwm_fp.data.common import sha256,write_json
from cwm_fp.lddg.static import load_static
from cwm_fp.lddg.trace import Dataflow,FRAMEWORK_PROFILE
from cwm_fp.models.local_dataset import (SourceBundle,LocalObservationReplay,load,replay_inputs,
    identity_check,prefix_probe,task_for,signature)
from cwm_fp.models.prefix_observer import tokens
from cwm_fp.models.prefix_dataset import suffix_variants
from cwm_fp.models.field_evidence import classify_read,read_value,visible_read

OLD=Path('/workspace/codes/wanghaining_cwm_fp/phase05C2_v1')
ANCHOR='23e96a1bfc51a218b7ba50a1840c27a14e1d6cad9c1ad36aa2a45dbd498b0b01'

def csvout(p,rows):
    with p.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in r.items()})

def observer_until(text,test,cut):
    ob=LocalObservationReplay(test)
    for t in tokens(text):
        if t['cut']>cut:break
        ob.push(t)
    return ob

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',required=True);a=p.parse_args()
    if not Path('/.dockerenv').exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Container CPU required')
    out=a.root/a.out
    if out.exists():raise ValueError('Never overwrite audit')
    out.mkdir();start=time.monotonic();src=SourceBundle(OLD/'inputs',ANCHOR);framework=load_static(src.checked('framework/static.tsv'))
    protocol=load(a.root/'transfer/protocol.json')
    if src.protocol['selected']!=protocol['selected']:raise ValueError('Changed cohort')
    runtime=load(a.root/'runtime_code_v1/runtime_readiness.json')
    if not runtime['build_source_and_library_closure_available']:raise ValueError('Emitter source evidence closure missing')
    oldchecks=load(OLD/'production_code_v2/checks.json');oldtraces=load(OLD/'production_code_v2/trace_status.json')
    oldinputs={r['query_id']:r for r in (json.loads(s) for s in (OLD/'production_code_v2/inputs.jsonl').read_text().splitlines())}
    oldhashes={str(f.relative_to(OLD)):sha256(f) for f in [OLD/'inputs/source_manifest.json',OLD/'production_code_v2/manifest.json',
         OLD/'production_code_v2/inputs.jsonl',OLD/'production_code_v2/targets.jsonl',OLD/'production_code_v2/checks.json']}
    first=[];allfields=[];traces=[];isolation=[];strict_equal=0;event_counts=Counter();candidate_counts=Counter();prefix_proofs=[]
    producer=a.root/'transfer/runtime/producer-src/main/java/ppfl/instrumentation'
    source_quotes={}
    markers={'TraceTransformer.java':['oi.form == 178','!mi.isStaticInitializer()'],
       'CallBackIndex.java':['identityHashCode(i)','writeStack(",stack=Obj:','writeStack(",stack=I:'],
       'opcode/GetStaticInst.java':['###RET@','insertByteCodeAfter'],
       'opcode/GetFieldInst.java':['dispatchByDesc','insertByteCodeAfter'],
       'opcode/OpcodeInst.java':["case 'L':",'return cbi.tsindex_object;']}
    for name,needles in markers.items():
        path=producer/name;lines=path.read_text(encoding='utf8').splitlines()
        source_quotes[name]={'sha256':sha256(path),'path':str(path),'lines':[{'line':i+1,'text':s} for i,s in enumerate(lines) if any(n in s for n in needles)]}
    write_json(out/'emitter_evidence.json',{'agent_sha256':runtime['producer_agent_sha256'],'agent_classes_checked':runtime['agent_classes_checked'],
        'sources':source_quotes,'note':'original build-source hashes and actual agent classes checked; JVM/EvoSuite runtime transforms not re-executed'})
    product_cache={}
    for sel in src.protocol['selected']:
        product_cache[sel['batch_product']]=src.product(sel)
    for ti,t in enumerate(src.manifest['traces']):
        static,ids,binding,receipt=product_cache[t['batch_product']]
        status,results,reasons=src.execution(t,binding)
        if reasons:raise ValueError('Frozen source execution not verified')
        log=src.checked(t['run']+'/instrumented/run.log');text=log.read_text(encoding='utf8')
        rep=replay_inputs(text,t['identity'][3],static,framework,ids);raw=rep['observer'].full;barrier=rep['barrier']
        old=oldtraces[ti]
        if rep['parse_error'] or len(raw)!=old['physical_instruction_events'] or signature(barrier)!=signature(old['trusted_until']):raise AssertionError('Strict event/barrier changed')
        for s in rep['states']:
            if s['learning_input']!=oldinputs[f'{ti:03d}:{s["event_index"]}']['learning_input']:raise AssertionError('Strict input changed')
            strict_equal+=1;event_counts[s['scope']]+=1;candidate_counts[(s['scope'],task_for(s['state']['opcode']))]+=1
        toklist=list(tokens(text));tok_by_clock={tuple(x['clock']):x for x in toklist}
        first_index=barrier['event_index'] if barrier and barrier['reason']=='field_read_without_observed_definition' else None
        for i,r in enumerate(raw):
            if r['opcode'] not in (178,180):continue
            ok,pid,inst=identity_check(r,static,ids);member=static['fields'].get((*r['domain'],r['bci']))
            if not member:raise AssertionError('Missing field descriptor')
            scope=pid.get('scope') if pid else 'unresolved'
            # Only prefix-valid reads receive the existing interpreter's frame/stack witness.
            prevalid=barrier is None or i<=barrier['event_index']
            earlier=[];receiver=None;frame=None;frame_valid=False;receiver_valid=r['opcode']==178;prior_write=None;flow=None;prestate=None
            if prevalid:
                preob=observer_until(text,t['identity'][3],rep['cuts'][i])
                prestate,flow,gaps=prefix_probe(preob,static,framework,ids)
                earlier=[g for g in gaps if g['event_index']<i];frame=prestate['frame_id'];frame_valid=not earlier and frame is not None
                if r['opcode']==180 and flow and flow.frames and flow.frames[-1].stack:
                    v=flow.frames[-1].stack[-1]
                    receiver={'value_id':v.id,'definition_event':v.definition,'allocation_token':v.ref,'certain':v.certain,
                         'evidence':'actual NEW + stack/local/call propagation; never Obj hash'}
                    receiver_valid=bool(v.certain and v.ref and v.ref.startswith('allocation:') and v.ref not in flow.escaped)
                ref=receiver['allocation_token'] if receiver else 'static'
                prior=flow.heap.get((ref,member)) if flow else None
                if prior:prior_write={'value_id':prior.id,'definition_event':prior.definition,'certain':prior.certain}
            else:earlier=[barrier]
            assoc=True;callback=r['fields'].get('stack');complete=r['completed_clock']
            if r['opcode']==178:
                assoc=r.get('completion_marker_kind')=='static_field_access_completed_no_value' and callback is None
            else:
                assoc=callback is not None and r['callback_clock']==[r['pre_clock'][0],1]
            ev=classify_read(opcode=r['opcode'],descriptor=member[2],raw_callback=callback,pre_clock=r['pre_clock'],
                 callback_clock=r['callback_clock'],completed_clock=complete,identity_valid=ok,frame_stack_valid=frame_valid,
                 receiver_valid=receiver_valid,association_valid=assoc,earlier_gaps=earlier,write_provenance_complete=bool(prior_write))
            following=raw[i+1] if i+1<len(raw) else None;use=None
            if following and prevalid:
                # Evidence-only consumption witness. Does not make an uncertain value certain or emit a DDG edge.
                f2=Dataflow(static,semantics_profile=FRAMEWORK_PROFILE).run(copy.deepcopy(raw[:i+2]))
                defined=[v.id for v in f2.values if v.definition==i]
                consumed=set(defined)&set(f2.events[i+1]['use_ids'])
                extra=[g for g in f2.gaps if g['reason'] not in {'unterminated_frame','field_read_without_observed_definition'}]
                use={'event_index':i+1,'opcode':following['opcode'],'clock':following['pre_clock'],
                    'consumed_read_occurrence_ids':sorted(consumed),'stack_occurrence_link_verified':bool(consumed) and not extra,
                    'value_transition_certified':False,'reason':'read payload unqualified' if not ev['observed_value_valid'] else 'separate local chain certification required',
                    'other_gaps':extra}
            row={**dict(zip(('system_id','bug_version_id','product_id','test_id'),t['identity'])),
                'trace_index':ti,'event_index':i,'scope':scope,'opcode':r['opcode'],'field_owner':member[0],
                'field_name':member[1],'field_descriptor':member[2],'frame_id':frame,'pre_clock':r['pre_clock'],
                'completed_clock':complete,'callback_clock':r['callback_clock'],'raw_callback':callback,
                'completion_association_verified':assoc and frame_valid and ok and receiver_valid,
                'callback_association_verified':callback is not None and assoc and frame_valid and ok and receiver_valid,'receiver':receiver,
                'prior_write':prior_write,'earlier_gaps':earlier,'first_use':use,
                'class_initialization':'getstatic may trigger natural class init; emitter omits clinit runtime instructions; no complete initialization proof' if r['opcode']==178 else 'no class-init transition at this instance read; setup/default writes not observed',
                'observed_thread_ids_before_read':sorted({x['thread_id'] for x in raw[:i+1]}),
                'exception_at_read_observed':False if complete and assoc and frame_valid else None,
                'external_setup_or_concurrent_effects_certified':False,
                'source_log':'../phase05C2_v1/inputs/'+t['run']+'/instrumented/run.log',
                'source_log_sha256':sha256(log),'raw_physical_line':r['raw_log_line'],'static_instruction_id':pid.get('instruction_id') if pid else None,
                'emitter_evidence':'emitter_evidence.json','is_first_field_gap':i==first_index,**ev}
            allfields.append(row)
            if i!=first_index:continue
            first.append(row);prefix_proofs.append({'identity':t['identity'],'event_index':i,'pre_state':prestate,
                'receiver':receiver,'frame':frame,'prior_gaps':earlier,'prior_write':prior_write,'first_use':use})
            # Actual physical reads are checked before and after completion independently.
            pretoken=tok_by_clock[tuple(r['pre_clock'])];comptoken=tok_by_clock[tuple(complete)]
            pre=observer_until(text,t['identity'][3],pretoken['cut']);post=observer_until(text,t['identity'][3],comptoken['cut'])
            if pre.full[i]['fields'].get('stack') is not None or pre.full[i]['callback_clock'] is not None:raise AssertionError('Future read value before callback')
            if post.full[i]['fields'].get('stack')!=callback or post.full[i]['completed_clock']!=complete:raise AssertionError('Completion pairing changed')
            checks={'identity':t['identity'],'event_index':i,'pre_cut':pretoken['cut'],'completed_cut':comptoken['cut'],
                'pre_value_absent':True,'post_raw_callback_equal':True,'suffix_variants':0,'deleted_callback_rejected':None,
                'full_input_suffix_checks':0,'class_A_transition_checks':None,'class_A_reason':'no actual A-class read certified'}
            for cut in (pretoken['cut'],comptoken['cut']):
                expected=observer_until(text,t['identity'][3],cut).full
                for name,suffix in suffix_variants(text[cut:]).items():
                    got=observer_until(text[:cut]+suffix,t['identity'][3],cut).full
                    if got!=expected:raise AssertionError('Future suffix altered observed field evidence')
                    checks['suffix_variants']+=1
            # Compare actual strict pre-state under suffix mutations too, not a hash-rejection shortcut.
            expected=rep['states'][i]
            for name,suffix in suffix_variants(text[pretoken['cut']:]).items():
                got=replay_inputs(text[:pretoken['cut']]+suffix,t['identity'][3],static,framework,ids,stop_cut=pretoken['cut'])['states'][-1]
                if got['learning_input']!=expected['learning_input'] or got['state']!=expected['state']:raise AssertionError('Prefix input affected by suffix')
                checks['full_input_suffix_checks']+=1
            if callback is not None:
                callbacktok=tok_by_clock[tuple(r['callback_clock'])]
                begin=callbacktok['cut']-len(callbacktok['text'])
                altered=text[:begin]+text[callbacktok['cut']:]
                deleted=LocalObservationReplay(t['identity'][3])
                for tok in tokens(altered):deleted.push(tok)
                if deleted.full[i]['fields'].get('stack') is not None:raise AssertionError('Deleted callback reappeared')
                absent=classify_read(opcode=r['opcode'],descriptor=member[2],raw_callback=None,pre_clock=r['pre_clock'],callback_clock=None,
                    completed_clock=deleted.full[i]['completed_clock'],identity_valid=ok,frame_stack_valid=frame_valid,
                    receiver_valid=receiver_valid,association_valid=True,earlier_gaps=earlier)
                if absent['observed_value_valid']:raise AssertionError('Missing callback accepted')
                checks['deleted_callback_rejected']=True
            isolation.append(checks)
        traces.append({'identity':t['identity'],'trace_index':ti,'source_execution_verified':True,'events':len(raw),
             'strict_barrier_unchanged':True,'strict_product_admitted':old['local_product_admitted'],
             'first_gap':barrier,'field_first_gap':first_index is not None,'full_product_admission':False})
        print('TRACE',ti,t['identity'][3],len(raw),'field_first',first_index,flush=True)
    if len(first)!=15:raise AssertionError('Original field denominator changed')
    cats=Counter(r['category'] for r in first)
    if cats['A']:
        raise RuntimeError('Actual A evidence found: conditional local mode requires separate implementation and verification before report')
    for rel,h in oldhashes.items():
        if sha256(OLD/rel)!=h:raise AssertionError('Frozen old result changed')
    for rel in src.files:src.checked(rel)
    csvout(out/'field_evidence.csv',first);csvout(out/'all_field_reads.csv',allfields)
    write_json(out/'field_evidence.json',first);write_json(out/'prefix_witnesses.json',prefix_proofs)
    write_json(out/'boundary_checks.json',isolation);write_json(out/'trace_status.json',traces)
    # No fake dataset is manufactured when the protocol's A-class gate is empty.
    candidate_rows=[]
    for line in (OLD/'production_code_v2/evidence.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['task'] not in ('branch','arithmetic'):continue
        candidate_rows.append({'query_id':r['query_id'],'identity':r['identity'],'scope':r['scope'],'task':r['task'],
            'strict_admitted':r['local_transition_admitted'],'observation_conditioned_admitted':False,
            'conditioned_mode_status':'NOT_ACTIVATED_NO_A_CLASS_ANCHOR','evidence':'../phase05C2_v1/production_code_v2/evidence.jsonl#'+r['query_id']})
    write_json(out/'candidate_admission.json',candidate_rows)
    checks={'stage_id':'05C3A','field_denominator':15,'first_field_categories':{k:cats[k] for k in 'ABCDE'},
      'first_field_reasons':dict(Counter(r['reason'] for r in first)),
      'first_field_scope':dict(Counter(r['scope'] for r in first)),
      'all_field_reads':len(allfields),'all_field_categories':dict(Counter(r['category'] for r in allfields)),
      'all_field_raw_scalar_callbacks':sum(read_value(r['raw_callback'],r['field_descriptor'])['valid'] for r in allfields),
      'strict_input_rows_equal':strict_equal,'event_counts':dict(event_counts),
      'candidate_counts':{'|'.join(k):v for k,v in candidate_counts.items()},
      'strict_product_admitted':sum(t['strict_product_admitted'] for t in traces),'conditioned_mode_admitted':0,
      'mode_status':'NOT_ACTIVATED_NO_A_CLASS_ANCHOR','covered_consistency':None,
      'consistency_reason':'no certified read-value anchor; no new input-target pairs',
      'boundary_checks':len(isolation),'physical_suffix_checks':sum(r['suffix_variants'] for r in isolation),
      'pre_input_suffix_checks':sum(r['full_input_suffix_checks'] for r in isolation),
      'callback_deletions_checked':sum(r['deleted_callback_rejected'] is True for r in isolation),
      'upstream_write_complete':False,'current_heap_complete':False,'full_product_admission_changed':False,
      'initial_state_recollected':False,'model_trained':False,'java_runs':0,
      'original_result_hashes_unchanged':oldhashes,'source_files_checked':len(src.files),
      'resources':{'seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
          'free_bytes':shutil.disk_usage(a.root).free,'container':os.uname().nodename}}
    write_json(out/'checks.json',checks);write_json(out/'source_access.json',{'files':sorted(set(src.access)),
       'labels_read':False,'only_24_selected_logs':True,'email_val_test_expanded':False})
    print(json.dumps(checks,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
