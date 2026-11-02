"""Bounded CPU-only Email local-state production in the existing container."""
import argparse,csv,json,os,platform,resource,shutil,sys,time
from collections import Counter,defaultdict
from pathlib import Path
from cwm_fp.data.common import sha256,write_json
from cwm_fp.lddg.static import load_static
from cwm_fp.lddg.effects import STATIC_SHA
from cwm_fp.models.local_dataset import (SourceBundle,load,replay_inputs,task_for,target_for,signature,tensor_input,read_input)
from cwm_fp.models.local_state import TRUSTED,semantic_check
from cwm_fp.models.prefix_dataset import suffix_variants

def emit(f,r):f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
def csvout(p,rows):
    with p.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['no_rows']);w.writeheader();w.writerows(rows)

def validate_operands(s,pre):
    checks=0
    if s['current_instruction_executed']:raise AssertionError('Not pre-instruction')
    for v in s['operands']:
        if v['source']=='UNKNOWN':
            if v['value'] is not None:raise AssertionError('Invented unknown')
            continue
        if v['available_at'] is None or v['available_at']>s['clock']:raise AssertionError('Future operand')
        if v['definition_event'] is None or v['definition_event']>=s['event_index']:raise AssertionError('Future/current definition')
        source=pre[v['definition_event']]
        if v['source']=='DIRECT_OBSERVED' and (source['callback_clock']!=v['available_at'] or source['callback_clock']>=s['clock']):raise AssertionError('Wrong callback time')
        if v['source']=='SEMANTICALLY_DERIVED' and (not v['derivation'] or source['completed_clock'] is None or source['completed_clock']>s['clock']):raise AssertionError('Copy before completion')
        checks+=1
    return checks

def aggregate(rows):
    result={}
    for scope in sorted({r['scope'] for r in rows}):
        result[scope]={}
        for task in ('branch','arithmetic','other'):
            rr=[r for r in rows if r['scope']==scope and r['task']==task];c=Counter();src=Counter();why=Counter();ops=defaultdict(Counter)
            for r in rr:
                c['trusted_prefix']+=r['trusted_prefix_available'];c['input_verified']+=r['local_input_verified'];c['target_verified']+=r['local_target_verified'];c['admitted']+=r['local_transition_admitted']
                vs=r['state']['operands'];n=sum(v['source'] in TRUSTED for v in vs)
                c['no_operands' if not vs else 'all_primary_known' if n==len(vs) else 'partial_primary_known' if n else 'all_primary_unknown']+=1
                c['contains_derived']+=any(v['source']=='SEMANTICALLY_DERIVED' for v in vs)
                c['checked']+=r['consistent'] is not None;c['consistent']+=r['consistent'] is True;c['inconsistent']+=r['consistent'] is False
                c['direct_only_complete']+=bool(vs) and all(v['source']=='DIRECT_OBSERVED' for v in vs)
                c['static_added_complete']+=bool(vs) and n==len(vs) and any(v['source']=='STATIC_CONSTANT' for v in vs)
                for v in vs:src[v['source']]+=1
                for reason in r['reasons']:why[reason]+=1
                ops[str(r['state']['opcode'])][str(r['target']['value']) if r['target']['mask'] else 'UNVERIFIED']+=1
            result[scope][task]={'denominator':len(rr),'counts':dict(c),'operand_sources':dict(src),'rejections':dict(why),'opcodes_and_targets':dict(ops),
                'symbolic_coverage':c['checked']/len(rr) if rr else None,'covered_consistency':c['consistent']/c['checked'] if c['checked'] else None,
                'empty_consistency_reason':'no supported trusted input-target pairs' if not c['checked'] else None}
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--anchor',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if not Path('/.dockerenv').exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Existing container CPU-only execution required')
    if a.out.exists():raise ValueError('Never overwrite production')
    a.out.mkdir(parents=True);start=time.monotonic();source=SourceBundle(a.source,a.anchor);framework=load_static(source.checked('framework/static.tsv'))
    if sha256(a.source/'framework/static.tsv')!=STATIC_SHA:raise ValueError('Framework bytecode changed')
    traces=[];rows=[];identity_rows=[];isolation=[];source_checks=[];counts=Counter();first_gaps=Counter();prod=source.protocol['selected']
    for selected in prod:
        static,ids,binding,idreceipt=source.product(selected)
        write_json(a.out/(selected['batch_product']+'_identity.json'),{'identity':selected['identity'],**idreceipt,'identities':list(ids.values())})
        for ti,t in [(i,t) for i,t in enumerate(source.manifest['traces']) if t['batch_product']==selected['batch_product']]:
            if time.monotonic()-start>6200:raise RuntimeError('Producer reserve budget')
            status,results,sourcereasons=source.execution(t,binding);ok=not sourcereasons
            log=source.checked(t['run']+'/instrumented/run.log')
            if log.stat().st_size>33554432:raise RuntimeError('Frozen per-trace size budget')
            text=log.read_text(encoding='utf8');replay=replay_inputs(text,t['identity'][3],static,framework,ids,source_ok=ok)
            observer=replay['observer'];barrier=replay['barrier'];boundary=barrier['clock'] if barrier else None
            if len(observer.full)>20000:raise RuntimeError('Frozen event budget')
            if barrier:first_gaps[barrier['reason']]+=1
            td=a.out/'traces'/f'{ti:03d}';td.mkdir(parents=True)
            with (td/'observations.jsonl').open('w',encoding='utf8') as f:
                for tok in replay['observations']:emit(f,tok)
            states=replay['states'];byevent={s['event_index']:s for s in states};admitted=[]
            for item in states:
                i=item['event_index'];r=observer.full[i];s=item['state'];task=task_for(r['opcode']);nxt=observer.full[i+1] if i+1<len(observer.full) else None
                frame=s['frame_id'];nframe=byevent.get(i+1,{}).get('state',{}).get('frame_id')
                trusted=ok and not replay['parse_error'] and (boundary is None or r['pre_clock']<boundary) and item['identity_verified'] and not s['prior_gaps']
                inputok=trusted and bool(s['operands']) and all(v['source'] in TRUSTED for v in s['operands'])
                target=target_for(r,nxt,static,frame,nframe,boundary)
                # Source failure or parse failure cannot generate certified targets.
                if not ok or replay['parse_error']:target.update(value=None,mask=False,reason='source_or_physical_stream_unverified')
                prediction,why=semantic_check(s) if inputok and task!='other' else (None,'input_not_primary_verified' if task!='other' else 'not_supported_transition_task')
                consistent=(prediction==target['value']) if prediction is not None and target['mask'] else None
                isadmitted=inputok and target['mask'] and consistent is True
                reasons=[]
                if not trusted:reasons.extend(sourcereasons or [barrier['reason'] if barrier else 'untrusted_prefix'])
                if not inputok:reasons.append('local_input_not_primary_verified')
                if not target['mask']:reasons.append(target['reason'])
                if why:reasons.append(why)
                if consistent is False:reasons.append('symbolic_inconsistency')
                q=f'{ti:03d}:{i}';row={'query_id':q,'identity':t['identity'],'partition':'train','trace_index':ti,'event_index':i,'scope':item['scope'],'task':task,
                    'source_execution_verified':ok,'trusted_prefix_available':trusted,'local_input_verified':inputok,'local_target_verified':bool(target['mask']),
                    'local_transition_admitted':bool(isadmitted),'full_product_admission':False,'state':s,'learning_input':item['learning_input'],
                    'target':target,'prediction':prediction,'consistent':consistent,'reasons':sorted(set(reasons)),
                    'evidence':{'source_log':t['run']+'/instrumented/run.log','source_log_sha256':sha256(log),'physical_cut':item['cut'],
                                'occurred_at':r['pre_clock'],'observed_at':r['pre_clock'],'evidence_available_at':r['pre_clock'],
                                'frame_id':frame,'initial_heap_state':'UNKNOWN','dependency_prefix':item['evidence'],
                                'identity_ref':selected['batch_product']+'_identity.json','original_lineinfo':[list(r['domain']),r['bci']]}}
                if trusted:
                    counts['operand_time_checks']+=validate_operands(s,observer.full)
                    for v,e,role in item['evidence']['links']:
                        if e>=i:raise AssertionError('Future dependency use')
                tensor_input(row['learning_input'])
                rows.append(row)
                if isadmitted:admitted.append(i)
            # Every admitted query plus every trajectory's first/middle/last,
            # and all observed call/branch/return/handler boundaries. No hash
            # rejection substitutes for destructive suffix isolation.
            choices=set(admitted)|set(replay['critical'])
            if states:choices.update((0,len(states)//2,len(states)-1))
            for i in sorted(choices):
                item=byevent[i];cut=item['cut'];prefix=text[:cut]
                actual=replay_inputs(prefix,t['identity'][3],static,framework,ids,source_ok=ok)
                expected=signature(item['learning_input']);got=actual['states'][-1]
                if signature(got['learning_input'])!=expected or got['state']!=item['state']:raise AssertionError('Independent truncated replay mismatch')
                for name,suffix in suffix_variants(text[cut:]).items():
                    changed=replay_inputs(prefix+suffix,t['identity'][3],static,framework,ids,source_ok=ok,stop_cut=cut)
                    if signature(changed['states'][-1]['learning_input'])!=expected or changed['states'][-1]['state']!=item['state']:raise AssertionError('Suffix changed pre-state')
                    counts['suffix_mutations']+=1
                isolation.append({'identity':t['identity'],'trace_index':ti,'event_index':i,'cut':cut,'input_sha256':expected,
                    'state_available':item['state']['frame_id'] is not None,'variants':5,'passed':True,'admitted_query':i in admitted})
                counts['truncated_replays']+=1
            for link in replay['links']:identity_rows.append({'identity':t['identity'],'trace_index':ti,**link})
            ts={'identity':t['identity'],'trace_index':ti,'partition':'train','registered_tests':t['registered_tests'],
                'source_execution_verified':ok,'original_result':results['original']['outcome'],'instrumented_result':results['instrumented']['outcome'],
                'physical_instruction_events':len(observer.full),'product_events':sum(x['scope']=='product' for x in replay['links']),
                'identity_verified_events':sum(x['identity_verified'] for x in replay['links']),
                'trusted_event_prefix':sum(x['trusted_prefix_available'] for x in rows if x['trace_index']==ti),
                'trusted_until':barrier,'parse_error':replay['parse_error'],'local_admitted':len(admitted),
                'local_product_admitted':sum(x['local_transition_admitted'] and x['scope']=='product' for x in rows if x['trace_index']==ti),
                'full_product_admission':False,'source_rejections':sourcereasons,'isolation_locations':len(choices),
                'source_log':t['run']+'/instrumented/run.log','source_status':t['status'],
                'opaque_log_records':len(observer.opaque_after_start),'raw_stream_unchanged':True}
            traces.append(ts);write_json(td/'status.json',ts);source_checks.append({'identity':t['identity'],'verified':ok,'reasons':sourcereasons})
            print('TRACE',ti,t['identity'][3],len(observer.full),'trusted',ts['trusted_event_prefix'],'admitted',len(admitted),'gap',barrier['reason'] if barrier else None,flush=True)
    with (a.out/'inputs.jsonl').open('w',encoding='utf8') as inf,(a.out/'targets.jsonl').open('w',encoding='utf8') as tf,(a.out/'evidence.jsonl').open('w',encoding='utf8') as ef:
        for r in rows:
            emit(inf,{'query_id':r['query_id'],'learning_input':r['learning_input']})
            emit(tf,{'query_id':r['query_id'],**r['target'],'local_transition_admitted':r['local_transition_admitted']})
            emit(ef,{k:v for k,v in r.items() if k not in {'learning_input','target','prediction','consistent'}})
    with (a.out/'event_of_static_instruction.jsonl').open('w',encoding='utf8') as f:
        for r in identity_rows:emit(f,r)
    write_json(a.out/'trace_status.json',traces);write_json(a.out/'isolation_checks.json',isolation)
    manifest={'schema':'local_transition_dataset_v1','source_manifest_sha256':a.anchor,'protocol_sha256':source.manifest['protocol_sha256'],
        'queries':len(rows),'inputs':{'path':'inputs.jsonl','sha256':sha256(a.out/'inputs.jsonl')},
        'targets':{'path':'targets.jsonl','sha256':sha256(a.out/'targets.jsonl')},'evidence':{'path':'evidence.jsonl','sha256':sha256(a.out/'evidence.jsonl')}}
    write_json(a.out/'manifest.json',manifest);anchor=sha256(a.out/'manifest.json')
    # Rename the target file temporarily in a separate copy, proving the public
    # input reader does not open it. The real frozen targets remain untouched.
    readerroot=a.out/'reader_probe';readerroot.mkdir();shutil.copyfile(a.out/'inputs.jsonl',readerroot/'inputs.jsonl');shutil.copyfile(a.out/'manifest.json',readerroot/'manifest.json')
    for r in rows:
        tensors=read_input(readerroot,r['query_id'],expected_manifest_sha256=anchor)
        if tensors['value'].dtype.name!='int64':raise AssertionError('Precision lost')
    write_json(a.out/'input_reader_probe.json',{'queries':len(rows),'different_root':str(readerroot),'target_file_absent':True,'input_reads_passed':len(rows),'manifest_sha256':anchor})
    statusrows=[]
    for r in rows:
        s=r['state'];statusrows.append({**dict(zip(('system_id','bug_version_id','product_id','test_id'),r['identity'])),
            **{k:r[k] for k in ('partition','trace_index','event_index','query_id','scope','task','source_execution_verified','trusted_prefix_available','local_input_verified','local_target_verified','local_transition_admitted','full_product_admission')},
            'opcode':s['opcode'],'operation':s['operation'],'operand_count':s['arity'],'sources':'|'.join(v['source'] for v in s['operands']),
            'primary_known':sum(v['source'] in TRUSTED for v in s['operands']),'consistent':r['consistent'],'rejection_codes':'|'.join(r['reasons']),
            'evidence':'evidence.jsonl#'+r['query_id'],'source_log':r['evidence']['source_log']})
    csvout(a.out/'local_state_status.csv',statusrows)
    csvout(a.out/'trace_status.csv',[{**dict(zip(('system_id','bug_version_id','product_id','test_id'),t['identity'])),
        **{k:v for k,v in t.items() if k not in {'identity','trusted_until','parse_error','source_rejections'}},
        'rejection_codes':'|'.join(t['source_rejections']) if t['source_rejections'] else (t['trusted_until']['reason'] if t['trusted_until'] else ''),
        'trusted_until':json.dumps(t['trusted_until'],ensure_ascii=False),'evidence':f"traces/{t['trace_index']:03d}/status.json"} for t in traces])
    checks={'schema':'phase05C2_checks_v1','source_manifest_sha256':a.anchor,'protocol_sha256':source.manifest['protocol_sha256'],
        'source_execution':source_checks,'summary':aggregate(rows),'first_trust_gaps':dict(first_gaps),'checks':dict(counts),
        'input_reader':load(a.out/'input_reader_probe.json'),'identity_links':{'events':len(identity_rows),'verified':sum(r['identity_verified'] for r in identity_rows)},
        'resources':{'seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                     'free_bytes':shutil.disk_usage(a.out).free,'python':sys.version,'python_path':sys.executable,'platform':platform.platform(),
                     'container_id':platform.node(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu_used':False},
        'no_new_Java_runs':True,'no_model_training':True,'full_product_admission_changed':False}
    write_json(a.out/'checks.json',checks);write_json(a.out/'source_access.json',{'files':sorted(set(source.access)),'labels_read':False,'selected_traces_only':True})
    print(json.dumps({'traces':len(traces),'queries':len(rows),'summary':checks['summary'],'checks':dict(counts),'seconds':checks['resources']['seconds']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
