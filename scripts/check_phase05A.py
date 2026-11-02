"""Independent raw-record/old-graph verifier, separate from target production.

Reads ONLY the selected train/val source files. Does not trust producer-written
availability fields, counters, final frame guesses, or success flags alone.
"""
from pathlib import Path
import sys,time,re
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cwm_fp.models.prefix_dataset import OUT,PACKAGE,ANCHOR,FRAMEWORK,load,lines,guard,budget_unchecked
from cwm_fp.models.prefix_observer import ObservationReplay,PrefixBuilder,tokens
from cwm_fp.lddg.trace import parse_trace,observed_scalar
from cwm_fp.lddg.static import load_static
from cwm_fp.data.common import sha256,write_json
from cwm_fp.models.graph_package import digest

def main():
    start=time.monotonic();p=load(OUT/'production.json');access=guard(load(PACKAGE/'manifest.json'),True)
    framework=load_static(FRAMEWORK);result=[];counter=Counter();graph_cache={}
    for row in p['status']:
        if row['status']!='PASS':continue
        d=Path(row['evidence_path']);source=load(d/'source.json');path=Path(source['trace_path'])
        assert sha256(path)==source['trace_sha256'];text=path.read_text(encoding='utf8')
        assert sha256(Path(source['static_path']))==source['static_sha256']==sha256(d/'static.tsv')
        s=load_static(d/'static.tsv');bundle=load(d/'static.json');builder=PrefixBuilder(s,bundle,framework)
        if row['source_graph_path'] not in graph_cache:
            graph_cache={row['source_graph_path']:{g['test_id']:g for g in lines(row['source_graph_path'])}}
        g=graph_cache[row['source_graph_path']][row['test_id']];assert digest(g)==source['source_graph_sha256']
        full=parse_trace(path,row['test_id'],physical_evidence=True);projected=[r for r in full if r['domain'][0]!='org.evosuite.runtime.EvoAssertions']
        old_values={n['id']:n['attrs'] for n in g['nodes'] if n['type']=='value'}
        old_event={n['attrs']['event_index']:n['attrs'] for n in g['nodes'] if n['id'].startswith('e:')}
        old_edges={(e['value_id'],e['definition_event'],e['event_index'],e['role']) for e in g['edges'] if e['type']=='data_dep'}
        indices={tuple(r['clock']):r for r in lines(d/'prefix_index.jsonl')}
        certified=g['control_flow_evidence']['assessment'].get('certified_transfers',[])
        checks=lines(d/'checks.jsonl');expected_checks={tuple(r['clock']):r for r in checks}
        obs=ObservationReplay(row['test_id']);observed_callbacks={};checked=Counter();latest= None
        for t in tokens(text):
            obs.push(t)
            if tuple(t['clock']) not in expected_checks:continue
            snap=builder.build(obs);latest=snap
            assert snap.fingerprint()==expected_checks[tuple(t['clock'])]['tensor_sha256']
            now=[r for r in obs.full if r['domain'][0]!='org.evosuite.runtime.EvoAssertions']
            assert len(snap.model['event_nodes'])==len(now)
            for i,r in enumerate(now):
                final=projected[i];assert (r['domain'],r['bci'],r['opcode'])==(final['domain'],final['bci'],final['opcode'])
                if 182<=r['opcode']<=186:
                    assert r.get('call_returned')==(bool(final['call_returned']) and final.get('return_log_line',10**12)<=t['clock'][0])
                if r['fields'].get('stack'):
                    # Real callback must physically be to the left of the cut.
                    original=text.splitlines()[r['raw_log_line']-1]
                    assert ',stack='+r['fields']['stack'] in original
                    assert r['raw_log_line']<t['clock'][0] or t['clock'][1]==1
                assert snap.audit['events'][i]['frame'] in (None,old_event[i]['frame_id'])
            for v in snap.audit['values']:
                original=old_values[v['value_id']]
                assert v['definition_event']==original['definition_event']
                if v['observed']:
                    r=now[v['definition_event']];obsval=observed_scalar(r['fields'].get('stack'))
                    assert obsval['observed'] and obsval['value']==v['value']==original['value']
                    # Refuse any hidden loss from converting an observed long.
                    j=v['local_node'];assert snap.model['value'][j,0]==v['value']
                    if isinstance(v['value'],int):assert int(snap.model['value'][j,0])==v['value']
                else:assert v['value'] is None
            for e in snap.audit['dynamic_dependencies']:
                assert (e['value_id'],e['definition_event'],e['use_event'],e['role']) in old_edges
                assert now[e['use_event']]['completed_clock'] is not None
            for transfer in snap.audit['exception_transfers']:
                assert transfer in certified
                assert now[transfer['target_event']]['pre_clock']<=t['clock']
                checked['exception_after_handler']+=1
            for transfer in certified:
                if len(now)<=transfer['target_event']:
                    assert all(v['value_id']!=transfer['exception_value_id'] for v in snap.audit['values'])
                    checked['exception_withheld_before_handler']+=1
            for effect in snap.audit['completed_effects']:
                assert effect['return_log_line']<=t['clock'][0]
                checked['effect_after_return']+=1
            if t['kind']=='instruction':
                if now[-1]['pre_clock']==t['clock']:
                    i=len(now)-1;assert not any(v['definition_event']==i for v in snap.audit['values'])
                    assert not snap.model['value_observed'][snap.model['event_nodes'][-1],0]
                    checked['instruction_before_callback']+=1
                    if 182<=now[-1]['opcode']<=186:assert i in snap.audit['pending_calls'];checked['call_entry_pending']+=1
            if t['kind']=='callback':checked['callback_positions']+=1
            if t['kind']=='return_marker':checked['return_marker_positions']+=1
            if any(r['domain'][0]=='org.evosuite.runtime.EvoAssertions' for r in obs.full):
                for effect in g.get('framework_effects',[]):
                    if effect['return_log_line']>t['clock'][0]:
                        assert all(e['event_index']!=effect['event_index'] for e in snap.audit['completed_effects'])
                        checked['framework_withheld_before_return']+=1
            assert not snap.model['x'][:,3:5].any()  # no final assertion/output feature
            if tuple(t['clock']) in indices:assert indices[tuple(t['clock'])]['input_sha256']==snap.fingerprint()
            checked['all_physical_positions']+=1
        target=lines(d/'targets.jsonl')
        assert len(target)==len(projected)==len(indices)
        for i,tr in enumerate(target):
            if tr['A_mask']:assert tr['A_available_at']>tr['anchor']
            if tr['B_mask']:
                assert old_event[i]['frame_id']==old_event[i+1]['frame_id']
                r=projected[i];nxt=projected[i+1];inst=s['methods'][r['domain']][r['bci']]
                assert nxt['bci'] in inst['successors'] and len(inst['successors'])==2
                assert tr['B']==('fallthrough' if nxt['bci']==inst['next'] else 'jump')
                checked['independent_branch_frame_target']+=1
        row_result={'identity':source['identity'],'partition':row['partition'],'checks':dict(checked),
            'final_dependency_subset_count':len(latest.audit['dynamic_dependencies']),
            'old_data_dependency_count':len(old_edges),'all_checks_passed':True}
        result.append(row_result);counter.update(checked)
        if len(result)%30==0:print('INDEPENDENT',len(result),flush=True)
    write_json(OUT/'independent_validation.json',{'status':'PASS','traces':len(result),'checks':dict(counter),'rows':result,
        'seconds':time.monotonic()-start,'resources':budget_unchecked(),'source_access':access})
    print('INDEPENDENT_PASS',len(result),dict(counter),flush=True)

if __name__=='__main__':main()
