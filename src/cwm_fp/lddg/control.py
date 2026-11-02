"""Certify observed paths, never infer catch from astore or invent execution.

The instrumenter is reused, including original BCIs, entries and RET@ records.
Exception tables come from original class bytes. Legacy normal_path_v1 remains
unchanged; opt-in null_field_exception_v1 independently replays one exact subset.
"""
from __future__ import annotations

from .static import identity_digest

CONTROL_SCHEMA = 'normal_path_v1'
RETURNS = set(range(172, 178))
INVOKES = set(range(182, 187))


def is_assertion(target):
    return (len(target) == 3 and target[0] in {'org.junit.Assert', 'junit.framework.Assert'}
            and target[1].startswith(('assert', 'fail')))


def replay_normal_path(method_records, events, complete, outcome, failure_types=(), *, null_field_transfers=None):
    """Independent replay of instruction adjacency and call/return nesting.

    Without independently replayed null-field proofs, only normal paths pass.
    B locates an uncertified transfer; C denotes ambiguous/incomplete evidence.
    """
    errors=[];transitions=[];handler_visits=[];frames=[];seen_frames=set();executed_handlers=[]
    verified={t['target_event']:t for t in (null_field_transfers or [])}
    aborted={i for t in verified.values() for i in t['aborted_calls']};certified=[]
    methods={};terminal=None
    def fail(index,reason):errors.append({'event_index':index,'reason':reason})
    for m in method_records:
        d=tuple(m['domain']);instructions={x['bci']:x for x in m['instructions']}
        if d in methods or len(instructions)!=len(m['instructions']):fail(-1,'duplicate_static_method_or_instruction')
        handlers=m['handlers'];bounds=set(instructions)|{m['code_length']}
        if (m['handler_count']!=len(handlers) or [h['table_index'] for h in handlers]!=list(range(len(handlers)))
            or any(not (h['start'] in instructions and h['end'] in bounds and h['start']<h['end']
                        and h['handler'] in instructions and isinstance(h['catch_type'],str) and h['catch_type']) for h in handlers)):
            fail(-1,'incomplete_or_invalid_ExceptionTable')
        if not instructions or any(s not in instructions and s!=-1 for x in instructions.values() for s in x['successors']):
            fail(-1,'incomplete_normal_CFG')
        methods[d]=(m,instructions)
    if not complete:fail(-1,'incomplete_execution_evidence')
    if not events or [e['event_index'] for e in events]!=list(range(len(events))):fail(-1,'missing_or_reordered_events')
    if len({e['thread_id'] for e in events})!=1:fail(-1,'multiple_threads_without_happens_before')
    for e in events:
        d=tuple(e['domain'])
        if d in methods and methods[d][0]['handlers'] and list(d) not in executed_handlers:executed_handlers.append(list(d))
    observed_handler_pcs=[{'event_index':e['event_index'],'domain':e['domain'],'bci':e['bci']}
        for e in events if tuple(e['domain']) in methods
        and any(h['handler']==e['bci'] for h in methods[tuple(e['domain'])][0]['handlers'])]
    exceptional_transition=False;checked_count=0
    for i,e in enumerate(events):
        checked_count+=1
        d=tuple(e['domain']);item=methods.get(d)
        if not item:fail(i,'missing_original_method_evidence');break
        m,rows=item;inst=rows.get(e['bci'])
        if inst is None or inst['opcode']!=e['opcode']:fail(i,'opcode_or_location_mismatch');break
        if m['handlers'] and list(d) not in executed_handlers:executed_handlers.append(list(d))
        matching=[h for h in m['handlers'] if h['handler']==e['bci']]
        transfer=verified.get(i)
        if transfer is not None:
            target=next((j for j,f in enumerate(frames) if f['id']==transfer['target_frame']),None)
            if (not matching or target is None or not frames or frames[-1]['id']!=transfer['source_frame']
                or transfer['source_event']!=i-1 or frames[-1]['last']['event_index']!=i-1
                or frames[target]['domain']!=d or transfer['handler'] not in matching
                or [f['id'] for f in reversed(frames[target+1:])]!=transfer['unwound_frames']
                or frames[target]['last']['event_index']!=transfer['protected_event']):
                fail(i,'exception_proof_frame_or_handler_mismatch');break
            frames=frames[:target+1];frames[-1]['last']=None
            certified.append(transfer);exceptional_transition=True
            handler_visits.append({'event_index':i,'domain':list(d),'handler':e['bci'],
                'protected_event':transfer['protected_event'],'matching_handlers':[transfer['handler']],
                'normal_predecessor_possible':False,'exception_object_and_unwinding_certified':True})
        elif matching:
            # Locate the protected active instruction, including a suspended caller.
            active=next((f for f in reversed(frames) if f['domain']==d),None)
            source=active['last'] if active else None
            protected=[h for h in matching if source is not None and h['start']<=source['bci']<h['end']]
            normal_possible=bool(source and e['bci'] in rows[source['bci']]['successors'])
            exceptional_transition=bool(protected and not normal_possible)
            handler_visits.append({'event_index':i,'domain':list(d),'handler':e['bci'],
                'protected_event':source['event_index'] if source else None,'matching_handlers':protected,
                'normal_predecessor_possible':normal_possible,'exception_object_and_unwinding_certified':False})
            fail(i,'observed_handler_transition_not_certified' if exceptional_transition else 'ambiguous_handler_entry')
            break
        if e['entry']:
            if e['frame_id'] in seen_frames or e['bci']!=min(rows):fail(i,'invalid_or_reused_invocation_entry');break
            if frames:
                caller=frames[-1]['last'];caller_rows=methods[frames[-1]['domain']][1]
                target=caller_rows[caller['bci']].get('call_target',[])
                if caller['opcode'] not in INVOKES or target!=list(d) or not (caller['call_returned'] or caller['event_index'] in aborted):
                    fail(i,'unmatched_call_or_missing_normal_return_marker');break
                transitions.append({'source_event':caller['event_index'],'target_event':i,'kind':'traced_call'})
            elif i!=0:fail(i,'unexpected_new_root_frame');break
            frames.append({'id':e['frame_id'],'domain':d,'last':None});seen_frames.add(e['frame_id'])
        if not frames or (frames[-1]['id'],frames[-1]['domain'])!=(e['frame_id'],d):
            fail(i,'unmatched_or_exceptional_frame_transition');break
        frame=frames[-1];previous=frame['last']
        if previous is not None:
            prev=rows[previous['bci']]
            if e['bci'] not in prev['successors']:
                fail(i,'non_normal_CFG_transition');break
            if previous['opcode'] in INVOKES and not previous['call_returned']:
                fail(i,'missing_normal_call_return_marker');break
            transitions.append({'source_event':previous['event_index'],'target_event':i,
                                'kind':'normal_call_resume' if previous['opcode'] in INVOKES else 'normal_CFG'})
        frame['last']=e
        if e['opcode'] in INVOKES:
            target=inst.get('call_target',[])
            if not target or target!=e.get('call_target'):fail(i,'missing_or_mismatched_call_target');break
            next_event=events[i+1] if i+1<len(events) else None
            enters=bool(next_event and next_event['entry'] and next_event['domain']==target)
            if not enters and not e['call_returned']:
                if (i==len(events)-1 and len(frames)==1 and is_assertion(target) and outcome=='FAIL' and complete
                    and list(failure_types)==['java.lang.AssertionError']):
                    terminal={'event_index':i,'kind':'observed_external_assertion_failure','exception_propagation_certified':False}
                else:fail(i,'unobserved_call_completion_or_exception');break
        if e['opcode']==191:fail(i,'athrow_exception_object_and_unwinding_unsupported');break
        if e['opcode'] in RETURNS:
            frames.pop()
            if frames and not frames[-1]['last']['call_returned']:
                fail(i,'normal_return_without_observed_return_marker');break
            if i+1<len(events):
                nxt=events[i+1]
                if not frames or nxt['entry'] or nxt['frame_id']!=frames[-1]['id']:
                    fail(i,'normal_return_does_not_resume_caller');break
                transitions.append({'source_event':i,'target_event':i+1,'kind':'normal_return',
                                    'caller_event':frames[-1]['last']['event_index']})
            elif frames:fail(i,'trace_ends_inside_callee')
            else:terminal={'event_index':i,'kind':'normal_root_return','exception_propagation_certified':False}
    if frames and terminal is None and not errors:fail(len(events)-1,'unterminated_invocation')
    if len(certified)!=len(verified):fail(-1,'unconsumed_exception_transfer_proof')
    accepted=not errors and terminal is not None
    result={'accepted':accepted,'classification':('NPE_FIELD_CERTIFIED' if certified else 'A' if executed_handlers else 'NORMAL_NO_EXECUTED_HANDLER_METHOD') if accepted
            else 'B_PATH_LOCATED_SEMANTICS_UNSUPPORTED' if exceptional_transition else 'C_UNRESOLVED_OR_UNSUPPORTED',
            'scope':'normal-path-only','exception_control_supported':False,'errors':errors,
            'executed_methods_with_exception_handlers':executed_handlers,'handler_visits':handler_visits,
            'observed_handler_pcs':observed_handler_pcs,
            'normal_transitions':transitions,'terminal':terminal,'events_checked':checked_count,'events_recorded':len(events),
            'normal_invocations':len(seen_frames),'certified_exception_transitions':len(certified)}
    if null_field_transfers is not None:
        result.update(scope='normal-and-exact-null-field-catch',certified_transfers=certified)
    return result


def control_evidence(raw, static, flow_events, status, *, semantics_profile=None, framework_evidence=None):
    domains=sorted({r['domain'] for r in raw});methods=[]
    for domain in domains:
        table=static['exception_tables'].get(domain)
        if table is None:raise ValueError('Missing complete original ExceptionTable metadata')
        methods.append({'domain':list(domain),'class_sha256':static['classes'][domain[0]]['class_sha256'],**table,
            'instructions':[{'bci':b,'opcode':inst['opcode'],'next':inst['next'],'successors':sorted(inst['successors']),
                             **({'call_target':list(static['calls'][(*domain,b)])} if (*domain,b) in static['calls'] else {})}
                            for b,inst in sorted(static['methods'][domain].items())]})
    events=[]
    for r,e in zip(raw,flow_events):
        record={k:r[k] for k in ('event_index','bci','opcode','thread_id','entry','call_returned')}
        record.update(domain=list(r['domain']),frame_id=e['frame_id'])
        if r['opcode'] in INVOKES:
            f=r['fields'];record['call_target']=[f.get('callclass','').replace('/','.'),f.get('callname',''),f.get('calltype','')]
        events.append(record)
    complete=bool(status['complete'] and status['parity'] and status['matches_historical_outcome'])
    outcome=status['runs']['instrumented']['result']['outcome']
    failure_types=[line.split('\t')[1] for line in status['runs']['instrumented']['result'].get('failures',[])
                   if line.startswith('FAILURE\t') and len(line.split('\t'))>=2]
    proof={'methods':methods,'events':events,'execution_complete_and_consistent':complete,'test_outcome':outcome,
           'failure_types':failure_types,
           'trace_sha256':status['artifacts_sha256']['instrumented/run.log']}
    if semantics_profile is not None:
        from .trace import NULL_FIELD_PROFILE,FRAMEWORK_PROFILE
        if semantics_profile not in {NULL_FIELD_PROFILE,FRAMEWORK_PROFILE}:raise ValueError('Unsupported semantic profile')
        if semantics_profile==FRAMEWORK_PROFILE:proof['framework_evidence']=framework_evidence
        for m in methods:
            d=tuple(m['domain'])
            m['method_flags']=static.get('method_flags',{}).get(d)
            for inst in m['instructions']:
                if (*d,inst['bci']) in static['fields']:inst['field_member']=list(static['fields'][(*d,inst['bci'])])
        proof['raw_events']=[{**r,'domain':list(r['domain'])} for r in raw]
        assessment,flow=replay_null_field_proof(proof)
        proof['dataflow_sha256']=flow_digest(flow)
        return {'schema':semantics_profile,'proof':proof,'proof_sha256':identity_digest(proof),'assessment':assessment}
    return {'schema':CONTROL_SCHEMA,'proof':proof,'proof_sha256':identity_digest(proof),
            'assessment':replay_normal_path(methods,events,complete,outcome,failure_types)}


def flow_digest(flow):
    from dataclasses import asdict
    return identity_digest({'values':[asdict(v) for v in flow.values],'links':flow.links,'events':flow.events,
                            'gaps':flow.gaps,'transfers':flow.exception_transfers,
                            **({'framework_effects':flow.framework_effects} if flow.framework_effects else {})})


def replay_null_field_proof(proof):
    """Consumer repeats null provenance, field effects and frame unwinding.

    Serialized assessment flags and the transfer list are never authority.
    The source bytecode tables and complete raw stream are retained for replay.
    """
    from .trace import Dataflow, NULL_FIELD_PROFILE, FRAMEWORK_PROFILE
    static={'methods':{},'fields':{},'exception_tables':{},'method_flags':{}}
    for m in proof['methods']:
        d=tuple(m['domain']);static['methods'][d]={i['bci']:{**i,'successors':set(i['successors'])} for i in m['instructions']}
        static['method_flags'][d]=m.get('method_flags')
        static['exception_tables'][d]={k:m[k] for k in ('code_length','handler_count','handlers')}
        for i in m['instructions']:
            if 'field_member' in i:static['fields'][(*d,i['bci'])]=tuple(i['field_member'])
    raw=[{**r,'domain':tuple(r['domain'])} for r in proof['raw_events']]
    calls={};profile=NULL_FIELD_PROFILE
    if 'framework_evidence' in proof:
        from .effects import project_and_verify
        projected,calls,_=project_and_verify(proof['framework_evidence'])
        if identity_digest(projected)!=identity_digest(raw):raise ValueError('Framework projection differs from actual trace')
        if proof['framework_evidence']['trace_sha256']!=proof['trace_sha256']:raise ValueError('Framework trace binding mismatch')
        profile=FRAMEWORK_PROFILE
    flow=Dataflow(static,semantics_profile=profile,framework_calls=calls).run(raw)
    assessment=replay_normal_path(proof['methods'],proof['events'],proof['execution_complete_and_consistent'],
        proof['test_outcome'],proof['failure_types'],null_field_transfers=flow.exception_transfers)
    if len(flow.events)!=len(proof['events']) or any(
        any(a[k]!=b[k] for k in ('event_index','bci','opcode','thread_id','entry','call_returned','frame_id'))
        or list(a['domain'])!=b['domain'] for a,b in zip(flow.events,proof['events'])):
        assessment['errors'].append({'event_index':-1,'reason':'raw_event_frame_or_location_mismatch'});assessment['accepted']=False
    return assessment,flow


def control_evidence_errors(graph):
    """Consumers recompute the path checks and compare every dynamic endpoint."""
    errors=[];evidence=graph.get('control_flow_evidence',{});q=graph.get('quality',{})
    from .trace import NULL_FIELD_PROFILE, FRAMEWORK_PROFILE
    schema=evidence.get('schema')
    if graph.get('provenance',{}).get('control_schema')!=schema or schema not in {CONTROL_SCHEMA,NULL_FIELD_PROFILE,FRAMEWORK_PROFILE}:
        return ['missing_or_unsupported_control_evidence_schema']
    try:
        proof=evidence['proof']
        if ('framework_evidence' in proof)!=(schema==FRAMEWORK_PROFILE):raise ValueError('Semantic profile/effect proof mismatch')
        if evidence['proof_sha256']!=identity_digest(proof):errors.append('control_proof_digest_mismatch')
        if proof['trace_sha256']!=graph['provenance']['trace_sha256']:errors.append('control_trace_binding_mismatch')
        if schema in {NULL_FIELD_PROFILE,FRAMEWORK_PROFILE}:
            if schema==FRAMEWORK_PROFILE and 'framework_evidence' not in proof:raise ValueError('Missing external call evidence')
            replay,flow=replay_null_field_proof(proof)
            if schema==FRAMEWORK_PROFILE:
                if graph.get('framework_effects')!=flow.framework_effects:errors.append('framework_effect_replay_mismatch')
                expected_effects=bool(flow.framework_effects) and all(e['accepted'] for e in flow.framework_effects)
                if q.get('framework_effects_valid')!=expected_effects:errors.append('framework_effect_quality_mismatch')
                from .effects import project_and_verify
                projected,_,full=project_and_verify(proof['framework_evidence'])
                if q.get('physical_trace_events')!=len(full) or q.get('framework_internal_events')!=len(full)-len(projected):
                    errors.append('framework_event_accounting_mismatch')
            if proof['dataflow_sha256']!=flow_digest(flow):errors.append('dataflow_replay_digest_mismatch')
            if q.get('unsupported_dynamic')!=flow.gaps:errors.append('dataflow_gap_summary_mismatch')
            if q.get('control_path_verified')!=replay['accepted']:errors.append('control_quality_summary_mismatch')
            values={v.id:v for v in flow.values} if replay['accepted'] else {}
            actual={n['id']:n['attrs'] for n in graph['nodes'] if n['type']=='value'}
            if set(actual)!=set(values):errors.append('dataflow_value_set_mismatch')
            for name,v in values.items():
                a=actual.get(name,{})
                expected_attrs={**v.observed,'definition_event':v.definition,'reaching_definition_supported':v.certain,
                                'runtime_reference_known':v.ref is not None}
                if v.ref and v.ref.startswith('jvm_npe:'):
                    expected_attrs['evidence_available_at_event']=next((t['target_event'] for t in flow.exception_transfers if t['exception_value_id']==v.id),v.definition)
                    expected_attrs['reference_semantics']='verified_JVM_NPE_reference; no_numeric_value'
                if any(a.get(k)!=val for k,val in expected_attrs.items()):
                    errors.append('dataflow_value_evidence_mismatch')
            links=flow.links if replay['accepted'] else []
            expected={(v,'e:'+str(i),role,i,'dynamic') for v,i,role in links}
            got={(e['source'],e['target'],e.get('role'),e.get('event_index'),e.get('evidence')) for e in graph['edges'] if e['type']=='consumes'}
            if expected!=got:errors.append('dataflow_use_endpoint_mismatch')
            expected={('e:'+str(values[v].definition),'e:'+str(i),v,role,values[v].definition,i,'dynamic') for v,i,role in links if values[v].definition is not None and values[v].definition<i}
            got={(e['source'],e['target'],e.get('value_id'),e.get('role'),e.get('definition_event'),e.get('event_index'),e.get('evidence')) for e in graph['edges'] if e['type']=='data_dep'}
            if expected!=got:errors.append('dataflow_dependency_endpoint_mismatch')
            expected={('e:'+str(v.definition),v.id,v.definition,'dynamic') for v in values.values() if v.definition is not None}
            got={(e['source'],e['target'],e.get('event_index'),e.get('evidence')) for e in graph['edges'] if e['type']=='produces'}
            if expected!=got:errors.append('dataflow_definition_endpoint_mismatch')
        else:replay=replay_normal_path(proof['methods'],proof['events'],proof['execution_complete_and_consistent'],proof['test_outcome'],proof['failure_types'])
        if replay!=evidence['assessment']:errors.append('control_assessment_disagrees_with_replay')
        dynamic=sorted((n for n in graph['nodes'] if n['type']=='statement' and 'event_index' in n['attrs']),key=lambda n:n['attrs']['event_index'])
        if len(dynamic)!=len(proof['events']):errors.append('control_event_count_mismatch')
        class_hash={tuple(m['domain']):m['class_sha256'] for m in proof['methods']}
        for n,r in zip(dynamic,proof['events']):
            a=n['attrs'];domain=[a['composed_class'],a['composed_method'],a['descriptor']]
            if (domain!=r['domain'] or any(a[k]!=r[k] for k in ('event_index','frame_id','thread_id','opcode')) or
                a['bytecode_offset']!=r['bci'] or a.get('program_identity',{}).get('locator',{}).get('class_sha256')!=class_hash[tuple(domain)]):
                errors.append('control_event_identity_mismatch:'+n['id'])
        outputs=[n['attrs'].get('result') for n in graph['nodes'] if n['type']=='output' and n['attrs'].get('kind')=='original_test_result']
        if outputs!=[proof['test_outcome']]:errors.append('control_terminal_result_mismatch')
        normal=bool(replay['accepted'] and replay['certified_exception_transitions']==0)
        if q.get('normal_path_verified')!=normal or q.get('control_semantics_scope')!=replay['scope']:
            errors.append('control_quality_summary_mismatch')
        if q.get('executed_methods_with_exception_handlers')!=replay['executed_methods_with_exception_handlers']:
            errors.append('control_handler_summary_mismatch')
        if q.get('exception_control_supported') is not False:errors.append('exception_semantics_overclaim')
        if not replay['accepted']:errors.append('normal_path_not_certified')  # legacy public rejection key; scope is explicit above
    except (KeyError,TypeError,ValueError,IndexError,OSError):errors.append('malformed_control_evidence')
    return sorted(set(errors))
