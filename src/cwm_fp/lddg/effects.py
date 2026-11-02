"""Version-pinned, observed-path contract for one framework call, not generic purity.

Library opcodes remain in the uncompressed evidence stream. The LDDG retains
product/test events and treats the framework region as an explicitly audited
call boundary. No product callback is removed. Any extra path is rejected.
"""
from pathlib import Path
from functools import lru_cache
from cwm_fp.data.common import sha256
from .trace import parse_trace

HELPER='org.evosuite.runtime.EvoAssertions'
DESC='(Ljava/lang/String;Ljava/lang/Throwable;)V'
JAR_SHA='231b939688bf69603374e8698ec2c2c4b6ca3aa030d1bcc7fa5073da4ee0a2eb'
RT_SHA='6eab9457e892290ffb7876da77855d082e4cdd3a548bbb83204624d4d6cb22b1'
STATIC_SHA='9e820f127b0266182adbed59a374f943f594d18761d637ee9bc8731334c08920'
ENVIRONMENT_PINS={
    'jre/bin/server/jvm.dll':'f4bc7e131f5ece73e05cfb39c3068fa98c0b2452c251cdb9fa06943353b8e935',
    'bin/java.exe':'dd49fa4479c27d79a46028b8a1553db4d5ae2e3f10cc411052b2cc74be6784c1',
    'phase03_bank_v2/agent.jar':'d451e8a256852215163433f4d38d7eaeb6f2728873af7f98b83da58637ca8ebf'}
VERIFY=[0,1,4,12,13,16,24,25,26,29]
ASSERT=[0,1,4,5,6,7,11,12,13,14,15,16,20,21,24,26,27,29,32,35]

@lru_cache(maxsize=128)
def _hash(path,mtime,size):return sha256(Path(path))
def checked_hash(path):
    p=Path(path);s=p.stat();return _hash(str(p.resolve()),s.st_mtime_ns,s.st_size)

def make_evidence(path,test_id,status,framework_static):
    artifacts=status['inputs']['runtime_binding']['artifacts_sha256']
    def artifact(suffix,expected):
        matches=[(p,h) for p,h in artifacts.items() if p.replace('\\','/').endswith(suffix)]
        if len(matches)!=1 or matches[0][1]!=expected:raise ValueError('Uncertified framework/runtime binary')
        return {'path':matches[0][0],'sha256':matches[0][1]}
    return {'schema':'evosuite_1_0_6_observed_equal_path_v1','trace_path':str(path),'test_id':test_id,
            'trace_sha256':sha256(path),'jar':artifact('evosuite-standalone-runtime-1.0.6.jar',JAR_SHA),
            'runtime':artifact('jre/lib/rt.jar',RT_SHA),'static_path':str(framework_static),
            'static_sha256':sha256(framework_static),
            'runtime_binding':status['inputs']['runtime_binding'],
            'scope':'product/test heap plus conservative exception-reference invalidation; library/VM private state not T2'}

def project_and_verify(evidence):
    """Independently read real trace and original binary-bound CFG; fail closed."""
    from .static import load_static
    if evidence['schema']!='evosuite_1_0_6_observed_equal_path_v1':raise ValueError('Unknown effect evidence schema')
    for key,expected in [('jar',JAR_SHA),('runtime',RT_SHA)]:
        b=evidence[key]
        if b['sha256']!=expected or checked_hash(b['path'])!=expected:raise ValueError('Effect binary mismatch')
        if evidence['runtime_binding']['artifacts_sha256'].get(b['path'])!=expected:raise ValueError('Unbound runtime binary')
    if HELPER not in evidence['runtime_binding']['selected_classes']:raise ValueError('Helper was not selected for instrumentation')
    for suffix,expected in ENVIRONMENT_PINS.items():
        matches=[(p,h) for p,h in evidence['runtime_binding']['artifacts_sha256'].items() if p.replace('\\','/').endswith(suffix)]
        if len(matches)!=1 or matches[0][1]!=expected or checked_hash(matches[0][0])!=expected:raise ValueError('Uncertified VM/agent environment')
    if evidence['static_sha256']!=STATIC_SHA or checked_hash(evidence['static_path'])!=STATIC_SHA:raise ValueError('Effect bytecode mismatch')
    if checked_hash(evidence['trace_path'])!=evidence['trace_sha256']:raise ValueError('Effect trace changed')
    static=load_static(Path(evidence['static_path']))
    full=parse_trace(Path(evidence['trace_path']),evidence['test_id'],physical_evidence=True)
    projected,calls=project_observed_events(full,static)
    return projected,calls,full

def project_observed_events(full,static,*,allow_pending=False):
    """Replay an already observed region; no file reads or future lookup.

    The caller must separately authenticate bytecode/binaries. Pending mode is
    for the prefix observer only: unfinished regions yield no effect certificate.
    It never turns an unsupported completed region into an accepted effect.
    """
    projected=[];calls={}
    for r in full:
        if r['domain'][0]==HELPER:continue  # retained verbatim in trace, certified below as library region
        e={**r,'event_index':len(projected)};projected.append(e)
        f=r['fields']
        if (f.get('callclass'),f.get('callname'),f.get('calltype'))!=(HELPER,'verifyException',DESC):continue
        reason=None;start=r['physical_event_index'];end=r.get('return_after_physical_event',start)
        region=full[start+1:end+1]
        wanted=[('verifyException',b) for b in VERIFY[:-1]]+[('assertThrownBy',b) for b in ASSERT]+[('verifyException',29)]
        if not r['call_returned']:reason='missing_normal_return'
        elif [(x['domain'][1],x['bci']) for x in region]!=wanted or any(x['domain'][0]!=HELPER or x['domain'][2]!=DESC for x in region):reason='unsupported_internal_path_or_callback'
        elif any(x['thread_id']!=r['thread_id'] for x in region):reason='different_thread'
        else:
            for x in region:
                inst=static['methods'][x['domain']].get(x['bci'])
                if not inst or (inst['opcode'],inst['line'])!=(x['opcode'],x['line']):reason='internal_bytecode_mismatch';break
                if 182<=x['opcode']<=186:
                    target=static['calls'][(*x['domain'],x['bci'])]
                    if tuple(x['fields'].get(k) for k in ('callclass','callname','calltype'))!=target or not x['call_returned']:
                        reason='internal_call_or_return_mismatch';break
                    if not (x['raw_log_line']<x.get('return_log_line',0)<r.get('return_log_line',0)):
                        reason='internal_return_order_mismatch';break
            if [x['entry'] for x in region]!=[i in (0,9) for i in range(30)]:reason='internal_entry_mismatch'
        if allow_pending and not r['call_returned']:continue
        calls[e['event_index']]={'accepted_path':reason is None,'reason':reason,'physical_call_event':start,
            'physical_return_after_event':end,'return_log_line':r.get('return_log_line'),'internal_events':len(region),
            'observed_scalar_callbacks':[{'physical_event_index':x['event_index'],'stack':x['fields']['stack']}
                                         for x in region if x['fields'].get('stack','').startswith(('I:','Z:','J:','D:','F:'))],
            'path':'stack_top_class_equal_early_return' if reason is None else 'UNSUPPORTED'}
    # Every removed library opcode must belong to an actual outer call region.
    covered={j for c in calls.values() for j in range(c['physical_call_event']+1,c['physical_return_after_event']+1)}
    if not allow_pending and any(r['domain'][0]==HELPER and r['event_index'] not in covered for r in full):
        raise ValueError('Unbounded/orphan framework events')
    return projected,calls

def apply_effect(index,args,raw,call,gaps,escaped):
    reason=call['reason']
    if not reason:
        if gaps:reason='uncertified_prefix'
        elif len(args)!=2 or not all(v.certain for v in args):reason='unknown_argument_definition'
        elif not args[1].ref or not args[1].ref.startswith('jvm_npe:'):reason='not_verified_JVM_null_field_NPE'
        elif args[1].ref in escaped:reason='exception_previously_escaped'
        elif args[0].definition is None or raw[args[0].definition]['opcode'] not in (18,19) or raw[args[0].definition]['fields'].get('stack')!='Str:':reason='string_literal_argument_not_verified'
    return {**call,'event_index':index,'accepted':reason is None,'reason':reason,
            'argument_value_ids':[v.id for v in args],'return_type':'void','scalar_return_observed':False,
            'possible_writes':['exception_private_stackTrace_cache','fresh_library_stack_array'],
            'unknown_state':['cache_materialization_branch','VM_private_stack_metadata','class_resolution_bookkeeping'],
            'product_callback':False if not reason else None,'product_heap_write':'none_on_verified_path' if not reason else 'UNKNOWN',
            'evidence_available_at':'actual_framework_return_marker; never at call entry'}
