"""Pre-instruction operand diagnostics. No targets or complete traces accepted.

The frozen interpreter appends an event after frame setup, BEFORE consuming
operands. A private list hook stops at that boundary. No current callback,
successor, result or full-graph field is available to this adapter.
"""
from __future__ import annotations
import copy
from cwm_fp.lddg.trace import Dataflow, FRAMEWORK_PROFILE
from cwm_fp.lddg.effects import project_observed_events

BRANCH_NAMES=dict(enumerate(('ifeq','ifne','iflt','ifge','ifgt','ifle','if_icmpeq','if_icmpne',
    'if_icmplt','if_icmpge','if_icmpgt','if_icmple','if_acmpeq','if_acmpne'),153))|{198:'ifnull',199:'ifnonnull'}
ARITH_NAMES={96:'iadd',100:'isub',104:'imul',108:'idiv',112:'irem',116:'ineg',120:'ishl',122:'ishr',124:'iushr',126:'iand',128:'ior',130:'ixor'}
UNSUPPORTED_NAMES={133:'i2l'}
TRUSTED={'DIRECT_OBSERVED','STATIC_CONSTANT'}

class _AtBoundary(BaseException):pass

class _BoundaryList(list):
    def __init__(self,index):super().__init__();self.index=index;self.current=None
    def append(self,event):
        if len(self)==self.index:
            self.current=event
            raise _AtBoundary()
        super().append(event)

def before_instruction(observer,static,framework):
    if not observer.full or observer.full[-1]['pre_clock']!=observer.last_clock:
        raise ValueError('Only an actual pre-instruction boundary is accepted')
    if observer.full[-1]['callback_clock'] is not None or 'stack' in observer.full[-1]['fields']:
        raise ValueError('Current result cannot enter pre-state')
    raw,calls=project_observed_events(copy.deepcopy(observer.full),framework,allow_pending=True)
    flow=Dataflow(static,semantics_profile=FRAMEWORK_PROFILE,framework_calls=calls)
    hook=_BoundaryList(len(raw)-1);flow.events=hook
    try:flow.run(raw)
    except _AtBoundary:pass
    else:raise ValueError('Interpreter did not reach declared pre-execution hook')
    if any(v.definition==len(raw)-1 for v in flow.values):raise AssertionError('Current definition leaked')
    return flow,raw,hook.current

def int32(value):return ((value+(1<<31))%(1<<32))-(1<<31)

def branch(op,values):
    if 153<=op<=164:
        if any(type(v) is not int or not -(1<<31)<=v<(1<<31) for v in values):raise ValueError('Not JVM int32')
        if len(values)!=(1 if op<159 else 2):raise ValueError('Wrong operand arity')
        a,b=(values[0],0) if op<159 else values
        return (a==b,a!=b,a<b,a>=b,a>b,a<=b)[(op-153)%6]
    if op in (198,199) and values==[None]:return op==198
    raise ValueError('Reference equality, non-null identity and other comparisons unsupported')

def arithmetic(op,values):
    if op not in ARITH_NAMES or len(values)!=(1 if op==116 else 2):raise ValueError('Unsupported exact arithmetic')
    if any(type(v) is not int or not -(1<<31)<=v<(1<<31) for v in values):raise ValueError('Not JVM int32')
    a=values[0];b=values[1] if len(values)>1 else None
    if op==96:z=a+b
    elif op==100:z=a-b
    elif op==104:z=a*b
    elif op in (108,112):
        if b==0:raise ValueError('Zero divisor: no normal result')
        q=(abs(a)//abs(b))*(-1 if (a<0)!=(b<0) else 1)
        z=q if op==108 else a-q*b
    elif op==116:z=-a
    elif op==120:z=a<<(b&31)
    elif op==122:z=a>>(b&31)
    elif op==124:z=(a&0xffffffff)>>(b&31)
    elif op==126:z=a&b
    elif op==128:z=a|b
    else:z=a^b
    return int32(z)

def operand_evidence(value,flow,raw,clock,seen=None):
    """Resolve a historical value version, never an assumed current heap slot."""
    out={'value_id':value.id,'definition_event':value.definition,'source':'UNKNOWN','value':None,
         'dtype':None,'available_at':None,'reason':None,'static_alternative':None,'derivation':[]}
    if not value.certain or value.definition is None:
        out['reason']='uncertain_or_external_definition';return out
    if flow.gaps:
        out['reason']='prior_semantic_gap:'+flow.gaps[0]['reason'];return out
    r=raw[value.definition];op=r['opcode'];constant=None
    if 2<=op<=8:constant={'value':op-3,'dtype':'I','available_at':r['pre_clock'],'rule':'iconst_opcode_literal'}
    elif op==1:constant={'value':None,'dtype':'reference_null','available_at':r['pre_clock'],'rule':'aconst_null_opcode'}
    # bipush/sipush/ldc are deliberately not guessed: TSV does not retain their
    # literal payload; their real prior callbacks may still be direct evidence.
    out['static_alternative']=constant
    ob=value.observed
    if ob.get('observed'):
        at=r['callback_clock'];dt=ob.get('dtype');v=ob.get('value')
        ranges={'I':(-(1<<31),(1<<31)-1),'B':(-128,127),'S':(-32768,32767),'Z':(0,1)}
        if dt in ranges and isinstance(v,(int,bool)) and ranges[dt][0]<=v<=ranges[dt][1] and at is not None and at<clock:
            out.update(source='DIRECT_OBSERVED',value=int(v),dtype='I',available_at=at);return out
        out['reason']='unsupported_type_range_or_observation_time';return out
    if constant and constant['available_at']<clock:
        out.update(source='STATIC_CONSTANT',value=constant['value'],dtype=constant['dtype'],available_at=constant['available_at']);return out
    # Restricted copy-only derivation; no numerical execution, heap assumption,
    # floating-point conversion, branch-result inversion or identity-hash use.
    seen=set() if seen is None else set(seen)
    if value.id not in seen and (op==89 or 21<=op<=45 or 54<=op<=78 or 172<=op<=176):
        seen.add(value.id)
        ids=[v for v,e,role in flow.links if e==value.definition and role in ('alias','local_read','operand','return_value')]
        if len(set(ids))==1:
            prior=next((v for v in flow.values if v.id==ids[0]),None)
            if prior is not None and prior.definition is not None and prior.definition<value.definition:
                ev=operand_evidence(prior,flow,raw,clock,seen)
                if ev['source']!='UNKNOWN' and r['completed_clock'] is not None and r['completed_clock']<=clock:
                    out.update(source='SEMANTICALLY_DERIVED',value=ev['value'],dtype=ev['dtype'],available_at=max(r['completed_clock'],ev['available_at']),
                        derivation=[{'rule':'verified_copy_only','source_value_id':prior.id,'source_category':ev['source']},*ev['derivation']]);return out
    out['reason']='no_prior_supported_scalar_callback_or_literal';return out

def capture(observer,static,framework):
    flow,raw,event=before_instruction(observer,static,framework);op=event['opcode']
    arity=(2 if 159<=op<=166 else 1) if op in BRANCH_NAMES else (1 if op==116 else 2) if op in ARITH_NAMES else int(event['fields'].get('popnum','0'))
    frame=flow.frames[-1];values=frame.stack[-arity:] if arity else []
    operands=[operand_evidence(v,flow,raw,observer.last_clock) for v in values]
    while len(operands)<arity:operands.insert(0,{'source':'UNKNOWN','value':None,'dtype':None,'reason':'operand_stack_short','value_id':None,'definition_event':None,'available_at':None,'static_alternative':None,'derivation':[]})
    return {'opcode':op,'operation':BRANCH_NAMES.get(op,ARITH_NAMES.get(op,UNSUPPORTED_NAMES.get(op,'unsupported_'+str(op)))),
        'arity':arity,'operand_order':'bottom_to_top_value1_then_value2','operands':operands,
        'clock':observer.last_clock,'frame_id':frame.id,'event_index':len(raw)-1,
        'prior_gaps':flow.gaps,'current_instruction_executed':False},flow,raw

def semantic_check(state):
    if any(v['source'] not in TRUSTED for v in state['operands']):return None,'insufficient_primary_evidence'
    try:
        vs=[v['value'] for v in state['operands']];op=state['opcode']
        if 153<=op<=164 and any(v['dtype']!='I' for v in state['operands']):return None,'wrong_operand_type'
        if op in ARITH_NAMES and any(v['dtype']!='I' for v in state['operands']):return None,'wrong_operand_type'
        return (branch(op,vs) if op in BRANCH_NAMES else arithmetic(op,vs)),None
    except ValueError as e:return None,str(e)
