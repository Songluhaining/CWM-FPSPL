"""Bounded adapter for inspected BRFL opcode records; no coverage-to-dataflow inference."""
from __future__ import annotations
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

NULL_FIELD_PROFILE = 'null_field_exception_v1'
FRAMEWORK_PROFILE = 'framework_effect_v1'
KNOWN_NULL = 'jvm:null'

def parse_trace(path:Path, test_id:str, max_events=20000, max_bytes=33554432, *, physical_evidence=False):
    if path.stat().st_size>max_bytes:raise ValueError("Trace file-size limit exceeded")
    clazz,method=test_id.rsplit('.',1);marker='###'+clazz+'::'+method
    active=False;entry=False;events=[];calls={}
    for line_number,raw in enumerate(path.read_text(encoding='utf-8',errors='replace').splitlines(),1):
        line=raw.strip()
        if line==marker:active=True;entry=True;continue
        if not active:continue
        if line.startswith('###RET@'):
            m=re.search(r'lineinfo=([^,]+)',line)
            if m and m[1] in calls and calls[m[1]]:
                event=events[calls[m[1]].pop()];event['call_returned']=True
                if physical_evidence:event.update(return_log_line=line_number,return_after_physical_event=len(events)-1)
            continue
        if line.startswith('###'):
            entry=True;continue
        if not line.startswith('opcode='):continue
        pieces=dict((k.strip(),v.strip()) for item in line.split(',') if '=' in item for k,v in [item.split('=',1)])
        info=pieces.get('lineinfo','').split('#')
        if len(info)!=5 or not re.fullmatch(r'\d+\([\w]+\)',pieces['opcode']):raise ValueError("Malformed opcode location")
        if 'thread' not in pieces:raise ValueError("No runtime thread identity; cannot certify ordered dataflow")
        op=int(pieces['opcode'].split('(')[0])
        event={'event_index':len(events),'domain':tuple(info[:3]),'line':int(info[3]),'bci':int(info[4]),'opcode':op,
               'thread_id':int(pieces['thread']),'entry':entry,'fields':{k:v for k,v in pieces.items() if k in {
                   'popnum','pushnum','load','store','nextinst','branchbyte','stack','calltype','callclass','callname','field','CONST'}},
               'call_returned':False}
        entry=False;events.append(event)
        if physical_evidence:event.update(physical_event_index=event['event_index'],raw_log_line=line_number)
        if 182<=op<=186:calls.setdefault(pieces['lineinfo'],[]).append(len(events)-1)
        if len(events)>max_events:raise ValueError("Event count limit exceeded")
    if not events:raise ValueError("No ordered opcode trace for requested original test; coverage XML is not DDG")
    return events


def descriptor_args(signature):
    prefix=signature[1:signature.index(')')]
    matches=re.findall(r'\[*L[^;]+;|\[*[ZBCSIFDJ]',prefix)
    if ''.join(matches)!=prefix:raise ValueError("Invalid JVM descriptor")
    return matches


def observed_scalar(raw):
    if not raw or ':' not in raw:return {'observed':False}
    dtype,text=raw.split(':',1)
    if dtype not in {'I','S','B','Z','J','F','D'} or not text:return {'observed':False,'dtype':dtype}
    try:
        value=text=='true' if dtype=='Z' and text in {'true','false'} else float(text) if dtype in {'F','D'} else int(text)
    except ValueError:return {'observed':False,'dtype':dtype}
    if isinstance(value,float) and not math.isfinite(value):return {'observed':False,'dtype':dtype,'reason':'nonfinite'}
    return {'observed':True,'dtype':dtype,'value':value,'origin':'producer_stack_callback'}


@dataclass
class Value:
    id:str
    definition:int|None
    ref:str|None=None
    width:int=1
    observed:dict=field(default_factory=lambda:{'observed':False})
    certain:bool=True


@dataclass
class Frame:
    id:int
    domain:tuple
    stack:list=field(default_factory=list)
    locals:dict=field(default_factory=dict)
    caller_event:int|None=None


class Dataflow:
    """Exact supported single-thread stack/local/call/allocated-object field subset.

    Object identities are NEW event tokens propagated through references, never
    identityHashCode. Arrays, unknown/native effects, unsupported JVM categories
    and general exception unwinding make a trace ineligible. The opt-in
    null-field profile certifies only evidenced exact NPE handlers; unknown
    calls and all other unsupported semantics still make the graph ineligible.
    The separate framework profile consumes independently verified, pinned
    call-region evidence; it invalidates exception-private state conservatively.
    """
    def __init__(self, static, *, semantics_profile=None, framework_calls=None):
        if semantics_profile not in {None, NULL_FIELD_PROFILE, FRAMEWORK_PROFILE}:raise ValueError('Unknown semantics profile')
        self.static=static;self.values=[];self.links=[];self.events=[];self.gaps=[];self.heap={};self.frames=[]
        self.frame_counter=0;self.pending=None;self.escaped=set()
        self.semantics_profile=semantics_profile;self.exception_transfers=[]
        self.framework_calls=framework_calls or {};self.framework_effects=[]

    def null_field_handler(self, raw, index, receiver):
        """Prove only exact NPE catch, from a certain ACONST_NULL reaching value.

        No hash-code null inference, catch-all/finally, hierarchy guessing, skipped
        events, missing tables, or already-tainted prefix is accepted.
        """
        if (self.semantics_profile not in {NULL_FIELD_PROFILE,FRAMEWORK_PROFILE} or self.gaps or not receiver.certain
                or receiver.ref!=KNOWN_NULL or index+1>=len(raw)):return None
        nxt=raw[index+1]
        if nxt['entry'] or raw[index]['fields'].get('stack'):return None
        for pos in range(len(self.frames)-1,-1,-1):
            frame=self.frames[pos]
            flags=self.static.get('method_flags',{}).get(frame.domain)
            if flags is None or flags & (32|256):return None  # monitor release/native unwinding not certified
            active=index if pos==len(self.frames)-1 else self.frames[pos+1].caller_event
            if active is None:return None
            table=self.static.get('exception_tables',{}).get(frame.domain)
            if table is None or table['handler_count']!=len(table['handlers']):return None
            handlers=[h for h in table['handlers'] if h['start']<=raw[active]['bci']<h['end']]
            if not handlers:continue
            h=handlers[0]
            if (h['catch_type'].replace('.','/')!='java/lang/NullPointerException'
                    or nxt['domain']!=frame.domain or nxt['bci']!=h['handler']):return None
            unwound=self.frames[pos+1:]
            if any(f.caller_event is None or raw[f.caller_event]['call_returned'] for f in unwound):return None
            return pos,{'source_event':index,'target_event':index+1,'kind':'verified_null_field_exception',
                'throw_opcode':raw[index]['opcode'],'exception_type':'java.lang.NullPointerException',
                'receiver_value_id':receiver.id,'source_frame':self.frames[-1].id,'target_frame':frame.id,
                'protected_event':active,'handler':h,'unwound_frames':[f.id for f in reversed(unwound)],
                'aborted_calls':[f.caller_event for f in reversed(unwound)],
                'exception_reference_evidence':'JVM_null_field_rule_and_observed_exact_handler; not_scalar_observation'}
        return None

    def new(self,index,ref=None,width=1,observation=None,certain=True):
        v=Value('v:'+str(len(self.values)),index,ref,width,observation or {'observed':False},certain)
        self.values.append(v);return v

    def gap(self,event,reason):
        self.gaps.append({'event_index':event,'reason':reason})

    def use(self,value,event,role):
        if value.definition is not None and value.certain:self.links.append((value.id,event,role))

    def run(self, raw):
        if len({r['thread_id'] for r in raw})!=1:
            self.gap(0,'multiple_threads_without_happens_before; all_dynamic_dependencies_disabled')
            return self
        for i,r in enumerate(raw):
            domain=r['domain'];op=r['opcode'];f=r['fields'];where=(*domain,r['bci'])
            if r['entry'] or not self.frames:
                frame=Frame(self.frame_counter,domain);self.frame_counter+=1
                if self.pending:
                    event,args,expected=self.pending
                    if domain!=expected: self.gap(i,'call_target_mismatch')
                    else:
                        slot=0
                        for arg in args:
                            frame.locals[slot]=arg;slot+=arg.width
                        frame.caller_event=event
                    self.pending=None
                elif self.frames:self.gap(i,'unmatched_method_entry')
                self.frames.append(frame)
            frame=self.frames[-1]
            if frame.domain!=domain:
                self.gap(i,'exception_or_unmatched_frame_transition')
                self.frames=[Frame(self.frame_counter,domain)];self.frame_counter+=1;frame=self.frames[-1]
            event={**r,'frame_id':frame.id,'use_ids':[],'def_ids':[],'branch_taken':None,'value_observed':False}
            self.events.append(event)
            used=[];defined=[]
            def pop():
                if frame.stack:return frame.stack.pop()
                self.gap(i,'operand_stack_underflow');return self.new(None,certain=False)
            def consume(v,role='operand'):
                used.append(v.id);self.use(v,i,role);return v
            def define(ref=None,width=1,obs=None,certain=True):
                v=self.new(i,ref,width,obs,certain);defined.append(v.id);return v
            obs=observed_scalar(f.get('stack'))
            event['value_observed']=obs['observed'];event['value_summary']=obs
            next_raw=raw[i+1] if i+1<len(raw) else None
            try:
                if op==0:pass
                elif op==187:
                    # Uninitialized reference is an allocation event, not a fabricated scalar.
                    frame.stack.append(define(ref='allocation:'+str(i)))
                elif op==89:
                    v=pop()
                    if v.width!=1:self.gap(i,'invalid_dup_category')
                    frame.stack.extend([v,define(v.ref,v.width,certain=v.certain)]);consume(v,'alias')
                elif op in {87,88}:
                    v=consume(pop())
                    if op==88 and v.width==1:consume(pop())
                elif op in {90,91,92,93,94,95}:
                    self.gap(i,'unsupported_stack_permutation');frame.stack.clear()
                elif op==132:
                    slot=int(f['store']);v=frame.locals.get(slot)
                    if v:consume(v,'local_read')
                    else:self.gap(i,'undefined_local_read')
                    frame.locals[slot]=define()
                elif 21<=op<=45:
                    slot=int(f['load']);v=frame.locals.get(slot)
                    if v:consume(v,'local_read')
                    else:
                        v=self.new(None,certain=False);frame.locals[slot]=v
                        # Root test's this is an external input; data value remains unknown.
                        if not (frame.id==0 and slot==0):self.gap(i,'undefined_local_read')
                    frame.stack.append(define(v.ref,v.width,obs,v.certain))
                elif 54<=op<=78:
                    v=consume(pop());frame.locals[int(f['store'])]=define(v.ref,v.width,certain=v.certain)
                elif op in {178,179,180,181}:
                    member=self.static['fields'].get(where)
                    value=consume(pop()) if op in {179,181} else None
                    receiver=consume(pop(),'receiver') if op in {180,181} else None
                    ref=receiver.ref if receiver else 'static'
                    slot=(ref,member)
                    caught=self.null_field_handler(raw,i,receiver) if receiver and member and op in {180,181} else None
                    if caught is not None:
                        pos,transfer=caught
                        # The actual faulting event creates a JVM exception reference.
                        # It is not a normal field result/write or an invented opcode.
                        exception=define(ref='jvm_npe:'+str(i))
                        transfer['exception_value_id']=exception.id
                        self.exception_transfers.append(transfer)
                        self.frames=self.frames[:pos+1];self.frames[-1].stack=[exception];self.pending=None
                    elif ref==KNOWN_NULL:
                        self.gap(i,'null_field_exception_not_certified')
                        if value is None:frame.stack.append(define(obs=obs,certain=False))
                    elif not member or not ref or ref in self.escaped:
                        self.gap(i,'unresolved_or_escaped_heap_identity')
                        if value is None:frame.stack.append(define(obs=obs,certain=False))
                    elif value is not None:
                        self.heap[slot]=define(value.ref,value.width,certain=value.certain)
                    else:
                        reaching=self.heap.get(slot)
                        if reaching:consume(reaching,'field_read')
                        else:self.gap(i,'field_read_without_observed_definition')
                        frame.stack.append(define(reaching.ref if reaching else None,2 if member[2] in {'J','D'} else 1,obs,
                                                  bool(reaching and reaching.certain)))
                elif 182<=op<=186:
                    signature=f['calltype'];types=descriptor_args(signature)
                    args=[consume(pop(),'call_argument') for _ in types][::-1]
                    if op!=184:args.insert(0,consume(pop(),'receiver'))
                    expected=(f['callclass'].replace('/','.'),f['callname'],signature)
                    traced=bool(next_raw and next_raw['entry'] and next_raw['domain']==expected)
                    is_assert=expected[0] in {'org.junit.Assert','junit.framework.Assert'} and expected[1].startswith(('assert','fail'))
                    event['assertion']=is_assert
                    if is_assert:event['assertion_passed']=bool(r['call_returned'])
                    if traced:self.pending=(i,args,expected)
                    else:
                        benign=(expected==('java.lang.Object','<init>','()V')) or is_assert
                        if self.semantics_profile==FRAMEWORK_PROFILE and i in self.framework_calls:
                            from .effects import apply_effect
                            effect=apply_effect(i,args,raw,self.framework_calls[i],self.gaps,self.escaped)
                            self.framework_effects.append(effect)
                            if effect['accepted']:
                                benign=True
                                # getStackTrace may materialize exception-private state. It is
                                # not observed as a write event; future heap reads must reject.
                                self.escaped.add(args[1].ref)
                            else:self.gap(i,'framework_effect_precondition:'+effect['reason'])
                        if not benign:
                            self.gap(i,'untraced_call_effects');self.escaped.update(v.ref for v in args if v.ref)
                        result=signature.split(')',1)[1]
                        if result!='V':frame.stack.append(define(width=2 if result in {'J','D'} else 1,certain=False))
                elif 172<=op<=177:
                    result=consume(pop(),'return_value') if op!=177 else None
                    event['return_value_id']=result.id if result else None
                    finished=self.frames.pop()
                    if self.frames and result:
                        self.frames[-1].stack.append(define(result.ref,result.width,certain=result.certain))
                elif (153<=op<=166) or op in {198,199,170,171}:
                    for _ in range(int(f.get('popnum','1'))):consume(pop(),'branch_operand')
                    if next_raw and next_raw['domain']==domain and not next_raw['entry']:
                        successor=next_raw['bci'];info=self.static['methods'].get(domain,{}).get(r['bci'])
                        if info and successor in info['successors']:
                            event['branch_taken']=successor!=int(f.get('nextinst','-1'))
                            event['observed_successor']=successor
                        else:self.gap(i,'branch_successor_not_in_original_CFG')
                    else:self.gap(i,'branch_successor_unobserved')
                elif op in {167,200}:pass
                elif op==191:
                    consume(pop(),'throw');self.gap(i,'exception_unwinding_not_supported')
                elif 46<=op<=53 or 79<=op<=86 or op in {188,189,190,192,193,194,195,196,197,186}:
                    self.gap(i,'unsupported_array_type_or_monitor_opcode');frame.stack.clear()
                elif 1<=op<=20 or 96<=op<=152:
                    count=int(f.get('popnum','0'))
                    operands=[consume(pop()) for _ in range(max(0,count))]
                    if any(v.width!=1 for v in operands):self.gap(i,'category2_arithmetic_not_certified')
                    width=2 if obs.get('dtype') in {'J','D'} or op in {9,10,14,15,20} else 1
                    frame.stack.append(define(ref=KNOWN_NULL if op==1 and self.semantics_profile in {NULL_FIELD_PROFILE,FRAMEWORK_PROFILE} else None,
                                              width=width,obs=obs,certain=all(v.certain for v in operands)))
                else:
                    self.gap(i,'unsupported_opcode_'+str(op));frame.stack.clear()
            except (KeyError,ValueError) as exc:
                self.gap(i,'invalid_opcode_metadata:'+str(exc));frame.stack.clear()
            event['use_ids']=list(dict.fromkeys(used));event['def_ids']=defined
        # A failed final assertion legitimately aborts a test without normal return.
        if self.frames and not (self.events and self.events[-1].get('assertion') and not self.events[-1].get('call_returned')):
            self.gap(len(raw)-1,'unterminated_frame')
        return self
