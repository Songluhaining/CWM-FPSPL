"""Prefix-only observation replay. Never accepts a final graph or its attributes.

Clock: original text line + ordered subrecord (pre-instruction, then callback).
The legacy JVM dataflow engine is reused on *only* the observed prefix. Its
terminal and look-ahead conclusions are not treated as observations. Numeric
values here are historical value-version observations, not a complete live heap.
"""
from __future__ import annotations
import copy
import hashlib
import json
import re
from dataclasses import dataclass
import numpy as np
from cwm_fp.lddg.trace import Dataflow, FRAMEWORK_PROFILE, observed_scalar
from cwm_fp.lddg.effects import HELPER, project_observed_events
from .graph_encoder import FEATURES, OPCODES, RELATIONS, opcode_group


def tokens(text):
    """Tokenize physical records, retaining exact character cut points for tests.

    No normalization can move a post-instruction callback into its pre-record.
    Non-observation logger text is retained as an opaque gap; no terminal status
    is converted into a learning feature or an implicit normal-end target.
    """
    offset=0
    for line_no,line in enumerate(text.splitlines(keepends=True),1):
        content=line.rstrip('\r\n');pos=content.find(',stack=')
        chunks=[content] if pos<0 else [content[:pos],content[pos:]]
        used=0
        for sub,chunk in enumerate(chunks):
            used+=len(chunk);s=chunk.strip()
            if s.startswith('opcode='):kind='instruction'
            elif s.startswith(',stack='):kind='callback'
            elif s.startswith('###RET@'):kind='return_marker'
            elif s.startswith('###'):kind='entry'
            else:kind='opaque'
            yield {'clock':[line_no,sub], 'kind':kind,'text':s,'cut':offset+used}
        offset+=len(line)


def fields(text):
    return dict((k.strip(),v.strip()) for item in text.split(',') if '=' in item
                for k,v in [item.split('=',1)])


class ObservationReplay:
    """Incremental physical observer. No source suffix/target interface exists."""
    def __init__(self,test_id):
        self.marker='###'+'::'.join(test_id.rsplit('.',1))
        self.active=False;self.entry=False;self.full=[];self.pending={}
        self.observations=[];self.last_clock=None;self.opaque_after_start=[]

    def push(self,t):
        if self.last_clock is not None and t['clock']<=self.last_clock:raise ValueError('Nonmonotone observation clock')
        self.last_clock=t['clock']
        s=t['text'];kind=t['kind']
        if not self.active:
            if s==self.marker:self.active=True;self.entry=True
            return
        self.observations.append(t)
        if kind=='entry':self.entry=True;return
        if kind=='instruction':
            f=fields(s);loc=f.get('lineinfo','').split('#')
            if len(loc)!=5 or not re.fullmatch(r'\d+\([\w]+\)',f.get('opcode','')) or 'thread' not in f:
                raise ValueError('Malformed/unordered physical instruction')
            i=len(self.full);op=int(f['opcode'].split('(')[0])
            r={'event_index':i,'physical_event_index':i,'raw_log_line':t['clock'][0],
               'domain':tuple(loc[:3]),'line':int(loc[3]),'bci':int(loc[4]),'opcode':op,
               'thread_id':int(f['thread']),'entry':self.entry,'call_returned':False,
               'fields':{k:v for k,v in f.items() if k in {'popnum','pushnum','load','store','nextinst','branchbyte','calltype','callclass','callname','field','CONST'}},
               'pre_clock':t['clock'],'callback_clock':None,'completed_clock':None}
            # Arrival of the next real instruction proves the prior ordinary
            # instruction left its pre-boundary. Calls require their RET marker.
            if self.full:
                prev=self.full[-1]
                if not 182<=prev['opcode']<=186 and prev['completed_clock'] is None:
                    prev['completed_clock']=t['clock']
            self.full.append(r);self.entry=False
            if 182<=op<=186 or op==187:self.pending.setdefault(f['lineinfo'],[]).append(i)
        elif kind=='callback':
            if not self.full:raise ValueError('Orphan callback')
            r=self.full[-1]
            if r['callback_clock'] is not None:raise ValueError('Ambiguous multiple callbacks')
            r['fields']['stack']=s.split('=',1)[1];r['callback_clock']=t['clock'];r['completed_clock']=t['clock']
        elif kind=='return_marker':
            f=fields(s[7:]);where=f.get('lineinfo');pending=self.pending.get(where,[])
            if not pending:raise ValueError('Orphan return marker')
            r=self.full[pending.pop()];r.update(call_returned=True,return_log_line=t['clock'][0],
                return_after_physical_event=len(self.full)-1,completed_clock=t['clock'])
            if self.full and 172<=self.full[-1]['opcode']<=177:
                self.full[-1]['completed_clock']=t['clock']
        elif s:self.opaque_after_start.append(t['clock'])

    @classmethod
    def from_text(cls,text,test_id):
        replay=cls(test_id)
        for t in tokens(text):replay.push(t)
        return replay


def serial_static(static,catalog):
    """All original compiled instructions, independent of this run's coverage."""
    byloc={(*tuple(p['locator'][k] for k in ('declaring_class','method','descriptor')),p['locator']['bytecode_offset']):p for p in catalog}
    instructions=[];controls=[]
    product_classes={p['locator']['declaring_class'] for p in catalog if p['scope']=='product'}
    for d,rows in sorted(static['methods'].items()):
        if d[0] not in product_classes:continue
        for b,r in sorted(rows.items()):
            p=byloc.get((*d,b))
            if p is None:raise ValueError('Static identity catalog incomplete; cannot select executed subset')
            if p['provenance_type']=='UNRESOLVED':raise ValueError('Unresolved static identity')
            instructions.append({'domain':list(d),'bci':b,'opcode':r['opcode'],'next':r['next'],
                'successors':sorted(r['successors']),'identity':p})
        controls.extend([[list(d),a,b] for a,b in sorted(static['controls'].get(d,[]))])
    return {'schema':'prefix_static_v1','instructions':instructions,'controls':controls,
        'static_initial_state':'unknown; pre-test setup excluded, not zero initialized',
        'features_config':'masked; constraints unverified',
        'dynamic_identity_lookup':catalog}


@dataclass
class Snapshot:
    model:dict
    audit:dict

    def fingerprint(self):
        h=hashlib.sha256()
        for k,v in sorted(self.model.items()):
            a=np.asarray(v);h.update(k.encode());h.update(str((a.shape,a.dtype)).encode());h.update(a.tobytes())
        return h.hexdigest()


class PrefixBuilder:
    def __init__(self,static,bundle,framework_static):
        self.static=static;self.bundle=bundle;self.framework_static=framework_static
        self.rows=bundle['instructions'];self.positions={(*r['domain'],r['bci']):i for i,r in enumerate(self.rows)}
        self.identities={tuple(p['locator'][k] for k in ('declaring_class','method','descriptor','bytecode_offset')):p
                         for p in bundle['dynamic_identity_lookup']}
        self.base=np.zeros((len(self.rows),len(FEATURES)),np.float32)
        for i,r in enumerate(self.rows):
            self.base[i,1]=1;self.base[i,5+OPCODES.index(opcode_group(r['opcode']))]=1
            self.base[i,14+('SOURCE_BACKED','COMPILER_GENERATED','UNRESOLVED').index(r['identity']['provenance_type'])]=1
            self.base[i,18]=1  # static available, never final executed
        self.static_edges=[(self.positions[(*d,a)],self.positions[(*d,b)],1) for d,a,b in bundle['controls']]

    def build(self,observer):
        # Copy because the old interpreter attaches derived records; no complete
        # graph, final outcome, final frames or post-hoc proof enters this API.
        full=copy.deepcopy(observer.full)
        raw,calls=project_observed_events(full,self.framework_static,allow_pending=True)
        if not raw:raise ValueError('No product/test instruction yet')
        if len({r['thread_id'] for r in full})!=1:raise ValueError('Multiple threads')
        flow=Dataflow(self.static,semantics_profile=FRAMEWORK_PROFILE,framework_calls=calls).run(raw)
        # Earliest incomplete semantics limits dependencies, not actual pre-events.
        cut=len(raw)
        for gap in flow.gaps:
            if gap['reason']=='unterminated_frame':continue  # expected for a prefix
            cut=min(cut,gap['event_index'])
        completed={i for i,r in enumerate(raw) if r['completed_clock'] is not None and i<cut}
        values=[v for v in flow.values if v.definition in completed and v.certain]
        # Unknown input definitions remain unknown; no invented input observation.
        values_by_id={v.id:v for v in values}
        sn=len(self.rows);en=len(raw);vn=len(values)
        x=np.zeros((sn+en+vn,22),np.float32);x[:sn]=self.base
        edges=list(self.static_edges);availability=[None]*sn;event_audit=[]
        for i,r in enumerate(raw):
            loc=(*r['domain'],r['bci'])
            p=self.identities.get(loc);inst=self.static['methods'].get(r['domain'],{}).get(r['bci'])
            if p is None or inst is None or p['opcode']!=r['opcode'] or inst['opcode']!=r['opcode'] or p['provenance_type']=='UNRESOLVED':raise ValueError('Instruction/identity mismatch')
            x[sn+i,1]=1;x[sn+i,5+OPCODES.index(opcode_group(r['opcode']))]=1
            x[sn+i,14+('SOURCE_BACKED','COMPILER_GENERATED','UNRESOLVED').index(p['provenance_type'])]=1
            x[sn+i,18]=1;x[sn+i,17]=int(i in completed);x[sn+i,19]=1
            availability.append(r['pre_clock'])
            if i:edges.append((sn+i-1,sn+i,6))
            event_audit.append({'local_node':sn+i,'event_index':i,'physical_event':r['physical_event_index'],
                'occurred_at':r['pre_clock'],'occurred_meaning':'instruction pre-observation, not completed result',
                'observed_at':r['pre_clock'],'evidence_available_at':r['pre_clock'],
                'completion_available_at':r['completed_clock'] if i in completed else None,
                'static_node':self.positions.get(loc),'frame':flow.events[i]['frame_id'] if i<=cut else None})
        vpos={v.id:sn+en+j for j,v in enumerate(values)}
        numeric=np.zeros((len(x),1),np.float64);mask=np.zeros((len(x),1),np.bool_)
        value_audit=[]
        for v in values:
            j=vpos[v.id];r=raw[v.definition];at=r['completed_clock']
            # A null-field exception reference exists only after the actual handler.
            transfer=next((t for t in flow.exception_transfers if t['exception_value_id']==v.id),None)
            if transfer:at=raw[transfer['target_event']]['pre_clock']
            x[j,2]=1;x[j,18]=1;availability.append(at)
            edges.append((sn+v.definition,j,3))
            obs=v.observed
            if obs.get('observed'):
                if r['callback_clock'] is None:raise ValueError('Numeric observation without callback')
                numeric[j,0]=obs['value'];mask[j,0]=True;at=max(at,r['callback_clock'])
            value_audit.append({'local_node':j,'value_id':v.id,'definition_event':v.definition,
                'occurred_at':None,'occurred_interval':[r['pre_clock'],at], 'observed_at':r['callback_clock'] if obs.get('observed') else None,
                'evidence_available_at':at,'observed':bool(obs.get('observed')),'dtype':obs.get('dtype'),
                'value':obs.get('value') if obs.get('observed') else None,
                'missing_reason':None if obs.get('observed') else 'no_numeric_callback; symbolic/derived_identity_only'})
        dynamic_edges=[]
        for val,i,role in flow.links:
            if val not in vpos or i not in completed:continue
            v=values_by_id[val]
            edges.extend([(vpos[val],sn+i,4),(sn+v.definition,sn+i,2)])
            dynamic_edges.append({'definition_event':v.definition,'use_event':i,'role':role,
                'evidence_available_at':raw[i]['completed_clock'],'value_id':val})
        ei=np.asarray([(a,b) for a,b,_ in edges],np.int64).reshape(-1,2).T
        et=np.asarray([r for _,_,r in edges],np.int64)
        degrees=np.bincount(ei[1],minlength=len(x)).astype(np.float32).reshape(-1,1)
        current=np.zeros((len(x),1),np.bool_);current[sn+en-1]=True
        pending=[i for i,r in enumerate(raw) if 182<=r['opcode']<=186 and not r['call_returned']]
        # Current heap values are intentionally not exposed. Historical observations
        # cannot be interpreted as unchanged heap while an external call is pending.
        model={'x':x,'edge_index':ei,'edge_type':et,'value':numeric,'value_observed':mask,
            'in_degree':degrees,'current_event':current,'event_nodes':np.arange(sn,sn+en,dtype=np.int64),
            'pending_call_nodes':np.asarray([sn+i for i in pending],np.int64),
            'heap_state_known':np.asarray([False],np.bool_)}
        audit={'clock':observer.last_clock,'events':event_audit,'values':value_audit,'dynamic_dependencies':dynamic_edges,
            'semantics_certified_before_event':cut,'pending_calls':pending,
            'completed_effects':[e for e in flow.framework_effects if e['accepted']],
            'exception_transfers':flow.exception_transfers,'opaque_observations':observer.opaque_after_start,
            'static_nodes':sn,'dynamic_events':en,'value_versions':vn,
            'semantic_gaps':flow.gaps,'availability':availability}
        validate_snapshot(Snapshot(model,audit))
        return Snapshot(model,audit)


def validate_snapshot(snapshot):
    m,a=snapshot.model,snapshot.audit;k=a['clock'];n=len(m['x'])
    if any(t is not None and t>k for t in a['availability']):raise ValueError('Future node')
    for e in a['dynamic_dependencies']:
        if not 0<=e['definition_event']<e['use_event']<a['dynamic_events'] or e['evidence_available_at']>k:raise ValueError('Future dependency')
    for v in a['values']:
        if v['evidence_available_at']>k or (v['observed'] and (v['observed_at'] is None or v['observed_at']>k)):raise ValueError('Future value')
    if m['edge_index'].size and (m['edge_index'].min()<0 or m['edge_index'].max()>=n):raise ValueError('Invalid endpoint')
    if m['x'][:a['static_nodes'],17].any() or m['x'][:,20:22].any():raise ValueError('Post-hoc static/config state')
    if np.any(m['value'][~m['value_observed']]!=0):raise ValueError('Missing value placeholder not masked')
    if not np.array_equal(m['in_degree'][:,0],np.bincount(m['edge_index'][1],minlength=n)):raise ValueError('Future normalization')
    return True
