"""Join actual bounded trace, original bytecode CFG and explicit SPL source maps."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from cwm_fp.data.common import ROOT, KEY, indexed, read_csv, sha256, write_json
from . import EXTRACTOR_VERSION
from .static import (SourceEvidence, load_static, mapping, program_identities, trusted_program_identity,
                     IDENTITY_SCHEMA, event_identity_id, instruction_identity_id)
from .trace import Dataflow, parse_trace
from .control import CONTROL_SCHEMA, control_evidence, control_evidence_errors


def sid(domain,bci):return 's:'+'#'.join(map(str,(*domain,bci)))


def graph_for(raw,static,mapped,features,lineage,status,identities=None, *, semantics_profile=None, framework_evidence=None):
    calls={};full=None
    if framework_evidence is not None:
        from .effects import project_and_verify
        from .trace import FRAMEWORK_PROFILE
        if semantics_profile!=FRAMEWORK_PROFILE:raise ValueError('Framework evidence requires versioned semantics profile')
        projected,calls,full=project_and_verify(framework_evidence)
        if projected!=raw:raise ValueError('Raw projection mismatch')
    flow=Dataflow(static,semantics_profile=semantics_profile,framework_calls=calls).run(raw);nodes=[];edges=[]
    path_evidence=control_evidence(raw,static,flow.events or [{**r,'frame_id':None} for r in raw],status,semantics_profile=semantics_profile,framework_evidence=framework_evidence) if static.get('control_schema')==CONTROL_SCHEMA and identities is not None else None
    # Do not emit interpreter guesses as runtime def-use after unverified exceptions.
    # Actual instruction records and scalar callbacks remain; raw logs are untouched.
    withheld=path_evidence is not None and not path_evidence['assessment']['accepted']
    if withheld:
        flow.values=[];flow.links=[]
        for event in flow.events:event['use_ids']=[];event['def_ids']=[]
    def node(identity,kind,attrs):nodes.append({'id':identity,'type':kind,'attrs':attrs})
    def edge(a,b,kind,evidence,**attrs):edges.append({'source':a,'target':b,'type':kind,'evidence':evidence,**attrs})
    for feature,enabled in sorted(features.items()):node('f:'+feature,'feature',{'enabled':enabled,'constraints_verified':False})
    for where,row in sorted(mapped.items()):
        domain=where[:3];bci=where[3];attrs={**row,'available':True,'executed':False}
        if identities is not None:attrs['opcode']=static['methods'][domain][bci]['opcode']
        node(sid(domain,bci),'statement',attrs)
        if row['feature']:edge('f:'+row['feature'],sid(domain,bci),'feature_controls','static',origin=row.get('mapping_note','coverage_location_and_roles'),mapping_confidence=row['mapping_confidence'])
    existing={n['id'] for n in nodes}
    for domain,controls in static['controls'].items():
        for a,b in sorted(controls):
            if sid(domain,a) in existing and sid(domain,b) in existing:
                edge(sid(domain,a),sid(domain,b),'control_dep','static',origin='original_bytecode_normal_CFG_postdominance',exception_control=False)
    events=flow.events or [{**r,'frame_id':None,'use_ids':[],'def_ids':[],'value_observed':False} for r in raw]
    previous={};mismatches=[]
    trace_hash=status.get('artifacts_sha256',{}).get('instrumented/run.log')
    event_identity_fields={}
    for r in events:
        i=r['event_index'];where=(*r['domain'],r['bci']);row=mapped.get(where)
        inst=static['methods'].get(r['domain'],{}).get(r['bci'])
        if not inst or (inst['line'],inst['opcode'])!=(r['line'],r['opcode']):mismatches.append(i)
        identity_attrs={}
        if identities is not None:
            identity=identities[where]
            identity_attrs={'program_identity':identity,'static_instruction_id':identity['instruction_id'],
                            'canonical_event_id':event_identity_id(lineage,trace_hash,r,identity['instruction_id'])
                            if trusted_program_identity(identity) and trace_hash else None}
            event_identity_fields[i]=identity_attrs
        node('e:'+str(i),'statement',{'executed':True,'event_index':i,'thread_id':r['thread_id'],'frame_id':r['frame_id'],
            **identity_attrs,
            'static_statement_id':sid(r['domain'],r['bci']) if row else None,
            'canonical_statement_id':row['canonical_statement_id'] if row else None,'mapping_confidence':row['mapping_confidence'] if row else 'test_or_unmapped',
            'canonical_source_line_id':row.get('canonical_source_line_id') if row else None,
            'composed_source_line_id':row.get('composed_source_line_id') if row else None,
            'source_construct':row.get('source_construct') if row else None,'source_statement_verified':False,'statement_ordinal':None,
            'composed_class':r['domain'][0],'composed_method':r['domain'][1],'descriptor':r['domain'][2],
            'composed_line':r['line'],'bytecode_offset':r['bci'],'opcode':r['opcode']})
        if row and row['feature']:edge('f:'+row['feature'],'e:'+str(i),'feature_controls','static',origin=row.get('mapping_note','coverage_location_and_roles'))
        for a,b in sorted(static['controls'].get(r['domain'],set())):
            if b==r['bci'] and sid(r['domain'],a) in existing:
                edge(sid(r['domain'],a),'e:'+str(i),'control_dep','static',origin='postdominance_instantiated_target',exception_control=False)
        thread=r['thread_id']
        if thread in previous:edge('e:'+str(previous[thread]),'e:'+str(i),'temporal','dynamic',event_index=i,origin='same_thread_record_order')
        previous[thread]=i
        if r.get('assertion'):
            node('a:'+str(i),'assertion',{'observed':True,'passed':r['assertion_passed'],'kind':r['fields']['callname'],
                                       'expected_actual_values_complete':False})
            edge('e:'+str(i),'a:'+str(i),'observes','dynamic',event_index=i,origin='observed_assertion_invocation_and_return')
    # Populate static executed flags only from actual matched locations.
    executed={sid(r['domain'],r['bci']) for r in raw}
    for n in nodes:
        if n['id'].startswith('s:'):n['attrs']['executed']=n['id'] in executed
    values={v.id:v for v in flow.values}
    for v in flow.values:
        node(v.id,'value',{**v.observed,'runtime_reference_known':v.ref is not None,'reaching_definition_supported':v.certain,
                          'definition_event':v.definition,'missing_reason':None if v.observed['observed'] else 'no_scalar_callback_for_this_definition'})
        if v.ref and v.ref.startswith('jvm_npe:'):
            nodes[-1]['attrs'].update(reference_semantics='verified_JVM_NPE_reference; no_numeric_value',
                                    evidence_available_at_event=next(t['target_event'] for t in flow.exception_transfers if t['exception_value_id']==v.id)
                                    if any(t['exception_value_id']==v.id for t in flow.exception_transfers) else v.definition)
        if v.definition is not None:edge('e:'+str(v.definition),v.id,'produces','dynamic',event_index=v.definition,origin='executed_opcode_definition')
    for val,event,role in flow.links:
        v=values[val]
        edge(val,'e:'+str(event),'consumes','dynamic',event_index=event,origin='supported_runtime_stack_local_or_field',role=role)
        if v.definition is not None and v.definition<event:
            edge('e:'+str(v.definition),'e:'+str(event),'data_dep','dynamic',event_index=event,definition_event=v.definition,
                 value_id=val,role=role,origin='runtime_reaching_definition')
    result=status['runs']['instrumented']['result']
    node('o:test','output',{'observed':status['complete'],'kind':'original_test_result','result':result['outcome']})
    if events:edge('e:'+str(events[-1]['event_index']),'o:test','observes','dynamic',event_index=events[-1]['event_index'],origin='runner_completion_record; not a claim of data dependence')
    product_events=[r for r in raw if (*r['domain'],r['bci']) in mapped]
    unmap=sum(not mapped[(*r['domain'],r['bci'])]['canonical_statement_id'] for r in product_events)
    static_unmap=sum(not r['canonical_statement_id'] for r in mapped.values())
    handlers=[d for d in {r['domain'] for r in raw} if static['handlers'].get(d)]
    executed_cfg_gaps={k:v for k,v in static['gaps'].items() if k in {'|'.join(r['domain']) for r in raw}}
    control_ok=path_evidence['assessment']['accepted'] if path_evidence is not None else not handlers
    eligible=bool(status['complete'] and status['parity'] and status['matches_historical_outcome'] and not flow.gaps
                  and not mismatches and control_ok and not executed_cfg_gaps and product_events)
    quality={'candidate_tier':'GRAPH','verified_tier':'T1' if eligible else 'T0_WITH_PARTIAL_TRACE',
             'canonical_mapping_granularity':'source_line','statement_ordinal_verified':False,
             'eligible_graph_structure':eligible,'canonical_mapping_complete':unmap==0 and static_unmap==0,
             'dynamic_product_mapping_complete':bool(product_events) and unmap==0,
             'canonical_mapped_product_instructions':len(mapped)-static_unmap,'unmapped_product_instructions':static_unmap,
             'canonical_mapped_product_events':len(product_events)-unmap,
             'unmapped_product_events':unmap,'bytecode_mismatches':mismatches,'unsupported_dynamic':flow.gaps,
             'executed_methods_with_exception_handlers':[list(d) for d in handlers], 'exception_control_supported':False,
             'executed_CFG_gaps':executed_cfg_gaps,
             'normal_cfg_gaps':static['gaps'],'events':len(raw),'product_events':len(product_events),
             'nodes':dict(Counter(n['type'] for n in nodes)),'edges':dict(Counter(e['type'] for e in edges)),
             'observed_scalar_events':sum(r.get('value_observed',False) for r in events),
             'state_value_training_enabled':False,'intervention_training_enabled':False,'constraints_verified':False}
    graph={**lineage,'tier':'GRAPH','nodes':nodes,'edges':edges,'quality':quality,
           'provenance':{'producer':'BRFL_bounded_callback_adapter','extractor_version':EXTRACTOR_VERSION,'coverage_only':False,
                         'evidence_scope':'single_thread_supported_opcodes_normal_control_only','learnable':eligible}}
    if framework_evidence is not None:
        graph['framework_effects']=flow.framework_effects
        graph['provenance']['extractor_version']='phase03B4-0.6'
        quality.update(physical_trace_events=len(full),framework_internal_events=len(full)-len(raw),
                       framework_effects_valid=bool(flow.framework_effects) and all(e['accepted'] for e in flow.framework_effects),
                       event_scope='product_and_test_projection; framework_body_retained_in_effect_evidence')
    if identities is not None:
        graph['instruction_catalog']=[identities[k] for k in sorted(identities)]
        graph['provenance'].update(identity_schema=IDENTITY_SCHEMA,trace_sha256=trace_hash)
        quality.update(identity_metrics(graph)[0],dependency_semantics_valid=eligible,suite_completeness=None)
        quality['identity_schema']=IDENTITY_SCHEMA
    if path_evidence is not None:
        graph['control_flow_evidence']=path_evidence
        graph['provenance']['control_schema']=path_evidence['schema']
        graph['provenance']['evidence_scope']='single_thread_supported_opcodes_verified_normal_path; no_exception_semantics'
        quality.update(normal_path_verified=path_evidence['assessment']['accepted'],control_semantics_scope='normal-path-only',
                       exception_path_classification=path_evidence['assessment']['classification'],
                       dynamic_dependency_emission='withheld_unverified_control_path' if withheld else 'supported_dataflow',
                       executed_methods_with_exception_handlers=path_evidence['assessment']['executed_methods_with_exception_handlers'])
        if semantics_profile:
            assessment=path_evidence['assessment']
            graph['provenance']['evidence_scope']='single_thread; normal_paths_and_exact_null_field_NPE_catch; other_exceptions_unsupported'
            quality.update(control_path_verified=assessment['accepted'],control_semantics_scope=assessment['scope'],
                           normal_path_verified=bool(assessment['accepted'] and not assessment['certified_exception_transitions']),
                           certified_null_field_transitions=assessment['certified_exception_transitions'])
    normalized=[]
    for r in events:
        normalized.append({**lineage,'event_index':r['event_index'],'event_type':'assertion' if r.get('assertion') else 'branch' if r.get('branch_taken') is not None else 'statement_enter',
                           **event_identity_fields.get(r['event_index'],{}),
                           'statement_id':'e:'+str(r['event_index']),'thread_id':r['thread_id'],'frame_id':r['frame_id'],
                           'composed_location':list(r['domain'])+[r['line'],r['bci']],'opcode':r['opcode'],'def_ids':r['def_ids'],'use_ids':r['use_ids'],
                           'value_observed':r.get('value_observed',False),'value_summary':r.get('value_summary',{'observed':False}),
                           'branch_taken':r.get('branch_taken'),'observed_successor':r.get('observed_successor'),
                           'assertion_passed':r.get('assertion_passed')})
    return graph,normalized


def identity_metrics(graph):
    """Recompute identity coverage from records; consumers do not trust summary flags."""
    errors=[];catalog={};nodes=graph['nodes'];static=[];dynamic=[];valid={}
    lineage={k:graph[k] for k in (*KEY,'test_id')};trace_hash=graph['provenance'].get('trace_sha256')
    for item in graph.get('instruction_catalog',[]):
        locator=instruction_identity_id(item.get('locator',{}))
        if locator in catalog:errors.append('duplicate_instruction_catalog_entry')
        catalog[locator]=item
    seen_events=set()
    for n in nodes:
        if n['type']!='statement':continue
        a=n['attrs'];p=a.get('program_identity',{});loc=p.get('locator',{});is_dynamic='event_index' in a
        ok=trusted_program_identity(p) and catalog.get(instruction_identity_id(loc))==p
        ok=ok and all(a.get(k)==loc.get(v) for k,v in [('composed_class','declaring_class'),('composed_method','method'),('descriptor','descriptor'),('bytecode_offset','bytecode_offset')])
        ok=ok and a.get('opcode')==p.get('opcode')
        if is_dynamic:
            dynamic.append(n)
            actual=a.get('canonical_event_id')
            ok=ok and bool(trace_hash) and a.get('static_instruction_id')==p.get('instruction_id') and actual==event_identity_id(lineage,trace_hash,a,p.get('instruction_id'))
            if actual in seen_events:ok=False
            seen_events.add(actual)
            link=a.get('static_statement_id')
            if p.get('scope')=='product':
                target=next((x for x in nodes if x['id']==link),None)
                ok=ok and target is not None and target['attrs'].get('program_identity')==p and 'event_index' not in target['attrs']
        else:
            static.append(n);ok=ok and p.get('scope')=='product'
        valid[n['id']]=bool(ok)
        if not ok:errors.append('unresolved_or_inconsistent_program_identity:'+n['id'])
    total=len(static);nsource=sum(valid[n['id']] and n['attrs']['program_identity']['provenance_type']=='SOURCE_BACKED' for n in static)
    ngen=sum(valid[n['id']] and n['attrs']['program_identity']['provenance_type']=='COMPILER_GENERATED' for n in static)
    product_dynamic=[n for n in dynamic if n['attrs'].get('program_identity',{}).get('scope')=='product']
    metrics={'source_mapping_coverage':nsource/total if total else 0.,'generated_identity_coverage':ngen/total if total else 0.,
             'source_backed_instruction_count':nsource,'compiler_generated_instruction_count':ngen,
             'program_instruction_count':total,'unresolved_identity_count':total-nsource-ngen,
             'dynamic_event_identity_coverage':sum(valid[n['id']] for n in dynamic)/len(dynamic) if dynamic else 0.,
             'dynamic_product_identity_coverage':sum(valid[n['id']] for n in product_dynamic)/len(product_dynamic) if product_dynamic else 0.,
             'unresolved_dynamic_event_count':sum(not valid[n['id']] for n in dynamic),
             'program_identity_complete':bool(static and dynamic and not errors),
             'identity_metric_scope':'static_product_instructions; dynamic_all_recorded_product_and_test_events'}
    return metrics,errors


def identity_errors(graph):
    if graph.get('provenance',{}).get('identity_schema')!=IDENTITY_SCHEMA:return ['unsupported_program_identity_schema']
    try:metrics,errors=identity_metrics(graph)
    except (KeyError,TypeError,ValueError):return ['malformed_program_identity_records']
    q=graph.get('quality',{})
    if any(q.get(k)!=v for k,v in metrics.items()):errors.append('identity_summary_disagrees_with_records')
    if not metrics['program_identity_complete']:errors.append('incomplete_program_identity')
    if q.get('dependency_semantics_valid')!=q.get('eligible_graph_structure'):errors.append('contradictory_dependency_semantics_status')
    return errors


def validate_graph(graph,events):
    ids={n['id'] for n in graph['nodes']}
    if len(ids)!=len(graph['nodes']):raise ValueError("Duplicate graph nodes")
    order=[e['event_index'] for e in events]
    if order!=list(range(len(events))):raise ValueError("Event order is not complete and unique")
    for edge in graph['edges']:
        if edge['source'] not in ids or edge['target'] not in ids:raise ValueError("Unknown graph endpoint")
        if edge['type']=='data_dep':
            if edge['evidence']!='dynamic' or edge['definition_event']>=edge['event_index']:raise ValueError("Invalid runtime reaching-definition order")
    if graph.get('quality',{}).get('eligible_graph_structure'):
        q=graph['quality']
        if any(q[k] for k in ('bytecode_mismatches','unsupported_dynamic','executed_CFG_gaps')):
            raise ValueError('Graph with evidence gaps cannot be eligible')
        if graph.get('provenance',{}).get('control_schema') or 'control_flow_evidence' in graph:
            if control_evidence_errors(graph):raise ValueError('Graph lacks verified normal-path evidence')
        elif q['executed_methods_with_exception_handlers']:raise ValueError('Legacy exception-control evidence gap')
        if q['verified_tier']!='T1' or not graph['provenance']['learnable']:raise ValueError('Contradictory graph eligibility')
    encoded=json.dumps(graph)
    for forbidden in ('"buggy"','"label"','"label_provenance"','"source_label"','"obsvalue"','"posterior"'):
        if forbidden in encoded:raise ValueError("Forbidden metadata in graph")
    if graph.get('provenance',{}).get('identity_schema'):
        metrics,_=identity_metrics(graph)
        if any(graph['quality'].get(k)!=v for k,v in metrics.items()):raise ValueError('Inconsistent identity quality summary')
        by_event={n['attrs']['event_index']:n['attrs'] for n in graph['nodes'] if 'event_index' in n['attrs']}
        for e in events:
            for k in ('program_identity','static_instruction_id','canonical_event_id'):
                if e.get(k)!=by_event[e['event_index']].get(k):raise ValueError('Event/static identity association mismatch')


def extract(work, output=None, source_evidence_path=None, identity_binding_path=None, *, semantics_profile=None):
    from cwm_fp.baselines.boundaries import install_io_guard
    install_io_guard('features')
    output=output or work
    output.mkdir(parents=True,exist_ok=True)
    evidence=SourceEvidence(source_evidence_path) if source_evidence_path else None
    binding=json.loads(identity_binding_path.read_text(encoding='utf-8')) if identity_binding_path else None
    if binding is not None and evidence is None:raise ValueError('Identity certification requires actual AST evidence')
    collection=json.loads((work/'collection.json').read_text(encoding='utf-8'))
    products=indexed(read_csv(ROOT/'data/processed/manifests/products.csv',KEY))
    allgraphs=[];allevents=[];allmaps=[];maps={};quality=[]
    suite_expected={};suite_collected={}
    if binding is not None:
        for test in read_csv(ROOT/'data/processed/manifests/tests.csv',(*KEY,'test_id')):
            suite_expected.setdefault(tuple(test[k] for k in KEY),set()).add(test['test_id'])
        for record in collection['records']:suite_collected.setdefault(tuple(record[k] for k in KEY),set()).add(record['test_id'])
    for record in collection['records']:
        lineage={k:record[k] for k in (*KEY,'test_id')};product=record['product_id'];variant=Path(record['variant'])
        features=json.loads(products[tuple(lineage[k] for k in KEY)]['features_json'])
        if product not in maps:
            static=load_static(work/product/'static.tsv');rows,by_inst,config=mapping(variant,static,features,evidence)
            allmaps.extend({**{k:lineage[k] for k in KEY},**r} for r in rows);maps[product]=(static,by_inst,config)
        static,by_inst,config=maps[product];stage=Path(record['directory'])
        raw=parse_trace(stage/'instrumented/run.log',record['test_id'],collection['config']['max_events'],collection['config']['max_output_bytes'])
        identities=program_identities(variant,static,by_inst,evidence,binding,required=[(*r['domain'],r['bci']) for r in raw]) if binding is not None else None
        graph,events=graph_for(raw,static,by_inst,features,lineage,record['status'],identities,semantics_profile=semantics_profile)
        if identities is not None:
            k=tuple(lineage[x] for x in KEY);expected=suite_expected[k];collected=suite_collected[k]
            graph['quality']['suite_completeness']={'complete':expected==collected,'expected_tests':len(expected),
                'collected_tests':len(collected),'basis':'registered_original_suite_not_absolute_historical_completeness'}
        graph['provenance'].update({'trace_sha256':sha256(stage/'instrumented/run.log'),'status_sha256':sha256(stage/'status.json'),
                                    'static_sha256':sha256(work/product/'static.tsv'),'roles_sha256':sha256(variant/'roles.meta'),
                                    'agent_sha256':record['status']['inputs']['agent_sha256']})
        if evidence:graph['provenance']['source_evidence_sha256']=evidence.digest
        validate_graph(graph,events);allgraphs.append(graph);allevents.extend(events)
        quality.append({**lineage,**graph['quality']})
        print(product,record['test_id'],graph['quality']['verified_tier'],graph['quality']['edges'],graph['quality']['unsupported_dynamic'],flush=True)
    if binding is not None:
        for row in allmaps:
            if row['executable']:
                where=(row['composed_class'],row['composed_method'],row['descriptor'],row['bytecode_offset'])
                row['program_identity']=maps[row['product_id']][1][where]['program_identity']
    for name,rows in [('graphs.jsonl',allgraphs),('events.jsonl',allevents),('source_map.jsonl',allmaps)]:
        (output/name).write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows),encoding='utf-8')
    write_json(output/'quality_report.json',{'extractor_version':EXTRACTOR_VERSION,'tests':quality,'source_map_records':len(allmaps),
               'source_map_counts':dict(Counter(r.get('mapping_confidence',r.get('status','unknown')) for r in allmaps)),
               'graphs':len(allgraphs),'events':len(allevents),'T1_tests':sum(q['eligible_graph_structure'] for q in quality),
               'T2_tests':0,'T3_tests':0,'raw_feature_model_constraints_verified':False})
    write_json(output/'extractor_version.json',{'version':EXTRACTOR_VERSION,'source_sha256':{p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(Path(__file__).parent.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}})


def main():
    p=argparse.ArgumentParser();p.add_argument('--work-dir',type=Path,default=ROOT/'data/processed/lddg/phase03_bank_v2')
    p.add_argument('--output-dir',type=Path);p.add_argument('--source-evidence',type=Path)
    p.add_argument('--identity-binding',type=Path)
    a=p.parse_args();extract(a.work_dir.resolve(),a.output_dir,a.source_evidence,a.identity_binding)

if __name__=='__main__':main()
