"""Bounded phase05A production/target/checker entry point; no classifier labels."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path
import numpy as np
from cwm_fp.data.common import ROOT, sha256, write_json
from cwm_fp.baselines.boundaries import install_io_guard
from cwm_fp.lddg.static import load_static
from cwm_fp.lddg.trace import parse_trace
from cwm_fp.lddg.effects import project_and_verify, HELPER, STATIC_SHA
from .graph_package import ProductPackage, digest
from .graph_encoder import OPCODES, FEATURES, opcode_group
from .package_training import process_memory
from .prefix_observer import ObservationReplay, PrefixBuilder, serial_static, tokens, validate_snapshot

PACKAGE=Path('D:/dataset/whn/cwm_fp_generated/data/processed/lddg/phase04D1_bank_v1/package')
ANCHOR='71770793f4c7a063c084591f5ebf184d21368296895b3ebe9e3e4d77a7d41289'
OUT=Path('D:/dataset/whn/cwm_fp_generated/data/processed/prefix/phase05A_v1')
PROTOCOL=ROOT/'outputs/phase05A_protocol.json'
FRAMEWORK=ROOT/'data/processed/lddg/phase03B4_framework_v1/framework/static.tsv'
SOURCE_FILES=['src/cwm_fp/models/prefix_observer.py','src/cwm_fp/models/prefix_dataset.py','src/cwm_fp/lddg/effects.py','src/cwm_fp/lddg/trace.py','src/cwm_fp/lddg/static.py','src/cwm_fp/models/graph_encoder.py','src/cwm_fp/models/graph_package.py']

def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def lines(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf8').splitlines() if x]

def guard(manifest,allow_validation):
    install_io_guard('phase05A_no_product_targets')
    denied={str((PACKAGE/g[k]['path']).resolve()).lower() for p in manifest['products']
        if p['partitions']['grouped']=='test' or (p['partitions']['grouped']=='val' and not allow_validation)
        for g in p['tests'] for k in ('metadata','tensor')}
    access=[]
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        name=str(Path(args[0]).resolve());low=name.lower().replace('\\','/')
        if name.lower() in denied:raise PermissionError('Sealed trajectory partition')
        if '/labels/' in low:raise PermissionError('No FP/TP targets in phase05A')
        if low.endswith('/run.log') or '/package/graphs/' in low:access.append(name)
    sys.addaudithook(hook)
    return access

def static_path(source,graph):
    product=graph['product_id'];wanted=graph['provenance']['static_sha256']
    roots=[Path(source['graph_path']).parent.parent,
        Path('D:/dataset/whn/cwm_fp_generated/data/processed/lddg/phase03B2_v1'),
        ROOT/'data/processed/lddg/phase03B1_bank0001_v1',ROOT/'data/processed/lddg/phase03A_mapping_v1']
    for root in roots:
        candidates=[root/product/'static.tsv'] if (root/product/'static.tsv').exists() else list(root.glob('p*/'+product+'/static.tsv'))
        for p in candidates:
            if sha256(p)==wanted:return p
    raise ValueError('Original static artifact not located by bound hash')

def source_product(p,receipt):
    source=receipt['source_paths'][json.dumps(p['identity'])];path=Path(source['graph_path'])
    if sha256(path)!=source['audit']['input_sha256']['graphs']:raise ValueError('Original graph file changed')
    gs={g['test_id']:g for g in lines(path)}
    if set(gs)!=set(p['registered_tests']):raise ValueError('Source suite differs')
    first=gs[p['registered_tests'][0]];sp=static_path(source,first);s=load_static(sp)
    collection=load(sp.parent.parent/'collection.json')
    records={r['test_id']:r for r in collection['records']}
    return source,gs,sp,s,records

def source_trace(g,ref,pack,records):
    tensor=pack.load_test(ref)
    att=tensor.metadata['collector_attestation']
    if digest(g)!=att['source_graph_sha256']:raise ValueError('Source graph does not match portable receipt')
    proof=g['control_flow_evidence']['proof'];e=proof.get('framework_evidence')
    path=Path(e['trace_path']) if e else Path(records[g['test_id']]['directory'])/'instrumented/run.log'
    if sha256(path)!=att['trace_sha256']:raise ValueError('Original physical trace changed')
    if e:projected,_,full=project_and_verify(e)
    else:full=parse_trace(path,g['test_id'],physical_evidence=True);projected=full
    if len(projected)!=ref['events']:raise ValueError('Physical/projection count mismatch')
    return path,path.read_text(encoding='utf8'),full,tensor

def freeze():
    if PROTOCOL.exists():raise ValueError('Protocol already frozen')
    receipt=load(PACKAGE.parent/'export_receipt.json')
    if receipt['manifest_sha256']!=ANCHOR:raise ValueError('External receipt mismatch')
    pack=ProductPackage(PACKAGE,expected_manifest_sha256=ANCHOR);access=guard(pack.manifest,False)
    p=next(p for p in pack.products if p['partitions']['grouped']=='train')
    source,gs,sp,s,records=source_product(p,receipt);ref=p['tests'][0];g=gs[ref['identity'][3]]
    path,text,full,_=source_trace(g,ref,pack,records)
    observer=ObservationReplay.from_text(text,g['test_id'])
    if len(observer.full)!=len(full):raise ValueError('Training physical format mismatch')
    bundle=serial_static(s,g['instruction_catalog']);framework=load_static(FRAMEWORK)
    PrefixBuilder(s,bundle,framework).build(observer)
    protocol={'stage_id':'05A','protocol':'grouped_bank_pilot_v1','schema':'physical_prefix_dataset_v1',
        'source_manifest_sha256':ANCHOR,'external_receipt_sha256':sha256(PACKAGE.parent/'export_receipt.json'),
        'source_sha256':{f:sha256(ROOT/f) for f in SOURCE_FILES},
        'public_sha256':{f:sha256(ROOT/f) for f in ['data/processed/manifests/products.csv','data/processed/manifests/tests.csv','data/processed/splits/splits.csv']},
        'task_vocabulary':list(OPCODES),'node_features':list(FEATURES),
        'actual_scope':{'train_products':17,'train_traces':283,'val_products':5,'val_traces':85,'test':'manifest identities only'},
        'anchor_definition':'Immediately after a product/test pre-instruction record, before its appended callback; one anchor per such actual record',
        'observation_clock':'1-based physical log line plus 0-based ordered subrecord; instruction pre then scalar callback; no timestamps fabricated',
        'A':'semantic category of the next observed product/test pre-instruction; intervening callbacks/returns/framework regions explicitly audited, not claimed a unit JVM transition',
        'B':'current conditional opcode153..166,198,199; two certified CFG successors; real next same-domain/frame instruction; target jump versus fallthrough',
        'end_target':'disabled; raw root return is a pre-record; no completion observation in this stream',
        'numeric':'historical real scalar callback values plus independent masks; local arithmetic pairs only with all operands directly observed',
        'static_input_policy':'all compiled PRODUCT instructions and normal static control dependence; test CFG for target checks; per-event test identity lookup only on actual arrival; no executed-subset static construction',
        'field_visibility_rules':{'executed':'0 for all static nodes; dynamic completion only after callback/actual successor/return',
            'values':'callback only; unknown placeholder0 with independent false mask',
            'data_dependencies':'legacy interpreter on observed prefix only; incomplete semantics withheld',
            'framework_effects':'only after real RET and observed-region checks; pending heap unknown',
            'null_exception':'only after actual handler arrival; symbolic reference not a scalar',
            'final_assertion_and_output':'excluded at all input boundaries','config':'masked',
            'identity':'audit only; local connection indices, no learned unique-ID columns',
            'normalization':'current visible edges only; no fit statistics'},
        'rejections':['unverified source','unknown clock','identity/opcode mismatch','multiple threads','orphan/malformed callback or return','budget'],
        'masks':['no later observed instruction','unproven branch boundary/frame/CFG','incomplete numeric observation'],
        'diagnostics':{'majority':'train targets only; ties fixed vocabulary order','static_successor':'non-call/non-return one normal successor; otherwise abstain; no fitted rules'},
        'checks':'all anchors timing/endpoints; all physical observations independent truncated replay and five suffix mutations (stronger than first/middle/last + critical)',
        'budgets':{'seconds':3600,'output_bytes':2*1024**3,'min_free_bytes':4*1024**3,'process_rss_bytes':4*1024**3},
        'training_format_probe':{'identity':ref['identity'],'trace_sha256':sha256(path),'physical_instructions':len(full),'static_product_instructions':len(bundle['instructions'])},
        'validation_trajectories_opened_before_freeze':0,'classification_target_access':'none permitted','created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    write_json(PROTOCOL,protocol);write_json(OUT/'protocol_receipt.json',{'sha256':sha256(PROTOCOL)})
    write_json(OUT/'freeze_access.json',{'paths':access,'product_targets_opened':0,'validation_trajectories_opened':0})
    for f in SOURCE_FILES:
        dest=OUT/'executed_source'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dest)
    print('FROZEN',sha256(PROTOCOL),flush=True)

def targets(observer,static):
    """Post-observation target side. Never called by PrefixBuilder."""
    full=observer.full;raw=[r for r in full if r['domain'][0]!=HELPER];rows=[]
    for i,r in enumerate(raw):
        nxt=raw[i+1] if i+1<len(raw) else None;op=r['opcode'];inst=static['methods'][r['domain']][r['bci']]
        a=opcode_group(nxt['opcode']) if nxt else None
        gap=full[r['physical_event_index']+1:nxt['physical_event_index']] if nxt else []
        condition=153<=op<=166 or op in (198,199)
        b=None;why='not_conditional'
        if condition:
            if not nxt:why='no_observed_successor'
            elif nxt['domain']!=r['domain'] or nxt['entry']:why='different_or_unknown_frame'
            elif len(inst['successors'])!=2 or nxt['bci'] not in inst['successors']:why='unverified_binary_successor'
            else:b='fallthrough' if nxt['bci']==inst['next'] else 'jump';why=None
        unique=next(iter(inst['successors'])) if len(inst['successors'])==1 and not (172<=op<=177 or 182<=op<=186) else None
        pred=opcode_group(static['methods'][r['domain']][unique]['opcode']) if unique is not None and unique>=0 else None
        rows.append({'event_index':i,'anchor':r['pre_clock'],'A':a,'A_mask':a is not None,'A_reason':None if a else 'no_later_observation; NOT_EOS',
            'A_available_at':nxt['pre_clock'] if nxt else None,'B':b,'B_mask':b is not None,'B_reason':why,
            'B_available_at':nxt['pre_clock'] if b is not None else None,'deterministic_prediction':pred,
            'interval_kind':'call' if 182<=op<=186 else 'return' if 172<=op<=177 else 'conditional' if condition else 'ordinary',
            'intervening_framework_instructions':len(gap),'current_opcode':op,
            'branch_audit_key':[list(r['domain']),r['bci']], 'projected_step_is_unit_JVM_step':not gap and not (182<=op<=186 or 172<=op<=177)})
    return rows

def suffix_variants(suffix):
    """Artificial internal test copies ONLY; never authenticated as real inputs."""
    import re
    return {'deleted':'', 'changed_values':re.sub(r'stack=([ISBZJFD]):[^\r\n,]*',r'stack=\1:987654',suffix),
        'changed_return_or_exception':suffix.replace('###RET@','###BROKEN@').replace('NullPointerException','IllegalStateException'),
        'changed_terminal':suffix+'\nFINAL_TEST_STATUS=FAIL\n',
        'extended':suffix+'\n###Artificial::extension\nopcode=3(iconst_0),pushnum=1,lineinfo=Artificial#extension#()V#1#0,nextinst=1,thread=1,stack=I:71\n'}

def check_prefix(builder,incremental,text,t,test_id):
    expected=builder.build(incremental);cut=t['cut'];prefix=text[:cut]
    # Parse a genuinely truncated raw text independently, not a full-graph filter.
    replay=ObservationReplay.from_text(prefix,test_id);actual=builder.build(replay)
    if expected.fingerprint()!=actual.fingerprint():raise AssertionError('Truncated replay input differs')
    for suffix in suffix_variants(text[cut:]).values():
        # Stream tokenizer over a destructive copy, stop by the physical cut BEFORE
        # consuming any suffix; no source-hash rejection is used as proof.
        changed=ObservationReplay(test_id)
        for tok in tokens(prefix+suffix):
            if tok['cut']>cut:break
            changed.push(tok)
        if builder.build(changed).fingerprint()!=expected.fingerprint():raise AssertionError('Suffix changed prefix tensor')
    return expected

def write_jsonl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')

def produce():
    protocol=load(PROTOCOL)
    if sha256(PROTOCOL)!=load(OUT/'protocol_receipt.json')['sha256']:raise ValueError('Protocol modified')
    for f,h in protocol['source_sha256'].items():
        if sha256(ROOT/f)!=h:raise ValueError('Source changed after task freeze')
    for f,h in protocol['public_sha256'].items():
        if sha256(ROOT/f)!=h:raise ValueError('Public split/manifest changed')
    if (OUT/'production.json').exists():raise ValueError('Do not overwrite finished production')
    pack=ProductPackage(PACKAGE,expected_manifest_sha256=ANCHOR);access=guard(pack.manifest,True)
    receipt=load(PACKAGE.parent/'export_receipt.json');framework=load_static(FRAMEWORK)
    if sha256(FRAMEWORK)!=STATIC_SHA:raise ValueError('Framework bytecode pin mismatch')
    started=time.monotonic();status=[];all_targets=[];checks=[];duplicates=defaultdict(list);branches=defaultdict(set)
    resources={'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,'gpu_training':False,
        'java_executions':0,'world_model_training':False,'monitor':'per-trace actual worker OS peak RSS and elapsed/disk checks; not OS hard quota'}
    def budget():
        mem=process_memory();free=shutil.disk_usage(OUT).free
        total=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
        # Actual worker lifetime peak, not Windows virtualenv launcher RSS.
        peak=mem.get('peak_rss_bytes') or mem.get('peak_working_set_bytes')
        if time.monotonic()-started>3600 or total>2*1024**3 or free<4*1024**3 or (peak is not None and peak>4*1024**3):raise RuntimeError('phase05A resource budget reached')
        return {'process_memory':mem,'output_bytes':total,'free_bytes':free}
    try:
        for part in ('train','val'):
            for product in [p for p in pack.products if p['partitions']['grouped']==part]:
                source,gs,sp,static,records=source_product(product,receipt)
                for ref in product['tests']:
                    budget();number=len(status);ident=ref['identity'];g=gs[ident[3]];dest=OUT/'traces'/f'{number:04d}'
                    if dest.exists():raise ValueError('Incomplete previous attempt exists; explicit new attempt required')
                    dest.mkdir(parents=True)
                    row={'system_id':ident[0],'bug_version_id':ident[1],'product_id':ident[2],'test_id':ident[3],'partition':part,
                        'source_events':ref['events'],'evidence_path':str(dest),'source_graph_path':source['graph_path'],
                        'accepted_prefixes':0,'rejected_prefixes':0,'status':'STARTED'}
                    try:
                        path,text,physical,tensor=source_trace(g,ref,pack,records)
                        bundle=serial_static(static,g['instruction_catalog']);builder=PrefixBuilder(static,bundle,framework)
                        obs=ObservationReplay(ident[3]);index=[];trace_checks=[];field_times=[];numeric_pairs=[];last=None
                        for t in tokens(text):
                            obs.push(t)
                            if not obs.active or not obs.full or not any(r['domain'][0]!=HELPER for r in obs.full):continue
                            # Empty logger lines need not duplicate the same tensor.
                            if t['kind']=='opaque' and not t['text']:continue
                            snap=check_prefix(builder,obs,text,t,ident[3]);last=snap
                            trace_checks.append({'clock':t['clock'],'kind':t['kind'],'replay':True,'suffix_cases':5,'tensor_sha256':snap.fingerprint()})
                            if t['kind']=='instruction' and obs.full[-1]['domain'][0]!=HELPER:
                                i=snap.audit['dynamic_events']-1
                                item={'event_index':i,'clock':t['clock'],'cut':t['cut'],'input_sha256':snap.fingerprint(),
                                    'observed_prefix_events':i+1,'observed_numeric_versions':int(snap.model['value_observed'].sum()),
                                    'value_versions':snap.audit['value_versions'],'semantic_masked_from_event':snap.audit['semantics_certified_before_event']}
                                index.append(item);duplicates[item['input_sha256']].append([part,*ident,i])
                                if i in (0,ref['events']//2,ref['events']-1):
                                    np.savez_compressed(dest/f'example_{i}.npz',**snap.model)
                                    write_json(dest/f'example_{i}.audit.json',snap.audit)
                            if t['kind']=='callback' and obs.full[-1]['domain'][0]!=HELPER:
                                i=snap.audit['dynamic_events']-1;op=obs.full[-1]['opcode']
                                vals={v['value_id']:v for v in snap.audit['values']}
                                ins=[vals[d['value_id']] for d in snap.audit['dynamic_dependencies'] if d['use_event']==i]
                                outs=[v for v in vals.values() if v['definition_event']==i and v['observed']]
                                if 96<=op<=152 and ins and outs and all(v['observed'] and v['observed_at']<obs.full[-1]['pre_clock'] for v in ins):
                                    numeric_pairs.append({'event_index':i,'opcode':op,'input_values':ins,'result_values':outs,'result_observed_at':t['clock']})
                        if len(obs.full)!=len(physical):raise ValueError('Physical instruction loss')
                        projected=[r for r in obs.full if r['domain'][0]!=HELPER]
                        for r,e in zip(projected,tensor.metadata['events']):
                            if [*r['domain'],r['line'],r['bci']]!=e['composed_location'] or r['opcode']!=e['opcode']:raise ValueError('Frozen event correspondence mismatch')
                        if len(index)!=ref['events']:raise ValueError('Missing anchor')
                        tt=targets(obs,static)
                        for target,item in zip(tt,index):
                            if target['anchor']!=item['clock']:raise ValueError('Target-anchor mismatch')
                            for task in ('A','B'):
                                if target[task+'_mask'] and not target[task+'_available_at']>item['clock']:raise ValueError('Already observed prediction target')
                            target['partition']=part;target['trace_index']=number;all_targets.append(target)
                            if target['B_mask']:
                                # Audit alignment within a product/version (no forced cross-version equality).
                                branches[json.dumps([ident[:3],target['branch_audit_key']])].add(target['B'])
                        field_times=last.audit['events']+last.audit['values']
                        write_json(dest/'static.json',bundle);shutil.copy2(sp,dest/'static.tsv')
                        write_jsonl(dest/'observations.jsonl',obs.observations);write_jsonl(dest/'prefix_index.jsonl',index)
                        write_jsonl(dest/'targets.jsonl',tt);write_jsonl(dest/'checks.jsonl',trace_checks)
                        write_json(dest/'field_times.json',field_times);write_json(dest/'numeric_pairs.json',numeric_pairs)
                        write_json(dest/'source.json',{'identity':ident,'partition':part,'trace_path':str(path),'trace_sha256':sha256(path),
                            'static_path':str(sp),'static_sha256':sha256(sp),'package_reference':ref,'source_graph_sha256':digest(g),
                            'input_excludes_final_results':True,'pre_test_setup':'excluded; initial state unknown'})
                        row.update(status='PASS',physical_instructions=len(obs.full),physical_observations=len(obs.observations),
                            accepted_prefixes=len(index),A_valid=sum(t['A_mask'] for t in tt),A_masked=sum(not t['A_mask'] for t in tt),
                            B_valid=sum(t['B_mask'] for t in tt),B_masked=sum(not t['B_mask'] for t in tt),
                            scalar_observations=sum(bool(r.get('fields',{}).get('stack','').startswith(('I:','S:','B:','Z:','J:','F:','D:'))) for r in projected),
                            final_visible_value_versions=last.audit['value_versions'],final_visible_numeric_versions=int(last.model['value_observed'].sum()),
                            numeric_pairs=len(numeric_pairs),check_positions=len(trace_checks),suffix_checks=5*len(trace_checks),
                            calls=sum(182<=r['opcode']<=186 for r in projected),conditionals=sum(153<=r['opcode']<=166 or r['opcode'] in (198,199) for r in projected),
                            exception_transfers=len(last.audit['exception_transfers']),framework_effects=len(last.audit['completed_effects']),
                            time_boundary_issues='root completion unobserved; no EOS target; incomplete calls/data withheld at their pending boundaries',
                            missing_numeric_versions=last.audit['value_versions']-int(last.model['value_observed'].sum()),source_trace_path=str(path))
                    except Exception as exc:
                        row.update(status='REJECTED',reason=f'{type(exc).__name__}: {exc}',rejected_prefixes=ref['events'])
                        write_json(dest/'failure.json',row)
                    status.append(row);checks.append({'trace':number,'status':row['status'],'positions':row.get('check_positions',0)})
                    write_json(OUT/'checkpoint.json',{'complete_traces':len(status),'rows':status,'elapsed_seconds':time.monotonic()-started})
                    if number%20==0:print('TRACE',number,part,row['status'],row.get('reason',''),flush=True)
        resources.update(budget(),seconds=time.monotonic()-started)
    except Exception as exc:
        resources.update(seconds=time.monotonic()-started,stopped_reason=str(exc),**budget_unchecked())
    write_json(OUT/'production.json',{'status':status,'targets':all_targets,'checks':checks,'resources':resources,
        'duplicates':{k:v for k,v in duplicates.items() if len(v)>1},
        'branch_point_count':len(branches),'branch_points_with_both_results':sum(len(v)==2 for v in branches.values()),
        'source_access':{'paths':access,'classification_targets_opened':0,'test_trajectories_opened':0}})
    print('PRODUCTION',Counter(r['status'] for r in status),flush=True)

def budget_unchecked():
    return {'process_memory':process_memory(),'free_bytes':shutil.disk_usage(OUT).free,
            'output_bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--action',choices=['freeze','produce'],required=True)
    args=parser.parse_args();freeze() if args.action=='freeze' else produce()

if __name__=='__main__':main()
