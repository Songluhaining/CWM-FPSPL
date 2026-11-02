"""Read-only corpus/shape and train-partition-only readiness checks. No fit."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import shutil
from statistics import median
import sys
from cwm_fp.data.common import ROOT,KEY,key,read_csv,sha256,write_json
from .batch import read_plan,product_dir
from .suite import load,lines,validate_saved
from .runner import tree_usage

FOLDS=('grouped','loso_BankAccountTP','loso_Email')
def split_path(fold):return ROOT/'data/processed/splits'/('splits.csv' if fold=='grouped' else fold+'.csv')

def sources(c,out):
    result=[]
    for i,p in enumerate(c['products']):
        work=product_dir(out,i)
        result.append({'index':i,'identity':p['identity'],'observed_result':p['observed_result'],
            'partitions':p['partitions'],'registered_tests':len(p['tests']),'work':work,'graphs':work/'g/graphs.jsonl',
            'reused':False})
    products={key(p):p for p in read_csv(ROOT/'data/processed/manifests/products.csv',KEY)}
    splits={fold:{key(p):p['split'] for p in read_csv(split_path(fold),(*KEY,'split'))} for fold in FOLDS}
    frozen=load(out/'frozen_inputs_sha256.json')
    for name in c['reuse_collections']:
        work=Path(name)
        for file in ('graphs.jsonl','events.jsonl','source_map.jsonl','collection.json','runtime_binding.json'):
            path=work/file
            if frozen.get(path.relative_to(ROOT).as_posix())!=sha256(path):raise ValueError('Frozen reuse changed')
        coll=load(work/'collection.json');identities={key(r) for r in coll['records']}
        if len(identities)!=1:raise ValueError('Expected one validated reused product')
        identity=next(iter(identities))
        for r in coll['records']:validate_saved(r)
        for n,h in load(work/'runtime_binding.json')['artifacts_sha256'].items():
            if sha256(Path(n))!=h:raise ValueError('Reused runtime binding changed')
        result.append({'index':'reuse03B1','identity':list(identity),'observed_result':products[identity]['observed_result'],
            'partitions':{f:s[identity] for f,s in splits.items()},'registered_tests':int(products[identity]['test_count']),
            'work':work,'graphs':work/'graphs.jsonl','reused':True})
    if len({tuple(x['identity']) for x in result})!=len(result):raise ValueError('Duplicate selected/reused full product key')
    return result

def admission(c,out):
    from cwm_fp.models.graph_data import load_corpus,tensor_products
    from cwm_fp.models.graph_encoder import Limits,collate
    entries=sources(c,out);audits=[];eligible_paths=[]
    for entry in entries:
        result={k:v for k,v in entry.items() if k not in ('graphs','work')};result['folds']={}
        gpath=entry['graphs'];qpath=entry['work']/'g/quality_report.json'
        if not entry['reused'] and not qpath.exists():
            result['blocked']='unsealed_or_missing_extraction';audits.append(result);continue
        if not gpath.exists():result['blocked']='missing_graphs';audits.append(result);continue
        result['graph_sha256']=sha256(gpath);result['graph_bytes']=gpath.stat().st_size
        for fold in FOLDS:
            try:
                products,splits,graphs,eligible,audit=load_corpus(ROOT/'data/processed/manifests/products.csv',ROOT/'data/processed/manifests/tests.csv',gpath,
                    split_path(fold),holdout=fold[5:] if fold.startswith('loso_') else None)
                product_record=next((r for r in audit['products'] if tuple(r['identity'])==tuple(entry['identity'])),None)
                record={'eligible':tuple(entry['identity']) in eligible,'product':product_record,
                        'identity_and_semantics_qualified_tests':audit['identity_and_semantics_qualified']}
                result['folds'][fold]=record
                if record['eligible']:
                    if fold=='grouped':eligible_paths.append(gpath)
                    default=tensor_products(products,splits,eligible,'both',Limits());b=collate(default)
                    record['default_encoding']={'shape':list(b.x.shape),'edge_shape':list(b.edge_index.shape),**default[0].audit}
                    assert b.x.shape[1]==22
                    assert not (b.node_test[b.edge_index[0]]!=b.node_test[b.edge_index[1]]).any()
                    gs=eligible[tuple(entry['identity'])]
                    limits=Limits(max_nodes=max(len(g['nodes']) for g in gs),max_events=max(g['quality']['events'] for g in gs),
                                  max_tests=len(gs),max_edges=max(len(g['edges']) for g in gs))
                    full=tensor_products(products,splits,eligible,'both',limits);fb=collate(full)
                    record['lossless_interface_check']={'limits':vars(limits),'shape':list(fb.x.shape),'edge_shape':list(fb.edge_index.shape),**full[0].audit}
                    assert all(full[0].audit[n]==0 for n in ('lost_test_fraction','lost_event_fraction','lost_edge_fraction'))
                    assert fb.x.shape[1]==22 and not (fb.node_test[fb.edge_index[0]]!=fb.node_test[fb.edge_index[1]]).any()
                result['folds'][fold]=record
            except Exception as e:
                # Encoding capacity is distinct from whole-suite semantic admission.
                if fold in result['folds']:
                    result['folds'][fold]['encoding_error']=type(e).__name__+': '+str(e)
                else:result['folds'][fold]={'eligible':False,'error':type(e).__name__+': '+str(e)}
        audits.append(result);print('ADMISSION',entry['identity'],{f:r['eligible'] for f,r in result['folds'].items()},flush=True)
    aggregates={}
    for fold in FOLDS:
        eligible=[r for r in audits if r.get('folds',{}).get(fold,{}).get('eligible')]
        aggregates[fold]={'eligible_by_partition':{p:sum(r['partitions'][fold]==p for r in eligible) for p in ('train','val','test')},
            'versions_by_partition':{p:len({tuple(r['identity'][:2]) for r in eligible if r['partitions'][fold]==p}) for p in ('train','val','test')},
            'eligible_identities':[r['identity'] for r in eligible], 'labels_read':False}
    # The existing training entry consumes a single <=128 MiB file. Preserve
    # all eligible graphs and actually check that entry when its bound permits.
    total=sum(p.stat().st_size for p in eligible_paths);combined={'bytes':total,'within_existing_128MiB_gate':total<=134217728,'products':len(eligible_paths)}
    if eligible_paths and total<=134217728:
        target=out/'admitted_graphs.jsonl'
        if target.exists():raise ValueError('Combined dataset already exists; preserve old artifact')
        with target.open('xb') as dest:
            for source in eligible_paths:
                with source.open('rb') as f:shutil.copyfileobj(f,dest)
        combined['sha256']=sha256(target)
        for fold in FOLDS:
            *_,a=load_corpus(ROOT/'data/processed/manifests/products.csv',ROOT/'data/processed/manifests/tests.csv',target,split_path(fold),
                            holdout=fold[5:] if fold.startswith('loso_') else None)
            aggregates[fold]['combined_official_audit']=a
    write_json(out/'admission.json',{'products':audits,'folds':aggregates,'combined_input':combined,'model_training_runs':0})

def labels(c,out,fold):
    """Run in its own process, after immutable selection and graph admission."""
    from cwm_fp.data.labels import load_targets
    from cwm_fp.baselines.boundaries import install_io_guard
    allowed=ROOT/'data/processed/labels/full'/fold/'train.csv'
    install_io_guard('readiness_train_only',[allowed])
    audit=load(out/'admission.json');keys={tuple(k) for k in audit['folds'][fold]['eligible_identities']}
    splits={key(p):p['split'] for p in read_csv(split_path(fold),(*KEY,'split'))}
    train={k for k in keys if splits[k]=='train'};targets={};opened=[]
    if train:
        targets=load_targets(allowed,ROOT/'data/processed/manifests/products.csv',split_path(fold),partition='train',purpose='train')
        opened=[str(allowed)]
    counts=Counter(targets.get(k,'MISSING') for k in train)
    write_json(out/'readiness'/f'{fold}.json',{'fold':fold,'train_eligible_products':len(train),'train_target_counts':dict(counts),
        'train_has_both_FP_TP':bool(counts['FP'] and counts['TP']),'label_files_opened':opened,'validation_test_label_files_opened':[],
        'selection_sha256':sha256(out/'plan.json'),'admission_sha256':sha256(out/'admission.json'),
        'label_qualification':'existing_proxy_targets_not_independently_verified_truth','model_training_runs':0})
    print('TRAIN TARGET AVAILABILITY',fold,dict(counts),flush=True)

def distribution(values):
    xs=sorted(values)
    if not xs:return {'count':0,'sum':0,'min':None,'median':None,'p95':None,'max':None}
    return {'count':len(xs),'sum':sum(xs),'min':xs[0],'median':median(xs),
            'p95':xs[min(len(xs)-1,int(.95*(len(xs)-1)))],'max':xs[-1]}

def graph_reasons(q):
    reasons=set()
    if not q.get('program_identity_complete'):reasons.add('unresolved_program_identity')
    if not q.get('control_path_verified',q.get('normal_path_verified')):reasons.add('unverified_control_path')
    for gap in q.get('unsupported_dynamic',[]):
        reason=gap.get('reason','unknown')
        if 'array' in reason:reasons.add('array_semantics')
        elif 'exception' in reason:reasons.add('exception_semantics')
        elif 'call' in reason or 'native' in reason:reasons.add('call_or_library_semantics')
        elif 'category2' in reason:reasons.add('wide_value_semantics')
        elif 'thread' in reason:reasons.add('concurrency_semantics')
        else:reasons.add('other_dynamic_semantics')
    if q.get('bytecode_mismatches'):reasons.add('bytecode_mismatch')
    if q.get('executed_CFG_gaps'):reasons.add('executed_control_graph_gap')
    return sorted(reasons)

def statistics(c,out):
    from cwm_fp.models.graph_encoder import evidence_errors
    entries=sources(c,out);products=[];all_commands=[]
    for entry in entries:
        work=entry['work'];row={k:v for k,v in entry.items() if k not in ('work','graphs')}
        coll=load(work/'collection.json')['records'] if (work/'collection.json').exists() else []
        row.update(recorded_tests=len(coll),complete_tests=0,consistent_tests=0,actual_original_failed_tests=0,
                   parity_disagreements=0,historical_disagreements=0,execution_problem_tests=0,execution_reasons={},graph_tests=[])
        ec=Counter();execrows=[]
        for r in coll:
            s=r['status'];row['complete_tests']+=int(s['complete'])
            row['consistent_tests']+=int(s['complete'] and s['parity'] and s['matches_historical_outcome'])
            row['actual_original_failed_tests']+=int(s['runs']['original']['result']['outcome']=='FAIL')
            row['parity_disagreements']+=int(s['complete'] and not s['parity'])
            row['historical_disagreements']+=int(s['complete'] and not s['matches_historical_outcome'])
            reasons=set()
            er={'test_id':r['test_id'],'modes':{}}
            if s['complete'] and not s['parity']:
                er['parity_difference']={m:run['result'] for m,run in s['runs'].items()}
            for mode,run in s['runs'].items():
                if run['execution']['reason']:reasons.add(run['execution']['reason'])
                if run['result']['outcome']=='UNKNOWN':reasons.add(run['result'].get('reason','unknown_completion'))
                directory=Path(r['directory'])/mode;log=(directory/'run.log')
                if log.exists() and 'CWM_TRUNCATED' in log.read_text(encoding='utf-8',errors='replace'):reasons.add('event_limit_truncation')
                er['modes'][mode]={'seconds':run['execution']['seconds'],'run_log_bytes':log.stat().st_size if log.exists() else None,
                    'result':run['result']['outcome'],'resource':run['execution'].get('resource'),'directory_usage':tree_usage(directory)}
            row['execution_problem_tests']+=bool(reasons);ec.update(reasons);execrows.append(er)
        row['executions']=execrows;row['execution_reasons']=dict(ec)
        row['execution_coverage_complete']=len(coll)==entry['registered_tests'] and row['complete_tests']==entry['registered_tests']
        row['complete_execution_suite']=len(coll)==entry['registered_tests'] and row['consistent_tests']==entry['registered_tests']
        if not entry['reused'] and (work/'prepare.json').exists():row['preparation']=load(work/'prepare.json')
        graphfile=entry['graphs'];valid_graphs=[]
        if graphfile.exists():
            with graphfile.open(encoding='utf-8') as f:
                for line in f:
                    try:g=json.loads(line)
                    except json.JSONDecodeError:row['partial_graph_file']=True;break
                    q=g['quality'];vals=[n for n in g['nodes'] if n['type']=='value'];asserts=[n for n in g['nodes'] if n['type']=='assertion']
                    errors=evidence_errors(g)
                    qualified=not errors
                    record={'test_id':g['test_id'],'qualified_claim':bool(qualified),'quality':q,'categories':graph_reasons(q),
                        'independent_evidence_errors':errors,'edge_types':dict(Counter(e['type'] for e in g['edges'])),
                        'nodes':len(g['nodes']),'edges':len(g['edges']),'events':q['events'],'values':len(vals),
                        'observed_values':sum(n['attrs'].get('observed',False) for n in vals),'assertions':len(asserts),
                        'assertion_values_complete':sum(n['attrs'].get('expected_actual_values_complete',False) for n in asserts),
                        'output_observed':any(n['type']=='output' and n['attrs'].get('observed') for n in g['nodes']),
                        'structured_input_binding':False}
                    valid_graphs.append(record)
        row['graph_tests']=valid_graphs;row['graphs']=len(valid_graphs)
        row['graph_file_bytes']=graphfile.stat().st_size if graphfile.exists() else 0
        row['directory_usage']=tree_usage(work);products.append(row)
        if not entry['reused']:
            for f in work.rglob('*.command.json'):
                rec=load(f);all_commands.append({'path':str(f),'seconds':rec['seconds'],'reason':rec['reason'],
                    'resource':rec.get('resource'),'test_process':'Phase03TestRunner' in rec['argv']})
    systems={}
    for system in ('BankAccountTP','Email'):
        ps=[p for p in products if p['identity'][0]==system];new=[p for p in ps if not p['reused']]
        gs=[g for p in ps for g in p['graph_tests']];reasons=Counter(r for g in gs for r in g['categories'])
        systems[system]={'new_versions':len({p['identity'][1] for p in new}),'all_versions_including_reuse':len({p['identity'][1] for p in ps}),
            'new_PASS_products':sum(p['observed_result']=='PASS' for p in new),'new_FAIL_products':sum(p['observed_result']=='FAIL' for p in new),
            'new_registered_tests':sum(p['registered_tests'] for p in new),'new_complete_tests':sum(p['complete_tests'] for p in new),
            'all_registered_tests_executed_products':sum(p['execution_coverage_complete'] for p in ps),
            'complete_execution_products':sum(p['complete_execution_suite'] for p in ps),
            'graph_distributions':{name:distribution([g[name] for g in gs]) for name in ('nodes','edges','events','values')},
            'graph_reason_test_counts_nonexclusive':dict(reasons),'values':sum(g['values'] for g in gs),
            'observed_values':sum(g['observed_values'] for g in gs),'assertions':sum(g['assertions'] for g in gs),
            'complete_assertion_values':sum(g['assertion_values_complete'] for g in gs),
            'qualified_graphs':sum(g['qualified_claim'] for g in gs),
            'qualified_graph_distributions':{name:distribution([g[name] for g in gs if g['qualified_claim']]) for name in ('nodes','edges','events','values')},
            'edge_types_all_candidates':dict(sum((Counter(g['edge_types']) for g in gs),Counter())),
            'edge_types_qualified_graphs':dict(sum((Counter(g['edge_types']) for g in gs if g['qualified_claim']),Counter()))}
    write_json(out/'statistics.json',{'products':products,'systems':systems,'new_commands':all_commands,
        'out_usage_at_statistics':tree_usage(out),'disk_free_bytes':shutil.disk_usage(out).free,'collection_budget':load(out/'collection_budget.json')})
    print('STATISTICS', {s:(v['new_complete_tests'],v['complete_execution_products']) for s,v in systems.items()},flush=True)

def preservation(c,out):
    for listname,relative in [('frozen_inputs_sha256.json',True),('readonly_source_sha256.json',False)]:
        rows=load(out/listname)
        for n,h in rows.items():
            if sha256(ROOT/n if relative else Path(n))!=h:raise ValueError('Frozen file changed: '+n)
    for p in c['products']:
        if sha256(Path(p['source_variant_path'])/'roles.meta')!=p['roles_sha256']:
            raise ValueError('Original configuration roles changed')
        for n,h in p['source_sha256'].items():
            if sha256(Path(n))!=h:raise ValueError('Original source changed')
    checkpoint=load(out/'checkpoint_evidence.json')
    for n,h in checkpoint['status_sha256'].items():
        if sha256(Path(n))!=h:raise ValueError('Resumed checkpoint changed')
    write_json(out/'preservation.json',{'valid':True,'frozen_files':len(load(out/'frozen_inputs_sha256.json')),
        'readonly_source_files':len(load(out/'readonly_source_sha256.json')),'plan_sha256':sha256(out/'plan.json'),
        'original_selected_sources_unchanged':True,'original_configuration_roles_unchanged':True,'first_two_pairs_unchanged':True})
    print('Frozen artifacts, source inputs, split files and initial checkpoint unchanged',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['admission','labels','statistics','preserve'],required=True)
    p.add_argument('--fold',choices=FOLDS);a=p.parse_args();c,out=read_plan()
    if a.stage=='labels':labels(c,out,a.fold)
    else:
        if a.stage!='preserve':
            from cwm_fp.baselines.boundaries import install_io_guard
            install_io_guard('features')
        if a.stage=='admission':admission(c,out)
        elif a.stage=='statistics':statistics(c,out)
        else:preservation(c,out)

if __name__=='__main__':main()
