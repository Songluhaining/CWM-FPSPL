"""Evaluator-only CSV export after all configurations and predictions are frozen."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from cwm_fp.data.common import ROOT, KEY, indexed, key, read_csv, sha256, write_csv, write_json
from cwm_fp.data.labels import load_targets
from .boundaries import install_io_guard
from .evaluator import metric_rows, seed_summary
from .pipeline import configuration, context, split_file

METRICS=('fp_precision','fp_recall','fp_f1','accuracy','balanced_accuracy','average_precision','pr_auc','roc_auc')


def flat_metrics(rows):
    m=metric_rows(rows)
    flat={n:m.get(n) for n in ('n','FP','TP',*METRICS)}
    matrix=m.get('confusion_matrix',[[None,None],[None,None]])
    flat.update(dict(zip(('tn','fp','fn','tp'),matrix[0]+matrix[1])))
    flat['single_class']=len({r['y_true'] for r in rows})<2
    return flat


def export(c):
    install_io_guard('evaluator')
    # All main models must have completed their one-time held-out evaluation.
    result=json.loads((c['metrics_dir']/'summary.json').read_text(encoding='utf-8'))
    dest=ROOT/c.get('export_dir','outputs/phaseBASE4_results')
    if not dest.resolve().is_relative_to((ROOT/'outputs').resolve()):
        raise ValueError('Export must be under project outputs')
    dest.mkdir(exist_ok=False)
    products=indexed(read_csv(c['products'],KEY))
    by_id={p['sample_id']:p for p in products.values()}
    label_audit=json.loads((c['labels']/'label_audit.json').read_text(encoding='utf-8'))
    versions={(r['system_id'],r['bug_version_id']):r for r in label_audit['versions']}
    run_rows=[];predictions=[];version_scores=[];strata=[];split_status=[];intervals=[]
    prediction_fields=None
    for fold in c['folds']:
        _,split,_=context(c,fold)
        for partition in ('train','val','test'):
            targets=load_targets(c['labels']/fold/(partition+'.csv'),c['products'],split_file(c,fold),
                partition=partition,purpose='evaluate')
            for system in sorted({k[0] for k in targets}):
                keys=[k for k in targets if k[0]==system]
                counts=Counter(targets[k] for k in keys)
                split_status.append({'fold':fold,'partition':partition,'system_id':system,
                    'versions':len({k[:2] for k in keys}),'products':len(keys),'F':counts['F'],
                    'FP':counts['FP'],'TP':counts['TP'],'all_targets_known':all(targets[k] in {'F','FP','TP'} for k in keys)})
        for seed in c['seeds']:
            for model in c['models']:
                path=c['metrics_dir']/fold/str(seed)/(model+'.csv')
                detail=json.loads(path.with_name(model+'_metrics.json').read_text(encoding='utf-8'))
                ci=detail['partitions']['test']['cluster_bootstrap']
                ci_row={'fold':fold,'model':model,'seed':seed,'clusters':ci.get('clusters'),
                    'replicates':ci.get('requested_replicates'),'valid_two_class_replicates':ci.get('valid_two_class_replicates')}
                for name in ('fp_f1','average_precision','roc_auc','balanced_accuracy'):
                    bounds=ci.get('intervals',{}).get(name) or [None,None]
                    ci_row[name+'_lower95'],ci_row[name+'_upper95']=bounds
                intervals.append(ci_row)
                rows=read_csv(path)
                rows=[r | {'y_true':int(r['y_true']),'y_pred':int(r['y_pred']),
                    'y_score':float(r['y_score']),'seed':int(r['seed'])} for r in rows]
                for r in rows:
                    r.update(protocol=c['protocol'], evidence_file=path.relative_to(ROOT).as_posix())
                predictions.extend(rows)
                for partition in ('train','val','test'):
                    local=[r for r in rows if r['partition']==partition]
                    for system in ['ALL']+sorted({r['system_id'] for r in local}):
                        subset=local if system=='ALL' else [r for r in local if r['system_id']==system]
                        run_rows.append({'protocol':c['protocol'],'fold':fold,'partition':partition,
                            'system_id':system,'model':model,'seed':seed,
                            'versions':len({(r['system_id'],r['bug_version_id']) for r in subset}),**flat_metrics(subset)})
                    for g in sorted({(r['system_id'],r['bug_version_id']) for r in local}):
                        subset=[r for r in local if (r['system_id'],r['bug_version_id'])==g]
                        version_scores.append({'protocol':c['protocol'],'fold':fold,'partition':partition,
                            'system_id':g[0],'bug_version_id':g[1],'base_feature_fault':versions[g]['base_feature_fault'],
                            'model':model,'seed':seed,**flat_metrics(subset)})
                    for base in (True,False):
                        subset=[r for r in local if versions[(r['system_id'],r['bug_version_id'])]['base_feature_fault']==base]
                        strata.append({'fold':fold,'partition':partition,'model':model,'seed':seed,
                            'base_feature_fault':base,**flat_metrics(subset)})

    pool=c['work_dir']/'source_pool'
    if pool.exists():
        pool_receipt=json.loads((pool/'receipt.json').read_text(encoding='utf-8'))
        targets={}
        for partition in ('train','val','test'):
            targets.update(load_targets(c['labels']/'grouped'/(partition+'.csv'),c['products'],split_file(c,'grouped'),partition=partition,purpose='evaluate'))
        for receipt in pool_receipt['receipts']:
            seed=receipt['seed'];path=pool/f'{seed}_predictions.csv'
            if sha256(path)!=receipt['predictions_sha256']: raise ValueError('Source pool prediction changed')
            rows=[]
            for p in read_csv(path):
                source=by_id[p['sample_id']];k=key(source)
                if source['observed_result']!='PASS': raise ValueError('F in binary evaluation')
                rows.append(p | dict(zip(KEY,k)) | {'seed':seed,'y_true':int(targets[k]=='FP'),
                    'y_score':float(p['y_score']),'y_pred':int(p['y_pred']),'origin':'natural' if targets[k]=='FP' else 'unknown',
                    'protocol':'source_pool_transductive_v1','split_id':'not_a_holdout_experiment',
                    'evidence_file':path.relative_to(ROOT).as_posix()})
            expected={p['sample_id'] for p in products.values() if p['observed_result']=='PASS'}
            if len(rows)!=len(expected) or {r['sample_id'] for r in rows}!=expected: raise ValueError('Incomplete source pool prediction')
            predictions.extend(rows)
            for system in ['ALL']+sorted({r['system_id'] for r in rows}):
                subset=rows if system=='ALL' else [r for r in rows if r['system_id']==system]
                run_rows.append({'protocol':'source_pool_transductive_v1','fold':'all_unlabeled_pool',
                    'partition':'all_unlabeled_pool','system_id':system,'model':'pulp_source_pool','seed':seed,
                    'versions':len({(r['system_id'],r['bug_version_id']) for r in subset}),**flat_metrics(subset)})
    means=[]
    groups=defaultdict(list)
    for row in run_rows:
        groups[tuple(row[n] for n in ('protocol','fold','partition','system_id','model'))].append(row)
    for group,items in sorted(groups.items()):
        row=dict(zip(('protocol','fold','partition','system_id','model'),group))
        row.update(n=items[0]['n'],FP=items[0]['FP'],TP=items[0]['TP'],versions=items[0]['versions'],seeds=len(items))
        for metric in METRICS:
            s=seed_summary([r[metric] for r in items]);row[metric+'_mean']=s['mean'];row[metric+'_std']=s['std']
        means.append(row)
    version_status=[]
    for group,a in sorted(versions.items()):
        local=[p for k,p in products.items() if k[:2]==group]
        version_status.append({'system_id':group[0],'bug_version_id':group[1],'products':len(local),
            'tests':sum(int(p['test_count']) for p in local),'F':a['label_counts'].get('F',0),
            'FP':a['label_counts'].get('FP',0),'TP':a['label_counts'].get('TP',0),
            'base_feature_fault':a['base_feature_fault'],'excluded':False,'status':'features_and_predictions_complete',
            'evidence_file':str(c['work_dir'].relative_to(ROOT)/'observations_audit.json')})
    tables={'metrics_by_seed.csv':run_rows,'metrics_mean_std.csv':means,
        'heldout_results.csv':[r for r in means if r['partition']=='test'],
        'source_pulp_pool.csv':[r for r in means if r['partition']=='all_unlabeled_pool'],
        'version_metrics.csv':version_scores,'base_feature_metrics.csv':strata,
        'version_status.csv':version_status,'split_status.csv':split_status,'product_predictions.csv':predictions,
        'cluster_confidence_intervals.csv':intervals}
    method_names={'clap6_svm':'CLAP六维支持向量（统一划分适配）','pulp50_pu_source':'PULP源码正例与无标签学习（训练区拟合）',
        'pulp50_lr':'PULP五十维逻辑回归对照','pulp50_svm':'PULP五十维支持向量对照','coverage2_lr':'测试数与覆盖率逻辑回归对照',
        'pulp_source_pool':'PULP源码全量无标签池'}
    def chinese(row):
        return {'评价方式':'按版本分组' if row['fold']=='grouped' else '全量无标签池（非泛化）' if row['fold']=='all_unlabeled_pool' else '整系统留出',
            '测试系统':row['system_id'],'方法':method_names[row['model']],'产品数':row['n'],'独立缺陷版本数':row['versions'],
            '假通过产品数':row['FP'],'真通过产品数':row['TP'],'随机种子数':row['seeds'],
            '假通过精确率均值':row['fp_precision_mean'],'假通过召回率均值':row['fp_recall_mean'],
            '假通过F1均值':row['fp_f1_mean'],'假通过F1标准差':row['fp_f1_std'],
            '平均精确率AP均值':row['average_precision_mean'],'ROC面积均值':row['roc_auc_mean'],
            '平衡准确率均值':row['balanced_accuracy_mean'],'准确率均值':row['accuracy_mean']}
    tables['四系统基线结果.csv']=[chinese(r) for r in tables['heldout_results.csv']
        if r['fold']=='grouped' or r['system_id']!='ALL']
    tables['PULP全量无标签池结果.csv']=[chinese(r) for r in tables['source_pulp_pool.csv']]
    for name,rows in tables.items():
        write_csv(dest/name,rows,rows[0].keys())
    product_status=[{**dict(zip(KEY,k)),'sample_id':p['sample_id'],'original_result':p['observed_result'],
        'tests':int(p['test_count']),'features_complete':True,'binary_target_eligible':p['observed_result']=='PASS',
        'exclusion_reason':'observed_failure_not_FP_TP_target' if p['observed_result']=='FAIL' else '',
        'evidence_file':'outputs/phaseBASE4_results/product_predictions.csv'} for k,p in sorted(products.items())]
    prefix=c.get('report_prefix','phaseBASE4')
    write_csv(ROOT/'outputs'/f'{prefix}_product_status.csv',product_status,product_status[0].keys())
    write_csv(ROOT/'outputs'/f'{prefix}_graph_status.csv',[],(*KEY,'test_id','status','reason','evidence_file'))
    write_json(dest/'export_receipt.json',{'files':{n:sha256(dest/n) for n in tables},'rows':{n:len(r) for n,r in tables.items()},
        'all_versions':len(versions),'all_products':len(products),'excluded_versions':0,
        'historical_label_matches':label_audit['matched_existing_original_view_products'],
        'graph_status':'not_applicable: no LDDG extraction/training in this baseline run'})
    write_json(dest/'workbook_data.json',{'heldout':tables['heldout_results.csv'],'source_pool':tables['source_pulp_pool.csv'],
        'seeds':run_rows,'versions':version_status,'splits':split_status})
    print(json.dumps({'output':str(dest),'rows':{n:len(r) for n,r in tables.items()}}))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();export(configuration(args.config))


if __name__=='__main__': main()
