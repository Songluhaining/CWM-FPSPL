"""A separately authorized holdout evaluation of frozen 04R1 models.

No graph fitting, checkpoint reselection, changed cohort, or new architecture.
Tabular baselines reuse BASE4 feature bytes and the same 17 supervised products.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from cwm_fp.data.common import ROOT, KEY, key, read_csv, sha256, write_csv, write_json
from cwm_fp.data.labels import load_targets
from cwm_fp.baselines.boundaries import install_io_guard


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def checked_rows(rows, cohort):
    expected = {r['sample_id']: r for r in cohort}
    if len(rows) != len(expected) or {r['sample_id'] for r in rows} != set(expected):
        raise ValueError('Prediction cohort incomplete or duplicated')
    for r in rows:
        source = expected[r['sample_id']]
        if tuple(r['identity']) != tuple(source['identity']) or r['partition'] != source['partitions']['grouped']:
            raise ValueError('Prediction identity or partition changed')
    return rows


def verify_protocol(path):
    p = read(path)
    if p['stage_id'] != 'EVAL1' or not p['test_evaluation_authorized']:
        raise ValueError('Explicit holdout evaluation protocol required')
    if sha256(path) != read(Path(path).with_suffix('.receipt.json'))['sha256']:
        raise ValueError('Protocol changed')
    for name, digest in p['frozen_files'].items():
        if sha256(Path(name)) != digest:
            raise ValueError('Frozen dependency changed: ' + name)
    return p


def prepare(out, protocol_file):
    import numpy as np
    install_io_guard('features')
    old_path = ROOT/'outputs/phase04R1_protocol.json'
    old = read(old_path)
    old_run = Path(old['paths']['out'])
    selections_path = old_run/'validation_results.json'
    selection = read(selections_path)
    if selection['protocol_sha256'] != sha256(old_path) or selection['test_evaluated']:
        raise ValueError('Historical selection provenance mismatch')
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT/'data/processed/baselines/four_systems_v1'
    products = read_csv(source/'manifests/products.csv', KEY)
    old_products = {key(r): r for r in read_csv(Path(old['paths']['products']), KEY)}
    old_splits = {key(r): r for r in read_csv(Path(old['paths']['splits']), (*KEY, 'split'))}
    by_key = {key(r): r for r in products}
    cohort = old['cohort']
    wanted = {tuple(r['identity']): r for r in cohort}
    if Counter(r['partitions']['grouped'] for r in cohort) != {'train':17, 'val':5, 'test':6}:
        raise ValueError('Wrong original cohort')
    for k, row in wanted.items():
        for name in ('sample_id','suite_id'):
            if row[name] != by_key[k][name] or by_key[k][name] != old_products[k][name]:
                raise ValueError('Old/new suite or product identity mismatch')
        if old_splits[k]['split'] != row['partitions']['grouped']:
            raise ValueError('Historical split changed')
    train_versions = {k[:2] for k,r in wanted.items() if r['partitions']['grouped']=='train'}
    anchors = [r for r in products if key(r)[:2] in train_versions and r['observed_result']=='FAIL']
    selected = sorted([by_key[k] for k in wanted]+anchors,key=key)
    split = [{**dict(zip(KEY,key(r))),'split':wanted[key(r)]['partitions']['grouped'] if key(r) in wanted else 'train'} for r in selected]
    write_csv(out/'products.csv',selected,selected[0].keys())
    write_csv(out/'splits/splits.csv',split,(*KEY,'split'))
    source_audit=read(source/'grouped/features_audit.json')
    feature_path=source/'grouped/features.npz'
    if sha256(feature_path)!=source_audit['features_sha256']:
        raise ValueError('Source features changed')
    index={sid:i for i,sid in enumerate(source_audit['sample_ids'])}
    ids=[index[r['sample_id']] for r in selected]
    with np.load(feature_path,allow_pickle=False) as arrays:
        (out/'grouped').mkdir()
        np.savez_compressed(out/'grouped/features.npz',**{n:arrays[n][ids] for n in arrays.files})
    write_json(out/'grouped/features_audit.json',{'products_sha256':sha256(out/'products.csv'),
        'split_sha256':sha256(out/'splits/splits.csv'),'features_sha256':sha256(out/'grouped/features.npz'),
        'sample_ids':[r['sample_id'] for r in selected],'source_feature_sha256':sha256(feature_path),
        'scope':'exact BASE4 version-context features, frozen 04R1 PASS cohort and TRAIN-version F anchors'})
    labels=ROOT/'data/processed/labels/eval1_matched_v1'
    if labels.exists(): raise ValueError('New label namespace required')
    config={'schema_version':1,'run_name':'phaseEVAL1','protocol':'version_context_v1','suite_view':'original',
        'products':str(out/'products.csv'),'tests':old['paths']['tests'],'split_dir':str(out/'splits'),
        'work_dir':str(out),'metrics_dir':str(ROOT/'outputs/metrics/eval1_matched_v1'),
        'labels':str(labels),'sources':str(ROOT/'configs/baseline_four_sources.local.json'),
        'folds':['grouped'],'seeds':old['seeds'],'c_grid':[.1,1.,10.],
        'models':['clap6_svm','pulp50_lr','pulp50_svm','coverage2_lr'],
        'bootstrap_replicates':200,'pu_estimators':30,'pu_sampling_ratio':.05,
        'paper_original_protocol':'matched 04R1 cohort; BASE4 version context; source-rule adapted baselines',
        'bounds':{'max_seconds_per_command':600,'minimum_free_bytes':1073741824}}
    write_json(out/'baseline_config.json',config)
    frozen=[old_path,selections_path,old_run/'prediction_index.json',old_run/'stats_lr_parameters.npz',
        old_run/'stats_lr_audit.json',old_run/'prior.json',source/'manifests/products.csv',feature_path,
        source/'grouped/features_audit.json',Path(old['paths']['package'])/'manifest.json',out/'baseline_config.json',
        out/'products.csv',out/'splits/splits.csv',out/'grouped/features.npz']
    frozen += [Path(old['paths'][name]) for name in ('products','tests','splits')]
    cp=[]
    for r in selection['selected']:
        path=old_run/r['checkpoint']['path']
        if sha256(path)!=r['checkpoint']['sha256'] or sha256(old_run/r['val_file'])!=r['val_sha256']:
            raise ValueError('Selected checkpoint or validation output changed')
        frozen.extend([path,path.with_suffix('.receipt.json'),old_run/r['val_file']])
        cp.append({k:r[k] for k in ('method','seed','epoch','checkpoint','val_file','val_sha256')})
    frozen += [ROOT/'src/cwm_fp/models'/n for n in ('matched_evaluation.py','graph_classifier.py','graph_encoder.py','graph_package.py')]
    p={'stage_id':'EVAL1','protocol':'matched_04R1_holdout_v1','test_evaluation_authorized':True,
        'authorization':'User requested actual evaluation of current runnable model against baselines',
        'out':str(out),'old_protocol':str(old_path),'old_run':str(old_run),'cohort':cohort,
        'selected_graph_checkpoints':cp,'baseline_config':str(out/'baseline_config.json'),
        'source_features':str(feature_path),'source_audit':str(source/'grouped/features_audit.json'),
        'source_products':str(source/'manifests/products.csv'),'graph_atol':1e-6,
        'seeds':old['seeds'],'supervised_training_products':17,'F_anchors':len(anchors),
        'source_pu_unlabeled_training_products':sum(key(r)[:2] in train_versions and r['observed_result']=='PASS' for r in products),
        'source_pu_scope':'All original F and unlabeled PASS of the same 9 TRAIN versions; no held-out fit; extra unlabeled pool disclosed',
        'source_pu_reason':'Source 5% sampling rounds 17 U products to zero, so exact17 PU is unavailable; declared extra-unlabeled comparison',
        'graph_optimizer_updates':0,'new_java_tests':0,'model_seconds_max':600,'new_bytes_max':268435456,
        'selection':'reuse frozen 04R1 graph selections; BASE4 validation F1 rule for tabular candidates; no test tuning',
        'historical_exposure':'Bank/Email prior development; BASE4 previously evaluated overlapping products under different models/splits',
        'frozen_files':{str(f):sha256(f) for f in frozen}}
    write_json(protocol_file,p)
    write_json(protocol_file.with_suffix('.receipt.json'),{'sha256':sha256(protocol_file),'frozen_at':time.time(),'test_labels_read':False})
    print(json.dumps({k:p[k] for k in ('supervised_training_products','F_anchors','source_pu_unlabeled_training_products','graph_optimizer_updates')}))


def export_labels(path,partition):
    p=verify_protocol(path);old=read(p['old_protocol']);out=Path(p['out']);config=read(p['baseline_config'])
    source=Path(old['paths']['train_labels']).with_name(partition+'.csv')
    dest=Path(config['labels'])/'grouped'/(partition+'.csv')
    # This worker may verify only its own newly exported partition, never a
    # different held-out file. The source partition is independently validated.
    install_io_guard('subset_labels',[source,dest])
    if partition=='test':
        verify_predictions(p)
    selected={key(r):r for r in read_csv(out/'products.csv',KEY)}
    split={key(r):r['split'] for r in read_csv(out/'splits/splits.csv',(*KEY,'split'))}
    targets=load_targets(source,Path(old['paths']['products']),Path(old['paths']['splits']),
        partition=partition,purpose='train' if partition=='train' else 'evaluate')
    rows=[dict(zip(KEY,k))|{'label':targets[k],'suite_id':r['suite_id'],'suite_view':'original',
        'origin':'natural' if targets[k]=='FP' else 'unknown'} for k,r in selected.items() if split[k]==partition]
    if dest.exists():
        if read_csv(dest)!=rows: raise ValueError('Existing exported targets differ')
    else:
        write_csv(dest,rows,(*KEY,'label','suite_id','suite_view','origin'))
    write_json(out/(partition+'_target_receipt.json'),{'partition':partition,'source_sha256':sha256(source),
        'export_sha256':sha256(dest),'retained_products':len(rows),'purpose':'fit' if partition=='train' else 'evaluator_only'})
    print(json.dumps({'partition':partition,'products':len(rows)}))


def graph_predict(path):
    import numpy as np
    import torch
    from .graph_package import ProductPackage
    from .graph_classifier import GraphClassifier
    from .train_graph import predict
    from .package_training import load_checkpoint, optimizer_for, process_memory
    from .package_comparison import product_statistics
    p=verify_protocol(path);old=read(p['old_protocol']);out=Path(p['out']);old_run=Path(p['old_run'])
    install_io_guard('inference');torch.set_num_threads(1);start=time.monotonic()
    package=ProductPackage(Path(old['paths']['package']),expected_manifest_sha256=old['manifest_sha256'])
    if {tuple(r['identity']) for r in package.products}!={tuple(r['identity']) for r in p['cohort']}:
        raise ValueError('Package cohort changed')
    items=[package.materialize(r,'grouped') for r in package.products]
    checks=[];all_rows=[];graph_status=[]
    for item in items:
        for t in item.tests:
            graph_status.append(dict(zip(KEY,item.identity))|{'test_id':t.metadata['identity'][3],
                'partition':item.partition,'nodes':len(t.x),'edges':len(t.edge_type),'events':len(t.metadata['events']),
                'status':'verified_lossless_package','reason':'','evidence_file':str(Path(old['paths']['package'])/'manifest.json')})
    write_csv(ROOT/'outputs/phaseEVAL1_graph_status.csv',graph_status,graph_status[0].keys())
    for r in p['selected_graph_checkpoints']:
        model=GraphClassifier(r['method'],hidden=16,layers=2)
        optimizer=optimizer_for(model)
        cp=old_run/r['checkpoint']['path'];receipt=read(cp.with_suffix('.receipt.json'))
        expected={'kind':'04R1_fixed_train_validation','protocol_sha256':sha256(Path(p['old_protocol'])),
            'method':r['method'],'seed':r['seed'],'epoch':r['epoch'],'model':old['models'][r['method']]}
        if receipt['contract']!=expected: raise ValueError('Historical checkpoint contract changed')
        load_checkpoint(cp,r['checkpoint']['sha256'],model,optimizer,expected)
        before={k:v.clone() for k,v in model.state_dict().items()}
        rows=checked_rows(predict(model,items,batch_products=2),p['cohort'])
        prior={x['sample_id']:x for x in read(old_run/r['val_file'])}
        delta=max(abs(x['y_score']-prior[x['sample_id']]['y_score']) for x in rows if x['partition']=='val')
        if delta>p['graph_atol']: raise ValueError('Restored model differs from frozen validation predictions')
        if any(not torch.equal(before[k],v) for k,v in model.state_dict().items()): raise ValueError('Inference changed parameters')
        all_rows += [x|{'model':r['method'],'seed':r['seed'],'checkpoint_epoch':r['epoch']} for x in rows]
        checks.append({'model':r['method'],'seed':r['seed'],'epoch':r['epoch'],'validation_max_abs_difference':delta,'optimizer_updates':0})
        if time.monotonic()-start>p['model_seconds_max']: raise TimeoutError('Inference budget')
    with np.load(old_run/'stats_lr_parameters.npz',allow_pickle=False) as a:
        if sha256(old_run/'stats_lr_parameters.npz')!=read(old_run/'stats_lr_audit.json')['parameters_sha256']:
            raise ValueError('Statistics model changed')
        z=(np.asarray(product_statistics(items))-a['mean'])/a['scale']
        logits=(z@a['coef'].T+a['intercept']).ravel();scores=1/(1+np.exp(-logits))
    for name,values in [('Stats-LR',scores),('Prior',[read(old_run/'prior.json')['p_train']]*len(items))]:
        prior={x['sample_id']:x for x in read(old_run/(name+'_fixed_0_val.json'))}
        rows=[{'sample_id':x.sample_id,'identity':list(x.identity),'partition':x.partition,'y_score':float(s),
            'y_pred':int(s>=.5),'score_kind':'probability_uncalibrated','model':name,'seed':None,'checkpoint_epoch':0} for x,s in zip(items,values)]
        if max(abs(x['y_score']-prior[x['sample_id']]['y_score']) for x in rows if x['partition']=='val')>p['graph_atol']:
            raise ValueError('Statistics/prior restore mismatch')
        all_rows+=checked_rows(rows,p['cohort'])
    dest=out/'graph_predictions.json'
    if dest.exists(): raise ValueError('Predictions already sealed')
    write_json(dest,all_rows)
    write_json(out/'graph_prediction_receipt.json',{'sha256':sha256(dest),'models':11,'products_per_model':28,
        'checks':checks,'seconds':time.monotonic()-start,'process_memory':process_memory(),'device':'cpu','optimizer_updates':0})
    print(json.dumps({'models':11,'predictions':len(all_rows),'seconds':time.monotonic()-start}))


def pu_predict(path):
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from cwm_fp.baselines.vendor import FrozenSourcePU
    p=verify_protocol(path);out=Path(p['out']);install_io_guard('features');start=time.monotonic()
    products=read_csv(Path(p['source_products']),KEY);by={r['sample_id']:r for r in products}
    audit=read(p['source_audit']);order=audit['sample_ids']
    train_versions={tuple(r['identity'][:2]) for r in p['cohort'] if r['partitions']['grouped']=='train'}
    training=[i for i,sid in enumerate(order) if key(by[sid])[:2] in train_versions]
    failed=[i for i in training if by[order[i]]['observed_result']=='FAIL']
    unlabeled=[i for i in training if by[order[i]]['observed_result']=='PASS']
    with np.load(p['source_features'],allow_pickle=False) as a: raw=a['pulp50']
    scaler=MinMaxScaler().fit(raw[training]);x=scaler.transform(raw)
    idx={sid:i for i,sid in enumerate(order)};query=[idx[r['sample_id']] for r in p['cohort']]
    all_rows=[]
    for seed in p['seeds']:
        model=FrozenSourcePU(seed,30,.05).fit(x[failed],x[unlabeled]);scores=model.score(x[query])
        rows=[{'sample_id':r['sample_id'],'identity':r['identity'],'partition':r['partitions']['grouped'],
            'model':'pulp_pu_extra_train_unlabeled','seed':seed,'y_score':float(s),'y_pred':int(s>.5),
            'score_kind':'probability_uncalibrated'} for r,s in zip(p['cohort'],scores)]
        all_rows+=checked_rows(rows,p['cohort'])
    dest=out/'pu_predictions.json'
    if dest.exists(): raise ValueError('PU already executed')
    write_json(dest,all_rows)
    write_json(out/'pu_prediction_receipt.json',{'sha256':sha256(dest),'seconds':time.monotonic()-start,
        'F_anchors':len(failed),'unlabeled_PASS':len(unlabeled),'fit_keys':[list(key(by[order[i]])) for i in training],
        'target_labels_read':False,'exact17_source_pu_status':'unavailable: floor(17*0.05)=0'})
    print(json.dumps({'models':3,'F':len(failed),'U':len(unlabeled),'seconds':time.monotonic()-start}))


def verify_predictions(p):
    out=Path(p['out']);c=read(p['baseline_config'])
    for stem in ('graph','pu'):
        if sha256(out/(stem+'_predictions.json'))!=read(out/(stem+'_prediction_receipt.json'))['sha256']:
            raise ValueError('Predictions changed before test evaluation')
    seal=read(out/'all_predictions_sealed.json')
    for name,digest in seal['files'].items():
        if sha256(Path(name))!=digest: raise ValueError('Prediction seal mismatch')
    expected={r['sample_id'] for r in p['cohort']}
    for seed in c['seeds']:
        for model in c['models']:
            rows=read_csv(out/'grouped'/str(seed)/(model+'_predictions.csv'))
            if len(rows)!=28 or {r['sample_id'] for r in rows}!=expected: raise ValueError('Baseline cohort mismatch')


def seal(path):
    p=verify_protocol(path);out=Path(p['out']);c=read(p['baseline_config'])
    files=[out/(s+'_predictions.json') for s in ('graph','pu')]
    files += [out/'grouped'/str(seed)/(model+'_predictions.csv') for seed in c['seeds'] for model in c['models']]
    dest=out/'all_predictions_sealed.json'
    if dest.exists(): raise ValueError('Predictions already frozen')
    write_json(dest,{'files':{str(f):sha256(f) for f in files},'time':time.time(),'test_targets_opened':False})
    verify_predictions(p)
    print('ALL_PREDICTIONS_FROZEN')


def evaluate(path):
    import numpy as np
    from cwm_fp.baselines.evaluator import metric_rows, cluster_interval, seed_summary
    from .graph_metrics import binary_metrics
    p=verify_protocol(path);verify_predictions(p);out=Path(p['out']);c=read(p['baseline_config'])
    install_io_guard('evaluator')
    dest=ROOT/'outputs/phaseEVAL1_results'
    dest.mkdir(exist_ok=False)
    cohort={r['sample_id']:r for r in p['cohort']};truth={}
    for part in ('train','val','test'):
        targets=load_targets(Path(c['labels'])/'grouped'/(part+'.csv'),Path(c['products']),
            Path(c['split_dir'])/'splits.csv',partition=part,purpose='evaluate')
        truth.update({k:v for k,v in targets.items() if k in {tuple(r['identity']) for r in p['cohort']}})
    rows=[]
    for stem in ('graph','pu'):
        for r in read(out/(stem+'_predictions.json')):
            rows.append(r|dict(zip(KEY,r['identity']))|{'y_true':int(truth[tuple(r['identity'])]=='FP'),
                'evidence_file':str(out/(stem+'_predictions.json'))})
    for seed in c['seeds']:
        for model in c['models']:
            source=out/'grouped'/str(seed)/(model+'_predictions.csv')
            for r in read_csv(source):
                item=cohort[r['sample_id']];k=tuple(item['identity'])
                rows.append(r|dict(zip(KEY,k))|{'identity':list(k),'seed':seed,'y_score':float(r['y_score']),
                    'y_pred':int(r['y_pred']),'y_true':int(truth[k]=='FP'),'evidence_file':str(source)})
    from collections import defaultdict
    groups=defaultdict(list)
    for r in rows:groups[r['model'],r['seed']].append(r)
    metrics=[];intervals=[]
    for (model,seed),items in groups.items():
        checked_rows(items,p['cohort'])
        for part in ('train','val','test'):
            local=[r for r in items if r['partition']==part]
            m=metric_rows(local)
            if local[0]['score_kind'].startswith('probability'):
                m['log_loss']=binary_metrics([r['y_true'] for r in local],[r['y_score'] for r in local])['log_loss']
            else:m['log_loss']=None
            row={'model':model,'seed':seed,'partition':part,'versions':len({tuple(r['identity'][:2]) for r in local}),
                **{k:m.get(k) for k in ('n','FP','TP','fp_precision','fp_recall','fp_f1','accuracy','balanced_accuracy','average_precision','pr_auc','roc_auc','brier','log_loss')}}
            row.update(dict(zip(('tn','fp','fn','tp'),sum(m['confusion_matrix'],[]))))
            metrics.append(row)
            if part=='test':intervals.append({'model':model,'seed':seed,**cluster_interval(local,200,seed or p['seeds'][0])})
    means=[]
    names=('fp_precision','fp_recall','fp_f1','accuracy','balanced_accuracy','average_precision','roc_auc','brier','log_loss')
    for model in sorted({r['model'] for r in metrics}):
        for part in ('train','val','test'):
            selected=[r for r in metrics if r['model']==model and r['partition']==part]
            row={'model':model,'partition':part,'seeds':len(selected),**{k:selected[0][k] for k in ('n','FP','TP','versions')}}
            for name in names:
                s=seed_summary([r[name] for r in selected]);row[name+'_mean']=s['mean'];row[name+'_std']=s['std']
            means.append(row)
    fields=(*KEY,'sample_id','model','seed','partition','score_kind','y_true','y_pred','y_score','evidence_file')
    write_csv(dest/'product_predictions.csv',[{k:r.get(k) for k in fields} for r in rows],fields)
    write_csv(dest/'metrics_by_seed.csv',metrics,metrics[0].keys())
    write_csv(dest/'metrics_mean_std.csv',means,means[0].keys())
    write_json(dest/'cluster_intervals.json',intervals)
    names_cn={'rgcn':'现有关系图分类器','untyped':'合并关系图对照','nodeonly':'无边节点对照',
        'Prior':'训练比例常数','Stats-LR':'图统计逻辑回归','clap6_svm':'CLAP六维支持向量适配',
        'pulp50_lr':'PULP五十维逻辑回归','pulp50_svm':'PULP五十维支持向量','coverage2_lr':'测试数与覆盖率对照',
        'pulp_pu_extra_train_unlabeled':'PULP源码（额外训练区无标签池）'}
    chinese=[{'方法':names_cn[r['model']],'分区':{'train':'训练','val':'验证','test':'测试'}[r['partition']],
        '产品数':r['n'],'独立版本数':r['versions'],'假通过数':r['FP'],'真通过数':r['TP'],'种子数':r['seeds'],
        '假通过精确率':r['fp_precision_mean'],'假通过召回率':r['fp_recall_mean'],'假通过F1均值':r['fp_f1_mean'],
        'F1标准差':r['fp_f1_std'],'平均精确率':r['average_precision_mean'],'ROC面积':r['roc_auc_mean'],
        '准确率':r['accuracy_mean'],'平衡准确率':r['balanced_accuracy_mean'],'概率对数损失':r['log_loss_mean'],
        '说明':'使用额外训练区无标签产品，单列解释' if 'extra_train' in r['model'] else '同17监督训练产品、5验证产品、6测试产品'} for r in means]
    write_csv(dest/'当前模型与基线对比.csv',chinese,chinese[0].keys())
    write_json(dest/'evaluation_receipt.json',{'prediction_groups':len(groups),'prediction_rows':len(rows),'metrics':means,
        'truth_counts':{part:dict(Counter(truth[tuple(r['identity'])] for r in p['cohort'] if r['partitions']['grouped']==part)) for part in ('train','val','test')},
        'files':{f.name:sha256(f) for f in dest.iterdir() if f.is_file()},'test_tuning':False,'new_graph_optimizer_updates':0})
    print(json.dumps({'models':len({r['model'] for r in metrics}),'prediction_groups':len(groups),'rows':len(rows),'output':str(dest)}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','labels','graph_predict','pu_predict','seal','evaluate'])
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--partition',choices=['train','val','test'])
    a=parser.parse_args()
    if a.action=='prepare': prepare(a.out,a.protocol)
    elif a.action=='labels': export_labels(a.protocol,a.partition)
    else: globals()[a.action](a.protocol)


if __name__=='__main__': main()
