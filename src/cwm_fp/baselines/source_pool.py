"""Separate, explicitly transductive PULP source F/unlabeled pool comparison."""
import time
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from cwm_fp.data.common import sha256, write_csv, write_json
from .boundaries import install_io_guard
from .vendor import FrozenSourcePU


def fit_source_pool(c):
    from .pipeline import context, load_features, score_rows, SCORE_FIELDS
    install_io_guard('inference')  # No FP/TP labels, even on TRAIN.
    products, _, ordered = context(c, 'grouped')
    values = load_features(c, 'grouped', ordered)['pulp50']
    x = np.empty_like(values)
    for group in sorted({k[:2] for k in ordered}):
        indexes = [i for i,k in enumerate(ordered) if k[:2]==group]
        x[indexes] = MinMaxScaler().fit_transform(values[indexes])
    failing = [i for i,k in enumerate(ordered) if products[k]['observed_result']=='FAIL']
    passing = [i for i,k in enumerate(ordered) if products[k]['observed_result']=='PASS']
    dest = c['work_dir']/'source_pool'
    dest.mkdir(exist_ok=False)
    receipts = []
    for seed in c['seeds']:
        started=time.monotonic()
        model=FrozenSourcePU(seed,c['pu_estimators'],c['pu_sampling_ratio']).fit(x[failing],x[passing])
        scores=model.score(x[passing]);pred=(scores>.5).astype(int)
        rows=score_rows(products,ordered,passing,scores,pred,'probability_uncalibrated',
            'all_unlabeled_pool','pulp_source_pool',seed)
        file=dest/f'{seed}_predictions.csv';write_csv(file,rows,SCORE_FIELDS)
        joblib.dump(model,dest/f'{seed}_model.joblib')
        receipts.append({'seed':seed,'seconds':time.monotonic()-started,'predictions_sha256':sha256(file)})
    write_json(dest/'receipt.json',{'protocol':'source_pool_transductive_v1','oracle_labels_read':False,
        'failing':len(failing),'unlabeled':len(passing),'receipts':receipts,
        'normalization':'original per-version 50D min/max, all observed F/U products',
        'not_generalization':True, 'source_algorithm':'unchanged 5% negative sample, accuracy weighting, 30 SVMs'})
