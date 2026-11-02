"""Small dependency-free evaluator (including tied scores and clustered intervals)."""
from __future__ import annotations

import math
import random


def binary_metrics(y, score) -> dict:
    if len(y) != len(score) or any(v not in (0, 1) for v in y) or any(not math.isfinite(s) or not 0 <= s <= 1 for s in score):
        raise ValueError("Invalid binary targets/probabilities")
    if not y:
        return {"n": 0, "status": "unavailable", "pr_auc": None, "average_precision": None}
    n, positives = len(y), sum(y)
    pred = [int(s >= .5) for s in score]
    tp = sum(a == b == 1 for a, b in zip(y, pred)); tn = sum(a == b == 0 for a, b in zip(y, pred))
    fp, fn = n - positives - tn, positives - tp
    precision, recall = tp / max(1, tp + fp), tp / positives if positives else None
    result = {"n": n, "status": "ok", "FP": positives, "TP": n - positives,
              "fp_precision": precision, "fp_recall": recall, "fp_f1": 2 * tp / max(1, 2 * tp + fp + fn),
              "accuracy": (tp + tn) / n, "confusion_matrix": [[tn, fp], [fn, tp]],
              "balanced_accuracy": ((recall + tn / (n - positives)) / 2) if 0 < positives < n else None,
              "pr_auc": None, "average_precision": None, "roc_auc": None,
              "brier": sum((s - a) ** 2 for s, a in zip(score, y)) / n,
              "log_loss": -sum(a * math.log(max(s, 1e-12)) + (1-a) * math.log(max(1-s, 1e-12)) for a, s in zip(y, score)) / n,
              "probability_status": "uncalibrated", "unavailable_reasons": []}
    ece = 0.
    for bin_id in range(10):
        ids = [i for i, s in enumerate(score) if min(9, int(s * 10)) == bin_id]
        if ids: ece += abs(sum(score[i] - y[i] for i in ids)) / n
    result["ece_10_equal_width"] = ece
    if 0 < positives < n:
        tied = {}
        for a, s in zip(y, score):
            group = tied.setdefault(s, [0, 0]); group[a] += 1
        previous_r, previous_p, previous_f, pos, neg = 0., 1., 0., 0, 0
        pr, ap, roc = 0., 0., 0.
        for s in sorted(tied, reverse=True):
            negatives_here, positives_here = tied[s]
            pos += positives_here; neg += negatives_here
            r, p, f = pos / positives, pos / (pos + neg), neg / (n - positives)
            pr += (r - previous_r) * (p + previous_p) / 2
            ap += (r - previous_r) * p
            roc += (f - previous_f) * (r + previous_r) / 2
            previous_r, previous_p, previous_f = r, p, f
        result.update(pr_auc=pr, average_precision=ap, roc_auc=roc)
    else:
        result["unavailable_reasons"].append("Both classes are required for AUC and balanced accuracy")
    return result


def clustered_interval(rows, seed, replicates=200):
    groups = {}
    for r in rows: groups.setdefault(tuple(r["identity"][:2]), []).append(r)
    if len(groups) < 2:
        return {"status": "unavailable", "reason": "fewer than two version clusters"}
    values, rng, keys = [], random.Random(seed), sorted(groups)
    for _ in range(replicates):
        sample = [r for k in rng.choices(keys, k=len(keys)) for r in groups[k]]
        m = binary_metrics([r["y_true"] for r in sample], [r["y_score"] for r in sample])
        if m["pr_auc"] is not None: values.append(m)
    def quantile(numbers, q):
        v = sorted(numbers); p = q * (len(v)-1); i = int(p)
        return v[i] + (p-i) * (v[min(i+1, len(v)-1)]-v[i])
    return {"method": "percentile bootstrap by (system_id,bug_version_id)", "clusters": len(keys),
            "requested": replicates, "valid_two_class": len(values), "level": .95,
            "intervals": {k: [quantile([v[k] for v in values], .025), quantile([v[k] for v in values], .975)] if values else None
                          for k in ("fp_f1", "pr_auc", "average_precision", "roc_auc", "balanced_accuracy")}}


def diagnostic_metrics(rows,targets):
    """04R1: explicit binary loss, AP (not trapezoidal PR area), version counts."""
    keys=[tuple(r['identity']) for r in rows]
    if len(set(keys))!=len(keys) or set(keys)!=set(targets):raise ValueError('Exact qualified prediction/target join required')
    y=[int(targets[k]=='FP') for k in keys];scores=[float(r['y_score']) for r in rows]
    if any(targets[k] not in ('FP','TP') for k in keys):raise ValueError('Only FP/TP targets allowed')
    m=binary_metrics(y,scores);m.pop('pr_auc',None)
    m['fp_recall']=m['fp_recall'] if m['fp_recall'] is not None else 0.
    mean=sum(scores)/len(scores)
    m['probability_summary']={'min':min(scores),'max':max(scores),'mean':mean,'std_population':math.sqrt(sum((s-mean)**2 for s in scores)/len(scores))}
    m['predicted_counts']={'FP':sum(s>=.5 for s in scores),'TP':sum(s<.5 for s in scores)}
    m['class_versions']={c:len({k[:2] for k in keys if targets[k]==c}) for c in ('FP','TP')}
    m['versions']=len({k[:2] for k in keys});m['confusion_axes']='rows true [TP=0,FP=1]; columns predicted [TP=0,FP=1]'
    m['single_class_ranking_applicable']=len(set(y))==2
    m['by_class_probabilities']={}
    for c in ('FP','TP'):
        vals=[s for s,k in zip(scores,keys) if targets[k]==c]
        m['by_class_probabilities'][c]=({'n':len(vals),'min':min(vals),'max':max(vals),'mean':sum(vals)/len(vals),
            'std_population':math.sqrt(sum((s-sum(vals)/len(vals))**2 for s in vals)/len(vals))} if vals else None)
    m['by_version']=[]
    for group in sorted({k[:2] for k in keys}):
        inds=[i for i,k in enumerate(keys) if k[:2]==group];sub=binary_metrics([y[i] for i in inds],[scores[i] for i in inds])
        m['by_version'].append({'identity':list(group),'products':len(inds),'FP':sub['FP'],'TP':sub['TP'],'log_loss':sub['log_loss'],
            'errors':sum(int(scores[i]>=.5)!=y[i] for i in inds)})
    return m
