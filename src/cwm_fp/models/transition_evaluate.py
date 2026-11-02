"""Fixed validation selection and post-hoc observation-only diagnostics."""
from __future__ import annotations
from collections import defaultdict,Counter
import json,math,time
import numpy as np
from cwm_fp.data.common import ROOT,write_json,write_csv,sha256
from .transition_data import load,read_targets
from .transition_model import KINDS
from .transition_train import SEEDS,verify_protocol

EPS=1e-12
def cat_metrics(y,p):
    if not len(y):return {'n':0,'accuracy':None,'macro_F1':None,'NLL':None}
    y=np.asarray(y,int);pred=np.argmax(p,1);f=[]
    for k in range(9):
        tp=np.sum((y==k)&(pred==k));den=np.sum(y==k)+np.sum(pred==k);f.append(2*tp/den if den else 0.)
    return {'n':len(y),'accuracy':float(np.mean(pred==y)),'macro_F1':float(np.mean(f)),'NLL':float(-np.log(np.clip(p[np.arange(len(y)),y],EPS,1)).mean())}

def bin_metrics(y,p):
    if not len(y):return {'n':0,'NLL':None,'Brier':None,'accuracy':None,'balanced_accuracy':None,'tn':0,'fp':0,'fn':0,'tp':0,'fallthrough_support':0,'jump_support':0}
    y=np.asarray(y,int);pred=p>=.5;tn=int(np.sum((y==0)&~pred));fp=int(np.sum((y==0)&pred));fn=int(np.sum((y==1)&~pred));tp=int(np.sum((y==1)&pred))
    recalls=[tp/(tp+fn)] if tp+fn else []
    if tn+fp:recalls.append(tn/(tn+fp))
    return {'n':len(y),'NLL':float(-(y*np.log(np.clip(p,EPS,1))+(1-y)*np.log(np.clip(1-p,EPS,1))).mean()),'Brier':float(np.mean((p-y)**2)),
        'accuracy':float(np.mean(pred==y)),'balanced_accuracy':float(np.mean(recalls)),'tn':tn,'fp':fp,'fn':fn,'tp':tp,'fallthrough_support':tn+fp,'jump_support':tp+fn}

def grouped_metric(rows,targets,pa,pb,mask,task):
    valid=np.array([t[task]>=0 for t in targets])&mask;ids=np.flatnonzero(valid);groups=defaultdict(list)
    for i in ids:groups[tuple(rows[i]['identity'][:2])].append(i)
    fn=cat_metrics if task=='A' else bin_metrics;pp=pa if task=='A' else pb;yy=np.array([t[task] for t in targets])
    micro=fn(yy[ids],pp[ids]);vm=[fn(yy[ii],pp[ii]) for ii in groups.values()]
    keys=('accuracy','macro_F1','NLL') if task=='A' else ('accuracy','balanced_accuracy','NLL','Brier')
    macro={k:float(np.mean([v[k] for v in vm])) if vm else None for k in keys}
    macro.update(n=len(ids),version_count=len(groups));micro['version_count']=len(groups)
    return micro,macro

def checkpoint_score(rows,targets,pa,pb):
    abstain=np.array([not t['covered'] for t in targets]);allmask=np.ones(len(rows),bool)
    _,a=grouped_metric(rows,targets,pa,pb,abstain,'A');_,b=grouped_metric(rows,targets,pa,pb,allmask,'B')
    if a['NLL'] is None or b['NLL'] is None:return None,a,b
    return .5*a['NLL']/math.log(9)+.5*b['NLL']/math.log(2),a,b

def choose_checkpoint(candidates):
    best=20;score=None
    for epoch,value in sorted(candidates):
        if value is not None and (score is None or value<score-1e-6):best=epoch;score=value
    return best,score

def fit_baselines(rows,targets):
    counts=np.zeros(9);conditional=np.zeros((9,9));tables={k:{'A':defaultdict(Counter),'B':defaultdict(Counter)} for k in KINDS};branches=[]
    for r,t in zip(rows,targets):
        if t['A']>=0:counts[t['A']]+=1;conditional[r['category'],t['A']]+=1
        if t['B']>=0:branches.append(t['B'])
        for k in KINDS:
            for task in ('A','B'):
                if t[task]>=0:tables[k][task][r['signatures'][k]][t[task]]+=1
    prior=counts/counts.sum();c=(conditional+1)/(conditional.sum(1,keepdims=True)+9);c[conditional.sum(1)==0]=prior
    return prior,c,float(np.mean(branches)),tables

def baseline_predict(rows,fit,kind):
    prior,conditional,branch,tables=fit
    a=np.tile(prior,(len(rows),1)) if kind=='Prior' else conditional[[r['category'] for r in rows]].copy();b=np.full(len(rows),branch)
    if kind.startswith('Exact-'):
        view=kind[6:]
        for i,r in enumerate(rows):
            for task in ('A','B'):
                counter=tables[view][task].get(r['signatures'][view])
                if counter:
                    if task=='A':a[i]=[counter.get(j,0)/sum(counter.values()) for j in range(9)]
                    else:b[i]=counter.get(1,0)/sum(counter.values())
    return a,b

def diagnostics(train,targets,val):
    fit=fit_baselines(train,targets);tables=fit[3];d={'views':{},'interpretation':'Exact ordered tensor signatures, not graph isomorphism or functional equivalence. Empirical conflicts are representation ambiguity, not intrinsic program noise or Bayes error.'}
    seen={}
    for k in KINDS:
        train_groups=defaultdict(list)
        for i,r in enumerate(train):train_groups[r['signatures'][k]].append(i)
        train_keys=set(train_groups);seen[k]=np.array([r['signatures'][k] in train_keys for r in val]);shared={r['signatures'][k] for r in val}&train_keys
        item={'train_unique_groups':len(train_groups),'train_repeated_groups':sum(len(g)>1 for g in train_groups.values()),
            'train_repeated_occurrences':sum(len(g) for g in train_groups.values() if len(g)>1),'cross_partition_groups':len(shared),'val_seen_prefixes':int(seen[k].sum()),'val_unseen_prefixes':int((~seen[k]).sum()),'tasks':{}}
        for task in ('A','B'):
            groups=tables[k][task];total=sum(sum(c.values()) for c in groups.values());conflicts=[];majority=entropy=0
            for key,c in groups.items():
                n=sum(c.values());majority+=max(c.values());entropy+=-sum(v*math.log(v/n) for v in c.values())
                if len(c)>1:conflicts.append({'signature':key,'n':n,'counts':dict(c),'probability':{str(y):v/n for y,v in c.items()}})
            item['tasks'][task]={'valid_targets':total,'conflicting_groups':len(conflicts),'affected_targets':sum(c['n'] for c in conflicts),
                'empirical_majority_accuracy_upper_bound':majority/total if total else None,'empirical_conditional_entropy_nats':entropy/total if total else None,'conflict_distributions':conflicts}
        d['views'][k]=item
    common=np.logical_and.reduce([~seen[k] for k in KINDS]);d['common_unseen_prefixes']=int(common.sum())
    return d,seen,fit

def evaluate(a,budget):
    protocol,protocol_sha=verify_protocol(a.out,a.device);start=time.monotonic()
    indexes={p:load(a.out/'cache'/f'index_{p}.json') for p in ('train','val')};rows={p:indexes[p]['rows'] for p in indexes};targets={p:read_targets(a.out,indexes[p]) for p in indexes}
    diag,seen,fit=diagnostics(rows['train'],targets['train'],rows['val']);common=np.logical_and.reduce([~seen[k] for k in KINDS])
    metrics=[];selection=[];curves=load(a.out/'learning_curves.json');chosen={};group_metrics=[]
    def append_results(name,seed,epoch,part,pa,pb,view=None):
        rr=rows[part];tt=targets[part];n=len(rr);allmask=np.ones(n,bool)
        views={'all':allmask}
        if part=='val':
            views.update(full_seen=seen['rgcn'],full_unseen=~seen['rgcn'],common_unseen=common)
            if view:views.update(own_seen=seen[view],own_unseen=~seen[view])
        both=defaultdict(set)
        for t in tt:
            if t['B']>=0:both[json.dumps(t['branch_point'],sort_keys=True)].add(t['B'])
        strata={'all':allmask,'A_covered':np.array([t['covered'] for t in tt]),'A_abstain':np.array([not t['covered'] for t in tt]),
            **{s:np.array([t['boundary']==s for t in tt]) for s in ('conditional','call','return')},
            'certified_exception':np.array([t['exception'] for t in tt]),'framework_interval':np.array([t['framework_interval'] for t in tt]),
            'both_branch_outcomes':np.array([len(both[json.dumps(t['branch_point'],sort_keys=True)])==2 for t in tt])}
        for subset,mask in views.items():
            for stratum,sm in strata.items():
                for task in ('A','B'):
                    if task=='B' and stratum not in ('all','both_branch_outcomes'):continue
                    if task=='A' and stratum=='both_branch_outcomes':continue
                    micro,macro=grouped_metric(rr,tt,pa,pb,mask&sm,task)
                    for aggregation,mm in (('event_micro',micro),('version_macro',macro)):
                        metrics.append({'model':name,'seed':seed,'epoch':epoch,'partition':part,'subset':subset,'stratum':stratum,'task':task,'aggregation':aggregation,**mm})
        for level,width in (('version',2),('product',3),('trajectory',4)):
            groups=defaultdict(list)
            for i,(r,t) in enumerate(zip(rr,tt)):
                if t['B']>=0:groups[tuple(r['identity'][:width])].append(i)
            for identity,ids in groups.items():group_metrics.append({'model':name,'seed':seed,'epoch':epoch,'partition':part,'level':level,'identity':json.dumps(identity,ensure_ascii=False),**bin_metrics([tt[i]['B'] for i in ids],pb[ids])})
    for name in ('Prior','Category-Markov','Exact-rgcn','Exact-no_dep','Exact-seq_gru'):
        for part in ('train','val'):
            pa,pb=baseline_predict(rows[part],fit,name);append_results(name,None,None,part,pa,pb,name[6:] if name.startswith('Exact-') else 'rgcn')
        budget()
    static={}
    for part in ('train','val'):
        rr=rows[part];tt=targets[part];groups=defaultdict(list)
        for r,t in zip(rr,tt):
            if t['A']>=0:groups[tuple(r['identity'][:2])].append(t)
        def cfg(ts):
            valid=[t for t in ts if t['A']>=0];covered=[t for t in valid if t['static_prediction']>=0];correct=sum(t['A']==t['static_prediction'] for t in covered)
            return {'A_valid':len(valid),'covered':len(covered),'correct':correct,'coverage':len(covered)/len(valid),'covered_accuracy':correct/len(covered) if covered else None,'correct_overall':correct/len(valid)}
        micro=cfg(tt);versions=[{'identity':list(k),**cfg(v)} for k,v in groups.items()];macro={k:float(np.mean([v[k] for v in versions])) for k in ('coverage','covered_accuracy','correct_overall')}
        static[part]={'event_micro':micro,'version_macro':macro,'versions':versions,'extra_information':'normal CFG exact static lookup not supplied to neural models; abstains, exceptions retained'}
        for aggregation,mm in (('event_micro',micro),('version_macro',macro)):metrics.append({'model':'Normal-CFG','seed':None,'epoch':None,'partition':part,'subset':'all','stratum':'all','task':'A','aggregation':aggregation,**mm,'version_count':len(versions)})
    progress=load(a.out/'training_progress.json')
    for run in progress['completed']:
        kind=run['model'];seed=run['seed'];path=a.out/'runs'/f'{kind}_{seed}';best=None;bestscore=None;candidates=[]
        for epoch in (5,10,20):
            file=path/f'epoch{epoch:02d}_val.npz'
            with np.load(file) as z:pa,pb=z['A'],z['B']
            score,ma,mb=checkpoint_score(rows['val'],targets['val'],pa,pb);candidates.append({'epoch':epoch,'score':score,'A_abstain_version_macro':ma,'B_version_macro':mb})
            for c in curves:
                if c['model']==kind and c['seed']==seed and c['epoch']==epoch:c['val_selection_score']=score;c['val_A_abstain_NLL']=ma['NLL'];c['val_B_NLL']=mb['NLL']
        best,bestscore=choose_checkpoint([(c['epoch'],c['score']) for c in candidates])
        selection.append({'model':kind,'seed':seed,'selected_epoch':best,'score':bestscore,'selection_valid':bestscore is not None,'candidates':candidates})
        for part in ('train','val'):
            with np.load(path/f'epoch{best:02d}_{part}.npz') as z:pa,pb=z['A'],z['B']
            append_results(kind,seed,best,part,pa,pb,kind)
        budget()
    diag.update(static_reference=static,train_priors={'A':fit[0].tolist(),'B_jump':fit[2]},selection=selection,
        common_unseen_A_valid=int(sum(common[i] and t['A']>=0 for i,t in enumerate(targets['val']))),common_unseen_B_valid=int(sum(common[i] and t['B']>=0 for i,t in enumerate(targets['val']))),
        metric_rules={'multiclass_macro_F1':'all fixed9 categories; absent precision/recall denominator ->0','balanced_accuracy':'mean recall of observed classes; supports reported',
            'NLL_probability_clip':EPS,'version_macro':'event means within each eligible version, then equal versions','branch_threshold':.5,
            'both_outcomes':'posthoc within each partition, exact source product/domain/BCI point, not features','own_unseen':'view-specific; not used alone to rank models'})
    write_json(a.out/'evaluation.json',{'selection':selection,'seconds':time.monotonic()-start,'protocol_sha256':protocol_sha,'completed_runs':len(progress['completed'])})
    def csv(path,records):
        fields=list(dict.fromkeys(k for r in records for k in r));write_csv(path,records,fields)
    csv(ROOT/'outputs/phase05B_metrics.csv',metrics);csv(ROOT/'outputs/phase05B_learning_curves.csv',curves);csv(a.out/'branch_group_metrics.csv',group_metrics)
    write_json(ROOT/'outputs/phase05B_shortcut_diagnostics.json',diag);write_json(a.out/'metrics.json',metrics)
    print(json.dumps({'selections':selection,'common_unseen':diag['common_unseen_prefixes'],'seconds':time.monotonic()-start}),flush=True)
