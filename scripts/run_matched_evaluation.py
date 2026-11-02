"""Bounded driver for the existing-model holdout comparison; no neural fitting."""
from pathlib import Path
import json
import os
import subprocess
import time
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--statistics-python',required=True)
    p.add_argument('--graph-python',required=True)
    p.add_argument('--protocol',type=Path)
    p.add_argument('--run-id',default='phaseEVAL1')
    p.add_argument('--resume-after',choices=['freeze'])
    a=p.parse_args()
    out=ROOT/'data/processed/baselines/eval1_matched_v1'
    protocol=a.protocol or ROOT/'outputs/phaseEVAL1_protocol.json'
    module='cwm_fp.models.matched_evaluation'
    config=out/'baseline_config.json'
    commands=[('freeze',a.statistics_python,['-m',module,'prepare','--protocol',str(protocol),'--out',str(out)]),
        ('train_targets',a.statistics_python,['-m',module,'labels','--protocol',str(protocol),'--partition','train']),
        ('graph_prediction',a.graph_python,['-m',module,'graph_predict','--protocol',str(protocol)]),
        ('baseline_fit',a.statistics_python,['-m','cwm_fp.baselines.pipeline','fit','--config',str(config)]),
        ('source_pu',a.statistics_python,['-m',module,'pu_predict','--protocol',str(protocol)]),
        ('validation_targets',a.statistics_python,['-m',module,'labels','--protocol',str(protocol),'--partition','val']),
        ('baseline_selection',a.statistics_python,['-m','cwm_fp.baselines.pipeline','select','--config',str(config)]),
        ('baseline_prediction',a.statistics_python,['-m','cwm_fp.baselines.pipeline','predict','--config',str(config)]),
        ('seal',a.statistics_python,['-m',module,'seal','--protocol',str(protocol)]),
        ('test_targets',a.statistics_python,['-m',module,'labels','--protocol',str(protocol),'--partition','test']),
        ('evaluation',a.statistics_python,['-m',module,'evaluate','--protocol',str(protocol)])]
    env=os.environ.copy()
    env.update(PYTHONPATH=str(ROOT/'src'),PYTHONUTF8='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    if a.resume_after=='freeze':commands=commands[1:]
    receipt=ROOT/'outputs'/(a.run_id+'_commands.json')
    if receipt.exists():raise ValueError('Run exists; inspect evidence instead of repeating test evaluation')
    rows=[];start=time.monotonic()
    for name,exe,args in commands:
        log=ROOT/'outputs'/(a.run_id+'_'+name+'.log');t=time.monotonic();cmd=[exe,'-X','utf8','-B',*args]
        with log.open('w',encoding='utf-8') as stream:
            try:code=subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=600).returncode
            except subprocess.TimeoutExpired:code=None
        rows.append({'name':name,'argv':cmd,'exit_code':code,'seconds':time.monotonic()-t,'log':str(log)})
        receipt.write_text(json.dumps({'commands':rows,'wall_seconds':time.monotonic()-start,
            'completed':name=='evaluation' and code==0,'graph_updates':0},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(name,code,round(rows[-1]['seconds'],3),flush=True)
        if code!=0:return 1
    return 0


if __name__=='__main__':sys.exit(main())
