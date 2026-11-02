"""Bounded phase05B entry; no classification labels, Java or test partition."""
from __future__ import annotations
import argparse,json,platform,sys,time,traceback,shutil
from pathlib import Path
import numpy as np
import torch
from cwm_fp.data.common import ROOT,sha256,write_json
from .transition_budget import Budget
from .transition_data import access_guard,cache_partition,read_cache,read_targets,hierarchy_weights,load

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cache_train','cache_val','preflight','train','evaluate','isolation'])
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cpu',choices=['cpu','cuda']);a=p.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    a.out.mkdir(parents=True,exist_ok=True);budget=Budget(a.out);exit_code=0;error=None
    try:
        allowed={'train'} if a.mode in ('cache_train','preflight') else {'train','val'}
        access=access_guard(a.source,allowed)
        env={'platform':platform.platform(),'python':sys.version,'executable':sys.executable,'torch':str(torch.__version__),
            'numpy':np.__version__,'device':a.device,'threads':1,'cuda_available':torch.cuda.is_available(),'actual_GPU_training':False}
        if not (a.out/'environment.json').exists():write_json(a.out/'environment.json',env)
        if a.mode.startswith('cache_'):
            if a.mode=='cache_val' and not (ROOT/'outputs/phase05B_protocol.json').exists():raise ValueError('Freeze protocol before val cache')
            index=cache_partition(a.source,a.out,a.mode.split('_')[1],budget)
            write_json(a.out/(a.mode+'_receipt.json'),{'index_sha256':sha256(a.out/'cache'/('index_'+a.mode.split('_')[1]+'.json'))})
            print(json.dumps(index['stats']),flush=True)
        elif a.mode=='preflight':
            from .transition_train import preflight
            preflight(a,budget)
        elif a.mode=='train':
            from .transition_train import train
            train(a,budget)
        elif a.mode=='evaluate':
            from .transition_evaluate import evaluate
            evaluate(a,budget)
        elif a.mode=='isolation':
            from .transition_checks import isolation
            isolation(a,budget)
        write_json(a.out/(a.mode+'_target_access.json'),access)
    except Exception as exc:
        exit_code=1;error=type(exc).__name__+': '+str(exc);traceback.print_exc()
    finally:budget.finish([sys.executable,'-m','cwm_fp.models.transition_run',*sys.argv[1:]],exit_code,error)
    raise SystemExit(exit_code)

if __name__=='__main__':main()
