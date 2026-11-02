"""Declared bounded phase03 acceptance: units, 6 benchmark pairs, synthetic Java, extraction."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cwm_fp.data.common import sha256,write_json


def main():
    config=ROOT/'configs/lddg_smoke.json'
    record={'started_at':datetime.datetime.now().astimezone().isoformat(),'commands':[],
            'config_sha256':sha256(config),'scope':'phase03 only, no model training','success':False,
            'code_sha256':{p.relative_to(ROOT).as_posix():sha256(p) for p in sorted((ROOT/'src/cwm_fp/lddg').rglob('*')) if p.is_file() and '__pycache__' not in p.parts},
            'python':sys.version,'executable':sys.executable}
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    work=ROOT/json.loads(config.read_text(encoding='utf-8'))['work_dir']
    stages=[('unit_tests',['-m','unittest','discover','-s','tests','-v'],120),
            ('collection',['-m','cwm_fp.lddg.runner','--config',str(config)],600),
            ('java_fixtures',['scripts/run_phase03_fixtures.py'],600),
            ('extraction',['-m','cwm_fp.lddg.extract','--work-dir',str(work)],120),
            ('verification',['scripts/verify_phase03_artifacts.py'],120)]
    for name,args,timeout in stages:
        command=[sys.executable,'-X','utf8','-B',*args];log=ROOT/'outputs'/('phase03_'+name+'.log');start=time.monotonic()
        error=None
        with log.open('w',encoding='utf-8') as out:
            try:code=subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=timeout).returncode
            except subprocess.TimeoutExpired as e:code=124;error=str(e)
        record['commands'].append({'name':name,'argv':command,'exit_code':code,'timeout_seconds':timeout,'timeout_error':error,
                                   'seconds':round(time.monotonic()-start,3),'log':str(log),'log_sha256':sha256(log)})
        write_json(ROOT/'outputs/phase03_run.json',record);print(name,code,record['commands'][-1]['seconds'],flush=True)
        if code:raise RuntimeError('Phase03 bounded acceptance failed: '+str(log))
    record.update(success=True,completed_at=datetime.datetime.now().astimezone().isoformat())
    write_json(ROOT/'outputs/phase03_run.json',record)

if __name__=='__main__':main()
