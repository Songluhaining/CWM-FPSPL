"""Cumulative stage budget measured inside the actual computation process."""
import json,shutil,time
from pathlib import Path
from cwm_fp.data.common import write_json
from .package_training import process_memory

class Budget:
    def __init__(self,out):
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        self.path=self.out/'resource_ledger.json'
        self.state=json.loads(self.path.read_text(encoding='utf8')) if self.path.exists() else {'seconds':0.,'updates':0,'commands':[]}
        self.start=time.monotonic();self.last_scan=0.;self.bytes=0
    def __call__(self):
        elapsed=time.monotonic()-self.start
        if self.state['seconds']+elapsed>7200:raise RuntimeError('BUDGET_SECONDS')
        if self.state['updates']>15000:raise RuntimeError('BUDGET_UPDATES')
        memory=process_memory()['peak_working_set_bytes']
        if memory is not None and memory>8*2**30:raise RuntimeError('BUDGET_MEMORY')
        if elapsed-self.last_scan>15 or self.last_scan==0:
            self.bytes=sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file());self.last_scan=elapsed
            if self.bytes>4*2**30:raise RuntimeError('BUDGET_OUTPUT')
            if shutil.disk_usage(self.out).free<4*2**30:raise RuntimeError('BUDGET_FREE_DISK')
    def update(self):self.state['updates']+=1;self()
    def finish(self,command,exit_code,error=None):
        self.state['seconds']+=time.monotonic()-self.start
        self.state['commands'].append({'command':command,'exit_code':exit_code,'error':error,'seconds':time.monotonic()-self.start,
            'memory':process_memory(),'output_bytes':sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file()),'disk_free_bytes':shutil.disk_usage(self.out).free})
        write_json(self.path,self.state)
