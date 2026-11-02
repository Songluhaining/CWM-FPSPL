"""Container command ledger and bounded process-group monitor (no secrets)."""
import argparse,hashlib,json,os,signal,shutil,subprocess,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--cwd',type=Path,required=True);p.add_argument('--name',required=True)
    p.add_argument('--max-rss-bytes',type=int,default=8*1024**3)
    p.add_argument('--max-stage-bytes',type=int,default=None)
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    if not Path('/.dockerenv').exists():raise RuntimeError('Container required')
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    logs=a.root/'command_logs';logs.mkdir(exist_ok=True);record=logs/(a.name+'.json')
    if record.exists():raise ValueError('Never overwrite an executed command receipt')
    prior=sum(json.loads(f.read_text())['seconds'] for f in logs.glob('*.json'));remaining=7200-prior
    if remaining<=0:raise RuntimeError('Cumulative experiment time exhausted')
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(a.cwd/'src'),TMPDIR=str(a.root/'environment/tmp'),
        XDG_CACHE_HOME=str(a.root/'environment/cache'),CUDA_CACHE_PATH=str(a.root/'environment/cache/cuda'),TORCH_HOME=str(a.root/'environment/cache/torch'))
    start=time.monotonic();peak=0;reason=None;stage_bytes=None;next_disk_scan=start
    with (logs/(a.name+'.log')).open('w') as f:
        proc=subprocess.Popen(command,cwd=a.cwd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        while proc.poll() is None:
            try:
                stat=Path(f'/proc/{proc.pid}/status').read_text()
                rss=next(int(s.split()[1])*1024 for s in stat.splitlines() if s.startswith('VmRSS:'));peak=max(peak,rss)
            except (FileNotFoundError,StopIteration):pass
            if a.max_stage_bytes and time.monotonic()>=next_disk_scan:
                stage_bytes=sum(p.stat().st_size for p in a.root.rglob('*') if p.is_file());next_disk_scan=time.monotonic()+10
            if peak>a.max_rss_bytes:reason='process_RSS_above_declared_limit'
            elif stage_bytes is not None and stage_bytes>a.max_stage_bytes:reason='stage_bytes_above_declared_limit'
            elif time.monotonic()-start>remaining:reason='cumulative_experiment_7200s'
            elif shutil.disk_usage(a.root).free<10*1024**3:reason='disk_below_10GiB'
            if reason:os.killpg(proc.pid,signal.SIGTERM);break
            time.sleep(.5)
        try:code=proc.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
    rec={'command':command,'cwd':str(a.cwd),'container':os.uname().nodename,'pid':proc.pid,'exit_code':code,'seconds':time.monotonic()-start,
        'cumulative_seconds_before':prior,'sampled_peak_process_rss_bytes':peak,'termination_reason':reason,
        'monitor':'0.5s main-process RSS and available disk; process group terminated on budget; not cgroup hard quota; child RAM not separately sampled',
        'max_rss_bytes':a.max_rss_bytes,'max_stage_bytes':a.max_stage_bytes,'last_measured_stage_bytes':stage_bytes,
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'gpu_uuid_selection':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'log_sha256':hashlib.sha256((logs/(a.name+'.log')).read_bytes()).hexdigest()}
    record.write_text(json.dumps(rec,indent=2));print(json.dumps(rec),flush=True)
    raise SystemExit(code)
if __name__=='__main__':main()
