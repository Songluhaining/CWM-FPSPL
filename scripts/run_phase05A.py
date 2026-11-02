"""Record bounded CPU commands, retaining failed attempts; never rerun Java."""
import os,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cwm_fp.lddg.runner import command
from cwm_fp.models import __name__ as _unused
OUT=Path('D:/dataset/whn/cwm_fp_generated/data/processed/prefix/phase05A_v1')
PY=ROOT.parent/'work7/FP-SPL/.venv/Scripts/python.exe'
name=sys.argv[1];args=sys.argv[2:]
if (OUT/(name+'.log')).exists():raise SystemExit('Use a new command name; historical logs are immutable')
spent=sum(json.loads(p.read_text(encoding='utf8'))['seconds'] for p in OUT.glob('*.command.json'))
remaining=3600-spent
if remaining<=0:raise SystemExit('Cumulative phase05A execution budget exhausted')
os.environ['PYTHONPATH']=str(ROOT/'src')+os.pathsep+str(ROOT/'tests')
result=command([str(PY),'-X','utf8',*args],ROOT,OUT/(name+'.log'),seconds=min(3500,remaining),max_bytes=32*1024**2)
print(result)
raise SystemExit(result['exit_code'])
