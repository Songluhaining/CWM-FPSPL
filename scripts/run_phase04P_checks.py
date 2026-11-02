"""Bounded phase-local checks; tolerate only disappearing temporary-file stats."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cwm_fp.lddg import runner
OUT=ROOT/'data/processed/graph_classifier/phase04P_v1'

def stable_usage(root):
    size=0;count=0
    def missing(error):
        if not isinstance(error,FileNotFoundError) or not Path(error.filename).is_relative_to(OUT/'temp'):raise error
    for folder,_,names in os.walk(root,onerror=missing):
        for name in names:
            p=Path(folder)/name
            try:size+=p.stat().st_size;count+=1
            except FileNotFoundError as error:missing(error)
    return {'bytes':size,'files':count}

if __name__=='__main__':
    runner.tree_usage=stable_usage
    for script,name in [('test_phase04P.py','unit_final_v3'),('audit_phase04P_integrity.py','integrity')]:
        result=runner.command(['E:/Myself/投稿/work7/FP-SPL/.venv/Scripts/python.exe','-X','utf8',str(ROOT/'scripts'/script)],
            ROOT,OUT/(name+'.log'),seconds=600,max_bytes=8*1024**2,
            budget={'root':str(OUT),'max_total_bytes':512*1024**2,'max_total_files':10000,'min_free_bytes':1024**3},
            max_rss_bytes=int(1.5*1024**3))
        print(name,result,flush=True);print((OUT/(name+'.log')).read_text(encoding='utf-8')[-2000:],flush=True)
        if result['exit_code']!=0 or result['reason']:raise SystemExit(1)
