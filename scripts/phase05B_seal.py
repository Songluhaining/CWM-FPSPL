"""Final immutable evidence inventory; no model fit or target access."""
from pathlib import Path
import csv,json,sys,shutil,time
from cwm_fp.data.common import ROOT,sha256,write_json
from cwm_fp.models.transition_data import load,verify_source
from cwm_fp.models.transition_train import verify_protocol

out=Path(sys.argv[1]);source=Path(sys.argv[2]);start=time.monotonic();protocol,digest=verify_protocol(out,'cpu');verify_source(source)
def reject_constant(value):raise ValueError('Nonstandard JSON constant: '+value)
for name in ('phase05B_protocol.json','phase05B_review_packet.json','phase05B_shortcut_diagnostics.json'):
    json.loads((ROOT/'outputs'/name).read_text(encoding='utf8'),parse_constant=reject_constant)
with (ROOT/'outputs/phase05B_learning_curves.csv').open(encoding='utf8',newline='') as f:curves=list(csv.DictReader(f))
assert len(curves)==180 and sum(int(r['updates']) for r in curves)==12780
assert all(int(r['prefixes'])==13597 and int(r['traces'])==283 for r in curves)
assert len({(r['model'],r['seed']) for r in curves})==9
for rel,n in [('phase05B_product_status.csv',22),('phase05B_graph_status.csv',368)]:
    with (ROOT/'outputs'/rel).open(encoding='utf8',newline='') as f:assert len(list(csv.DictReader(f)))==n
logs=out/'logs';logs.mkdir(exist_ok=True)
for p in (ROOT/'outputs').glob('phase05B*.log'):shutil.copy2(p,logs/p.name)
archive=out/'code_as_executed'
for base in (ROOT/'src/cwm_fp',):
    for p in base.rglob('*.py'):
        dest=archive/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
for p in list((ROOT/'scripts').glob('phase05B*.py'))+[ROOT/'tests/test_transition.py']:
    dest=archive/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
index={p.relative_to(out).as_posix():{'sha256':sha256(p),'bytes':p.stat().st_size} for p in out.rglob('*') if p.is_file() and p.name not in ('artifact_index.json','final_audit.json')}
write_json(out/'artifact_index.json',index)
outputs={p.relative_to(ROOT).as_posix():sha256(p) for p in (ROOT/'outputs').glob('phase05B*') if p.is_file() and p.suffix!='.log'}
sizes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file());output_bytes=sum(p.stat().st_size for p in (ROOT/'outputs').glob('phase05B*') if p.is_file())
ledger=load(out/'resource_ledger.json');report=load(ROOT/'outputs/phase05B_review_packet.json');elapsed=time.monotonic()-start
assert ledger['seconds']+load(out/'report_generation.json')['seconds']+report['tests_executed']['regression']['seconds']+elapsed<7200
assert sizes+output_bytes<4*2**30 and shutil.disk_usage(out).free>=4*2**30
write_json(out/'final_audit.json',{'stage_id':'05B','status':'PASS','frozen_protocol_sha256':digest,'frozen_code_unchanged':True,
    'completed_runs':9,'completed_epochs':180,'formal_updates':12780,'engineering_updates':3,'no_model_tuning_after_validation':True,
    'artifact_count':len(index),'artifact_index_sha256':sha256(out/'artifact_index.json'),'outputs':outputs,
    'source05A_unchanged':True,'new_Java_tests':0,'classification_target_reads':0,'test_trajectory_expansions':0,'Linux_GPU_verified':False,
    'worker_commands_seconds':ledger['seconds'],'report_seconds':load(out/'report_generation.json')['seconds'],'seal_seconds':elapsed,
    'runtime_bytes_before_final_audit':sizes,'repository_output_bytes':output_bytes,'disk_free_bytes':shutil.disk_usage(out).free,
    'phase_status_sha256':sha256(ROOT/'outputs/phase_status.md'),'contract_sha256':sha256(ROOT/'DATA_CONTRACTS.md'),'stages_sha256':sha256(ROOT/'STAGES.md')})
print('SEALED',len(index),'files',sizes+output_bytes,'bytes', 'seconds',ledger['seconds'])
