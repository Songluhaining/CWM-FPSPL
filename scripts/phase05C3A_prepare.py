"""Local file-only preparation: frozen selection, hashes and transfer package.

No trace replay, experimental statistics, Java execution or model access here.
"""
import argparse, hashlib, json, shutil, tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STAGE=Path('D:/dataset/whn/cwm_fp_generated/data/processed/local_state/phase05C3A_v1')
OLD=STAGE.parent/'phase05C2_v1/remote_evidence'
PRODUCER=ROOT/'data/processed/lddg/phase03_bank_v2'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)

def prepare():
    if STAGE.exists():raise ValueError('Stage already exists')
    STAGE.mkdir();out=STAGE/'transfer';out.mkdir();files={};missing=[]
    def add(src,rel,expected=None):
        src=Path(src)
        if not src.is_file() or expected and digest(src)!=expected:
            missing.append({'source':str(src),'expected_sha256':expected,'reason':'missing_or_changed'});return
        dst=out/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        files[rel]={'source':str(src),'sha256':digest(src),'bytes':src.stat().st_size}
    old=load(OLD/'inputs/protocol.json')
    protocol={'stage_id':'05C3A','schema':'field_read_evidence_protocol_v1',
      'old_protocol_sha256':digest(OLD/'inputs/protocol.json'),
      'old_source_manifest_sha256':digest(OLD/'inputs/source_manifest.json'),
      'source_root':'/workspace/codes/wanghaining_cwm_fp/phase05C2_v1',
      'selected':old['selected'],'partition':'grouped/train','selection':'exact inherited 24, no additions/replacement',
      'first_field_gap_denominator':15,'strict_product_branch_denominator':46,'strict_product_arithmetic_denominator':21,
      'classification':{'A':'verified read value and time, only write provenance absent',
        'B':'read completed, no reliable scalar/reference value',
        'C':'callback-to-instruction/frame/type/receiver association unreliable',
        'D':'earlier semantic/identity/stack/frame/control gap', 'E':'other evidenced reason'},
      'reference_policy':'Obj hash (including 0) is not a typed null/reference identity observation; no heap merging',
      'new_mode':'conditional only if actual class A exists; strict results unchanged',
      'primary_supported_types':['I','B','S','Z'],'future_callback_input':False,
      'java_replay_tests':['EmailSystem.Email_ESTest.test20','EmailSystem.Email_ESTest.test07'],
      'java_replay_product':old['selected'][0]['identity'],
      'java_gate':'all required runtime/runner/dependency/lifecycle/temp-path conditions verified before any test process',
      'budgets':{'max_java_test_processes':4,'rss_bytes':4*1024**3,'research_bytes':2*1024**3,
        'min_free_bytes':10*1024**3,'threads':2,'command_seconds':7200,'gpu_processes':0,'optimizer_updates':0},
      'forbidden':['FP/TP','fault locations','Email val/test traces','new instrumentation','initial heap recollection','model fitting'],
      'full_product_admission_changed':False}
    write(out/'protocol.json',protocol);files['protocol.json']={'source':'frozen local protocol','sha256':digest(out/'protocol.json'),'bytes':(out/'protocol.json').stat().st_size}
    shutil.copyfile(out/'protocol.json',ROOT/'outputs/phase05C3A_protocol.json')
    for name in ('DATA_CONTRACTS.md','STAGES.md','outputs/phase_status.md'):
        p=STAGE/'source_before'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
    for name in ('review_packet.md','review_packet.json','checks.json','protocol.json','trace_status.csv','local_state_status.csv','product_status.csv','graph_status.csv'):
        add(ROOT/('outputs/phase05C2_'+name),'prior/phase05C2_'+name)
    for name in ('administrator_guide.lake','receipt.json'):
        add(STAGE.parent/'phase05C1_v1/guide'/name,'guide/'+name)
    add(Path('E:/Program Files/Java/jdk-1.8/release'),'runtime/original_jdk_release.txt')
    build=load(PRODUCER/'producer-build-status.json')
    add(PRODUCER/'producer-build-status.json','runtime/producer-build-status.json')
    for rel,h in build['artifacts'].items():add(PRODUCER/rel,'runtime/'+rel,h)
    for src,h in build['inputs']['sources'].items():
        p=Path(src);rel=('producer-src/'+p.relative_to(PRODUCER/'producer-src').as_posix()) if p.is_relative_to(PRODUCER/'producer-src') else 'build_helpers/'+p.name
        add(p,'runtime/'+rel,h)
    for name,h in build['inputs']['libraries'].items():add(ROOT/'third_party/existing_instrumenter/lib'/name,'runtime/lib/'+name,h)
    for name in ('producer-build.command.json','producer-jar.command.json','producer-sources.txt'):
        add(PRODUCER/name,'runtime/'+name)
    add(ROOT/'third_party/existing_instrumenter/src/main/resources/META-INF/MANIFEST.MF','runtime/MANIFEST.MF')
    add(ROOT/'outputs/phase03_producer_provenance.json','runtime/original_producer_provenance.json')
    for name in ('CallBackIndex.java','TraceTransformer.java','opcode/GetStaticInst.java','opcode/GetFieldInst.java','opcode/OpcodeInst.java'):
        add(Path('E:/Myself/投稿/work5/code/BRFL/src/main/java/ppfl/instrumentation')/name,'reference_brfl/'+name)
    add(ROOT/'src/cwm_fp/lddg/runner.py','runtime/runner_current.py')
    write(out/'transfer_manifest.json',{'files':files,'missing_or_changed':missing,
          'policy':'file-only copy; all runtime interpretation and tests occur in existing container'})
    archive=STAGE/'transfer.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:tar.add(out,arcname='transfer')
    print(json.dumps({'archive':str(archive),'sha256':digest(archive),'bytes':archive.stat().st_size,
        'files':len(files),'missing_or_changed':missing},ensure_ascii=False))

def snapshot(name):
    if not name.startswith('code_v') or not name[6:].isdigit():raise ValueError('Versioned snapshot required')
    out=STAGE/name
    if out.exists():raise ValueError('Snapshot exists')
    out.mkdir();files={}
    paths=[p for d in ('src','tests') for p in (ROOT/d).rglob('*') if p.is_file() and p.suffix in {'.py','.json'}]
    paths += list((ROOT/'scripts').glob('phase05C3A*'))
    paths += [ROOT/'scripts/phase05C2_regressions.py',ROOT/'scripts/phase05C1_execute.py']
    paths += [ROOT/n for n in ('AGENTS.md','PROJECT_SPEC.md','DATA_CONTRACTS.md','STAGES.md','EVALUATION_PROTOCOL.md','INSTRUMENTATION_PLAN.md')]
    for p in paths:
        rel=p.relative_to(ROOT).as_posix();dst=out/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst);files[rel]=digest(p)
    write(out/'code_receipt.json',{'stage_id':'05C3A','files':files})
    archive=STAGE/(name+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:tar.add(out,arcname=name)
    print(json.dumps({'archive':str(archive),'sha256':digest(archive),'files':len(files)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action');a=p.parse_args()
    prepare() if a.action=='prepare' else snapshot(a.action)
