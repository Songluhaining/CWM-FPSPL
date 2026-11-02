"""Package a new immutable code snapshot; no local experiment execution."""
import argparse,json,shutil,tarfile
from pathlib import Path
from phase05C2_export import ROOT,STAGE,digest,write

p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args()
if not a.name.startswith('code_v') or not a.name[6:].isdigit():raise ValueError('Versioned code snapshot required')
dest=STAGE/a.name
if dest.exists():raise ValueError('Snapshot exists')
dest.mkdir();files={}
paths=[f for directory in ('src','tests') for f in (ROOT/directory).rglob('*') if f.is_file() and f.suffix in {'.py','.json'}]
paths += [f for f in (ROOT/'scripts').glob('phase05C2*') if f.is_file()]
paths += [ROOT/'scripts/phase05C1_execute.py']
paths += [ROOT/f for f in ('third_party/existing_instrumenter/src/main/java/ppfl/instrumentation/TraceTransformer.java',
                          'third_party/existing_instrumenter/src/main/java/ppfl/instrumentation/opcode/GetStaticInst.java')]
paths += [ROOT/f for f in ('AGENTS.md','PROJECT_SPEC.md','DATA_CONTRACTS.md','STAGES.md','EVALUATION_PROTOCOL.md','INSTRUMENTATION_PLAN.md')]
for f in paths:
    rel=f.relative_to(ROOT).as_posix();target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target);files[rel]=digest(f)
write(dest/'code_receipt.json',{'files':files,'stage_id':'05C2'})
archive=STAGE/(a.name+'.tar.gz')
with tarfile.open(archive,'w:gz') as tf:tf.add(dest,arcname=a.name)
print(json.dumps({'archive':str(archive),'sha256':digest(archive),'files':len(files),'bytes':archive.stat().st_size}))
