"""Prepare a new, immutable train/validation-only Linux transfer archive.

This packaging command does not replay observations or execute experiments.
Original manifests remain byte-identical; a separate receipt binds the subset.
"""
import argparse, hashlib, json, shutil, tarfile, time
from pathlib import Path

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    srca=Path('D:/dataset/whn/cwm_fp_generated/data/processed/prefix/phase05A_v1')
    srcb=Path('D:/dataset/whn/cwm_fp_generated/data/processed/transitions/phase05B_cpu_v1')
    if a.out.exists():raise ValueError('Never overwrite a sealed export')
    a.out.mkdir(parents=True);files={};start=time.monotonic()
    def add(src,rel,expected=None):
        h=digest(src)
        if expected and h!=expected:raise ValueError('Frozen source hash mismatch: '+str(src))
        dest=a.out/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
        if digest(dest)!=h:raise ValueError('Copy changed')
        files[rel]={'sha256':h,'bytes':src.stat().st_size}
    for sub in ('src','tests'):
        for f in (root/sub).rglob('*.py'):add(f,'code/'+f.relative_to(root).as_posix())
    for f in ['AGENTS.md','PROJECT_SPEC.md','DATA_CONTRACTS.md','EVALUATION_PROTOCOL.md','INSTRUMENTATION_PLAN.md','STAGES.md']:
        add(root/f,'code/'+f)
    for f in (root/'scripts').glob('phase05C1*'):
        if f.is_file():add(f,'code/scripts/'+f.name)
    add(root/'scripts/phase05B_linux_acceptance.py','code/scripts/phase05B_linux_acceptance.py')
    for f in ['phase05A_protocol.json','phase05A_review_packet.json','phase05B_protocol.json','phase05B_review_packet.json']:
        add(root/'outputs'/f,'code/outputs/'+f)
    m=json.loads((srca/'prefix_manifest.json').read_text());idx=json.loads((srca/'artifact_index.json').read_text())
    if any(t['partition'] not in ('train','val') for t in m['traces']):raise ValueError('Test trace prohibited')
    refs={f:None for f in ['prefix_manifest.json','protocol_receipt.json','final_audit.json','artifact_index.json']}
    refs[m['framework_static']['path']]=m['framework_static']['sha256']
    for t in m['traces']:
        for ref in [*t['input_files'].values(),t['targets']]:refs[ref['path']]=ref['sha256']
        rel=f"traces/{t['trace_index']:04d}/numeric_pairs.json";refs[rel]=idx[rel]['sha256']
    for f,h in refs.items():add(srca/f,'data/05A/'+f,h)
    refs={f:None for f in ['protocol_receipt.json','final_audit.json','artifact_index.json','cache_val_receipt.json']}
    for part in ('train','val'):
        rel=f'cache/index_{part}.json';refs[rel]=None
        index=json.loads((srcb/rel).read_text())
        for ref in [*index['files'].values(),index['targets']]:refs[ref['path']]=ref['sha256']
    for kind in ('seq_gru','no_dep','rgcn'):
        for suffix in ('epoch20.pt','epoch20_train.npz'):refs[f'runs/{kind}_20261101/{suffix}']=None
    for f,h in refs.items():add(srcb/f,'data/05B/'+f,h)
    receipt={'schema':'phase05C1_portable_subset_v1','files':files,'partitions':['train','val'],
        'test_trajectories_copied':False,'product_targets_copied':False,'original_manifests_unchanged':True,
        'path_resolution':'all operational references relative; absolute provenance strings audit-only',
        'seconds':time.monotonic()-start,'bytes':sum(f['bytes'] for f in files.values())}
    (a.out/'export_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    archive=a.out.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tf:tf.add(a.out,arcname='snapshot')
    print(json.dumps({'archive':str(archive),'sha256':digest(archive),'files':len(files),'bytes':receipt['bytes'],'compressed_bytes':archive.stat().st_size,'seconds':time.monotonic()-start}))
if __name__=='__main__':main()
