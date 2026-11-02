"""Metadata-only selection and portable file packaging; no trace replay locally."""
import argparse,csv,hashlib,json,shutil,tarfile,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BATCH=Path('D:/dataset/whn/cwm_fp_generated/data/processed/lddg/phase03B2_v1')
STAGE=Path('D:/dataset/whn/cwm_fp_generated/data/processed/local_state/phase05C2_v1')
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def rank(identity):return hashlib.sha256(('05C2|20261101|'+'|'.join(identity)).encode()).hexdigest()

def select():
    if STAGE.exists():raise ValueError('New stage only')
    plan=load(BATCH/'plan.json')
    if digest(BATCH/'plan.json')!='80dd7078584edd111a23152d5efde2b27b8928f0dcc37a2cbb9f32c46ac3423d':raise ValueError('Frozen batch changed')
    splitpath=ROOT/'data/processed/splits/splits.csv'
    if digest(splitpath)!='516b6a4112694a914163dbc72c9449046b5012dcda8ea95a2025541a5433402f':raise ValueError('Split changed')
    with splitpath.open(encoding='utf8') as f:splits={tuple(r[k] for k in ('system_id','bug_version_id','product_id')):r['split'] for r in csv.DictReader(f)}
    candidates=[(i,p) for i,p in enumerate(plan['products']) if p['identity'][0]=='Email' and splits[tuple(p['identity'])]=='train']
    versions=sorted({tuple(p['identity'][:2]) for _,p in candidates},key=rank)[:3];selected=[]
    for version in versions:
        for i,p in sorted([(i,p) for i,p in candidates if tuple(p['identity'][:2])==version],key=lambda x:rank(x[1]['identity']))[:2]:
            tests=sorted(p['tests'],key=lambda t:rank(p['identity']+[t['test_id']]))[:12]
            selected.append({'identity':p['identity'],'batch_product':f'p{i:03d}','partition':'train','registered_tests':len(p['tests']),
                             'suite_id':p['suite_id'],'observed_result':p['observed_result'],'tests':tests,'selection_hash':rank(p['identity'])})
    protocol={'stage_id':'05C2','schema':'email_local_mechanism_protocol_v1','seed':20261101,'namespace':'05C2',
        'selection':'SHA256(05C2|20261101|complete identity), version then product then test; no outcome/quality/label filtering; no replacement',
        'source_batch_sha256':digest(BATCH/'plan.json'),'split_sha256':digest(splitpath),'access_partition':'grouped/train',
        'max_versions':3,'max_products_per_version':2,'max_traces_per_product':12,'max_traces':72,'selected':selected,
        'budgets':{'java_runs':0,'optimizer_updates':0,'gpu_processes':0,'experiment_seconds':7200,'rss_bytes':8*1024**3,'threads':2,
                   'research_bytes':4*1024**3,'transfer_input_bytes':8*1024**3,'min_free_bytes':10*1024**3},
        'primary_sources':['DIRECT_OBSERVED','STATIC_CONSTANT'],'derived_sources_report_separately':True,
        'semantics':'unchanged phase05C1 int32 operations and literal-null comparisons; i2l/float/reference equality unsupported',
        'trust':'monotone first unsupported identity/frame/stack/alias/control/dependency boundary; no recovery after scalar callbacks',
        'candidate_denominators':'every observed conditional, JVM arithmetic/conversion, and other product instruction; framework/test separately',
        'targets':'independent target stream; same-frame static successor for branches, later direct scalar callback for arithmetic; both sides inside trust',
        'isolation':'all admitted queries independently truncated replay; each trace first/middle/last and all call/branch/exception boundaries suffix mutation',
        'full_product_admission':'inherited false for selected Email products; not recalculated or relaxed',
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    write(STAGE/'protocol.json',protocol);write(ROOT/'outputs/phase05C2_protocol.json',protocol)
    write(STAGE/'development_exposure.json',{'stage_id':'05C2','protocol_sha256':digest(STAGE/'protocol.json'),
        'versions':[list(v) for v in versions],'reason':'Email mechanism development; cannot subsequently claim these versions as blind unseen-system evidence',
        'unexpanded_email_batch_versions':[p['identity'][:2] for p in plan['products'] if p['identity'][0]=='Email' and tuple(p['identity'][:2]) not in versions]})
    for name in ('DATA_CONTRACTS.md','STAGES.md','outputs/phase_status.md'):
        dest=STAGE/'source_before'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    print(json.dumps({'protocol_sha256':digest(STAGE/'protocol.json'),'selected':selected},ensure_ascii=False,indent=2))

def export():
    start=time.monotonic();protocol=load(STAGE/'protocol.json');out=STAGE/'transfer_v1'
    if out.exists():raise ValueError('Never overwrite export')
    out.mkdir();files={};pathmap={};runtime={};traces=[]
    def add(src,rel,expected=None):
        src=Path(src);h=digest(src)
        if expected and h!=expected:raise ValueError('Frozen artifact changed: '+str(src))
        dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
        if digest(dest)!=h:raise ValueError('Copy changed')
        files[rel]={'sha256':h,'bytes':src.stat().st_size,'original_path':str(src)};pathmap[str(src).replace('\\','/')]=rel
    add(STAGE/'protocol.json','protocol.json');add(STAGE/'development_exposure.json','development_exposure.json')
    add(ROOT/'data/processed/splits/splits.csv','metadata/splits.csv',protocol['split_sha256'])
    add(BATCH/'plan.json','metadata/source_plan.json',protocol['source_batch_sha256'])
    for p in protocol['selected']:
        work=BATCH/p['batch_product'];base='email/'+p['batch_product'];coll=load(work/'collection.json');recs={r['test_id']:r for r in coll['records']}
        # Collection carries public identities/outcomes, not trace suffixes or labels.
        for name in ('collection.json','prepare.json','compilation_binding.json','source_evidence.json','source_ast.jsonl','g/source_map.jsonl'):
            add(work/name,base+'/'+name)
        prep=load(work/'prepare.json')
        for rel,h in prep['artifacts'].items():add(work/rel,base+'/'+rel,h)
        binding=load(work/'compilation_binding.json')
        for src,h in binding['source_sha256'].items():
            rel='sources/'+hashlib.sha256(src.encode()).hexdigest()[:16]+'/'+Path(src).name
            add(src,rel,h)
        for t in p['tests']:
            rec=recs[t['test_id']];run=Path(rec['directory']);rel=base+'/runs/'+run.name;status=load(run/'status.json')
            if status!=rec['status']:raise ValueError('Collection/status disagreement')
            add(run/'status.json',rel+'/status.json')
            for f,h in status['artifacts_sha256'].items():add(run/f,rel+'/'+f,h)
            for name,h in status['inputs']['runtime_binding']['artifacts_sha256'].items():
                if name not in runtime:
                    if digest(name)!=h:raise ValueError('Runtime binding changed: '+name)
                    runtime[name]={'sha256':h,'verified_locally_during_file_export':True,'copied':str(Path(name)).replace('\\','/') in pathmap}
            traces.append({'identity':p['identity']+[t['test_id']],'partition':'train','batch_product':p['batch_product'],
                           'run':rel,'status':rel+'/status.json','expected_result':t['result'],'registered_tests':p['registered_tests']})
    add(ROOT/'data/processed/lddg/phase03B4_framework_v1/framework/static.tsv','framework/static.tsv')
    for name in ('phase05C1_review_packet.json','phase05C1_semantic_checks.json','phase05C1_local_state_status.csv'):
        add(ROOT/'outputs'/name,'bank_reference/'+name)
    # No Bank trajectory is transported: compare to frozen diagnostic rows only on the server.
    write(out/'source_manifest.json',{'schema':'email_local_source_v1','protocol_sha256':digest(out/'protocol.json'),
        'files':files,'path_map':pathmap,'traces':traces,'runtime_binding_local_receipt':runtime,
        'runtime_policy':'class bytes and source copied; old Java runtime/JAR hashes verified during export, no Java execution here',
        'bytes':sum(x['bytes'] for x in files.values()),'seconds':time.monotonic()-start})
    if sum(x['bytes'] for x in files.values())>8*1024**3:raise RuntimeError('Input transfer budget')
    archive=STAGE/'inputs_v1.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:tf.add(out,arcname='inputs')
    write(STAGE/'transfer_receipt.json',{'archive':str(archive),'sha256':digest(archive),'bytes':archive.stat().st_size,
        'source_manifest_sha256':digest(out/'source_manifest.json'),'files':len(files),'uncompressed_bytes':sum(x['bytes'] for x in files.values()),'seconds':time.monotonic()-start})
    print((STAGE/'transfer_receipt.json').read_text())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['select','export']);a=p.parse_args()
    select() if a.action=='select' else export()
