"""Container-only runtime prerequisite audit, with no Java test execution."""
import argparse,hashlib,json,os,platform,resource,shutil,struct,sys,time,zipfile
from pathlib import Path
from cwm_fp.data.common import sha256,write_json
from cwm_fp.models.local_dataset import SourceBundle,load

OLD=Path('/workspace/codes/wanghaining_cwm_fp/phase05C2_v1')
ANCHOR='23e96a1bfc51a218b7ba50a1840c27a14e1d6cad9c1ad36aa2a45dbd498b0b01'

def search_java():
    bases=['/usr/lib/jvm','/usr/java','/opt/java','/opt/jdk','/usr/local/java','/opt','/usr/local','/workspace']
    out=[];found=[]
    for name in bases:
        base=Path(name);out.append({'path':name,'exists':base.exists(),'max_depth':5})
        if not base.is_dir():continue
        for d,ds,fs in os.walk(base):
            depth=len(Path(d).relative_to(base).parts)
            ds[:]=[n for n in ds if n not in {'.git','node_modules','.cache','site-packages','miniconda3','anaconda3'} and not Path(d,n).is_symlink()]
            if depth>=5:ds.clear()
            for f in fs:
                if f in ('java','javac','jar') or ('jdk' in f.lower() and 'linux' in f.lower() and f.endswith(('.gz','.xz','.zip'))):
                    found.append(str(Path(d,f)))
    return {'searched':out,'found':sorted(set(found)),
            'path_commands':{n:shutil.which(n) for n in ('java','javac','jar','mvn')},
            'permission_scope':'standard software locations and /workspace = this user mount only; no other home directories'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',required=True);a=p.parse_args()
    if not Path('/.dockerenv').exists():raise RuntimeError('Container only')
    out=a.root/a.out
    if out.exists():raise ValueError('Never overwrite runtime audit')
    out.mkdir();t0=time.monotonic();transfer=a.root/'transfer';tm=load(transfer/'transfer_manifest.json')
    for rel,rec in tm['files'].items():
        if sha256(transfer/rel)!=rec['sha256']:raise ValueError('Transfer hash mismatch '+rel)
    guide=load(transfer/'guide/receipt.json')
    if sha256(transfer/'guide/administrator_guide.lake').lower()!=guide['content_sha256'].lower():raise ValueError('Guide receipt mismatch')
    src=SourceBundle(OLD/'inputs',ANCHOR);protocol=load(transfer/'protocol.json')
    if protocol['selected']!=src.protocol['selected']:raise ValueError('Selection changed')
    build=load(transfer/'runtime/producer-build-status.json');verified=[]
    for rel,h in build['artifacts'].items():
        if sha256(transfer/'runtime'/rel)!=h:raise ValueError('Old producer artifact changed')
        verified.append(rel)
    jar=transfer/'runtime/agent.jar'
    with zipfile.ZipFile(jar) as z:
        compared=[]
        for rel,h in build['artifacts'].items():
            if rel.startswith('producer-classes/'):
                member=rel[len('producer-classes/'):]
                if hashlib.sha256(z.read(member)).hexdigest()!=h:raise ValueError('Agent/class mismatch '+member)
                compared.append(member)
        manifest=z.read('META-INF/MANIFEST.MF').decode('utf8')
    # Recover the old exporter from its existing frozen backup, not new code.
    supplement=a.root/'supplement'
    recovered={}
    if (supplement/'receipt.json').exists():
        for r in load(supplement/'receipt.json')['files']:
            if sha256(supplement/r['relative_path'])!=r['sha256']:raise ValueError('Supplement mismatch')
            recovered[r['original_path']]=r
    missing=[];source_proofs=[]
    for original,h in build['inputs']['sources'].items():
        norm=original.replace('\\','/');suffix=norm.split('/producer-src/',1)[-1]
        rel='runtime/producer-src/'+suffix if '/producer-src/' in norm else 'runtime/build_helpers/'+norm.rsplit('/',1)[-1]
        candidate=transfer/rel
        if candidate.is_file() and sha256(candidate)==h:source_proofs.append({'original':original,'sha256':h,'path':str(candidate)})
        elif original in recovered and recovered[original]['sha256']==h:source_proofs.append(recovered[original])
        else:missing.append({'name':original,'expected_sha256':h,'reason':'original build source unavailable'})
    for name,h in build['inputs']['libraries'].items():
        if sha256(transfer/'runtime/lib'/name)!=h:raise ValueError('Build dependency changed')
    classes=[];replays=[];lifecycle=[]
    for sel in src.protocol['selected']:
        static,ids,binding,idreceipt=src.product(sel)
        for original,h in binding['class_sha256'].items():
            data=src.resolve(original).read_bytes()
            if data[:4]!=b'\xca\xfe\xba\xbe':raise ValueError('Not class')
            classes.append({'original_path':original,'sha256':h,'major_version':struct.unpack('>H',data[6:8])[0]})
        for t in src.manifest['traces']:
            if t['batch_product']!=sel['batch_product']:continue
            if t['identity'][:3]!=protocol['java_replay_product'] or t['identity'][3] not in protocol['java_replay_tests']:continue
            status,results,reasons=src.execution(t,binding)
            if reasons:raise ValueError('Representative execution evidence changed')
            for original,h in status['inputs']['runtime_binding']['artifacts_sha256'].items():
                n=original.replace('\\','/')
                if '/producer-classes/' in n:path=transfer/'runtime/producer-classes'/n.split('/producer-classes/',1)[1]
                elif n.endswith('/agent.jar'):path=jar
                elif '/existing_instrumenter/lib/' in n:path=transfer/'runtime/lib'/n.rsplit('/',1)[1]
                elif original.replace('\\','/') in src.paths:path=src.resolve(original)
                else:continue # original Windows VM intentionally not executable in Linux
                if sha256(path)!=h:raise ValueError('Execution-bound artifact mismatch '+original)
            replays.append({'identity':t['identity'],'source_status':t['status'],'historical_results':results,
               'original_commands':{m:status['runs'][m]['execution'] for m in ('original','instrumented')},
               'limits':status['inputs']['limits'],'selected_classes':status['inputs']['runtime_binding']['selected_classes'],
               'linux_test_executed':False,'linux_result':None,'reason':'runtime prerequisites not yet satisfied'})
    for rel,rec in src.files.items():
        original=rec['original_path'].replace('\\','/')
        if original.endswith('Email_ESTest_scaffolding.java') and 'model_m_ca4_0011/' in original:
            lines=src.checked(rel).read_text().splitlines()
            lifecycle=[{'line':i+1,'text':s} for i,s in enumerate(lines) if any(x in s for x in ('@Before','@After','setProperty','initializeClasses','resetClasses','resetRuntime','activate','deactivate','ClassResetter','killAndJoin'))]
    j=search_java();java_candidates=[s for s in j['found'] if Path(s).name=='java']
    # Never execute a test automatically with an unqualified runtime version.
    missing_runtime=not java_candidates and not j['path_commands']['java']
    missing.append({'name':'Linux x86_64 JDK 8','required_basis':'original Oracle Java 1.8.0_381-b09 Windows; class major 52; EvoSuite runtime 1.0.6',
      'reason':'no Linux java/javac/jar found in bounded search' if missing_runtime else 'found runtime needs version and compatibility qualification',
      'user_action_required':missing_runtime,'delivery':'/workspace/codes/wanghaining_cwm_fp/phase05C3A_v1/environment/incoming',
      'checksum':'Linux archive SHA256 unknown until approved distribution supplied; Windows VM digest is not interchangeable'})
    readiness={'stage_id':'05C3A','status':'BLOCKED','guide_receipt':guide,'ssh_and_container_reused':True,
      'container_id':platform.node(),'python':sys.executable,'python_version':sys.version,'os':platform.platform(),
      'host_environment_unchanged':True,'gpu_used':False,'java_search':j,
      'JAVA_RUNTIME_READY':False,'ORIGINAL_TEST_REPLAY_VERIFIED':False,'AGENT_REBUILD_READY':False,
      'USER_ACTION_REQUIRED':missing_runtime,'java_test_processes_executed':0,'replay_status':'NOT_EXECUTED_PREREQUISITES_BLOCKED',
      'original_jdk_release':(transfer/'runtime/original_jdk_release.txt').read_text(),
      'producer_agent_sha256':sha256(jar),'agent_manifest':manifest,'agent_classes_checked':len(compared),
      'producer_build_sources':source_proofs,'producer_build_dependencies':build['inputs']['libraries'],
      'build_source_and_library_closure_available':len(source_proofs)==len(build['inputs']['sources']),
      'test_classes':classes,'representative_tests':replays,'lifecycle':lifecycle,
      'java_entry':'Phase03TestRunner: JUnitCore.run(Request.method(...)); all inherited Before/After/Class rules retained',
      'linux_differences':[
        'Windows executable/VM cannot execute in Linux; Java 8 compatibility not measured',
        'Classpath separator must be colon and paths mapped; original runner hardcodes java.exe, requires small Linux adapter after prerequisite approval',
        'Scaffolding explicitly restores java.io.tmpdir=/tmp; root filesystem has zero free bytes; TMPDIR alone cannot prove Java temporary files stay in /workspace',
        'Historical user.dir/user.home properties are virtual test inputs, not authorization to access those physical directories',
        'Historical logs contain EvoSuite class-reset warnings; PASS alone does not prove reset instrumentation completeness',
        'Original process inherited environment not exhaustively captured; do not claim exact Windows environment replication'],
      'temp_path_readiness':'NOT_VERIFIED; require lifecycle-compatible workspace temp handling before Java execution; no mount/property/source change performed',
      'missing_or_unqualified':missing,'original_missing_copy_attempt':tm['missing_or_changed'],
      'supplemental_recovery':recovered,'transfer_files_verified':len(tm['files']),
      'resources':{'seconds':time.monotonic()-t0,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'workspace_free_bytes':shutil.disk_usage(a.root).free,'root_free_bytes':shutil.disk_usage('/').free}}
    write_json(out/'runtime_readiness.json',readiness)
    write_json(out/'source_access.json',{'files':sorted(set(src.access)),'labels_read':False,'test_partition_read':False})
    print(json.dumps({'runtime_ready':False,'replay_processes':0,'build_source_closure':readiness['build_source_and_library_closure_available'],
       'java_search':j,'missing':missing},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
