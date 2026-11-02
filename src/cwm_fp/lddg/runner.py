"""Reuse the audited BRFL producer in a bounded copy; original trees stay read-only."""
from __future__ import annotations
import argparse
import difflib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from cwm_fp.data.common import ROOT, KEY, indexed, key, read_csv, sha256, write_json

JAVA = Path(__file__).parent / "java"


def tree_usage(root):
    files=[p for p in Path(root).rglob('*') if p.is_file()]
    return {'bytes':sum(p.stat().st_size for p in files),'files':len(files)}


def budget_check(policy, reserve_bytes=0):
    """A polled safety bound, not a filesystem quota; never delete old results."""
    root=Path(policy['root']);usage=tree_usage(root)
    usage['free_bytes']=shutil.disk_usage(root).free
    reason=None
    if usage['bytes']+reserve_bytes>policy['max_total_bytes']:reason='collection_byte_budget'
    elif usage['files']>=policy['max_total_files']:reason='collection_file_budget'
    elif usage['free_bytes']-reserve_bytes<policy['min_free_bytes']:reason='minimum_free_disk'
    return reason,usage


def command(args, cwd, log, seconds=180, max_bytes=33554432, monitor_files=False,
            *, max_files=None, budget=None, max_rss_bytes=None):
    """Poll bounded child; no shell; never infer test outcome from exit status."""
    cwd.mkdir(parents=True, exist_ok=True)
    start=time.monotonic(); reason=None;peak_rss=0;peak_os=0;samples=0;disk_samples=[]
    try:
        import psutil
    except ImportError:
        psutil=None
    if max_rss_bytes and psutil is None:raise RuntimeError('RSS enforcement requires psutil')
    def limits():
        nonlocal reason,peak_rss,peak_os,samples
        if time.monotonic()-start>seconds:reason='timeout'
        elif log.stat().st_size>max_bytes:reason='output_limit'
        elif monitor_files:
            usage=tree_usage(cwd)
            if usage['bytes']>max_bytes:reason='file_size_limit'
            elif max_files is not None and usage['files']>max_files:reason='file_count_limit'
        if psutil and process.poll() is None:
            try:
                mem=psutil.Process(process.pid).memory_info();samples+=1
                peak_rss=max(peak_rss,mem.rss);peak_os=max(peak_os,getattr(mem,'peak_wset',0))
                if max_rss_bytes and mem.rss>max_rss_bytes:reason='process_memory_limit'
            except psutil.Error:pass
        if budget:
            why,sample=budget_check(budget);disk_samples.append(sample)
            reason=reason or why
    if budget:
        reason,sample=budget_check(budget,reserve_bytes=max_bytes);disk_samples.append(sample)
    process=None
    with log.open("wb") as out:
        if not reason:
            process=subprocess.Popen([str(x) for x in args], cwd=cwd, stdout=out, stderr=subprocess.STDOUT)
            while process.poll() is None:
                limits()
                if reason:
                    if psutil:
                        try:
                            for child in psutil.Process(process.pid).children(recursive=True):child.kill()
                        except psutil.Error:pass
                    process.kill();break
                time.sleep(.05)
            code=process.wait(timeout=5)
            if reason is None:limits()  # Fast exits cannot bypass disk/file checks.
        else:code=None
    record={"argv":[str(x) for x in args],"cwd":str(cwd),"exit_code":code,"reason":reason,
            "seconds":round(time.monotonic()-start,3),"log_sha256":sha256(log),
            'resource':{'memory_samples':samples,'peak_sampled_rss_bytes':peak_rss or None,
                        'peak_os_working_set_bytes':peak_os or None,'memory_scope':'direct_child_process',
                        'poll_seconds':.05,'launched':process is not None,
                        'min_free_bytes':min((s['free_bytes'] for s in disk_samples),default=None),
                        'max_collection_bytes':max((s['bytes'] for s in disk_samples),default=None),
                        'max_collection_files':max((s['files'] for s in disk_samples),default=None)}}
    write_json(log.with_suffix(".command.json"),record)
    return record


def runtime_binding(c, producer, classes, agent, build, selected_classes):
    java=Path(c['java_home'])
    paths=[agent,java/'bin/java.exe',java/'bin/server/jvm.dll',java/'jre/bin/server/jvm.dll',java/'jre/lib/rt.jar']
    paths += sorted(build.rglob('*.class'))+sorted(classes.rglob('*.class'))+sorted((producer/'lib').glob('*.jar'))
    return {'artifacts_sha256':{str(p.resolve()):sha256(p) for p in paths if p.is_file()},
            'selected_classes':sorted(selected_classes),'resource_policy':c.get('resource_policy'),
            'max_files_per_run':c.get('max_files_per_run'),'max_rss_bytes':c.get('max_rss_bytes'),
            'runner_sha256':sha256(Path(__file__))}


def checked(args, cwd, log):
    record=command(args,cwd,log)
    if record["exit_code"] or record["reason"]:raise RuntimeError(f"Bounded command failed: {log}")


def snapshot(source:Path):
    target=ROOT/"third_party/existing_instrumenter"
    paths=sorted((source/"src").rglob("*"))+sorted((source/"lib").glob("*.jar"))
    hashes={}
    for p in paths:
        if not p.is_file():continue
        rel=p.relative_to(source); dest=target/rel
        if dest.exists() and sha256(dest)!=sha256(p):raise ValueError(f"Producer snapshot changed: {rel}")
        if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        hashes[rel.as_posix()]=sha256(p)
    write_json(ROOT/"outputs/phase03_producer_provenance.json",{"source_root":str(source),"files_sha256":hashes})
    return target


def build_producer(c, work):
    source=snapshot(Path(c["producer_root"])); patched=work/"producer-src"
    # No deletion and no modifications to the source snapshot.
    for p in (source/"src").rglob("*"):
        if p.is_file():
            dest=patched/p.relative_to(source/"src");dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    callback=patched/"main/java/ppfl/instrumentation/CallBackIndex.java"
    original=callback.read_text(encoding="utf-8");changed=original
    changed=changed.replace("static int loglimit = 1200000;",'static int loglimit = Integer.getInteger("cwm.maxEvents", 20000);')
    changed=changed.replace("System.exit(0);",'System.err.println("CWM_TRUNCATED_EVENT_LIMIT");\n            System.err.flush();\n            Runtime.getRuntime().halt(86);')
    changed=changed.replace("        writeLogString(s);",'        if (s.trim().startsWith("opcode=")) s += ",thread=" + Thread.currentThread().getId();\n        writeLogString(s);')
    if changed==original or "System.exit(0);" in changed:raise ValueError("Audited producer patch no longer matches")
    callback.write_text(changed,encoding="utf-8")
    patch="".join(difflib.unified_diff(original.splitlines(True),changed.splitlines(True),fromfile="original/CallBackIndex.java",tofile="bounded/CallBackIndex.java"))
    (ROOT/"compatibility/phase03_callback.diff").write_text(patch,encoding="utf-8")
    java=Path(c["java_home"])/"bin";classes=work/"producer-classes";classes.mkdir(parents=True,exist_ok=True)
    # Source-only parsing is a separate JDK-tools utility, never part of the agent.
    sources=sorted((patched/"main/java").rglob("*.java"))+sorted(p for p in JAVA.glob("*.java") if p.name!='Phase03SourceExport.java')
    response=work/"producer-sources.txt"
    response.write_text("\n".join('"'+Path(os.path.relpath(p,work)).as_posix()+'"' for p in sources),encoding="ascii")
    agent=work/"agent.jar";stamp=work/'producer-build-status.json'
    inputs={'sources':{str(p):sha256(p) for p in sources},'libraries':{p.name:sha256(p) for p in sorted((source/'lib').glob('*.jar'))},
            'javac_sha256':sha256(java/'javac.exe')}
    if stamp.exists():
        old=json.loads(stamp.read_text(encoding='utf-8'))
        if old['inputs']==inputs:
            for name,h in old['artifacts'].items():
                if sha256(work/name)!=h:raise ValueError('Producer build artifact changed')
            return source,classes,agent
    checked([java/"javac.exe","-g","-encoding","UTF-8","-cp",source/"lib/*","-d",classes,"@"+str(response)],work,work/"producer-build.log")
    checked([java/"jar.exe","cfm",agent,source/"src/main/resources/META-INF/MANIFEST.MF","-C",classes,"."],work,work/"producer-jar.log")
    write_json(stamp,{'inputs':inputs,'artifacts':{p.relative_to(work).as_posix():sha256(p) for p in [agent,*sorted(classes.rglob('*.class'))]}})
    return source,classes,agent


def completion(path:Path, identity:str):
    if not path.exists():return {"outcome":"UNKNOWN","reason":"missing_completion"}
    lines=path.read_text(encoding="utf-8").splitlines()
    ends=[x.split("\t") for x in lines if x.startswith("COMPLETE\t")]
    if len(ends)!=1 or ends[0][1]!=identity or ends[0][2]!="1" or ends[0][4]!="0":
        return {"outcome":"UNKNOWN","reason":"invalid_completion"}
    failures=[x for x in lines if x.startswith("FAILURE\t")]
    n=int(ends[0][3])
    if n!=len(failures):return {"outcome":"UNKNOWN","reason":"incomplete_failures"}
    return {"outcome":"FAIL" if n else "PASS","failures":failures,"run_count":1}


def execute_pair(c, work, producer, classes, agent, variant, identity, selected_classes, build, run_dir, expected):
    java=Path(c["java_home"])/"bin/java.exe";test_class,method=identity.rsplit(".",1)
    cp=os.pathsep.join(map(str,[build/"main",build/"tests",classes,producer/"lib/*"]))
    records={}
    inputs={"identity":identity,"expected":expected,"agent_sha256":sha256(agent),
            "limits":{k:c[k] for k in ('max_seconds_per_test','max_events','max_output_bytes','java_heap_mb')},
            "sources":{str(p):sha256(p) for folder in ("src","test") for p in sorted((variant/folder).rglob("*.java"))}}
    if c.get('strict_runtime_binding'):
        inputs['runtime_binding']=runtime_binding(c,producer,classes,agent,build,selected_classes+[test_class])
    status_path=run_dir/"status.json"
    if status_path.exists():
        cached=json.loads(status_path.read_text(encoding="utf-8"))
        if cached["inputs"]==inputs and cached.get("complete"):
            if c.get('strict_runtime_binding'):
                actual={p.relative_to(run_dir).as_posix() for p in run_dir.rglob('*') if p.is_file() and p!=status_path}
                if actual!=set(cached['artifacts_sha256']):raise ValueError('Resume artifact set changed')
            for name,hash_value in cached["artifacts_sha256"].items():
                if sha256(run_dir/name)!=hash_value:raise ValueError("Resume artifact changed")
            return cached
        raise ValueError(f"Existing incomplete/different run; declare a new run directory: {run_dir}")
    for mode in ("original","instrumented"):
        stage=run_dir/mode;stage.mkdir(parents=True,exist_ok=True)
        if c.get('strict_runtime_binding') and any(stage.iterdir()):
            raise ValueError(f'Unsealed prior execution; choose new run ID: {stage}')
        result=stage/"result.tsv"
        if result.exists():raise ValueError(f"Unsealed prior execution; choose new run ID: {stage}")
        args=[java,"-Xmx"+str(c["java_heap_mb"])+"m","-noverify"]
        if mode=="instrumented":
            args += ["-Dcwm.maxEvents="+str(c["max_events"]),f"-javaagent:{agent}=instrumentingclass={':'.join(selected_classes+[test_class])},logfile=single.log,project=phase03"]
        args += ["-cp",cp,"Phase03TestRunner",test_class,method,result]
        rec=command(args,stage,stage/"run.log",c["max_seconds_per_test"],c["max_output_bytes"],monitor_files=True,
                    max_files=c.get('max_files_per_run'),budget=c.get('resource_policy'),max_rss_bytes=c.get('max_rss_bytes'))
        outcome=completion(result,identity)
        text=(stage/"run.log").read_text(encoding="utf-8",errors="replace")
        if rec["reason"] or rec["exit_code"] not in (0,1) or "CWM_TRUNCATED" in text:
            outcome={"outcome":"UNKNOWN","reason":rec["reason"] or "abnormal_exit"}
        if "[Agent] bytecode transform failed" in text:
            outcome={"outcome":"UNKNOWN","reason":"producer_transform_failure"}
        records[mode]={"execution":rec,"result":outcome}
    a,b=[records[x]["result"] for x in ("original","instrumented")]
    parity=a==b and a["outcome"] in {"PASS","FAIL"}
    artifacts={p.relative_to(run_dir).as_posix():sha256(p) for p in sorted(run_dir.rglob("*")) if p.is_file()}
    record={"inputs":inputs,"runs":records,"parity":parity,"matches_historical_outcome":a["outcome"]==expected,
            "complete":all(x["result"]["outcome"]!="UNKNOWN" for x in records.values()),"artifacts_sha256":artifacts}
    write_json(status_path,record);return record


def collect(c):
    work=(ROOT/c["work_dir"]).resolve()
    if not work.is_relative_to(ROOT/"data/processed/lddg"):raise ValueError("Unsafe output root")
    work.mkdir(parents=True,exist_ok=True)
    if len(c["tests"])>c["max_tests"]:raise ValueError("Test budget exceeded")
    producer,classes,agent=build_producer(c,work)
    from cwm_fp.baselines.boundaries import install_io_guard
    install_io_guard('features')
    products=indexed(read_csv(ROOT/"data/processed/manifests/products.csv",KEY))
    tests={(*key(r),r["test_id"]):r for r in read_csv(ROOT/"data/processed/manifests/tests.csv",(*KEY,"test_id"))}
    records=[];builds={}
    for product,identity in c["tests"]:
        k=(c["system_id"],c["bug_version_id"],product);row=products[k];test=tests[(*k,identity)]
        if row["suite_view"]!="original":raise ValueError("Only original suites")
        variant=Path(row["source_variant_path"]);dest=work/product;dest.mkdir(exist_ok=True)
        if product not in builds:
            build=dest/"build";java=Path(c["java_home"])/"bin";names=[]
            for part,folder in (("main","src"),("tests","test")):
                output=build/part;output.mkdir(parents=True,exist_ok=True)
                sources=sorted((variant/folder).rglob("*.java"))
                cp=os.pathsep.join(map(str,[build/"main",producer/"lib/*"]))
                checked([java/"javac.exe","-g","-encoding","UTF-8","-cp",cp,"-d",output,*sources],dest,dest/(part+"-build.log"))
            classfiles=sorted((build/"main").rglob("*.class"))
            names=[p.relative_to(build/"main").as_posix()[:-6].replace("/",".") for p in classfiles]
            cp=os.pathsep.join(map(str,[classes,producer/"lib/*"]))
            checked([java/"java.exe","-cp",cp,"Phase03StaticExport",dest/"static.tsv",*sorted(build.rglob('*.class'))],dest,dest/"static-export.log")
            builds[product]=(build,names)
        build,names=builds[product]
        record=execute_pair(c,work,producer,classes,agent,variant,identity,names,build,dest/identity,test["result"])
        records.append({"system_id":k[0],"bug_version_id":k[1],"product_id":product,"test_id":identity,
                        "variant":str(variant),"directory":str(dest/identity),"status":record})
        print(f"{product}/{identity}: parity={record['parity']}, historical={record['matches_historical_outcome']}",flush=True)
    write_json(work/"collection.json",{"records":records,"config":c})


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=ROOT/"configs/lddg_smoke.json")
    a=p.parse_args();collect(json.loads(a.config.read_text(encoding="utf-8")))

if __name__=="__main__":main()
