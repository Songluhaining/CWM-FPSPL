"""Bounded four-system material preparation and isolated benchmark oracle export.

The oracle action is never imported by features or fitting. Original failures
stay F; base-feature and single-class versions are retained. No Java test is run.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

from cwm_fp.data.benchmark_adapter import load_sources, scan, PRODUCT_FIELDS, TEST_FIELDS
from cwm_fp.data.common import ROOT, KEY, digest, indexed, key, read_csv, sha256, write_csv, write_json
from cwm_fp.data.labels import FIELDS, export_partition_labels
from cwm_fp.data.splits import build_all
from .vendor import pure_functions, VENDOR

BASE_FEATURES = {"BankAccount", "Base", "ExamDB", "Compress"}


def slicing_input(raw):
    """Exact read closure of inspected FileUtil/StatementContainer bytecode.

    Source bytes plus SET of count>0 feature lines: the JAR does not use the
    aggregate XML's timestamps, buggy attributes, count magnitude or roles.meta.
    System dependencies and JAR hashes are added by the caller.
    """
    source = {p.relative_to(raw/'src').as_posix():sha256(p)
              for p in (raw/'src').rglob('*') if p.is_file()}
    executed = set()
    aggregates = {}
    for name in ['failed','passed']:
        path = raw/'coverage'/f'spectrum_{name}_coverage.xml'
        if path.exists():
            aggregates[name] = sha256(path)
            for line in ET.parse(path).iter('line'):
                if int(line.get('count', '0')) > 0:
                    executed.add(line.attrib['featureClass']+'.'+line.attrib['featureLineNum'])
    return {'source':source,'executed':sorted(executed)}, aggregates


def slices_cached(config):
    import psutil
    products = read_csv(ROOT/config['products'], KEY)
    sources = {s.system_id:s for s in load_sources(ROOT/config['sources'])}
    output = ROOT/config['work_dir']/'slicing_cached'
    output.mkdir(parents=True, exist_ok=True)
    java, jar = Path(config['java']), ROOT/config['slicer']
    jdk = java.parent.parent
    bindings = {str(p):sha256(p) for p in [java,jar,jdk/'jre/lib/rt.jar',jdk/'jre/lib/jce.jar']}
    groups = {}
    for p in products:
        if p['system_id'] not in config['new_slicing_systems']: continue
        raw = Path(p['source_variant_path'])
        closure, aggregates = slicing_input(raw)
        signature = digest([p['system_id'],closure,bindings])
        groups.setdefault(signature, []).append((p,closure,aggregates))
    plan = {'bindings':bindings,'jobs':{h:[key(p) for p,_,_ in group] for h,group in groups.items()},
        'workers':config['slicing_workers'], 'budget_seconds':config['bounds']['slicing_seconds'],
        'closure':'all source bytes + actual count-positive feature-line set; bytecode-audited'}
    plan_path = output/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text(encoding='utf-8')) != json.loads(json.dumps(plan)):
        raise ValueError('Frozen slicing plan changed')
    write_json(plan_path, plan)
    started = time.monotonic()
    print(f"frozen slicing jobs={len(groups)}, products={sum(map(len,groups.values()))}, workers={config['slicing_workers']}",flush=True)

    def run_job(signature, group):
        dest = output/signature
        receipt = dest/'receipt.json'
        if receipt.exists():
            result = json.loads(receipt.read_text(encoding='utf-8'))
            if result['exit_code'] != 0: raise RuntimeError('Failed attempt retained; new attempt required')
            for path, expected in result['outputs'].items():
                if sha256(Path(path)) != expected: raise ValueError('Cache fingerprint mismatch')
            return result | {'cache_reused':True}
        if time.monotonic()-started >= config['bounds']['slicing_seconds']: raise TimeoutError('Slicing budget exhausted')
        p, closure, aggregate_hashes = group[0]
        raw = Path(p['source_variant_path'])
        isolated = dest/'variants'/p['product_id']
        isolated.mkdir(parents=True, exist_ok=False)
        shutil.copytree(raw/'src', isolated/'src')
        (isolated/'coverage').mkdir()
        for name in aggregate_hashes:
            file=f'spectrum_{name}_coverage.xml'
            shutil.copyfile(raw/'coverage'/file,isolated/'coverage'/file)
        command=[str(java),'-Xmx512m','-XX:ParallelGCThreads=1','-XX:ConcGCThreads=1',
            '-Dvariants_dir='+(dest/'variants').as_posix(),'-jar',str(jar)]
        tick=time.monotonic(); peak=0; killed=None
        if shutil.disk_usage(dest).free < config['bounds']['minimum_free_bytes']: raise OSError('Disk reserve reached')
        with (dest/'java.log').open('xb') as log:
            process=subprocess.Popen(command,cwd=dest,stdout=log,stderr=subprocess.STDOUT)
            monitor=psutil.Process(process.pid)
            while process.poll() is None:
                try: peak=max(peak, monitor.memory_info().rss)
                except psutil.NoSuchProcess: pass
                if time.monotonic()-tick>config['bounds']['slicing_product_seconds'] or peak>1024**3:
                    killed='timeout' if peak<=1024**3 else 'memory';process.kill();process.wait();break
                time.sleep(.2)
        result={'signature':signature,'exit_code':process.returncode,'killed':killed,'seconds':time.monotonic()-tick,
            'peak_rss_bytes':peak,'command':command,'source_identity':key(p),'closure':closure,
            'aggregate_file_hashes':aggregate_hashes,'bindings':bindings,'outputs':{}}
        for direction in ['forward','backward']:
            file=isolated/f'slice.all_statements.{direction}_slicing.log'
            if file.exists():
                json.loads(file.read_text(encoding='utf-8-sig'))
                result['outputs'][str(file)]=sha256(file)
        write_json(receipt,result)
        if result['exit_code'] or len(result['outputs'])!=2: raise RuntimeError(f'Slicing job failed: {receipt}')
        return result

    receipts=[]
    with ThreadPoolExecutor(max_workers=config['slicing_workers']) as pool:
        jobs={pool.submit(run_job,h,g):(h,g) for h,g in groups.items()}
        for future in as_completed(jobs):
            h,group=jobs[future];result=future.result();receipts.append(result)
            for p,closure,aggregates in group:
                raw=Path(p['source_variant_path']); material=sources[p['system_id']].label_root/p['bug_version_id']/'variants'/p['product_id']
                if not (material/'src').exists(): shutil.copytree(raw/'src',material/'src')
                # Check source byte identity again before using any cached slice.
                if slicing_input(raw)[0]!=closure: raise ValueError('Raw slicing input changed')
                for file,expected in result['outputs'].items():
                    target=material/Path(file).name
                    if target.exists() and sha256(target)!=expected: raise ValueError('Existing output changed')
                    if not target.exists(): shutil.copyfile(file,target)
                write_json(material/'slice_provenance.json', {'job_receipt':str(output/h/'receipt.json'),
                    'signature':h,'input_identity':key(p),'aggregate_hashes':aggregates})
            write_json(output/'progress.json', {'completed_jobs':len(receipts),'total_jobs':len(groups),
                'completed_products':sum(len(groups[r['signature']]) for r in receipts),
                'wall_seconds':time.monotonic()-started,'aggregate_job_seconds':sum(r['seconds'] for r in receipts)})
            print(f"sliced {len(receipts)}/{len(groups)}, restored_products={len(group)}, seconds={result['seconds']:.2f}",flush=True)
    write_json(output/'summary.json', {'jobs':len(groups),'products':sum(map(len,groups.values())),
        'wall_seconds':time.monotonic()-started,'job_seconds':sum(r['seconds'] for r in receipts),
        'maximum_single_process_rss':max(r['peak_rss_bytes'] for r in receipts),'successful':True})


def cache_verify(config):
    """Independently rerun one equivalent but different raw product per system."""
    base=ROOT/config['work_dir']/'slicing_cached'
    plan=json.loads((base/'plan.json').read_text(encoding='utf-8'))
    products=indexed(read_csv(ROOT/config['products'],KEY))
    results=[]
    for system in config['new_slicing_systems']:
        h,members=next((h,g) for h,g in sorted(plan['jobs'].items())
                      if g[0][0]==system and len(g)>1 and (base/h/'receipt.json').exists())
        original=json.loads((base/h/'receipt.json').read_text(encoding='utf-8'))
        if original['exit_code']!=0: raise ValueError('Original slicing failed')
        p=products[tuple(members[1])];raw=Path(p['source_variant_path'])
        folder=base/'independent_validation'/system
        variant=folder/'variants'/p['product_id']
        variant.mkdir(parents=True,exist_ok=False)
        shutil.copytree(raw/'src',variant/'src');(variant/'coverage').mkdir()
        closure,aggregates=slicing_input(raw)
        if closure!=original['closure']: raise ValueError('Cache input closure mismatch')
        for name in aggregates:
            file=f'spectrum_{name}_coverage.xml';shutil.copyfile(raw/'coverage'/file,variant/'coverage'/file)
        command=[config['java'],'-Xmx512m','-Dvariants_dir='+(folder/'variants').as_posix(),'-jar',str(ROOT/config['slicer'])]
        tick=time.monotonic()
        with (folder/'java.log').open('xb') as log:
            code=subprocess.run(command,cwd=folder,stdout=log,stderr=subprocess.STDOUT,timeout=120).returncode
        comparisons=[]
        for source in original['outputs']:
            a=json.loads(Path(source).read_text(encoding='utf-8-sig'))
            file=variant/Path(source).name;b=json.loads(file.read_text(encoding='utf-8-sig'))
            normalize=lambda d:{k:sorted(set(v)) for k,v in d.items()}
            comparisons.append({'file':file.name,'semantic_equal':normalize(a)==normalize(b),
                'original_sha256':sha256(Path(source)),'replica_sha256':sha256(file)})
        results.append({'system':system,'cache_key':h,'original_identity':members[0],'replica_identity':members[1],
            'exit_code':code,'command':command,'seconds':time.monotonic()-tick,'comparisons':comparisons})
        write_json(base/'independent_validation.json',results)
        if code or not all(x['semantic_equal'] for x in comparisons): raise ValueError('Independent slice mismatch')
    print('independent cache comparison passed for both systems')


def source_target(observed, features, buggy_statements, present):
    """PULP membership oracle for ORIGINAL suites, including base features.

    PULP read_info checks presence in the aggregate XML, not execution counts.
    It must not be silently described as a newly verified failure oracle.
    """
    if observed == "FAIL":
        return "F"
    if observed != "PASS":
        raise ValueError("Unknown observed result")
    candidates = [s for s in buggy_statements
                  if s.split('.', 1)[0] in BASE_FEATURES or features.get(s.split('.', 1)[0], False)]
    return "FP" if any(s in present for s in candidates) else "TP"


def manifest(config):
    out = ROOT / config["manifest_dir"]
    if out.exists():
        raise ValueError("Manifest output already exists")
    products, tests, audit = scan(load_sources(ROOT / config["sources"]),
        max_versions=config["bounds"]["max_versions"], max_products=config["bounds"]["max_products"])
    if len(tests) > config["bounds"]["max_tests"]:
        raise ValueError("Test inventory bound exceeded")
    write_csv(out / "products.csv", products, PRODUCT_FIELDS)
    write_csv(out / "tests.csv", tests, TEST_FIELDS)
    write_json(out / "manifest_audit.json", audit)
    split_audit = build_all(out / "products.csv", out / "tests.csv", ROOT / config["split_dir"], config["split_seed"],
                            source_train_fraction=config.get('source_train_fraction'))
    print(json.dumps({"versions": audit["versions"], "products": audit["products"],
        "tests": audit["tests"], "systems": audit["systems"], "splits": list(split_audit["folds"])}, ensure_ascii=False))


def slices(config):
    products = read_csv(ROOT / config["products"], KEY)
    sources = {s.system_id: s for s in load_sources(ROOT / config["sources"])}
    selected = config["new_slicing_systems"]
    output = ROOT / config["work_dir"] / "slicing"
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / "receipt.json"
    record = json.loads(receipt.read_text()) if receipt.exists() else {"completed": {}, "failures": [], "elapsed_seconds": 0}
    java, jar = Path(config["java"]), ROOT / config["slicer"]
    budget = config["bounds"]["slicing_seconds"]
    for system, version in sorted({key(p)[:2] for p in products if p['system_id'] in selected}):
        label = system + '/' + version
        if label in record['completed']:
            for file, expected in record['completed'][label]['outputs'].items():
                if sha256(Path(file)) != expected:
                    raise ValueError("Completed slicing output changed")
            continue
        if record['elapsed_seconds'] >= budget:
            raise TimeoutError("Cumulative static slicing budget exhausted")
        source = sources[system]
        target = source.label_root / version
        # Only source, aggregate coverage and roles are required by the JAR.
        for p in (p for p in products if key(p)[:2] == (system, version)):
            raw, dest = Path(p['source_variant_path']), target / 'variants' / p['product_id']
            for file in sorted((raw / 'src').rglob('*')):
                if file.is_file():
                    copied = dest / file.relative_to(raw)
                    copied.parent.mkdir(parents=True, exist_ok=True)
                    if copied.exists() and sha256(file) != sha256(copied):
                        raise ValueError("Source copy changed")
                    if not copied.exists():
                        shutil.copyfile(file, copied)
            for rel in ['roles.meta', 'coverage/spectrum_passed_coverage.xml', 'coverage/spectrum_failed_coverage.xml']:
                if (raw / rel).is_file():
                    copied = dest / rel
                    copied.parent.mkdir(parents=True, exist_ok=True)
                    if not copied.exists():
                        shutil.copyfile(raw / rel, copied)
        if shutil.disk_usage(target).free < config['bounds']['minimum_free_bytes']:
            raise OSError("Minimum disk reserve reached")
        attempt = len(record['completed']) + len(record['failures'])
        log = output / f'{attempt:03d}_{system}_{version}.log'
        command = [str(java), '-Xmx512m', '-Dvariants_dir=' + (target/'variants').as_posix(), '-jar', str(jar)]
        started = time.monotonic()
        with log.open('xb') as stream:
            try:
                result = subprocess.run(command, cwd=output, stdout=stream, stderr=subprocess.STDOUT,
                    timeout=min(config['bounds']['slicing_version_seconds'], budget-record['elapsed_seconds']))
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
        elapsed = time.monotonic()-started
        record['elapsed_seconds'] += elapsed
        outputs = {}
        missing = []
        for p in (p for p in products if key(p)[:2] == (system, version)):
            for name in ['forward', 'backward']:
                file = target/'variants'/p['product_id']/f'slice.all_statements.{name}_slicing.log'
                if not file.exists():
                    missing.append(str(file))
                else:
                    json.loads(file.read_text(encoding='utf-8-sig'))
                    outputs[str(file)] = sha256(file)
        entry = {'command': command, 'exit_code': code, 'seconds': elapsed, 'log': str(log),
                 'outputs': outputs, 'missing': missing, 'jar_sha256': sha256(jar), 'java_sha256': sha256(java)}
        if code or missing:
            record['failures'].append(entry | {'version': label})
        else:
            record['completed'][label] = entry
        write_json(receipt, record)
        print(f"slicing {label}: exit={code}, outputs={len(outputs)}, missing={len(missing)}, seconds={elapsed:.2f}", flush=True)
        if code or missing:
            raise RuntimeError("Static slicing failed; evidence retained")


def oracle(config):
    products = indexed(read_csv(ROOT / config['products'], KEY))
    out = ROOT / config['labels']
    if out.exists():
        raise ValueError("Sealed label export already exists")
    source_file = VENDOR/'PULP/suspicious_statements_manager/SuspiciousStatementManager.py'
    funcs = pure_functions(source_file, ['get_multiple_buggy_statements'],
        {'join_path': lambda *p: str(Path(*p))})
    sources = {s.system_id: s for s in load_sources(ROOT / config['sources'])}
    rows, audits, disagreements = [], [], []
    matched_existing = 0
    for group in sorted({k[:2] for k in products}):
        members = [p for k, p in products.items() if k[:2] == group]
        raw_version = Path(members[0]['source_variant_path']).parent.parent
        statements = funcs['get_multiple_buggy_statements'](group[1], str(raw_version))
        if len(statements) != 1:
            raise ValueError("Only single-fault versions authorized")
        mutant = raw_version / (group[1]+'.mutant.log')
        old_path = sources[group[0]].label_root / group[1] / 'variant_labels.csv'
        old = {r['VARIANT']:r for r in read_csv(old_path)} if old_path.exists() else {}
        labels = Counter()
        for p in members:
            present = set()
            if p['observed_result'] == 'PASS':
                path = Path(p['source_variant_path'])/'coverage/spectrum_passed_coverage.xml'
                for line in ET.parse(path).iter('line'):
                    present.add(line.attrib['featureClass']+'.'+line.attrib['featureLineNum'])
            target = source_target(p['observed_result'], json.loads(p['features_json']), statements, present)
            if p['product_id'] in old:
                expected = 'F' if p['observed_result']=='FAIL' else old[p['product_id']]['LABEL']
                if target != expected:
                    disagreements.append(dict(zip(KEY,key(p))) | {'new':target,'old_original_view':expected})
                else:
                    matched_existing += 1
            labels[target] += 1
            row = {f: '' for f in FIELDS}
            row.update({**dict(zip(KEY,key(p))), 'label':target, 'origin':'natural' if target=='FP' else 'unknown',
                'label_source':'pulp_original_suite_all_features_v1', 'label_source_path':str(mutant),
                'label_source_sha256':sha256(mutant), 'source_label':target, 'source_origin':'natural' if target=='FP' else 'unknown',
                'decision':'original_F_preserved' if target=='F' else 'source_membership_rule_no_base_exclusion',
                'label_provenance':'PULP membership rule; not a newly demonstrated failure',
                'suite_id':p['suite_id'],'suite_view':'original',
                'quality':'observed_failure' if target=='F' else 'benchmark_fault_membership_oracle'})
            rows.append(row)
        audits.append({'system_id':group[0], 'bug_version_id':group[1], 'products':len(members),
            'base_feature_fault':any(s.split('.')[0] in BASE_FEATURES for s in statements),
            'excluded':False, 'label_counts':dict(labels), 'mutant_log_sha256':sha256(mutant)})
    if disagreements:
        write_json(out/'disagreements.json', disagreements)
        raise ValueError('Existing label mismatch; no training export produced')
    write_csv(out/'labels.csv', rows, FIELDS)
    exports = export_partition_labels(rows, ROOT/config['products'],
        [ROOT/config['split_dir']/('splits.csv' if f=='grouped' else f+'.csv') for f in config['folds']], out)
    write_json(out/'label_audit.json', {'versions':audits, 'partition_files':exports,
        'source_parser_sha256':sha256(source_file), 'all_original_failures_retained':True,
        'matched_existing_original_view_products':matched_existing, 'disagreements':disagreements,
        'base_feature_exclusion':False, 'single_class_exclusion':False,
        'rule':'PULP aggregate membership, not hitcount threshold; no failed tests removed'})
    # No test target counts are returned to development/selection.
    print(json.dumps({'sealed_products':len(rows),'versions':len(audits),'excluded_versions':0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['manifest','slices','slices_cached','cache_verify','oracle'])
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    {'manifest':manifest, 'slices':slices, 'slices_cached':slices_cached,
     'cache_verify':cache_verify, 'oracle':oracle}[args.action](config)


if __name__ == '__main__':
    main()
