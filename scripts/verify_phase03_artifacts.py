"""Read-only provenance/identity/schema checks plus reuse of sealed completed runs."""
import json
from collections import Counter,defaultdict
from pathlib import Path
import sys
import jsonschema
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cwm_fp.data.common import sha256,write_json
from cwm_fp.lddg.extract import validate_graph,extract
from cwm_fp.lddg.runner import execute_pair
from cwm_fp.lddg.static import load_static
from cwm_fp.lddg.trace import Dataflow,parse_trace


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def jsonl(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]


def main():
    c=load(ROOT/'configs/lddg_smoke.json');w=ROOT/c['work_dir'];collection=load(w/'collection.json')
    schemas={name:jsonschema.Draft202012Validator(load(ROOT/'src/cwm_fp/lddg'/(name+'.schema.json'))) for name in ('graphs','events')}
    graphs=jsonl(w/'graphs.jsonl');events=jsonl(w/'events.jsonl');grouped=defaultdict(list)
    fields=('system_id','bug_version_id','product_id','test_id')
    for e in events:schemas['events'].validate(e);grouped[tuple(e[k] for k in fields)].append(e)
    for g in graphs:schemas['graphs'].validate(g);validate_graph(g,grouped[tuple(g[k] for k in fields)])
    assert len(graphs)==len(c['tests'])==6
    before={p:sha256(p) for p in (w/'graphs.jsonl',w/'events.jsonl',w/'source_map.jsonl',w/'quality_report.json')}
    # Cache reuse must validate hashes and run no JVM; exact existing results remain unchanged.
    resumed=0;raw_hashes={}
    for r in collection['records']:
        status=r['status'];assert status['complete'] and status['parity'] and status['matches_historical_outcome']
        variant=Path(r['variant']);stage=Path(r['directory']);product=r['product_id'];build=w/product/'build'
        for name,h in status['inputs']['sources'].items():assert sha256(Path(name))==h;raw_hashes[name]=h
        for name,h in status['artifacts_sha256'].items():assert sha256(stage/name)==h
        names=[p.relative_to(build/'main').as_posix()[:-6].replace('/','.') for p in sorted((build/'main').rglob('*.class'))]
        result=execute_pair(c,w,ROOT/'third_party/existing_instrumenter',w/'producer-classes',w/'agent.jar',variant,r['test_id'],names,build,stage,status['inputs']['expected'])
        assert result==status;resumed+=1
    vendor=load(ROOT/'outputs/phase03_producer_provenance.json')
    for name,h in vendor['files_sha256'].items():
        assert sha256(ROOT/'third_party/existing_instrumenter'/name)==h
        assert sha256(Path(vendor['source_root'])/name)==h
    old=load(ROOT/'outputs/phase02_artifacts.json');unchanged=0
    for name,info in old['files'].items():
        if name.startswith(('data/processed/baselines/','outputs/metrics/phase02_v1/','src/cwm_fp/baselines/','configs/baselines.yaml','tests/test_baselines.py')):
            assert sha256(ROOT/name)==info['sha256'];unchanged+=1
    # No fitting: repeat only deterministic parsing of frozen evidence and compare bytes.
    extract(w)
    for p,h in before.items():assert sha256(p)==h
    fixture=load(ROOT/'outputs/phase03_java_fixtures.json');assert fixture['passed']==11
    fixture_root=ROOT/'data/processed/lddg/phase03_synthetic_v2'
    fixture_static=load_static(fixture_root/'static.tsv');reparsed=0
    for item in fixture['cases']:
        if item['fixture']=='event_limit':continue
        name=item['fixture'];raw=parse_trace(fixture_root/name/'instrumented/run.log','phase03fixture.Phase03FixtureTest.'+name)
        flow=Dataflow(fixture_static).run(raw)
        assert bool(flow.gaps)==(name in {'unsupportedArray','missingStringValue'})
        reparsed+=1
    q=load(w/'quality_report.json')
    write_json(ROOT/'outputs/phase03_verification.json',{'valid':True,'graphs':len(graphs),'events':len(events),
        'T1_supported_scope_graphs':q['T1_tests'],'T2_graphs':0,'T3_graphs':0,'completed_run_reuses_without_new_test_execution':resumed,
        'source_and_test_java_files_unchanged':len(raw_hashes),'vendor_files_verified':len(vendor['files_sha256']),
        'phase02_artifacts_unchanged':unchanged,'frozen_extraction_byte_identical':True,'json_schema_valid':True,
        'original_test_outcomes':dict(Counter(r['status']['runs']['original']['result']['outcome'] for r in collection['records'])),
        'max_events_measured':max(t['events'] for t in q['tests']), 'source_map_counts':q['source_map_counts'],
        'all_canonical_mappings_complete':all(g['quality']['canonical_mapping_complete'] for g in graphs),
        'fixture_cases':11,'frozen_java_fixture_traces_reparsed':reparsed,'model_training_runs':0})
    print('Phase03 provenance, schemas, parity, resume, unchanged baselines and deterministic extraction: PASS')

if __name__=='__main__':main()
