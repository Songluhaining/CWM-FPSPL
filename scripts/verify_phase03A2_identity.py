"""Phase03A2: six frozen real traces + four bounded synthetic constructor tests.

No model fitting, labels, new benchmark test or benchmark recompilation mutation.
"""
import argparse
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cwm_fp.data.common import KEY,sha256,write_json,indexed,read_csv
from cwm_fp.lddg.runner import checked,execute_pair,completion
from cwm_fp.lddg.static import SourceEvidence,load_static,mapping,program_identities,trusted_program_identity
from cwm_fp.lddg.extract import extract,graph_for,validate_graph,identity_errors
from cwm_fp.lddg.trace import Dataflow,parse_trace
from verify_phase03A_mapping import SELECTED,edge_checks

OLD=ROOT/'data/processed/lddg/phase03A_mapping_v1'
BYTECODE=ROOT/'data/processed/lddg/phase03_bank_v2'
OUT=ROOT/'data/processed/lddg/phase03A2_identity_v1'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def key(r):return tuple(r[k] for k in (*KEY,'test_id'))


def preserve():
    frozen=load(OUT/'frozen_inputs_sha256.json')
    for name,h in frozen.items():assert sha256(ROOT/name)==h,name
    write_json(OUT/'preservation.json',{'frozen_data_files_unchanged':len(frozen),'valid':True})
    print('Frozen data files unchanged:',len(frozen))


def tools_build(c):
    java=Path(c['java_home'])/'bin';dest=OUT/'tools';dest.mkdir(exist_ok=True)
    cp=os.pathsep.join(map(str,[ROOT/'third_party/existing_instrumenter/lib/*',Path(c['java_home'])/'lib/tools.jar']))
    checked([java/'javac.exe','-encoding','UTF-8','-cp',cp,'-d',dest,
             ROOT/'src/cwm_fp/lddg/java/Phase03StaticExport.java',ROOT/'src/cwm_fp/lddg/java/Phase03SourceExport.java'],OUT,OUT/'tools-build.log')
    return os.pathsep.join([str(dest),cp])


def source_export(c,cp,sources,dest):
    sources=sorted(set(sources))
    checked([Path(c['java_home'])/'bin/java.exe','-Xmx256m','-cp',cp,'Phase03SourceExport',dest/'source_ast.jsonl',*sources],dest,dest/'source-export.log')
    write_json(dest/'source_evidence.json',{'producer':'JDK8_JavacTask_parse_only',
               'source_sha256':{str(p.resolve()):sha256(p) for p in sources},'records':lines(dest/'source_ast.jsonl')})
    return SourceEvidence(dest/'source_evidence.json')


def export_static(c,cp,classes,dest):
    checked([Path(c['java_home'])/'bin/java.exe','-Xmx256m','-cp',cp,'Phase03StaticExport',dest/'static.tsv',*classes],dest,dest/'static-export.log')
    return load_static(dest/'static.tsv')


def binding_for(source,class_files,**evidence):
    return {'source_sha256':{name:sha256(Path(name)) for name in source.text},
            'class_sha256':{str(p.resolve()):sha256(p) for p in sorted(set(class_files))},**evidence}


def real(c,cp):
    collection=load(OLD/'collection.json');records=collection['records']
    assert [(r['product_id'],r['test_id']) for r in records]==SELECTED
    assert {r['system_id'] for r in records}=={'BankAccountTP'} and {r['bug_version_id'] for r in records}=={'_MultipleBugs_.NOB_1.ID_1'}
    sources={Path(p) for p in load(OLD/'source_evidence.json')['source_sha256']}
    all_classes=[];recompiled=[];seen=set()
    for r in records:
        stage=Path(r['directory']);status=load(stage/'status.json');assert status==r['status']
        for name,h in status['inputs']['sources'].items():assert sha256(Path(name))==h;sources.add(Path(name))
        for name,h in status['artifacts_sha256'].items():assert sha256(stage/name)==h
        assert status['complete'] and status['parity'] and status['matches_historical_outcome']
        for mode in ('original','instrumented'):assert completion(stage/mode/'result.tsv',r['test_id'])==status['runs'][mode]['result']
        product=r['product_id']
        if product in seen:continue
        seen.add(product);dest=OUT/product;dest.mkdir(exist_ok=True);variant=Path(r['variant'])
        # Recompile only existing main sources into isolation; require byte identity.
        build=dest/'recompiled-main';build.mkdir(exist_ok=True)
        checked([Path(c['java_home'])/'bin/javac.exe','-g','-encoding','UTF-8','-cp',ROOT/'third_party/existing_instrumenter/lib/*',
                 '-d',build,*sorted((variant/'src').rglob('*.java'))],dest,dest/'main-recompile.log')
        original=BYTECODE/product/'build'
        for f in (original/'main').rglob('*.class'):
            assert sha256(f)==sha256(build/f.relative_to(original/'main'))
            recompiled.append(str(f))
        classes=sorted(original.rglob('*.class'));all_classes.extend(classes)
        current=export_static(c,cp,classes,dest);previous=load_static(OLD/product/'static.tsv')
        for field in ('methods','controls','handlers','fields','allocations','gaps'):assert current[field]==previous[field],field
    source=source_export(c,cp,sources,OUT)
    binding=binding_for(source,all_classes,product_classes_recompiled_byte_identically=recompiled,
                        source_collection_sha256=sha256(OLD/'collection.json'),javac_sha256=sha256(Path(c['java_home'])/'bin/javac.exe'))
    write_json(OUT/'compilation_binding.json',binding)
    write_json(OUT/'collection.json',collection|{'identity_source_collection':str(OLD/'collection.json')})
    extract(OUT,OUT,OUT/'source_evidence.json',OUT/'compilation_binding.json')
    import jsonschema
    schemas={n:jsonschema.Draft202012Validator(load(ROOT/'src/cwm_fp/lddg'/(n+'.schema.json'))) for n in ('graphs','events')}
    before=lines(OLD/'graphs.jsonl');after=lines(OUT/'graphs.jsonl');events=lines(OUT/'events.jsonl');old_events=lines(OLD/'events.jsonl')
    assert len(after)==6 and len(events)==len(old_events)==332
    grouped=defaultdict(list)
    for e,old in zip(events,old_events):
        schemas['events'].validate(e);assert all(e[k]==v for k,v in old.items());grouped[key(e)].append(e)
    checks=[]
    for old,g,r in zip(before,after,records):
        schemas['graphs'].validate(g);validate_graph(g,grouped[key(g)])
        assert [n['id'] for n in old['nodes']]==[n['id'] for n in g['nodes']]
        assert old['edges']==g['edges'],'Identity normalization changed relationships'
        assert [n for n in old['nodes'] if n['type']=='output']==[n for n in g['nodes'] if n['type']=='output']
        static=load_static(OUT/r['product_id']/'static.tsv');raw=parse_trace(Path(r['directory'])/'instrumented/run.log',r['test_id'])
        flow=Dataflow(static).run(raw);edge_report=edge_checks(g,grouped[key(g)],static,flow)
        assert not identity_errors(g),identity_errors(g)
        assert g['quality']['dependency_semantics_valid']==old['quality']['eligible_graph_structure']
        checks.append({**{k:r[k] for k in (*KEY,'test_id')},'quality':g['quality'],'before_quality':old['quality'],
                       'edge_checks':edge_report,'same_event_order_nodes_edges_and_test_result':True})
    maps=lines(OUT/'source_map.jsonl');old_maps=lines(OLD/'source_map.jsonl');assert len(maps)==len(old_maps)==494
    for old,new in zip(old_maps,maps):
        # AST evidence now includes test sources and class-kind metadata; its hash changes.
        assert all(new[k]==v for k,v in old.items() if k!='source_evidence_sha256'),'Existing line mapping changed'
        if not old['executable']:assert 'program_identity' not in new
    instructions=[m for m in maps if m['executable']];counts=Counter(m['program_identity']['provenance_type'] for m in instructions)
    assert counts=={'SOURCE_BACKED':446,'COMPILER_GENERATED':15}
    assert all(trusted_program_identity(m['program_identity']) for m in instructions)
    generated=[m for m in instructions if m['program_identity']['provenance_type']=='COMPILER_GENERATED']
    assert all(m['canonical_source_line_id'] is None and m['program_identity']['evidence']['default_constructor']['acc_synthetic'] is False for m in generated)
    assert sum(e['program_identity']['provenance_type']=='COMPILER_GENERATED' for e in events)==6
    report={'valid':True,'real_test_executions_this_phase':0,'frozen_real_traces_reprocessed':6,'events':332,'product_events':253,
            'product_instruction_identity_counts':dict(counts),'source_mapped_instructions':446,'source_mapped_product_events':247,
            'compiled_product_classes_verified_byte_identical':len(recompiled),'source_evidence_files':len(source.text),
            'source_anchors_retained_without_events':33,'control_edges':sum(x['edge_checks']['control_endpoints_checked'] for x in checks),
            'data_edges':sum(x['edge_checks']['data_edges'] for x in checks),'per_test':checks,'model_training_runs':0}
    write_json(OUT/'verification.json',report);print('Real identity verification:',counts,'6 graphs; 332 events; all edges unchanged')


def fixtures(c,cp):
    dest=OUT/'synthetic';dest.mkdir(exist_ok=True);variant=dest/'variants/fixture'
    for folder in ('src','test'):(variant/folder).mkdir(parents=True,exist_ok=True)
    base=dest/'features/Base';base.mkdir(parents=True,exist_ok=True)
    for name in ('Implicit.java','Explicit.java','WithField.java'):
        for target in (variant/'src'/name,base/name):shutil.copyfile(ROOT/'tests/java/identity'/name,target)
    shutil.copyfile(ROOT/'tests/java/identity/IdentityFixtureTest.java',variant/'test/IdentityFixtureTest.java')
    (variant/'roles.meta').write_text(json.dumps({'features':['Base'],'roles':{}}),encoding='utf-8');(variant/'coverage').mkdir(exist_ok=True)
    build=dest/'build';producer=ROOT/'third_party/existing_instrumenter'
    for part,folder in [('main','src'),('tests','test')]:
        (build/part).mkdir(parents=True,exist_ok=True)
        checked([Path(c['java_home'])/'bin/javac.exe','-g','-encoding','UTF-8','-cp',os.pathsep.join(map(str,[build/'main',producer/'lib/*'])),
                 '-d',build/part,*sorted((variant/folder).glob('*.java'))],dest,dest/(part+'-build.log'))
    source=source_export(c,cp,list(variant.rglob('*.java'))+list(base.glob('*.java')),dest)
    static=export_static(c,cp,sorted(build.rglob('*.class')),dest);binding=binding_for(source,list(build.rglob('*.class')),synthetic=True)
    write_json(dest/'compilation_binding.json',binding)
    rows,mapped,_=mapping(variant,static,{'Base':True},source)
    outcomes=[];new_executions=0
    for test in ('implicitConstructor','explicitConstructor','repeatedConstruction','initializedField'):
        identity='identityfixture.IdentityFixtureTest.'+test;stage=dest/test;cached=(stage/'status.json').exists()
        status=execute_pair(c,BYTECODE,producer,BYTECODE/'producer-classes',BYTECODE/'agent.jar',variant,identity,
                            ['identityfixture.Implicit','identityfixture.Explicit','identityfixture.WithField'],build,stage,'PASS')
        new_executions+=0 if cached else 2
        assert status['complete'] and status['parity'] and status['matches_historical_outcome']
        raw=parse_trace(stage/'instrumented/run.log',identity)
        identities=program_identities(variant,static,mapped,source,binding,[(*r['domain'],r['bci']) for r in raw])
        lineage={'system_id':'synthetic','bug_version_id':'identity-fixture','product_id':'fixture','test_id':identity}
        graph,events=graph_for(raw,static,mapped,{'Base':True},lineage,status,identities)
        validate_graph(graph,events);assert not identity_errors(graph),identity_errors(graph)
        assert graph['quality']['dependency_semantics_valid']
        implicit=[n for n in graph['nodes'] if n['type']=='statement' and n['attrs'].get('composed_class')=='identityfixture.Implicit']
        explicit=[n for n in graph['nodes'] if n['type']=='statement' and n['attrs'].get('composed_class')=='identityfixture.Explicit']
        assert all(n['attrs']['program_identity']['provenance_type']=='COMPILER_GENERATED' for n in implicit)
        assert all(n['attrs']['program_identity']['provenance_type']=='SOURCE_BACKED' for n in explicit)
        repeated=[n['attrs'] for n in implicit if 'event_index' in n['attrs']]
        if test=='repeatedConstruction':
            assert len(repeated)==6 and len({n['canonical_event_id'] for n in repeated})==6
            assert len({n['static_instruction_id'] for n in repeated})==3 and len({n['frame_id'] for n in repeated})==2
        assert all(not x['evidence']['default_constructor']['acc_synthetic'] for x in identities.values() if x['provenance_type']=='COMPILER_GENERATED')
        write_json(stage/'graph.json',graph);write_json(stage/'events.json',events)
        outcomes.append({'test':test,'synthetic':True,'real_execution':True,'parity':True,'events':len(events),'valid':True})
        print('Constructor fixture:',test,'PASS')
    write_json(dest/'source_map.json',rows)
    write_json(dest/'verification.json',{'valid':True,'cases':outcomes,'new_test_executions_this_command':new_executions,
                                       'source_backed_and_generated_both_have_ACC_SYNTHETIC_false':True,'benchmark_experiments':False})


def admission():
    from cwm_fp.models.graph_data import load_corpus
    from cwm_fp.models.graph_encoder import encode_test,evidence_errors
    config=load(ROOT/'configs/graph_classifier.json');audits={}
    for fold in config['folds']:
        split=ROOT/config['split_dir']/('splits.csv' if fold=='grouped' else fold+'.csv')
        _,_,graphs,eligible,audit=load_corpus(ROOT/config['products'],ROOT/config['tests'],OUT/'graphs.jsonl',split,
                                           holdout=fold[5:] if fold.startswith('loso_') else None)
        assert not eligible;assert audit['identity_and_semantics_qualified']==4 and audit['program_identity_complete']==6
        audits[fold]=audit
    encoded=[];rejected=[]
    for graph in lines(OUT/'graphs.jsonl'):
        if evidence_errors(graph):
            try:encode_test(graph,'both')
            except ValueError:rejected.append(graph['test_id'])
            else:raise AssertionError('Exception semantics silently admitted')
        else:
            tensor=encode_test(graph,'both')
            encoded.append({'product_id':graph['product_id'],'test_id':graph['test_id'],'shape':list(tensor.x.shape),
                            'feature_schema':tensor.metadata['feature_schema'],'preview_only':tensor.metadata['preview_only'],
                            'state_observations_used_as_classifier_features':False})
    assert len(encoded)==4 and len(rejected)==2
    write_json(OUT/'admission.json',audits);write_json(OUT/'encoding_check.json',{'encoded_tests':encoded,'rejected_tests':rejected,
               'eligible_products':0,'model_training_runs':0,'model_instances_created':0,'labels_read':False})
    print('Identity+semantics test graphs: 4 accepted, 2 exception graphs rejected; complete products: 0; no training')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['real','fixtures','admission','preservation'],required=True);args=parser.parse_args()
    if args.stage=='preservation':preserve()
    else:
        from cwm_fp.baselines.boundaries import install_io_guard
        install_io_guard('features')
        c=load(OLD/'collection.json')['config']
        if args.stage=='admission':admission()
        else:
            cp=tools_build(c)
            if args.stage=='real':real(c,cp)
            else:fixtures(c,cp)
