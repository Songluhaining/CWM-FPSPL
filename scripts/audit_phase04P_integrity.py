"""Compare every packaged tensor and observation to the five frozen originals."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from collections import Counter
from cwm_fp.data.common import sha256,write_json
from cwm_fp.models.graph_package import ProductPackage,read_metadata,portable,ABSOLUTE,digest
from cwm_fp.models.graph_encoder import encode_test_lossless
from cwm_fp.models.graph_data import tensor_products
from cwm_fp.baselines.boundaries import install_io_guard
from verify_phase04P import OUT,OLD,seal,load,lines

def absolute_strings(x):
    if isinstance(x,str):return int(bool(ABSOLUTE.match(x)))
    if isinstance(x,list):return sum(map(absolute_strings,x))
    if isinstance(x,dict):return sum(absolute_strings(k)+absolute_strings(v) for k,v in x.items())
    return 0

def main():
    install_io_guard('features');plan=seal();receipt=load(OUT/'export_receipt.json')
    pkg=ProductPackage(OUT/'package',expected_manifest_sha256=receipt['manifest_sha256'])
    rows=[];default_entries=[];counts=Counter();paths=absolute_strings(pkg.manifest)
    for selected,p in zip(plan['products'],pkg.products):
        assert selected['identity']==p['identity']
        original={g['test_id']:g for g in lines(OLD/selected['name']/'graphs.jsonl')}
        events=lines(OLD/selected['name']/'events.jsonl')
        key=tuple(p['identity'])
        expected_product=tensor_products({key:{'sample_id':p['sample_id'],'observed_result':p['observed_result']}},
            {key:{'split':p['partitions']['grouped']}},{key:[original[name] for name in p['registered_tests']]})[0]
        assert len(expected_product.tests)==len(p['registered_tests'])
        assert all(expected_product.audit[k]==0 for k in ('lost_test_fraction','lost_node_fraction','lost_event_fraction','lost_edge_fraction'))
        default_entries.append({'identity':p['identity'],**expected_product.audit})
        for ref,t,expected in zip(p['tests'],pkg.iter_tests(p),expected_product.tests):
            g=original[ref['identity'][3]];raw=[e for e in events if e['test_id']==g['test_id']]
            meta=read_metadata(pkg.root/ref['metadata']['path'],16*1024**2)
            for name in ('x','edge_index','edge_type','coverage'):assert torch.equal(getattr(expected,name),getattr(t,name)),name
            assert meta['tensor_metadata']==portable(expected.metadata,{})
            assert meta['events']==portable(raw,{})
            assert meta['instruction_catalog']==portable(g['instruction_catalog'],{})
            assert meta['framework_effects']==portable(g.get('framework_effects',[]),{})
            assert meta['quality']==g['quality']
            assert meta['collector_attestation']['source_graph_sha256']==digest(g)
            paths+=absolute_strings(meta)
            for n in g['nodes']:
                if n['type']=='value':counts['value_nodes']+=1;counts['observed_value_nodes']+=int(n['attrs'].get('observed') is True)
                if n['type']=='assertion':counts['assertion_nodes']+=1
            counts['graphs']+=1;counts['events']+=len(raw)
            rows.append({'identity':ref['identity'],'all_tensors_equal':True,'all_node_and_edge_records_equal':True,
                'all_events_equal':True,'static_catalog_equal':True,'framework_effects_equal':True,'quality_equal':True})
    assert paths==0
    write_json(OUT/'integrity_verification.json',{'graphs':rows,'default_entry_products':default_entries,'counts':dict(counts),'absolute_paths_in_package':paths,
        'frozen_source_files_verified':len(plan['source_sha256']),'manifest_sha256':receipt['manifest_sha256'],
        'all_frozen_input_fingerprints_unchanged':True,'new_java_executions':0,'labels_read':False})
    print('SOURCE IDENTICAL',len(rows),'graphs; absolute paths',paths)

if __name__=='__main__':main()
