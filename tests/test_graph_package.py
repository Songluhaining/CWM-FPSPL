"""No fitting. Real five-product checks and explicitly artificial corruption cases."""
import copy
import gzip
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from cwm_fp.data.common import ROOT,sha256
from cwm_fp.models.graph_package import ProductPackage,ReadBudget,safe_path,write,forward_stream,write_test,read_metadata
from cwm_fp.models.graph_encoder import encode_test_lossless,Limits,collate
from cwm_fp.models.graph_classifier import GraphClassifier
from cwm_fp.models.graph_data import tensor_products
from cwm_fp.models.train_graph import check_partitions
from cwm_fp.models.graph_smoke import synthetic_graph
from cwm_fp.models.prefix_contract import PrefixCursor,validate_prefix

OUT=ROOT/'data/processed/graph_classifier/phase04P_v1';PKG=OUT/'package'
def load(p):return json.loads(p.read_text(encoding='utf-8'))

class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=OUT/'temp');self.root=Path(self.temp.name)
        self.m=load(PKG/'manifest.json');self.m['products']=[self.m['products'][-1]]
        for p in self.m['products']:
            for g in p['tests']:
                for k in ('tensor','metadata'):
                    dest=self.root/g[k]['path'];dest.parent.mkdir(exist_ok=True);shutil.copyfile(PKG/g[k]['path'],dest)
        self.anchor=self.save()
    def tearDown(self):self.temp.cleanup()
    def save(self):write(self.root/'manifest.json',self.m);return sha256(self.root/'manifest.json')
    def reader(self):return ProductPackage(self.root,expected_manifest_sha256=self.save())
    def meta(self,change):
        g=self.m['products'][0]['tests'][0];p=self.root/g['metadata']['path'];d=read_metadata(p,16*1024**2);change(d)
        if p.suffix=='.gz':p.write_bytes(gzip.compress(json.dumps(d).encode(),mtime=0))
        else:write(p,d)
        g['metadata'].update(sha256=sha256(p),bytes=p.stat().st_size)
    def test_complete_eight(self):
        p=self.reader();self.assertEqual(8,len(list(p.iter_tests(p.products[0]))))
    def test_single_test_artificial_registration(self):
        # A synthetic container contract exercise, never added to the research package.
        p=self.m['products'][0];p['tests']=p['tests'][:1];p['registered_tests']=p['registered_tests'][:1]
        self.meta(lambda d:d['quality']['suite_completeness'].update(expected_tests=1,collected_tests=1))
        reader=self.reader();self.assertEqual(1,len(list(reader.iter_tests(reader.products[0]))))
    def test_over_sixteen_and_variable_sizes(self):
        p=ProductPackage(PKG,expected_manifest_sha256=sha256(PKG/'manifest.json'));row=next(r for r in p.products if len(r['tests'])==20)
        ts=list(p.iter_tests(row));self.assertEqual(20,len(ts));self.assertGreater(len({len(t.x) for t in ts}),1)
    def test_resume_exact_suffix(self):
        p=self.reader();row=p.products[0];all_tests=list(p.iter_tests(row));resume=list(p.iter_tests(row,start=3))
        self.assertEqual(5,len(resume))
        for a,b in zip(all_tests[3:],resume):self.assertTrue(torch.equal(a.edge_index,b.edge_index))
    def test_incomplete_suite(self):
        self.m['products'][0]['tests'].pop()
        with self.assertRaises(ValueError):self.reader()
    def test_shortened_registration_cannot_reuse_full_suite_claim(self):
        p=self.m['products'][0];p['tests']=p['tests'][:1];p['registered_tests']=p['registered_tests'][:1]
        reader=self.reader()
        with self.assertRaises(ValueError):list(reader.iter_tests(reader.products[0]))
    def test_missing_file(self):
        (self.root/self.m['products'][0]['tests'][0]['tensor']['path']).unlink()
        with self.assertRaises((ValueError,OSError)):self.reader()
    def test_corruption(self):
        p=self.root/self.m['products'][0]['tests'][0]['metadata']['path'];data=p.read_bytes();p.write_bytes(data[:-1]+b'!')
        reader=self.reader()
        with self.assertRaises(ValueError):list(reader.iter_tests(reader.products[0]))
    def test_rehashed_manifest_requires_external_anchor(self):
        self.m['products'][0]['partitions']['grouped']='train';self.save()
        with self.assertRaises(ValueError):ProductPackage(self.root,expected_manifest_sha256=self.anchor)
    def test_path_escape(self):
        for s in ['../x','C:/x','/x','graphs\\x']:
            with self.subTest(s=s),self.assertRaises(ValueError):safe_path(self.root,s)
    def test_unqualified_metadata_rejected(self):
        self.meta(lambda d:d['quality'].update(eligible_graph_structure=False));reader=self.reader()
        with self.assertRaises(ValueError):list(reader.iter_tests(reader.products[0]))
    def test_legacy_manifest_rejected(self):
        self.m['feature_schema']='semantic_v1'
        with self.assertRaises(ValueError):self.reader()
    def test_19_column_tensor_rejected(self):
        g=self.m['products'][0]['tests'][0];p=self.root/g['tensor']['path']
        with np.load(p,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        a['x']=a['x'][:,:19];np.savez(p,**a);g['tensor'].update(sha256=sha256(p),bytes=p.stat().st_size);reader=self.reader()
        with self.assertRaises(ValueError):list(reader.iter_tests(reader.products[0]))
    def test_same_names_do_not_merge_products(self):
        p=ProductPackage(PKG,expected_manifest_sha256=sha256(PKG/'manifest.json'))
        all_ids=[tuple(t['identity']) for row in p.products for t in row['tests']]
        self.assertEqual(75,len(set(all_ids)));self.assertLess(len({k[3] for k in all_ids}),len(all_ids))
    def test_foreign_product_record(self):
        self.m['products'][0]['tests'][0]['identity'][1]='other-version'
        with self.assertRaises(ValueError):self.reader()
    def test_suite_order_is_not_silently_changed(self):
        self.m['products'][0]['tests'].reverse()
        with self.assertRaises(ValueError):self.reader()
    def test_shard_capacity_rejects_not_samples(self):
        with self.assertRaises(ValueError):ProductPackage(self.root,expected_manifest_sha256=self.anchor,budget=ReadBudget(max_shard_bytes=1))
    def test_graph_capacity_rejects_not_samples(self):
        g=synthetic_graph(('s','v','p'),'t',0)
        with self.assertRaises(ValueError):encode_test_lossless(g,capacity=Limits(1,1,1,1))
    def test_unverified_source_export_rejected(self):
        g=synthetic_graph(('s','v','p'),'t',0);g['quality']['eligible_graph_structure']=False
        with self.assertRaises(ValueError):write_test(self.root,99,g,[],{}, {})
    def test_all_relations_preserved(self):
        g=synthetic_graph(('s','v','p'),'t',0);t=encode_test_lossless(g)
        self.assertEqual(len(g['edges']),t.edge_index.shape[1]);self.assertEqual(7,len(set(t.edge_type.tolist())))
    def test_frozen_model_equivalence_and_train_mode_refusal(self):
        p=self.reader();row=p.products[0];torch.manual_seed(20261101);torch.set_num_threads(1);model=GraphClassifier().eval()
        before={k:v.clone() for k,v in model.state_dict().items()}
        with torch.inference_mode():a=model(collate([p.materialize(row,'grouped')]))
        b=forward_stream(model,p,row,micro_tests=3);torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
        model.train()
        with self.assertRaises(ValueError):forward_stream(model,p,row)
    def test_materialization_budget(self):
        p=self.reader()
        with self.assertRaises(ValueError):p.materialize(p.products[0],'grouped',max_tensor_bytes=1)
    def test_default_product_entry_preserves_twenty_tests_and_all_edges(self):
        k=('synthetic','v','p');graphs=[synthetic_graph(k,'t%02d'%i,0) for i in range(20)]
        result=tensor_products({k:{'sample_id':'synthetic-only','observed_result':'PASS'}},{k:{'split':'train'}},{k:graphs})[0]
        self.assertEqual(20,len(result.tests))
        self.assertEqual(sum(len(g['edges']) for g in graphs),sum(t.edge_index.shape[1] for t in result.tests))
        self.assertTrue(all(result.audit[k]==0 for k in ('lost_node_fraction','lost_edge_fraction','lost_event_fraction','lost_test_fraction')))
    def test_formal_partition_gate_rejects_explicit_legacy_sampling(self):
        k=('synthetic','v','p');graphs=[synthetic_graph(k,'t%02d'%i,0) for i in range(20)]
        result=tensor_products({k:{'sample_id':'synthetic-only','observed_result':'PASS'}},{k:{'split':'train'}},{k:graphs},limits=Limits())[0]
        self.assertGreater(result.audit['lost_test_fraction'],0)
        with self.assertRaisesRegex(ValueError,'Lossy'):
            check_partitions([result],{k:'train',('synthetic','v2','p'):'val',('synthetic','v3','p'):'test'})
    def test_cli_defaults_to_full_without_executing_training(self):
        from cwm_fp.models.train_graph import main
        with patch('cwm_fp.models.train_graph.configuration',return_value={}),patch('cwm_fp.models.train_graph.real_train',return_value=3) as entry:
            self.assertEqual(3,main(['--mode','train']))
            self.assertEqual('full',entry.call_args.args[4])

class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.cursor=PrefixCursor(('s','v','p','t'),2)
        self.r={'test_identity':['s','v','p','t'],'evidence_kind':'observed','available_after_event':2,'event_index':1,'observed':False,'value':None}
    def test_valid_missing_value(self):self.assertTrue(validate_prefix(self.cursor,[self.r]))
    def test_future_event_or_use(self):
        for patch in [{'available_after_event':3},{'event_index':3},{'use_event':3},{'call_return_event':4},{'depends_on_events':[4]}]:
            with self.subTest(patch=patch),self.assertRaises(ValueError):validate_prefix(self.cursor,[self.r|patch])
    def test_unknown_visibility(self):
        with self.assertRaises(ValueError):validate_prefix(self.cursor,[self.r|{'available_after_event':None}])
    def test_future_outcome_and_posterior(self):
        for k in ['test_result','assertion_passed','posterior','quality','whole_trace_statistics']:
            with self.subTest(k=k),self.assertRaises(ValueError):validate_prefix(self.cursor,[self.r|{'nested':{k:True}}])
    def test_static_structure_has_no_future_executed_flag(self):
        r={'test_identity':self.r['test_identity'],'evidence_kind':'static','attrs':{'executed':True}}
        with self.assertRaises(ValueError):validate_prefix(self.cursor,[r])
    def test_missing_value_not_zero(self):
        with self.assertRaises(ValueError):validate_prefix(self.cursor,[self.r|{'value':0}])
    def test_missing_mask_or_invalid_temporal_reference(self):
        for change in [{'observed':None,'value':3},{'depends_on_events':[-1]},{'use_event':True}]:
            with self.subTest(change=change),self.assertRaises(ValueError):validate_prefix(self.cursor,[self.r|change])
    def test_cross_test_and_predicted_rejected(self):
        for patch in [{'test_identity':['s','v','p','other']},{'evidence_kind':'predicted'}]:
            with self.assertRaises(ValueError):validate_prefix(self.cursor,[self.r|patch])

if __name__=='__main__':unittest.main()
