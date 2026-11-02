"""Boundaries and complete-product semantics; fitting is separately budgeted."""
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
import subprocess
import sys
from cwm_fp.models.package_training import validate_options,check_training_batch,representation_audit,reject_target_fields,compare
from cwm_fp.models.graph_smoke import synthetic_graph
from cwm_fp.models.graph_encoder import encode_test_lossless,ProductTensor,FEATURE_SCHEMA
from cwm_fp.models.graph_package import forward_train_stream
import torch


def product(k=('SyntheticOnly','v','p')):
    t=encode_test_lossless(synthetic_graph(k,'t',0,coverage_signal=True))
    return ProductTensor(k,'s','train','PASS',[t],{'tests_before':1})


class PackageTrainingTests(unittest.TestCase):
    def options(self):
        return SimpleNamespace(package=Path('package'),manifest_sha256='1'*64,train_labels=Path('private/grouped/train.csv'),train_label_sha256='2'*64,
            products=Path('products.csv'),tests=Path('tests.csv'),splits=Path('splits.csv'),split_sha256='3'*64,out=Path('result'),
            protocol='grouped_bank_pilot_v1',partition='train',fold='grouped',kind='rgcn',feature_schema=FEATURE_SCHEMA,
            relation_mode='full',ablation='full',max_updates=400,max_model_seconds=1800,epochs=10,micro_steps=100,synthetic_steps=40,batch_products=2)
    def test_explicit_authorized_options(self):validate_options(self.options())
    def test_validation_and_test_paths_rejected_before_open(self):
        for p in ('val','test'):
            a=self.options();a.train_labels=Path('private/grouped/'+p+'.csv')
            with self.assertRaises(ValueError):validate_options(a)
            a=self.options();a.partition=p
            with self.assertRaises(ValueError):validate_options(a)
    def test_other_protocol_train_targets_refused(self):
        a=self.options();a.train_labels=Path('private/loso_Email/train.csv')
        with self.assertRaises(ValueError):validate_options(a)
    def test_schema_and_budget_rejection(self):
        for field,val in [('feature_schema','semantic_v1'),('relation_mode','both'),('max_updates',401),('max_model_seconds',1801),('epochs',11),('micro_steps',101)]:
            a=self.options();setattr(a,field,val)
            with self.subTest(field=field),self.assertRaises(ValueError):validate_options(a)
    def test_wrong_target_join(self):
        p=product()
        for targets in ({},{('SyntheticOnly','other','p'):'FP'},{p.identity:'F'}):
            with self.assertRaises(ValueError):check_training_batch([p],targets)
    def test_missing_test_or_bad_quality(self):
        p=product()
        for q in (replace(p,tests=[]),replace(p,audit={'tests_before':2}),replace(p,audit={'tests_before':1,'lost_event_fraction':.1}),replace(p,partition='val')):
            with self.assertRaises(ValueError):check_training_batch([q],{p.identity:'FP'})
        p.tests[0].metadata['preview_only']=True
        with self.assertRaises(ValueError):check_training_batch([p],{p.identity:'FP'})
    def test_cross_product_graph_refused(self):
        p=product();p.tests=[product(('SyntheticOnly','v2','p')).tests[0]]
        with self.assertRaises(ValueError):check_training_batch([p],{p.identity:'FP'})
    def test_metadata_target_whitelist(self):
        for key in ('label','fault_location','label_provenance','removed_tests'):
            with self.assertRaises(ValueError):reject_target_fields({'nested':[{'nested':{key:1}}]})
    def test_exact_opposing_inputs_reported_without_removal(self):
        a=product();b=product(('SyntheticOnly','v2','p'))
        r=representation_audit([a,b],{a.identity:'FP',b.identity:'TP'})
        self.assertEqual(1,r['canonical_product_tensor']['opposite_label_groups']);self.assertFalse(r['removed_conflicting_samples'])
    def test_shared_cached_object_is_not_semantic_duplicate(self):
        a=product();b=replace(a,identity=('SyntheticOnly','v2','p'))
        with self.assertRaises(ValueError):representation_audit([a,b],{a.identity:'FP',b.identity:'TP'})
    def test_relative_error_does_not_override_absolute_tolerance(self):
        r=compare(torch.tensor([1e-10]),torch.tensor([0.]));self.assertTrue(r['passed']);self.assertGreater(r['max_relative_difference'],.9)
    def test_stream_refuses_heldout_before_model_call(self):
        for part in ('val','test'):
            with self.assertRaises(ValueError):forward_train_stream(None,None,{'partitions':{'grouped':part}})
    def test_actual_guard_blocks_heldout_and_other_protocol(self):
        code="""
from pathlib import Path
from tempfile import TemporaryDirectory
from cwm_fp.models.package_training import access_guard
with TemporaryDirectory() as d:
    root=Path(d)/'labels'; p=root/'grouped/train.csv'; p.parent.mkdir(parents=True); p.write_text('artificial')
    reads,blocked=access_guard(p)
    assert p.read_text()=='artificial'
    for name in ('grouped/val.csv','grouped/test.csv','loso_Email/train.csv'):
        try:(root/name).read_text()
        except PermissionError:pass
        else:raise AssertionError('Target boundary failed')
    assert sum(reads.values())==1 and len(blocked)==3
"""
        r=subprocess.run([sys.executable,'-X','utf8','-c',code],capture_output=True,text=True,timeout=20)
        self.assertEqual(0,r.returncode,r.stdout+r.stderr)


if __name__=='__main__':unittest.main()
