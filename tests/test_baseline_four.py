"""Synthetic regression tests; not empirical classification results."""
import copy
import sys
from pathlib import Path
import unittest
import tempfile
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'tests'))
from test_baselines import observation
from cwm_fp.baselines.features import TrainReferenceFeatures
from cwm_fp.baselines.full_dataset import source_target, slicing_input
from cwm_fp.baselines.evaluator import binary_metrics, seed_summary
from cwm_fp.baselines.pipeline import configuration
from cwm_fp.baselines.observations import test_coverage, collect_product, cached_product
from cwm_fp.data.benchmark_adapter import Source
from cwm_fp.data.common import digest


class FullScopeTests(unittest.TestCase):
    def test_base_fault_retained_and_original_failure_never_converted(self):
        for feature in ['Base','BankAccount','ExamDB','Compress']:
            s = feature+'.A.7'
            self.assertEqual(source_target('PASS',{},[s],{s}), 'FP')
            self.assertEqual(source_target('PASS',{},[s],set()), 'TP')
            self.assertEqual(source_target('FAIL',{},[s],{s}), 'F')

    def test_optional_feature_and_statement_membership_both_required(self):
        s='Option.A.3'
        self.assertEqual(source_target('PASS',{'Option':False},[s],{s}), 'TP')
        self.assertEqual(source_target('PASS',{'Option':True},[s],{s}), 'FP')
        self.assertEqual(source_target('PASS',{'Option':True},[s],set()), 'TP')

    def test_all_positive_version_is_preserved_not_removed(self):
        values=[source_target('PASS',{},['Base.A.1'],{'Base.A.1'}) for _ in range(8)]
        self.assertEqual(values,['FP']*8)
        metric=binary_metrics([1]*8,[.8]*8,[1]*8,'probability_uncalibrated')
        self.assertEqual(metric['n'],8)
        self.assertIsNone(metric['roc_auc'])
        self.assertEqual(seed_summary([None,None,None])['available_seeds'],0)

    def test_one_version_scope_cannot_mix_duplicate_statement_identities(self):
        with self.assertRaisesRegex(ValueError,'exactly one'):
            TrainReferenceFeatures().fit_version_context([observation('v1','p'),observation('v2','p')])
        with self.assertRaisesRegex(ValueError,'exactly one'):
            TrainReferenceFeatures().fit_version_context([observation('v1','p',system='A'),observation('v1','p',system='B')])

    def test_target_version_context_is_nonzero_and_label_free(self):
        local=[observation('v2','f','FAIL'),observation('v2','p')]
        model=TrainReferenceFeatures().fit_version_context(local)
        before=model.one(local[1])
        self.assertTrue(np.any(before[1]))
        other=copy.deepcopy(local)
        for o in other:
            o.identity=('DifferentSystem','DifferentVersion',o.identity[2])
        after=TrainReferenceFeatures().fit_version_context(other).one(other[1])
        for a,b in zip(before,after): np.testing.assert_array_equal(a,b)

    def test_spectrum_uses_failing_products_as_original_ranking_does(self):
        f,p=observation('v1','f','FAIL'),observation('v1','p')
        first=TrainReferenceFeatures().fit_version_context([f,p]).systems['SyntheticA']['raw']
        p.passed=[frozenset({'Base.A.1'})]*100
        second=TrainReferenceFeatures().fit_version_context([f,p]).systems['SyntheticA']['raw']
        self.assertEqual(set(first),set(second))
        for k in first: np.testing.assert_array_equal(first[k],second[k])

    def test_slicer_cache_matches_audited_read_fields_not_oracle_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);(path/'src').mkdir();(path/'coverage').mkdir()
            source=path/'src/A.java';source.write_text('class A {}')
            xml=path/'coverage/spectrum_passed_coverage.xml'
            xml.write_text('<coverage><line count="1" featureClass="Base.A" featureLineNum="1" buggy="false"/></coverage>')
            before,hashes=slicing_input(path)
            xml.write_text('<coverage time="different"><line count="8" featureClass="Base.A" featureLineNum="1" buggy="true"/></coverage>')
            after,new_hashes=slicing_input(path)
            self.assertEqual(before,after);self.assertNotEqual(hashes,new_hashes)
            xml.write_text('<coverage><line count="0" featureClass="Base.A" featureLineNum="1"/></coverage>')
            self.assertNotEqual(before,slicing_input(path)[0])
            source.write_text('class A {int x;}')
            self.assertNotEqual(after['source'],slicing_input(path)[0]['source'])

    def test_no_failing_anchors_is_explicit_not_an_exclusion(self):
        local=[observation('v3','p1'),observation('v3','p2')]
        model=TrainReferenceFeatures().fit_version_context(local)
        arrays=model.transform(local)
        self.assertEqual(arrays['pulp50'].shape,(2,50))
        self.assertTrue(np.isfinite(arrays['pulp50']).all())
        self.assertEqual(model.systems['SyntheticA']['fail_tests'],0)

    def test_source_count_policy_handles_unmapped_branch_records_without_guessing(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'coverage.xml'
            path.write_text('<coverage><project><package><file path="A.java"><line num="1" count="1"/>'
                            '<line num="83" truecount="1" falsecount="0"/></file></package></project></coverage>')
            from collections import Counter
            diagnostics=Counter()
            mapped,_=test_coverage(path,{('A.java','1'):'Base.A.1'},hit_policy='source_count_only',diagnostics=diagnostics)
            self.assertEqual(mapped,frozenset({'Base.A.1'}))
            self.assertEqual(diagnostics['unmapped_branch_only_records'],1)
            with self.assertRaisesRegex(ValueError,'no mapping'):
                test_coverage(path,{('A.java','1'):'Base.A.1'})
            path.write_text('<coverage><project><package><file path="A.java"><line num="99" count="1"/></file></package></project></coverage>')
            with self.assertRaisesRegex(ValueError,'no mapping'):
                test_coverage(path,{('A.java','1'):'Base.A.1'},hit_policy='source_count_only')

    def test_product_checkpoint_matches_live_and_rejects_changed_coverage(self):
        # Windows app-container Temp can reject an atomic rename despite a
        # successful write. Exercise the real workspace checkpoint filesystem.
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as directory:
            root=Path(directory);raw=root/'raw';derived=root/'derived'
            self.assertTrue(root.resolve().is_relative_to((ROOT/'outputs').resolve()))
            variant=raw/'v1/variants/p1';copy=derived/'v1/variants/p1'
            for v in (variant,copy):
                (v/'src').mkdir(parents=True);(v/'src/A.java').write_text('class A {}')
            (variant/'coverage/passed').mkdir(parents=True)
            xml='<coverage><project><package><file path="A.java"><line num="1" count="1" featureClass="Base.A" featureLineNum="1"/></file></package></project></coverage>'
            (variant/'coverage/spectrum_passed_coverage.xml').write_text(xml)
            test=variant/'coverage/passed/t.xml';test.write_text(xml)
            for direction in ['forward','backward']:
                (copy/f'slice.all_statements.{direction}_slicing.log').write_text(json.dumps({'Base.A.1':['Base.A.1']}))
            source=Source('BankAccountTP','BankAccountTP',raw,derived,'unknown','synthetic')
            p={'system_id':'BankAccountTP','bug_version_id':'v1','product_id':'p1','sample_id':digest(['synthetic']),
               'suite_view':'original','label':'','label_provenance':'','origin':'unknown','source_variant_path':str(variant),
               'suite_id':'s','observed_result':'PASS'}
            tests=[{'test_id':'t','result':'PASS','coverage_path':str(test),'suite_id':'s'}]
            live=collect_product(p,tests,source,{})
            cached=cached_product(p,tests,source,{},root/'cache')
            restored=cached_product(p,tests,source,{},root/'cache')
            self.assertEqual(live,cached);self.assertEqual(live,restored)
            test.write_text(xml.replace('count="1"','count="2"'))
            with self.assertRaisesRegex(ValueError,'coverage content changed'):
                cached_product(p,tests,source,{},root/'cache')


if __name__=='__main__': unittest.main()
