"""Quality dimensions and label boundaries; no target labels are opened."""
from pathlib import Path
import unittest
from cwm_fp.lddg.batch_audit import distribution,graph_reasons,FOLDS,split_path
from cwm_fp.baselines.boundaries import forbidden_read
from cwm_fp.data.common import ROOT

class QualityTests(unittest.TestCase):
    def q(self):return {'program_identity_complete':True,'normal_path_verified':True,'unsupported_dynamic':[]}

    def test_missing_identity_and_semantics_are_independent(self):
        q=self.q();q['program_identity_complete']=False
        self.assertEqual(graph_reasons(q),['unresolved_program_identity'])

    def test_normal_failure_boundary_does_not_imply_exception_semantics(self):
        q=self.q();q['original_result']='FAIL'
        self.assertEqual(graph_reasons(q),[])

    def test_array_and_library_gaps_are_separate(self):
        q=self.q();q['unsupported_dynamic']=[{'reason':'unsupported_array_opcode'},{'reason':'untraced_call_effects'}]
        self.assertEqual(set(graph_reasons(q)),{'array_semantics','call_or_library_semantics'})

    def test_multiple_gaps_do_not_double_count_one_category(self):
        q=self.q();q['unsupported_dynamic']=[{'reason':'unsupported_array'},{'reason':'unsupported_array_load'}]
        self.assertEqual(graph_reasons(q),['array_semantics'])

    def test_empty_distribution_does_not_fabricate_zero_measurement(self):
        d=distribution([]);self.assertEqual(d['count'],0);self.assertIsNone(d['median']);self.assertIsNone(d['max'])

    def test_even_median_and_sum(self):
        d=distribution([4,1,3,2]);self.assertEqual(d['median'],2.5);self.assertEqual(d['sum'],10)

class TargetBoundaryTests(unittest.TestCase):
    def test_each_fold_allows_only_its_train_file(self):
        for fold in FOLDS:
            allowed=ROOT/'data/processed/labels/full'/fold/'train.csv'
            self.assertFalse(forbidden_read(allowed,'readiness_train_only',{allowed.resolve()}))
            for partition in ('val','test'):
                self.assertTrue(forbidden_read(allowed.with_name(partition+'.csv'),'readiness_train_only',{allowed.resolve()}))
            self.assertTrue(forbidden_read(allowed.parent.parent/'labels.csv','readiness_train_only',{allowed.resolve()}))

    def test_cross_fold_label_scope_is_not_shared(self):
        allowed=ROOT/'data/processed/labels/full/grouped/train.csv'
        self.assertTrue(forbidden_read(ROOT/'data/processed/labels/full/loso_Email/train.csv','readiness_train_only',{allowed.resolve()}))

    def test_existing_split_names_are_preserved(self):
        self.assertEqual(split_path('grouped').name,'splits.csv')
        self.assertEqual(split_path('loso_Email').name,'loso_Email.csv')

if __name__=='__main__':unittest.main()
