import copy
import json
from pathlib import Path
import tempfile
import unittest

from cwm_fp.data.common import ROOT, sha256, write_json
from cwm_fp.models.matched_evaluation import checked_rows, verify_protocol


class MatchedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.cohort=[{'sample_id':'s','identity':['A','v','p'],'partitions':{'grouped':'test'}}]
        self.rows=[{'sample_id':'s','identity':['A','v','p'],'partition':'test','y_score':.3}]

    def test_exact_cohort_accepts_authorized_test_prediction(self):
        self.assertEqual(checked_rows(self.rows,self.cohort),self.rows)

    def test_missing_rejected(self):
        with self.assertRaises(ValueError):checked_rows([],self.cohort)

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):checked_rows(self.rows*2,self.cohort)

    def test_other_product_rejected(self):
        r=copy.deepcopy(self.rows);r[0]['identity'][2]='other'
        with self.assertRaises(ValueError):checked_rows(r,self.cohort)

    def test_partition_relabel_rejected(self):
        r=copy.deepcopy(self.rows);r[0]['partition']='train'
        with self.assertRaises(ValueError):checked_rows(r,self.cohort)

    def test_export_guard_can_hash_own_partition_but_not_other_targets(self):
        from cwm_fp.baselines.boundaries import forbidden_read
        source=(ROOT/'data/processed/labels/full/grouped/train.csv').resolve()
        dest=(ROOT/'data/processed/labels/eval1_matched_v1/grouped/train.csv').resolve()
        allowed={source,dest}
        self.assertFalse(forbidden_read(dest,'subset_labels',allowed))
        self.assertTrue(forbidden_read(dest.with_name('test.csv'),'subset_labels',allowed))
        self.assertTrue(forbidden_read(source.with_name('val.csv'),'subset_labels',allowed))

    def test_frozen_dependency_and_protocol_changes_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as d:
            folder=Path(d);source=folder/'source';source.write_text('frozen')
            protocol=folder/'protocol.json'
            p={'stage_id':'EVAL1','test_evaluation_authorized':True,'frozen_files':{str(source):sha256(source)}}
            write_json(protocol,p);write_json(protocol.with_suffix('.receipt.json'),{'sha256':sha256(protocol)})
            self.assertEqual(verify_protocol(protocol)['stage_id'],'EVAL1')
            source.write_text('changed')
            with self.assertRaises(ValueError):verify_protocol(protocol)
            source.write_text('frozen');p['test_evaluation_authorized']=False;write_json(protocol,p)
            with self.assertRaises(ValueError):verify_protocol(protocol)


if __name__=='__main__':unittest.main()
