"""Selection/partition/budget checks without Java, targets or model fitting."""
import copy
import unittest
from unittest.mock import patch
from cwm_fp.lddg.cohort import select,rank,product_reservation,QUOTAS
from cwm_fp.baselines.boundaries import forbidden_read
from cwm_fp.data.common import ROOT

def pool():
    return [{'identity':['BankAccountTP',f'{part}_v{v}',f'p{p}'],'partition':part,'observed_result':'PASS',
             'previously_collected':False,'registered_tests':10+p} for part in QUOTAS for v in range(10) for p in range(4)]

class CohortTests(unittest.TestCase):
    def test_deterministic_under_input_order(self):self.assertEqual(select(pool()),select(list(reversed(pool()))))
    def test_quotas_and_three_per_version(self):
        from collections import Counter
        rows=select(pool());self.assertEqual(36,len(rows));self.assertEqual(12,len({tuple(p['identity'][:2]) for p in rows}))
        self.assertEqual({'train':8,'val':2,'test':2},{s:len({p['identity'][1] for p in rows if p['partition']==s}) for s in QUOTAS})
        self.assertEqual({3},set(Counter(p['identity'][1] for p in rows).values()))
    def test_partition_round_robin(self):self.assertEqual(['train','val','test']*6,[p['partition'] for p in select(pool())[:18]])
    def test_whole_original_test_count_not_ranked(self):
        p=pool();a=[r['identity'] for r in select(p)]
        for i,r in enumerate(p):r['registered_tests']=1000-i
        self.assertEqual(a,[r['identity'] for r in select(p)])
    def test_collected_products_never_selected(self):
        p=pool();p[0]['previously_collected']=True;self.assertNotIn(p[0]['identity'],[r['identity'] for r in select(p)])
    def test_unseen_versions_preferred(self):
        p=pool()
        for r in p:
            if r['identity'][1]=='train_v0':r['previously_collected']=r['identity'][2]=='p0'
        self.assertNotIn('train_v0',[r['identity'][1] for r in select(p)])
    def test_missing_partition_not_reassigned(self):
        p=[r for r in pool() if r['partition']!='val'];rows=select(p)
        self.assertEqual(0,sum(r['partition']=='val' for r in rows));self.assertEqual(24,sum(r['partition']=='train' for r in rows))
    def test_original_fail_never_selected(self):
        p=pool()
        for r in p:r['observed_result']='FAIL'
        self.assertEqual([],select(p))
    def test_targets_and_quality_cannot_enter_selector(self):
        for field in ('label','quality','fault_location','test_class','features'):
            p=pool();p[0][field]='forbidden'
            with self.subTest(field=field),self.assertRaises(ValueError):select(p)
    def test_full_product_reservation(self):self.assertEqual((2100,40),product_reservation({'tests':[{}]*20}))
    def test_frozen_rank_namespace(self):self.assertEqual(rank('product','s','v','p'),rank('product','s','v','p'));self.assertNotEqual(rank('product','s','v','p'),rank('version','s','v','p'))
    def test_only_grouped_train_target_allowed(self):
        allowed={ (ROOT/'data/processed/labels/full/grouped/train.csv').resolve() }
        for fold in ('grouped','loso_Email','loso_BankAccountTP'):
            for part in ('train','val','test'):
                path=ROOT/f'data/processed/labels/full/{fold}/{part}.csv'
                self.assertEqual(not(fold=='grouped' and part=='train'),forbidden_read(path,'readiness_train_only',allowed))

if __name__=='__main__':unittest.main()
