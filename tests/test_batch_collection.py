"""Frozen-scope and resource checks; no Java executions or model fitting."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from cwm_fp.lddg.batch import read_plan,product_dir,CollectionBudget,local_config
from cwm_fp.lddg.suite import load
from cwm_fp.data.common import write_json

class PlanTests(unittest.TestCase):
    def test_frozen_plan_is_original_and_label_blind(self):
        c,_=read_plan();self.assertEqual(c['seed'],20261101);self.assertEqual(c['label_files_read_before_freeze'],[])
        self.assertEqual(len(c['products']),11);self.assertEqual(c['new_registered_tests'],401)
        self.assertEqual(sum(p['observed_result']=='FAIL' for p in c['products']),2)

    def test_both_systems_and_three_original_partitions(self):
        c,_=read_plan()
        self.assertEqual({(p['identity'][0],p['partitions']['grouped']) for p in c['products']},
                         {(s,f) for s in ['BankAccountTP','Email'] for f in ['train','val','test']})

    def test_versions_never_cross_partitions(self):
        c,_=read_plan()
        for fold in ['grouped','loso_BankAccountTP','loso_Email']:
            by={}
            for p in c['products']:by.setdefault(tuple(p['identity'][:2]),set()).add(p['partitions'][fold])
            self.assertTrue(all(len(v)==1 for v in by.values()))

    def test_same_product_name_has_distinct_work_directory(self):
        c,out=read_plan();names=[p['identity'][2] for p in c['products']]
        self.assertLess(len(set(names)),len(names))
        self.assertEqual(len({product_dir(out,i) for i in range(len(names))}),len(names))

    def test_local_quotas_fit_global_frozen_budget(self):
        c,out=read_plan();lc=local_config(c,product_dir(out,0))
        self.assertLess(len(c['products'])*lc['resource_policy']['max_total_bytes']+268435456,c['resource_policy']['max_total_bytes'])
        self.assertEqual(lc['resource_policy']['min_free_bytes'],c['resource_policy']['min_free_bytes'])

    def test_complete_test_counts_fit_process_budget(self):
        c,_=read_plan();self.assertLessEqual(sum(len(p['tests'])*2 for p in c['products']),c['new_java_test_process_limit'])
        for p in c['products']:self.assertEqual(len({t['test_id'] for t in p['tests']}),len(p['tests']))

class BudgetTests(unittest.TestCase):
    def setUp(self):
        _,out=read_plan();parent=out/'unit_tmp';parent.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=parent);self.root=Path(self.tmp.name).resolve()
        assert self.root.is_relative_to(out.resolve())
        self.addCleanup(self.tmp.cleanup)
        self.c={'resource_policy':{},'collection_time_budget_seconds':100,'new_java_test_process_limit':4}
        patcher=patch('cwm_fp.lddg.batch.budget_check',return_value=(None,{}));patcher.start();self.addCleanup(patcher.stop)

    def test_time_reservation_blocks_before_execution(self):
        b=CollectionBudget(self.c,self.root);called=[]
        with self.assertRaisesRegex(RuntimeError,'time_budget'):b.run('x',101,2,lambda:called.append(1))
        self.assertFalse(called)

    def test_process_budget_persists_across_resume(self):
        b=CollectionBudget(self.c,self.root);b.run('x',10,4,lambda:None);b=CollectionBudget(self.c,self.root)
        with self.assertRaisesRegex(RuntimeError,'process_budget'):b.run('y',10,2,lambda:None)

    def test_interrupted_reservation_is_charged_and_retained(self):
        write_json(self.root/'collection_budget.json',{'seconds':3,'reserved_test_processes':2,
            'interrupted_reservations':[],'pending':{'name':'x','seconds':30,'test_processes':2}})
        b=CollectionBudget(self.c,self.root);self.assertEqual(b.state['seconds'],33)
        self.assertEqual(len(b.state['interrupted_reservations']),1);self.assertIsNone(b.state['pending'])

    def test_failure_preserves_budget_charge(self):
        b=CollectionBudget(self.c,self.root)
        def fail():raise ValueError('deliberate unit exception')
        with self.assertRaises(ValueError):b.run('x',10,2,fail)
        saved=load(b.path);self.assertIsNone(saved['pending']);self.assertEqual(saved['reserved_test_processes'],2)

    def test_disk_rejection_prevents_callback(self):
        b=CollectionBudget(self.c,self.root);called=[]
        with patch('cwm_fp.lddg.batch.budget_check',return_value=('minimum_free_disk',{})):
            with self.assertRaisesRegex(RuntimeError,'minimum_free_disk'):b.run('x',10,2,lambda:called.append(1))
        self.assertFalse(called)

    def test_resource_rejected_sealed_extraction_stays_rejected_on_resume(self):
        from cwm_fp.lddg.batch import extract_batch
        c={**self.c,'products':[{'identity':['Demo','v','p']}]}
        work=product_dir(self.root,0);(work/'g').mkdir(parents=True)
        write_json(work/'collection.json',{})
        write_json(work/'g/quality_report.json',{'artifact_sha256':{},'inputs_sha256':{},'source_sha256':{}})
        failure={'index':0,'status':'failed','execution':{'reason':'file_size_limit'}}
        write_json(work/'extraction_attempt.json',failure)
        extract_batch(c,self.root)
        self.assertEqual(load(self.root/'extraction_summary.json')['products'],[failure])

if __name__=='__main__':unittest.main()
