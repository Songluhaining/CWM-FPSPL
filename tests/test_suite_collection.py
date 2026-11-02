"""Bounded subprocess and sealed-cache tests; no extra Java benchmark runs."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from cwm_fp.lddg.runner import command,budget_check,execute_pair,runtime_binding
from cwm_fp.data.common import write_json,sha256

class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.policy={'root':str(self.root),'max_total_bytes':1000000,'max_total_files':100,'min_free_bytes':0}

    def test_preflight_byte_budget_prevents_launch(self):
        policy=self.policy|{'max_total_bytes':10}
        r=command([sys.executable,'-c','raise Exception("must not run")'],self.root,self.root/'run.log',max_bytes=100,budget=policy)
        self.assertEqual(r['reason'],'collection_byte_budget');self.assertFalse(r['resource']['launched'])

    def test_minimum_free_space_prevents_launch(self):
        with patch('cwm_fp.lddg.runner.shutil.disk_usage') as disk:
            disk.return_value.free=100
            reason,_=budget_check(self.policy|{'min_free_bytes':101})
        self.assertEqual(reason,'minimum_free_disk')

    def test_total_file_budget_detected(self):
        (self.root/'x').write_text('x')
        self.assertEqual(budget_check(self.policy|{'max_total_files':1})[0],'collection_file_budget')

    def test_fast_exit_file_count_cannot_bypass_limit(self):
        r=command([sys.executable,'-c','from pathlib import Path; [Path(str(i)).write_text("x") for i in range(5)]'],
                  self.root,self.root/'run.log',monitor_files=True,max_files=3)
        self.assertEqual(r['reason'],'file_count_limit')

    def test_running_collection_bytes_stop_child(self):
        r=command([sys.executable,'-c','from pathlib import Path; import time; Path("x").write_bytes(b"x"*10000); time.sleep(10)'],
                  self.root,self.root/'run.log',max_bytes=100,budget=self.policy|{'max_total_bytes':500})
        self.assertEqual(r['reason'],'collection_byte_budget');self.assertLess(r['seconds'],3)

    def test_rss_limit_stops_child_and_records_memory(self):
        r=command([sys.executable,'-c','import time; time.sleep(10)'],self.root,self.root/'run.log',max_rss_bytes=1)
        self.assertEqual(r['reason'],'process_memory_limit');self.assertGreater(r['resource']['memory_samples'],0)

class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.c={'java_home':str(self.root/'java'),'java_heap_mb':512,'max_seconds_per_test':30,'max_events':20000,
                'max_output_bytes':33554432,'strict_runtime_binding':True}
        self.producer=self.root/'producer';self.classes=self.root/'classes';self.build=self.root/'build';self.variant=self.root/'variant';self.stage=self.root/'run'
        for p in [self.root/'java/bin/java.exe',self.producer/'lib/a.jar',self.classes/'Runner.class',self.build/'main/A.class',self.build/'tests/T.class',self.variant/'src/A.java',self.root/'agent.jar']:
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
        self.agent=self.root/'agent.jar'

    def pair(self):
        return execute_pair(self.c,self.root,self.producer,self.classes,self.agent,self.variant,'T.test',['A'],self.build,self.stage,'PASS')

    def fake_command(self,args,cwd,log,*a,**kw):
        log.write_text('fixture')
        Path(args[-1]).write_text('COMPLETE\tT.test\t1\t0\t0\n')
        return {'reason':None,'exit_code':0}

    def seal(self):
        with patch('cwm_fp.lddg.runner.command',side_effect=self.fake_command) as run:
            r=self.pair();self.assertEqual(run.call_count,2)
        return r

    def test_cache_reuses_without_execution(self):
        old=self.seal()
        with patch('cwm_fp.lddg.runner.command',side_effect=AssertionError('rerun forbidden')):
            self.assertEqual(old,self.pair())

    def test_class_change_invalidates_cache(self):
        self.seal();(self.build/'main/A.class').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'different run'):self.pair()

    def test_library_change_invalidates_cache(self):
        self.seal();(self.producer/'lib/a.jar').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'different run'):self.pair()

    def test_trace_corruption_rejected(self):
        self.seal();(self.stage/'instrumented/run.log').write_text('changed')
        with self.assertRaisesRegex(ValueError,'artifact changed'):self.pair()

    def test_added_file_rejected(self):
        self.seal();(self.stage/'extra').write_text('x')
        with self.assertRaisesRegex(ValueError,'artifact set'):self.pair()

    def test_unsealed_interrupt_preserved_and_not_reexecuted(self):
        (self.stage/'original').mkdir(parents=True);p=self.stage/'original/run.log';p.write_text('interrupted')
        with patch('cwm_fp.lddg.runner.command',side_effect=AssertionError('must not run')):
            with self.assertRaisesRegex(ValueError,'Unsealed'):self.pair()
        self.assertEqual(p.read_text(),'interrupted')

    def test_changed_whitelist_changes_binding(self):
        a=runtime_binding(self.c,self.producer,self.classes,self.agent,self.build,['A'])
        b=runtime_binding(self.c,self.producer,self.classes,self.agent,self.build,['B'])
        self.assertNotEqual(a,b)

if __name__=='__main__':unittest.main()
