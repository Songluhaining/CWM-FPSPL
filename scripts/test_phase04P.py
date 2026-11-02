"""Non-fitting synthetic regressions plus five-product portable-data checks."""
import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/processed/graph_classifier/phase04P_v1'
(OUT/'temp').mkdir(exist_ok=True);os.environ['TEMP']=os.environ['TMP']=str(OUT/'temp')
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests'),str(ROOT/'scripts')]
import torch
sys.path.append('D:/SoftInstall/Lib/site-packages')
from cwm_fp.data.common import write_json
from test_phase03A2_graph_regression import NAMES
loader=unittest.TestLoader();names=['test_graph_classifier.GraphTests.'+n for n in NAMES if n!='test_real_data_blocked_without_labels']
suite=unittest.TestSuite(loader.loadTestsFromName(n) for n in names);suite.addTest(loader.loadTestsFromName('test_graph_package'))
def ids(s):
    for t in s:
        if isinstance(t,unittest.TestSuite):yield from ids(t)
        else:yield t.id()
names=list(ids(suite));start=time.monotonic()
with (OUT/'unit_tests.log').open('w',encoding='utf-8') as log,patch('test_graph_classifier.fit',side_effect=AssertionError('Fitting prohibited in04P')):
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
write_json(OUT/'unit_tests.json',{'tests_run':result.testsRun,'unique_tests':len(set(names)),'test_ids':names,'errors':len(result.errors),
                               'failures':len(result.failures),'skipped':len(result.skipped),'seconds':time.monotonic()-start,'exit_code':0 if result.wasSuccessful() else 1,'fit_executions':0})
print(result)
for t,e in result.failures+result.errors:print(t.id(),e[-2000:])
raise SystemExit(not result.wasSuccessful())
