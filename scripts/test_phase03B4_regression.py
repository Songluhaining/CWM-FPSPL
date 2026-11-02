"""Exact previous 188 non-fitting checks plus the new framework-effect tests."""
import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed/lddg/phase03B4_framework_v1'
(OUT/'temp').mkdir(exist_ok=True)
os.environ['TEMP']=os.environ['TMP']=str(OUT/'temp')
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
import torch
sys.path.append('D:/SoftInstall/Lib/site-packages')
from cwm_fp.data.common import write_json
old=json.loads(Path('D:/dataset/whn/cwm_fp_generated/data/processed/lddg/phase03B3_bank_v1/unit_test_results.json').read_text(encoding='utf-8'))
loader=unittest.TestLoader();suite=unittest.TestSuite(loader.loadTestsFromName(n) for n in old['test_ids'])
new=loader.loadTestsFromName('test_framework_effects');suite.addTest(new)
def ids(suite):
    for t in suite:
        if isinstance(t,unittest.TestSuite):yield from ids(t)
        else:yield t.id()
names=list(ids(suite));start=time.monotonic()
with (OUT/'unit_tests.log').open('w',encoding='utf-8') as log,patch('test_graph_classifier.fit',side_effect=AssertionError('Phase03B4 forbids fitting')):
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
write_json(OUT/'unit_test_results.json',{'tests_run':result.testsRun,'unique_tests':len(set(names)),'test_ids':names,
    'successful':result.wasSuccessful(),'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
    'seconds':time.monotonic()-start,'argv':[sys.executable,*sys.argv],'exit_code':0 if result.wasSuccessful() else 1,'model_fitting_forbidden':True})
print(result,flush=True)
if not result.wasSuccessful():print('\n'.join(t.id()+': '+err[-1500:] for t,err in result.failures+result.errors))
raise SystemExit(not result.wasSuccessful())
