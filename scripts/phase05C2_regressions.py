"""Run only CPU prefix/state regressions; no optimizer or Java invocation."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
suite=unittest.TestSuite()
for name in ('test_local_state','test_prefix_observer','test_prefix_reader'):
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
