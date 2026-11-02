"""Bounded data-only regression; no Java rerun, model forward, or training."""
import json
from pathlib import Path
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from cwm_fp.lddg.cohort import OUT,OLD,read_plan
from cwm_fp.lddg.suite import load
from cwm_fp.data.common import sha256,write_json
from cwm_fp.baselines.boundaries import install_io_guard


class FrozenEvidenceTests(unittest.TestCase):
    def test_current_cohort_seal(self):
        s=load(OUT/'cohort_seal.json')
        for name,h in s['files_sha256'].items():
            self.assertEqual(h,sha256(Path(name)),name)

    def test_selection_and_original_inputs_unchanged(self):
        plan=read_plan()
        expected={}
        for p in plan['products']:
            expected.update(p['source_sha256'])
            expected[str(Path(p['source_variant_path'])/'roles.meta')]=p['roles_sha256']
        for path,h in expected.items():self.assertEqual(h,sha256(Path(path)),path)

    def test_old_five_products_and_all_records_preserved(self):
        from cwm_fp.models.graph_package import ProductPackage,read_metadata
        old=ProductPackage(OLD/'package',expected_manifest_sha256=load(OLD/'export_receipt.json')['manifest_sha256'])
        new=ProductPackage(OUT/'package',expected_manifest_sha256=load(OUT/'export_receipt.json')['manifest_sha256'])
        by={tuple(p['identity']):p for p in new.products};count=0
        for p in old.products:
            q=by[tuple(p['identity'])]
            self.assertEqual(p['registered_tests'],q['registered_tests']);self.assertEqual(p['partitions'],q['partitions'])
            for a,b in zip(p['tests'],q['tests']):
                left=read_metadata(OLD/'package'/a['metadata']['path'],16*1024**2)
                right=read_metadata(OUT/'package'/b['metadata']['path'],16*1024**2)
                for field in ('tensor_metadata','events','quality','instruction_catalog','framework_effects'):
                    self.assertEqual(left[field],right[field],(p['identity'],a['identity'],field))
                count+=1
        self.assertEqual(75,count)


def preservation():
    plan=read_plan();expected=dict(plan['frozen_input_sha256'])
    for p in plan['products']:
        expected.update(p['source_sha256'])
        expected[str(Path(p['source_variant_path'])/'roles.meta')]=p['roles_sha256']
    for status in OUT.glob('p*/r/t*/status.json'):
        record=load(status)
        expected.update(record['inputs']['sources'])
        expected.update(record['inputs']['runtime_binding']['artifacts_sha256'])
        expected.update({str(status.parent/name):h for name,h in record['artifacts_sha256'].items()})
    # Compare historical data and reports with their existing final indexes;
    # mutable current source is deliberately not compared to old stage code.
    historical=0
    for name in ('phase03B4_artifacts.json','phase04P_artifacts.json'):
        doc=load(ROOT/'outputs'/name)
        for path,record in doc['files'].items():
            if path.startswith(('data/processed/','outputs/')) and path!='outputs/phase_status.md':
                expected[str(ROOT/path)]=record['sha256'];historical+=1
    mismatch=[path for path,h in expected.items() if not Path(path).is_file() or sha256(Path(path))!=h]
    result={'checked_files':len(expected),'historical_index_entries_checked':historical,'mismatches':mismatch,
        'scope':'Frozen manifests/splits and semantic implementations; selected source/test/feature files and roles; current paired-run runtime bindings and sealed logs; existing03B4/04P data/reports. Target files not opened.',
        'plan_sha256':sha256(OUT/'cohort_plan.json'),'files_sha256':expected}
    write_json(OUT/'preservation.json',result)
    if mismatch:raise AssertionError('Preservation mismatch: '+repr(mismatch))


def main():
    install_io_guard('features');start=time.monotonic();(OUT/'temp').mkdir(exist_ok=True)
    import test_phase04D1_cohort as cohort
    import test_graph_package as package
    # Existing test fixtures copy read-only04P data into this stage's scratch.
    # Do not create/delete temporary files in the frozen04P directory.
    package.OUT=OUT
    suite=unittest.TestSuite();loader=unittest.defaultTestLoader;ids=[]
    for cls in (cohort.CohortTests,package.PackageTests,package.PrefixTests,FrozenEvidenceTests):
        for name in loader.getTestCaseNames(cls):
            if name=='test_frozen_model_equivalence_and_train_mode_refusal':continue
            case=cls(name);suite.addTest(case);ids.append(case.id())
    with (OUT/'unit_tests.log').open('w',encoding='utf-8') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    check_error=None
    try:preservation()
    except Exception as e:check_error=str(e)
    ok=result.wasSuccessful() and check_error is None
    write_json(OUT/'unit_tests.json',{'command':[sys.executable,str(Path(__file__).resolve())],
        'exit_code':0 if ok else 1,'tests_run':result.testsRun,'passed':result.testsRun-len(result.errors)-len(result.failures)-len(result.skipped),
        'errors':[(t.id(),e) for t,e in result.errors],'failures':[(t.id(),e) for t,e in result.failures],
        'skipped':result.skipped,'test_ids':ids,'seconds':time.monotonic()-start,'preservation_error':check_error,
        'excluded_existing_test':'model forward equivalence;04D1 is data-only','Java_tests_rerun':0,'model_forward_runs':0,'optimizer_or_backward_runs':0,
        'log':str(OUT/'unit_tests.log'),'log_sha256':sha256(OUT/'unit_tests.log')})
    print('REGRESSION',result.testsRun,'success',ok,'preservation_error',check_error,flush=True)
    return 0 if ok else 1


if __name__=='__main__':raise SystemExit(main())
