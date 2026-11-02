"""Phase02 synthetic unit tests; these scores are never reported as experiments."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cwm_fp.baselines.boundaries import forbidden_read
from cwm_fp.baselines.evaluator import binary_metrics, cluster_interval
from cwm_fp.baselines.features import TrainReferenceFeatures, PULP50
from cwm_fp.baselines.observations import Observation, aggregate_mapping, test_coverage, ddu_complement
from cwm_fp.baselines.vendor import FrozenSourcePU, VENDOR, clap_fit, pure_functions
from cwm_fp.data.common import KEY, digest, write_csv


def observation(version, product, result="PASS", system="SyntheticA"):
    identity = (system, version, product)
    present = frozenset({"Base.A.1", "Base.A.2", "Base.A.3"})
    passed = [frozenset({"Base.A.1"}), frozenset({"Base.A.1", "Base.A.2"})]
    failed = [frozenset({"Base.A.2", "Base.A.3"})] if result == "FAIL" else []
    return Observation(identity, digest(identity), result, present, frozenset().union(*passed, *failed),
                       passed, failed, {s: frozenset({s}) for s in present}, .5, 2/3)


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.observations = [observation(v, p, r) for v in ("v1", "v2", "v3", "v4")
                             for p, r in (("p1", "FAIL"), ("p2", "PASS"))]
        rows = [dict(zip(KEY, o.identity)) | {"split": "train" if o.identity[1] in {"v1", "v2"} else "val" if o.identity[1] == "v3" else "test"}
                for o in self.observations]
        self.path = self.root / "split.csv"
        write_csv(self.path, rows, (*KEY, "split"))
        self.train = self.observations[:4]

    def test_only_train_may_fit_reference_statistics(self):
        with self.assertRaisesRegex(ValueError, "TRAIN"):
            TrainReferenceFeatures().fit(self.observations, self.path)
        model = TrainReferenceFeatures().fit(self.train, self.path)
        self.assertEqual(model.train_ids, {o.identity for o in self.train})

    def test_shapes_and_schema_exactly_six_fifty_two(self):
        model = TrainReferenceFeatures().fit(self.train, self.path)
        arrays = model.transform(self.observations)
        self.assertEqual(arrays["clap6"].shape, (8, 6))
        self.assertEqual(arrays["pulp50"].shape, (8, 50))
        self.assertEqual(arrays["coverage2"].shape, (8, 2))
        self.assertEqual(len(set(PULP50)), 50)
        self.assertTrue(all(np.isfinite(x).all() for x in arrays.values()))

    def test_heldout_products_cannot_update_training_vocabulary_or_others(self):
        model = TrainReferenceFeatures().fit(self.train, self.path)
        query = self.observations[-1]
        expected = model.one(query)
        unrelated = observation("v4", "new")
        unrelated.present = frozenset({"Secret.X.999"})
        unrelated.passed = [unrelated.present]
        unrelated.covered = unrelated.present
        model.one(unrelated)
        for before, after in zip(expected, model.one(query)):
            np.testing.assert_array_equal(before, after)
        self.assertNotIn("Secret.X.999", model.systems["SyntheticA"]["scores"])

    def test_unseen_system_does_not_match_same_spelling_statements(self):
        model = TrainReferenceFeatures().fit(self.train, self.path)
        query = observation("v1", "q", system="SyntheticB")
        clap, pulp, coverage = model.one(query)
        np.testing.assert_array_equal(pulp, np.zeros(50))
        self.assertEqual(clap[0], query.ddu_complement)
        self.assertEqual(coverage[0], 2)

    def test_original_source_reflection_matches_cached_formula(self):
        names = ["jaccard_similarity", "convert_execution_to_set", "check_correctness_reflectability"]
        ns = pure_functions(VENDOR / "PULP/consistent_testing_manager/FPMatricsCaculation.py", names)
        failed = {"f": {"t": [{"id": "a"}, {"id": "b"}]}}
        passed = {"f": {"t": [{"id": "a"}, {"id": "b"}], "u": [{"id": "a"}]}}
        query = {"q1": [{"id": "a"}], "q2": [{"id": "b"}, {"id": "a"}]}
        from collections import Counter
        model = TrainReferenceFeatures().fit(self.train, self.path)
        result = model._reflection(Counter({frozenset('ab'): 1}), Counter({frozenset('ab'): 1, frozenset('a'): 1}), [frozenset('a'), frozenset('ab')])
        for i in range(10):
            expected = ns["check_correctness_reflectability"](failed, passed, query, {}, i/10)
            self.assertAlmostEqual(result[i], expected)

    def test_ddu_matches_original_on_regular_matrix(self):
        names = ["matrix_density", "density", "same_pattern", "diversity", "uniqueness", "ddu_system_level"]
        ns = pure_functions(VENDOR / "CLAP/consistent_testing_manager/DDU.py", names)
        matrix = {"t1": {"a": 1, "b": 0}, "t2": {"a": 1, "b": 1}, "t3": {"a": 1, "b": 0}}
        paths = [frozenset(k for k, v in row.items() if v) for row in matrix.values()]
        self.assertAlmostEqual(ddu_complement(paths, {'a', 'b'}), 1 - ns["ddu_system_level"](copy.deepcopy(matrix)))

    def test_original_rank_return_shape_incompatibility_is_reproducible(self):
        ns = pure_functions(VENDOR / "PULP/consistent_testing_manager/FPMatricsCaculation.py", ["normalized_dict"])
        source_ranking_return = {"Base.A.1": .5}
        with self.assertRaisesRegex(AttributeError, "keys"):
            ns["normalized_dict"](next(iter(source_ranking_return)))

    def test_performance_caches_preserve_feature_values(self):
        # Compare the pre-optimization implementation on all synthetic query
        # classes, including self-excluded F references and unseen systems.
        path = ROOT / 'outputs/phase02_attempt1/features.py'
        if not path.is_file():
            self.skipTest('historical implementation archive not present')
        namespace = {'__name__':'cwm_fp.baselines.historical_features','__package__':'cwm_fp.baselines'}
        exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),namespace)
        old = namespace['TrainReferenceFeatures']().fit(self.train,self.path)
        new = TrainReferenceFeatures().fit(self.train,self.path)
        queries = self.observations + [observation('u','p',system='Unseen')]
        before, after = old.transform(queries), new.transform(queries)
        for family in before:
            np.testing.assert_array_equal(before[family],after[family])


class ObservationTests(unittest.TestCase):
    def test_oracle_attributes_do_not_change_features_and_true_branch_is_counted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            agg = root / "aggregate.xml"
            template = '<coverage><project><package><file path="A.java"><line num="1" featureClass="Base.A" featureLineNum="8" count="0" buggy="{}"/></file></package></project></coverage>'
            agg.write_text(template.format('true'))
            a = aggregate_mapping([agg])
            agg.write_text(template.format('false'))
            b = aggregate_mapping([agg])
            self.assertEqual(a, b)
            test = root / "test.xml"
            test.write_text('<coverage><project><package><file path="/src/A.java"><line num="1" falsecount="0" truecount="1" oracle_bug_line="8"/></file></package></project></coverage>')
            mapped, composed = test_coverage(test, a)
            self.assertEqual(mapped, frozenset({'Base.A.8'}))
            self.assertEqual(composed, frozenset({('A.java', '1')}))

    def test_unknown_executed_location_fails_instead_of_fabricating_mapping(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "x.xml"
            path.write_text('<coverage><project><package><file path="/src/A.java"><line num="9" count="1"/></file></package></project></coverage>')
            with self.assertRaisesRegex(ValueError, "no mapping"):
                test_coverage(path, {('A.java', '1'): 'Base.A.8'})


class ModelTests(unittest.TestCase):
    def test_pu_artifact_loads_in_fresh_inference_process(self):
        import joblib
        import os
        with tempfile.TemporaryDirectory() as folder:
            model = FrozenSourcePU(2, estimators=2).fit(np.array([[.9], [1.]]), np.linspace(0, 1, 40).reshape(-1, 1))
            path = Path(folder) / 'model.joblib'
            joblib.dump({'model': model}, path)
            code = '''import sys,numpy as np
from cwm_fp.baselines.boundaries import install_io_guard
from cwm_fp.baselines.vendor import load_frozen_artifact
install_io_guard('inference')
model=load_frozen_artifact(sys.argv[1])['model']
assert np.isfinite(model.score(np.array([[.5]]))).all()
print('frozen_load_ok')
'''
            result = subprocess.run([sys.executable,'-X','utf8','-B','-c',code,str(path)],capture_output=True,text=True,timeout=30,
                                    env=dict(os.environ,PYTHONPATH=str(ROOT/'src')))
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('frozen_load_ok',result.stdout)

    def test_clap_training_statements_reused_without_evaluation_labels(self):
        x = np.array([[0.], [.1], [.9], [1.]])
        model = clap_fit(x, np.array([0, 0, 1, 1]))
        np.testing.assert_array_equal(model.predict(x), [0, 0, 1, 1])
        self.assertEqual(model.kernel, 'linear')

    def test_original_pu_frozen_prediction_matches_source_and_is_repeatable(self):
        failing = np.array([[.9, 1.], [1., .9], [.95, .95]])
        passing = np.column_stack([np.linspace(0, 1, 40), np.linspace(1, 0, 40)])
        a = FrozenSourcePU(5, estimators=3).fit(failing, passing)
        b = FrozenSourcePU(5, estimators=3).fit(failing, passing)
        np.testing.assert_allclose(a.score(passing), b.score(passing))
        np.testing.assert_array_equal((a.score(passing) > .5).astype(int), a.source._ensemble_predict())
        self.assertIsNone(a.source.ground_truth_target)
        self.assertIsNone(a.source.ground_truth_passing_target)

    def test_original_pu_insufficient_sampling_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "sampled passing negative"):
            FrozenSourcePU(1).fit(np.ones((1, 2)), np.zeros((3, 2)))


class LeakageAndMetricTests(unittest.TestCase):
    def test_label_origin_oracle_and_metrics_access_boundaries(self):
        train = (ROOT / 'data/processed/labels/full/grouped/train.csv').resolve()
        val = ROOT / 'data/processed/labels/full/grouped/val.csv'
        self.assertFalse(forbidden_read(train, 'train', {train}))
        self.assertTrue(forbidden_read(val, 'train', {train}))
        self.assertTrue(forbidden_read(train, 'inference', set()))
        self.assertTrue(forbidden_read(ROOT / 'outputs/metrics/scores.csv', 'inference', set()))
        self.assertTrue(forbidden_read(ROOT / 'x.mutant.log', 'features', set()))
        self.assertTrue(forbidden_read(ROOT / 'attributes.csv', 'features', set()))
        self.assertFalse(forbidden_read(val, 'evaluator', set()))

    def test_real_subprocess_hook_blocks_metrics_and_test_truth(self):
        code = '''from pathlib import Path
from cwm_fp.baselines.boundaries import install_io_guard
from cwm_fp.data.common import ROOT
install_io_guard('inference')
for p in [ROOT/'outputs/metrics/nonexistent.csv',ROOT/'data/processed/labels/full/grouped/test.csv']:
 try: p.open()
 except PermissionError: pass
 else: raise AssertionError('leak')
print('blocked')
'''
        import os
        result = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-c', code], env=dict(os.environ, PYTHONPATH=str(ROOT/'src')), capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('blocked', result.stdout)

    def test_switching_fold_revokes_prior_training_label_access(self):
        import os
        code = '''from cwm_fp.baselines.boundaries import install_io_guard
from cwm_fp.data.common import ROOT
old=(ROOT/'data/processed/labels/policy-test/old.csv').resolve()
new=(ROOT/'data/processed/labels/policy-test/new.csv').resolve()
gate=install_io_guard('train',[old])
gate.clear(); gate.add(new)
try: old.open()
except PermissionError: pass
else: raise AssertionError('prior fold label access was not revoked')
try: new.open()
except PermissionError: raise AssertionError('current fold label was denied')
except FileNotFoundError: pass
print('fold_scoped')
'''
        result = subprocess.run([sys.executable,'-X','utf8','-B','-c',code],env=dict(os.environ,PYTHONPATH=str(ROOT/'src')),
                                capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('fold_scoped',result.stdout)

    def test_fp_is_positive_and_f_never_a_third_metric_class(self):
        result = binary_metrics([1, 0, 1, 0], [.9, .7, .6, .1], [1, 1, 1, 0])
        self.assertAlmostEqual(result['fp_precision'], 2/3)
        self.assertEqual(result['fp_recall'], 1.)
        self.assertEqual(result['confusion_matrix'], [[1, 1], [0, 2]])
        with self.assertRaisesRegex(ValueError, 'invalid binary'):
            binary_metrics([2, 0], [1., 0.], [1, 0])

    def test_single_class_and_empty_origin_strata_are_not_fake_auc(self):
        self.assertIsNone(binary_metrics([1, 1], [.8, .6], [1, 1])['roc_auc'])
        self.assertEqual(binary_metrics([], [], [])['status'], 'unavailable')
        self.assertIsNone(binary_metrics([0, 1], [-1., 1.], [0, 1])['brier'])

    def test_bootstrap_resamples_versions_and_is_deterministic(self):
        rows = [{'system_id':'A','bug_version_id':v,'y_true':y,'y_pred':y,'y_score':.9 if y else .1}
                for v in ('v1','v2','v3') for y in (0, 1)]
        a = cluster_interval(rows, 20, 4)
        self.assertEqual(a, cluster_interval(rows, 20, 4))
        self.assertEqual(a['clusters'], 3)
        self.assertEqual(a['intervals']['fp_f1'], [1.,1.])


if __name__ == '__main__':
    unittest.main()
