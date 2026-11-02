"""Evaluator tests run in the existing base environment without torch."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cwm_fp.data.common import sha256, write_json
from cwm_fp.models.graph_evaluator import evaluate_once, score_rows
from cwm_fp.models.graph_metrics import binary_metrics, clustered_interval


class GraphMetricsTests(unittest.TestCase):
    def test_perfect_and_single_class(self):
        m = binary_metrics([0, 1], [.1, .9])
        self.assertEqual((m["pr_auc"], m["roc_auc"], m["average_precision"]), (1., 1., 1.))
        self.assertIsNone(binary_metrics([1, 1], [.2, .3])["pr_auc"])
        self.assertIsNone(binary_metrics([], [])["pr_auc"])

    def test_tied_constant_scores_are_not_ranking_skill(self):
        m = binary_metrics([0, 1, 1, 0], [.4] * 4)
        self.assertEqual(m["average_precision"], .5)
        self.assertEqual(m["roc_auc"], .5)
        self.assertEqual(m["pr_auc"], .75)

    def test_metrics_match_existing_evaluator(self):
        try:
            from cwm_fp.baselines.evaluator import binary_metrics as reference
        except ImportError:
            self.skipTest("Cross-check needs base-environment sklearn/scipy")
        import random
        rng = random.Random(7)
        for _ in range(8):
            y = [0, 1] + [rng.randrange(2) for _ in range(18)]
            s = [rng.choice([.1, .3, .5, .9]) for _ in y]
            actual, expected = binary_metrics(y, s), reference(y, s, [int(v >= .5) for v in s], "probability_uncalibrated")
            for k in ("pr_auc", "average_precision", "roc_auc", "accuracy", "balanced_accuracy", "fp_f1", "brier", "ece_10_equal_width"):
                self.assertAlmostEqual(actual[k], expected[k], places=12, msg=k)

    def test_invalid_metrics(self):
        for y, s in [([0], [float("nan")]), ([1], [1.1]), ([2], [.5]), ([0], [])]:
            with self.assertRaises(ValueError): binary_metrics(y, s)

    def test_once_only_and_frozen_prediction_guard(self):
        with tempfile.TemporaryDirectory() as d:
            p, out = Path(d) / "pred.json", Path(d) / "result.json"
            write_json(p, [{"score": .5}]); digest = sha256(p)
            called = []
            def evaluator(rows): called.append(True); return {"ok": True}
            self.assertTrue(evaluate_once(out, p, digest, evaluator)["ok"])
            with self.assertRaises(FileExistsError): evaluate_once(out, p, digest, evaluator)
            self.assertEqual(len(called), 1)
            p.write_text("[]")
            with self.assertRaises(ValueError): evaluate_once(Path(d)/"different.json", p, digest, evaluator)
            self.assertEqual(len(called), 1)

    def test_cohort_and_identity_mismatch(self):
        truth = {"a": {"identity": ["s", "v", "p"], "y_true": 1}}
        r = {"sample_id": "a", "identity": ["s", "v", "p"], "partition": "test", "y_score": .8, "y_pred": 1, "score_kind": "probability_uncalibrated"}
        self.assertEqual(score_rows([r], truth, partition="test", expected_ids=["a"])["n"], 1)
        for rows in ([r, r], [r | {"partition": "val"}], [r | {"y_true": 1}], [r | {"identity": ["s", "other_version", "p"]}]):
            with self.assertRaises(ValueError): score_rows(rows, truth, partition="test", expected_ids=["a"])

    def test_cluster_bootstrap(self):
        rows = [{"identity": ["s", v, str(y)], "y_true": y, "y_score": .1 if y == 0 else .9} for v in ("v1", "v2") for y in (0, 1)]
        a = clustered_interval(rows, 42, 20)
        self.assertEqual(a, clustered_interval(rows, 42, 20))
        self.assertEqual(a["intervals"]["pr_auc"], [1., 1.])
        self.assertEqual(clustered_interval(rows[:2], 42)["status"], "unavailable")
