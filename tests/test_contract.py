"""Starter pack smoke tests. These do not constitute model evaluation."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import validate_contract as vc
import make_splits as sp


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.example = ROOT / "data" / "examples"
        self.products = self.example / "products.example.csv"

    def test_products(self):
        p, err = vc.validate_products(self.products)
        self.assertFalse(err)
        self.assertEqual(len(p), 6)

    def test_tests(self):
        p, err = vc.validate_products(self.products)
        self.assertFalse(err)
        t, err = vc.validate_tests(self.example / "tests.example.csv", p)
        self.assertFalse(err)
        self.assertEqual(len(t), 6)

    def test_graphs(self):
        p, _ = vc.validate_products(self.products)
        t, _ = vc.validate_tests(self.example / "tests.example.csv", p)
        n, err = vc.validate_graphs(self.example / "graphs.example.jsonl", t)
        self.assertFalse(err)
        self.assertEqual(n, 1)

    def test_events_interventions(self):
        _, err1 = vc.validate_events(self.example / "events.example.jsonl")
        _, err2 = vc.validate_interventions(self.example / "interventions.example.jsonl")
        self.assertFalse(err1 + err2)

    def test_group_splits_no_version_overlap(self):
        rows = sp.make_split(self.products, strategy="grouped", seed=42)
        groups = {}
        for x in rows:
            key = (x["system_id"], x["bug_version_id"])
            groups.setdefault(key, set()).add(x["split"])
        self.assertTrue(all(len(s) == 1 for s in groups.values()))
        self.assertEqual({x["split"] for x in rows}, {"train", "val", "test"})

    def test_loso(self):
        rows = sp.make_split(self.products, strategy="loso", holdout="SyntheticB", seed=42)
        self.assertTrue(all(x["split"] == "test" for x in rows if x["system_id"] == "SyntheticB"))
        self.assertTrue(all(x["split"] != "test" for x in rows if x["system_id"] != "SyntheticB"))

    def test_simulated_intervention_cannot_assert_verified_failure(self):
        path = self.example / "interventions.example.jsonl"
        n, err = vc.validate_interventions(path)
        self.assertEqual(n, 1)
        self.assertFalse(err)


if __name__ == "__main__":
    unittest.main()
