"""Phase 01 regression tests use synthetic files, never external benchmark truth."""
import csv
import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cwm_fp.data import benchmark_adapter as ba
from cwm_fp.data import labels, splits
from cwm_fp.data.common import KEY, SYSTEMS, generated_dir, indexed, read_csv, sha256, write_csv
from cwm_fp.data.inputs import load_inputs
from cwm_fp.data.validation import audit, validate_sealed


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        specs = []
        for system, alias in (("BankAccountTP", "BankAccountTP"), ("Email", "Email-FH-JML")):
            raw, sealed = self.root / system, self.root / (system + "_labels")
            specs.append({"system_alias": alias, "root": str(raw), "label_root": str(sealed),
                          "label_source": "synthetic_fixture", "dataset_family": "synthetic"})
            for i in range(1, 4):
                version = f"_MultipleBugs_.NOB_1.ID_{i}"
                version_dir = raw / version
                report = []
                lab = []
                for n, observed, value, converted in ((1, "PASS", "TP", ""), (2, "FAIL", "F", ""),
                                                       (3, "FAIL", "FP", "1"), (4, "PASS", "FP", "")):
                    pid = f"model_m_ca4_{n:04d}"
                    variant = version_dir / "variants" / pid
                    (variant / "test").mkdir(parents=True)
                    coverage = variant / "coverage"
                    (coverage / "passed").mkdir(parents=True)
                    (coverage / "passed/T.test0.coverage.xml").write_text("<coverage/>")
                    (coverage / "spectrum_passed_coverage.xml").write_text('<coverage buggy="true"/>')
                    if observed == "FAIL":
                        (coverage / "failed").mkdir()
                        (coverage / "failed/T.test1.coverage.xml").write_text("<coverage/>")
                        (coverage / "spectrum_failed_coverage.xml").write_text('<coverage buggy="true"/>')
                    report.append({"Product\\Feature": pid, "Base": " T ", "Lock": " F ",
                                   "__TEST_OUTPUT__": "__PASSED__" if observed == "PASS" else "__FAILED__"})
                    lab.append({"VARIANT": pid, "LABEL": value, "FP transformed from F": converted})
                fields = ("Product\\Feature", "Base", "Lock", "__TEST_OUTPUT__")
                write_csv(version_dir / "config.report.csv", report, fields)
                write_csv(sealed / version / "config.report.csv", report, fields)
                write_csv(sealed / version / "variant_labels.csv", lab, ("VARIANT", "LABEL", "FP transformed from F"))
        self.config = self.root / "sources.json"
        self.config.write_text(json.dumps({"sources": specs}), encoding="utf-8")
        self.sources = ba.load_sources(self.config)
        self.products, self.tests, self.metadata = ba.scan(self.sources, max_versions=6, max_products=24)
        self.products_path, self.tests_path = self.root / "products.csv", self.root / "tests.csv"
        write_csv(self.products_path, self.products, ba.PRODUCT_FIELDS)
        write_csv(self.tests_path, self.tests, ba.TEST_FIELDS)
        self.label_rows, self.label_audit = labels.import_labels(self.products_path, self.config)
        self.labels_path = self.root / "labels.csv"
        write_csv(self.labels_path, self.label_rows, labels.FIELDS)
        self.split_path = self.root / "splits.csv"
        write_csv(self.split_path, splits.make_split(self.products_path, seed=42), (*KEY, "split"))

    def test_full_identity_does_not_mix_same_version_names_across_systems(self):
        self.assertEqual(len(indexed(self.products)), 24)
        self.assertEqual(len({p["sample_id"] for p in self.products}), 24)
        self.assertEqual({p["system_id"] for p in self.products}, {"BankAccountTP", "Email"})

    def test_adapter_is_read_only_and_does_not_read_oracle_files(self):
        def snapshot():
            return {str(p): sha256(p) for source in self.sources for p in source.root.rglob("*") if p.is_file()}
        before = snapshot()
        original = Path.open
        def guarded(path, *args, **kwargs):
            self.assertNotIn("mutant.log", str(path))
            self.assertNotIn("spectrum_", path.name)  # inventory only, no XML body
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guarded):
            products, tests, _ = ba.scan(self.sources, max_versions=6, max_products=24)
        self.assertEqual(products, self.products)
        self.assertEqual(tests, self.tests)
        self.assertEqual(snapshot(), before)

    def test_explicit_smoke_has_one_version_three_products(self):
        products, tests, meta = ba.scan(self.sources, max_versions=1, max_products=3, smoke=True)
        self.assertEqual(meta["versions"], 1)
        self.assertEqual(len(products), 3)
        self.assertEqual(len(tests), 5)

    def test_bounds_fail_without_truncating_dataset(self):
        with self.assertRaisesRegex(ValueError, "bound"):
            ba.scan(self.sources, max_versions=1, max_products=24)
        with self.assertRaisesRegex(ValueError, "bound"):
            ba.scan(self.sources, max_versions=6, max_products=2)

    def test_duplicate_report_product_rejected(self):
        report = self.sources[0].root / "_MultipleBugs_.NOB_1.ID_1/config.report.csv"
        rows = read_csv(report)
        write_csv(report, rows + rows[:1], rows[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ba.scan(self.sources, max_versions=6, max_products=30)

    def test_completed_report_only_supported_and_joined_to_labels(self):
        for source in self.sources:
            for root in (source.root, source.label_root):
                for report in list(root.glob("*/config.report.csv")):
                    report.with_name("config.report.csv.done").write_bytes(report.read_bytes())
                    report.unlink()
        products, tests, _ = ba.scan(self.sources, max_versions=6, max_products=24)
        self.assertTrue(all(p["feature_config_path"].endswith(".done") for p in products))
        write_csv(self.products_path, products, ba.PRODUCT_FIELDS)
        rows, _ = labels.import_labels(self.products_path, self.config)
        self.assertEqual(len(rows), 24)
        self.assertEqual(len(tests), len(self.tests))

    def test_conflicting_intermediate_report_rejected(self):
        version = self.sources[0].root / "_MultipleBugs_.NOB_1.ID_1"
        (version / "config.report.csv.done").write_text("different completed report")
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            ba.scan(self.sources, max_versions=6, max_products=24)

    def test_report_directory_mismatch_rejected(self):
        report = self.sources[0].root / "_MultipleBugs_.NOB_1.ID_1/config.report.csv"
        rows = read_csv(report)
        write_csv(report, rows[1:], rows[0])
        with self.assertRaisesRegex(ValueError, "directories/report differ"):
            ba.scan(self.sources, max_versions=6, max_products=24)

    def test_failed_test_in_passing_product_rejected(self):
        folder = self.sources[0].root / "_MultipleBugs_.NOB_1.ID_1/variants/model_m_ca4_0001/coverage/failed"
        folder.mkdir()
        (folder / "T.test1.coverage.xml").write_text("<coverage/>")
        with self.assertRaisesRegex(ValueError, "contradiction"):
            ba.scan(self.sources, max_versions=6, max_products=24)

    def test_test_identity_collision_rejected(self):
        folder = self.sources[0].root / "_MultipleBugs_.NOB_1.ID_1/variants/model_m_ca4_0002/coverage/failed"
        (folder / "T.test0.coverage.xml").write_text("<coverage/>")
        with self.assertRaisesRegex(ValueError, "Repeated/inconsistent"):
            ba.scan(self.sources, max_versions=6, max_products=24)

    def test_converted_fp_preserves_original_failure(self):
        converted = [r for r in self.label_rows if r["source_transformed_from_f"] == "1"]
        self.assertEqual(len(converted), 6)
        self.assertTrue(all(r["label"] == "F" and r["source_origin"] == "simulated" and r["origin"] == "unknown" for r in converted))
        self.assertFalse(validate_sealed(indexed(self.label_rows), indexed(self.products)))

    def test_unknown_origin_never_inferred_from_suite_size(self):
        self.assertTrue(all(r["origin"] == "unknown" for r in self.label_rows))
        self.assertEqual(self.label_audit["origin"], {"unknown": 24})

    def test_audited_nonconversion_and_original_pass_support_natural_origin(self):
        config = json.loads(self.config.read_text())
        for source in config["sources"]:
            source["label_source"] = "local_pulp_fault_location_proxy"
        self.config.write_text(json.dumps(config))
        rows, _ = labels.import_labels(self.products_path, self.config)
        self.assertEqual(sum(r["origin"] == "natural" for r in rows), 6)

    def test_source_report_mismatch_rejected(self):
        path = self.sources[0].label_root / "_MultipleBugs_.NOB_1.ID_1/config.report.csv"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "differs from original"):
            labels.import_labels(self.products_path, self.config)

    def test_unexplained_fail_to_fp_rejected(self):
        path = self.sources[0].label_root / "_MultipleBugs_.NOB_1.ID_1/variant_labels.csv"
        rows = read_csv(path)
        rows[2]["FP transformed from F"] = ""
        write_csv(path, rows, rows[0])
        with self.assertRaisesRegex(ValueError, "without explicit conversion"):
            labels.import_labels(self.products_path, self.config)

    def test_missing_label_stays_unknown_for_pass(self):
        path = self.sources[0].label_root / "_MultipleBugs_.NOB_1.ID_1/variant_labels.csv"
        path.unlink()
        rows, _ = labels.import_labels(self.products_path, self.config)
        selected = [r for r in rows if r["system_id"] == "BankAccountTP" and r["bug_version_id"].endswith("ID_1")]
        self.assertEqual(sum(r["label"] == "" for r in selected), 2)
        self.assertTrue(all(r["origin"] == "unknown" for r in selected))

    def test_inference_never_opens_label_or_raw_paths(self):
        original = Path.open
        def guarded(path, *args, **kwargs):
            self.assertEqual(path, self.products_path)
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guarded):
            rows = load_inputs(self.products_path)
        self.assertEqual(set(dataclasses.asdict(rows[0])), {"sample_id", "features", "observed_result"})
        self.assertEqual(rows[0].features, {"Base": True, "Lock": False})

    def test_inference_rejects_label_and_forbidden_config_columns(self):
        rows = read_csv(self.products_path)
        rows[0]["label"] = "FP"
        write_csv(self.products_path, rows, ba.PRODUCT_FIELDS)
        with self.assertRaisesRegex(ValueError, "sealed"):
            load_inputs(self.products_path)
        rows[0]["label"] = ""
        rows[0]["features_json"] = '{"buggy": true}'
        write_csv(self.products_path, rows, ba.PRODUCT_FIELDS)
        with self.assertRaisesRegex(ValueError, "forbidden"):
            load_inputs(self.products_path)

    def test_simulated_view_disabled(self):
        rows = read_csv(self.products_path)
        rows[0]["suite_view"] = "simulated"
        write_csv(self.products_path, rows, ba.PRODUCT_FIELDS)
        with self.assertRaisesRegex(ValueError, "Simulated"):
            load_inputs(self.products_path)

    def test_conversion_and_fault_metadata_cannot_hide_in_feature_config(self):
        rows = read_csv(self.products_path)
        for name in ("source_transformed_from_f", "label_source_path", "true_fault_location", "mutant_log", "bug_count"):
            rows[0]["features_json"] = json.dumps({"Base": True, name: True})
            write_csv(self.products_path, rows, ba.PRODUCT_FIELDS)
            with self.assertRaisesRegex(ValueError, "forbidden"):
                load_inputs(self.products_path)

    def test_train_targets_cannot_read_other_partition_or_all_labels(self):
        for partition in ("val", "test"):
            with self.assertRaisesRegex(ValueError, "cannot access"):
                labels.load_targets(self.root / "DOES_NOT_EXIST", self.products_path, self.split_path,
                                    partition=partition, purpose="train")
        with self.assertRaisesRegex(ValueError, "partition only"):
            labels.load_targets(self.labels_path, self.products_path, self.split_path, partition="train", purpose="train")
        labels.export_partition_labels(self.label_rows, self.products_path, [self.split_path], self.root / "sealed")
        for partition in ("train", "val", "test"):
            targets = labels.load_targets(self.root / f"sealed/grouped/{partition}.csv", self.products_path,
                                         self.split_path, partition=partition,
                                         purpose="train" if partition == "train" else "evaluate")
            assignments = indexed(read_csv(self.split_path))
            self.assertTrue(all(assignments[k]["split"] == partition for k in targets))

    def test_grouping_and_test_inheritance(self):
        assignments = indexed(read_csv(self.split_path))
        counts = splits.test_assignments(self.tests, indexed(self.products), assignments)
        self.assertEqual(sum(counts.values()), len(self.tests))
        groups = {}
        for k, row in assignments.items():
            groups.setdefault(k[:2], set()).add(row["split"])
        self.assertTrue(all(len(v) == 1 for v in groups.values()))
        broken = {k: dict(r) for k, r in assignments.items()}
        first = next(iter(broken))
        broken[first]["split"] = "test" if broken[first]["split"] == "train" else "train"
        with self.assertRaisesRegex(ValueError, "overlap"):
            splits.validate_split(indexed(self.products), broken)

    def test_split_determinism_independent_of_row_order(self):
        before = splits.make_split(self.products_path, seed=42)
        write_csv(self.products_path, reversed(self.products), ba.PRODUCT_FIELDS)
        self.assertEqual(before, splits.make_split(self.products_path, seed=42))

    def test_duplicate_product_key_rejected_before_split(self):
        write_csv(self.products_path, self.products + self.products[:1], ba.PRODUCT_FIELDS)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            splits.make_split(self.products_path)

    def test_training_only_guard_blocks_val_test_and_unknown(self):
        assignments = indexed(read_csv(self.split_path))
        train = [k for k, row in assignments.items() if row["split"] == "train"]
        forbidden = [k for k, row in assignments.items() if row["split"] != "train"]
        for operation in ("normalization", "vocabulary", "suspiciousness", "pretraining"):
            splits.require_training_keys(train, self.split_path, operation)
            for k in forbidden + [("missing", "v", "p")]:
                with self.assertRaisesRegex(ValueError, "TRAIN"):
                    splits.require_training_keys(train + [k], self.split_path, operation)

    def test_unavailable_four_systems_are_not_fabricated(self):
        out = self.root / "splits"
        result = splits.build_all(self.products_path, self.tests_path, out, 42)
        self.assertFalse(result["complete_six_systems"])
        self.assertEqual(sum(f["status"] == "unavailable" for f in result["folds"].values()), 4)
        self.assertFalse((out / "loso_GPL.csv").exists())
        for system in ("BankAccountTP", "Email"):
            assignments = indexed(read_csv(out / f"loso_{system}.csv"))
            splits.validate_split(indexed(self.products), assignments, holdout=system)

    def test_stale_missing_system_split_rejected(self):
        out = self.root / "stale_splits"
        out.mkdir()
        (out / "loso_GPL.csv").write_text("stale")
        with self.assertRaisesRegex(ValueError, "Stale split"):
            splits.build_all(self.products_path, self.tests_path, out, 42)

    def test_one_group_cannot_be_product_split(self):
        rows = [p for p in self.products if p["system_id"] == "BankAccountTP" and p["bug_version_id"].endswith("ID_1")]
        write_csv(self.products_path, rows, ba.PRODUCT_FIELDS)
        with self.assertRaisesRegex(ValueError, "at least 3"):
            splits.make_split(self.products_path)

    def test_real_style_validation_and_paper_gap(self):
        report = audit(self.products_path, self.tests_path, self.labels_path, [self.split_path], check_paths=True)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["systems"]["BankAccountTP"]["versions"], 3)
        self.assertEqual(report["systems"]["GPL"]["products"], 0)
        self.assertEqual(report["systems"]["BankAccountTP"]["missing_coverage_files"], 0)
        self.assertEqual(report["paper_total"]["F"] - report["paper_row_sums"]["F"], 109)

    def test_output_cannot_target_raw_data(self):
        with self.assertRaisesRegex(ValueError, "Output must be"):
            generated_dir(self.sources[0].root)


class SixSystemTests(unittest.TestCase):
    def test_all_six_loso_folds_with_explicitly_synthetic_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            products = [dict(zip(KEY, (s, f"synthetic-v{i}", f"p{j}"))) | {"suite_id": "synthetic"}
                        for s in SYSTEMS for i in range(3) for j in range(2)]
            tests = [p | {"test_id": "synthetic-test"} for p in products]
            write_csv(root / "products.csv", products, (*KEY, "suite_id"))
            write_csv(root / "tests.csv", tests, (*KEY, "suite_id", "test_id"))
            result = splits.build_all(root / "products.csv", root / "tests.csv", root / "out", 42)
            self.assertTrue(result["complete_six_systems"])
            self.assertEqual(len(result["folds"]), 7)
            for system in SYSTEMS:
                split = indexed(read_csv(root / f"out/loso_{system}.csv"))
                splits.validate_split(indexed(products), split, holdout=system)
                self.assertEqual({k[0] for k, r in split.items() if r["split"] == "test"}, {system})


if __name__ == "__main__":
    unittest.main()
