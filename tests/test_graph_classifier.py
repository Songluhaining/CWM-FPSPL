"""Meaningful graph, evidence, gradient and supervised boundary checks."""
import copy
import json
import sys
import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest("Phase04 graph tests require the separately recorded torch environment") from exc

from cwm_fp.data.common import KEY, ROOT, digest, write_csv, write_json
from cwm_fp.models.graph_data import load_corpus
from cwm_fp.models.graph_classifier import GraphClassifier
from cwm_fp.models.graph_encoder import FEATURES, NODE_TYPES, RELATIONS, GraphEncoder, Limits, RelationLayer, collate, encode_test, mean_pool
from cwm_fp.models.graph_smoke import fixture, synthetic_graph
from cwm_fp.models.train_graph import TrainOptions, check_partitions, configuration, fit, predict, readiness
from cwm_fp.models.graph_metrics import binary_metrics
from cwm_fp.models.graph_attribution import attribution


class GraphTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(7)
        self.graph = synthetic_graph(("s", "v", "p"), "test", 1)

    def test_white_list_and_no_raw_values(self):
        a = encode_test(self.graph, "both")
        g = copy.deepcopy(self.graph)
        g["system_id"], g["bug_version_id"], g["product_id"] = "other", "NOB_99", "known_FP_name"
        for n in g["nodes"]:
            n["attrs"].update(value=999999, composed_line=123, canonical_statement_id="oracle_name_not_embedded")
        b = encode_test(g, "both")
        self.assertTrue(torch.equal(a.x, b.x)); self.assertEqual(a.x.shape[1], len(FEATURES))

    def test_forbidden_fields_rejected_recursively(self):
        for k in ("label", "label_provenance", "buggy", "removed_tests", "origin", "posterior"):
            g = copy.deepcopy(self.graph); g["nodes"][0]["attrs"][k] = "FP"
            with self.assertRaises(ValueError): encode_test(g, "both")

    def test_rename_ids_preserves_tensors(self):
        a = encode_test(self.graph, "both"); g = copy.deepcopy(self.graph)
        mapping = {n["id"]: str(i+1000) for i, n in enumerate(g["nodes"])}
        for n in g["nodes"]: n["id"] = mapping[n["id"]]
        for e in g["edges"]: e["source"], e["target"] = mapping[e["source"]], mapping[e["target"]]
        b = encode_test(g, "both")
        for name in ("x", "edge_index", "edge_type", "coverage"): self.assertTrue(torch.equal(getattr(a,name), getattr(b,name)))

    def test_config_requires_constraint_verification(self):
        g = copy.deepcopy(self.graph); g["quality"]["constraints_verified"] = False
        self.assertTrue(torch.equal(encode_test(g, "both").x[:, -2:], torch.zeros(len(g["nodes"]), 2)))
        g["quality"]["constraints_verified"] = True; g["nodes"][0]["attrs"]["enabled"] = "False"
        with self.assertRaises(ValueError): encode_test(g, "both")

    def test_evidence_gate_and_preview(self):
        for field, value in (("eligible_graph_structure", False), ("verified_tier", "T0"), ("canonical_mapping_complete", False), ("unsupported_dynamic", ["array"])):
            g = copy.deepcopy(self.graph); g["quality"][field] = value
            with self.assertRaises(ValueError): encode_test(g, "both")
        g = copy.deepcopy(self.graph); g["quality"]["canonical_mapping_complete"] = False
        self.assertTrue(encode_test(g, "both", preview=True).metadata["preview_only"])
        g["quality"]["unsupported_dynamic"] = ["exception"]
        with self.assertRaises(ValueError): encode_test(g, "both", preview=True)

    def test_wrong_edge_evidence_and_order_rejected(self):
        g = copy.deepcopy(self.graph); g["edges"][2]["evidence"] = "static"
        with self.assertRaises(ValueError): encode_test(g, "both")
        g["edges"][2]["evidence"] = "dynamic"; g["edges"][2]["definition_event"] = 2
        with self.assertRaises(ValueError): encode_test(g, "both")

    def test_unknown_endpoint_type_and_duplicate_node(self):
        for alteration in (lambda g:g["edges"][0].update(source="absent"), lambda g:g["nodes"].append(g["nodes"][0]), lambda g:g["nodes"][0].update(type="bug")):
            g = copy.deepcopy(self.graph); alteration(g)
            with self.assertRaises(ValueError): encode_test(g, "both")

    def test_ablation_removes_indirect_data_paths(self):
        t = encode_test(self.graph, "cdg")
        self.assertFalse({"data_dep", "produces", "consumes", "temporal"}.intersection(RELATIONS[i] for i in t.edge_type.tolist()))
        self.assertFalse(t.x[:, NODE_TYPES.index("value")].any())
        t = encode_test(self.graph, "ddg")
        self.assertFalse({"feature_controls", "control_dep", "temporal"}.intersection(RELATIONS[i] for i in t.edge_type.tolist()))
        self.assertEqual(encode_test(self.graph, "coverage").edge_type.numel(), 0)

    def test_sampling_loss_accounting(self):
        a = encode_test(self.graph, "both", Limits(max_nodes=7, max_events=1))
        m = a.metadata
        self.assertLessEqual(len(a.x), 7); self.assertEqual(m["events_after"], 1)
        self.assertAlmostEqual(m["lost_event_fraction"], 2/3)
        self.assertEqual(m["edges_after_sampling"], sum(e["source"] in {n["id"] for n in m["node_records"]} and e["target"] in {n["id"] for n in m["node_records"]} for e in self.graph["edges"]))
        products, _, _ = fixture("both", Limits(max_tests=1))
        self.assertEqual(products[0].audit["lost_test_fraction"], .5)
        self.assertEqual(products[0].audit["lost_event_fraction"], .5)
        capped=encode_test(self.graph,"both",Limits(max_edges=2))
        self.assertEqual(capped.metadata["edges_after_sampling"],2)
        self.assertAlmostEqual(capped.metadata["lost_edge_fraction"],.8)

    def test_value_callback_origin_is_evidence_not_label_provenance(self):
        g=copy.deepcopy(self.graph)
        value=next(n for n in g["nodes"] if n["type"]=="value")
        value["attrs"]["origin"]="producer_stack_callback"
        a=encode_test(g,"both")
        self.assertTrue(torch.equal(a.x,encode_test(self.graph,"both").x))
        value["attrs"]["origin"]="natural"
        with self.assertRaises(ValueError):encode_test(g,"both")

    def test_hierarchical_pooling_is_not_all_node_average(self):
        x = torch.tensor([[0.], [10.], [10.], [20.]])
        t = mean_pool(x, torch.tensor([0, 1, 1, 2]), 3)
        p = mean_pool(t, torch.tensor([0, 0, 1]), 2)
        self.assertEqual(p.tolist(), [[5.], [20.]])

    def test_batch_has_no_cross_test_edges(self):
        products, _, _ = fixture("both"); b = collate(products[:3])
        self.assertTrue(torch.equal(b.node_test[b.edge_index[0]], b.node_test[b.edge_index[1]]))
        self.assertEqual(b.n_products, 3)

    def test_node_permutation_invariance(self):
        products, _, _ = fixture("both"); p = products[0]; a = collate([p])
        gs = []
        for t in p.tests:
            g = synthetic_graph(tuple(t.metadata["identity"][:3]), t.metadata["identity"][-1], 0)
            g["nodes"].reverse(); gs.append(encode_test(g, "both"))
        b = collate([replace(p, tests=gs)])
        for kind in ("agnostic", "rgcn"):
            model = GraphClassifier(kind)
            self.assertTrue(torch.allclose(model(a), model(b), atol=1e-6))

    def test_agnostic_ignores_node_relation_types(self):
        products, _, _ = fixture("both"); b = collate(products[:1]); model = GraphClassifier("agnostic")
        x = b.x.clone(); x[:, :len(NODE_TYPES)] = torch.rand_like(x[:, :len(NODE_TYPES)])
        self.assertTrue(torch.equal(model(b), model(replace(b, x=x, edge_type=(b.edge_type+1) % len(RELATIONS)))))

    def test_graph_logits_can_distinguish_topology(self):
        products, _, _ = fixture("both"); b = collate(products[:2])
        self.assertTrue(torch.equal(products[0].tests[0].x, products[1].tests[0].x))
        self.assertTrue(torch.equal(products[0].tests[0].coverage, products[1].tests[0].coverage))
        values = GraphClassifier("rgcn")(b)
        self.assertGreater(abs(float(values[0]-values[1])), 1e-7)

    def test_forward_backward_all_models(self):
        products, _, _ = fixture("both"); b = collate(products[:2])
        for kind in ("coverage", "agnostic", "rgcn"):
            model = GraphClassifier(kind); loss = torch.nn.functional.binary_cross_entropy_with_logits(model(b), torch.tensor([0., 1.]))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None), 0.)

    def test_relation_gradient_matches_finite_difference(self):
        layer = RelationLayer(2, 2).double()
        x = torch.tensor([[.4,.7],[1.1,.3],[.2,.9]], dtype=torch.double, requires_grad=True)
        edge = torch.tensor([[0,1],[1,2]]); relation = torch.tensor([0,1])
        gate = torch.ones(2, dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda a,b:layer(a,edge,relation,b),(x,gate)))
        layer(x,edge,relation).sum().backward(); actual = layer.bases.grad[0,0,0].item()
        with torch.no_grad():
            initial = layer.bases[0,0,0].item(); step=1e-5
            layer.bases[0,0,0] = initial+step; plus=layer(x,edge,relation).sum().item()
            layer.bases[0,0,0] = initial-step; minus=layer(x,edge,relation).sum().item()
            layer.bases[0,0,0] = initial
        self.assertAlmostEqual(actual,(plus-minus)/(2*step),places=6)

    def test_train_normalizer_rejects_heldout_failed_preview(self):
        products, _, assignments = fixture("coverage", coverage_signal=True)
        train = [p for p in products if p.partition == "train"]
        for p in [next(p for p in products if p.partition=="val"), replace(train[0], observed_result="FAIL"), replace(train[0], audit={"preview_only":True})]:
            with self.assertRaises(ValueError): GraphClassifier("coverage").fit_normalization([p], assignments)
        model=GraphClassifier("coverage"); model.fit_normalization(train,assignments)
        self.assertEqual(len(model.normalization_audit["contributors"]),8)
        with self.assertRaises(ValueError): model.fit_normalization(train,assignments)

    def test_partition_and_test_ownership_checks(self):
        products, _, assignments = fixture("both")
        p=products[0]
        with self.assertRaises(ValueError): check_partitions([replace(p, observed_result="FAIL")],assignments)
        with self.assertRaises(ValueError): check_partitions([replace(p, tests=products[1].tests)],assignments)
        mixed=assignments | {(p.identity[0],p.identity[1],"new"):"val"}
        with self.assertRaises(ValueError): check_partitions(products,mixed)

    def test_frozen_checkpoint_and_validation_early_stop(self):
        products, labels, assignments=fixture("both")
        train=[p for p in products if p.partition=="train"]; val=[p for p in products if p.partition=="val"]
        seen=[]
        def evaluator(rows, remaining):
            self.assertTrue(all(r["partition"]=="val" for r in rows)); seen.append(True)
            return {"log_loss":float(len(seen)),"pr_auc":None}
        model,report=fit("rgcn",train,{p.identity:labels[p.identity] for p in train},val,assignments,evaluator,TrainOptions(max_epochs=6,patience=2))
        self.assertEqual(report["selected_epoch"],1); self.assertEqual(report["epochs"],3)
        self.assertEqual(report["stop_reason"],"validation_patience")
        self.assertEqual(report["class_weights"],{"TP":1.,"FP":1.})
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"model.pt"; torch.save(model.state_dict(),path)
            restored=GraphClassifier("rgcn"); restored.load_state_dict(torch.load(path,weights_only=True))
            self.assertEqual(predict(model,val),predict(restored,val))

    def test_training_refuses_single_class_or_wrong_targets(self):
        products, _, assignments=fixture("both"); train=[p for p in products if p.partition=="train"]; val=[p for p in products if p.partition=="val"]
        for targets in ({p.identity:"TP" for p in train},{p.identity:"F" for p in train},{}):
            with self.assertRaises(ValueError): fit("rgcn",train,targets,val,assignments,lambda *_:None)

    def test_training_time_and_parameter_limits(self):
        products, labels, assignments=fixture("both"); train=[p for p in products if p.partition=="train"]; val=[p for p in products if p.partition=="val"]
        targets={p.identity:labels[p.identity] for p in train}
        with self.assertRaises(TimeoutError): fit("rgcn",train,targets,val,assignments,lambda *_:None,TrainOptions(max_seconds=1e-12))
        with self.assertRaises(ValueError): fit("rgcn",train,targets,val,assignments,lambda *_:None,TrainOptions(max_parameters=1))
        with self.assertRaises(ValueError): TrainOptions(max_epochs=501)

    def test_attribution_does_not_change_model(self):
        products, _, _=fixture("both"); model=GraphClassifier("rgcn")
        before=predict(model,products[:1])
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/"a"; report=attribution(model,products[0],out,synthetic=True)
            self.assertTrue(out.with_suffix(".svg").is_file()); self.assertTrue(report["synthetic"])
        self.assertEqual(before,predict(model,products[:1]))

    def test_real_data_blocked_without_labels(self):
        c=configuration(ROOT/"configs/graph_classifier.json")
        if not c["graphs"].is_file(): self.skipTest("Local measured phase03 corpus not present")
        result=readiness(c)
        self.assertEqual(result["status"],"blocked")
        for fold,a in result["folds"].items():
            self.assertEqual((a["graphs"],a["version_groups"],a["original_PASS_with_graphs"]),(6,1,1))
            self.assertFalse(a["labels_read"]); self.assertEqual(sum(a["eligible_by_partition"].values()),0)
        self.assertEqual(result["folds"]["grouped"]["raw_PASS_by_partition"],{"val":1})

    def test_original_suite_join_and_partial_cohort_exclusion(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); products=[]; tests=[]; splits=[]; graphs=[]
            for part in ("train","val","test"):
                for i in range(2):
                    k=("s",part+"_version",str(i)); lineage=dict(zip(KEY,k))
                    products.append(lineage|{"observed_result":"PASS","sample_id":digest(k),"suite_view":"original","suite_id":digest(k),"label":"","label_provenance":"","origin":"unknown"})
                    splits.append(lineage|{"split":part})
                    for j in range(2):
                        tests.append(lineage|{"test_id":str(j),"result":"PASS","suite_id":digest(k)})
                        graphs.append(synthetic_graph(k,str(j),i))
            for name,rows in (("products",products),("tests",tests),("splits",splits)):
                write_csv(root/(name+".csv"),rows,tuple(rows[0]))
            def load():
                (root/"graphs.jsonl").write_text("".join(json.dumps(g)+"\n" for g in graphs),encoding="utf-8")
                return load_corpus(root/"products.csv",root/"tests.csv",root/"graphs.jsonl",root/"splits.csv")
            audit=load()[-1]; self.assertTrue(audit["ready"])
            removed=graphs.pop()
            _,_,_,eligible,audit=load()
            self.assertEqual(len(eligible),5); self.assertTrue(audit["ready"])
            missing=next(r for r in audit["products"] if r["identity"]==[removed[k] for k in KEY])
            self.assertIn("incomplete_original_test_collection",missing["reasons"])
            graphs.append(copy.deepcopy(graphs[0]))
            with self.assertRaises(ValueError):load()

    def test_training_is_seed_reproducible(self):
        products,labels,assignments=fixture("both")
        train=[p for p in products if p.partition=="train"]; val=[p for p in products if p.partition=="val"]
        def evaluate(rows,remaining):
            return binary_metrics([int(labels[tuple(r["identity"])]=="FP") for r in rows],[r["y_score"] for r in rows])
        opts=TrainOptions(max_epochs=4)
        m1,r1=fit("rgcn",train,{p.identity:labels[p.identity] for p in train},val,assignments,evaluate,opts)
        m2,r2=fit("rgcn",train,{p.identity:labels[p.identity] for p in train},val,assignments,evaluate,opts)
        self.assertEqual(predict(m1,val),predict(m2,val));self.assertEqual(r1["history"],r2["history"])

    def test_separate_training_validation_and_once_only_evaluation_processes(self):
        # Fully artificial corpus. No held-out test target file even exists
        # during fitting; the evaluator receives it only after model freeze.
        processed=ROOT/"data/processed/graph_classifier"; processed.mkdir(parents=True,exist_ok=True)
        metrics=ROOT/"outputs/metrics"; metrics.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=processed,prefix="phase04_unit_") as d, tempfile.TemporaryDirectory(dir=metrics,prefix="phase04_unit_") as m:
            root=Path(d).resolve(); metric_root=Path(m).resolve()
            self.assertTrue(root.is_relative_to(processed.resolve())); self.assertTrue(metric_root.is_relative_to(metrics.resolve()))
            products=[]; tests=[]; splits=[]; graphs=[]; targets={p:[] for p in ("train","val","test")}
            for part in targets:
                for i in range(2):
                    k=("SyntheticOnly",part+"_version",str(i)); lineage=dict(zip(KEY,k)); suite=digest(k)
                    products.append(lineage|{"observed_result":"PASS","sample_id":digest(k),"suite_view":"original","suite_id":suite,"label":"","label_provenance":"","origin":"unknown"})
                    splits.append(lineage|{"split":part}); tests.append(lineage|{"test_id":"t","result":"PASS","suite_id":suite})
                    graphs.append(synthetic_graph(k,"t",i))
                    targets[part].append(lineage|{"label":"FP" if i else "TP","suite_id":suite,"origin":"unknown"})
            for name,rows in (("products",products),("tests",tests),("splits",splits)):
                write_csv(root/(name+".csv"),rows,tuple(rows[0]))
            (root/"graphs.jsonl").write_text("".join(json.dumps(g)+"\n" for g in graphs),encoding="utf-8")
            for part in ("train","val"):
                write_csv(root/"labels/grouped"/(part+".csv"),targets[part],tuple(targets[part][0]))
            c=json.loads((ROOT/"configs/graph_classifier.json").read_text(encoding="utf-8"))
            c.update({k:str(root/(k+".csv")) for k in ("products","tests")})
            c.update(graphs=str(root/"graphs.jsonl"),split_dir=str(root),labels=str(root/"labels"),folds=["grouped"],work_dir=str(root/metric_root.name))
            c["training"].update(max_epochs=3,max_seconds=30.)
            config=root/"config.json";write_json(config,c)
            env=os.environ.copy();env["PYTHONPATH"]=str(ROOT/"src");env["PYTHONUTF8"]="1"
            def command(args):return subprocess.run([sys.executable,*args],cwd=ROOT,env=env,capture_output=True,text=True,encoding="utf-8",timeout=60)
            trained=command(["-m","cwm_fp.models.train_graph","--mode","train","--config",str(config)])
            self.assertEqual(trained.returncode,0,trained.stdout+trained.stderr)
            dest=Path(c["work_dir"])/"grouped/20261030/rgcn_both"
            self.assertTrue((dest/"frozen.json").is_file());self.assertFalse((root/"labels/grouped/test.csv").exists())
            write_csv(root/"labels/grouped/test.csv",targets["test"],tuple(targets["test"][0]))
            args=["-m","cwm_fp.models.graph_evaluator","--config",str(config),"--fold","grouped","--partition","test","--predictions",str(dest/"test_predictions.json")]
            evaluated=command(args);self.assertEqual(evaluated.returncode,0,evaluated.stdout+evaluated.stderr)
            self.assertEqual(json.loads(evaluated.stdout)["metrics"]["n"],2)
            repeated=command(args);self.assertNotEqual(repeated.returncode,0);self.assertIn("FileExistsError",repeated.stderr)
