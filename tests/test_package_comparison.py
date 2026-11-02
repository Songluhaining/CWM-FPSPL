"""Artificial, bounded tests; no real held-out model calls or target reads."""
import copy
from dataclasses import replace
import math
from pathlib import Path
import subprocess
import sys
import unittest
import torch
from cwm_fp.models.graph_encoder import collate,RelationLayer
from cwm_fp.models.graph_classifier import GraphClassifier
from cwm_fp.models.graph_metrics import diagnostic_metrics
from cwm_fp.models.package_comparison import product_statistics,select_checkpoint,load_protocol,predict_records,representation_summary
from test_package_training import product


class ComparisonTests(unittest.TestCase):
    def test_nodeonly_ignores_edges_preserves_features(self):
        torch.manual_seed(1);m=GraphClassifier('nodeonly',16,2);batch=collate([product()]);seen=[]
        h=m.encoder.input.register_forward_pre_hook(lambda mod,args:seen.append(args[0].clone()))
        x=m(batch);y=m(replace(batch,edge_index=batch.edge_index[:,:0],edge_type=batch.edge_type[:0]));h.remove()
        self.assertTrue(torch.equal(x,y));self.assertTrue(torch.equal(seen[0],batch.x))
        x.sum().backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
    def test_untyped_merges_only_relations(self):
        m=GraphClassifier('untyped',16,2);b=collate([product()]);seen=[]
        h=m.encoder.input.register_forward_pre_hook(lambda mod,args:seen.append(args[0].clone()))
        a=m(b);c=m(replace(b,edge_type=(b.edge_type+1)%7));h.remove()
        self.assertTrue(torch.equal(a,c));self.assertTrue(torch.equal(seen[0],b.x))
    def test_message_direction_and_multiplicity(self):
        layer=RelationLayer(1,1)
        with torch.no_grad():
            layer.self_map.weight.zero_();layer.self_map.bias.zero_();layer.bases.fill_(.5);layer.coeff.fill_(1)
        x=torch.tensor([[1.],[3.],[0.]])
        edges=torch.tensor([[0,1],[2,2]]);types=torch.zeros(2,dtype=torch.long)
        a=layer(x,edges,types);b=layer(x,torch.tensor([[0,0,1],[2,2,2]]),torch.zeros(3,dtype=torch.long))
        self.assertAlmostEqual(2.,a[2].item());self.assertAlmostEqual(5/3,b[2].item(),places=6)
        self.assertEqual(0.,layer(x,edges.flip(0),types)[2].item())
    def test_parameter_counts(self):
        for k,count in [('nodeonly',1201),('untyped',2229),('rgcn',2253)]:
            self.assertEqual(count,sum(p.numel() for p in GraphClassifier(k,16,2).parameters()))
    def test_statistics_test_equal_weight_and_fixed_columns(self):
        a=product();a.tests[0].metadata['events']=[{}]*a.tests[0].metadata['events_before']
        b=copy.deepcopy(a.tests[0]);a.tests=[a.tests[0],b]
        row=product_statistics([a])[0]
        self.assertEqual(16,len(row));self.assertAlmostEqual(math.log(3),row[0]);self.assertAlmostEqual(1.,sum(row[-5:]))
        self.assertAlmostEqual(math.log1p(len(b.x)),row[1])
    def test_earlier_candidate_wins_tolerance(self):
        a={'epoch':10,'metrics':{'log_loss':.5}};b={'epoch':50,'metrics':{'log_loss':.5-5e-7}}
        self.assertIs(a,select_checkpoint([b,a]));b['metrics']['log_loss']=.49;self.assertIs(b,select_checkpoint([a,b]))
    def test_single_class_metrics_explicit(self):
        rows=[{'identity':['s','v','p'],'y_score':.8}];m=diagnostic_metrics(rows,{('s','v','p'):'FP'})
        self.assertAlmostEqual(-math.log(.8),m['log_loss']);self.assertIsNone(m['roc_auc']);self.assertIsNone(m['average_precision'])
        self.assertEqual([[0,0],[0,1]],m['confusion_matrix']);self.assertNotIn('pr_auc',m)
    def test_tied_AP_and_zero_denominator(self):
        rows=[{'identity':['s','v',str(i)],'y_score':.3} for i in range(2)]
        m=diagnostic_metrics(rows,{('s','v','0'):'FP',('s','v','1'):'TP'})
        self.assertEqual(.5,m['average_precision']);self.assertEqual(.5,m['roc_auc']);self.assertEqual(0,m['fp_precision'])
        self.assertEqual(0,m['fp_recall']);self.assertEqual(1,m['versions'])
    def test_duplicate_prediction_rejected(self):
        r={'identity':['s','v','p'],'y_score':.7}
        with self.assertRaises(ValueError):diagnostic_metrics([r,r],{('s','v','p'):'FP'})
    def test_no_test_predictions_or_heldout_representations(self):
        p=product();p.partition='test'
        with self.assertRaises(ValueError):predict_records(None,[p])
        p.partition='val'
        with self.assertRaises(ValueError):representation_summary(None,[p])
    def test_protocol_hash_required_before_any_read(self):
        with self.assertRaises(ValueError):load_protocol(Path('does_not_exist'),None)
    def test_actual_evaluator_target_guard(self):
        code="""
from pathlib import Path
from tempfile import TemporaryDirectory
from cwm_fp.models.package_comparison import label_guard
with TemporaryDirectory() as d:
 p=Path(d)/'labels/grouped/val.csv';p.parent.mkdir(parents=True);p.write_text('artificial')
 reads,blocked=label_guard('04R1_validation_only',p)
 assert p.read_text()=='artificial'
 for q in [p.with_name('test.csv'),p.with_name('train.csv'),p.parent.parent/'loso_Email/train.csv']:
  try:q.read_text()
  except PermissionError:pass
  else:raise AssertionError('allowed forbidden target')
 assert sum(reads.values())==1
"""
        r=subprocess.run([sys.executable,'-X','utf8','-c',code],capture_output=True,text=True,timeout=20)
        self.assertEqual(0,r.returncode,r.stdout+r.stderr)


if __name__=='__main__':unittest.main()
