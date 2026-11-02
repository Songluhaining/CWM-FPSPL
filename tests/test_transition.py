import copy,unittest
import numpy as np
import torch
from cwm_fp.models.transition_model import StructureInput,TransitionModel,validate_input,signature,weighted_loss,view_edges
from cwm_fp.models.transition_data import hierarchy_weights
from cwm_fp.models.transition_train import trajectory_batches
from cwm_fp.models.transition_evaluate import choose_checkpoint,cat_metrics,bin_metrics,fit_baselines,baseline_predict

def example(n=3):
    x=np.zeros((2+n+1,22),np.uint8);x[:2,1]=1;x[:2,6]=1;x[:2,14]=1
    x[2:2+n,1]=1;x[2:2+n,19]=1;x[2:2+n,5]=1;x[2:1+n,17]=1;x[-1,2]=1;x[-1,18]=1
    edges=[(0,1,1),(2,len(x)-1,3),(len(x)-1,3,4),(2,3,2)]+[(i,i+1,6) for i in range(2,1+n)]
    return StructureInput(x,np.array([[s for s,d,r in edges],[d for s,d,r in edges]],np.int32),np.array([r for s,d,r in edges],np.uint8),1+n)

class TransitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_shared_static_forward_gradient(self):
        for kind in ('no_dep','rgcn'):
            torch.manual_seed(1);m=TransitionModel(kind);ins=[example(),example(5),example()]
            a=m(ins);sum(z.sum() for z in a).backward();grads={k:p.grad.clone() for k,p in m.named_parameters()}
            m.zero_grad();b=m(ins,share_static=False);sum(z.sum() for z in b).backward()
            for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-6)
            for k,p in m.named_parameters():torch.testing.assert_close(grads[k],p.grad,atol=2e-5,rtol=2e-5)
    def test_batch_padding_and_trajectory_isolation(self):
        for kind in ('seq_gru','no_dep','rgcn'):
            torch.manual_seed(4);m=TransitionModel(kind).eval();a=m([example()]);b=m([example(12),example(),example(5)])
            for x,y in zip(a,b):torch.testing.assert_close(x[0],y[1],atol=2e-6,rtol=2e-6)
    def test_node_whitelist_and_query(self):
        inp=example();validate_input(inp)
        with self.assertRaises(TypeError):TransitionModel('rgcn')([{'x':inp,'target':0}])
        with self.assertRaises(ValueError):validate_input(StructureInput(inp.x,inp.edge_index,inp.edge_type,2))
        bad=inp.x.copy();bad[0,17]=1
        with self.assertRaises(ValueError):validate_input(StructureInput(bad,inp.edge_index,inp.edge_type,inp.anchor))
    def test_all_dependency_bypasses_removed(self):
        inp=example();ei,et=view_edges(inp,'no_dep');self.assertEqual(set(et),{6})
        self.assertEqual(len(inp.x),6)
        modified=StructureInput(inp.x,inp.edge_index[:,:-1],inp.edge_type[:-1],inp.anchor)
        self.assertNotEqual(signature(inp,'rgcn'),signature(modified,'rgcn'))
        self.assertEqual(signature(inp,'seq_gru'),signature(modified,'seq_gru'))
    def test_same_graph_initialization(self):
        torch.manual_seed(8);a=TransitionModel('no_dep');torch.manual_seed(8);b=TransitionModel('rgcn')
        self.assertTrue(all(torch.equal(a.state_dict()[k],v) for k,v in b.state_dict().items()))
    def test_hierarchical_weights_empty_invalid_and_last_batch(self):
        rows=[];ts=[]
        for ver,npd in [('v1',2),('v2',1)]:
            for pd in range(npd):
                for tr in range(pd+1):
                    tid=len(rows)
                    for pos in range(tr+4):
                        rows.append({'identity':['S',ver,str(pd),str(tr)],'trace_index':tid,'partition':'train'})
                        ts.append({'A':0 if pos<3 else -1,'B':pos%2 if pos<2 else -1,'covered':pos%2==0})
        w=hierarchy_weights(rows,ts);np.testing.assert_allclose(w.sum(0),1)
        for ver in ('v1','v2'):np.testing.assert_allclose(w[[r['identity'][1]==ver for r in rows]].sum(0),.5)
        batches=trajectory_batches(rows,np.random.default_rng(0));self.assertEqual(sorted(sum(batches,[])),list(range(len(rows))))
        rows[0]['partition']='val'
        with self.assertRaises(ValueError):hierarchy_weights(rows,ts)
    def test_microbatch_gradient_equals_full_and_invalid_not_negative(self):
        torch.manual_seed(3);a=torch.randn(7,9,requires_grad=True);b=torch.randn(7,requires_grad=True)
        targets=torch.tensor([[1,0],[2,1],[3,-1],[4,-1],[5,0],[-1,-1],[0,1]],dtype=torch.float32)
        weights=torch.zeros((7,3));weights[[0,2],0]=.5;weights[[1,3,4,6],1]=.25;weights[[0,1,4,6],2]=.25
        full,_=weighted_loss((a,b),targets,weights,2);full.backward();ga=a.grad.clone();gb=b.grad.clone();a.grad=None;b.grad=None
        for start in range(0,7,3):
            s=slice(start,start+3);v,_=weighted_loss((a[s],b[s]),targets[s],weights[s],2);v.backward()
        torch.testing.assert_close(a.grad,ga);torch.testing.assert_close(b.grad,gb);self.assertEqual(float(a.grad[5].abs().sum()),0)
    def test_empty_parts_keep_fixed_divisor(self):
        a=torch.zeros((2,9),requires_grad=True);b=torch.zeros(2,requires_grad=True)
        loss,parts=weighted_loss((a,b),torch.tensor([[-1.,1.],[-1.,-1.]]),torch.tensor([[0.,0.,1.],[0.,0.,0.]]))
        self.assertAlmostEqual(float(loss),np.log(2)/3,places=6);loss.backward();self.assertEqual(float(b.grad[1]),0)
    def test_selection_fixed_tolerance_and_empty(self):
        self.assertEqual(choose_checkpoint([(5,1.),(10,.9999995),(20,.99999)]),(20,.99999))
        self.assertEqual(choose_checkpoint([(5,1.),(10,.9999995),(20,1.1)]),(5,1.))
        self.assertEqual(choose_checkpoint([(5,None),(20,None)]),(20,None))
    def test_metrics_and_train_only_baseline(self):
        self.assertEqual(cat_metrics([0,1],np.eye(9)[:2])['accuracy'],1.)
        self.assertEqual(bin_metrics([0,1],np.array([.2,.8]))['balanced_accuracy'],1.)
        rows=[{'category':0,'signatures':{k:'x' for k in ('rgcn','no_dep','seq_gru')}}]*2
        fit=fit_baselines(rows,[{'A':1,'B':0},{'A':2,'B':1}]);pa,pb=baseline_predict(rows,fit,'Exact-rgcn')
        np.testing.assert_allclose(pa[0,1:3],[.5,.5]);np.testing.assert_allclose(pb,.5)
        self.assertAlmostEqual(fit[1][0,1],2/11)
    def test_final_trajectory_batch(self):
        rows=[{'trace_index':i//2} for i in range(18)];b=trajectory_batches(rows,np.random.default_rng(1))
        self.assertEqual([len(v) for v in b],[8,8,2]);self.assertEqual(sorted(sum(b,[])),list(range(18)))

if __name__=='__main__':unittest.main()
