"""Artificial records only; never counted as research trajectories."""
import copy
import unittest
import numpy as np
from cwm_fp.models.prefix_observer import ObservationReplay, PrefixBuilder, tokens, serial_static, validate_snapshot
from cwm_fp.models.prefix_dataset import targets, check_prefix

D=('Demo','run','()V')
def fixture():
    ops=[(3,1,[1]),(59,2,[2]),(26,3,[3]),(153,4,[4,5]),(4,5,[5]),(177,-1,[-1])]
    static={'methods':{D:{i:{'opcode':op,'bci':i,'next':n,'successors':set(ss),'line':1} for i,(op,n,ss) in enumerate(ops)}},
        'fields':{},'controls':{D:{(3,4)}},'method_flags':{D:8},'exception_tables':{D:{'code_length':6,'handler_count':0,'handlers':[]}}}
    ids=[{'locator':dict(zip(('declaring_class','method','descriptor','bytecode_offset'),(*D,i))),
        'opcode':op,'provenance_type':'SOURCE_BACKED','scope':'product'} for i,(op,_,_) in enumerate(ops)]
    b=PrefixBuilder(static,serial_static(static,ids),{'methods':{}})
    text='###Demo::run\n'+''.join([
        'opcode=3(iconst_0),pushnum=1,lineinfo=Demo#run#()V#1#0,nextinst=1,thread=1,stack=I:0\n',
        'opcode=59(istore_0),popnum=1,store=0,lineinfo=Demo#run#()V#1#1,nextinst=2,thread=1\n',
        'opcode=26(iload_0),pushnum=1,load=0,lineinfo=Demo#run#()V#1#2,nextinst=3,thread=1,stack=I:0\n',
        'opcode=153(ifeq),popnum=1,lineinfo=Demo#run#()V#1#3,nextinst=4,thread=1\n',
        'opcode=177(return),popnum=-1,lineinfo=Demo#run#()V#1#5,nextinst=-1,thread=1\n'])
    return b,text

class PrefixObserverTests(unittest.TestCase):
    def setUp(self):self.b,self.text=fixture()
    def snap(self,kind='instruction',index=0):
        obs=ObservationReplay('Demo.run');n=0
        for t in tokens(self.text):
            obs.push(t)
            if t['kind']==kind:
                if n==index:return self.b.build(obs),obs,t
                n+=1
        self.fail('missing artificial checkpoint')
    def test_same_line_callback_not_previsible(self):
        pre,_,_=self.snap();post,_,_=self.snap('callback')
        self.assertEqual(pre.model['value_observed'].sum(),0);self.assertEqual(post.model['value_observed'].sum(),1)
        self.assertEqual(post.audit['values'][0]['observed_at'],[2,1])
    def test_real_zero_distinct_from_missing(self):
        s,_,_=self.snap('callback');j=s.audit['values'][0]['local_node']
        self.assertEqual(s.model['value'][j,0],0);self.assertTrue(s.model['value_observed'][j,0])
    def test_no_static_executed_aggregation(self):
        s,_,_=self.snap('instruction',4);self.assertFalse(s.model['x'][:6,17].any())
    def test_unexecuted_static_branch_preserved(self):
        s,_,_=self.snap();self.assertEqual(s.audit['static_nodes'],6);self.assertEqual(s.model['edge_index'][:,0].tolist(),[3,4])
    def test_current_instruction_has_no_result(self):
        s,_,_=self.snap('instruction',2);self.assertFalse(s.model['x'][8,17]);self.assertFalse(any(v['definition_event']==2 for v in s.audit['values']))
    def test_branch_target_only_before_result(self):
        obs=ObservationReplay.from_text(self.text,'Demo.run');t=targets(obs,self.b.static)[3]
        self.assertEqual(t['B'],'jump');self.assertGreater(t['B_available_at'],t['anchor'])
    def test_root_return_is_not_completion(self):
        s,obs,_=self.snap('instruction',4);self.assertIsNone(obs.full[-1]['completed_clock']);self.assertFalse(targets(obs,self.b.static)[-1]['A_mask'])
    def test_unknown_numeric_is_masked(self):
        s,_,_=self.snap('instruction',3);self.assertTrue(any(v['value'] is None for v in s.audit['values']))
        self.assertTrue(np.all(s.model['value'][~s.model['value_observed']]==0))
    def test_future_suffix_variants_at_every_boundary(self):
        obs=ObservationReplay('Demo.run')
        for t in tokens(self.text):
            obs.push(t)
            if obs.full:check_prefix(self.b,obs,self.text,t,'Demo.run')
    def test_target_changes_without_input_change(self):
        old,_,t=self.snap('instruction',3);changed=self.text.replace('#1#5,nextinst=-1','#1#4,nextinst=5').replace('opcode=177(return)','opcode=4(iconst_1),pushnum=1')
        obs=ObservationReplay.from_text(changed[:t['cut']],'Demo.run');self.assertEqual(self.b.build(obs).fingerprint(),old.fingerprint())
        self.assertNotEqual(targets(ObservationReplay.from_text(changed,'Demo.run'),self.b.static)[3]['B'], 'jump')
    def test_future_endpoint_refused(self):
        s,_,_=self.snap();s.model['edge_index'][0,0]=999
        with self.assertRaises(ValueError):validate_snapshot(s)
    def test_future_value_time_refused(self):
        s,_,_=self.snap('callback');s.audit['values'][0]['observed_at']=[999,0]
        with self.assertRaises(ValueError):validate_snapshot(s)
    def test_stale_full_degree_refused(self):
        s,_,_=self.snap();s.model['in_degree'][0]=123
        with self.assertRaises(ValueError):validate_snapshot(s)
    def test_unique_identity_strings_not_in_model(self):
        s,_,_=self.snap();self.assertTrue(all(np.asarray(v).dtype.kind in 'bif' for v in s.model.values()))
    def test_sealed_config_and_no_terminal_features(self):
        s,_,_=self.snap('instruction',4);self.assertFalse(s.model['x'][:,20:].any())
        self.assertFalse(s.model['x'][:,3:5].any())
    def test_orphan_callback_rejected(self):
        with self.assertRaises(ValueError):ObservationReplay.from_text('###Demo::run\n,stack=I:3','Demo.run')
    def test_multithread_rejected(self):
        obs=ObservationReplay.from_text(self.text.replace('#1#5,nextinst=-1,thread=1','#1#5,nextinst=-1,thread=2'),'Demo.run')
        with self.assertRaises(ValueError):self.b.build(obs)
    def test_pre_test_setup_not_initial_state(self):
        prefix='###Setup::run\nopcode=3(iconst_0),lineinfo=Setup#run#()V#1#0,nextinst=1,thread=9,stack=I:99\n'
        obs=ObservationReplay.from_text(prefix+self.text,'Demo.run');self.assertEqual(len(obs.full),5)
        self.assertFalse(self.b.build(obs).model['heap_state_known'][0])
    def test_repeated_event_instances_not_collapsed(self):
        text=self.text+self.text.split('###Demo::run\n')[1]
        obs=ObservationReplay.from_text(text,'Demo.run');self.assertEqual(len(obs.full),10)
        s=self.b.build(obs);self.assertEqual(len(set(s.model['event_nodes'].tolist())),10)
    def test_unknown_identity_refused(self):
        self.b.identities[(*D,0)]['provenance_type']='UNRESOLVED'
        with self.assertRaises(ValueError):self.snap()

if __name__=='__main__':unittest.main()
