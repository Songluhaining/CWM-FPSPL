"""Artificial boundary cases; never counted as Email research data."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from cwm_fp.data.common import sha256
from cwm_fp.models.local_dataset import (SourceBundle,tensor_input,read_input,target_for,replay_inputs,learning_input,LocalObservationReplay)
from cwm_fp.models.prefix_observer import tokens,ObservationReplay
from test_prefix_observer import fixture

class LocalDatasetTests(unittest.TestCase):
    def make(self):
        b,text=fixture()
        for d in b.static['methods']:
            b.static['exception_tables'][d]['handlers']=[]
        return b,text,{(*p['locator'].values(),):p for p in b.bundle['dynamic_identity_lookup']}
    def test_integer_precision_and_missing_masks(self):
        x={'opcode':100,'arity':2,'operands':[{'dtype':'I','source':'DIRECT_OBSERVED','primary_known':True,'value':2147483647},
            {'dtype':None,'source':'UNKNOWN','primary_known':False,'value':None}]}
        a=tensor_input(x);self.assertEqual(int(a['value'][0]),2147483647);self.assertEqual(a['known'].tolist(),[True,False])
    def test_getstatic_completion_is_not_value_or_method_return(self):
        text='###Demo::run\nopcode=178(getstatic),pushnum=1,field=Demo#counter,lineinfo=Demo#run#()V#1#0,nextinst=3,thread=1\n###RET@opcode=178(getstatic),pushnum=1,field=Demo#counter,lineinfo=Demo#run#()V#1#0,nextinst=3\n'
        r=LocalObservationReplay('Demo.run')
        for t in tokens(text):r.push(t)
        self.assertIsNotNone(r.full[0]['completed_clock']);self.assertFalse(r.full[0]['call_returned'])
        self.assertIsNone(r.full[0]['callback_clock']);self.assertNotIn('stack',r.full[0]['fields'])
        # The old observer is not silently reinterpreted.
        with self.assertRaises(ValueError):ObservationReplay.from_text(text,'Demo.run')
    def test_getstatic_wrong_completion_opcode_rejected(self):
        text='###Demo::run\nopcode=178(getstatic),pushnum=1,lineinfo=Demo#run#()V#1#0,nextinst=3,thread=1\n###RET@opcode=187(new),lineinfo=Demo#run#()V#1#0\n'
        r=LocalObservationReplay('Demo.run')
        with self.assertRaises(ValueError):
            for t in tokens(text):r.push(t)
    def test_repeated_getstatic_completion_retains_instances(self):
        one='opcode=178(getstatic),pushnum=1,lineinfo=Demo#run#()V#1#0,nextinst=3,thread=1\n###RET@opcode=178(getstatic),lineinfo=Demo#run#()V#1#0\n'
        r=LocalObservationReplay('Demo.run')
        for t in tokens('###Demo::run\n'+one+one):r.push(t)
        self.assertEqual(len(r.full),2);self.assertNotEqual(r.full[0]['completed_clock'],r.full[1]['completed_clock'])
    def test_source_mask_spoof_and_audit_fields_rejected(self):
        for extra in ('target','product_id','method','frame_id'):
            with self.assertRaises(ValueError):tensor_input({'opcode':153,'arity':0,'operands':[],extra:0})
        with self.assertRaises(ValueError):tensor_input({'opcode':153,'arity':1,'operands':[{'dtype':'I','source':'UNKNOWN','primary_known':True,'value':0}]})
    def test_unknown_not_invented_zero(self):
        with self.assertRaises(ValueError):tensor_input({'opcode':153,'arity':1,'operands':[{'dtype':'I','source':'UNKNOWN','primary_known':False,'value':0}]})
    def test_null_separate_from_integer_zero(self):
        a=tensor_input({'opcode':198,'arity':1,'operands':[{'dtype':'reference_null','source':'STATIC_CONSTANT','primary_known':True,'value':None}]})
        self.assertTrue(a['null'][0]);self.assertTrue(a['known'][0])
    def test_target_crossing_boundary_is_masked(self):
        d=('Demo','run','()V');b,text=fixture();r={'opcode':153,'domain':d,'bci':3,'pre_clock':[5,0]};n={'domain':d,'bci':5,'entry':False,'pre_clock':[6,0]}
        t=target_for(r,n,b.static,0,0,[6,0]);self.assertFalse(t['mask']);self.assertIsNone(t['value'])
        self.assertTrue(target_for(r,n,b.static,0,0,None)['value'])
    def test_unverified_frame_cannot_supply_branch_target(self):
        b,text=fixture();d=('Demo','run','()V')
        r={'opcode':153,'domain':d,'bci':3,'pre_clock':[5,0]};n={'domain':d,'bci':5,'entry':False,'pre_clock':[6,0]}
        self.assertFalse(target_for(r,n,b.static,None,None,None)['mask'])
    def test_reader_does_not_open_targets_at_new_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'inputs.jsonl').write_text(json.dumps({'query_id':'x','learning_input':{'opcode':0,'arity':0,'operands':[]}})+'\n')
            (root/'manifest.json').write_text(json.dumps({'schema':'local_transition_dataset_v1','inputs':{'path':'inputs.jsonl','sha256':sha256(root/'inputs.jsonl')},'targets':{'path':'FORBIDDEN'}}))
            self.assertEqual(int(read_input(root,'x',expected_manifest_sha256=sha256(root/'manifest.json'))['opcode']),0)
            with self.assertRaises(ValueError):read_input(root,'x',expected_manifest_sha256='0'*64)
    def test_portable_path_must_exist_in_hashed_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f=root/'source.java';f.write_text('class X {}')
            b=SourceBundle.__new__(SourceBundle);b.root=root;b.files={'source.java':{'sha256':sha256(f)}};b.paths={'D:/old/X.java':'source.java'};b.access=[]
            self.assertEqual(b.resolve('D:\\old\\X.java'),f)
            with self.assertRaises(ValueError):b.resolve('D:/other/X.java')
            f.write_text('changed')
            with self.assertRaises(ValueError):b.resolve('D:/old/X.java')
    def test_unresolved_identity_is_permanent_prefix_boundary(self):
        b,text,ids=self.make()
        def check(r,*args):return r['bci']!=1,{'scope':'product','instruction_id':'test'},b.static['methods'][r['domain']][r['bci']]
        with patch('cwm_fp.models.local_dataset.identity_check',side_effect=check):
            result=replay_inputs(text,'Demo.run',b.static,b.framework_static,ids)
        self.assertEqual(result['barrier']['event_index'],1)
        self.assertEqual(result['states'][3]['state']['operands'][0]['source'],'UNKNOWN')
    def test_prior_array_gap_never_recovers_after_callback(self):
        b,text,ids=self.make();text=text.replace('opcode=59(istore_0)','opcode=188(newarray)')
        b.static['methods'][('Demo','run','()V')][1]['opcode']=188
        with patch('cwm_fp.models.local_dataset.identity_check',return_value=(True,{'scope':'product','instruction_id':'test'},{})):
            result=replay_inputs(text,'Demo.run',b.static,b.framework_static,ids)
        self.assertEqual(result['barrier']['reason'],'unsupported_array_type_or_monitor_opcode')
        self.assertEqual(result['states'][3]['state']['operands'][0]['source'],'UNKNOWN')
    def test_source_failure_blocks_all_prefixes(self):
        b,text,ids=self.make()
        with patch('cwm_fp.models.local_dataset.identity_check',return_value=(True,{'scope':'product','instruction_id':'test'},{})):
            result=replay_inputs(text,'Demo.run',b.static,b.framework_static,ids,source_ok=False)
        self.assertEqual(result['barrier']['clock'],[0,0]);self.assertIsNone(result['states'][3]['state']['frame_id'])
    def test_suffix_isolation_independent_truncated_input(self):
        b,text,ids=self.make()
        with patch('cwm_fp.models.local_dataset.identity_check',return_value=(True,{'scope':'product','instruction_id':'test'},{})):
            whole=replay_inputs(text,'Demo.run',b.static,b.framework_static,ids);r=whole['states'][3];cut=r['cut']
            short=replay_inputs(text[:cut],'Demo.run',b.static,b.framework_static,ids)
            changed=replay_inputs(text[:cut]+'\n,stack=I:199\nFINAL=FAIL','Demo.run',b.static,b.framework_static,ids,stop_cut=cut)
        self.assertEqual(r['learning_input'],short['states'][-1]['learning_input']);self.assertEqual(r['learning_input'],changed['states'][-1]['learning_input'])

if __name__=='__main__':unittest.main()
