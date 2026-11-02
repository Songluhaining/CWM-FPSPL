"""Artificial records, not research observations or new Java collections."""
import copy,unittest
from cwm_fp.models.local_state import capture,branch,arithmetic,semantic_check,before_instruction
from cwm_fp.models.prefix_observer import ObservationReplay,tokens
from test_prefix_observer import fixture

class LocalStateTests(unittest.TestCase):
    def at(self,text,index):
        observer=ObservationReplay('Demo.run');n=0
        for t in tokens(text):
            observer.push(t)
            if t['kind']=='instruction':
                if n==index:return observer
                n+=1
        self.fail('Missing boundary')
    def test_actual_pre_boundary_and_frame(self):
        b,text=fixture();s,flow,raw=capture(self.at(text,3),b.static,b.framework_static)
        self.assertEqual(s['operands'][0]['source'],'DIRECT_OBSERVED')
        self.assertEqual(s['operands'][0]['value'],0);self.assertEqual(semantic_check(s),(True,None))
        self.assertEqual(len(flow.events),3);self.assertEqual(len(raw),4)
        self.assertFalse(any(v.definition==3 for v in flow.values));self.assertEqual(s['frame_id'],0)
    def test_current_callback_refused(self):
        b,text=fixture();obs=self.at(text,2)
        obs.push({'clock':[4,1],'kind':'callback','text':',stack=I:999'})
        with self.assertRaises(ValueError):capture(obs,b.static,b.framework_static)
    def test_future_target_or_callback_cannot_change_input(self):
        b,text=fixture();original=capture(self.at(text,3),b.static,b.framework_static)[0]
        for future in ('', '\nFINAL=FAIL\n', '\n,stack=I:999\n'):
            changed=text[:text.index('opcode=177')]+future
            self.assertEqual(capture(self.at(changed,3),b.static,b.framework_static)[0],original)
    def test_static_constant_is_not_measurement(self):
        b,text=fixture();text=text.replace(',stack=I:0','')
        # Before store, the iconst literal has no runtime callback.
        s,_,_=capture(self.at(text,1),b.static,b.framework_static)
        self.assertEqual(s['operands'][0]['source'],'STATIC_CONSTANT')
    def test_copy_derivation_separate_from_direct(self):
        b,text=fixture();text=text.replace(',stack=I:0','')
        s,_,_=capture(self.at(text,3),b.static,b.framework_static)
        self.assertEqual(s['operands'][0]['source'],'SEMANTICALLY_DERIVED')
        self.assertEqual(semantic_check(s),(None,'insufficient_primary_evidence'))
    def test_unknown_reference_hash_not_numeric(self):
        b,text=fixture();text=text.replace(',stack=I:0',',stack=Obj:12345')
        s,_,_=capture(self.at(text,3),b.static,b.framework_static)
        self.assertNotEqual(s['operands'][0]['source'],'DIRECT_OBSERVED')
    def test_return_copy_is_derived_not_new_observation(self):
        b,_=fixture()
        text='###Demo::run\n'+'\n'.join([
            'opcode=184(invokestatic),calltype=()I,callclass=Demo,callname=callee,lineinfo=Demo#run#()V#1#0,nextinst=1,thread=1',
            '###Demo::callee',
            'opcode=4(iconst_1),pushnum=1,lineinfo=Demo#callee#()I#1#0,nextinst=1,thread=1,stack=I:1',
            'opcode=172(ireturn),popnum=1,lineinfo=Demo#callee#()I#1#1,nextinst=-1,thread=1',
            '###RET@lineinfo=Demo#run#()V#1#0',
            'opcode=154(ifne),popnum=1,lineinfo=Demo#run#()V#1#1,nextinst=2,thread=1'])
        s,_,_=capture(self.at(text,3),b.static,b.framework_static)
        self.assertEqual(s['operands'][0]['source'],'SEMANTICALLY_DERIVED')
        self.assertEqual(s['operands'][0]['value'],1)
        self.assertEqual(s['operands'][0]['derivation'][0]['source_category'],'DIRECT_OBSERVED')
        self.assertIsNone(semantic_check(s)[0])
    def test_binary_order(self):
        self.assertTrue(branch(161,[1,2]));self.assertFalse(branch(161,[2,1]))
        self.assertEqual(arithmetic(100,[2,5]),-3)
        self.assertEqual(arithmetic(108,[7,-2]),-3)
        self.assertEqual(arithmetic(112,[-7,2]),-1)
    def test_stack_order_is_captured_before_binary_consume(self):
        b,_=fixture()
        text='###Demo::run\n'+'\n'.join([
            'opcode=4(iconst_1),pushnum=1,lineinfo=Demo#run#()V#1#0,nextinst=1,thread=1,stack=I:1',
            'opcode=5(iconst_2),pushnum=1,lineinfo=Demo#run#()V#1#1,nextinst=2,thread=1,stack=I:2',
            'opcode=161(if_icmplt),popnum=2,lineinfo=Demo#run#()V#1#2,nextinst=3,thread=1'])
        s,flow,_=capture(self.at(text,2),b.static,b.framework_static)
        self.assertEqual([v['value'] for v in s['operands']],[1,2])
        self.assertEqual(len(flow.frames[-1].stack),2)
        self.assertEqual(semantic_check(s),(True,None))
    def test_overflow_shift_width_and_minint(self):
        self.assertEqual(arithmetic(96,[2147483647,1]),-2147483648)
        self.assertEqual(arithmetic(108,[-2147483648,-1]),-2147483648)
        self.assertEqual(arithmetic(112,[-2147483648,-1]),0)
        self.assertEqual(arithmetic(124,[-1,32]),-1)
        self.assertEqual(arithmetic(124,[-1,1]),2147483647)
        self.assertEqual(arithmetic(116,[-2147483648]),-2147483648)
    def test_unsupported_types_and_comparisons(self):
        for op,vs in [(165,[1,1]),(153,[1.0]),(153,[2**31]),(198,[123])]:
            with self.assertRaises(ValueError):branch(op,vs)
        for op,vs in [(98,[1,2]),(97,[1,2]),(108,[1,0]),(96,[float('nan'),1])]:
            with self.assertRaises(ValueError):arithmetic(op,vs)
        self.assertTrue(branch(198,[None]));self.assertFalse(branch(199,[None]))
    def test_repeated_execution_has_independent_frame_and_value_ids(self):
        b,text=fixture();twice=text+text
        a,_,_=capture(self.at(twice,3),b.static,b.framework_static)
        z,_,_=capture(self.at(twice,8),b.static,b.framework_static)
        self.assertNotEqual(a['frame_id'],z['frame_id']);self.assertNotEqual(a['operands'][0]['value_id'],z['operands'][0]['value_id'])
        self.assertEqual(a['operands'][0]['value'],z['operands'][0]['value'])

if __name__=='__main__':unittest.main()
