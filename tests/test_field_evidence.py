"""Artificial physical evidence boundaries, not real Email samples."""
import unittest
from cwm_fp.models.field_evidence import classify_read,read_value,visible_read

def example(**kw):
    args=dict(opcode=180,descriptor='I',raw_callback='I:7',pre_clock=[8,0],callback_clock=[8,1],completed_clock=[8,1],
              identity_valid=True,frame_stack_valid=True,receiver_valid=True,association_valid=True,earlier_gaps=[])
    return classify_read(**(args|kw))

class FieldEvidenceTests(unittest.TestCase):
    def test_read_value_not_write_lineage(self):
        r=example();self.assertEqual(r['category'],'A');self.assertTrue(r['observed_value_valid'])
        for k in ('upstream_write_provenance_complete','current_heap_complete','local_transition_valid','full_product_admission'):self.assertFalse(r[k])
    def test_completion_only_static(self):
        r=example(opcode=178,raw_callback=None,callback_clock=None,completed_clock=[9,0]);self.assertEqual(r['category'],'B')
    def test_object_hash_zero_is_not_null_tag(self):
        r=example(descriptor='Ljava/lang/String;',raw_callback='Obj:0');self.assertEqual(r['category'],'B');self.assertIsNone(r['value'])
    def test_hash_not_numeric_or_object_merge(self):
        for raw in ('Obj:1','Obj:123','Obj:-7'):
            self.assertFalse(read_value(raw,'Ljava/lang/Object;')['valid'])
    def test_unknown_receiver(self):self.assertEqual(example(receiver_valid=False)['category'],'C')
    def test_unknown_frame(self):self.assertEqual(example(frame_stack_valid=False)['category'],'C')
    def test_wrong_descriptor(self):self.assertEqual(example(raw_callback='S:7')['category'],'C')
    def test_earlier_call_or_cast_never_recovers(self):
        for why in ('untraced_call_effects','unsupported_array_type_or_monitor_opcode','unresolved_identity','multiple_threads'):
            self.assertEqual(example(earlier_gaps=[why])['category'],'D')
    def test_before_or_at_callback_not_visible(self):
        r=example()
        self.assertFalse(visible_read(r,[8,0])['observed']);self.assertFalse(visible_read(r,[8,1])['observed'])
        self.assertEqual(visible_read(r,[9,0])['value'],7)
    def test_deleted_callback_rejects(self):
        self.assertEqual(example(raw_callback=None,callback_clock=None,completed_clock=[9,0])['category'],'B')
    def test_time_and_duplicate_association(self):
        self.assertEqual(example(callback_clock=[7,0])['category'],'C')
        self.assertEqual(example(association_valid=False)['category'],'C')
    def test_bool_and_int_boundaries(self):
        self.assertEqual(read_value('Z:false','Z')['value'],0)
        self.assertEqual(read_value('Z:true','Z')['value'],1)
        self.assertFalse(read_value('Z:1','Z')['valid'])
        for n in (-2147483648,2147483647):self.assertEqual(read_value('I:'+str(n),'I')['value'],n)
        self.assertFalse(read_value('I:2147483648','I')['valid'])
    def test_unknown_is_not_zero(self):
        r=example(raw_callback=None,callback_clock=None,completed_clock=[9,0]);self.assertIsNone(visible_read(r,[10,0])['value'])
    def test_separate_read_occurrences(self):
        first=example();again=example(pre_clock=[18,0],callback_clock=[18,1],completed_clock=[18,1])
        self.assertNotEqual(first['available_at'],again['available_at']);self.assertFalse(visible_read(again,[9,0])['observed'])
    def test_unsupported_types_do_not_upgrade(self):
        for dt in ('J','F','D','C'):
            self.assertEqual(example(descriptor=dt,raw_callback=dt+':7')['category'],'B')
    def test_known_write_is_not_missing_provenance(self):
        r=example(write_provenance_complete=True)
        self.assertEqual(r['category'],'E');self.assertTrue(r['observed_value_valid'])
        self.assertTrue(r['upstream_write_provenance_complete']);self.assertFalse(r['local_transition_valid'])

if __name__=='__main__':unittest.main()
