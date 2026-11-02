"""Replay, tamper, side-effect and boundary checks using real saved Java evidence."""
import copy
import json
from pathlib import Path
import unittest

from cwm_fp.data.common import ROOT
from cwm_fp.lddg.control import control_evidence_errors
from cwm_fp.lddg.effects import project_and_verify,apply_effect
from cwm_fp.lddg.static import identity_digest
from cwm_fp.lddg.trace import Value

OUT=ROOT/'data/processed/lddg/phase03B4_framework_v1'
def load(p):return json.loads(p.read_text(encoding='utf-8'))

class FrameworkEffectsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph=load(OUT/'synthetic/read.graph.json')
        cls.evidence=cls.graph['control_flow_evidence']['proof']['framework_evidence']
        cls.fixtures={r['case']:r for r in load(OUT/'synthetic/verification.json')}

    def changed(self,mutate,reason=None):
        g=copy.deepcopy(self.graph);mutate(g)
        e=g['control_flow_evidence'];e['proof_sha256']=identity_digest(e['proof'])
        errors=control_evidence_errors(g);self.assertTrue(errors)
        if reason:self.assertIn(reason,errors)

    def test_real_body_replay(self):self.assertEqual([],control_evidence_errors(self.graph))
    def test_binary_hash_mismatch(self):
        e=copy.deepcopy(self.evidence);e['jar']['sha256']='0'*64
        with self.assertRaises(ValueError):project_and_verify(e)
    def test_runtime_binding_mismatch(self):
        e=copy.deepcopy(self.evidence);e['runtime_binding']['artifacts_sha256']={}
        with self.assertRaises(ValueError):project_and_verify(e)
    def test_unselected_instrumentation(self):
        e=copy.deepcopy(self.evidence);e['runtime_binding']['selected_classes']=[]
        with self.assertRaises(ValueError):project_and_verify(e)
    def test_trace_hash_mismatch(self):
        e=copy.deepcopy(self.evidence);e['trace_sha256']='0'*64
        with self.assertRaises(ValueError):project_and_verify(e)
    def test_forged_projected_return(self):
        self.changed(lambda g:g['control_flow_evidence']['proof']['raw_events'][-2].update(call_returned=False))
    def test_forged_effect_flag(self):
        self.changed(lambda g:g['framework_effects'][0].update(accepted=False),'framework_effect_replay_mismatch')
    def test_removed_effect(self):self.changed(lambda g:g.update(framework_effects=[]),'framework_effect_replay_mismatch')
    def test_quality_flag(self):self.changed(lambda g:g['quality'].update(framework_effects_valid=False),'framework_effect_quality_mismatch')
    def test_count_flag(self):self.changed(lambda g:g['quality'].update(physical_trace_events=1),'framework_event_accounting_mismatch')
    def test_schema_downgrade(self):
        self.changed(lambda g:(g['provenance'].update(control_schema='null_field_exception_v1'),g['control_flow_evidence'].update(schema='null_field_exception_v1')))
    def test_dependency_tamper(self):
        self.changed(lambda g:next(e for e in g['edges'] if e['type']=='data_dep').update(definition_event=999),'dataflow_dependency_endpoint_mismatch')
    def test_exception_state_not_numeric(self):
        effect=self.graph['framework_effects'][0]
        self.assertIn('exception_private_stackTrace_cache',effect['possible_writes'])
        self.assertFalse(effect['scalar_return_observed']);self.assertEqual('void',effect['return_type'])
        self.assertGreater(effect['physical_return_after_event'],effect['physical_call_event'])
    def test_no_missing_values_filled(self):
        for n in self.graph['nodes']:
            if n['type']=='value' and not n['attrs'].get('observed'):self.assertNotIn('value',n['attrs'])
    def test_repeated_instances(self):
        g=load(OUT/'synthetic/repeated.graph.json');effects=g['framework_effects']
        self.assertEqual(3,len(effects));self.assertEqual(3,len({e['argument_value_ids'][1] for e in effects}))
        self.assertTrue(all(e['accepted'] for e in effects));self.assertEqual([],control_evidence_errors(g))
    def test_same_exception_rejected(self):
        self.assertFalse(self.fixtures['repeatedSameException']['certified'])
        self.assertIn('exception_previously_escaped',str(self.fixtures['repeatedSameException']['errors']))
    def test_callback_preserved_and_rejected(self):
        g=load(OUT/'synthetic/callback.graph.json')
        self.assertTrue(any(n['attrs'].get('composed_method')=='callback' and n['attrs'].get('executed') for n in g['nodes']))
        self.assertFalse(g['quality']['eligible_graph_structure'])
        self.assertEqual('PASS',self.fixtures['callback']['original_result']['outcome'])
    def test_effects_after_real_state_writes(self):
        g=load(OUT/'synthetic/stateAfter.graph.json')
        self.assertTrue(g['quality']['eligible_graph_structure'])
        self.assertTrue(any(e['type']=='data_dep' and e.get('role')=='field_read' for e in g['edges']))
        self.assertEqual([],control_evidence_errors(g))
    def test_unsupported_paths_and_original_failure(self):
        for name in ['unexpected','wrongOrigin','callback','reflection','arrayAfter','nullInvoke']:
            with self.subTest(name=name):self.assertFalse(self.fixtures[name]['certified']);self.assertTrue(self.fixtures[name]['parity'])
        self.assertEqual('FAIL',self.fixtures['noException']['original_result']['outcome'])
    def test_unknown_or_non_NPE_arguments_rejected(self):
        raw,calls,_=project_and_verify(self.evidence);i=next(iter(calls))
        result=apply_effect(i,[Value('v:x',0),Value('v:y',1,ref='allocation:1')],raw,calls[i],[],set())
        self.assertFalse(result['accepted']);self.assertEqual('not_verified_JVM_null_field_NPE',result['reason'])

if __name__=='__main__':unittest.main()
