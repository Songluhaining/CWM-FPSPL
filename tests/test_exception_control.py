"""Evidence replay and safe rejection; uses separately executed Java fixtures."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cwm_fp.lddg.control import replay_normal_path,control_evidence_errors
from cwm_fp.lddg.static import identity_digest

OUT=ROOT/'data/processed/lddg/phase03A3_exception_v1'
OLD=ROOT/'data/processed/lddg/phase03A2_identity_v1'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def fixture(name):return load(OUT/'synthetic'/name/'graph.json')
def replay(p):return replay_normal_path(p['methods'],p['events'],p['execution_complete_and_consistent'],p['test_outcome'],p['failure_types'])


class ExceptionControlTests(unittest.TestCase):
    def proof(self,name='normalTryCatch'):
        return copy.deepcopy(fixture(name)['control_flow_evidence']['proof'])

    def test_six_real_graphs_preserve_nodes_edges_and_static_identities(self):
        for old,g in zip(lines(OLD/'graphs.jsonl'),lines(OUT/'graphs.jsonl')):
            for k in ('nodes','edges','instruction_catalog'):self.assertEqual(old[k],g[k])
            self.assertFalse(control_evidence_errors(g))
            self.assertTrue(g['quality']['normal_path_verified'])
            self.assertFalse(g['quality']['exception_control_supported'])

    def test_events_source_maps_and_33_non_executable_anchors_unchanged(self):
        for name in ('events.jsonl','source_map.jsonl'):self.assertEqual(lines(OLD/name),lines(OUT/name))
        rows=lines(OUT/'source_map.jsonl')
        anchors=[r for r in rows if not r['executable']]
        self.assertEqual(len(anchors),33)
        self.assertTrue(all('program_identity' not in r for r in anchors))

    def test_transaction_paths_have_real_finally_tables_without_handler_execution(self):
        for g in (lines(OUT/'graphs.jsonl')[3],lines(OUT/'graphs.jsonl')[5]):
            e=g['control_flow_evidence'];p=e['proof']
            m=next(m for m in p['methods'] if m['domain'][1]=='transfer')
            self.assertEqual([(h['start'],h['end'],h['handler'],h['catch_type']) for h in m['handlers']],
                             [(10,18,97,'*'),(29,42,97,'*'),(53,72,97,'*'),(83,86,97,'*'),(97,99,97,'*')])
            events=[r for r in p['events'] if r['domain']==m['domain']]
            self.assertNotIn(97,[r['bci'] for r in events]);self.assertEqual(events[-1]['opcode'],172)
            self.assertEqual(e['assessment']['classification'],'A')
            self.assertEqual(e['assessment']['terminal']['kind'],'observed_external_assertion_failure')
            self.assertEqual(p['failure_types'],['java.lang.AssertionError'])

    def test_exception_tables_do_not_reject_observed_normal_paths(self):
        for case in ('normalTryCatch','normalFinally'):
            a=fixture(case)['control_flow_evidence']['assessment']
            self.assertTrue(a['accepted']);self.assertEqual(a['classification'],'A')
            self.assertFalse(a['observed_handler_pcs']);self.assertEqual(a['certified_exception_transitions'],0)

    def test_ordinary_astore_is_not_an_exception_handler(self):
        g=fixture('ordinaryStore');p=g['control_flow_evidence']['proof']
        self.assertTrue(any(e['opcode'] in (58,75,76,77,78) for e in p['events']))
        self.assertTrue(replay(p)['accepted']);self.assertFalse(replay(p)['handler_visits'])

    def test_five_unsupported_cases_are_rejected_and_do_not_emit_fabricated_def_use(self):
        for name in ('sameMethodThrow','callerCatch','exceptionalFinally','untracedException','implicitDivide'):
            g=fixture(name);self.assertTrue(control_evidence_errors(g))
            self.assertFalse(g['quality']['eligible_graph_structure'])
            self.assertEqual(g['quality']['dynamic_dependency_emission'],'withheld_unverified_control_path')
            self.assertFalse(any(e['type'] in {'data_dep','produces','consumes'} for e in g['edges']))
            self.assertTrue(all(not e['def_ids'] and not e['use_ids'] for e in load(OUT/'synthetic'/name/'events.json')))

    def test_implicit_exception_handler_located_but_not_certified(self):
        a=fixture('implicitDivide')['control_flow_evidence']['assessment']
        self.assertEqual(a['classification'],'B_PATH_LOCATED_SEMANTICS_UNSUPPORTED')
        self.assertTrue(a['handler_visits'][0]['matching_handlers'])
        self.assertFalse(a['handler_visits'][0]['exception_object_and_unwinding_certified'])

    def test_early_rejection_does_not_claim_to_have_checked_the_entire_trace(self):
        a=fixture('sameMethodThrow')['control_flow_evidence']['assessment']
        self.assertEqual(a['events_checked'],a['errors'][0]['event_index']+1)
        self.assertLess(a['events_checked'],a['events_recorded'])

    def test_explicit_and_cross_method_handlers_remain_visible_after_rejection(self):
        for name in ('sameMethodThrow','callerCatch','exceptionalFinally'):
            a=fixture(name)['control_flow_evidence']['assessment']
            self.assertTrue(a['observed_handler_pcs']);self.assertFalse(a['accepted'])

    def test_missing_handler_rows_fail_even_on_normal_path(self):
        p=self.proof();m=next(m for m in p['methods'] if m['handlers']);m['handlers']=[]
        self.assertFalse(replay(p)['accepted'])

    def test_invalid_handler_range_or_order_fails(self):
        for key,value in [('start',9999),('end',-1),('handler',9999),('table_index',9),('catch_type','')]:
            p=self.proof();next(m for m in p['methods'] if m['handlers'])['handlers'][0][key]=value
            self.assertFalse(replay(p)['accepted'])

    def test_missing_normal_return_marker_fails(self):
        p=self.proof();next(e for e in p['events'] if e['call_returned'])['call_returned']=False
        self.assertFalse(replay(p)['accepted'])

    def test_reordered_events_or_threads_fail(self):
        p=self.proof();p['events'][1]['event_index']=0;self.assertFalse(replay(p)['accepted'])
        p=self.proof();p['events'][1]['thread_id']='different-thread';self.assertFalse(replay(p)['accepted'])

    def test_unknown_frame_or_reused_invocation_identity_fails(self):
        p=self.proof();p['events'][3]['frame_id']=9876;self.assertFalse(replay(p)['accepted'])
        p=self.proof();e=next(e for e in p['events'][1:] if e['entry']);e['frame_id']=p['events'][0]['frame_id']
        self.assertFalse(replay(p)['accepted'])

    def test_illegal_normal_successor_cannot_be_accepted(self):
        p=self.proof();first=p['events'][0];m=next(m for m in p['methods'] if m['domain']==first['domain'])
        next(i for i in m['instructions'] if i['bci']==first['bci'])['successors']=[-1]
        self.assertFalse(replay(p)['accepted'])

    def test_wrong_opcode_or_call_target_fails(self):
        p=self.proof();p['events'][0]['opcode']=191;self.assertFalse(replay(p)['accepted'])
        p=self.proof();next(e for e in p['events'] if 'call_target' in e)['call_target']=['fake','fake','()V']
        self.assertFalse(replay(p)['accepted'])

    def test_incomplete_trace_cannot_use_assertion_terminal_exception(self):
        p=copy.deepcopy(lines(OUT/'graphs.jsonl')[3]['control_flow_evidence']['proof'])
        p['execution_complete_and_consistent']=False;self.assertFalse(replay(p)['accepted'])

    def test_unreturned_call_is_not_automatically_a_failed_assertion(self):
        base=lines(OUT/'graphs.jsonl')[3]['control_flow_evidence']['proof']
        for key,value in [('test_outcome','PASS'),('failure_types',[]),('failure_types',['java.lang.NullPointerException'])]:
            p=copy.deepcopy(base);p[key]=value;self.assertFalse(replay(p)['accepted'])

    def test_trace_digest_and_proof_digest_are_checked(self):
        g=copy.deepcopy(lines(OUT/'graphs.jsonl')[0]);g['control_flow_evidence']['proof_sha256']='bad'
        self.assertIn('control_proof_digest_mismatch',control_evidence_errors(g))
        g=copy.deepcopy(lines(OUT/'graphs.jsonl')[0]);g['provenance']['trace_sha256']='bad'
        self.assertIn('control_trace_binding_mismatch',control_evidence_errors(g))

    def test_consumer_replays_instead_of_trusting_claimed_accepted_flag(self):
        g=fixture('implicitDivide');g['control_flow_evidence']['assessment']['accepted']=True
        g['quality']['normal_path_verified']=True
        self.assertIn('control_assessment_disagrees_with_replay',control_evidence_errors(g))
        self.assertIn('normal_path_not_certified',control_evidence_errors(g))

    def test_rehashed_proof_still_cannot_change_event_identity(self):
        g=copy.deepcopy(lines(OUT/'graphs.jsonl')[0]);e=g['control_flow_evidence'];e['proof']['events'][0]['frame_id']=9876
        e['proof_sha256']=identity_digest(e['proof'])
        self.assertTrue(any(x.startswith('control_event_identity_mismatch') for x in control_evidence_errors(g)))

    def test_exception_semantics_overclaim_is_rejected(self):
        g=copy.deepcopy(lines(OUT/'graphs.jsonl')[0]);g['quality']['exception_control_supported']=True
        self.assertIn('exception_semantics_overclaim',control_evidence_errors(g))

    def test_audit_transitions_have_valid_forward_event_endpoints(self):
        for g in lines(OUT/'graphs.jsonl'):
            e=g['control_flow_evidence'];events=e['proof']['events'];a=e['assessment']
            for edge in a['normal_transitions']:
                self.assertLess(edge['source_event'],edge['target_event'])
                self.assertLess(edge['target_event'],len(events))
                self.assertIn(edge['kind'],('normal_CFG','normal_call_resume','normal_return','traced_call'))
                if edge['kind']=='normal_return':self.assertLess(edge['caller_event'],edge['source_event'])
            self.assertEqual(len({e['canonical_event_id'] for e in lines(OUT/'events.jsonl')}),332)

    def test_all_java_pairs_pass_without_changing_original_semantics(self):
        v=load(OUT/'synthetic/verification.json');self.assertEqual(len(v['cases']),8)
        self.assertTrue(all(x['parity'] and x['result']['outcome']=='PASS' for x in v['cases']))
        self.assertEqual(v['total_new_java_test_executions'],16)


if __name__=='__main__':unittest.main()
