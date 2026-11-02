"""Actual Java observations, precise exception transfer and fail-closed replay."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cwm_fp.lddg.control import control_evidence_errors,replay_null_field_proof,control_evidence
from cwm_fp.lddg.trace import Dataflow,parse_trace,NULL_FIELD_PROFILE
from cwm_fp.lddg.static import identity_digest,load_static
OUT=Path('D:/dataset/whn/cwm_fp_generated/data/processed/lddg/phase03B3_bank_v1')
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def graph(name='callerRead'):return load(OUT/'synthetic'/name/'graph.json')
def flow(name):
    g=graph(name);return replay_null_field_proof(g['control_flow_evidence']['proof'])[1]
def rehash(g):g['control_flow_evidence']['proof_sha256']=identity_digest(g['control_flow_evidence']['proof'])


class NullFieldTests(unittest.TestCase):
    def test_all_13_real_java_pairs_complete_and_equal(self):
        v=load(OUT/'synthetic/verification.json');self.assertEqual(v['paired_tests'],13)
        for c in v['cases']:
            for key in ('complete','parity','matches_historical_outcome'):self.assertTrue(c['status'][key])
            self.assertEqual(c['status']['runs']['original']['result'],c['status']['runs']['instrumented']['result'])

    def test_seven_supported_and_six_unsupported(self):
        for c in load(OUT/'synthetic/verification.json')['cases']:
            g=graph(c['case']);self.assertEqual(g['quality']['eligible_graph_structure'],c['expected_supported'])
            if c['expected_supported']:self.assertFalse(control_evidence_errors(g))
            self.assertFalse(g['quality']['exception_control_supported'])

    def test_actual_scalar_boundaries_are_callbacks(self):
        observed=[v.observed['value'] for v in flow('normalReadWrite').values if v.observed['observed']]
        for x in (-2147483648,0,2147483647):self.assertIn(x,observed)
        for case,x in [('sameRead',11),('sameWrite',12),('callerRead',13),('callerWrite',14),('aliasedNull',7)]:
            self.assertIn(x,[v.observed['value'] for v in flow(case).values if v.observed['observed']])

    def test_repeated_faults_preserve_distinct_dynamic_instances(self):
        g=graph('repeated');ts=g['control_flow_evidence']['assessment']['certified_transfers']
        self.assertEqual(len(ts),6);self.assertEqual(len({t['exception_value_id'] for t in ts}),6)
        events={n['attrs']['event_index']:n['attrs'] for n in g['nodes'] if 'event_index' in n['attrs']}
        self.assertEqual(len({events[t['source_event']]['canonical_event_id'] for t in ts}),6)
        self.assertEqual(len({events[t['source_event']]['static_instruction_id'] for t in ts}),2)

    def test_no_scalar_or_normal_field_result_fabricated(self):
        for name in ('sameRead','sameWrite','callerRead','callerWrite','repeated','aliasedNull'):
            f=flow(name)
            for t in f.exception_transfers:
                defs=[v for v in f.values if v.definition==t['source_event']]
                self.assertEqual([v.id for v in defs],[t['exception_value_id']]);self.assertFalse(defs[0].observed['observed'])
                self.assertEqual(f.events[t['source_event']]['value_observed'],False)
                self.assertIn((t['exception_value_id'],t['target_event'],'operand'),f.links)

    def test_handler_preserves_caller_and_unwinds_only_real_frames(self):
        for name in ('sameRead','sameWrite','callerRead','callerWrite'):
            f=flow(name);t=f.exception_transfers[0]
            self.assertEqual(f.events[t['target_event']]['frame_id'],t['target_frame'])
            if name.startswith('same'):self.assertEqual(t['source_frame'],t['target_frame']);self.assertFalse(t['unwound_frames'])
            else:self.assertNotEqual(t['source_frame'],t['target_frame']);self.assertEqual(len(t['unwound_frames']),1)
            self.assertFalse(f.gaps)

    def test_definition_use_edges_have_strict_past_endpoints(self):
        for c in load(OUT/'synthetic/verification.json')['cases']:
            if not c['expected_supported']:continue
            g=graph(c['case']);ids={n['id'] for n in g['nodes']}
            for e in g['edges']:
                self.assertIn(e['source'],ids);self.assertIn(e['target'],ids)
                if e['type']=='data_dep':self.assertLess(e['definition_event'],e['event_index'])

    def test_null_hash_callback_without_aconst_null_is_not_proof(self):
        p=copy.deepcopy(graph()['control_flow_evidence']['proof'])
        for r in p['raw_events']:
            if r['opcode']==1:r['opcode']=3  # keep the real Obj:0 callback: still must refuse
        a,f=replay_null_field_proof(p);self.assertFalse(f.exception_transfers);self.assertFalse(a['accepted'])

    def test_wrong_handler_type_or_catch_all_is_refused(self):
        for typ in ('java.lang.RuntimeException','*','unresolved.Exception'):
            p=copy.deepcopy(graph()['control_flow_evidence']['proof'])
            for m in p['methods']:
                for h in m['handlers']:h['catch_type']=typ
            a,f=replay_null_field_proof(p);self.assertFalse(f.exception_transfers);self.assertFalse(a['accepted'])

    def test_synchronized_frame_unwinding_not_certified(self):
        p=copy.deepcopy(graph()['control_flow_evidence']['proof'])
        for m in p['methods']:m['method_flags']|=32
        a,f=replay_null_field_proof(p);self.assertFalse(f.exception_transfers);self.assertFalse(a['accepted'])

    def test_missing_or_forged_normal_return_marker_not_accepted(self):
        g=graph();p=g['control_flow_evidence']['proof'];t=g['control_flow_evidence']['assessment']['certified_transfers'][0]
        for seq in ('events','raw_events'):p[seq][t['aborted_calls'][0]]['call_returned']=True
        rehash(g);self.assertTrue(control_evidence_errors(g))

    def test_forged_transfer_assessment_does_not_override_replay(self):
        g=graph();g['control_flow_evidence']['assessment']['certified_transfers'][0]['target_frame']=999
        self.assertIn('control_assessment_disagrees_with_replay',control_evidence_errors(g))

    def test_forged_value_and_dependency_are_refused(self):
        g=graph();v=next(n for n in g['nodes'] if n['type']=='value');v['attrs']['definition_event']=999
        self.assertIn('dataflow_value_evidence_mismatch',control_evidence_errors(g))
        g=graph();e=next(e for e in g['edges'] if e['type']=='data_dep');e['target']='e:0'
        self.assertIn('dataflow_dependency_endpoint_mismatch',control_evidence_errors(g))
        g=graph();e=next(e for e in g['edges'] if e['type']=='data_dep');e['event_index']+=1
        self.assertIn('dataflow_dependency_endpoint_mismatch',control_evidence_errors(g))
        g=graph();e=next(e for e in g['edges'] if e['type']=='produces');e['evidence']='static'
        self.assertIn('dataflow_definition_endpoint_mismatch',control_evidence_errors(g))

    def test_library_side_effects_not_cleared_by_verified_exception(self):
        g=graph('externalEffectsRejected');self.assertTrue(g['quality']['control_path_verified'])
        self.assertFalse(g['quality']['eligible_graph_structure'])
        self.assertIn('untraced_call_effects',[x['reason'] for x in g['quality']['unsupported_dynamic']])

    def test_legacy_profile_still_rejects_actual_exception(self):
        work=OUT/'synthetic';static=load_static(work/'static.tsv');test='nullfieldfixture.NullFieldFixtureTest.callerRead'
        raw=parse_trace(work/'callerRead/instrumented/run.log',test);f=Dataflow(static).run(raw)
        e=control_evidence(raw,static,f.events,load(work/'callerRead/status.json'))
        self.assertEqual(e['schema'],'normal_path_v1');self.assertFalse(e['assessment']['accepted'])

    def test_real_regression_and_14_partial_certifications(self):
        rows=load(OUT/'comparison.json');new=[r for r in rows if r['batch_product']!='reuse03B1']
        self.assertEqual(len(new),140);self.assertEqual(sum(r['qualified_before'] for r in new),124)
        self.assertEqual(sum(r['qualified_after'] for r in new),124)
        self.assertEqual(sum(r['certified_null_field_transfers'] for r in new),14)
        self.assertTrue(all(r['old_qualified_nodes_edges_events_exactly_unchanged'] for r in rows if r['qualified_before']))
        for r in new:
            if r['certified_null_field_transfers']:
                self.assertEqual([x['reason'] for x in r['new_dynamic_gaps']],['untraced_call_effects'])
                self.assertFalse(r['qualified_after'])

    def test_scoped_control_does_not_claim_a_normal_or_general_exception_path(self):
        from cwm_fp.lddg.batch_audit import graph_reasons
        g=graph();q=g['quality']
        self.assertTrue(q['control_path_verified']);self.assertFalse(q['normal_path_verified'])
        self.assertFalse(q['exception_control_supported']);self.assertNotIn('unverified_control_path',graph_reasons(q))

    def test_exception_reference_evidence_is_available_only_at_handler_observation(self):
        g=graph();transfer=g['control_flow_evidence']['assessment']['certified_transfers'][0]
        value=next(n for n in g['nodes'] if n['id']==transfer['exception_value_id'])
        self.assertEqual(value['attrs']['evidence_available_at_event'],transfer['target_event'])
        self.assertGreater(value['attrs']['evidence_available_at_event'],value['attrs']['definition_event'])
        self.assertFalse(value['attrs']['observed'])
        value['attrs']['evidence_available_at_event']=value['attrs']['definition_event']
        self.assertIn('dataflow_value_evidence_mismatch',control_evidence_errors(g))


if __name__=='__main__':unittest.main()
