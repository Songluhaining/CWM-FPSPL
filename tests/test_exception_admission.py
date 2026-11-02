"""No fitting: revised control proof versus existing model admission gates."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cwm_fp.models.graph_encoder import encode_test
from cwm_fp.models.graph_data import load_corpus

OUT=ROOT/'data/processed/lddg/phase03A3_exception_v1'
def graphs():return [json.loads(s) for s in (OUT/'graphs.jsonl').read_text(encoding='utf-8').splitlines()]


class ExceptionAdmissionTests(unittest.TestCase):
    def test_added_control_proof_does_not_change_existing_numeric_features(self):
        import torch
        old=[json.loads(s) for s in (ROOT/'data/processed/lddg/phase03A2_identity_v1/graphs.jsonl').read_text(encoding='utf-8').splitlines()]
        for i in (0,1,2,4):
            a=encode_test(old[i],'both');b=encode_test(graphs()[i],'both')
            self.assertTrue(torch.equal(a.x,b.x));self.assertTrue(torch.equal(a.edge_index,b.edge_index))
            self.assertTrue(torch.equal(a.edge_type,b.edge_type))

    def test_six_normal_path_graphs_encode_without_event_or_edge_sampling_loss(self):
        for g in graphs():
            t=encode_test(g,'both');self.assertEqual(t.x.shape[1],22)
            self.assertFalse(t.metadata['preview_only'])
            self.assertEqual(t.metadata['lost_event_fraction'],0)
            self.assertEqual(t.metadata['lost_edge_fraction'],0)

    def test_handler_method_cannot_downgrade_to_legacy_by_removing_new_proof(self):
        for keys in [('control_flow_evidence',),('control_schema',),('control_flow_evidence','control_schema')]:
            g=copy.deepcopy(graphs()[3])
            for k in keys:
                if k=='control_schema':del g['provenance'][k]
                else:del g[k]
            with self.assertRaises(ValueError):encode_test(g,'both')

    def test_unsupported_exceptions_rejected_even_in_preview_and_with_fake_quality_flags(self):
        for name in ('sameMethodThrow','callerCatch','exceptionalFinally','untracedException','implicitDivide'):
            g=json.loads((OUT/'synthetic'/name/'graph.json').read_text(encoding='utf-8'))
            g['quality'].update(eligible_graph_structure=True,verified_tier='T1',dependency_semantics_valid=True)
            g['provenance']['learnable']=True
            for preview in (False,True):
                with self.assertRaises(ValueError):encode_test(g,'both',preview=preview)

    def test_identity_checks_remain_mandatory(self):
        g=copy.deepcopy(graphs()[3]);n=next(n for n in g['nodes'] if n['type']=='statement')
        n['attrs']['program_identity']['provenance_type']='UNRESOLVED'
        with self.assertRaises(ValueError):encode_test(g,'both')

    def test_full_exception_claim_cannot_be_accepted(self):
        g=copy.deepcopy(graphs()[3]);g['quality']['exception_control_supported']=True
        with self.assertRaises(ValueError):encode_test(g,'both')

    def test_no_product_admitted_without_complete_original_suite(self):
        m=ROOT/'data/processed/manifests';s=ROOT/'data/processed/splits/splits.csv'
        _,_,_,eligible,a=load_corpus(m/'products.csv',m/'tests.csv',OUT/'graphs.jsonl',s)
        self.assertFalse(eligible);self.assertEqual(a['identity_and_semantics_qualified'],6)
        self.assertEqual(len(a['products']),3)
        for p in a['products']:
            self.assertFalse(p['suite_completeness']['complete'])
            self.assertIn('incomplete_original_test_collection',p['reasons'])


if __name__=='__main__':unittest.main()
