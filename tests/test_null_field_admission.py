"""Stage04 checks only, never model fitting."""
import copy
import unittest
from test_null_field_semantics import graph,load,OUT
from cwm_fp.models.graph_encoder import encode_test,Limits


class NullFieldAdmissionTests(unittest.TestCase):
    def test_supported_exception_graphs_use_existing_22_features_losslessly(self):
        for name in ('sameRead','sameWrite','callerRead','callerWrite','repeated','aliasedNull'):
            g=graph(name);t=encode_test(g,'both',Limits(max_nodes=len(g['nodes']),max_events=g['quality']['events'],max_edges=len(g['edges'])))
            self.assertEqual(t.x.shape[1],22);self.assertEqual(t.metadata['lost_event_fraction'],0);self.assertEqual(t.metadata['lost_edge_fraction'],0)

    def test_unknown_side_effects_cannot_be_admitted_by_flipping_flags(self):
        g=graph('externalEffectsRejected');g['quality'].update(eligible_graph_structure=True,verified_tier='T1',dependency_semantics_valid=True,unsupported_dynamic=[])
        g['provenance']['learnable']=True
        for preview in (False,True):
            with self.assertRaises(ValueError):encode_test(g,'both',preview=preview)

    def test_all_negative_java_cases_rejected(self):
        for name in ('nullInvokeRejected','castRejected','broadCatchRejected','finallyRejected','externalEffectsRejected','divideRejected'):
            with self.assertRaises(ValueError):encode_test(graph(name),'both')

    def test_formal_real_admission_retains_only_old_complete_product(self):
        a=load(OUT/'admission.json')
        self.assertEqual(len(a['products']),8)
        for p in a['products']:
            for r in p['folds'].values():self.assertEqual(r['eligible'],p['batch_product']=='reuse03B1')

    def test_full_exception_support_overclaim_rejected(self):
        g=graph();g['quality']['exception_control_supported']=True
        with self.assertRaises(ValueError):encode_test(g,'both')


if __name__=='__main__':unittest.main()
