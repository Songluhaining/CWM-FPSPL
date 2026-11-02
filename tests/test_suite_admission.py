"""Actual eight-graph interface and negative gates, never fit or read labels."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from cwm_fp.lddg.suite import OUT,ROOT,TARGET,lines,load
from cwm_fp.models.graph_data import load_corpus,tensor_products
from cwm_fp.models.graph_encoder import Limits,collate

class CompleteSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graphs=lines(OUT/'graphs.jsonl')
        cls.products=ROOT/'data/processed/manifests/products.csv'
        cls.tests=ROOT/'data/processed/manifests/tests.csv'
        cls.split=ROOT/'data/processed/splits/splits.csv'

    def corpus(self,graphs):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'graphs.jsonl';p.write_text(''.join(json.dumps(g)+'\n' for g in graphs),encoding='utf-8')
            return load_corpus(self.products,self.tests,p,self.split)

    def test_complete_eight_product_eligible_but_experiment_not_ready(self):
        *_,eligible,audit=self.corpus(self.graphs)
        self.assertEqual(set(eligible),{TARGET});self.assertFalse(audit['ready'])
        self.assertEqual(audit['eligible_by_partition'],{'train':0,'val':1,'test':0})

    def test_missing_test_cannot_be_admitted(self):
        *_,eligible,audit=self.corpus(self.graphs[:-1])
        self.assertFalse(eligible);self.assertIn('incomplete_original_test_collection',audit['products'][0]['reasons'])

    def test_duplicate_test_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.corpus(self.graphs+[self.graphs[0]])

    def test_unknown_test_rejected(self):
        gs=copy.deepcopy(self.graphs);gs[0]['test_id']='main.Account_ESTest.not_registered'
        with self.assertRaisesRegex(ValueError,'unknown'):self.corpus(gs)

    def test_forged_suite_count_rejected(self):
        gs=copy.deepcopy(self.graphs);gs[0]['quality']['suite_completeness']['collected_tests']=7
        *_,eligible,audit=self.corpus(gs);self.assertFalse(eligible)
        self.assertIn('suite_completeness_claim_disagrees_with_manifest',audit['products'][0]['reasons'])

    def test_one_unknown_identity_rejects_whole_product(self):
        gs=copy.deepcopy(self.graphs);gs[0]['instruction_catalog'][0]['provenance_type']='UNRESOLVED'
        *_,eligible,audit=self.corpus(gs);self.assertFalse(eligible)

    def test_one_unverified_control_path_rejects_whole_product(self):
        gs=copy.deepcopy(self.graphs);gs[0]['control_flow_evidence']['assessment']['accepted']=False
        *_,eligible,audit=self.corpus(gs);self.assertFalse(eligible)

    def test_all_eight_retained_no_cross_test_dynamic_edges(self):
        products,splits,_,eligible,_=self.corpus(self.graphs)
        items=tensor_products(products,splits,eligible,'both',Limits());batch=collate(items)
        self.assertEqual(len(items[0].tests),8);self.assertEqual(batch.x.shape[1],22)
        for name in ('lost_test_fraction','lost_event_fraction','lost_edge_fraction'):
            self.assertEqual(items[0].audit[name],0)
        for a,b in batch.edge_index.T.tolist():self.assertEqual(batch.node_test[a],batch.node_test[b])

    def test_dynamic_ids_unique_but_static_ids_align_by_real_evidence(self):
        events=lines(OUT/'events.jsonl');ids=[e['canonical_event_id'] for e in events]
        self.assertEqual(len(ids),len(set(ids)))
        common=set.intersection(*[{x['instruction_id'] for x in g['instruction_catalog'] if x['scope']=='product'} for g in self.graphs])
        self.assertEqual(len(common),47)

if __name__=='__main__':unittest.main()
