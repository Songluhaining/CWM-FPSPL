"""No fitting: identity gates, schema compatibility and semantic-only encoding."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from cwm_fp.lddg.static import instruction_identity_id,identity_digest,event_identity_id
from cwm_fp.lddg.extract import identity_metrics
from cwm_fp.models.graph_encoder import encode_test,evidence_errors,dependency_errors,FEATURES,FEATURES_V1,FEATURE_SCHEMA,Limits,collate,ProductTensor
from cwm_fp.models.graph_classifier import GraphClassifier
from cwm_fp.models.graph_data import load_corpus,tensor_products

OUT=ROOT/'data/processed/lddg/phase03A2_identity_v1'
def graphs():return [json.loads(s) for s in (OUT/'graphs.jsonl').read_text(encoding='utf-8').splitlines()]


class IdentityAdmissionTests(unittest.TestCase):
    def test_generated_instructions_accepted_without_inventing_source_lines(self):
        g=graphs()[0];self.assertFalse(g['quality']['canonical_mapping_complete'])
        t=encode_test(g,'both');self.assertFalse(t.metadata['preview_only']);self.assertEqual(t.x.shape[1],22)
        self.assertGreater(t.x[:,FEATURES.index('program_compiler_generated')].sum(),0)

    def test_exception_graphs_still_rejected_after_identity_completion(self):
        for g in (graphs()[3],graphs()[5]):
            self.assertTrue(g['quality']['program_identity_complete'])
            for preview in (True,False):
                with self.assertRaises(ValueError):encode_test(g,'both',preview=preview)

    def test_unknown_or_tampered_identity_rejected_even_with_claimed_complete_flag(self):
        g=copy.deepcopy(graphs()[0]);n=next(n for n in g['nodes'] if n['id'].startswith('s:'))
        n['attrs']['program_identity']['provenance_type']='UNRESOLVED'
        for preview in (True,False):
            with self.assertRaises(ValueError):encode_test(g,'both',preview=preview)

    def test_identity_schema_cannot_be_removed_to_fall_back_to_old_gate(self):
        g=copy.deepcopy(graphs()[0]);del g['provenance']['identity_schema'];g['quality']['canonical_mapping_complete']=True
        with self.assertRaises(ValueError):encode_test(g,'both')

    def test_identity_and_dependency_quality_remain_independent(self):
        g=copy.deepcopy(graphs()[0]);n=next(n for n in g['nodes'] if n['id'].startswith('s:'))
        n['attrs']['program_identity']['provenance_type']='UNRESOLVED'
        self.assertFalse(dependency_errors(g));self.assertTrue(evidence_errors(g))

    def test_runtime_values_are_separate_and_missing_is_explicit(self):
        g=graphs()[0];a=encode_test(g,'both');changed=copy.deepcopy(g)
        for n in changed['nodes']:
            if n['type']=='value' and n['attrs'].get('observed'):n['attrs']['value']=123456
        b=encode_test(changed,'both');self.assertTrue(torch.equal(a.x,b.x))
        self.assertTrue(a.metadata['ordered_event_nodes'])
        self.assertTrue(any(x['observed'] for x in a.metadata['state_observations']))
        self.assertTrue(all(x['value'] is None for x in a.metadata['state_observations'] if not x['observed']))
        self.assertFalse(a.metadata['state_observations_used_as_classifier_features'])

    def test_position_product_and_method_names_are_not_numeric_features(self):
        # Rename only audit identity coordinates and consistently re-key certificates;
        # semantic origin, opcodes, relationships and runtime order stay identical.
        g=graphs()[0];a=encode_test(g,'both');b=copy.deepcopy(g)
        b['system_id']='renamed';b['bug_version_id']='other-version';b['product_id']='other-product';b['test_id']='other-test'
        def renamed(p):
            p=copy.deepcopy(p);p['locator']['declaring_class']='renamed.'+p['locator']['declaring_class'];p['locator']['method']='renamed_'+p['locator']['method'] if p['locator']['method']!='<init>' else '<init>'
            p['locator']['bytecode_offset']+=1000;p['instruction_id']=instruction_identity_id(p['locator']);return p
        b['instruction_catalog']=[renamed(p) for p in b['instruction_catalog']]
        by_locator={(p['locator']['declaring_class'],p['locator']['method'],p['locator']['descriptor'],p['locator']['bytecode_offset']):p for p in b['instruction_catalog']}
        for n in b['nodes']:
            if n['type']!='statement':continue
            attr=n['attrs'];p=renamed(attr['program_identity']);loc=p['locator'];attr['program_identity']=p
            attr.update(composed_class=loc['declaring_class'],composed_method=loc['method'],bytecode_offset=loc['bytecode_offset'],composed_line=99999)
            if 'event_index' in attr:
                attr['static_instruction_id']=p['instruction_id'];attr['canonical_event_id']=event_identity_id({k:b[k] for k in ('system_id','bug_version_id','product_id','test_id')},b['provenance']['trace_sha256'],attr,p['instruction_id'])
        b['quality'].update(identity_metrics(b)[0]);encoded=encode_test(b,'both')
        self.assertTrue(torch.equal(a.x,encoded.x));self.assertTrue(torch.equal(a.edge_index,encoded.edge_index));self.assertTrue(torch.equal(a.edge_type,encoded.edge_type))

    def test_legacy_19_column_schema_is_explicit_and_not_silently_mixed(self):
        g=json.loads((ROOT/'data/processed/lddg/phase03A_mapping_v1/graphs.jsonl').read_text(encoding='utf-8').splitlines()[0])
        old=encode_test(g,'both',preview=True,feature_schema='semantic_v1');new=encode_test(g,'both',preview=True)
        self.assertEqual(old.x.shape[1],19);self.assertEqual(new.x.shape[1],22)
        self.assertTrue(torch.equal(old.x[:,:14],new.x[:,:14]));self.assertTrue(torch.equal(old.x[:,-5:],new.x[:,-5:]))
        p=ProductTensor(('s','v','p'),'sample','train','PASS',[old,new],{})
        with self.assertRaises(ValueError):collate([p])
        legacy_model=GraphClassifier(feature_schema='semantic_v1');legacy_model.load_state_dict(copy.deepcopy(legacy_model.state_dict()))
        current_model=GraphClassifier()
        with self.assertRaises(RuntimeError):current_model.load_state_dict(legacy_model.state_dict())

    def test_single_qualified_test_cannot_become_training_product(self):
        g=graphs()[0];k=tuple(g[x] for x in ('system_id','bug_version_id','product_id'))
        with self.assertRaisesRegex(ValueError,'complete original product suite'):
            tensor_products({k:{'observed_result':'PASS'}},{k:{'split':'train'}},{k:[g]},'both',Limits())

    def test_real_products_rejected_for_suites_and_semantics_not_generated_nodes(self):
        m=ROOT/'data/processed/manifests';s=ROOT/'data/processed/splits/splits.csv'
        _,_,_,eligible,audit=load_corpus(m/'products.csv',m/'tests.csv',OUT/'graphs.jsonl',s)
        self.assertFalse(eligible);self.assertEqual(audit['program_identity_complete'],6);self.assertEqual(audit['identity_and_semantics_qualified'],4)
        for p in audit['products']:
            self.assertIn('incomplete_original_test_collection',p['reasons']);self.assertNotIn('incomplete_canonical_mapping',p['reasons'])
            self.assertFalse(p['suite_completeness']['complete'])


if __name__=='__main__':unittest.main()
