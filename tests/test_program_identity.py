"""Identity evidence tests using the separately executed synthetic Java fixtures."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cwm_fp.lddg.static import SourceEvidence,load_static,mapping,program_identities,trusted_program_identity
from cwm_fp.lddg.extract import identity_errors,identity_metrics,validate_graph

OUT=ROOT/'data/processed/lddg/phase03A2_identity_v1'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.root=OUT/'synthetic';self.variant=self.root/'variants/fixture'
        self.source=SourceEvidence(self.root/'source_evidence.json');self.static=load_static(self.root/'static.tsv')
        self.binding=load(self.root/'compilation_binding.json')
        self.rows,self.mapped,_=mapping(self.variant,self.static,{'Base':True},self.source)

    def identities(self):return program_identities(self.variant,self.static,self.mapped,self.source,self.binding)

    def test_no_synthetic_flag_is_required_for_real_default_constructor(self):
        ids=self.identities();selected=[v for k,v in ids.items() if k[0]=='identityfixture.Implicit']
        self.assertEqual(len(selected),3)
        self.assertTrue(all(v['provenance_type']=='COMPILER_GENERATED' and trusted_program_identity(v) for v in selected))
        self.assertTrue(all(v['evidence']['default_constructor']['acc_synthetic'] is False for v in selected))

    def test_explicit_constructor_with_same_opcode_shape_is_source_backed(self):
        a=('identityfixture.Implicit','<init>','()V');b=('identityfixture.Explicit','<init>','()V')
        self.assertEqual([i['opcode'] for i in self.static['methods'][a].values()],[i['opcode'] for i in self.static['methods'][b].values()])
        self.static['method_flags'][b]|=4096  # A flag alone must not override source evidence.
        selected=[v for k,v in self.identities().items() if k[0]==b[0]]
        self.assertTrue(all(v['provenance_type']=='SOURCE_BACKED' for v in selected))

    def test_explicit_field_stays_source_backed_inside_default_constructor(self):
        selected=[v['provenance_type'] for k,v in self.identities().items() if k[0]=='identityfixture.WithField']
        self.assertEqual(selected.count('COMPILER_GENERATED'),3);self.assertEqual(selected.count('SOURCE_BACKED'),3)

    def test_absent_AST_is_unresolved_even_with_synthetic_flag(self):
        self.source.records=[r for r in self.source.records if r['class']!='identityfixture.Implicit']
        self.static['method_flags'][('identityfixture.Implicit','<init>','()V')]=4096
        selected=[v for k,v in self.identities().items() if k[0]=='identityfixture.Implicit']
        self.assertTrue(all(v['provenance_type']=='UNRESOLVED' and v['instruction_id'] is None for v in selected))

    def test_wrong_artifact_binding_cannot_certify_generated_identity(self):
        self.binding['class_sha256']={}
        self.assertTrue(all(not trusted_program_identity(v) for v in self.identities().values()))

    def test_unsupported_superclass_is_not_certified_by_missing_constructor(self):
        self.static['classes']['identityfixture.Implicit']['superclass']='example.UnknownBase'
        self.assertTrue(all(v['provenance_type']=='UNRESOLVED' for k,v in self.identities().items() if k[0]=='identityfixture.Implicit'))

    def test_non_linear_constructor_is_not_guessed(self):
        self.static['methods'][('identityfixture.Implicit','<init>','()V')][0]['successors']={1,4}
        self.assertTrue(all(v['provenance_type']=='UNRESOLVED' for k,v in self.identities().items() if k[0]=='identityfixture.Implicit'))

    def test_type_flag_alone_cannot_promote_unknown(self):
        v=copy.deepcopy(next(iter(self.identities().values())))
        v['provenance_type']='COMPILER_GENERATED';v['evidence']={}
        self.assertFalse(trusted_program_identity(v))

    def test_missing_bytecode_call_evidence_does_not_create_default_identity(self):
        self.static['calls']={}
        self.assertTrue(all(v['provenance_type']=='UNRESOLVED' for k,v in self.identities().items() if k[0]=='identityfixture.Implicit'))

    def test_same_instruction_repeated_has_different_runtime_id_and_frame(self):
        events=load(self.root/'repeatedConstruction/events.json')
        generated=[e for e in events if e['program_identity']['locator']['declaring_class']=='identityfixture.Implicit']
        self.assertEqual(len(generated),6);self.assertEqual(len({e['static_instruction_id'] for e in generated}),3)
        self.assertEqual(len({e['canonical_event_id'] for e in generated}),6);self.assertEqual(len({e['frame_id'] for e in generated}),2)

    def test_cross_class_equal_opcodes_do_not_force_same_instruction(self):
        ids=self.identities();a=ids[('identityfixture.Implicit','<init>','()V',0)];b=ids[('identityfixture.Explicit','<init>','()V',0)]
        self.assertNotEqual(a['instruction_id'],b['instruction_id'])

    def test_real_446_sources_15_generated_and_33_anchors_are_preserved(self):
        rows=lines(OUT/'source_map.jsonl');programs=[r for r in rows if r['executable']];anchors=[r for r in rows if not r['executable']]
        self.assertEqual(sum(r['program_identity']['provenance_type']=='SOURCE_BACKED' for r in programs),446)
        generated=[r for r in programs if r['program_identity']['provenance_type']=='COMPILER_GENERATED']
        self.assertEqual(len(generated),15);self.assertTrue(all(r['canonical_source_line_id'] is None for r in generated))
        self.assertEqual(len(anchors),33);self.assertTrue(all('program_identity' not in r for r in anchors))

    def test_all_real_event_static_links_and_quality_are_consistent(self):
        graphs=lines(OUT/'graphs.jsonl');events=lines(OUT/'events.jsonl')
        for g in graphs:
            self.assertFalse(identity_errors(g))
            selected=[e for e in events if (e['product_id'],e['test_id'])==(g['product_id'],g['test_id'])]
            validate_graph(g,selected)
            self.assertEqual(identity_metrics(g)[0]['dynamic_event_identity_coverage'],1)

    def test_false_quality_flag_cannot_hide_changed_static_identity(self):
        g=copy.deepcopy(lines(OUT/'graphs.jsonl')[0]);node=next(n for n in g['nodes'] if n['id'].startswith('s:'))
        node['attrs']['program_identity']['instruction_id']='instruction-v1:incorrect'
        g['quality']['program_identity_complete']=True
        self.assertTrue(identity_errors(g))

    def test_event_reassignment_is_rejected(self):
        g=copy.deepcopy(lines(OUT/'graphs.jsonl')[0]);event=next(n for n in g['nodes'] if n['id'].startswith('e:'))
        event['attrs']['static_instruction_id']='another-instruction'
        self.assertTrue(identity_errors(g))


if __name__=='__main__':unittest.main()
