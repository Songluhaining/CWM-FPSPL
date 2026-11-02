"""Source-mapping regression tests. Synthetic Java is parsed, never executed."""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cwm_fp.data.common import sha256
from cwm_fp.lddg.static import SourceEvidence, java_tokens, mapping, source_descriptor
from cwm_fp.lddg.extract import graph_for


class SourceMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.tmp.cleanup)
        cls.root=Path(cls.tmp.name);cls.v=cls.root/'variants/product'
        (cls.v/'src').mkdir(parents=True);(cls.v/'coverage').mkdir()
        (cls.root/'features/Base').mkdir(parents=True)
        cls.feature=cls.root/'features/Base/A.java';cls.product=cls.v/'src/A.java'
        # Both overloads have distinct signatures; a field contains significant string whitespace.
        cls.feature.write_text('package p;\nclass A {\n int value = 1;\n String text = "a b";\n void f(int x) {\n value = x;\n }\n void f(boolean x) {\n value = x ? 1 : 0;\n }\n static class Inner {\n void g() {\n }\n }\n}\n',encoding='utf-8')
        cls.product.write_text('package p;\nclass A {\n int value = 1;\n String text = "a b";\n //__feature_mapping__ [Base] [5:7]\n private void f__wrappee__Base(int x) {\n value = x;\n }\n //__feature_mapping__ [Base] [8:10]\n void f(boolean x) {\n value = x ? 1 : 0;\n }\n static class Inner {\n //__feature_mapping__ [Base] [12:13]\n void g() {\n }\n }\n}\n',encoding='utf-8')
        (cls.v/'roles.meta').write_text(json.dumps({'features':['Base'],'roles':{'f':{'Base':'f__wrappee__Base'},'g':{'Base':'g'}}}))
        cls.java=Path(json.loads((ROOT/'configs/lddg_smoke.json').read_text())['java_home'])
        cls.build=cls.root/'parser';cls.build.mkdir()
        subprocess.run([str(cls.java/'bin/javac.exe'),'-encoding','UTF-8','-cp',str(cls.java/'lib/tools.jar'),'-d',str(cls.build),
                        str(ROOT/'src/cwm_fp/lddg/java/Phase03SourceExport.java')],check=True,capture_output=True,timeout=30)
        cls.evidence=cls.export([cls.product,cls.feature])

    @classmethod
    def export(cls,paths):
        raw=cls.root/'ast.jsonl'
        subprocess.run([str(cls.java/'bin/java.exe'),'-Xmx128m','-cp',os.pathsep.join(map(str,[cls.build,cls.java/'lib/tools.jar'])),
                        'Phase03SourceExport',str(raw),*map(str,paths)],check=True,capture_output=True,timeout=30)
        payload={'records':[json.loads(s) for s in raw.read_text().splitlines()], 'source_sha256':{str(p):sha256(p) for p in paths}}
        dest=cls.root/'evidence.json';dest.write_text(json.dumps(payload),encoding='utf-8')
        return SourceEvidence(dest)

    def state(self,domain,line,op=177):
        return {'methods':{domain:{0:{'line':line,'opcode':op},1:{'line':line,'opcode':op}}},
                'fields':{},'controls':{},'handlers':{},'gaps':{},'allocations':{}}

    def test_wrapper_signature_and_line_mapping_without_coverage(self):
        d=('p.A','f__wrappee__Base','(I)V');st=self.state(d,8)
        rows,by,_=mapping(self.v,st,{'Base':True},self.evidence)
        self.assertEqual(rows[0]['original'],['Base','p.A',7])
        self.assertEqual(rows[0]['source_construct'],'implicit_return_at_explicit_method_end')
        self.assertIsNone(rows[0]['statement_ordinal']);self.assertFalse(rows[0]['source_statement_verified'])
        self.assertTrue(rows[0]['canonical_source_line_id'].startswith('line|'))
        self.assertEqual(len(by),2)  # Same source line never collapses bytecode offsets.

    def test_wrong_overload_cannot_inherit_other_method_mapping(self):
        d=('p.A','f__wrappee__Base','(Z)V')
        rows,_,_=mapping(self.v,self.state(d,8),{'Base':True},self.evidence)
        self.assertTrue(all(r['canonical_statement_id'] is None for r in rows))

    def test_inner_class_is_resolved_from_AST_not_dollar_filename(self):
        d=('p.A$Inner','g','()V')
        rows,_,_=mapping(self.v,self.state(d,16),{'Base':True},self.evidence)
        self.assertEqual(len(rows),2);self.assertEqual(rows[0]['original'],['Base','p.A$Inner',13])

    def test_field_requires_real_qualified_store(self):
        d=('p.A','<init>','()V');st=self.state(d,3,181)
        rows,_,_=mapping(self.v,st,{'Base':True},self.evidence)
        self.assertIsNone(rows[0]['canonical_statement_id'])
        st['fields'][(*d,0)]=('p.A','value','Z')
        rows,_,_=mapping(self.v,st,{'Base':True},self.evidence)
        self.assertIsNone(rows[0]['canonical_statement_id'])
        st['fields'][(*d,0)]=('p.A','value','I')
        rows,_,_=mapping(self.v,st,{'Base':True},self.evidence)
        self.assertEqual(rows[0]['original'],['Base','p.A',3])

    def test_implicit_constructor_is_not_invented_source_method(self):
        d=('p.A','<init>','()V')
        rows,_,_=mapping(self.v,self.state(d,2,183),{'Base':True},self.evidence)
        self.assertEqual(rows[0]['source_construct'],'implicit_default_constructor')
        self.assertIsNone(rows[0]['canonical_statement_id'])

    def test_ambiguous_feature_field_stays_unknown(self):
        other=self.root/'features/Other/A.java';other.parent.mkdir(exist_ok=True)
        other.write_bytes(self.feature.read_bytes());roles=self.v/'roles.meta';saved=roles.read_bytes()
        try:
            roles.write_text(json.dumps({'features':['Base','Other'],'roles':{'f':{'Base':'f__wrappee__Base'},'g':{'Base':'g'}}}))
            evidence=self.export([self.product,self.feature,other]);d=('p.A','<init>','()V');st=self.state(d,3,181)
            st['fields'][(*d,0)]=('p.A','value','I')
            rows,_,_=mapping(self.v,st,{'Base':True,'Other':True},evidence)
            self.assertEqual(rows[0]['mapping_note'],'UNKNOWN_ambiguous_equal_feature_members')
            self.assertIsNone(rows[0]['canonical_source_line_id'])
        finally:roles.write_bytes(saved)

    def test_changed_body_cannot_use_old_annotation_as_proof(self):
        saved=self.product.read_bytes()
        try:
            self.product.write_bytes(saved.replace(b'value = x;',b'value = x+1;'))
            evidence=self.export([self.product,self.feature]);d=('p.A','f__wrappee__Base','(I)V')
            rows,_,_=mapping(self.v,self.state(d,8),{'Base':True},evidence)
            self.assertIsNone(rows[0]['canonical_statement_id'])
        finally:self.product.write_bytes(saved)

    def test_stale_source_hash_is_rejected(self):
        saved=self.product.read_bytes();self.export([self.product,self.feature])
        try:
            self.product.write_bytes(saved+b'\n')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):SourceEvidence(self.root/'evidence.json')
        finally:self.product.write_bytes(saved)

    def test_unknown_type_is_not_guessed(self):
        self.assertIsNone(source_descriptor({'class':'p.A','params':'Missing','type':'void'}, {'p.A'}))

    def test_literals_and_comments_do_not_create_false_token_matches(self):
        self.assertNotEqual(java_tokens('"a b"'),java_tokens('"ab"'))
        self.assertEqual([t for t,l in java_tokens('x /* example */ + 1')],['x','+','1'])
        with self.assertRaisesRegex(ValueError,'Unicode escape'):java_tokens(r'\u0061')

    def test_unexecuted_unknown_static_location_does_not_pass_full_mapping_gate(self):
        from test_lddg import MappingAndSchemaTests
        raw,st,mapped,status=MappingAndSchemaTests().graph_fixture()
        mapped[(*raw[0]['domain'],99)]={'canonical_statement_id':None,'mapping_confidence':'unmapped','feature':None}
        graph,events=graph_for(raw,st,mapped,{}, {},status)
        self.assertTrue(graph['quality']['dynamic_product_mapping_complete'])
        self.assertFalse(graph['quality']['canonical_mapping_complete'])
        self.assertEqual(len(events),2)  # No artificial event for the unexecuted location.

    def test_repeat_execution_events_keep_distinct_node_identities(self):
        from test_lddg import MappingAndSchemaTests
        raw,st,mapped,status=MappingAndSchemaTests().graph_fixture()
        # Repeated actual trace entries at one bytecode offset keep event identity.
        first=copy.deepcopy(raw[0]);first.update(event_index=1,entry=False)
        raw=[raw[0],first,{**raw[1],'event_index':2}]
        graph,events=graph_for(raw,st,mapped,{}, {},status)
        self.assertEqual([e['statement_id'] for e in events],['e:0','e:1','e:2'])
        nodes={n['id']:n for n in graph['nodes']}
        self.assertEqual(nodes['e:0']['attrs']['static_statement_id'],nodes['e:1']['attrs']['static_statement_id'])


if __name__=='__main__':unittest.main()
