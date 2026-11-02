"""Synthetic unit inputs; real Java parity is checked by the separate bounded runner."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cwm_fp.lddg.static import control_dependence,load_static,mapping
from cwm_fp.lddg.trace import Dataflow,parse_trace,observed_scalar,descriptor_args
from cwm_fp.lddg.runner import completion,command
from cwm_fp.lddg.extract import validate_graph,graph_for


def event(i,op,fields=None,entry=False,domain=('Synthetic','method','()V'),thread=1):
    return {'event_index':i,'opcode':op,'fields':fields or {},'entry':entry,'domain':domain,'line':i+1,'bci':i,
            'thread_id':thread,'call_returned':False}


def static_empty():return {'fields':{},'methods':{},'allocations':{},'handlers':{},'controls':{},'gaps':{}}


class ControlTests(unittest.TestCase):
    def test_diamond_is_control_dependence_not_CFG(self):
        self.assertEqual(control_dependence({0:{1,2},1:{3},2:{3},3:{-1}}),{(0,1),(0,2)})

    def test_loop_retains_self_control_and_body(self):
        edges=control_dependence({0:{1,2},1:{0},2:{-1}})
        self.assertIn((0,1),edges);self.assertIn((0,0),edges);self.assertNotIn((0,2),edges)

    def test_infinite_region_and_unknown_target_fail_closed(self):
        with self.assertRaises(ValueError):control_dependence({0:{0}})
        with self.assertRaises(ValueError):control_dependence({0:{1}})

    def test_sequence_has_no_control_edge(self):self.assertEqual(control_dependence({0:{1},1:{-1}}),set())


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name)/'trace.log'

    def test_coverage_xml_never_creates_dynamic_graph(self):
        self.p.write_text('<coverage><line num="1" count="5"/></coverage>')
        with self.assertRaisesRegex(ValueError,'coverage XML is not DDG'):parse_trace(self.p,'Synthetic.test')

    def test_missing_thread_rejected(self):
        self.p.write_text('###Synthetic::test\nopcode=177(return),lineinfo=Synthetic#test#()V#1#0')
        with self.assertRaisesRegex(ValueError,'thread identity'):parse_trace(self.p,'Synthetic.test')

    def test_event_and_size_limits(self):
        self.p.write_text('###Synthetic::test\nopcode=177(return),lineinfo=Synthetic#test#()V#1#0,thread=2\n')
        with self.assertRaisesRegex(ValueError,'Event count'):parse_trace(self.p,'Synthetic.test',max_events=0)
        with self.assertRaisesRegex(ValueError,'file-size'):parse_trace(self.p,'Synthetic.test',max_bytes=1)

    def test_unknown_metadata_and_ret_marks_are_not_extra_events(self):
        self.p.write_text('###Synthetic::test\nopcode=184(invokestatic),lineinfo=Synthetic#test#()V#1#0,calltype=()V,callclass=org.junit.Assert,callname=assertTrue,thread=2,buggy=true,origin=simulated\n###RET@opcode=184(invokestatic),lineinfo=Synthetic#test#()V#1#0\n')
        rows=parse_trace(self.p,'Synthetic.test');self.assertEqual(len(rows),1);self.assertTrue(rows[0]['call_returned'])
        self.assertNotIn('buggy',str(rows));self.assertNotIn('simulated',str(rows))

    def test_values_are_only_actual_scalar_callbacks(self):
        self.assertEqual(observed_scalar('I:12')['value'],12)
        self.assertFalse(observed_scalar('Obj:123')['observed']);self.assertFalse(observed_scalar('Str:')['observed'])
        self.assertFalse(observed_scalar('D:NaN')['observed']);self.assertFalse(observed_scalar(None)['observed'])

    def test_descriptor_preserves_arrays_and_wide_slots(self):
        self.assertEqual(descriptor_args('(I[JLjava/lang/String;D)V'),['I','[J','Ljava/lang/String;','D'])


class RuntimeDependencyTests(unittest.TestCase):
    def test_stack_copy_has_fresh_definition_and_preserves_object_identity(self):
        rows=[event(0,187,entry=True),event(1,89),event(2,87),event(3,87),event(4,177)]
        flow=Dataflow(static_empty()).run(rows);self.assertFalse(flow.gaps)
        self.assertNotEqual(flow.values[0].id,flow.values[1].id)
        self.assertEqual(flow.values[0].ref,flow.values[1].ref)
        self.assertEqual(flow.values[1].definition,1)

    def test_local_write_versions_and_reaching_read(self):
        rows=[event(0,3,{'pushnum':'1'},True),event(1,60,{'store':'1'}),event(2,4,{'pushnum':'1'}),
              event(3,60,{'store':'1'}),event(4,27,{'load':'1'}),event(5,172)]
        flow=Dataflow(static_empty()).run(rows);self.assertFalse(flow.gaps)
        used=flow.events[4]['use_ids'][0];v=next(v for v in flow.values if v.id==used)
        self.assertEqual(v.definition,3);self.assertNotEqual(flow.events[1]['def_ids'],flow.events[3]['def_ids'])

    def test_missing_values_stay_missing_on_store(self):
        flow=Dataflow(static_empty()).run([event(0,3,{'stack':'I:0'},True),event(1,60,{'store':'1'}),event(2,177)])
        self.assertTrue(flow.values[0].observed['observed']);self.assertFalse(flow.values[1].observed['observed'])

    def test_cross_thread_without_happens_before_disables_dependencies(self):
        flow=Dataflow(static_empty()).run([event(0,3,entry=True),event(1,177,thread=2)])
        self.assertFalse(flow.links);self.assertIn('happens_before',flow.gaps[0]['reason'])

    def test_unobserved_local_does_not_gain_invented_def_use(self):
        flow=Dataflow(static_empty()).run([event(0,27,{'load':'1'},True),event(1,172)])
        self.assertTrue(flow.gaps);self.assertFalse(flow.links)

    def test_arrays_are_explicitly_unsupported(self):
        flow=Dataflow(static_empty()).run([event(0,188,entry=True),event(1,177)])
        self.assertTrue(any('unsupported_array' in g['reason'] for g in flow.gaps))

    def test_category2_arithmetic_is_not_silently_certified(self):
        flow=Dataflow(static_empty()).run([event(0,9,{'stack':'J:0'},True),event(1,10,{'stack':'J:1'}),
                                         event(2,97,{'popnum':'2','pushnum':'1','stack':'J:1'}),event(3,173)])
        self.assertTrue(any('category2' in g['reason'] for g in flow.gaps))

    def test_branch_uses_observed_successor_not_nextinst(self):
        st=static_empty();domain=('Synthetic','method','()V')
        st['methods'][domain]={1:{'successors':{2,4}}}
        rows=[event(0,3,entry=True),event(1,153,{'popnum':'1','nextinst':'2'}),event(2,177)];rows[2]['bci']=4
        flow=Dataflow(st).run(rows);self.assertTrue(flow.events[1]['branch_taken']);self.assertEqual(flow.events[1]['observed_successor'],4)


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.p=self.root/'result.tsv'

    def test_exit_zero_without_completion_is_unknown(self):self.assertEqual(completion(self.p,'C.test')['outcome'],'UNKNOWN')

    def test_ignored_or_wrong_test_not_counted_pass(self):
        self.p.write_text('COMPLETE\tC.test\t0\t0\t1\n');self.assertEqual(completion(self.p,'C.test')['outcome'],'UNKNOWN')
        self.p.write_text('COMPLETE\tC.other\t1\t0\t0\n');self.assertEqual(completion(self.p,'C.test')['outcome'],'UNKNOWN')

    def test_explicit_complete_failure_and_pass(self):
        self.p.write_text('FAILURE\tjava.lang.AssertionError\tQQ==\nCOMPLETE\tC.test\t1\t1\t0\n')
        self.assertEqual(completion(self.p,'C.test')['outcome'],'FAIL')
        self.p.write_text('COMPLETE\tC.test\t1\t0\t0\n');self.assertEqual(completion(self.p,'C.test')['outcome'],'PASS')

    def test_timeout_is_recorded_and_process_stopped(self):
        record=command([sys.executable,'-c','import time; time.sleep(10)'],self.root,self.root/'timeout.log',seconds=.1)
        self.assertEqual(record['reason'],'timeout');self.assertLess(record['seconds'],3)

    def test_output_limit_is_recorded(self):
        record=command([sys.executable,'-u','-c','import time; print("X"*10000,flush=True); time.sleep(10)'],self.root,self.root/'large.log',max_bytes=100)
        self.assertEqual(record['reason'],'output_limit')

    def test_fast_exit_cannot_bypass_output_limit(self):
        record=command([sys.executable,'-c','print("X"*10000)'],self.root,self.root/'fast.log',max_bytes=100)
        self.assertEqual(record['reason'],'output_limit')

    def test_generated_file_is_included_in_size_limit(self):
        record=command([sys.executable,'-c','from pathlib import Path; Path("trace.dat").write_bytes(b"X"*10000)'],
                       self.root,self.root/'file.log',max_bytes=100,monitor_files=True)
        self.assertEqual(record['reason'],'file_size_limit')


class MappingAndSchemaTests(unittest.TestCase):
    def graph_fixture(self):
        raw=[event(0,3,entry=True),event(1,172)];domain=raw[0]['domain'];st=static_empty()
        st['methods'][domain]={i:{'line':i+1,'opcode':r['opcode']} for i,r in enumerate(raw)}
        mapped={(*domain,i):{'canonical_statement_id':'Synthetic|method|'+str(i),'mapping_confidence':'synthetic','feature':None} for i in range(2)}
        status={'complete':True,'parity':True,'matches_historical_outcome':True,'runs':{'instrumented':{'result':{'outcome':'PASS'}}}}
        return raw,st,mapped,status

    def test_exception_control_and_CFG_gaps_block_training(self):
        raw,st,mapped,status=self.graph_fixture();domain=raw[0]['domain']
        st['handlers'][domain]=[[0,1,1]]
        graph,_=graph_for(raw,st,mapped,{}, {},status);self.assertFalse(graph['provenance']['learnable'])
        st['handlers']={};st['gaps']['|'.join(domain)]='no finite exit analysis'
        graph,_=graph_for(raw,st,mapped,{}, {},status);self.assertFalse(graph['provenance']['learnable'])

    def test_missing_completion_or_parity_blocks_graph_training(self):
        for flag in ['complete','parity','matches_historical_outcome']:
            raw,st,mapped,status=self.graph_fixture();status[flag]=False
            graph,_=graph_for(raw,st,mapped,{}, {},status);self.assertFalse(graph['provenance']['learnable'])

    def test_real_style_two_feature_variants_preserve_canonical_method(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'features/Base').mkdir(parents=True);(root/'features/Base/A.java').write_text('class A {\nint update() {return 1;}\n}\n')
            ids=[]
            for number,method in [(1,'update'),(2,'update__wrappee__Base')]:
                v=root/'variants'/str(number);(v/'coverage').mkdir(parents=True);(v/'src').mkdir();(v/'src/A.java').write_text('class A {}')
                (v/'roles.meta').write_text(json.dumps({'features':['Base'],'roles':{'update':{'Base':method}}}))
                (v/'coverage/spectrum_passed_coverage.xml').write_text('<coverage><project><package name="p"><file path="A.java"><line num="9" featureClass="Base.A" featureLineNum="2" buggy="true"/><line num="10" featureClass="Base.A" featureLineNum="3"/></file></package></project></coverage>')
                st=static_empty();st['methods']={('p.A',method,'()I'):{0:{'line':9}}}
                rows,by,quality=mapping(v,st,{'Base':True});ids.append(by[('p.A',method,'()I',0)]['canonical_statement_id'])
                self.assertTrue(any(r.get('status')=='unmapped_or_non_executable' for r in rows))
                self.assertNotIn('buggy',json.dumps(rows));self.assertFalse(quality['constraints_verified'])
                with self.assertRaises(ValueError):mapping(v,st,{'Other':True})
            self.assertEqual(ids[0],ids[1]);self.assertIsNotNone(ids[0])

    def test_future_or_label_edges_rejected(self):
        graph={'nodes':[{'id':'e:0'},{'id':'e:1'}],'edges':[{'source':'e:1','target':'e:0','type':'data_dep','evidence':'dynamic','definition_event':1,'event_index':0}]}
        with self.assertRaises(ValueError):validate_graph(graph,[{'event_index':0},{'event_index':1}])
        graph['edges']=[];graph['label']='FP'
        with self.assertRaises(ValueError):validate_graph(graph,[{'event_index':0},{'event_index':1}])

if __name__=='__main__':unittest.main()
