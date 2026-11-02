"""Bounded regression with no optimizer step or real held-out prediction."""
import sys,time,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from cwm_fp.data.common import write_json,sha256
from cwm_fp.baselines.boundaries import install_io_guard
OUT=Path(sys.argv[1])

def main():
    install_io_guard('features');start=time.monotonic();(OUT/'temp').mkdir(exist_ok=True)
    import test_graph_package as package
    import test_graph_classifier as graph
    import test_package_training as training
    package.OUT=OUT
    # Include model-independent graph regressions; all real held-out paths are integrity only.
    names=['test_white_list_and_no_raw_values','test_forbidden_fields_rejected_recursively','test_rename_ids_preserves_tensors',
        'test_config_requires_constraint_verification','test_evidence_gate_and_preview','test_wrong_edge_evidence_and_order_rejected',
        'test_unknown_endpoint_type_and_duplicate_node','test_ablation_removes_indirect_data_paths','test_sampling_loss_accounting',
        'test_value_callback_origin_is_evidence_not_label_provenance','test_hierarchical_pooling_is_not_all_node_average','test_batch_has_no_cross_test_edges',
        'test_train_normalizer_rejects_heldout_failed_preview','test_partition_and_test_ownership_checks']
    loader=unittest.defaultTestLoader;suite=unittest.TestSuite();ids=[]
    for cls in (training.PackageTrainingTests,package.PackageTests,package.PrefixTests):
        for name in loader.getTestCaseNames(cls):
            # Keep the original 58-case command reproducible; the later access-hook
            # test has its own bounded subprocess command and evidence log.
            if name in ('test_frozen_model_equivalence_and_train_mode_refusal','test_actual_guard_blocks_heldout_and_other_protocol'):continue
            t=cls(name);suite.addTest(t);ids.append(t.id())
    for name in names:
        t=graph.GraphTests(name);suite.addTest(t);ids.append(t.id())
    with (OUT/'unit_tests.log').open('w',encoding='utf8') as log:r=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    report={'tests_run':r.testsRun,'passed':r.testsRun-len(r.failures)-len(r.errors),'failures':[(t.id(),s) for t,s in r.failures],
        'errors':[(t.id(),s) for t,s in r.errors],'seconds':time.monotonic()-start,'test_ids':ids,'optimizer_updates':0,
        'real_heldout_model_executions':0,'exit_code':0 if r.wasSuccessful() else 1,'log_sha256':sha256(OUT/'unit_tests.log')}
    write_json(OUT/'unit_tests.json',report);print('UNIT_TESTS',r.testsRun,'passed',r.wasSuccessful(),flush=True)
    return report['exit_code']

if __name__=='__main__':raise SystemExit(main())
