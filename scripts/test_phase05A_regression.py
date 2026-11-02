"""No Java reruns, model fitting, product targets, or held-out source traces."""
import sys,unittest,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from cwm_fp.models.prefix_dataset import OUT,guard,load,PACKAGE
from cwm_fp.data.common import write_json
guard(load(PACKAGE/'manifest.json'),False)
import test_lddg,test_null_field_semantics,test_exception_control,test_graph_classifier,test_graph_package,test_package_training
loader=unittest.defaultTestLoader;suite=unittest.TestSuite()
for module in (test_lddg,):suite.addTests(loader.loadTestsFromModule(module))
for cls in (test_null_field_semantics.NullFieldTests,test_exception_control.ExceptionControlTests,
            test_graph_package.PrefixTests,test_package_training.PackageTrainingTests):
    for name in loader.getTestCaseNames(cls):
        if name in {'test_real_regression_and_14_partial_certifications','test_six_real_graphs_preserve_nodes_edges_and_static_identities',
                    'test_events_source_maps_and_33_non_executable_anchors_unchanged','test_transaction_paths_have_real_finally_tables_without_handler_execution',
                    'test_actual_guard_blocks_heldout_and_other_protocol','test_audit_transitions_have_valid_forward_event_endpoints'}:continue
        suite.addTest(cls(name))
for name in ['test_white_list_and_no_raw_values','test_forbidden_fields_rejected_recursively','test_rename_ids_preserves_tensors',
    'test_config_requires_constraint_verification','test_evidence_gate_and_preview','test_wrong_edge_evidence_and_order_rejected',
    'test_unknown_endpoint_type_and_duplicate_node','test_ablation_removes_indirect_data_paths','test_sampling_loss_accounting',
    'test_value_callback_origin_is_evidence_not_label_provenance','test_hierarchical_pooling_is_not_all_node_average','test_batch_has_no_cross_test_edges',
    'test_train_normalizer_rejects_heldout_failed_preview','test_partition_and_test_ownership_checks']:
    suite.addTest(test_graph_classifier.GraphTests(name))
start=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(suite)
write_json(OUT/'regression.json',{'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
    'skipped':len(result.skipped),'seconds':time.monotonic()-start,'model_fits':0,'new_java_executions':0})
raise SystemExit(0 if result.wasSuccessful() else 1)
