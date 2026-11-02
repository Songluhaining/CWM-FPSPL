"""Relevant regression without real classification-label or test-stream access."""
import json,sys,time,unittest,shutil
from pathlib import Path
import torch
from cwm_fp.data.common import ROOT,write_json
from cwm_fp.models.package_training import process_memory
sys.path.insert(0,str(ROOT/'tests'));torch.set_num_threads(1)
out=Path(sys.argv[1]);source=Path(sys.argv[2]);start=time.monotonic()
fixture=out/'regression_train_fixture';fixture.mkdir(exist_ok=True)
manifest=json.loads((source/'prefix_manifest.json').read_text(encoding='utf8'))
shutil.copy2(source/'prefix_manifest.json',fixture/'prefix_manifest.json')
for ref in [*manifest['traces'][0]['input_files'].values(),manifest['framework_static']]:
    p=fixture/ref['path'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/ref['path'],p)
import test_prefix_reader
test_prefix_reader.OUT=fixture
suite=unittest.TestSuite();loader=unittest.TestLoader()
for name in ('test_transition','test_prefix_observer','test_prefix_reader','test_package_training','test_package_comparison','test_graph_package'):
    suite.addTests(loader.loadTestsFromName(name))
for name in ('white_list_and_no_raw_values','forbidden_fields_rejected_recursively','rename_ids_preserves_tensors','config_requires_constraint_verification',
 'evidence_gate_and_preview','wrong_edge_evidence_and_order_rejected','unknown_endpoint_type_and_duplicate_node','ablation_removes_indirect_data_paths',
 'sampling_loss_accounting','value_callback_origin_is_evidence_not_label_provenance','hierarchical_pooling_is_not_all_node_average','batch_has_no_cross_test_edges',
 'node_permutation_invariance','agnostic_ignores_node_relation_types','graph_logits_can_distinguish_topology','forward_backward_all_models','relation_gradient_matches_finite_difference',
 'train_normalizer_rejects_heldout_failed_preview','partition_and_test_ownership_checks','original_suite_join_and_partial_cohort_exclusion'):
    suite.addTests(loader.loadTestsFromName('test_graph_classifier.GraphTests.test_'+name))
result=unittest.TextTestRunner(verbosity=2).run(suite)
write_json(out/'regression_tests.json',{'command':[sys.executable,*sys.argv],'tests_run':result.testsRun,'passed':result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
    'failures':[(str(x),v) for x,v in result.failures],'errors':[(str(x),v) for x,v in result.errors],'skipped':[(str(x),v) for x,v in result.skipped],
    'exit_code':0 if result.wasSuccessful() else 1,'seconds':time.monotonic()-start,'memory':process_memory(),'optimizer_updates':0,
    'real_classification_targets_read':0,'real_test_partition_streams_read':0,'real_Java_executions':0})
raise SystemExit(0 if result.wasSuccessful() else 1)
