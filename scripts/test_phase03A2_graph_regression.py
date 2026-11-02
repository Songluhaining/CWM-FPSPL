"""The 18 phase04 regression checks that do not fit or evaluate a trained model."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT / 'src'))
from test_graph_classifier import GraphTests

NAMES = (
    'test_white_list_and_no_raw_values',
    'test_forbidden_fields_rejected_recursively',
    'test_rename_ids_preserves_tensors',
    'test_config_requires_constraint_verification',
    'test_evidence_gate_and_preview',
    'test_wrong_edge_evidence_and_order_rejected',
    'test_unknown_endpoint_type_and_duplicate_node',
    'test_ablation_removes_indirect_data_paths',
    'test_sampling_loss_accounting',
    'test_value_callback_origin_is_evidence_not_label_provenance',
    'test_hierarchical_pooling_is_not_all_node_average',
    'test_batch_has_no_cross_test_edges',
    'test_node_permutation_invariance',
    'test_agnostic_ignores_node_relation_types',
    'test_graph_logits_can_distinguish_topology',
    'test_partition_and_test_ownership_checks',
    'test_real_data_blocked_without_labels',
    'test_original_suite_join_and_partial_cohort_exclusion',
)

if __name__ == '__main__':
    with patch('test_graph_classifier.fit', side_effect=AssertionError('Model fitting forbidden in phase03A2')):
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.TestSuite(GraphTests(name) for name in sorted(NAMES)))
    raise SystemExit(not result.wasSuccessful())
