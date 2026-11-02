"""Bounded single-graph representation checks; never admit partial products or fit."""
from collections import Counter
import json
from cwm_fp.baselines.boundaries import install_io_guard
from cwm_fp.data.common import write_json
from cwm_fp.lddg.batch import read_plan
from cwm_fp.lddg.batch_audit import sources
from cwm_fp.models.graph_encoder import encode_test, evidence_errors, Limits


def main():
    install_io_guard('features')
    config, out = read_plan()
    records, unsupported_ops = [], Counter()
    for entry in sources(config, out):
        if not entry['graphs'].exists():
            continue
        with entry['graphs'].open(encoding='utf-8') as stream:
            for line in stream:
                try:
                    graph = json.loads(line)
                except json.JSONDecodeError:
                    break  # An interrupted file stays unsealed and inadmissible.
                events = {n['attrs']['event_index']: n['attrs']['opcode'] for n in graph['nodes']
                          if n['type'] == 'statement' and 'event_index' in n['attrs']}
                # The producer's combined reason includes arrays, casts and monitors.
                # Count actual opcodes rather than claiming every such gap is an array.
                for opcode in {events.get(g['event_index']) for g in graph['quality']['unsupported_dynamic']
                               if g['reason'] == 'unsupported_array_type_or_monitor_opcode'}:
                    unsupported_ops[(entry['identity'][0], opcode)] += 1
                if evidence_errors(graph):
                    continue
                record = {'identity': [*entry['identity'], graph['test_id']], 'reused': entry['reused'],
                          'original_product_result': entry['observed_result'], 'product_admission_claim': False}
                try:
                    for name, limits in [('default', Limits()), ('full_limits', Limits(
                            max_nodes=max(1, len(graph['nodes'])), max_events=max(1, graph['quality']['events']),
                            max_edges=max(1, len(graph['edges'])), max_tests=1))]:
                        tensor = encode_test(graph, 'both', limits)
                        meta = tensor.metadata
                        assert tensor.x.shape[1] == 22
                        record[name] = {'shape': list(tensor.x.shape), 'edge_shape': list(tensor.edge_index.shape),
                            **{k: meta[k] for k in ('nodes_before', 'nodes_after_sampling', 'events_before',
                                'events_after', 'edges_before', 'edges_after_sampling', 'edges_after_ablation',
                                'lost_event_fraction', 'lost_edge_fraction')},
                            'state_observation_records': len(meta['state_observations'])}
                    record['full_limits_no_sampling_loss'] = all(record['full_limits'][a] == record['full_limits'][b]
                        for a, b in [('nodes_before', 'nodes_after_sampling'), ('events_before', 'events_after'),
                                     ('edges_before', 'edges_after_sampling')])
                except Exception as exc:
                    record['error'] = type(exc).__name__ + ': ' + str(exc)
                records.append(record)
    write_json(out/'interface_checks.json', {'graphs': records,
        'unsupported_array_type_monitor_opcode_test_counts': [
            {'system': s, 'opcode': op, 'tests': count} for (s, op), count in sorted(unsupported_ops.items())],
        'scope': 'qualified single graphs, not partial-product training; both relation family; temporal deliberately excluded',
        'training_runs': 0, 'labels_read': False})
    print('QUALIFIED SINGLE GRAPH INTERFACES', len(records), 'errors', sum('error' in r for r in records),
          'no sampling loss at full limits', sum(r.get('full_limits_no_sampling_loss', False) for r in records), flush=True)


if __name__ == '__main__':
    main()
