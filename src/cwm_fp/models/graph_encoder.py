"""Explicit numeric graph tensors. Identifiers stay in audit metadata only."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch
from torch import Tensor, nn
from cwm_fp.lddg.extract import identity_errors
from cwm_fp.lddg.static import IDENTITY_SCHEMA, IDENTITY_TYPES
from cwm_fp.lddg.control import control_evidence_errors

NODE_TYPES = ("feature", "statement", "value", "assertion", "output")
RELATIONS = ("feature_controls", "control_dep", "data_dep", "produces", "consumes", "observes", "temporal")
OPCODES = ("constant", "load", "store", "arithmetic", "branch", "call", "return", "field", "other")
FEATURES_V1 = (*("node_" + t for t in NODE_TYPES), *("opcode_" + t for t in OPCODES),
            "executed", "available", "event_instance", "config_enabled", "config_known")
FEATURE_SCHEMA = 'semantic_identity_v2'
FEATURES = (*FEATURES_V1[:-5], *("program_"+t.lower() for t in IDENTITY_TYPES), *FEATURES_V1[-5:])


def feature_names(schema):
    if schema==FEATURE_SCHEMA:return FEATURES
    if schema=='semantic_v1':return FEATURES_V1
    raise ValueError('Unsupported feature schema')
ABLATIONS = {"cdg": {"feature_controls", "control_dep", "observes"},
             "ddg": {"data_dep", "produces", "consumes", "observes"},
             "both": set(RELATIONS) - {"temporal"}, "coverage": set()}
ABLATIONS['full'] = set(RELATIONS)  # Complete data transport; historical ablations are unchanged.
# Temporal edges are deliberately excluded from all four controls: sequence
# adjacency must not reintroduce the removed dependence family.
FORBIDDEN = {"label", "label_provenance", "source_label", "buggy", "fault_location",
             "mutant", "patch", "removed_tests", "obsvalue", "posterior", "y_true"}


def check_metadata(value, *, edge=False, value_attrs=False):
    if isinstance(value, dict):
        if FORBIDDEN.intersection(value):
            raise ValueError("Oracle/label fields in graph input")
        callback_origin = value_attrs and value.get("origin") == "producer_stack_callback"
        if "origin" in value and not edge and not callback_origin:
            raise ValueError("Label-origin metadata in graph input")
        for k, v in value.items():
            if k == "edges":
                for e in v:
                    check_metadata(e, edge=True)
            else:
                check_metadata(v, edge=edge, value_attrs=k == "attrs" and value.get("type") == "value")
    elif isinstance(value, list):
        for v in value:
            check_metadata(v, edge=edge)


def dependency_errors(graph: dict) -> list[str]:
    """Independent structure/semantics dimension; not a standalone admission gate."""
    q, p = graph.get("quality", {}), graph.get("provenance", {})
    errors = []
    if graph.get("tier") != "GRAPH" or p.get("coverage_only") is not False:
        errors.append("not_a_verified_graph_source")
    if q.get("eligible_graph_structure") is not True or q.get("verified_tier") != "T1" or p.get("learnable") is not True:
        errors.append("unsupported_or_incomplete_structure")
    if any(q.get(k) for k in ("bytecode_mismatches", "unsupported_dynamic", "executed_CFG_gaps")):
        errors.append("contradictory_evidence_gaps")
    if p.get('control_schema') or 'control_flow_evidence' in graph:
        errors.extend(control_evidence_errors(graph))
    elif q.get('executed_methods_with_exception_handlers'):
        errors.append('contradictory_evidence_gaps')
    if p.get('identity_schema') and q.get('dependency_semantics_valid') is not True:
        errors.append('dependency_semantics_not_verified')
    return errors


def evidence_errors(graph: dict, *, require_mapping=True) -> list[str]:
    q, p = graph.get('quality', {}), graph.get('provenance', {})
    errors = dependency_errors(graph)
    has_identity = bool(p.get('identity_schema') or graph.get('instruction_catalog') or
                        any('program_identity' in n.get('attrs',{}) for n in graph.get('nodes',[])))
    if has_identity:
        errors.extend(identity_errors(graph))
    elif require_mapping and q.get("canonical_mapping_complete") is not True:
        errors.append("incomplete_canonical_mapping")
    return errors


def opcode_group(value) -> str:
    # JVM opcode numbers, fixed semantic vocabulary; no fitted names/locations.
    if not isinstance(value, int):
        return "other"
    if 1 <= value <= 20: return "constant"
    if 21 <= value <= 53: return "load"
    if 54 <= value <= 86: return "store"
    if 96 <= value <= 152: return "arithmetic"
    if 153 <= value <= 171 or value in (198, 199, 200, 201): return "branch"
    if 182 <= value <= 186: return "call"
    if 172 <= value <= 177: return "return"
    if 178 <= value <= 181: return "field"
    return "other"


@dataclass(frozen=True)
class Limits:
    max_nodes: int = 512
    max_events: int = 128
    max_tests: int = 16
    max_edges: int = 4096

    def __post_init__(self):
        if min(self.max_nodes, self.max_events, self.max_tests, self.max_edges) < 1:
            raise ValueError("Sampling limits must be positive")


@dataclass
class TestTensor:
    x: Tensor
    edge_index: Tensor
    edge_type: Tensor
    coverage: Tensor
    metadata: dict


def encode_test_lossless(graph, *, capacity=Limits(4096,4096,4096,65536)):
    """Refuse oversize input; preserve every node/event and all seven relations."""
    n,e,t=len(graph['nodes']),len(graph['edges']),graph['quality']['events']
    if n>capacity.max_nodes or e>capacity.max_edges or t>capacity.max_events:
        raise ValueError('Single graph exceeds lossless capacity; sampling is forbidden')
    result=encode_test(graph,'full',Limits(max(1,n),max(1,t),1,max(1,e)))
    if (len(result.x),result.edge_index.shape[1],result.metadata['events_after'])!=(n,e,t):
        raise ValueError('Encoder omitted nodes/edges/events; cannot certify lossless input')
    result.metadata['lossless']=True
    return result


@dataclass
class ProductTensor:
    identity: tuple[str, str, str]
    sample_id: str
    partition: str
    observed_result: str
    tests: list[TestTensor]
    audit: dict


def encode_test(graph: dict, ablation: str, limits=Limits(), *, preview=False, feature_schema=FEATURE_SCHEMA) -> TestTensor:
    """Preview relaxes mapping only; unsupported dynamic semantics still fail."""
    check_metadata(graph)
    if ablation not in ABLATIONS:
        raise ValueError("Unknown dependence ablation")
    if errors := evidence_errors(graph, require_mapping=not preview):
        raise ValueError("; ".join(errors))
    nodes = graph["nodes"]
    by_id = {n["id"]: n for n in nodes}
    if not nodes or len(by_id) != len(nodes) or any(n["type"] not in NODE_TYPES for n in nodes):
        raise ValueError("Empty graph, duplicate node or unknown node type")
    events = [n for n in nodes if n["type"] == "statement" and "event_index" in n["attrs"]]
    order = [n["attrs"]["event_index"] for n in events]
    if (not order or any(type(i) is not int or i < 0 for i in order) or sorted(order) != list(range(len(order)))
            or graph["quality"].get("events") != len(order)):
        raise ValueError("Invalid/incomplete/duplicate event order or count")
    selected_events = {n["id"] for n in sorted(events, key=lambda n: n["attrs"]["event_index"])[:limits.max_events]}
    selected_indices = {by_id[i]["attrs"]["event_index"] for i in selected_events}
    event_ids = {n["id"] for n in events}
    # Stable source order; no label-, name-, version-, or attention-based selection.
    candidates = [n for n in nodes if (n["id"] not in event_ids or n["id"] in selected_events)
                  and (n["type"] != "value" or n["attrs"].get("definition_event") in selected_indices)]
    sampled = candidates[:limits.max_nodes]
    # Isolated value nodes still reveal DDG-derived counts under mean pooling.
    # Remove them as well as all value/data edges in the CDG-only control.
    kept = [n for n in sampled if ablation != "cdg" or n["type"] != "value"]
    sampled_ids = {n["id"] for n in sampled}
    positions = {n["id"]: i for i, n in enumerate(kept)}
    names=feature_names(feature_schema)
    x = torch.zeros((len(kept), len(names)))
    for i, n in enumerate(kept):
        a, t = n["attrs"], n["type"]
        x[i, NODE_TYPES.index(t)] = 1
        if t == "statement":
            x[i, len(NODE_TYPES) + OPCODES.index(opcode_group(a.get("opcode")))] = 1
            x[i, -5] = float(a.get("executed") is True)
            x[i, -4] = float(a.get("available") is True)
            x[i, -3] = float("event_index" in a)
            if feature_schema==FEATURE_SCHEMA and graph['provenance'].get('identity_schema')==IDENTITY_SCHEMA:
                x[i,len(NODE_TYPES)+len(OPCODES)+IDENTITY_TYPES.index(a['program_identity']['provenance_type'])]=1
        if t == "feature" and a.get("constraints_verified") is True and graph["quality"].get("constraints_verified") is True:
            if type(a.get("enabled")) is not bool:
                raise ValueError("Verified feature state must be boolean")
            x[i, -2:] = torch.tensor([float(a["enabled"]), 1.])
    edges, encoded_edges, selected_edge_count = [], [], 0
    for e in graph["edges"]:
        if e["type"] not in RELATIONS or e["source"] not in by_id or e["target"] not in by_id:
            raise ValueError("Unknown edge type/endpoint")
        required = "static" if e["type"] in {"feature_controls", "control_dep"} else "dynamic"
        if e.get("evidence") != required:
            raise ValueError("Unsupported edge evidence")
        if e["type"] == "data_dep":
            da, ua = by_id[e["source"]]["attrs"], by_id[e["target"]]["attrs"]
            if not (type(e.get("definition_event")) is int and type(e.get("event_index")) is int
                    and 0 <= e["definition_event"] < e["event_index"]
                    and da.get("event_index") == e["definition_event"] and ua.get("event_index") == e["event_index"]):
                raise ValueError("Invalid reaching definition order/endpoint")
        present = e["source"] in positions and e["target"] in positions
        within_window = e["source"] in sampled_ids and e["target"] in sampled_ids
        sampled_edge = within_window and selected_edge_count < limits.max_edges
        selected_edge_count += int(sampled_edge)
        if sampled_edge and present and e["type"] in ABLATIONS[ablation]:
            edges.append((positions[e["source"]], positions[e["target"]], RELATIONS.index(e["type"])))
            encoded_edges.append(e)
    static = [n for n in nodes if n["type"] == "statement" and "event_index" not in n["attrs"]]
    executed = sum(n["attrs"].get("executed") is True for n in static)
    cov = torch.tensor([len(static), executed, executed / max(1, len(static)), len(events)], dtype=torch.float32)
    cov[[0, 1, 3]] = torch.log1p(cov[[0, 1, 3]])
    kept_events = sum(n["id"] in selected_events for n in kept)
    audit = {"preview_only": preview, 'feature_schema':feature_schema, 'feature_names':list(names),
             'identity_schema':graph['provenance'].get('identity_schema','legacy_mapping_only'),
             'ordered_event_nodes':[n['id'] for n in sorted(kept,key=lambda n:n['attrs'].get('event_index',-1)) if 'event_index' in n['attrs']],
             # Explicit observation side channel for later state work; not classifier features.
             'state_observations':[{'node_id':n['id'],'observed':n['attrs'].get('observed') is True,
                                    'dtype':n['attrs'].get('dtype'),'value':n['attrs'].get('value') if n['attrs'].get('observed') is True else None,
                                    'missing_reason':n['attrs'].get('missing_reason')}
                                   for n in kept if n['type']=='value'],
             'state_observations_used_as_classifier_features':False,
             "nodes_before": len(nodes), "nodes_after_sampling": len(sampled), "nodes_after": len(kept),
             "events_before": len(events), "events_after": kept_events,
             "edges_before": len(graph["edges"]), "edges_after_sampling": selected_edge_count,
             "edges_after_ablation": len(edges), "ablation": ablation,
             "lost_event_fraction": 1 - kept_events / len(events),
             "lost_edge_fraction": 1 - selected_edge_count / len(graph["edges"]) if graph["edges"] else 0.,
             "config_states_enabled": graph["quality"].get("constraints_verified") is True,
             "node_records": kept, "edge_records": encoded_edges,
             "identity": [graph[k] for k in ("system_id", "bug_version_id", "product_id", "test_id")]}
    return TestTensor(x, torch.tensor([e[:2] for e in edges], dtype=torch.long).reshape(-1, 2).T.contiguous(),
                      torch.tensor([e[2] for e in edges], dtype=torch.long), cov, audit)


@dataclass
class Batch:
    x: Tensor
    edge_index: Tensor
    edge_type: Tensor
    node_test: Tensor
    test_product: Tensor
    coverage: Tensor
    n_products: int

    def to(self, device):
        return replace(self, **{k: v.to(device) for k, v in vars(self).items() if isinstance(v, Tensor)})


def collate(products: list[ProductTensor]) -> Batch:
    if not products or any(not p.tests for p in products):
        raise ValueError("Cannot pool an empty product/test collection")
    if len({p.identity for p in products}) != len(products):
        raise ValueError("Duplicate product in batch")
    if len({(g.x.shape[1],g.metadata.get('feature_schema','semantic_v1')) for p in products for g in p.tests})!=1:
        raise ValueError('Cannot mix feature schema versions in a batch')
    xs, es, rs, nt, tp, cov = [], [], [], [], [], []
    offset, test = 0, 0
    for pi, p in enumerate(products):
        for g in p.tests:
            if not len(g.x): raise ValueError("Empty test graph")
            xs.append(g.x); es.append(g.edge_index + offset); rs.append(g.edge_type)
            nt.append(torch.full((len(g.x),), test, dtype=torch.long)); tp.append(pi); cov.append(g.coverage)
            offset += len(g.x); test += 1
    return Batch(torch.cat(xs), torch.cat(es, dim=1), torch.cat(rs), torch.cat(nt),
                 torch.tensor(tp), torch.stack(cov), len(products))


def mean_pool(x: Tensor, index: Tensor, n: int) -> Tensor:
    total = x.new_zeros((n, x.shape[1])).index_add(0, index, x)
    return total / torch.bincount(index, minlength=n).clamp_min(1).to(x.dtype).unsqueeze(1)


class RelationLayer(nn.Module):
    """Incoming per-relation mean, basis-decomposed weights and a self map."""
    def __init__(self, width: int, relations: int, bases=2):
        super().__init__()
        self.self_map = nn.Linear(width, width)
        self.bases = nn.Parameter(torch.empty(bases, width, width))
        self.coeff = nn.Parameter(torch.empty(relations, bases))
        nn.init.xavier_uniform_(self.bases); nn.init.xavier_uniform_(self.coeff)

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        weights = torch.einsum("rb,bio->rio", self.coeff, self.bases)
        out = self.self_map(x)
        for r in range(len(weights)):
            mask = edge_type == r
            src, dst = edge_index[:, mask]
            messages = x[src] @ weights[r]
            if edge_weight is not None: messages = messages * edge_weight[mask, None]
            # Edge weights are attribution gates, not a change to degree normalization.
            out = out + mean_pool(messages, dst, len(x))
        return torch.relu(out)


class SelfLayer(nn.Module):
    """Node-only control: the same self transformation, no message parameters."""
    def __init__(self,width):
        super().__init__();self.self_map=nn.Linear(width,width)
    def forward(self,x,edge_index,edge_type,edge_weight=None):
        return torch.relu(self.self_map(x))


class GraphEncoder(nn.Module):
    def __init__(self, kind: Literal["agnostic", "rgcn", "nodeonly", "untyped"], hidden=16, layers=2, feature_schema=FEATURE_SCHEMA):
        super().__init__()
        if kind not in {"agnostic", "rgcn", "nodeonly", "untyped"}: raise ValueError("Unknown graph encoder")
        self.kind = kind
        self.feature_schema=feature_schema
        self.input = nn.Linear(len(feature_names(feature_schema)), hidden)
        self.layers = nn.ModuleList(SelfLayer(hidden) if kind=='nodeonly' else
            RelationLayer(hidden, 1 if kind in {"agnostic","untyped"} else len(RELATIONS)) for _ in range(layers))

    def forward(self, batch: Batch, edge_weight=None):
        x = batch.x
        if x.shape[1]!=self.input.in_features:raise ValueError('Checkpoint/input feature schema mismatch; explicitly select legacy semantic_v1 when needed')
        if self.kind == "agnostic":
            # Remove node types as well as relation types; retain opcode semantics.
            x = torch.cat((torch.zeros_like(x[:, :len(NODE_TYPES)]), x[:, len(NODE_TYPES):]), dim=1)
        h = torch.relu(self.input(x))
        relation = torch.zeros_like(batch.edge_type) if self.kind in {"agnostic","untyped"} else batch.edge_type
        for layer in self.layers: h = layer(h, batch.edge_index, relation, edge_weight)
        tests = mean_pool(h, batch.node_test, len(batch.test_product))
        return mean_pool(tests, batch.test_product, batch.n_products)
