"""Bounded supervised classifier, train-only coverage scaling, two-level pooling."""
from __future__ import annotations

import torch
from torch import nn

from .graph_encoder import Batch, GraphEncoder, ProductTensor, collate, mean_pool, FEATURE_SCHEMA


class GraphClassifier(nn.Module):
    def __init__(self, kind="rgcn", hidden=16, layers=2, feature_schema=FEATURE_SCHEMA):
        super().__init__()
        if kind not in {"coverage", "agnostic", "rgcn", "nodeonly", "untyped"}: raise ValueError("Unknown classifier")
        if not 1 <= hidden <= 128 or not 1 <= layers <= 4: raise ValueError("Unbounded architecture")
        self.kind = kind
        self.register_buffer("coverage_mean", torch.zeros(4))
        self.register_buffer("coverage_scale", torch.ones(4))
        self.encoder = (nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
                        if kind == "coverage" else GraphEncoder(kind, hidden, layers,feature_schema))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.normalization_audit = {"fit": False, "graph_features": "fixed semantic/boolean vocabulary; no fitted statistics"}

    def fit_normalization(self, products: list[ProductTensor], assignments: dict):
        if self.normalization_audit["fit"]: raise ValueError("Normalization already frozen")
        if not products or any(p.partition != "train" or assignments.get(p.identity) != "train"
                               or p.observed_result != "PASS" or p.audit.get("preview_only")
                               or any(g.metadata.get("preview_only") for g in p.tests) for p in products):
            raise ValueError("Normalization accepts only eligible TRAIN PASS products")
        if self.kind == "coverage":
            values = collate(products).coverage
            self.coverage_mean.copy_(values.mean(0))
            self.coverage_scale.copy_(values.std(0, unbiased=False).clamp_min(1e-6))
        self.normalization_audit = {"fit": True, "contributors": [list(p.identity) for p in products],
                                    "mean": self.coverage_mean.tolist(), "scale": self.coverage_scale.tolist(),
                                    "graph_features": "fixed semantic/boolean vocabulary; no fitted statistics"}

    def encode(self, batch: Batch, edge_weight=None):
        if self.kind == "coverage":
            h = self.encoder((batch.coverage - self.coverage_mean) / self.coverage_scale)
            h = mean_pool(h, batch.test_product, batch.n_products)
        else:
            h = self.encoder(batch, edge_weight)
        return h

    def forward(self, batch: Batch, edge_weight=None):
        return self.head(self.encode(batch,edge_weight)).squeeze(-1)
