"""Local logit sensitivity; neither attention nor a program intervention."""
from __future__ import annotations

import html
from dataclasses import replace
from pathlib import Path

import torch

from cwm_fp.data.common import write_json
from .graph_encoder import collate


def attribution(model, product, out: Path, *, synthetic: bool):
    if model.kind == "coverage": raise ValueError("Coverage model has no edge attribution")
    batch = collate([product]).to(next(model.parameters()).device)
    x = batch.x.clone().detach().requires_grad_(True)
    weights = torch.ones(len(batch.edge_type), device=x.device, requires_grad=True)
    model.eval()
    score = model(replace(batch, x=x), weights)[0]
    gx, ge = torch.autograd.grad(score, (x, weights), allow_unused=True)
    nodes, edges, offset = [], [], 0
    for t in product.tests:
        for n in t.metadata["node_records"]:
            nodes.append({"test_id": t.metadata["identity"][-1], "id": n["id"], "type": n["type"],
                          "location": n["attrs"].get("canonical_statement_id"),
                          "sensitivity": float(gx[len(nodes)].abs().sum()) if gx is not None else 0.})
        for j, e in enumerate(t.metadata["edge_records"]):
            edges.append({"source": e["source"], "target": e["target"], "type": e["type"],
                          "test_id": t.metadata["identity"][-1], "evidence": e["evidence"],
                          "sensitivity": float(ge[offset+j]) if ge is not None else 0.})
        offset += len(t.metadata["edge_records"])
    result = {"synthetic": synthetic, "identity": list(product.identity), "logit": float(score.detach()),
              "method": "absolute input gradient for nodes; signed logit derivative w.r.t. edge message gates (fixed degrees)",
              "interpretation": "model sensitivity only; no physical rerun or causal effect", "nodes": nodes, "edges": edges}
    write_json(out.with_suffix(".json"), result)
    top = sorted((n for n in nodes if n["type"] == "statement"), key=lambda n: -n["sensitivity"])[:10]
    top_edges = sorted(edges, key=lambda e: -abs(e["sensitivity"]))[:10]
    rows = []
    largest = max((n["sensitivity"] for n in top), default=1.) or 1.
    for i, n in enumerate(top):
        y = 85 + i * 29
        label = html.escape(f'{n["test_id"]} / {n["id"]}')
        rows.append(f'<text x="16" y="{y}" font-size="12">{label}</text><rect x="450" y="{y-12}" width="{300*n["sensitivity"]/largest:.2f}" height="14" fill="#397b93"/><text x="760" y="{y}" font-size="12">{n["sensitivity"]:.5f}</text>')
    base = 100 + len(top) * 29
    for i, e in enumerate(top_edges):
        label = html.escape(f'{e["test_id"]}: {e["source"]} → {e["target"]} ({e["type"]})  {e["sensitivity"]:+.5f}')
        rows.append(f'<text x="16" y="{base+30+i*23}" font-size="12">{label}</text>')
    title = "合成图验收：语句与边的模型敏感度" if synthetic else "语句与边的模型敏感度"
    out.with_suffix(".svg").write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{base+70+len(top_edges)*23}" viewBox="0 0 1000 {base+70+len(top_edges)*23}"><rect width="100%" height="100%" fill="white"/><g font-family="Microsoft YaHei, sans-serif" fill="#172f3a"><text x="16" y="30" font-size="20">{title}</text><text x="16" y="53" font-size="13">仅反映模型局部敏感度，不代表真实程序干预或因果效应。</text>{"".join(rows)}</g></svg>', encoding="utf-8")
    return result
