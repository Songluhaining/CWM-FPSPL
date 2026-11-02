"""Render measured synthetic-model sensitivities using the base plotting env."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / "configs/graph_classifier.json").read_text(encoding="utf-8"))
path = ROOT / config["work_dir"] / "synthetic/attribution.json"
record = json.loads(path.read_text(encoding="utf-8"))
assert record["synthetic"] is True
nodes = sorted((n for n in record["nodes"] if n["type"] == "statement"), key=lambda n: -n["sensitivity"])[:10]
edges = sorted(record["edges"], key=lambda e: -abs(e["sensitivity"]))[:10]
plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False, "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), layout="constrained")
labels = [f'{n["test_id"]} / {n["id"]}' for n in nodes]
axes[0].barh(labels[::-1], [n["sensitivity"] for n in nodes][::-1], color="#397b93")
axes[0].set_title("语句输入的梯度绝对值之和")
names = {"feature_controls": "特征控制", "control_dep": "控制依赖", "data_dep": "数据依赖", "produces": "定义值", "consumes": "读取值", "observes": "观测"}
labels = [f'{e["test_id"]} / {e["source"]}→{e["target"]}（{names[e["type"]]}）' for e in edges]
axes[1].barh(labels[::-1], [e["sensitivity"] for e in edges][::-1], color="#b67835")
axes[1].set_title("边消息权重的有符号梯度")
for ax in axes:
    ax.grid(axis="x", alpha=.2); ax.set_axisbelow(True); ax.set_xlabel("对模型分类打分的局部敏感度")
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("合成图验收：固定首个种子的关系图卷积模型\n仅为模型敏感度，不代表真实程序干预或因果效应", fontsize=14)
out = ROOT / "outputs/phase04_attribution.png"
fig.savefig(out, dpi=150); plt.close(fig)
print(out)
