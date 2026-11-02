"""Load unchanged pure source and a minimal, explicit CLAP fit-only adapter."""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

from sklearn import svm

from cwm_fp.data.common import ROOT

VENDOR = ROOT / "third_party"


def pure_functions(path: Path, names: list[str], namespace: dict | None = None) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError(f"Source functions missing: {path}: {names}")
    ns = dict(namespace or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


def spectrum_module():
    path = VENDOR / "PULP/spectrum_manager/Spectrum_Expression.py"
    spec = importlib.util.spec_from_file_location("cwm_vendor_spectrum", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clap_fit(x, y):
    """Execute the original constructor/fit statements, excluding evaluator code."""
    path = VENDOR / "CLAP/PassingVariants_Classification.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "classify_by_svm")
    expected = "clf = svm.SVC(kernel='linear')\nclf.fit(X_train, y_train)"
    if "\n".join(ast.unparse(n) for n in fn.body[:2]) != expected:
        raise ValueError("CLAP source changed; review fit-only adapter")
    ns = {"svm": svm, "X_train": x, "y_train": y}
    exec(compile(ast.Module(body=fn.body[:2], type_ignores=[]), str(path), "exec"), ns)
    return ns["clf"]


def pulp_class():
    # This source imports only BaseClassifier, numpy and sklearn; no outer CLI.
    path = str(VENDOR / "PULP")
    if path not in sys.path:
        sys.path.insert(0, path)
    from fp_detection.PUBaggingLearning import PUBaggingLearning
    return PUBaggingLearning


def load_frozen_artifact(path):
    """Register the unchanged source class in a fresh inference process."""
    import joblib
    pulp_class()
    return joblib.load(path)


class FrozenSourcePU:
    """Original 5%-negative / accuracy-weighted algorithm, fitted on TRAIN only."""
    def __init__(self, seed: int, estimators: int = 30, ratio: float = .05):
        self.seed, self.estimators, self.ratio = seed, estimators, ratio

    def fit(self, failing, passing):
        if not len(failing) or int(len(passing) * self.ratio) < 1:
            raise ValueError("Original PU needs F anchors and >=1 sampled passing negative")
        self.source = pulp_class()(n_estimators=self.estimators, sampling_ratio=self.ratio, random_state=self.seed)
        # Do not call prepare_data: it unnecessarily accepts hidden FP/TP truth.
        self.source.failing_feature = failing
        self.source.passing_feature = passing
        self.source._train_pu_bagging()
        return self

    def score(self, x):
        import numpy as np
        return np.sum([w * model.predict_proba(x)[:, list(model.classes_).index(1)]
                       for model, w in zip(self.source.estimators, self.source.estimator_weights)], axis=0)
