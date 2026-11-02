"""CLAP-6 / PULP-50 source-rule features with strictly TRAIN-only references.

The reference scope is deliberately different from the legacy per-version
protocol. This module does not claim an exact paper feature reproduction.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np

from cwm_fp.data.splits import require_training_keys
from .observations import Observation
from .vendor import spectrum_module

FORMULAS = ("jaccard", "op2", "tarantula", "barinel", "ochiai", "russell_rao", "dstar", "gp02", "gp03", "gp19")
FAMILIES = ("bscp", "bug_involving_statements", "code_coverage", "correctness_reflectability", "incorrectness_verifiability")
PULP50 = tuple(f"{family}_{i}" for family in FAMILIES for i in range(10))
CLAP6 = ("DDU", "code_coverage", "incorrectness_verifiability", "correctness_reflectability", "bscp", "bug_involving_statements")
COVERAGE2 = ("original_test_count", "statement_coverage_ratio")


@lru_cache(maxsize=100000)
def jaccard(a: frozenset, b: frozenset) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def sorted_paths(paths):
    return sorted(paths, key=lambda x: tuple(sorted(x)))


class TrainReferenceFeatures:
    def fit(self, observations: list[Observation], split_path):
        require_training_keys([o.identity for o in observations], split_path, "suspiciousness")
        return self._build(observations)

    def _build(self, observations: list[Observation], *, spectrum_failing_only=False):
        """Internal calculation; public fit enforces its declared evidence scope."""
        self.train_ids = {o.identity for o in observations}
        self.systems = {}
        self.risk_cache, self.query_cache, self.reference_similarity_cache = {}, {}, {}
        self.containment_cache = {}
        self.diagnostics = Counter()
        module = spectrum_module()
        self.formulas = [getattr(module, f"{name}_calculation") for name in FORMULAS]
        for system in sorted({o.identity[0] for o in observations}):
            local = [o for o in observations if o.identity[0] == system]
            spectrum = [o for o in local if o.observed_result == "FAIL"] if spectrum_failing_only else local
            vocabulary = sorted(frozenset().union(*(o.present for o in (spectrum or local))))
            failed, passed = Counter(), Counter()
            for o in spectrum:
                for path in o.failed:
                    failed.update(path)
                for path in o.passed:
                    passed.update(path)
            nf, npass = sum(len(o.failed) for o in spectrum), sum(len(o.passed) for o in spectrum)
            raw = np.array([[fn(failed[s], passed[s], nf, npass) for fn in self.formulas] for s in vocabulary], dtype=float)
            low, high = raw.min(axis=0), raw.max(axis=0)
            normalized = (raw - low) / np.where(high > low, high - low, 1)
            references = {o.identity: o for o in local if o.observed_result == "FAIL"}
            fpaths, ppaths, slices = Counter(), Counter(), defaultdict(Counter)
            for o in references.values():
                fpaths.update(o.failed)
                ppaths.update(o.passed)
                for s in frozenset().union(*o.failed):
                    if s in o.slices:
                        slices[s][o.slices[s]] += 1
            self.systems[system] = {
                "scores": dict(zip(vocabulary, normalized)), "raw": dict(zip(vocabulary, raw)),
                "median": np.median(normalized, axis=0), "lexical_max_scores": normalized[-1],
                "fpaths": fpaths, "ppaths": ppaths, "slices": slices, "references": references,
                "train_products": len(local), "train_tests": nf + npass,
                "fail_tests": nf, "pass_tests": npass, "statement_vocabulary": len(vocabulary),
                "score_min": low.tolist(), "score_max": high.tolist()}
            self.systems[system]["denominator_cache"] = {}
            self.systems[system]["ratio_cache"] = {}
        return self


    def fit_version_context(self, observations: list[Observation]):
        """Explicit target-version context, never a strict-inductive fit.

        Only observed original outcomes/coverage/slices are accepted by the
        Observation schema. Calling this does not allow any FP/TP oracle input.
        A separate instance per version prevents cross-version identity merging.
        """
        if len({o.identity[:2] for o in observations}) != 1:
            raise ValueError("Version context must contain exactly one complete version")
        return self._build(observations, spectrum_failing_only=True)

    def _reflection(self, failed: Counter, passed: Counter, query: list[frozenset]):
        # Equivalent to the source's two successive >threshold Jaccard filters,
        # using unique reference paths, while preserving query-test multiplicity.
        signature = (frozenset(p for p, n in failed.items() if n), frozenset(p for p, n in passed.items() if n))
        if signature not in self.risk_cache:
            fp, pp = signature
            risks = []
            for p in sorted_paths(pp):
                # Cache each complete reference-failure comparison vector. A
                # query F product can remove paths, but cannot change pairwise
                # Jaccard values. This avoids LRU thrashing across exclusions.
                if p not in self.reference_similarity_cache:
                    self.reference_similarity_cache[p] = {}
                memo = self.reference_similarity_cache[p]
                for f in fp:
                    if f not in memo:
                        memo[f] = jaccard(p, f)
                risks.append((p, max((memo[f] for f in fp), default=0.0)))
            self.risk_cache[signature] = risks
        risks = self.risk_cache[signature]
        strengths = []
        for q in query:
            k = (signature, q)
            if k not in self.query_cache:
                if q not in self.reference_similarity_cache:
                    self.reference_similarity_cache[q] = {}
                memo = self.reference_similarity_cache[q]
                for p, _ in risks:
                    if p not in memo:
                        memo[p] = jaccard(q, p)
                self.query_cache[k] = max((min(risk, memo[p]) for p, risk in risks), default=0.0)
            strengths.append(self.query_cache[k])
        return np.array([sum(v > i / 10 for v in strengths) / len(query) if query else 0.0 for i in range(10)])

    def one(self, o: Observation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pulp = np.zeros((5, 10))
        clap = np.array([o.ddu_complement, 0., 0., 0., 0., 0.])
        coverage = np.array([len(o.passed) + len(o.failed), o.coverage_ratio])
        state = self.systems.get(o.identity[0])
        if state is None:
            self.diagnostics["products_without_same_system_training_reference"] += 1
            return clap, pulp.ravel(), coverage
        scores, raw = state["scores"], state["raw"]
        known = sorted(o.present & scores.keys())
        self.diagnostics["query_statements_unseen_in_train"] += len(o.present - scores.keys())
        if known:
            pulp[0] = np.mean([scores[s] for s in known], axis=0)
            clap[4] = float(np.mean([raw[s][4] for s in known]))
        failed, passed = state["fpaths"].copy(), state["ppaths"].copy()
        own = state["references"].get(o.identity)
        own_failed = frozenset()
        if own:
            failed.subtract(own.failed)
            passed.subtract(own.passed)
            own_failed = frozenset().union(*own.failed)
        failed = +failed
        passed = +passed
        all_failed = frozenset().union(*failed) if failed else frozenset()
        suspicious = all_failed & o.present
        covered_suspicious = suspicious & o.covered
        clap[1] = len(suspicious - o.covered) / len(suspicious) if suspicious else 0.
        unweighted_slice = []
        weighted_slice = []
        for s in sorted(suspicious):
            if s not in o.slices:
                continue
            candidates = [value for value, count in state["slices"].get(s, {}).items()
                          if count - int(own is not None and s in own_failed and own.slices.get(s) == value) > 0]
            if candidates:
                best = max(jaccard(o.slices[s], value) for value in candidates)
                unweighted_slice.append(best)
                weighted_slice.append(best * scores.get(s, np.zeros(10)))
        if unweighted_slice:
            clap[5] = float(np.mean(unweighted_slice))
            # Preserve the current source's lexical max-key denominator, not
            # silently substitute the numeric maximum or a paper formula.
            denominator = state["lexical_max_scores"]
            pulp[1] = np.divide(np.mean(weighted_slice, axis=0), denominator,
                                out=np.zeros(10), where=denominator != 0)
        for j in range(10):
            high = {s for s in known if scores[s][j] > state["median"][j]}
            pulp[2, j] = len(high | covered_suspicious) / len(o.present) if o.present else 0.
        pulp[3] = self._reflection(failed, passed, o.passed)
        clap[3] = pulp[3, 8]
        contained, matched, nref = 0, 0, sum(failed.values())
        query_paths = frozenset(o.passed)
        for path in sorted_paths(failed):
            count = failed[path]
            cache_key = (o.present, path)
            if cache_key not in state["ratio_cache"]:
                overlap = o.present & path
                if path not in state["denominator_cache"]:
                    state["denominator_cache"][path] = np.sum([scores.get(s, np.zeros(10)) for s in sorted(path)], axis=0) if path else np.zeros(10)
                denominator = state["denominator_cache"][path]
                numerator = np.sum([scores.get(s, np.zeros(10)) for s in sorted(overlap)], axis=0) if overlap else np.zeros(10)
                ratio = np.divide(numerator, denominator, out=np.zeros(10), where=denominator != 0)
                state["ratio_cache"][cache_key] = (ratio, len(overlap), int(np.count_nonzero(denominator == 0)))
            ratio, overlap_size, zero_count = state["ratio_cache"][cache_key]
            if path and overlap_size / len(path) > .8:
                contained += count
                match_key = (query_paths, path)
                if match_key not in self.containment_cache:
                    self.containment_cache[match_key] = any(len(path & p) / len(path) > .8 for p in o.passed)
                if self.containment_cache[match_key]:
                    matched += count
            self.diagnostics["zero_reference_score_denominators"] += zero_count
            pulp[4] += count * ratio
        if nref:
            pulp[4] /= nref
        clap[2] = 1 - matched / contained if contained else 0.
        if not np.all(np.isfinite(pulp)):
            raise ValueError("Non-finite PULP features")
        return clap, pulp.ravel(), coverage

    def transform(self, observations: list[Observation]):
        rows = []
        for i, o in enumerate(observations):
            rows.append(self.one(o))
            if (i + 1) % 500 == 0:
                print(f"fold-local feature rows: {i+1}/{len(observations)}", flush=True)
        return {"clap6": np.stack([r[0] for r in rows]), "pulp50": np.stack([r[1] for r in rows]),
                "coverage2": np.stack([r[2] for r in rows])}

    def audit(self):
        fields = ("train_products", "train_tests", "fail_tests", "pass_tests", "statement_vocabulary", "score_min", "score_max")
        return {"training_keys": sorted(self.train_ids), "systems": {s: {f: state[f] for f in fields} for s, state in self.systems.items()},
                "diagnostics": dict(self.diagnostics), "reference_policy": "all TRAIN F products, same-system namespace, exclude query product from reference paths/slices",
                "unseen_system_policy": "no same-system statement prior: 50 zeros; CLAP keeps own DDU; coverage2 remains observed",
                "undefined_ratios": "zero with counts in diagnostics, not invented observations",
                "rank_formulas": FORMULAS, "pulp_columns": PULP50, "clap_columns": CLAP6,
                "deviations": ["train-fold reference pool replaces legacy same-version evidence", "TRAIN-only vocabulary and score normalization", "empty/zero denominators explicitly become zero", "branch hits use count OR truecount OR falsecount", "source lexical-max slice denominator and GP19 retained"]}
