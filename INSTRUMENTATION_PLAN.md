# Java 8 LDDG Instrumentation Adapter — Engineering Design

This document is a **proposed adapter**. It does NOT imply that these classes or logs exist in the current repository; Codex must inspect user-supplied instrumenter before coding. If the existing SMARTFL/SPL adaptation includes `SplGraphBuilder`, `TraceTransformer`, `FeatureLocationMapper`, `initSplTestsFromRunLogs`, or analogous implementations, inspect and reuse their real interfaces.

## Goals
For each ORIGINAL test of each product, obtain events with consistent source locations and enough runtime information to connect actual dynamic definitions to actual uses. Static CDG must be extracted from a proper CFG/postdominance analysis and joined with the instantiated dynamic statements. Some nodes are present statically but not executed; distinguish `available` vs `executed`.

## Minimum collection levels
A. Existing `opcode/lineinfo` trace + coverage: may map source lines and bytecode instructions but is **not sufficient alone to claim precise interprocedural DDG or actual values**.
B. Dynamic dataflow: each event has test ID, thread, invocation/call frame, bytecode instruction ID, source mapping, variable read/write ID, runtime reaching definition (including local/field/array when supported), event time or order. Explicitly label any omitted alias/heap cases.
C. Value aware: typed safe summaries or actual allowed scalar values, branch predicates/results, test input, output/assertion values. Record privacy-safe encoding and missing-value masks.
D. Interventions: a runner capable of changing legal test inputs or typed runtime values and reproducing the test in the **same unmodified buggy program**, generating before/after trace pairs.

## Proposed producer/adapter pipeline
`FeatureHouse variant + original tests + config` → `Java8 instrumented test runner` → per-test `events.jsonl` → `dynamic_def_use.jsonl` → combined `graphs.jsonl` (static CDG + feature control + actual observed DDG) → quality audit.

- Preserve `<system>/<bug version>/<variant>/<test>` lineage; JUnit names may contain `Class::method` or other separators; normalize in a dedicated tested function, never with ad hoc string splitting in many modules.
- Static control dependence: obtain from precise method CFG and postdominator relationship. Separate exception-control and normal flow as far as tool permits.
- Dynamic local def-use: track definition events and references through actual executed bytecode, including stack/local-frame context. For fields and arrays, trace object/field/array identities or declare coverage gap; `s1 read x`, `s2 writes x` in source text does not alone prove runtime dependence.
- Edges into output/exception/assertion are crucial for the falsification mechanism. Add distinguishable observation/propagation-sink node for each assertion or externally visible return.
- Feature-level edges require actual feature-to-composed-statement map, including `roles.meta` or a tested source-location mapper. Store mapping confidence and collisions.
- Loops: same source statement may execute many times; keep event-instance keys; graph pooling may later compress them, but raw logs must preserve order.
- Missing source lines: compiler may optimize e.g. constant branches (`if (true)`) out of bytecode. Do not insert a fictional line event to match a known bug location. Flag and quantify such exclusions. Any separate recompilation experiment must validate semantic equivalence and be clearly reported.
- Runtime instrumentation can change timing, especially for concurrency. For initial work select deterministic single-thread test subsets and report unsupported cases.

## Acceptance tests to ask Codex to implement
1. Local writes and reads: `a = input; b = a + 1; c = b * 2`, both expected dynamic edges present.
2. Dead branch: one branch not executed, no dynamic values/edges from that branch, static CDG retained.
3. Loop: 3 iterations yield distinct value versions and correct iteration-specific uses.
4. Method call and return: callee return dependency and call frame context tracked.
5. Heap alias: object field write then read through alias (supported or explicitly marked unsupported; never silently wrong).
6. SPL feature mapping: two products with different features produce consistent original feature statement identity, but different available/active node sets.
7. Oracle test parity: uninstrumented and instrumented versions produce same original test results; compare source mapping and bounded coverage changes.
8. Performance ceiling: configurable max events/test, duration/test, memory and file-size limits; truncation marked and affected traces masked from supervised transition loss.

## Ambiguities to resolve by inspection
- How are source line and `opcode` records encoded? Are dynamic values and `def-use` already emitted?
- How does the current variant compiler preserve feature source location? Is `roles.meta` sufficient for all statements?
- Does the trace collector distinguish per-test runs, invocation IDs, and reentrant methods?
- Can instrumented code rerun arbitrary tests without rebuilding all variants?
- Are compiled `.class` files built for Java 8? Which compiler/ASM/agent version is installed?

If any answer is unknown, report it as unknown; do not fabricate a parser.
