# AdaptiveRAG — Development Plan

> **Status:** Roadmap v2.0 (revised for minimum research-valid scope)
> **Derived from:** [`prod-spec.md`](prod-spec.md) · [`architecture.md`](architecture.md)
> **Project:** AdaptiveRAG — Adaptive Retrieval-Augmented Generation using Query Complexity and Context Optimization
> **Purpose:** Convert the architecture into an executable, dependency-ordered implementation roadmap

---

## Document Role

```text
prod-spec.md        →  WHAT and WHY
architecture.md     →  HOW the system is structured
development-plan.md →  WHEN / IN WHAT ORDER / HOW TO VERIFY
src/                →  actual code
```

This plan does not restate the architecture or product specification. It tells the team what to do next, in what order, and how to verify it.

---

## Current Repository State

As of commit `674dab7` (M6.1 acceptance), the repository contains:

```text
adaptive-rag-query-aware-retrieval/
├── configs/config.yaml              ← Working configuration
├── datasets/                        ← Dataset loading, splitting, corpus, index builds
│   ├── load_dataset.py              ← PopQA + HotpotQA download/normalize
│   ├── prepare_data.py              ← Train/val/test splits (70/15/15)
│   ├── build_corpus.py              ← Fixed chunking (512 tokens, 50 overlap)
│   ├── build_index.py               ← ChromaDB index construction
│   ├── fixtures_loader.py           ← Local dev fixture loader
│   └── [raw/, corpus/, splits/]     ← Data artifacts (git-ignored large files)
├── src/
│   ├── utils/                       ← Config, logger (complete)
│   ├── retriever/                   ← Embeddings, vector store, retriever (complete)
│   ├── generator/                   ← ResponseGenerator with SmolLM (complete)
│   ├── optimizer/                   ← ContextOptimizer with LLMLingua-2 (complete)
│   ├── pipeline/adaptive_rag.py     ← Pipeline: supports llm_only + fixed_rag modes
│   ├── router/                      ← __init__.py only — NOT YET IMPLEMENTED
│   └── evaluation/                  ← __init__.py only — NOT YET IMPLEMENTED
├── tests/                           ← 73 tests passing (M6.1 baseline)
├── scripts/run_pipeline.py          ← CLI entry point for baselines
└── docs/
```

**What works:** Baselines A (LLM Only) and B (Fixed RAG) are functional end-to-end.
**What does not exist yet:** Evaluation metrics, benchmark runner, complexity router, adaptive controller, Baseline C (Always-Compress), Baseline D (AdaptiveRAG).

---

## Execution Principles

1. **Vertical slices** — prefer a thin end-to-end slice over many isolated subsystems.
2. **Verify before proceeding** — do not advance until the current milestone's acceptance gate is met.
3. **Configuration-driven** — all experiment-sensitive parameters live in `configs/config.yaml`.
4. **Fairness first** — all four baselines must run on the same index, corpus, embedding model, generator, and test split.
5. **Research first** — every milestone directly supports the scientific question; engineering ceremony is minimized.

---

## Completed Milestone History (Do Not Modify)

| Milestone | Description | Status |
|---|---|---|
| M0.1 | Repository structure | ✅ Complete |
| M0.2 | Configuration system | ✅ Complete |
| M0.3 | Logging + pytest infrastructure | ✅ Complete |
| M0.4 | Dev fixtures | ✅ Complete |
| M1.1 | Dataset loading (PopQA + HotpotQA) | ✅ Complete |
| M1.2 | Train/val/test splits | ✅ Complete |
| M1.3 | Corpus preparation and chunking | ✅ Complete |
| M1.4 | Embedding model (BAAI/bge-small-en-v1.5) | ✅ Complete |
| M1.5 | Vector index (ChromaDB) | ✅ Complete |
| M1.6 | Top-K retrieval | ✅ Complete |
| M2.1 | Generator model (SmolLM-135M-Instruct) | ✅ Complete |
| M2.2 | Prompt construction | ✅ Complete |
| M3.1 | Query preprocessor | ✅ Complete |
| M3.2 | ⭐ First vertical slice | ✅ Complete |
| M4.1 | Baseline A — LLM Only | ✅ Complete |
| M5.1 | Baseline B — Fixed RAG | ✅ Complete |
| M6.1 | LLMLingua-2 context compression | ✅ Complete |

**ADR resolutions recorded by implementation:**
- **ADR-001** Generator: `HuggingFaceTB/SmolLM-135M-Instruct` (selected for CPU/low-VRAM compatibility; replaces Phi-3-mini)
- **ADR-003** Embeddings: `BAAI/bge-small-en-v1.5` ✓
- **ADR-004** Vector store: ChromaDB ✓
- **ADR-006** Compression budget: 0.5 (initial default; validate on val set)
- **ADR-002, ADR-005, ADR-007** remain open (router model, K values, confidence threshold)

---

## Remaining Roadmap Overview

| Milestone | Name | What It Produces |
|---|---|---|
| **M7** | Evaluation Infrastructure | All metrics + benchmark runner validated |
| **M8** | Compression Fallback + Baseline C | Always-Compress pipeline complete + evaluated |
| **M9** | Complexity Router | Trained router with SIMPLE/COMPLEX classification, independently evaluated |
| **M10** | Adaptive Controller + Baseline D + CLI Demo | Full AdaptiveRAG pipeline complete + evaluated + mandatory demo trace |
| **M11** | Primary Benchmark | All 4 baselines compared on identical test conditions |
| **M12** | Ablation + Error Analysis | Compression ablation + representative failure inspection |
| **MVP STOP ✋** | Research package | Analysis, reproducibility metadata, report packaging |

**Target: 6 meaningful remaining milestones + stop condition.**

All remaining milestones flow sequentially:

```text
M7 (Evaluation Infrastructure)
  ↓
M8 (Compression Fallback + Baseline C)
  ↓
M9 (Complexity Router)
  ↓
M10 (Adaptive Controller + Baseline D)
  ↓
M11 (Primary Benchmark — all 4 baselines)
  ↓
M12 (Ablation + Error Analysis)
  ↓
MVP STOP ✋
```

---

---

# M7 — Evaluation Infrastructure

> **Objective:** Implement and validate all evaluation metrics and the benchmark runner before any large-scale experiment is executed. Evaluation must be correct before results are meaningful.

**Prerequisites:** M6.1 complete.

**Implementation scope:**

### M7.1 — Answer Quality + Efficiency Metrics

Implement `src/evaluation/metrics.py`:

- `exact_match(prediction, reference) → float` — binary; case-normalized.
- `token_f1(prediction, reference) → float` — token-level overlap.
- `compression_ratio(original_tokens, compressed_tokens) → float`.
- `count_tokens(text) → int` — using a shared tokenizer or word-split approximation (must be consistent across all baselines).

Implement `src/evaluation/latency.py`:

- `measure_latency(fn, *args) → (result, latency_ms)` — wraps any callable.

**Tests:** Unit tests for known input/output pairs (all four functions). Edge cases: empty string, identical strings, zero tokens.

### M7.2 — Router Quality Metrics

Implement `router_metrics(predictions, labels) → RouterMetrics` in `src/evaluation/metrics.py`:

- Accuracy, precision (per-class), recall (per-class), F1 (per-class + macro), confusion matrix.

**Tests:** Known prediction/label pairs producing expected per-class values.

### M7.3 — Benchmark Runner

Implement `src/evaluation/benchmark.py`:

- `run_benchmark(pipeline_fn, examples, config) → BenchmarkResult`
  - Accepts a list of examples from `datasets/splits/test.jsonl`.
  - Calls `pipeline_fn(query)` for each example.
  - Computes per-example and aggregate EM, F1, latency stats, token stats.
  - Writes results to `results/<experiment_name>/`:
    - `config.yaml` — copy of run configuration
    - `metrics.json` — aggregate metrics
    - `predictions.jsonl` — per-query: query, reference, prediction, all metadata fields
    - `latency.csv` — per-query latency breakdown

Implement `scripts/evaluate.py` — CLI entry point to run any baseline through the benchmark.

**Tests:** Run benchmark on 10 fixture examples through the Fixed RAG baseline. Verify output files are created and metrics are non-null.

**Out of scope for M7:**
- Optional metrics (semantic similarity, RAGAS, LLM-as-judge) — remain optional post-MVP.
- MRR, nDCG — deferred; only Recall@K is required where ground-truth document IDs are available.
- Real-data benchmark run — reserved for M11.

**Acceptance gate:**
- [ ] EM and F1 produce correct values on known examples.
- [ ] Router metrics produce correct confusion matrix on known examples.
- [ ] Benchmark runner executes Fixed RAG baseline on fixture data and saves machine-readable output.
- [ ] Latency measurement wraps all pipeline stages.
- [ ] All unit and integration tests pass.

**Deliverable:** Validated evaluation infrastructure. Any subsequent benchmark is now interpretable.

---

---

# M8 — Compression Fallback + Baseline C (Always-Compress RAG)

> **Objective:** Add the compression failure fallback path to the existing `ContextOptimizer`, wire the compressor into the pipeline, and complete Baseline C (Always-Compress RAG).

**Prerequisites:** M7 complete.

**Implementation scope:**

### M8.1 — Compression Failure Fallback

Extend `src/optimizer/context_optimizer.py`:

- If `compress_prompt` raises an exception: return `OptimizedContext { text=original_context, status=FAILED }`, log the failure, and do NOT re-raise.
- Pipeline must continue with uncompressed context on compression failure.

**Tests:**
- Unit test: simulate compressor exception via mock → original context returned, status=FAILED.
- Verify pipeline does not crash on compression failure.

### M8.2 — Baseline C Pipeline Mode

Extend `src/pipeline/adaptive_rag.py` to support `baseline="always_compress"` mode:

```text
Query → Preprocessor → Retriever(K=K_complex) → ContextOptimizer → Generator → Answer
```

- K = `config.retrieval.k_complex` (default: 10).
- Compression always invoked.
- `ExecutionMetadata` must include: `compression_applied`, `original_context_tokens`, `compressed_context_tokens`, `compression_ratio`, `compression_latency_ms`.

Extend `scripts/run_pipeline.py` with `--baseline=always_compress`.

**Tests:**
- Integration test (5 fixture queries): verify `compression_applied=true` for all.
- Verify `compressed_context_tokens < original_context_tokens`.
- Verify all latency fields (including `compression_latency_ms`) are populated.
- Verify results save to `results/baseline_c/`.

**Out of scope for M8:**
- Adaptive routing (M10).
- Benchmark at full scale (M11).

**Acceptance gate:**
- [ ] Compression failure does not crash the pipeline; original context is preserved and failure is logged.
- [ ] Always-Compress pipeline runs end-to-end on fixture data.
- [ ] All metadata fields populated (including compression fields).
- [ ] Results saved in machine-readable format.
- [ ] All tests pass (73 existing + new M8 tests).

**Deliverable:** Baselines A, B, C all functional. Compression failure fallback verified. Ready for router implementation.

---

---

# M9 — Complexity Router

> **Objective:** Implement, train, and independently evaluate a lightweight binary classifier that produces SIMPLE/COMPLEX labels from query text. Router must be validated before being wired into the adaptive pipeline.

**Prerequisites:** M8 complete (data splits already exist from M1.2).

**Implementation scope:**

### M9.1 — Router Training Data

Create labeled (query, complexity_label) pairs from existing data splits.

- Source: `datasets/splits/train.jsonl`, `datasets/splits/val.jsonl`.
- Labels: PopQA examples → `SIMPLE` (proxy); HotpotQA examples → `COMPLEX` (proxy).
- Store as `datasets/splits/router_train.jsonl`, `datasets/splits/router_val.jsonl`.
- **Critical:** Labels must be explicit fields; dataset source must NOT be an input feature to the classifier.
- **Critical:** No test queries in router training data (leakage check must pass).

**ADR-002 resolution required before training:** Confirm router model. Default direction: `microsoft/deberta-v3-small`. The model must be trainable on available hardware. If not feasible, select another lightweight encoder classifier and update the ADR.

**Tests:** Label distribution logged. Leakage check: confirm no test query appears in `router_train.jsonl`.

### M9.2 — Router Interface + Training

Implement `src/router/query_classifier.py`:

- `classify(query: str) → ComplexityResult { label: SIMPLE|COMPLEX, confidence: float, latency_ms: float }`
- Model name loaded from config (`config.models.router.name`).
- Checkpoint path loaded from config (`config.models.router.checkpoint`).
- Latency measured per inference call.

Implement `scripts/train_router.py`:

- Fine-tunes the encoder classifier on `router_train.jsonl`.
- Logs validation accuracy per epoch.
- Saves best checkpoint to `models/router/`.
- Records training time and hardware.

**Tests (structural, before training):** Verify interface with a randomly initialized model — label is always SIMPLE or COMPLEX, confidence in [0, 1], latency measured.

### M9.3 — Router Evaluation

Implement `scripts/evaluate_router.py` (or extend `scripts/evaluate.py`):

- Evaluate trained router on `router_val.jsonl` using `router_metrics()` from M7.
- Output: accuracy, precision, recall, F1, confusion matrix saved to `results/router_eval/metrics.json`.
- Log 5 false-positive and 5 false-negative examples for manual inspection.
- Verify router is not trivially learning dataset membership by inspecting error distribution.

Measure and record router inference latency (per-query).

**ADR-005 decision point:** Review K values (K_simple=2, K_complex=10) against validation data. Adjust if clearly suboptimal. Record decision.

**Out of scope for M9:**
- Optional heuristic router comparison (M13.3 — remains deferred).
- Router threshold optimization if argmax classification is sufficient.

**Acceptance gate:**
- [ ] Router training completes without OOM.
- [ ] Best checkpoint saved to `models/router/`.
- [ ] Validation accuracy > 50% (at minimum better than random; document actual value).
- [ ] Router metrics computed and saved (accuracy, precision, recall, F1, confusion matrix).
- [ ] False-classification examples accessible for inspection.
- [ ] Router latency measured per inference.
- [ ] Leakage check passes (no test data in training).
- [ ] All tests pass.

**Deliverable:** Trained and independently evaluated complexity router. Ready for integration into the adaptive pipeline.

---

---

# M10 — Adaptive Controller + Baseline D (AdaptiveRAG)

> **Objective:** Implement the adaptive routing controller and wire all components into the full AdaptiveRAG pipeline. Both SIMPLE and COMPLEX routes must be explicitly verified.

**Prerequisites:** M9 complete (trained router checkpoint available).

**Implementation scope:**

### M10.1 — Adaptive Routing Controller

Implement `src/router/routing_logic.py`:

- Input: `ComplexityResult`
- Output: `RoutingDecision { complexity: SIMPLE|COMPLEX, retrieval_k: int, compression_enabled: bool }`
- Routing policy (from architecture):
  - SIMPLE → `retrieval_k = K_simple`, `compression_enabled = false`
  - COMPLEX → `retrieval_k = K_complex`, `compression_enabled = true`
- All K values read from config. Controller is deterministic. Not an LLM agent.

**Tests:** Unit tests for both SIMPLE and COMPLEX inputs. K values change when config changes.

### M10.2 — AdaptiveRAG End-to-End Pipeline

Extend `src/pipeline/adaptive_rag.py` with `baseline="adaptive"` mode:

```text
SIMPLE path:  Query → Preprocessor → Router → Controller(K=2, compress=false)
              → Retriever(K=2) → Generator → Answer

COMPLEX path: Query → Preprocessor → Router → Controller(K=10, compress=true)
              → Retriever(K=10) → ContextOptimizer → Generator → Answer
```

Extend `scripts/run_pipeline.py` with `--baseline=adaptive`.

`ExecutionMetadata` must include all fields from the architecture spec:
`router_latency_ms`, `complexity_label`, `router_confidence`, `retrieval_k`, `compression_applied`, `original_context_tokens`, `compressed_context_tokens`, `compression_ratio`, `compression_latency_ms`, `generation_latency_ms`, `total_latency_ms`.

**Explicit route verification tests (using dev fixtures):**
- Known simple-proxy query: verify router=SIMPLE, K=2, compression_applied=false, answer non-empty.
- Known complex-proxy query: verify router=COMPLEX, K=10, compression_applied=true, compressed_tokens < original_tokens, answer non-empty.
- Router failure simulation: verify fallback route applied, logged, pipeline continues.
- Compression failure simulation: verify original context preserved, pipeline continues.
- Empty retrieval result: verify structured failure logged, pipeline does not fabricate evidence.

### M10.3 — CLI Demonstration Trace (Mandatory)

Extend `scripts/run_pipeline.py` to produce a formatted per-query decision trace when `--baseline=adaptive` is invoked. This is a thin wrapper over the existing pipeline — no new pipeline logic is introduced.

Required output format:

```text
Query:                <user query>
Predicted Complexity: SIMPLE / COMPLEX
Retrieval K:          2 / 10
Compression Applied:  Yes / No
Original Tokens:      <N>
Compressed Tokens:    <N>  (if applicable)
Answer:               <generated answer>
─────────────────────────────────────
Router Latency:       <N> ms
Retrieval Latency:    <N> ms
Compression Latency:  <N> ms  (if applicable)
Generation Latency:   <N> ms
Total Latency:        <N> ms
```

**Constraint:** Demo output must be populated from `ExecutionMetadata` — it must not introduce separate logic or shadow variables.

**Tests:** Run one simple-proxy and one complex-proxy fixture query through `--baseline=adaptive`; verify all fields are printed and non-null.

**Out of scope for M10:**
- Full benchmark run (M11).
- Ablation experiments (M12).

**Acceptance gate:**
- [ ] Both SIMPLE and COMPLEX routes execute correctly end-to-end.
- [ ] K selection matches routing decision.
- [ ] Compression gate matches routing decision.
- [ ] All metadata fields populated for both paths.
- [ ] Router failure and compression failure paths tested.
- [ ] Integration tests for both routes pass.
- [ ] CLI demo trace prints all required fields for both SIMPLE and COMPLEX queries.
- [ ] All tests pass.

**Deliverable:** Baseline D (AdaptiveRAG) functional end-to-end, with CLI demo trace. All four baselines (A, B, C, D) are now complete. The mandatory functional demonstration of the complete system is done.

---

---

# M11 — Primary Benchmark

> **Objective:** Run all four baselines on the full held-out test set under identical conditions. Produce the primary comparative results table. This is the central empirical output of the project.

**Prerequisites:** M10 complete. All four baselines functional.

**Fairness pre-check (must be verified before any results are recorded):**

The following must be identical across all baselines:
- [ ] Same query set (identical `test.jsonl` examples for all four baselines).
- [ ] Same corpus, chunking parameters, embedding model, vector index.
- [ ] Same generator model (`HuggingFaceTB/SmolLM-135M-Instruct`).
- [ ] Same prompt template.
- [ ] Same generation parameters (temperature, max_new_tokens, seed).
- [ ] Same hardware and environment.
- [ ] Same evaluation scripts.
- [ ] Validation data was used for any threshold/K tuning; test data is fresh.
- [ ] Latency measurement methodology consistent (document warm vs. cold cache state).
- [ ] Failures handled consistently (failed examples marked as failed, not skipped).

**Implementation scope:**

Run `scripts/evaluate.py` for each baseline on the full test split (sample limit from config, default: ~500–1,000 per dataset source). Produce:

```text
results/
├── baseline_a/    (config.yaml, metrics.json, predictions.jsonl, latency.csv)
├── baseline_b/    (same structure)
├── baseline_c/    (same + compression fields)
├── baseline_d/    (same + routing.csv with per-query: complexity_label, retrieval_k, compression_applied)
└── summary.csv    (comparative table across all 4 baselines)
```

**Per-baseline metrics collected:**
- Answer quality: EM, Token-level F1
- Efficiency: mean and p95 router latency (Baseline D only), retrieval latency, compression latency (Baselines C, D), generation latency, total latency
- Token statistics: retrieved context tokens, compressed context tokens (Baselines C, D), final prompt tokens, compression ratio (Baselines C, D)
- Router quality (Baseline D only): accuracy, precision, recall, F1, confusion matrix (from M9.3, applied to test routing decisions)

**Acceptance gate:**
- [ ] All four baselines produce result artifacts without error.
- [ ] No baseline uses validation or training data.
- [ ] All result files are machine-readable (JSON/JSONL/CSV).
- [ ] Experiment configuration recorded alongside results (git commit hash, model identifiers, dataset split, sample count, hardware, random seed).
- [ ] Summary comparison table generated.
- [ ] Fairness pre-check passed.

**Deliverable:** Primary benchmark results. First complete comparative numbers for all research questions.

> **Note on sample size:** The default `sample_limit: 1000` in config covers the evaluation set. Final size should be whatever fits compute constraints while producing statistically distinguishable results — document the actual count used.

---

---

# M12 — Ablation + Error Analysis

> **Objective:** Run the primary compression ablation to isolate the value of compression, and inspect a representative set of failure cases to support honest research interpretation.

**Prerequisites:** M11 complete (primary benchmark results available).

**Implementation scope:**

### M12.1 — Ablation A: Compression Value

**This ablation is mandatory.** It isolates the contribution of compression from retrieval depth.

| Condition | Retrieval K | Compression | Purpose |
|---|---|---|---|
| A1 | 10 | No compression | K=10 uncompressed |
| A2 | 10 | Compression | K=10 compressed (= Baseline C) |

A2 is already Baseline C — no additional pipeline work needed, just correct configuration. A1 requires running the pipeline with K=10 and `compression_enabled=false`.

Results to `results/ablation_compression/`. Compute difference in EM, F1, token usage, and latency between A1 and A2.

This directly answers RQ3 (selective compression) and RQ4 (compression economics).

### M12.2 — Error Analysis (Representative)

Inspect representative failure examples from the M11 benchmark results. This is a lightweight analysis workflow — no dedicated tooling is required beyond the existing `predictions.jsonl` output.

For each of the following categories, identify and log at least 3–5 representative examples from the test results:

| Category | What to Inspect |
|---|---|
| Router mistakes (false SIMPLE) | Complex query routed SIMPLE → was K=2 insufficient? |
| Router mistakes (false COMPLEX) | Simple query routed COMPLEX → unnecessary overhead |
| Retrieval failures | Answer wrong despite routing being correct — evidence not retrieved |
| Compression information loss | Answer correct without compression, wrong after compression |
| Generator failures | Evidence present, answer still wrong |
| Cases where AdaptiveRAG helps | Examples where Baseline D clearly outperforms Baseline B |
| Cases where AdaptiveRAG offers no advantage | Examples where Baseline B matches or beats Baseline D |

Document findings in `results/error_analysis/findings.md`. Patterns should be referenced in the research write-up.

### M12.3 — Reproducibility Metadata

Record and verify reproducibility metadata for at least one baseline (Fixed RAG recommended):

1. Confirm `requirements.txt` has pinned versions.
2. Confirm model identifiers are stored in every `results/<baseline>/config.yaml`.
3. Confirm dataset splits are version-controlled (or their construction is deterministic and seeded).
4. Confirm random seed is recorded (`seed: 42` in config).
5. Confirm hardware information is logged in experiment output.
6. Confirm git commit hash is captured in result metadata.

A full clean-environment rerun is **not required** if all of the above are verifiable. The goal is an auditable record, not a redundant execution.

**Ablation B (optional):** Adaptive K vs Fixed K (K=2/10 routing vs fixed K=5, no compression). Implement only if time remains after M12.1 and M12.2 are complete.

**Out of scope for M12:**
- Optional heuristic router comparison.
- Additional parameter sweeps.
- Semantic similarity or RAGAS-style evaluation (remain optional post-MVP).
- MRR, nDCG (remain deferred).

**Acceptance gate:**
- [ ] Ablation A results saved with correct configuration metadata.
- [ ] Error analysis findings documented for all 7 categories.
- [ ] Reproducibility metadata verified for Fixed RAG baseline.
- [ ] Results are machine-readable.

**Deliverable:** Ablation results + representative error analysis. The project now has all empirical content needed for honest research interpretation.

---

---

# MVP Hard Stop ✋

**The MVP is complete when ALL of the following are true.**

After this point: stop adding features. The next stage is analysis, writing, tables, plots, and presentation — not new engineering.

### Core Pipeline
- [ ] Query preprocessing works.
- [ ] Complexity router works (trained and evaluated independently).
- [ ] SIMPLE and COMPLEX routes execute correctly.
- [ ] Retrieval K changes according to route.
- [ ] SIMPLE route bypasses compression.
- [ ] COMPLEX route invokes compression.
- [ ] Compression failure fallback works.
- [ ] Fixed generator produces answers on all routes.

### Baselines
- [ ] Baseline A (LLM Only) produces results.
- [ ] Baseline B (Fixed RAG) produces results.
- [ ] Baseline C (Always-Compress RAG) produces results.
- [ ] Baseline D (AdaptiveRAG) produces results.

### Evaluation
- [ ] Same test set used across all four baselines.
- [ ] EM and token-level F1 measured.
- [ ] Router quality measured (accuracy, precision, recall, F1, confusion matrix).
- [ ] Stage-level latency measured for all stages.
- [ ] Token statistics measured (retrieved, compressed, final prompt).
- [ ] Compression ratio measured.
- [ ] Results are machine-readable with full experiment metadata.

### Research
- [ ] Ablation A (compression value) complete.
- [ ] Representative failure cases analyzed (7 categories, ≥3–5 examples each).
- [ ] Results support honest interpretation of whether AdaptiveRAG helps or hurts.

### Engineering
- [ ] Core logic has tests.
- [ ] Configuration is externalized.
- [ ] Reproducibility metadata recorded.
- [ ] Git commit hash captured in at least one result artifact.
- [ ] CLI demo trace functional (implemented in M10.3; verified for SIMPLE and COMPLEX paths).

> **Once this checklist is complete: STOP adding features.**
> Do not add GraphRAG, agents, multimodal RAG, web search, dynamic chunking, generator fine-tuning, complex UI, distributed infrastructure, or any capability not directly required by the research question.
>
> The remaining work after this point is: analysis, plots/tables, report sections, writing.

---

---

# Post-MVP: Research Packaging (After MVP Stop)

> Only after the MVP checklist is complete. These are not implementation milestones — they are documentation, presentation, and communication tasks.

**CLI Demo:** Completed in M10.3 as part of the mandatory AdaptiveRAG milestone. No additional demo work is required post-MVP.

**Result tables and plots:** Generate comparison tables (summary.csv already produced in M11) and any visualizations from `notebooks/` using existing result artifacts.

**Report support:** The deliverables from each milestone map directly to report sections:

| Report Section | Source |
|---|---|
| Proposed Methodology | Architecture + this plan |
| Dataset and Preprocessing | M1.x implementation + data documentation |
| Implementation | src/ code + pipeline description |
| Experimentation and Results | M11 benchmark results + M12 ablation + error analysis |
| Functional Demonstration | CLI demo trace from M10.3 |
| GitHub Repository | Complete repository with pinned dependencies |

---

---

## Summary: Remaining Phase Sequence

```text
M7:  Evaluation Infrastructure (metrics + benchmark runner)
     ↓
M8:  Compression Fallback + Baseline C (Always-Compress RAG)
     ↓
M9:  Complexity Router (data prep → train → evaluate)
     ↓
M10: Adaptive Controller + Baseline D (AdaptiveRAG) + CLI Demo [mandatory]
     ↓
M11: Primary Benchmark (all 4 baselines on identical test conditions)
     ↓
M12: Ablation + Error Analysis + Reproducibility Metadata
     ↓
     MVP STOP ✋
     ↓
     Post-MVP: Report + Plots + Writing
```

---

## ADR Status

| Decision | Status | Resolution |
|---|---|---|
| **ADR-001** Generator model | ✅ Resolved | `HuggingFaceTB/SmolLM-135M-Instruct` |
| **ADR-002** Router model | 🔲 Open | Direction: `microsoft/deberta-v3-small`; must confirm at M9.1 |
| **ADR-003** Embedding model | ✅ Resolved | `BAAI/bge-small-en-v1.5` |
| **ADR-004** Vector store | ✅ Resolved | ChromaDB |
| **ADR-005** K values | 🔲 Open (defaults set) | K_simple=2, K_complex=10, K_baseline=5; validate at M9.3 |
| **ADR-006** Compression budget | ✅ Resolved (initial) | 0.5; validated at M12 ablation |
| **ADR-007** Router confidence threshold | 🔲 Open | Determine after M9.3 router evaluation |
| **ADR-008** External inference fallback | ✅ Not needed | Local SmolLM confirmed feasible on CPU |

---

## Risks and Mitigations

| Risk | Milestone | Detection | Mitigation |
|---|---|---|---|
| Router OOM during training | M9.2 | Training script error | Reduce batch size; try lighter encoder (update ADR-002) |
| Router learns dataset identity | M9.3 | Error distribution analysis | Mix datasets in training; inspect false-classification examples |
| Test leakage into router training | M9.1 | Leakage check script | Enforce split separation at data preparation time |
| Compression degrades answer quality | M12 | Ablation A results | Report result honestly; analyze in error analysis |
| SmolLM generates poor quality answers | M11 | Benchmark EM/F1 results | Report result honestly; discuss as limitation |
| Latency inconsistency across baselines | M11 | Fairness pre-check | Document cache state; run under identical conditions |
