# AdaptiveRAG — Development Plan

> **Status:** Roadmap v1.0  
> **Derived from:** [`prod-spec.md`](prod-spec.md) · [`architecture.md`](architecture.md)  
> **Project:** AdaptiveRAG — Adaptive Retrieval-Augmented Generation using Query Complexity and Context Optimization  
> **Purpose:** Convert the architecture into an executable, dependency-ordered implementation roadmap

---

## Document Role

```text
prod-spec.md        →  WHAT and WHY
architecture.md     →  HOW the system is structured
development-plan.md →  WHEN / IN WHAT ORDER / HOW TO VERIFY
implementation      →  actual code
```

This plan does not restate the architecture or product specification. It tells the team what to do next, in what order, and how to verify it.

---

## Current Repository State

As of the creation of this plan, the repository contains:

```text
adaptive-rag-query-aware-retrieval/
├── .git/
└── docs/
    ├── prod-spec.md
    ├── architecture.md
    └── development-plan.md   ← this file
```

**There is no application code, no scaffolding, no configuration, no dependencies, no tests.**

All implementation milestones begin from this baseline.

---

## Hardware Integration

> **This is a pure software project.**  
> No physical hardware (microcontrollers, sensors, actuators, robotic arms) is required or planned. All milestones are software-only.  
> Model inference runs on consumer GPU or CPU — see ADR-001 and ADR-008 in `architecture.md`.

---

## Execution Principles

1. **Small milestones** — every milestone produces one testable outcome.
2. **Verify before proceeding** — do not advance until the current milestone's definition of done is met.
3. **Vertical slices** — prefer a thin end-to-end slice over building every subsystem in isolation.
4. **Dependency-aware** — do not build the adaptive system before its components work independently.
5. **Baselines first** — build all baselines before the adaptive comparison.
6. **Evaluation early** — build metric infrastructure before large-scale experiments.
7. **Configuration-driven** — no magic constants; every parameter lives in config.

---

## Phase Overview

| Phase | Name | What It Produces |
|---|---|---|
| **0** | Foundation | Working repo structure, config, dev fixtures |
| **1** | Data Pipeline | Loadable, split, chunked datasets + vector index |
| **2** | Generator Interface | Generator loads and produces answers |
| **3** | First Vertical Slice ⭐ | query → retrieve → generate → answer |
| **4** | Baseline A — LLM Only | Generator answers without retrieval |
| **5** | Baseline B — Fixed RAG | Fixed-K retrieval + generator, evaluated |
| **6** | Context Compression | LLMLingua integrated and measured |
| **7** | Baseline C — Always-Compress RAG | Full compress pipeline, evaluated |
| **8** | Evaluation Infrastructure | All metrics validated on controlled examples |
| **9** | Complexity Router | Trained router producing SIMPLE/COMPLEX labels |
| **10** | Adaptive Controller | Both routing paths work and are verified |
| **11** | Baseline D — AdaptiveRAG | Full adaptive pipeline, evaluated |
| **12** | Experiment Fairness Checkpoint | All baselines verified on identical conditions |
| **13** | Primary Benchmark | Comparative results for all four systems |
| **14** | Ablations | Compression and adaptive-K ablations |
| **15** | Error Analysis | Representative failures inspected |
| **16** | Reproducibility Checkpoint | Full rerun verified from config |
| **17** | MVP Hard Stop ✋ | MVP acceptance criteria satisfied |
| **18** | Demo (Optional, Post-MVP) | Lightweight interactive demonstration |

---

## Parallel Work Opportunities

The following tasks do not block each other and can proceed in parallel once Phase 0 is complete:

| Track A | Track B |
|---|---|
| Data loading + chunking (M1.x) | Evaluation metric implementation (M8.x) |
| Generator interface (M2.x) | Config scaffolding (M0.x) |
| Unit-test infrastructure (M0.3) | Router data preparation (M9.1) |

All other phases have hard sequential dependencies.

---

## Milestone Notation

- **M{phase}.{step}** — milestone identifier
- **ADR Required** — an open architectural decision must be resolved before starting
- **⭐ First Vertical Slice** — minimum viable end-to-end path

---

---

# Phase 0 — Foundation

> **Goal:** A usable repository structure, working configuration system, and dev fixtures so all subsequent implementation can begin cleanly.

---

### M0.1 — Repository Structure

**Objective:** Create the directory layout defined in `architecture.md §15`.

**Prerequisites:** None.

**ADRs Required:** None.

**Expected directories/files:**

```text
src/router/
src/retriever/
src/optimizer/
src/generator/
src/pipeline/
src/evaluation/
src/utils/
datasets/
scripts/
configs/
tests/
results/
notebooks/
requirements.txt
```

**Expected behavior:** `import src.utils.config` works without error.

**Tests / Verification:** Directory structure matches `architecture.md §15.1`.

**Definition of Done:**
- [ ] All directories exist.
- [ ] All `__init__.py` files present in `src/` subdirectories.
- [ ] `requirements.txt` lists known dependencies (even if unpinned initially).

**Deliverable:** Repository structure matching the architecture's module layout.

---

### M0.2 — Configuration System

**Objective:** Implement `src/utils/config.py` — loads YAML config, validates required keys, exposes typed values.

**Prerequisites:** M0.1.

**ADRs Required:**
- **ADR-005** (K values): Use defaults K_simple=2, K_complex=10, K_baseline=5 for now.

**Expected files:**
- `src/utils/config.py`
- `configs/config.yaml` (with schema defined in `architecture.md §10.1`)

**Expected behavior:** `config = load_config("configs/config.yaml")` returns a populated config object. Invalid config raises a clear error immediately.

**Tests / Verification:**
- Unit test: valid config loads correctly.
- Unit test: missing required key raises an informative error.
- Unit test: K values match defaults.

**Definition of Done:**
- [ ] Config loads from YAML without error.
- [ ] All parameters from `architecture.md §10.1` are accessible.
- [ ] Invalid config fails fast with a human-readable message.
- [ ] Unit tests pass.

**Deliverable:** Working configuration system; `configs/config.yaml` with all known parameters.

---

### M0.3 — Logging and Test Infrastructure

**Objective:** Implement `src/utils/logger.py` and set up `pytest` so tests can run from day one.

**Prerequisites:** M0.1.

**ADRs Required:** None.

**Expected files:**
- `src/utils/logger.py`
- `tests/conftest.py`
- `tests/test_config.py` (from M0.2)

**Expected behavior:** `pytest tests/` runs and exits cleanly (even if all tests are stubs initially).

**Definition of Done:**
- [ ] `pytest` runs without import errors.
- [ ] Logger produces structured output with timestamp and component name.
- [ ] M0.2 config unit tests pass within this infrastructure.

**Deliverable:** Functional test harness and logging utility.

---

### M0.4 — Dev Fixtures

**Objective:** Create a tiny local fixture dataset (10–20 examples each for SIMPLE-proxy and COMPLEX-proxy queries) that can be used for all integration tests without downloading large datasets.

**Prerequisites:** M0.1, M0.2.

**ADRs Required:** None.

**Expected files:**
- `datasets/fixtures/simple_queries.json`
- `datasets/fixtures/complex_queries.json`

**Format:** Each example: `{ "query": "...", "reference_answer": "...", "complexity_label": "SIMPLE"|"COMPLEX" }`.

**Expected behavior:** Fixtures load with a helper function; no network dependency.

**Definition of Done:**
- [ ] 10+ SIMPLE-proxy examples.
- [ ] 10+ COMPLEX-proxy examples (multi-hop style).
- [ ] Helper function loads fixtures without network access.
- [ ] Fixtures are deterministic and version-controlled.

**Deliverable:** Local dev fixtures usable in all subsequent integration tests.

---

### Checkpoint A — Foundation Ready

> **Gate:** M0.1 + M0.2 + M0.3 + M0.4 complete. Config loads, pytest runs, fixtures exist. All subsequent work can begin.

**Risks at this phase:**

| Risk | Detection | Fallback |
|---|---|---|
| Dependency conflicts in requirements.txt | `pip install` errors | Pin specific versions; test in clean venv |
| Config schema gaps discovered later | Missing key errors during integration | Update schema; config system must not fail silently |

---

---

# Phase 1 — Data Pipeline

> **Goal:** Load PopQA and HotpotQA, create train/validation/test splits, chunk the corpus, and build the vector index.

---

### M1.1 — Dataset Loading

**Objective:** Implement `datasets/load_dataset.py` — downloads and normalizes PopQA and HotpotQA. Preserves question, reference answer, supporting document information, and dataset split.

**Prerequisites:** M0.2 (config for dataset parameters).

**ADRs Required:**
- **ADR-005**: Sample limits per dataset (initial: ~1,000 per dataset from product spec).

**Expected files:**
- `datasets/load_dataset.py`

**Expected behavior:**
- Running the script produces normalized JSON/JSONL files in `datasets/raw/`.
- Each entry retains: `query`, `answer`, `supporting_docs` (where available), `dataset_source`, `split`.

**Tests / Verification:**
- Verify at least one PopQA and one HotpotQA example loaded with all required fields.
- Verify counts match configured sample limits.

**Definition of Done:**
- [ ] PopQA loads without error.
- [ ] HotpotQA loads without error.
- [ ] Both datasets normalized to a common schema.
- [ ] No reference answers leaked into query fields.

**Deliverable:** Raw normalized datasets in `datasets/raw/`.

---

### M1.2 — Train / Validation / Test Splits

**Objective:** Implement `datasets/prepare_data.py` — creates reproducible TRAIN/VALIDATION/TEST splits.

**Prerequisites:** M1.1.

**ADRs Required:** None (split ratios are implementation decisions; start with 70/15/15 or similar).

**Expected files:**
- `datasets/prepare_data.py`
- `datasets/splits/` (train.jsonl, val.jsonl, test.jsonl)

**Critical constraint from product spec:** Test examples must not be used for tuning router thresholds or K values.

**Expected behavior:** Fixed random seed produces identical splits on every run.

**Tests / Verification:**
- Verify no query appears in both train and test.
- Verify split counts are logged and recorded.
- Verify random seed reproduces identical splits.

**Definition of Done:**
- [ ] Three splits produced (TRAIN, VALIDATION, TEST).
- [ ] No query overlap between TRAIN and TEST.
- [ ] Split creation is deterministic (seeded).
- [ ] Split metadata (counts, seed) is saved alongside splits.

**Deliverable:** Reproducible data splits; leakage verified.

---

### M1.3 — Corpus Preparation and Chunking

**Objective:** Implement corpus construction and fixed chunking.

**Prerequisites:** M1.1, M0.2 (chunk_size and chunk_overlap from config).

**ADRs Required:** None (chunk_size and chunk_overlap are configurable defaults to be validated later).

**Expected files:**
- `datasets/prepare_data.py` (extended)
- `datasets/corpus/` (chunked documents)

**Expected behavior:**
- Supporting documents are chunked with configurable `chunk_size` and `chunk_overlap`.
- Each chunk retains `document_id`, `chunk_id`, `text`, `source_metadata`.
- All baselines will use this identical corpus.

**Tests / Verification:**
- Verify chunks have expected fields.
- Verify chunk_overlap produces expected overlap.
- Verify total chunk count is logged.

**Definition of Done:**
- [ ] Corpus chunked and saved.
- [ ] `chunk_size` and `chunk_overlap` read from config (not hard-coded).
- [ ] Each chunk has `document_id`, `chunk_id`, `text`.
- [ ] Chunking is deterministic.

**Deliverable:** Chunked corpus ready for indexing.

---

### M1.4 — Embedding Model Integration

**Objective:** Implement `src/retriever/embeddings.py` — loads the embedding model and encodes text.

**Prerequisites:** M0.2.

**ADRs Required:**
- **ADR-003**: Embedding model. **Current direction: `BAAI/bge-small-en-v1.5`.** Must be confirmed based on resource usage and retrieval quality on validation set before indexing.

**Expected files:**
- `src/retriever/embeddings.py`

**Expected behavior:**
- `embed(text: str) -> vector` works.
- `embed_batch(texts: list) -> list[vector]` works.
- Model loads from config-specified name.

**Tests / Verification:**
- Encode a known query; verify output is a float vector of expected dimensionality.
- Verify two semantically similar queries produce closer vectors than two dissimilar ones (sanity check).

**Definition of Done:**
- [ ] Embedding model loads from config.
- [ ] Single and batch encoding work.
- [ ] Embedding dimensionality logged.
- [ ] Unit test passes.

**Deliverable:** Embedding model interface; embeddings can be produced.

---

### M1.5 — Vector Index Construction

**Objective:** Implement `src/retriever/vector_store.py` — builds and persists the ChromaDB index from the chunked corpus.

**Prerequisites:** M1.3, M1.4.

**ADRs Required:**
- **ADR-004**: Vector store. **Current direction: ChromaDB.** Confirm before building large index.

**Expected files:**
- `src/retriever/vector_store.py`
- `datasets/build_index.py`
- `data/index/` (persisted ChromaDB files)

**Expected behavior:**
- Running `build_index.py` produces a persistent ChromaDB collection.
- Collection stores chunk embeddings and metadata.
- Index is deterministic: same corpus + same embedding model = same index.

**Tests / Verification:**
- Insert 5 known chunks; verify they are retrievable.
- Verify persisted index can be reloaded without rebuilding.

**Definition of Done:**
- [ ] Index built and persisted.
- [ ] Index reloads correctly without re-embedding.
- [ ] Metadata (document_id, chunk_id, text) retrievable with embeddings.
- [ ] Index build time logged.

**Deliverable:** Persistent vector index ready for retrieval.

---

### M1.6 — Top-K Retrieval

**Objective:** Implement `src/retriever/retriever.py` — exposes `retrieve(query, top_k) -> RankedDocuments`.

**Prerequisites:** M1.4, M1.5.

**Expected files:**
- `src/retriever/retriever.py`

**Expected behavior:**
- Given a query string and integer K, returns K ranked chunks.
- Each result includes `document_id`, `chunk_id`, `text`, `score`, `rank`.
- Different K values return different counts.

**Tests / Verification:**
- Known query + known corpus → verify expected top-1 chunk is returned.
- Verify K=2 returns exactly 2 results.
- Verify K=10 returns exactly 10 results (or all if corpus < 10).
- Verify results are ranked by score.

**Definition of Done:**
- [ ] `retrieve(query, top_k)` returns correctly ranked chunks.
- [ ] All required metadata fields present in each result.
- [ ] Retrieval is deterministic given the same index and query.
- [ ] Retrieval latency is measured and logged.
- [ ] Unit + integration tests pass.

**Deliverable:** Working retriever interface. The retrieval layer is now independently functional.

---

### Checkpoint B — Data & Retrieval Ready

> **Gate:** M1.1–M1.6 complete. Data is loaded, split, chunked, indexed, and retrievable. The retrieval layer works independently. Evaluation metrics for retrieval can be implemented in parallel.

**Risks at this phase:**

| Risk | Detection | Fallback |
|---|---|---|
| Embedding model too large for hardware | OOM error during M1.4 | Switch to smaller model; requires ADR-003 update |
| ChromaDB indexing too slow | Time log during M1.5 | Evaluate FAISS; requires ADR-004 update |
| Chunk size produces poor retrieval | Retrieval sanity check in M1.6 | Adjust chunk_size in config and rebuild |

---

---

# Phase 2 — Generator Interface

> **Goal:** The generator model loads and produces answers from query + context. No routing or retrieval yet.

---

### M2.1 — Generator Model Loading

**Objective:** Implement `src/generator/response_generator.py` — loads the generator LLM and runs inference.

**Prerequisites:** M0.2.

**ADRs Required:**
- **ADR-001**: Generator model. **Current direction: Phi-3-mini-4k-instruct.** Must be confirmed based on VRAM, inference stability, and context capacity before this milestone can begin. This is a blocking decision.

**Expected files:**
- `src/generator/response_generator.py`

**Expected behavior:**
- Model loads from config-specified identifier.
- `generate(query, context) -> GeneratedAnswer` works.
- Temperature, max_new_tokens, and seed are read from config.

**Tests / Verification:**
- Load model; run one inference with a short context; verify a non-empty string is returned.
- Verify generation is deterministic with a fixed seed.
- Log VRAM usage and generation latency.

**Definition of Done:**
- [ ] Generator loads from config.
- [ ] `generate(query, context)` returns a non-empty answer string.
- [ ] Generation latency measured and logged.
- [ ] VRAM usage recorded.
- [ ] Temperature, max_new_tokens, seed are config-driven.

**Deliverable:** Generator interface that can be used by any baseline.

---

### M2.2 — Prompt Construction

**Objective:** Implement the fixed prompt template from `architecture.md §4.9`.

**Prerequisites:** M2.1.

**Expected files:**
- `src/generator/response_generator.py` (extended)

**Prompt contract:**

```text
Question:
    <query>

Context:
    <retrieved or compressed context>
```

**Expected behavior:** Prompt is assembled identically for every baseline. Prompt template is not embedded in pipeline logic — it lives in the generator module.

**Tests / Verification:**
- Unit test: given query + context strings, prompt is correctly assembled.
- Verify template does not change between SIMPLE and COMPLEX paths.

**Definition of Done:**
- [ ] Prompt assembled correctly from query + context.
- [ ] Template is identical across simple and complex paths (verified by unit test).
- [ ] Prompt template is not hard-coded in pipeline logic.

**Deliverable:** Consistent prompt construction used by all baselines.

---

---

# Phase 3 — First Vertical Slice ⭐

> **Goal:** A minimal end-to-end path: query → preprocess → retrieve (fixed K) → generate → answer. No router, no compression, no evaluation. Just prove data flows.

---

### M3.1 — Query Preprocessor

**Objective:** Implement `src/router/query_classifier.py` or a standalone `preprocessor.py` — validates and normalizes the input query.

**Prerequisites:** M0.2.

**Expected files:**
- `src/router/complexity_estimator.py` or `src/utils/preprocessor.py`

**Expected behavior:**
- Empty query rejected with a structured error.
- Whitespace normalized.
- Original query preserved alongside normalized query.
- Preprocessor does not alter semantic meaning.

**Tests / Verification:**
- Unit test: empty string → structured error.
- Unit test: `"  What is X?  "` → `"What is X?"`.
- Unit test: original query preserved in output.

**Definition of Done:**
- [ ] Empty/invalid queries rejected with a clear error.
- [ ] Normalized query returned.
- [ ] Original query preserved.
- [ ] All unit tests pass.

**Deliverable:** Preprocessor module usable by all pipeline paths.

---

### M3.2 — Minimal Pipeline (First Vertical Slice)

**Objective:** Implement `src/pipeline/adaptive_rag.py` with a minimal pipeline that wires preprocessor → retriever → generator.

**Prerequisites:** M1.6, M2.2, M3.1.

**ADRs Required:** ADR-001 (generator model) must already be resolved from M2.1.

**Expected files:**
- `src/pipeline/adaptive_rag.py`
- `scripts/run_pipeline.py`

**Expected behavior:**

```text
Query string
    → Preprocessor (validate + normalize)
    → Retriever (top_k = baseline_k from config)
    → Generator (query + retrieved context)
    → Answer + ExecutionMetadata
```

No router, no controller, no compression. Fixed K from config.

**Tests / Verification:**
- Use dev fixtures (M0.4): run a known simple-proxy query.
- Verify answer string is non-empty.
- Verify ExecutionMetadata contains: query, retrieval_k, retrieval_latency_ms, generation_latency_ms, total_latency_ms.
- Verify all metadata is populated (not None/zero).

**Definition of Done:**
- [ ] End-to-end query → answer works.
- [ ] ExecutionMetadata populated with latency fields.
- [ ] Retrieval K read from config.
- [ ] Integration test passes using dev fixtures.

**Deliverable:** ⭐ **FIRST VERTICAL SLICE** — basic data flow proven end-to-end.

---

### Checkpoint B+ — First Vertical Slice Ready

> **Gate:** M3.2 complete. The fundamental query → retrieve → generate → answer path works. Integration problems can now be detected early.

---

---

# Phase 4 — Baseline A: LLM Only

> **Goal:** Generator produces answers without any retrieval. This is the simplest baseline and validates generator quality in isolation.

---

### M4.1 — LLM-Only Pipeline

**Objective:** Implement the LLM-only baseline — query goes directly to generator with no retrieved context.

**Prerequisites:** M2.2, M3.1.

**Expected files:**
- `src/pipeline/adaptive_rag.py` (extend with baseline mode)
- `scripts/run_pipeline.py` (extend with `--baseline=llm_only` flag)

**Expected behavior:**

```text
Query → Preprocessor → Generator (context = empty or minimal prompt) → Answer
```

No retrieval call. No compression.

**Tests / Verification:**
- Run 5 fixture queries; verify answers are non-empty.
- Verify retrieval was NOT called (no retrieval latency in metadata).
- Verify `ExecutionMetadata.retrieval_k = 0`.

**Definition of Done:**
- [ ] LLM-only pipeline runs without retrieval.
- [ ] Metadata correctly shows no retrieval.
- [ ] Answers are non-empty strings.
- [ ] Results can be saved to `results/baseline_a/`.

**Deliverable:** Baseline A (LLM Only) functional. First research baseline established.

---

---

# Phase 5 — Baseline B: Standard Fixed RAG

> **Goal:** Fixed-K retrieval + generator pipeline is implemented, evaluated, and results are saved.

---

### M5.1 — Fixed RAG Pipeline

**Objective:** Wire the existing preprocessor + retriever + generator into a clean fixed-RAG baseline with configurable K.

**Prerequisites:** M3.2 (first vertical slice), M4.1 (to confirm pipeline extension pattern).

**Expected files:**
- `src/pipeline/adaptive_rag.py` (baseline B mode)
- `scripts/run_pipeline.py` (extend with `--baseline=fixed_rag`)

**Expected behavior:**

```text
Query → Preprocessor → Retriever(K=baseline_k) → Generator → Answer
```

`baseline_k` = 5 from config (configurable).

**Tests / Verification:**
- Run 5 fixture queries with K=5; verify exactly 5 chunks are retrieved.
- Verify chunk metadata flows through to ExecutionMetadata.
- Verify results saved to `results/baseline_b/`.

**Definition of Done:**
- [ ] Fixed RAG pipeline runs with configurable K.
- [ ] Retrieved chunk metadata present in ExecutionMetadata.
- [ ] Results saved in machine-readable format.
- [ ] Integration test passes.

**Deliverable:** Baseline B (Standard Fixed RAG) functional.

---

### Checkpoint C — Retrieval Baseline Ready

> **Gate:** M5.1 complete. Fixed RAG with retrieval works end-to-end. The retrieval-to-generation path is proven.

---

---

# Phase 6 — Context Compression

> **Goal:** LLMLingua is integrated, compression is measured, and compression failure fallback works.

---

### M6.1 — LLMLingua Integration

**Objective:** Implement `src/optimizer/context_optimizer.py` — wraps LLMLingua to compress retrieved context.

**Prerequisites:** M0.2 (compression settings from config).

**ADRs Required:**
- **ADR-006**: Compression budget. Use a reasonable initial default (e.g., ratio=0.5); final value to be selected using validation data later.

**Expected files:**
- `src/optimizer/context_optimizer.py`

**Expected behavior:**

```text
compress(query, retrieved_context) → OptimizedContext {
    text, original_tokens, compressed_tokens, compression_ratio, latency_ms, status
}
```

**Tests / Verification:**
- Pass a known context string (50+ tokens); verify compressed output is shorter.
- Verify `original_tokens > compressed_tokens`.
- Verify `compression_ratio = original_tokens / compressed_tokens`.
- Verify `compression_latency_ms > 0`.
- Verify `status = SUCCESS`.

**Definition of Done:**
- [ ] `compress()` returns an `OptimizedContext` with all fields populated.
- [ ] Token counts are correct.
- [ ] Compression ratio is calculated correctly.
- [ ] Compression latency is measured.
- [ ] All unit tests pass.

**Deliverable:** Compression interface working independently.

---

### M6.2 — Compression Failure Fallback

**Objective:** Verify that if compression fails, the original context is preserved and failure is logged.

**Prerequisites:** M6.1.

**Expected behavior:**
- If LLMLingua raises an exception: `status = FAILED`, original context returned, failure logged.
- Pipeline continues with uncompressed context.

**Tests / Verification:**
- Unit test: simulate compressor exception → verify original context returned.
- Verify `compression_applied = false` in ExecutionMetadata on failure.
- Verify failure is logged.

**Definition of Done:**
- [ ] Compressor failure does not crash the pipeline.
- [ ] Original context returned on failure.
- [ ] Failure status recorded in metadata.
- [ ] Unit test for failure path passes.

**Deliverable:** Compression with reliable failure fallback.

---

---

# Phase 7 — Baseline C: Always-Compress RAG

> **Goal:** Full always-compress pipeline: retrieve K=10 → compress → generate → answer, evaluated and saved.

---

### M7.1 — Always-Compress Pipeline

**Objective:** Wire retriever + compressor + generator into the always-compress baseline.

**Prerequisites:** M5.1, M6.2.

**Expected files:**
- `src/pipeline/adaptive_rag.py` (baseline C mode)
- `scripts/run_pipeline.py` (extend with `--baseline=always_compress`)

**Expected behavior:**

```text
Query → Preprocessor → Retriever(K=10) → Compressor → Generator → Answer
```

Compression always applied. K=10 from config.

**Tests / Verification:**
- Run 5 fixture queries; verify all have `compression_applied = true`.
- Verify `compressed_context_tokens < original_context_tokens`.
- Verify answer is non-empty.
- Verify all latency fields populated (including `compression_latency_ms`).
- Verify results saved to `results/baseline_c/`.

**Definition of Done:**
- [ ] Always-compress pipeline runs end-to-end.
- [ ] All compression metadata fields populated.
- [ ] Results saved in machine-readable format.
- [ ] Integration test passes.

**Deliverable:** Baseline C (Always-Compress RAG) functional.

---

### Checkpoint D — Compression Ready

> **Gate:** M7.1 complete. All non-adaptive baselines (A, B, C) are functional. The compression path is proven.

---

---

# Phase 8 — Evaluation Infrastructure

> **Goal:** All metrics (answer quality, retrieval quality, router quality, efficiency) are implemented and validated on controlled examples before any large-scale experiments.

> **Note:** Evaluation metric implementation (M8.x) can proceed **in parallel** with Phases 1–7 after Phase 0 is complete.

---

### M8.1 — Answer Quality Metrics

**Objective:** Implement `src/evaluation/metrics.py` — Exact Match and token-level F1.

**Prerequisites:** M0.1.

**Expected files:**
- `src/evaluation/metrics.py`
- `tests/test_metrics.py`

**Expected behavior:**
- `exact_match(prediction, reference) -> float` returns 1.0 or 0.0.
- `token_f1(prediction, reference) -> float` returns overlap score.

**Tests / Verification:**
- `exact_match("Paris", "Paris")` → 1.0.
- `exact_match("Paris", "London")` → 0.0.
- `token_f1("the cat sat", "cat sat on mat")` → expected overlap value.
- Edge cases: empty strings, case sensitivity rules.

**Definition of Done:**
- [ ] EM and F1 produce correct values on known examples.
- [ ] Edge cases handled.
- [ ] Unit tests pass.

**Deliverable:** Answer quality metric functions.

---

### M8.2 — Retrieval Quality Metrics

**Objective:** Implement Recall@K, MRR, nDCG in `src/evaluation/metrics.py`.

**Prerequisites:** M8.1.

**Expected behavior:**
- `recall_at_k(retrieved_ids, relevant_ids, k) -> float`
- `mrr(retrieved_ids, relevant_ids) -> float`

**Tests / Verification:**
- `recall_at_k(["a", "b", "c"], ["a"], k=3)` → 1.0.
- `recall_at_k(["b", "c"], ["a"], k=2)` → 0.0.
- MRR: relevant doc at rank 2 → MRR = 0.5.

**Definition of Done:**
- [ ] Recall@K, MRR correct on known examples.
- [ ] Functions handle empty relevant-set gracefully.
- [ ] Unit tests pass.

**Deliverable:** Retrieval quality metric functions.

---

### M8.3 — Router Quality Metrics

**Objective:** Implement accuracy, precision, recall, F1, and confusion matrix for binary SIMPLE/COMPLEX classification.

**Prerequisites:** M8.1.

**Expected behavior:**
- `router_metrics(predictions, labels) -> RouterMetrics { accuracy, precision, recall, f1, confusion_matrix }`.

**Tests / Verification:**
- Known predictions vs labels → verify each metric value manually.
- Confusion matrix has correct SIMPLE/COMPLEX cell counts.

**Definition of Done:**
- [ ] All router metrics correct on known examples.
- [ ] Confusion matrix populated correctly.
- [ ] Unit tests pass.

**Deliverable:** Router quality metric functions.

---

### M8.4 — Efficiency and Latency Instrumentation

**Objective:** Implement `src/evaluation/latency.py` — timing utilities and token counting.

**Prerequisites:** M0.1.

**Expected files:**
- `src/evaluation/latency.py`

**Expected behavior:**
- `measure_latency(fn) -> (result, latency_ms)` — wraps any function and returns its wall-clock time.
- `count_tokens(text, tokenizer) -> int` — token count using model tokenizer.
- `compression_ratio(original_tokens, compressed_tokens) -> float`.

**Tests / Verification:**
- `measure_latency(time.sleep, 0.1)` → latency ≈ 100ms (±5ms).
- `compression_ratio(1000, 250)` → 4.0.
- `count_tokens("Hello world", tokenizer)` → expected count.

**Definition of Done:**
- [ ] Latency measurement wrapper works.
- [ ] Token counting works.
- [ ] Compression ratio formula correct.
- [ ] Unit tests pass.

**Deliverable:** Latency and token measurement utilities, usable by all pipeline stages.

---

### M8.5 — Benchmark Runner

**Objective:** Implement `src/evaluation/benchmark.py` — runs a batch of examples through any pipeline and produces an `EvaluationMetrics` result.

**Prerequisites:** M8.1–M8.4, M5.1 (at least one working baseline to test against).

**Expected files:**
- `src/evaluation/benchmark.py`
- `scripts/evaluate.py`

**Expected behavior:**
- Accepts a list of examples and a pipeline callable.
- Runs each example; collects answer, metadata, latency.
- Computes aggregate EM, F1, latency stats, token stats.
- Saves results to `results/<experiment_name>/`.

**Tests / Verification:**
- Run benchmark on 10 fixture examples through the fixed-RAG baseline.
- Verify output file is created.
- Verify metrics are non-null.
- Verify result is machine-readable (JSON/JSONL).

**Definition of Done:**
- [ ] Benchmark runner executes any pipeline callable.
- [ ] Aggregate metrics computed correctly.
- [ ] Results saved with experiment metadata.
- [ ] Integration test with fixed-RAG fixture passes.

**Deliverable:** Benchmark runner; can now evaluate any baseline.

---

### Checkpoint E — Evaluation Infrastructure Ready

> **Gate:** M8.1–M8.5 complete. All metric functions are validated. The benchmark runner can evaluate any pipeline. Only now is large-scale evaluation meaningful.

---

---

# Phase 9 — Complexity Router

> **Goal:** A trained binary classifier produces SIMPLE/COMPLEX labels with measured accuracy and latency. Router is independently validated.

---

### M9.1 — Router Training Data Preparation

**Objective:** Create labeled (query, complexity_label) pairs for router training from the dataset splits.

**Prerequisites:** M1.2 (train/val/test splits).

**ADRs Required:**
- **ADR-002**: Router model. **Current direction: DeBERTa-v3-small.** Must be confirmed before training begins. Key criterion: training feasibility on available hardware.

**Expected files:**
- `datasets/prepare_data.py` (extended)
- `datasets/splits/router_train.jsonl`
- `datasets/splits/router_val.jsonl`

**Labeling strategy from product spec:**
- PopQA examples → initial SIMPLE proxy
- HotpotQA examples → initial COMPLEX proxy
- Mix datasets; do NOT use dataset identity as an explicit feature.
- Hold out test set; do NOT use test data for label assignment.

**Leakage check:** Verify no test queries appear in router training data.

**Tests / Verification:**
- Label distribution is logged (SIMPLE count, COMPLEX count).
- No test query appears in router train set.
- Labels are saved as a structured field, not inferred from file path.

**Definition of Done:**
- [ ] Router train and validation splits created.
- [ ] Label distribution logged.
- [ ] Data leakage check passes (no test examples in train).
- [ ] Labels are explicit fields, not derived from dataset identity at runtime.

**Deliverable:** Labeled router training data.

---

### M9.2 — Router Model Interface

**Objective:** Implement `src/router/query_classifier.py` and `src/router/complexity_estimator.py` — wraps the encoder classifier for SIMPLE/COMPLEX prediction.

**Prerequisites:** M9.1, M0.2.

**Expected files:**
- `src/router/query_classifier.py`
- `src/router/complexity_estimator.py`

**Expected behavior:**
- `classify(query: str) -> ComplexityResult { label: SIMPLE|COMPLEX, confidence: float }`.
- Model name loaded from config.
- Inference latency measured.

**Tests / Verification (before training):**
- Verify interface with a randomly initialized model (just structural test).
- Verify `label` is always `SIMPLE` or `COMPLEX`.
- Verify `confidence` is in [0, 1].
- Verify latency is measured.

**Definition of Done:**
- [ ] `classify()` interface works structurally.
- [ ] Output fields are validated.
- [ ] Latency measured.
- [ ] Unit test (with mock model) passes.

**Deliverable:** Router interface (model weights not yet meaningful — trained next).

---

### M9.3 — Router Training

**Objective:** Fine-tune the lightweight encoder classifier on the prepared router training data.

**Prerequisites:** M9.1, M9.2.

**Expected files:**
- `scripts/train_router.py`
- `models/router/` (trained checkpoint)

**Expected behavior:**
- Training completes without OOM.
- Validation accuracy is logged per epoch.
- Best checkpoint is saved.
- Training time is recorded.

**ADR check:** If training OOMs or takes unreasonably long, ADR-002 may need to be revisited.

**Definition of Done:**
- [ ] Training completes.
- [ ] Best checkpoint saved to `models/router/`.
- [ ] Validation accuracy ≥ reasonable baseline (>50% — better than random).
- [ ] Training time and hardware recorded.

**Deliverable:** Trained router checkpoint.

---

### M9.4 — Router Evaluation

**Objective:** Evaluate the trained router on the held-out validation split. Produce router quality metrics.

**Prerequisites:** M9.3, M8.3.

**Expected files:**
- `scripts/evaluate.py` (extended for router evaluation)
- `results/router_eval/metrics.json`

**Expected behavior:**
- Accuracy, precision, recall, F1, confusion matrix computed on validation set.
- Error examples (false SIMPLE, false COMPLEX) are logged for inspection.

**Tests / Verification:**
- Verify confusion matrix sums to total validation set size.
- Inspect 5 false-positive and 5 false-negative examples manually.
- Verify router is not simply learning dataset identity (check error distribution).

**Definition of Done:**
- [ ] Router metrics computed on held-out validation data.
- [ ] Confusion matrix logged.
- [ ] False-classification examples are accessible for inspection.
- [ ] Router latency measured and recorded (per inference).

**Deliverable:** Router validation results; router is independently evaluated.

---

### Checkpoint E+ — Router Ready

> **Gate:** M9.4 complete. Router is trained, validated, and independently measured. Ready for integration into the controller.

---

---

# Phase 10 — Adaptive Controller and AdaptiveRAG

> **Goal:** The routing controller wires the router to the retriever and conditional compressor. Both the SIMPLE and COMPLEX paths are explicitly verified.

---

### M10.1 — Adaptive Routing Controller

**Objective:** Implement `src/router/routing_logic.py` — converts `ComplexityResult` to `RoutingDecision`.

**Prerequisites:** M9.2 (router interface).

**Routing policy (from `architecture.md §4.4`):**

| Complexity | retrieval_k | compression_enabled |
|---|---|---|
| SIMPLE | `K_simple` (config, default=2) | `false` |
| COMPLEX | `K_complex` (config, default=10) | `true` |

**ADRs Required:**
- **ADR-005**: K values. Use defaults K_simple=2, K_complex=10 until validated on validation data.

**Expected files:**
- `src/router/routing_logic.py`

**Expected behavior:**
- `ComplexityResult { label: SIMPLE }` → `RoutingDecision { retrieval_k: 2, compression_enabled: false }`.
- `ComplexityResult { label: COMPLEX }` → `RoutingDecision { retrieval_k: 10, compression_enabled: true }`.
- All values read from config.

**Tests / Verification:**
- Unit test: SIMPLE input → K_simple, compression=false.
- Unit test: COMPLEX input → K_complex, compression=true.
- Unit test: K values change when config changes.

**Definition of Done:**
- [ ] Controller is deterministic (same input → same output always).
- [ ] K values read from config.
- [ ] Unit tests for both routes pass.

**Deliverable:** Adaptive routing controller; the core policy is testable in isolation.

---

### M10.2 — AdaptiveRAG End-to-End Pipeline

**Objective:** Connect all components: preprocessor → router → controller → retriever → conditional compressor → generator.

**Prerequisites:** M10.1, M6.2, M9.3, M3.2.

**Expected files:**
- `src/pipeline/adaptive_rag.py` (baseline D / full adaptive mode)
- `scripts/run_pipeline.py` (extend with `--baseline=adaptive`)

**Expected behavior:**

```text
SIMPLE query:
    Preprocessor → Router(SIMPLE) → Controller(K=2, compress=false)
    → Retriever(K=2) → Generator → Answer

COMPLEX query:
    Preprocessor → Router(COMPLEX) → Controller(K=10, compress=true)
    → Retriever(K=10) → Compressor → Generator → Answer
```

**Tests / Verification (explicit route verification):**
- Take a known simple-proxy query from dev fixtures:
  - Verify router returns SIMPLE.
  - Verify retrieval_k = K_simple (2).
  - Verify compression_applied = false.
  - Verify answer is non-empty.
- Take a known complex-proxy query from dev fixtures:
  - Verify router returns COMPLEX.
  - Verify retrieval_k = K_complex (10).
  - Verify compression_applied = true.
  - Verify compressed_tokens < original_tokens.
  - Verify answer is non-empty.
- Verify all ExecutionMetadata fields populated for both paths.

**Definition of Done:**
- [ ] Both SIMPLE and COMPLEX routes execute correctly.
- [ ] K selection matches routing decision.
- [ ] Compression gate matches routing decision.
- [ ] All metadata fields populated.
- [ ] Integration tests for both routes pass.

**Deliverable:** Baseline D (AdaptiveRAG) functional end-to-end.

---

### Checkpoint F — AdaptiveRAG Ready

> **Gate:** M10.2 complete. All four baselines (A, B, C, D) are functional. All pipeline paths are proven. Ready for controlled benchmarking.

---

---

# Phase 11 — Experimental Fairness Checkpoint

> **Goal:** Verify that all four baselines run under identical conditions before interpreting any results.

---

### M11.1 — Fairness Verification

**Objective:** Run all four baselines on the same held-out test examples and verify identical conditions.

**Prerequisites:** M4.1, M5.1, M7.1, M10.2, M8.5.

**Verification checklist (from `architecture.md §14.2`):**

- [ ] Same query set (identical test split examples for all baselines).
- [ ] Same dataset split (test, not validation).
- [ ] Same corpus.
- [ ] Same chunking parameters.
- [ ] Same embedding model.
- [ ] Same vector index.
- [ ] Same generator model.
- [ ] Same prompt template.
- [ ] Same generation parameters (temperature, max_new_tokens, seed).
- [ ] Same hardware.
- [ ] Same evaluation scripts.
- [ ] Validation data was used for any threshold tuning; test data is fresh.
- [ ] Latency methodology is consistent (cold-cache vs warm-cache documented).
- [ ] Failures handled consistently across all baselines.

**Definition of Done:**
- [ ] All checklist items verified programmatically where possible.
- [ ] Any discrepancy is fixed before proceeding to primary benchmark.

**Deliverable:** Signed-off fairness checklist. Benchmarking can begin.

---

### Checkpoint G — Benchmark Ready

> **Gate:** M11.1 complete. Fairness is verified. Results from this point forward are interpretable.

---

---

# Phase 12 — Primary Benchmark

> **Goal:** Run all four baselines on the full test set and produce comparable results.

---

### M12.1 — Full Benchmark Run

**Objective:** Execute all four baselines on the test split (~1,000 PopQA + ~1,000 HotpotQA examples or configured sample_limit).

**Prerequisites:** M11.1.

**Expected files:**
- `results/baseline_a/` — LLM Only results
- `results/baseline_b/` — Fixed RAG results
- `results/baseline_c/` — Always-Compress results
- `results/baseline_d/` — AdaptiveRAG results
- `results/summary.csv` — comparative table

**Each result directory must contain:**
```text
config.yaml
metrics.json
predictions.jsonl
latency.csv
routing.csv (for adaptive baseline)
```

**Expected behavior:**
- All four baselines run to completion.
- Per-system: EM, F1, Recall@K (where applicable), router metrics (for AdaptiveRAG), avg/p95 latency, prompt tokens, compression ratio.

**Definition of Done:**
- [ ] All four baselines produce result artifacts.
- [ ] No baseline uses validation data.
- [ ] Results are machine-readable.
- [ ] Experiment config recorded alongside results.
- [ ] Summary table generated.

**Deliverable:** Primary benchmark results. First comparative numbers available.

---

---

# Phase 13 — Ablations

> **Goal:** Run the required ablation studies to isolate the contribution of individual components.

---

### M13.1 — Ablation A: Compression Value

**Objective:** Isolate the effect of compression by holding K constant.

**Prerequisite:** M12.1.

**Experiment design:**

| Condition | Retrieval K | Compression |
|---|---|---|
| A1 | 10 | No compression |
| A2 | 10 | Compression |

This directly isolates whether compression helps or hurts when retrieval depth is identical.

**Definition of Done:**
- [ ] Both conditions run on the same test examples.
- [ ] Results saved as `results/ablation_compression/`.
- [ ] Difference in EM, F1, tokens, latency computed.

---

### M13.2 — Ablation B: Adaptive K Value

**Objective:** Isolate the effect of adaptive retrieval depth vs fixed retrieval depth.

**Experiment design:**

| Condition | Retrieval K |
|---|---|
| B1 | Fixed K=5 (no routing) |
| B2 | Adaptive K (K=2 or K=10 via router) |

No compression in either condition, to isolate K effect.

**Definition of Done:**
- [ ] Both conditions run on the same test examples.
- [ ] Results saved as `results/ablation_adaptive_k/`.
- [ ] Difference in EM, F1, latency computed.

---

### M13.3 — Optional Ablation C: Router Strategy

**Objective:** Compare learned router vs a simple heuristic router (e.g., query length threshold).

**Note:** This is optional. Only implement if bandwidth allows.

**Definition of Done (if pursued):**
- [ ] Heuristic router implemented.
- [ ] Same test examples used.
- [ ] Results compared.

**Deliverable:** Ablation results quantifying the contribution of compression and adaptive K independently.

---

---

# Phase 14 — Error Analysis

> **Goal:** Understand when and why the system fails, not just aggregate metrics.

---

### M14.1 — Error Analysis Suite

**Objective:** Inspect representative failure examples across all failure categories.

**Prerequisites:** M12.1.

**Failure categories (from `prod-spec.md`):**

| Category | Examples to Inspect |
|---|---|
| Routing failures (false SIMPLE) | Complex query routed as SIMPLE → low K insufficient |
| Routing failures (false COMPLEX) | Simple query routed as COMPLEX → unnecessary overhead |
| Retrieval failures | Required evidence not retrieved |
| Compression failures | Evidence present before compression, absent after |
| Generation failures | Evidence retrieved but answer still wrong |
| Efficiency failures | Compression saved tokens but increased total latency |
| Adaptation helps | Examples where AdaptiveRAG clearly outperforms Fixed RAG |
| Adaptation hurts | Examples where AdaptiveRAG underperforms Fixed RAG |

**For each category:**
- Identify ≥5 representative examples.
- Log query, routing decision, retrieved chunks, (compressed) context, generated answer, reference answer, failure reason.

**Definition of Done:**
- [ ] All 8 failure categories have at least 5 examples inspected.
- [ ] Findings documented in `results/error_analysis/`.
- [ ] Patterns (if any) recorded for research interpretation.

**Deliverable:** Error analysis report enabling honest research interpretation.

---

---

# Phase 15 — Reproducibility Checkpoint

> **Goal:** Verify that experiments can be rerun from configuration alone and produce equivalent results.

---

### M15.1 — Reproducibility Verification

**Objective:** Run at least one baseline from scratch in a clean environment and verify results match.

**Prerequisites:** M12.1.

**Verification steps:**

1. Fresh virtual environment.
2. Install from `requirements.txt` (pinned versions).
3. Download models using config identifiers.
4. Rebuild index using `scripts/build_index.py`.
5. Run Fixed RAG benchmark using same config.
6. Compare EM and F1 against original run.

**Definition of Done:**
- [ ] Results match within expected floating-point tolerance.
- [ ] Model identifiers recorded in experiment metadata.
- [ ] Dataset splits versioned and recorded.
- [ ] Random seeds recorded.
- [ ] Hardware recorded in experiment metadata.
- [ ] `git commit` hash captured (or equivalent).

**Deliverable:** Reproducibility verified; experiment is auditable.

---

---

# Phase 16 — MVP Hard Stop ✋

### MVP Acceptance Criteria Checklist

The MVP is complete when **ALL** of the following are true:

**Core Pipeline:**
- [ ] Query input works.
- [ ] Query preprocessing works.
- [ ] Complexity router works (trained and evaluated).
- [ ] SIMPLE and COMPLEX routes work.
- [ ] Retrieval K changes according to route.
- [ ] SIMPLE route bypasses compression.
- [ ] COMPLEX route invokes compression.
- [ ] Fixed generator produces answers.

**Baselines:**
- [ ] Baseline A (LLM Only) works.
- [ ] Baseline B (Fixed RAG) works.
- [ ] Baseline C (Always-Compress) works.
- [ ] Baseline D (AdaptiveRAG) works.

**Evaluation:**
- [ ] Same evaluation set used across all baselines.
- [ ] EM and F1 measured.
- [ ] Retrieval quality measured where applicable.
- [ ] Router metrics measured.
- [ ] Prompt/context tokens measured.
- [ ] Compression ratio measured.
- [ ] End-to-end latency measured.
- [ ] Stage-level latency measured.

**Research:**
- [ ] At least one meaningful ablation complete (Ablation A).
- [ ] Router errors can be inspected.
- [ ] Compression errors can be inspected.
- [ ] Representative failure cases analyzed.
- [ ] Results are comparable across baselines.

**Engineering:**
- [ ] Core logic has tests.
- [ ] Configuration is externalized.
- [ ] Experiments produce machine-readable artifacts.
- [ ] Model/dataset/config are recorded per experiment.
- [ ] At least one experiment is reproducible from config.

---

> **Once this checklist is complete: STOP adding features.**  
> Do not add GraphRAG, agents, multimodal RAG, web search, dynamic chunking, generator fine-tuning, complex UI, or distributed infrastructure.  
> The next stage is research interpretation, documentation, and writing — not new features.

---

---

# Phase 17 — Final Integration Validation

> **Goal:** Complete end-to-end system verification before research packaging.

---

### M17.1 — Final System Verification

**Objective:** Run a complete end-to-end test of the full AdaptiveRAG pipeline covering all paths, failure modes, and edge cases.

**Prerequisites:** M16 checklist complete.

**Test cases to run:**

| Test Case | Expected Outcome |
|---|---|
| Simple-proxy query, SIMPLE route | K=2, compression=false, answer non-empty |
| Complex-proxy query, COMPLEX route | K=10, compression=true, compressed_tokens < original |
| Router failure simulation | Fallback route applied, logged, pipeline continues |
| Empty retrieval result simulation | Structured failure, logged, no fabricated evidence |
| Compression failure simulation | Original context preserved, failure logged, pipeline continues |
| Invalid query | Rejected with structured error at preprocessor |

**Definition of Done:**
- [ ] All test cases pass.
- [ ] All metadata fields populated.
- [ ] All failure paths handled correctly.
- [ ] Result artifacts produced.

**Deliverable:** Fully verified end-to-end system.

---

---

# Phase 18 — Demo (Optional, Post-MVP)

> Only after the research pipeline is stable and the MVP checklist is complete.

---

### M18.1 — Lightweight Interactive Demo

**Objective:** Simple script or minimal interface that accepts a query and shows the pipeline decision trace.

**Prerequisites:** M16 MVP complete.

**Expected output for a given query:**

```text
Query:               <user query>
Predicted Complexity: SIMPLE / COMPLEX
Retrieval K:         2 / 10
Compression Applied: Yes / No
Original Tokens:     <N>
Compressed Tokens:   <N> (if applicable)
Answer:              <generated answer>
Router Latency:      <N> ms
Retrieval Latency:   <N> ms
Compression Latency: <N> ms (if applicable)
Generation Latency:  <N> ms
Total Latency:       <N> ms
```

**Constraint:** Demo must NOT delay or block the research pipeline. It is a thin wrapper over the existing pipeline.

**Definition of Done:**
- [ ] Demo runs from CLI.
- [ ] Output includes all fields above.
- [ ] No new pipeline logic introduced.

**Deliverable:** A simple CLI demo for presentations or paper figures.

---

### Checkpoint I — Final Research Package Ready

> **Gate:** M17.1 complete (and optionally M18.1). All results are available, reproducible, and ready for research interpretation and writing.

---

---

## Summary: Development Sequence

```text
Phase 0:  Foundation (repo, config, logging, fixtures)
          ↓
Phase 1:  Data pipeline + vector index
          ↓
Phase 2:  Generator interface
          ↓
Phase 3:  ⭐ First Vertical Slice (query → retrieve → generate)
          ↓
Phase 4:  Baseline A (LLM Only)
          ↓
Phase 5:  Baseline B (Fixed RAG)
          ↓ (parallel: Phase 8 — Evaluation Infrastructure)
Phase 6:  Context Compression (LLMLingua)
          ↓
Phase 7:  Baseline C (Always-Compress RAG)
          ↓
Phase 8:  Evaluation Infrastructure (metrics, benchmark runner)
          ↓
Phase 9:  Complexity Router (data prep → train → evaluate)
          ↓
Phase 10: Adaptive Controller + AdaptiveRAG
          ↓
Phase 11: Experimental Fairness Checkpoint
          ↓
Phase 12: Primary Benchmark (all 4 baselines, test set)
          ↓
Phase 13: Ablations
          ↓
Phase 14: Error Analysis
          ↓
Phase 15: Reproducibility Checkpoint
          ↓
Phase 16: ✋ MVP Hard Stop
          ↓
Phase 17: Final Integration Validation
          ↓
Phase 18: Demo (optional)
```

---

## ADR Dependency Summary

| Decision | Milestone Blocked | Status |
|---|---|---|
| **ADR-001** Generator model | M2.1 | Open — must confirm before Phase 2 |
| **ADR-002** Router model | M9.1 | Open — must confirm before Phase 9 |
| **ADR-003** Embedding model | M1.4 | Open — must confirm before Phase 1 |
| **ADR-004** Vector store | M1.5 | Open — must confirm before Phase 1 |
| **ADR-005** K values | M0.2, M10.1 | Use defaults; validate on val set |
| **ADR-006** Compression budget | M6.1 | Use initial default; validate on val set |
| **ADR-007** Router threshold | M9.4 | Determine after router validation |
| **ADR-008** External inference | M2.1 | Only if local generation infeasible |

---

## Risks and Mitigations

| Risk | Phase | Detection | Mitigation |
|---|---|---|---|
| Generator OOM | 2 | M2.1 load test | Quantize model; switch to lighter alternative (ADR-001) |
| Embedding model OOM | 1 | M1.4 load test | Switch to lighter embedding model (ADR-003) |
| ChromaDB indexing slow | 1 | M1.5 time log | Evaluate FAISS (ADR-004) |
| Compressor incompatible | 6 | M6.1 integration test | Check LLMLingua version; pin dependency |
| Router learns dataset identity | 9 | M9.4 error analysis | Mix datasets in training; inspect error distribution |
| Test leakage | 1 | M1.2 leakage check | Enforce split separation; check in M9.1 |
| Latency inconsistency | 12 | M11.1 fairness check | Document cache state; run under same conditions |
| Compression degrades quality | 13 | M13.1 ablation | Report result honestly; discuss in error analysis |
| Generator quality insufficient | 12 | M12.1 results review | Consider alternative generator (ADR-001 update) |
