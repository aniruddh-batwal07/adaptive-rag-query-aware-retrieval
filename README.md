# AdaptiveRAG

A system-level empirical investigation of query-complexity-aware retrieval and conditional context compression.

## Overview

Retrieval-Augmented Generation (RAG) systems typically apply a fixed retrieval and context formulation policy to every query. This uniform approach can retrieve excessive context for simple factual questions or insufficient evidence for complex multi-hop queries, leading to token bloat and processing inefficiencies.

AdaptiveRAG investigates an adaptive policy where a lightweight query-complexity classifier determines the execution path:

- **SIMPLE Branch:** Retrieves $K = 2$ chunks and skips compression.
- **COMPLEX Branch:** Retrieves $K = 10$ chunks and applies chunk-wise context compression via LLMLingua-2.

*Note:* This project is an empirical system-level evaluation of coupling adaptive retrieval with selective compression under controlled execution. It does not claim individual architectural novelty for DistilBERT, BGE, ChromaDB, or LLMLingua-2.

## Research Question

Can a lightweight pre-retrieval complexity classifier efficiently optimize the RAG pipeline by adaptively varying retrieval depth and selectively triggering context compression, and under what conditions does this yield a net latency improvement?

## Architecture

```text
User Query
   |
   v
Preprocessing
   |
   v
Complexity Router (DistilBERT)
   |
   +------> [SIMPLE]  (K=2, No Compression)
   |
   +------> [COMPLEX] (K=10, LLMLingua-2 Compression)
   |
   v
BGE Embeddings / ChromaDB Retrieval
   |
   v
Generator (SmolLM-135M-Instruct)
   |
   v
Final Answer + Evaluation Telemetry
```

## Components / Technology Stack

- **Implementation:** Python
- **Complexity Router:** Fine-tuned `distilbert-base-uncased` (and `bert-base-uncased` for comparison)
- **Embeddings:** `BAAI/bge-small-en-v1.5`
- **Vector Database:** ChromaDB
- **Context Compressor:** `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank` (applied chunk-wise due to 512-token constraints)
- **Generator:** `HuggingFaceTB/SmolLM-135M-Instruct`
- **Execution Environment:** CPU-only local execution

## Datasets

The project uses subsets of PopQA (factual/simple proxy) and HotpotQA (multi-hop/complex proxy) to establish complexity labels:

- **160 training samples** for fine-tuning the router.
- **40 development samples** for validation during training.
- **298-sample router evaluation artifact** for robust standalone classifier evaluation.
- **10-query held-out end-to-end benchmark** (5 PopQA + 5 HotpotQA) strictly reserved for primary system evaluation.

## Baselines

The AdaptiveRAG policy is evaluated against three fixed conditions:
1. **LLM-only:** Generation without any retrieved evidence.
2. **Fixed RAG:** Static retrieval at $K = 5$ without compression.
3. **Always-Compress:** Static retrieval at $K = 10$ with forced LLMLingua-2 compression on all queries.

## Key Results

### Classifier Comparison (298-sample robust evaluation)
The system evaluated DistilBERT against BERT to select the optimal router. DistilBERT was selected for the primary benchmark due to its superior complex-query recall and much lower inference overhead.

**DistilBERT:**
- Accuracy: 99.33%
- Precision: 99.35%
- Recall: 99.35%
- F1: 99.35%
- Inference Latency: 12.42 ms

**BERT:**
- Accuracy: 97.32%
- Precision: 100.00%
- Recall: 94.84%
- F1: 97.35%
- Inference Latency: 22.30 ms

### Primary Benchmark (10-query held-out evaluation)
- **LLM-only:** 11.877 s
- **Fixed RAG:** 18.472 s
- **Always-Compress:** 19.510 s
- **AdaptiveRAG:** 16.514 s

### Compression Ablation ($K=10$)
- **Token Reduction:** 51.46% average reduction in context length
- **Compression Overhead:** 13.488 s average penalty
- **Generation Savings:** 21.419 s average savings
- **Net Latency Reduction:** 8.806 s average improvement under the tested CPU constraints.

*Important:* Answer-quality superiority across the retrieval strategies is **not established**. The primary benchmark recorded Exact Match (EM) = 0.0 across all retrieval-augmented conditions, as the weak SmolLM-135M generator could not reliably synthesize the retrieved text into correct factual answers.

## Classifier Downstream Effect
While the 298-sample evaluation demonstrated clear classifier differences (e.g., BERT suffering 8 false-negative COMPLEX queries), evaluating the full pipeline with the BERT router on the small 10-query benchmark yielded identical routing decisions to DistilBERT. This indicates the 10-query set is too small to reliably expose routing edge cases downstream.

## Repository Structure
```
AdaptiveRAG/
├── configs/            # YAML configuration for models, paths, and hyperparameters
├── datasets/           # Data loaders, indexing scripts, and PopQA/HotpotQA splits
├── docs/               # System architecture and design documentation
├── paper/              # Final IEEE conference paper (LaTeX source and figures)
├── results/            # Primary benchmark outputs, comparisons, and ablation artifacts
├── scripts/            # Executable workflows (training, evaluation, benchmarking)
├── src/                # Core pipeline logic, evaluation metrics, and model wrappers
└── tests/              # Pytest suite verifying component behavior
```

## Reproducibility
The final repository contains the recorded benchmarks, ablation metrics, and classifier comparisons in the `results/` directory, specifically under `results/primary_benchmark/`, `results/m12/`, and `results/router_comparison/`.

To run evaluations or examine pipeline logic, reference the CLI scripts in `scripts/`. Note that full end-to-end execution requires the downloaded local HuggingFace weights and a populated ChromaDB index (not checked into source control).

## Running the Project
The primary execution entry points are:
```bash
# Evaluate the standalone trained router against the evaluation set
python scripts/evaluate_router.py

# Run the downstream classifier comparison script
python scripts/compare_routers.py

# Run the 10-query pipeline benchmark
python scripts/run_primary_benchmark.py

# Execute the compression ablation study
python scripts/run_m12_ablation.py
```

## Testing
The `src` components are verified using `pytest`. The current suite confirms the modularity of the pipeline, data loaders, preprocessor logic, and the adaptive routing controller (23 test modules passing).

```bash
pytest tests/
```

## Limitations
- **10-Query Benchmark:** Functionally demonstrates the pipeline but is insufficient for statistical significance.
- **Proxy Complexity Labels:** Training labels based on PopQA/HotpotQA membership may promote shortcut learning rather than genuine semantic complexity awareness.
- **Generator Bottleneck:** SmolLM-135M's limited reasoning capacity prevents meaningful conclusions regarding factual answer quality.
- **CPU Execution:** Heavily inflates generation times, producing a favorable compression economy that may not generalize to GPU environments.
- **Sequence-Length Constraints:** LLMLingua-2's 512-token limit necessitates independent chunk-wise compression, scaling the compression overhead linearly with $K$.

## Research Artifacts
- **Primary Benchmark:** `results/primary_benchmark/`
- **Compression Ablation:** `results/m12/`
- **Router Comparison:** `results/router_comparison/`
- **Literature Survey:** `results/literature_comparison.csv`
- **IEEE Paper Source:** `paper/`
