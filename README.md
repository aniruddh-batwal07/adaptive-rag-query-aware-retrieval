# AdaptiveRAG: Adaptive Retrieval-Augmented Generation

AdaptiveRAG is an implementation of a retrieval-augmented generation (RAG) system that uses query complexity classification to adaptively route queries. Instead of applying a fixed retrieval depth and full context payload to every query, it dynamically chooses between a lightweight path for simple queries and a deeper, context-compressed path for complex queries.

## Research Question

Can we reduce the computational cost and context bloat of RAG systems by dynamically routing queries based on complexity, retrieving more documents only when necessary, and compressing the retrieved context to preserve only the most relevant information?

## Architecture Overview

The system processes queries through a preprocessing step and then classifies them using a complexity router. An adaptive controller uses this classification to decide the retrieval depth (K) and whether to apply context compression.

### Main Components
- **Query Preprocessor**: Normalizes the query.
- **Complexity Router**: A binary classifier that predicts whether a query is `SIMPLE` or `COMPLEX`.
- **Adaptive Controller**: Routes the query based on the prediction.
- **Retriever**: Uses ChromaDB to fetch relevant chunks based on the routed depth (K).
- **Context Compressor**: LLMLingua-2 compresses the context for complex queries.
- **Generator**: SmolLM-135M generates the final answer from the prompt.

## Models Used
- **Generator**: `HuggingFaceTB/SmolLM-135M-Instruct`
- **Embeddings**: `BAAI/bge-small-en-v1.5`
- **Compressor**: `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`
- **Router**: `distilbert-base-uncased`
- **Vector Store**: ChromaDB

## Adaptive Policy
- **SIMPLE Path**: K = 2, No compression
- **COMPLEX Path**: K = 10, Context compression applied

## Evaluated Baselines
1. **LLM-only**: Generator without any retrieval.
2. **Fixed RAG**: K = 5, no compression.
3. **Always-compress**: K = 10, compression always applied.
4. **AdaptiveRAG**: Adaptive policy based on router classification.

## CLI Usage

Run a specific baseline using the provided CLI:

```bash
python scripts/run_pipeline.py --baseline adaptive --query "What is the capital of France?"
```

## Evaluation Setup

The system was evaluated on a held-out test subset consisting of 10 queries (5 PopQA + 5 HotpotQA).

**Metrics tracked:**
- Exact Match (EM)
- Token-level F1
- Stage-level and total latency
- Token/context metrics (retrieved, compressed, final prompt sizes)
- Compression metrics (compression ratio)

## M12 Compression Ablation

An ablation study was conducted to isolate the value of compression by comparing K=10 uncompressed vs K=10 compressed (Baseline C). This highlights the trade-offs between context size, answer quality, and latency overhead introduced by compression.

## Research Limitations

- **Dataset Size**: Evaluation was constrained to a 10-query subset for MVP validation.
- **Generator Quality**: SmolLM-135M produced very poor absolute answer-quality scores across all baselines.
- **Compressor Constraints**: LLMLingua-2 has a 512-token input limitation, requiring chunk-wise compression. This caused compression cost to scale linearly with K.
- **Scope**: Results describe this specific experimental setup and should not be taken as universal conclusions on the viability of Adaptive RAG.

## Important Results

Detailed metrics and predictions can be found in the `results/` directory:
- [Primary Benchmark Summary](results/primary_benchmark/summary.csv)
- [M12 Analysis Notes](results/m12/analysis.md)
- [M12 Ablation Aggregate](results/m12/ablation_aggregate.json)
- [M12 Adaptive Analysis](results/m12/adaptive_analysis.json)
- [M12 Reproducibility Metadata](results/m12/reproducibility.json)

