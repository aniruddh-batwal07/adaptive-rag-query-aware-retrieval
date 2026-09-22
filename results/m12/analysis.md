# Milestone 12 Analysis

## 1. Experimental Setup
This milestone performs an ablation study to isolate the effect of context compression in the AdaptiveRAG pipeline, utilizing the `datasets/splits/test_mini_10.jsonl` held-out dataset. We evaluate two conditions under a fixed `K=10` retrieval setting:
* **Condition A:** RAG WITHOUT compression (`fixed_rag` with `k=10`).
* **Condition B:** RAG WITH compression (`always_compress` with `k=10`).
All other parameters (embeddings, generator, prompt, temperature) remain identical. 

## 2. Compression Ablation
The ablation study precisely isolated the effect of the compression step by retaining `K=10` in both conditions. The resulting metrics show exactly how applying `LLMLingua-2` alters tokens and latency.

## 3. Adaptive SIMPLE vs COMPLEX behavior
Based on the existing M11 `adaptive` pipeline results, the router successfully partitioned queries into two groups:
* **SIMPLE (5 queries):** The pipeline retrieved `K=2` chunks and bypassed compression, maintaining a low overhead.
* **COMPLEX (5 queries):** The pipeline retrieved `K=10` chunks and invoked compression to handle the larger context size before generation.

## 4. Observed quality results
Across the benchmark, the measured Quality (Exact Match and Token F1) was exceptionally low for all configurations.
* **EM:** 0.0 for all queries.
* **Token F1:** Ranged between 0.0 and 0.03 across runs.
These low scores reflect issues with the generator model rather than necessarily the retrieval or compression, as the model struggled to provide concise and relevant answers.

## 5. Latency/token trade-offs
* **Token Reduction Percentage:** Compression reduced the context by 51.4%, effectively halving the number of tokens passed to the generator.
* **Compression Latency:** The overhead to perform the chunk-wise compression averaged ~13.5 seconds per query.
* **Generation Latency Savings:** By supplying a smaller context to the generator, the pipeline saved ~21.4 seconds per query during the generation phase.
* **Total Latency Difference:** The generation latency savings offset the compression latency, resulting in a net decrease in total latency of ~8.8 seconds per query.

## 6. Failure/error observations
An evidence-based review of `predictions.jsonl` reveals that the `SmolLM-135M-Instruct` generator frequently fell into repeating loops (e.g., repeating years like "1865 1885..." or repeating identical sentences). For example, in the "Leslie Knope" query, it generated the phrase "Leslie Knope Amy Poehler first episode..." repeatedly. Because of this pathological generation behavior, the predictions fail to align with the concise reference answers, driving the exact match scores to zero regardless of the quality of the retrieved context.

## 7. Correction of Compression Model Limitation
During the initial M12 ablation, a critical research-validity flaw was identified: `LLMLingua-2` (which is BERT-based) has a strict 512-token input sequence limit. Because `K=10` retrieval yields roughly 10 chunks (totaling ~1200+ tokens on average), concatenating them into a single context string caused the compressor to silently truncate and discard over half of the retrieved chunks before compression even began.
* **Invalidation:** The initial M12 ablation result was invalid. It falsely compared a full 1200-token uncompressed context against a compressed version of only the first 512 tokens.
* **Correction:** To resolve this safely without changing the model or the retrieval K, the `ContextOptimizer` and `Pipeline` were modified to preserve all 10 retrieved chunks. Instead of concatenating them prior to compression, the pipeline now passes the list of chunks directly to `LLMLingua-2`. The compressor iterates over each chunk independently (well within the 512-token limit per chunk), compresses them, and recombines them into a single optimized context.
* **Result:** The updated ablation (Condition B) is now research-valid, as all K=10 chunks are mathematically represented in the pre-compression input.

## 8. Reproducibility
A machine-readable reproducibility artifact `reproducibility.json` has been generated, logging the dataset split, sample count, random seed, model configurations, and git commit hash to ensure the experiment can be perfectly replicated.

## 9. Limitations
1. **Generator Quality:** The severe limitations of the 135M parameter generator obscure the true downstream impact of retrieval and compression on answer correctness.
2. **Small Sample Size:** The 10-query set provides a functional verification but lacks statistical power for drawing sweeping conclusions.
3. **Compression Scalability:** While chunk-wise compression resolves the 512-token sequence limit, it increases compressor work linearly as K grows.

## 10. Research interpretation
The results demonstrate the mechanical functioning of the AdaptiveRAG architecture: routing, variable `K` retrieval, and conditional compression all execute as designed. By correcting the LLMLingua sequence limit flaw via chunk-wise compression, we verified that context compression delivers downstream generation latency savings that outweigh the overhead of running the compressor. Note that the measured latency/token result is descriptive for this specific experiment and configuration, and does not represent a universal claim about all RAG setups. Furthermore, quality improvements from context optimization cannot be validated with the current generator. The architecture is sound, but the generator model requires upgrading for production use.
