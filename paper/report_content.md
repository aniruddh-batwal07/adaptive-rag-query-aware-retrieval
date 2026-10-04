# Adaptive Retrieval-Augmented Generation Using Query Complexity and Context Optimization

*Note: This is the markdown equivalent of the LaTeX paper source.*

## Abstract
Retrieval-Augmented Generation (RAG) systems typically apply a fixed retrieval policy to every query...

## 1. Introduction
Retrieval-Augmented Generation (RAG) has emerged as a foundational paradigm...

## 2. Background and Related Work
**RAG.** Lewis et al. introduced RAG...
**Literature Comparison.** We added a comprehensive literature comparison...

## 3. Problem Formulation
Let $q$ denote the input query. The complexity router classifies...

## 4. Proposed AdaptiveRAG Method
**System Architecture:** The architecture uses a direct branching strategy (SIMPLE vs COMPLEX) instead of a monolithic Adaptive Controller...
**Query Complexity Router:** DistilBERT vs BERT comparison...

## 5. Experimental Setup
**Datasets and Splits:** 160 training samples, 40 development samples, 298-sample router evaluation artifact, and a 10-query held-out end-to-end benchmark.

## 6. Experimental Results
**Classifier Comparison:** DistilBERT outperformed BERT in accuracy and recall...
**Primary Benchmark:** All strategies achieved EM=0.0 due to the weak SmolLM-135M generator.
**Downstream Classifier Effect:** Both models made identical routing decisions on the 10-query test set, indicating the set is too small to capture edge cases.

## 7. Compression Ablation Study
Compression reduced context tokens by 51.5% and yielded a net latency improvement of 8.8s...

## 8. Error Analysis and Limitations
- Generator quality bottleneck
- Small 10-query benchmark
- Shortcut learning from proxy labels

## 9. Conclusion
Adaptive routing is architecturally feasible. DistilBERT is the superior router for this task...
