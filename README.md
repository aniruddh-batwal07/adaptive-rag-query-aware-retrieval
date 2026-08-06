# AdaptiveRAG

<div align="center">
  <!-- Placeholder for project banner or architecture image -->
  <img src="docs/assets/banner_placeholder.png" alt="AdaptiveRAG Banner" width="800">
  <br>
  <strong>Adaptive Retrieval-Augmented Generation using Query Complexity and Context Optimization</strong>
</div>
<br>

<div align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-F9AB00.svg" alt="HuggingFace"></a>
</div>

---

## Project Description

**Motivation:** As Large Language Models (LLMs) scale, inference cost and latency have become primary bottlenecks in production environments. Retrieval-Augmented Generation (RAG) mitigates factual hallucinations, but its computational overhead remains heavily dependent on the volume of context processed.

**Limitations of Conventional RAG:** Standard RAG pipelines apply a static retrieval strategy (e.g., always retrieving the top $k=5$ chunks) regardless of query characteristics. For simple factual queries, this wastes compute and spikes token costs. For complex, multi-hop queries, injecting massive amounts of uncompressed text dilutes the model's attention—a limitation widely recognized as the "lost in the middle" phenomenon.

**Proposed Solution:** AdaptiveRAG introduces a dynamically routed framework. By integrating a sub-50ms query complexity classifier prior to retrieval, the system accurately estimates the cognitive load required to answer a prompt. It routes simple queries through a minimal-latency pipeline and reserves heavy compute—including deep retrieval and active context compression—for complex, multi-hop reasoning tasks. This targeted compute allocation establishes an optimal frontier between answer quality, latency, and token efficiency.

---

## Key Features

*   **Query Complexity Estimation:** A lightweight pre-retrieval sequence classifier to determine input difficulty.
*   **Adaptive Retrieval Controller:** A conditional router that dictates the downstream RAG pipeline based on query classification.
*   **Dynamic Top-K Retrieval:** Adjusts the retrieval depth (e.g., $k=2$ vs. $k=10$) relative to the query's information needs.
*   **Context Optimization & Conditional Compression:** Uses perplexity-based token pruning to distill redundant information out of large document sets, applied exclusively when mathematically viable.
*   **Retrieval-Augmented Generation:** Grounded answer synthesis using small, highly capable instruction-tuned LLMs.
*   **Comprehensive Evaluation Pipeline:** Built-in benchmarking for end-to-end telemetry, accuracy, and latency tracking.

---

## System Architecture

<div align="center">
  <!-- Placeholder for system architecture diagram -->
  <img src="docs/assets/architecture_diagram.png" alt="System Architecture Diagram" width="800">
</div>
<br>

**Pipeline Overview:**

1.  **Ingestion:** A user query enters the system and is parsed by the Query Analysis Module.
2.  **Classification:** The Query Complexity Estimator assigns a binary route (`Simple` or `Complex`).
3.  **Retrieval:** The Adaptive Retrieval Controller fetches dense embeddings from the vector database. Simple queries fetch a small context window; complex queries fetch a broad context pool.
4.  **Optimization:** If routed to the complex path, the Context Optimization Module distills the retrieved chunks, removing high-entropy tokens to maximize signal density.
5.  **Generation:** The final synthesized context is passed to the LLM to generate the final response.

---

## Repository Structure

```text
adaptive-rag/
├── data/                  # Ignored by git; place raw JSON/CSV datasets here
├── docs/                  # Architecture diagrams and supplementary documentation
├── src/                   # Main source code
│   ├── config/            # Hyperparameters and path configurations
│   ├── modules/
│   │   ├── router.py      # Query Complexity Estimator (Classifier)
│   │   ├── retriever.py   # Embedding and Vector DB interaction
│   │   ├── optimizer.py   # Context Optimization (LLMLingua wrapper)
│   │   └── generator.py   # LLM inference engine
│   ├── pipeline.py        # Core AdaptiveRAG orchestrator
│   └── utils.py           # Logging, token counting, and I/O helpers
├── eval/                  # Evaluation and benchmarking scripts
│   └── run_metrics.py     # Calculates EM, F1, Latency, Recall, etc.
├── notebooks/             # Jupyter notebooks for EDA and prototyping
├── tests/                 # Unit tests for individual pipeline components
├── requirements.txt       # Pinned dependency list
└── README.md
```

---

## Technology Stack

| Category              | Technologies Used                                      |
| --------------------- | ------------------------------------------------------ |
| Core Frameworks       | Python, PyTorch, HuggingFace Transformers              |
| Orchestration         | LangChain                                              |
| Retrieval & DB        | ChromaDB, Sentence Transformers                        |
| Context Compression   | LLMLingua                                              |
| Language Models       | Phi-3 (Generator), scikit-learn / DeBERTa (Router)     |

---

## Datasets

The framework is benchmarked against two distinct dataset distributions to validate the routing mechanism:

| Dataset   | Type       | Purpose in this Research                                                                                                                                                            |
| --------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PopQA     | Single-hop | Serves as the proxy for simple, factual queries. Used to validate that the pipeline can reduce latency and token usage by bypassing unnecessary context optimization.                |
| HotpotQA  | Multi-hop  | Serves as the proxy for complex reasoning queries. Used to validate that expansive retrieval paired with context compression improves answer quality without overflowing the context window. |

---

## Experimental Plan

### Baselines for Comparison

*   **Naive RAG:** Fixed $k=5$ retrieval, no context optimization.
*   **Always-Compress RAG:** Fixed $k=10$ retrieval, forces all queries through the context optimization module.

### Evaluation Metrics

**Answer Quality:**

*   **Exact Match (EM):** String matching against ground truth.
*   **F1-Score:** Harmonic mean of precision and recall for generated tokens.

**Retrieval Performance:**

*   **Recall@K:** Measures if the ground-truth context is within the retrieved chunks.
*   **MRR (Mean Reciprocal Rank):** Evaluates the ranking quality of the vector search.
*   **nDCG (Normalized Discounted Cumulative Gain):** Measures ranking quality considering relevance grading.

**Efficiency & Telemetry:**

*   **End-to-End Latency:** Measured in milliseconds from query ingestion to the first generated token.
*   **Token Usage:** Total prompt tokens sent to the generator API/GPU.
*   **Compression Ratio:** The percentage of tokens reduced by the Context Optimizer.

---

## Installation

**1. Clone the repository:**

```bash
git clone https://github.com/your-username/adaptive-rag.git
cd adaptive-rag
```

**2. Create and activate a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## Running the Project

> **Note:** Implementation is currently in progress. The following commands outline the intended usage pattern.

**Run a single query through the pipeline:**

```bash
python src/pipeline.py --query "Who wrote the novel 1984?"
```

**Run the full evaluation benchmark:**

```bash
python eval/run_metrics.py --dataset data/hotpotqa_subset.json --baseline adaptive
```

---

## Current Project Status

- [x] **Research & Proposal:** Problem formulation and literature survey complete.
- [x] **Architecture Design:** System pipeline and module responsibilities defined.
- [ ] **Data Pipeline Setup:** Ingestion scripts for PopQA and HotpotQA.
- [ ] **Router Implementation:** Train/prompt the query complexity estimator.
- [ ] **Retriever & DB Integration:** Setup ChromaDB and Sentence Transformers.
- [ ] **Context Optimizer Integration:** Incorporate LLMLingua.
- [ ] **Generator Integration:** Connect the Phi-3 inference engine.
- [ ] **Evaluation:** Run benchmarks against Naive and Always-Compress baselines.
- [ ] **Documentation:** Finalize report and publish results.

---

## Project Roadmap

<!-- Add milestones and timeline here as the project progresses -->

---

## Team Members

*   **[Name Placeholder 1]** - [Role/Focus Placeholder]
*   **[Name Placeholder 2]** - [Role/Focus Placeholder]

---

## References

This project builds upon foundational Retrieval-Augmented Generation literature and is highly inspired by recent 2023–2024 studies on dynamic RAG routing (e.g., Adaptive-RAG), token-level perplexity pruning (e.g., LongLLMLingua), and active retrieval mechanisms.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
