# AdaptiveRAG — Product & Research Specification

> **Status:** Initial Source of Truth for Implementation  
> **Version:** 1.0  
> **Project:** AdaptiveRAG  
> **Full Title:** Adaptive Retrieval-Augmented Generation using Query Complexity and Context Optimization  
> **Repository:** https://github.com/aniruddh-batwal07/adaptive-rag-query-aware-retrieval  
> **Implementation Status:** Core implementation not yet started at the time this specification was created  
> **Primary Purpose:** Engineering specification + ML system design contract + research experiment contract

---

## 1. Document Purpose

This document is the authoritative specification for the AdaptiveRAG project.

It defines:

- the problem being solved;
- the research question and hypotheses;
- the intended system architecture;
- the behavior of every major component;
- dataset and experiment design;
- baseline systems;
- evaluation methodology;
- research-validity requirements;
- testing and reproducibility requirements;
- scope boundaries;
- acceptance criteria;
- definition of done;
- future decision rules.

- This document is intentionally more detailed than `README.md`.

### Relationship with `README.md`

`README.md` is the concise, user-facing project overview and repository entry point.

`prod-spec.md` is the engineering and research source of truth.

The intended relationship is:

```text
README.md
    |
    |-- project overview
    |-- quick explanation
    |-- setup / usage
    |-- repository introduction
    |
    v

prod-spec.md
    |
    |-- requirements
    |-- architecture
    |-- research methodology
    |-- experiment design
    |-- baselines
    |-- evaluation
    |-- acceptance criteria
    |-- scope boundaries
    |-- decisions
    |
    v

Implementation
    |
    |-- code
    |-- tests
    |-- experiments
    |-- results
```

The README created for the earlier academic DA1 submission should not be interpreted as proof that the system described there is already implemented.

The core implementation begins from this specification.

## 2. Executive Summary

AdaptiveRAG is a research-oriented Retrieval-Augmented Generation system that investigates whether retrieval and context-processing effort should be dynamically allocated according to the estimated complexity of a user query.

Conventional RAG pipelines commonly use a fixed retrieval policy:

every query
    -> retrieve fixed K
    -> pass retrieved context
    -> generate answer

This one-size-fits-all strategy may be inefficient.

A simple factual question may not need a large retrieved context, while a complex multi-hop question may require substantially more evidence.

At the same time, retrieving more context is not automatically better. Larger contexts can:

increase token usage;
increase processing cost;
increase latency;
contain irrelevant information;
make useful evidence harder for the generator to use effectively.

AdaptiveRAG therefore proposes an adaptive pipeline in which:

query
    -> estimate query complexity
    -> select retrieval depth
    -> conditionally compress retrieved context
    -> generate answer

The core initial policy is:

SIMPLE QUERY
    -> small K
    -> skip context compression
    -> generate

COMPLEX QUERY
    -> larger K
    -> compress context
    -> generate

The project does not claim that adaptive routing or context compression are individually novel.

Prior research has already explored:

Retrieval-Augmented Generation;
query/question-complexity-aware retrieval;
dynamic retrieval;
context compression;
long-context optimization.

The research contribution of AdaptiveRAG is the empirical investigation of a complexity-gated selective context compression strategy and its end-to-end quality-efficiency trade-off.

The central question is:

Can dynamically allocating retrieval and context optimization according to query complexity provide a better quality-efficiency trade-off than fixed RAG pipelines?

The project is intentionally designed as a controlled research study rather than a feature-heavy "advanced RAG" product.

## Project Goals
### Primary Goal

Build a reproducible RAG system that dynamically chooses retrieval depth and whether to apply context compression according to estimated query complexity.

### Research Goal

Determine experimentally whether adaptive retrieval and selective context compression can reduce unnecessary computation/token usage while maintaining acceptable or improved answer quality.

### Engineering Goal

Build the system as a modular, testable, configuration-driven ML application so that individual components can be replaced without redesigning the entire pipeline.

### Research Communication Goal

Produce sufficiently rigorous and reproducible results that the final work can support an undergraduate research paper or technical report, depending on the strength of the experimental findings.

The project must not assume that publication is guaranteed.

## Non-Goals

The project is deliberately constrained.

### The following are not part of the core MVP:

GraphRAG;
knowledge graphs;
multimodal RAG;
web-search RAG;
autonomous web browsing;
multi-agent RAG;
agentic planning;
RLHF;
full generator fine-tuning;
training an LLM from scratch;
training custom embedding models;
inventing a new context compression algorithm;
dynamic semantic re-chunking;
privacy-preserving RAG;
hallucination detection as a separate subsystem;
large-scale reinforcement-learning routing;
enterprise authentication;
distributed production serving;
multi-user collaboration;
elaborate web frontend;
mobile application;
Kubernetes/cloud-native deployment;
complex monitoring infrastructure;
unrelated AI capabilities.

These features may be technically interesting, but adding them would weaken the clarity of the research question and make attribution of experimental results harder.

## Research Context
### Retrieval-Augmented Generation

Retrieval-Augmented Generation provides a mechanism for conditioning generation on information retrieved from an external knowledge source.

The basic conceptual pipeline is:

Query
   |
   v
Retriever
   |
   v
Relevant Context
   |
   v
## Generator
   |
   v
Answer

AdaptiveRAG starts from this standard paradigm.

### Complexity-Aware Retrieval

Prior work such as Adaptive-RAG demonstrates that queries can be associated with different retrieval requirements and that a system can adapt its strategy according to estimated question complexity.

AdaptiveRAG builds on this concept rather than claiming to invent complexity-aware routing.

The project extends the design space by asking an additional question:

Should context compression itself be conditional on the query's complexity?

### Long Context and Context Selection

Research including Lost in the Middle motivates caution around simply increasing the amount of retrieved material passed to a language model.

More context is not always equivalent to more useful context.

This supports the project's focus on:

retrieve sufficient information
+
control the amount of context actually passed to the generator
### Context Compression

Prior work such as:

LLMLingua;
LongLLMLingua;
RECOMP;

demonstrates that retrieved/prompt context can be compressed or selectively represented to reduce token usage and potentially improve efficiency.

AdaptiveRAG does not attempt to replace these algorithms.

Instead, it asks:

When is compression worth paying for?

Compression itself has computational cost.

Therefore:

compression savings

must be evaluated against:

compression overhead

rather than assuming that compression automatically improves total latency.

## Research Gap and Contribution Framing
### Central Gap

Prior literature provides strong evidence for two separate ideas:

adapt retrieval behavior according to query/question characteristics;
compress long or retrieved contexts.

AdaptiveRAG investigates the combination of these ideas through:

Complexity-Gated Selective Context Compression

The intended decision policy is:

query complexity
      |
      +---- SIMPLE
      |       |
      |       +--> retrieve smaller K
      |       +--> bypass compressor
      |
      +---- COMPLEX
              |
              +--> retrieve larger K
              +--> apply context compression

The contribution is therefore primarily:

architectural;
experimental;
empirical.

It is not a claim of inventing a new compression algorithm.

### Contribution Statement

The intended contribution should be described approximately as:

AdaptiveRAG investigates a query-complexity-aware RAG architecture that dynamically adjusts retrieval depth and selectively applies context compression, and evaluates whether this strategy provides a better answer-quality/latency/token-efficiency trade-off than fixed RAG and always-compress RAG baselines.

This wording should be preferred over stronger claims such as:

"first adaptive RAG system";
"first system to use query complexity";
"novel compression algorithm";
"guaranteed latency reduction".

Such claims require evidence and should not appear without verification.

## Central Research Question

The primary research question is:

Does query-aware adaptive retrieval combined with selective context compression produce a better quality-efficiency trade-off than conventional fixed RAG pipelines?

A detailed experimental formulation is:

We investigate whether dynamically choosing retrieval depth and selectively applying context compression based on query complexity can maintain or improve answer quality while reducing latency and token usage compared with fixed RAG and always-compress RAG baselines.

## Secondary Research Questions
RQ1 — Query Routing

How accurately can a lightweight classifier estimate whether a query requires a relatively small or large retrieval context?

RQ2 — Retrieval Adaptation

Does adapting retrieval depth according to estimated query complexity improve the quality-efficiency trade-off compared with fixed retrieval depth?

RQ3 — Selective Compression

Does selectively applying context compression reduce context/token usage without causing unacceptable answer-quality degradation?

RQ4 — Compression Economics

When does the computational cost of compression outweigh the savings obtained from processing a smaller context?

RQ5 — End-to-End Benefit

Does the complete AdaptiveRAG pipeline outperform fixed RAG in terms of the combined quality/latency/token trade-off?

RQ6 — Failure Conditions

For what types of queries does AdaptiveRAG fail or provide no meaningful advantage?

## Research Hypotheses

These are hypotheses to test, not guarantees.

H1

Query-aware adaptive retrieval with selective context compression can achieve a better quality-efficiency trade-off than fixed RAG.

H2

Relatively simple queries can often be answered adequately with a smaller retrieval depth, reducing unnecessary context and computation.

H3

Complex multi-hop queries can benefit from a larger retrieval depth.

H4

Context compression can reduce token usage for larger retrieved contexts, but the value depends on whether compression overhead and information loss are outweighed by downstream savings.

H5

A lightweight router can provide useful routing decisions with substantially less decision-making overhead than asking a large generator model to make every routing decision.

H5 is specifically an efficiency hypothesis and must be measured rather than assumed.

## Core System Concept

The system has two main execution paths.

### Simple Path
Query
  |
  v
Complexity Router
  |
  v
SIMPLE
  |
  v
Retrieve K_simple
  |
  v
Skip Compression
  |
  v
Fixed Generator
  |
  v
Answer

Initial proposed value:

K_simple = 2

This is a configurable experimental default, not an immutable value.

### Complex Path
Query
  |
  v
Complexity Router
  |
  v
COMPLEX
  |
  v
Retrieve K_complex
  |
  v
## Context Compression
  |
  v
Fixed Generator
  |
  v
Answer

Initial proposed value:

K_complex = 10

Again, this is an initial experimental default.

## End-to-End Architecture

The conceptual architecture is:

                           +----------------------+
                           |      User Query      |
                           +----------+-----------+
                                      |
                                      v
                           +----------------------+
                           | Query Preprocessing  |
                           +----------+-----------+
                                      |
                                      v
                           +----------------------+
                           | Query Complexity     |
                           | Estimator / Router   |
                           +----------+-----------+
                                      |
                       +--------------+--------------+
                       |                             |
                    SIMPLE                        COMPLEX
                       |                             |
                       v                             v
                +-------------+               +-------------+
                | Small K     |               | Large K     |
                | Retrieval   |               | Retrieval   |
                +------+------+               +------+------+
                       |                             |
                       |                             v
                       |                    +----------------+
                       |                    | Context        |
                       |                    | Compression    |
                       |                    +-------+--------+
                       |                             |
                       +--------------+--------------+
                                      |
                                      v
                           +----------------------+
                           |   Fixed Generator    |
                           |        LLM           |
                           +----------+-----------+
                                      |
                                      v
                           +----------------------+
                           |       Answer         |
                           +----------+-----------+
                                      |
                                      v
                           +----------------------+
                           | Metrics / Telemetry  |
                           | / Experiment Logs    |
                           +----------------------+

The implementation may split this architecture into more modules.

The behavioral contract is authoritative.

## System Components

The MVP contains the following logical components:

Query Preprocessor
Query Complexity Estimator
## Adaptive Routing Controller
Embedding Model
Vector Index / Store
Retriever
Context Optimizer / Compressor
## Generator
## Pipeline Orchestrator
Evaluation System
Experiment Logger
Result Writer
## Query Preprocessing
### Purpose

Normalize the input query without changing its semantic meaning.

### Responsibilities
validate the input;
normalize whitespace;
handle malformed/empty queries;
optionally normalize trivial formatting;
preserve the original query for logging/evaluation.
### Requirements

The preprocessor must be deterministic.

The preprocessor must not:

add answers;
remove important semantic information;
rewrite the question using an LLM;
make routing decisions.
## Query Complexity Estimator / Router
### Purpose

The router determines whether the query should take the low-compute/simple route or the high-information/complex route.

Minimum output:

SIMPLE
or
COMPLEX

Optional output:

confidence / probability
### Preferred Implementation

The preferred direction is a lightweight encoder classifier.

A candidate is:

DeBERTa-v3-small

However, the specification does NOT guarantee that this exact model must remain the final implementation if practical experiments show that another lightweight classifier is better.

The router should remain replaceable.

### Important Latency Constraint

Do not encode the old DA1 proposal's "<50 ms" router statement as a guaranteed requirement.

The correct engineering requirement is:

The router should be lightweight enough that its overhead can be measured and potentially justified by downstream savings.

Actual latency must be measured experimentally.

No latency number should be considered achieved until benchmarked.

### Router Interface

Conceptually:

### Input:
    query: string

### Output:
    complexity_label: SIMPLE | COMPLEX
    confidence: optional float
    metadata: optional dictionary

A conceptual representation:

{
  "label": "COMPLEX",
  "confidence": 0.91
}

The exact internal type may be implemented using a typed Python object.

## Query Complexity Validity

This is a major research-validity concern.

The project initially uses:

PopQA     -> relatively simple/factual proxy
HotpotQA  -> relatively complex/multi-hop proxy

This is useful for creating an initial benchmark, but it is NOT sufficient evidence that complexity is a universal property of a dataset.

A naive classifier may learn:

"PopQA style" = SIMPLE
"HotpotQA style" = COMPLEX

rather than learning:

"This individual query requires less/more retrieval effort."

This is a serious threat to research validity.

### Mitigation Strategy

The implementation should, where practical:

mix datasets rather than maintaining completely isolated routing populations;
create proper train/validation/test partitions;
evaluate on held-out queries;
inspect query-level characteristics;
examine router errors;
manually inspect a subset of routing examples;
avoid using dataset identity as an explicit feature;
preserve the ability to create mixed evaluation sets.

The goal is to test whether the router captures query characteristics rather than merely dataset membership.

## Complexity Labels

The initial routing problem is binary:

SIMPLE
COMPLEX

The first implementation should NOT require:

EASY
MEDIUM
HARD
VERY HARD

or arbitrary multi-level routing.

Binary routing is sufficient for the MVP and provides a cleaner experiment.

If later evidence strongly supports multi-level routing, that should be treated as a post-MVP extension and documented as a new decision.

## Adaptive Routing Controller
### Purpose

Translate the router's prediction into a concrete pipeline execution policy.

Conceptually:

ComplexityResult
       |
       v
RoutingDecision

The routing controller should be deterministic.

It should NOT be an LLM agent.

### Initial Routing Policy
SIMPLE
retrieval_k = K_simple
compression = false
COMPLEX
retrieval_k = K_complex
compression = true

Initial experimental defaults:

K_simple  = 2
K_complex = 10
### Routing Decision Contract

Conceptually:

RoutingDecision
    complexity
    retrieval_k
    compression_enabled
    optional compression configuration

Example:

{
  "complexity": "SIMPLE",
  "retrieval_k": 2,
  "compression_enabled": false
}

or:

{
  "complexity": "COMPLEX",
  "retrieval_k": 10,
  "compression_enabled": true
}
## Retrieval System
### Retrieval Strategy

The MVP uses dense semantic retrieval.

The system should contain:

Embedding Model
       +
Vector Index / Store
       +
Top-K Retrieval

Current preferred embedding direction:

BAAI/bge-small-en-v1.5

The exact model version must be configuration-driven.

### Vector Store

The DA1 proposal selected:

ChromaDB

as the current local vector store direction.

The project should preserve this choice unless actual implementation requirements provide a strong reason to use another local index.

FAISS can be considered internally as an alternative if justified.

The specification should not allow silent architectural changes.

Any significant change should be recorded in the decision log.

### Retriever Interface

Conceptually:

retrieve(
    query,
    top_k
)
    ->
    ranked documents/chunks

Each result should retain useful metadata:

document_id
chunk_id
text
similarity/relevance score if available
source metadata
rank
## Retrieval Fairness

The retriever must remain constant across the main comparisons.

All baselines should use:

the same embedding model;
the same corpus;
the same chunking;
the same index;
the same similarity calculation;
the same retrieval implementation.

The primary experiment should change retrieval depth and compression policy, not secretly change the retriever itself.

## Dataset Strategy

The initial benchmark uses:

PopQA
HotpotQA

The datasets serve different research purposes.

### PopQA

Use as a factual/open-domain question-answering benchmark and as a practical proxy for queries that may often require relatively limited retrieval depth.

It should not be described as universally or objectively "simple."

### HotpotQA

Use as a multi-hop/multi-document question-answering benchmark.

HotpotQA is particularly useful because it contains questions requiring reasoning across supporting information.

It should not be described as universally or objectively "complex" merely because it is a multi-hop dataset.

### Initial Dataset Size

The DA1 plan proposed:

~1,000 PopQA examples
~1,000 HotpotQA examples

This is a practical starting point for development and rapid experimentation.

The final benchmark size should be configurable.

The project should initially prioritize:

clean experiments
+
fast iteration
+
reproducibility

over unnecessarily large evaluation sets.

## Data Leakage Requirements

The benchmark must avoid leakage.

Requirements:

training and test queries must be separated;
router training data must not include final test examples;
test performance must not be used to select hyperparameters;
retrieval corpus construction must be carefully documented;
reference answers must not be injected into retrieval inputs;
supporting-document information must not accidentally leak directly into query features.

Any preprocessing that can expose evaluation information must be documented and reviewed.

## Document Chunking

The primary experiments should use a fixed chunking strategy.

Potential parameters:

chunk_size
chunk_overlap

These must be configurable.

All major baselines must use the same chunking.

### Why Fixed Chunking?

If the adaptive pipeline changes:

query complexity;
retrieval K;
chunking;
compression;
generator;

at the same time, then we cannot determine why the performance changed.

Therefore the main experiment should keep chunking fixed.

## Dynamic Re-Chunking

Dynamic semantic/query-aware re-chunking is explicitly OUT OF SCOPE for the MVP.

It was considered as an interesting research direction but rejected for the initial project because it adds:

implementation complexity;
index-management complexity;
additional experimental variables;
harder attribution.

Do not reintroduce it during normal implementation.

## Context Compression
### Purpose

Reduce unnecessary context/token volume before generation when the adaptive route determines that compression is worthwhile.

### Preferred Technology

The preferred implementation is:

LLMLingua

or a compatible implementation/library version of the chosen LLMLingua approach.

The exact package/API must be verified during implementation.

### Role of Compression

The compressor is NOT itself the proposed research contribution.

The proposed contribution is:

when to invoke compression

rather than:

how to invent a better compression algorithm.
### Compression Policy

The default MVP policy is:

SIMPLE
    -> compression OFF

COMPLEX
    -> compression ON

This conditional behavior is central to the experiment.

### Compression Metadata

The optimizer should provide enough information to measure:

input token count
output token count
compression ratio
compression latency
compression status
fallback status

Conceptually:

compression_ratio =
    original_tokens / compressed_tokens

where the exact definition must be documented consistently.

## Compression Validity

Compression can cause information loss.

Therefore the project must NOT assume:

compressed context = equally useful context

It must experimentally evaluate whether compression preserves useful evidence.

A compression result that uses fewer tokens but causes major answer degradation is not automatically a success.

## Generator
### Principle

Use a fixed generator LLM for the main comparisons.

This is essential for experimental fairness.

The generator should NOT be fine-tuned during the MVP.

### Preferred Model Direction

The DA1 proposal considered:

Phi-3-mini-4k-instruct

as the preferred lightweight generator.

A practical alternative is a small Llama 3-class instruct model with suitable quantization.

The final generator must be chosen based on actual:

VRAM;
runtime;
context capacity;
inference stability;
reproducibility.

The final model identifier must be stored in configuration and experiment metadata.

### Generation Configuration

At minimum, make the following configurable:

temperature
max_new_tokens
seed where supported

The exact settings must remain identical across the main baselines.

## Core Prompting Contract

The generator receives conceptually:

Question:
    <user query>

Context:
    <retrieved or compressed context>

The prompt format should remain fixed across the main experiments.

Do not change generator prompting between baseline systems in a way that could influence the result.

## Pipeline Orchestrator

The orchestrator coordinates the full system.

Conceptually:

query
  |
  v
preprocess
  |
  v
router
  |
  v
routing controller
  |
  v
retrieve
  |
  +-----------------------+
  |                       |
simple                  complex
  |                       |
skip compression       compress
  |                       |
  +-----------+-----------+
              |
              v
          generator
              |
              v
            answer
              |
              v
         telemetry

The orchestrator should remain deterministic and modular.

It should not become an autonomous agent.

## Full Request Execution Contract

Each execution should conceptually produce:

Answer
+
Execution Metadata

Execution metadata should include enough information to understand the decision path.

Example:

{
  "query": "example query",
  "complexity": "COMPLEX",
  "router_confidence": 0.91,
  "retrieval_k": 10,
  "compression_applied": true,
  "original_context_tokens": 4200,
  "compressed_context_tokens": 1250,
  "compression_ratio": 3.36,
  "router_latency_ms": 12.4,
  "retrieval_latency_ms": 34.8,
  "compression_latency_ms": 48.2,
  "generation_latency_ms": 620.1,
  "total_latency_ms": 715.5
}

This is an informational schema, not necessarily the exact production data type.

## Required Baselines

The baseline design is one of the most important parts of the project.

The main research comparison must include at least four systems.

### Baseline A — LLM Only

Pipeline:

Query
  |
  v
## Generator
  |
  v
Answer

No retrieval.

### Purpose

Determine the value of external retrieval itself.

### Baseline B — Standard Fixed RAG

Pipeline:

Query
  |
  v
Retrieve fixed K
  |
  v
## Generator

Initial proposed:

K = 5

No compression.

### Purpose

Represent a conventional fixed RAG system.

### Baseline C — Always-Compress RAG

Pipeline:

Query
  |
  v
Retrieve K=10
  |
  v
Compress
  |
  v
## Generator

Compression is always enabled.

### Purpose

Determine whether compression itself explains observed efficiency/quality differences.

### Baseline D — AdaptiveRAG

Pipeline:

Query
  |
  v
Complexity Router
  |
  +------ SIMPLE ------> K=2 -> no compression
  |
  +------ COMPLEX -----> K=10 -> compression
  |
  v
## Generator
### Purpose

This is the proposed system.

## Optional Control Baseline

A rule-based adaptive router may be added if practical.

Example:

heuristics
    ->
simple / complex
    ->
adaptive retrieval/compression

Its purpose would be to answer:

Is the learned classifier actually necessary, or does a simple routing rule produce similar behavior?

This is useful but is NOT required for MVP completion.

## Baseline Fairness Rules

Every major baseline must use the same:

benchmark queries;
document corpus;
chunks;
embedding model;
vector index;
retrieval implementation;
generator model;
generator prompt;
generation parameters;
hardware/environment where possible;
evaluation scripts.

Only the intended experimental policy should change.

## Evaluation Framework

Evaluation must be treated as a first-class part of the system.

The project evaluates four major dimensions:

1. Answer Quality
## Retrieval Quality
## Router Quality
## Efficiency

No single metric is sufficient.

## Answer Quality Metrics

**Primary metrics:**

### Exact Match

Useful where reference-answer matching is appropriate.

### Token-Level F1

Measures overlap between generated and reference answers.

These should be primary reference-based answer metrics where supported by the datasets.

### Supplementary Evaluation

Optional:

semantic similarity;
RAGAS;
carefully designed LLM-based judging.

These should remain supplementary.

The system must not depend on an expensive LLM judge to establish every primary result.

## Retrieval Quality Metrics

Where ground-truth supporting information permits meaningful retrieval evaluation, measure:

Recall@K;
MRR;
nDCG where appropriate.

Retrieval metrics should be reported separately from generation metrics.

This allows the research team to distinguish:

bad retrieval

from:

good retrieval but poor generation
## Router Metrics

**Measure:**

Accuracy;
Precision;
Recall;
F1;
confusion matrix.

Where confidence scores are available, calibration may be analyzed later.

Router evaluation must not rely only on the overall accuracy number.

The project should inspect which classes are systematically misclassified.

## Efficiency Metrics

At minimum measure:

router latency;
retrieval latency;
compression latency;
generation latency;
total end-to-end latency;
retrieved context tokens;
compressed context tokens;
final prompt tokens;
compression ratio;
retrieval K.

Where useful, calculate:

token reduction %
latency reduction %

relative to the selected fixed baseline.

## Latency Measurement

A central research question is whether context compression actually saves total time after its own computation cost is included.

Therefore:

Total Latency
    =
    Router
    +
    Retrieval
    +
    Compression
    +
    Generation
    +
    Other Measured Overhead

The exact code-level implementation may differ, but these conceptual stages should remain measurable.

### Important Rule

Do not report only:

total latency

Also report the breakdown.

This allows analysis such as:

router adds little overhead;
compression dominates runtime;
generation savings outweigh compression;
compression saves tokens but not wall-clock time.

Any of these outcomes can be scientifically useful.

## Token Efficiency

Track:

retrieved_context_tokens
compressed_context_tokens
final_prompt_tokens

Potential derived values:

token_reduction_percent
compression_ratio

The project should investigate whether reducing context actually leads to practical improvements rather than merely reporting compression ratios in isolation.

## Quality-Efficiency Trade-Off

The project should not optimize one metric in isolation.

For example:

90% token reduction
+
major answer-quality degradation

is not an automatic success.

The intended objective is a favorable trade-off among:

### Answer Quality
Latency
Token Usage
### Retrieval Quality

The final analysis should therefore compare systems across multiple dimensions.

Where useful, present quality-efficiency trade-off plots or Pareto-style comparisons.

## Primary Research Experiment

The main experiment is:

LLM Only
vs
Standard Fixed RAG
vs
Always-Compress RAG
vs
AdaptiveRAG

The principal question is:

Does AdaptiveRAG provide a better quality-efficiency trade-off?

## Experimental Segmentation

Results should be analyzed at more than one level.

At minimum:

overall

and:

simple/proxy-simple queries
complex/proxy-complex queries

The analysis should also investigate router behavior across these groups.

This allows answers to questions such as:

Does AdaptiveRAG help mostly on simple queries?
Does it preserve complex-query quality?
Is the cost of compression justified on complex queries?
Does router misclassification create most failures?
## Required Ablation Studies

At least one strong ablation is mandatory.

### Ablation A — Compression Value

Keep retrieval depth fixed for complex queries:

K = 10

Compare:

K=10 + no compression

against:

K=10 + compression

This isolates the effect of compression.

### Ablation B — Adaptive Retrieval Depth

Compare:

fixed K

against:

complexity-dependent K

This isolates the value of adaptive retrieval depth.

### Optional Ablation C — Router Strategy

Compare:

learned router

against:

simple heuristic router

only if feasible.

## Hyperparameter Sensitivity

The project may later investigate:

K_simple;
K_complex;
compression budget;
compression ratio;
router threshold.

However, excessive parameter sweeps are discouraged during the MVP.

The project should prioritize:

clear questions
+
small number of meaningful experiments

over:

large quantity of shallow experiments
## Experimental Data Splits

Use clear data roles.

Conceptually:

TRAIN
    ->
router training / preparation

VALIDATION
    ->
thresholds
K selection
compression settings
other experimental tuning

TEST
    ->
final comparison

The test set must not be used for tuning.

Any deviation must be documented.

## Reproducibility Requirements

Every major experiment must record:

experiment name;
dataset;
split;
sample count;
model identifiers;
retrieval parameters;
router parameters;
compression parameters;
generation parameters;
random seed;
hardware;
software/dependency environment;
git commit where practical;
metrics;
artifact locations.

A researcher should be able to understand exactly which configuration produced a result.

## Experiment Metadata

Each run should have a unique identifier.

Conceptually:

experiment_id
timestamp
git_commit
config
hardware
models
dataset
split
results

This prevents results from becoming anonymous CSV files with unclear origins.

## Caching Considerations

Caching can heavily distort latency measurements.

The benchmark system should distinguish, where relevant:

cold-cache

and:

warm-cache

Do not compare:

cached AdaptiveRAG

against:

uncached baseline

and call the result scientifically valid.

Cache behavior must be documented.

## Hardware Fairness

Latency comparisons should ideally use the same hardware and environment.

Record:

CPU;
RAM;
GPU model;
VRAM;
operating environment;
framework/library versions.

Do not mix local and cloud/API latency measurements as though they represent the same deployment condition.

If external inference is later necessary, clearly label it.

## External API Fallback

If local hardware cannot run the selected generator reliably, an external inference API may be considered as a practical fallback.

However:

this is not the primary architecture;
API latency must be measured separately;
all major baselines should use the same generator/API configuration;
API/network latency must not be confused with pure model inference latency;
cost should be documented;
reproducibility should be maintained as far as possible.

The external provider should be treated as an implementation contingency rather than a research contribution.

## Compute Constraints

The project is intended for a small undergraduate team with limited compute.

Possible environments include:

consumer GPU;
NVIDIA RTX 3060-class hardware;
NVIDIA RTX 4090-class hardware;
Google Colab T4;
comparable resource-constrained environments.

The system should favor:

lightweight models;
frozen pretrained components;
quantization where appropriate;
manageable benchmark subsets;
local/reproducible execution.
## Model Training Constraints

The MVP should avoid expensive training.

The intended architecture primarily uses frozen pretrained components.

The likely trainable component is:

query complexity classifier

if a learned router is used.

Do not train:

the generator LLM;
a large custom retriever;
a new compression model.
## Router Training

If DeBERTa-v3-small or a similar encoder is selected, router training must be lightweight enough for the available environment.

The earlier DA1 document estimated approximately 2,000 samples and under one hour of T4 training.

That estimate should NOT be treated as a guaranteed engineering requirement.

Actual training time must be measured.

The training dataset size should be configurable.

## Model Selection Principles

Model choice should follow these priorities:

Research validity.
Reproducibility.
Sufficient quality.
Reasonable compute requirements.
Stability.
Ease of experimentation.

Do not choose a model merely because it is the newest model.

Model upgrades during development must be treated as experiment changes, not invisible improvements.

## Software Architecture Principles

The codebase should follow:

modularity;
separation of concerns;
typed interfaces where useful;
configuration-driven behavior;
deterministic experiment execution where practical;
explicit logging;
testability;
replaceable models;
reproducibility.

Avoid putting core research logic only inside notebooks.

Notebooks may be used for analysis and visualization, but the actual pipeline must operate through reusable Python modules/scripts.

## Target Repository Architecture

A conceptual target structure is:

adaptive-rag-query-aware-retrieval/
│
├── README.md
├── prod-spec.md
├── LICENSE
├── requirements.txt
├── pyproject.toml                  # if introduced
│
├── configs/
│   ├── config.yaml
│   └── experiments/
│
├── datasets/
│   ├── load_dataset.py
│   ├── prepare_data.py
│   └── build_index.py
│
├── src/
│   ├── router/
│   │   ├── query_classifier.py
│   │   ├── complexity_estimator.py
│   │   └── routing_logic.py
│   │
│   ├── retriever/
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   └── retriever.py
│   │
│   ├── optimizer/
│   │   └── context_optimizer.py
│   │
│   ├── generator/
│   │   └── response_generator.py
│   │
│   ├── pipeline/
│   │   └── adaptive_rag.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── benchmark.py
│   │   └── latency.py
│   │
│   └── utils/
│       ├── config.py
│       └── logger.py
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── build_index.py
│   ├── train_router.py
│   ├── run_pipeline.py
│   ├── evaluate.py
│   └── run_benchmark.py
│
├── notebooks/
│
├── tests/
│
├── results/
│
├── assets/
│
└── docs/

This structure is a logical target, not a requirement to create every file immediately.

The actual repository structure may evolve as implementation begins.

Any major architectural deviation must be documented.

## Component Contracts
### Router
Query
    ->
ComplexityResult
### Retriever
Query + K
    ->
RankedDocuments
### Controller
ComplexityResult
    ->
RoutingDecision
### Optimizer
Query + RetrievedContext
    ->
OptimizedContext
### Generator
Query + Context
    ->
GeneratedAnswer
### Pipeline
Query
    ->
Answer + ExecutionMetadata
### Evaluation
Predictions + References + RetrievalMetadata
    ->
EvaluationMetrics
## Configuration Requirements

Configuration should control experiment-sensitive values.

Potential structure:

models:
  router:
    name: ...
  embeddings:
    name: ...
  compressor:
    name: ...
  generator:
    name: ...

retrieval:
  k_simple: 2
  k_complex: 10
  baseline_k: 5
  chunk_size: ...
  chunk_overlap: ...

routing:
  threshold: ...

compression:
  enabled: true
  budget: ...

generation:
  temperature: ...
  max_new_tokens: ...
  seed: ...

evaluation:
  dataset: ...
  split: ...
  sample_limit: ...

runtime:
  device: ...
  batch_size: ...

Exact values are implementation decisions.

Avoid magic constants in source code.

## Functional Requirements
FR-001 — Query Input

The system SHALL accept a natural-language query.

FR-002 — Query Validation

The system SHALL validate the input before execution.

FR-003 — Query Complexity Estimation

The system SHALL estimate the query complexity.

FR-004 — Binary Routing

The MVP SHALL support SIMPLE and COMPLEX routes.

FR-005 — Adaptive Retrieval

The system SHALL select retrieval depth according to the route.

FR-006 — Simple-Path Compression Bypass

The SIMPLE route SHALL bypass context compression by default.

FR-007 — Complex-Path Compression

The COMPLEX route SHALL invoke context compression by default.

FR-008 — Fixed Generator

The generator SHALL remain fixed across primary experimental baselines.

FR-009 — Configurable K

Retrieval K values SHALL be configurable.

FR-010 — Retrieval Metadata

The system SHALL preserve useful retrieval metadata.

FR-011 — Compression Metadata

The system SHALL record compression statistics where compression is applied.

FR-012 — Latency Instrumentation

The system SHALL record latency for major pipeline stages.

FR-013 — Token Instrumentation

The system SHALL record context/token measurements required for efficiency analysis.

FR-014 — Baseline Support

The system SHALL support the required baseline pipelines.

FR-015 — Evaluation Support

The system SHALL support running common benchmark data through all systems.

FR-016 — Router Evaluation

The system SHALL support independent router evaluation.

FR-017 — Retrieval Evaluation

The system SHALL support retrieval evaluation where reference evidence permits it.

FR-018 — Ablations

The system SHALL support at least one meaningful ablation.

FR-019 — Result Persistence

The system SHALL save experiment results in machine-readable form.

FR-020 — Failure Logging

The system SHALL record important failures/fallbacks.

FR-021 — Reproducibility

The system SHALL support reproducible benchmark configuration.

## Non-Functional Requirements
NFR-001 — Modularity

Major pipeline components SHOULD be independently replaceable.

NFR-002 — Configurability

Experiment parameters SHOULD be externalized.

NFR-003 — Reproducibility

The same configuration SHOULD reproduce equivalent experimental conditions.

NFR-004 — Testability

Core routing and evaluation logic MUST be testable independently.

NFR-005 — Observability

Experiment execution MUST expose sufficient metadata for later analysis.

NFR-006 — Research Traceability

Results SHOULD be traceable to configuration, model versions, dataset split, and code version.

NFR-007 — Compute Awareness

The system SHOULD remain usable on the project's intended resource constraints.

NFR-008 — Controlled Scope

New features MUST NOT be introduced if they materially obscure the core experiment without explicit specification changes.

## Error Handling
### Router Failure

Fallback to a safe/default routing policy if configured.

The fallback event must be logged.

### Retrieval Failure

If retrieval produces no usable context:

handle the situation explicitly;
do not fabricate evidence;
record the failure.
### Compression Failure

If compression fails:

preserve the original context where safe;
continue if the pipeline design allows;
log the failure;
record that compression was not successfully applied.
### Generator Failure

The system should:

return a structured failure;
preserve experiment metadata;
avoid silently marking the example as successful.
### Configuration Failure

Invalid configurations should fail fast with a clear message.

## Logging Requirements

At minimum capture:

timestamp
run_id
query_id
complexity
router_confidence
retrieval_k
compression_applied
token counts
latency breakdown
answer status
fallback/error status

For experiments also capture:

dataset
split
model versions
hardware
config
git commit if practical
## Testing Strategy

The objective is to test the research-critical behavior.

### Unit Tests

Test:

query normalization;
router result validation;
routing decisions;
K selection;
compression gating;
configuration parsing;
token calculations;
compression-ratio calculations;
metric calculations.
### Integration Tests

Test:

query
    ->
router
    ->
controller
    ->
retriever

and:

query
    ->
AdaptiveRAG
    ->
generator

using a tiny local/test dataset or mocks where appropriate.

### Failure Tests

Test:

empty query;
invalid configuration;
empty retrieval results;
compression failure;
generator failure;
router failure/fallback.
### Regression Testing

Core research logic should have tests so later refactoring does not accidentally change:

SIMPLE -> low K -> no compression
COMPLEX -> high K -> compression

without the change being noticed.

## Experiment Execution Workflow

The intended development/research workflow is:

1. Prepare data
2. Build corpus
3. Chunk data
4. Build retrieval index
5. Prepare router data
6. Train/prepare router
7. Validate router
8. Implement LLM-only baseline
9. Implement fixed RAG
10. Implement always-compress RAG
11. Implement AdaptiveRAG
12. Validate end-to-end pipeline
13. Run primary benchmark
14. Run ablations
15. Perform error analysis
## Generate plots/tables
## Save reproducible experiment artifacts
## Freeze the core MVP

Do not start with frontend work.

Do not start with optional extensions.

The experiment must work before polish.

## Development Order

The preferred implementation sequence is:

Phase 1 — Data
dataset loading;
normalization;
train/validation/test construction;
corpus preparation.
Phase 2 — Retrieval
chunking;
embeddings;
vector index;
top-K retrieval.
Phase 3 — Generator
fixed generator interface;
controlled generation;
simple RAG generation.
Phase 4 — Standard Baselines
LLM-only;
fixed RAG;
always-compress RAG.
Phase 5 — Compression
LLMLingua integration;
token measurement;
compression metadata;
failure fallback.
Phase 6 — Router
dataset/label preparation;
lightweight router;
validation;
router metrics.
Phase 7 — Adaptive Controller
complexity decision;
K selection;
conditional compression.
Phase 8 — End-to-End AdaptiveRAG
connect all components;
verify execution;
verify metadata.
Phase 9 — Benchmarking
common benchmark;
latency;
tokens;
quality;
retrieval;
router metrics.
Phase 10 — Ablations
compression ablation;
adaptive-K ablation.
Phase 11 — Analysis
error analysis;
trade-off analysis;
visualization;
reproducibility verification.
Phase 12 — Optional Demo

Only after the research pipeline is stable.

## Minimum Viable Demo

A demo is optional until the core research system works.

The eventual simple interface should show:

Query
Predicted Complexity
Retrieval K
Compression Applied?
Original Context Tokens
Compressed Context Tokens
Answer
Latency

This is sufficient.

A sophisticated frontend is not part of the MVP.

## Research Validity Risks

The following risks must be actively monitored.

Risk 1 — Dataset Shortcut Learning
Problem

The router could learn dataset style rather than query complexity.

Mitigation
mixed datasets;
held-out test data;
query-level analysis;
manual inspection;
avoid explicit dataset-ID features.
Risk 2 — Compression Information Loss
Problem

Compression may remove critical evidence.

Mitigation

Compare:

raw retrieved context
vs
compressed context

using both:

retrieval metrics;
answer metrics.
Risk 3 — Compression Overhead
Problem

Compression could save generation tokens but take more time than it saves.

Mitigation

**Measure:**

compression latency
+
generation latency

separately.

Risk 4 — Router Overhead
Problem

The router may add enough latency to erase downstream savings.

Mitigation

Measure router latency.

Do not assume it is negligible.

Risk 5 — Generator Confounding
Problem

Changing the generator changes the experiment.

Mitigation

Use one fixed generator for primary comparisons.

Risk 6 — Retrieval Confounding
Problem

Changing retrieval implementation between systems would make results uninterpretable.

Mitigation

Same retriever/index/embeddings across primary baselines.

Risk 7 — Test Leakage
Problem

Using test results to select parameters produces optimistic results.

Mitigation

Use validation data for tuning.

Risk 8 — Hardware/Caching Effects
Problem

Latency can vary heavily with environment and cache state.

Mitigation
benchmark under consistent conditions;
document hardware;
repeat measurements;
record cache state where relevant.
Risk 9 — LLM-Judge Bias
Problem

An LLM evaluator may favor particular answer styles or generators.

Mitigation

Keep reference-based metrics primary.

Use LLM-based evaluation as supplementary.

## Failure Analysis Requirements

Aggregate metrics are insufficient.

The system should support inspecting examples where:

Routing failures
actual/simple-proxy -> predicted complex
actual/complex-proxy -> predicted simple
Retrieval failures
required evidence not retrieved
Compression failures
evidence present before compression
but absent after compression
Generation failures
relevant evidence present
but answer incorrect
Efficiency failures
compression saved tokens
but did not save latency

This analysis is important for research quality.

## Experimental Fairness Checklist

Every main comparison must satisfy:

[ ] Same query set
[ ] Same dataset split
[ ] Same corpus
[ ] Same chunking
[ ] Same embedding model
[ ] Same vector index
[ ] Same generator
[ ] Same prompt template
[ ] Same generation settings
[ ] Same hardware where possible
[ ] Same evaluation code
[ ] Validation used for tuning
[ ] Test reserved for final comparison
[ ] Latency methodology consistent
[ ] Cache behavior documented
[ ] Failures handled consistently
## Result Artifacts

The preferred result layout is:

results/
└── <experiment_name>/
    ├── config.yaml
    ├── metrics.json
    ├── predictions.jsonl
    ├── latency.csv
    ├── routing.csv
    ├── summary.csv
    ├── logs/
    └── plots/

The exact structure may evolve.

The core properties must remain:

machine-readable;
traceable;
reproducible.
## Required Final Result Tables

The eventual benchmark should be able to generate a comparison similar to:

System	EM	F1	Retrieval Metric	Avg Latency	P95 Latency	Avg Prompt Tokens	Token Reduction
LLM Only	—	—	—	—	—	—	—
Fixed RAG	—	—	—	—	—	—	—
Always-Compress RAG	—	—	—	—	—	—	—
AdaptiveRAG	—	—	—	—	—	—	—

The actual numbers must be generated from experiments.

No placeholder values should be presented as results.

## Potential Research Figures

Useful final plots may include:

Answer quality vs latency.
Answer quality vs prompt tokens.
Average tokens by system.
Latency breakdown by system.
Compression ratio distribution.
Router confusion matrix.
Performance by query-complexity group.
Compression ablation.
Adaptive-K ablation.

These are analysis artifacts, not mandatory UI features.

## Success Criteria

The project should judge success at multiple levels.

Engineering Success

The system correctly executes:

query
-> routing
-> retrieval
-> optional compression
-> generation

without manual intervention.

Experimental Success

The four primary systems can be run under controlled conditions.

Measurement Success

The project can measure:

answer quality;
retrieval quality;
router quality;
latency;
token usage;
compression behavior.
Research Success

The project can answer:

Under what circumstances does adaptive retrieval + selective compression help, hurt, or provide no meaningful advantage?

A negative result can still satisfy research success if the experiment is rigorous and interpretable.

## MVP Definition of Done

The MVP is complete when ALL of the following are true.

Core Pipeline
 Query input works.
 Query preprocessing works.
 Complexity router works.
 SIMPLE and COMPLEX routes work.
 Retrieval K changes according to route.
 SIMPLE route bypasses compression.
 COMPLEX route invokes compression.
 Fixed generator produces answers.
Baselines
 LLM-only baseline works.
 Fixed RAG baseline works.
 Always-compress baseline works.
 AdaptiveRAG works.
Evaluation
 Same evaluation set can run through every baseline.
 Answer quality is measured.
 Retrieval quality is measured where applicable.
 Router metrics are measured.
 Prompt/context tokens are measured.
 Compression ratio is measured.
 End-to-end latency is measured.
 Stage-level latency is measured.
Research
 At least one meaningful ablation is complete.
 Router errors can be inspected.
 Compression errors can be inspected.
 Representative failure cases can be analyzed.
 Results can be compared fairly.
Engineering
 Core logic has tests.
 Configuration is externalized.
 Experiments produce machine-readable artifacts.
 Environment/model/dataset configuration is recorded.
 Main experiment is reproducible.
## HARD STOP

Once the MVP Definition of Done is satisfied:

STOP adding core features.

Do not expand the project merely because more features are possible.

Do not automatically add:

GraphRAG
Agents
Multi-agent workflows
Multimodal retrieval
Web search
Privacy layer
Hallucination detector
RL routing
Dynamic chunking
Multiple generators
Large-scale fine-tuning
Fancy frontend
Cloud infrastructure

At this stage, the priority becomes:

better experiments
+
ablations
+
error analysis
+
reproducibility
+
research writing

This boundary exists to preserve research quality.

## Optional Post-MVP Extensions

Only after the hard-stop conditions are satisfied, the following may be considered:

Extension A — Confidence-Aware Fallback

If router confidence is low:

fallback to safer retrieval strategy
Extension B — Multi-Level Routing

Potentially:

SIMPLE
MEDIUM
COMPLEX
Extension C — Additional Datasets

Introduce broader query domains to evaluate generalization.

Extension D — Multiple Compressors

Compare LLMLingua with another established compression method.

Extension E — Cost Modeling

Estimate operational/API cost.

These are future work, not MVP requirements.

## Research Paper Readiness

The implementation should preserve enough evidence for later paper writing.

A possible final paper structure:

1. Introduction
2. Related Work
3. Problem Definition
4. AdaptiveRAG Method
5. Query Complexity Router
6. Adaptive Retrieval Policy
7. Selective Context Compression
8. Experimental Setup
9. Results
10. Ablation Studies
## Error Analysis
## Limitations
## Conclusion

The paper contribution should focus on:

adaptive allocation of retrieval effort;
selective compression;
empirical quality-efficiency trade-offs;
understanding when the strategy works.

The paper should explicitly discuss limitations.

## Limitations to Expect

Potential final limitations include:

binary complexity classification is an approximation;
dataset-defined complexity may not generalize perfectly;
lightweight router may make mistakes;
compression can remove useful evidence;
latency results depend on hardware;
generator choice affects results;
compression library/model choice affects results;
local experiments may not represent production-scale workloads.

These limitations do not weaken the project by themselves.

A research paper should acknowledge them.

## Research Claim Discipline

Throughout implementation and documentation, distinguish clearly among:

Established

What prior research has already demonstrated.

Proposed

What AdaptiveRAG intends to test or implement.

Measured

What our actual experiments demonstrate.

Hypothesized

What we expect may happen but have not yet verified.

Never transform:

hypothesis

into:

fact

before experimentation.

## Statements to Avoid

Do not write or hard-code statements such as:

The router runs under 50 ms.

unless actually benchmarked under a defined environment.

Do not write:

AdaptiveRAG reduces latency.

before experiments demonstrate it.

Prefer:

AdaptiveRAG is designed to investigate whether latency can be reduced.

Do not write:

PopQA is the simple-query dataset.

Prefer:

PopQA is used as a proxy for relatively simple/factual query characteristics.

Do not write:

HotpotQA represents all complex questions.

Prefer:

HotpotQA provides a useful multi-hop benchmark for evaluating queries requiring broader evidence.

Do not write:

Our approach is the first of its kind.

unless independently verified through rigorous literature review.

## DA1 Historical Context

The earlier academic proposal established the following initial design:

Project:
Adaptive Retrieval-Augmented Generation using Query Complexity and Context Optimization

Router:
DeBERTa-v3-small direction

Simple:
K = 2

Complex:
K = 10

Embedding:
BGE-small-en-v1.5 direction

Vector Store:
ChromaDB

Compression:
LLMLingua

Generator:
Phi-3-mini direction

Datasets:
PopQA
HotpotQA

Initial benchmark:
~1,000 examples per dataset

Baseline:
fixed K=5 RAG

Always-compress:
K=10 + compression

Main ablation:
K=10 raw vs K=10 compressed

These choices are the starting point of implementation.

They are not all permanent decisions.

Actual implementation evidence may justify changes.

If changed, update the decision log.

## Corrections to Historical Proposal Claims

Some statements in the original DA1 proposal were intentionally written as proposal-level claims.

They must not be treated as guaranteed project outcomes.

Specifically:

Router Latency

Earlier target:

<50 ms

Current specification:

Measure router latency and determine whether the overhead is sufficiently low to justify adaptation.
Simple Query Latency

Earlier target:

sub-200 ms

Current specification:

Latency is an experimental outcome, not a guaranteed requirement.
Dataset Complexity

Earlier framing:

PopQA = simple
HotpotQA = complex

Current specification:

They are proxies for different query characteristics and require validity analysis.
Compression Benefit

Earlier framing:

compression saves time

Current specification:

measure whether compression savings outweigh compression overhead.
Router Training Time

Earlier estimate:

less than one hour

Current specification:

training must remain compute-feasible; actual time is measured.
VRAM

Earlier estimate:

8–12 GB

Current specification:

resource usage must be measured on the actual environment.

These corrections are necessary for research honesty.

## Architecture Decision Principles

When a technical decision is not yet fixed, use the following priorities:

1. Experimental validity
2. Simplicity
3. Reproducibility
4. Compute feasibility
## Maintainability
## Performance
## Convenience

Do not select technology merely because it is fashionable.

## Decision Log

Every major architecture or research-methodology change should be documented here.

Use:

### DEC-XXX — <Decision>

### Date:
### Status:
### Decision:
### Reason:
### Alternatives considered:
### Impact:

Examples of decisions requiring an entry:

changing generator;
changing embedding model;
replacing ChromaDB;
changing routing strategy;
changing dataset composition;
adding/removing a baseline;
changing primary evaluation metrics;
adding a new research variable.

Small bug fixes do not require decision-log entries.

## Open Decisions

The following should remain explicitly open until implementation provides sufficient evidence.

OD-001 — Final Generator
### Current preference

Phi-3-mini-class small instruct model.

Alternatives

Llama 3-class small instruct model.

### Decision criteria
available VRAM;
inference stability;
context capacity;
reproducibility;
answer quality;
latency.
OD-002 — Final Router Model
### Current preference

DeBERTa-v3-small.

Alternatives

Another lightweight encoder classifier.

### Decision criteria
classification quality;
inference overhead;
training feasibility.
OD-003 — Final Embedding Model Version
### Current preference

BGE-small-en-v1.5 class model.

### Decision criteria
retrieval quality;
resource use;
availability;
reproducibility.
OD-004 — Compression Budget
### Current status

To be selected using validation experiments.

### Decision criteria
token reduction;
answer quality;
compression latency.
OD-005 — Final K Values
### Initial defaults
K_simple = 2
K_complex = 10
K_baseline = 5

These are starting values.

Final values may be tuned using validation data.

OD-006 — Router Threshold

If the learned model outputs confidence/probability, a threshold may be introduced.

The threshold must be selected using validation data.

OD-007 — Vector Store
### Current preference

ChromaDB.

Alternative

FAISS/local vector index.

### Decision criteria
simplicity;
indexing speed;
reproducibility;
retrieval performance;
integration complexity.
## Change Management

Any change that materially alters:

research question;
core architecture;
baseline methodology;
evaluation methodology;
dataset strategy;
generator;
router;
retrieval strategy;
compression methodology;
MVP scope;

must be reflected in prod-spec.md.

The code should then be updated to match the specification.

Do not allow code and specification to silently diverge.

## Source-of-Truth Rule

For future AI-assisted development:

Before implementing a new major feature, the coding agent should:

read prod-spec.md;
identify the relevant requirement;
inspect current implementation;
implement the smallest change satisfying the requirement;
update documentation/decision log if the design materially changes;
run relevant tests;
avoid adding unrelated functionality.

The agent must not reconstruct project intent from an old chat conversation when the information already exists here.

## AI Coding Agent Rules

Any future coding agent working on AdaptiveRAG should follow:

Rule 1

Read prod-spec.md before major implementation work.

Rule 2

Do not invent product requirements.

Rule 3

Do not add features merely because they are technically interesting.

Rule 4

Do not modify experimental methodology silently.

Rule 5

Do not fabricate benchmark results.

Rule 6

Do not claim implementation completion unless code actually works.

Rule 7

Keep research-critical logic modular.

Rule 8

Preserve reproducibility.

Rule 9

Prefer small validated changes.

Rule 10

Stop at the MVP boundary unless explicitly instructed to continue.

## Repository Documentation Strategy

The documentation hierarchy should eventually look approximately like:

README.md
    ->
What is this project?
How do I install/run it?
What does the system do?

prod-spec.md
    ->
What exactly are we building?
Why?
What are the requirements?
How is it evaluated?
What are the boundaries?

docs/
    ->
Detailed implementation notes
experiment reports
research notes
architecture decisions

Do not duplicate every implementation detail into README.md.

## Security / Privacy Scope

This is not a privacy-preserving RAG project.

The MVP does not include:

PII detection;
data sanitization;
privacy-preserving embeddings;
confidential retrieval;
access-control-aware retrieval.

However, basic software hygiene still applies.

Do not commit:

secrets;
API keys;
credentials;
private user data.

Use environment variables for secrets when external APIs are introduced.

## Performance Interpretation

Performance must be interpreted carefully.

A reduction in:

prompt tokens

does not automatically imply:

lower wall-clock latency

A reduction in:

retrieval K

does not automatically imply:

same answer quality

A higher:

retrieval recall

does not automatically imply:

higher generation quality

The research system must measure the complete chain.

## Expected Research Outcomes

There are several scientifically valid possible outcomes.

Outcome A — Strong Positive

AdaptiveRAG:

maintains/improves answer quality;
reduces tokens;
reduces or maintains latency.

This would strongly support the hypothesis.

Outcome B — Token Positive, Latency Neutral

AdaptiveRAG reduces token usage but compression overhead offsets latency gains.

This would still be valuable because it reveals an efficiency trade-off.

Outcome C — Quality Trade-Off

AdaptiveRAG is faster/cheaper but loses answer quality.

This suggests routing/compression needs improvement.

Outcome D — No Meaningful Benefit

AdaptiveRAG performs similarly to fixed RAG with additional complexity.

This is still a useful experimental result if explained rigorously.

The project must not manipulate experiments to force Outcome A.

## Minimum Error Analysis Set

At the end of the main benchmark, inspect representative examples from:

1. Correct SIMPLE route
2. Correct COMPLEX route
3. False SIMPLE
4. False COMPLEX
5. Compression success
6. Compression failure
7. Retrieval failure
## Generator failure despite correct retrieval
## Cases where adaptation helps
## Cases where adaptation hurts

These examples can later become part of the research paper.

## Final Research Narrative

The eventual project story should be understandable in one sequence:

Traditional RAG
    |
    v
Fixed retrieval depth
    |
    v
Same compute for every query
    |
    v
Potential inefficiency
    |
    +------------------------+
    |                        |
simple queries          complex queries
    |                        |
need less evidence       may need more evidence
    |                        |
    +-----------+------------+
                |
                v
       Adaptive query routing
                |
       +--------+--------+
       |                 |
    SIMPLE            COMPLEX
       |                 |
      low K             high K
       |                 |
  no compression     compression
       |                 |
       +--------+--------+
                |
                v
          fixed generator
                |
                v
        quality / latency /
        token evaluation

The research question is whether this allocation of computation provides a better overall trade-off.

## Project Definition in One Paragraph

AdaptiveRAG is a research-oriented Retrieval-Augmented Generation framework that estimates the complexity of an incoming question and dynamically allocates retrieval and context-processing effort. The initial system uses a lightweight binary router to classify queries as SIMPLE or COMPLEX. SIMPLE queries use a lower retrieval depth and bypass context compression, while COMPLEX queries retrieve more evidence and apply an established context-compression method before generation. A fixed language model generates answers across all experimental conditions. The system is evaluated against LLM-only, fixed-RAG, and always-compress RAG baselines using answer quality, retrieval quality, router quality, latency, and token-efficiency metrics, with controlled ablations used to isolate the contributions of adaptive retrieval and selective compression. The central research objective is to determine whether this adaptive allocation strategy offers a better quality-efficiency trade-off than a conventional one-size-fits-all RAG pipeline.

## Definition of "Done"

The project is not considered finished merely because:

a chatbot works;
a RAG pipeline answers questions;
a UI looks good;
a model can be demonstrated.

The project is finished when the research experiment works.

That means:

System works
+
Baselines work
+
Adaptive routing works
+
Compression gating works
+
Evaluation works
+
Ablations work
+
Metrics are reproducible
+
Errors can be analyzed
+
Results can support a defensible research conclusion

That is the actual definition of completion.

## Final Hard Boundary

The core project ends here:

Query
  ->
Complexity Router
  ->
Adaptive K
  ->
Selective Compression
  ->
Fixed LLM
  ->
Evaluation

with:

LLM-only
Fixed RAG
Always-compress RAG
AdaptiveRAG

as the main experimental comparison.

Everything beyond this is secondary.

The project should become deeper through:

better measurement
better experiments
better ablations
better analysis

rather than through uncontrolled feature expansion.

## Final Checklist Before Considering the Specification Stable
[ ] Project objective is unambiguous
[ ] Research question is unambiguous
[ ] Hypotheses are explicit
[ ] Research gap is conservatively framed
[ ] Complexity routing is defined
[ ] Simple path is defined
[ ] Complex path is defined
[ ] Retrieval behavior is defined
[ ] Compression behavior is defined
[ ] Generator behavior is defined
[ ] Datasets are defined
[ ] Dataset shortcut risk is documented
[ ] Baselines are defined
[ ] Evaluation metrics are defined
[ ] Latency methodology is defined
[ ] Token methodology is defined
[ ] Ablation methodology is defined
[ ] Experimental fairness is defined
[ ] Reproducibility is defined
[ ] Failure handling is defined
[ ] Testing strategy is defined
[ ] Compute constraints are defined
[ ] MVP boundary is defined
[ ] Hard stop is defined
[ ] Future extensions are separated
[ ] Open decisions are separated
[ ] Change management is defined
[ ] AI coding-agent rules are defined
## Authoritative Project Principle

From this point onward, AdaptiveRAG should be developed according to the following principle:

Build the smallest rigorous system capable of answering the research question convincingly.

The objective is not to build the largest RAG system possible.

The objective is not to collect the most fashionable components.

The objective is to build a controlled adaptive RAG system whose behavior can be measured, compared, explained, reproduced, and eventually written up as research.

That principle governs all future implementation decisions.

One correction I intentionally made compared with the earlier DA1 material is that this document **does not inherit the DA1 proposal's speculative claims as facts**. The DA1 report explicitly proposed DeBERTa, K=2/K=10, ChromaDB, BGE-small, LLMLingua, Phi-3, PopQA/HotpotQA, and the compression ablation, so those remain our starting architecture, but things like `<50 ms`, guaranteed sub-200 ms latency, guaranteed token/latency savings, and treating PopQA/HotpotQA as absolute complexity labels are now correctly framed as things to **measure or validate**. :contentReference[oaicite:0]{index=0}

This is the document I would now treat as the **canonical foundation for the implementation phase**.
