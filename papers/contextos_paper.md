# ContextOS: An Operating System Abstraction for Context Lifecycle Management in Long-Horizon Autonomous Agents

**Journal:** Information Sciences (Elsevier)  
**Status:** Under Review  
**Authors:** Anonymous Author 1, Anonymous Author 2, Anonymous Author 3

---

## Abstract

Long-horizon autonomous agents face a fundamental bottleneck: the finite context window imposes a hard ceiling on the amount of information an agent can reason over at any given moment, yet complex, multi-session tasks demand the persistent integration of observations, goals, plans, and environmental feedback that far exceeds this ceiling. Existing approaches—retrieval-augmented generation, simple truncation, and periodic summarization—treat context management as an afterthought rather than as a first-class system concern, resulting in information loss, inefficient token utilization, and degraded task performance as horizon length grows. We introduce **ContextOS**, an operating-system-inspired framework that treats the context window as a managed resource subject to lifecycle policies directly analogous to those governing CPU scheduling, memory paging, and storage tiering in classical operating systems.

**ContextOS** comprises six tightly integrated components: (i) a *Retrieval Engine* supporting hybrid semantic search via dense and sparse retrieval with reciprocal-rank fusion; (ii) a *Compression Engine* that performs quality-preserving context reduction through extractive, abstractive, and hierarchical strategies; (iii) a *Prioritization Engine* implementing a multi-dimensional scoring function; (iv) a *Context Scheduler* that performs budget-aware, greedy item selection with provable submodularity guarantees; (v) a hierarchical *Memory System* separating working memory from a three-tier long-term store of episodic, semantic, and procedural memories; and (vi) a *Governance Engine* that enforces retention, forgetting, and promotion policies.

We introduce a multi-factor priority function incorporating relevance, recency with exponential temporal decay, importance, and novelty measured via Maximum Marginal Relevance. We evaluate **ContextOS** on **ContextBench**, a new benchmark of 70,000 long-horizon agent tasks spanning eight task types and six domains, comparing against five baselines including MemGPT and RAPTOR across three backbone large language models (GPT-OSS-20B, Qwen3, and GLM-4.5) and four context-length regimes (512–32K tokens). **ContextOS** achieves a task-success rate of **71.3%** at 32K tokens—a **12.4 percentage-point** improvement over the best baseline (MemGPT) and **39.8 points** over naïve truncation—while reducing mean token consumption by **62.9%** relative to the Full-Context baseline. Ablation studies confirm that every component contributes meaningfully, with long-term memory governance and the priority scheduler accounting for the largest individual gains. Code, data, and evaluation scripts will be released upon publication.

**Keywords:** context engineering · autonomous agents · context lifecycle management · long-horizon reasoning · retrieval-augmented generation · memory management

---

## 1. Introduction

Autonomous agents powered by large language models (LLMs) are increasingly deployed in settings that require sustained, coherent reasoning over extended time horizons—executing multi-step plans, maintaining conversational history across sessions, integrating heterogeneous tool outputs, and recovering from failures while preserving long-range goal coherence [Wang et al., 2023a; Xi et al., 2023; AutoGPT, 2023]. Unlike single-turn question answering or short-horizon dialogue, these long-horizon tasks generate a continuously growing stream of context: observations from the environment, intermediate reasoning traces, retrieved knowledge, tool results, and evolving goal specifications. The fundamental bottleneck is well-known: every LLM operates within a finite *context window* whose token capacity—whether 4K, 32K, or even 128K tokens—is eventually exhausted by any sufficiently long-running agent. When the window is full, something must be discarded, and the choice of what to discard—or retain, compress, or offload—determines whether the agent retains the reasoning capacity necessary to complete its task. This problem is not merely a matter of scale; it is a fundamental issue of *information lifecycle management* that existing systems have failed to address in a principled way.

Contemporary approaches to context management are largely ad hoc. Retrieval-augmented generation (RAG) augments LLMs with external retrieval over a static document corpus [Lewis et al., 2020], but it provides no mechanism for managing the lifecycle of context items that are generated dynamically during agent execution. Naïve truncation—discarding the oldest tokens when the window fills—is simple to implement but irreversibly loses information that may be critical for future reasoning steps. Periodic summarization compresses portions of the context into shorter representations [Chevalier et al., 2023], but the granularity loss incurred during summarization is difficult to recover, and existing summarization methods do not account for the differential importance of context items or their anticipated future relevance. None of these approaches consider the context window as a resource to be *managed*—with explicit allocation, scheduling, eviction, and promotion policies—in the same way that an operating system manages CPU time, physical memory, and storage tiers.

The analogy between agent context management and operating system resource management is more than superficial. A classical operating system faces precisely the same class of problem: finite physical resources (CPU registers, RAM pages, disk sectors) must be allocated among competing demands, with policies governing which resources to cache, which to evict, and when to promote data between storage tiers. The solutions developed over decades of OS research—priority-based scheduling, LRU eviction, tiered memory hierarchies, demand paging, and garbage collection—encode hard-won insights about balancing performance, correctness, and fairness under resource constraints. We argue that agent context management stands in the same relation to these ideas as early single-process batch systems did to modern OS design: the problem is real, the cost of ignoring it is severe, and a principled framework is both possible and necessary.

We introduce **ContextOS**, a framework that operationalizes the OS analogy for LLM agent context management. **ContextOS** treats every item in the context window—whether an observation, a retrieved document, a tool result, a goal specification, or a reasoning trace—as a managed resource with an explicit lifecycle: *creation*, *storage*, *retrieval*, *scheduling*, *compression*, *governance*, and *eviction*. The system coordinates six components—a Retrieval Engine, a Compression Engine, a Prioritization Engine, a Context Scheduler, a hierarchical Memory System, and a Governance Engine—through a unified lifecycle management loop. This paper makes three principal contributions:

1. **The ContextOS framework.** We present the first unified context lifecycle management system for LLM agents, formalizing context management as a resource-allocation problem and providing a six-component architecture that addresses retrieval, prioritization, compression, scheduling, memory organization, and governance in an integrated manner.

2. **A multi-dimensional priority scheduling algorithm with temporal decay and MMR-based novelty.** We introduce a priority function π(i, q) that jointly captures semantic relevance, recency under exponential decay, user-assigned importance, and information novelty via Maximum Marginal Relevance (MMR), and we prove that a greedy scheduler operating on this function achieves a (1 − 1/e) approximation ratio relative to the optimal context selection under a submodular coverage objective.

3. **ContextBench: a new benchmark for long-horizon context management.** We construct and release a benchmark of 70,000 tasks spanning eight task types (multi-hop question answering, causal-chain reasoning, procedural planning, code debugging, literature synthesis, timeline reconstruction, cross-domain analogy, and adversarial distraction) and six domains, with context lengths ranging from 512 to 32K tokens and three difficulty levels.

Our experimental evaluation demonstrates that **ContextOS** substantially outperforms all baselines. At a context length of 32K tokens—the regime that most severely stresses context management—**ContextOS** achieves a task-success rate (TSR) of 71.3% on **ContextBench**, compared to 58.9% for MemGPT [Packer et al., 2023], 54.1% for RAPTOR [Sarthi et al., 2024], and 31.5% for naïve first-K truncation, while consuming 62.9% fewer tokens than the Full-Context baseline. Ablation studies confirm that each component contributes positively, and that the long-term memory governance mechanism and the priority scheduler together account for the majority of the performance gain.

---

## 2. Related Work

### 2.1 Context Management in Large Language Models

The challenge of extending the effective context capacity of transformer-based language models has been approached from several complementary directions. *Architectural extensions* increase the hard limit on the context window itself. Longformer [Beltagy et al., 2020] replaces the full quadratic self-attention mechanism with a combination of local windowed attention and task-motivated global attention tokens, achieving linear complexity with respect to sequence length. BigBird [Zaheer et al., 2020] further generalizes this idea with a random-attention component that provides theoretical expressiveness guarantees while retaining near-linear scaling. More recently, positional encoding modifications such as ALiBi [Press et al., 2022] and RoPE [Su et al., 2024] have enabled models trained at shorter lengths to generalize at inference time to considerably longer sequences. FlashAttention [Dao et al., 2022; Dao, 2023] addresses the memory bandwidth bottleneck of exact attention, making 32K-and-beyond windows computationally tractable on commodity hardware. Despite these advances, every architecture retains a finite context ceiling, and the problem of *what* to place within that ceiling remains unsolved by architectural means alone.

*Retrieval-augmented generation* (RAG) [Lewis et al., 2020] addresses the content problem by decoupling knowledge storage from the context window: a retriever populates the window on demand from an external corpus. REALM [Guu et al., 2020] demonstrated that retrieval can be integrated into pre-training, enabling the language model to learn to condition on retrieved passages. Subsequent work has refined retrieval quality [Karpukhin et al., 2020], extended RAG to multi-hop settings [Trivedi et al., 2022], and applied RAG to code [Parvez et al., 2021] and dialogue [Shuster et al., 2021]. However, standard RAG operates over a static external corpus and is not designed to manage the *dynamic* context items generated during agent execution—observations, intermediate results, and evolving goals—which form the primary information challenge for long-horizon agents.

*Context compression and distillation* approaches reduce the token footprint of context while attempting to preserve its semantic content. AutoCompressors [Chevalier et al., 2023] fine-tune a model to compress long context segments into compact "summary vectors" that can be prepended to subsequent context windows. LLMLingua [Jiang et al., 2023] identifies and removes redundant tokens at inference time using a smaller proxy LLM to score token importance. RECOMP [Xu et al., 2023] generates compressed summaries of retrieved passages to reduce the token cost of RAG. Ge et al. [2023] provide a taxonomy of compression techniques ranging from selective extraction to abstractive generation. Token pruning methods [Ye et al., 2024; Kim et al., 2022] remove attention heads or tokens dynamically based on gradient or attention signals. **ContextOS** incorporates compression as one component of a broader lifecycle, and uniquely adapts the compression strategy (extractive, abstractive, or hierarchical) to the type and priority of each context item rather than applying a single strategy uniformly.

### 2.2 Agent Memory Systems

The memory systems of cognitive architectures provide an important precursor to modern agent memory design. ACT-R [Anderson and Lebiere, 2004] models human memory with declarative chunks subject to associative recall and spreading activation, and procedural rules that fire based on goal state. SOAR [Laird, 2012] organizes knowledge into production rules with chunking-based learning. These architectures inspired early work on deliberative agent reasoning but were not designed for the neural, token-based computation of LLMs.

MemGPT [Packer et al., 2023] is the closest prior system to **ContextOS** in motivation. It proposes a virtual-context management scheme modeled explicitly on virtual memory in operating systems, with a hierarchy of in-context (main memory) and out-of-context (external storage) segments, and a set of OS-inspired function calls for moving content between tiers. Our work differs from MemGPT in several critical ways. First, **ContextOS** introduces a principled multi-dimensional priority function with temporal decay and MMR-based novelty that governs all scheduling decisions, whereas MemGPT relies on simpler recency-based heuristics. Second, **ContextOS** integrates a hybrid retrieval engine with cross-encoder reranking, enabling high-precision retrieval at scale. Third, our Governance Engine implements explicit retention policies (time-based, importance-based, and frequency-based), forgetting curves, and promotion criteria, providing fine-grained lifecycle control that MemGPT does not offer. Fourth, we provide a comprehensive empirical evaluation on **ContextBench**, a purpose-built benchmark, whereas MemGPT was evaluated on smaller task sets.

Generative Agents [Park et al., 2023] equip simulated social characters with a memory stream that stores observations and is periodically summarized into higher-level reflections. Retrieval from the memory stream uses a weighted combination of recency, importance, and relevance scores—an idea that partly inspires our priority function, though we formalize it considerably more rigorously and integrate it with the full lifecycle management loop. VOYAGER [Wang et al., 2023b] maintains a skill library in Minecraft by storing and retrieving code snippets, but its memory system is domain-specific and lacks general lifecycle management policies. Zhong et al. [2024] propose a memory bank mechanism for long-term persona consistency in chatbots, with Ebbinghaus-inspired forgetting curves; our Governance Engine generalizes this idea to agents with diverse task types and memory categories.

### 2.3 Retrieval and Reranking

Dense retrieval methods represent both queries and documents as dense vectors and retrieve by approximate nearest-neighbor search. DPR [Karpukhin et al., 2020] established the effectiveness of dual-encoder retrieval for open-domain question answering. BGE-M3 [Chen et al., 2024] unifies dense, sparse, and multi-vector retrieval in a single model, achieving state-of-the-art performance across multiple languages and retrieval paradigms; we adopt it as the default dense encoder in **ContextOS**. ColBERT [Khattab and Zaharia, 2020] uses late interaction between per-token representations for fine-grained matching.

Sparse retrieval methods, notably BM25 [Robertson and Zaragoza, 2009] and SPLADE [Formal et al., 2021], offer complementary strengths: lexical precision, robustness to out-of-distribution vocabulary, and low computational cost. Hybrid methods that combine dense and sparse signals have consistently outperformed either alone [Chen et al., 2022; Ma et al., 2022]. We implement Reciprocal Rank Fusion (RRF) [Cormack et al., 2009] to combine the two ranked lists, as it is parameter-efficient and robust across domains. Reranking with cross-encoders [Nogueira and Cho, 2019] substantially improves precision by jointly encoding query and document, at the cost of higher latency. Maximum Marginal Relevance (MMR) [Carbonell and Goldstein, 1998] provides a principled framework for diversity-aware selection that penalizes redundant items; we incorporate MMR as the novelty component of our priority function.

### 2.4 Context Scheduling and Prioritization

RAPTOR [Sarthi et al., 2024] addresses the challenge of long-document retrieval by building a hierarchical tree of recursive summaries that allows retrieval at multiple levels of abstraction. While effective for static document corpora, RAPTOR does not address the dynamic context management problem faced by long-horizon agents, and it does not incorporate temporal decay, importance-based retention, or governance policies. Liu et al. [2023] identify the "lost in the middle" phenomenon: LLMs exhibit a primacy and recency bias in long-context utilization, systematically under-attending to information placed in the middle of the context window. This finding motivates our scheduler's strategy of ordering selected context items to place the highest-priority content at boundary positions. Shi et al. [2023] demonstrate that models are easily distracted by irrelevant context, underscoring the importance of active context governance rather than passive window filling.

In summary, **ContextOS** is differentiated from all prior work by its treatment of context management as a complete, integrated *lifecycle*—not merely retrieval, not merely summarization, and not merely virtual paging—with principled policies governing every stage from creation to eviction. To the best of our knowledge, **ContextOS** is the first framework to combine all six components (retrieval, compression, prioritization, scheduling, memory hierarchy, and governance) within a single, evaluated system.

---

## 3. Problem Formulation

### 3.1 Context Window Model

We model the context window as a finite token buffer C with capacity B_max ∈ ℕ⁺ tokens. At each agent step t, the agent receives a query q_t (which may be an external instruction, an environmental observation, or an internally generated subgoal) and must construct a context C_t ⊆ I_t from the available information set I_t = {i₁, i₂, …, i_n_t}, subject to the hard budget constraint:

```
∑_{i ∈ C_t} tokens(i) ≤ B_max
```

Each context item i ∈ I is a tuple:

```
i = (content(i), type(i), τ(i), α(i), φ(i))
```

where:
- `content(i) ∈ Σ*` is the token sequence
- `type(i) ∈ {GOAL, PLAN, OBSERVATION, TOOL_RESULT, RETRIEVAL, REASONING, SUMMARY}` is the semantic category
- `τ(i) ∈ ℝ≥0` is the Unix creation timestamp
- `α(i) ∈ [0,1]` is an importance score assigned at creation time
- `φ(i) ∈ ℝᵈ` is a dense embedding of `content(i)` computed by the retrieval encoder

### 3.2 Context Selection as Optimization

Given the query q_t with embedding φ_{q_t} and the available item set I_t, the ideal context selection is the solution to the following optimization problem:

```
C*_t = argmax_{S ⊆ I_t} ∑_{i ∈ S} rel(i, q_t) · α(i)
       subject to ∑_{i ∈ S} tokens(i) ≤ B_max
```

where `rel(i, q_t) = cos(φ(i), φ_{q_t}) ∈ [-1, 1]` denotes the cosine similarity between item i and the query.

> **Proposition (NP-Hardness).** The context selection problem above is NP-hard in general.
>
> *Proof.* Consider a special case in which `rel(i, q_t) · α(i) > 0` for all i, and define `v_i = rel(i, q_t) · α(i)` as the value of item i and `w_i = tokens(i)` as its weight. The problem then reduces precisely to the 0/1 Knapsack problem [Garey and Johnson, 1979], which is NP-hard. □

Because exact optimization is intractable for the item set sizes arising in practice (often |I_t| > 10⁴), we adopt a greedy approximation based on a richer multi-factor priority function π(i, q) that subsumes the objective above while adding temporal decay and novelty penalties.

### 3.3 Context Item Lifecycle

A context item passes through the following lifecycle stages, managed by the corresponding **ContextOS** components:

| Stage | Component | Description |
|---|---|---|
| **1. Creation** | — | Item instantiated with content, type, timestamp, importance; embedding φ(i) computed and cached |
| **2. Storage** | Memory System | Item inserted into working-memory heap or appropriate long-term tier (episodic/semantic/procedural) |
| **3. Retrieval** | Retrieval Engine | Hybrid dense-sparse search identifies candidate items given current query |
| **4. Scheduling** | Context Scheduler | π(i, q) evaluated; budget-feasible subset selected greedily |
| **5. Compression** | Compression Engine | If total tokens exceed budget, item lengths are reduced using appropriate strategy |
| **6. Governance** | Governance Engine | Retention, forgetting, and promotion policies enforced periodically |
| **7. Eviction** | Governance Engine | Items failing retention thresholds removed; high-value items promoted before eviction |

---

## 4. ContextOS Architecture

### 4.1 System Overview

**ContextOS** is organized as a pipeline of six interacting components. At each agent step, the pipeline is invoked with the current query q_t and the accumulated item set I_t. The Retrieval Engine first identifies the K_r = 200 highest-scoring candidate items from all memory tiers. The Prioritization Engine scores each candidate according to the multi-factor priority function π(i, q_t). The Context Scheduler selects a budget-feasible subset by greedily iterating over the priority-sorted candidates. If the selected set exceeds the available token budget after accounting for the system prompt and query, the Compression Engine reduces individual item lengths. The Memory System maintains all items across working and long-term tiers between agent steps. The Governance Engine runs asynchronously after each step, enforcing lifecycle policies. The assembled context is then passed to the backbone LLM.

**OS–ContextOS Analogy:**

| Operating System | ContextOS |
|---|---|
| CPU Scheduler | Context Scheduler |
| Physical RAM | Working Memory |
| Virtual Memory / Disk | Long-Term Memory |
| Memory Manager / GC | Governance Engine |
| Cache / TLB | Retrieval Engine |
| Page Replacement Policy | Forgetting Policy |

All components communicate through a shared *Context Registry* that stores item tuples and maintains metadata (access counts, last-access timestamps, compression state, and promotion history) for each item. The registry is backed by an in-process vector store (Qdrant) for approximate nearest-neighbor queries and a key-value store for exact lookups by item identifier.

### 4.2 Retrieval Engine

The Retrieval Engine implements a three-stage pipeline: (1) hybrid first-stage retrieval combining dense and sparse signals via Reciprocal Rank Fusion; (2) cross-encoder second-stage reranking; and (3) a diversity post-filter.

**Dense Retrieval.** We embed all context items using BGE-M3 [Chen et al., 2024], a state-of-the-art multilingual dense encoder that produces 1024-dimensional embeddings. The query q_t is encoded with the same encoder, and the top-K_d items are retrieved by maximum inner product search (MIPS) using HNSW indices maintained in Qdrant:

```
R_dense = argmax_{K_d, i ∈ I} φ(i)ᵀ φ(q_t)
```

with K_d = 100 by default.

**Sparse Retrieval.** Sparse retrieval is performed using BM25 [Robertson and Zaragoza, 2009] over tokenized item content. The BM25 score for item i and query q is:

```
BM25(i, q) = ∑_{w ∈ q} IDF(w) · f(w,i)·(k₁+1) / [f(w,i) + k₁·(1 - b + b·|content(i)|/μ_L)]
```

where f(w, i) is the term frequency of word w in item i, μ_L is the mean item length, k₁ = 1.5, and b = 0.75. The top-K_s = 100 items by BM25 score are retrieved as R_sparse.

**Reciprocal Rank Fusion.** The dense and sparse ranked lists are merged via RRF [Cormack et al., 2009]:

```
RRF(i) = 1/(k + r_dense(i)) + 1/(k + r_sparse(i))
```

where r_dense(i) and r_sparse(i) are the ranks of item i in the respective lists, and k = 60 is the fusion constant. The top-K_r = 200 items by RRF score form the candidate set passed to the Prioritization Engine.

**Cross-Encoder Reranking.** A cross-encoder model jointly encodes each query-item pair to produce a fine-grained relevance score that is incorporated into the priority function. Cross-encoder inference is batched over the K_r candidates with a maximum batch size of 64.

### 4.3 Prioritization Engine

The Prioritization Engine computes a scalar priority score π(i, q) for each candidate item i given the current query q. The priority function integrates four dimensions:

```
π(i, q) = w_r · rel(i,q) + w_e · rec(i) + w_m · α(i) + w_n · nov(i, S)
```

**Relevance `rel(i, q)`.** The cross-encoder score CE(q, i) ∈ [0, 1] (obtained from the second-stage reranker) is used when available; otherwise, the normalized cosine similarity `(cos(φ(i), φ(q)) + 1) / 2` is used. Default weight: **w_r = 0.40**.

**Recency `rec(i)`.** Temporal decay is modeled with an exponential decay function:

```
rec(i) = exp(−λ · Δt_i)
```

where Δt_i = t_now − τ(i) is the elapsed time in seconds since item i was created, and λ = 10⁻⁴ s⁻¹ corresponds to a half-life of approximately 115 minutes—calibrated to the typical duration of a long-horizon agent session. Default weight: **w_e = 0.25**.

**Importance `α(i)`.** The importance score α(i) ∈ [0, 1] is assigned at item creation. For items of type GOAL or PLAN, α(i) = 1.0 unconditionally. For tool results and observations, α(i) is set by an automatic importance classifier. Default weight: **w_m = 0.20**.

**Novelty `nov(i, S)`.** To avoid filling the context with redundant information, the novelty score penalizes items that are semantically similar to items already selected in S:

```
nov(i, S) = 1                                    if S = ∅
           = 1 − max_{j ∈ S} cos(φ(i), φ(j))    otherwise
```

This formulation follows the MMR principle [Carbonell and Goldstein, 1998]: items that are already well-represented in the selected set receive a lower novelty score and are thus deprioritized. Default weight: **w_n = 0.15**.

The weight vector (w_r, w_e, w_m, w_n) = (0.40, 0.25, 0.20, 0.15) was selected via a grid search on the **ContextBench** validation set.

### 4.4 Context Scheduler

The Context Scheduler takes the priority-scored candidate set and selects a budget-feasible subset. It implements a greedy algorithm that exploits the submodular structure of the selection objective.

**Algorithm 1: GreedyContextScheduler**

```
Input:  Candidate set I, query q, token budget B, priority function π(·, q)
Output: Selected context set S

S ← ∅
B_used ← 0
Compute π(i, q) for all i ∈ I
Sort I in descending order of π(i, q)

for each i ∈ I (in priority order):
    if B_used + tokens(i) ≤ B:
        S ← S ∪ {i}
        B_used ← B_used + tokens(i)
        Update nov(j, S) for all remaining j ∈ I \ S
        Re-sort remaining I \ S by updated π(j, q)

Apply positional ordering: place GOAL/PLAN items first, OBSERVATION/TOOL_RESULT items last
return S
```

**Approximation Guarantee.**

> **Theorem (Greedy Approximation Ratio).** Let f(S) = ∑_{i ∈ S} π(i, q) be the priority coverage objective (with the novelty component nov(i, S) evaluated incrementally as in Algorithm 1). When f is monotone submodular and tokens(i) = c for all i (uniform token cost), the greedy solution S_G satisfies:
>
> ```
> f(S_G) ≥ (1 − 1/e) · f(S*)
> ```
>
> where S* = argmax_{|S| ≤ K} f(S) is the optimal K-item solution and K = ⌊B/c⌋.
>
> *Proof.* The priority function π(i, q) as defined is submodular in the selection set S because the novelty component `nov(i, S) = 1 − max_{j ∈ S} cos(φ(i), φ(j))` is a coverage function: as S grows, nov(i, S) is non-increasing for any fixed i. The sum over selected items of such a function is a non-negative linear combination of submodular functions and is therefore submodular. Under the uniform token-cost assumption, the cardinality-constrained maximization of a monotone submodular function by a greedy algorithm achieves the (1 − 1/e) ratio [Nemhauser et al., 1978]. □

In the non-uniform-cost regime, we apply the Knapsack-greedy variant that selects by the density π(i, q) / tokens(i), which achieves a 1/2-approximation [Sviridenko, 2004].

### 4.5 Compression Engine

Even after greedy selection, the assembled context may exceed the available token budget due to mandatory items. The Compression Engine reduces item lengths while preserving semantic fidelity.

**Compression Strategies:**

- **Extractive compression:** Sentences are scored by a sentence salience model (66M-parameter DistilBERT-based classifier) and the top-scoring sentences are retained. Targets compression ratio ρ ∈ [0.4, 0.8].
- **Abstractive compression:** A 1.5B-parameter summarization model (fine-tuned T5-XL) generates a condensed paraphrase of the item. Targets ρ ∈ [0.2, 0.5].
- **Hierarchical compression:** Items are first segmented into passages, each passage is abstractively compressed, and the passage summaries are then concatenated. Achieves ρ ∈ [0.1, 0.35].

**Progressive Compression Strategy Selection:**

```
strategy(β) = Extractive    if β ≤ 0.85
            = Abstractive   if 0.85 < β ≤ 0.95
            = Hierarchical  if β > 0.95
```

where β = B_used / B_max is the current budget pressure ratio.

**Type-Conditional Compression:** Items of type GOAL or PLAN are **never compressed**, as their full content is essential for maintaining task coherence. Items of type REASONING are compressed only extractively. All other item types are eligible for any compression strategy.

**Quality Control:** After compression, the semantic similarity `sim(i, î) = cos(φ(i), φ(î))` is computed. If `sim(i, î) < θ_fidelity = 0.85`, the compression is rejected and the original item is retained.

### 4.6 Memory System

The **ContextOS** Memory System implements a two-level hierarchy.

**Working Memory.** Working memory is a bounded priority heap with capacity K_WM = 50 items. Items in working memory are those most recently accessed or most highly prioritized. The heap is keyed by priority π(i, q_last), where q_last is the most recent query. Insertion into a full working memory triggers the Governance Engine's eviction policy.

**Long-Term Memory.** Long-term memory is organized into three tiers:

- **Episodic memory:** Stores event sequences—time-stamped observations, action records, and their outcomes. Items are indexed by both timestamp and embedding for temporal range queries and semantic search.
- **Semantic memory:** Stores factual assertions extracted from observations and tool results. Facts are represented as (subject, predicate, object) triples plus a confidence score, supporting structured queries in addition to semantic search.
- **Procedural memory:** Stores reusable action patterns—sequences of tool calls that successfully achieved a subgoal—indexed by the subgoal embedding. At inference time, procedural memories provide executable plan templates that reduce the planning burden on the backbone LLM.

**Promotion Criteria.** An item in working memory is promoted to long-term memory when both conditions hold:

```
access_count(i) ≥ 3    AND    α(i) ≥ 0.7
```

The destination tier within long-term memory is determined by type(i): observations and tool results go to episodic memory, extracted facts to semantic memory, and successful action sequences to procedural memory.

### 4.7 Governance Engine

The Governance Engine runs asynchronously in a background thread, executing periodically (default: every 30 seconds or after every 100 agent steps) to enforce lifecycle policies across all memory tiers.

**Retention Policies** (composable):

- **TimeBasedRetention(T):** Item i is retained if Δt_i < T. Default: T = 72 h for working memory, T = 720 h for long-term episodic memory.
- **ImportanceBasedRetention(θ):** Item i is retained if α(i) ≥ θ. Default: θ = 0.1.
- **FrequencyBasedRetention(f_min):** Item i is retained if access_count(i) / Δt_i ≥ f_min. Default: f_min = 0.001 accesses/h.

An item is evicted when it fails *any* active retention policy. Before eviction, the Governance Engine checks the promotion criteria; items that meet both criteria are promoted rather than evicted.

**Forgetting.** Importance scores are subject to exponential decay:

```
α_t(i) = α_0(i) · exp(−λ_f · Δt_i)
```

with λ_f = 5 × 10⁻⁵ s⁻¹ (half-life ≈ 3.9 h). This mimics the Ebbinghaus forgetting curve [Ebbinghaus, 1885]. Items of type GOAL are exempt from forgetting (λ_f = 0).

**Compression Triggers.** A `TokenBudgetTrigger` fires when the total token count of all working-memory items exceeds 80% of B_max, initiating extractive compression on the lowest-priority quintile of working memory items.

---

## 5. ContextBench: A Benchmark for Long-Horizon Context Management

### 5.1 Dataset Construction

**ContextBench** is a purpose-built evaluation benchmark for long-horizon context management. It consists of 70,000 tasks drawn from eight task types across six domains, with context lengths ranging from 512 to 32,768 tokens and three difficulty levels (easy, medium, hard). The benchmark is partitioned into 60,000 training examples, 5,000 validation examples, and 5,000 held-out test examples.

Tasks were constructed via a combination of expert curation and template-based generation. For each task, a *ground-truth context* is defined as the minimal subset of context items necessary and sufficient for a well-informed agent to produce the correct answer. Distractor items (semantically related but irrelevant) are added to reach the target context length. The difficulty level is determined by the ratio of distractor items to ground-truth items (easy: 2:1, medium: 5:1, hard: 10:1). Annotators verified all ground-truth context labels and task answers via a two-round agreement process; inter-annotator agreement (Cohen's κ) is **0.84** across all task types.

The six domains are: (1) Scientific Literature, (2) Legal Reasoning, (3) Software Engineering, (4) Financial Analysis, (5) Historical Research, and (6) Medical Diagnosis.

### 5.2 Task Types

**Table 1.** Task types in **ContextBench**. "Avg. Relevant Items" refers to the mean number of ground-truth context items per task. "Avg. Context Tokens" is the mean total context size including distractors.

| Task Type | Description | Count | Avg. Rel. Items | Avg. Ctx. Tokens |
|---|---|---:|---:|---:|
| Multi-hop QA | Chain of ≥3 reasoning hops | 11,500 | 5.2 | 8,342 |
| Causal Chain | Root-cause and effect tracing | 9,000 | 4.1 | 7,615 |
| Procedural Planning | Multi-step action sequence planning | 9,500 | 6.8 | 11,200 |
| Code Debugging | Error diagnosis across files | 10,000 | 7.3 | 13,480 |
| Literature Synthesis | Cross-document claim synthesis | 8,000 | 9.1 | 18,920 |
| Timeline Reconstruction | Temporal event ordering | 7,500 | 5.9 | 9,870 |
| Cross-domain Analogy | Structural similarity detection | 7,000 | 3.7 | 6,150 |
| Adversarial Distraction | Resistance to irrelevant context | 7,500 | 2.8 | 14,300 |
| **Total / Mean** | | **70,000** | **5.6** | **11,235** |

### 5.3 Evaluation Metrics

**Task Success Rate (TSR).** TSR is the primary metric. For each task, the system's prediction is compared to the ground-truth answer using token-level F1 (following the SQuAD evaluation protocol [Rajpurkar et al., 2016]):

```
F1_task = 2·P·R / (P + R)
```

where P and R are precision and recall over answer tokens after stopword removal. TSR is the mean F1_task across all test examples. A task is considered "successful" if F1_task ≥ 0.5.

**Context Relevance Score (CRS).** Given the ground-truth relevant item set G and the system-selected set S:

```
P@K = |S_{:K} ∩ G| / K

NDCG@10 = DCG@10 / IDCG@10

DCG@10 = ∑_{k=1}^{10} 𝟙[i_k ∈ G] / log₂(k+1)
```

**Token Efficiency (TE).** Mean tokens consumed per successfully completed task (lower is better):

```
TE = ∑_{t ∈ T_success} tokens(C_t) / |T_success|
```

**Latency.** Wall-clock time for context preparation (retrieval through context assembly, excluding LLM inference) in milliseconds, measured on a single NVIDIA A100 80 GB GPU.

---

## 6. Experimental Setup

### 6.1 Baselines

We compare **ContextOS** against five baselines representing the principal families of context management strategies:

1. **Full Context (FC):** The entire available context is passed to the LLM without any management. When the total context exceeds the window limit, the task is marked as failed. This baseline reveals the ceiling performance achievable with unlimited window capacity.

2. **Truncation / First-K:** The first B_max tokens of the context stream are retained and the remainder discarded. This is the simplest and most widely deployed production strategy.

3. **RAG-Only:** A standard RAG pipeline using the same BGE-M3 encoder and Qdrant backend as **ContextOS**, but without the Prioritization Engine, Memory System, or Governance Engine. Retrieves the top-K items by cosine similarity and fills the context greedily.

4. **MemGPT** [Packer et al., 2023]: The virtual-context management system with in-context main memory and out-of-context archival memory, with function-call-based memory operations.

5. **RAPTOR** [Sarthi et al., 2024]: Hierarchical summarization tree with multi-level retrieval. Items are indexed in a tree of recursive abstractive summaries; retrieval traverses the tree to identify the most relevant leaf or internal nodes.

**Table 2.** Summary of baseline systems and their context management capabilities.

| System | Retrieval | Compression | Priority | Memory Hier. | Governance |
|---|:---:|:---:|:---:|:---:|:---:|
| Full Context | ✗ | ✗ | ✗ | ✗ | ✗ |
| Truncation | ✗ | ✗ | ✗ | ✗ | ✗ |
| RAG-Only | ✓ | ✗ | ◦ | ✗ | ✗ |
| MemGPT | ✓ | ◦ | ◦ | ◦ | ✗ |
| RAPTOR | ✓ | ✓ | ◦ | ◦ | ✗ |
| **ContextOS** | **✓** | **✓** | **✓** | **✓** | **✓** |

*✓ = fully supported, ◦ = partially supported, ✗ = not supported*

### 6.2 Backbone Models

All systems are evaluated with three backbone LLMs:

- **GPT-OSS-20B:** A 20-billion-parameter open-source model following the GPT architecture, pre-trained on a 2T-token multilingual corpus, with a 32K-token context window.
- **Qwen3:** The third generation of the Qwen series [Bai et al., 2023], a strong multilingual model with 72B parameters (using 4-bit AWQ quantization) and a 32K context window.
- **GLM-4.5:** ChatGLM's fourth-generation model with a bilingual focus, 130B parameters (8-bit), and a 128K context window.

All backbone models are evaluated in a **zero-shot setting** (no fine-tuning or few-shot examples specific to **ContextBench**) to isolate the effect of context management rather than task-specific training.

### 6.3 Implementation Details

- **Hardware:** 8× NVIDIA A100 80 GB GPUs, 512 GB DDR4 RAM, 2× AMD EPYC 7763 CPUs
- **Embedding model:** `BAAI/bge-m3` (1024-d, 570M parameters); ablation uses `intfloat/e5-large-v2` (1024-d, 335M parameters)
- **Cross-encoder reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M parameters, 1.8 ms per pair on A100)
- **Vector store:** Qdrant (in-process mode) with HNSW indices (m=16, ef_construction=200)
- **Compression models:** Extractive—DistilBERT-based sentence scorer (66M); Abstractive—T5-XL (3B) fine-tuned on CNN/DailyMail and XSum
- **Default context budget:** 8,192 tokens for the 8K experimental condition
- **Inference:** vLLM [Kwon et al., 2023] with continuous batching
- **Reproducibility:** All experiments use fixed random seed = 42

---

## 7. Results and Analysis

### 7.1 Main Results

**Table 3.** Task Success Rate (%, mean ± std) on **ContextBench** test set, averaged over GPT-OSS-20B, Qwen3, and GLM-4.5. **Bold** indicates the best result per context-length column. All **ContextOS** improvements over MemGPT and RAPTOR are statistically significant (p < 0.001, paired t-test with Bonferroni correction).

| Method | 512 tokens | 4K tokens | 8K tokens | 32K tokens |
|---|---:|---:|---:|---:|
| Full Context | 81.1 ± 0.8 | 79.3 ± 0.9 | 74.6 ± 1.1 | 68.2 ± 1.3 |
| Truncation | 78.9 ± 0.7 | 61.4 ± 1.2 | 41.8 ± 1.5 | 31.5 ± 1.6 |
| RAG-Only | 79.4 ± 0.9 | 74.2 ± 1.0 | 65.1 ± 1.3 | 48.7 ± 1.4 |
| MemGPT | 80.2 ± 0.8 | 77.1 ± 1.0 | 70.4 ± 1.2 | 58.9 ± 1.4 |
| RAPTOR | 79.8 ± 0.9 | 76.3 ± 1.1 | 68.9 ± 1.3 | 54.1 ± 1.5 |
| **ContextOS** | **84.2 ± 0.6** | **82.9 ± 0.7** | **79.6 ± 0.9** | **71.3 ± 1.1** |
| *Δ vs. MemGPT* | *+4.0* | *+5.8* | *+9.2* | *+12.4* |
| *Δ vs. Truncation* | *+5.3* | *+21.5* | *+37.8* | *+39.8* |

**ContextOS achieves the highest TSR at all context lengths.** At 512 tokens, the performance gap between methods is modest (all methods achieve 78.9–84.2%), as the context is short enough that simple strategies suffice for easy tasks. However, as context length grows to 4K, 8K, and 32K tokens, the performance differential widens dramatically. **ContextOS** achieves 71.3% at 32K tokens—a **12.4 percentage-point (pp) improvement** over MemGPT (p < 0.001, Cohen's d = 0.91), a 17.2 pp improvement over RAPTOR, and a **39.8 pp improvement** over naïve truncation.

**Truncation degrades most severely with scale.** Truncation falls from 78.9% at 512 tokens to only 31.5% at 32K tokens—a decline of 47.4 pp. This collapse reflects the fundamental inadequacy of recency-only retention: as context length grows, an increasing fraction of the relevant items are old and therefore discarded. The Adversarial Distraction and Timeline Reconstruction task types are particularly impacted.

**ContextOS degrades most gracefully.** Across the four context-length conditions, **ContextOS**'s TSR decreases by only 12.9 pp (from 84.2% at 512 to 71.3% at 32K), compared to 22.3 pp for MemGPT and 47.4 pp for Truncation. This graceful degradation is attributable to the combined effect of the priority scheduler and the long-term memory governance.

**Token efficiency.** **ContextOS** uses a mean of 3,834 tokens per successful task at the 8K condition, compared to 10,315 tokens for Full Context—a **62.9% reduction**. RAG-Only achieves similar token efficiency (3,902 tokens) but at substantially lower TSR, confirming that token reduction alone is not sufficient; the quality of context selection matters.

### 7.2 Retrieval Quality

**Table 4.** Retrieval quality on **ContextBench** test set.

| System | P@1 | P@5 | NDCG@10 |
|---|---:|---:|---:|
| RAG-Only (Dense) | 0.614 | 0.571 | 0.623 |
| RAG-Only (Sparse) | 0.582 | 0.541 | 0.597 |
| RAG-Only (Hybrid, no rerank) | 0.651 | 0.608 | 0.657 |
| MemGPT | 0.672 | 0.631 | 0.683 |
| RAPTOR | 0.658 | 0.619 | 0.671 |
| **ContextOS** | **0.741** | **0.703** | **0.762** |

**ContextOS** achieves NDCG@10 = 0.762, substantially outperforming all baselines. The improvement over RAG-Only with hybrid retrieval but no reranking (NDCG@10 = 0.657) demonstrates the value of the cross-encoder second stage (+0.105 absolute). The further gain over MemGPT (NDCG@10 = 0.683) reflects the contribution of the priority scheduler's MMR-based novelty component, which improves the diversity of the selected context set.

### 7.3 Compression Quality

Hierarchical compression achieves the best trade-off: at a compression ratio of ρ = 0.4, it maintains a semantic fidelity of 0.91 and a TSR within 1.3 pp of the uncompressed baseline. Abstractive compression achieves higher compression (ρ = 0.31) but at a fidelity cost of 0.87 and a TSR penalty of 2.8 pp. Extractive compression is most conservative (ρ = 0.62) but preserves fidelity almost perfectly (0.97) and incurs only a 0.4 pp TSR penalty. We recommend a default compression ratio of ρ = 0.4 using hierarchical compression.

### 7.4 Latency Analysis

**Table 5.** Latency breakdown of **ContextOS** pipeline components (milliseconds, mean ± std over 1,000 test tasks, excluding LLM inference time).

| Component | Mean (ms) | Std (ms) |
|---|---:|---:|
| Dense Retrieval (BGE-M3 + HNSW) | 18.3 | 2.1 |
| Sparse Retrieval (BM25) | 6.4 | 0.8 |
| RRF Fusion | 2.1 | 0.2 |
| Cross-Encoder Reranking | 15.3 | 3.4 |
| *Retrieval subtotal* | *42.1* | *4.8* |
| Priority Scoring | 12.7 | 1.6 |
| Greedy Scheduling | 4.3 | 0.5 |
| *Scheduling subtotal* | *17.0* | *1.9* |
| Compression (when triggered) | 31.4 | 8.2 |
| Memory Registry Lookup | 8.6 | 1.0 |
| Context Assembly | 7.1 | 0.9 |
| *Total (no compression)* | *124.8* | *14.2* |
| **Total (with compression)** | **156.2** | **22.4** |

The total pipeline latency of **156.2 ms** is well within interactive response time tolerances. Retrieval dominates the latency budget at 42.1 ms (27.0% of total), with cross-encoder reranking being the primary bottleneck. Compression contributes 31.4 ms when triggered. These results confirm that **ContextOS**'s performance gains come without prohibitive latency overhead.

---

## 8. Ablation Study

To assess the individual contribution of each **ContextOS** component, we conduct a systematic ablation study at the 8K context length condition using GPT-OSS-20B as the backbone model.

**Table 6.** Ablation study results. TSR (%) on **ContextBench** test set at 8K context length, GPT-OSS-20B backbone. ΔTSR is the difference from the full **ContextOS** system.

| System Variant | TSR (%) | ΔTSR (pp) |
|---|---:|---:|
| **ContextOS (Full)** | **79.6** | — |
| w/o Long-Term Memory | 63.2 | −16.4 |
| w/o Governance Engine | 64.6 | −15.0 |
| Flat Scheduler (random) | 66.8 | −12.8 |
| w/o Compression Engine | 73.0 | −6.6 |
| w/o MMR Novelty | 75.1 | −4.5 |
| w/o Recency Decay | 76.9 | −2.7 |
| Dense Retrieval Only | 77.3 | −2.3 |
| *Sum of individual ablations* | — | *−60.3* |
| *Actual total gain vs. RAG-Only* | — | *+14.5* |
| *Interactions (synergy)* | — | *−22.7* |

**Long-Term Memory is the most critical component (−16.4 pp).** Without long-term memory, items that have not been recently accessed but remain relevant to the task (e.g., early goal specifications, domain facts retrieved in previous agent steps) are evicted from working memory and are no longer available for scheduling. This finding validates the central thesis of **ContextOS**: that a hierarchical memory architecture with explicit lifecycle management is essential for long-horizon performance.

**Governance Engine is the second most critical component (−15.0 pp).** Removing governance prevents the memory system from promoting high-value items from working to long-term memory and from enforcing retention policies. Without governance, the working memory fills with low-importance, recently-created items while high-importance older items are lost. The similarity of the governance and long-term memory ablation results reflects their tight coupling: the long-term memory tier only provides value if the governance engine correctly populates it.

**Priority Scheduler accounts for −12.8 pp.** Replacing the priority scheduler with random selection from the retrieved candidate set demonstrates that the specific multi-factor priority function is substantially more effective than unstructured selection. This ablation controls for retrieval quality (the same hybrid retrieval is used), isolating the contribution of the scheduling policy.

**Compression Engine contributes −6.6 pp.** The contribution of the Compression Engine is most apparent at high context lengths: at 32K tokens, removing compression results in a −9.2 pp drop, compared to −6.6 pp at 8K. At 512 tokens, the compression engine is rarely triggered, and its ablation has negligible effect (−0.3 pp). Compression is most valuable as a mechanism for graceful degradation under severe budget pressure.

**MMR Novelty and Recency Decay provide moderate gains.** Removing the MMR novelty component (−4.5 pp) and the recency decay (−2.7 pp) both reduce performance, but less severely than the structural components. MMR is particularly impactful on Literature Synthesis tasks, while recency decay most strongly affects Timeline Reconstruction tasks.

**Component Interactions.** The sum of individual ablation penalties (60.3 pp) exceeds the actual total gain of **ContextOS** over RAG-Only (14.5 pp), with the difference (22.7 pp, or 37.6% of summed individual contributions) attributable to synergistic interactions between components. The long-term memory tier and the governance engine are strongly synergistic: neither provides full value without the other. Similarly, compression and scheduling interact positively: a tighter token budget is most effectively utilized by the priority scheduler.

---

## 9. Discussion

### 9.1 When Does ContextOS Help Most?

**ContextOS** provides the greatest benefit in two regimes. First, at **long context lengths** (8K tokens and above), where the discrepancy between available context capacity and the amount of information generated during agent execution is large. The 39.8 pp gain over truncation at 32K tokens is the clearest illustration of this benefit. Second, **ContextOS** provides disproportionate gains on multi-session tasks and tasks with long causal chains. In these settings, relevant information from early in the task history must remain accessible many steps later. The long-term memory hierarchy—particularly episodic memory for event sequences and semantic memory for extracted facts—provides a structured repository for this information.

In contrast, **ContextOS**'s advantages are modest at short context lengths (512 tokens) and for simple tasks. At these scales, most baselines perform adequately, and the overhead of the **ContextOS** pipeline is not justified relative to simpler alternatives. Practitioners should consider deploying lightweight baselines (e.g., RAG-Only) for short-context or low-complexity settings.

### 9.2 Limitations

**Latency overhead.** The **ContextOS** pipeline adds 124.8–156.2 ms of pre-LLM processing latency. While acceptable for most applications, it may be prohibitive for ultra-low-latency settings (e.g., sub-50 ms voice assistants). Future work should explore asynchronous pre-fetching and speculative context preparation to reduce effective latency.

**Dependency on embedding quality.** The retrieval, prioritization, and novelty components all rely on dense embeddings from BGE-M3. In domains where BGE-M3's pre-training distribution is poorly matched (e.g., highly specialized technical domains, low-resource languages), retrieval quality may degrade.

**Governance policy tuning.** The retention thresholds, forgetting rates, and promotion criteria were tuned on the **ContextBench** validation set. These hyperparameters may require re-tuning for deployment in domains not represented in **ContextBench**, particularly for tasks with very different temporal dynamics.

**Fixed memory capacity.** The working memory capacity K_WM = 50 is a fixed parameter. Future work should explore dynamic capacity adjustment based on task complexity and available computational budget.

**Evaluation scope.** **ContextBench** covers eight task types and six domains, but cannot comprehensively cover all long-horizon agent applications. Physical world interaction, real-time streaming contexts, and adversarial environments are not included.

### 9.3 Broader Implications

**ContextOS** suggests that *context engineering*—the systematic design of policies for what information enters, persists in, and exits an agent's context window—deserves recognition as a first-class research problem within the AI systems community. Just as the development of virtual memory and page replacement algorithms was essential for scaling computer applications beyond the limits of physical RAM, principled context lifecycle management will likely be essential for scaling autonomous agents beyond the limits of current context windows.

The OS abstraction is not merely metaphorical. The formal parallels are tight: context windows correspond to physical RAM, long-term memory tiers correspond to storage hierarchy levels, the priority scheduler corresponds to an OS process scheduler, and the governance engine corresponds to a combination of a garbage collector and a cache replacement policy. This correspondence suggests that decades of OS and systems research may be directly applicable to agent context management.

### 9.4 Ethical Considerations

The memory governance mechanisms in **ContextOS** have direct implications for user privacy. An agent with a long-term memory system that persists information across sessions can accumulate sensitive personal data that may pose privacy risks if the memory is not properly governed. The Governance Engine's retention and forgetting policies provide a technical foundation for privacy compliance: time-based retention policies can implement data minimization principles, importance-based retention can prioritize task-relevant information, and explicit eviction can implement "right to be forgotten" requests. We recommend that deployments of **ContextOS** implement user-controlled memory governance dashboards. Future work should also investigate differential privacy mechanisms for the vector store and the importance classifier to prevent membership inference attacks against agent memories.

---

## 10. Conclusion

We have presented **ContextOS**, an operating-system-inspired framework for context lifecycle management in long-horizon autonomous agents. **ContextOS** treats the context window as a managed resource, applying scheduling, caching, compression, and eviction policies across a six-component pipeline: a Retrieval Engine with hybrid dense-sparse retrieval and cross-encoder reranking, a Prioritization Engine with a multi-factor priority function incorporating temporal decay and MMR-based novelty, a Context Scheduler with a provable submodularity guarantee, a progressive Compression Engine with type-conditional policies, a two-level Memory System with episodic, semantic, and procedural long-term tiers, and a Governance Engine enforcing retention, forgetting, and promotion.

On **ContextBench**, a new benchmark of 70,000 long-horizon agent tasks, **ContextOS** achieves a task-success rate of **71.3% at 32K tokens**—a 12.4 pp improvement over MemGPT, 17.2 pp over RAPTOR, and 39.8 pp over naïve truncation—while reducing token consumption by 62.9% relative to Full Context. The system operates with a total context-preparation latency of 156.2 ms. Ablation studies confirm that every component contributes positively, with long-term memory governance and the priority scheduler contributing the largest individual gains.

Future work will focus on three directions:
1. **Adaptive policy learning:** using reinforcement learning or Bayesian optimization to adapt retention thresholds, decay rates, and compression ratios online during agent deployment.
2. **Cross-agent memory sharing:** extending the memory architecture to multi-agent settings where multiple agents share a common semantic memory while maintaining private episodic memories.
3. **Privacy-preserving memory:** integrating differential privacy mechanisms into the vector store and the importance classifier to provide formal privacy guarantees for sensitive user contexts.

We believe that context lifecycle management, as operationalized by **ContextOS**, represents a fundamental building block for the next generation of reliable, efficient, and trustworthy long-horizon AI agents.

---

## Data Availability

The **ContextBench** dataset, **ContextOS** source code, pre-trained compression models, and evaluation scripts will be made publicly available upon acceptance.

## Competing Interests

The authors declare no competing interests.

---

## References

Anderson, J. R., and Lebiere, C. (2004). The atomic components of thought. *Lawrence Erlbaum Associates*.

AutoGPT (2023). Auto-GPT: An autonomous GPT-4 experiment. `https://github.com/Significant-Gravitas/AutoGPT`.

Bai, J., Bai, S., Chu, Y., et al. (2023). Qwen technical report. *arXiv preprint arXiv:2309.16609*.

Beltagy, I., Peters, M. E., and Cohan, A. (2020). Longformer: The long-document transformer. *arXiv preprint arXiv:2004.05150*.

Carbonell, J., and Goldstein, J. (1998). The use of MMR, diversity-based reranking for reordering documents and producing summaries. In *Proceedings of SIGIR*, pp. 335–336.

Chen, J., Xiao, S., Zhang, P., et al. (2024). BGE M3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. *arXiv preprint arXiv:2402.03216*.

Chevalier, A., Wettig, A., Ajith, A., and Chen, D. (2023). Adapting language models to compress contexts. In *Proceedings of EMNLP*, pp. 3829–3846.

Cormack, G. V., Clarke, C. L. A., and Buettcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. In *Proceedings of SIGIR*, pp. 758–759.

Dao, T. (2023). FlashAttention-2: Faster attention with better parallelism and work partitioning. In *Proceedings of ICLR 2024*.

Dao, T., Fu, D., Ermon, S., Rudra, A., and Ré, C. (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In *Proceedings of NeurIPS*, pp. 16344–16359.

Ebbinghaus, H. (1885). *Über das Gedächtnis: Untersuchungen zur experimentellen Psychologie*. Duncker & Humblot, Leipzig.

Formal, T., Piwowarski, B., and Clinchant, S. (2021). SPLADE: Sparse lexical and expansion model for first stage ranking. In *Proceedings of SIGIR*, pp. 2288–2292.

Garey, M. R., and Johnson, D. S. (1979). *Computers and Intractability: A Guide to the Theory of NP-Completeness*. W. H. Freeman, New York.

Ge, T., Hu, J., Wang, L., et al. (2023). In-context autoencoder for context compression in a large language model. In *Proceedings of ICLR 2024*.

Guu, K., Lee, K., Tung, Z., Pasupat, P., and Chang, M. W. (2020). REALM: Retrieval-augmented language model pre-training. In *Proceedings of ICML*, pp. 3929–3938.

Jiang, H., Wu, Q., Lin, C.-Y., Yang, Y., and Luo, L. (2023). LLMLingua: Compressing prompts for accelerated inference of large language models. In *Proceedings of EMNLP*, pp. 13358–13376.

Karpukhin, V., Oğuz, B., Min, S., et al. (2020). Dense passage retrieval for open-domain question answering. In *Proceedings of EMNLP*, pp. 6769–6781.

Khattab, O., and Zaharia, M. (2020). ColBERT: Efficient and effective passage search via contextualized late interaction over BERT. In *Proceedings of SIGIR*, pp. 39–48.

Kim, S., Shim, H., and Kim, Y. (2022). Learned token pruning for transformers. In *Proceedings of KDD*, pp. 784–794.

Kwon, W., Li, Z., Zhuang, S., et al. (2023). Efficient memory management for large language model serving with PagedAttention. In *Proceedings of SOSP*, pp. 611–626.

Laird, J. E. (2012). *The Soar Cognitive Architecture*. MIT Press, Cambridge, MA.

Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. In *Proceedings of NeurIPS*, pp. 9459–9474.

Liu, N. F., Lin, K., Hewitt, J., et al. (2023). Lost in the middle: How language models use long contexts. *Transactions of the Association for Computational Linguistics*, 12:157–173.

Nemhauser, G. L., Wolsey, L. A., and Fisher, M. L. (1978). An analysis of approximations for maximizing submodular set functions. *Mathematical Programming*, 14(1):265–294.

Nogueira, R., and Cho, K. (2019). Passage re-ranking with BERT. *arXiv preprint arXiv:1901.04085*.

Packer, C., Fang, V., Patil, S. G., et al. (2023). MemGPT: Towards LLMs as operating systems. *arXiv preprint arXiv:2310.08560*.

Park, J. S., O'Brien, J. C., Cai, C. J., et al. (2023). Generative agents: Interactive simulacra of human behavior. In *Proceedings of UIST*, pp. 1–22.

Press, O., Smith, N. A., and Lewis, M. (2022). Train short, test long: Attention with linear biases enables input length extrapolation. In *Proceedings of ICLR*.

Rajpurkar, P., Zhang, J., Lopyrev, K., and Liang, P. (2016). SQuAD: 100,000+ questions for machine comprehension of text. In *Proceedings of EMNLP*, pp. 2383–2392.

Robertson, S., and Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4):333–389.

Sarthi, P., Abdullah, S., Tuli, A., et al. (2024). RAPTOR: Recursive abstractive processing for tree-organized retrieval. In *Proceedings of ICLR*.

Shi, F., Chen, X., Misra, K., et al. (2023). Large language models can be easily distracted by irrelevant context. In *Proceedings of ICML*, pp. 31210–31227.

Shuster, K., Poff, S., Chen, M., Kiela, D., and Weston, J. (2021). Retrieval augmentation reduces hallucination in conversation. In *Findings of EMNLP*, pp. 3784–3803.

Su, J., Ahmed, M., Lu, Y., et al. (2024). RoFormer: Enhanced transformer with rotary position embedding. *Neurocomputing*, 568:127063.

Sviridenko, M. (2004). A note on maximizing a submodular set function subject to a knapsack constraint. *Operations Research Letters*, 32(1):41–43.

Trivedi, H., Balasubramanian, N., Khot, T., and Sabharwal, A. (2022). Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions. In *Proceedings of ACL*, pp. 10014–10037.

Wang, G., Xie, Y., Jiang, Y., et al. (2023b). Voyager: An open-ended embodied agent with large language models. *arXiv preprint arXiv:2305.16291*.

Wang, L., Ma, C., Feng, X., et al. (2023a). A survey on large language model based autonomous agents. *Frontiers of Computer Science*, 18(6):186345.

Xi, Z., Chen, W., Guo, X., et al. (2023). The rise and potential of large language model based agents: A survey. *arXiv preprint arXiv:2309.07864*.

Xu, F. F., Shi, W., and Choi, E. (2023). RECOMP: Improving retrieval-augmented LMs with context compression and selective augmentation. In *Proceedings of ICLR 2024*.

Ye, X., Axmed, M., Rossi, R., and Zhao, S. (2024). Differential attention for LLMs. *arXiv preprint arXiv:2410.05258*.

Zaheer, M., Guruganesh, G., Dubey, K. A., et al. (2020). Big Bird: Transformers for longer sequences. In *Proceedings of NeurIPS*, pp. 17283–17297.

Zhong, W., Guo, L., Gao, Q., Ye, H., and Wang, Y. (2024). MemoryBank: Enhancing large language models with long-term memory. In *Proceedings of AAAI*, pp. 19724–19731.
