# ContextOS

> **An Operating System for Context-Aware Autonomous Agents**

[![Research](https://img.shields.io/badge/Research-Context%20Engineering-blue)]()
[![Open Source](https://img.shields.io/badge/Open%20Source-Community-green)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-red)]()

ContextOS is an open-source research framework for **context lifecycle management** in autonomous AI agents.

It treats context as a first-class system resource, providing mechanisms for:

* Retrieval
* Compression
* Prioritization
* Scheduling
* Memory Management
* Governance

The goal is to enable reliable **long-horizon agents** operating across large contexts, persistent memory, and multi-session workflows.

---

## Core Idea

> Better context management produces better agents.

Just as operating systems manage CPU, memory, and storage, ContextOS manages:

| Operating System | ContextOS          |
| ---------------- | ------------------ |
| CPU Scheduler    | Context Scheduler  |
| RAM              | Working Memory     |
| Storage          | Long-Term Memory   |
| Memory Manager   | Context Governance |
| Cache            | Retrieval Layer    |

---

## Architecture

```text
User
 │
 ▼
Context Orchestrator
 │
 ├── Retrieval Engine
 ├── Compression Engine
 ├── Governance Engine
 │
 ▼
Context Scheduler
 │
 ▼
Working Memory
 │
 ▼
Long-Term Memory
 │
 ▼
Agent Runtime
```

---

## Research Areas

* Context Engineering
* Agent Memory Systems
* Context Scheduling
* Context Governance
* Long-Horizon Agents
* Multi-Agent Memory Sharing

---

## Core Modules

#### Retrieval Engine : Semantic retrieval, reranking, and memory access.

#### Compression Engine : Context reduction while preserving information.

#### Prioritization Engine : Ranks context by relevance, importance, recency, and novelty.

#### Working Memory : Short-term storage for goals, plans, observations, and tool outputs.

#### Long-Term Memory : Episodic, semantic, and procedural memory systems.

#### Governance Engine : Retention, forgetting, promotion, and compression policies.

#### Context Scheduler : Determines what enters context and when.

---

## Tech Stack

### Frameworks

* LangGraph
* FastAPI
* Next.js

### Storage

* PostgreSQL + pgvector
* Qdrant

### Observability

* Langfuse
* OpenTelemetry
* MLflow

### Models

* GPT-OSS-20B
* GPT-OSS-120B
* DeepSeek-V3.1
* Qwen3
* Qwen3-Coder
* GLM-4.5
* BGE-M3
* E5-Large-V2

---

## Research Questions

1. What information should enter an agent's context?
2. How should memories be prioritized?
3. When should memories be compressed or forgotten?
4. Can context scheduling improve agent performance?
5. How can context quality be measured?
6. How can agents operate across million-token histories?

---

## Repository Structure

```text
context-os/
├── apps/
├── core/
├── memory/
├── benchmarks/
├── experiments/
├── datasets/
├── docs/
├── papers/
├── notebooks/
├── examples/
└── tests/
```

---

## License

Apache License 2.0
