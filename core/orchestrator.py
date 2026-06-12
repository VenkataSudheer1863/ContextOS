"""
ContextOS Core Orchestrator
Coordinates all context management subsystems.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
import yaml
import os


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    TOOL_OUTPUT = "tool_output"
    OBSERVATION = "observation"
    GOAL = "goal"
    PLAN = "plan"


@dataclass
class ContextItem:
    id: str
    content: str
    memory_type: MemoryType
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5
    relevance: float = 0.0
    access_count: int = 0
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    def update_access(self):
        self.access_count += 1
        self.metadata["last_accessed"] = time.time()


@dataclass
class AgentState:
    agent_id: str
    current_task: str
    task_type: str = "general"
    session_id: str = ""
    iteration: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextStats:
    total_items_retrieved: int = 0
    total_items_compressed: int = 0
    total_tokens_used: int = 0
    total_tokens_saved: int = 0
    compression_ratio: float = 0.0
    retrieval_precision: float = 0.0
    scheduling_overhead_ms: float = 0.0
    governance_evictions: int = 0


@dataclass
class ProcessedContext:
    items: List[ContextItem]
    total_tokens: int
    stats: ContextStats
    compressed: bool
    context_string: str


class ContextOrchestrator:
    """
    Central coordinator for context lifecycle management.
    Integrates retrieval, scheduling, compression, and governance
    into a unified context processing pipeline.
    """

    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict] = None):
        if config is None:
            config = self._load_config(config_path)
        self.config = config
        self._stats = ContextStats()
        self._session_stats: Dict[str, ContextStats] = {}
        self._initialized = False
        logger.info("ContextOrchestrator initializing...")

    def _load_config(self, path: Optional[str]) -> Dict:
        default = os.path.join(os.path.dirname(__file__), "..", "config", "default_config.yaml")
        target = path or default
        if os.path.exists(target):
            with open(target) as f:
                return yaml.safe_load(f)
        return self._default_config()

    def _default_config(self) -> Dict:
        return {
            "context": {"max_tokens": 8192, "compression_ratio": 0.4, "retrieval_top_k": 20},
            "scheduler": {"algorithm": "greedy",
                          "weights": {"relevance": 0.4, "recency": 0.25, "importance": 0.2, "novelty": 0.15},
                          "decay_lambda": 0.0001},
            "governance": {"forgetting_threshold": 0.1, "promotion_threshold": 0.7, "compression_trigger": 0.8},
        }

    def initialize(self):
        """Lazy initialization of heavy components."""
        if self._initialized:
            return
        try:
            from .retrieval_engine import RetrievalEngine
            from .compression_engine import CompressionEngine
            from .prioritization_engine import PrioritizationEngine
            from .governance_engine import GovernanceEngine
            from .scheduler import ContextScheduler
            from ..memory.working_memory import WorkingMemory
            from ..memory.long_term_memory import LongTermMemory

            self.retrieval = RetrievalEngine(self.config)
            self.compression = CompressionEngine(self.config)
            self.prioritization = PrioritizationEngine(self.config)
            self.governance = GovernanceEngine(self.config)
            self.scheduler = ContextScheduler(self.config)
            self.working_memory = WorkingMemory(self.config)
            self.long_term_memory = LongTermMemory(self.config)
            self._initialized = True
            logger.info("ContextOrchestrator fully initialized.")
        except ImportError as e:
            logger.warning(f"Some components not available: {e}. Running in mock mode.")
            self._initialized = True

    def process_query(
        self,
        query: str,
        agent_state: AgentState,
        candidate_items: Optional[List[ContextItem]] = None,
    ) -> ProcessedContext:
        """
        Main pipeline: retrieve -> prioritize -> schedule -> compress -> govern.
        """
        t0 = time.perf_counter()
        self.initialize()

        # 1. Retrieve candidates if not provided
        if candidate_items is None:
            if hasattr(self, "long_term_memory"):
                candidate_items = self.long_term_memory.retrieve_for_query(
                    query, top_k=self.config["context"]["retrieval_top_k"]
                )
            else:
                candidate_items = []

        # Add working memory items
        if hasattr(self, "working_memory"):
            wm_items = self.working_memory.get_all()
            candidate_items = candidate_items + wm_items

        # 2. Score and prioritize
        if hasattr(self, "prioritization") and candidate_items:
            query_emb = None
            if hasattr(self, "retrieval"):
                try:
                    query_emb = self.retrieval.encode(query)
                except Exception:
                    pass
            candidate_items = self.prioritization.score_all(candidate_items, query_emb, query)

        # 3. Schedule: select items within token budget
        max_tokens = self.config["context"]["max_tokens"]
        if hasattr(self, "scheduler"):
            scheduled_items, token_count = self.scheduler.schedule(candidate_items, max_tokens)
        else:
            scheduled_items = candidate_items[:20]
            token_count = sum(len(i.content.split()) * 4 // 3 for i in scheduled_items)

        # 4. Compress if over budget
        compressed = False
        trigger = self.config.get("governance", {}).get("compression_trigger", 0.8)
        if token_count > max_tokens * trigger and hasattr(self, "compression"):
            scheduled_items = self.compression.compress(scheduled_items, int(max_tokens * trigger))
            token_count = sum(len(i.content.split()) * 4 // 3 for i in scheduled_items)
            compressed = True
            self._stats.total_items_compressed += len(scheduled_items)

        # 5. Apply governance
        if hasattr(self, "governance"):
            scheduled_items = self.governance.apply_context_policies(scheduled_items)

        # 6. Build context string
        context_str = self._build_context_string(scheduled_items, agent_state)

        overhead = (time.perf_counter() - t0) * 1000
        stats = ContextStats(
            total_items_retrieved=len(candidate_items),
            total_items_compressed=len(scheduled_items) if compressed else 0,
            total_tokens_used=token_count,
            compression_ratio=1 - token_count / max(token_count * 2, 1) if compressed else 1.0,
            scheduling_overhead_ms=overhead,
        )
        self._stats.scheduling_overhead_ms += overhead
        self._stats.total_tokens_used += token_count

        logger.debug(
            f"Context processed: {len(scheduled_items)} items, {token_count} tokens, {overhead:.1f}ms"
        )
        return ProcessedContext(
            items=scheduled_items,
            total_tokens=token_count,
            stats=stats,
            compressed=compressed,
            context_string=context_str,
        )

    def store_observation(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.OBSERVATION,
        importance: float = 0.5,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Store a new observation in memory."""
        self.initialize()
        import uuid

        item = ContextItem(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata or {},
        )
        item.token_count = len(content.split()) * 4 // 3
        if hasattr(self, "working_memory"):
            self.working_memory.add(item, priority=importance)
        if hasattr(self, "governance") and hasattr(self, "long_term_memory"):
            if self.governance.should_promote(item):
                self.long_term_memory.store(item)
        return item.id

    def run_governance_cycle(self) -> List[ContextItem]:
        """Periodic governance: retention, forgetting, promotion."""
        self.initialize()
        if hasattr(self, "governance") and hasattr(self, "long_term_memory"):
            evicted = self.governance.run_forgetting_cycle(self.long_term_memory)
            self._stats.governance_evictions += len(evicted)
            logger.info(f"Governance cycle: evicted {len(evicted)} items.")
            return evicted
        return []

    def _build_context_string(self, items: List[ContextItem], state: AgentState) -> str:
        sections: Dict[MemoryType, List[str]] = {t: [] for t in MemoryType}
        for item in items:
            sections[item.memory_type].append(item.content)

        parts = [f"# Agent Context for Task: {state.current_task}"]
        order = [
            MemoryType.GOAL,
            MemoryType.PLAN,
            MemoryType.OBSERVATION,
            MemoryType.TOOL_OUTPUT,
            MemoryType.EPISODIC,
            MemoryType.SEMANTIC,
        ]
        labels = {
            MemoryType.GOAL: "## Goals",
            MemoryType.PLAN: "## Plans",
            MemoryType.OBSERVATION: "## Recent Observations",
            MemoryType.TOOL_OUTPUT: "## Tool Outputs",
            MemoryType.EPISODIC: "## Relevant Episodes",
            MemoryType.SEMANTIC: "## Background Knowledge",
        }
        for mtype in order:
            if sections[mtype]:
                parts.append(labels.get(mtype, f"## {mtype.value.title()}"))
                parts.extend(f"- {c}" for c in sections[mtype])
        return "\n".join(parts)

    def get_stats(self) -> ContextStats:
        return self._stats
