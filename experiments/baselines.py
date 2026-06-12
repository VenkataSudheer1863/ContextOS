"""
ContextOS Baseline Context Management Methods
=============================================
Self-contained implementations of all baseline methods used in experiments.
Each baseline receives a query, a list of ContextItem candidates, and a
max_tokens budget, and returns the selected (and possibly transformed)
list of ContextItem objects.
"""
from __future__ import annotations

import math
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Lightweight ContextItem definition (mirrors core.orchestrator.ContextItem
# but has no external dependencies so baselines are fully standalone).
# ---------------------------------------------------------------------------

@dataclass
class ContextItem:
    """Minimal context item used by all baseline methods."""
    id: str
    content: str
    memory_type: str = "episodic"
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5
    relevance: float = 0.0
    access_count: int = 0
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        if self.token_count == 0 and self.content:
            self.token_count = self._estimate_tokens(self.content)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """GPT-style approximation: ~4 chars per token."""
        return max(1, len(text) // 4)

    def age_seconds(self) -> float:
        return time.time() - self.timestamp


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseContextMethod(ABC):
    """
    Abstract base for all context management strategies.
    Subclasses implement the selection / transformation pipeline and must
    return a list of ContextItem objects whose total token_count fits within
    *max_tokens*.
    """

    @abstractmethod
    def process(
        self,
        query: str,
        candidate_items: List[ContextItem],
        max_tokens: int,
    ) -> List[ContextItem]:
        """
        Select and/or transform candidate_items to fit within max_tokens.

        Parameters
        ----------
        query : str
            The current user / agent query driving context selection.
        candidate_items : list[ContextItem]
            Pool of items available for inclusion.
        max_tokens : int
            Hard upper bound on the total token budget.

        Returns
        -------
        list[ContextItem]
            Ordered list of items to include in the context window.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this method."""

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _total_tokens(items: List[ContextItem]) -> int:
        return sum(i.token_count for i in items)

    @staticmethod
    def _truncate_to_budget(
        items: List[ContextItem], max_tokens: int
    ) -> List[ContextItem]:
        """Greedily include items until budget is exhausted."""
        selected: List[ContextItem] = []
        used = 0
        for item in items:
            if used + item.token_count <= max_tokens:
                selected.append(item)
                used += item.token_count
        return selected

    @staticmethod
    def _keyword_score(query: str, content: str) -> float:
        """
        Simple TF-based keyword overlap score in absence of an embedder.
        Returns a value in [0, 1].
        """
        q_tokens = set(re.findall(r"\w+", query.lower()))
        c_tokens = re.findall(r"\w+", content.lower())
        if not q_tokens or not c_tokens:
            return 0.0
        matches = sum(1 for t in c_tokens if t in q_tokens)
        return min(matches / len(q_tokens), 1.0)

    @staticmethod
    def _recency_score(item: ContextItem, decay_lambda: float = 0.0001) -> float:
        """Exponential recency decay."""
        return math.exp(-decay_lambda * item.age_seconds())


# ---------------------------------------------------------------------------
# 1. FullContextMethod
# ---------------------------------------------------------------------------

class FullContextMethod(BaseContextMethod):
    """
    Returns all candidate items in their original order up to the token
    limit.  No selection, ranking, or compression is applied.  This
    represents the naive 'stuff everything into the context window' baseline.
    """

    @property
    def name(self) -> str:
        return "FullContext"

    def process(
        self,
        query: str,
        candidate_items: List[ContextItem],
        max_tokens: int,
    ) -> List[ContextItem]:
        return self._truncate_to_budget(candidate_items, max_tokens)


# ---------------------------------------------------------------------------
# 2. TruncationMethod (First-K)
# ---------------------------------------------------------------------------

class TruncationMethod(BaseContextMethod):
    """
    Keeps only the first K tokens worth of items.  Items are taken in
    chronological / insertion order (oldest first), simulating the 'just
    truncate the prompt' approach used in many production systems.
    """

    def __init__(self, keep_newest: bool = False):
        """
        Parameters
        ----------
        keep_newest : bool
            If True, sort by timestamp descending (newest first) before
            truncating.  Default is oldest-first (keep_newest=False).
        """
        self.keep_newest = keep_newest

    @property
    def name(self) -> str:
        return "TruncationFirstK"

    def process(
        self,
        query: str,
        candidate_items: List[ContextItem],
        max_tokens: int,
    ) -> List[ContextItem]:
        if not candidate_items:
            return []

        ordered = sorted(
            candidate_items,
            key=lambda x: x.timestamp,
            reverse=self.keep_newest,
        )
        return self._truncate_to_budget(ordered, max_tokens)


# ---------------------------------------------------------------------------
# 3. RAGOnlyMethod
# ---------------------------------------------------------------------------

class RAGOnlyMethod(BaseContextMethod):
    """
    Pure dense-retrieval baseline: rank items by relevance to the query
    (keyword overlap when no embedder is available) and include the top-K
    items within the token budget.  No scheduling, compression, or governance
    is applied.
    """

    def __init__(self, use_importance_tiebreak: bool = True):
        self.use_importance_tiebreak = use_importance_tiebreak

    @property
    def name(self) -> str:
        return "RAGOnly"

    def process(
        self,
        query: str,
        candidate_items: List[ContextItem],
        max_tokens: int,
    ) -> List[ContextItem]:
        if not candidate_items:
            return []

        # Score each item
        scored: List[Tuple[float, ContextItem]] = []
        for item in candidate_items:
            # Use pre-computed relevance if available, otherwise compute
            rel = item.relevance if item.relevance > 0 else self._keyword_score(
                query, item.content
            )
            tiebreak = item.importance if self.use_importance_tiebreak else 0.0
            scored.append((rel + 0.01 * tiebreak, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        ranked = [item for _, item in scored]
        return self._truncate_to_budget(ranked, max_tokens)


# ---------------------------------------------------------------------------
# 4. SummarizationOnlyMethod
# ---------------------------------------------------------------------------

class SummarizationOnlyMethod(BaseContextMethod):
    """
    Compresses every candidate item via extractive summarization to fit
    within the token budget.  Simulates systems that solely rely on
    summarization pipelines without retrieval-based selection.

    In the absence of a live LLM, compression is simulated by retaining
    only the first *ratio* fraction of each item's content (sentence-level).
    """

    def __init__(self, compression_ratio: float = 0.40):
        if not 0.0 < compression_ratio <= 1.0:
            raise ValueError("compression_ratio must be in (0, 1]")
        self.compression_ratio = compression_ratio

    @property
    def name(self) -> str:
        return "SummarizationOnly"

    def process(
        self,
        query: str,
        candidate_items: List[ContextItem],
        max_tokens: int,
    ) -> List[ContextItem]:
        if not candidate_items:
            return []

        compressed = [self._compress_item(item) for item in candidate_items]
        # After compression, greedily fill budget
        return self._truncate_to_budget(compressed, max_tokens)

    def _compress_item(self, item: ContextItem) -> ContextItem:
        """Return a new ContextItem with content shrunk by compression_ratio."""
        sentences = re.split(r"(?<=[.!?])\s+", item.content.strip())
        keep_n = max(1, math.ceil(len(sentences) * self.compression_ratio))
        compressed_content = " ".join(sentences[:keep_n])

        import copy
        new_item = copy.copy(item)
        new_item.content = compressed_content
        new_item.token_count = ContextItem._estimate_tokens(compressed_content)
        new_item.metadata = dict(item.metadata)
        new_item.metadata["compressed"] = True
        new_item.metadata["original_tokens"] = item.token_count
        return new_item


# ---------------------------------------------------------------------------
# 5. MemGPTMethod
# ---------------------------------------------------------------------------

class MemGPTMethod(BaseContextMethod):
    """
    Simulates MemGPT's hierarchical memory approach (Packer et al., 2023).

    Memory tiers
    ------------
    - main_ctx  : Items currently in the LLM context (working set).
    - archival  : Long-term storage outside the context window.
    - recall    : Conversation history buffer.

    Selection logic
    ---------------
    1. High-importance items are always promoted to main_ctx.
    2. Remaining budget is filled with recall (recent) items.
    3. Archival items are retrieved on relevance only when budget permits.
    4. If over budget, the least important items are swapped to archival.
    """

    IMPORTANCE_THRESHOLD: float = 0.7   # items always kept in main context
    RECALL_FRACTION: float = 0.30        # fraction of budget reserved for recent
    ARCHIVAL_FRACTION: float = 0.20      # fraction of budget for archival retrieval

    def __init__(
        self,
        importance_threshold: float = IMPORTANCE_THRESHOLD,
        recall_fraction: float = RECALL_FRACTION,
        archival_fraction: float = ARCHIVAL_FRACTION,
    ):
        self.importance_threshold = importance_threshold
        self.recall_fraction = recall_fraction
        self.archival_fraction = archival_fraction

    @property
    def name(self) -> str:
        return "MemGPT"

    def process(
        self,
        query: str,
        candidate_items: List[ContextItem],
        max_tokens: int,
    ) -> List[ContextItem]:
        if not candidate_items:
            return []

        # --- Tier assignment ---
        main_ctx: List[ContextItem] = []
        recall: List[ContextItem] = []
        archival: List[ContextItem] = []

        # Sort by timestamp descending to identify recent items
        by_recency = sorted(candidate_items, key=lambda x: x.timestamp, reverse=True)
        recent_ids = {item.id for item in by_recency[:max(1, len(by_recency) // 3)]}

        for item in candidate_items:
            if item.importance >= self.importance_threshold:
                main_ctx.append(item)
            elif item.id in recent_ids:
                recall.append(item)
            else:
                archival.append(item)

        # --- Budget allocation ---
        recall_budget = int(max_tokens * self.recall_fraction)
        archival_budget = int(max_tokens * self.archival_fraction)
        main_budget = max_tokens - recall_budget - archival_budget

        selected: List[ContextItem] = []

        # Fill main context (high-importance)
        main_ctx.sort(key=lambda x: x.importance, reverse=True)
        selected.extend(self._truncate_to_budget(main_ctx, main_budget))

        used = self._total_tokens(selected)
        remaining = max_tokens - used

        # Fill recall (recent items)
        recall.sort(key=lambda x: x.timestamp, reverse=True)
        recall_selected = self._truncate_to_budget(
            recall, min(recall_budget, remaining)
        )
        selected.extend(recall_selected)
        used = self._total_tokens(selected)
        remaining = max_tokens - used

        # Fill archival (relevance-ranked)
        if remaining > 0 and archival:
            scored_archival = sorted(
                archival,
                key=lambda x: (
                    x.relevance if x.relevance > 0
                    else self._keyword_score(query, x.content)
                ),
                reverse=True,
            )
            archival_selected = self._truncate_to_budget(
                scored_archival, min(archival_budget, remaining)
            )
            selected.extend(archival_selected)

        return selected


# ---------------------------------------------------------------------------
# 6. RAPTORMethod
# ---------------------------------------------------------------------------

class RAPTORMethod(BaseContextMethod):
    """
    Simulates RAPTOR's recursive tree-based summarization approach
    (Sarthi et al., 2024).

    Algorithm
    ---------
    1. Cluster items into semantic groups (simulated by memory_type).
    2. Summarize each cluster into a leaf node (extractive, ratio-based).
    3. If the total summary tokens still exceed budget, recursively merge
       and re-summarize pairs of clusters (bottom-up tree construction).
    4. Traverse the tree top-down to fill the context budget.

    The actual clustering and LLM summarization are simulated here via
    deterministic heuristics so that the baseline runs without inference.
    """

    MAX_TREE_DEPTH: int = 4
    CLUSTER_COMPRESSION_RATIO: float = 0.35

    def __init__(
        self,
        max_depth: int = MAX_TREE_DEPTH,
        compression_ratio: float = CLUSTER_COMPRESSION_RATIO,
    ):
        self.max_depth = max_depth
        self.compression_ratio = compression_ratio

    @property
    def name(self) -> str:
        return "RAPTOR"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(
        self,
        query: str,
        candidate_items: List[ContextItem],
        max_tokens: int,
    ) -> List[ContextItem]:
        if not candidate_items:
            return []

        # Step 1: cluster by memory_type (proxy for semantic similarity)
        clusters = self._cluster_items(candidate_items)

        # Step 2: build tree bottom-up
        tree_nodes = self._build_tree(clusters, max_tokens)

        # Step 3: retrieve from tree with query relevance
        selected = self._retrieve_from_tree(tree_nodes, query, max_tokens)
        return selected

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cluster_items(
        self, items: List[ContextItem]
    ) -> Dict[str, List[ContextItem]]:
        clusters: Dict[str, List[ContextItem]] = defaultdict(list)
        for item in items:
            clusters[item.memory_type].append(item)
        return dict(clusters)

    def _summarize_cluster(
        self, cluster_id: str, items: List[ContextItem]
    ) -> ContextItem:
        """
        Extractive cluster summary: concatenate top sentences from each item,
        keeping *compression_ratio* of the total content.
        """
        all_sentences: List[str] = []
        for item in sorted(items, key=lambda x: x.importance, reverse=True):
            sentences = re.split(r"(?<=[.!?])\s+", item.content.strip())
            all_sentences.extend(sentences)

        keep_n = max(1, math.ceil(len(all_sentences) * self.compression_ratio))
        summary_text = " ".join(all_sentences[:keep_n])

        avg_importance = sum(i.importance for i in items) / len(items)
        avg_relevance = sum(i.relevance for i in items) / len(items)

        return ContextItem(
            id=f"raptor_node_{cluster_id}",
            content=summary_text,
            memory_type=cluster_id,
            importance=avg_importance,
            relevance=avg_relevance,
            token_count=ContextItem._estimate_tokens(summary_text),
            metadata={
                "raptor_node": True,
                "source_ids": [i.id for i in items],
                "cluster": cluster_id,
                "depth": 1,
            },
        )

    def _build_tree(
        self,
        clusters: Dict[str, List[ContextItem]],
        max_tokens: int,
    ) -> List[ContextItem]:
        """
        Build a recursive summarization tree.  Returns leaf-level summaries
        initially; if still over budget, merges clusters iteratively.
        """
        # Level 0: raw items per cluster summarized into leaf nodes
        leaves = [
            self._summarize_cluster(cid, items)
            for cid, items in clusters.items()
        ]

        current_level = leaves
        depth = 1

        while (
            self._total_tokens(current_level) > max_tokens
            and depth < self.max_depth
            and len(current_level) > 1
        ):
            # Merge pairs of least-important nodes
            current_level = self._merge_level(current_level, depth)
            depth += 1

        return current_level

    def _merge_level(
        self, nodes: List[ContextItem], depth: int
    ) -> List[ContextItem]:
        """Pair adjacent nodes and merge-summarize them."""
        nodes_sorted = sorted(nodes, key=lambda x: x.importance)
        merged: List[ContextItem] = []
        i = 0
        while i < len(nodes_sorted):
            if i + 1 < len(nodes_sorted):
                a, b = nodes_sorted[i], nodes_sorted[i + 1]
                combined_text = a.content + " " + b.content
                sentences = re.split(r"(?<=[.!?])\s+", combined_text.strip())
                keep_n = max(1, math.ceil(len(sentences) * self.compression_ratio))
                merged_text = " ".join(sentences[:keep_n])
                merged_node = ContextItem(
                    id=f"raptor_merge_d{depth}_{i}",
                    content=merged_text,
                    memory_type="merged",
                    importance=(a.importance + b.importance) / 2,
                    relevance=(a.relevance + b.relevance) / 2,
                    token_count=ContextItem._estimate_tokens(merged_text),
                    metadata={
                        "raptor_node": True,
                        "depth": depth,
                        "merged_from": [a.id, b.id],
                    },
                )
                merged.append(merged_node)
                i += 2
            else:
                merged.append(nodes_sorted[i])
                i += 1
        return merged

    def _retrieve_from_tree(
        self,
        nodes: List[ContextItem],
        query: str,
        max_tokens: int,
    ) -> List[ContextItem]:
        """Score tree nodes by query relevance and fill budget."""
        scored = sorted(
            nodes,
            key=lambda x: (
                x.relevance if x.relevance > 0
                else self._keyword_score(query, x.content)
            )
            + 0.1 * x.importance,
            reverse=True,
        )
        return self._truncate_to_budget(scored, max_tokens)


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, type] = {
    "full_context": FullContextMethod,
    "truncation": TruncationMethod,
    "rag_only": RAGOnlyMethod,
    "summarization_only": SummarizationOnlyMethod,
    "memgpt": MemGPTMethod,
    "raptor": RAPTORMethod,
}


def get_baseline(name: str, **kwargs) -> BaseContextMethod:
    """
    Instantiate a baseline method by registry key.

    Parameters
    ----------
    name : str
        One of: 'full_context', 'truncation', 'rag_only',
        'summarization_only', 'memgpt', 'raptor'.
    **kwargs
        Forwarded to the constructor.

    Returns
    -------
    BaseContextMethod
    """
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown baseline '{name}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[key](**kwargs)


def list_baselines() -> List[str]:
    """Return all registered baseline names."""
    return list(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uuid

    def _make_item(content: str, importance: float = 0.5, mtype: str = "episodic") -> ContextItem:
        return ContextItem(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=mtype,
            importance=importance,
        )

    samples = [
        _make_item("The quick brown fox jumps over the lazy dog.", 0.8, "observation"),
        _make_item("Paris is the capital of France and a major European city.", 0.6, "semantic"),
        _make_item("User asked to summarize the quarterly report.", 0.9, "goal"),
        _make_item("Tool output: file written successfully to /tmp/out.txt.", 0.4, "tool_output"),
        _make_item("Previous step: retrieved 42 rows from the database.", 0.5, "episodic"),
        _make_item("Background: SQL queries require proper indexing for performance.", 0.3, "semantic"),
    ]

    query = "summarize quarterly report database"
    budget = 200

    for method_name in list_baselines():
        method = get_baseline(method_name)
        result = method.process(query, samples, budget)
        tokens = sum(i.token_count for i in result)
        print(f"[{method.name:22s}] items={len(result):2d}  tokens={tokens:4d}")
