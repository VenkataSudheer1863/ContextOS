"""
ContextOS Core Scheduler
Priority-based scheduling with diversity-aware selection strategies.
"""
from __future__ import annotations

import heapq
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

# ContextItem is defined in orchestrator; import lazily to avoid circular deps.
from .orchestrator import ContextItem


# ---------------------------------------------------------------------------
# Internal heap entry
# ---------------------------------------------------------------------------

@dataclass(order=True)
class _HeapEntry:
    """Wrapper for max-heap via negated priority."""
    neg_priority: float
    item_id: str = field(compare=False)
    item: ContextItem = field(compare=False)

    @classmethod
    def from_item(cls, item: ContextItem) -> "_HeapEntry":
        priority = item.metadata.get("final_priority", item.relevance)
        return cls(neg_priority=-priority, item_id=item.id, item=item)


# ---------------------------------------------------------------------------
# Scheduling statistics
# ---------------------------------------------------------------------------

@dataclass
class SchedulingStats:
    total_calls: int = 0
    total_items_considered: int = 0
    total_items_selected: int = 0
    total_tokens_scheduled: int = 0
    total_tokens_saved: int = 0
    avg_priority_selected: float = 0.0
    avg_priority_dropped: float = 0.0
    strategy_used: str = ""
    last_schedule_ms: float = 0.0

    def update(
        self,
        selected: List[ContextItem],
        dropped: List[ContextItem],
        tokens_scheduled: int,
        tokens_available: int,
        strategy: str,
        elapsed_ms: float,
    ) -> None:
        self.total_calls += 1
        self.total_items_considered += len(selected) + len(dropped)
        self.total_items_selected += len(selected)
        self.total_tokens_scheduled += tokens_scheduled
        self.total_tokens_saved += max(0, tokens_available - tokens_scheduled)
        self.strategy_used = strategy
        self.last_schedule_ms = elapsed_ms

        if selected:
            self.avg_priority_selected = sum(
                i.metadata.get("final_priority", i.relevance) for i in selected
            ) / len(selected)
        if dropped:
            self.avg_priority_dropped = sum(
                i.metadata.get("final_priority", i.relevance) for i in dropped
            ) / len(dropped)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_items_considered": self.total_items_considered,
            "total_items_selected": self.total_items_selected,
            "total_tokens_scheduled": self.total_tokens_scheduled,
            "total_tokens_saved": self.total_tokens_saved,
            "avg_priority_selected": round(self.avg_priority_selected, 4),
            "avg_priority_dropped": round(self.avg_priority_dropped, 4),
            "strategy_used": self.strategy_used,
            "last_schedule_ms": round(self.last_schedule_ms, 3),
        }


# ---------------------------------------------------------------------------
# Base scheduler
# ---------------------------------------------------------------------------

class BaseScheduler(ABC):
    """Abstract base class for all scheduling strategies."""

    @abstractmethod
    def schedule(
        self,
        items: List[ContextItem],
        max_tokens: int,
    ) -> Tuple[List[ContextItem], int]:
        """
        Select a subset of items whose total token count fits within max_tokens.

        Returns
        -------
        selected_items : list of ContextItem
        total_tokens   : int  — total token count of selected items
        """
        ...

    @staticmethod
    def _token_count(item: ContextItem) -> int:
        if item.token_count and item.token_count > 0:
            return item.token_count
        # Fallback: ~4/3 tokens per whitespace-separated word
        return max(1, int(len(item.content.split()) * 4 / 3))

    @staticmethod
    def _build_heap(items: List[ContextItem]) -> List[_HeapEntry]:
        heap = [_HeapEntry.from_item(i) for i in items]
        heapq.heapify(heap)
        return heap


# ---------------------------------------------------------------------------
# Strategy 1 — GreedyScheduler
# ---------------------------------------------------------------------------

class GreedyScheduler(BaseScheduler):
    """
    Select items in descending priority order until the token budget is filled.
    Simple, deterministic, and O(n log n).
    """

    def schedule(
        self,
        items: List[ContextItem],
        max_tokens: int,
    ) -> Tuple[List[ContextItem], int]:
        heap = self._build_heap(items)
        selected: List[ContextItem] = []
        used = 0

        while heap:
            entry = heapq.heappop(heap)
            tokens = self._token_count(entry.item)
            if used + tokens <= max_tokens:
                selected.append(entry.item)
                used += tokens

        logger.debug(
            f"GreedyScheduler: {len(selected)}/{len(items)} items, {used}/{max_tokens} tokens"
        )
        return selected, used


# ---------------------------------------------------------------------------
# Strategy 2 — DiversityScheduler  (MMR-based)
# ---------------------------------------------------------------------------

class DiversityScheduler(BaseScheduler):
    """
    Marginal Maximal Relevance selection: balances priority with diversity.

    MMR score = lambda * priority(i) - (1 - lambda) * max_sim(i, selected)

    Requires item embeddings; falls back to GreedyScheduler when embeddings
    are absent.
    """

    def __init__(self, mmr_lambda: float = 0.6) -> None:
        self.mmr_lambda = mmr_lambda

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return dot / (na * nb)

    def _max_sim_to_selected(
        self,
        item: ContextItem,
        selected: List[ContextItem],
    ) -> float:
        if not selected or item.embedding is None:
            return 0.0
        sims = [
            self._cosine(item.embedding, s.embedding)
            for s in selected
            if s.embedding is not None
        ]
        return max(sims) if sims else 0.0

    def schedule(
        self,
        items: List[ContextItem],
        max_tokens: int,
    ) -> Tuple[List[ContextItem], int]:
        if not items:
            return [], 0

        has_embeddings = any(i.embedding for i in items)
        if not has_embeddings:
            logger.warning("DiversityScheduler: no embeddings found, falling back to GreedyScheduler.")
            return GreedyScheduler().schedule(items, max_tokens)

        remaining = list(items)
        selected: List[ContextItem] = []
        used = 0

        while remaining:
            best: Optional[ContextItem] = None
            best_mmr = float("-inf")

            for item in remaining:
                tokens = self._token_count(item)
                if used + tokens > max_tokens:
                    continue
                priority = item.metadata.get("final_priority", item.relevance)
                sim = self._max_sim_to_selected(item, selected)
                mmr = self.mmr_lambda * priority - (1.0 - self.mmr_lambda) * sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best = item

            if best is None:
                break

            selected.append(best)
            used += self._token_count(best)
            remaining.remove(best)

        logger.debug(
            f"DiversityScheduler(lambda={self.mmr_lambda}): "
            f"{len(selected)}/{len(items)} items, {used}/{max_tokens} tokens"
        )
        return selected, used


# ---------------------------------------------------------------------------
# Strategy 3 — ThresholdScheduler
# ---------------------------------------------------------------------------

class ThresholdScheduler(BaseScheduler):
    """
    Only include items whose final_priority exceeds a dynamic threshold.

    The threshold can be:
    - absolute: a fixed value in [0, 1]
    - percentile: derived from the distribution of all item priorities

    Falls back to top-K greedy if nothing passes the threshold.
    """

    def __init__(
        self,
        threshold: float = 0.4,
        use_percentile: bool = False,
        percentile: float = 50.0,
        fallback_k: int = 5,
    ) -> None:
        self.threshold = threshold
        self.use_percentile = use_percentile
        self.percentile = percentile
        self.fallback_k = fallback_k

    def _effective_threshold(self, items: List[ContextItem]) -> float:
        if not self.use_percentile:
            return self.threshold
        priorities = sorted(
            [i.metadata.get("final_priority", i.relevance) for i in items]
        )
        if not priorities:
            return self.threshold
        idx = int(len(priorities) * self.percentile / 100)
        idx = min(idx, len(priorities) - 1)
        return priorities[idx]

    def schedule(
        self,
        items: List[ContextItem],
        max_tokens: int,
    ) -> Tuple[List[ContextItem], int]:
        effective = self._effective_threshold(items)
        filtered = [
            i for i in items
            if i.metadata.get("final_priority", i.relevance) >= effective
        ]

        if not filtered:
            logger.warning(
                f"ThresholdScheduler: no items above threshold {effective:.3f}, "
                f"using greedy fallback (top-{self.fallback_k})"
            )
            filtered = sorted(
                items,
                key=lambda i: i.metadata.get("final_priority", i.relevance),
                reverse=True,
            )[: self.fallback_k]

        # Greedy fill within budget from filtered set
        heap = self._build_heap(filtered)
        selected: List[ContextItem] = []
        used = 0

        while heap:
            entry = heapq.heappop(heap)
            tokens = self._token_count(entry.item)
            if used + tokens <= max_tokens:
                selected.append(entry.item)
                used += tokens

        logger.debug(
            f"ThresholdScheduler(threshold={effective:.3f}): "
            f"{len(selected)}/{len(items)} items, {used}/{max_tokens} tokens"
        )
        return selected, used


# ---------------------------------------------------------------------------
# Main ContextScheduler
# ---------------------------------------------------------------------------

class ContextScheduler:
    """
    Priority-based context scheduler.

    Priority formula
    ----------------
    priority = w_rel * relevance
             + w_rec * recency_score
             + w_imp * importance
             + w_nov * novelty

    Temporal decay
    --------------
    recency_score = exp(-lambda * age_seconds)

    Novelty
    -------
    Computed via MMR: 1 - max_cosine_sim(item, already_selected_items)

    Parameters
    ----------
    config : dict
        Expected shape mirrors ContextOrchestrator._default_config():
        {
            "scheduler": {
                "algorithm": "greedy" | "diversity" | "threshold",
                "weights": {
                    "relevance": 0.40,
                    "recency":   0.25,
                    "importance":0.20,
                    "novelty":   0.15,
                },
                "decay_lambda": 0.0001,
                "mmr_lambda":   0.6,
                "threshold":    0.4,
            }
        }
    """

    _STRATEGIES = {
        "greedy": GreedyScheduler,
        "diversity": DiversityScheduler,
        "threshold": ThresholdScheduler,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = (config or {}).get("scheduler", {})
        w = cfg.get("weights", {})
        self.w_rel = float(w.get("relevance", 0.40))
        self.w_rec = float(w.get("recency", 0.25))
        self.w_imp = float(w.get("importance", 0.20))
        self.w_nov = float(w.get("novelty", 0.15))
        self.decay_lambda = float(cfg.get("decay_lambda", 0.0001))
        self.algorithm = cfg.get("algorithm", "greedy")
        self.mmr_lambda = float(cfg.get("mmr_lambda", 0.6))
        self.threshold = float(cfg.get("threshold", 0.4))
        self.stats = SchedulingStats()
        logger.info(
            f"ContextScheduler ready — algorithm={self.algorithm}, "
            f"weights=(rel={self.w_rel}, rec={self.w_rec}, "
            f"imp={self.w_imp}, nov={self.w_nov}), "
            f"decay_lambda={self.decay_lambda}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schedule(
        self,
        items: List[ContextItem],
        max_tokens: int,
    ) -> Tuple[List[ContextItem], int]:
        """
        Score every item, then select the best subset within max_tokens.

        Returns
        -------
        (selected_items, total_tokens)
        """
        if not items:
            return [], 0

        t0 = time.perf_counter()

        # Step 1: compute and attach priorities
        scored = self._score_items(items)

        # Step 2: pick strategy and run
        strategy_instance = self._build_strategy()
        selected, used = strategy_instance.schedule(scored, max_tokens)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        dropped = [i for i in scored if i not in selected]
        self.stats.update(selected, dropped, used, max_tokens, self.algorithm, elapsed_ms)

        logger.info(
            f"Scheduled {len(selected)}/{len(items)} items using {self.algorithm!r} "
            f"({used}/{max_tokens} tokens) in {elapsed_ms:.2f}ms"
        )
        return selected, used

    def set_algorithm(self, algorithm: str) -> None:
        """Switch scheduling strategy at runtime."""
        if algorithm not in self._STRATEGIES:
            raise ValueError(
                f"Unknown algorithm {algorithm!r}. Choose from: {list(self._STRATEGIES)}"
            )
        self.algorithm = algorithm
        logger.debug(f"Scheduler algorithm switched to {algorithm!r}")

    def set_weights(
        self,
        relevance: Optional[float] = None,
        recency: Optional[float] = None,
        importance: Optional[float] = None,
        novelty: Optional[float] = None,
    ) -> None:
        """Update priority weights at runtime."""
        if relevance is not None:
            self.w_rel = relevance
        if recency is not None:
            self.w_rec = recency
        if importance is not None:
            self.w_imp = importance
        if novelty is not None:
            self.w_nov = novelty
        logger.debug(
            f"Weights updated: rel={self.w_rel}, rec={self.w_rec}, "
            f"imp={self.w_imp}, nov={self.w_nov}"
        )

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.to_dict()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recency_score(self, item: ContextItem) -> float:
        """Temporal decay: exp(-lambda * age_seconds)."""
        age = item.age_seconds()
        return math.exp(-self.decay_lambda * age)

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return dot / (na * nb)

    def _novelty_score(
        self,
        item: ContextItem,
        peers: List[ContextItem],
    ) -> float:
        """
        MMR-based novelty: 1 - max_cosine_sim(item, peers).
        Returns 1.0 if no embeddings are available (maximum novelty assumed).
        """
        if item.embedding is None or not peers:
            return 1.0
        sims = [
            self._cosine(item.embedding, p.embedding)
            for p in peers
            if p is not item and p.embedding is not None
        ]
        return 1.0 - max(sims) if sims else 1.0

    def _score_items(self, items: List[ContextItem]) -> List[ContextItem]:
        """
        Compute final_priority for every item and store it in metadata.
        Novelty for each item is relative to all other items in the batch.
        """
        for item in items:
            recency = self._recency_score(item)
            novelty = self._novelty_score(item, items)
            priority = (
                self.w_rel * item.relevance
                + self.w_rec * recency
                + self.w_imp * item.importance
                + self.w_nov * novelty
            )
            item.metadata["final_priority"] = round(priority, 6)
            item.metadata["recency_score"] = round(recency, 6)
            item.metadata["novelty_score"] = round(novelty, 6)

        return items

    def _build_strategy(self) -> BaseScheduler:
        if self.algorithm == "diversity":
            return DiversityScheduler(mmr_lambda=self.mmr_lambda)
        if self.algorithm == "threshold":
            return ThresholdScheduler(threshold=self.threshold)
        return GreedyScheduler()
