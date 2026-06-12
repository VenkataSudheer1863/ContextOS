"""
ContextOS Governance Engine
===========================
Implements context lifecycle governance including retention policies,
forgetting policies, promotion policies, and compression triggers.
"""

from __future__ import annotations

import math
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

@dataclass
class ContextItem:
    """Represents a single item stored in the context memory."""

    id: str
    content: str
    importance: float = 0.5
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    memory_tier: str = "working"          # "working" | "episodic" | "semantic"
    tags: List[str] = field(default_factory=list)
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def age_hours(self) -> float:
        return self.age_seconds / 3600.0

    @property
    def idle_seconds(self) -> float:
        """Seconds since last access."""
        return time.time() - self.last_accessed

    @property
    def idle_hours(self) -> float:
        return self.idle_seconds / 3600.0

    def touch(self) -> None:
        """Update last_accessed and increment access_count."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class GovernanceStats:
    """Accumulated statistics produced by GovernanceEngine."""

    total_evicted: int = 0
    total_promoted: int = 0
    total_compressed: int = 0
    avg_retention_hours: float = 0.0

    # Internal accumulator for computing running average
    _retention_samples: List[float] = field(default_factory=list, repr=False)

    def record_retention(self, hours: float) -> None:
        self._retention_samples.append(hours)
        self.avg_retention_hours = (
            sum(self._retention_samples) / len(self._retention_samples)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_evicted": self.total_evicted,
            "total_promoted": self.total_promoted,
            "total_compressed": self.total_compressed,
            "avg_retention_hours": round(self.avg_retention_hours, 4),
        }


# ---------------------------------------------------------------------------
# Memory store protocol (duck-typed; no hard dependency on implementation)
# ---------------------------------------------------------------------------

class MemoryStoreProtocol(Protocol):
    """Minimal interface expected from a memory store."""

    def get_all(self) -> List[ContextItem]: ...
    def remove(self, item_id: str) -> None: ...
    def update(self, item: ContextItem) -> None: ...


# ---------------------------------------------------------------------------
# Retention Policies
# ---------------------------------------------------------------------------

class BaseRetentionPolicy(ABC):
    """Abstract base class for all retention policies."""

    @abstractmethod
    def should_retain(self, item: ContextItem) -> bool:
        """Return True if the item should be retained."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class TimeBasedRetention(BaseRetentionPolicy):
    """Retain items that were accessed within *max_age_hours*."""

    def __init__(self, max_age_hours: float = 72.0) -> None:
        self.max_age_hours = max_age_hours

    def should_retain(self, item: ContextItem) -> bool:
        return item.idle_hours <= self.max_age_hours

    def __repr__(self) -> str:
        return f"TimeBasedRetention(max_age_hours={self.max_age_hours})"


class ImportanceBasedRetention(BaseRetentionPolicy):
    """Retain items whose importance score exceeds *threshold*."""

    def __init__(self, threshold: float = 0.1) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = threshold

    def should_retain(self, item: ContextItem) -> bool:
        return item.importance > self.threshold

    def __repr__(self) -> str:
        return f"ImportanceBasedRetention(threshold={self.threshold})"


class FrequencyBasedRetention(BaseRetentionPolicy):
    """Retain items that have been accessed at least *min_access_count* times."""

    def __init__(self, min_access_count: int = 2) -> None:
        if min_access_count < 0:
            raise ValueError("min_access_count must be >= 0")
        self.min_access_count = min_access_count

    def should_retain(self, item: ContextItem) -> bool:
        return item.access_count >= self.min_access_count

    def __repr__(self) -> str:
        return f"FrequencyBasedRetention(min_access_count={self.min_access_count})"


class CompositeRetention(BaseRetentionPolicy):
    """Combine multiple retention policies with "any" or "all" logic."""

    _VALID_MODES = frozenset({"any", "all"})

    def __init__(self, *policies: BaseRetentionPolicy, mode: str = "any") -> None:
        if not policies:
            raise ValueError("At least one child policy is required")
        if mode not in self._VALID_MODES:
            raise ValueError(f"mode must be one of {self._VALID_MODES}")
        self.policies = list(policies)
        self.mode = mode

    def should_retain(self, item: ContextItem) -> bool:
        results = (p.should_retain(item) for p in self.policies)
        if self.mode == "any":
            return any(results)
        # mode == "all"
        return all(p.should_retain(item) for p in self.policies)

    def __repr__(self) -> str:
        policy_repr = ", ".join(repr(p) for p in self.policies)
        return f"CompositeRetention({policy_repr}, mode={self.mode!r})"


# ---------------------------------------------------------------------------
# Forgetting Policies
# ---------------------------------------------------------------------------

class BaseForgettingPolicy(ABC):
    """Abstract base class for all forgetting (eviction) policies."""

    @abstractmethod
    def should_evict(self, item: ContextItem) -> bool:
        """Return True if the item should be evicted."""

    def apply_decay(self, item: ContextItem) -> ContextItem:
        """Optionally mutate importance before eviction check (override if needed)."""
        return item

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ExponentialDecayForgetting(BaseForgettingPolicy):
    """Apply exponential importance decay; evict if decayed value falls below threshold."""

    def __init__(
        self,
        lambda_: float = 0.0001,
        threshold: float = 0.05,
    ) -> None:
        if lambda_ < 0:
            raise ValueError("lambda_ must be >= 0")
        self.lambda_ = lambda_
        self.threshold = threshold

    def decayed_importance(self, item: ContextItem) -> float:
        return item.importance * math.exp(-self.lambda_ * item.age_seconds)

    def apply_decay(self, item: ContextItem) -> ContextItem:
        item.importance = self.decayed_importance(item)
        return item

    def should_evict(self, item: ContextItem) -> bool:
        return self.decayed_importance(item) < self.threshold

    def __repr__(self) -> str:
        return (
            f"ExponentialDecayForgetting(lambda_={self.lambda_}, "
            f"threshold={self.threshold})"
        )


class LRUForgetting(BaseForgettingPolicy):
    """Evict least-recently-used items when the store exceeds *capacity*."""

    def __init__(self, capacity: int = 1000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = capacity
        # Populated by select_evictions(); individual should_evict queries are
        # context-free and therefore always return False unless the item is
        # explicitly marked for eviction.
        self._evict_ids: frozenset = frozenset()

    def select_evictions(self, items: List[ContextItem]) -> List[ContextItem]:
        """Return the LRU items that must be evicted to fit within capacity."""
        if len(items) <= self.capacity:
            return []
        # Sort ascending by last_accessed (oldest first)
        sorted_items = sorted(items, key=lambda x: x.last_accessed)
        overflow = len(items) - self.capacity
        victims = sorted_items[:overflow]
        self._evict_ids = frozenset(v.id for v in victims)
        return victims

    def should_evict(self, item: ContextItem) -> bool:
        return item.id in self._evict_ids

    def __repr__(self) -> str:
        return f"LRUForgetting(capacity={self.capacity})"


class ThresholdForgetting(BaseForgettingPolicy):
    """Evict items whose importance score falls below *quality_threshold*."""

    def __init__(self, quality_threshold: float = 0.05) -> None:
        if not 0.0 <= quality_threshold <= 1.0:
            raise ValueError("quality_threshold must be in [0, 1]")
        self.quality_threshold = quality_threshold

    def should_evict(self, item: ContextItem) -> bool:
        return item.importance < self.quality_threshold

    def __repr__(self) -> str:
        return f"ThresholdForgetting(quality_threshold={self.quality_threshold})"


# ---------------------------------------------------------------------------
# Promotion Policies
# ---------------------------------------------------------------------------

class BasePromotionPolicy(ABC):
    """Abstract base class for memory-tier promotion policies."""

    @abstractmethod
    def should_promote(self, item: ContextItem) -> bool:
        """Return True if the item qualifies for promotion."""

    @abstractmethod
    def target_tier(self) -> str:
        """Return the target memory tier for promoted items."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class WorkingToLongTermPromotion(BasePromotionPolicy):
    """Promote working-memory items to long-term (semantic) memory.

    Criteria: access_count >= 3 AND importance >= 0.7.
    """

    def __init__(
        self,
        min_access_count: int = 3,
        min_importance: float = 0.7,
    ) -> None:
        self.min_access_count = min_access_count
        self.min_importance = min_importance

    def should_promote(self, item: ContextItem) -> bool:
        return (
            item.memory_tier == "working"
            and item.access_count >= self.min_access_count
            and item.importance >= self.min_importance
        )

    def target_tier(self) -> str:
        return "semantic"

    def __repr__(self) -> str:
        return (
            f"WorkingToLongTermPromotion("
            f"min_access_count={self.min_access_count}, "
            f"min_importance={self.min_importance})"
        )


class EpisodicPromotion(BasePromotionPolicy):
    """Promote important episodic-memory items to semantic memory.

    Criteria: memory_tier == "episodic" AND importance >= threshold.
    """

    def __init__(self, importance_threshold: float = 0.6) -> None:
        self.importance_threshold = importance_threshold

    def should_promote(self, item: ContextItem) -> bool:
        return (
            item.memory_tier == "episodic"
            and item.importance >= self.importance_threshold
        )

    def target_tier(self) -> str:
        return "semantic"

    def __repr__(self) -> str:
        return f"EpisodicPromotion(importance_threshold={self.importance_threshold})"


# ---------------------------------------------------------------------------
# Compression Triggers
# ---------------------------------------------------------------------------

class BaseCompressionTrigger(ABC):
    """Abstract base class for context-compression triggers."""

    @abstractmethod
    def should_compress(self, context_tokens: int, budget: int, **kwargs: Any) -> bool:
        """Return True if compression should be triggered."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class TokenBudgetTrigger(BaseCompressionTrigger):
    """Trigger compression when token usage exceeds *budget_fraction* of budget."""

    def __init__(self, budget_fraction: float = 0.8) -> None:
        if not 0.0 < budget_fraction <= 1.0:
            raise ValueError("budget_fraction must be in (0, 1]")
        self.budget_fraction = budget_fraction

    def should_compress(self, context_tokens: int, budget: int, **kwargs: Any) -> bool:
        if budget <= 0:
            return False
        return (context_tokens / budget) >= self.budget_fraction

    def __repr__(self) -> str:
        return f"TokenBudgetTrigger(budget_fraction={self.budget_fraction})"


class CapacityTrigger(BaseCompressionTrigger):
    """Trigger compression when the number of context items exceeds *max_items*."""

    def __init__(self, max_items: int = 500) -> None:
        if max_items <= 0:
            raise ValueError("max_items must be > 0")
        self.max_items = max_items

    def should_compress(
        self,
        context_tokens: int,
        budget: int,
        item_count: int = 0,
        **kwargs: Any,
    ) -> bool:
        return item_count > self.max_items

    def __repr__(self) -> str:
        return f"CapacityTrigger(max_items={self.max_items})"


class TimedTrigger(BaseCompressionTrigger):
    """Trigger compression once per *interval_hours* regardless of load."""

    def __init__(self, interval_hours: float = 6.0) -> None:
        if interval_hours <= 0:
            raise ValueError("interval_hours must be > 0")
        self.interval_hours = interval_hours
        self._last_triggered: float = time.time()   # start clock from now

    def should_compress(self, context_tokens: int, budget: int, **kwargs: Any) -> bool:
        now = time.time()
        elapsed_hours = (now - self._last_triggered) / 3600.0
        if elapsed_hours >= self.interval_hours:
            self._last_triggered = now
            return True
        return False

    def reset(self) -> None:
        """Reset the timer (useful for testing)."""
        self._last_triggered = 0.0

    def __repr__(self) -> str:
        return f"TimedTrigger(interval_hours={self.interval_hours})"


# ---------------------------------------------------------------------------
# Governance Engine
# ---------------------------------------------------------------------------

@dataclass
class GovernanceConfig:
    """Configuration for GovernanceEngine.  All fields are optional."""

    # Retention
    max_age_hours: float = 72.0
    importance_threshold: float = 0.1
    min_access_count: int = 2
    retention_mode: str = "any"           # "any" | "all"

    # Forgetting
    decay_lambda: float = 0.0001
    decay_evict_threshold: float = 0.05
    lru_capacity: int = 1000
    quality_threshold: float = 0.05

    # Promotion
    working_min_access: int = 3
    working_min_importance: float = 0.7
    episodic_promote_threshold: float = 0.6

    # Compression
    token_budget_fraction: float = 0.8
    max_items: int = 500
    compress_interval_hours: float = 6.0


class GovernanceEngine:
    """
    Central governance engine for the ContextOS memory subsystem.

    Responsibilities
    ----------------
    * Apply configurable retention / forgetting policies to context items.
    * Run periodic forgetting cycles against a memory store.
    * Evaluate whether individual items should be promoted to a higher tier.
    * Decide when context compression should be triggered.
    * Accumulate lifecycle statistics.
    """

    def __init__(self, config: Optional[GovernanceConfig] = None) -> None:
        self.config = config or GovernanceConfig()
        self._stats = GovernanceStats()

        # --- Build retention policies ---
        self._retention_policy = CompositeRetention(
            TimeBasedRetention(max_age_hours=self.config.max_age_hours),
            ImportanceBasedRetention(threshold=self.config.importance_threshold),
            FrequencyBasedRetention(min_access_count=self.config.min_access_count),
            mode=self.config.retention_mode,
        )

        # --- Build forgetting policies ---
        self._forgetting_policies: List[BaseForgettingPolicy] = [
            ExponentialDecayForgetting(
                lambda_=self.config.decay_lambda,
                threshold=self.config.decay_evict_threshold,
            ),
            LRUForgetting(capacity=self.config.lru_capacity),
            ThresholdForgetting(quality_threshold=self.config.quality_threshold),
        ]

        # --- Build promotion policies ---
        self._promotion_policies: List[BasePromotionPolicy] = [
            WorkingToLongTermPromotion(
                min_access_count=self.config.working_min_access,
                min_importance=self.config.working_min_importance,
            ),
            EpisodicPromotion(
                importance_threshold=self.config.episodic_promote_threshold,
            ),
        ]

        # --- Build compression triggers ---
        self._compression_triggers: List[BaseCompressionTrigger] = [
            TokenBudgetTrigger(budget_fraction=self.config.token_budget_fraction),
            CapacityTrigger(max_items=self.config.max_items),
            TimedTrigger(interval_hours=self.config.compress_interval_hours),
        ]

        logger.info(
            "GovernanceEngine initialised | retention=%s | forgetting_policies=%d "
            "| promotion_policies=%d | compression_triggers=%d",
            self.config.retention_mode,
            len(self._forgetting_policies),
            len(self._promotion_policies),
            len(self._compression_triggers),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_context_policies(self, items: List[ContextItem]) -> List[ContextItem]:
        """
        Filter *items* through the active retention policy.

        Items that fail the retention check are marked as candidates for
        eviction (importance set to 0) and excluded from the returned list.
        Stats are updated accordingly.

        Parameters
        ----------
        items:
            The list of ContextItem objects to evaluate.

        Returns
        -------
        List[ContextItem]
            Only the items that pass the retention policy.
        """
        retained: List[ContextItem] = []
        evicted_count = 0

        for item in items:
            if self._retention_policy.should_retain(item):
                retained.append(item)
                self._stats.record_retention(item.age_hours)
            else:
                evicted_count += 1
                self._stats.total_evicted += 1
                logger.debug(
                    "apply_context_policies: evicting item id=%s (age=%.2fh, "
                    "importance=%.3f, access_count=%d)",
                    item.id,
                    item.age_hours,
                    item.importance,
                    item.access_count,
                )

        if evicted_count:
            logger.info(
                "apply_context_policies: retained %d / %d items (%d evicted)",
                len(retained),
                len(items),
                evicted_count,
            )

        return retained

    def run_forgetting_cycle(self, memory_store: Any) -> List[str]:
        """
        Execute a full forgetting cycle against *memory_store*.

        Steps
        -----
        1. Retrieve all items from the store.
        2. Apply exponential decay to importance scores (mutates in place and
           persists via store.update).
        3. Collect eviction candidates from every forgetting policy.
        4. For LRUForgetting, run select_evictions to determine overflow victims.
        5. Remove eviction candidates from the store.
        6. Update statistics.

        Parameters
        ----------
        memory_store:
            Object satisfying :class:`MemoryStoreProtocol`.

        Returns
        -------
        List[str]
            IDs of all evicted items.
        """
        all_items: List[ContextItem] = memory_store.get_all()
        if not all_items:
            return []

        # Step 1: Apply decay (mutates importance in-place)
        for policy in self._forgetting_policies:
            for item in all_items:
                policy.apply_decay(item)
            # Persist decayed importance back to store
            for item in all_items:
                try:
                    memory_store.update(item)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not update item %s: %s", item.id, exc)

        # Step 2: LRU — pre-compute eviction set based on capacity
        for policy in self._forgetting_policies:
            if isinstance(policy, LRUForgetting):
                policy.select_evictions(all_items)

        # Step 3: Collect eviction candidates
        evict_ids: List[str] = []
        evict_id_set: set = set()

        for item in all_items:
            for policy in self._forgetting_policies:
                if policy.should_evict(item) and item.id not in evict_id_set:
                    evict_ids.append(item.id)
                    evict_id_set.add(item.id)
                    self._stats.total_evicted += 1
                    self._stats.record_retention(item.age_hours)
                    logger.debug(
                        "run_forgetting_cycle: evicting id=%s via %s",
                        item.id,
                        policy.__class__.__name__,
                    )
                    break  # No need to check further policies for this item

        # Step 4: Remove from store
        for item_id in evict_ids:
            try:
                memory_store.remove(item_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to remove item %s from store: %s", item_id, exc)

        if evict_ids:
            logger.info(
                "run_forgetting_cycle: evicted %d items from %d total",
                len(evict_ids),
                len(all_items),
            )

        return evict_ids

    def should_promote(self, item: ContextItem) -> bool:
        """
        Evaluate whether *item* qualifies for memory-tier promotion.

        If any promotion policy matches, the item's tier is updated and
        promotion statistics are incremented.

        Parameters
        ----------
        item:
            The ContextItem to evaluate.

        Returns
        -------
        bool
            True if the item was (or should be) promoted.
        """
        for policy in self._promotion_policies:
            if policy.should_promote(item):
                old_tier = item.memory_tier
                item.memory_tier = policy.target_tier()
                self._stats.total_promoted += 1
                logger.debug(
                    "should_promote: item id=%s promoted %s -> %s via %s",
                    item.id,
                    old_tier,
                    item.memory_tier,
                    policy.__class__.__name__,
                )
                return True
        return False

    def should_compress(
        self,
        context_tokens: int,
        budget: int,
        item_count: int = 0,
    ) -> bool:
        """
        Decide whether context compression should be triggered.

        Evaluates all registered compression triggers; returns True if any
        trigger fires.  Increments compression statistics on a positive result.

        Parameters
        ----------
        context_tokens:
            Current total token count in the active context.
        budget:
            Maximum allowed token budget.
        item_count:
            Current number of items in the context (used by CapacityTrigger).

        Returns
        -------
        bool
            True if at least one trigger recommends compression.
        """
        # Evaluate ALL triggers so side-effects (e.g. TimedTrigger clock reset)
        # are applied even when an earlier trigger already fired.
        fired_trigger: Optional[str] = None
        for trigger in self._compression_triggers:
            if trigger.should_compress(
                context_tokens, budget, item_count=item_count
            ) and fired_trigger is None:
                fired_trigger = trigger.__class__.__name__

        if fired_trigger:
            self._stats.total_compressed += 1
            logger.info(
                "should_compress: triggered by %s "
                "(tokens=%d, budget=%d, items=%d)",
                fired_trigger,
                context_tokens,
                budget,
                item_count,
            )
            return True
        return False

    def get_governance_stats(self) -> GovernanceStats:
        """Return a snapshot of accumulated governance statistics."""
        return self._stats

    # ------------------------------------------------------------------
    # Advanced / helper methods
    # ------------------------------------------------------------------

    def evaluate_all_promotions(
        self, items: List[ContextItem]
    ) -> Tuple[List[ContextItem], List[ContextItem]]:
        """
        Batch-evaluate promotion eligibility for a list of items.

        Returns
        -------
        Tuple[List[ContextItem], List[ContextItem]]
            (promoted_items, unchanged_items)
        """
        promoted: List[ContextItem] = []
        unchanged: List[ContextItem] = []
        for item in items:
            if self.should_promote(item):
                promoted.append(item)
            else:
                unchanged.append(item)
        return promoted, unchanged

    def apply_decay_to_items(self, items: List[ContextItem]) -> List[ContextItem]:
        """
        Apply exponential decay to importance scores for a list of items
        without evicting them.  Useful for scoring/ranking tasks.

        Returns the same list with mutated importance values.
        """
        decay_policy = next(
            (
                p
                for p in self._forgetting_policies
                if isinstance(p, ExponentialDecayForgetting)
            ),
            None,
        )
        if decay_policy is None:
            return items
        for item in items:
            decay_policy.apply_decay(item)
        return items

    def score_item(self, item: ContextItem) -> float:
        """
        Compute a combined governance score for *item* in [0, 1].

        The score incorporates importance, recency, and frequency signals.
        Higher is better.
        """
        # Recency score: decays to 0 as idle_hours approaches max_age_hours
        max_age = max(self.config.max_age_hours, 1e-9)
        recency = max(0.0, 1.0 - (item.idle_hours / max_age))

        # Frequency score: saturates at min_access_count * 2
        saturation = max(1, self.config.min_access_count * 2)
        frequency = min(item.access_count / saturation, 1.0)

        # Weighted combination
        score = 0.5 * item.importance + 0.3 * recency + 0.2 * frequency
        return round(min(max(score, 0.0), 1.0), 6)

    def rank_items(
        self, items: List[ContextItem], descending: bool = True
    ) -> List[ContextItem]:
        """Return *items* sorted by governance score."""
        return sorted(items, key=self.score_item, reverse=descending)

    def reset_stats(self) -> None:
        """Reset accumulated statistics (useful between test runs)."""
        self._stats = GovernanceStats()

    def describe(self) -> Dict[str, Any]:
        """Return a human-readable description of the engine configuration."""
        return {
            "retention_policy": repr(self._retention_policy),
            "forgetting_policies": [repr(p) for p in self._forgetting_policies],
            "promotion_policies": [repr(p) for p in self._promotion_policies],
            "compression_triggers": [repr(t) for t in self._compression_triggers],
            "stats": self._stats.to_dict(),
        }

    def __repr__(self) -> str:
        return (
            f"GovernanceEngine("
            f"retention_mode={self.config.retention_mode!r}, "
            f"lru_capacity={self.config.lru_capacity}, "
            f"max_age_hours={self.config.max_age_hours})"
        )
