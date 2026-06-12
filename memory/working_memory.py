"""
ContextOS Working Memory
Short-term, capacity-bounded storage for active agent context:
goals, plans, observations, and tool outputs.

Design
------
* Fixed capacity with priority-based eviction (min-heap on priority score).
* Each slot in the heap is a (priority, insertion_order, item_id) tuple so that
  items with equal priority are broken by insertion order (FIFO), keeping the
  heap stable without comparing ContextItem objects directly.
* All public methods are thread-safe via a single reentrant lock.
"""

from __future__ import annotations

import heapq
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loguru import logger

# ---------------------------------------------------------------------------
# Import shared types from the core orchestrator.  The relative import works
# whether the package is used as ``context_os.memory`` or run directly.
# ---------------------------------------------------------------------------
try:
    from core.orchestrator import ContextItem, MemoryType
except ImportError:
    from ..core.orchestrator import ContextItem, MemoryType


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------

@dataclass
class WorkingMemoryStats:
    """Snapshot of working-memory utilisation."""
    used: int
    capacity: int
    by_type: Dict[str, int] = field(default_factory=dict)

    @property
    def utilisation(self) -> float:
        """Fraction of capacity currently in use (0.0–1.0)."""
        return self.used / self.capacity if self.capacity > 0 else 0.0

    def __repr__(self) -> str:  # pragma: no cover
        pct = f"{self.utilisation * 100:.1f}%"
        return (
            f"WorkingMemoryStats(used={self.used}/{self.capacity} [{pct}], "
            f"by_type={self.by_type})"
        )


# ---------------------------------------------------------------------------
# Internal heap entry
# ---------------------------------------------------------------------------

class _HeapEntry:
    """
    A min-heap entry that orders by (priority, insertion_order) so that
    the *lowest* priority item surfaces first (cheapest to evict).

    We store the item by reference; the heap only holds this wrapper.
    Python's heapq is a *min*-heap, so we negate nothing — lower numeric
    priority means "evict me first".
    """

    __slots__ = ("priority", "order", "item_id")

    def __init__(self, priority: float, order: int, item_id: str) -> None:
        self.priority = priority
        self.order = order
        self.item_id = item_id

    # Comparison operators for heapq
    def __lt__(self, other: "_HeapEntry") -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.order < other.order  # earlier insertion evicted first on tie

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _HeapEntry):
            return NotImplemented
        return self.priority == other.priority and self.order == other.order

    def __repr__(self) -> str:  # pragma: no cover
        return f"_HeapEntry(priority={self.priority:.3f}, order={self.order}, id={self.item_id!r})"


# ---------------------------------------------------------------------------
# WorkingMemory
# ---------------------------------------------------------------------------

class WorkingMemory:
    """
    Fixed-capacity, priority-managed working memory for an autonomous agent.

    Parameters
    ----------
    config : dict, optional
        A ContextOS config dict.  ``config["working_memory"]["capacity"]``
        sets the slot limit (default 50).
    capacity : int, optional
        Direct capacity override; takes precedence over *config*.

    Thread-safety
    -------------
    All public methods acquire ``self._lock`` (a ``threading.RLock``) before
    touching internal state, making this class safe to use from multiple
    threads (e.g. a tool-execution thread adding observations while the main
    loop reads context).
    """

    # Default priority assigned when the caller does not supply one.
    # It sits at the midpoint of [0, 1] so that explicit priorities above or
    # below stand out clearly.
    _DEFAULT_PRIORITY: float = 0.5

    def __init__(
        self,
        config: Optional[Dict] = None,
        capacity: Optional[int] = None,
    ) -> None:
        cfg = (config or {}).get("working_memory", {})
        if capacity is not None:
            self._capacity: int = max(1, capacity)
        else:
            self._capacity = max(1, int(cfg.get("capacity", 50)))

        # Primary storage: id -> ContextItem
        self._items: Dict[str, ContextItem] = {}

        # Heap of _HeapEntry objects; always reflects current _items.
        # We use "lazy deletion": when an item is removed we mark it invalid
        # and skip it during eviction rather than rebuilding the heap.
        self._heap: List[_HeapEntry] = []

        # item_id -> _HeapEntry so we can invalidate on removal
        self._entry_map: Dict[str, _HeapEntry] = {}

        # Monotonically increasing counter for FIFO tie-breaking
        self._insertion_counter: int = 0

        self._lock = threading.RLock()

        logger.debug(f"WorkingMemory initialised (capacity={self._capacity}).")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._items)

    @property
    def is_full(self) -> bool:
        with self._lock:
            return len(self._items) >= self._capacity

    # ------------------------------------------------------------------
    # Core mutation methods
    # ------------------------------------------------------------------

    def add(
        self,
        item: ContextItem,
        priority: Optional[float] = None,
    ) -> bool:
        """
        Add *item* to working memory.

        If an item with the same ``item.id`` already exists it is replaced
        in-place (priority updated).  If memory is full the lowest-priority
        item is evicted to make room.

        Parameters
        ----------
        item : ContextItem
            The context item to store.
        priority : float, optional
            Override priority in [0, 1]; defaults to ``item.importance`` or
            ``_DEFAULT_PRIORITY`` if that is zero.

        Returns
        -------
        bool
            ``True`` if the item was stored (eviction may have occurred),
            ``False`` only if an unrecoverable internal error prevented storage.
        """
        if priority is None:
            priority = item.importance if item.importance else self._DEFAULT_PRIORITY
        priority = float(max(0.0, min(1.0, priority)))

        with self._lock:
            # Replace existing entry
            if item.id in self._items:
                self._replace(item, priority)
                logger.debug(f"WorkingMemory: replaced item {item.id!r}.")
                return True

            # Evict if at capacity
            if len(self._items) >= self._capacity:
                evicted = self._evict_lowest_priority_locked()
                if evicted:
                    logger.debug(
                        f"WorkingMemory: evicted {evicted.id!r} "
                        f"(priority={self._entry_map.get(evicted.id, _HeapEntry(0,0,''))!r}) "
                        f"to make room."
                    )

            self._store_new(item, priority)
            logger.debug(
                f"WorkingMemory: added {item.id!r} type={item.memory_type.value} "
                f"priority={priority:.3f} (size={len(self._items)}/{self._capacity})."
            )
            return True

    def remove(self, id: str) -> bool:
        """
        Remove the item with the given *id*.

        Returns ``True`` if the item was found and removed, ``False`` otherwise.
        """
        with self._lock:
            if id not in self._items:
                return False
            del self._items[id]
            # Lazily invalidate the heap entry
            if id in self._entry_map:
                # We cannot remove from the middle of a heap cheaply; instead
                # we overwrite priority with +inf so it never surfaces as the
                # minimum while the entry is still referenced.
                entry = self._entry_map.pop(id)
                entry.priority = float("inf")
                entry.item_id = "__invalidated__"
            logger.debug(f"WorkingMemory: removed {id!r}.")
            return True

    def clear(self) -> None:
        """Remove all items and reset internal state."""
        with self._lock:
            self._items.clear()
            self._heap.clear()
            self._entry_map.clear()
            self._insertion_counter = 0
            logger.debug("WorkingMemory: cleared.")

    def evict_lowest_priority(self) -> Optional[ContextItem]:
        """
        Public interface: evict and return the lowest-priority item.

        Returns ``None`` if memory is empty.
        """
        with self._lock:
            return self._evict_lowest_priority_locked()

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_all(self) -> List[ContextItem]:
        """Return all stored items ordered by descending priority."""
        with self._lock:
            if not self._items:
                return []
            # Build a sorted list: highest priority first
            id_to_priority = {e.item_id: e.priority for e in self._entry_map.values()}
            items = list(self._items.values())
            items.sort(key=lambda x: id_to_priority.get(x.id, 0.0), reverse=True)
            for item in items:
                item.update_access()
            return items

    def get_by_type(self, memory_type: MemoryType) -> List[ContextItem]:
        """Return all items of a given ``MemoryType``, highest priority first."""
        with self._lock:
            id_to_priority = {e.item_id: e.priority for e in self._entry_map.values()}
            matching = [
                item for item in self._items.values()
                if item.memory_type == memory_type
            ]
            matching.sort(key=lambda x: id_to_priority.get(x.id, 0.0), reverse=True)
            for item in matching:
                item.update_access()
            return matching

    def get_by_id(self, id: str) -> Optional[ContextItem]:
        """Return the item with *id*, or ``None``."""
        with self._lock:
            item = self._items.get(id)
            if item is not None:
                item.update_access()
            return item

    # ------------------------------------------------------------------
    # Serialisation / display
    # ------------------------------------------------------------------

    def to_context_string(self) -> str:
        """
        Render working memory as a human-readable (and LLM-readable) string,
        grouped by memory type in a logical order.
        """
        with self._lock:
            if not self._items:
                return "[WorkingMemory: empty]"

            type_order = [
                MemoryType.GOAL,
                MemoryType.PLAN,
                MemoryType.OBSERVATION,
                MemoryType.TOOL_OUTPUT,
                MemoryType.WORKING,
                MemoryType.EPISODIC,
                MemoryType.SEMANTIC,
                MemoryType.PROCEDURAL,
            ]

            section_labels: Dict[MemoryType, str] = {
                MemoryType.GOAL: "Goals",
                MemoryType.PLAN: "Plans",
                MemoryType.OBSERVATION: "Observations",
                MemoryType.TOOL_OUTPUT: "Tool Outputs",
                MemoryType.WORKING: "Working Notes",
                MemoryType.EPISODIC: "Episodic Memories",
                MemoryType.SEMANTIC: "Semantic Facts",
                MemoryType.PROCEDURAL: "Procedures",
            }

            # Group items
            groups: Dict[MemoryType, List[ContextItem]] = {t: [] for t in MemoryType}
            id_to_priority = {e.item_id: e.priority for e in self._entry_map.values()}
            for item in self._items.values():
                groups[item.memory_type].append(item)

            # Sort each group by priority descending
            for mtype in groups:
                groups[mtype].sort(
                    key=lambda x: id_to_priority.get(x.id, 0.0), reverse=True
                )

            lines: List[str] = [
                f"=== Working Memory ({len(self._items)}/{self._capacity} slots) ==="
            ]
            for mtype in type_order:
                items_in_group = groups.get(mtype, [])
                if not items_in_group:
                    continue
                label = section_labels.get(mtype, mtype.value.title())
                lines.append(f"\n[{label}]")
                for i, item in enumerate(items_in_group, start=1):
                    age = item.age_seconds()
                    age_str = (
                        f"{age:.0f}s ago" if age < 60
                        else f"{age / 60:.1f}m ago"
                    )
                    prio = id_to_priority.get(item.id, 0.0)
                    preview = item.content[:120].replace("\n", " ")
                    if len(item.content) > 120:
                        preview += "…"
                    lines.append(
                        f"  {i}. [{prio:.2f}] {preview}  ({age_str})"
                    )
            return "\n".join(lines)

    def get_stats(self) -> WorkingMemoryStats:
        """Return a snapshot of current memory utilisation."""
        with self._lock:
            by_type: Dict[str, int] = {}
            for item in self._items.values():
                key = item.memory_type.value
                by_type[key] = by_type.get(key, 0) + 1
            return WorkingMemoryStats(
                used=len(self._items),
                capacity=self._capacity,
                by_type=by_type,
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _store_new(self, item: ContextItem, priority: float) -> None:
        """Insert a brand-new item (caller holds the lock)."""
        self._insertion_counter += 1
        entry = _HeapEntry(priority, self._insertion_counter, item.id)
        self._items[item.id] = item
        self._entry_map[item.id] = entry
        heapq.heappush(self._heap, entry)

    def _replace(self, item: ContextItem, priority: float) -> None:
        """Replace an existing item in-place (caller holds the lock)."""
        # Invalidate old heap entry
        old_entry = self._entry_map.pop(item.id, None)
        if old_entry is not None:
            old_entry.priority = float("inf")
            old_entry.item_id = "__invalidated__"

        # Overwrite item
        self._items[item.id] = item

        # Push fresh entry
        self._insertion_counter += 1
        new_entry = _HeapEntry(priority, self._insertion_counter, item.id)
        self._entry_map[item.id] = new_entry
        heapq.heappush(self._heap, new_entry)

    def _evict_lowest_priority_locked(self) -> Optional[ContextItem]:
        """
        Pop the lowest-priority *valid* entry from the heap and remove the
        corresponding item.  Skips invalidated (lazily-deleted) entries.
        Caller must hold the lock.
        """
        while self._heap:
            entry = heapq.heappop(self._heap)
            # Skip invalidated entries
            if entry.item_id == "__invalidated__" or entry.item_id not in self._items:
                continue
            # Verify the entry still matches (it might have been replaced)
            current_entry = self._entry_map.get(entry.item_id)
            if current_entry is not entry:
                # Stale heap entry from a replace operation
                continue
            item = self._items.pop(entry.item_id)
            self._entry_map.pop(entry.item_id, None)
            return item
        return None

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def __contains__(self, id: str) -> bool:
        with self._lock:
            return id in self._items

    def __repr__(self) -> str:  # pragma: no cover
        with self._lock:
            return f"WorkingMemory(size={len(self._items)}, capacity={self._capacity})"
