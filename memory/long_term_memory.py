"""
ContextOS Long-Term Memory
Persistent, structured memory comprising three complementary subsystems:

  EpisodicMemory   — autobiographical episodes with temporal context
  SemanticMemory   — generalised facts and world knowledge
  ProceduralMemory — learned action sequences and their success rates

All three are unified under LongTermMemory, which dispatches to the correct
subsystem based on a ContextItem's MemoryType and exposes a consistent
retrieve / store / persist API to the rest of ContextOS.

Similarity search uses numpy cosine similarity so that no external vector
database is required for development / testing.  The design deliberately
isolates the similarity logic so it can be swapped for pgvector or Qdrant
in production with minimal changes.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NUMPY_AVAILABLE = False
    logger.warning("numpy not found; similarity-based retrieval will be disabled.")

try:
    from core.orchestrator import ContextItem, MemoryType
except ImportError:
    from ..core.orchestrator import ContextItem, MemoryType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Return the cosine similarity between two embedding vectors."""
    if not _NUMPY_AVAILABLE:
        return 0.0
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _top_k_by_similarity(
    query_embedding: List[float],
    candidates: List[Tuple[str, List[float]]],  # (id, embedding)
    top_k: int,
) -> List[Tuple[str, float]]:
    """Return the top-k (id, score) pairs sorted by cosine similarity desc."""
    if not _NUMPY_AVAILABLE or not candidates:
        return []
    scores: List[Tuple[float, str]] = []
    for cid, emb in candidates:
        if emb:
            scores.append((_cosine_similarity(query_embedding, emb), cid))
    scores.sort(reverse=True)
    return [(cid, score) for score, cid in scores[:top_k]]


def _make_context_item(
    id: str,
    content: str,
    memory_type: MemoryType,
    importance: float,
    timestamp: float,
    embedding: Optional[List[float]],
    metadata: Optional[Dict[str, Any]],
) -> ContextItem:
    item = ContextItem(
        id=id,
        content=content,
        memory_type=memory_type,
        timestamp=timestamp,
        importance=importance,
        metadata=metadata or {},
        embedding=embedding,
    )
    item.token_count = len(content.split()) * 4 // 3
    return item


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class LongTermMemoryStats:
    episodic_count: int = 0
    semantic_count: int = 0
    procedural_count: int = 0
    total_count: int = 0
    avg_importance: float = 0.0
    avg_semantic_confidence: float = 0.0
    avg_procedural_success_rate: float = 0.0

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"LongTermMemoryStats("
            f"episodic={self.episodic_count}, "
            f"semantic={self.semantic_count}, "
            f"procedural={self.procedural_count}, "
            f"total={self.total_count})"
        )


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------

@dataclass
class _Episode:
    id: str
    content: str
    context_snapshot: Dict[str, Any]
    timestamp: float
    importance: float
    embedding: Optional[List[float]] = None
    consolidated_into: Optional[str] = None  # id of the episode this was merged into


class EpisodicMemory:
    """
    Autobiographical memory that records *what happened* during agent
    execution together with the surrounding context snapshot.

    Episodes are stored in an in-memory list.  Retrieval supports both
    embedding-based similarity search and simple recency ordering.

    Thread-safety: all public methods are protected by a ``threading.RLock``.
    """

    def __init__(self) -> None:
        self._episodes: Dict[str, _Episode] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_episode(
        self,
        content: str,
        context_snapshot: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
        importance: float = 0.5,
        embedding: Optional[List[float]] = None,
        id: Optional[str] = None,
    ) -> str:
        """
        Record a new episode.

        Parameters
        ----------
        content : str
            Human-readable description of the episode.
        context_snapshot : dict, optional
            Arbitrary state snapshot at the time of the episode (task,
            agent_id, tool outputs, etc.).
        timestamp : float, optional
            Unix timestamp; defaults to ``time.time()``.
        importance : float
            Salience score in [0, 1].
        embedding : list of float, optional
            Pre-computed semantic embedding for similarity retrieval.
        id : str, optional
            Explicit ID; auto-generated if omitted.

        Returns
        -------
        str
            The episode ID.
        """
        episode_id = id or str(uuid.uuid4())
        episode = _Episode(
            id=episode_id,
            content=content,
            context_snapshot=context_snapshot or {},
            timestamp=timestamp or time.time(),
            importance=float(max(0.0, min(1.0, importance))),
            embedding=embedding,
        )
        with self._lock:
            self._episodes[episode_id] = episode
            logger.debug(f"EpisodicMemory: added episode {episode_id!r}.")
        return episode_id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def retrieve_similar(
        self,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> List[ContextItem]:
        """
        Return the *top_k* episodes most similar to *query_embedding*.

        Episodes without an embedding are excluded from similarity ranking.
        Falls back to recency ordering if numpy is unavailable.
        """
        with self._lock:
            active = [e for e in self._episodes.values() if e.consolidated_into is None]

        if not _NUMPY_AVAILABLE or not query_embedding:
            # Fallback: most recent episodes
            active.sort(key=lambda e: e.timestamp, reverse=True)
            return [self._episode_to_item(e) for e in active[:top_k]]

        candidates = [(e.id, e.embedding or []) for e in active if e.embedding]
        ranked = _top_k_by_similarity(query_embedding, candidates, top_k)
        id_to_episode = {e.id: e for e in active}
        result: List[ContextItem] = []
        for eid, score in ranked:
            ep = id_to_episode.get(eid)
            if ep:
                item = self._episode_to_item(ep)
                item.relevance = score
                result.append(item)
        return result

    def retrieve_recent(self, n: int = 10) -> List[ContextItem]:
        """Return the *n* most recent episodes."""
        with self._lock:
            active = [
                e for e in self._episodes.values() if e.consolidated_into is None
            ]
        active.sort(key=lambda e: e.timestamp, reverse=True)
        return [self._episode_to_item(e) for e in active[:n]]

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def consolidate_similar_episodes(self, threshold: float = 0.92) -> int:
        """
        Merge highly similar episodes to reduce redundancy.

        Episodes whose pairwise cosine similarity exceeds *threshold* are
        merged: the lower-importance episode is absorbed into the higher-
        importance one.  The surviving episode gets:

        * ``importance = max(a.importance, b.importance)``
        * ``content`` extended with a consolidation note

        Returns the number of merges performed.
        """
        if not _NUMPY_AVAILABLE:
            logger.warning("EpisodicMemory: numpy required for consolidation; skipped.")
            return 0

        with self._lock:
            active = [
                e for e in self._episodes.values()
                if e.consolidated_into is None and e.embedding
            ]

        merges = 0
        merged_ids: set = set()

        for i, ep_a in enumerate(active):
            if ep_a.id in merged_ids:
                continue
            for ep_b in active[i + 1:]:
                if ep_b.id in merged_ids:
                    continue
                sim = _cosine_similarity(ep_a.embedding, ep_b.embedding)  # type: ignore[arg-type]
                if sim >= threshold:
                    # Keep the more important episode
                    survivor, absorbed = (
                        (ep_a, ep_b) if ep_a.importance >= ep_b.importance
                        else (ep_b, ep_a)
                    )
                    with self._lock:
                        survivor.importance = max(survivor.importance, absorbed.importance)
                        survivor.content = (
                            survivor.content
                            + f"\n[Consolidated from: {absorbed.content[:80]}]"
                        )
                        absorbed.consolidated_into = survivor.id
                    merged_ids.add(absorbed.id)
                    merges += 1
                    logger.debug(
                        f"EpisodicMemory: merged {absorbed.id!r} -> {survivor.id!r} "
                        f"(sim={sim:.3f})."
                    )
        return merges

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def all_ids(self) -> List[str]:
        with self._lock:
            return list(self._episodes.keys())

    def delete(self, id: str) -> bool:
        with self._lock:
            if id in self._episodes:
                del self._episodes[id]
                return True
            return False

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "episodes": [
                    {
                        "id": e.id,
                        "content": e.content,
                        "context_snapshot": e.context_snapshot,
                        "timestamp": e.timestamp,
                        "importance": e.importance,
                        "embedding": e.embedding,
                        "consolidated_into": e.consolidated_into,
                    }
                    for e in self._episodes.values()
                ]
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodicMemory":
        em = cls()
        for ep_data in data.get("episodes", []):
            ep = _Episode(**ep_data)
            em._episodes[ep.id] = ep
        return em

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _episode_to_item(self, ep: _Episode) -> ContextItem:
        return _make_context_item(
            id=ep.id,
            content=ep.content,
            memory_type=MemoryType.EPISODIC,
            importance=ep.importance,
            timestamp=ep.timestamp,
            embedding=ep.embedding,
            metadata={"context_snapshot": ep.context_snapshot},
        )


# ---------------------------------------------------------------------------
# SemanticMemory
# ---------------------------------------------------------------------------

@dataclass
class _Fact:
    id: str
    content: str
    source: str
    confidence: float
    timestamp: float
    embedding: Optional[List[float]] = None
    last_updated: Optional[float] = None


class SemanticMemory:
    """
    Generalised factual knowledge base.

    Facts are discrete, source-attributed assertions with confidence scores.
    Retrieval is text-based (keyword substring) when no embedding is
    supplied, and embedding-cosine otherwise.

    Thread-safety: protected by ``threading.RLock``.
    """

    def __init__(self) -> None:
        self._facts: Dict[str, _Fact] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def store_fact(
        self,
        fact: str,
        source: str = "unknown",
        confidence: float = 1.0,
        embedding: Optional[List[float]] = None,
        id: Optional[str] = None,
    ) -> str:
        """
        Store a factual assertion.

        Parameters
        ----------
        fact : str
            The fact statement.
        source : str
            Provenance string (URL, document title, tool name …).
        confidence : float
            Belief strength in [0, 1].
        embedding : list of float, optional
            Semantic embedding for similarity retrieval.
        id : str, optional
            Explicit ID; auto-generated otherwise.

        Returns
        -------
        str
            Fact ID.
        """
        fact_id = id or str(uuid.uuid4())
        record = _Fact(
            id=fact_id,
            content=fact,
            source=source,
            confidence=float(max(0.0, min(1.0, confidence))),
            timestamp=time.time(),
            embedding=embedding,
        )
        with self._lock:
            self._facts[fact_id] = record
            logger.debug(f"SemanticMemory: stored fact {fact_id!r}.")
        return fact_id

    def update_fact(
        self,
        fact_id: str,
        new_content: Optional[str] = None,
        confidence: Optional[float] = None,
        embedding: Optional[List[float]] = None,
    ) -> bool:
        """
        Update content and/or confidence of an existing fact.

        Returns ``True`` if the fact was found and updated.
        """
        with self._lock:
            fact = self._facts.get(fact_id)
            if fact is None:
                return False
            if new_content is not None:
                fact.content = new_content
            if confidence is not None:
                fact.confidence = float(max(0.0, min(1.0, confidence)))
            if embedding is not None:
                fact.embedding = embedding
            fact.last_updated = time.time()
            logger.debug(f"SemanticMemory: updated fact {fact_id!r}.")
            return True

    def delete_fact(self, fact_id: str) -> bool:
        """Delete a fact by ID.  Returns ``True`` if found."""
        with self._lock:
            if fact_id in self._facts:
                del self._facts[fact_id]
                logger.debug(f"SemanticMemory: deleted fact {fact_id!r}.")
                return True
            return False

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def retrieve_facts(
        self,
        query: str,
        top_k: int = 10,
        query_embedding: Optional[List[float]] = None,
    ) -> List[ContextItem]:
        """
        Retrieve the most relevant facts for *query*.

        Strategy:
        1. If *query_embedding* is supplied, rank by cosine similarity.
        2. Otherwise, rank by case-insensitive keyword overlap, then confidence.
        """
        with self._lock:
            facts = list(self._facts.values())

        if not facts:
            return []

        if query_embedding and _NUMPY_AVAILABLE:
            candidates = [(f.id, f.embedding or []) for f in facts if f.embedding]
            ranked_ids = {cid: score for cid, score in _top_k_by_similarity(
                query_embedding, candidates, top_k
            )}
            # Include unembedded facts at score=0 if slots remain
            id_to_fact = {f.id: f for f in facts}
            result: List[ContextItem] = []
            seen = set()
            for fid, score in sorted(ranked_ids.items(), key=lambda x: -x[1]):
                fact = id_to_fact[fid]
                item = self._fact_to_item(fact)
                item.relevance = score
                result.append(item)
                seen.add(fid)
            # Pad with remaining facts (by confidence) if under top_k
            if len(result) < top_k:
                remaining = sorted(
                    [f for f in facts if f.id not in seen],
                    key=lambda f: f.confidence,
                    reverse=True,
                )
                for fact in remaining[:top_k - len(result)]:
                    result.append(self._fact_to_item(fact))
            return result[:top_k]

        # Keyword fallback
        query_lower = query.lower()
        tokens = set(query_lower.split())

        def score_fn(f: _Fact) -> float:
            text_lower = f.content.lower()
            overlap = sum(1 for t in tokens if t in text_lower)
            return overlap + f.confidence * 0.1

        ranked = sorted(facts, key=score_fn, reverse=True)
        return [self._fact_to_item(f) for f in ranked[:top_k]]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def all_ids(self) -> List[str]:
        with self._lock:
            return list(self._facts.keys())

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "facts": [
                    {
                        "id": f.id,
                        "content": f.content,
                        "source": f.source,
                        "confidence": f.confidence,
                        "timestamp": f.timestamp,
                        "embedding": f.embedding,
                        "last_updated": f.last_updated,
                    }
                    for f in self._facts.values()
                ]
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticMemory":
        sm = cls()
        for fd in data.get("facts", []):
            fact = _Fact(**fd)
            sm._facts[fact.id] = fact
        return sm

    def _fact_to_item(self, fact: _Fact) -> ContextItem:
        return _make_context_item(
            id=fact.id,
            content=fact.content,
            memory_type=MemoryType.SEMANTIC,
            importance=fact.confidence,
            timestamp=fact.timestamp,
            embedding=fact.embedding,
            metadata={"source": fact.source, "confidence": fact.confidence},
        )


# ---------------------------------------------------------------------------
# ProceduralMemory
# ---------------------------------------------------------------------------

@dataclass
class _Procedure:
    id: str
    trigger: str
    action_sequence: List[str]
    success_rate: float
    invocation_count: int
    success_count: int
    timestamp: float
    trigger_embedding: Optional[List[float]] = None


class ProceduralMemory:
    """
    Memory for learned action sequences: *if trigger then do [actions]*.

    Procedures track their empirical success rate and can be ranked by
    both trigger similarity and historical performance.

    Thread-safety: protected by ``threading.RLock``.
    """

    def __init__(self) -> None:
        self._procedures: Dict[str, _Procedure] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def store_procedure(
        self,
        trigger: str,
        action_sequence: List[str],
        success_rate: float = 1.0,
        trigger_embedding: Optional[List[float]] = None,
        id: Optional[str] = None,
    ) -> str:
        """
        Store or register an action sequence associated with a *trigger*.

        Parameters
        ----------
        trigger : str
            Natural-language description of the condition that activates
            this procedure (e.g. ``"user asks to summarise a document"``).
        action_sequence : list of str
            Ordered list of action descriptions or tool names.
        success_rate : float
            Prior success rate in [0, 1]; defaults to 1.0 for new procedures.
        trigger_embedding : list of float, optional
            Embedding of the trigger string for semantic matching.
        id : str, optional
            Explicit ID; auto-generated otherwise.

        Returns
        -------
        str
            Procedure ID.
        """
        proc_id = id or str(uuid.uuid4())
        proc = _Procedure(
            id=proc_id,
            trigger=trigger,
            action_sequence=list(action_sequence),
            success_rate=float(max(0.0, min(1.0, success_rate))),
            invocation_count=0,
            success_count=0,
            timestamp=time.time(),
            trigger_embedding=trigger_embedding,
        )
        with self._lock:
            self._procedures[proc_id] = proc
            logger.debug(f"ProceduralMemory: stored procedure {proc_id!r}.")
        return proc_id

    def update_success_rate(self, procedure_id: str, success: bool) -> bool:
        """
        Record the outcome of a procedure invocation.

        Updates ``invocation_count``, ``success_count``, and recomputes the
        Laplace-smoothed ``success_rate``.

        Returns ``True`` if the procedure was found.
        """
        with self._lock:
            proc = self._procedures.get(procedure_id)
            if proc is None:
                return False
            proc.invocation_count += 1
            if success:
                proc.success_count += 1
            # Laplace smoothing: (successes + 1) / (total + 2)
            proc.success_rate = (proc.success_count + 1) / (proc.invocation_count + 2)
            logger.debug(
                f"ProceduralMemory: updated {procedure_id!r} success_rate="
                f"{proc.success_rate:.3f} ({proc.success_count}/{proc.invocation_count})."
            )
            return True

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def retrieve_matching(
        self,
        trigger: str,
        top_k: int = 5,
        trigger_embedding: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return the *top_k* procedures whose trigger best matches the query.

        Ranking combines semantic similarity (if embeddings are available)
        and historical success rate.

        Returns
        -------
        list of dict
            Each dict contains: ``id``, ``trigger``, ``action_sequence``,
            ``success_rate``, ``invocation_count``, ``similarity``.
        """
        with self._lock:
            procs = list(self._procedures.values())

        if not procs:
            return []

        # Compute raw scores
        sim_scores: Dict[str, float] = {}

        if trigger_embedding and _NUMPY_AVAILABLE:
            candidates = [
                (p.id, p.trigger_embedding or [])
                for p in procs if p.trigger_embedding
            ]
            for pid, score in _top_k_by_similarity(trigger_embedding, candidates, len(procs)):
                sim_scores[pid] = score
        else:
            # Keyword overlap fallback
            query_tokens = set(trigger.lower().split())
            for p in procs:
                trig_tokens = set(p.trigger.lower().split())
                overlap = len(query_tokens & trig_tokens)
                union = len(query_tokens | trig_tokens)
                sim_scores[p.id] = overlap / union if union else 0.0

        # Combined score: 0.6 * similarity + 0.4 * success_rate
        def combined(p: _Procedure) -> float:
            return 0.6 * sim_scores.get(p.id, 0.0) + 0.4 * p.success_rate

        ranked = sorted(procs, key=combined, reverse=True)[:top_k]

        return [
            {
                "id": p.id,
                "trigger": p.trigger,
                "action_sequence": p.action_sequence,
                "success_rate": p.success_rate,
                "invocation_count": p.invocation_count,
                "similarity": sim_scores.get(p.id, 0.0),
            }
            for p in ranked
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def all_ids(self) -> List[str]:
        with self._lock:
            return list(self._procedures.keys())

    def delete(self, id: str) -> bool:
        with self._lock:
            if id in self._procedures:
                del self._procedures[id]
                return True
            return False

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "procedures": [
                    {
                        "id": p.id,
                        "trigger": p.trigger,
                        "action_sequence": p.action_sequence,
                        "success_rate": p.success_rate,
                        "invocation_count": p.invocation_count,
                        "success_count": p.success_count,
                        "timestamp": p.timestamp,
                        "trigger_embedding": p.trigger_embedding,
                    }
                    for p in self._procedures.values()
                ]
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProceduralMemory":
        pm = cls()
        for pd_ in data.get("procedures", []):
            proc = _Procedure(**pd_)
            pm._procedures[proc.id] = proc
        return pm


# ---------------------------------------------------------------------------
# LongTermMemory (facade)
# ---------------------------------------------------------------------------

class LongTermMemory:
    """
    Unified long-term memory that delegates storage and retrieval to the
    three specialised subsystems based on ``ContextItem.memory_type``.

    Dispatch table
    --------------
    MemoryType.EPISODIC   -> EpisodicMemory
    MemoryType.PROCEDURAL -> ProceduralMemory
    Everything else       -> SemanticMemory  (SEMANTIC, WORKING, GOAL, PLAN, …)

    Persistence
    -----------
    ``save_to_disk(path)`` serialises all three subsystems to a single JSON
    file.  ``load_from_disk(path)`` restores them.  The format is a plain
    JSON object with three top-level keys: ``"episodic"``, ``"semantic"``,
    ``"procedural"``.

    Thread-safety
    -------------
    The facade itself is protected by an ``RLock``; each subsystem also
    maintains its own lock for fine-grained concurrency.
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        self._config = config or {}
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.procedural = ProceduralMemory()
        self._lock = threading.RLock()
        logger.debug("LongTermMemory initialised.")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def store(self, item: ContextItem) -> str:
        """
        Store *item* in the appropriate subsystem.

        For ``MemoryType.EPISODIC``, ``item.metadata`` may contain a
        ``"context_snapshot"`` dict that is forwarded to
        ``EpisodicMemory.add_episode``.

        For ``MemoryType.PROCEDURAL``, ``item.metadata`` may contain
        ``"trigger"`` (str) and ``"action_sequence"`` (list of str).  If
        absent, ``item.content`` is used as the trigger with an empty
        action sequence.

        Returns
        -------
        str
            The ID of the stored record (may differ from ``item.id`` if a
            new UUID was generated internally).
        """
        if item.memory_type == MemoryType.EPISODIC:
            return self.episodic.add_episode(
                content=item.content,
                context_snapshot=item.metadata.get("context_snapshot", {}),
                timestamp=item.timestamp,
                importance=item.importance,
                embedding=item.embedding,
                id=item.id,
            )

        if item.memory_type == MemoryType.PROCEDURAL:
            trigger = item.metadata.get("trigger", item.content)
            action_seq = item.metadata.get("action_sequence", [])
            success_rate = item.metadata.get("success_rate", 1.0)
            return self.procedural.store_procedure(
                trigger=trigger,
                action_sequence=action_seq,
                success_rate=success_rate,
                trigger_embedding=item.embedding,
                id=item.id,
            )

        # Default: semantic
        return self.semantic.store_fact(
            fact=item.content,
            source=item.metadata.get("source", "unknown"),
            confidence=item.importance,
            embedding=item.embedding,
            id=item.id,
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def retrieve_for_query(
        self,
        query: str,
        top_k: int = 20,
        query_embedding: Optional[List[float]] = None,
    ) -> List[ContextItem]:
        """
        Retrieve the most relevant items across *all* subsystems for *query*.

        The result list is sorted by descending ``relevance`` (similarity
        score when embeddings are available, keyword score otherwise) and
        truncated to *top_k*.

        Parameters
        ----------
        query : str
            Natural-language query string.
        top_k : int
            Maximum total items returned.
        query_embedding : list of float, optional
            Pre-computed embedding of *query*; passed through to subsystems
            that support similarity search.

        Returns
        -------
        list of ContextItem
        """
        per_system = max(1, top_k // 3)

        ep_items = self.episodic.retrieve_similar(
            query_embedding or [], top_k=per_system
        ) if query_embedding else self.episodic.retrieve_recent(n=per_system)

        sem_items = self.semantic.retrieve_facts(
            query=query,
            top_k=per_system,
            query_embedding=query_embedding,
        )

        # Procedural: retrieve matching, convert to ContextItem
        proc_matches = self.procedural.retrieve_matching(
            trigger=query,
            top_k=per_system,
            trigger_embedding=query_embedding,
        )
        proc_items = [
            _make_context_item(
                id=m["id"],
                content=f"Procedure trigger: {m['trigger']}\nActions: "
                        + " -> ".join(m["action_sequence"]),
                memory_type=MemoryType.PROCEDURAL,
                importance=m["success_rate"],
                timestamp=time.time(),
                embedding=None,
                metadata=m,
            )
            for m in proc_matches
        ]

        combined = ep_items + sem_items + proc_items
        # Sort by relevance desc, then importance desc as tiebreak
        combined.sort(key=lambda x: (x.relevance, x.importance), reverse=True)

        # Update access counts
        for item in combined[:top_k]:
            item.update_access()

        logger.debug(
            f"LongTermMemory: retrieved {len(combined[:top_k])} items for query."
        )
        return combined[:top_k]

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def get_all_ids(self) -> List[str]:
        """Return every stored ID across all subsystems."""
        return (
            self.episodic.all_ids()
            + self.semantic.all_ids()
            + self.procedural.all_ids()
        )

    def delete(self, id: str) -> bool:
        """
        Delete item with *id* from whichever subsystem holds it.

        Returns ``True`` if found and deleted.
        """
        if self.episodic.delete(id):
            return True
        if self.semantic.delete_fact(id):
            return True
        if self.procedural.delete(id):
            return True
        return False

    def get_stats(self) -> LongTermMemoryStats:
        """Return aggregate statistics across all subsystems."""
        ep_ids = self.episodic.all_ids()
        sem_ids = self.semantic.all_ids()
        proc_ids = self.procedural.all_ids()

        ep_count = len(ep_ids)
        sem_count = len(sem_ids)
        proc_count = len(proc_ids)
        total = ep_count + sem_count + proc_count

        # Average importance from episodic
        with self.episodic._lock:
            ep_importances = [
                e.importance for e in self.episodic._episodes.values()
                if e.consolidated_into is None
            ]
        avg_imp = sum(ep_importances) / len(ep_importances) if ep_importances else 0.0

        # Average confidence from semantic
        with self.semantic._lock:
            confidences = [f.confidence for f in self.semantic._facts.values()]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        # Average success rate from procedural
        with self.procedural._lock:
            rates = [p.success_rate for p in self.procedural._procedures.values()]
        avg_rate = sum(rates) / len(rates) if rates else 0.0

        return LongTermMemoryStats(
            episodic_count=ep_count,
            semantic_count=sem_count,
            procedural_count=proc_count,
            total_count=total,
            avg_importance=avg_imp,
            avg_semantic_confidence=avg_conf,
            avg_procedural_success_rate=avg_rate,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_to_disk(self, path: str) -> None:
        """
        Serialise all three subsystems to a single JSON file at *path*.

        Creates parent directories as needed.  Writes atomically by
        first writing to a ``.tmp`` file then renaming.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")

        payload = {
            "version": "1.0",
            "saved_at": time.time(),
            "episodic": self.episodic.to_dict(),
            "semantic": self.semantic.to_dict(),
            "procedural": self.procedural.to_dict(),
        }

        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

        # Atomic rename
        tmp.replace(target)
        logger.info(f"LongTermMemory: saved to {target}.")

    def load_from_disk(self, path: str) -> None:
        """
        Restore all subsystems from a JSON file previously written by
        ``save_to_disk``.

        Raises ``FileNotFoundError`` if *path* does not exist.
        Raises ``ValueError`` if the file format is unrecognised.
        """
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"LongTermMemory: snapshot not found at {source}.")

        with open(source, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        version = payload.get("version", "unknown")
        if version != "1.0":
            raise ValueError(
                f"LongTermMemory: unsupported snapshot version {version!r}. "
                "Expected '1.0'."
            )

        with self._lock:
            self.episodic = EpisodicMemory.from_dict(payload.get("episodic", {}))
            self.semantic = SemanticMemory.from_dict(payload.get("semantic", {}))
            self.procedural = ProceduralMemory.from_dict(payload.get("procedural", {}))

        logger.info(
            f"LongTermMemory: loaded from {source} "
            f"(episodic={len(self.episodic.all_ids())}, "
            f"semantic={len(self.semantic.all_ids())}, "
            f"procedural={len(self.procedural.all_ids())})."
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.get_all_ids())

    def __repr__(self) -> str:  # pragma: no cover
        stats = self.get_stats()
        return (
            f"LongTermMemory(episodic={stats.episodic_count}, "
            f"semantic={stats.semantic_count}, "
            f"procedural={stats.procedural_count})"
        )
