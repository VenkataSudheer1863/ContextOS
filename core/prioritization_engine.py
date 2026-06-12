"""
ContextOS Prioritization Engine
Multi-signal scoring with task-adaptive weights and normalization.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .orchestrator import ContextItem


# ---------------------------------------------------------------------------
# Task-type weight presets
# ---------------------------------------------------------------------------

TASK_WEIGHT_PRESETS: Dict[str, Dict[str, float]] = {
    "factual": {
        "relevance":  0.50,
        "recency":    0.15,
        "importance": 0.25,
        "novelty":    0.10,
    },
    "creative": {
        "relevance":  0.25,
        "recency":    0.10,
        "importance": 0.20,
        "novelty":    0.45,
    },
    "coding": {
        "relevance":  0.45,
        "recency":    0.20,
        "importance": 0.25,
        "novelty":    0.10,
    },
    "conversational": {
        "relevance":  0.35,
        "recency":    0.35,
        "importance": 0.15,
        "novelty":    0.15,
    },
    "general": {
        "relevance":  0.40,
        "recency":    0.25,
        "importance": 0.20,
        "novelty":    0.15,
    },
}


# ---------------------------------------------------------------------------
# TF-IDF fallback for relevance when embeddings are absent
# ---------------------------------------------------------------------------

class _TFIDFScorer:
    """
    Lightweight TF-IDF cosine similarity without external dependencies.
    Recomputes the IDF corpus on every call — suitable for small batches.
    """

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        import re
        return re.findall(r"[a-z0-9]+", text.lower())

    def _tf(self, tokens: List[str]) -> Dict[str, float]:
        counts: Dict[str, int] = defaultdict(int)
        for t in tokens:
            counts[t] += 1
        total = len(tokens) or 1
        return {w: c / total for w, c in counts.items()}

    def score(self, query: str, item_content: str, corpus: List[str]) -> float:
        """Return TF-IDF cosine similarity between query and item_content."""
        all_docs = corpus + [query]
        df: Dict[str, int] = defaultdict(int)
        tokenized = [self._tokenize(d) for d in all_docs]
        for tokens in tokenized:
            for w in set(tokens):
                df[w] += 1

        n = len(all_docs)
        idf = {w: math.log((n + 1) / (c + 1)) + 1.0 for w, c in df.items()}

        def tfidf_vec(tokens: List[str]) -> Dict[str, float]:
            tf = self._tf(tokens)
            return {w: tf[w] * idf.get(w, 1.0) for w in tf}

        query_tokens = self._tokenize(query)
        item_tokens = self._tokenize(item_content)

        qv = tfidf_vec(query_tokens)
        iv = tfidf_vec(item_tokens)

        common = set(qv) & set(iv)
        dot = sum(qv[w] * iv[w] for w in common)
        nq = math.sqrt(sum(v * v for v in qv.values()))
        ni = math.sqrt(sum(v * v for v in iv.values()))
        if nq < 1e-10 or ni < 1e-10:
            return 0.0
        return dot / (nq * ni)


# ---------------------------------------------------------------------------
# Score distribution container (plain dict helper — no extra imports needed)
# ---------------------------------------------------------------------------


def _empty_distribution() -> Dict[str, Any]:
    return {
        "count": 0,
        "min": float("inf"),
        "max": float("-inf"),
        "mean": 0.0,
        "std": 0.0,
        "signal_means": {
            "relevance": 0.0,
            "recency": 0.0,
            "importance": 0.0,
            "novelty": 0.0,
        },
    }


# ---------------------------------------------------------------------------
# PrioritizationEngine
# ---------------------------------------------------------------------------

class PrioritizationEngine:
    """
    Multi-signal priority scorer for ContextItem objects.

    Scoring signals
    ---------------
    * relevance  — cosine similarity (embedding) or TF-IDF (text fallback)
    * recency    — exponential decay: exp(-lambda * age_seconds)
    * importance — item.importance field (user/system assigned, [0, 1])
    * novelty    — 1 - max_cosine_sim(item, existing_context)

    Final score
    -----------
    priority = w_rel * relevance
             + w_rec * recency
             + w_imp * importance
             + w_nov * novelty

    Weights are adjusted per task type using TASK_WEIGHT_PRESETS.
    All individual signals are normalized to [0, 1] before weighting.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = (config or {}).get("scheduler", {})
        w = cfg.get("weights", {})
        self._base_weights: Dict[str, float] = {
            "relevance":  float(w.get("relevance",  0.40)),
            "recency":    float(w.get("recency",    0.25)),
            "importance": float(w.get("importance", 0.20)),
            "novelty":    float(w.get("novelty",    0.15)),
        }
        self.decay_lambda: float = float(cfg.get("decay_lambda", 0.0001))
        self._tfidf = _TFIDFScorer()
        self._score_history: List[Dict[str, float]] = []
        logger.info(
            f"PrioritizationEngine ready — "
            f"decay_lambda={self.decay_lambda}, "
            f"base_weights={self._base_weights}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_item(
        self,
        item: ContextItem,
        query_embedding: Optional[List[float]],
        query_text: str,
        existing_context: Optional[List[ContextItem]] = None,
        task_type: str = "general",
    ) -> float:
        """
        Compute a single item's priority score.

        Parameters
        ----------
        item             : ContextItem to score
        query_embedding  : dense vector for the current query (may be None)
        query_text       : raw query string used for TF-IDF fallback
        existing_context : items already committed to context (for novelty)
        task_type        : one of the keys in TASK_WEIGHT_PRESETS

        Returns
        -------
        priority : float in approximately [0, 1]
        """
        weights = self._task_weights(task_type)

        rel = self._relevance_score(item, query_embedding, query_text, existing_context or [])
        rec = self._recency_score(item)
        imp = self._importance_score(item)
        nov = self._novelty_score(item, existing_context or [])

        priority = (
            weights["relevance"]  * rel
            + weights["recency"]    * rec
            + weights["importance"] * imp
            + weights["novelty"]    * nov
        )

        # Persist signals for later inspection / normalization
        item.metadata.update({
            "relevance_score":  round(rel, 6),
            "recency_score":    round(rec, 6),
            "importance_score": round(imp, 6),
            "novelty_score":    round(nov, 6),
            "final_priority":   round(priority, 6),
            "task_type":        task_type,
        })
        self._score_history.append({
            "relevance":  rel,
            "recency":    rec,
            "importance": imp,
            "novelty":    nov,
            "priority":   priority,
        })
        return priority

    def score_all(
        self,
        items: List[ContextItem],
        query_embedding: Optional[List[float]],
        query_text: str,
        existing_context: Optional[List[ContextItem]] = None,
        task_type: str = "general",
    ) -> List[ContextItem]:
        """
        Score every item and return them sorted by descending priority.

        Parameters
        ----------
        items            : candidate ContextItems
        query_embedding  : dense query vector (may be None)
        query_text       : raw query for TF-IDF fallback
        existing_context : items already in context window
        task_type        : task classification for weight adjustment

        Returns
        -------
        List[ContextItem] sorted by final_priority descending
        """
        if not items:
            return []

        corpus = [i.content for i in items]
        context = list(existing_context or [])

        for item in items:
            self.score_item(
                item,
                query_embedding,
                query_text,
                context,
                task_type=task_type,
            )

        items_sorted = sorted(
            items,
            key=lambda i: i.metadata.get("final_priority", 0.0),
            reverse=True,
        )
        logger.debug(
            f"Scored {len(items_sorted)} items for task_type={task_type!r}; "
            f"top priority={items_sorted[0].metadata.get('final_priority', 0):.4f}"
            if items_sorted else "No items to score."
        )
        return items_sorted

    def normalize_scores(self, items: List[ContextItem]) -> List[ContextItem]:
        """
        Min-max normalize final_priority across items to the [0, 1] range.
        Modifies items in-place and returns them.
        """
        if not items:
            return items

        scores = [i.metadata.get("final_priority", 0.0) for i in items]
        lo, hi = min(scores), max(scores)

        if hi - lo < 1e-10:
            for item in items:
                item.metadata["final_priority"] = 1.0
            logger.debug("normalize_scores: all items have equal priority; set to 1.0")
            return items

        for item in items:
            raw = item.metadata.get("final_priority", 0.0)
            item.metadata["final_priority"] = round((raw - lo) / (hi - lo), 6)

        logger.debug(f"normalize_scores: range [{lo:.4f}, {hi:.4f}] -> [0, 1]")
        return items

    def get_score_distribution(self) -> Dict[str, Any]:
        """
        Return statistical summary of all scores computed since construction.

        Returns
        -------
        Dict with keys: count, min, max, mean, std, signal_means
        """
        dist = _empty_distribution()
        if not self._score_history:
            return dist

        priorities = [h["priority"] for h in self._score_history]
        dist["count"] = len(priorities)
        dist["min"] = round(min(priorities), 6)
        dist["max"] = round(max(priorities), 6)
        mean = sum(priorities) / len(priorities)
        dist["mean"] = round(mean, 6)
        variance = sum((p - mean) ** 2 for p in priorities) / len(priorities)
        dist["std"] = round(math.sqrt(variance), 6)

        for signal in ("relevance", "recency", "importance", "novelty"):
            vals = [h[signal] for h in self._score_history]
            dist["signal_means"][signal] = round(sum(vals) / len(vals), 6)

        return dist

    def reset_history(self) -> None:
        """Clear accumulated score history."""
        self._score_history.clear()
        logger.debug("PrioritizationEngine score history cleared.")

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def _relevance_score(
        self,
        item: ContextItem,
        query_embedding: Optional[List[float]],
        query_text: str,
        context: List[ContextItem],
    ) -> float:
        """
        Cosine similarity if both embeddings are available, else TF-IDF.
        """
        if query_embedding and item.embedding:
            return self._cosine(query_embedding, item.embedding)

        # TF-IDF fallback
        corpus = [i.content for i in context] if context else [item.content]
        return self._tfidf.score(query_text, item.content, corpus)

    def _recency_score(self, item: ContextItem) -> float:
        """Exponential temporal decay: exp(-lambda * age_seconds)."""
        age = item.age_seconds()
        return math.exp(-self.decay_lambda * age)

    @staticmethod
    def _importance_score(item: ContextItem) -> float:
        """Clamp item.importance to [0, 1]."""
        return max(0.0, min(1.0, item.importance))

    def _novelty_score(
        self,
        item: ContextItem,
        existing_context: List[ContextItem],
    ) -> float:
        """
        1 - max_cosine_sim(item, existing_context).
        Returns 1.0 (fully novel) when embeddings are unavailable or
        existing_context is empty.
        """
        if not existing_context or item.embedding is None:
            return 1.0

        sims = [
            self._cosine(item.embedding, ctx.embedding)
            for ctx in existing_context
            if ctx.embedding is not None
        ]
        return 1.0 - max(sims) if sims else 1.0

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def _task_weights(self, task_type: str) -> Dict[str, float]:
        """
        Return weights for the given task type.
        Falls back to 'general' for unknown task types.
        """
        preset = TASK_WEIGHT_PRESETS.get(task_type)
        if preset is None:
            logger.warning(
                f"Unknown task_type={task_type!r}; using 'general' weights."
            )
            preset = TASK_WEIGHT_PRESETS["general"]
        return preset

    def get_weights_for_task(self, task_type: str) -> Dict[str, float]:
        """Expose weight preset for inspection or debugging."""
        return dict(self._task_weights(task_type))

    def register_task_preset(
        self, task_type: str, weights: Dict[str, float]
    ) -> None:
        """
        Register a custom weight preset for a new task type.

        Parameters
        ----------
        task_type : identifier string
        weights   : dict with keys relevance, recency, importance, novelty
                    Values should sum to approximately 1.0.
        """
        required = {"relevance", "recency", "importance", "novelty"}
        missing = required - set(weights)
        if missing:
            raise ValueError(f"Weight preset missing keys: {missing}")
        TASK_WEIGHT_PRESETS[task_type] = {k: float(v) for k, v in weights.items()}
        logger.info(f"Registered task preset {task_type!r}: {weights}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return dot / (na * nb)

    @staticmethod
    def _top_k_items(
        items: List[ContextItem],
        k: int,
    ) -> List[ContextItem]:
        """Return the k highest-priority items."""
        return sorted(
            items,
            key=lambda i: i.metadata.get("final_priority", 0.0),
            reverse=True,
        )[:k]
