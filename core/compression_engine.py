"""
ContextOS — Compression Engine
================================
Implements multiple context-compression strategies for long-horizon agents.

Strategies
----------
- ExtractiveCompressor   : TF-IDF sentence ranking
- AbstractiveCompressor  : LLM-based summarisation (httpx), falls back to extractive
- HierarchicalCompressor : Three-level progressive removal
- ProgressiveCompressor  : Iterative tightening with quality floor

The CompressionEngine selects the right strategy based on item type,
importance score, and remaining token budget.
"""

from __future__ import annotations

import math
import os
import re
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Optional heavy dependencies — gracefully degrade when absent
# ---------------------------------------------------------------------------
try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HTTPX_AVAILABLE = False
    warnings.warn("httpx not installed — AbstractiveCompressor will fall back to extractive.", stacklevel=1)

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _NUMPY_AVAILABLE = False

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
    _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover
    _TIKTOKEN_AVAILABLE = False
    _TIKTOKEN_ENC = None


# ---------------------------------------------------------------------------
# Domain types — lightweight to avoid circular imports
# ---------------------------------------------------------------------------

class ItemType(str, Enum):
    GOAL = "GOAL"
    PLAN = "PLAN"
    OBSERVATION = "OBSERVATION"
    TOOL_OUTPUT = "TOOL_OUTPUT"
    EPISODIC = "EPISODIC"
    SEMANTIC = "SEMANTIC"
    PROCEDURAL = "PROCEDURAL"
    WORKING = "WORKING"


@dataclass
class ContextItem:
    """A single unit of context managed by ContextOS."""

    content: str
    item_type: ItemType = ItemType.WORKING
    importance: float = 0.5          # 0.0 – 1.0
    recency: float = 1.0             # 0.0 – 1.0  (1 = most recent)
    metadata: Dict = field(default_factory=dict)
    item_id: Optional[str] = None
    compressed: bool = False
    original_content: Optional[str] = None


@dataclass
class CompressionMetrics:
    """Quality metrics produced after compressing a piece of text."""

    original_tokens: int
    compressed_tokens: int
    ratio: float                     # compressed / original   (lower = more compressed)
    rouge_l: float = 0.0             # estimated ROUGE-L  (0–1)
    semantic_similarity: float = 0.0 # cosine similarity if embeddings available (0–1)

    @property
    def compression_rate(self) -> float:
        """Fraction of tokens removed (0 = no compression, 1 = all removed)."""
        return 1.0 - self.ratio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """Split *text* into sentences, keeping non-empty ones."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


def _tfidf_scores(sentences: List[str]) -> List[float]:
    """Return a TF-IDF importance score for each sentence."""
    if not sentences:
        return []

    # Build term-frequency per sentence
    tf_per_sent: List[Counter] = []
    for sent in sentences:
        tokens = re.findall(r"\b\w+\b", sent.lower())
        tf_per_sent.append(Counter(tokens))

    # Document frequency across sentences
    all_terms: set = set()
    for tf in tf_per_sent:
        all_terms |= set(tf.keys())

    n = len(sentences)
    df: Dict[str, int] = {}
    for term in all_terms:
        df[term] = sum(1 for tf in tf_per_sent if term in tf)

    idf: Dict[str, float] = {
        term: math.log((n + 1) / (cnt + 1)) + 1.0
        for term, cnt in df.items()
    }

    scores: List[float] = []
    for tf in tf_per_sent:
        total = sum(tf.values()) or 1
        score = sum((freq / total) * idf.get(term, 1.0) for term, freq in tf.items())
        scores.append(score)

    # Normalise to [0, 1]
    max_s = max(scores) if scores else 1.0
    return [s / (max_s or 1.0) for s in scores]


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors."""
    if not _NUMPY_AVAILABLE:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        return dot / ((norm_a * norm_b) or 1e-9)
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1e-9
    return float(np.dot(va, vb) / denom)


def _bow_vector(text: str, vocab: Dict[str, int]) -> List[float]:
    """Simple bag-of-words vector over *vocab*."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    vec = [0.0] * len(vocab)
    for tok in tokens:
        if tok in vocab:
            vec[vocab[tok]] += 1.0
    return vec


def _estimate_rouge_l(reference: str, hypothesis: str) -> float:
    """Lightweight ROUGE-L approximation using LCS length."""
    ref_tokens = re.findall(r"\b\w+\b", reference.lower())
    hyp_tokens = re.findall(r"\b\w+\b", hypothesis.lower())
    if not ref_tokens or not hyp_tokens:
        return 0.0
    m, n = len(ref_tokens), len(hyp_tokens)
    # DP table — O(m*n) but both are short after compression
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    precision = lcs / n if n else 0.0
    recall = lcs / m if m else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return round(f1, 4)


def _keyword_extract(text: str, top_k: int = 12) -> str:
    """Extract the *top_k* most frequent content words as a compact summary."""
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "to",
        "of", "in", "on", "at", "for", "with", "by", "from", "as",
        "that", "this", "it", "its", "i", "we", "you", "he", "she",
        "they", "and", "or", "but", "not", "no", "so", "if", "then",
    }
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    freq = Counter(tok for tok in tokens if tok not in stopwords)
    keywords = [word for word, _ in freq.most_common(top_k)]
    return "[Keywords] " + ", ".join(keywords) if keywords else text[:120]


# ===========================================================================
# Base compressor
# ===========================================================================

class BaseCompressor:
    """Abstract interface every strategy must implement."""

    def compress(self, text: str, target_ratio: float) -> str:
        """Return a compressed version of *text* at roughly *target_ratio* length."""
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__


# ===========================================================================
# 1. Extractive Compressor
# ===========================================================================

class ExtractiveCompressor(BaseCompressor):
    """Select top sentences by TF-IDF score while always keeping first/last."""

    def __init__(self, preserve_ratio: float = 0.5):
        if not 0.0 < preserve_ratio <= 1.0:
            raise ValueError("preserve_ratio must be in (0, 1]")
        self.preserve_ratio = preserve_ratio

    def compress(self, text: str, target_ratio: float) -> str:  # noqa: D102
        ratio = min(target_ratio, self.preserve_ratio)
        sentences = _split_sentences(text)
        n = len(sentences)

        if n <= 2:
            return text  # nothing to remove

        scores = _tfidf_scores(sentences)
        n_keep = max(2, math.ceil(n * ratio))

        # Always keep first and last
        anchors = {0, n - 1}
        middle_indices = list(range(1, n - 1))

        # Sort middle sentences by score (descending), keep top ones
        middle_ranked = sorted(middle_indices, key=lambda i: scores[i], reverse=True)
        slots_for_middle = max(0, n_keep - len(anchors))
        selected_middle = set(middle_ranked[:slots_for_middle])

        kept = sorted(anchors | selected_middle)
        return " ".join(sentences[i] for i in kept)


# ===========================================================================
# 2. Abstractive Compressor
# ===========================================================================

class AbstractiveCompressor(BaseCompressor):
    """LLM-based summarisation via the Anthropic Messages API (httpx)."""

    SUMMARISE_TEMPLATE = (
        "Summarize the following preserving all key facts:\n{text}\n\nSummary:"
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-6",
        api_base: str = "https://api.anthropic.com",
        timeout: float = 30.0,
        fallback: Optional[BaseCompressor] = None,
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self._fallback = fallback or ExtractiveCompressor()

    def compress(self, text: str, target_ratio: float) -> str:  # noqa: D102
        if not _HTTPX_AVAILABLE or not self.api_key:
            return self._fallback.compress(text, target_ratio)

        prompt = self.SUMMARISE_TEMPLATE.format(text=text)
        # Approximate max tokens for response based on target ratio
        src_words = len(text.split())
        max_tokens = max(64, int(src_words * target_ratio * 1.3))

        try:
            response = httpx.post(
                f"{self.api_base}/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            summary = data["content"][0]["text"].strip()
            return summary if summary else self._fallback.compress(text, target_ratio)

        except Exception:
            return self._fallback.compress(text, target_ratio)


# ===========================================================================
# 3. Hierarchical Compressor
# ===========================================================================

class HierarchicalCompressor(BaseCompressor):
    """Three-level compression chosen by how much budget pressure exists.

    Level 1 (mild)    — deduplicate highly similar sentences (cosine sim > 0.92)
    Level 2 (medium)  — TF-IDF sentence extraction
    Level 3 (heavy)   — keyword extraction only
    """

    _DEDUP_THRESHOLD = 0.92
    _LEVEL_THRESHOLDS = (0.70, 0.40)   # > 0.70 → L1; > 0.40 → L2; else → L3

    def __init__(self):
        self._extractive = ExtractiveCompressor()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_vocab(self, sentences: List[str]) -> Dict[str, int]:
        all_words = re.findall(r"\b\w+\b", " ".join(sentences).lower())
        return {w: i for i, w in enumerate(sorted(set(all_words)))}

    def _deduplicate(self, sentences: List[str]) -> List[str]:
        """Remove sentences whose BoW cosine similarity to any earlier kept sentence
        exceeds the threshold."""
        if len(sentences) <= 2:
            return sentences
        vocab = self._build_vocab(sentences)
        kept: List[str] = [sentences[0]]
        kept_vecs: List[List[float]] = [_bow_vector(sentences[0], vocab)]
        for sent in sentences[1:]:
            vec = _bow_vector(sent, vocab)
            redundant = any(
                _cosine_similarity(vec, kv) >= self._DEDUP_THRESHOLD
                for kv in kept_vecs
            )
            if not redundant:
                kept.append(sent)
                kept_vecs.append(vec)
        # Always keep last sentence
        if sentences[-1] not in kept:
            kept.append(sentences[-1])
        return kept

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------

    def _choose_level(self, target_ratio: float) -> int:
        if target_ratio > self._LEVEL_THRESHOLDS[0]:
            return 1
        if target_ratio > self._LEVEL_THRESHOLDS[1]:
            return 2
        return 3

    def compress(self, text: str, target_ratio: float) -> str:  # noqa: D102
        level = self._choose_level(target_ratio)

        if level == 1:
            sentences = _split_sentences(text)
            deduped = self._deduplicate(sentences)
            return " ".join(deduped)

        if level == 2:
            return self._extractive.compress(text, target_ratio)

        # Level 3 — keywords only
        return _keyword_extract(text, top_k=max(8, int(len(text.split()) * 0.15)))


# ===========================================================================
# 4. Progressive Compressor
# ===========================================================================

class ProgressiveCompressor(BaseCompressor):
    """Iteratively tighten compression until the token budget is met or the
    minimum quality threshold would be violated."""

    def __init__(
        self,
        initial_ratio: float = 0.70,
        step: float = 0.10,
        minimum_quality_threshold: float = 0.25,
        engine: Optional["CompressionEngine"] = None,
    ):
        if not 0.0 < minimum_quality_threshold < 1.0:
            raise ValueError("minimum_quality_threshold must be in (0, 1)")
        self.initial_ratio = initial_ratio
        self.step = step
        self.minimum_quality_threshold = minimum_quality_threshold
        self._engine = engine  # used for token estimation
        self._hierarchical = HierarchicalCompressor()
        self._extractive = ExtractiveCompressor()

    def compress(self, text: str, target_ratio: float) -> str:  # noqa: D102
        ratio = min(self.initial_ratio, target_ratio)
        best = text
        while ratio >= self.minimum_quality_threshold:
            candidate = self._hierarchical.compress(text, ratio)
            if not candidate.strip():
                break
            best = candidate
            # If the candidate already satisfies the target ratio, stop early
            est_candidate = len(candidate.split())
            est_original = len(text.split()) or 1
            achieved = est_candidate / est_original
            if achieved <= target_ratio:
                break
            ratio = round(ratio - self.step, 4)

        return best


# ===========================================================================
# Compression Engine — orchestrator
# ===========================================================================

# Item types that must never be compressed
_PROTECTED_TYPES = {ItemType.GOAL, ItemType.PLAN}
# Item types eligible for aggressive compression
_COMPRESSIBLE_TYPES = {ItemType.OBSERVATION, ItemType.EPISODIC, ItemType.TOOL_OUTPUT}
# High importance threshold above which only extractive is used
_HIGH_IMPORTANCE = 0.75


class CompressionEngine:
    """Main engine that selects and applies compression strategies.

    Parameters
    ----------
    llm_api_key:
        Optional API key forwarded to AbstractiveCompressor.
    llm_model:
        Model identifier used for abstractive summarisation.
    abstractive_enabled:
        When False the abstractive strategy is never used (useful in tests /
        offline environments).
    """

    def __init__(
        self,
        llm_api_key: Optional[str] = None,
        llm_model: str = "claude-sonnet-4-6",
        abstractive_enabled: bool = True,
    ):
        self._extractive = ExtractiveCompressor()
        self._hierarchical = HierarchicalCompressor()
        self._progressive = ProgressiveCompressor(engine=self)
        self._abstractive: Optional[AbstractiveCompressor] = (
            AbstractiveCompressor(api_key=llm_api_key, model=llm_model)
            if abstractive_enabled
            else None
        )

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in *text*.

        Uses tiktoken (cl100k_base) when available, otherwise approximates as
        ``len(words) * 1.35`` which is accurate to within ~10 % for English.
        """
        if not text:
            return 0
        if _TIKTOKEN_AVAILABLE and _TIKTOKEN_ENC is not None:
            try:
                return len(_TIKTOKEN_ENC.encode(text))
            except Exception:
                pass
        # Fallback heuristic
        return max(1, round(len(text.split()) * 1.35))

    # ------------------------------------------------------------------
    # Single-item compression
    # ------------------------------------------------------------------

    def compress_item(self, item: ContextItem, ratio: float) -> ContextItem:
        """Return a new ContextItem whose content is compressed to *ratio*.

        Protected types (GOAL, PLAN) are returned unchanged.
        The original content is preserved in ``item.original_content``.
        """
        if item.item_type in _PROTECTED_TYPES:
            return item  # never touch these

        if ratio >= 1.0:
            return item  # nothing to do

        ratio = max(0.10, ratio)
        strategy = self._select_strategy(item, ratio)
        compressed_text = strategy.compress(item.content, ratio)

        new_item = ContextItem(
            content=compressed_text,
            item_type=item.item_type,
            importance=item.importance,
            recency=item.recency,
            metadata={**item.metadata, "compression_strategy": strategy.name()},
            item_id=item.item_id,
            compressed=True,
            original_content=item.original_content or item.content,
        )
        return new_item

    # ------------------------------------------------------------------
    # Batch compression
    # ------------------------------------------------------------------

    def compress(
        self,
        items: List[ContextItem],
        target_tokens: int,
    ) -> List[ContextItem]:
        """Compress a list of context items to fit within *target_tokens*.

        Algorithm
        ---------
        1. Never compress protected types (GOAL, PLAN).
        2. Sort compressible items by importance ascending (compress least
           important first).
        3. Iteratively compress items until the total token count fits within
           the budget.
        4. Stop early if already within budget.
        """
        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive")

        current_total = sum(self.estimate_tokens(it.content) for it in items)
        if current_total <= target_tokens:
            return items  # already fits

        # Separate protected vs. compressible items
        protected = [it for it in items if it.item_type in _PROTECTED_TYPES]
        compressible = [it for it in items if it.item_type not in _PROTECTED_TYPES]

        protected_tokens = sum(self.estimate_tokens(it.content) for it in protected)
        budget_for_compressible = target_tokens - protected_tokens

        if budget_for_compressible <= 0:
            # Cannot fit even with full compression — return protected only
            warnings.warn(
                "Protected items alone exceed target_tokens; returning protected items only.",
                stacklevel=2,
            )
            return protected

        compressible_tokens = sum(self.estimate_tokens(it.content) for it in compressible)
        if compressible_tokens <= budget_for_compressible:
            return items  # protected items are the only issue and we already warned

        # Sort: compress unimportant items first
        sorted_compressible = sorted(compressible, key=lambda it: it.importance)

        result_map: Dict[int, ContextItem] = {}
        remaining_budget = budget_for_compressible
        remaining_tokens = compressible_tokens

        for idx, item in enumerate(sorted_compressible):
            item_tokens = self.estimate_tokens(item.content)

            if remaining_tokens <= remaining_budget:
                # Budget met — copy remaining items as-is
                for j in range(idx, len(sorted_compressible)):
                    result_map[j] = sorted_compressible[j]
                break

            # Compute the ideal ratio for this item
            fraction_of_compressible = item_tokens / (remaining_tokens or 1)
            item_budget = max(10, int(fraction_of_compressible * remaining_budget))
            ratio = item_budget / (item_tokens or 1)
            ratio = max(0.10, min(ratio, 0.95))

            compressed_item = self.compress_item(item, ratio)
            result_map[idx] = compressed_item

            old_tokens = item_tokens
            new_tokens = self.estimate_tokens(compressed_item.content)
            remaining_budget -= new_tokens
            remaining_tokens -= old_tokens
        else:
            # Ensure all items are in result_map
            for j in range(len(sorted_compressible)):
                if j not in result_map:
                    result_map[j] = sorted_compressible[j]

        compressed_compressible = [result_map[j] for j in range(len(sorted_compressible))]

        # Restore original ordering
        original_order: Dict[int, ContextItem] = {}
        comp_iter = iter(compressed_compressible)
        for original_idx, item in enumerate(items):
            if item.item_type in _PROTECTED_TYPES:
                original_order[original_idx] = item
            else:
                original_order[original_idx] = next(comp_iter)

        return [original_order[i] for i in range(len(items))]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_compression(
        self,
        original: str,
        compressed: str,
    ) -> CompressionMetrics:
        """Compute quality metrics for a (original, compressed) pair.

        Semantic similarity uses a simple bag-of-words cosine if numpy is
        available; otherwise it is set to 0.0.
        """
        orig_tokens = self.estimate_tokens(original)
        comp_tokens = self.estimate_tokens(compressed)
        ratio = comp_tokens / (orig_tokens or 1)
        rouge_l = _estimate_rouge_l(original, compressed)

        semantic_sim = 0.0
        if _NUMPY_AVAILABLE and original and compressed:
            all_words = re.findall(r"\b\w+\b", (original + " " + compressed).lower())
            vocab = {w: i for i, w in enumerate(sorted(set(all_words)))}
            v_orig = _bow_vector(original, vocab)
            v_comp = _bow_vector(compressed, vocab)
            semantic_sim = round(_cosine_similarity(v_orig, v_comp), 4)

        return CompressionMetrics(
            original_tokens=orig_tokens,
            compressed_tokens=comp_tokens,
            ratio=round(ratio, 4),
            rouge_l=rouge_l,
            semantic_similarity=semantic_sim,
        )

    # ------------------------------------------------------------------
    # Internal: strategy selection
    # ------------------------------------------------------------------

    def _select_strategy(self, item: ContextItem, ratio: float) -> BaseCompressor:
        """Pick the best compression strategy for *item* and *ratio*."""

        # High importance items: use extractive only (safe, lossless order)
        if item.importance >= _HIGH_IMPORTANCE:
            return self._extractive

        # Compressible types allow heavier strategies
        if item.item_type in _COMPRESSIBLE_TYPES:
            if ratio <= 0.35:
                # Very tight budget — hierarchical will drop to keyword level
                return self._hierarchical
            if ratio <= 0.65 and self._abstractive is not None:
                return self._abstractive
            return self._progressive

        # Default: progressive (iterative tightening with quality floor)
        return self._progressive


# ===========================================================================
# Convenience factory
# ===========================================================================

def build_engine(
    api_key: Optional[str] = None,
    offline: bool = False,
) -> CompressionEngine:
    """Create a CompressionEngine with sensible defaults.

    Parameters
    ----------
    api_key:
        Anthropic API key.  Reads ``ANTHROPIC_API_KEY`` env var if omitted.
    offline:
        When True, disables abstractive (LLM) compression entirely.
    """
    return CompressionEngine(
        llm_api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
        abstractive_enabled=not offline,
    )


# ===========================================================================
# Quick smoke-test  (python compression_engine.py)
# ===========================================================================

if __name__ == "__main__":
    import textwrap

    sample = textwrap.dedent("""
        The context window of large language models is a fundamental constraint
        that limits how much information an agent can reason over at once.
        Modern agents need to handle long-horizon tasks spanning many turns.
        Context compression is essential for fitting relevant information into
        limited token budgets without losing critical facts.
        Extractive methods select the most important sentences.
        Abstractive methods generate new, shorter summaries.
        Hierarchical methods apply multiple levels of reduction.
        Progressive methods iteratively tighten compression until the budget
        is satisfied or a minimum quality threshold is reached.
        ContextOS combines all of these into a unified engine that selects
        the best strategy based on item type, importance, and token budget.
    """).strip()

    engine = build_engine(offline=True)

    items = [
        ContextItem(content="Retrieve all open tickets.", item_type=ItemType.GOAL, importance=0.95),
        ContextItem(content="Step 1: query DB. Step 2: filter. Step 3: rank.", item_type=ItemType.PLAN, importance=0.90),
        ContextItem(content=sample, item_type=ItemType.OBSERVATION, importance=0.40),
        ContextItem(content=sample + " " + sample, item_type=ItemType.EPISODIC, importance=0.30),
    ]

    print("=== Original token counts ===")
    for it in items:
        print(f"  [{it.item_type.value}] {engine.estimate_tokens(it.content)} tokens")

    compressed = engine.compress(items, target_tokens=120)

    print("\n=== After compression (target=120 tokens) ===")
    for it in compressed:
        print(f"  [{it.item_type.value}] compressed={it.compressed}  "
              f"tokens={engine.estimate_tokens(it.content)}")
        print(f"    {it.content[:120]}...")

    print("\n=== Metrics for observation item ===")
    orig_obs = items[2].content
    comp_obs = compressed[2].content
    metrics = engine.evaluate_compression(orig_obs, comp_obs)
    print(f"  ratio={metrics.ratio}  rouge_l={metrics.rouge_l}  "
          f"semantic_similarity={metrics.semantic_similarity}")
