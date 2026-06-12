"""
ContextOS Retrieval Engine
Implements hybrid retrieval combining dense (embedding), sparse (BM25),
and cross-encoder reranking with Reciprocal Rank Fusion.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Optional dependencies — graceful fallbacks
# ---------------------------------------------------------------------------
try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False
    logger.debug("faiss not available; dense retrieval falls back to numpy cosine search.")

try:
    from rank_bm25 import BM25Okapi  # type: ignore
    _BM25_AVAILABLE = True
except ImportError:
    BM25Okapi = None  # type: ignore
    _BM25_AVAILABLE = False
    logger.debug("rank-bm25 not available; sparse retrieval falls back to TF-IDF-style scoring.")

try:
    from sentence_transformers import CrossEncoder  # type: ignore
    _CE_AVAILABLE = True
except ImportError:
    CrossEncoder = None  # type: ignore
    _CE_AVAILABLE = False
    logger.debug("sentence-transformers CrossEncoder not available; rerank uses score-based fallback.")

# Local
from ..utils.embeddings import EmbeddingManager
from .orchestrator import ContextItem


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60  # standard RRF constant


# ---------------------------------------------------------------------------
# Tokenisation helper (used by BM25 fallback and sparse retrieval)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Lower-case word tokeniser — no external dependency required."""
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# RetrievalEngine
# ---------------------------------------------------------------------------

class RetrievalEngine:
    """
    Hybrid retrieval engine for ContextOS.

    Combines dense vector search, sparse BM25 keyword search, and
    cross-encoder reranking, fused via Reciprocal Rank Fusion (RRF).

    Parameters
    ----------
    config : dict, optional
        ContextOS configuration dict.  Relevant keys::

            retrieval:
              embedding_model: "BAAI/bge-m3"   # or "intfloat/e5-large-v2"
              cross_encoder:   "cross-encoder/ms-marco-MiniLM-L-6-v2"
              cache_size:      2048
              batch_size:      64
              rrf_k:           60
              alpha:           0.5              # default dense/sparse blend

    embedding_model : str, optional
        Override the embedding model from config.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        cfg = (config or {}).get("retrieval", {})
        model_name = embedding_model or cfg.get("embedding_model", DEFAULT_MODEL)
        cache_size = cfg.get("cache_size", 2048)
        self._batch_size: int = cfg.get("batch_size", 64)
        self._rrf_k: int = cfg.get("rrf_k", RRF_K)
        self._alpha: float = cfg.get("alpha", 0.5)
        self._cross_encoder_name: str = cfg.get("cross_encoder", DEFAULT_CROSS_ENCODER)

        self._embedding_manager = EmbeddingManager(
            model_name=model_name,
            cache_size=cache_size,
            normalize=True,
        )
        self._cross_encoder: Optional[Any] = None  # lazy
        logger.info(
            f"RetrievalEngine initialised | model={model_name} "
            f"faiss={_FAISS_AVAILABLE} bm25={_BM25_AVAILABLE} ce={_CE_AVAILABLE}"
        )

    # ------------------------------------------------------------------
    # Public: single-text encoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> np.ndarray:
        """
        Encode *text* into a normalised embedding vector.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        np.ndarray
            Shape (D,), L2-normalised.
        """
        return self._embedding_manager.encode(text, batch_size=1)

    # ------------------------------------------------------------------
    # Dense retrieval
    # ------------------------------------------------------------------

    def dense_retrieve(
        self,
        query: str,
        corpus_items: List[ContextItem],
        top_k: int = 10,
    ) -> List[ContextItem]:
        """
        Retrieve top-k items by dense vector similarity (FAISS or cosine).

        Parameters
        ----------
        query : str
        corpus_items : List[ContextItem]
        top_k : int

        Returns
        -------
        List[ContextItem]
            Scored and sorted by relevance (highest first), length <= top_k.
        """
        if not corpus_items:
            return []
        top_k = min(top_k, len(corpus_items))

        query_emb = self.encode(query)
        corpus_embs = self._get_corpus_embeddings(corpus_items)

        if _FAISS_AVAILABLE:
            scores = self._faiss_search(query_emb, corpus_embs, top_k)
        else:
            scores = self._embedding_manager.batch_cosine_similarity(query_emb, corpus_embs)

        ranked = self._rank_by_scores(corpus_items, scores, top_k)
        logger.debug(f"dense_retrieve: top_k={top_k}, corpus={len(corpus_items)}")
        return ranked

    # ------------------------------------------------------------------
    # Sparse retrieval
    # ------------------------------------------------------------------

    def sparse_retrieve(
        self,
        query: str,
        corpus_items: List[ContextItem],
        top_k: int = 10,
    ) -> List[ContextItem]:
        """
        Retrieve top-k items using BM25 keyword matching.

        Falls back to simple TF-IDF-style term-overlap scoring if
        rank-bm25 is not installed.

        Parameters
        ----------
        query : str
        corpus_items : List[ContextItem]
        top_k : int

        Returns
        -------
        List[ContextItem]
            Scored and sorted by BM25 score (highest first), length <= top_k.
        """
        if not corpus_items:
            return []
        top_k = min(top_k, len(corpus_items))

        query_tokens = _tokenize(query)
        corpus_texts = [item.content for item in corpus_items]

        if _BM25_AVAILABLE:
            tokenized_corpus = [_tokenize(t) for t in corpus_texts]
            bm25 = BM25Okapi(tokenized_corpus)
            scores = np.array(bm25.get_scores(query_tokens), dtype=np.float32)
        else:
            scores = self._tfidf_scores(query_tokens, corpus_texts)

        ranked = self._rank_by_scores(corpus_items, scores, top_k)
        logger.debug(f"sparse_retrieve: top_k={top_k}, corpus={len(corpus_items)}")
        return ranked

    # ------------------------------------------------------------------
    # Hybrid retrieval (RRF fusion)
    # ------------------------------------------------------------------

    def hybrid_retrieve(
        self,
        query: str,
        corpus_items: List[ContextItem],
        top_k: int = 10,
        alpha: Optional[float] = None,
    ) -> List[ContextItem]:
        """
        Hybrid retrieval via Reciprocal Rank Fusion of dense + sparse lists.

        Parameters
        ----------
        query : str
        corpus_items : List[ContextItem]
        top_k : int
        alpha : float, optional
            Weight for dense ranking in RRF (0 = pure sparse, 1 = pure dense).
            Defaults to the engine-level alpha (0.5).

        Returns
        -------
        List[ContextItem]
            Fused and re-sorted results, length <= top_k.
        """
        if not corpus_items:
            return []
        alpha = self._alpha if alpha is None else alpha
        retrieve_n = min(len(corpus_items), max(top_k * 2, 20))

        dense_list = self.dense_retrieve(query, corpus_items, top_k=retrieve_n)
        sparse_list = self.sparse_retrieve(query, corpus_items, top_k=retrieve_n)

        fused = self._reciprocal_rank_fusion(
            [dense_list, sparse_list],
            weights=[alpha, 1.0 - alpha],
        )
        result = fused[:top_k]
        logger.debug(
            f"hybrid_retrieve: top_k={top_k}, alpha={alpha:.2f}, "
            f"dense={len(dense_list)}, sparse={len(sparse_list)}, fused={len(fused)}"
        )
        return result

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: List[ContextItem],
        top_k: int = 5,
    ) -> List[ContextItem]:
        """
        Rerank *candidates* against *query* using a cross-encoder (if
        available) or score-based interpolation as fallback.

        Parameters
        ----------
        query : str
        candidates : List[ContextItem]
        top_k : int

        Returns
        -------
        List[ContextItem]
            Top-k items after reranking, highest score first.
        """
        if not candidates:
            return []
        top_k = min(top_k, len(candidates))

        if _CE_AVAILABLE:
            try:
                return self._cross_encoder_rerank(query, candidates, top_k)
            except Exception as exc:
                logger.warning(f"Cross-encoder rerank failed ({exc}); falling back to score-based.")

        return self._score_based_rerank(query, candidates, top_k)

    # ------------------------------------------------------------------
    # RRF
    # ------------------------------------------------------------------

    def _reciprocal_rank_fusion(
        self,
        ranked_lists: List[List[ContextItem]],
        weights: Optional[List[float]] = None,
        k: Optional[int] = None,
    ) -> List[ContextItem]:
        """
        Fuse multiple ranked lists via Reciprocal Rank Fusion.

        RRF score for item d: sum_i weight_i / (k + rank_i(d))

        Parameters
        ----------
        ranked_lists : List[List[ContextItem]]
            Each inner list is already sorted highest-first.
        weights : List[float], optional
            Per-list weights (uniform if None).
        k : int, optional
            RRF constant (default: self._rrf_k = 60).

        Returns
        -------
        List[ContextItem]
            Items sorted by descending fused RRF score.
        """
        k = k or self._rrf_k
        n = len(ranked_lists)
        if weights is None:
            weights = [1.0 / n] * n
        else:
            total = sum(weights) or 1.0
            weights = [w / total for w in weights]

        rrf_scores: Dict[str, float] = defaultdict(float)
        item_map: Dict[str, ContextItem] = {}

        for ranked_list, weight in zip(ranked_lists, weights):
            for rank, item in enumerate(ranked_list, start=1):
                rrf_scores[item.id] += weight / (k + rank)
                item_map[item.id] = item

        sorted_ids = sorted(rrf_scores.keys(), key=lambda iid: rrf_scores[iid], reverse=True)
        result = []
        for iid in sorted_ids:
            item = item_map[iid]
            item.relevance = rrf_scores[iid]
            result.append(item)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_corpus_embeddings(self, items: List[ContextItem]) -> np.ndarray:
        """
        Return a (N, D) embedding matrix for *items*, using stored
        embeddings where available and encoding the rest in batch.
        """
        texts_to_encode: List[Tuple[int, str]] = []
        emb_list: List[Optional[np.ndarray]] = [None] * len(items)

        for i, item in enumerate(items):
            if item.embedding is not None:
                emb_list[i] = np.asarray(item.embedding, dtype=np.float32)
            else:
                texts_to_encode.append((i, item.content))

        if texts_to_encode:
            indices, texts = zip(*texts_to_encode)
            batch_embs = self._embedding_manager.encode(
                list(texts), batch_size=self._batch_size
            )
            if batch_embs.ndim == 1:
                batch_embs = batch_embs[np.newaxis, :]
            for rel_i, abs_i in enumerate(indices):
                emb = batch_embs[rel_i]
                emb_list[abs_i] = emb
                # Cache back onto the item so future calls are free
                items[abs_i].embedding = emb.tolist()

        # Guaranteed non-None at this point
        return np.stack(
            [e if e is not None else np.zeros(1024, dtype=np.float32) for e in emb_list],
            axis=0,
        )

    def _faiss_search(
        self,
        query_emb: np.ndarray,
        corpus_embs: np.ndarray,
        top_k: int,
    ) -> np.ndarray:
        """Use FAISS flat inner-product index for fast nearest-neighbour search."""
        dim = corpus_embs.shape[1]
        index = faiss.IndexFlatIP(dim)  # type: ignore[union-attr]
        index.add(corpus_embs.astype(np.float32))
        distances, indices = index.search(query_emb[np.newaxis, :].astype(np.float32), top_k)
        # Build a full scores array aligned with corpus_embs rows
        scores = np.zeros(corpus_embs.shape[0], dtype=np.float32)
        for rank, idx in enumerate(indices[0]):
            if idx >= 0:
                scores[idx] = float(distances[0][rank])
        return scores

    @staticmethod
    def _tfidf_scores(query_tokens: List[str], corpus_texts: List[str]) -> np.ndarray:
        """
        Simple TF-IDF-style term-overlap scorer used when rank-bm25 is
        unavailable.  Not a rigorous TF-IDF; just a reasonable proxy.
        """
        query_set = set(query_tokens)
        n_docs = len(corpus_texts)
        # IDF: log(N / (df + 1))
        df: Dict[str, int] = defaultdict(int)
        tokenized = [_tokenize(t) for t in corpus_texts]
        for tokens in tokenized:
            for term in set(tokens):
                if term in query_set:
                    df[term] += 1

        scores = np.zeros(n_docs, dtype=np.float32)
        for i, tokens in enumerate(tokenized):
            tf: Dict[str, int] = defaultdict(int)
            for tok in tokens:
                if tok in query_set:
                    tf[tok] += 1
            score = 0.0
            for term, freq in tf.items():
                idf = math.log((n_docs + 1) / (df[term] + 1)) + 1.0
                score += (freq / (freq + 0.5)) * idf
            scores[i] = score
        return scores

    @staticmethod
    def _rank_by_scores(
        items: List[ContextItem],
        scores: np.ndarray,
        top_k: int,
    ) -> List[ContextItem]:
        """Return top-k items sorted by descending score, updating relevance."""
        scored = sorted(
            zip(scores.tolist(), items), key=lambda x: x[0], reverse=True
        )
        result = []
        for score, item in scored[:top_k]:
            item.relevance = float(score)
            result.append(item)
        return result

    def _load_cross_encoder(self) -> Any:
        """Lazy-load the cross-encoder model."""
        if self._cross_encoder is None:
            if not _CE_AVAILABLE:
                raise RuntimeError("sentence-transformers is not installed.")
            logger.info(f"Loading cross-encoder: {self._cross_encoder_name}")
            self._cross_encoder = CrossEncoder(self._cross_encoder_name)
        return self._cross_encoder

    def _cross_encoder_rerank(
        self,
        query: str,
        candidates: List[ContextItem],
        top_k: int,
    ) -> List[ContextItem]:
        """Rerank using a cross-encoder model."""
        model = self._load_cross_encoder()
        pairs = [[query, item.content] for item in candidates]
        scores: np.ndarray = np.array(model.predict(pairs), dtype=np.float32)
        return self._rank_by_scores(candidates, scores, top_k)

    def _score_based_rerank(
        self,
        query: str,
        candidates: List[ContextItem],
        top_k: int,
    ) -> List[ContextItem]:
        """
        Fallback reranker: recompute dense similarity and blend with the
        existing relevance score (importance + relevance).
        """
        query_emb = self.encode(query)
        corpus_embs = self._get_corpus_embeddings(candidates)
        dense_scores = self._embedding_manager.batch_cosine_similarity(query_emb, corpus_embs)

        # Normalise existing relevance scores to [0, 1]
        rel_scores = np.array([c.relevance for c in candidates], dtype=np.float32)
        r_max = rel_scores.max() if rel_scores.max() > 0 else 1.0
        rel_scores = rel_scores / r_max

        # Blend: 70% fresh dense similarity, 30% prior relevance
        blended = 0.7 * dense_scores + 0.3 * rel_scores
        return self._rank_by_scores(candidates, blended, top_k)

    # ------------------------------------------------------------------
    # Batch encode convenience
    # ------------------------------------------------------------------

    def batch_encode(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Encode a list of texts and return a (N, D) embedding matrix.

        Parameters
        ----------
        texts : List[str]
        batch_size : int, optional
            Overrides the engine-level batch size.

        Returns
        -------
        np.ndarray
            Shape (N, D).
        """
        bs = batch_size or self._batch_size
        return self._embedding_manager.encode(texts, batch_size=bs)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RetrievalEngine(model={self._embedding_manager.model_name!r}, "
            f"alpha={self._alpha}, rrf_k={self._rrf_k}, "
            f"faiss={_FAISS_AVAILABLE}, bm25={_BM25_AVAILABLE}, ce={_CE_AVAILABLE})"
        )
