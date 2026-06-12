"""
ContextOS Embedding Manager
Wraps sentence-transformers with LRU caching, device auto-detection,
and batch encoding helpers.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import List, Optional, Union

import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Optional heavy dependency — graceful degradation
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _ST_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore
    _ST_AVAILABLE = False
    logger.warning(
        "sentence-transformers not installed. "
        "EmbeddingManager will operate in stub mode — "
        "install via: pip install sentence-transformers"
    )

# Supported model identifiers
SUPPORTED_MODELS = {
    "bge-m3": "BAAI/bge-m3",
    "e5-large": "intfloat/e5-large-v2",
    "BAAI/bge-m3": "BAAI/bge-m3",
    "intfloat/e5-large-v2": "intfloat/e5-large-v2",
}

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_BATCH_SIZE = 64
DEFAULT_CACHE_SIZE = 2048  # number of single-text embeddings to cache


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _auto_device() -> str:
    """Return the best available compute device string."""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            logger.debug("EmbeddingManager: using CUDA device.")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.debug("EmbeddingManager: using MPS device.")
            return "mps"
    except ImportError:
        pass
    logger.debug("EmbeddingManager: using CPU device.")
    return "cpu"


def _text_hash(text: str) -> str:
    """Stable SHA-256 hex digest for a text string (cache key)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# EmbeddingManager
# ---------------------------------------------------------------------------

class EmbeddingManager:
    """
    Manages sentence-transformer models with lazy loading, LRU embedding
    cache, and device auto-detection.

    Parameters
    ----------
    model_name : str
        HuggingFace model id or shorthand key (see SUPPORTED_MODELS).
    device : str, optional
        Force a specific device ("cuda", "mps", "cpu").  Auto-detected
        when *None*.
    cache_size : int
        Maximum number of individual text embeddings held in the LRU cache.
    normalize : bool
        Whether to L2-normalise embeddings by default in encode().
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        cache_size: int = DEFAULT_CACHE_SIZE,
        normalize: bool = True,
    ) -> None:
        self.model_name: str = SUPPORTED_MODELS.get(model_name, model_name)
        self.device: str = device if device is not None else _auto_device()
        self.normalize: bool = normalize
        self._model: Optional[object] = None  # loaded lazily
        self._dim: Optional[int] = None
        # Per-instance LRU cache keyed by (model_name, text_hash)
        self._cache: dict[str, np.ndarray] = {}
        self._cache_size = cache_size
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info(
            f"EmbeddingManager created | model={self.model_name} "
            f"device={self.device} cache_size={cache_size}"
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, model_name: Optional[str] = None) -> None:
        """
        Explicitly (pre-)load the sentence-transformer model.

        If *model_name* is provided the manager switches to that model
        and clears the embedding cache.
        """
        if model_name is not None:
            resolved = SUPPORTED_MODELS.get(model_name, model_name)
            if resolved != self.model_name:
                logger.info(f"EmbeddingManager: switching model {self.model_name} -> {resolved}")
                self.model_name = resolved
                self._model = None
                self._dim = None
                self._cache.clear()

        if self._model is not None:
            return  # already loaded

        if not _ST_AVAILABLE:
            logger.error(
                "Cannot load model: sentence-transformers is not installed."
            )
            return

        logger.info(f"Loading sentence-transformer model: {self.model_name} on {self.device}")
        try:
            self._model = SentenceTransformer(self.model_name, device=self.device)
            # Infer embedding dimension from a probe
            probe = self._model.encode(["probe"], convert_to_numpy=True)
            self._dim = probe.shape[1]
            logger.info(
                f"Model loaded successfully | dim={self._dim} device={self.device}"
            )
        except Exception as exc:
            logger.error(f"Failed to load model '{self.model_name}': {exc}")
            self._model = None
            raise

    @property
    def embedding_dim(self) -> Optional[int]:
        return self._dim

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Encode one or more texts into embedding vectors.

        Parameters
        ----------
        texts : str | List[str]
            Input text(s).
        batch_size : int
            Number of texts to encode per forward pass.
        normalize : bool, optional
            Override instance-level normalize flag for this call.

        Returns
        -------
        np.ndarray
            Shape (N, D) for list input or (D,) for single string.
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        do_normalize = self.normalize if normalize is None else normalize

        # Split into cached vs. uncached
        results: List[Optional[np.ndarray]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for i, text in enumerate(texts):
            key = f"{self.model_name}:{_text_hash(text)}"
            if key in self._cache:
                results[i] = self._cache[key]
                self._cache_hits += 1
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
                self._cache_misses += 1

        # Encode uncached texts in batches
        if uncached_texts:
            if self._model is None:
                self.load_model()

            if self._model is None:
                # Stub: return zero vectors
                dim = self._dim or 1024
                stub_embs = np.zeros((len(uncached_texts), dim), dtype=np.float32)
                for idx, emb in zip(uncached_indices, stub_embs):
                    results[idx] = emb
            else:
                raw: np.ndarray = self._model.encode(  # type: ignore[union-attr]
                    uncached_texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                for rel_i, (abs_i, text) in enumerate(zip(uncached_indices, uncached_texts)):
                    emb = raw[rel_i]
                    if do_normalize:
                        emb = self._l2_normalize_single(emb)
                    key = f"{self.model_name}:{_text_hash(text)}"
                    self._evict_if_needed()
                    self._cache[key] = emb
                    results[abs_i] = emb

        # Apply normalisation to cached results if caller requests it
        final: List[np.ndarray] = []
        for emb in results:
            if emb is None:
                emb = np.zeros(self._dim or 1024, dtype=np.float32)
            if do_normalize:
                emb = self._l2_normalize_single(emb)
            final.append(emb)

        output = np.stack(final, axis=0)  # (N, D)
        return output[0] if single else output

    # ------------------------------------------------------------------
    # Similarity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two 1-D vectors.

        Returns a float in [-1, 1].  Returns 0.0 for zero vectors.
        """
        a = np.asarray(a, dtype=np.float32).ravel()
        b = np.asarray(b, dtype=np.float32).ravel()
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def batch_cosine_similarity(
        query: np.ndarray,
        corpus: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between a single query vector and every
        row of *corpus*.

        Parameters
        ----------
        query : np.ndarray
            Shape (D,).
        corpus : np.ndarray
            Shape (N, D).

        Returns
        -------
        np.ndarray
            Shape (N,) with similarity scores in [-1, 1].
        """
        query = np.asarray(query, dtype=np.float32).ravel()
        corpus = np.asarray(corpus, dtype=np.float32)
        if corpus.ndim == 1:
            corpus = corpus[np.newaxis, :]

        q_norm = float(np.linalg.norm(query))
        if q_norm == 0.0:
            return np.zeros(corpus.shape[0], dtype=np.float32)

        c_norms = np.linalg.norm(corpus, axis=1)
        c_norms = np.where(c_norms == 0.0, 1e-10, c_norms)  # avoid div/0
        dots = corpus @ query
        return (dots / (c_norms * q_norm)).astype(np.float32)

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """
        L2-normalise a batch of embedding vectors in-place.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape (N, D) or (D,).

        Returns
        -------
        np.ndarray
            L2-normalised copy with the same shape.
        """
        emb = np.asarray(embeddings, dtype=np.float32)
        single = emb.ndim == 1
        if single:
            emb = emb[np.newaxis, :]
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1e-10, norms)
        result = emb / norms
        return result[0] if single else result

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Evict all cached embeddings."""
        self._cache.clear()
        logger.debug("EmbeddingManager: embedding cache cleared.")

    def cache_stats(self) -> dict:
        """Return a dict with cache hit/miss counters."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": round(hit_rate, 4),
            "size": len(self._cache),
            "capacity": self._cache_size,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Simple FIFO eviction when cache exceeds capacity."""
        if len(self._cache) >= self._cache_size:
            # Remove the oldest key (insertion order preserved in Python 3.7+)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

    @staticmethod
    def _l2_normalize_single(v: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            return v
        return v / norm

    def __repr__(self) -> str:
        loaded = self._model is not None
        return (
            f"EmbeddingManager(model={self.model_name!r}, "
            f"device={self.device!r}, loaded={loaded}, "
            f"cache_size={self._cache_size})"
        )
