"""
ContextOS — retrieval_eval.py
==============================
Evaluator for context retrieval methods.

Implements the standard suite of information-retrieval metrics from scratch
(no scikit-learn or other IR libraries required):

    Precision@K     — fraction of top-K results that are relevant
    Recall@K        — fraction of relevant docs found in top-K
    NDCG@K          — position-discounted relevance gain
    MRR             — Mean Reciprocal Rank of first relevant item
    MAP             — Mean Average Precision
    F1@K            — harmonic mean of Precision@K and Recall@K

Public API
----------
RetrievalEvaluator
    .compute_precision_at_k(retrieved, relevant, k) -> float
    .compute_recall_at_k(retrieved, relevant, k) -> float
    .compute_ndcg(retrieved, relevant, k=10) -> float
    .compute_mrr(retrieved, relevant) -> float
    .compute_map(retrieved, relevant) -> float
    .compute_f1_at_k(retrieved, relevant, k) -> float
    .evaluate_retriever(retriever, test_queries) -> RetrievalResults
    .compare_retrievers(retriever_dict, test_queries) -> Dict[str, RetrievalResults]
    .generate_report(results_dict) -> str
"""

from __future__ import annotations

import math
import re
import statistics
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RetrievalQuery:
    """A single retrieval test query with ground-truth relevant IDs."""

    query_id: str
    query_text: str
    relevant_ids: List[str]              # unordered set of relevant doc IDs
    graded_relevance: Optional[Dict[str, int]] = None  # doc_id -> relevance grade (0..3)
    domain: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResults:
    """Aggregated retrieval evaluation results for one retriever."""

    retriever_name: str
    n_queries: int

    # Core metrics (macro-averaged across queries)
    mean_precision_at_1: float
    mean_precision_at_5: float
    mean_precision_at_10: float
    mean_recall_at_10: float
    mean_ndcg_at_10: float
    mean_ndcg_at_5: float
    mean_mrr: float
    mean_map: float
    mean_f1_at_5: float

    # Spread
    std_ndcg_at_10: float
    std_mrr: float

    # Latency
    mean_latency_ms: float
    total_latency_ms: float

    # Breakdowns
    by_domain: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_query_ndcg: List[float] = field(default_factory=list)
    per_query_mrr: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Retriever protocol
# ---------------------------------------------------------------------------

class RetrieverProtocol(Protocol):
    """Minimal protocol that retrieval systems must satisfy."""

    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """Return a list of document IDs ordered by relevance (best first)."""
        ...


RetrieverCallable = Callable[[str, int], List[str]]


def _call_retriever(
    retriever: Any,
    query: str,
    top_k: int,
) -> List[str]:
    """Call a retriever regardless of whether it uses .retrieve or __call__."""
    if hasattr(retriever, "retrieve"):
        return retriever.retrieve(query, top_k=top_k)
    return retriever(query, top_k)


# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------

_CORPUS: Dict[str, str] = {
    "doc_001": "Reciprocal Rank Fusion combines multiple ranked lists into one.",
    "doc_002": "BM25 is a sparse retrieval function based on term frequency.",
    "doc_003": "Dense retrieval uses embedding vectors for semantic similarity search.",
    "doc_004": "NDCG measures ranking quality with position discounting.",
    "doc_005": "Cross-encoder models re-rank candidates by scoring pairs jointly.",
    "doc_006": "Extractive compression selects the most important sentences.",
    "doc_007": "Abstractive compression generates a summary using a language model.",
    "doc_008": "Hierarchical compression applies multiple levels of reduction.",
    "doc_009": "FAISS provides fast approximate nearest-neighbour search.",
    "doc_010": "Precision at K measures the fraction of top-K results that are relevant.",
    "doc_011": "Recall at K measures how many relevant items appear in top-K results.",
    "doc_012": "Mean Reciprocal Rank is the average of 1/rank of the first relevant item.",
    "doc_013": "ContextOS integrates retrieval and compression in a unified engine.",
    "doc_014": "Token budget management is critical for long-horizon AI agents.",
    "doc_015": "Progressive compression iteratively tightens reduction until quality floor.",
    "doc_016": "Sentence embeddings enable semantic search beyond keyword matching.",
    "doc_017": "Hybrid retrieval fuses dense and sparse signals via RRF.",
    "doc_018": "MAP is the mean of per-query average precision across a test set.",
    "doc_019": "Context prioritization ranks items by importance, recency, and relevance.",
    "doc_020": "Goal and Plan items are never compressed in ContextOS.",
}

_QUERIES_DATA: List[Tuple[str, List[str], str]] = [
    ("What is Reciprocal Rank Fusion?", ["doc_001", "doc_017"], "general"),
    ("How does BM25 work?", ["doc_002"], "general"),
    ("Explain dense retrieval with embeddings.", ["doc_003", "doc_009", "doc_016"], "general"),
    ("What is NDCG?", ["doc_004", "doc_010"], "general"),
    ("What is cross-encoder reranking?", ["doc_005"], "general"),
    ("Describe extractive compression.", ["doc_006"], "compression"),
    ("What is abstractive summarization?", ["doc_007"], "compression"),
    ("How does hierarchical compression work?", ["doc_008", "doc_015"], "compression"),
    ("What is FAISS used for?", ["doc_009", "doc_003"], "retrieval"),
    ("Define Precision at K.", ["doc_010", "doc_011"], "metrics"),
    ("What is Recall at K?", ["doc_011", "doc_010"], "metrics"),
    ("Explain Mean Reciprocal Rank.", ["doc_012"], "metrics"),
    ("What does ContextOS do?", ["doc_013", "doc_014", "doc_019"], "general"),
    ("Why is token budget important?", ["doc_014", "doc_013"], "general"),
    ("What is progressive compression?", ["doc_015", "doc_008"], "compression"),
    ("How do sentence embeddings help retrieval?", ["doc_016", "doc_003"], "retrieval"),
    ("What is hybrid retrieval?", ["doc_017", "doc_001", "doc_002"], "retrieval"),
    ("What is MAP in information retrieval?", ["doc_018", "doc_004"], "metrics"),
    ("How are items prioritized in ContextOS?", ["doc_019", "doc_013"], "general"),
    ("Which item types are never compressed?", ["doc_020", "doc_006"], "compression"),
]


def _build_synthetic_queries(n: int = 20, seed: int = 42) -> List[RetrievalQuery]:
    """Build synthetic retrieval test queries from the hard-coded pool."""
    import random
    rng = random.Random(seed)
    pool = _QUERIES_DATA[:n] if n <= len(_QUERIES_DATA) else _QUERIES_DATA
    queries: List[RetrievalQuery] = []
    for i, (qtext, rel_ids, domain) in enumerate(pool):
        # Add graded relevance: primary = grade 2, secondary = grade 1
        graded: Dict[str, int] = {}
        for j, rid in enumerate(rel_ids):
            graded[rid] = 2 if j == 0 else 1
        queries.append(RetrievalQuery(
            query_id=f"q_{i:03d}",
            query_text=qtext,
            relevant_ids=rel_ids,
            graded_relevance=graded,
            domain=domain,
        ))
    return queries


# ---------------------------------------------------------------------------
# Core metric implementations
# ---------------------------------------------------------------------------

def _dcg(gains: List[float], k: int) -> float:
    """Discounted Cumulative Gain at position k (log base 2)."""
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def _ideal_gains(
    retrieved: List[str],
    relevant: List[str],
    graded: Optional[Dict[str, int]] = None,
    k: int = 10,
) -> List[float]:
    """Return sorted (descending) gains representing the ideal ranking."""
    relevant_set = set(relevant)
    if graded:
        all_gains = [graded.get(doc_id, 0) for doc_id in retrieved]
    else:
        all_gains = [1.0 if doc_id in relevant_set else 0.0 for doc_id in retrieved]
    return sorted(all_gains, reverse=True)[:k]


def _query_gains(
    retrieved: List[str],
    relevant: List[str],
    graded: Optional[Dict[str, int]] = None,
    k: int = 10,
) -> List[float]:
    """Return gains at each position in retrieved (up to k)."""
    relevant_set = set(relevant)
    if graded:
        return [graded.get(doc_id, 0) for doc_id in retrieved[:k]]
    return [1.0 if doc_id in relevant_set else 0.0 for doc_id in retrieved[:k]]


# ---------------------------------------------------------------------------
# RetrievalEvaluator
# ---------------------------------------------------------------------------

class RetrievalEvaluator:
    """
    Evaluates retrieval quality using standard IR metrics.

    Parameters
    ----------
    default_k : int
        Default top-K used when evaluating a retriever.
    n_synthetic : int
        Number of synthetic queries to generate when no test_queries provided.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        default_k: int = 10,
        n_synthetic: int = 20,
        seed: int = 42,
    ) -> None:
        self._default_k = default_k
        self._n_synthetic = n_synthetic
        self._seed = seed

    # ------------------------------------------------------------------
    # Individual metric methods
    # ------------------------------------------------------------------

    def compute_precision_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int,
    ) -> float:
        """
        Precision at K: fraction of the top-K retrieved documents that are relevant.

        P@K = |relevant ∩ retrieved[:K]| / K

        Parameters
        ----------
        retrieved : List[str]
            Ordered list of retrieved document IDs (best first).
        relevant : List[str]
            Ground-truth relevant document IDs (unordered).
        k : int
            Cut-off depth.

        Returns
        -------
        float in [0, 1]
        """
        if k <= 0:
            return 0.0
        relevant_set = set(relevant)
        hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant_set)
        return hits / k

    def compute_recall_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int,
    ) -> float:
        """
        Recall at K: fraction of all relevant documents that appear in top-K results.

        R@K = |relevant ∩ retrieved[:K]| / |relevant|

        Parameters
        ----------
        retrieved : List[str]
        relevant : List[str]
        k : int

        Returns
        -------
        float in [0, 1]
        """
        if not relevant or k <= 0:
            return 0.0
        relevant_set = set(relevant)
        hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant_set)
        return hits / len(relevant_set)

    def compute_ndcg(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int = 10,
        graded_relevance: Optional[Dict[str, int]] = None,
    ) -> float:
        """
        Normalized Discounted Cumulative Gain at K.

        NDCG@K = DCG@K / IDCG@K

        Supports both binary (0/1) and graded relevance.

        Parameters
        ----------
        retrieved : List[str]
            Ordered list of retrieved document IDs.
        relevant : List[str]
            Ground-truth relevant document IDs.
        k : int
            Cut-off depth.
        graded_relevance : Dict[str, int], optional
            Map from doc_id to integer relevance grade (0..3).
            If provided, graded DCG is used instead of binary.

        Returns
        -------
        float in [0, 1]
        """
        if not retrieved or not relevant:
            return 0.0
        gains = _query_gains(retrieved, relevant, graded_relevance, k)
        ideal = _ideal_gains(retrieved, relevant, graded_relevance, k)
        dcg = _dcg(gains, k)
        idcg = _dcg(ideal, k)
        return (dcg / idcg) if idcg > 0 else 0.0

    def compute_mrr(
        self,
        retrieved: List[str],
        relevant: List[str],
    ) -> float:
        """
        Mean Reciprocal Rank (single-query version).

        MRR = 1 / rank_of_first_relevant_document

        If no relevant document is found, returns 0.

        Parameters
        ----------
        retrieved : List[str]
        relevant : List[str]

        Returns
        -------
        float in [0, 1]
        """
        relevant_set = set(relevant)
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant_set:
                return 1.0 / rank
        return 0.0

    def compute_map(
        self,
        retrieved: List[str],
        relevant: List[str],
    ) -> float:
        """
        Average Precision (AP) for a single query.

        AP = sum_k P@k * rel(k) / |relevant|

        where rel(k) = 1 if the k-th result is relevant, else 0.

        Parameters
        ----------
        retrieved : List[str]
        relevant : List[str]

        Returns
        -------
        float in [0, 1]
        """
        if not relevant:
            return 0.0
        relevant_set = set(relevant)
        cumulative_precision = 0.0
        hits = 0
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant_set:
                hits += 1
                cumulative_precision += hits / rank
        return cumulative_precision / len(relevant_set)

    def compute_f1_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int,
    ) -> float:
        """
        F1 score at K: harmonic mean of Precision@K and Recall@K.

        Parameters
        ----------
        retrieved : List[str]
        relevant : List[str]
        k : int

        Returns
        -------
        float in [0, 1]
        """
        p = self.compute_precision_at_k(retrieved, relevant, k)
        r = self.compute_recall_at_k(retrieved, relevant, k)
        denom = p + r
        return (2 * p * r / denom) if denom > 0 else 0.0

    # ------------------------------------------------------------------
    # Full retriever evaluation
    # ------------------------------------------------------------------

    def evaluate_retriever(
        self,
        retriever: Any,
        test_queries: Optional[List[RetrievalQuery]] = None,
        top_k: Optional[int] = None,
        retriever_name: Optional[str] = None,
    ) -> RetrievalResults:
        """
        Evaluate a retriever over all test queries and aggregate metrics.

        Parameters
        ----------
        retriever : retriever object or callable
            Must expose .retrieve(query_text, top_k) -> List[str] or
            be callable as (query_text, top_k) -> List[str].
        test_queries : List[RetrievalQuery], optional
            If None, synthetic queries are generated.
        top_k : int, optional
            Depth at which to retrieve; defaults to self._default_k.
        retriever_name : str, optional

        Returns
        -------
        RetrievalResults
        """
        if test_queries is None:
            test_queries = _build_synthetic_queries(self._n_synthetic, self._seed)

        k = top_k if top_k is not None else self._default_k
        name = retriever_name or getattr(
            retriever, "name", lambda: type(retriever).__name__
        )()

        p1_scores: List[float] = []
        p5_scores: List[float] = []
        p10_scores: List[float] = []
        r10_scores: List[float] = []
        ndcg10_scores: List[float] = []
        ndcg5_scores: List[float] = []
        mrr_scores: List[float] = []
        map_scores: List[float] = []
        f1_5_scores: List[float] = []
        latencies: List[float] = []
        domain_buckets: Dict[str, Dict[str, List[float]]] = {}

        for query in test_queries:
            t0 = time.perf_counter()
            try:
                retrieved = _call_retriever(retriever, query.query_text, k)
            except Exception as exc:
                warnings.warn(
                    f"Retriever {name} raised exception on query {query.query_id}: {exc}"
                )
                retrieved = []
            elapsed_ms = (time.perf_counter() - t0) * 1000

            rel = query.relevant_ids
            graded = query.graded_relevance

            p1 = self.compute_precision_at_k(retrieved, rel, 1)
            p5 = self.compute_precision_at_k(retrieved, rel, 5)
            p10 = self.compute_precision_at_k(retrieved, rel, 10)
            r10 = self.compute_recall_at_k(retrieved, rel, 10)
            ndcg10 = self.compute_ndcg(retrieved, rel, 10, graded)
            ndcg5 = self.compute_ndcg(retrieved, rel, 5, graded)
            mrr = self.compute_mrr(retrieved, rel)
            ap = self.compute_map(retrieved, rel)
            f1_5 = self.compute_f1_at_k(retrieved, rel, 5)

            p1_scores.append(p1)
            p5_scores.append(p5)
            p10_scores.append(p10)
            r10_scores.append(r10)
            ndcg10_scores.append(ndcg10)
            ndcg5_scores.append(ndcg5)
            mrr_scores.append(mrr)
            map_scores.append(ap)
            f1_5_scores.append(f1_5)
            latencies.append(elapsed_ms)

            # Domain breakdown
            domain = query.domain
            if domain not in domain_buckets:
                domain_buckets[domain] = {
                    "ndcg_10": [], "mrr": [], "map": [], "p_5": []
                }
            domain_buckets[domain]["ndcg_10"].append(ndcg10)
            domain_buckets[domain]["mrr"].append(mrr)
            domain_buckets[domain]["map"].append(ap)
            domain_buckets[domain]["p_5"].append(p5)

        by_domain: Dict[str, Dict[str, float]] = {}
        for domain, metrics in domain_buckets.items():
            by_domain[domain] = {
                "mean_ndcg_10": statistics.mean(metrics["ndcg_10"]),
                "mean_mrr": statistics.mean(metrics["mrr"]),
                "mean_map": statistics.mean(metrics["map"]),
                "mean_p_5": statistics.mean(metrics["p_5"]),
                "n_queries": len(metrics["ndcg_10"]),
            }

        def _safe_mean(lst: List[float]) -> float:
            return statistics.mean(lst) if lst else 0.0

        def _safe_std(lst: List[float]) -> float:
            return statistics.stdev(lst) if len(lst) > 1 else 0.0

        return RetrievalResults(
            retriever_name=name,
            n_queries=len(test_queries),
            mean_precision_at_1=_safe_mean(p1_scores),
            mean_precision_at_5=_safe_mean(p5_scores),
            mean_precision_at_10=_safe_mean(p10_scores),
            mean_recall_at_10=_safe_mean(r10_scores),
            mean_ndcg_at_10=_safe_mean(ndcg10_scores),
            mean_ndcg_at_5=_safe_mean(ndcg5_scores),
            mean_mrr=_safe_mean(mrr_scores),
            mean_map=_safe_mean(map_scores),
            mean_f1_at_5=_safe_mean(f1_5_scores),
            std_ndcg_at_10=_safe_std(ndcg10_scores),
            std_mrr=_safe_std(mrr_scores),
            mean_latency_ms=_safe_mean(latencies),
            total_latency_ms=sum(latencies),
            by_domain=by_domain,
            per_query_ndcg=ndcg10_scores,
            per_query_mrr=mrr_scores,
        )

    def compare_retrievers(
        self,
        retriever_dict: Dict[str, Any],
        test_queries: Optional[List[RetrievalQuery]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, RetrievalResults]:
        """
        Evaluate and compare multiple retrievers on the same test query set.

        Parameters
        ----------
        retriever_dict : Dict[str, retriever]
        test_queries : List[RetrievalQuery], optional
        top_k : int, optional

        Returns
        -------
        Dict[str, RetrievalResults]
        """
        if test_queries is None:
            test_queries = _build_synthetic_queries(self._n_synthetic, self._seed)
        results: Dict[str, RetrievalResults] = {}
        for name, retriever in retriever_dict.items():
            results[name] = self.evaluate_retriever(
                retriever,
                test_queries=test_queries,
                top_k=top_k,
                retriever_name=name,
            )
        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_report(
        self,
        results_dict: Dict[str, RetrievalResults],
    ) -> str:
        """
        Generate a formatted comparison report.

        Parameters
        ----------
        results_dict : Dict[str, RetrievalResults]

        Returns
        -------
        str
        """
        sep = "=" * 88
        thin = "-" * 88
        lines = [
            sep,
            "  ContextOS Retrieval Evaluator — Comparison Report",
            sep,
            f"  {'Retriever':<22} {'P@1':>5} {'P@5':>5} {'P@10':>6} "
            f"{'R@10':>6} {'NDCG@5':>7} {'NDCG@10':>8} {'MRR':>6} "
            f"{'MAP':>6} {'F1@5':>6} {'Lat(ms)':>8}",
            thin,
        ]
        for name, res in results_dict.items():
            lines.append(
                f"  {name:<22} "
                f"{res.mean_precision_at_1:>5.3f} "
                f"{res.mean_precision_at_5:>5.3f} "
                f"{res.mean_precision_at_10:>6.3f} "
                f"{res.mean_recall_at_10:>6.3f} "
                f"{res.mean_ndcg_at_5:>7.4f} "
                f"{res.mean_ndcg_at_10:>8.4f} "
                f"{res.mean_mrr:>6.4f} "
                f"{res.mean_map:>6.4f} "
                f"{res.mean_f1_at_5:>6.4f} "
                f"{res.mean_latency_ms:>8.2f}"
            )
        lines.append(sep)

        # Domain breakdown
        all_domains = sorted({d for res in results_dict.values() for d in res.by_domain})
        if all_domains:
            lines.append("\n  NDCG@10 by Domain")
            lines.append(thin)
            dom_hdr = f"  {'Retriever':<22} " + "  ".join(f"{d:>14}" for d in all_domains)
            lines.append(dom_hdr)
            for name, res in results_dict.items():
                vals = "  ".join(
                    f"{res.by_domain.get(d, {}).get('mean_ndcg_10', 0.0):>14.4f}"
                    for d in all_domains
                )
                lines.append(f"  {name:<22} {vals}")
            lines.append(sep)

        if len(results_dict) >= 2:
            best_ndcg = max(results_dict.items(), key=lambda kv: kv[1].mean_ndcg_at_10)
            best_mrr = max(results_dict.items(), key=lambda kv: kv[1].mean_mrr)
            best_map = max(results_dict.items(), key=lambda kv: kv[1].mean_map)
            lines.append("\n  Best per Metric")
            lines.append(thin)
            lines.append(f"  Best NDCG@10 : {best_ndcg[0]} ({best_ndcg[1].mean_ndcg_at_10:.4f})")
            lines.append(f"  Best MRR     : {best_mrr[0]} ({best_mrr[1].mean_mrr:.4f})")
            lines.append(f"  Best MAP     : {best_map[0]} ({best_map[1].mean_map:.4f})")
            lines.append(sep)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build synthetic test queries
    queries = _build_synthetic_queries(n=20)

    # Oracle retriever: returns known relevant IDs
    class OracleRetriever:
        def name(self) -> str:
            return "oracle"

        def retrieve(self, query_text: str, top_k: int = 10) -> List[str]:
            for q in queries:
                if q.query_text == query_text:
                    return q.relevant_ids + [f"extra_{i}" for i in range(top_k - len(q.relevant_ids))]
            return [f"doc_{i:03d}" for i in range(1, top_k + 1)]

    # Random retriever: returns shuffled doc IDs
    import random as _random
    _rng = _random.Random(99)

    class RandomRetriever:
        def name(self) -> str:
            return "random"

        def retrieve(self, query_text: str, top_k: int = 10) -> List[str]:
            pool = list(_CORPUS.keys())
            _rng.shuffle(pool)
            return pool[:top_k]

    # BM25-style keyword retriever
    class KeywordRetriever:
        def name(self) -> str:
            return "keyword"

        def _score(self, query: str, doc_text: str) -> float:
            q_tokens = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
            d_tokens = re.findall(r"[a-zA-Z0-9]+", doc_text.lower())
            if not d_tokens:
                return 0.0
            hits = sum(1 for t in d_tokens if t in q_tokens)
            return hits / len(d_tokens)

        def retrieve(self, query_text: str, top_k: int = 10) -> List[str]:
            scored = [
                (self._score(query_text, text), doc_id)
                for doc_id, text in _CORPUS.items()
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            return [doc_id for _, doc_id in scored[:top_k]]

    evaluator = RetrievalEvaluator(default_k=10, n_synthetic=20)

    results = evaluator.compare_retrievers(
        {
            "oracle": OracleRetriever(),
            "keyword": KeywordRetriever(),
            "random": RandomRetriever(),
        },
        test_queries=queries,
    )

    print(evaluator.generate_report(results))

    # Single metric examples
    ev = RetrievalEvaluator()
    retrieved = ["doc_001", "doc_003", "doc_002", "doc_009", "doc_010"]
    relevant = ["doc_001", "doc_002", "doc_009"]
    print(f"\nPrecision@3 : {ev.compute_precision_at_k(retrieved, relevant, 3):.4f}")
    print(f"Recall@3    : {ev.compute_recall_at_k(retrieved, relevant, 3):.4f}")
    print(f"NDCG@5      : {ev.compute_ndcg(retrieved, relevant, 5):.4f}")
    print(f"MRR         : {ev.compute_mrr(retrieved, relevant):.4f}")
    print(f"MAP         : {ev.compute_map(retrieved, relevant):.4f}")
    print(f"F1@3        : {ev.compute_f1_at_k(retrieved, relevant, 3):.4f}")
