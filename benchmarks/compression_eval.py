"""
ContextOS — compression_eval.py
================================
Evaluator for context compression methods.

All ROUGE-L and information-retention metrics are implemented from scratch
with no external dependencies beyond the standard library and (optional) numpy.

Public API
----------
CompressionEvaluator
    .compute_rouge_l(hypothesis, reference) -> float
    .compute_compression_ratio(original_tokens, compressed_tokens) -> float
    .compute_information_retention(original, compressed) -> float
    .evaluate_compressor(compressor, test_data) -> CompressionResults
    .compare_compressors(compressor_dict, test_data) -> Dict[str, CompressionResults]
"""

from __future__ import annotations

import math
import re
import statistics
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
    _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_AVAILABLE = False
    _TIKTOKEN_ENC = None

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]
    _NUMPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Token / text utilities
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Lower-case word tokeniser."""
    return _WORD_RE.findall(text.lower())


def _estimate_tokens(text: str) -> int:
    """Estimate BPE token count; falls back to word heuristic."""
    if not text:
        return 0
    if _TIKTOKEN_AVAILABLE and _TIKTOKEN_ENC is not None:
        try:
            return len(_TIKTOKEN_ENC.encode(text))
        except Exception:
            pass
    return max(1, round(len(text.split()) * 1.35))


def _lcs_length(a: List[str], b: List[str]) -> int:
    """
    Length of the Longest Common Subsequence between token lists a and b.
    O(min(len(a), len(b))) space.
    """
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Always iterate over the shorter list in the inner loop
    if m < n:
        a, b = b, a
        m, n = n, m
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def _bag_of_words(text: str) -> Dict[str, int]:
    """Return term-frequency dict for the text."""
    freq: Dict[str, int] = {}
    for tok in _tokenize(text):
        freq[tok] = freq.get(tok, 0) + 1
    return freq


def _cosine_bow(text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts using bag-of-words vectors."""
    bow_a = _bag_of_words(text_a)
    bow_b = _bag_of_words(text_b)
    if not bow_a or not bow_b:
        return 0.0
    all_terms = set(bow_a) | set(bow_b)
    dot = sum(bow_a.get(t, 0) * bow_b.get(t, 0) for t in all_terms)
    norm_a = math.sqrt(sum(v * v for v in bow_a.values()))
    norm_b = math.sqrt(sum(v * v for v in bow_b.values()))
    denom = norm_a * norm_b
    return dot / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CompressionSample:
    """A single (original_text, compressed_text) evaluation pair."""

    sample_id: str
    original: str
    compressed: Optional[str] = None       # filled in by evaluate_compressor
    ground_truth_summary: Optional[str] = None
    domain: str = "general"                # "code" | "dialogue" | "documentation" | "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionResults:
    """Aggregated evaluation results for one compressor on a dataset."""

    compressor_name: str
    n_samples: int

    # Quality
    mean_rouge_l: float
    std_rouge_l: float
    mean_information_retention: float
    std_information_retention: float
    mean_semantic_similarity: float

    # Efficiency
    mean_compression_ratio: float          # compressed_tokens / original_tokens
    std_compression_ratio: float
    mean_original_tokens: float
    mean_compressed_tokens: float
    total_tokens_saved: int

    # Performance
    mean_latency_ms: float
    total_latency_ms: float

    # Breakdowns
    by_domain: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_sample_rouge_l: List[float] = field(default_factory=list)
    per_sample_ratios: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Compressor protocol — any callable (str, float) -> str qualifies
# ---------------------------------------------------------------------------

class CompressorProtocol(Protocol):
    """Protocol that compression strategies must satisfy."""

    def compress(self, text: str, target_ratio: float) -> str:
        ...


CompressorCallable = Callable[[str, float], str]


def _call_compressor(
    compressor: Any,
    text: str,
    target_ratio: float,
) -> str:
    """
    Call a compressor regardless of whether it follows the Protocol
    (has a .compress method) or is a bare callable.
    """
    if hasattr(compressor, "compress"):
        return compressor.compress(text, target_ratio)
    return compressor(text, target_ratio)


# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------

_SAMPLE_TEXTS = [
    (
        "The context window of large language models represents a fundamental constraint. "
        "Modern agents need to process long documents spanning hundreds of pages. "
        "Without effective compression, agents cannot retain critical information across "
        "multi-step reasoning chains. Extractive methods select important sentences. "
        "Abstractive methods generate concise summaries using a language model. "
        "Hierarchical methods apply progressive reduction at multiple levels of granularity. "
        "ContextOS integrates all three strategies into a unified compression engine.",
        "ContextOS integrates extractive, abstractive, and hierarchical compression strategies.",
        "documentation",
    ),
    (
        "User: How do I reset my password? "
        "Agent: Please click the 'Forgot Password' link on the login page. "
        "User: I don't see that link. "
        "Agent: It is located below the login button, in small grey text. "
        "User: Found it, thanks. "
        "Agent: You're welcome. Please check your email for the reset link.",
        "User asked how to reset password; agent directed them to 'Forgot Password' link below login button.",
        "dialogue",
    ),
    (
        "def compute_ndcg(retrieved, relevant, k=10):\n"
        "    relevant_set = set(relevant)\n"
        "    gains = [1.0 if r in relevant_set else 0.0 for r in retrieved[:k]]\n"
        "    ideal = sorted([1.0 if r in relevant_set else 0.0 for r in retrieved], reverse=True)[:k]\n"
        "    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))\n"
        "    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))\n"
        "    return dcg / idcg if idcg > 0 else 0.0",
        "Function computes NDCG@k using binary relevance.",
        "code",
    ),
    (
        "Reciprocal Rank Fusion (RRF) is a score fusion technique for combining multiple "
        "ranked lists into a single consolidated ranking. It was proposed by Cormack et al. "
        "in 2009 and has since become a standard baseline in information retrieval. "
        "The formula assigns a score of 1/(k + rank) to each document in each list, "
        "where k is a smoothing constant (commonly set to 60). "
        "Documents appearing in multiple lists receive cumulative scores. "
        "RRF is parameter-free beyond the k constant and is robust to score distributions.",
        "RRF combines ranked lists using score 1/(k+rank), default k=60.",
        "documentation",
    ),
    (
        "Precision at K measures the fraction of the top-K retrieved documents that are "
        "relevant. Recall at K measures the fraction of all relevant documents that appear "
        "in the top-K results. The F1 score at K balances precision and recall. "
        "NDCG discounts relevance by log position, rewarding systems that place relevant "
        "items higher. MRR is the mean reciprocal rank of the first relevant item. "
        "These metrics together characterise both efficiency and completeness of retrieval.",
        "Retrieval metrics include Precision@K, Recall@K, F1, NDCG, and MRR.",
        "general",
    ),
]


def _build_synthetic_test_data(n: int = 30, seed: int = 42) -> List[CompressionSample]:
    """Generate a synthetic compression test dataset."""
    import random
    rng = random.Random(seed)
    samples: List[CompressionSample] = []
    pool_size = len(_SAMPLE_TEXTS)
    for i in range(n):
        original, summary, domain = _SAMPLE_TEXTS[i % pool_size]
        # Vary length by repetition
        multiplier = 1 + (i % 3)
        extended = (original + " ") * multiplier
        samples.append(CompressionSample(
            sample_id=f"sample_{i:04d}",
            original=extended.strip(),
            ground_truth_summary=summary,
            domain=domain,
            metadata={"template_idx": i % pool_size, "multiplier": multiplier},
        ))
    return samples


# ---------------------------------------------------------------------------
# CompressionEvaluator
# ---------------------------------------------------------------------------

class CompressionEvaluator:
    """
    Evaluates and compares context compression methods.

    Parameters
    ----------
    target_ratio : float
        Default compression ratio applied when evaluating a compressor.
        0.5 means compress to 50% of original length.
    n_synthetic : int
        Number of synthetic samples to generate when no test_data is provided.
    seed : int
        Random seed for synthetic data.
    """

    def __init__(
        self,
        target_ratio: float = 0.5,
        n_synthetic: int = 30,
        seed: int = 42,
    ) -> None:
        if not 0.0 < target_ratio <= 1.0:
            raise ValueError("target_ratio must be in (0, 1]")
        self._target_ratio = target_ratio
        self._n_synthetic = n_synthetic
        self._seed = seed

    # ------------------------------------------------------------------
    # Core metric methods (all public)
    # ------------------------------------------------------------------

    def compute_rouge_l(self, hypothesis: str, reference: str) -> float:
        """
        Compute ROUGE-L F1 score between hypothesis and reference.

        ROUGE-L measures the longest common subsequence (LCS) of token sequences.
        No external dependencies are required — the LCS is computed with a
        standard DP algorithm.

        Parameters
        ----------
        hypothesis : str
            The compressed / predicted text.
        reference : str
            The gold-standard reference text.

        Returns
        -------
        float
            ROUGE-L F1 score in [0, 1].
        """
        h_tokens = _tokenize(hypothesis)
        r_tokens = _tokenize(reference)
        if not h_tokens or not r_tokens:
            return 0.0
        lcs_len = _lcs_length(h_tokens, r_tokens)
        precision = lcs_len / len(h_tokens)
        recall = lcs_len / len(r_tokens)
        denom = precision + recall
        f1 = (2 * precision * recall / denom) if denom > 0 else 0.0
        return round(f1, 6)

    def compute_compression_ratio(
        self,
        original_tokens: int,
        compressed_tokens: int,
    ) -> float:
        """
        Compute the compression ratio as compressed_tokens / original_tokens.

        A value of 1.0 means no compression. A value of 0.3 means the
        compressed output uses 30% of the original token budget.

        Parameters
        ----------
        original_tokens : int
        compressed_tokens : int

        Returns
        -------
        float
            Ratio in [0, ∞). Clamped to [0, 1] for typical use.
        """
        if original_tokens <= 0:
            return 0.0
        ratio = compressed_tokens / original_tokens
        return round(ratio, 6)

    def compute_information_retention(
        self,
        original: str,
        compressed: str,
    ) -> float:
        """
        Compute information retention of a compressed text relative to the original.

        Combines three signals:
        1. Token overlap F1      — exact lexical coverage
        2. ROUGE-L               — sequence-level coverage
        3. Cosine (bag-of-words) — semantic similarity proxy

        The three signals are averaged with weights 0.3 / 0.4 / 0.3.

        Parameters
        ----------
        original : str
            The full original text.
        compressed : str
            The compressed version.

        Returns
        -------
        float
            Information retention score in [0, 1].
        """
        if not original or not compressed:
            return 0.0

        # Signal 1: token overlap F1
        orig_set = set(_tokenize(original))
        comp_set = set(_tokenize(compressed))
        if not orig_set or not comp_set:
            overlap_f1 = 0.0
        else:
            common = len(orig_set & comp_set)
            prec = common / len(comp_set)
            rec = common / len(orig_set)
            denom = prec + rec
            overlap_f1 = (2 * prec * rec / denom) if denom > 0 else 0.0

        # Signal 2: ROUGE-L
        rouge = self.compute_rouge_l(compressed, original)

        # Signal 3: bag-of-words cosine similarity
        cosine = _cosine_bow(original, compressed)

        retention = 0.3 * overlap_f1 + 0.4 * rouge + 0.3 * cosine
        return round(min(retention, 1.0), 6)

    # ------------------------------------------------------------------
    # Compressor evaluation
    # ------------------------------------------------------------------

    def evaluate_compressor(
        self,
        compressor: Any,
        test_data: Optional[List[CompressionSample]] = None,
        target_ratio: Optional[float] = None,
        compressor_name: Optional[str] = None,
    ) -> CompressionResults:
        """
        Run a compressor over all test samples and collect quality metrics.

        Parameters
        ----------
        compressor : compressor object or callable
            Must expose .compress(text, ratio) or be callable as (text, ratio) -> str.
        test_data : List[CompressionSample], optional
            If None, synthetic test data is generated automatically.
        target_ratio : float, optional
            Compression ratio to target per sample. Defaults to self._target_ratio.
        compressor_name : str, optional
            Display name; auto-detected from compressor if omitted.

        Returns
        -------
        CompressionResults
        """
        if test_data is None:
            test_data = _build_synthetic_test_data(self._n_synthetic, self._seed)

        ratio = target_ratio if target_ratio is not None else self._target_ratio
        name = compressor_name or getattr(
            compressor, "name", lambda: type(compressor).__name__
        )()

        rouge_scores: List[float] = []
        retention_scores: List[float] = []
        cosine_scores: List[float] = []
        compression_ratios: List[float] = []
        original_token_counts: List[int] = []
        compressed_token_counts: List[int] = []
        latencies: List[float] = []
        domain_buckets: Dict[str, Dict[str, List[float]]] = {}

        for sample in test_data:
            t0 = time.perf_counter()
            try:
                compressed = _call_compressor(compressor, sample.original, ratio)
            except Exception as exc:
                warnings.warn(
                    f"Compressor {name} raised exception on sample {sample.sample_id}: {exc}"
                )
                compressed = sample.original  # no compression
            elapsed_ms = (time.perf_counter() - t0) * 1000

            sample.compressed = compressed

            orig_tok = _estimate_tokens(sample.original)
            comp_tok = _estimate_tokens(compressed)

            # Use ground truth summary as reference if available, else original
            reference = sample.ground_truth_summary or sample.original
            rouge = self.compute_rouge_l(compressed, reference)
            retention = self.compute_information_retention(sample.original, compressed)
            cosine = _cosine_bow(sample.original, compressed)
            cr = self.compute_compression_ratio(orig_tok, comp_tok)

            rouge_scores.append(rouge)
            retention_scores.append(retention)
            cosine_scores.append(cosine)
            compression_ratios.append(cr)
            original_token_counts.append(orig_tok)
            compressed_token_counts.append(comp_tok)
            latencies.append(elapsed_ms)

            # Domain breakdown
            domain = sample.domain
            if domain not in domain_buckets:
                domain_buckets[domain] = {"rouge_l": [], "retention": [], "ratio": []}
            domain_buckets[domain]["rouge_l"].append(rouge)
            domain_buckets[domain]["retention"].append(retention)
            domain_buckets[domain]["ratio"].append(cr)

        # Aggregate domain stats
        by_domain: Dict[str, Dict[str, float]] = {}
        for domain, metrics in domain_buckets.items():
            by_domain[domain] = {
                "mean_rouge_l": statistics.mean(metrics["rouge_l"]),
                "mean_retention": statistics.mean(metrics["retention"]),
                "mean_ratio": statistics.mean(metrics["ratio"]),
                "n_samples": len(metrics["rouge_l"]),
            }

        n = len(test_data)
        total_saved = sum(o - c for o, c in zip(original_token_counts, compressed_token_counts))
        total_latency = sum(latencies)

        return CompressionResults(
            compressor_name=name,
            n_samples=n,
            mean_rouge_l=statistics.mean(rouge_scores) if rouge_scores else 0.0,
            std_rouge_l=statistics.stdev(rouge_scores) if len(rouge_scores) > 1 else 0.0,
            mean_information_retention=statistics.mean(retention_scores) if retention_scores else 0.0,
            std_information_retention=statistics.stdev(retention_scores) if len(retention_scores) > 1 else 0.0,
            mean_semantic_similarity=statistics.mean(cosine_scores) if cosine_scores else 0.0,
            mean_compression_ratio=statistics.mean(compression_ratios) if compression_ratios else 0.0,
            std_compression_ratio=statistics.stdev(compression_ratios) if len(compression_ratios) > 1 else 0.0,
            mean_original_tokens=statistics.mean(original_token_counts) if original_token_counts else 0.0,
            mean_compressed_tokens=statistics.mean(compressed_token_counts) if compressed_token_counts else 0.0,
            total_tokens_saved=max(0, total_saved),
            mean_latency_ms=statistics.mean(latencies) if latencies else 0.0,
            total_latency_ms=total_latency,
            by_domain=by_domain,
            per_sample_rouge_l=rouge_scores,
            per_sample_ratios=compression_ratios,
        )

    def compare_compressors(
        self,
        compressor_dict: Dict[str, Any],
        test_data: Optional[List[CompressionSample]] = None,
        target_ratio: Optional[float] = None,
    ) -> Dict[str, CompressionResults]:
        """
        Evaluate and compare multiple compressors on the same test dataset.

        Parameters
        ----------
        compressor_dict : Dict[str, compressor]
            Mapping of display-name -> compressor object or callable.
        test_data : List[CompressionSample], optional
        target_ratio : float, optional

        Returns
        -------
        Dict[str, CompressionResults]
            Keyed by compressor name.
        """
        if test_data is None:
            test_data = _build_synthetic_test_data(self._n_synthetic, self._seed)

        results: Dict[str, CompressionResults] = {}
        for name, compressor in compressor_dict.items():
            results[name] = self.evaluate_compressor(
                compressor,
                test_data=test_data,
                target_ratio=target_ratio,
                compressor_name=name,
            )
        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_comparison_report(
        self,
        results: Dict[str, CompressionResults],
    ) -> str:
        """
        Generate a formatted comparison report for multiple compressors.

        Parameters
        ----------
        results : Dict[str, CompressionResults]

        Returns
        -------
        str
        """
        sep = "=" * 80
        thin = "-" * 80
        lines = [
            sep,
            "  ContextOS Compression Evaluator — Comparison Report",
            sep,
            f"  {'Compressor':<24} {'ROUGE-L':>8} {'Retention':>10} "
            f"{'SemanticSim':>12} {'CompRatio':>10} {'TokSaved':>9} {'Lat(ms)':>8}",
            thin,
        ]
        for name, res in results.items():
            lines.append(
                f"  {name:<24} "
                f"{res.mean_rouge_l:>8.4f} "
                f"{res.mean_information_retention:>10.4f} "
                f"{res.mean_semantic_similarity:>12.4f} "
                f"{res.mean_compression_ratio:>10.4f} "
                f"{res.total_tokens_saved:>9d} "
                f"{res.mean_latency_ms:>8.2f}"
            )
        lines.append(sep)

        # Domain breakdown for each compressor
        all_domains: List[str] = sorted(
            {d for res in results.values() for d in res.by_domain}
        )
        if all_domains:
            lines.append("\n  ROUGE-L by Domain")
            lines.append(thin)
            dom_header = f"  {'Compressor':<24} " + " ".join(
                f"{d:>14}" for d in all_domains
            )
            lines.append(dom_header)
            for name, res in results.items():
                dom_vals = " ".join(
                    f"{res.by_domain.get(d, {}).get('mean_rouge_l', 0.0):>14.4f}"
                    for d in all_domains
                )
                lines.append(f"  {name:<24} {dom_vals}")
            lines.append(sep)

        if len(results) >= 2:
            best_rouge = max(results.items(), key=lambda kv: kv[1].mean_rouge_l)
            best_ret = max(results.items(), key=lambda kv: kv[1].mean_information_retention)
            best_eff = min(results.items(), key=lambda kv: kv[1].mean_compression_ratio)
            lines.append("\n  Best per Metric")
            lines.append(thin)
            lines.append(f"  Best ROUGE-L       : {best_rouge[0]} ({best_rouge[1].mean_rouge_l:.4f})")
            lines.append(f"  Best Retention     : {best_ret[0]} ({best_ret[1].mean_information_retention:.4f})")
            lines.append(f"  Best Efficiency    : {best_eff[0]} (ratio={best_eff[1].mean_compression_ratio:.4f})")
            lines.append(sep)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal inline compressor stubs for testing
    class TruncateCompressor:
        """Baseline: truncate to target_ratio of words."""

        def name(self) -> str:
            return "truncate"

        def compress(self, text: str, target_ratio: float) -> str:
            words = text.split()
            keep = max(1, int(len(words) * target_ratio))
            return " ".join(words[:keep])

    class IdentityCompressor:
        """No-op: return text unchanged."""

        def name(self) -> str:
            return "identity"

        def compress(self, text: str, target_ratio: float) -> str:
            return text

    evaluator = CompressionEvaluator(target_ratio=0.5, n_synthetic=20)
    test_data = _build_synthetic_test_data(n=20)

    results = evaluator.compare_compressors(
        {
            "truncate": TruncateCompressor(),
            "identity": IdentityCompressor(),
        },
        test_data=test_data,
    )

    print(evaluator.generate_comparison_report(results))

    # Test individual metrics
    ev = CompressionEvaluator()
    hyp = "ContextOS integrates extractive and abstractive compression."
    ref = "ContextOS integrates extractive, abstractive, and hierarchical compression strategies."
    print(f"\nROUGE-L: {ev.compute_rouge_l(hyp, ref):.4f}")
    print(f"Retention: {ev.compute_information_retention(ref, hyp):.4f}")
    print(f"Compression ratio (200 -> 80 tokens): {ev.compute_compression_ratio(200, 80):.4f}")
