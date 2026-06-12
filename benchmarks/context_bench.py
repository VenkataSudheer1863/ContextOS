"""
ContextOS — context_bench.py
============================
Main benchmark evaluation suite for ContextOS research.

Implements ContextBenchEvaluator with full support for:
- Task success rate (exact + fuzzy match)
- Context relevance scoring (Precision@K, NDCG)
- Token efficiency statistics
- Stratified reporting by difficulty and context length
"""

from __future__ import annotations

import math
import re
import statistics
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]
    _NUMPY_AVAILABLE = False

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
    _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_AVAILABLE = False
    _TIKTOKEN_ENC = None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkExample:
    """A single benchmark example with all metadata needed for evaluation."""

    example_id: str
    query: str
    ground_truth: str
    context: str = ""
    relevant_context_ids: List[str] = field(default_factory=list)
    difficulty: str = "medium"           # "easy" | "medium" | "hard"
    context_length_tier: str = "2k"      # "512" | "2k" | "8k" | "32k"
    task_type: str = "qa"                # "qa" | "state_tracking" | "goal_completion"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenEfficiencyStats:
    """Token usage statistics for a single method across all tasks."""

    mean_tokens_per_task: float
    std_tokens_per_task: float
    min_tokens: float
    max_tokens: float
    total_tokens: int
    efficiency_ratio: float             # useful_tokens / total_tokens  (0-1)
    tasks_evaluated: int


@dataclass
class BenchmarkResults:
    """Complete results for a single evaluated method."""

    method_name: str
    task_success_rate: float
    context_relevance_score: float
    ndcg_at_10: float
    precision_at_1: float
    precision_at_5: float
    token_efficiency: float
    mean_tokens_per_task: float
    std_tokens_per_task: float
    latency_ms: float
    by_difficulty: Dict[str, float] = field(default_factory=dict)
    by_context_length: Dict[str, float] = field(default_factory=dict)
    n_examples: int = 0
    exact_match_rate: float = 0.0
    fuzzy_match_rate: float = 0.0


# ---------------------------------------------------------------------------
# Helpers — tokenisation and string utilities
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Lower-case word tokeniser with no external dependencies."""
    return _TOKEN_RE.findall(text.lower())


def _estimate_tokens(text: str) -> int:
    """Estimate token count via tiktoken or word-count heuristic."""
    if not text:
        return 0
    if _TIKTOKEN_AVAILABLE and _TIKTOKEN_ENC is not None:
        try:
            return len(_TIKTOKEN_ENC.encode(text))
        except Exception:
            pass
    return max(1, round(len(text.split()) * 1.35))


def _normalize_text(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _lcs_length(a: List[str], b: List[str]) -> int:
    """Compute length of the Longest Common Subsequence of two token lists."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Use O(min(m,n)) space
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


def _token_overlap_f1(pred: str, gold: str) -> float:
    """Compute token-level F1 between prediction and gold answer."""
    pred_tokens = _tokenize(pred)
    gold_tokens = _tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_set = set(pred_tokens)
    gold_set = set(gold_tokens)
    common = len(pred_set & gold_set)
    if common == 0:
        return 0.0
    precision = common / len(pred_set)
    recall = common / len(gold_set)
    return 2 * precision * recall / (precision + recall)


def _rouge_l_score(prediction: str, reference: str) -> float:
    """ROUGE-L F1 score between prediction and reference strings."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _dcg(relevances: List[float], k: int) -> float:
    """Discounted Cumulative Gain at position k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        dcg += rel / math.log2(i + 1)
    return dcg


# ---------------------------------------------------------------------------
# Synthetic dataset generation (used when no real dataset is available)
# ---------------------------------------------------------------------------

_DIFFICULTIES = ["easy", "medium", "hard"]
_CONTEXT_TIERS = ["512", "2k", "8k", "32k"]
_TASK_TYPES = ["qa", "state_tracking", "goal_completion"]

_SYNTHETIC_TEMPLATES = [
    {
        "query": "What is the primary goal of context compression in AI systems?",
        "ground_truth": "To fit relevant information within the token budget of an LLM without losing critical facts.",
        "context": (
            "Context compression reduces the size of information stored in an agent's context window. "
            "The primary goal is to fit relevant information within the token budget of an LLM without "
            "losing critical facts. Techniques include extractive, abstractive, and hierarchical compression. "
            "ContextOS implements all three strategies adaptively based on content type and importance."
        ),
        "task_type": "qa",
    },
    {
        "query": "Which retrieval strategy combines dense and sparse methods?",
        "ground_truth": "Hybrid retrieval using Reciprocal Rank Fusion (RRF).",
        "context": (
            "Dense retrieval uses embedding-based similarity. Sparse retrieval uses BM25 keyword matching. "
            "Hybrid retrieval using Reciprocal Rank Fusion (RRF) combines both. "
            "Cross-encoder reranking can be applied as a final stage to improve precision."
        ),
        "task_type": "qa",
    },
    {
        "query": "Track the state: user opened file A, then edited line 10, then saved.",
        "ground_truth": "File A, line 10 edited and saved.",
        "context": (
            "Session log: User opened file A at 10:01. User navigated to line 10 at 10:02. "
            "User edited line 10 at 10:03. User saved file A at 10:04. "
            "Current state: File A, line 10 edited and saved."
        ),
        "task_type": "state_tracking",
    },
    {
        "query": "What is the RRF constant used in ContextOS by default?",
        "ground_truth": "60",
        "context": (
            "Reciprocal Rank Fusion (RRF) is parameterised by a constant k. "
            "The ContextOS default value is RRF_K = 60, following the original paper. "
            "This constant controls the trade-off between top-rank emphasis and tail items."
        ),
        "task_type": "qa",
    },
    {
        "query": "What compression strategy is used for items with importance >= 0.75?",
        "ground_truth": "Extractive compression only.",
        "context": (
            "Items with importance below 0.75 may receive hierarchical or abstractive compression. "
            "High importance items (importance >= 0.75) use extractive compression only, "
            "ensuring the most significant context is preserved with minimal loss. "
            "Protected types GOAL and PLAN are never compressed."
        ),
        "task_type": "qa",
    },
    {
        "query": "Complete the goal: retrieve all unresolved issues and rank by priority.",
        "ground_truth": "Query for unresolved issues, sort by priority score descending.",
        "context": (
            "Goal: retrieve all unresolved issues and rank by priority. "
            "Step 1: Query the issue tracker for status=open. "
            "Step 2: Fetch priority scores for each issue. "
            "Step 3: Sort issues by priority score descending. "
            "Query for unresolved issues, sort by priority score descending."
        ),
        "task_type": "goal_completion",
    },
    {
        "query": "What does NDCG stand for?",
        "ground_truth": "Normalized Discounted Cumulative Gain.",
        "context": (
            "Information retrieval metrics include Precision, Recall, MRR, and NDCG. "
            "NDCG stands for Normalized Discounted Cumulative Gain. "
            "It accounts for the position of relevant documents in the ranking, "
            "giving higher weight to relevant items that appear earlier."
        ),
        "task_type": "qa",
    },
    {
        "query": "How does ProgressiveCompressor determine when to stop compressing?",
        "ground_truth": "When the achieved compression ratio meets the target or the quality floor is reached.",
        "context": (
            "ProgressiveCompressor iterates by reducing the ratio by a fixed step size at each round. "
            "It stops when the achieved compression ratio meets the target or "
            "the quality floor (minimum_quality_threshold) is reached. "
            "The best candidate seen so far is returned."
        ),
        "task_type": "qa",
    },
]


def _generate_synthetic_dataset(n: int = 200, seed: int = 42) -> List[BenchmarkExample]:
    """Generate a synthetic benchmark dataset for evaluation."""
    # Simple deterministic pseudo-randomness without numpy dependency
    def _lcg(state: int) -> Tuple[int, int]:
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state, state

    rng_state = seed
    examples: List[BenchmarkExample] = []
    tpl_count = len(_SYNTHETIC_TEMPLATES)

    for i in range(n):
        rng_state, r1 = _lcg(rng_state)
        rng_state, r2 = _lcg(rng_state)
        rng_state, r3 = _lcg(rng_state)
        rng_state, r4 = _lcg(rng_state)

        tpl = _SYNTHETIC_TEMPLATES[i % tpl_count]
        difficulty = _DIFFICULTIES[r1 % 3]
        tier = _CONTEXT_TIERS[r2 % 4]

        # Generate plausible relevant context IDs
        n_relevant = 1 + (r3 % 4)
        relevant_ids = [f"ctx_{i}_{j}" for j in range(n_relevant)]

        # Repeat context to simulate different length tiers
        multiplier = {"512": 1, "2k": 3, "8k": 10, "32k": 40}.get(tier, 1)
        padded_context = (tpl["context"] + " ") * multiplier

        examples.append(BenchmarkExample(
            example_id=f"example_{i:04d}",
            query=tpl["query"],
            ground_truth=tpl["ground_truth"],
            context=padded_context.strip(),
            relevant_context_ids=relevant_ids,
            difficulty=difficulty,
            context_length_tier=tier,
            task_type=tpl["task_type"],
            metadata={"template_idx": i % tpl_count, "seed": r4},
        ))

    return examples


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class ContextBenchEvaluator:
    """
    End-to-end benchmark evaluator for ContextOS methods.

    Parameters
    ----------
    dataset_path : str, optional
        Path to a JSON/JSONL benchmark file.  If None, a synthetic dataset
        is generated in memory.
    fuzzy_threshold : float
        ROUGE-L score above which a prediction is counted as a fuzzy match.
    n_synthetic : int
        Number of synthetic examples to generate when no dataset_path given.
    seed : int
        Random seed for synthetic data generation.
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        fuzzy_threshold: float = 0.5,
        n_synthetic: int = 200,
        seed: int = 42,
    ) -> None:
        self._dataset_path = dataset_path
        self._fuzzy_threshold = fuzzy_threshold
        self._n_synthetic = n_synthetic
        self._seed = seed
        self._examples: List[BenchmarkExample] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_benchmark(self, split: str = "test") -> List[BenchmarkExample]:
        """Load or generate benchmark examples.

        Parameters
        ----------
        split : str
            Dataset split to load ("train" | "val" | "test").
            Only "test" is fully supported in the synthetic fallback.

        Returns
        -------
        List[BenchmarkExample]
        """
        if self._dataset_path is not None:
            self._examples = self._load_from_file(self._dataset_path, split)
        else:
            n = self._n_synthetic
            if split == "train":
                n = int(n * 0.7)
                seed = self._seed
            elif split == "val":
                n = int(n * 0.15)
                seed = self._seed + 1
            else:
                n = int(n * 0.15)
                seed = self._seed + 2
            self._examples = _generate_synthetic_dataset(n=max(n, 10), seed=seed)
        return self._examples

    def evaluate_method(
        self,
        method_name: str,
        predictions: List[str],
        ground_truths: List[str],
        retrieved_ids_list: Optional[List[List[str]]] = None,
        relevant_ids_list: Optional[List[List[str]]] = None,
        token_counts: Optional[List[int]] = None,
        latency_ms: float = 0.0,
        examples: Optional[List[BenchmarkExample]] = None,
    ) -> BenchmarkResults:
        """
        Compute full benchmark results for one method.

        Parameters
        ----------
        method_name : str
            Human-readable identifier for the method being evaluated.
        predictions : List[str]
            Predicted answer strings (one per example).
        ground_truths : List[str]
            Gold-standard answer strings.
        retrieved_ids_list : List[List[str]], optional
            Per-query list of retrieved context IDs (for retrieval metrics).
        relevant_ids_list : List[List[str]], optional
            Per-query list of gold relevant context IDs.
        token_counts : List[int], optional
            Token budget consumed per task.
        latency_ms : float
            Total evaluation latency in milliseconds.
        examples : List[BenchmarkExample], optional
            Full example objects (used for stratified reporting).

        Returns
        -------
        BenchmarkResults
        """
        if len(predictions) != len(ground_truths):
            raise ValueError(
                f"predictions length ({len(predictions)}) != "
                f"ground_truths length ({len(ground_truths)})"
            )

        n = len(predictions)
        if n == 0:
            warnings.warn("evaluate_method called with empty predictions list.")
            return BenchmarkResults(
                method_name=method_name,
                task_success_rate=0.0,
                context_relevance_score=0.0,
                ndcg_at_10=0.0,
                precision_at_1=0.0,
                precision_at_5=0.0,
                token_efficiency=0.0,
                mean_tokens_per_task=0.0,
                std_tokens_per_task=0.0,
                latency_ms=latency_ms,
            )

        # Task success rate
        tsr = self.compute_task_success_rate(predictions, ground_truths)
        exact_rate = self._exact_match_rate(predictions, ground_truths)
        fuzzy_rate = self._fuzzy_match_rate(predictions, ground_truths)

        # Context relevance (Precision@K and NDCG)
        if retrieved_ids_list and relevant_ids_list:
            crs = self.compute_context_relevance_score(
                retrieved_ids_list[0] if len(retrieved_ids_list) == 1 else [],
                relevant_ids_list[0] if len(relevant_ids_list) == 1 else [],
            )
            ndcg_at_10 = self._mean_ndcg(retrieved_ids_list, relevant_ids_list, k=10)
            p_at_1 = self._mean_precision_at_k(retrieved_ids_list, relevant_ids_list, k=1)
            p_at_5 = self._mean_precision_at_k(retrieved_ids_list, relevant_ids_list, k=5)
            # Average context relevance across all queries
            crs = statistics.mean(
                self.compute_context_relevance_score(r, g)
                for r, g in zip(retrieved_ids_list, relevant_ids_list)
            ) if retrieved_ids_list else 0.0
        else:
            crs = 0.0
            ndcg_at_10 = 0.0
            p_at_1 = 0.0
            p_at_5 = 0.0

        # Token efficiency
        if token_counts:
            method_results = {
                "method": method_name,
                "token_counts": token_counts,
                "predictions": predictions,
                "ground_truths": ground_truths,
            }
            eff_stats = self.compute_token_efficiency(method_results)
            token_eff = eff_stats.efficiency_ratio
            mean_tok = eff_stats.mean_tokens_per_task
            std_tok = eff_stats.std_tokens_per_task
        else:
            # Estimate from prediction length
            estimated = [_estimate_tokens(p) for p in predictions]
            token_eff = 0.0
            mean_tok = statistics.mean(estimated) if estimated else 0.0
            std_tok = statistics.stdev(estimated) if len(estimated) > 1 else 0.0

        # Stratified reporting
        by_difficulty: Dict[str, float] = {}
        by_context_length: Dict[str, float] = {}

        if examples and len(examples) == n:
            by_difficulty = self._stratify_success(
                predictions, ground_truths, examples, "difficulty"
            )
            by_context_length = self._stratify_success(
                predictions, ground_truths, examples, "context_length_tier"
            )

        return BenchmarkResults(
            method_name=method_name,
            task_success_rate=tsr,
            context_relevance_score=crs,
            ndcg_at_10=ndcg_at_10,
            precision_at_1=p_at_1,
            precision_at_5=p_at_5,
            token_efficiency=token_eff,
            mean_tokens_per_task=mean_tok,
            std_tokens_per_task=std_tok,
            latency_ms=latency_ms,
            by_difficulty=by_difficulty,
            by_context_length=by_context_length,
            n_examples=n,
            exact_match_rate=exact_rate,
            fuzzy_match_rate=fuzzy_rate,
        )

    def compute_task_success_rate(
        self,
        predictions: List[str],
        ground_truths: List[str],
    ) -> float:
        """
        Compute combined task success rate using exact + fuzzy match.

        A prediction is considered successful if it either:
        - Exactly matches the gold answer (after normalisation), OR
        - Achieves a ROUGE-L score >= fuzzy_threshold against the gold answer.

        Parameters
        ----------
        predictions : List[str]
        ground_truths : List[str]

        Returns
        -------
        float
            Success rate in [0, 1].
        """
        if not predictions:
            return 0.0
        successes = 0
        for pred, gold in zip(predictions, ground_truths):
            norm_pred = _normalize_text(pred)
            norm_gold = _normalize_text(gold)
            if norm_pred == norm_gold:
                successes += 1
                continue
            rouge = _rouge_l_score(pred, gold)
            if rouge >= self._fuzzy_threshold:
                successes += 1
        return successes / len(predictions)

    def compute_context_relevance_score(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
    ) -> float:
        """
        Compute context relevance as average of Precision@K across K=1,3,5,10
        and NDCG@10.

        Parameters
        ----------
        retrieved_ids : List[str]
            Ordered list of retrieved context chunk IDs (best first).
        relevant_ids : List[str]
            Set of ground-truth relevant context chunk IDs.

        Returns
        -------
        float
            Composite relevance score in [0, 1].
        """
        if not retrieved_ids or not relevant_ids:
            return 0.0

        relevant_set = set(relevant_ids)

        def _p_at_k(k: int) -> float:
            hits = sum(1 for rid in retrieved_ids[:k] if rid in relevant_set)
            return hits / k if k > 0 else 0.0

        p1 = _p_at_k(1)
        p3 = _p_at_k(3)
        p5 = _p_at_k(5)
        p10 = _p_at_k(10)

        # Binary NDCG@10
        gains = [1.0 if rid in relevant_set else 0.0 for rid in retrieved_ids[:10]]
        ideal_gains = sorted(
            [1.0 if rid in relevant_set else 0.0 for rid in retrieved_ids],
            reverse=True,
        )[:10]
        ndcg = _dcg(gains, 10) / (_dcg(ideal_gains, 10) or 1.0)

        # Weighted composite
        composite = 0.2 * p1 + 0.2 * p3 + 0.2 * p5 + 0.1 * p10 + 0.3 * ndcg
        return round(min(composite, 1.0), 4)

    def compute_token_efficiency(
        self,
        method_results: Dict[str, Any],
    ) -> TokenEfficiencyStats:
        """
        Compute token efficiency statistics from a method results dict.

        Expected keys in method_results:
            token_counts : List[int]   — tokens consumed per task
            predictions  : List[str]   — predicted answers
            ground_truths: List[str]   — gold answers

        Parameters
        ----------
        method_results : Dict[str, Any]

        Returns
        -------
        TokenEfficiencyStats
        """
        token_counts: List[int] = method_results.get("token_counts", [])
        predictions: List[str] = method_results.get("predictions", [])
        ground_truths: List[str] = method_results.get("ground_truths", [])

        if not token_counts:
            # Estimate from prediction length
            token_counts = [_estimate_tokens(p) for p in predictions]

        if not token_counts:
            return TokenEfficiencyStats(
                mean_tokens_per_task=0.0,
                std_tokens_per_task=0.0,
                min_tokens=0.0,
                max_tokens=0.0,
                total_tokens=0,
                efficiency_ratio=0.0,
                tasks_evaluated=0,
            )

        mean_tok = statistics.mean(token_counts)
        std_tok = statistics.stdev(token_counts) if len(token_counts) > 1 else 0.0
        total = sum(token_counts)

        # Efficiency: tokens consumed by correct predictions / total tokens
        if predictions and ground_truths and len(predictions) == len(token_counts):
            useful_tokens = 0
            for tok, pred, gold in zip(token_counts, predictions, ground_truths):
                score = _rouge_l_score(pred, gold)
                if score >= self._fuzzy_threshold:
                    useful_tokens += tok
            efficiency = useful_tokens / total if total > 0 else 0.0
        else:
            efficiency = 0.0

        return TokenEfficiencyStats(
            mean_tokens_per_task=round(mean_tok, 2),
            std_tokens_per_task=round(std_tok, 2),
            min_tokens=float(min(token_counts)),
            max_tokens=float(max(token_counts)),
            total_tokens=total,
            efficiency_ratio=round(efficiency, 4),
            tasks_evaluated=len(token_counts),
        )

    def run_full_evaluation(
        self,
        method_results_dict: Dict[str, Dict[str, Any]],
        examples: Optional[List[BenchmarkExample]] = None,
    ) -> Dict[str, BenchmarkResults]:
        """
        Run evaluation for multiple methods in one call.

        Parameters
        ----------
        method_results_dict : Dict[str, Dict]
            Keys are method names; values are dicts with at minimum:
                predictions   : List[str]
                ground_truths : List[str]
            And optionally:
                retrieved_ids_list : List[List[str]]
                relevant_ids_list  : List[List[str]]
                token_counts       : List[int]
                latency_ms         : float

        examples : List[BenchmarkExample], optional
            Full example objects for stratified analysis.

        Returns
        -------
        Dict[str, BenchmarkResults]
        """
        results: Dict[str, BenchmarkResults] = {}
        for method_name, method_data in method_results_dict.items():
            t0 = time.perf_counter()
            predictions = method_data.get("predictions", [])
            ground_truths = method_data.get("ground_truths", [])
            retrieved = method_data.get("retrieved_ids_list")
            relevant = method_data.get("relevant_ids_list")
            token_counts = method_data.get("token_counts")
            latency_ms = method_data.get(
                "latency_ms",
                (time.perf_counter() - t0) * 1000,
            )
            results[method_name] = self.evaluate_method(
                method_name=method_name,
                predictions=predictions,
                ground_truths=ground_truths,
                retrieved_ids_list=retrieved,
                relevant_ids_list=relevant,
                token_counts=token_counts,
                latency_ms=latency_ms,
                examples=examples,
            )
        return results

    def generate_report(self, results_dict: Dict[str, BenchmarkResults]) -> str:
        """
        Generate a human-readable evaluation report.

        Parameters
        ----------
        results_dict : Dict[str, BenchmarkResults]

        Returns
        -------
        str
            Formatted multi-line report string.
        """
        sep = "=" * 72
        thin = "-" * 72
        lines: List[str] = []

        lines.append(sep)
        lines.append("  ContextOS Benchmark Evaluation Report")
        lines.append(sep)

        header = (
            f"{'Method':<24} {'TSR':>6} {'EM':>6} {'FM':>6} "
            f"{'CRS':>6} {'NDCG@10':>8} {'P@1':>5} {'P@5':>5} "
            f"{'TokEff':>7} {'MeanTok':>8} {'Lat(ms)':>8}"
        )
        lines.append(header)
        lines.append(thin)

        for method_name, res in results_dict.items():
            row = (
                f"{method_name:<24} "
                f"{res.task_success_rate:>6.3f} "
                f"{res.exact_match_rate:>6.3f} "
                f"{res.fuzzy_match_rate:>6.3f} "
                f"{res.context_relevance_score:>6.3f} "
                f"{res.ndcg_at_10:>8.4f} "
                f"{res.precision_at_1:>5.3f} "
                f"{res.precision_at_5:>5.3f} "
                f"{res.token_efficiency:>7.4f} "
                f"{res.mean_tokens_per_task:>8.1f} "
                f"{res.latency_ms:>8.1f}"
            )
            lines.append(row)

        lines.append(sep)

        # Stratified breakdown
        for method_name, res in results_dict.items():
            if res.by_difficulty or res.by_context_length:
                lines.append(f"\n  Stratified Results — {method_name}")
                lines.append(thin)
            if res.by_difficulty:
                lines.append("  By difficulty:")
                for diff, score in sorted(res.by_difficulty.items()):
                    lines.append(f"    {diff:<12}: {score:.3f}")
            if res.by_context_length:
                lines.append("  By context length tier:")
                for tier, score in sorted(res.by_context_length.items()):
                    lines.append(f"    {tier:<8}: {score:.3f}")

        if len(results_dict) >= 2:
            lines.append(f"\n{sep}")
            lines.append("  Best Method per Metric")
            lines.append(thin)
            metrics = {
                "Task Success Rate": lambda r: r.task_success_rate,
                "Context Relevance": lambda r: r.context_relevance_score,
                "NDCG@10": lambda r: r.ndcg_at_10,
                "Token Efficiency": lambda r: r.token_efficiency,
            }
            for metric_name, fn in metrics.items():
                best = max(results_dict.items(), key=lambda kv: fn(kv[1]))
                lines.append(f"  {metric_name:<22}: {best[0]} ({fn(best[1]):.4f})")

        lines.append(sep)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _exact_match_rate(
        self, predictions: List[str], ground_truths: List[str]
    ) -> float:
        hits = sum(
            1 for p, g in zip(predictions, ground_truths)
            if _normalize_text(p) == _normalize_text(g)
        )
        return hits / len(predictions) if predictions else 0.0

    def _fuzzy_match_rate(
        self, predictions: List[str], ground_truths: List[str]
    ) -> float:
        hits = sum(
            1 for p, g in zip(predictions, ground_truths)
            if _rouge_l_score(p, g) >= self._fuzzy_threshold
        )
        return hits / len(predictions) if predictions else 0.0

    def _mean_ndcg(
        self,
        retrieved_list: List[List[str]],
        relevant_list: List[List[str]],
        k: int = 10,
    ) -> float:
        scores = []
        for retrieved, relevant in zip(retrieved_list, relevant_list):
            relevant_set = set(relevant)
            gains = [1.0 if r in relevant_set else 0.0 for r in retrieved[:k]]
            ideal = sorted(
                [1.0 if r in relevant_set else 0.0 for r in retrieved],
                reverse=True,
            )[:k]
            ndcg = _dcg(gains, k) / (_dcg(ideal, k) or 1.0)
            scores.append(ndcg)
        return statistics.mean(scores) if scores else 0.0

    def _mean_precision_at_k(
        self,
        retrieved_list: List[List[str]],
        relevant_list: List[List[str]],
        k: int,
    ) -> float:
        scores = []
        for retrieved, relevant in zip(retrieved_list, relevant_list):
            relevant_set = set(relevant)
            hits = sum(1 for r in retrieved[:k] if r in relevant_set)
            scores.append(hits / k)
        return statistics.mean(scores) if scores else 0.0

    def _stratify_success(
        self,
        predictions: List[str],
        ground_truths: List[str],
        examples: List[BenchmarkExample],
        attribute: str,
    ) -> Dict[str, float]:
        groups: Dict[str, Tuple[List[str], List[str]]] = {}
        for pred, gold, ex in zip(predictions, ground_truths, examples):
            key = getattr(ex, attribute, "unknown")
            if key not in groups:
                groups[key] = ([], [])
            groups[key][0].append(pred)
            groups[key][1].append(gold)
        return {
            key: self.compute_task_success_rate(preds, golds)
            for key, (preds, golds) in groups.items()
        }

    @staticmethod
    def _load_from_file(
        path: str, split: str
    ) -> List[BenchmarkExample]:
        """Load benchmark examples from a JSON/JSONL file."""
        import json
        import os

        examples: List[BenchmarkExample] = []
        if not os.path.exists(path):
            warnings.warn(f"Dataset path not found: {path}. Using synthetic data.")
            return _generate_synthetic_dataset()

        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read().strip()

        # Try JSONL first, then JSON array
        try:
            raw_list = [json.loads(line) for line in content.splitlines() if line.strip()]
        except json.JSONDecodeError:
            raw_list = json.loads(content)

        # Filter by split if 'split' key exists
        for i, raw in enumerate(raw_list):
            if "split" in raw and raw["split"] != split:
                continue
            examples.append(BenchmarkExample(
                example_id=raw.get("id", f"ex_{i:04d}"),
                query=raw.get("query", raw.get("question", "")),
                ground_truth=raw.get("ground_truth", raw.get("answer", "")),
                context=raw.get("context", ""),
                relevant_context_ids=raw.get("relevant_ids", []),
                difficulty=raw.get("difficulty", "medium"),
                context_length_tier=raw.get("context_length_tier", "2k"),
                task_type=raw.get("task_type", "qa"),
                metadata=raw.get("metadata", {}),
            ))
        return examples


# ---------------------------------------------------------------------------
# CLI entry point for quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    evaluator = ContextBenchEvaluator(n_synthetic=40)
    examples = evaluator.load_benchmark(split="test")
    print(f"Loaded {len(examples)} examples.")

    # Simulate two methods: oracle and a simple baseline
    import random
    random.seed(0)

    oracle_preds = [ex.ground_truth for ex in examples]
    baseline_preds = [
        ex.ground_truth if random.random() > 0.5 else "I don't know."
        for ex in examples
    ]
    ground_truths = [ex.ground_truth for ex in examples]

    results = evaluator.run_full_evaluation(
        {
            "oracle": {
                "predictions": oracle_preds,
                "ground_truths": ground_truths,
                "token_counts": [_estimate_tokens(ex.context) for ex in examples],
                "latency_ms": 10.0,
            },
            "baseline": {
                "predictions": baseline_preds,
                "ground_truths": ground_truths,
                "token_counts": [_estimate_tokens(p) * 2 for p in baseline_preds],
                "latency_ms": 25.0,
            },
        },
        examples=examples,
    )

    print(evaluator.generate_report(results))
