"""
ContextOS — long_horizon_bench.py
==================================
Long-horizon benchmark for evaluating ContextOS across multiple context
length tiers and task types that require sustained multi-step reasoning.

Context length tiers tested:
    512 tokens  — short session
    2 048 tokens — single conversation
    8 192 tokens — multi-session
    32 768 tokens — long project context

Task categories:
    multi_session_qa   — Q&A spanning multiple sessions
    state_tracking     — Track mutable state over many steps
    goal_completion    — Achieve a compound goal across many turns
"""

from __future__ import annotations

import math
import random
import statistics
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
    _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_AVAILABLE = False
    _TIKTOKEN_ENC = None


# ---------------------------------------------------------------------------
# Token estimation helper
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    if _TIKTOKEN_AVAILABLE and _TIKTOKEN_ENC is not None:
        try:
            return len(_TIKTOKEN_ENC.encode(text))
        except Exception:
            pass
    return max(1, round(len(text.split()) * 1.35))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTEXT_LENGTH_TIERS: List[int] = [512, 2048, 8192, 32768]

TIER_NAMES: Dict[int, str] = {
    512: "512",
    2048: "2k",
    8192: "8k",
    32768: "32k",
}

TASK_TYPES: List[str] = ["multi_session_qa", "state_tracking", "goal_completion"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LongHorizonTask:
    """A single long-horizon evaluation task."""

    task_id: str
    task_type: str                        # "multi_session_qa" | "state_tracking" | "goal_completion"
    context: str                          # Full context at the target length tier
    query: str                            # Question or final step query
    ground_truth: str                     # Expected answer / final state
    intermediate_states: List[str] = field(default_factory=list)
    context_length_target: int = 2048     # Target token count
    actual_context_tokens: int = 0        # Actual token count after generation
    difficulty: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LongHorizonResult:
    """Results for a single (method, context_length) evaluation run."""

    method_name: str
    context_length: int
    tier_name: str

    # Core metrics
    task_success_rate: float
    state_tracking_accuracy: float
    goal_completion_rate: float
    multi_session_qa_score: float

    # Efficiency
    mean_tokens_used: float
    context_utilization: float           # tokens_used / context_length_target

    # Quality
    mean_rouge_l: float
    consistency_score: float             # answer consistency across equivalent queries

    # Latency
    mean_latency_ms: float
    n_tasks: int

    by_task_type: Dict[str, float] = field(default_factory=dict)
    per_task_scores: List[float] = field(default_factory=list)


@dataclass
class ScalingAnalysis:
    """Scaling behaviour of methods across context length tiers."""

    methods: List[str]
    tiers: List[int]

    # results[method_name][context_length] = LongHorizonResult
    results: Dict[str, Dict[int, LongHorizonResult]] = field(default_factory=dict)

    # Derived scaling statistics
    scaling_slopes: Dict[str, float] = field(default_factory=dict)     # TSR drop per log2 doubling
    degradation_rates: Dict[str, float] = field(default_factory=dict)  # relative perf loss 512→32k
    best_at_each_tier: Dict[int, str] = field(default_factory=dict)    # best method per tier
    summary_table: str = ""


# ---------------------------------------------------------------------------
# Synthetic task generators
# ---------------------------------------------------------------------------

# A small pool of factual Q&A pairs used to fill contexts
_QA_POOL = [
    ("What is RRF?", "Reciprocal Rank Fusion, a score-fusion technique."),
    ("What does NDCG measure?", "Normalized Discounted Cumulative Gain for ranked lists."),
    ("What is the default BM25 k1 parameter?", "1.5"),
    ("What is extractive compression?", "Selecting important sentences from the original text."),
    ("What is the ContextOS token budget strategy?", "Progressive compression with quality floor."),
    ("What is the HierarchicalCompressor Level 3?", "Keyword extraction only."),
    ("Name two protected ContextOS item types.", "GOAL and PLAN."),
    ("What model does AbstractiveCompressor use by default?", "claude-sonnet-4-6"),
    ("What is the default RRF constant in ContextOS?", "60"),
    ("What metric combines precision and recall?", "F1 score."),
    ("What is sentence embedding?", "A dense vector representation of a sentence."),
    ("What does BM25 stand for?", "Best Match 25, a probabilistic retrieval function."),
    ("What is a cross-encoder?", "A model that scores query-document pairs jointly."),
    ("What is FAISS?", "Facebook AI Similarity Search, a library for dense vector search."),
    ("What compression ratio triggers keyword-only extraction?", "Below 0.40"),
]

_STATE_TRANSITIONS = [
    ("open file {name}", "File {name} opened."),
    ("edit line {n} to '{val}'", "Line {n} = {val}."),
    ("save file {name}", "File {name} saved."),
    ("close file {name}", "File {name} closed."),
    ("create variable {name} = {val}", "var {name} = {val}."),
    ("increment {name}", "{name} += 1."),
    ("append '{val}' to list {name}", "{name}.append('{val}')."),
    ("delete item {name}", "{name} deleted."),
]

_GOAL_STEPS = [
    "Query the database for open tasks.",
    "Filter tasks by priority > 3.",
    "Sort tasks by deadline ascending.",
    "Assign each task to the least-loaded agent.",
    "Send notification to each assigned agent.",
    "Log assignment results to the audit trail.",
    "Return the summary of assigned tasks.",
]


def _fill_to_tokens(base: str, target_tokens: int, filler_unit: str) -> str:
    """
    Repeat filler_unit after base until the text reaches approximately
    target_tokens. Returns the padded text.
    """
    current = _estimate_tokens(base)
    if current >= target_tokens:
        return base
    unit_tokens = max(1, _estimate_tokens(filler_unit))
    repetitions = math.ceil((target_tokens - current) / unit_tokens)
    padding = (" " + filler_unit.strip()) * repetitions
    return base + padding


def _generate_multi_session_qa_task(
    task_id: str,
    target_tokens: int,
    rng: random.Random,
    difficulty: str = "medium",
) -> LongHorizonTask:
    """Generate a multi-session Q&A task filled to target_tokens."""
    n_sessions = max(2, target_tokens // 512)
    qa_pairs = rng.choices(_QA_POOL, k=n_sessions)

    session_lines: List[str] = []
    for i, (q, a) in enumerate(qa_pairs, start=1):
        session_lines.append(f"[Session {i}] User: {q} Assistant: {a}")

    # The query is the question from the last session
    query_q, query_a = qa_pairs[-1]
    context_base = " ".join(session_lines[:-1])  # all but last session
    context = _fill_to_tokens(context_base, target_tokens, session_lines[-1])

    return LongHorizonTask(
        task_id=task_id,
        task_type="multi_session_qa",
        context=context,
        query=query_q,
        ground_truth=query_a,
        intermediate_states=[f"Session {i}" for i in range(1, n_sessions)],
        context_length_target=target_tokens,
        actual_context_tokens=_estimate_tokens(context),
        difficulty=difficulty,
    )


def _generate_state_tracking_task(
    task_id: str,
    target_tokens: int,
    rng: random.Random,
    difficulty: str = "medium",
) -> LongHorizonTask:
    """Generate a state-tracking task filled to target_tokens."""
    n_steps = max(3, target_tokens // 256)
    transitions = rng.choices(_STATE_TRANSITIONS, k=n_steps)

    def _fill(template: str, rng: random.Random) -> str:
        return template.format(
            name=rng.choice(["fileA", "fileB", "config", "log"]),
            n=rng.randint(1, 100),
            val=rng.choice(["true", "42", "hello", "None"]),
        )

    steps: List[Tuple[str, str]] = []
    for action_tpl, result_tpl in transitions:
        action = _fill(action_tpl, rng)
        result = _fill(result_tpl, rng)
        steps.append((action, result))

    log_lines = [f"Step {i+1}: {a} -> {r}" for i, (a, r) in enumerate(steps)]
    final_state = steps[-1][1]

    context_base = " ".join(log_lines[:-1])
    context = _fill_to_tokens(context_base, target_tokens, log_lines[-1])

    return LongHorizonTask(
        task_id=task_id,
        task_type="state_tracking",
        context=context,
        query=f"What is the result of: {steps[-1][0]}",
        ground_truth=final_state,
        intermediate_states=[r for _, r in steps[:-1]],
        context_length_target=target_tokens,
        actual_context_tokens=_estimate_tokens(context),
        difficulty=difficulty,
    )


def _generate_goal_completion_task(
    task_id: str,
    target_tokens: int,
    rng: random.Random,
    difficulty: str = "medium",
) -> LongHorizonTask:
    """Generate a goal-completion task filled to target_tokens."""
    n_steps = min(len(_GOAL_STEPS), max(3, target_tokens // 512))
    steps = _GOAL_STEPS[:n_steps]

    goal = "Complete the task pipeline: " + " Then ".join(steps[:-1]) + "."
    final_step = steps[-1]

    plan_text = " ".join(f"Step {i+1}: {s}" for i, s in enumerate(steps[:-1]))
    context_base = f"Goal: {goal} Plan: {plan_text}"
    context = _fill_to_tokens(context_base, target_tokens, plan_text)

    return LongHorizonTask(
        task_id=task_id,
        task_type="goal_completion",
        context=context,
        query=f"What is the final step to complete the goal?",
        ground_truth=final_step,
        intermediate_states=steps[:-1],
        context_length_target=target_tokens,
        actual_context_tokens=_estimate_tokens(context),
        difficulty=difficulty,
    )


_GENERATORS: Dict[str, Callable] = {
    "multi_session_qa": _generate_multi_session_qa_task,
    "state_tracking": _generate_state_tracking_task,
    "goal_completion": _generate_goal_completion_task,
}


def _generate_tasks_for_tier(
    context_length: int,
    n_per_type: int = 10,
    seed: int = 42,
) -> List[LongHorizonTask]:
    """Generate n_per_type tasks for each task type at a given context length."""
    rng = random.Random(seed + context_length)
    difficulties = ["easy", "medium", "hard"]
    tasks: List[LongHorizonTask] = []
    for task_type, gen_fn in _GENERATORS.items():
        for i in range(n_per_type):
            difficulty = difficulties[i % 3]
            task_id = f"{task_type}_{TIER_NAMES[context_length]}_{i:03d}"
            task = gen_fn(task_id, context_length, rng, difficulty)
            tasks.append(task)
    return tasks


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

import re as _re


def _tokenize(text: str) -> List[str]:
    return _re.findall(r"[a-zA-Z0-9]+", text.lower())


def _lcs(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
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


def _rouge_l(pred: str, ref: str) -> float:
    p_toks = _tokenize(pred)
    r_toks = _tokenize(ref)
    if not p_toks or not r_toks:
        return 0.0
    lcs_len = _lcs(p_toks, r_toks)
    prec = lcs_len / len(p_toks)
    rec = lcs_len / len(r_toks)
    denom = prec + rec
    return (2 * prec * rec / denom) if denom > 0 else 0.0


def _exact_match(pred: str, ref: str) -> bool:
    return pred.strip().lower() == ref.strip().lower()


# ---------------------------------------------------------------------------
# Method protocol — any callable that maps (query, context) -> str
# ---------------------------------------------------------------------------

MethodCallable = Callable[[str, str], str]


# ---------------------------------------------------------------------------
# LongHorizonBenchmark
# ---------------------------------------------------------------------------

class LongHorizonBenchmark:
    """
    Evaluates ContextOS methods across four context-length tiers.

    Parameters
    ----------
    n_tasks_per_type : int
        Number of tasks to generate per task-type per tier.
    seed : int
        Global random seed.
    rouge_l_threshold : float
        ROUGE-L cutoff for counting a prediction as "successful".
    tiers : List[int], optional
        Override the default context length tiers.
    """

    def __init__(
        self,
        n_tasks_per_type: int = 10,
        seed: int = 42,
        rouge_l_threshold: float = 0.4,
        tiers: Optional[List[int]] = None,
    ) -> None:
        self._n_per_type = n_tasks_per_type
        self._seed = seed
        self._threshold = rouge_l_threshold
        self._tiers = tiers or CONTEXT_LENGTH_TIERS
        # Pre-generate tasks per tier
        self._tasks: Dict[int, List[LongHorizonTask]] = {
            length: _generate_tasks_for_tier(length, n_per_type=n_tasks_per_type, seed=seed)
            for length in self._tiers
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tasks(self, context_length: int) -> List[LongHorizonTask]:
        """Return the tasks for a given context length tier."""
        if context_length not in self._tasks:
            raise ValueError(
                f"Unknown context length {context_length}. "
                f"Available: {list(self._tasks.keys())}"
            )
        return self._tasks[context_length]

    def evaluate_at_length(
        self,
        method: MethodCallable,
        context_length: int,
        method_name: str = "method",
    ) -> LongHorizonResult:
        """
        Evaluate a method at a single context length tier.

        Parameters
        ----------
        method : Callable[[str, str], str]
            Function that takes (query, context) and returns a predicted answer.
        context_length : int
            One of CONTEXT_LENGTH_TIERS.
        method_name : str
            Display name for the method.

        Returns
        -------
        LongHorizonResult
        """
        tasks = self.get_tasks(context_length)
        if not tasks:
            raise ValueError(f"No tasks found for context_length={context_length}")

        all_scores: List[float] = []
        rouge_scores: List[float] = []
        tokens_used: List[int] = []
        latencies: List[float] = []
        by_type: Dict[str, List[float]] = {tt: [] for tt in TASK_TYPES}
        consistency_pairs: List[Tuple[str, str]] = []

        for task in tasks:
            t0 = time.perf_counter()
            try:
                prediction = method(task.query, task.context)
            except Exception as exc:
                warnings.warn(f"Method raised exception on task {task.task_id}: {exc}")
                prediction = ""
            elapsed = (time.perf_counter() - t0) * 1000

            rouge = _rouge_l(prediction, task.ground_truth)
            success = float(
                _exact_match(prediction, task.ground_truth) or rouge >= self._threshold
            )

            all_scores.append(success)
            rouge_scores.append(rouge)
            tokens_used.append(_estimate_tokens(task.context))
            latencies.append(elapsed)
            by_type[task.task_type].append(success)

            # Collect consecutive predictions for consistency check
            consistency_pairs.append((prediction, task.ground_truth))

        # Aggregate by task type
        by_type_agg: Dict[str, float] = {}
        for tt, scores in by_type.items():
            by_type_agg[tt] = statistics.mean(scores) if scores else 0.0

        # Consistency: std dev of scores for same task type (lower = more consistent)
        consistency_values: List[float] = []
        for tt, scores in by_type.items():
            if len(scores) > 1:
                consistency_values.append(1.0 - min(1.0, statistics.stdev(scores)))
        consistency_score = statistics.mean(consistency_values) if consistency_values else 1.0

        mean_tokens = statistics.mean(tokens_used) if tokens_used else 0.0
        context_utilization = mean_tokens / context_length if context_length > 0 else 0.0

        return LongHorizonResult(
            method_name=method_name,
            context_length=context_length,
            tier_name=TIER_NAMES.get(context_length, str(context_length)),
            task_success_rate=statistics.mean(all_scores) if all_scores else 0.0,
            state_tracking_accuracy=by_type_agg.get("state_tracking", 0.0),
            goal_completion_rate=by_type_agg.get("goal_completion", 0.0),
            multi_session_qa_score=by_type_agg.get("multi_session_qa", 0.0),
            mean_tokens_used=mean_tokens,
            context_utilization=round(context_utilization, 4),
            mean_rouge_l=statistics.mean(rouge_scores) if rouge_scores else 0.0,
            consistency_score=round(consistency_score, 4),
            mean_latency_ms=statistics.mean(latencies) if latencies else 0.0,
            n_tasks=len(tasks),
            by_task_type=by_type_agg,
            per_task_scores=all_scores,
        )

    def run_scaling_analysis(
        self,
        methods: Dict[str, MethodCallable],
    ) -> ScalingAnalysis:
        """
        Run evaluation for all methods across all context length tiers and
        produce a ScalingAnalysis.

        Parameters
        ----------
        methods : Dict[str, Callable]
            Method name -> callable mapping.

        Returns
        -------
        ScalingAnalysis
        """
        method_names = list(methods.keys())
        analysis = ScalingAnalysis(
            methods=method_names,
            tiers=self._tiers,
        )

        for method_name, method_fn in methods.items():
            analysis.results[method_name] = {}
            for tier in self._tiers:
                result = self.evaluate_at_length(method_fn, tier, method_name=method_name)
                analysis.results[method_name][tier] = result

        # Compute scaling slopes (linear regression of TSR over log2(context_length))
        log_tiers = [math.log2(t) for t in self._tiers]
        for method_name in method_names:
            tsr_values = [
                analysis.results[method_name][t].task_success_rate
                for t in self._tiers
            ]
            slope = _linear_slope(log_tiers, tsr_values)
            analysis.scaling_slopes[method_name] = round(slope, 6)

            # Degradation: (TSR at 512 - TSR at 32k) / TSR at 512
            tsr_min = analysis.results[method_name][self._tiers[0]].task_success_rate
            tsr_max = analysis.results[method_name][self._tiers[-1]].task_success_rate
            degradation = ((tsr_min - tsr_max) / tsr_min) if tsr_min > 0 else 0.0
            analysis.degradation_rates[method_name] = round(max(0.0, degradation), 4)

        # Best method per tier
        for tier in self._tiers:
            best_name = max(
                method_names,
                key=lambda n: analysis.results[n][tier].task_success_rate,
            )
            analysis.best_at_each_tier[tier] = best_name

        analysis.summary_table = _format_scaling_table(analysis)
        return analysis

    def generate_tier_report(self, result: LongHorizonResult) -> str:
        """Generate a formatted report for a single tier evaluation."""
        sep = "-" * 60
        lines = [
            sep,
            f"  Long-Horizon Eval: {result.method_name} @ {result.tier_name} tokens",
            sep,
            f"  Tasks evaluated  : {result.n_tasks}",
            f"  Task success rate: {result.task_success_rate:.4f}",
            f"  State tracking   : {result.state_tracking_accuracy:.4f}",
            f"  Goal completion  : {result.goal_completion_rate:.4f}",
            f"  Multi-session QA : {result.multi_session_qa_score:.4f}",
            f"  Mean ROUGE-L     : {result.mean_rouge_l:.4f}",
            f"  Consistency      : {result.consistency_score:.4f}",
            f"  Context util.    : {result.context_utilization:.4f}",
            f"  Mean latency(ms) : {result.mean_latency_ms:.2f}",
            sep,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_slope(x: List[float], y: List[float]) -> float:
    """Ordinary least-squares slope dy/dx."""
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den = sum((xi - mean_x) ** 2 for xi in x)
    return num / den if den != 0 else 0.0


def _format_scaling_table(analysis: ScalingAnalysis) -> str:
    """Format a scaling analysis summary table."""
    sep = "=" * 70
    thin = "-" * 70
    tier_headers = "  ".join(f"{TIER_NAMES.get(t, str(t)):>8}" for t in analysis.tiers)
    lines = [
        sep,
        "  Long-Horizon Scaling Analysis — Task Success Rate",
        sep,
        f"  {'Method':<24}  {tier_headers}  {'Slope':>8}  {'Degr.':>6}",
        thin,
    ]
    for method in analysis.methods:
        tsr_vals = "  ".join(
            f"{analysis.results[method][t].task_success_rate:>8.4f}"
            for t in analysis.tiers
        )
        slope = analysis.scaling_slopes.get(method, 0.0)
        degrad = analysis.degradation_rates.get(method, 0.0)
        lines.append(f"  {method:<24}  {tsr_vals}  {slope:>8.5f}  {degrad:>6.4f}")

    lines.append(thin)
    lines.append("  Best method per tier:")
    for tier in analysis.tiers:
        best = analysis.best_at_each_tier.get(tier, "N/A")
        lines.append(f"    {TIER_NAMES.get(tier, str(tier)):>8}: {best}")
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bench = LongHorizonBenchmark(n_tasks_per_type=5, seed=0)

    # Oracle method: returns ground truth from context if found
    def oracle_method(query: str, context: str) -> str:
        # Simplified: look for a sentence that answers the query
        words = set(query.lower().split())
        for sent in context.split("."):
            sent_words = set(sent.lower().split())
            if len(words & sent_words) > 2:
                return sent.strip()
        return context[:80].strip()

    # Baseline: always returns the first 80 chars
    def baseline_method(query: str, context: str) -> str:
        return context[:80].strip()

    analysis = bench.run_scaling_analysis({
        "oracle": oracle_method,
        "baseline": baseline_method,
    })
    print(analysis.summary_table)

    # Single-tier report
    result = bench.evaluate_at_length(oracle_method, 2048, method_name="oracle")
    print(bench.generate_tier_report(result))
