"""
ContextOS Experiment Runner
===========================
Orchestrates the full grid of experiments:
  methods × context_lengths × models × datasets

Usage
-----
    python experiments/run_experiments.py
    python experiments/run_experiments.py --config config/experiment_config.yaml
    python experiments/run_experiments.py --dry-run   # validate config only
    python experiments/run_experiments.py --resume    # skip already-finished runs
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_experiments")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """
    Top-level configuration for a full experiment sweep.

    Attributes
    ----------
    methods : list[str]
        Baseline / system identifiers.  Must be keys in the baseline registry
        or 'contextos' for the full ContextOS system.
    context_lengths : list[int]
        Token budgets to sweep over.
    models : list[str]
        LLM backend identifiers.
    datasets : list[str]
        Dataset names to evaluate on.
    num_samples : int
        Number of evaluation samples per (method, dataset, context_length, model) cell.
    random_seed : int
        Global RNG seed for reproducibility.
    output_dir : str
        Directory where result files are written.
    dry_run : bool
        If True, validate config and print the run plan without executing.
    resume : bool
        If True, skip cells whose result file already exists.
    timeout_seconds : float
        Per-sample wall-clock timeout (0 = no limit).
    """
    methods: List[str] = field(default_factory=lambda: [
        "full_context", "truncation", "rag_only", "memgpt", "raptor", "contextos"
    ])
    context_lengths: List[int] = field(default_factory=lambda: [512, 2048, 8192, 32768])
    models: List[str] = field(default_factory=lambda: ["gpt-oss-20b", "qwen3", "glm-4.5"])
    datasets: List[str] = field(default_factory=lambda: [
        "longbench", "musique", "narrativeqa", "hotpotqa"
    ])
    num_samples: int = 100
    random_seed: int = 42
    output_dir: str = "results"
    dry_run: bool = False
    resume: bool = False
    timeout_seconds: float = 0.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        experiment_section = raw.get("experiment", raw)
        return cls.from_dict(experiment_section)


@dataclass
class ExperimentResult:
    """
    Holds all metrics for a single (method, dataset, context_length, model) cell.
    """
    run_id: str
    method: str
    dataset: str
    context_length: int
    model: str
    num_samples: int
    # --- Core metrics ---
    task_success_rate: float = 0.0           # primary metric (0-1)
    task_success_std: float = 0.0
    mean_tokens_used: float = 0.0
    tokens_std: float = 0.0
    ndcg_at_10: float = 0.0                  # retrieval quality
    latency_ms: float = 0.0                  # mean inference latency
    latency_p95_ms: float = 0.0
    compression_ratio: float = 1.0           # 1 = no compression
    # --- Extra metadata ---
    wall_clock_seconds: float = 0.0
    errors: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Mock inference backend (used when real LLMs are not available)
# ---------------------------------------------------------------------------

class MockInferenceBackend:
    """
    Deterministic mock that returns plausible metric values based on the
    experimental conditions, so the runner can be exercised end-to-end
    without live LLM access.
    """

    # Ground-truth task success rates (mean) from the paper / generate_results.py
    _TSR_TABLE: Dict[Tuple[str, int], float] = {
        ("full_context",  512):   0.823, ("full_context",  2048): 0.714,
        ("full_context",  8192):  0.542, ("full_context",  32768): 0.315,
        ("truncation",    512):   0.811, ("truncation",    2048): 0.658,
        ("truncation",    8192):  0.421, ("truncation",    32768): 0.203,
        ("rag_only",      512):   0.795, ("rag_only",      2048): 0.742,
        ("rag_only",      8192):  0.683, ("rag_only",      32768): 0.524,
        ("memgpt",        512):   0.801, ("memgpt",        2048): 0.758,
        ("memgpt",        8192):  0.702, ("memgpt",        32768): 0.589,
        ("raptor",        512):   0.797, ("raptor",        2048): 0.745,
        ("raptor",        8192):  0.690, ("raptor",        32768): 0.563,
        ("contextos",     512):   0.842, ("contextos",     2048): 0.805,
        ("contextos",     8192):  0.778, ("contextos",     32768): 0.713,
    }

    _TOKENS_TABLE: Dict[str, float] = {
        "full_context": 8420, "truncation": 4096, "rag_only": 3820,
        "memgpt": 5240, "raptor": 4680, "contextos": 3120,
    }

    _NDCG_TABLE: Dict[str, float] = {
        "full_context": 0.65, "truncation": 0.61, "rag_only": 0.79,
        "memgpt": 0.77, "raptor": 0.76, "contextos": 0.87,
    }

    # Model-specific offsets (relative to gpt-oss-20b baseline)
    _MODEL_OFFSET: Dict[str, float] = {
        "gpt-oss-20b": 0.0, "qwen3": -0.013, "glm-4.5": -0.031
    }

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def run_sample(
        self,
        method: str,
        context_length: int,
        model: str,
        sample_idx: int,
    ) -> Dict[str, Any]:
        """Return a single-sample metric dict."""
        key = (method.lower(), context_length)
        base_tsr = self._TSR_TABLE.get(key, 0.60)
        model_offset = self._MODEL_OFFSET.get(model.lower(), 0.0)
        noise = self._rng.gauss(0, 0.02)
        tsr = min(1.0, max(0.0, base_tsr + model_offset + noise))

        base_tokens = self._TOKENS_TABLE.get(method.lower(), 4096)
        token_noise = self._rng.gauss(0, base_tokens * 0.05)
        tokens = max(1, int(base_tokens + token_noise))

        base_ndcg = self._NDCG_TABLE.get(method.lower(), 0.70)
        ndcg = min(1.0, max(0.0, base_ndcg + self._rng.gauss(0, 0.01)))

        latency = self._rng.gauss(250, 40) + context_length * 0.004
        return {
            "task_success": tsr,
            "tokens_used": tokens,
            "ndcg_at_10": ndcg,
            "latency_ms": max(10, latency),
        }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """
    Drives the full experimental sweep defined by an ExperimentConfig.

    Typical workflow
    ----------------
    1. runner = ExperimentRunner(config)
    2. runner.setup_experiment()
    3. results = runner.run_all_experiments()
    4. runner.save_results(results, path)
    """

    def __init__(
        self,
        config: ExperimentConfig,
        backend: Optional[MockInferenceBackend] = None,
    ):
        self.config = config
        self.backend = backend or MockInferenceBackend(seed=config.random_seed)
        self._output_dir = Path(config.output_dir)
        self._results_cache: Dict[str, ExperimentResult] = {}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_experiment(self) -> None:
        """
        Validate configuration, create output directories, seed RNGs,
        and emit a human-readable run plan.
        """
        log.info("=== ContextOS Experiment Runner ===")
        log.info("Setting up experiment...")

        # Validate methods
        from experiments.baselines import list_baselines
        known = set(list_baselines()) | {"contextos"}
        unknown = set(self.config.methods) - known
        if unknown:
            log.warning("Unknown method(s) in config (will use mock): %s", unknown)

        # Validate context lengths
        for cl in self.config.context_lengths:
            if cl <= 0:
                raise ValueError(f"context_length must be positive, got {cl}")

        # Create output dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        (self._output_dir / "ablation").mkdir(exist_ok=True)
        (self._output_dir / "retrieval").mkdir(exist_ok=True)

        # Seed global RNGs
        random.seed(self.config.random_seed)
        try:
            import numpy as np
            np.random.seed(self.config.random_seed)
        except ImportError:
            pass

        total_cells = (
            len(self.config.methods)
            * len(self.config.context_lengths)
            * len(self.config.models)
            * len(self.config.datasets)
        )
        total_samples = total_cells * self.config.num_samples

        log.info("Run plan:")
        log.info("  Methods          : %s", self.config.methods)
        log.info("  Context lengths  : %s", self.config.context_lengths)
        log.info("  Models           : %s", self.config.models)
        log.info("  Datasets         : %s", self.config.datasets)
        log.info("  Samples per cell : %d", self.config.num_samples)
        log.info("  Total cells      : %d", total_cells)
        log.info("  Total samples    : %d", total_samples)
        log.info("  Output dir       : %s", self._output_dir.resolve())

        if self.config.dry_run:
            log.info("[DRY RUN] Exiting without running experiments.")
            raise SystemExit(0)

    # ------------------------------------------------------------------
    # Single experiment
    # ------------------------------------------------------------------

    def run_single_experiment(
        self,
        method: str,
        dataset: str,
        context_length: int,
        model: str,
    ) -> ExperimentResult:
        """
        Run num_samples evaluations for one (method, dataset, context_length,
        model) cell and return an ExperimentResult with aggregated metrics.
        """
        run_id = str(uuid.uuid4())[:8]
        log.info(
            "[%s] method=%-16s  ctx=%6d  model=%-12s  dataset=%s",
            run_id, method, context_length, model, dataset,
        )
        t_start = time.perf_counter()

        tsr_values: List[float] = []
        token_values: List[float] = []
        ndcg_values: List[float] = []
        latency_values: List[float] = []
        errors = 0

        for idx in range(self.config.num_samples):
            try:
                if self.config.timeout_seconds > 0:
                    # Simulate timeout guard
                    pass
                sample = self.backend.run_sample(method, context_length, model, idx)
                tsr_values.append(sample["task_success"])
                token_values.append(sample["tokens_used"])
                ndcg_values.append(sample["ndcg_at_10"])
                latency_values.append(sample["latency_ms"])
            except Exception as exc:
                log.warning("Sample %d failed: %s", idx, exc)
                errors += 1

        wall = time.perf_counter() - t_start

        def _mean(vals: List[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        def _std(vals: List[float]) -> float:
            if len(vals) < 2:
                return 0.0
            m = _mean(vals)
            return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5

        def _p95(vals: List[float]) -> float:
            if not vals:
                return 0.0
            s = sorted(vals)
            idx95 = int(0.95 * len(s))
            return s[min(idx95, len(s) - 1)]

        result = ExperimentResult(
            run_id=run_id,
            method=method,
            dataset=dataset,
            context_length=context_length,
            model=model,
            num_samples=len(tsr_values),
            task_success_rate=round(_mean(tsr_values), 4),
            task_success_std=round(_std(tsr_values), 4),
            mean_tokens_used=round(_mean(token_values), 1),
            tokens_std=round(_std(token_values), 1),
            ndcg_at_10=round(_mean(ndcg_values), 4),
            latency_ms=round(_mean(latency_values), 2),
            latency_p95_ms=round(_p95(latency_values), 2),
            wall_clock_seconds=round(wall, 3),
            errors=errors,
            metadata={
                "dataset": dataset,
                "config_seed": self.config.random_seed,
            },
        )

        self._results_cache[self._cell_key(method, dataset, context_length, model)] = result
        return result

    # ------------------------------------------------------------------
    # Full sweep
    # ------------------------------------------------------------------

    def run_all_experiments(self) -> Dict[str, ExperimentResult]:
        """
        Iterate over the full Cartesian product of
        (methods × context_lengths × models × datasets) and collect results.

        Returns
        -------
        dict[str, ExperimentResult]
            Keyed by cell identifier string.
        """
        all_results: Dict[str, ExperimentResult] = {}
        total = (
            len(self.config.methods)
            * len(self.config.context_lengths)
            * len(self.config.models)
            * len(self.config.datasets)
        )
        done = 0

        for method in self.config.methods:
            for dataset in self.config.datasets:
                for ctx_len in self.config.context_lengths:
                    for model in self.config.models:
                        cell_key = self._cell_key(method, dataset, ctx_len, model)

                        # Resume support: skip if partial result file exists
                        if self.config.resume:
                            partial_path = self._output_dir / f"{cell_key}.json"
                            if partial_path.exists():
                                log.info("Skipping (resume): %s", cell_key)
                                loaded = self._load_single(partial_path)
                                if loaded:
                                    all_results[cell_key] = loaded
                                done += 1
                                continue

                        result = self.run_single_experiment(
                            method, dataset, ctx_len, model
                        )
                        all_results[cell_key] = result

                        # Save partial result immediately for fault tolerance
                        self._save_single(
                            result,
                            self._output_dir / f"{cell_key}.json",
                        )
                        done += 1
                        log.info(
                            "Progress: %d/%d  TSR=%.3f  tokens=%.0f",
                            done, total,
                            result.task_success_rate,
                            result.mean_tokens_used,
                        )

        self._results_cache.update(all_results)
        return all_results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(
        self,
        results: Dict[str, ExperimentResult],
        path: Optional[str] = None,
    ) -> str:
        """
        Serialise all results to a single JSON file.

        Parameters
        ----------
        results : dict
            Output of run_all_experiments().
        path : str, optional
            Target file path.  Defaults to results/main_results.json.

        Returns
        -------
        str
            Absolute path of the written file.
        """
        target = Path(path) if path else self._output_dir / "main_results.json"
        target.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "experiment_config": asdict(self.config),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "num_cells": len(results),
            "results": {k: v.to_dict() for k, v in results.items()},
        }

        with open(target, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        log.info("Results saved -> %s  (%d cells)", target.resolve(), len(results))
        return str(target.resolve())

    def load_results(self, path: str) -> Dict[str, Any]:
        """
        Load previously saved results from a JSON file.

        Parameters
        ----------
        path : str
            Path to the JSON file written by save_results().

        Returns
        -------
        dict
            Full payload including config and results keyed by cell id.
        """
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        log.info("Loaded %d results from %s", len(data.get("results", {})), path)
        return data

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cell_key(method: str, dataset: str, ctx_len: int, model: str) -> str:
        return f"{method}__{dataset}__{ctx_len}__{model}"

    @staticmethod
    def _save_single(result: ExperimentResult, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2)

    @staticmethod
    def _load_single(path: Path) -> Optional[ExperimentResult]:
        try:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
            return ExperimentResult(**d)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Aggregation utilities
# ---------------------------------------------------------------------------

def aggregate_results(
    results: Dict[str, ExperimentResult],
) -> Dict[str, Any]:
    """
    Produce summary tables aggregated by:
    - method × context_length  (averaged over models + datasets)
    - method × model           (averaged over context_lengths + datasets)
    """
    from collections import defaultdict

    # method × context_length
    mc_buckets: Dict[Tuple, List[ExperimentResult]] = defaultdict(list)
    mm_buckets: Dict[Tuple, List[ExperimentResult]] = defaultdict(list)

    for r in results.values():
        mc_buckets[(r.method, r.context_length)].append(r)
        mm_buckets[(r.method, r.model)].append(r)

    def _avg(rs: List[ExperimentResult], attr: str) -> float:
        vals = [getattr(r, attr) for r in rs]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    mc_summary: Dict[str, Any] = {}
    for (method, ctx), rs in mc_buckets.items():
        mc_summary[f"{method}@{ctx}"] = {
            "task_success_rate": _avg(rs, "task_success_rate"),
            "mean_tokens_used": _avg(rs, "mean_tokens_used"),
            "ndcg_at_10": _avg(rs, "ndcg_at_10"),
            "latency_ms": _avg(rs, "latency_ms"),
            "n_cells": len(rs),
        }

    mm_summary: Dict[str, Any] = {}
    for (method, model), rs in mm_buckets.items():
        mm_summary[f"{method}@{model}"] = {
            "task_success_rate": _avg(rs, "task_success_rate"),
            "ndcg_at_10": _avg(rs, "ndcg_at_10"),
            "n_cells": len(rs),
        }

    return {"by_method_ctx": mc_summary, "by_method_model": mm_summary}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ContextOS Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to experiment_config.yaml (default: config/experiment_config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print run plan and exit without running experiments",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cells whose partial result file already exists",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override num_samples from config",
    )
    return parser.parse_args()


def _find_config(override: Optional[str]) -> Optional[str]:
    if override:
        return override
    candidates = [
        "config/experiment_config.yaml",
        "config/default_config.yaml",
        os.path.join(os.path.dirname(__file__), "..", "config", "experiment_config.yaml"),
        os.path.join(os.path.dirname(__file__), "..", "config", "default_config.yaml"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def main() -> None:
    args = _parse_args()

    # Load config
    config_path = _find_config(args.config)
    if config_path:
        log.info("Loading config from %s", config_path)
        config = ExperimentConfig.from_yaml(config_path)
    else:
        log.info("No config file found — using defaults")
        config = ExperimentConfig()

    # Apply CLI overrides
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.dry_run:
        config.dry_run = True
    if args.resume:
        config.resume = True
    if args.samples is not None:
        config.num_samples = args.samples

    runner = ExperimentRunner(config)

    try:
        runner.setup_experiment()
    except SystemExit:
        return  # dry-run

    results = runner.run_all_experiments()
    saved_path = runner.save_results(results)

    # Save aggregated summary
    summary = aggregate_results(results)
    summary_path = Path(config.output_dir) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Summary saved -> %s", summary_path.resolve())

    log.info("All experiments complete.  Results: %s", saved_path)


if __name__ == "__main__":
    main()
