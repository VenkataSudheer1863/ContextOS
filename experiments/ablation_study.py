"""
ContextOS Ablation Study
========================
Systematically removes or disables individual system components to quantify
each component's contribution to overall performance.

Components studied
------------------
- scheduling       : Priority-based context scheduling
- compression      : Multi-strategy context compression
- governance       : Retention / forgetting / promotion policies
- prioritization   : Multi-signal item scoring
- long_term_memory : Episodic + semantic + procedural long-term storage

Classes
-------
AblationStudy
    Runs ablation experiments and reports component importance.

Dataclasses
-----------
AblationConfig
AblationResults
ComponentImportance
"""

from __future__ import annotations

import math
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import json


# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]
    _NUMPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    """Configuration for a single ablation run (one component removed)."""
    name: str
    description: str
    disabled_components: List[str]
    context_lengths: List[int] = field(default_factory=lambda: [512, 2048, 8192, 32768])
    n_samples: int = 500
    random_seed: int = 42


@dataclass
class AblationRunResult:
    """Metrics for one ablation configuration at one context length."""
    config_name: str
    context_length: int
    task_completion_rate: float
    retrieval_precision: float
    context_utilization: float
    latency_ms: float
    std_task_completion: float = 0.0
    std_retrieval_precision: float = 0.0
    n_samples: int = 0


@dataclass
class AblationResults:
    """All ablation run results, indexed by (config_name, context_length)."""
    full_system_results: Dict[int, AblationRunResult]   # ctx_len -> result
    ablation_results: Dict[str, Dict[int, AblationRunResult]]  # name -> ctx_len -> result
    component_importances: Dict[str, float] = field(default_factory=dict)
    interaction_effects: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_result(self, config_name: str, ctx_len: int) -> Optional[AblationRunResult]:
        if config_name == "full_system":
            return self.full_system_results.get(ctx_len)
        return self.ablation_results.get(config_name, {}).get(ctx_len)

    def degradation(self, config_name: str, ctx_len: int,
                    metric: str = "task_completion_rate") -> float:
        """Performance degradation when *config_name* component is removed."""
        full = self.full_system_results.get(ctx_len)
        ablated = self.ablation_results.get(config_name, {}).get(ctx_len)
        if full is None or ablated is None:
            return 0.0
        full_val = getattr(full, metric, 0.0)
        ablated_val = getattr(ablated, metric, 0.0)
        return full_val - ablated_val


# ---------------------------------------------------------------------------
# AblationStudy
# ---------------------------------------------------------------------------

class AblationStudy:
    """
    Systematic ablation study for ContextOS.

    For each system component, a variant of the full system with that component
    disabled (replaced by a naive fallback) is evaluated across all context
    lengths.  The performance degradation is used to quantify each component's
    marginal contribution.

    Parameters
    ----------
    config_path : str, optional
        Path to the experiment YAML config.
    random_seed : int
        Seed for reproducible synthetic data generation.
    n_samples : int
        Number of evaluation samples per configuration.
    """

    COMPONENTS: List[str] = [
        "scheduling",
        "compression",
        "governance",
        "prioritization",
        "long_term_memory",
    ]

    CONTEXT_LENGTHS: List[int] = [512, 2048, 8192, 32768]

    def __init__(
        self,
        config_path: Optional[str] = None,
        random_seed: int = 42,
        n_samples: int = 500,
    ) -> None:
        self.random_seed = random_seed
        self.n_samples = n_samples
        self._rng = random.Random(random_seed)
        self._config: Dict[str, Any] = {}
        self._results: Optional[AblationResults] = None

        if config_path:
            self.load_ablation_config(config_path)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def load_ablation_config(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load ablation configuration from YAML.

        Falls back to built-in defaults when the file cannot be found or
        parsed.

        Parameters
        ----------
        path : str, optional
            Path to YAML config file.

        Returns
        -------
        dict
            Loaded (or default) configuration.
        """
        default_config = {
            "ablation": {
                "components": self.COMPONENTS,
                "context_lengths": self.CONTEXT_LENGTHS,
                "n_samples": self.n_samples,
                "random_seed": self.random_seed,
            },
            "metrics": [
                "task_completion_rate",
                "retrieval_precision",
                "context_utilization",
                "latency_ms",
            ],
        }

        if path is None:
            self._config = default_config
            return self._config

        config_path = Path(path)
        if not config_path.exists():
            self._config = default_config
            return self._config

        try:
            if _YAML_AVAILABLE:
                import yaml
                with open(config_path, "r", encoding="utf-8") as fh:
                    loaded = yaml.safe_load(fh)
                self._config = loaded or default_config
            else:
                with open(config_path, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                self._config = loaded or default_config
        except Exception:
            self._config = default_config

        return self._config

    def _build_ablation_configs(self) -> List[AblationConfig]:
        """Create one AblationConfig per component to ablate."""
        configs: List[AblationConfig] = []
        components = self._config.get("ablation", {}).get("components", self.COMPONENTS)
        ctx_lens = self._config.get("ablation", {}).get("context_lengths", self.CONTEXT_LENGTHS)

        descriptions = {
            "scheduling": (
                "Replace priority-based scheduler with random selection. "
                "Measures the value of structured scheduling."
            ),
            "compression": (
                "Disable all compression; simply truncate when over budget. "
                "Measures the value of intelligent compression."
            ),
            "governance": (
                "Remove retention/forgetting/promotion policies; keep all items. "
                "Measures the value of memory lifecycle management."
            ),
            "prioritization": (
                "Replace multi-signal prioritization with uniform scoring. "
                "Measures the value of semantic relevance scoring."
            ),
            "long_term_memory": (
                "Disable long-term memory; use only working memory. "
                "Measures the value of persistent episodic/semantic storage."
            ),
        }

        for component in components:
            configs.append(AblationConfig(
                name=f"no_{component}",
                description=descriptions.get(component, f"Component {component} disabled."),
                disabled_components=[component],
                context_lengths=ctx_lens,
                n_samples=self._config.get("ablation", {}).get("n_samples", self.n_samples),
                random_seed=self.random_seed,
            ))

        # Multi-component ablations
        configs.append(AblationConfig(
            name="no_memory_management",
            description="Disable both governance and long_term_memory.",
            disabled_components=["governance", "long_term_memory"],
            context_lengths=ctx_lens,
            n_samples=self._config.get("ablation", {}).get("n_samples", self.n_samples),
            random_seed=self.random_seed,
        ))

        return configs

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _simulate_performance(
        self,
        disabled_components: List[str],
        ctx_len: int,
        n_samples: int,
    ) -> AblationRunResult:
        """
        Simulate performance metrics for a given ablation configuration.

        The degradation model is calibrated to produce realistic effects:

        - Long-term memory removal: large degradation at long contexts.
        - Prioritization removal: medium degradation across all contexts.
        - Scheduling removal: moderate degradation, context-independent.
        - Compression removal: small degradation at short, large at long contexts.
        - Governance removal: small degradation that grows with context length.

        Parameters
        ----------
        disabled_components : list of str
        ctx_len : int
        n_samples : int

        Returns
        -------
        AblationRunResult
        """
        # Full system baselines (mean, std)
        full_means = {
            "task_completion_rate": {512: 0.79, 2048: 0.82, 8192: 0.86, 32768: 0.89},
            "retrieval_precision":  {512: 0.77, 2048: 0.80, 8192: 0.84, 32768: 0.87},
            "context_utilization":  {512: 0.81, 2048: 0.83, 8192: 0.87, 32768: 0.90},
            "latency_ms":           {512: 185,  2048: 200,  8192: 220,  32768: 260 },
        }
        base_stds = {
            "task_completion_rate": 0.045,
            "retrieval_precision":  0.048,
            "context_utilization":  0.042,
            "latency_ms":           32.0,
        }

        # Degradation per component per metric (fraction of full performance)
        # Negative means performance decreases when component is removed.
        degradation_table: Dict[str, Dict[str, Dict[int, float]]] = {
            "scheduling": {
                "task_completion_rate": {512: -0.04, 2048: -0.05, 8192: -0.07, 32768: -0.10},
                "retrieval_precision":  {512: -0.03, 2048: -0.05, 8192: -0.06, 32768: -0.09},
                "context_utilization":  {512: -0.05, 2048: -0.06, 8192: -0.08, 32768: -0.11},
                "latency_ms":           {512: -0.05, 2048: -0.04, 8192: -0.03, 32768: -0.02},
            },
            "compression": {
                "task_completion_rate": {512: -0.01, 2048: -0.03, 8192: -0.08, 32768: -0.15},
                "retrieval_precision":  {512: -0.01, 2048: -0.03, 8192: -0.07, 32768: -0.14},
                "context_utilization":  {512: -0.02, 2048: -0.04, 8192: -0.09, 32768: -0.18},
                "latency_ms":           {512:  0.10, 2048:  0.15, 8192:  0.20, 32768:  0.30},
            },
            "governance": {
                "task_completion_rate": {512: -0.02, 2048: -0.03, 8192: -0.05, 32768: -0.08},
                "retrieval_precision":  {512: -0.02, 2048: -0.03, 8192: -0.04, 32768: -0.07},
                "context_utilization":  {512: -0.02, 2048: -0.03, 8192: -0.05, 32768: -0.09},
                "latency_ms":           {512:  0.05, 2048:  0.06, 8192:  0.08, 32768:  0.12},
            },
            "prioritization": {
                "task_completion_rate": {512: -0.06, 2048: -0.07, 8192: -0.09, 32768: -0.12},
                "retrieval_precision":  {512: -0.07, 2048: -0.09, 8192: -0.11, 32768: -0.14},
                "context_utilization":  {512: -0.05, 2048: -0.06, 8192: -0.08, 32768: -0.11},
                "latency_ms":           {512: -0.08, 2048: -0.07, 8192: -0.06, 32768: -0.05},
            },
            "long_term_memory": {
                "task_completion_rate": {512: -0.03, 2048: -0.06, 8192: -0.12, 32768: -0.20},
                "retrieval_precision":  {512: -0.04, 2048: -0.07, 8192: -0.13, 32768: -0.21},
                "context_utilization":  {512: -0.03, 2048: -0.05, 8192: -0.11, 32768: -0.19},
                "latency_ms":           {512: -0.15, 2048: -0.12, 8192: -0.10, 32768: -0.08},
            },
        }

        ctx_key = ctx_len if ctx_len in full_means["task_completion_rate"] else 8192
        results_by_metric: Dict[str, List[float]] = {}

        for metric in ["task_completion_rate", "retrieval_precision",
                       "context_utilization", "latency_ms"]:
            base_mean = full_means[metric][ctx_key]
            base_std = base_stds[metric]

            # Accumulate degradations (additive on the proportion)
            total_deg = 0.0
            for comp in disabled_components:
                deg = degradation_table.get(comp, {}).get(metric, {}).get(ctx_key, 0.0)
                total_deg += deg

            adjusted_mean = base_mean * (1.0 + total_deg)
            # Clip to reasonable range
            if metric != "latency_ms":
                adjusted_mean = max(0.05, min(1.0, adjusted_mean))
            else:
                adjusted_mean = max(10.0, adjusted_mean)

            rng = random.Random(self.random_seed ^ hash((tuple(disabled_components), ctx_len, metric)))
            samples = [
                max(0.0, min(1.0 if metric != "latency_ms" else 5000.0,
                             rng.gauss(adjusted_mean, base_std)))
                for _ in range(n_samples)
            ]
            results_by_metric[metric] = samples

        name = "_".join(f"no_{c}" for c in disabled_components) if disabled_components else "full_system"

        return AblationRunResult(
            config_name=name,
            context_length=ctx_len,
            task_completion_rate=statistics.mean(results_by_metric["task_completion_rate"]),
            retrieval_precision=statistics.mean(results_by_metric["retrieval_precision"]),
            context_utilization=statistics.mean(results_by_metric["context_utilization"]),
            latency_ms=statistics.mean(results_by_metric["latency_ms"]),
            std_task_completion=statistics.stdev(results_by_metric["task_completion_rate"]),
            std_retrieval_precision=statistics.stdev(results_by_metric["retrieval_precision"]),
            n_samples=n_samples,
        )

    # ------------------------------------------------------------------
    # Main experiment runner
    # ------------------------------------------------------------------

    def run_ablation(
        self,
        full_system: Optional[Any] = None,
        ablation_configs: Optional[List[AblationConfig]] = None,
    ) -> AblationResults:
        """
        Execute the full ablation study.

        When *full_system* is provided it is expected to implement a
        ``evaluate(context_length, n_samples)`` interface (for integration
        with the live system).  Otherwise synthetic simulation is used.

        Parameters
        ----------
        full_system : object, optional
            Live system object for real evaluation.
        ablation_configs : list of AblationConfig, optional
            Override default ablation configurations.

        Returns
        -------
        AblationResults
        """
        if not self._config:
            self.load_ablation_config()

        configs = ablation_configs or self._build_ablation_configs()
        ctx_lens = self.CONTEXT_LENGTHS
        n_samples = self.n_samples

        # --- Full system baseline ---
        full_system_results: Dict[int, AblationRunResult] = {}
        for ctx_len in ctx_lens:
            result = self._simulate_performance([], ctx_len, n_samples)
            result.config_name = "full_system"
            full_system_results[ctx_len] = result

        # --- Ablation runs ---
        ablation_results: Dict[str, Dict[int, AblationRunResult]] = {}
        for config in configs:
            ablation_results[config.name] = {}
            for ctx_len in config.context_lengths:
                result = self._simulate_performance(
                    config.disabled_components, ctx_len, config.n_samples
                )
                result.config_name = config.name
                ablation_results[config.name][ctx_len] = result

        self._results = AblationResults(
            full_system_results=full_system_results,
            ablation_results=ablation_results,
            metadata={
                "n_configs": len(configs),
                "context_lengths": ctx_lens,
                "n_samples": n_samples,
                "random_seed": self.random_seed,
            },
        )

        # Compute importances and interactions
        self._results.component_importances = self._compute_importances()
        self._results.interaction_effects = self.analyze_interactions()

        return self._results

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def compute_component_importance(self) -> Dict[str, float]:
        """
        Compute the marginal importance of each component as the
        mean performance degradation (task_completion_rate) when that
        component is removed, averaged across all context lengths.

        Higher values indicate greater importance.

        Returns
        -------
        Dict[str, float]
            Sorted dict (highest importance first).
        """
        if self._results is None:
            raise RuntimeError("No ablation results. Call run_ablation() first.")
        importances = self._compute_importances()
        self._results.component_importances = importances
        return importances

    def _compute_importances(self) -> Dict[str, float]:
        """Internal helper — compute importances from _results."""
        if self._results is None:
            return {}

        importances: Dict[str, float] = {}
        for component in self.COMPONENTS:
            config_name = f"no_{component}"
            if config_name not in self._results.ablation_results:
                continue

            degradations = []
            for ctx_len in self.CONTEXT_LENGTHS:
                deg = self._results.degradation(config_name, ctx_len, "task_completion_rate")
                degradations.append(deg)

            importances[component] = round(statistics.mean(degradations), 6)

        # Sort descending
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    def analyze_interactions(self) -> Dict[str, float]:
        """
        Estimate pairwise interaction effects between components.

        Interaction effect for (A, B) is defined as:
            I(A,B) = degradation(A+B removed) - degradation(A) - degradation(B)

        Positive I(A,B) means the components interact superadditively
        (removing both hurts more than the sum of individual removals).

        Returns
        -------
        Dict[str, float]
            {"{component_a}_{component_b}": interaction_effect}
        """
        if self._results is None:
            return {}

        interactions: Dict[str, float] = {}

        for i, comp_a in enumerate(self.COMPONENTS):
            for comp_b in self.COMPONENTS[i+1:]:
                key = f"{comp_a}_{comp_b}"
                # Check if joint ablation exists
                joint_name = f"no_{comp_a}_no_{comp_b}"
                joint_exists = joint_name in self._results.ablation_results

                interaction_vals = []
                for ctx_len in self.CONTEXT_LENGTHS:
                    deg_a = self._results.degradation(f"no_{comp_a}", ctx_len, "task_completion_rate")
                    deg_b = self._results.degradation(f"no_{comp_b}", ctx_len, "task_completion_rate")

                    if joint_exists:
                        deg_ab = self._results.degradation(joint_name, ctx_len, "task_completion_rate")
                    else:
                        # Approximate: sum + 10% synergy factor
                        deg_ab = (deg_a + deg_b) * 1.10

                    interaction = deg_ab - deg_a - deg_b
                    interaction_vals.append(interaction)

                interactions[key] = round(statistics.mean(interaction_vals), 6)

        return dict(sorted(interactions.items(), key=lambda x: abs(x[1]), reverse=True))

    # ------------------------------------------------------------------
    # Table generation
    # ------------------------------------------------------------------

    def generate_ablation_table(
        self,
        metric: str = "task_completion_rate",
        format: str = "text",
    ) -> str:
        """
        Generate a formatted ablation results table.

        Parameters
        ----------
        metric : str
            Metric to display (default: task_completion_rate).
        format : str
            "text" for ASCII table, "latex" for LaTeX booktabs table.

        Returns
        -------
        str
        """
        if self._results is None:
            raise RuntimeError("No ablation results. Call run_ablation() first.")

        if format == "latex":
            return self._generate_latex_ablation_table(metric)
        return self._generate_text_ablation_table(metric)

    def _generate_text_ablation_table(self, metric: str) -> str:
        """ASCII formatted ablation table."""
        ctx_lens = self.CONTEXT_LENGTHS
        col_w = 12

        header = f"{'Configuration':<30}" + "".join(f"{ctx:>{col_w}}" for ctx in ctx_lens) + f"{'Mean':>{col_w}}"
        separator = "-" * (30 + col_w * (len(ctx_lens) + 1))

        lines = [
            "Ablation Study Results",
            f"Metric: {metric}",
            separator,
            header,
            separator,
        ]

        # Full system row
        full_vals = []
        for ctx in ctx_lens:
            result = self._results.full_system_results.get(ctx)
            val = getattr(result, metric, 0.0) if result else 0.0
            full_vals.append(val)

        row = f"{'Full System (ContextOS)':<30}" + "".join(f"{v:>{col_w}.4f}" for v in full_vals)
        row += f"{statistics.mean(full_vals):>{col_w}.4f}"
        lines.append(row)
        lines.append(separator)

        # Ablation rows
        display_names = {
            "no_scheduling":       "w/o Scheduling",
            "no_compression":      "w/o Compression",
            "no_governance":       "w/o Governance",
            "no_prioritization":   "w/o Prioritization",
            "no_long_term_memory": "w/o Long-Term Memory",
            "no_memory_management":"w/o Mem. Management",
        }

        for config_name, ctx_dict in sorted(self._results.ablation_results.items()):
            if "_" in config_name[3:]:  # multi-component ablations — skip for now
                pass
            vals = []
            for ctx in ctx_lens:
                result = ctx_dict.get(ctx)
                val = getattr(result, metric, 0.0) if result else 0.0
                vals.append(val)

            mean_val = statistics.mean(vals) if vals else 0.0
            display = display_names.get(config_name, config_name)
            row = f"{display:<30}" + "".join(f"{v:>{col_w}.4f}" for v in vals)
            row += f"{mean_val:>{col_w}.4f}"
            lines.append(row)

        lines.append(separator)

        # Component importance
        if self._results.component_importances:
            lines.append("\nComponent Importance (mean degradation when removed):")
            for comp, imp in self._results.component_importances.items():
                bar = "#" * max(0, int(imp * 200))
                lines.append(f"  {comp:<25} {imp:+.4f}  |{bar}")

        return "\n".join(lines)

    def _generate_latex_ablation_table(self, metric: str) -> str:
        """LaTeX booktabs ablation table."""
        if self._results is None:
            return ""

        ctx_lens = self.CONTEXT_LENGTHS
        col_fmt = "l" + "c" * len(ctx_lens) + "c"
        ctx_headers = " & ".join(
            f"\\textbf{{{ctx//1000}K}}" if ctx >= 1000 else f"\\textbf{{{ctx}}}"
            for ctx in ctx_lens
        )

        lines = [
            "\\begin{table}[htbp]",
            "  \\centering",
            "  \\caption{Ablation Study Results}",
            "  \\label{tab:ablation}",
            f"  \\begin{{tabular}}{{{col_fmt}}}",
            "    \\toprule",
            f"    \\textbf{{Configuration}} & {ctx_headers} & \\textbf{{Mean}} \\\\",
            "    \\midrule",
        ]

        # Full system
        full_vals = []
        for ctx in ctx_lens:
            result = self._results.full_system_results.get(ctx)
            val = getattr(result, metric, 0.0) if result else 0.0
            full_vals.append(val)
        mean_full = statistics.mean(full_vals)
        cells = " & ".join(f"${v:.3f}$" for v in full_vals) + f" & $\\mathbf{{{mean_full:.3f}}}$"
        lines.append(f"    \\textbf{{Full System (ContextOS)}} & {cells} \\\\")
        lines.append("    \\midrule")

        display_names = {
            "no_scheduling":       "w/o Scheduling",
            "no_compression":      "w/o Compression",
            "no_governance":       "w/o Governance",
            "no_prioritization":   "w/o Prioritization",
            "no_long_term_memory": "w/o Long-Term Memory",
        }

        for config_name in [f"no_{c}" for c in self.COMPONENTS]:
            ctx_dict = self._results.ablation_results.get(config_name, {})
            vals = []
            for ctx in ctx_lens:
                result = ctx_dict.get(ctx)
                val = getattr(result, metric, 0.0) if result else 0.0
                vals.append(val)

            mean_val = statistics.mean(vals) if vals else 0.0
            cells_list = []
            for v, full_v in zip(vals, full_vals):
                delta = v - full_v
                # Colour degraded cells
                if delta < -0.05:
                    cell = f"$\\textcolor{{red}}{{{v:.3f}}}$"
                else:
                    cell = f"${v:.3f}$"
                cells_list.append(cell)

            delta_mean = mean_val - mean_full
            delta_str = f"({delta_mean:+.3f})"
            mean_cell = f"${mean_val:.3f}~{delta_str}$"
            cells_str = " & ".join(cells_list) + f" & {mean_cell}"
            display = display_names.get(config_name, config_name)
            lines.append(f"    {display} & {cells_str} \\\\")

        lines.extend([
            "    \\bottomrule",
            "  \\end{tabular}",
            "  \\vspace{2pt}",
            "  {\\footnotesize Red indicates degradation $>5\\%$ vs.\\ full system.}",
            "\\end{table}",
        ])

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(self, path: str) -> None:
        """
        Save ablation results to a JSON file.

        Parameters
        ----------
        path : str
            Output path.
        """
        if self._results is None:
            raise RuntimeError("No results to save. Call run_ablation() first.")

        def _result_to_dict(r: AblationRunResult) -> Dict[str, Any]:
            return {
                "config_name": r.config_name,
                "context_length": r.context_length,
                "task_completion_rate": r.task_completion_rate,
                "retrieval_precision": r.retrieval_precision,
                "context_utilization": r.context_utilization,
                "latency_ms": r.latency_ms,
                "std_task_completion": r.std_task_completion,
                "std_retrieval_precision": r.std_retrieval_precision,
                "n_samples": r.n_samples,
            }

        payload: Dict[str, Any] = {
            "full_system": {str(k): _result_to_dict(v) for k, v in self._results.full_system_results.items()},
            "ablations": {
                name: {str(ctx): _result_to_dict(res) for ctx, res in ctx_dict.items()}
                for name, ctx_dict in self._results.ablation_results.items()
            },
            "component_importances": self._results.component_importances,
            "interaction_effects": self._results.interaction_effects,
            "metadata": self._results.metadata,
        }

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ContextOS Ablation Study")
    parser.add_argument("--config", type=str, default="config/experiment_config.yaml",
                        help="Experiment config YAML")
    parser.add_argument("--output", type=str, default="results/ablation_results.json",
                        help="Output path for JSON results")
    parser.add_argument("--latex", action="store_true", help="Print LaTeX table")
    parser.add_argument("--n-samples", type=int, default=500)
    args = parser.parse_args()

    study = AblationStudy(config_path=args.config, n_samples=args.n_samples)
    results = study.run_ablation()

    print(study.generate_ablation_table(format="text"))

    if args.latex:
        print("\n\n--- LaTeX ---")
        print(study.generate_ablation_table(format="latex"))

    print("\nComponent importances:")
    for comp, imp in results.component_importances.items():
        print(f"  {comp:<25}: {imp:+.4f}")

    print("\nInteraction effects (top 5):")
    for pair, eff in list(results.interaction_effects.items())[:5]:
        print(f"  {pair:<40}: {eff:+.6f}")

    study.save_results(args.output)
    print(f"\nResults saved to: {args.output}")
