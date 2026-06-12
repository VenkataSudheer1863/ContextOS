"""
ContextBench Dataset Downloader and Validator
==============================================

Entrypoint for setting up all ContextBench dataset subsets locally.

Usage
-----
    # Check status, run missing generation scripts, then validate
    python datasets/download_datasets.py

    # Only check status (no generation)
    python datasets/download_datasets.py --check-only

    # Regenerate a specific subset even if it already exists
    python datasets/download_datasets.py --force --subset compression_bench

    # Skip JSON line validation (faster)
    python datasets/download_datasets.py --no-validate

    # Quiet output (summary table only)
    python datasets/download_datasets.py --quiet

Workflow
--------
1. Reads SUBSET_SPECS to determine the expected files and line-count bounds.
2. For each (subset, split) pair, checks whether the file exists and whether its
   line count and byte size fall within the expected bounds.
3. Runs the corresponding generate_<subset>.py script for any subset whose files
   are missing or invalid (unless --check-only is set).
4. Re-validates after generation.
5. Prints a formatted summary table and exits with code 0 on success or 1 if
   any subset remains invalid.

Dataset Files (all flat JSONL in datasets/ directory)
------------------------------------------------------
  context_bench_{train,val,test}.jsonl       60K / 5K / 5K examples
  retrieval_bench_{train,val,test}.jsonl     25K / 2.5K / 2.5K examples
  compression_bench_{train,val,test}.jsonl   25K / 2.5K / 2.5K examples
  embedding_finetune_{train,val,test}.jsonl  120K / 10K / 10K examples
  llm_finetune_{train,val,test}.jsonl        25K / 2.5K / 2.5K examples
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASETS_DIR = Path(__file__).parent
PROJECT_ROOT = DATASETS_DIR.parent


# ---------------------------------------------------------------------------
# Dataset specification
# ---------------------------------------------------------------------------

@dataclass
class SplitSpec:
    split: str
    expected_lines: int
    min_bytes: int    # lower bound — generation must produce at least this much
    max_bytes: int    # upper bound — safety check for runaway generation


@dataclass
class SubsetSpec:
    name: str
    generate_script: str  # filename relative to DATASETS_DIR
    splits: List[SplitSpec]
    description: str = ""

    def file_path(self, split: str) -> Path:
        """Return the expected JSONL path for this subset and split."""
        return DATASETS_DIR / f"{self.name}_{split}.jsonl"


# Size bounds are intentionally wide to accommodate variation in content length.
# context_bench: ~1.5 KB/example on average (10-50 items, templates)
# retrieval_bench: ~3 KB/example (50-100 corpus items)
# compression_bench: ~2 KB/example (200-1000 word passages)
# embedding_finetune: ~0.5 KB/example (anchor + positive + negative)
# llm_finetune: ~0.8 KB/example (instruction + serialized context + output)

SUBSET_SPECS: List[SubsetSpec] = [
    SubsetSpec(
        name="context_bench",
        generate_script="generate_context_bench.py",
        description="Multi-task context management (8 task types, 6 domains, 10-50 items/example)",
        splits=[
            SplitSpec("train", expected_lines=60000, min_bytes=30_000_000,  max_bytes=600_000_000),
            SplitSpec("val",   expected_lines=5000,  min_bytes=2_500_000,   max_bytes=50_000_000),
            SplitSpec("test",  expected_lines=5000,  min_bytes=2_500_000,   max_bytes=50_000_000),
        ],
    ),
    SubsetSpec(
        name="retrieval_bench",
        generate_script="generate_retrieval_bench.py",
        description="Retrieval evaluation (4 domains, easy/medium/hard difficulty, 50-100 corpus items/example)",
        splits=[
            SplitSpec("train", expected_lines=25000, min_bytes=25_000_000,  max_bytes=500_000_000),
            SplitSpec("val",   expected_lines=2500,  min_bytes=2_500_000,   max_bytes=50_000_000),
            SplitSpec("test",  expected_lines=2500,  min_bytes=2_500_000,   max_bytes=50_000_000),
        ],
    ),
    SubsetSpec(
        name="compression_bench",
        generate_script="generate_compression_bench.py",
        description="Compression fidelity (4 domains, 200-1000 word passages, ~40% reference compression)",
        splits=[
            SplitSpec("train", expected_lines=25000, min_bytes=20_000_000,  max_bytes=400_000_000),
            SplitSpec("val",   expected_lines=2500,  min_bytes=2_000_000,   max_bytes=40_000_000),
            SplitSpec("test",  expected_lines=2500,  min_bytes=2_000_000,   max_bytes=40_000_000),
        ],
    ),
    SubsetSpec(
        name="embedding_finetune",
        generate_script="generate_embedding_finetune.py",
        description="Embedding fine-tuning triplets (anchor, positive, hard-negative)",
        splits=[
            SplitSpec("train", expected_lines=120000, min_bytes=20_000_000, max_bytes=400_000_000),
            SplitSpec("val",   expected_lines=10000,  min_bytes=1_500_000,  max_bytes=35_000_000),
            SplitSpec("test",  expected_lines=10000,  min_bytes=1_500_000,  max_bytes=35_000_000),
        ],
    ),
    SubsetSpec(
        name="llm_finetune",
        generate_script="generate_llm_finetune.py",
        description="LLM instruction fine-tuning (Alpaca-style instruction/input/output triples)",
        splits=[
            SplitSpec("train", expected_lines=25000, min_bytes=10_000_000,  max_bytes=200_000_000),
            SplitSpec("val",   expected_lines=2500,  min_bytes=1_000_000,   max_bytes=20_000_000),
            SplitSpec("test",  expected_lines=2500,  min_bytes=1_000_000,   max_bytes=20_000_000),
        ],
    ),
]


# ---------------------------------------------------------------------------
# File inspection and validation
# ---------------------------------------------------------------------------

@dataclass
class FileStatus:
    subset: str
    split: str
    path: Path
    exists: bool
    line_count: int = 0
    byte_size: int = 0
    expected_lines: int = 0
    min_bytes: int = 0
    max_bytes: int = 0
    json_valid: bool = True
    json_error_line: Optional[int] = None
    errors: List[str] = field(default_factory=list)

    # Tolerance: allow +/- 2% deviation on line count (minimum tolerance: 5 lines)
    _LINE_TOLERANCE_PCT: float = 0.02

    @property
    def lines_ok(self) -> bool:
        if not self.exists:
            return False
        tol = max(5, int(self.expected_lines * self._LINE_TOLERANCE_PCT))
        return abs(self.line_count - self.expected_lines) <= tol

    @property
    def size_ok(self) -> bool:
        if not self.exists:
            return False
        return self.min_bytes <= self.byte_size <= self.max_bytes

    @property
    def ok(self) -> bool:
        return self.exists and self.lines_ok and self.size_ok and self.json_valid

    @property
    def status_label(self) -> str:
        if self.ok:
            return "OK"
        if not self.exists:
            return "MISSING"
        if not self.json_valid:
            return "JSON_ERR"
        if not self.lines_ok:
            return "LINE_ERR"
        if not self.size_ok:
            return "SIZE_ERR"
        return "INVALID"


def inspect_file(
    path: Path,
    spec: SplitSpec,
    subset_name: str,
    validate_json: bool = True,
) -> FileStatus:
    """Inspect a JSONL file and return its FileStatus."""
    status = FileStatus(
        subset=subset_name,
        split=spec.split,
        path=path,
        exists=path.exists(),
        expected_lines=spec.expected_lines,
        min_bytes=spec.min_bytes,
        max_bytes=spec.max_bytes,
    )

    if not status.exists:
        return status

    try:
        status.byte_size = path.stat().st_size
    except OSError as exc:
        status.errors.append(f"Cannot stat file: {exc}")
        return status

    line_count = 0
    json_error_line: Optional[int] = None

    try:
        with path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                line_count += 1
                if validate_json and json_error_line is None:
                    try:
                        json.loads(raw)
                    except json.JSONDecodeError as exc:
                        json_error_line = lineno
                        status.json_valid = False
                        status.json_error_line = lineno
                        status.errors.append(
                            f"JSON decode error at line {lineno}: {exc}"
                        )
    except OSError as exc:
        status.errors.append(f"Cannot read file: {exc}")
        return status

    status.line_count = line_count

    if not status.lines_ok:
        diff = status.line_count - status.expected_lines
        direction = "extra" if diff > 0 else "missing"
        status.errors.append(
            f"Line count {status.line_count:,} vs expected {status.expected_lines:,} "
            f"({abs(diff):,} {direction} lines; tolerance "
            f"+/-{max(5, int(status.expected_lines * FileStatus._LINE_TOLERANCE_PCT)):,})"
        )

    if not status.size_ok:
        if status.byte_size >= 1_000_000:
            human = f"{status.byte_size / 1_000_000:.1f} MB"
        elif status.byte_size >= 1_000:
            human = f"{status.byte_size / 1_000:.1f} KB"
        else:
            human = f"{status.byte_size} B"
        status.errors.append(
            f"File size {human} outside expected range "
            f"[{status.min_bytes / 1_000_000:.1f} MB, "
            f"{status.max_bytes / 1_000_000:.0f} MB]"
        )

    return status


def validate_all(
    specs: List[SubsetSpec],
    validate_json: bool = True,
    quiet: bool = False,
) -> Dict[str, Dict[str, FileStatus]]:
    """Validate all subsets. Returns nested dict {subset_name: {split: FileStatus}}."""
    results: Dict[str, Dict[str, FileStatus]] = {}

    for subset in specs:
        results[subset.name] = {}
        for split_spec in subset.splits:
            path = subset.file_path(split_spec.split)
            status = inspect_file(
                path, split_spec, subset.name, validate_json=validate_json
            )
            results[subset.name][split_spec.split] = status
            if not quiet and status.errors:
                for err in status.errors:
                    print(f"  [WARN] {subset.name}/{split_spec.split}: {err}")

    return results


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def run_generation_script(
    subset: SubsetSpec,
    quiet: bool = False,
) -> Tuple[bool, str]:
    """Run the generation script for *subset*. Returns (success, message)."""
    script_path = DATASETS_DIR / subset.generate_script
    if not script_path.exists():
        msg = f"Generation script not found: {script_path}"
        if not quiet:
            print(f"  [ERROR] {msg}")
        return False, msg

    if not quiet:
        print(f"  Running: python {script_path.name} ...", flush=True)

    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=quiet,
        text=True,
    )
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        msg = f"Script exited with code {result.returncode} after {elapsed:.1f}s"
        if not quiet:
            print(f"  [ERROR] {msg}")
            if result.stderr:
                # Print first 2000 chars of stderr for diagnostics
                print("  --- stderr ---")
                print(result.stderr[:2000])
                print("  --- end stderr ---")
        return False, msg

    msg = f"Completed in {elapsed:.1f}s"
    if not quiet:
        print(f"  [OK] {msg}", flush=True)
    return True, msg


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

_COL = {
    "subset": 20,
    "split":   6,
    "status":  9,
    "lines":  20,
    "size":   14,
    "json":    8,
}
_TOTAL_WIDTH = sum(_COL.values()) + len(_COL) * 3 - 1
_SEP = "-" * _TOTAL_WIDTH


def _fmt_lines(s: FileStatus) -> str:
    if not s.exists:
        return "--"
    mark = "" if s.lines_ok else " !"
    return f"{s.line_count:,} / {s.expected_lines:,}{mark}"


def _fmt_size(s: FileStatus) -> str:
    if not s.exists:
        return "--"
    b = s.byte_size
    if b >= 1_000_000_000:
        human = f"{b / 1_000_000_000:.2f} GB"
    elif b >= 1_000_000:
        human = f"{b / 1_000_000:.1f} MB"
    elif b >= 1_000:
        human = f"{b / 1_000:.1f} KB"
    else:
        human = f"{b} B"
    mark = "" if s.size_ok else " !"
    return f"{human}{mark}"


def _fmt_json(s: FileStatus) -> str:
    if not s.exists:
        return "--"
    return "OK" if s.json_valid else f"ERR@{s.json_error_line}"


def print_summary_table(results: Dict[str, Dict[str, FileStatus]]) -> None:
    header = (
        f"{'Subset':<{_COL['subset']}} | "
        f"{'Split':<{_COL['split']}} | "
        f"{'Status':<{_COL['status']}} | "
        f"{'Lines (actual/exp)':<{_COL['lines']}} | "
        f"{'Size':<{_COL['size']}} | "
        f"{'JSON':<{_COL['json']}}"
    )
    print()
    print("ContextBench — Dataset Validation Summary")
    print(_SEP)
    print(header)
    print(_SEP)

    all_ok = True
    total_ok = 0
    total_count = 0
    for subset_name, splits in results.items():
        for split_name, status in splits.items():
            total_count += 1
            if status.ok:
                total_ok += 1
            else:
                all_ok = False
            row = (
                f"{subset_name:<{_COL['subset']}} | "
                f"{split_name:<{_COL['split']}} | "
                f"{status.status_label:<{_COL['status']}} | "
                f"{_fmt_lines(status):<{_COL['lines']}} | "
                f"{_fmt_size(status):<{_COL['size']}} | "
                f"{_fmt_json(status):<{_COL['json']}}"
            )
            print(row)

    print(_SEP)
    overall = "ALL VALID" if all_ok else f"{total_ok}/{total_count} valid"
    print(f"Result: {overall}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _valid_subset_names() -> List[str]:
    return [s.name for s in SUBSET_SPECS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="download_datasets.py",
        description="Check, generate, and validate ContextBench dataset subsets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only report status; do not run any generation scripts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run generation scripts even for subsets that already appear valid.",
    )
    parser.add_argument(
        "--subset",
        nargs="+",
        metavar="SUBSET",
        help=(
            "Only process these subsets. "
            f"Valid names: {', '.join(_valid_subset_names())}."
        ),
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip per-line JSON validation (faster, but does not catch malformed records).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file progress messages; print only the summary table.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Filter specs to requested subsets
    active_specs = SUBSET_SPECS
    if args.subset:
        requested = set(args.subset)
        valid = set(_valid_subset_names())
        unknown = requested - valid
        if unknown:
            print(f"[ERROR] Unknown subset(s): {', '.join(sorted(unknown))}")
            print(f"        Valid names: {', '.join(sorted(valid))}")
            return 1
        active_specs = [s for s in SUBSET_SPECS if s.name in requested]

    validate_json = not args.no_validate

    # --- Initial check ---
    if not args.quiet:
        n = len(active_specs)
        print(f"\nContextBench — Checking {n} subset(s)...\n")

    results = validate_all(active_specs, validate_json=validate_json, quiet=args.quiet)

    if args.check_only:
        print_summary_table(results)
        all_ok = all(fs.ok for splits in results.values() for fs in splits.values())
        return 0 if all_ok else 1

    # --- Determine which subsets need generation ---
    needs_generation: List[SubsetSpec] = []
    for spec in active_specs:
        subset_ok = all(
            fs.ok for fs in results.get(spec.name, {}).values()
        )
        if args.force or not subset_ok:
            needs_generation.append(spec)

    if not needs_generation:
        if not args.quiet:
            print("All requested subsets are already valid. Nothing to generate.")
        print_summary_table(results)
        return 0

    # --- Generation pass ---
    if not args.quiet:
        print(f"Generating {len(needs_generation)} subset(s):\n")

    generation_failures: List[str] = []
    for spec in needs_generation:
        if not args.quiet:
            print(f"[{spec.name}] {spec.description}")
        success, msg = run_generation_script(spec, quiet=args.quiet)
        if not success:
            generation_failures.append(f"{spec.name}: {msg}")
        if not args.quiet:
            print()

    if generation_failures:
        print("[ERROR] The following generation scripts failed:")
        for failure in generation_failures:
            print(f"  - {failure}")
        print()

    # --- Post-generation validation ---
    if not args.quiet:
        print("Validating generated files...\n")

    results = validate_all(active_specs, validate_json=validate_json, quiet=args.quiet)
    print_summary_table(results)

    all_ok = all(fs.ok for splits in results.values() for fs in splits.values())
    if generation_failures or not all_ok:
        print(
            "[FAIL] One or more subsets failed validation. "
            "Review the errors above and re-run the relevant generation script manually."
        )
        return 1

    print("[PASS] All subsets are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
