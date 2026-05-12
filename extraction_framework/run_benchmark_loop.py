"""
Repeat the full benchmark N times back-to-back.
Each pass is stored as a separate session in results/ and tagged with
run_number (1..N_RUNS) in the SQLite database for easy cross-run queries.

Usage:
    python run_benchmark_loop.py            # uses default N_RUNS = 30
    python run_benchmark_loop.py --runs 5   # override N_RUNS from CLI
"""
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from extraction_framework.benchmark_runner import BenchmarkRunner
from extraction_framework.run_benchmark import run

# ── CONFIGURE HERE ────────────────────────────────────────────────────────────
DEFAULT_N_RUNS = 37
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the benchmark N times back-to-back.")
    parser.add_argument("--runs", type=int, default=DEFAULT_N_RUNS,
                        help=f"Number of benchmark passes (default: {DEFAULT_N_RUNS})")
    args = parser.parse_args()
    n_runs = args.runs

    load_dotenv()
    BASE_DIR = Path(__file__).parent.parent
    RESULTS_DIR = BASE_DIR / "extraction_framework" / "results"

    # One shared runner so the text cache persists across all N runs
    runner = BenchmarkRunner(results_dir=RESULTS_DIR)

    print(f"Starting repeated benchmark: {n_runs} run(s)")
    print("=" * 60)

    for i in range(1, n_runs + 1):
        print(f"\n{'#' * 60}")
        print(f"  BENCHMARK RUN {i} / {n_runs}")
        print(f"{'#' * 60}")
        run(run_number=i, runner=runner)

    print(f"\n{'=' * 60}")
    print(f"ALL {n_runs} RUNS COMPLETE")
    print(f"Query results: SELECT * FROM extraction_results WHERE run_number BETWEEN 1 AND {n_runs}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
