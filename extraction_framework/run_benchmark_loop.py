"""
Repeat the full benchmark N times back-to-back.
Each pass is stored as a separate session in results/ and tagged with
run_number (1..N_RUNS) in the SQLite database for easy cross-run queries.

Change N_RUNS before starting a multi-run experiment.
"""
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from extraction_framework.benchmark_runner import BenchmarkRunner
from extraction_framework.run_benchmark import run

# ── CONFIGURE HERE ────────────────────────────────────────────────────────────
N_RUNS = 30
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    load_dotenv()
    BASE_DIR = Path(__file__).parent.parent
    RESULTS_DIR = BASE_DIR / "extraction_framework" / "results"

    # One shared runner so the text cache persists across all N runs
    runner = BenchmarkRunner(results_dir=RESULTS_DIR)

    print(f"Starting repeated benchmark: {N_RUNS} run(s)")
    print("=" * 60)

    for i in range(1, N_RUNS + 1):
        print(f"\n{'#' * 60}")
        print(f"  BENCHMARK RUN {i} / {N_RUNS}")
        print(f"{'#' * 60}")
        run(run_number=i, runner=runner)

    print(f"\n{'=' * 60}")
    print(f"ALL {N_RUNS} RUNS COMPLETE")
    print(f"Query results: SELECT * FROM extraction_results WHERE run_number BETWEEN 1 AND {N_RUNS}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
