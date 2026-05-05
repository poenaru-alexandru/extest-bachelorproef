#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run.sh  —  Start a benchmark in the background so SSH can be closed safely.
# MAKE SCRIPT EXECUTABLE:  chmod +x run.sh
#
# Usage:
#   ./run.sh                        # single pass (run_benchmark.py)
#   ./run.sh loop                   # 30 passes (default N_RUNS)
#   ./run.sh loop --runs 10         # 10 passes
#
# Logs are written to:  logs/benchmark_YYYYMMDD_HHMMSS.log
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

MODE="${1:-single}"
shift || true   # remaining args forwarded to the python script

if [[ "$MODE" == "loop" ]]; then
    PYTHON_SCRIPT="$SCRIPT_DIR/run_benchmark_loop.py"
    LOG_FILE="$LOG_DIR/benchmark_loop_${TIMESTAMP}.log"
else
    PYTHON_SCRIPT="$SCRIPT_DIR/run_benchmark.py"
    LOG_FILE="$LOG_DIR/benchmark_single_${TIMESTAMP}.log"
fi

echo "Starting benchmark (mode: $MODE) — logging to:"
echo "  $LOG_FILE"
echo ""

# nohup keeps the process alive after SSH disconnects.
# stdbuf -oL forces line-buffered stdout so progress is written immediately.
nohup stdbuf -oL python "$PYTHON_SCRIPT" "$@" >> "$LOG_FILE" 2>&1 &
PID=$!

echo "Process started with PID $PID"
echo "To follow live:  tail -f $LOG_FILE"
echo "To check status: ps -p $PID"
echo "To stop:         kill $PID"
