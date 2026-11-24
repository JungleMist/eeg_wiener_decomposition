#!/usr/bin/env bash
# Launch or resume shared EEG cross-validation. Put --detach immediately after RUN_DIR.
set -euo pipefail

if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
    cat <<'EOF'
Usage: bash scripts/run_tuep_shared_10fold_cv.sh RUN_DIR [--detach] [pipeline options]
Common options: --data-dir PATH --cache-dir PATH --device auto|cpu|cuda --workers N
Use --dataset tuab for TUAB. The dataset defaults to TUEP.
For a pilot, add --max-model-folds N. Rerun with the same settings to resume.
The launcher appends to RUN_DIR/run.log. Set TUEP_CONDA_ENV to change eeg_pipeline.
EOF
    exit 0
fi

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p -- "$1"
output="$(cd -- "$1" && pwd -P)"
shift
if [[ "${1:-}" == "--detach" ]]; then
    shift
    nohup bash "$root/scripts/run_tuep_shared_10fold_cv.sh" "$output" "$@" \
        </dev/null >/dev/null 2>&1 &
    printf '%s\n' "$!" > "$output/launcher.pid"
    printf 'Launcher PID %s\nLog %s/run.log\n' "$!" "$output"
    exit 0
fi

exec >> "$output/run.log" 2>&1
command -v conda >/dev/null || { printf 'conda is required on PATH\n' >&2; exit 1; }
mkdir -p -- "$output/.runtime/numba" "$output/.runtime/matplotlib"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$output/.runtime/numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$output/.runtime/matplotlib}"
export MNE_DONTWRITE_HOME="${MNE_DONTWRITE_HOME:-true}"
printf '\n%s Starting shared EEG 10-fold pipeline\n' "$(date '+%Y-%m-%d %H:%M:%S %z')"
exec conda run --no-capture-output -n "${TUEP_CONDA_ENV:-eeg_pipeline}" \
    python -u "$root/scripts/run_tuep_shared_10fold_cv.py" "$@" --output-dir "$output"
