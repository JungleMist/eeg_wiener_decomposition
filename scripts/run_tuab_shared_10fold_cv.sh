#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
    cat <<'EOF'
Usage: bash scripts/run_tuab_shared_10fold_cv.sh RUN_DIR [--detach] [pipeline options]
Common options: --data-dir PATH --cache-dir PATH --device auto|cpu|cuda --workers N
For a pilot, add --max-model-folds N. Rerun with the same settings to resume.
The launcher appends to RUN_DIR/run.log.
Monitor with: bash scripts/monitor_tuab_shared_10fold_cv.sh RUN_DIR [--follow]
EOF
    exit 0
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/run_tuep_shared_10fold_cv.sh" "$@" --dataset tuab
