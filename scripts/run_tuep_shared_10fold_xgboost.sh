#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
    printf 'Usage: bash scripts/%s RUN_DIR [--detach] [pipeline options]\n' "${0##*/}"
    printf '%s\n' 'Runs XGBoost only. Use a separate RUN_DIR from the two-model experiment.'
    printf '%s\n' 'Common options: --data-dir PATH --cache-dir PATH --device auto|cpu|cuda --workers N'
    exit 0
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/run_tuep_shared_10fold_cv.sh" "$@" --dataset tuep --models xgboost
