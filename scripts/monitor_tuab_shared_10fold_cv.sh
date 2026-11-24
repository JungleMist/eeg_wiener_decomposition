#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
    printf 'Usage: bash scripts/monitor_tuab_shared_10fold_cv.sh RUN_DIR [--follow]\n'
    exit 0
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/monitor_tuep_shared_10fold_cv.sh" "$@"
