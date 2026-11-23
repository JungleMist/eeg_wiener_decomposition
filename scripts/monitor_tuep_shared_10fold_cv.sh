#!/usr/bin/env bash
# Print the persisted status and recent logs. --follow waits for state changes.
set -euo pipefail
if [[ $# -eq 0 || "$1" == "--help" || "$1" == "-h" ]]; then
    printf 'Usage: bash scripts/monitor_tuep_shared_10fold_cv.sh RUN_DIR [--follow]\n'
    exit 0
fi
if [[ $# -gt 2 || ( $# -eq 2 && "$2" != "--follow" ) ]]; then
    printf 'Expected RUN_DIR [--follow]\n' >&2
    exit 2
fi
output="$1"
last=""
while true; do
    if [[ -s "$output/status.json" ]]; then
        snapshot="$(cat -- "$output/status.json")"
        if [[ "$last" != "$snapshot" ]]; then
            printf '%s\n' "$snapshot"
            last="$snapshot"
        fi
        # The JSON is written by the pipeline and atomically replaced.
        state="$(sed -n 's/.*"state": "\([^"]*\)".*/\1/p' <<< "$snapshot")"
        pid="$(sed -n 's/.*"pid": \([0-9][0-9]*\).*/\1/p' <<< "$snapshot")"
        case "$state" in
            completed|paused) break ;;
            failed|interrupted) tail -n 20 "$output/run.log" 2>/dev/null || true; exit 1 ;;
        esac
        if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
            if [[ "$(cat -- "$output/status.json")" != "$snapshot" ]]; then
                continue
            fi
            printf 'Pipeline PID %s is no longer running\n' "$pid" >&2
            tail -n 20 "$output/run.log" 2>/dev/null || true
            exit 1
        fi
    elif [[ "${2:-}" != "--follow" ]]; then
        printf 'No pipeline status yet\n'
        break
    elif [[ -s "$output/launcher.pid" ]] &&
        ! kill -0 "$(cat -- "$output/launcher.pid")" 2>/dev/null; then
        printf 'Launcher exited before creating status\n' >&2
        tail -n 20 "$output/run.log" 2>/dev/null || true
        exit 1
    fi
    [[ "${2:-}" == "--follow" ]] || break
    sleep 5
done
if [[ -s "$output/run.log" ]]; then
    tail -n 15 "$output/run.log"
fi
