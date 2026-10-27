#!/usr/bin/env bash

set -euo pipefail

usage() {
    printf 'Usage: %s JOB_ID [interval-seconds]\n' "$0" >&2
}

if [[ $# -lt 1 || $# -gt 2 || ! "$1" =~ ^[0-9]+$ ]]; then
    usage
    exit 2
fi

job_id="$1"
interval="${2:-60}"
if [[ ! "$interval" =~ ^[0-9]+$ ]] || (( interval < 1 )); then
    usage
    exit 2
fi

command -v squeue >/dev/null
command -v sacct >/dev/null

while true; do
    if ! queue_state="$(squeue -h -j "$job_id" -o '%T')"; then
        printf 'Unable to query squeue for job %s.\n' "$job_id" >&2
        exit 1
    fi

    if [[ -n "$queue_state" ]]; then
        printf '%s job=%s state=%s\n' "$(date --iso-8601=seconds)" "$job_id" "$queue_state"
        sleep "$interval"
        continue
    fi

    if ! accounting="$(sacct -n -X -j "$job_id" -o State,ExitCode)"; then
        printf 'Unable to query sacct for job %s.\n' "$job_id" >&2
        exit 1
    fi
    record="$(printf '%s\n' "$accounting" | awk 'NF { print; exit }')"
    if [[ -z "$record" ]]; then
        printf '%s job=%s state=not-yet-visible-in-accounting\n' \
            "$(date --iso-8601=seconds)" "$job_id"
        sleep "$interval"
        continue
    fi

    read -r state exit_code _ <<<"$record"
    printf '%s job=%s state=%s exit_code=%s\n' \
        "$(date --iso-8601=seconds)" "$job_id" "$state" "$exit_code"
    case "$state" in
        COMPLETED) exit 0 ;;
        PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|REQUEUED|RESIZING)
            sleep "$interval"
            ;;
        *) exit 1 ;;
    esac
done
