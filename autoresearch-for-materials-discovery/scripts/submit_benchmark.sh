#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
made_root="$(cd -- "$project_root/.." && pwd)"
: "${USER:?USER must be set to check the Slurm job limit}"

if [[ $# -gt 5 ]]; then
    printf 'Usage: %s [agent-config] [seed] [query-budget] [initial-q-values] [comma-separated-elements]\n' "$0" >&2
    exit 2
fi
if [[ $# -ge 5 ]]; then
    elements="$5"
    if [[ ! "$elements" =~ ^[A-Z][a-z]?(,[A-Z][a-z]?){1,4}$ ]]; then
        printf 'Element list must contain 2-5 comma-separated element symbols, e.g. Fe,Ni,Ti.\n' >&2
        exit 2
    fi
    IFS=, read -r -a element_symbols <<<"$elements"
    declare -A seen_elements=()
    for element in "${element_symbols[@]}"; do
        if [[ -n "${seen_elements[$element]:-}" ]]; then
            printf 'Element list contains duplicate symbol: %s\n' "$element" >&2
            exit 2
        fi
        seen_elements["$element"]=1
    done
fi

command -v sbatch >/dev/null
command -v squeue >/dev/null
command -v flock >/dev/null

mkdir -p "$project_root/results/slurm"
exec 9>"$project_root/results/.submit.lock"
if ! flock -n 9; then
    printf 'Another project submission check is in progress; retry shortly.\n' >&2
    exit 1
fi

active_jobs="$(squeue -h -u "$USER" -o '%i')"
active_count="$(printf '%s\n' "$active_jobs" | awk 'NF { count++ } END { print count+0 }')"
if (( active_count >= 1 )); then
    printf 'Refusing submission: %s job(s) are already pending or running; the cluster QOS currently allows one GPU job per user.\n' \
        "$active_count" >&2
    exit 1
fi

while IFS= read -r -d '' log_file; do
    gzip -1 -- "$log_file"
done < <(
    find "$project_root/results/slurm" -maxdepth 1 -type f \
        \( -name 'fars-*.out' -o -name 'fars-*.err' \) \
        ! -name '*.gz' -print0
)

export MADE_ROOT="$made_root"
export MADE_PROJECT_ROOT="$project_root"
cd "$made_root"
exec sbatch "$project_root/scripts/fars_benchmark.sbatch" "$@"
