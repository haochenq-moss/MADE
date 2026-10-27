#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
elements="${1:-Cd,Li,Nd,Ti,W}"
budget="${2:-60}"
seeds_csv="${3:-1,2,3}"
initializer="$project_root/results/mean-q-transfer/initial_q_values.json"

usage() {
    printf 'Usage: %s [ELEMENTS] [QUERY_BUDGET] [SEEDS_CSV]\n' "$0" >&2
    printf 'Example: %s Cd,Li,Nd,Ti,W 60 1,2,3\n' "$0" >&2
}

if (( $# > 3 )) ||
    [[ ! "$elements" =~ ^[A-Z][a-z]?(,[A-Z][a-z]?){1,4}$ ||
       ! "$budget" =~ ^[1-9][0-9]*$ ]] ||
    (( budget > 360 )) ||
    [[ ! -s "$initializer" ]]; then
    usage
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

IFS=, read -r -a seeds <<<"$seeds_csv"
if (( ${#seeds[@]} == 0 )); then
    usage
    exit 2
fi
declare -A seen_seeds=()
for seed in "${seeds[@]}"; do
    if [[ ! "$seed" =~ ^[0-9]+$ || -n "${seen_seeds[$seed]:-}" ]]; then
        printf 'Seeds must be unique non-negative integers.\n' >&2
        exit 2
    fi
    seen_seeds["$seed"]=1
done

manifest_dir="$project_root/results/paired-fars-${elements//,/-}-budget-${budget}"
manifest="$manifest_dir/job-manifest.csv"
mkdir -p "$manifest_dir"
if [[ ! -e "$manifest" ]]; then
    printf 'elements,budget,seed,arm,initializer,initializer_sha256,job_id\n' >"$manifest"
fi
initializer_sha256="$(sha256sum "$initializer" | awk '{print $1}')"

export FARS_LLM_MODEL="${PAIRED_FARS_LLM_MODEL:-${FARS_LLM_MODEL:-ollama_chat/qwen2.5-coder:7b}}"

submit_and_wait() {
    local arm="$1"
    local seed="$2"
    local submission job_id initializer_record initializer_hash

    submission="$("$script_dir/submit_benchmark.sh" \
        "$arm" "$seed" "$budget" "" "$elements")"
    if [[ "$submission" =~ Submitted[[:space:]]batch[[:space:]]job[[:space:]]([0-9]+) ]]; then
        job_id="${BASH_REMATCH[1]}"
    else
        printf 'Could not parse Slurm submission response: %s\n' "$submission" >&2
        exit 1
    fi

    initializer_record="none"
    initializer_hash="none"
    if [[ "$arm" == "fars_meanq_train" ]]; then
        initializer_record="$initializer"
        initializer_hash="$initializer_sha256"
    fi
    printf '%s,%s,%s,%s,%s,%s,%s\n' \
        "$elements" "$budget" "$seed" "$arm" "$initializer_record" \
        "$initializer_hash" "$job_id" \
        >>"$manifest"
    printf 'Submitted paired arm=%s seed=%s budget=%s job=%s\n' \
        "$arm" "$seed" "$budget" "$job_id"
    "$script_dir/watch_slurm_job.sh" "$job_id"
}

for seed in "${seeds[@]}"; do
    submit_and_wait fars_train "$seed"
    submit_and_wait fars_meanq_train "$seed"
done

printf 'Paired FARS comparison complete. Manifest: %s\n' "$manifest"
