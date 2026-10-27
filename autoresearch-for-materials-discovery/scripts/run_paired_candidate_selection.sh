#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
elements="${1:-Cd,Li,Nd,Ti,W}"
budget="${2:-60}"
seeds_csv="${3:-1,2,3}"
llm_model="${4:-deepseek/deepseek-flash}"

usage() {
    printf 'Usage: %s [ELEMENTS] [QUERY_BUDGET] [SEEDS_CSV] [LLM_MODEL]\n' "$0" >&2
    printf 'Example: %s Cd,Li,Nd,Ti,W 60 1,2,3 deepseek/deepseek-flash\n' "$0" >&2
}

if (( $# > 4 )) ||
    [[ ! "$elements" =~ ^[A-Z][a-z]?(,[A-Z][a-z]?){1,4}$ ||
       ! "$budget" =~ ^[1-9][0-9]*$ ]] ||
    (( budget > 360 )) ||
    [[ -z "$llm_model" ]]; then
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

manifest_dir="$project_root/results/paired-selection-${elements//,/-}-budget-${budget}"
manifest="$manifest_dir/job-manifest.csv"
mkdir -p "$manifest_dir"
if [[ ! -e "$manifest" ]]; then
    printf 'elements,budget,seed,arm,selection_policy,llm_model,agent_sha256,config_sha256,job_id\n' \
        >"$manifest"
fi

export FARS_LLM_MODEL="$llm_model"
agent_sha256="$(sha256sum "$project_root/src/autoresearch_for_materials_discovery/agent.py" | awk '{print $1}')"

has_completed_run() {
    local requested_arm="$1"
    local requested_seed="$2"
    local recorded_elements recorded_budget recorded_seed recorded_arm
    local selection_policy recorded_model recorded_agent_hash recorded_config_hash
    local job_id state result_dir

    while IFS=, read -r recorded_elements recorded_budget recorded_seed \
        recorded_arm selection_policy recorded_model recorded_agent_hash \
        recorded_config_hash job_id; do
        if [[ "$recorded_elements" != "$elements" ||
              "$recorded_budget" != "$budget" ||
              "$recorded_seed" != "$requested_seed" ||
              "$recorded_arm" != "$requested_arm" ||
              ! "$job_id" =~ ^[0-9]+$ ]]; then
            continue
        fi

        state="$(sacct -n -X -j "$job_id" -o State | awk 'NF { print $1; exit }')"
        result_dir="$project_root/results/${requested_arm}-system-${elements//,/-}-seed-${requested_seed}-job-${job_id}"
        if [[ "$state" == COMPLETED &&
              -s "$result_dir/summary/episodes.csv" &&
              -s "$result_dir/trajectories/episode_000.json" ]]; then
            return 0
        fi
    done < <(tail -n +2 "$manifest")

    return 1
}

submit_and_wait() {
    local arm="$1"
    local policy="$2"
    local seed="$3"
    local config_sha256
    local submission job_id
    config_sha256="$(sha256sum "$project_root/configs/agent/${arm}.yaml" | awk '{print $1}')"

    submission="$("$script_dir/submit_benchmark.sh" \
        "$arm" "$seed" "$budget" "" "$elements")"
    if [[ "$submission" =~ Submitted[[:space:]]batch[[:space:]]job[[:space:]]([0-9]+) ]]; then
        job_id="${BASH_REMATCH[1]}"
    else
        printf 'Could not parse Slurm submission response: %s\n' "$submission" >&2
        exit 1
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$elements" "$budget" "$seed" "$arm" "$policy" "$llm_model" \
        "$agent_sha256" "$config_sha256" "$job_id" >>"$manifest"
    printf 'Submitted paired arm=%s seed=%s policy=%s job=%s\n' \
        "$arm" "$seed" "$policy" "$job_id"
    "$script_dir/watch_slurm_job.sh" "$job_id"
}

for seed in "${seeds[@]}"; do
    if has_completed_run fars_train "$seed"; then
        printf 'Already completed baseline seed=%s; skipping.\n' "$seed"
    else
        submit_and_wait fars_train llm "$seed"
    fi
    if has_completed_run fars_balanced_train "$seed"; then
        printf 'Already completed balanced-selection seed=%s; skipping.\n' "$seed"
    else
        submit_and_wait fars_balanced_train balanced_formula "$seed"
    fi
done

printf 'Paired candidate-selection experiment complete. Manifest: %s\n' "$manifest"
