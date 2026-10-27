#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"

if [[ $# -ne 1 || ! "$1" =~ ^[1-9][0-9]*$ ]] || (( ${1:-0} > 360 )); then
    printf 'Usage: %s QUERY_BUDGET (1-360)\n' "$0" >&2
    exit 2
fi

budget="$1"
methods=(
    fars_train
    llm_orchestrator_control
    random_generator_baseline
    chemeleon_generative_baseline
)
seeds=(1 2 3)
manifest_dir="$project_root/results/repeated-budget-$budget"
manifest="$manifest_dir/job-manifest.csv"
mkdir -p "$manifest_dir"
if [[ ! -e "$manifest" ]]; then
    printf 'method,seed,budget,job_id\n' >"$manifest"
fi

has_completed_result() {
    local requested_method="$1"
    local requested_seed="$2"
    local requested_budget="$3"
    local recorded_method recorded_seed recorded_budget job_id state result_dir

    while IFS=, read -r recorded_method recorded_seed recorded_budget job_id; do
        if [[ "$recorded_method" != "$requested_method" ||
              "$recorded_seed" != "$requested_seed" ||
              "$recorded_budget" != "$requested_budget" ||
              ! "$job_id" =~ ^[0-9]+$ ]]; then
            continue
        fi

        state="$(sacct -n -X -j "$job_id" -o State | awk 'NF { print $1; exit }')"
        result_dir="$project_root/results/${requested_method}-seed-${requested_seed}-job-${job_id}"
        if [[ "$state" == COMPLETED &&
              -s "$result_dir/summary/episodes.csv" &&
              -s "$result_dir/trajectories/episode_000.json" ]]; then
            return 0
        fi
    done < <(tail -n +2 "$manifest")

    return 1
}

for method in "${methods[@]}"; do
    for seed in "${seeds[@]}"; do
        if has_completed_result "$method" "$seed" "$budget"; then
            printf 'Already completed method=%s seed=%s budget=%s; skipping.\n' \
                "$method" "$seed" "$budget"
            continue
        fi

        submission="$("$script_dir/submit_benchmark.sh" "$method" "$seed" "$budget")"
        if [[ "$submission" =~ Submitted[[:space:]]batch[[:space:]]job[[:space:]]([0-9]+) ]]; then
            job_id="${BASH_REMATCH[1]}"
        else
            printf 'Could not parse Slurm submission response: %s\n' "$submission" >&2
            exit 1
        fi

        printf '%s,%s,%s,%s\n' "$method" "$seed" "$budget" "$job_id" >>"$manifest"
        printf 'Submitted method=%s seed=%s budget=%s job=%s\n' \
            "$method" "$seed" "$budget" "$job_id"
        "$script_dir/watch_slurm_job.sh" "$job_id"
    done
done

printf 'All requested runs completed. Job manifest: %s\n' "$manifest"
