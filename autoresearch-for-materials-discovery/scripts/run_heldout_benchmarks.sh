#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
elements="${1:-}"
budget="${2:-60}"
seeds_csv="${3:-1,2,3}"
methods_csv="${4:-fars_train,llm_orchestrator_control,random_generator_baseline,chemeleon_generative_baseline,fars_meanq_train}"

usage() {
    printf 'Usage: %s ELEMENTS [QUERY_BUDGET] [SEEDS_CSV] [METHODS_CSV]\n' "$0" >&2
    printf 'Example: %s Fe,Ni,Ti 60 1,2,3\n' "$0" >&2
}

if (( $# > 4 )) ||
    [[ ! "$elements" =~ ^[A-Z][a-z]?(,[A-Z][a-z]?){1,4}$ ||
      ! "$budget" =~ ^[1-9][0-9]*$ ]] || (( budget > 360 )); then
    usage
    exit 2
fi
if (( $# >= 4 )) && [[ -z "$4" ]]; then
    printf 'Method list cannot be empty.\n' >&2
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

IFS=, read -r -a methods <<<"$methods_csv"
if (( ${#methods[@]} == 0 )); then
    printf 'At least one method is required.\n' >&2
    usage
    exit 2
fi
allowed_methods=(
    fars_train
    fars_meanq_train
    llm_orchestrator_control
    random_generator_baseline
    chemeleon_generative_baseline
)
declare -A seen_methods=()
for method in "${methods[@]}"; do
    allowed=false
    for allowed_method in "${allowed_methods[@]}"; do
        if [[ "$method" == "$allowed_method" ]]; then
            allowed=true
            break
        fi
    done
    if [[ "$allowed" != true || -n "${seen_methods[$method]:-}" ]]; then
        printf 'Unsupported or duplicate method: %s\n' "$method" >&2
        usage
        exit 2
    fi
    seen_methods["$method"]=1
done

contains_llm=false
for method in "${methods[@]}"; do
    if [[ "$method" == fars_train || "$method" == fars_meanq_train ||
          "$method" == llm_orchestrator_control ]]; then
        contains_llm=true
    fi
done
if [[ "$contains_llm" == true ]]; then
    export FARS_LLM_MODEL="${HELDOUT_LLM_MODEL:-ollama_chat/qwen2.5-coder:7b}"
fi

manifest_dir="$project_root/results/heldout-${elements//,/-}-budget-${budget}"
manifest="$manifest_dir/job-manifest.csv"
mkdir -p "$manifest_dir"
if [[ ! -e "$manifest" ]]; then
    printf 'elements,method,seed,budget,job_id\n' >"$manifest"
fi

has_completed_result() {
    local requested_method="$1"
    local requested_seed="$2"
    local recorded_elements recorded_method recorded_seed recorded_budget job_id state result_dir

    while IFS=, read -r recorded_elements recorded_method recorded_seed recorded_budget job_id; do
        if [[ "$recorded_elements" != "$elements" ||
              "$recorded_method" != "$requested_method" ||
              "$recorded_seed" != "$requested_seed" ||
              "$recorded_budget" != "$budget" ||
              ! "$job_id" =~ ^[0-9]+$ ]]; then
            continue
        fi

        state="$(sacct -n -X -j "$job_id" -o State | awk 'NF { print $1; exit }')"
        result_dir="$project_root/results/${requested_method}-system-${elements//,/-}-seed-${requested_seed}-job-${job_id}"
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
        if has_completed_result "$method" "$seed"; then
            printf 'Already completed system=%s method=%s seed=%s budget=%s; skipping.\n' \
                "$elements" "$method" "$seed" "$budget"
            continue
        fi

        submission="$("$script_dir/submit_benchmark.sh" \
            "$method" "$seed" "$budget" "" "$elements")"
        if [[ "$submission" =~ Submitted[[:space:]]batch[[:space:]]job[[:space:]]([0-9]+) ]]; then
            job_id="${BASH_REMATCH[1]}"
        else
            printf 'Could not parse Slurm submission response: %s\n' "$submission" >&2
            exit 1
        fi

        printf '%s,%s,%s,%s,%s\n' "$elements" "$method" "$seed" "$budget" "$job_id" >>"$manifest"
        printf 'Submitted held-out system=%s method=%s seed=%s budget=%s job=%s\n' \
            "$elements" "$method" "$seed" "$budget" "$job_id"
        "$script_dir/watch_slurm_job.sh" "$job_id"
    done
done

printf 'Held-out runs completed. Manifest: %s\n' "$manifest"
