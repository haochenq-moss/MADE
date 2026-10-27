#!/usr/bin/env bash
#SBATCH --job-name=fars
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:nvidia:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=autoresearch-for-materials-discovery/results/slurm/fars-%j.out
#SBATCH --error=autoresearch-for-materials-discovery/results/slurm/fars-%j.err

set -euo pipefail

usage() {
    printf 'Usage: sbatch fars_benchmark.sbatch [agent-config] [seed] [query-budget] [initial-q-values] [comma-separated-elements]\n' >&2
    printf 'Submit through scripts/submit_benchmark.sh to respect the cluster GPU limit.\n' >&2
}

if [[ $# -gt 5 ]]; then
    usage
    exit 2
fi

agent="${1:-fars}"
seed="${2:-1}"
budget="${3:-20}"
initial_q_values="${4:-}"
elements="${5:-Cd,Li,Nd,Ti,W}"
if [[ ! "$agent" =~ ^[a-zA-Z0-9_-]+$ ||
      ! "$seed" =~ ^[0-9]+$ ||
      ! "$budget" =~ ^[1-9][0-9]*$ ]] ||
    (( budget > 360 )); then
    usage
    exit 2
fi
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
made_root="${MADE_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the MADE repository root}}"
project_root="${MADE_PROJECT_ROOT:-$made_root/autoresearch-for-materials-discovery}"
if [[ "$agent" == "fars_meanq_train" && -z "$initial_q_values" ]]; then
    initial_q_values="$project_root/results/mean-q-transfer/initial_q_values.json"
fi
if [[ -n "$initial_q_values" ]]; then
    if [[ "$agent" != fars* || ! -s "$initial_q_values" ]]; then
        printf 'An initial Q-table must be a non-empty file and can only be used with a FARS config.\n' >&2
        exit 2
    fi
    initial_q_values="$(realpath -- "$initial_q_values")"
fi

python_executable="${MADE_PYTHON:-$made_root/.venv/bin/python}"
if [[ $# -ge 5 ]]; then
    system_label="${elements//,/-}"
    run_dir="$project_root/results/${agent}-system-${system_label}-seed-${seed}-job-${SLURM_JOB_ID:?Run through Slurm}"
else
    run_dir="$project_root/results/${agent}-seed-${seed}-job-${SLURM_JOB_ID:?Run through Slurm}"
fi
runner_work_dir="$project_root/results/runner-workdir/job-$SLURM_JOB_ID"
chemeleon_checkpoint_dir="$project_root/results/runner-workdir/shared-chemeleon/ckpts"

if [[ ! -f "$made_root/scripts/run_benchmark.py" ||
      ! -d "$project_root/configs/agent" ||
      ! -x "$python_executable" ]]; then
    printf 'MADE root, project config directory, or executable Python environment is missing.\n' >&2
    exit 2
fi

export PYTHONPATH="$project_root/src:$made_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export FARS_TOKEN_USAGE_LOG="$project_root/results/token-usage/job-${SLURM_JOB_ID:?}.jsonl"
ollama_bin="${OLLAMA_BIN:-$HOME/.local/bin/ollama}"
ollama_model="${FARS_OLLAMA_MODEL:-qwen2.5-coder:7b}"
llm_model="${FARS_LLM_MODEL:-ollama_chat/$ollama_model}"
export FARS_LLM_MODEL="$llm_model"
export FARS_LOCAL_CHECKPOINTS=1

mkdir -p "$runner_work_dir"
if [[ -e "$runner_work_dir/data" || -L "$runner_work_dir/data" ]]; then
    printf 'Runner work directory already contains a data path: %s\n' "$runner_work_dir/data" >&2
    exit 2
fi
ln -s "$made_root/data" "$runner_work_dir/data"
if [[ -d "$chemeleon_checkpoint_dir" ]]; then
    if [[ -e "$runner_work_dir/ckpts" || -L "$runner_work_dir/ckpts" ]]; then
        printf 'Runner work directory already contains a checkpoint path: %s\n' "$runner_work_dir/ckpts" >&2
        exit 2
    fi
    ln -s "$chemeleon_checkpoint_dir" "$runner_work_dir/ckpts"
fi
cd "$runner_work_dir"
printf 'Slurm job=%s host=%s agent=%s seed=%s python=%s\n' \
    "$SLURM_JOB_ID" "$(hostname)" "$agent" "$seed" "$python_executable"

"$python_executable" -c '
import sys
import torch
print("Python:", sys.version)
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable on the allocated node.")
if torch.cuda.device_count() != 1:
    raise SystemExit("Expected exactly one Slurm-visible GPU.")
print("GPU:", torch.cuda.get_device_name(0))
'

extra_overrides=()
if [[ "$agent" == "random_generator_baseline" ]]; then
    extra_overrides+=( "++agent.generator.base_generator.seed=$seed" )
fi
if [[ "$agent" == fars* || "$agent" == "llm_orchestrator_control" ||
      "$agent" == "branch_search_react" ]]; then
    extra_overrides+=( "agent.max_generated_per_episode=$((32 * budget))" )
fi
if [[ "$agent" == "branch_search_react" ]]; then
    extra_overrides+=( "++agent.generators.random.seed=$seed" )
fi
if [[ -n "$initial_q_values" ]]; then
    extra_overrides+=( "agent.initial_q_values_path=$initial_q_values" )
fi

llm_overrides=()
needs_llm=false
case "$agent" in
    fars*|llm_orchestrator_control|branch_search_react)
        needs_llm=true
        llm_overrides+=( "agent.llm_config.model=$llm_model" )
        if [[ "$agent" != "branch_search_react" ]]; then
            llm_overrides+=( "agent.scorers.llm.llm_config.model=$llm_model" )
        fi
        ;;
esac

if [[ "$needs_llm" == true ]] &&
    [[ "$llm_model" == ollama_chat/* || "$llm_model" == ollama/* ]]; then
    if [[ ! -x "$ollama_bin" ]]; then
        printf 'Ollama executable is missing: %s\n' "$ollama_bin" >&2
        exit 2
    fi
    ollama_model="${llm_model#*/}"
    ollama_port=$((20000 + SLURM_JOB_ID % 30000))
    export OLLAMA_HOST="127.0.0.1:$ollama_port"
    "$ollama_bin" serve >"$project_root/results/slurm/ollama-${SLURM_JOB_ID}.log" 2>&1 &
    ollama_pid=$!
    trap 'kill "$ollama_pid" 2>/dev/null || true; wait "$ollama_pid" 2>/dev/null || true' EXIT

    ollama_ready=false
    for _ in $(seq 1 30); do
        if ! kill -0 "$ollama_pid" 2>/dev/null; then
            printf 'The job-local Ollama server exited before becoming ready; see ollama-%s.log.\n' \
                "$SLURM_JOB_ID" >&2
            exit 1
        fi
        if curl --silent --fail --max-time 2 "http://$OLLAMA_HOST/api/tags" >/dev/null; then
            ollama_ready=true
            break
        fi
        sleep 2
    done
    if [[ "$ollama_ready" != true ]]; then
        printf 'Ollama did not become ready on the allocated node at %s.\n' "$OLLAMA_HOST" >&2
        exit 1
    fi
    "$ollama_bin" show "$ollama_model" >/dev/null
    printf 'LLM provider=Ollama model=%s; checkpoints are project-local.\n' "$ollama_model"
fi

"$python_executable" "$project_root/scripts/run_benchmark_with_env.py" \
    "$made_root/.env" \
    "$made_root/scripts/run_benchmark.py" \
    --config-dir "$project_root/configs" \
    agent="$agent" \
    "dataset.elements=[$elements]" \
    "environment.budget=$budget" \
    environment.start_with_all_stable=false \
    environment.include_near_stable_from_ground_truth=false \
    experiment.infra=local \
    experiment.num_episodes=1 \
    "experiment.seed=$seed" \
    "${extra_overrides[@]}" \
    "${llm_overrides[@]}" \
    "hydra.run.dir=$run_dir"
