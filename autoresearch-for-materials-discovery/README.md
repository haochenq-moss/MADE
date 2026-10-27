# FARS: RL-Scheduled Auto Research for Materials Discovery

This project adds a small tabular Q-learning scheduler above MADE's existing
GPT-5.1 ReAct orchestrator. The LLM and the upstream tool loop still perform
planning, conditional generation, buffer operations, optional surrogate
scoring, and final candidate selection. The controller only selects between
`EXPLORE` and `REFINE`; it does not replace or fine-tune a scientific model.

All project code, configs, tests, and generated results belong in this directory.
The MADE benchmark runner, environment, evaluator, oracle, and required baseline
configs are not modified.

## Verified starting point

- The project branch starts from the current local MADE `main` commit,
  `bffda0c9a1fa1904b35c0ef67b455881c975df86`. Per the student's instruction,
  local MADE rollover changes are accepted as the project base; no separate
  teaching-team source SHA is treated as a blocker.
- The existing MADE virtual environment reports Python 3.12.14, DSPy 3.0.4,
  and Hydra 1.3.2. No packages were installed or dependency manifests changed.
- The configured GPT-5.1 API account has no remaining credits, and the
  configured Anthropic credential returned an authentication error. The
  available local Ollama model `qwen2.5-coder:7b` passed a DSPy ReAct tool-use
  smoke check. Runs using it must be reported as an LLM-provider/model
  deviation from the original GPT-5.1 design; FARS and the matched LLM control
  are configured to use the same local model for a fairer comparison.
- Approved development system supplied by the student: `Cd-Li-Nd-Ti-W`.
- Instructor approval for local RL scheduler training was supplied by the student.
- The project branch is `autoresearch-materials-discovery`. It starts from the
  current local MADE `main` HEAD; pre-existing MADE-root worktree changes were
  left uncommitted and untouched.
- Three budget-60 repeats (seeds 1-3) completed for FARS, a matched LLM-only
  control, and the official random-generator and Chemeleon baselines on the
  development system with Qwen for the custom LLM methods. A later DeepSeek
  campaign completed three seeds each for FARS, the matched LLM-only control,
  and FARS-MQ. The current comparison plot and evaluation matrix are generated
  by `scripts/analyze_budget60.py` under
  `results/comparison-budget-60/`. The Qwen and DeepSeek results come from
  separate model campaigns and do not establish a causal model comparison or
  held-out performance. See [report.tex](./report.tex) for the updated results.
- An interactive dashboard for saved discovery trajectories, model/API usage,
  and research-resource statistics is available at
  `results/dashboard/index.html`. Rebuild it after new runs from the MADE root:
  ```bash
  PYTHONPATH="$PWD/autoresearch-for-materials-discovery/src:$PWD/src" \
    .venv/bin/python autoresearch-for-materials-discovery/scripts/build_experiment_dashboard.py
  ```
  The dashboard distinguishes oracle queries from LLM API calls and tokens;
  dollar costs are unavailable because billing data was not recorded.
- Workspace Slurm examples and the live scheduler confirm partition
  `MGPU-TC2`, one `gpu:nvidia` GPU, and a six-hour time limit. The project batch
  template follows the workspace's 4-CPU/32-GB pattern and omits account/QOS
  directives, as the examples do.
- The submission helper allows one pending/running job at a time. An initial
  two-job attempt showed the cluster QOS `QOSMaxGRESPerUser`, so experiments
  must run sequentially despite the two-job user limit. The status watcher
  polls Slurm accounting every 60 seconds by default until the job reaches a
  terminal state.
- Slurm runs use the official local runner with local checkpoint persistence;
  the project wrapper disables only the optional remote Modal volume commit.
  Checkpoints and temporary trajectories stay under the project `results/`
  directory. Each job starts its own Ollama server on a job-specific local
  port, avoiding accidental connections to another process on a shared node.
- Local MADE configuration currently specifies ORB v3 conservative INF OMAT,
  FIRE relaxation (500 steps, `fmax=0.02`), 0.1 eV/atom stability tolerance,
  and `start_with_all_stable: true`. These local values are not assumed to be
  course-frozen settings.

## Safety and measurement

- The policy code does not access `ground_truth_pd`, `phase_diagram_gt`, or
  dataset labels directly. It consumes only `env.get_state()` and the
  orchestrator's candidate buffer. The course-frozen initialization must still
  be confirmed: the local environment config currently starts with all stable
  structures, so the exact initial information visible to an agent is a
  course-protocol question, not something this project silently changes.
- The project Slurm runner explicitly overrides MADE's local
  `start_with_all_stable` and `include_near_stable_from_ground_truth` defaults
  to `false`. This prevents ground-truth structures from being included in the
  agent-visible initial phase diagram; the unchanged official evaluator still
  computes its metrics internally after candidate queries.
- The controller reward is the signed change in the official observed-hull
  count of novel stable structures. It is recomputed from the environment's
  observed phase diagram using MADE's `safe_e_above_hull` utility. Signed changes
  are retained because a later hull update can change a candidate's stability.
- Direct `create_structure` calls are disabled so generation is bounded. The
  custom variants share a maximum of 32 generated candidates and 32 surrogate
  scores per proposal, with an episode generation cap of 640 by default.
- A formal candidate query is still performed by MADE's unchanged runner and
  counts once per structure. Batching or surrogate scoring does not add formal
  oracle queries. Surrogate evaluations are a separate cost.
- `controller_memory` affects scheduler features only. The parent ReAct agent's
  official evaluation history and environment state remain intact in all
  controller-memory ablations.
- Online training updates arrive on the next policy call. The unchanged runner
  does not send a terminal feedback callback after the last query, so the final
  transition of each episode is not applied to the online Q table. Its outcome
  remains in the official raw trajectory. Do not describe this online schedule
  as a complete terminal-transition learner.
- The local random baseline's `RandomGenerator` owns a separate PRNG; seed it
  explicitly using the supported config field as shown below. The local
  Chemeleon generator exposes no seed parameter in its constructor, so do not
  claim bitwise reproducibility for Chemeleon runs without verifying its
  stochastic sampling path.
- FARS logs action, observed-state provenance, and (for training) completed
  signed-reward transitions to `controller-logs/`. The official trajectory
  remains the source of query outcomes. Slurm jobs record provider-reported
  LLM token usage, when available, in
  `results/token-usage/job-<job-id>.jsonl`; prompts, completions, and
  credentials are not logged. Cache-hit tokens are reported separately from
  non-cached API totals. These records do not estimate dollar cost, and usage
  from jobs started before token tracking was enabled cannot be recovered from
  their saved trajectories.

## Project configuration and commands

### Verified interface map

| Interface | Callable signature / behavior |
|---|---|
| `scripts/run_benchmark.py` | `run_episode(config, episode_id=0, wandb_run_name="benchmark", system_id=None)` instantiates the Hydra agent and calls `agent(state)` before `env.step(structure)`. |
| `src/made/agents/base.py` | `Agent.propose_composition_and_structure(state) -> tuple[Composition, Structure]`; `Agent.step(state)` calls `update_state(state)` then the proposal method. Checkpoint contract: `get_state()`, `load_state(state)`, and `update_state(state)`. |
| `src/made/agents/llm_react_orchestrator.py` | `LLMReActOrchestratorAgent.propose_composition_and_structure(state) -> tuple[Composition, Structure]`; existing tools include `generate_structures(generator_name, compositions="", num_candidates=10)`, `score_buffer(scorer_name, composition="")`, `list_compositions(k=10, mode="top", scorer_name="")`, `query_structures(composition, k=5, mode="top", scorer_name="", include_structure_details=False)`, `get_buffer_stats()`, and `select_for_evaluation(composition, structure_index=0, scorer_name="", reason="")`. |
| `src/made/envs/convex_hull.py` | `ConvexHullEnvironment.__init__(dataset, oracle, budget, start_with_all_stable=False, compute_elemental_from_oracle=True, stability_tolerance=1e-8, include_near_stable_from_ground_truth=False, structure_matcher_ltol=0.2, structure_matcher_stol=0.3, structure_matcher_angle_tol=5, structure_matcher_primitive_cell=True, filter_by_smact=False, max_stoichiometry=None)`; `step(proposal)` returns a post-query observation and done flag; `get_state()` returns the observed environment state. |
| `src/made/evaluation/metrics.py` | `DiscoveryCurveMetrics.calculate(metrics_history, *, baseline_metrics_history=None, metric_key="num_newly_discovered_structures", query_key="queries_used")`; official mSUN-related metric histories use `num_newly_discovered_stable` in the environment. |

The runner supports custom importable `_target_` classes through Hydra config
composition; no upstream policy registry change is needed. Custom policy source
is added to `PYTHONPATH`, and custom Hydra groups are made available with
`--config-dir`.

Run commands from the MADE repository root. They use the official
`scripts/run_benchmark.py` and the project-owned Hydra config search path.
`MADE_ROOT` points Hydra at the existing official configs; `MADE_PROJECT_ROOT`
keeps results inside this project folder.

```bash
export MADE_ROOT="$PWD"
export MADE_PROJECT_ROOT="$PWD/autoresearch-for-materials-discovery"
export PYTHONPATH="$MADE_PROJECT_ROOT/src:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

The approved development-system dry-run (configuration composition only; no
dataset fetch, model load, oracle query, or GPU work):

```bash
.venv/bin/python scripts/run_benchmark.py \
  --config-dir "$MADE_PROJECT_ROOT/configs" --cfg job agent=fars \
  dataset.elements='[Cd,Li,Nd,Ti,W]' environment.budget=20
```

The official runner commands, with course evaluation settings to be confirmed,
are:

The GPU batch template runs one official benchmark config in the existing MADE
virtual environment and writes output under this project. Submit each arm
sequentially (the site permits only one per-user GPU job at a time):

```bash
autoresearch-for-materials-discovery/scripts/submit_benchmark.sh fars_train 1
autoresearch-for-materials-discovery/scripts/watch_slurm_job.sh JOB_ID
autoresearch-for-materials-discovery/scripts/submit_benchmark.sh llm_orchestrator_control 1
autoresearch-for-materials-discovery/scripts/watch_slurm_job.sh JOB_ID
autoresearch-for-materials-discovery/scripts/submit_benchmark.sh random_generator_baseline 1
autoresearch-for-materials-discovery/scripts/watch_slurm_job.sh JOB_ID
autoresearch-for-materials-discovery/scripts/submit_benchmark.sh chemeleon_generative_baseline 1
```

The positional arguments are the agent config name, seed, and query budget;
defaults are `fars`, `1`, and `20`. The template uses the approved development
system. The helper checks the current user job count before submission; do not
bypass it with direct `sbatch` calls. After submission, monitor a job with:

```bash
autoresearch-for-materials-discovery/scripts/watch_slurm_job.sh JOB_ID
```

The watcher checks once per minute by default; pass a positive number of
seconds as the second argument to change the interval. The batch template
checks the allocated GPU, loads provider credentials from the MADE `.env`
without printing them, and explicitly disables initialization from hidden
ground-truth stable structures. It also installs a project-local Monty
deserialization alias for the pymatgen module path used by current Materials
Project API responses. Review each job's Hydra config and logs under `results/`
before interpreting or reporting its results.

To run the four evaluated arms (FARS online training, matched LLM-only control,
random-generator baseline, and Chemeleon baseline) with seeds 1, 2, and 3,
sequentially at budget 60:

```bash
bash autoresearch-for-materials-discovery/scripts/run_repeated_benchmarks.sh 60
```

The sequence uses only one active job at a time and records submitted job IDs
in `results/repeated-budget-60/job-manifest.csv`. For FARS and its matched
control, the episode generation cap scales with budget to 32 candidates per
query (1,920 total at budget 60); the 32-per-query generation and scoring caps
remain in place. All arms retain the prior system, seed-matched configuration,
safe initialization overrides, local model selection, and local checkpoint
behavior. The repeated-run script is resumable: it skips an arm only when its
manifested Slurm job completed and the expected summary and trajectory files
exist.

### ReAct branch-search selector

The experimental `branch_search_react` agent learns branch/action values from
observed binary discovery rewards. MADE's grid planner and `RandomGenerator`
build a bounded slate inside a first-element fraction region; MADE's ReAct
agent can select only from that slate. It receives no generation tools or
hidden ground-truth labels. The runner explicitly disables ground-truth
initialization and retains MADE's unchanged oracle and evaluator.

Submit one 60-query run with the previously used DeepSeek Flash model:

```bash
FARS_LLM_MODEL=deepseek/deepseek-flash \
  autoresearch-for-materials-discovery/scripts/submit_benchmark.sh \
  branch_search_react 1 60 "" Cd,Li,Nd,Ti,W
```

The selected seed, model, controller decisions, observed rewards, and Q-values
are recorded in the run's project-local Hydra directory.

### Held-out system evaluation

Use only a system list approved for held-out evaluation. The runner does not
choose systems or inspect hidden ground-truth labels. It accepts element
symbols explicitly, uses the unchanged MADE benchmark runner and evaluator,
and disables hidden-ground-truth initialization as in the development runs.

```bash
bash autoresearch-for-materials-discovery/scripts/run_heldout_benchmarks.sh \
  Fe,Ni,Ti 60 1,2,3
```

Arguments are element symbols, oracle-query budget, seed list, and an optional
method list. By default, it runs FARS, the matched LLM-only control, Random,
Chemeleon, and FARS-MQ sequentially. For the LLM-based methods, the default model is the locally installed
Ollama model `qwen2.5-coder:7b`. Each Slurm job starts Ollama on its allocated
node and uses `ollama_chat/qwen2.5-coder:7b`; no OpenAI API credentials are
needed. Override the model with `HELDOUT_LLM_MODEL=provider/model` only when
that provider is configured and available to the batch job. The runner
records job IDs in `results/heldout-<system>-budget-<budget>/job-manifest.csv`
and writes each run into a system-specific result directory.

To run only FARS for three seeds:

```bash
bash autoresearch-for-materials-discovery/scripts/run_heldout_benchmarks.sh \
  Fe,Ni,Ti 60 1,2,3 fars_train
```

Based on the completed development-system timings, one system with three
seeds is estimated at about 13.0 GPU-hours for all five methods. This includes
the three recent FARS-MQ runs; it excludes queue delays. The estimate can vary
with system size, model latency, and candidate-generation workload. The setup
does not submit or run held-out experiments by itself.

### Continuing FARS from previous runs

FARS can initialize its tabular scheduler from a prior `q_values.json` with
`agent.initial_q_values_path=...` while `train_scheduler=true`. It loads the
Q-values as the starting policy and continues online updates from new
post-query observations; it does not replay the previous trajectories.
`agent.q_values_path` remains the destination for the new run's updated table.
To pool independent seed runs, `scripts/average_q_values.py` averages
corresponding state/action values and treats missing entries as zero. The
warm-started campaign and its source table are recorded in
`results/continued-fars/`.

The named variant **FARS-MQ** (FARS with Mean-Q initialization) loads the
project's supplied merged table in
`results/mean-q-transfer/initial_q_values.json` by default, then continues
online Q-learning while writing updates to that run's own controller log.
At each scheduler decision, the controller log records whether the state key
matched the initializer exactly and cumulative coverage, including unmatched
keys.
FARS does not translate a key with a different feature value into a nearby
state. In each of the three prior budget-60 runs, only 3 of 60 decisions
matched an initializer key exactly. The initializer mostly described a `large`
candidate buffer, while warm-start decisions usually had an `empty` buffer.
This is a state-distribution mismatch, not evidence that these keys are
interchangeable.
The attached table's arithmetic aggregation and source runs were not included
with the attachment, so its aggregation provenance is not independently
verified. FARS-MQ names the initialization approach, not evidence that it
improves discovery performance. To reproduce it for one query:

```bash
autoresearch-for-materials-discovery/scripts/submit_benchmark.sh \
  fars_meanq_train 6 1
```

An explicit fourth argument overrides the default initializer. The original
input table is never used as the output path.

To make a paired fresh-start versus mean-Q comparison, use the same
development system, budget, and seed list for both arms:

```bash
bash autoresearch-for-materials-discovery/scripts/run_paired_fars_comparison.sh \
  Cd,Li,Nd,Ti,W 60 1,2,3
```

For each seed, the runner completes `fars_train` and then
`fars_meanq_train`, using the same model, episode settings, action epsilon,
generation/scoring caps, safe initialization flags, and seed. Only the
initializer differs. Job IDs and initializer path are recorded in a
pair-specific manifest under `results/paired-fars-*/`. The initializer remains
read-only and separate from each run's `controller-logs/q_values.json`.
Candidate proposal logs include selected compositions, generated/scored
counts, state matches, and the post-query observation/reward events. Do not
interpret the comparison as paired evidence until both arms complete for every
listed seed.

To test a formula-balancing candidate-selection tie-break against current FARS:

```bash
bash autoresearch-for-materials-discovery/scripts/run_paired_candidate_selection.sh \
  Cd,Li,Nd,Ti,W 60 1,2,3 deepseek/deepseek-flash
```

The paired runner executes `fars_train` and `fars_balanced_train` sequentially
for each seed, using the same model, system, budget, scheduler settings, and
generation/scoring caps. The treatment only prompts the ReAct agent to mildly
prefer less-queried formulas among comparable available candidates; it does not
block previously queried formulas or alter candidate generation, scoring, the
oracle, or evaluation. The agent logs the selection policy and selected
formula for each query. Job IDs and source/config hashes are recorded in a
pair-specific manifest. This is an exploratory three-seed comparison; keep
the default `llm` selection policy unless results support a follow-up
experiment.

In EXPLORE mode, FARS now makes previously queried compositions unavailable
for selection when the filter-approved buffer contains an unqueried
composition. Generation requests also prioritize unqueried formulas. If no
unqueried buffered or proposed composition is available, it falls back to the
existing composition rather than blocking the MADE ReAct loop. This behavior was evaluated in the three-seed DeepSeek budget-60 campaign on
one development system; those results do not by themselves establish that
composition-diverse selection improves discovery. REFINE behavior, candidate
filters, oracle, evaluator, and generation/scoring limits are unchanged.

Project-owned comparison configs are:

| Config | Purpose / caveat |
|---|---|
| `fars` | RL scheduler with full controller memory; training disabled by default. |
| `fars_train` | Enables scheduler-only online Q updates; base LLM remains fixed. |
| `llm_orchestrator_control` | ReAct LLM only, no RL scheduler, with the same generation/scoring caps as FARS. |
| `fars_fixed` | Predeclared 3:1 EXPLORE/REFINE schedule, independent of outcomes. |
| `fars_random_action` | Seeded random choice among feasible macro-actions. |
| `fars_no_memory` | Removes controller history features, not ReAct/environment memory. |
| `fars_no_scorer` | Removes surrogate-scoring tool access from the LLM tool list. |
| `fars_diversity_only` | Fixed EXPLORE with random generation and no scorer tool. |
| `fars_surrogate_greedy` | Fixed REFINE with explicit surrogate-ranking instructions; report separately. |
| `fars_exploit_diagnostic` | Fixed REFINE with no recovery instruction; a diagnostic, not evidence by itself. |

Do not claim the two required failure cases unless the predeclared development
trajectories satisfy the project's criteria. Threshold 0.01 eV/atom is not
configured; it requires explicit course approval and confirmation that the
official runner supports it as intended.

## Tests

From the MADE root, after the project and MADE dependencies are available:

```bash
PYTHONPATH="$PWD/autoresearch-for-materials-discovery/src:$PWD/src" \
  .venv/bin/python -m pytest autoresearch-for-materials-discovery/tests -q
```

Tests use fake tools only and are not scientific results.

## Paper-to-code evidence

| MADE v2 evidence | Local code/config | Status |
|---|---|---|
| §4.2.4: ReAct uses evaluation history, an unevaluated candidate buffer, tools, and feedback. | `src/made/agents/llm_react_orchestrator.py`, `configs/agent/llm_react_orchestrator.yaml` | Reused unchanged as the base; FARS adds scheduler prompts and bounded generator wrappers. |
| Appendix B.4.3: MACE-MP-0-medium ranks 1,024 cross-composition candidates in the separate MLIP pipeline. | `configs/agent/chemeleon_mlip_ranking_chain_filter.yaml`, `configs/agent/scorer/oracle.yaml`, `configs/oracle/mace.yaml` | Verified local config; not imposed on every policy. |
| Appendix C.5: removing MLIP scoring improved reported orchestrator metrics; removing generator tools reduced results; structural context mattered. | `configs/agent/fars_no_scorer.yaml` and the original orchestrator context config | Ablation config prepared; no local results yet. |
| §3.3 and Appendix C.5: mSUN is novel, unique and stable; the observed hull changes over a campaign. | `src/made/envs/convex_hull.py` (`_compute_metrics`, `step`); `src/made/utils/convex_hull_utils.py` (`safe_e_above_hull`) | Signed reward is recomputed against only the observed hull; no ground-truth metric is consumed by the controller. |

The uploaded paper's ablation table uses 0.1 eV/atom and budget 50; those are
paper settings, not authorization to change the course's fixed evaluation
conditions. Development and held-out system lists and any course-specific
evaluation overrides must come from the teaching team.

## Report outline

Use the supplied template, write in first person, and state that I completed the
project individually. Keep the main report within five pages:

1. **Abstract** — research question, FARS design, observed results only after runs.
2. **Introduction** — closed-loop discovery and the project objective.
3. **Method and novelty** — reused ReAct orchestration, FARS scheduler, memory,
   tools, budget accounting, and what is original versus inherited from MADE.
4. **Evaluation** — project branch and evaluation settings, approved development and held-out
   systems, seeds, official baselines, variants, metrics, and costs.
5. **Results and discussion** — leaderboard (mSUN@B, AUDC, AF, EF), query-based
   discovery curves, ablations, development/held-out comparison, and limitations.
   Include two genuine failure trajectories only if the predeclared criteria are
   met; otherwise report the evidence gap and propose controlled diagnostics.
6. **Conclusion and references** — findings and cited MADE/paper sources.
