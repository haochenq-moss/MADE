# FARS: RL-Scheduled Materials Discovery

**Individual project report**

I completed this project individually. I implemented and evaluated FARS, a
bounded tabular Q-learning scheduler layered over MADE's existing
ReAct-based materials-discovery agent, and a separate branch-search state-Q
method. This report summarizes completed 60-query runs on one development
system. Several methods have five runs, but some reuse seed labels; FARS
with Qwen has three runs. The results are exploratory and do not establish
generalization beyond this system.

## Abstract

I evaluated FARS, a Q-learning scheduler around MADE's ReAct agent, and
branch-search state-Q, which uses RandomGenerator and ORB without LLM calls.
Across five completed 60-query runs, branch-search had the largest mean
observed-hull mSUN@60 (25.2) and AUDC (862.6) among the compared methods.
These averages exceeded DeepSeek FARS (23.4; 744.1), FARS-MQ (22.6; 688.1),
and the two five-run generator baselines (8.6 and 8.8 mSUN@60). The
comparison is exploratory: run-to-run variation is substantial, several
groups reuse seed labels, and branch-search recorded no exact
ground-truth formula or structure matches. Therefore this is evidence of
more observed-hull discoveries in these runs, not proof of better confirmed
materials discovery.

The separate original-seed view tempers this result: on seeds 1–3,
branch-search averaged 21.00 mSUN@60 (26, 17, 20), below FARS-MQ's 22.67
(20, 21, 27) and above DeepSeek FARS's 18.67 (10, 13, 33). The five-run
available comparison therefore highlights a promising branch-search mean,
not a consistent win across every run selection.

## Introduction

Materials discovery is a sequential search problem: an agent proposes a
candidate, receives an evaluation, and uses the observation to decide what
to try next. MADE provides configurable agents, generators, oracles, and
closed-loop environments for studying this process [1, 2]. My objective was
to add a compact scheduler without replacing MADE's candidate-generation
workflow or changing its scientific oracle and evaluator.

## Method

I implemented FARS as a controller around MADE's existing
`LLMReActOrchestratorAgent`. The underlying ReAct agent remains responsible
for planning, tool selection, candidate generation, buffer operations,
optional scoring, and candidate selection. FARS chooses between two
macro-actions, `EXPLORE` and `REFINE`, and maintains a tabular Q-value for
the scheduler state and selected action. The LLM and scientific model are
not fine-tuned.

The controller reward is the signed change in the count of newly discovered
stable structures, recomputed against the *observed* phase diagram after a
query using MADE's hull utility. It does not read hidden labels or
ground-truth phase-diagram values. The custom variants cap generation at 32
candidates per proposal and 640 per episode, and cap distinct candidates
scored at 32 per proposal. MADE's unchanged runner performs the formal
queries and its unchanged evaluator computes post-run metrics. The project
runner explicitly disables adding ground-truth stable structures to the
agent's initial state.
The LLM-mediated reasoning-and-action pattern is related to ReAct [3];
FARS adds a separate tabular scheduler rather than fine-tuning the language
model. The Q-value update follows standard Q-learning [4].

The separate branch-search state-Q method tracks global observed yield and
stagnation together with each branch's depth, query count, and recent reward.
An epsilon-greedy tabular controller selects a feasible branch and an
`explore`, `expand`, or `exploit` action. The MADE planner and
`RandomGenerator` produce a candidate slate; the configured selector takes
the first candidate, and only that structure is evaluated by ORB. It uses
the binary observed reward `is_stable AND is_newly_discovered`, with
epsilon 0.1, learning rate 0.2, and discount 0.9. This branch-search
workflow makes no LLM or LLM API calls.

I have now added a separate, code-only method named **MADE-ReAct Branch
Search**. Its state-Q controller still chooses the branch and action; MADE's
planner and RandomGenerator create that branch's candidate slate, then MADE's
`LLMReActOrchestratorAgent` can inspect and select only from that slate using
selection tools. Candidate generation, branch constraints, evaluation, and
the observed binary Q reward remain controlled by the MADE branch-search
workflow. This variant has **not been run** and is not included in any
reported results or averages.

I reused MADE's tools, baseline implementations, environment, oracle, and
metrics. My contribution is the scheduler, its bounded wrappers,
project-owned configs and logs, and tests. The project branch
`autoresearch-materials-discovery` is based on local MADE revision
`bffda0c9a1fa1904b35c0ef67b455881c975df86`.

## Evaluation

I ran 60-query episodes on `Cd-Li-Nd-Ti-W`. Every run used the same local environment configuration,
including a 0.1 eV/atom stability tolerance, with both
`start_with_all_stable` and `include_near_stable_from_ground_truth` set to
false. I used the official MADE runner and the local ORB v3 conservative
oracle with FIRE relaxation. These are the settings in the selected local
MADE configuration; I do not present them as a separately confirmed
course-frozen configuration.
I ran with `experiment.infra=local`; my project wrapper disables only the
optional remote Modal volume commit while retaining local checkpoint writes.
The official runner and evaluator remain unchanged.

GPT-5.1 could not be used because its configured API account had no
remaining credits, and the configured Anthropic credential failed
authentication. With the student's approval, I used the local
`ollama_chat/qwen2.5-coder:7b` model for both FARS and its matched LLM-only
control. The random and Chemeleon arms use their official generator
baselines, not the LLM. All completed jobs ran on one NVIDIA A40 through
Slurm. For FARS and the matched control, I scaled the episode generation
cap to 1,920 candidates (32 per query); the per-query generation and scoring
caps remained 32. An initial random-baseline seed-2 attempt completed its
queries but failed while writing the final checkpoint because of a storage
quota. I compressed completed scheduler logs, retried that arm successfully,
and counted only the retry in the results.

I use MADE's final `novelty_stable_unique_novel_count` as the reported
stable-novel count at budget 60. AUDC is the official area under the
stable-discovery curve; I include the raw value and its normalized counterpart.
MADE's acceleration factor (AF) and enhancement factor (EF) are zero in
these outputs because no baseline history was supplied to the environment.
Those zeros are therefore unavailable comparisons, not evidence that the
methods are equally fast or effective.

Mean and standard deviation below are computed across completed runs
(sample standard deviation). The five-run groups include repeated campaigns
with reused seed labels for Random, Chemeleon, FARS (DeepSeek), and FARS-MQ
(DeepSeek); branch-search has five distinct seeds. FARS (Qwen) has three.

| Method | LLM / proposer | Runs | mSUN@60 (mean ± SD) | AUDC (mean ± SD) | nAUDC | Valid |
|---|---|---:|---:|---:|---:|---:|
| Random generator | None | 5 | 8.60 ± 2.88 | 326.70 ± 99.27 | 0.1815 | 85.7% |
| Chemeleon | None | 5 | 8.80 ± 2.86 | 280.20 ± 98.55 | 0.1557 | 99.3% |
| FARS | Qwen2.5-Coder 7B | 3 | 5.33 ± 2.08 | 192.33 ± 92.57 | 0.1069 | 100.0% |
| FARS | DeepSeek Flash | 5 | 23.40 ± 14.54 | 744.10 ± 437.74 | 0.4134 | 100.0% |
| FARS-MQ | DeepSeek Flash | 5 | 22.60 ± 2.88 | 688.10 ± 171.78 | 0.3823 | 100.0% |
| **Branch-search state-Q** | **None** | **5** | **25.20 ± 11.61** | **862.60 ± 372.86** | **0.4792** | 81.0% |

These are official observed-hull outcomes. nAUDC divides mean AUDC by 1,800.
MADE's EF/AF remain unavailable because the required baseline history was
not supplied. Per-run and summary values are in
[`six-methods-run-metrics.csv`](./results/comparison-budget-60/six-methods-run-metrics.csv)
and
[`six-methods-summary-metrics.csv`](./results/comparison-budget-60/six-methods-summary-metrics.csv).
The distinct-seed aggregates are in
[`six-methods-three-distinct-seed-summary.csv`](./results/comparison-budget-60/six-methods-three-distinct-seed-summary.csv).

## Results and discussion

Branch-search state-Q had the largest observed-hull mean mSUN@60 and AUDC
among these six arms, despite using no LLM calls. Its mean mSUN@60 exceeded
DeepSeek FARS by 1.80 and FARS-MQ by 2.60; their sample-SD ranges overlap
substantially, so these results do not establish a reliable statistical
advantage. It also exceeded the five-run Random and Chemeleon means, but
branch-search is a different search workflow, not a paired ablation of
LLM use. All five branch-search runs had zero exact ground-truth formula
or structure matches; mSUN counts stability on the observed hull and is
not a confirmed ground-truth discovery count. Branch-search's mean
distance-and-SMACT validity was 81.0%, lower than the other arms.

The five-run DeepSeek FARS and FARS-MQ results are descriptive repeat
campaign aggregates, not five independent seeds. Their comparison does not
isolate a causal effect of Q-table initialization. The three-run
Qwen-FARS result remains a separate model/configuration arm.

Each FARS seed made 60 controller decisions and recorded 59 post-query
transitions; the unchanged runner does not send a terminal feedback callback
after the final query. All 60 decision records per seed mark
`hidden_ground_truth_read: false`. Each saved Q-table now contains nonzero
values, though the terminal-update limitation remains. This demonstrates
that observed rewards updated the tabular
controller in these runs, but it does not establish convergence or
generalization.

The first random seed-2 job (46225) failed to save its final checkpoint
because of a disk-quota error; its retry (46227) completed and is the
reported original seed-2 result. All five Chemeleon runs completed.

## Limitations and conclusion

This is a one-system development experiment. I did not run
held-out systems or the prepared FARS ablations, and I make no
generalization claim. Several five-run groups reuse seed labels, and the
three-seed view is small. The local 0.1 eV/atom
tolerance and the local LLM are evaluation deviations that should be
considered before comparing these numbers with results from another
protocol. The AF and EF outputs also need a separately supplied baseline
history before they can support relative comparisons.

The branch-search result is promising: its five-run mean was 25.2
observed-hull stable discoveries with no LLM calls. However, the five-run
sample was variable and recorded no exact ground-truth matches. I therefore
claim an advantage in the reported observed-hull metrics only, not a
confirmed improvement in true materials discovery. A stronger follow-up
would use predeclared unique seeds, matched search workflows and candidate
budgets, more systems, and the official baseline history needed for EF/AF.

### Continued training from pooled Q-values

To test whether prior FARS learning could initialize a new campaign, I
computed the arithmetic mean of the three budget-60 FARS Q-tables,
treating missing state/action entries as zero, and used that table to
initialize a fourth 60-query training episode (Slurm job 47098). The
campaign made 60 decisions (33 `EXPLORE`, 27 `REFINE`) and completed with
0 stable, unique, novel discoveries and AUDC 0. Its 59 observed
transitions all had zero immediate signed discovery reward. This run
therefore shows that the warm-started policy executed, but this episode
supplied no nonzero immediate discovery reward and did not improve the
measured outcome. The saved Q-table has 23 nonzero entries, compared with
19 in the pooled initializer; because all immediate rewards were zero,
this change can arise from bootstrapping previously positive values and
should not be interpreted as new discovery evidence.

The run loaded and updated the pooled table through the same path, so the
pool file was overwritten during training. I reconstructed the original
pooled initializer from the three untouched seed Q-tables and preserved
the post-run table separately. For future runs, the launcher and agent
now accept a distinct `initial_q_values_path`; online updates are saved
to that new run's controller log, leaving the input Q-table unchanged.
The seed-4 result is separate from the original three-run Qwen FARS
evaluation and is not included in its averages. The [continued-run manifest](./results/continued-fars/job-manifest.csv)
links the initializer and saved post-run Q-table.

### FARS-MQ variant

I named the variant that initializes FARS from a merged/mean Q-table
**FARS-MQ** (FARS with Mean-Q initialization). It uses the user-supplied
table in `results/mean-q-transfer/initial_q_values.json` and continues
online learning; each run writes its updated values separately. The table
contains 25 states and 50 state-action values. The attachment did not include
its source runs or aggregation metadata, so I have not independently verified
that it is an arithmetic mean of the earlier three seed tables. FARS-MQ is a
method configuration, not a demonstrated performance improvement; the
one-query smoke result (Slurm job 47169) is only an execution check and is not
comparable to the budget-60 experiments. It completed one official query with
one distance/SMACT-invalid candidate and no stable novel discovery. The
controller trace confirms that the supplied Q-table was selected as the
initializer; this single outcome does not evaluate the method's performance.

## Expanded six-method comparison and side-by-side pipelines

The branch-search state-Q method tracks global observed yield and stagnation
together with each branch's depth, query count, and recent reward. A
state-conditioned epsilon-greedy Q-learner selects a feasible branch and
`explore`, `expand`, or `exploit` action. When expanding, it opens an
available first-element composition-fraction region. MADE's planner and
RandomGenerator create a candidate slate; the configured selector takes
the first candidate, and only that structure is evaluated by ORB. The
binary reward is one only when the observed structure is stable and newly
discovered. It makes no LLM or LLM API calls.

The paired illustration contrasts FARS with the new, unrun MADE-ReAct
Branch Search method. Purple highlights the MADE-specific ReAct selection
step, and each feedback path is shown as one straight arrow. The original
branch-search results above are from the LLM-free method, not this new
variant.

![FARS and MADE-ReAct Branch Search pipelines side by side; the new MADE-specific step is purple and feedback arrows are straight](./results/comparison-budget-60/fars-and-branch-search-side-by-side-pipelines.png)

The expanded comparison includes five completed runs for Random,
Chemeleon, DeepSeek FARS, DeepSeek FARS-MQ, and branch-search state-Q.
Qwen FARS has three runs. Random, Chemeleon, and the DeepSeek FARS groups
combine original seeds 1–3 with two additional runs that reuse seed labels
1 and 2; branch-search uses five distinct seeds. Means and sample standard
deviations are descriptive, not confidence intervals.
I report both views: all available completed campaigns (with reused seed
labels disclosed) and the separate seeds-1–3 plot as a distinct-seed
comparison. The five campaigns are not five independent seeds, so the
five-run mean should be read descriptively rather than as an independent
replication estimate.

| Seed | mSUN@60 | Official AUDC |
|---:|---:|---:|
| 1 | 26 | 764.0 |
| 2 | 17 | 468.5 |
| 3 | 20 | 636.0 |
| 4 | 45 | 1,423.5 |
| 5 | 18 | 1,021.0 |
| Mean ± sample SD | **25.20 ± 11.61** | **862.60 ± 372.86** |

![Six-method 60-query comparison using all available five-run groups](./results/comparison-budget-60/cumulative-discovery-curve.png)

The five-run plot includes the two generator baselines and Qwen FARS as
requested. FARS with Qwen has three runs; the other five groups have five.
Branch-search had the highest mean observed-hull mSUN@60 (25.2) and AUDC
(862.6), compared with DeepSeek FARS (23.4; 744.1), FARS-MQ (22.6; 688.1),
Random (8.6; 326.7), Chemeleon (8.8; 280.2), and Qwen FARS (5.33; 192.33).
Branch-search makes no LLM calls, but it uses a different search workflow,
and its sample standard deviation is substantial. This supports a promising
observed-hull result, not a confirmed causal or ground-truth advantage.

![Six-method discovery-curve comparison using only three distinct seeds per method](./results/comparison-budget-60/cumulative-discovery-curve-three-distinct-seeds.png)

This second plot includes only the original seeds 1–3 for every method and
excludes repeat campaigns. Across all five branch-search runs, no exact
ground-truth formula or structure matches were recorded. mSUN therefore
counts stability on the observed hull, not confirmed ground-truth
discoveries. Run-level values are in
[`six-methods-run-metrics.csv`](./results/comparison-budget-60/six-methods-run-metrics.csv);
curve means and standard deviations for both plot sets are in
[`six-methods-discovery-curves.csv`](./results/comparison-budget-60/six-methods-discovery-curves.csv).

## References

1. *MADE: Benchmark Environments for Closed-Loop Materials Discovery*.
   arXiv:2601.20996, 2026. [Paper](https://arxiv.org/abs/2601.20996).
2. MADE project repository and benchmark implementation.
   [Official repository](https://github.com/diffractivelabs/MADE).
3. S. Yao et al., “ReAct: Synergizing Reasoning and Acting in Language
   Models,” ICLR 2023. [Paper](https://arxiv.org/abs/2210.03629).
4. C. J. C. H. Watkins and P. Dayan, “Q-learning,” *Machine Learning*,
   8, 279–292 (1992). [https://doi.org/10.1007/BF00992698](https://doi.org/10.1007/BF00992698).
