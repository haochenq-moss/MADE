#!/usr/bin/env python3
"""Build budget-60 discovery curves, representative trajectories, and cost tables."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition
from pymatgen.core.structure import Structure

from made.utils.convex_hull_utils import safe_e_above_hull
from made.utils.structure_hash import structure_hash

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
OUTPUT = RESULTS / "comparison-budget-60"
BUDGET = 60

RUNS = {
    "FARS": [
        (1, 46040),
        (2, 46112),
        (3, 46166),
    ],
    "Matched LLM-only control": [
        (1, 46186),
        (2, 46199),
        (3, 46210),
    ],
    "Random-generator baseline": [
        (1, 46222),
        (2, 46227),
        (3, 46310),
    ],
    "Chemeleon baseline": [
        (1, 46313),
        (2, 46316),
        (3, 46322),
    ],
    "FARS-MQ": [
        (1, 47206),
        (2, 47237),
        (3, 47216),
    ],
    "FARS (DeepSeek)": [(1, 47268), (2, 47290), (3, 47307)],
    "Matched LLM-only control (DeepSeek)": [
        (1, 47323),
        (2, 47342),
        (3, 47361),
    ],
    "FARS-MQ (DeepSeek)": [(1, 47373), (2, 47398), (3, 47406)],
}
METHOD_KEYS = {
    "FARS": "fars_train",
    "Matched LLM-only control": "llm_orchestrator_control",
    "Random-generator baseline": "random_generator_baseline",
    "Chemeleon baseline": "chemeleon_generative_baseline",
    "FARS-MQ": "fars_meanq_train",
    "FARS (DeepSeek)": "fars_train",
    "Matched LLM-only control (DeepSeek)": "llm_orchestrator_control",
    "FARS-MQ (DeepSeek)": "fars_meanq_train",
}
REPRESENTATIVES = {
    "FARS": (3, 46166),
    "Matched LLM-only control": (3, 46210),
    "Random-generator baseline": (3, 46310),
    "Chemeleon baseline": (1, 46313),
    "FARS-MQ": (2, 47237),
    "FARS (DeepSeek)": (3, 47307),
    "Matched LLM-only control (DeepSeek)": (1, 47323),
    "FARS-MQ (DeepSeek)": (3, 47406),
}
DEEPSEEK_COMPARISON_METHODS = (
    "FARS (DeepSeek)",
    "Matched LLM-only control (DeepSeek)",
    "FARS-MQ (DeepSeek)",
)
RUN_DIRECTORIES = {
    47268: "fars_train-system-Cd-Li-Nd-Ti-W-seed-1-job-47268",
    47290: "fars_train-system-Cd-Li-Nd-Ti-W-seed-2-job-47290",
    47307: "fars_train-system-Cd-Li-Nd-Ti-W-seed-3-job-47307",
    47323: "llm_orchestrator_control-system-Cd-Li-Nd-Ti-W-seed-1-job-47323",
    47342: "llm_orchestrator_control-system-Cd-Li-Nd-Ti-W-seed-2-job-47342",
    47361: "llm_orchestrator_control-system-Cd-Li-Nd-Ti-W-seed-3-job-47361",
    47373: "fars_meanq_train-system-Cd-Li-Nd-Ti-W-seed-1-job-47373",
    47398: "fars_meanq_train-system-Cd-Li-Nd-Ti-W-seed-2-job-47398",
    47406: "fars_meanq_train-system-Cd-Li-Nd-Ti-W-seed-3-job-47406",
}
ELAPSED_SECONDS = {
    46040: 7429,
    46112: 6340,
    46166: 5870,
    46186: 6794,
    46199: 6430,
    46210: 6599,
    46222: 817,
    46227: 856,
    46310: 723,
    46313: 567,
    46316: 582,
    46322: 592,
    47206: 988,
    47237: 974,
    47216: 1077,
    47268: 4335,
    47290: 4167,
    47307: 4202,
    47323: 2880,
    47342: 2493,
    47361: 2164,
    47373: 3620,
    47398: 3500,
    47406: 3436,
}
CONTROLLER_RUNS = [
    ("FARS", 1, 46040, None),
    ("FARS", 2, 46112, None),
    ("FARS", 3, 46166, None),
    (
        "FARS pooled-Q continuation",
        4,
        47098,
        RESULTS / "repeated-budget-60/fars_q_values_initial.json",
    ),
    (
        "FARS pooled-Q smoke",
        5,
        47162,
        RESULTS / "repeated-budget-60/fars_q_values_initial.json",
    ),
    (
        "FARS-MQ smoke",
        6,
        47169,
        RESULTS / "mean-q-transfer/initial_q_values.json",
    ),
    (
        "FARS-MQ smoke",
        7,
        47175,
        RESULTS / "mean-q-transfer/initial_q_values.json",
    ),
    (
        "FARS-MQ",
        1,
        47206,
        RESULTS / "mean-q-transfer/initial_q_values.json",
    ),
    (
        "FARS-MQ",
        2,
        47237,
        RESULTS / "mean-q-transfer/initial_q_values.json",
    ),
    (
        "FARS-MQ",
        3,
        47216,
        RESULTS / "mean-q-transfer/initial_q_values.json",
    ),
    ("FARS (DeepSeek)", 1, 47268, None),
    ("FARS (DeepSeek)", 2, 47290, None),
    ("FARS (DeepSeek)", 3, 47307, None),
    (
        "FARS-MQ (DeepSeek)",
        1,
        47373,
        RESULTS / "mean-q-transfer/initial_q_values.json",
    ),
    (
        "FARS-MQ (DeepSeek)",
        2,
        47398,
        RESULTS / "mean-q-transfer/initial_q_values.json",
    ),
    (
        "FARS-MQ (DeepSeek)",
        3,
        47406,
        RESULTS / "mean-q-transfer/initial_q_values.json",
    ),
]


def run_dir(method: str, seed: int, job_id: int) -> Path:
    if job_id in RUN_DIRECTORIES:
        return RESULTS / RUN_DIRECTORIES[job_id]
    key = METHOD_KEYS[method]
    return RESULTS / f"{key}-seed-{seed}-job-{job_id}"


def token_usage_for_job(job_id: int) -> int | None:
    path = RESULTS / "token-usage" / f"job-{job_id}.jsonl"
    if not path.is_file():
        return None
    records = read_jsonl(path)
    summaries = [
        record for record in records if record.get("event") == "lm_usage_summary"
    ]
    if summaries:
        total = summaries[-1].get("api_token_totals_excluding_cache_hits", {}).get(
            "total_tokens"
        )
        return int(total) if isinstance(total, int) else None
    usage_records = [record for record in records if record.get("event") == "lm_usage"]
    if not usage_records:
        return None
    return sum(int(record.get("total_tokens") or 0) for record in usage_records)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def observed_hull_curve(path: Path) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    episode = load_json(path)
    trajectory = episode["trajectory"]
    final_state = episode["final_env_state"]
    tolerance = float(final_state["stability_tolerance"])

    proposals = [
        Structure.from_dict(row["proposal"])
        for row in trajectory
    ]
    proposal_hashes = [structure_hash(structure) for structure in proposals]

    # Reconstruct prefixes from MADE's final observed phase diagram, excluding
    # query structures that had not yet been submitted at each prefix.
    observed_entries = [
        PDEntry.from_dict(entry)
        for entry in final_state["phase_diagram_all_entries"]
    ]
    entries_by_hash: dict[str, PDEntry] = {}
    for entry in observed_entries:
        attribute = entry.attribute or {}
        if "structure" in attribute:
            hashed = structure_hash(Structure.from_dict(attribute["structure"]))
            entries_by_hash[hashed] = entry
    if not set(proposal_hashes).issubset(entries_by_hash):
        missing = len(set(proposal_hashes) - set(entries_by_hash))
        raise ValueError(f"{missing} queried structures absent from observed entries: {path}")

    queried_hashes = set(proposal_hashes)
    initial_entries = [
        entry
        for hashed, entry in entries_by_hash.items()
        if hashed not in queried_hashes
    ]
    cumulative: list[int] = []
    queried_unique: dict[str, PDEntry] = {}
    for index, row in enumerate(trajectory):
        hashed = proposal_hashes[index]
        if row.get("is_newly_discovered", False):
            queried_unique[hashed] = entries_by_hash[hashed]
        prefix_entries = initial_entries + [
            entries_by_hash[proposal_hashes[prefix]]
            for prefix in range(index + 1)
        ]
        observed_pd = PhaseDiagram(prefix_entries)
        stable_hashes = {
            candidate_hash
            for candidate_hash, entry in queried_unique.items()
            if safe_e_above_hull(observed_pd, entry) <= tolerance
        }
        cumulative.append(len(stable_hashes))

    metrics = episode["metrics"]
    official_msun = float(metrics["final/novelty_stable_unique_novel_count"])
    official_audc = float(metrics["final/discovery_curve_area_under_discovery_curve"])
    if not math.isclose(cumulative[-1], official_msun, abs_tol=1e-8):
        raise ValueError(f"mSUN curve mismatch in {path}: {cumulative[-1]} != {official_msun}")
    reconstructed_audc = trapezoid([0, *cumulative])
    if not math.isclose(reconstructed_audc, official_audc, abs_tol=1e-7):
        raise ValueError(
            f"AUDC curve mismatch in {path}: {reconstructed_audc} != {official_audc}"
        )
    return trajectory, cumulative, metrics


def trapezoid(values: list[float]) -> float:
    return sum(
        (left + right) / 2 for left, right in zip(values, values[1:], strict=False)
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def q_scores_for_run(
    label: str, seed: int, job_id: int, initial_path: Path | None
) -> list[dict[str, Any]]:
    directory = RESULTS / next(
        path.name
        for path in RESULTS.glob(f"*-seed-{seed}-job-{job_id}")
    )
    logs = list((directory / "controller-logs").glob("*.jsonl"))
    if len(logs) != 1:
        raise ValueError(f"Expected one controller log in {directory}")
    events = read_jsonl(logs[0])
    initial_q_values = load_json(initial_path) if initial_path else {}
    q_values = {
        state: dict(values) for state, values in initial_q_values.items()
    }
    decisions: dict[int, dict[str, Any]] = {}
    work: dict[int, dict[str, Any]] = {}
    outcomes: dict[int, dict[str, Any]] = {}
    transitions: dict[int, dict[str, Any]] = {}
    for event in events:
        kind = event["event"]
        if kind == "decision":
            decisions[int(event["decision_query"])] = event
        elif kind == "proposal_work":
            work[int(event["decision_query"])] = event
        elif kind == "observed_outcome":
            outcomes[int(event["query_count"])] = event
        elif kind == "transition":
            transitions[int(event["query_count"])] = event

    rows = []
    alpha, gamma = 0.2, 0.9
    for query in sorted(decisions):
        decision = decisions[query]
        state_key = decision["scheduler_state"]
        values = q_values.get(state_key, {})
        explore_q = float(values.get("EXPLORE", 0.0))
        refine_q = float(values.get("REFINE", 0.0))
        chosen = decision["controller_action"]
        chosen_q = explore_q if chosen == "EXPLORE" else refine_q
        other_q = refine_q if chosen == "EXPLORE" else explore_q
        maximum = max(explore_q, refine_q)
        if math.isclose(chosen_q, other_q, abs_tol=1e-12):
            choice_note = "Q tie; random tie-break or epsilon branch is not logged"
        elif math.isclose(chosen_q, maximum, abs_tol=1e-12):
            choice_note = "Q-greedy-consistent; epsilon branch is not logged"
        else:
            choice_note = "Lower-Q action; epsilon exploration is required"
        work_item = work.get(query, {})
        rows.append(
            {
                "method_run": label,
                "seed": seed,
                "job_id": job_id,
                "query": query,
                "state_key": state_key,
                "warm_start_exact_match": state_key in initial_q_values,
                "warm_start_initial_q_values": initial_q_values.get(state_key, {}),
                "chosen_action": chosen,
                "explore_q_at_selection": explore_q,
                "refine_q_at_selection": refine_q,
                "chosen_action_q": chosen_q,
                "other_action_q": other_q,
                "q_greedy_action": (
                    "EXPLORE"
                    if explore_q > refine_q
                    else "REFINE"
                    if refine_q > explore_q
                    else "TIE"
                ),
                "choice_note": choice_note,
                "epsilon": 0.2,
                "generated_candidates": work_item.get("generated_candidates", ""),
                "distinct_scored_candidates": work_item.get(
                    "distinct_scored_candidates", ""
                ),
                "selected_composition": work_item.get("selected_composition", ""),
                "composition_diversity_fallback": work_item.get(
                    "composition_diversity_fallback", ""
                ),
                "reward_after_query": outcomes.get(query, {}).get(
                    "signed_msun_increment", ""
                ),
                "hidden_ground_truth_read": decision.get(
                    "hidden_ground_truth_read", ""
                ),
            }
        )

        transition = transitions.get(query)
        if transition is not None:
            next_decision = decisions.get(query + 1)
            next_key = (
                next_decision["scheduler_state"] if next_decision else ""
            )
            next_actions = (
                next_decision["feasible_actions"]
                if next_decision
                else []
            )
            future = max(
                (float(q_values.get(next_key, {}).get(action, 0.0)) for action in next_actions),
                default=0.0,
            )
            action_values = q_values.setdefault(state_key, {})
            current = float(action_values.get(chosen, 0.0))
            reward = float(transition["reward"])
            action_values[chosen] = current + alpha * (
                reward + gamma * future - current
            )
    saved_table_path = directory / "controller-logs/q_values.json"
    if saved_table_path.exists():
        saved_table = load_json(saved_table_path)
        for state, actions in saved_table.items():
            for action, value in actions.items():
                reconstructed = float(q_values.get(state, {}).get(action, 0.0))
                if not math.isclose(reconstructed, float(value), abs_tol=1e-9):
                    raise ValueError(
                        f"Q-table reconstruction mismatch for {label}, seed {seed}, "
                        f"state {state}, action {action}: "
                        f"{reconstructed} != {value}"
                    )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_curve_plot(curves: dict[str, dict[int, list[int]]]) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    display_names = {
        "FARS": "FARS (Qwen; earlier code)",
        "Matched LLM-only control": "LLM-only (Qwen; earlier code)",
        "FARS-MQ": "FARS-MQ (Qwen)",
        "FARS (DeepSeek)": "FARS (DeepSeek; updated code)",
        "Matched LLM-only control (DeepSeek)": "LLM-only (DeepSeek)",
        "FARS-MQ (DeepSeek)": "FARS-MQ (DeepSeek)",
    }
    for method, seed_curves in curves.items():
        queries = list(range(1, BUDGET + 1))
        means = [
            sum(curve[index] for curve in seed_curves.values()) / len(seed_curves)
            for index in range(BUDGET)
        ]
        deviations = [
            math.sqrt(
                sum((curve[index] - means[index]) ** 2 for curve in seed_curves.values())
                / (len(seed_curves) - 1)
            )
            for index in range(BUDGET)
        ]
        axis.plot(
            queries,
            means,
            label=display_names.get(method, method),
            linewidth=2,
        )
        axis.fill_between(
            queries,
            [max(0, mean - deviation) for mean, deviation in zip(means, deviations, strict=True)],
            [mean + deviation for mean, deviation in zip(means, deviations, strict=True)],
            alpha=0.12,
        )
    axis.set(
        xlabel="Official oracle queries used",
        ylabel="Cumulative stable, unique, novel discoveries (mSUN)",
        title="Budget-60 discovery curves across historical and new runs",
        xlim=(0, BUDGET),
    )
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.text(
        0.5,
        0.01,
        "Historical Qwen FARS/control runs used earlier code; DeepSeek runs used updated code.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(OUTPUT / "cumulative-discovery-curve.png", dpi=180)
    plt.close(figure)


def write_selection_state_diagnostics() -> None:
    initializer_path = RESULTS / "mean-q-transfer/initial_q_values.json"
    initial_q_values = load_json(initializer_path)
    rows = []
    for method in ("FARS (DeepSeek)", "FARS-MQ (DeepSeek)"):
        for seed, job_id in RUNS[method]:
            directory = run_dir(method, seed, job_id)
            logs = list((directory / "controller-logs").glob("*.jsonl"))
            if len(logs) != 1:
                raise ValueError(f"Expected one controller log in {directory}")
            events = read_jsonl(logs[0])
            decisions = {
                int(event["decision_query"]): event
                for event in events
                if event.get("event") == "decision"
            }
            proposal_work = {
                int(event["decision_query"]): event
                for event in events
                if event.get("event") == "proposal_work"
            }
            trajectory, observed_curve, _ = observed_hull_curve(
                directory / "trajectories/episode_000.json"
            )
            structure_hashes = [
                structure_hash(Structure.from_dict(row["proposal"]))
                for row in trajectory
            ]
            selected = []
            state_keys = []
            hidden_ground_truth_values = []
            initializer_sources = set()
            for query, decision in sorted(decisions.items()):
                state_key = decision["scheduler_state"]
                state_keys.append(state_key)
                hidden_ground_truth_values.append(
                    decision.get("hidden_ground_truth_read")
                )
                if decision.get("warm_start_source"):
                    initializer_sources.add(decision["warm_start_source"])
                work = proposal_work.get(query, {})
                formula = work.get("selected_composition")
                if formula:
                    reduced_formula = Composition(formula).reduced_formula
                    selected.append(
                        {
                            "query": query,
                            "formula": reduced_formula,
                            "action": decision["controller_action"],
                            "reward": observed_curve[query - 1]
                            - (observed_curve[query - 2] if query > 1 else 0),
                        }
                    )

            if len(decisions) != BUDGET or len(selected) != BUDGET:
                raise ValueError(
                    f"Expected {BUDGET} selected candidate decisions for {method}, "
                    f"seed {seed}; found {len(decisions)} decisions and "
                    f"{len(selected)} selected formulas"
                )
            if method == "FARS-MQ (DeepSeek)" and initializer_sources != {
                str(initializer_path.resolve())
            }:
                raise ValueError(
                    f"FARS-MQ run {job_id} did not consistently record "
                    f"{initializer_path} as its warm-start source: "
                    f"{sorted(initializer_sources)}"
                )

            seen_formulas: set[str] = set()
            repeats = []
            formula_counts: dict[str, int] = defaultdict(int)
            seen_structures: set[str] = set()
            structure_repeats = 0
            for item in selected:
                formula_counts[item["formula"]] += 1
                if item["formula"] in seen_formulas:
                    repeats.append(item)
                seen_formulas.add(item["formula"])
            for candidate_hash in structure_hashes:
                if candidate_hash in seen_structures:
                    structure_repeats += 1
                seen_structures.add(candidate_hash)
            top_formulas = sorted(
                formula_counts.items(), key=lambda item: (-item[1], item[0])
            )[:3]
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "job_id": job_id,
                    "decisions": len(decisions),
                    "selected_queries_with_formula": len(selected),
                    "distinct_selected_reduced_formulas": len(formula_counts),
                    "repeated_formula_selections_after_first": len(repeats),
                    "distinct_selected_structures_by_made_hash": len(
                        seen_structures
                    ),
                    "exact_structure_repeat_selections_after_first": (
                        structure_repeats
                    ),
                    "repeated_formula_queries_explore": sum(
                        item["action"] == "EXPLORE" for item in repeats
                    ),
                    "repeated_formula_queries_refine": sum(
                        item["action"] == "REFINE" for item in repeats
                    ),
                    "repeated_formula_queries_with_positive_observed_msun": sum(
                        item["reward"] > 0 for item in repeats
                    ),
                    "top_selected_reduced_formulas": json.dumps(top_formulas),
                    "unique_scheduler_states": len(set(state_keys)),
                    "initializer_state_count": len(initial_q_values),
                    "exact_initializer_matched_decisions": sum(
                        state in initial_q_values for state in state_keys
                    ),
                    "distinct_visited_states_matching_initializer": len(
                        {state for state in state_keys if state in initial_q_values}
                    ),
                    "hidden_ground_truth_read_true": sum(
                        value is True for value in hidden_ground_truth_values
                    ),
                    "hidden_ground_truth_read_unreported": sum(
                        value is None for value in hidden_ground_truth_values
                    ),
                }
            )
    write_csv(OUTPUT / "candidate-selection-state-coverage.csv", rows)


def write_deepseek_seed_curves(curves: dict[str, dict[int, list[int]]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    queries = list(range(1, BUDGET + 1))
    for axis, method in zip(axes, DEEPSEEK_COMPARISON_METHODS, strict=True):
        seed_curves = curves[method]
        for seed, curve in sorted(seed_curves.items()):
            axis.plot(queries, curve, label=f"Seed {seed}", alpha=0.65, linewidth=1.5)
        means = [
            sum(curve[index] for curve in seed_curves.values()) / len(seed_curves)
            for index in range(BUDGET)
        ]
        axis.plot(queries, means, label="Mean", color="black", linewidth=2.5)
        axis.set_title(method)
        axis.set_xlabel("Official oracle queries")
        axis.set_xlim(1, BUDGET)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Cumulative stable, unique, novel discoveries (mSUN)")
    figure.suptitle("DeepSeek budget-60 discovery curves by seed")
    figure.tight_layout()
    figure.savefig(OUTPUT / "deepseek-three-methods-seed-curves.png", dpi=180)
    plt.close(figure)


def write_deepseek_token_cost_table() -> None:
    rows = []
    for method in DEEPSEEK_COMPARISON_METHODS:
        run_tokens = [
            (seed, token_usage_for_job(job_id)) for seed, job_id in RUNS[method]
        ]
        logged_tokens = [tokens for _, tokens in run_tokens if tokens is not None]
        mean_logged_tokens = sum(logged_tokens) / len(logged_tokens)
        estimated_total = round(mean_logged_tokens * len(RUNS[method]))
        rows.append(
            {
                "method": method,
                "runs": len(RUNS[method]),
                "token_logs_available": f"{len(logged_tokens)}/{len(RUNS[method])}",
                "observed_tokens_excluding_cache_hits": sum(logged_tokens),
                "mean_tokens_per_logged_run": round(mean_logged_tokens),
                "estimated_tokens_for_three_runs": estimated_total,
                "estimate_basis": "mean of available run logs multiplied by 3 runs",
                "currency_cost": (
                    "Not available: benchmark-level currency pricing was not recorded"
                ),
            }
        )
    write_csv(OUTPUT / "deepseek-token-cost-estimate.csv", rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    curves_by_method: dict[str, dict[int, list[int]]] = defaultdict(dict)
    metrics_by_run: dict[tuple[str, int], dict[str, Any]] = {}
    trajectory_cache: dict[tuple[str, int], tuple[list[dict[str, Any]], list[int]]] = {}

    for method, runs in RUNS.items():
        for seed, job_id in runs:
            directory = run_dir(method, seed, job_id)
            trajectory, curve, metrics = observed_hull_curve(
                directory / "trajectories/episode_000.json"
            )
            curves_by_method[method][seed] = curve
            metrics_by_run[(method, seed)] = metrics
            trajectory_cache[(method, seed)] = (trajectory, curve)

    curve_rows: list[dict[str, Any]] = []
    for method, seed_curves in curves_by_method.items():
        for seed, curve in seed_curves.items():
            for query, count in enumerate(curve, 1):
                curve_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "query_budget": query,
                        "cumulative_msun": count,
                    }
                )
        for query in range(1, BUDGET + 1):
            values = [curve[query - 1] for curve in seed_curves.values()]
            mean = sum(values) / len(values)
            sd = (
                math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
                if len(values) > 1
                else 0.0
            )
            curve_rows.append(
                {
                    "method": method,
                    "seed": "mean",
                    "query_budget": query,
                    "cumulative_msun": round(mean, 8),
                }
            )
            curve_rows.append(
                {
                    "method": method,
                    "seed": "sample_sd",
                    "query_budget": query,
                    "cumulative_msun": round(sd, 8),
                }
            )
    write_csv(OUTPUT / "cumulative-discovery-curves.csv", curve_rows)
    write_curve_plot(curves_by_method)
    write_deepseek_seed_curves(curves_by_method)
    write_deepseek_token_cost_table()
    write_selection_state_diagnostics()

    representative_rows: list[dict[str, Any]] = []
    for method, (seed, job_id) in REPRESENTATIVES.items():
        trajectory, curve = trajectory_cache[(method, seed)]
        for query, (observation, cumulative) in enumerate(
            zip(trajectory, curve, strict=True), 1
        ):
            representative_rows.append(
                {
                    "method": method,
                    "representative_seed": seed,
                    "job_id": job_id,
                    "query_budget": query,
                    "reduced_formula": observation["reduced_formula"],
                    "e_above_hull_at_query_eV_per_atom": observation["e_above_hull"],
                    "stable_at_query": observation["is_stable"],
                    "new_structure_at_query": observation["is_newly_discovered"],
                    "distance_smact_valid": observation[
                        "validity_distance_and_smact_valid_count"
                    ]
                    > 0,
                    "cumulative_msun_observed_hull": cumulative,
                }
            )
    write_csv(OUTPUT / "representative-run-trajectories.csv", representative_rows)

    action_rows: list[dict[str, Any]] = []
    for controller_run in CONTROLLER_RUNS:
        action_rows.extend(q_scores_for_run(*controller_run))
    write_csv(OUTPUT / "fars-action-q-scores.csv", action_rows)

    random_curve = [
        sum(curve[q] for curve in curves_by_method["Random-generator baseline"].values())
        / len(curves_by_method["Random-generator baseline"])
        for q in range(BUDGET)
    ]
    random_target = random_curve[-1]
    random_target_q = next(
        (index + 1 for index, value in enumerate(random_curve) if value >= random_target),
        None,
    )
    summary_rows = []
    for method in RUNS:
        run_metrics = [metrics_by_run[(method, seed)] for seed, _ in RUNS[method]]
        final_values = [
            float(metrics["final/novelty_stable_unique_novel_count"])
            for metrics in run_metrics
        ]
        audcs = [
            float(metrics["final/discovery_curve_area_under_discovery_curve"])
            for metrics in run_metrics
        ]
        mean_msun = sum(final_values) / len(final_values)
        mean_audc = sum(audcs) / len(audcs)
        ef = mean_msun / random_target if random_target else ""
        method_curve = [
            sum(curves_by_method[method][seed][q] for seed, _ in RUNS[method])
            / len(RUNS[method])
            for q in range(BUDGET)
        ]
        target_q = next(
            (index + 1 for index, value in enumerate(method_curve) if value >= random_target),
            None,
        )
        af = (
            random_target_q / target_q
            if random_target_q and target_q
            else 0.0
        )
        elapsed = [ELAPSED_SECONDS[job] for _, job in RUNS[method]]
        total_generated = []
        total_scored = []
        for seed, job_id in RUNS[method]:
            log = next(
                (run_dir(method, seed, job_id) / "controller-logs").glob("*.jsonl"),
                None,
            )
            if log is None:
                total_generated.append(1024)
                total_scored.append(0)
            else:
                events = read_jsonl(log)
                proposal_work = [event for event in events if event["event"] == "proposal_work"]
                total_generated.append(sum(event["generated_candidates"] for event in proposal_work))
                total_scored.append(sum(event["distinct_scored_candidates"] for event in proposal_work))
        summary_rows.append(
            {
                "method": method,
                "llm_model": (
                    "DeepSeek Flash"
                    if "DeepSeek" in method
                    else (
                        "Qwen2.5-Coder 7B"
                        if method in {"FARS", "Matched LLM-only control", "FARS-MQ"}
                        else "none"
                    )
                ),
                "llm_token_runs_available": sum(
                    token_usage_for_job(job_id) is not None
                    for _, job_id in RUNS[method]
                ),
                "llm_tokens_total_available_runs": sum(
                    total
                    for _, job_id in RUNS[method]
                    if (total := token_usage_for_job(job_id)) is not None
                ),
                "llm_tokens_mean_available_run": (
                    round(
                        sum(
                            total
                            for _, job_id in RUNS[method]
                            if (total := token_usage_for_job(job_id)) is not None
                        )
                        / sum(
                            token_usage_for_job(job_id) is not None
                            for _, job_id in RUNS[method]
                        ),
                        2,
                    )
                    if any(
                        token_usage_for_job(job_id) is not None
                        for _, job_id in RUNS[method]
                    )
                    else ""
                ),
                "representative_seed": REPRESENTATIVES[method][0],
                "representative_job_id": REPRESENTATIVES[method][1],
                "query_budget_per_run": BUDGET,
                "mSUN_at_60_mean": round(mean_msun, 6),
                "mSUN_at_60_sample_sd": round(
                    math.sqrt(sum((v - mean_msun) ** 2 for v in final_values) / (len(final_values) - 1)),
                    6,
                ),
                "AUDC_official_mean": round(mean_audc, 6),
                "AUDC_official_sample_sd": round(
                    math.sqrt(sum((v - mean_audc) ** 2 for v in audcs) / (len(audcs) - 1)),
                    6,
                ),
                "AUDC_normalized_mean": round(mean_audc / (BUDGET * BUDGET / 2), 8),
                "EF_vs_random_mean_mSUN_at_60": round(ef, 6),
                "AF_vs_random_to_random_final_mean": round(af, 6),
                "random_reference_target_mSUN": round(random_target, 6),
                "random_reference_target_query": random_target_q,
                "method_target_query": target_q or "",
                "MADE_saved_EF_AF": "0 (baseline history not supplied)",
                "oracle_queries_per_run": BUDGET,
                "generated_candidates_per_run_mean": round(sum(total_generated) / len(total_generated), 2),
                "scored_candidates_per_run_mean": round(sum(total_scored) / len(total_scored), 2),
                "wall_clock_minutes_per_run_mean": round(sum(elapsed) / len(elapsed) / 60, 2),
                "wall_clock_minutes_per_run_range": (
                    f"{min(elapsed) / 60:.2f}-{max(elapsed) / 60:.2f}"
                ),
                "allocated_gpu_hours_total": round(sum(elapsed) / 3600, 3),
                "llm_token_cost": "not available from provider logs",
            }
        )
    write_csv(OUTPUT / "metrics-and-research-cost.csv", summary_rows)
    write_csv(OUTPUT / "current-evaluation-matrix.csv", summary_rows)

    print(f"Analysis written to {OUTPUT}")
    print(f"Random reference: mSUN@60={random_target:.3f}; first reaches it at q={random_target_q}")
    for row in summary_rows:
        print(
            f"{row['method']}: mSUN@60={row['mSUN_at_60_mean']}, "
            f"AUDC={row['AUDC_official_mean']}, EF={row['EF_vs_random_mean_mSUN_at_60']}, "
            f"AF={row['AF_vs_random_to_random_final_mean']}"
        )


if __name__ == "__main__":
    main()
