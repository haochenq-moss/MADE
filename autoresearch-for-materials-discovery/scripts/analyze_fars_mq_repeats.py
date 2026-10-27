#!/usr/bin/env python3
"""Plot completed FARS-MQ repeats against seed-matched FARS runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from analyze_budget60 import observed_hull_curve, read_jsonl, token_usage_for_job

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
OUTPUT = RESULTS / "comparison-budget-60"
MANIFEST = (
    RESULTS
    / "replicate-fars-mq-Cd-Li-Nd-Ti-W-budget-60/job-manifest.csv"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No completed FARS-MQ repeat runs found in {MANIFEST}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_directory(method: str, seed: int, job_id: int) -> Path:
    path = RESULTS / (
        f"{method}-system-Cd-Li-Nd-Ti-W-seed-{seed}-job-{job_id}"
    )
    if not path.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {path}")
    return path


def load_run(method: str, seed: int, job_id: int) -> tuple[list[int], dict[str, Any]]:
    directory = run_directory(method, seed, job_id)
    trajectory_path = directory / "trajectories/episode_000.json"
    if not (directory / "summary/episodes.csv").is_file():
        raise FileNotFoundError(f"Run does not have a completed summary: {directory}")
    _, curve, metrics = observed_hull_curve(trajectory_path)
    episode = json.loads(trajectory_path.read_text(encoding="utf-8"))
    if len(episode["trajectory"]) != 60:
        raise ValueError(
            f"Expected 60 queries for job {job_id}; "
            f"found {len(episode['trajectory'])}"
        )
    proposal_log = next((directory / "controller-logs").glob("*.jsonl"), None)
    if proposal_log is None:
        raise FileNotFoundError(f"Missing controller event log: {directory}")
    work = [
        row
        for row in read_jsonl(proposal_log)
        if row.get("event") == "proposal_work"
    ]
    tokens = token_usage_for_job(job_id)
    result = {
        "method": method,
        "seed": seed,
        "job_id": job_id,
        "queries": len(episode["trajectory"]),
        "mSUN_at_60": int(metrics["final/novelty_stable_unique_novel_count"]),
        "AUDC": float(metrics["final/discovery_curve_area_under_discovery_curve"]),
        "nAUDC": float(
            metrics["final/discovery_curve_area_under_discovery_curve_normalized"]
        ),
        "generated_candidates": sum(
            int(row.get("generated_candidates", 0)) for row in work
        ),
        "scored_candidates": sum(
            int(row.get("distinct_scored_candidates", 0)) for row in work
        ),
        "provider_tokens_excluding_cache_hits": tokens,
    }
    return curve, result


def main() -> None:
    manifest_rows = read_csv(MANIFEST)
    if not manifest_rows:
        raise ValueError(f"Manifest is empty: {MANIFEST}")

    completed: list[tuple[int, int, list[int], dict[str, Any], list[int], dict[str, Any]]] = []
    for row in manifest_rows:
        seed, mq_job, fars_job = (
            int(row["seed"]),
            int(row["job_id"]),
            int(row["reference_fars_job_id"]),
        )
        mq_dir = run_directory("fars_meanq_train", seed, mq_job)
        if not (mq_dir / "summary/episodes.csv").is_file():
            continue
        mq_curve, mq_metrics = load_run("fars_meanq_train", seed, mq_job)
        fars_curve, fars_metrics = load_run("fars_train", seed, fars_job)
        completed.append(
            (seed, mq_job, mq_curve, mq_metrics, fars_curve, fars_metrics)
        )

    if not completed:
        raise ValueError("No complete FARS-MQ repeat runs are available yet.")

    original_rows = read_csv(OUTPUT / "cumulative-discovery-curves.csv")
    original_curve = [
        float(row["cumulative_msun"])
        for row in original_rows
        if row["method"] == "FARS-MQ (DeepSeek)"
        and row["seed"] == "mean"
    ]
    if len(original_curve) != 60:
        raise ValueError(
            "Expected the original 60-query DeepSeek FARS-MQ mean curve."
        )

    figure, axis = plt.subplots(figsize=(10, 6))
    queries = list(range(1, 61))
    axis.plot(
        queries,
        original_curve,
        color="black",
        linestyle=":",
        linewidth=2.5,
        label="Original FARS-MQ DeepSeek mean (3 seeds)",
    )
    colors = plt.get_cmap("tab10")
    for index, (seed, _, mq_curve, _, fars_curve, _) in enumerate(completed):
        color = colors(index % 10)
        axis.plot(
            queries,
            fars_curve,
            color=color,
            linestyle="--",
            alpha=0.7,
            linewidth=1.6,
            label=f"Fresh-start FARS seed {seed}",
        )
        axis.plot(
            queries,
            mq_curve,
            color=color,
            linewidth=2.2,
            label=f"FARS-MQ repeat seed {seed}",
        )
    axis.set(
        xlabel="Official oracle queries used",
        ylabel="Cumulative stable, unique, novel discoveries (mSUN)",
        title="FARS-MQ repeats versus seed-matched FARS",
        xlim=(1, 60),
    )
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT / "fars-mq-repeats-discovery-curve.png", dpi=180)
    plt.close(figure)

    metrics_rows: list[dict[str, Any]] = []
    matrix = read_csv(OUTPUT / "current-evaluation-matrix.csv")
    original_summary = next(
        row for row in matrix if row["method"] == "FARS-MQ (DeepSeek)"
    )
    metrics_rows.append(
        {
            "method": "Original FARS-MQ (DeepSeek) mean",
            "seed_or_summary": "3-seed mean",
            "job_id": "original campaign",
            "official_queries": 60,
            "mSUN_at_60": original_summary["mSUN_at_60_mean"],
            "AUDC": original_summary["AUDC_official_mean"],
            "nAUDC": original_summary["AUDC_normalized_mean"],
            "generated_candidates": original_summary[
                "generated_candidates_per_run_mean"
            ],
            "scored_candidates": original_summary[
                "scored_candidates_per_run_mean"
            ],
            "provider_tokens_excluding_cache_hits": (
                original_summary["llm_tokens_mean_available_run"]
            ),
            "token_note": "Mean across original 3 logged runs",
        }
    )
    for seed, _mq_job, _, mq_result, _, fars_result in completed:
        for method, result in (
            ("Fresh-start FARS (DeepSeek)", fars_result),
            ("FARS-MQ repeat (DeepSeek)", mq_result),
        ):
            metrics_rows.append(
                {
                    "method": method,
                    "seed_or_summary": seed,
                    "job_id": result["job_id"],
                    "official_queries": result["queries"],
                    "mSUN_at_60": result["mSUN_at_60"],
                    "AUDC": result["AUDC"],
                    "nAUDC": result["nAUDC"],
                    "generated_candidates": result["generated_candidates"],
                    "scored_candidates": result["scored_candidates"],
                    "provider_tokens_excluding_cache_hits": (
                        result["provider_tokens_excluding_cache_hits"]
                    ),
                    "token_note": (
                        "One run; provider tokens unavailable"
                        if result["provider_tokens_excluding_cache_hits"] is None
                        else "One run; provider-reported tokens"
                    ),
                }
            )
    write_csv(OUTPUT / "fars-mq-repeats-metrics.csv", metrics_rows)

    print(f"Analysis written to {OUTPUT}")
    for _, _, _, mq_metrics, _, fars_metrics in completed:
        print(
            f"Seed {mq_metrics['seed']}: FARS-MQ job {mq_metrics['job_id']} "
            f"mSUN={mq_metrics['mSUN_at_60']}, AUDC={mq_metrics['AUDC']}; "
            f"paired FARS job {fars_metrics['job_id']} "
            f"mSUN={fars_metrics['mSUN_at_60']}, AUDC={fars_metrics['AUDC']}"
        )


if __name__ == "__main__":
    main()
