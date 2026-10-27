#!/usr/bin/env python3
"""Build a self-contained dashboard from saved experiment episodes."""

from __future__ import annotations

import gzip
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
MADE_ROOT = PROJECT.parent
RESULTS = PROJECT / "results"
OUTPUT = RESULTS / "dashboard" / "index.html"
BRANCH_SEARCH_RESULTS = (
    Path.home()
    / ".cache"
    / "auto-research-made"
    / "studies"
    / "cd-li-nd-ti-w-branch-budget60"
    / "state-q-branch"
)

METHODS = (
    ("fars_meanq_train", "FARS-MQ"),
    ("fars_train", "FARS"),
    ("llm_orchestrator_control", "LLM-only control"),
    ("random_generator_baseline", "Random generator"),
    ("chemeleon_generative_baseline", "Chemeleon"),
    ("random", "Random generator"),
    ("llm-control", "LLM-only control"),
    ("chemeleon", "Chemeleon"),
)
ELAPSED_SECONDS = {
    46040: 7429, 46112: 6340, 46166: 5870, 46186: 6794, 46199: 6430,
    46210: 6599, 46222: 817, 46227: 856, 46310: 723, 46313: 567,
    46316: 582, 46322: 592, 47206: 988, 47237: 974, 47216: 1077,
    47268: 4335, 47290: 4167, 47307: 4202, 47323: 2880, 47342: 2493,
    47361: 2164, 47373: 3620, 47398: 3500, 47406: 3436,
}


def load_curve_function():
    sys.path[:0] = [str(MADE_ROOT / "src"), str(PROJECT / "src")]
    analysis_path = PROJECT / "scripts" / "analyze_budget60.py"
    spec = importlib.util.spec_from_file_location("budget60_analysis", analysis_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load curve reconstruction from {analysis_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.observed_hull_curve


def parse_method(directory: str, config: str) -> str:
    for prefix, label in METHODS:
        if directory.startswith(prefix):
            return label
    return "Other"


def is_balanced_formula(directory: str, config: str) -> bool:
    return (
        directory.startswith("fars_balanced_train")
        or "candidate_selection_policy: balanced_formula" in config
    )


def parse_model(config: str, usage_path: Path | None) -> str:
    if usage_path and usage_path.is_file():
        for line in usage_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            model = record.get("model")
            if model:
                return display_model(str(model))
    for model in re.findall(r"^\s*model:\s*([^\s#]+)", config, re.MULTILINE):
        if any(name in model.lower() for name in ("deepseek", "qwen", "ollama")):
            return display_model(model)
    return "No LLM"


def display_model(model: str) -> str:
    lowered = model.lower()
    if "deepseek" in lowered:
        return "DeepSeek Flash"
    if "qwen2.5-coder" in lowered:
        return "Qwen2.5-Coder 7B"
    if "gpt-5.1" in lowered:
        return "GPT-5.1"
    return model


def job_id_for(directory: str) -> int | None:
    match = re.search(r"job-(\d+)$", directory)
    return int(match.group(1)) if match else None


def read_token_usage(job_id: int | None) -> dict[str, Any]:
    if job_id is None:
        return {"calls": None, "tokens": None, "prompt": None, "completion": None}
    path = RESULTS / "token-usage" / f"job-{job_id}.jsonl"
    if not path.is_file():
        return {"calls": None, "tokens": None, "prompt": None, "completion": None}
    calls = tokens = prompt = completion = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") != "lm_usage":
            continue
        calls += 1
        tokens += int(record.get("total_tokens") or 0)
        prompt += int(record.get("prompt_tokens") or 0)
        completion += int(record.get("completion_tokens") or 0)
    if calls == 0:
        return {"calls": None, "tokens": None, "prompt": None, "completion": None}
    return {"calls": calls, "tokens": tokens, "prompt": prompt, "completion": completion}


def read_controller_work(directory: Path) -> dict[str, int | None]:
    logs = list((directory / "controller-logs").glob("*.jsonl"))
    if len(logs) != 1:
        return {"generated": None, "scored": None}
    generated = scored = 0
    found = False
    for line in logs[0].read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") == "proposal_work":
            found = True
            generated += int(record.get("generated_candidates") or 0)
            scored += int(record.get("distinct_scored_candidates") or 0)
    return {
        "generated": generated if found else None,
        "scored": scored if found else None,
    }


def elapsed_seconds(job_id: int | None) -> tuple[int | None, str | None]:
    if job_id in ELAPSED_SECONDS:
        return ELAPSED_SECONDS[job_id], "Slurm elapsed time"
    if job_id is None:
        return None, None
    paths = [
        path
        for path in (RESULTS / "slurm").glob(f"fars-{job_id}.*")
        if path.suffix in {".gz", ".out", ".err"}
    ]
    timestamps: list[int] = []
    timestamp_pattern = re.compile(r"\[(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\]")
    for path in paths:
        try:
            text = gzip.open(path, "rt", errors="replace").read() if path.suffix == ".gz" else path.read_text(errors="replace")
        except OSError:
            continue
        for match in timestamp_pattern.finditer(text):
            from datetime import datetime

            timestamps.append(
                int(datetime(*map(int, match.groups())).timestamp())
            )
    if len(timestamps) > 1:
        return max(timestamps) - min(timestamps), "Timestamped log span"
    return None, None


def read_elements(config: str, directory: str) -> str:
    path_match = re.search(r"system-([A-Za-z0-9-]+)-seed-", directory)
    if path_match:
        return path_match.group(1)
    inline = re.search(r"elements:\s*\[([^\]]+)\]", config)
    if inline:
        return "-".join(re.findall(r"[A-Z][a-z]?", inline.group(1)))
    lines = config.splitlines()
    for index, line in enumerate(lines):
        if re.match(r"^\s*elements:\s*$", line):
            elements: list[str] = []
            for item in lines[index + 1:]:
                match = re.match(r"^\s*-\s+([A-Z][a-z]?)\s*$", item)
                if match:
                    elements.append(match.group(1))
                elif item.strip():
                    break
            if elements:
                return "-".join(elements)
    return "Unknown system"


def has_terminal_failure(job_id: int) -> bool:
    for suffix in ("err", "err.gz", "out", "out.gz"):
        path = RESULTS / "slurm" / f"fars-{job_id}.{suffix}"
        if not path.is_file():
            continue
        try:
            text = gzip.open(path, "rt", errors="replace").read() if path.suffix == ".gz" else path.read_text(errors="replace")
        except OSError:
            continue
        if "Traceback (most recent call last):" in text or "Disk quota exceeded" in text:
            return True
    return False


def build_branch_search_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    job_ids = {1: 47625, 2: 47637, 3: 47637, 4: 47637, 5: 47637}
    for seed in range(1, 6):
        matches = list(
            (BRANCH_SEARCH_RESULTS / f"seed-{seed:02d}").glob(
                "attempt-*/**/results.json"
            )
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one completed branch-search result for seed {seed}, "
                f"found {len(matches)} under {BRANCH_SEARCH_RESULTS}"
            )
        data = json.loads(matches[0].read_text(encoding="utf-8"))
        snapshot = data["snapshot"]
        experiments = snapshot["experiments"]
        curve = [
            int(experiment["study_metrics"]["num_newly_discovered_stable"])
            for experiment in experiments
        ]
        queries = int(snapshot["official_queries"])
        metrics = snapshot["official_metrics"]
        if snapshot["status"] != "completed" or len(curve) != queries:
            raise RuntimeError(
                f"Branch-search seed {seed} is not a complete reconciled run: "
                f"status={snapshot['status']}, queries={queries}, curve={len(curve)}"
            )
        if curve[-1] != int(metrics["num_newly_discovered_stable"]):
            raise ValueError(f"Branch-search seed {seed} curve endpoint mismatch")
        records.append({
            "id": f"branch-search-state-q-seed-{seed}",
            "label": (
                f"Branch-search state-Q | no LLM | seed {seed} | "
                f"{queries} queries | job {job_ids[seed]}"
            ),
            "method": "Branch-search state-Q",
            "model": "No LLM",
            "seed": seed,
            "job": job_ids[seed],
            "system": "Cd-Li-Nd-Ti-W",
            "status": "Complete",
            "queries": queries,
            "curve": curve,
            "curveError": None,
            "metrics": {
                "msun": metrics["num_newly_discovered_stable"],
                "audc": metrics["discovery_curve_area_under_discovery_curve"],
                "naudc": metrics[
                    "discovery_curve_area_under_discovery_curve_normalized"
                ],
                "validity": metrics[
                    "validity_distance_and_smact_valid_fraction"
                ],
                "structures": metrics[
                    "diversity_all_structure_unique_structure_count"
                ],
                "compositions": metrics[
                    "diversity_all_composition_unique_composition_count"
                ],
            },
            "tokens": {
                "calls": None,
                "tokens": None,
                "prompt": None,
                "completion": None,
            },
            "runtimeSeconds": None,
            "runtimeSource": None,
            "gpuHours": None,
            "generated": None,
            "scored": None,
            "usd": None,
        })
    return records


def build_records() -> list[dict[str, Any]]:
    observed_hull_curve = load_curve_function()
    records: list[dict[str, Any]] = []
    for episode_path in sorted(RESULTS.glob("*/trajectories/episode_000.json")):
        directory = episode_path.parent.parent
        config_path = directory / ".hydra" / "config.yaml"
        config = config_path.read_text(encoding="utf-8", errors="replace") if config_path.is_file() else ""
        if is_balanced_formula(directory.name, config):
            continue
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        metrics = episode.get("metrics", {})
        trajectory = episode.get("trajectory", [])
        if not trajectory:
            continue
        final_queries = int(metrics.get("final/queries_used", len(trajectory)))
        if final_queries == 20 or len(trajectory) == 20:
            continue
        job_id = job_id_for(directory.name)
        usage = read_token_usage(job_id)
        method = parse_method(directory.name, config)
        model = parse_model(config, RESULTS / "token-usage" / f"job-{job_id}.jsonl" if job_id else None)
        try:
            _, curve, _ = observed_hull_curve(episode_path)
            curve_error = None
        except Exception as error:
            curve = None
            curve_error = f"{type(error).__name__}: {error}"
        summary_path = directory / "summary" / "episodes.csv"
        status = "Complete" if summary_path.is_file() and final_queries == len(trajectory) else "Partial / summary unavailable"
        runtime, runtime_source = elapsed_seconds(job_id)
        work = read_controller_work(directory)
        seed_match = re.search(r"seed-(\d+)", directory.name)
        seed = int(seed_match.group(1)) if seed_match else None
        records.append({
            "id": directory.name,
            "label": f"{method} | {model} | seed {seed if seed is not None else 'n/a'} | {final_queries} queries | job {job_id if job_id is not None else 'n/a'}",
            "method": method,
            "model": model,
            "seed": seed,
            "job": job_id,
            "system": read_elements(config, directory.name),
            "status": status,
            "queries": final_queries,
            "curve": curve,
            "curveError": curve_error,
            "metrics": {
                "msun": metrics.get("final/novelty_stable_unique_novel_count"),
                "audc": metrics.get("final/discovery_curve_area_under_discovery_curve"),
                "naudc": metrics.get("final/discovery_curve_area_under_discovery_curve_normalized"),
                "validity": metrics.get("final/validity_distance_and_smact_valid_fraction"),
                "structures": metrics.get("final/diversity_all_structure_unique_structure_count"),
                "compositions": metrics.get("final/diversity_all_composition_unique_composition_count"),
            },
            "tokens": usage,
            "runtimeSeconds": runtime,
            "runtimeSource": runtime_source,
            "gpuHours": round(runtime / 3600, 3) if runtime is not None else None,
            "generated": work["generated"],
            "scored": work["scored"],
            "usd": None,
        })
    records.extend(build_branch_search_records())
    known_ids = {record["job"] for record in records if record["job"] is not None}
    for workdir in sorted((RESULTS / "runner-workdir").glob("job-*")):
        match = re.fullmatch(r"job-(\d+)", workdir.name)
        trace_path = workdir / "trajectory_episode_000.json"
        if not match or not trace_path.is_file() or int(match.group(1)) in known_ids:
            continue
        job_id = int(match.group(1))
        if has_terminal_failure(job_id):
            continue
        directory = next(
            (path for path in RESULTS.glob(f"*-job-{job_id}") if path.is_dir()),
            None,
        )
        if directory is None:
            continue
        config_path = directory / ".hydra" / "config.yaml"
        config = config_path.read_text(encoding="utf-8", errors="replace") if config_path.is_file() else ""
        if is_balanced_formula(directory.name, config):
            continue
        raw_trace = json.loads(trace_path.read_text(encoding="utf-8"))
        trajectory = raw_trace.get("trajectory", []) if isinstance(raw_trace, dict) else raw_trace
        if not trajectory or len(trajectory) == 20:
            continue
        seed_match = re.search(r"seed-(\d+)", directory.name)
        seed = int(seed_match.group(1)) if seed_match else None
        usage = read_token_usage(job_id)
        method = parse_method(directory.name, config)
        runtime, runtime_source = elapsed_seconds(job_id)
        work = read_controller_work(directory)
        provisional_curve: list[int] = []
        discovered = 0
        for row in trajectory:
            if row.get("is_newly_discovered") and row.get("is_stable"):
                discovered += 1
            provisional_curve.append(discovered)
        records.append({
            "id": directory.name,
            "label": f"{method} — in progress | seed {seed if seed is not None else 'n/a'} | {len(trajectory)} queries | job {job_id}",
            "method": method,
            "model": parse_model(config, RESULTS / "token-usage" / f"job-{job_id}.jsonl"),
            "seed": seed,
            "job": job_id,
            "system": read_elements(config, directory.name),
            "status": "In progress (provisional)",
            "queries": len(trajectory),
            "curve": provisional_curve,
            "curveError": None,
            "curveProvisional": True,
            "metrics": {
                "msun": None,
                "audc": None,
                "naudc": None,
                "validity": None,
                "structures": None,
                "compositions": None,
            },
            "tokens": usage,
            "runtimeSeconds": runtime,
            "runtimeSource": runtime_source,
            "gpuHours": round(runtime / 3600, 3) if runtime is not None else None,
            "generated": work["generated"],
            "scored": work["scored"],
            "usd": None,
        })
    return records


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Materials discovery experiment dashboard</title>
<style>
:root{color-scheme:light;--ink:#17243a;--muted:#607086;--line:#d8e0ea;--panel:#fff;--bg:#f4f7fb;--blue:#276b9a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1440px;margin:auto;padding:28px 24px 48px}h1{font-size:clamp(1.7rem,3vw,2.4rem);margin:0 0 4px}h2{font-size:1.12rem;margin:0 0 12px}.sub{color:var(--muted);margin:0 0 20px}
.note{background:#fff8e7;border:1px solid #eddaa9;padding:12px 15px;border-radius:10px;margin-bottom:18px}
.panel,.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;box-shadow:0 2px 8px #13223808}
.filters{padding:16px;margin-bottom:16px;display:grid;grid-template-columns:repeat(4,minmax(170px,1fr));gap:14px;align-items:end}
label,legend{font-weight:650;font-size:.88rem}.field{display:flex;flex-direction:column;gap:6px}select,input{font:inherit;color:var(--ink)}select{border:1px solid var(--line);border-radius:8px;padding:8px;min-height:40px;background:white}select[multiple]{height:120px}input[type=range]{width:100%;accent-color:var(--blue)}.query-line{display:flex;justify-content:space-between;color:var(--muted);font-size:.86rem}
.cards{display:grid;grid-template-columns:repeat(6,minmax(130px,1fr));gap:12px;margin-bottom:16px}.card{padding:13px 15px}.card span{display:block;color:var(--muted);font-size:.78rem}.card strong{display:block;font-size:1.22rem;margin-top:3px;overflow-wrap:anywhere}
.panel{padding:16px;margin-bottom:16px}.chart-wrap{width:100%;overflow:hidden}svg{display:block;width:100%;height:auto;min-height:300px}.chart-legend{color:var(--muted);font-size:.83rem;margin-top:8px}
.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:.84rem;white-space:nowrap}th,td{text-align:right;padding:8px 9px;border-bottom:1px solid #e8edf3}th{position:sticky;top:0;background:#f7f9fc;color:#526174}th:first-child,td:first-child{text-align:left}
.small{font-size:.82rem;color:var(--muted)}.error{color:#a3362d}.pill{display:inline-block;padding:2px 7px;border-radius:99px;background:#edf3f8;font-size:.76rem}
@media(max-width:980px){.filters{grid-template-columns:repeat(2,minmax(160px,1fr))}.cards{grid-template-columns:repeat(3,minmax(120px,1fr))}}
@media(max-width:560px){main{padding:18px 12px}.filters{grid-template-columns:1fr}.cards{grid-template-columns:repeat(2,minmax(120px,1fr))}}
</style>
</head>
<body><main>
<h1>Materials discovery experiment dashboard</h1>
<p class="sub">Compare saved runs by method, model, seed and query budget. Curves use the observed-hull trajectory; recorded LLM usage is shown separately from oracle queries.</p>
<div class="note"><strong>Cost interpretation:</strong> Oracle queries and LLM API requests are different budgets. The dashboard reports API calls, prompt/completion tokens and available runtime/GPU-hours. Dollar costs were not recorded in the run data, so no monetary estimates are shown. Runtime uses Slurm elapsed time where available; otherwise it is an estimate from the timestamp span in the logs.</div>
<div class="note"><strong>Branch-search comparison:</strong> Branch-search state-Q uses RandomGenerator with the ORB oracle and no LLM. Its five 60-query runs are shown separately; this is a different search workflow from the LLM-controlled FARS methods, so the overlay is descriptive rather than a like-for-like comparison.</div>
<div class="note"><strong>Included runs:</strong> Campaigns with exactly 20 queries are excluded from this dashboard. Other run budgets are retained and grouped separately.</div>
<section class="panel filters">
 <div class="field"><label for="methodFilter">Method (multi-select)</label><select id="methodFilter" multiple></select></div>
 <div class="field"><label for="modelFilter">Model (multi-select)</label><select id="modelFilter" multiple></select></div>
 <div class="field"><label for="runFilter">Individual run</label><select id="runFilter"></select></div>
 <div class="field"><label for="queryLimit">Show through query</label><div class="query-line"><span>Query cutoff</span><strong id="queryValue"></strong></div><input id="queryLimit" type="range" min="1" value="60"></div>
</section>
<section class="cards" id="cards"></section>
<section class="panel"><h2>Discovery trajectory</h2><div class="chart-wrap" id="chart"></div><div class="chart-legend" id="chartNote"></div></section>
<section class="panel"><h2>Run statistics</h2><p class="small" id="tableNote"></p><div class="table-wrap" id="table"></div></section>
<p class="small" id="recordCount"></p>
</main>
<script>
const DATA = __DATA__;
const COLORS = {"FARS":"#123B63","FARS-MQ":"#8FC4E3","LLM-only control":"#7d8798","Random generator":"#FF8C00","Chemeleon":"#FFC04D","Branch-search state-Q":"#26934a","Other":"#8b5e83"};
const methodSelect=document.getElementById("methodFilter"),modelSelect=document.getElementById("modelFilter"),runSelect=document.getElementById("runFilter"),queryInput=document.getElementById("queryLimit");
function options(select,values){select.innerHTML=values.map(v=>`<option value="${esc(v)}" selected>${esc(v)}</option>`).join("")}
function esc(s){return String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}
function selected(select){return Array.from(select.selectedOptions).map(o=>o.value)}
function fmt(n,d=2){return n===null||n===undefined||!Number.isFinite(Number(n))?"Not recorded":Number(n).toLocaleString(undefined,{maximumFractionDigits:d})}
function sumAt(curve,q){return curve&&curve.length?curve[Math.min(q,curve.length)-1]??null:null}
function areaAt(curve,q){if(!curve||!curve.length||q<1)return null;let area=0,prev=0;for(let i=0;i<Math.min(q,curve.length);i++){area+=(prev+curve[i])/2;prev=curve[i]}return area}
const methods=[...new Set(DATA.runs.map(r=>r.method))].sort(),models=[...new Set(DATA.runs.map(r=>r.model))].sort();
options(methodSelect,methods);options(modelSelect,models);
queryInput.max=Math.max(1,...DATA.runs.map(r=>r.queries));queryInput.value=queryInput.max;
function filtered(){const ms=selected(methodSelect),mods=selected(modelSelect);return DATA.runs.filter(r=>ms.includes(r.method)&&mods.includes(r.model))}
function drawChart(runs,q,single){const el=document.getElementById("chart"),w=Math.max(600,el.clientWidth||900),h=420,pl=64,pr=22,pt=18,pb=48,pw=w-pl-pr,ph=h-pt-pb;
 const series=[];if(single){const r=runs.find(x=>x.id===single);if(r&&r.curve)series.push({name:r.label,color:COLORS[r.method]||COLORS.Other,model:r.model,mean:r.curve.slice(0,q),lo:null,hi:null,count:1})}
 else{const groups={};runs.filter(r=>r.curve).forEach(r=>{let key=`${r.method} — ${r.model} — ${r.queries} queries`;(groups[key]??=[]).push(r)});for(const [name,rs] of Object.entries(groups)){let n=Math.min(q,...rs.map(r=>r.curve.length)),mean=[],sd=[];for(let i=0;i<n;i++){let vals=rs.map(r=>r.curve[i]),m=vals.reduce((a,b)=>a+b,0)/vals.length;mean.push(m);sd.push(vals.length>1?Math.sqrt(vals.reduce((a,b)=>a+(b-m)**2,0)/(vals.length-1)):0)}let method=rs[0].method;series.push({name,color:COLORS[method]||COLORS.Other,model:rs[0].model,mean,lo:mean.map((v,i)=>Math.max(0,v-sd[i])),hi:mean.map((v,i)=>v+sd[i]),count:rs.length})}}
 let ymax=Math.max(1,...series.flatMap(s=>s.hi||s.mean));ymax=Math.ceil(ymax/5)*5;const X=i=>pl+(i/Math.max(1,q-1))*pw,Y=v=>pt+ph-(v/ymax)*ph;
 let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Discovery curves through query ${q}"><rect width="${w}" height="${h}" fill="white"/>`;
 for(let i=0;i<=4;i++){let v=ymax*i/4,y=Y(v);svg+=`<line x1="${pl}" y1="${y}" x2="${w-pr}" y2="${y}" stroke="#e4e9ef"/><text x="${pl-10}" y="${y+4}" text-anchor="end" fill="#637084" font-size="12">${v.toFixed(0)}</text>`}
 for(let i=0;i<=5;i++){let val=Math.round(q*i/5),x=pl+(val/Math.max(1,q))*pw;svg+=`<line x1="${x}" y1="${pt}" x2="${x}" y2="${pt+ph}" stroke="#eef1f5"/><text x="${x}" y="${h-17}" text-anchor="middle" fill="#637084" font-size="12">${val}</text>`}
 for(const s of series){if(s.lo){let poly=s.hi.map((v,i)=>`${X(i)},${Y(v)}`).concat(s.lo.map((_,i)=>{let j=s.lo.length-1-i;return `${X(j)},${Y(s.lo[j])}`})).join(" ");svg+=`<polygon points="${poly}" fill="${s.color}" opacity=".13"/>`}let pts=s.mean.map((v,i)=>`${X(i)},${Y(v)}`).join(" "),dash=s.model==="DeepSeek Flash"?"8 4":"";svg+=`<polyline points="${pts}" fill="none" stroke="${s.color}" stroke-width="2.6" stroke-dasharray="${dash}" stroke-linejoin="round" stroke-linecap="round"/>`}
 svg+=`<line x1="${pl}" y1="${pt+ph}" x2="${w-pr}" y2="${pt+ph}" stroke="#8290a1"/><text x="${pl+pw/2}" y="${h-1}" text-anchor="middle" fill="#46566b" font-size="13">Official oracle queries</text><text transform="translate(16 ${pt+ph/2}) rotate(-90)" text-anchor="middle" fill="#46566b" font-size="13">Cumulative mSUN</text></svg>`;
 const legend=series.map(s=>`<span style="display:inline-flex;align-items:center;gap:6px;margin:0 14px 6px 0"><i style="display:inline-block;width:18px;border-top:3px ${s.model==="DeepSeek Flash"?"dashed":"solid"} ${s.color}"></i>${esc(s.name)}${single?"":` (n=${s.count})`}</span>`).join("");
 el.innerHTML=series.length?`${svg}<div class="chart-legend">${legend}</div>`:'<p class="small">No discovery curves available for these filters.</p>';
 document.getElementById("chartNote").innerHTML=single?(runs.find(r=>r.id===single)?.curveProvisional?"Showing the selected in-progress run. Its line is provisional and may change when the final observed-hull analysis is available.":"Showing the selected run."):"Mean curves and ±1 sample standard deviation are grouped by method, model, and run query budget. 20-query pilot runs are excluded; other short runs remain separate by budget. DeepSeek curves are dashed; n is the run count.";
}
function drawCards(runs,q,single){let target=single?runs.find(r=>r.id===single):null;let active=target?[target]:runs;let ms=active.map(r=>sumAt(r.curve,q)).filter(Number.isFinite),tok=active.map(r=>r.tokens.tokens).filter(Number.isFinite),calls=active.map(r=>r.tokens.calls).filter(Number.isFinite),hours=active.map(r=>r.gpuHours).filter(Number.isFinite);let avg=a=>a.length?a.reduce((x,y)=>x+y,0)/a.length:null;
let items=[["Runs shown",active.length],["mSUN by query "+q,target?fmt(sumAt(target.curve,q),0):fmt(avg(ms),1)+(ms.length?" mean":"")],["Final mSUN",target?fmt(target.metrics.msun,0):fmt(avg(active.map(r=>r.metrics.msun).filter(Number.isFinite)),1)+(active.length?" mean":"")],["LLM tokens",target?fmt(target.tokens.tokens,0):fmt(avg(tok),0)+(tok.length?" mean":"")],["LLM API calls",target?fmt(target.tokens.calls,0):fmt(avg(calls),1)+(calls.length?" mean":"")],["GPU-hours",target?fmt(target.gpuHours,3):fmt(avg(hours),2)+(hours.length?" mean":"")]];
document.getElementById("cards").innerHTML=items.map(([a,b])=>`<div class="card"><span>${esc(a)}</span><strong>${esc(b)}</strong></div>`).join("")}
function drawTable(runs,q){
 const available=v=>v!==null&&v!==undefined&&v!==""&&Number.isFinite(Number(v));
 const columns=[
  {label:"Model",value:r=>r.model},
  {label:"System",value:r=>r.system},
  {label:"Seed",value:r=>r.seed??"n/a"},
  {label:"Job",value:r=>r.job??"n/a"},
  {label:"Status",value:r=>r.status},
  {label:"Queries shown/total",value:r=>`${fmt(Math.min(q,r.queries),0)}/${fmt(r.queries,0)}`},
  {label:"mSUN at cutoff",value:r=>sumAt(r.curve,q),format:v=>fmt(v,0),measured:true},
  {label:"Final mSUN",value:r=>r.metrics.msun,format:v=>fmt(v,0),measured:true},
  {label:"AUDC to cutoff",value:r=>areaAt(r.curve,q),format:v=>fmt(v,1),measured:true},
  {label:"Final AUDC",value:r=>r.metrics.audc,format:v=>fmt(v,1),measured:true},
  {label:"API calls",value:r=>r.tokens.calls,format:v=>fmt(v,0),measured:true},
  {label:"Prompt tokens",value:r=>r.tokens.prompt,format:v=>fmt(v,0),measured:true},
  {label:"Completion tokens",value:r=>r.tokens.completion,format:v=>fmt(v,0),measured:true},
  {label:"Total tokens",value:r=>r.tokens.tokens,format:v=>fmt(v,0),measured:true},
  {label:"Runtime",value:r=>r.runtimeSeconds,format:(v,r)=>`${fmt(v/60,1)} min (${esc(r.runtimeSource||"timing")})`,measured:true},
  {label:"GPU-hours",value:r=>r.gpuHours,format:v=>fmt(v,3),measured:true},
  {label:"Generated candidates",value:r=>r.generated,format:v=>fmt(v,0),measured:true},
  {label:"Scored candidates",value:r=>r.scored,format:v=>fmt(v,0),measured:true},
  {label:"Valid candidates",value:r=>r.metrics.validity,format:v=>`${fmt(Number(v)*100,1)}%`,measured:true},
 ];
 const grouped=new Map();
 runs.forEach(r=>{if(!grouped.has(r.method))grouped.set(r.method,[]);grouped.get(r.method).push(r)});
 const tables=[...grouped].map(([method,methodRuns])=>{
  const visible=columns.filter(c=>!c.measured||methodRuns.some(r=>available(c.value(r))));
  const rows=methodRuns.map(r=>`<tr>${visible.map(c=>{
   const value=c.value(r);
   const rendered=available(value)?(c.format?c.format(value,r):value):"Not recorded";
   return `<td>${esc(rendered)}</td>`;
  }).join("")}</tr>`).join("");
  return `<h3>${esc(method)}</h3><table><thead><tr>${visible.map(c=>`<th>${esc(c.label)}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table>`;
 }).join("");
 document.getElementById("table").innerHTML=tables||'<p class="small">No runs match these filters.</p>';
 document.getElementById("tableNote").textContent="Run-statistic tables are separated by method. A measured column is omitted only from methods with no values for it; available columns are retained for other methods. AUDC at cutoff is calculated from the displayed curve. In-progress values are provisional. USD cost was not recorded.";
document.getElementById("recordCount").textContent=`Loaded ${DATA.runs.length} saved experiment runs. ${DATA.errors.length?DATA.errors.length+" curve reconstruction(s) need attention; their run statistics remain listed.":""}`}
function render(){const q=Number(queryInput.value),runs=filtered(),old=runSelect.value,optionsHtml=['<option value="">All matching runs (mean by method)</option>',...runs.map(r=>`<option value="${esc(r.id)}">${esc(r.label)}</option>`)].join("");runSelect.innerHTML=optionsHtml;if(runs.some(r=>r.id===old))runSelect.value=old;const single=runSelect.value;document.getElementById("queryValue").textContent=q;drawChart(runs,q,single);drawCards(runs,q,single);drawTable(single?runs.filter(r=>r.id===single):runs,q)}
[methodSelect,modelSelect,runSelect,queryInput].forEach(el=>el.addEventListener("change",render));queryInput.addEventListener("input",render);window.addEventListener("resize",()=>render());render();
</script></body></html>"""


def main() -> None:
    records = build_records()
    errors = [
        {"run": record["id"], "error": record["curveError"]}
        for record in records
        if record["curveError"]
    ]
    payload = json.dumps({"runs": records, "errors": errors}, ensure_ascii=False)
    page = HTML.replace("__DATA__", payload.replace("</", "<\\/"))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(page, encoding="utf-8")
    print(f"Dashboard written to {OUTPUT} ({len(records)} runs)")
    for error in errors:
        print(f"Curve unavailable for {error['run']}: {error['error']}", file=sys.stderr)


if __name__ == "__main__":
    main()
