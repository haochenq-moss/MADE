"""FARS: an RL scheduler around MADE's existing LLM ReAct orchestrator."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core.structure import Structure

from made.agents import LLMReActOrchestratorAgent
from made.agents.base import Generator, Plan, Scorer, ScoreResult
from made.utils.convex_hull_utils import safe_e_above_hull
from made.utils.structure_hash import structure_hash

from .scheduler import ResearchAction, SchedulerObservation, TabularScheduler

logger = logging.getLogger(__name__)


class GenerationBudget:
    def __init__(self, per_query_cap: int, remaining_episode_cap: int):
        self.per_query_cap = per_query_cap
        self.remaining_episode_cap = remaining_episode_cap
        self.generated_this_query = 0

    def begin_query(self) -> None:
        self.generated_this_query = 0

    def available(self) -> int:
        return min(
            self.per_query_cap - self.generated_this_query,
            self.remaining_episode_cap,
        )

    def consume(self, count: int) -> None:
        if count > self.available():
            raise RuntimeError("FARS structure-generation cap exceeded.")
        self.generated_this_query += count
        self.remaining_episode_cap -= count


class ScoringBudget:
    def __init__(self, maximum: int):
        self.maximum = maximum
        self.seen_hashes: set[str] = set()

    def begin_query(self) -> None:
        self.seen_hashes.clear()

    def consume(self, candidates: list[Structure]) -> None:
        candidate_hashes = {structure_hash(candidate) for candidate in candidates}
        new_hashes = candidate_hashes - self.seen_hashes
        if len(self.seen_hashes) + len(new_hashes) > self.maximum:
            raise RuntimeError(
                f"FARS scoring cap is {self.maximum} distinct candidates per proposal."
            )
        self.seen_hashes.update(new_hashes)


class CappedGenerator(Generator):
    """Limit structure-generation calls without changing the underlying generator."""

    def __init__(
        self,
        generator: Generator,
        per_query_cap: int,
        remaining_episode_cap: int,
        budget: GenerationBudget | None = None,
        queried_compositions: set[str] | None = None,
        unqueried_buffer_available: bool = False,
    ):
        self.generator = generator
        self.budget = budget or GenerationBudget(per_query_cap, remaining_episode_cap)
        self.queried_compositions = queried_compositions
        self.unqueried_buffer_available = unqueried_buffer_available
        self.prioritized_composition_count = 0
        self.buffer_fallback_count = 0

    @property
    def remaining_episode_cap(self) -> int:
        return self.budget.remaining_episode_cap

    @property
    def generated_this_query(self) -> int:
        return self.budget.generated_this_query

    def begin_query(self) -> None:
        self.budget.begin_query()

    def generate(self, plan: Plan, state: dict[str, Any]) -> list[Structure]:
        remaining = self.budget.available()
        if remaining <= 0:
            raise RuntimeError("FARS structure-generation cap reached.")
        if not plan.compositions:
            raise ValueError("FARS generation requires explicit compositions.")

        selected_compositions = plan.compositions
        if self.queried_compositions is not None:
            unseen = [
                composition
                for composition in plan.compositions
                if composition.reduced_formula not in self.queried_compositions
            ]
            if unseen:
                selected_compositions = unseen
                self.prioritized_composition_count += len(unseen)
            elif self.unqueried_buffer_available:
                self.buffer_fallback_count += 1
                return []

        requested_per_composition = max(1, int(plan.num_candidates))
        selected_compositions = selected_compositions[:remaining]
        per_composition = min(
            requested_per_composition,
            max(1, remaining // len(selected_compositions)),
        )
        capped_plan = Plan(
            compositions=selected_compositions,
            num_candidates=per_composition,
            constraints=plan.constraints,
        )
        structures = self.generator.generate(capped_plan, state)
        if len(structures) > remaining:
            structures = structures[:remaining]
        self.budget.consume(len(structures))
        return structures

    def get_state(self) -> dict[str, Any]:
        return self.generator.get_state()

    def load_state(self, state: dict[str, Any]) -> None:
        load_state = getattr(self.generator, "load_state", None)
        if load_state is not None:
            load_state(state)

    def update_state(self, state: dict[str, Any]) -> None:
        self.generator.update_state(state)


class CappedScorer(Scorer):
    """Reject oversized surrogate batches rather than silently scoring a subset."""

    scorer_name = "FARS capped surrogate"

    def __init__(self, scorer: Scorer, budget: ScoringBudget):
        self.scorer = scorer
        self.budget = budget

    def _score_with_results(
        self, candidates: list[Structure], state: dict[str, Any]
    ) -> tuple[list[float], list[ScoreResult]]:
        self.budget.consume(candidates)
        return self.scorer.score_candidates(candidates, state, return_results=True)

    def get_state(self) -> dict[str, Any]:
        return self.scorer.get_state()

    def load_state(self, state: dict[str, Any]) -> None:
        self.scorer.load_state(state)

    def update_state(self, state: dict[str, Any]) -> None:
        self.scorer.update_state(state)


class CappedLLMOrchestratorControl(LLMReActOrchestratorAgent):
    """Matched LLM-only control with FARS's generation/scoring caps."""

    def __init__(
        self,
        *,
        max_candidates_per_query: int = 32,
        max_generated_per_episode: int = 640,
        max_scored_per_query: int = 32,
        controller_log_dir: str | None = None,
        **kwargs: Any,
    ):
        if max_candidates_per_query < 1 or max_generated_per_episode < 1:
            raise ValueError("Generation caps must be positive.")
        if max_scored_per_query < 1:
            raise ValueError("max_scored_per_query must be positive.")
        self._base_generators = dict(kwargs.get("generators") or {})
        self._generation_caps = (max_candidates_per_query, max_generated_per_episode)
        self._remaining_episode_cap = max_generated_per_episode
        self._max_scored_per_query = max_scored_per_query
        self._controller_log_dir = (
            Path(controller_log_dir) if controller_log_dir else None
        )
        self._run_id = uuid4().hex
        self._log_path: Path | None = None
        self._base_enabled_tools = list(kwargs.get("enabled_tools") or [])
        scorers = dict(kwargs.get("scorers") or {})
        self._scoring_budget = ScoringBudget(max_scored_per_query)
        kwargs["scorers"] = {
            name: CappedScorer(scorer, self._scoring_budget)
            for name, scorer in scorers.items()
        }
        super().__init__(**kwargs)
        self._base_generators = dict(self.generators)
        self._base_enabled_tools = list(self.enabled_tools)
        self._base_orchestration_instructions = self.signature_class.instructions

    def propose_composition_and_structure(
        self, state: dict[str, Any]
    ) -> tuple[Any, Structure]:
        per_query_cap, episode_cap = self._generation_caps
        budget = GenerationBudget(per_query_cap, self._remaining_episode_cap)
        budget.begin_query()
        self.generators = {
            name: CappedGenerator(generator, per_query_cap, episode_cap, budget)
            for name, generator in self._base_generators.items()
        }
        self._scoring_budget.begin_query()
        self.enabled_tools = [
            tool for tool in self._base_enabled_tools if tool != "create_structure"
        ]
        self.signature_class.instructions = (
            self._base_orchestration_instructions
            + "\n\nFor this matched control, use the enabled generation tools and "
            "candidate buffer; direct structure creation is unavailable. "
            "Generate no more than the configured candidate cap and select one "
            "candidate for the official query."
        )
        try:
            return super().propose_composition_and_structure(state)
        finally:
            self._remaining_episode_cap = budget.remaining_episode_cap
            self._write_control_log(
                {
                    "event": "proposal_work",
                    "run_id": self._run_id,
                    "query_count": int(state.get("query_count", 0)) + 1,
                    "policy": "llm_orchestrator_without_rl",
                    "generated_candidates": budget.generated_this_query,
                    "distinct_scored_candidates": len(self._scoring_budget.seen_hashes),
                    "react_max_iters": self.max_iters,
                }
            )
            self.generators = dict(self._base_generators)
            self.enabled_tools = list(self._base_enabled_tools)
            self.signature_class.instructions = self._base_orchestration_instructions

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["llm_control"] = {
            "remaining_episode_cap": self._remaining_episode_cap,
            "run_id": self._run_id,
        }
        return state

    def load_state(self, state: dict[str, Any]) -> None:
        parent_state = {
            key: value for key, value in state.items() if key != "llm_control"
        }
        super().load_state(parent_state)
        control_state = state.get("llm_control", {})
        self._remaining_episode_cap = int(
            control_state.get("remaining_episode_cap", self._remaining_episode_cap)
        )
        self._run_id = control_state.get("run_id", self._run_id)

    def _write_control_log(self, record: dict[str, Any]) -> None:
        if self._controller_log_dir is None:
            return
        if self._log_path is None:
            self._controller_log_dir.mkdir(parents=True, exist_ok=True)
            system = "-".join(self.chemical_system_elements) or "unknown-system"
            self._log_path = self._controller_log_dir / f"{system}-{self._run_id}.jsonl"
        with self._log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, sort_keys=True) + "\n")


class FARSAgent(LLMReActOrchestratorAgent):
    """Keep MADE's ReAct tool loop; use a small controller to choose its research mode."""

    def __init__(
        self,
        *,
        scheduler_mode: str = "q_learning",
        fixed_action: str = "EXPLORE",
        controller_memory: str = "full",
        candidate_selection_policy: str = "llm",
        scheduler_seed: int = 0,
        learning_rate: float = 0.2,
        discount: float = 0.9,
        epsilon: float = 0.1,
        train_scheduler: bool = False,
        q_values: Mapping[str, Mapping[str, float]] | None = None,
        max_candidates_per_query: int = 32,
        max_generated_per_episode: int = 640,
        max_scored_per_query: int = 32,
        episode_budget: int = 20,
        controller_log_dir: str | None = None,
        q_values_path: str | None = None,
        initial_q_values_path: str | None = None,
        **kwargs: Any,
    ):
        if controller_memory not in {"full", "compact", "none"}:
            raise ValueError("controller_memory must be full, compact, or none.")
        if candidate_selection_policy not in {"llm", "balanced_formula"}:
            raise ValueError(
                "candidate_selection_policy must be llm or balanced_formula."
            )
        if max_candidates_per_query < 1 or max_generated_per_episode < 1:
            raise ValueError("Generation caps must be positive.")
        if max_scored_per_query < 1:
            raise ValueError("max_scored_per_query must be positive.")
        if episode_budget < 1:
            raise ValueError("episode_budget must be positive.")

        requested_generators = kwargs.get("generators") or {}
        self._all_generators = {
            name: generator
            for name, generator in dict(requested_generators).items()
            if generator is not None
        }
        kwargs["generators"] = self._all_generators
        self._per_query_generation_cap = max_candidates_per_query
        self._episode_generation_cap = max_generated_per_episode
        self._max_scored_per_query = max_scored_per_query
        self._episode_budget = episode_budget
        self._q_values_path = Path(q_values_path) if q_values_path else None
        self._initial_q_values_path = (
            Path(initial_q_values_path) if initial_q_values_path else None
        )
        self._scheduler_mode = scheduler_mode
        self._fixed_action = ResearchAction(fixed_action)
        self._controller_memory_mode = controller_memory
        self._candidate_selection_policy = candidate_selection_policy
        self._scheduler = TabularScheduler(
            seed=scheduler_seed,
            learning_rate=learning_rate,
            discount=discount,
            epsilon=epsilon,
            training=train_scheduler,
            q_values=q_values,
        )
        if (
            self._initial_q_values_path is not None
            and self._q_values_path is not None
            and self._initial_q_values_path.resolve() == self._q_values_path.resolve()
        ):
            raise ValueError("Initial Q-table and run output path must be different.")
        if self._initial_q_values_path is not None:
            if not self._initial_q_values_path.is_file():
                raise FileNotFoundError(
                    f"Initial Q-table does not exist: {self._initial_q_values_path}"
                )
            with self._initial_q_values_path.open(encoding="utf-8") as weights_file:
                initial_values = json.load(weights_file)
            if not isinstance(initial_values, Mapping) or not initial_values:
                raise ValueError("Initial Q-table must be a non-empty state mapping.")
            self._scheduler.load_state(initial_values)
        elif self._q_values_path is not None and self._q_values_path.exists():
            with self._q_values_path.open(encoding="utf-8") as weights_file:
                self._scheduler.load_state(json.load(weights_file))
        self._initial_q_values = self._scheduler.get_state()
        self._warm_start_decision_keys: list[str] = []
        self._controller_history: list[dict[str, Any]] = []
        self._objective_ledger: list[dict[str, Any]] = []
        self._last_query_count = 0
        self._previous_objective = 0
        self._pending_transition: dict[str, Any] | None = None
        self._run_id = uuid4().hex
        self._controller_log_dir = (
            Path(controller_log_dir) if controller_log_dir else None
        )
        self._log_path: Path | None = None
        self._decision_records: list[dict[str, Any]] = []

        scorers = dict(kwargs.get("scorers") or {})
        self._scoring_budget = ScoringBudget(max_scored_per_query)
        kwargs["scorers"] = {
            name: CappedScorer(scorer, self._scoring_budget)
            for name, scorer in scorers.items()
        }
        super().__init__(**kwargs)
        self._all_generators = dict(self.generators)
        self._base_orchestration_instructions = self.signature_class.instructions

    def propose_composition_and_structure(
        self, state: dict[str, Any]
    ) -> tuple[Any, Structure]:
        observation = self._make_scheduler_observation(state)
        feasible = self._feasible_actions()
        selected_action = self._scheduler.choose(
            observation,
            feasible,
            mode=self._scheduler_mode,
            fixed_action=self._fixed_action,
        )

        generator_name = self._generator_for(selected_action)
        generation_budget = GenerationBudget(
            self._per_query_generation_cap,
            self._episode_generation_cap,
        )
        queried_compositions = {
            str(row["composition"])
            for row in self._controller_history
            if row.get("composition") and row["composition"] != "?"
        }
        visible_buffer = dict(self.buffer)
        hidden_buffer: dict[str, list[dict[str, Any]]] = {}
        diversity_fallback = False
        if selected_action == ResearchAction.EXPLORE:
            visible_buffer, hidden_buffer, diversity_fallback = (
                self._partition_exploration_buffer(
                    self.buffer, queried_compositions
                )
            )
            if hidden_buffer:
                self.buffer.clear()
                self.buffer.update(visible_buffer)
        capped_generator = CappedGenerator(
            self._all_generators[generator_name],
            self._per_query_generation_cap,
            self._episode_generation_cap,
            generation_budget,
            queried_compositions=(
                queried_compositions
                if selected_action == ResearchAction.EXPLORE
                else None
            ),
            unqueried_buffer_available=bool(hidden_buffer),
        )
        capped_generator.begin_query()

        self.generators = {generator_name: capped_generator}
        self._scoring_budget.begin_query()
        self.enabled_tools = [
            tool for tool in self.enabled_tools if tool != "create_structure"
        ]
        original_prompt = self.context_config.get("orchestration_prompt", "")
        unqueried_compositions = sorted(
            composition
            for composition in self.buffer
            if composition not in queried_compositions
        )
        mode_instructions = (
            f"\n\nFARS CONTROLLER MODE: {selected_action.value}.\n"
            + self._mode_instruction(selected_action)
            + "\nDo not use unsupported actions. Generate at most "
            + str(self._per_query_generation_cap)
            + " candidates this proposal. Select exactly one buffered structure "
            + "for the official environment to query.\n"
            + "Scoring is optional and has a strict limit of "
            + str(self._max_scored_per_query)
            + " distinct candidates for this proposal. MADE's score_buffer scores "
            + "every buffered structure for the selected composition (or the whole "
            + "buffer if composition is omitted). Check buffer size first. Call "
            + "score_buffer only when the complete requested batch fits within the "
            + "remaining scoring allowance; otherwise do not score that batch and "
            + "select using the available buffer/query tools. Do not retry an "
            + "oversized scoring request or score the same buffer repeatedly."
        )
        if selected_action == ResearchAction.EXPLORE:
            mode_instructions += (
                "\nIn EXPLORE mode, prioritize reduced compositions not yet queried. "
                "Previously queried reduced compositions: "
                + (", ".join(sorted(queried_compositions)) or "none")
                + ". Unqueried filter-approved compositions currently in the buffer: "
                + (", ".join(unqueried_compositions) or "none")
                + ". Select a previously queried composition only when no unqueried "
                "buffer composition or proposed composition is available."
            )
        mode_instructions += self._candidate_selection_guidance()

        state_key = observation.key()
        self._warm_start_decision_keys.append(state_key)
        exact_q_match = state_key in self._initial_q_values
        self._pending_transition = {
            "state_key": state_key,
            "action": selected_action.value,
            "query_count": int(state.get("query_count", 0)) + 1,
        }
        record = {
            "run_id": self._run_id,
            "decision_query": int(state.get("query_count", 0)) + 1,
            "controller_action": selected_action.value,
            "candidate_selection_policy": self._candidate_selection_policy,
            "scheduler_state": state_key,
            "warm_start_source": (
                str(self._initial_q_values_path)
                if self._initial_q_values_path is not None
                else None
            ),
            "warm_start_table_state_count": len(self._initial_q_values),
            "warm_start_exact_match": exact_q_match,
            "warm_start_q_values": dict(
                self._initial_q_values.get(state_key, {})
            ),
            "warm_start_coverage": SchedulerObservation.exact_transfer_coverage(
                self._initial_q_values, self._warm_start_decision_keys
            ),
            "feasible_actions": [action.value for action in feasible],
            "generator": generator_name,
            "controller_memory": self._controller_memory_mode,
            "provenance": self._provenance(state),
            "hidden_ground_truth_read": False,
        }
        self._decision_records.append(record)
        self._write_log({"event": "decision", **record})

        self.context_config["orchestration_prompt"] = (
            original_prompt + mode_instructions
        )
        self.signature_class.instructions = (
            self._base_orchestration_instructions + mode_instructions
        )
        result: tuple[Any, Structure] | None = None
        try:
            result = super().propose_composition_and_structure(state)
            return result
        finally:
            self._episode_generation_cap = capped_generator.remaining_episode_cap
            self.context_config["orchestration_prompt"] = original_prompt
            self.signature_class.instructions = self._base_orchestration_instructions
            if hidden_buffer:
                self.buffer.update(hidden_buffer)
            selected_composition = (
                result[1].composition.reduced_formula if result is not None else None
            )
            self._write_log(
                {
                    "event": "proposal_work",
                    "run_id": self._run_id,
                    "decision_query": int(state.get("query_count", 0)) + 1,
                    "generated_candidates": capped_generator.generated_this_query,
                    "distinct_scored_candidates": len(self._scoring_budget.seen_hashes),
                    "react_max_iters": self.max_iters,
                    "selected_composition": selected_composition,
                    "candidate_selection_policy": self._candidate_selection_policy,
                    "unqueried_buffer_compositions": unqueried_compositions,
                    "no_unqueried_buffer_at_start": diversity_fallback,
                    "composition_diversity_fallback": (
                        selected_action == ResearchAction.EXPLORE
                        and selected_composition in queried_compositions
                        and not hidden_buffer
                        and capped_generator.prioritized_composition_count == 0
                    ),
                    "prioritized_generation_compositions": (
                        capped_generator.prioritized_composition_count
                    ),
                    "generation_deferred_to_buffer": (
                        capped_generator.buffer_fallback_count
                    ),
                }
            )

    def _candidate_selection_guidance(self) -> str:
        if self._candidate_selection_policy == "llm":
            return ""
        if self._candidate_selection_policy != "balanced_formula":
            raise ValueError(
                "Unsupported candidate selection policy: "
                f"{self._candidate_selection_policy}"
            )

        queried_counts: dict[str, int] = {}
        for row in self._controller_history:
            composition = row.get("composition")
            if composition and composition != "?":
                queried_counts[composition] = queried_counts.get(composition, 0) + 1
        counts = ", ".join(
            f"{composition}={count}"
            for composition, count in sorted(queried_counts.items())
        )
        return (
            "\n\nCandidate-selection tie-break: among valid structures currently "
            "in the buffer with comparable available surrogate scores and "
            "consistent with the active FARS mode, mildly prefer a reduced "
            "composition with fewer prior official queries in this run. Do not "
            "reject a candidate solely because its composition was queried "
            "before, and do not prefer a lower-scoring candidate when the "
            "available score or active REFINE objective clearly favors another. "
            "Prior official query counts by reduced formula are: "
            + (counts or "no formulas queried yet")
            + ". A formula not listed has zero prior queries. These counts "
            "describe prior queried proposals, not hidden ground-truth labels."
        )

    def update_state(self, state: dict[str, Any]) -> None:
        super().update_state(state)
        query_count = int(state.get("query_count", 0))
        last_observation = state.get("last_observation")
        if query_count <= self._last_query_count or not last_observation:
            return

        proposal = last_observation.get("proposal")
        if isinstance(proposal, dict):
            structure = Structure.from_dict(proposal)
        elif isinstance(proposal, Structure):
            structure = proposal
        else:
            raise ValueError("Observed MADE query has no readable proposal structure.")

        self._objective_ledger.append(
            {
                "structure_hash": structure_hash(structure),
                "is_newly_discovered": bool(
                    last_observation.get("is_newly_discovered", False)
                ),
            }
        )
        objective = self._observed_msun_count(state)
        current_hash = self._objective_ledger[-1]["structure_hash"]
        self._controller_history.append(
            {
                "query_count": query_count,
                "composition": last_observation.get("reduced_formula", "?"),
                "e_above_hull_at_query": float(
                    last_observation.get("e_above_hull", float("inf"))
                ),
                "is_newly_discovered": self._objective_ledger[-1][
                    "is_newly_discovered"
                ],
                "stable_novel": current_hash
                in self._observed_stable_novel_hashes(state),
                "discovery_increment": objective - self._previous_objective,
            }
        )
        reward = objective - self._previous_objective
        self._write_log(
            {
                "event": "observed_outcome",
                "run_id": self._run_id,
                "query_count": query_count,
                "composition": last_observation.get("reduced_formula", "?"),
                "e_above_hull_at_query": float(
                    last_observation.get("e_above_hull", float("inf"))
                ),
                "is_stable_at_query": bool(last_observation.get("is_stable", False)),
                "is_newly_discovered": self._objective_ledger[-1][
                    "is_newly_discovered"
                ],
                "observed_msun_count": objective,
                "signed_msun_increment": reward,
                "outcome_source": "official post-query observation and observed phase diagram",
            }
        )
        self._previous_objective = objective

        if self._pending_transition is not None and self._scheduler.training:
            reward_for_update = float(reward)
            self._scheduler.update(
                self._pending_transition["state_key"],
                ResearchAction(self._pending_transition["action"]),
                reward_for_update,
                self._make_scheduler_observation(state).key(),
                self._feasible_actions(),
            )
            self._save_q_values()
            self._write_log(
                {
                    "event": "transition",
                    "run_id": self._run_id,
                    "query_count": query_count,
                    "reward": reward_for_update,
                    "observed_msun_count": objective,
                    "reward_source": "recomputed from observed phase diagram",
                }
            )
        self._last_query_count = query_count

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["fars"] = {
            "run_id": self._run_id,
            "scheduler_q_values": self._scheduler.get_state(),
            "warm_start_decision_keys": self._warm_start_decision_keys,
            "controller_history": self._controller_history,
            "objective_ledger": self._objective_ledger,
            "last_query_count": self._last_query_count,
            "previous_objective": self._previous_objective,
            "pending_transition": self._pending_transition,
            "decision_records": self._decision_records,
            "episode_generation_cap": self._episode_generation_cap,
        }
        return state

    def load_state(self, state: dict[str, Any]) -> None:
        parent_state = {key: value for key, value in state.items() if key != "fars"}
        super().load_state(parent_state)
        fars = state.get("fars", {})
        self._scheduler.load_state(fars.get("scheduler_q_values", {}))
        self._warm_start_decision_keys = fars.get("warm_start_decision_keys", [])
        self._controller_history = fars.get("controller_history", [])
        self._objective_ledger = fars.get("objective_ledger", [])
        self._last_query_count = int(fars.get("last_query_count", self.last_step))
        self._previous_objective = int(fars.get("previous_objective", 0))
        self._pending_transition = fars.get("pending_transition")
        self._decision_records = fars.get("decision_records", [])
        self._run_id = fars.get("run_id", self._run_id)
        self._episode_generation_cap = int(
            fars.get("episode_generation_cap", self._episode_generation_cap)
        )

    def _make_scheduler_observation(
        self, state: dict[str, Any]
    ) -> SchedulerObservation:
        query_count = int(state.get("query_count", 0))
        remaining = max(0, self._episode_budget - query_count)
        history = (
            self._controller_history if self._controller_memory_mode != "none" else []
        )
        recent = history[-5:]
        recent_yield = (
            sum(bool(row.get("stable_novel", False)) for row in recent) / len(recent)
            if recent
            else 0.0
        )
        queries_since_discovery = 0
        for row in reversed(history):
            if row.get("stable_novel", False):
                break
            queries_since_discovery += 1

        buffer_size = sum(len(entries) for entries in self.buffer.values())
        if self._controller_memory_mode == "compact":
            visited_regions = len(
                {row.get("composition") for row in self._controller_history[-10:]}
            )
        elif self._controller_memory_mode == "full":
            visited_regions = len(
                {row.get("composition") for row in self._controller_history}
            )
        else:
            visited_regions = 0

        return SchedulerObservation(
            remaining_fraction=remaining / self._episode_budget,
            recent_yield=recent_yield,
            queries_since_discovery=queries_since_discovery,
            buffer_size=buffer_size,
            visited_regions=visited_regions,
            query_count=query_count,
        )

    @staticmethod
    def _partition_exploration_buffer(
        buffer: Mapping[str, list[dict[str, Any]]],
        queried_compositions: set[str],
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], bool]:
        unseen = {
            composition: entries
            for composition, entries in buffer.items()
            if composition not in queried_compositions
        }
        if unseen:
            repeated = {
                composition: entries
                for composition, entries in buffer.items()
                if composition in queried_compositions
            }
            return unseen, repeated, False
        return dict(buffer), {}, True

    def _feasible_actions(self) -> tuple[ResearchAction, ...]:
        actions: list[ResearchAction] = []
        if self._all_generators:
            actions.append(ResearchAction.EXPLORE)
        if "chemeleon" in self._all_generators:
            actions.append(ResearchAction.REFINE)
        return tuple(actions)

    def _generator_for(self, action: ResearchAction) -> str:
        if action == ResearchAction.REFINE:
            return "chemeleon"
        if "random" in self._all_generators:
            return "random"
        return next(iter(self._all_generators))

    @staticmethod
    def _mode_instruction(action: ResearchAction) -> str:
        if action == ResearchAction.EXPLORE:
            return (
                "Explore: prioritize valid compositions not yet queried and "
                "composition-space coverage. Use only the enabled generator."
            )
        return (
            "Refine: condition Chemeleon generation on compositions with prior "
            "observed stable novel outcomes, while varying the structure."
        )

    def _observed_msun_count(self, state: dict[str, Any]) -> int:
        return len(self._observed_stable_novel_hashes(state))

    def _observed_stable_novel_hashes(self, state: dict[str, Any]) -> set[str]:
        raw_entries = state.get("phase_diagram_all_entries")
        if not isinstance(raw_entries, list):
            raise ValueError(
                "MADE state is missing its observed phase-diagram entries."
            )
        entries = [
            PDEntry.from_dict(entry) if isinstance(entry, dict) else entry
            for entry in raw_entries
        ]
        phase_diagram = PhaseDiagram(entries)
        by_hash: dict[str, PDEntry] = {}
        for entry in phase_diagram.all_entries:
            structure_data = getattr(entry, "attribute", None) or {}
            if "structure" in structure_data:
                candidate = Structure.from_dict(structure_data["structure"])
                by_hash[structure_hash(candidate)] = entry

        tolerance = float(state.get("stability_tolerance", 0.1))
        stable_hashes: set[str] = set()
        for record in self._objective_ledger:
            if not record["is_newly_discovered"]:
                continue
            entry = by_hash.get(record["structure_hash"])
            if (
                entry is not None
                and safe_e_above_hull(phase_diagram, entry) <= tolerance
            ):
                stable_hashes.add(record["structure_hash"])
        return stable_hashes

    def _provenance(self, state: Mapping[str, Any]) -> dict[str, Any]:
        observation = state.get("last_observation") or {}
        return {
            "query_count": int(state.get("query_count", 0)),
            "last_observation_fields": sorted(observation.keys()),
            "observed_phase_diagram_entry_count": len(
                state.get("phase_diagram_all_entries", [])
            ),
            "controller_history_length": len(self._controller_history),
            "buffer_composition_count": len(self.buffer),
            "sources": [
                "env.get_state.query_count",
                "env.get_state.last_observation",
                "env.get_state.phase_diagram_all_entries",
                "agent.candidate_buffer",
            ],
        }

    def _write_log(self, record: dict[str, Any]) -> None:
        if self._controller_log_dir is None:
            return
        if self._log_path is None:
            self._controller_log_dir.mkdir(parents=True, exist_ok=True)
            system = "-".join(self.chemical_system_elements) or "unknown-system"
            self._log_path = self._controller_log_dir / f"{system}-{self._run_id}.jsonl"
        with self._log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, sort_keys=True) + "\n")

    def _save_q_values(self) -> None:
        if self._q_values_path is None:
            return
        self._q_values_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._q_values_path.with_suffix(".tmp")
        with temporary_path.open("w", encoding="utf-8") as weights_file:
            json.dump(
                self._scheduler.get_state(), weights_file, sort_keys=True, indent=2
            )
        os.replace(temporary_path, self._q_values_path)
