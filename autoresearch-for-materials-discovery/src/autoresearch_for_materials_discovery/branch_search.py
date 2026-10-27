"""Observed-feedback branch search with MADE ReAct slate selection."""

from __future__ import annotations

import json
import logging
import os
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from pymatgen.core import Composition, Structure

from made.agents import LLMReActOrchestratorAgent
from made.agents.llm_react_orchestrator import (
    OrchestratorReActSignature,
    OrchestratorTools,
)
from made.agents.planners.grid_search import GridSearchPlanner

from .agent import CappedGenerator, GenerationBudget

logger = logging.getLogger(__name__)


class BranchSlateSelectionSignature(OrchestratorReActSignature):
    """ReAct selector restricted to structures in a pre-generated branch slate."""

    __doc__ = """You select one structure from an already generated candidate slate.

Do not generate structures, propose compositions, or use hidden ground-truth
labels. Use only the available buffer-inspection and selection tools. Select
exactly one structure from the current slate for the next official oracle
query. Prefer a candidate only when the observed history or its visible
structure details support that choice. If no ranking evidence is available,
select a valid candidate from the slate."""


class BranchSearchReActAgent(LLMReActOrchestratorAgent):
    """Learn branch/action values and use MADE ReAct only to select a slate item."""

    def __init__(
        self,
        *,
        scheduler_seed: int = 0,
        learning_rate: float = 0.2,
        discount: float = 0.9,
        epsilon: float = 0.1,
        episode_budget: int = 60,
        max_candidates_per_query: int = 32,
        max_generated_per_episode: int = 1920,
        max_stoichiometry: int = 20,
        num_fraction_bins: int = 5,
        controller_log_dir: str | None = None,
        q_values_path: str | None = None,
        q_values: Mapping[str, Mapping[str, float]] | None = None,
        **kwargs: Any,
    ):
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1].")
        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be in [0, 1].")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")
        if episode_budget < 1:
            raise ValueError("episode_budget must be positive.")
        if max_candidates_per_query < 1 or max_generated_per_episode < 1:
            raise ValueError("Candidate-generation caps must be positive.")
        if max_stoichiometry < 2:
            raise ValueError("max_stoichiometry must be at least 2.")
        if num_fraction_bins < 2:
            raise ValueError("num_fraction_bins must be at least 2.")

        generators = {
            name: generator
            for name, generator in dict(kwargs.get("generators") or {}).items()
            if generator is not None
        }
        if "random" not in generators:
            raise ValueError("Branch search requires MADE's 'random' generator.")
        kwargs["generators"] = generators
        super().__init__(**kwargs)

        self.signature_class = BranchSlateSelectionSignature
        self.enabled_tools = [
            "list_compositions",
            "query_structures",
            "get_buffer_stats",
            "select_for_evaluation",
        ]
        self._base_generators = dict(self.generators)
        self._episode_budget = episode_budget
        self._per_query_generation_cap = max_candidates_per_query
        self._episode_generation_cap = max_generated_per_episode
        self._max_stoichiometry = max_stoichiometry
        self._num_fraction_bins = num_fraction_bins
        self._learning_rate = learning_rate
        self._discount = discount
        self._epsilon = epsilon
        self._rng = random.Random(scheduler_seed)
        self._q_values = {
            str(state): {str(action): float(value) for action, value in actions.items()}
            for state, actions in (q_values or {}).items()
        }
        self._q_values_path = Path(q_values_path) if q_values_path else None
        self._controller_log_dir = (
            Path(controller_log_dir) if controller_log_dir else None
        )
        self._run_id = uuid4().hex
        self._log_path: Path | None = None

        self._planner: GridSearchPlanner | None = None
        self._compositions: list[Composition] = []
        self._first_element: str | None = None
        self._branches: dict[str, dict[str, Any]] = {}
        self._assigned_bins: set[int] = set()
        self._reward_history: list[int] = []
        self._query_history: list[dict[str, Any]] = []
        self._queried_compositions: set[str] = set()
        self._last_query_count = 0
        self._last_branch_id: str | None = None
        self._pending_transition: dict[str, Any] | None = None
        self._decision_records: list[dict[str, Any]] = []
        self._proposal_count = 0

    def update_state(self, state: dict[str, Any]) -> None:
        super().update_state(state)
        query_count = int(state.get("query_count", 0))

        if self._planner is None:
            self._initialize_composition_planner(state)
            self._last_query_count = query_count
            return

        last_observation = state.get("last_observation")
        if query_count <= self._last_query_count or not last_observation:
            return

        reward = int(
            bool(last_observation.get("is_stable", False))
            and bool(last_observation.get("is_newly_discovered", False))
        )
        branch_id = (
            self._pending_transition.get("branch_id")
            if self._pending_transition is not None
            else None
        )
        branch = self._branches.get(branch_id) if branch_id else None
        if branch is None:
            raise RuntimeError(
                "Observed query has no associated branch-search decision."
            )

        branch["query_count"] += 1
        branch["rewards"].append(reward)
        self._reward_history.append(reward)
        self._last_branch_id = branch_id
        composition = str(last_observation.get("reduced_formula", "?"))
        if composition != "?":
            self._queried_compositions.add(composition)
        self._query_history.append(
            {
                "query_count": query_count,
                "branch_id": branch_id,
                "composition": composition,
                "reward": reward,
                "is_stable": bool(last_observation.get("is_stable", False)),
                "is_newly_discovered": bool(
                    last_observation.get("is_newly_discovered", False)
                ),
            }
        )

        next_state_key = self._state_key(state)
        next_actions = self._feasible_actions()
        if self._pending_transition is not None:
            self._update_q_values(
                self._pending_transition["state_key"],
                self._pending_transition["action"],
                reward,
                next_state_key,
                next_actions,
            )
            self._write_log(
                {
                    "event": "outcome",
                    "run_id": self._run_id,
                    "query": query_count,
                    "branch_id": branch_id,
                    "reward": reward,
                    "observed_stable": bool(last_observation.get("is_stable", False)),
                    "observed_new": bool(
                        last_observation.get("is_newly_discovered", False)
                    ),
                    "hidden_ground_truth_read": False,
                }
            )
            self._save_q_values()

        self._last_query_count = query_count
        self._pending_transition = None

    def propose_composition_and_structure(
        self, state: dict[str, Any]
    ) -> tuple[Any, Structure]:
        if self._planner is None or self._first_element is None:
            raise RuntimeError("Branch composition planner was not initialized.")

        feasible_actions = self._feasible_actions()
        if not feasible_actions:
            raise RuntimeError("No branch/action option is feasible.")
        state_key = self._state_key(state)
        selected_action = self._choose_action(state_key, feasible_actions)
        action, branch_id = selected_action.split(":", maxsplit=1)

        if action == "EXPAND":
            bin_index = int(branch_id.removeprefix("bin-"))
            branch_id = f"branch-{bin_index:02d}"
            self._branches[branch_id] = {
                "bin_index": bin_index,
                "depth": 1,
                "query_count": 0,
                "rewards": [],
            }
            self._assigned_bins.add(bin_index)

        branch = self._branches[branch_id]
        selected_compositions = self._select_compositions(branch, action)
        if not selected_compositions:
            raise RuntimeError(f"No planner compositions available for {branch_id}.")

        budget = GenerationBudget(
            self._per_query_generation_cap, self._episode_generation_cap
        )
        budget.begin_query()
        capped_generator = CappedGenerator(
            self._base_generators["random"],
            self._per_query_generation_cap,
            self._episode_generation_cap,
            budget,
        )
        self.generators = {"random": capped_generator}
        tools = OrchestratorTools(
            generators=self.generators,
            static_filter=self.static_filter,
            uniqueness_filter=self.uniqueness_filter,
            scorers={},
            chemical_system_elements=self.chemical_system_elements,
            max_stoichiometry=self.max_stoichiometry,
            buffer=self.buffer,
            structure_cache=self.structure_cache,
            state=state,
        )

        existing_branch_buffer = {
            composition: entries
            for composition, entries in self.buffer.items()
            if self._composition_in_branch(entries[0]["structure"].composition, branch)
        }
        other_buffer = {
            composition: entries
            for composition, entries in self.buffer.items()
            if composition not in existing_branch_buffer
        }
        existing_branch_hashes = {
            entry["hash"]
            for entries in existing_branch_buffer.values()
            for entry in entries
        }
        self.buffer.clear()
        self.buffer.update(existing_branch_buffer)
        generation_result = tools.generate_structures(
            "random",
            compositions=", ".join(
                composition.formula.replace(" ", "")
                for composition in selected_compositions
            ),
            num_candidates=1,
        )
        if generation_result.startswith("Error during generation:"):
            self.buffer.update(other_buffer)
            self.generators = self._base_generators
            raise RuntimeError(generation_result)
        new_slate = [
            entry
            for entries in self.buffer.values()
            for entry in entries
            if entry["hash"] not in existing_branch_hashes
        ]
        if new_slate:
            slate = new_slate[: self._per_query_generation_cap]
        else:
            slate = [
                entry
                for entries in existing_branch_buffer.values()
                for entry in entries
            ]
            if len(slate) > self._per_query_generation_cap:
                slate = self._rng.sample(slate, self._per_query_generation_cap)
        self.buffer.clear()
        for entry in slate:
            self.buffer.setdefault(entry["composition"], []).append(entry)
        if not self.buffer:
            self.buffer.update(other_buffer)
            self.generators = self._base_generators
            raise RuntimeError(
                "Branch slate is empty after generation and MADE candidate filters: "
                f"{generation_result}"
            )

        self.enabled_tools = [
            "list_compositions",
            "query_structures",
            "get_buffer_stats",
            "select_for_evaluation",
        ]
        self.signature_class = BranchSlateSelectionSignature
        self._pending_transition = {
            "state_key": state_key,
            "action": selected_action,
            "branch_id": branch_id,
        }
        query_number = int(state.get("query_count", 0)) + 1
        decision = {
            "event": "decision",
            "run_id": self._run_id,
            "decision_query": query_number,
            "state_key": state_key,
            "action": selected_action,
            "branch_id": branch_id,
            "branch_bin": branch["bin_index"],
            "branch_bounds": self._bin_bounds(branch["bin_index"]),
            "feasible_actions": feasible_actions,
            "candidate_compositions": [
                composition.reduced_formula for composition in selected_compositions
            ],
            "slate_size": sum(len(entries) for entries in self.buffer.values()),
            "generated_candidates": budget.generated_this_query,
            "llm_model": self.llm_config.get("model"),
            "hidden_ground_truth_read": False,
        }
        self._decision_records.append(decision)
        self._write_log(decision)

        original_buffer = self.buffer
        original_enabled_tools = self.enabled_tools
        original_generators = self.generators
        try:
            result = super().propose_composition_and_structure(state)
            return result
        finally:
            self._proposal_count += 1
            self._episode_generation_cap = budget.remaining_episode_cap
            self.buffer = original_buffer
            self.buffer.update(other_buffer)
            self.enabled_tools = original_enabled_tools
            self.generators = original_generators
            selected_composition = (
                result[1].composition.reduced_formula
                if "result" in locals() and result is not None
                else None
            )
            self._write_log(
                {
                    "event": "selection",
                    "run_id": self._run_id,
                    "decision_query": query_number,
                    "branch_id": branch_id,
                    "selected_composition": selected_composition,
                    "generated_candidates": budget.generated_this_query,
                    "hidden_ground_truth_read": False,
                }
            )

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["branch_search"] = {
            "q_values": self._q_values,
            "branches": self._branches,
            "assigned_bins": sorted(self._assigned_bins),
            "reward_history": self._reward_history,
            "query_history": self._query_history,
            "queried_compositions": sorted(self._queried_compositions),
            "last_query_count": self._last_query_count,
            "last_branch_id": self._last_branch_id,
            "pending_transition": self._pending_transition,
            "episode_generation_cap": self._episode_generation_cap,
            "proposal_count": self._proposal_count,
            "rng_state": self._rng.getstate(),
            "compositions": [
                composition.formula for composition in self._compositions
            ],
            "first_element": self._first_element,
        }
        return state

    def load_state(self, state: dict[str, Any]) -> None:
        super().load_state(state)
        branch_state = state.get("branch_search")
        if not isinstance(branch_state, dict):
            return
        self._q_values = {
            str(key): {str(action): float(value) for action, value in values.items()}
            for key, values in branch_state.get("q_values", {}).items()
        }
        self._branches = {
            str(key): {
                **value,
                "bin_index": int(value["bin_index"]),
                "depth": int(value.get("depth", 1)),
                "query_count": int(value.get("query_count", 0)),
                "rewards": list(value.get("rewards", [])),
            }
            for key, value in branch_state.get("branches", {}).items()
        }
        self._assigned_bins = {
            int(index) for index in branch_state.get("assigned_bins", [])
        }
        self._reward_history = [
            int(value) for value in branch_state.get("reward_history", [])
        ]
        self._query_history = list(branch_state.get("query_history", []))
        self._queried_compositions = set(
            branch_state.get("queried_compositions", [])
        )
        self._last_query_count = int(branch_state.get("last_query_count", 0))
        self._last_branch_id = branch_state.get("last_branch_id")
        self._pending_transition = branch_state.get("pending_transition")
        self._episode_generation_cap = int(
            branch_state.get(
                "episode_generation_cap", self._episode_generation_cap
            )
        )
        self._proposal_count = int(branch_state.get("proposal_count", 0))
        if "rng_state" in branch_state:
            self._rng.setstate(_nested_tuple(branch_state["rng_state"]))
        self._compositions = [
            Composition(formula)
            for formula in branch_state.get("compositions", [])
        ]
        self._first_element = branch_state.get("first_element")

    def _initialize_composition_planner(self, state: dict[str, Any]) -> None:
        elements = state.get("elements", [])
        if len(elements) < 2:
            raise ValueError("Branch search requires at least two system elements.")
        self._first_element = str(elements[0])
        self._planner = GridSearchPlanner(
            max_stoichiometry=self._max_stoichiometry,
            num_compositions=1,
            num_candidates=1,
            seed=0,
            selection_strategy="random",
            filter_by_smact_validity=False,
            return_all_compositions=True,
        )
        self._planner.update_state(state)
        self._compositions = list(self._planner.propose(state).compositions)
        if not self._compositions:
            raise RuntimeError("MADE planner returned no candidate compositions.")

    def _feasible_actions(self) -> list[str]:
        actions = [
            f"EXPAND:bin-{index}"
            for index in range(self._num_fraction_bins)
            if index not in self._assigned_bins
            and any(
                self._composition_in_bin(composition, index)
                for composition in self._compositions
            )
        ]
        for branch_id, branch in self._branches.items():
            if any(
                self._composition_in_branch(composition, branch)
                for composition in self._compositions
            ):
                actions.append(f"EXPLORE:{branch_id}")
                if any(branch["rewards"]):
                    actions.append(f"EXPLOIT:{branch_id}")
        return actions

    def _choose_action(self, state_key: str, actions: list[str]) -> str:
        if self._rng.random() < self._epsilon:
            return self._rng.choice(actions)
        values = self._q_values.get(state_key, {})
        best = max(values.get(action, 0.0) for action in actions)
        return self._rng.choice(
            [action for action in actions if values.get(action, 0.0) == best]
        )

    def _update_q_values(
        self,
        state_key: str,
        action: str,
        reward: int,
        next_state_key: str,
        next_actions: list[str],
    ) -> None:
        state_values = self._q_values.setdefault(state_key, {})
        current = state_values.get(action, 0.0)
        future_values = self._q_values.get(next_state_key, {})
        future = (
            max(future_values.get(next_action, 0.0) for next_action in next_actions)
            if next_actions
            else 0.0
        )
        state_values[action] = current + self._learning_rate * (
            reward + self._discount * future - current
        )

    def _state_key(self, state: Mapping[str, Any]) -> str:
        query_count = int(state.get("query_count", 0))
        remaining_fraction = max(
            0.0, (self._episode_budget - query_count) / self._episode_budget
        )
        remaining_band = min(3, max(0, int(remaining_fraction * 4)))
        recent_rewards = self._reward_history[-5:]
        recent_yield = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
        yield_band = "high" if recent_yield >= 0.4 else "some" if recent_yield else "none"
        stagnation = 0
        for reward in reversed(self._reward_history):
            if reward:
                break
            stagnation += 1
        stagnation_band = "long" if stagnation >= 6 else "short" if stagnation >= 3 else "recent"

        branch = self._branches.get(self._last_branch_id or "")
        depth = int(branch["depth"]) if branch else 0
        branch_queries = int(branch["query_count"]) if branch else 0
        branch_query_band = "high" if branch_queries >= 8 else "some" if branch_queries >= 3 else "low"
        branch_rewards = list(branch["rewards"])[-5:] if branch else []
        branch_yield = (
            sum(branch_rewards) / len(branch_rewards) if branch_rewards else 0.0
        )
        branch_reward_band = (
            "positive" if branch_yield >= 0.4 else "some" if branch_yield else "none"
        )
        return "|".join(
            (
                str(remaining_band),
                yield_band,
                stagnation_band,
                str(depth),
                branch_query_band,
                branch_reward_band,
            )
        )

    def _select_compositions(
        self, branch: Mapping[str, Any], action: str
    ) -> list[Composition]:
        compositions = [
            composition
            for composition in self._compositions
            if self._composition_in_branch(composition, branch)
        ]
        reduced: dict[str, Composition] = {}
        for composition in compositions:
            reduced.setdefault(composition.reduced_formula, composition)
        compositions = list(reduced.values())

        if action in {"EXPLORE", "EXPAND"}:
            unqueried = [
                composition
                for composition in compositions
                if composition.reduced_formula not in self._queried_compositions
            ]
            if unqueried:
                compositions = unqueried
        elif action == "EXPLOIT":
            successful = {
                row["composition"]
                for row in self._query_history
                if row["reward"] == 1
            }
            promising = [
                composition
                for composition in compositions
                if composition.reduced_formula in successful
            ]
            if promising:
                compositions = promising

        self._rng.shuffle(compositions)
        return compositions[: self._per_query_generation_cap]

    def _composition_in_branch(
        self, composition: Composition, branch: Mapping[str, Any]
    ) -> bool:
        return self._composition_in_bin(composition, int(branch["bin_index"]))

    def _composition_in_bin(self, composition: Composition, index: int) -> bool:
        if self._first_element is None:
            return False
        low, high = self._bin_bounds(index)
        fraction = float(composition.get_atomic_fraction(self._first_element))
        return low <= fraction < high or (
            index == self._num_fraction_bins - 1 and fraction <= high
        )

    def _bin_bounds(self, index: int) -> tuple[float, float]:
        width = 1.0 / self._num_fraction_bins
        return index * width, (index + 1) * width

    def _write_log(self, record: dict[str, Any]) -> None:
        if self._controller_log_dir is None:
            return
        if self._log_path is None:
            self._controller_log_dir.mkdir(parents=True, exist_ok=True)
            system = "-".join(self.chemical_system_elements) or "unknown-system"
            self._log_path = (
                self._controller_log_dir / f"{system}-{self._run_id}.jsonl"
            )
        with self._log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, sort_keys=True) + "\n")

    def _save_q_values(self) -> None:
        if self._q_values_path is None:
            return
        self._q_values_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._q_values_path.with_suffix(".tmp")
        with temporary_path.open("w", encoding="utf-8") as q_file:
            json.dump(self._q_values, q_file, sort_keys=True, indent=2)
        os.replace(temporary_path, self._q_values_path)


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value
