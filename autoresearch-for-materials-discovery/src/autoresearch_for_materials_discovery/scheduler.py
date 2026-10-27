"""Small tabular controller for choosing feasible research macro-actions."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum


class ResearchAction(StrEnum):
    EXPLORE = "EXPLORE"
    REFINE = "REFINE"


@dataclass(frozen=True)
class SchedulerObservation:
    remaining_fraction: float
    recent_yield: float
    queries_since_discovery: int
    buffer_size: int
    visited_regions: int
    query_count: int

    def key(self) -> str:
        remaining = min(3, max(0, int(self.remaining_fraction * 4)))
        yield_band = (
            "high"
            if self.recent_yield >= 0.4
            else "some"
            if self.recent_yield > 0
            else "none"
        )
        stagnation = (
            "long"
            if self.queries_since_discovery >= 6
            else "short"
            if self.queries_since_discovery >= 3
            else "recent"
        )
        buffer = (
            "empty"
            if self.buffer_size == 0
            else "small"
            if self.buffer_size <= 32
            else "large"
        )
        regions = "few" if self.visited_regions <= 2 else "many"
        return "|".join((str(remaining), yield_band, stagnation, buffer, regions))

    @staticmethod
    def exact_transfer_coverage(
        table: Mapping[str, Mapping[str, float]], visited_keys: Sequence[str]
    ) -> dict[str, object]:
        """Report exact state-key overlap; never infer or project near matches."""
        matched = [key for key in visited_keys if key in table]
        unmatched = sorted(set(visited_keys) - set(table))
        return {
            "decision_count": len(visited_keys),
            "matched_decision_count": len(matched),
            "matched_state_count": len(set(matched)),
            "table_state_count": len(table),
            "unmatched_decision_count": len(visited_keys) - len(matched),
            "unmatched_state_count": len(unmatched),
            "unmatched_state_keys": unmatched,
        }


class TabularScheduler:
    """An epsilon-greedy Q-learning scheduler over EXPLORE and REFINE."""

    def __init__(
        self,
        *,
        seed: int = 0,
        learning_rate: float = 0.2,
        discount: float = 0.9,
        epsilon: float = 0.1,
        training: bool = False,
        q_values: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1].")
        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be in [0, 1].")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")

        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.training = training
        self._rng = random.Random(seed)
        self.q_values = {
            state: {action: float(value) for action, value in actions.items()}
            for state, actions in (q_values or {}).items()
        }

    def choose(
        self,
        observation: SchedulerObservation,
        feasible_actions: Sequence[ResearchAction],
        *,
        mode: str = "q_learning",
        fixed_action: ResearchAction = ResearchAction.EXPLORE,
    ) -> ResearchAction:
        actions = tuple(dict.fromkeys(feasible_actions))
        if not actions:
            raise RuntimeError("No feasible research action is available.")

        if mode == "fixed":
            if fixed_action not in actions:
                raise RuntimeError(f"Fixed action {fixed_action} is not feasible.")
            return fixed_action
        if mode == "random":
            return self._rng.choice(actions)
        if mode == "fixed_schedule":
            scheduled = (
                ResearchAction.REFINE
                if (observation.query_count + 1) % 4 == 0
                else ResearchAction.EXPLORE
            )
            return scheduled if scheduled in actions else actions[0]
        if mode != "q_learning":
            raise ValueError(f"Unsupported scheduler mode: {mode}")

        if self.training and self._rng.random() < self.epsilon:
            return self._rng.choice(actions)

        state_values = self.q_values.get(observation.key(), {})
        best_value = max(state_values.get(action.value, 0.0) for action in actions)
        best_actions = [
            action
            for action in actions
            if state_values.get(action.value, 0.0) == best_value
        ]
        return self._rng.choice(best_actions)

    def update(
        self,
        state_key: str,
        action: ResearchAction,
        reward: float,
        next_state_key: str | None,
        next_feasible_actions: Sequence[ResearchAction] = (),
    ) -> float:
        """Apply one Q-learning update and return the new action value."""
        current = self.q_values.setdefault(state_key, {}).get(action.value, 0.0)
        next_values = self.q_values.get(next_state_key or "", {})
        future = (
            max(
                next_values.get(candidate.value, 0.0)
                for candidate in next_feasible_actions
            )
            if next_state_key and next_feasible_actions
            else 0.0
        )
        updated = current + self.learning_rate * (
            float(reward) + self.discount * future - current
        )
        self.q_values[state_key][action.value] = updated
        return updated

    def get_state(self) -> dict[str, dict[str, float]]:
        return {state: values.copy() for state, values in self.q_values.items()}

    def load_state(self, values: Mapping[str, Mapping[str, float]]) -> None:
        self.q_values = {
            state: {action: float(value) for action, value in actions.items()}
            for state, actions in values.items()
        }
