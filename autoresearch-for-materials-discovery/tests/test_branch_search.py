import random

from autoresearch_for_materials_discovery.branch_search import (
    BranchSearchReActAgent,
)
from pymatgen.analysis.phase_diagram import PDEntry
from pymatgen.core import Composition, Lattice, Structure

from made.agents import LLMReActOrchestratorAgent
from made.agents.base import Generator, Plan


class FakeRandomGenerator(Generator):
    def generate(self, plan: Plan, state: dict) -> list[Structure]:
        structures = []
        for composition in plan.compositions:
            species = [
                str(element)
                for element, amount in composition.get_el_amt_dict().items()
                for _ in range(int(amount))
            ]
            for candidate_index in range(plan.num_candidates):
                structures.append(
                    Structure(
                        Lattice.cubic(4 + candidate_index / 10),
                        species,
                        [
                            [index / max(1, len(species)), 0, 0]
                            for index in range(len(species))
                        ],
                    )
                )
        return structures

    def get_state(self) -> dict:
        return {}

    def load_state(self, state: dict) -> None:
        return None

    def update_state(self, state: dict) -> None:
        return None


def test_fraction_bins_are_half_open_and_cover_full_interval():
    agent = BranchSearchReActAgent.__new__(BranchSearchReActAgent)
    agent._first_element = "Cd"
    agent._num_fraction_bins = 5

    assert not agent._composition_in_bin(Composition("Cd4Li"), 3)
    assert agent._composition_in_bin(Composition("Cd4Li"), 4)
    assert agent._composition_in_bin(Composition("CdLi4"), 1)
    assert agent._composition_in_bin(Composition("Cd"), 4)


def test_q_learning_update_uses_observed_reward_and_next_action_value():
    agent = BranchSearchReActAgent.__new__(BranchSearchReActAgent)
    agent._q_values = {
        "current": {"EXPAND:bin-0": 0.2},
        "next": {"EXPLORE:branch-0": 0.4, "EXPLOIT:branch-0": 0.8},
    }
    agent._learning_rate = 0.2
    agent._discount = 0.9

    agent._update_q_values(
        "current",
        "EXPAND:bin-0",
        reward=1,
        next_state_key="next",
        next_actions=["EXPLORE:branch-0", "EXPLOIT:branch-0"],
    )

    assert abs(agent._q_values["current"]["EXPAND:bin-0"] - 0.504) < 1e-12


def test_action_selection_uses_seeded_epsilon_greedy_values():
    agent = BranchSearchReActAgent.__new__(BranchSearchReActAgent)
    agent._rng = random.Random(7)
    agent._epsilon = 0
    agent._q_values = {"state": {"EXPLORE:branch-0": 0.1, "EXPAND:bin-1": 0.5}}

    assert (
        agent._choose_action(
            "state", ["EXPLORE:branch-0", "EXPAND:bin-1"]
        )
        == "EXPAND:bin-1"
    )


def test_state_key_uses_observed_controller_features_only():
    agent = BranchSearchReActAgent.__new__(BranchSearchReActAgent)
    agent._episode_budget = 60
    agent._reward_history = [0, 1, 0]
    agent._branches = {
        "branch-00": {
            "depth": 1,
            "query_count": 3,
            "rewards": [0, 1, 0],
        }
    }
    agent._last_branch_id = "branch-00"

    ordinary = agent._state_key({"query_count": 4})
    with_hidden_label = agent._state_key(
        {"query_count": 4, "ground_truth_pd": {"secret": "unused"}}
    )

    assert ordinary == with_hidden_label


def test_branch_react_selects_only_from_generated_region_slate(monkeypatch):
    elemental_entries = []
    for symbol in ("Cd", "Li"):
        structure = Structure(Lattice.cubic(3), [symbol], [[0, 0, 0]])
        elemental_entries.append(
            PDEntry(
                Composition(symbol),
                0,
                attribute={"structure": structure.as_dict()},
            ).as_dict()
        )
    initial_state = {
        "query_count": 0,
        "elements": ["Cd", "Li"],
        "phase_diagram_all_entries": elemental_entries,
        "last_observation": None,
        "stability_tolerance": 0.1,
    }

    monkeypatch.setattr(
        LLMReActOrchestratorAgent,
        "_setup_dspy",
        lambda agent: setattr(agent, "lm", None),
    )

    def select_first_slate_candidate(agent, state):
        assert agent.enabled_tools == [
            "list_compositions",
            "query_structures",
            "get_buffer_stats",
            "select_for_evaluation",
        ]
        assert agent.buffer
        assert sum(len(entries) for entries in agent.buffer.values()) <= (
            agent._per_query_generation_cap
        )
        assert all(
            agent._composition_in_branch(
                entries[0]["structure"].composition,
                agent._branches[agent._pending_transition["branch_id"]],
            )
            for entries in agent.buffer.values()
        )
        selected = next(iter(agent.buffer.values()))[0]["structure"]
        return selected.composition, selected

    monkeypatch.setattr(
        LLMReActOrchestratorAgent,
        "propose_composition_and_structure",
        select_first_slate_candidate,
    )
    agent = BranchSearchReActAgent(
        generators={"random": FakeRandomGenerator()},
        llm_config={},
        max_stoichiometry=4,
        max_candidates_per_query=4,
        max_generated_per_episode=8,
        episode_budget=60,
        epsilon=0,
    )

    agent.update_state(initial_state)
    _, selected = agent.propose_composition_and_structure(initial_state)
    pending = dict(agent._pending_transition)
    observed_state = {
        **initial_state,
        "query_count": 1,
        "last_observation": {
            "proposal": selected.as_dict(),
            "reduced_formula": selected.composition.reduced_formula,
            "e_above_hull": 0.0,
            "is_stable": True,
            "is_newly_discovered": True,
        },
        "ground_truth_pd": {"private": "must not be read"},
    }

    agent.update_state(observed_state)

    assert agent._q_values[pending["state_key"]][pending["action"]] > 0
    assert agent._query_history[-1]["reward"] == 1
    agent.propose_composition_and_structure(observed_state)
