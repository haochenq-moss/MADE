import json
from typing import Any

import pytest
from autoresearch_for_materials_discovery.agent import (
    CappedGenerator,
    FARSAgent,
    GenerationBudget,
    ScoringBudget,
)
from autoresearch_for_materials_discovery.scheduler import (
    ResearchAction,
    SchedulerObservation,
)
from pymatgen.analysis.phase_diagram import PDEntry
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from made.agents.base import Generator, Plan
from made.utils.structure_hash import structure_hash


class FakeGenerator(Generator):
    def __init__(self):
        self.last_plan: Plan | None = None

    def generate(self, plan: Plan, state: dict[str, Any]) -> list[Structure]:
        self.last_plan = plan
        structures = []
        for composition in plan.compositions:
            for _ in range(plan.num_candidates):
                structures.append(
                    Structure(
                        Lattice.cubic(4),
                        [str(element) for element in composition.elements],
                        [[0, 0, 0] for _ in composition.elements],
                    )
                )
        return structures

    def get_state(self) -> dict[str, Any]:
        return {}

    def load_state(self, state: dict[str, Any]) -> None:
        pass

    def update_state(self, state: dict[str, Any]) -> None:
        pass


def test_generator_caps_candidate_count_and_episode_remaining():
    underlying = FakeGenerator()
    generator = CappedGenerator(underlying, per_query_cap=5, remaining_episode_cap=7)
    generator.begin_query()
    plan = Plan(
        compositions=[Composition("Li"), Composition("O")],
        num_candidates=10,
    )
    generated = generator.generate(plan, {})
    assert len(generated) == 4
    assert generator.remaining_episode_cap == 3
    assert generator.generated_this_query == 4

    generated_next = generator.generate(plan, {})
    assert len(generated_next) == 1
    assert generator.generated_this_query == 5


def test_equivalent_scheduler_observations_have_identical_state_keys():
    first = SchedulerObservation(0.74, 0.4, 2, 32, 2, 5)
    same_buckets = SchedulerObservation(0.70, 0.45, 1, 16, 1, 9)

    assert first.key() == same_buckets.key()


def test_warm_start_coverage_counts_exact_keys_only():
    table = {
        "3|none|recent|large|few": {"EXPLORE": 0.2},
        "2|none|long|large|few": {"REFINE": 0.5},
    }
    coverage = SchedulerObservation.exact_transfer_coverage(
        table,
        [
            "3|none|recent|large|few",
            "3|none|recent|empty|few",
            "2|none|long|large|few",
        ],
    )

    assert coverage == {
        "decision_count": 3,
        "matched_decision_count": 2,
        "matched_state_count": 2,
        "table_state_count": 2,
        "unmatched_decision_count": 1,
        "unmatched_state_count": 1,
        "unmatched_state_keys": ["3|none|recent|empty|few"],
    }


def test_explore_generation_prioritizes_unqueried_proposed_compositions():
    underlying = FakeGenerator()
    generator = CappedGenerator(
        underlying,
        per_query_cap=8,
        remaining_episode_cap=8,
        queried_compositions={"Li"},
    )
    generator.begin_query()
    plan = Plan(
        compositions=[Composition("Li"), Composition("O"), Composition("Li2O")],
        num_candidates=1,
    )

    generated = generator.generate(plan, {})

    assert [
        composition.reduced_formula
        for composition in underlying.last_plan.compositions
    ] == ["O2", "Li2O"]
    assert len(generated) == 2
    assert generator.prioritized_composition_count == 2


def test_explore_generation_uses_buffer_when_no_unqueried_plan_composition():
    underlying = FakeGenerator()
    generator = CappedGenerator(
        underlying,
        per_query_cap=8,
        remaining_episode_cap=8,
        queried_compositions={"Li"},
        unqueried_buffer_available=True,
    )
    generator.begin_query()

    generated = generator.generate(
        Plan(compositions=[Composition("Li")], num_candidates=1), {}
    )

    assert generated == []
    assert underlying.last_plan is None
    assert generator.buffer_fallback_count == 1
    assert generator.generated_this_query == 0


def test_explore_generation_falls_back_when_no_unqueried_option_exists():
    generator = CappedGenerator(
        FakeGenerator(),
        per_query_cap=2,
        remaining_episode_cap=2,
        queried_compositions={"Li"},
    )
    generator.begin_query()

    generated = generator.generate(
        Plan(compositions=[Composition("Li")], num_candidates=1), {}
    )

    assert len(generated) == 1
    assert generator.generated_this_query == 1


def test_explore_buffer_hides_repeated_compositions_when_alternatives_exist():
    buffer = {"Li": [{"id": "repeat"}], "Li2O": [{"id": "unseen"}]}

    visible, hidden, fallback = FARSAgent._partition_exploration_buffer(
        buffer, {"Li"}
    )

    assert visible == {"Li2O": [{"id": "unseen"}]}
    assert hidden == {"Li": [{"id": "repeat"}]}
    assert fallback is False


def test_explore_buffer_falls_back_when_no_unqueried_composition_exists():
    buffer = {"Li": [{"id": "repeat"}]}

    visible, hidden, fallback = FARSAgent._partition_exploration_buffer(
        buffer, {"Li"}
    )

    assert visible == buffer
    assert hidden == {}
    assert fallback is True


def test_explore_proposal_masks_repeated_buffer_until_unqueried_selection(
    monkeypatch,
):
    from made.agents.llm_react_orchestrator import LLMReActOrchestratorAgent

    repeated = Structure(Lattice.cubic(3), ["Li"], [[0, 0, 0]])
    unseen = Structure(
        Lattice.cubic(4),
        ["Li", "Li", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
    )

    def entry(structure, composition):
        return {
            "structure": structure,
            "hash": structure_hash(structure),
            "composition": composition,
            "full_formula": structure.composition.formula.replace(" ", ""),
            "source": "test",
            "scores": {},
            "num_sites": len(structure),
        }

    def fake_react_proposal(agent, state):
        assert list(agent.buffer) == ["Li2O"]
        assert "Previously queried reduced compositions: Li" in (
            agent.signature_class.instructions
        )
        selected = agent.buffer["Li2O"].pop(0)["structure"]
        del agent.buffer["Li2O"]
        return selected.composition, selected

    monkeypatch.setattr(
        LLMReActOrchestratorAgent,
        "_setup_dspy",
        lambda agent: setattr(agent, "lm", None),
    )
    monkeypatch.setattr(
        LLMReActOrchestratorAgent,
        "propose_composition_and_structure",
        fake_react_proposal,
    )
    agent = FARSAgent(
        generators={"random": FakeGenerator()},
        llm_config={},
        scheduler_mode="fixed",
        fixed_action="EXPLORE",
    )
    agent.buffer = {
        "Li": [entry(repeated, "Li")],
        "Li2O": [entry(unseen, "Li2O")],
    }
    agent.structure_cache = {
        row["hash"]: row for entries in agent.buffer.values() for row in entries
    }
    agent._controller_history = [{"composition": "Li", "stable_novel": False}]

    selected = agent.propose_composition_and_structure(
        {"query_count": 1, "elements": ["Li", "O"]}
    )

    assert selected[1] is unseen
    assert list(agent.buffer) == ["Li"]


def test_shared_generator_budget_cannot_be_bypassed_by_switching_generators():
    budget = GenerationBudget(per_query_cap=6, remaining_episode_cap=6)
    random_generator = CappedGenerator(FakeGenerator(), 6, 6, budget)
    chemeleon_generator = CappedGenerator(FakeGenerator(), 6, 6, budget)
    budget.begin_query()

    plan = Plan(compositions=[Composition("Li")], num_candidates=4)
    assert len(random_generator.generate(plan, {})) == 4
    assert len(chemeleon_generator.generate(plan, {})) == 2
    assert budget.generated_this_query == 6
    assert budget.remaining_episode_cap == 0


def test_scoring_budget_counts_unique_candidates_and_stops_over_cap():
    first = Structure(Lattice.cubic(4), ["Li"], [[0, 0, 0]])
    second = Structure(Lattice.cubic(5), ["O"], [[0, 0, 0]])
    budget = ScoringBudget(maximum=2)
    budget.consume([first])
    budget.consume([first, second])
    assert len(budget.seen_hashes) == 2
    try:
        budget.consume([Structure(Lattice.cubic(6), ["Na"], [[0, 0, 0]])])
    except RuntimeError as error:
        assert "scoring cap" in str(error)
    else:
        raise AssertionError("Scoring budget should reject a third candidate.")


def test_provenance_only_names_observed_inputs():
    agent = FARSAgent.__new__(FARSAgent)
    agent._controller_history = []
    agent.buffer = {}
    provenance = agent._provenance(
        {
            "query_count": 3,
            "last_observation": {"is_stable": True, "e_above_hull": 0.0},
            "phase_diagram_all_entries": [],
            "ground_truth_pd": {"hidden": "must not be read"},
        }
    )
    assert provenance["query_count"] == 3
    assert "ground_truth_pd" not in provenance["sources"]
    assert "hidden" not in str(provenance)


def test_memory_ablation_changes_scheduler_features_without_touching_evaluator():
    agent = FARSAgent.__new__(FARSAgent)
    agent._controller_memory_mode = "none"
    agent._controller_history = [
        {"composition": "LiO", "stable_novel": True},
        {"composition": "Li2O", "stable_novel": False},
    ]
    agent.buffer = {}
    agent._episode_generation_cap = 640
    agent._episode_budget = 20
    agent._per_query_generation_cap = 32
    no_memory = agent._make_scheduler_observation({"query_count": 2})

    agent._controller_memory_mode = "full"
    full_memory = agent._make_scheduler_observation({"query_count": 2})
    assert no_memory.recent_yield == 0
    assert no_memory.visited_regions == 0
    assert full_memory.recent_yield > 0
    assert full_memory.visited_regions == 2


def test_balanced_formula_guidance_uses_only_prior_query_counts():
    agent = FARSAgent.__new__(FARSAgent)
    agent._candidate_selection_policy = "balanced_formula"
    agent._controller_history = [
        {"composition": "Li2O", "stable_novel": True},
        {"composition": "Li2O", "stable_novel": False},
        {"composition": "Li", "stable_novel": False},
    ]
    agent.buffer = {"Li2O": [], "Li": [], "O2": []}

    instruction = agent._candidate_selection_guidance()

    assert "Li=1, Li2O=2" in instruction
    assert "A formula not listed has zero prior queries" in instruction
    assert "Do not reject a candidate solely because its composition was queried" in (
        instruction
    )
    assert "hidden ground-truth labels" in instruction
    assert "stable_novel" not in instruction


def test_default_candidate_selection_has_no_balancing_instruction():
    agent = FARSAgent.__new__(FARSAgent)
    agent._candidate_selection_policy = "llm"

    assert agent._candidate_selection_guidance() == ""


def test_unsupported_candidate_selection_policy_is_rejected():
    with pytest.raises(ValueError, match="candidate_selection_policy"):
        FARSAgent(
            generators={},
            llm_config={},
            candidate_selection_policy="block_repeated_formula",
        )


def test_observed_msun_reward_can_decrease_after_hull_update():
    first = Structure(
        Lattice.cubic(4),
        ["Li", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    lower_energy_polymorph = Structure(
        Lattice.cubic(5),
        ["Li", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    elemental_li = Structure(Lattice.cubic(3), ["Li"], [[0, 0, 0]])
    elemental_o = Structure(Lattice.cubic(3), ["O"], [[0, 0, 0]])
    entries = [
        PDEntry("Li", 0, attribute={"structure": elemental_li.as_dict()}),
        PDEntry("O", 0, attribute={"structure": elemental_o.as_dict()}),
        PDEntry("LiO", -1, attribute={"structure": first.as_dict()}),
        PDEntry(
            "LiO",
            -1.8,
            attribute={"structure": lower_energy_polymorph.as_dict()},
        ),
    ]

    agent = FARSAgent.__new__(FARSAgent)
    agent._objective_ledger = [
        {"structure_hash": structure_hash(first), "is_newly_discovered": True}
    ]
    observed_state = {
        "phase_diagram_all_entries": [entry.as_dict() for entry in entries],
        "stability_tolerance": 0.1,
    }
    assert agent._observed_msun_count(observed_state) == 0


def test_agent_initializes_with_q_values_and_masks_refinement(monkeypatch, tmp_path):
    from made.agents.llm_react_orchestrator import LLMReActOrchestratorAgent

    weights = tmp_path / "q-values.json"
    weights.write_text(
        json.dumps({"state": {ResearchAction.REFINE.value: 0.75}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        LLMReActOrchestratorAgent,
        "_setup_dspy",
        lambda agent: setattr(agent, "lm", None),
    )
    agent = FARSAgent(
        generators={"random": FakeGenerator()},
        llm_config={},
        q_values_path=str(weights),
    )
    assert agent._scheduler.q_values["state"]["REFINE"] == 0.75
    assert agent._feasible_actions() == (ResearchAction.EXPLORE,)


def test_initial_q_values_are_loaded_without_overwriting_source(
    monkeypatch, tmp_path
):
    from made.agents.llm_react_orchestrator import LLMReActOrchestratorAgent

    initial_weights = tmp_path / "initial-q-values.json"
    output_weights = tmp_path / "controller-logs" / "q-values.json"
    initial_weights.write_text(
        json.dumps({"state": {ResearchAction.EXPLORE.value: 0.5}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        LLMReActOrchestratorAgent,
        "_setup_dspy",
        lambda agent: setattr(agent, "lm", None),
    )
    agent = FARSAgent(
        generators={"random": FakeGenerator()},
        llm_config={},
        q_values_path=str(output_weights),
        initial_q_values_path=str(initial_weights),
    )

    assert agent._scheduler.q_values["state"]["EXPLORE"] == 0.5
    agent._scheduler.update(
        "state",
        ResearchAction.EXPLORE,
        reward=1,
        next_state_key=None,
    )
    agent._save_q_values()

    assert json.loads(initial_weights.read_text(encoding="utf-8")) == {
        "state": {"EXPLORE": 0.5}
    }
    saved = json.loads(output_weights.read_text(encoding="utf-8"))
    assert saved["state"]["EXPLORE"] == 0.6


def test_warm_start_decision_log_reports_exact_state_coverage(
    monkeypatch, tmp_path
):
    from made.agents.llm_react_orchestrator import LLMReActOrchestratorAgent

    def fake_react_proposal(agent, state):
        generated = next(iter(agent.generators.values())).generate(
            Plan(compositions=[Composition("Li")], num_candidates=1), state
        )
        return generated[0].composition, generated[0]

    monkeypatch.setattr(
        LLMReActOrchestratorAgent,
        "_setup_dspy",
        lambda agent: setattr(agent, "lm", None),
    )
    monkeypatch.setattr(
        LLMReActOrchestratorAgent,
        "propose_composition_and_structure",
        fake_react_proposal,
    )
    state_key = SchedulerObservation(
        remaining_fraction=1,
        recent_yield=0,
        queries_since_discovery=0,
        buffer_size=0,
        visited_regions=0,
        query_count=0,
    ).key()
    initial_weights = tmp_path / "initializer.json"
    output_weights = tmp_path / "run" / "q_values.json"
    initial_weights.write_text(
        json.dumps({state_key: {"EXPLORE": 0.5}}), encoding="utf-8"
    )
    log_dir = tmp_path / "controller"
    agent = FARSAgent(
        generators={"random": FakeGenerator()},
        llm_config={},
        scheduler_mode="fixed",
        fixed_action="EXPLORE",
        initial_q_values_path=str(initial_weights),
        q_values_path=str(output_weights),
        controller_log_dir=str(log_dir),
    )

    agent.propose_composition_and_structure(
        {"query_count": 0, "elements": ["Li", "O"]}
    )

    records = [
        json.loads(line)
        for path in log_dir.glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    decision = next(record for record in records if record["event"] == "decision")
    assert decision["warm_start_exact_match"] is True
    assert decision["warm_start_coverage"]["matched_decision_count"] == 1
    assert decision["warm_start_coverage"]["unmatched_decision_count"] == 0
    assert initial_weights.exists()
    assert not output_weights.exists()


def test_missing_warm_start_table_is_an_explicit_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Initial Q-table does not exist"):
        FARSAgent(
            generators={"random": FakeGenerator()},
            llm_config={},
            initial_q_values_path=str(tmp_path / "missing.json"),
        )


def test_fars_proposal_routes_through_capped_existing_generator(monkeypatch):
    from made.agents.llm_react_orchestrator import LLMReActOrchestratorAgent

    generated_counts = []

    def fake_react_proposal(agent, state):
        assert "create_structure" not in agent.enabled_tools
        assert "FARS CONTROLLER MODE: EXPLORE" in agent.signature_class.instructions
        assert "strict limit of 32 distinct candidates" in agent.signature_class.instructions
        assert "complete requested batch fits within the remaining scoring allowance" in (
            agent.signature_class.instructions
        )
        generator = next(iter(agent.generators.values()))
        structures = generator.generate(
            Plan(compositions=[Composition("Li")], num_candidates=20),
            state,
        )
        generated_counts.append(len(structures))
        return structures[0].composition, structures[0]

    monkeypatch.setattr(
        LLMReActOrchestratorAgent,
        "_setup_dspy",
        lambda agent: setattr(agent, "lm", None),
    )
    monkeypatch.setattr(
        LLMReActOrchestratorAgent,
        "propose_composition_and_structure",
        fake_react_proposal,
    )
    agent = FARSAgent(
        generators={"random": FakeGenerator()},
        llm_config={},
        scheduler_mode="fixed",
        fixed_action="EXPLORE",
        max_candidates_per_query=3,
        max_generated_per_episode=5,
        context_config={"orchestration_prompt": "base instructions"},
    )
    state = {"query_count": 0, "elements": ["Li", "O"]}

    first = agent.propose_composition_and_structure(state)
    second = agent.propose_composition_and_structure({**state, "query_count": 1})

    assert len(first[1]) == 1
    assert len(second[1]) == 1
    assert generated_counts == [3, 2]
    assert agent._episode_generation_cap == 0
