import pytest
from autoresearch_for_materials_discovery.scheduler import (
    ResearchAction,
    SchedulerObservation,
    TabularScheduler,
)


def observation() -> SchedulerObservation:
    return SchedulerObservation(
        remaining_fraction=0.75,
        recent_yield=0.2,
        queries_since_discovery=2,
        buffer_size=8,
        visited_regions=3,
        query_count=5,
    )


def test_action_mask_prevents_infeasible_selection():
    scheduler = TabularScheduler(seed=4)
    assert (
        scheduler.choose(
            observation(),
            [ResearchAction.EXPLORE],
        )
        == ResearchAction.EXPLORE
    )
    with pytest.raises(RuntimeError, match="not feasible"):
        scheduler.choose(
            observation(),
            [ResearchAction.EXPLORE],
            mode="fixed",
            fixed_action=ResearchAction.REFINE,
        )


def test_q_update_uses_signed_reward_and_masked_next_actions():
    scheduler = TabularScheduler(learning_rate=0.5, discount=0.0)
    updated = scheduler.update(
        "state",
        ResearchAction.EXPLORE,
        reward=-1.0,
        next_state_key="next",
        next_feasible_actions=[ResearchAction.REFINE],
    )
    assert updated == pytest.approx(-0.5)
    assert scheduler.get_state()["state"]["EXPLORE"] == pytest.approx(-0.5)


def test_seeded_random_action_choice_is_reproducible():
    actions = [ResearchAction.EXPLORE, ResearchAction.REFINE]
    first = TabularScheduler(seed=23)
    second = TabularScheduler(seed=23)
    first_actions = [
        first.choose(observation(), actions, mode="random") for _ in range(10)
    ]
    second_actions = [
        second.choose(observation(), actions, mode="random") for _ in range(10)
    ]
    assert first_actions == second_actions


def test_fixed_schedule_switches_without_using_reward_history():
    scheduler = TabularScheduler(seed=1)
    actions = [ResearchAction.EXPLORE, ResearchAction.REFINE]
    selected = [
        scheduler.choose(
            SchedulerObservation(
                remaining_fraction=0.5,
                recent_yield=0,
                queries_since_discovery=0,
                buffer_size=0,
                visited_regions=0,
                query_count=count,
            ),
            actions,
            mode="fixed_schedule",
        )
        for count in range(4)
    ]
    assert selected == [
        ResearchAction.EXPLORE,
        ResearchAction.EXPLORE,
        ResearchAction.EXPLORE,
        ResearchAction.REFINE,
    ]


def test_observation_key_omits_raw_identifiers():
    assert observation().key() == observation().key()
    assert "query_count" not in observation().key()
