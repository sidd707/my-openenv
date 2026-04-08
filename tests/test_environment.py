"""
Tests for SafeActEnvironment.
Covers: reset returns valid observation, step increments count,
episode terminates at max_steps, invalid actions handled.
"""

import pytest

from models import AgentAction, SystemObservation, EpisodeState
from server.environment import SafeActEnvironment


@pytest.fixture
def env():
    return SafeActEnvironment()


class TestReset:
    def test_reset_returns_observation(self, env):
        obs = env.reset(seed=42, task_id="easy")
        assert isinstance(obs, SystemObservation)
        assert obs.done is False
        assert obs.task_id == "easy"
        assert len(obs.available_actions) > 0
        assert obs.steps_remaining > 0

    def test_reset_state_is_clean(self, env):
        env.reset(seed=42, task_id="easy")
        state = env.state
        assert isinstance(state, EpisodeState)
        assert state.step_count == 0
        assert state.irreversible_mistakes == 0
        assert state.total_reward == 0.0
        assert state.task_complete is False

    def test_reset_each_task(self, env):
        for task_id in ("easy", "medium", "hard"):
            obs = env.reset(seed=42, task_id=task_id)
            assert obs.task_id == task_id
            assert len(obs.available_actions) > 0

    def test_reset_invalid_task_falls_back(self, env):
        obs = env.reset(seed=42, task_id="nonexistent")
        assert obs.task_id == "easy"  # falls back to easy


class TestStep:
    def test_step_increments_count(self, env):
        env.reset(seed=42, task_id="easy")
        action = AgentAction(
            action_name="list_directory",
            parameters={"path": "/temp"},
            reasoning="Checking temp files",
        )
        obs = env.step(action)
        assert env.state.step_count == 1
        assert isinstance(obs, SystemObservation)

    def test_step_invalid_action(self, env):
        env.reset(seed=42, task_id="easy")
        action = AgentAction(
            action_name="nonexistent_action",
            parameters={},
            reasoning="Testing invalid",
        )
        obs = env.step(action)
        assert "Error" in obs.last_action_result
        assert env.state.step_count == 1

    def test_step_without_reset(self, env):
        action = AgentAction(action_name="test", parameters={}, reasoning="")
        obs = env.step(action)
        assert obs.done is True

    def test_episode_terminates_at_max_steps(self, env):
        env.reset(seed=42, task_id="easy")
        max_steps = env.state.max_steps

        safe_action = AgentAction(
            action_name="list_directory",
            parameters={"path": "/temp"},
            reasoning="Step test",
        )

        for i in range(max_steps):
            obs = env.step(safe_action)

        assert obs.done is True
        assert env.state.step_count == max_steps

    def test_safe_action_gives_positive_reward(self, env):
        env.reset(seed=42, task_id="easy")
        action = AgentAction(
            action_name="list_directory",
            parameters={"path": "/temp"},
            reasoning="Info gathering",
        )
        obs = env.step(action)
        assert obs.reward is not None
        assert obs.reward > 0

    def test_metadata_contains_risk_info(self, env):
        env.reset(seed=42, task_id="easy")
        action = AgentAction(
            action_name="list_directory",
            parameters={"path": "/temp"},
            reasoning="Checking",
        )
        obs = env.step(action)
        assert "action_risk" in obs.metadata
        assert "was_irreversible" in obs.metadata


class TestState:
    def test_state_reflects_progress(self, env):
        env.reset(seed=42, task_id="easy")
        action = AgentAction(
            action_name="list_directory",
            parameters={"path": "/temp"},
            reasoning="Checking",
        )
        env.step(action)
        state = env.state
        assert state.step_count == 1
        assert state.episode_id is not None


class TestGrader:
    def test_grade_without_episode(self, env):
        result = env.grade_episode()
        assert result.task_id == "unknown"
        assert result.score == 0.0

    def test_grade_after_reset(self, env):
        env.reset(seed=42, task_id="easy")
        result = env.grade_episode()
        assert 0.0 <= result.score <= 1.0
        assert result.task_id == "easy"
