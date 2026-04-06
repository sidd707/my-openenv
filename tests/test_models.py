"""Phase 0 — models.py contract tests. All 19 must fail until models.py is implemented."""

import pytest
from pydantic import ValidationError

# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_available_action(**overrides):
    defaults = {
        "name": "read_file",
        "description": "Reads a file and returns its contents",
        "parameters": {"path": "str"},
    }
    return {**defaults, **overrides}


def make_agent_action(**overrides):
    defaults = {
        "action_name": "read_file",
        "parameters": {"path": "/tmp/test.txt"},
        "reasoning": "I need to read the file before deciding to delete it",
    }
    return {**defaults, **overrides}


def make_system_observation(**overrides):
    from models import AvailableAction

    defaults = {
        "task_description": "Clean up the file system",
        "current_state": {"disk_used_mb": 400},
        "available_actions": [AvailableAction(**make_available_action())],
        "action_history": [],
        "steps_remaining": 20,
        "last_action_result": "",
        "reward": 0.0,
        "done": False,
    }
    return {**defaults, **overrides}


# ── Group 1: AgentAction contract ─────────────────────────────────────────────


class TestAgentActionContract:
    def test_agent_action_requires_action_name(self):
        from models import AgentAction

        data = make_agent_action()
        del data["action_name"]
        with pytest.raises(ValidationError):
            AgentAction(**data)

    def test_agent_action_requires_reasoning(self):
        from models import AgentAction

        data = make_agent_action()
        del data["reasoning"]
        with pytest.raises(ValidationError):
            AgentAction(**data)

    def test_agent_action_requires_parameters(self):
        from models import AgentAction

        data = make_agent_action()
        del data["parameters"]
        with pytest.raises(ValidationError):
            AgentAction(**data)

    def test_agent_action_rejects_empty_reasoning(self):
        from models import AgentAction

        with pytest.raises(ValidationError):
            AgentAction(**make_agent_action(reasoning=""))
        with pytest.raises(ValidationError):
            AgentAction(**make_agent_action(reasoning="   "))


# ── Group 2: AvailableAction safety contract ──────────────────────────────────


class TestAvailableActionSafety:
    def test_available_action_has_no_action_type_field(self):
        from models import AvailableAction

        assert "action_type" not in AvailableAction.model_fields

    def test_available_action_requires_name_description_parameters(self):
        from models import AvailableAction

        with pytest.raises(ValidationError):
            AvailableAction(description="desc", parameters={"a": "b"})
        with pytest.raises(ValidationError):
            AvailableAction(name="x", parameters={"a": "b"})
        with pytest.raises(ValidationError):
            AvailableAction(name="x", description="desc")

    def test_available_action_serializes_without_action_type(self):
        from models import AvailableAction

        a = AvailableAction(**make_available_action())
        dumped = a.model_dump()
        assert "action_type" not in dumped


# ── Group 3: SystemObservation contract ───────────────────────────────────────


class TestSystemObservationContract:
    def test_observation_has_no_action_type_in_available_actions(self):
        from models import SystemObservation

        obs = SystemObservation(**make_system_observation())
        dumped = obs.model_dump()
        for action_dict in dumped["available_actions"]:
            assert "action_type" not in action_dict

    def test_observation_reward_accepts_negative_values(self):
        from models import SystemObservation

        SystemObservation(**make_system_observation(reward=-0.5))

    def test_observation_reward_accepts_positive_values(self):
        from models import SystemObservation

        SystemObservation(**make_system_observation(reward=0.5))

    def test_observation_done_is_boolean(self):
        from models import SystemObservation

        obs = SystemObservation(**make_system_observation(done=True))
        assert isinstance(obs.done, bool)
        assert obs.done is True

    def test_observation_steps_remaining_accepts_zero_and_positive(self):
        from models import SystemObservation

        SystemObservation(**make_system_observation(steps_remaining=0))
        SystemObservation(**make_system_observation(steps_remaining=10))


# ── Group 4: ActionRecord internal model ──────────────────────────────────────


class TestActionRecordInternal:
    def test_action_record_tracks_reversibility(self):
        from models import ActionRecord

        r = ActionRecord(
            action_name="delete_file", was_irreversible=True, was_mistake=False, step=3
        )
        assert r.was_irreversible is True

    def test_action_record_tracks_mistake(self):
        from models import ActionRecord

        r = ActionRecord(
            action_name="delete_file", was_irreversible=True, was_mistake=True, step=3
        )
        assert r.was_mistake is True


# ── Group 5: EpisodeState defaults ────────────────────────────────────────────


class TestEpisodeStateDefaults:
    def test_episode_state_initializes_with_zero_mistakes(self):
        from models import EpisodeState

        s = EpisodeState(task_name="easy")
        assert s.irreversible_mistakes == 0

    def test_episode_state_history_starts_empty(self):
        from models import EpisodeState

        s = EpisodeState(task_name="easy")
        assert s.history == []

    def test_episode_state_total_reward_starts_at_zero(self):
        from models import EpisodeState

        s = EpisodeState(task_name="easy")
        assert s.total_reward == 0.0


# ── Group 6: Serialization round-trips ────────────────────────────────────────


class TestSerializationRoundTrips:
    def test_agent_action_round_trips_through_json(self):
        from models import AgentAction

        original = AgentAction(**make_agent_action())
        json_str = original.model_dump_json()
        restored = AgentAction.model_validate_json(json_str)
        assert restored == original

    def test_system_observation_round_trips_through_json(self):
        from models import SystemObservation

        original = SystemObservation(**make_system_observation())
        json_str = original.model_dump_json()
        restored = SystemObservation.model_validate_json(json_str)
        assert restored == original
