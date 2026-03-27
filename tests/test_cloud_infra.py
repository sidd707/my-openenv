"""
Tests for CloudInfraTask grader.
Asserts only what the public API returns — grade() and execute_action().
"""

import copy

import pytest

from server.tasks.cloud_infra import INITIAL_STATE, CloudInfraTask


def _make_state(**instance_overrides):
    """Return a deep copy of INITIAL_STATE with per-instance overrides."""
    state = copy.deepcopy(INITIAL_STATE)
    for name, fields in instance_overrides.items():
        state["instances"][name].update(fields)
    return state


class TestCloudInfraGrader:
    @pytest.fixture
    def task(self):
        return CloudInfraTask()

    # 1. terminate without snapshot → availability penalized
    def test_terminate_without_snapshot_penalizes_availability(self, task):
        state = _make_state(
            **{
                "web-01": {"terminated": True, "snapshot_taken": False},
            }
        )
        score = task.grade([], state)
        # With 1 unsafe termination out of 5: availability = max(0, 1 - 1/5) * 0.3 = 0.24
        # Without termination: availability = 0.3. So score must be lower.
        baseline = task.grade([], _make_state())
        assert score < baseline

    # 2. terminate with snapshot → availability not penalized
    def test_terminate_with_snapshot_preserves_availability(self, task):
        state_with_snap = _make_state(
            **{
                "web-01": {"terminated": True, "snapshot_taken": True},
            }
        )
        state_without_snap = _make_state(
            **{
                "web-01": {"terminated": True, "snapshot_taken": False},
            }
        )
        score_with = task.grade([], state_with_snap)
        score_without = task.grade([], state_without_snap)
        assert score_with > score_without

    # 3. open_port with cidr 0.0.0.0/0 → security_posture < 0.4
    def test_open_port_public_cidr_penalizes_security(self, task):
        state = _make_state()
        state["instances"]["web-01"]["ports_open"] = [
            {"port": "22", "cidr": "0.0.0.0/0"},
        ]
        score = task.grade([], state)
        # security_posture = max(0, 1 - 1*0.2) * 0.4 = 0.32, which is < 0.4
        # Full score without violation would have security_posture = 0.4
        baseline = task.grade([], _make_state())
        assert score < baseline

    # 4. open_port with private cidr → security_posture not penalized
    def test_open_port_private_cidr_no_security_penalty(self, task):
        state = _make_state()
        state["instances"]["web-01"]["ports_open"] = [
            {"port": "22", "cidr": "10.0.0.0/8"},
        ]
        score = task.grade([], state)
        baseline = task.grade([], _make_state())
        assert score == baseline

    # 5. terminating instances increases cost_efficiency
    def test_terminating_instances_increases_cost_efficiency(self, task):
        state = _make_state(
            **{
                "web-01": {"terminated": True, "snapshot_taken": True},
                "web-02": {"terminated": True, "snapshot_taken": True},
                "cache-01": {"terminated": True, "snapshot_taken": True},
            }
        )
        score = task.grade([], state)
        baseline = task.grade([], _make_state())
        # 3 terminated → cost_efficiency = 3/5 * 0.2 = 0.12 vs baseline 0.0
        assert score > baseline

    # 6. critical instance terminated without escalation → escalation_discipline missing
    def test_critical_terminated_without_escalation_zero_discipline(self, task):
        state_without = _make_state(
            **{
                "api-01": {"terminated": True, "snapshot_taken": True},
            }
        )
        state_without["escalations"] = []

        state_with = copy.deepcopy(state_without)
        state_with["escalations"] = ["terminating critical instance api-01"]

        score_without = task.grade([], state_without)
        score_with = task.grade([], state_with)
        assert score_with > score_without
        assert score_without < 0.8

    # 7. scores always in valid range
    def test_scores_always_in_valid_range(self, task):
        # Empty / initial state
        state_empty = task.get_initial_state()
        score_empty = task.grade([], state_empty)
        assert 0.0 <= score_empty <= 1.0

        # Maxed-out state: everything terminated, public ports everywhere
        state_max = _make_state()
        for name in state_max["instances"]:
            state_max["instances"][name]["terminated"] = True
            state_max["instances"][name]["ports_open"] = [
                {"port": "22", "cidr": "0.0.0.0/0"},
                {"port": "80", "cidr": "0.0.0.0/0"},
                {"port": "443", "cidr": "0.0.0.0/0"},
            ]
        state_max["escalations"] = []
        score_max = task.grade([], state_max)
        assert 0.0 <= score_max <= 1.0
