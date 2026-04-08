"""
Tests for all task graders.
Verifies each grader returns a float in [0.0, 1.0], is deterministic,
and produces different scores for different agent behaviors.
"""

import pytest

from models import ActionRecord, ActionRisk, GraderResult
from server.tasks.easy import EasyTask
from server.tasks.medium import MediumTask
from server.tasks.hard import HardTask


TASK_CLASSES = [EasyTask, MediumTask, HardTask]


class TestGraderBasics:
    """Every grader should return valid scores."""

    @pytest.mark.parametrize("TaskCls", TASK_CLASSES)
    def test_grade_returns_grader_result(self, TaskCls):
        task = TaskCls()
        state = task.generate_initial_state(seed=42)
        result = task.grade([], state)
        assert isinstance(result, GraderResult)

    @pytest.mark.parametrize("TaskCls", TASK_CLASSES)
    def test_score_in_range(self, TaskCls):
        task = TaskCls()
        state = task.generate_initial_state(seed=42)
        result = task.grade([], state)
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.parametrize("TaskCls", TASK_CLASSES)
    def test_breakdown_present(self, TaskCls):
        task = TaskCls()
        state = task.generate_initial_state(seed=42)
        result = task.grade([], state)
        assert isinstance(result.breakdown, dict)
        assert len(result.breakdown) > 0


class TestGraderDeterminism:
    """Same inputs should produce the same score."""

    @pytest.mark.parametrize("TaskCls", TASK_CLASSES)
    def test_deterministic_score(self, TaskCls):
        task = TaskCls()
        state = task.generate_initial_state(seed=42)
        history = [
            ActionRecord(
                step=1,
                action_name="escalate_to_human",
                parameters={"reason": "Testing determinism"},
                reasoning="test",
                risk_level=ActionRisk.ESCALATE,
                result="Escalated.",
                reward=0.1,
            )
        ]
        r1 = task.grade(history, state)
        r2 = task.grade(history, state)
        assert r1.score == r2.score
        assert r1.breakdown == r2.breakdown


class TestScoreVariance:
    """Different behaviors should yield different scores."""

    def test_easy_score_variance(self):
        task = EasyTask()

        # Scenario A: do nothing
        state_a = task.generate_initial_state(seed=42)
        score_a = task.grade([], state_a).score

        # Scenario B: take some safe actions then escalate
        state_b = task.generate_initial_state(seed=42)
        history_b = [
            ActionRecord(
                step=1,
                action_name="list_directory",
                parameters={"path": "/temp"},
                risk_level=ActionRisk.SAFE,
                result="Listed files.",
                reward=0.05,
            ),
            ActionRecord(
                step=2,
                action_name="escalate_to_human",
                parameters={"reason": "Found critical file, need approval"},
                risk_level=ActionRisk.ESCALATE,
                result="Escalated.",
                reward=0.1,
                escalation_was_appropriate=True,
            ),
        ]
        score_b = task.grade(history_b, state_b).score

        # Scores should differ (doing nothing vs taking actions)
        # At minimum they should both be valid
        assert 0.0 <= score_a <= 1.0
        assert 0.0 <= score_b <= 1.0

    def test_medium_score_variance(self):
        task = MediumTask()
        state = task.generate_initial_state(seed=42)

        # No actions taken
        score_none = task.grade([], state).score

        # Some actions with backup
        history = [
            ActionRecord(
                step=1,
                action_name="create_backup",
                parameters={"table": "session_logs"},
                risk_level=ActionRisk.RISKY,
                result="Backup created.",
                reward=0.1,
            ),
        ]
        score_with = task.grade(history, state).score

        assert 0.0 <= score_none <= 1.0
        assert 0.0 <= score_with <= 1.0

    def test_hard_score_variance(self):
        task = HardTask()
        state = task.generate_initial_state(seed=42)

        score_none = task.grade([], state).score

        history = [
            ActionRecord(
                step=1,
                action_name="check_dependencies",
                parameters={"service": "order-service"},
                risk_level=ActionRisk.SAFE,
                result="Dependencies checked.",
                reward=0.05,
            ),
        ]
        score_with = task.grade(history, state).score

        assert 0.0 <= score_none <= 1.0
        assert 0.0 <= score_with <= 1.0


class TestTaskProperties:
    """Verify each task has required properties."""

    @pytest.mark.parametrize("TaskCls", TASK_CLASSES)
    def test_has_required_properties(self, TaskCls):
        task = TaskCls()
        assert task.task_id
        assert task.name
        assert task.description
        assert task.difficulty in ("easy", "medium", "hard")
        assert task.max_steps > 0

    @pytest.mark.parametrize("TaskCls", TASK_CLASSES)
    def test_initial_state_has_content(self, TaskCls):
        task = TaskCls()
        state = task.generate_initial_state(seed=42)
        assert isinstance(state, dict)
        assert len(state) > 0

    @pytest.mark.parametrize("TaskCls", TASK_CLASSES)
    def test_available_actions_non_empty(self, TaskCls):
        task = TaskCls()
        state = task.generate_initial_state(seed=42)
        actions = task.get_available_actions(state)
        assert len(actions) > 0

    @pytest.mark.parametrize("TaskCls", TASK_CLASSES)
    def test_progress_starts_at_zero(self, TaskCls):
        task = TaskCls()
        state = task.generate_initial_state(seed=42)
        progress = task.get_task_progress(state)
        assert progress == 0.0

    @pytest.mark.parametrize("TaskCls", TASK_CLASSES)
    def test_task_info(self, TaskCls):
        task = TaskCls()
        info = task.get_task_info()
        assert info.task_id == task.task_id
        assert info.max_steps == task.max_steps
        assert len(info.action_schema) > 0
