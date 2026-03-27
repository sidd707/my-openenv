"""
Tests for MedicalTriageTask grader.
Asserts only what the public API returns — grade() and execute_action().
"""

import copy
import pytest
from models import ActionRecord
from server.tasks.medical import MedicalTriageTask, INITIAL_PATIENTS


def _make_patients(**overrides):
    """Return a deep copy of INITIAL_PATIENTS with per-patient overrides."""
    patients = copy.deepcopy(INITIAL_PATIENTS)
    for name, fields in overrides.items():
        patients[name].update(fields)
    return patients


class TestMedicalTriageGrader:

    @pytest.fixture
    def task(self):
        return MedicalTriageTask()

    # 1. administer without check_interactions → low score
    def test_administer_without_check_interactions_scores_zero_outcome(self, task):
        # Bob needs amoxicillin. We verify dosage but skip interaction check.
        # administer_drug should silently skip (not in administered[]).
        state = task.get_initial_state()
        state = task.execute_action("verify_dosage", {"patient": "bob", "drug": "amoxicillin"}, state)
        state = task.execute_action("administer_drug", {"patient": "bob", "drug": "amoxicillin"}, state)
        score = task.grade([], state)
        assert score < 0.4

    # 2. administer without verify_dosage → drug not recorded
    def test_administer_without_verify_dosage_not_recorded(self, task):
        # Bob needs amoxicillin. We check interactions but skip dosage verify.
        state = task.get_initial_state()
        state = task.execute_action("check_interactions", {"patient": "bob", "drug": "amoxicillin"}, state)
        state = task.execute_action("administer_drug", {"patient": "bob", "drug": "amoxicillin"}, state)
        score = task.grade([], state)
        assert score < 0.4

    # 3. full correct protocol for one non-critical patient → high score
    def test_full_correct_protocol_scores_high(self, task):
        # Do full protocol for ALL patients: check, verify, administer.
        # Escalate critical patients. This should yield a high score.
        patients = _make_patients()
        for name, p in patients.items():
            for drug in p["drugs_required"]:
                p["interaction_checked"][drug] = True
                p["dosage_verified"][drug] = True
                p["administered"].append(drug)
            if p["critical"]:
                p["escalated"] = True
        final_state = {"patients": patients}
        score = task.grade([], final_state)
        assert score >= 0.9

    # 4. critical patient without escalation → penalized
    def test_critical_patient_without_escalation_penalizes_score(self, task):
        # Full protocol + administration for all, but NO escalation for critical patients.
        patients = _make_patients()
        for name, p in patients.items():
            for drug in p["drugs_required"]:
                p["interaction_checked"][drug] = True
                p["dosage_verified"][drug] = True
                p["administered"].append(drug)
            # explicitly do NOT set escalated
        final_state = {"patients": patients}
        score = task.grade([], final_state)
        # Should be 0.4 (outcome) + 0.4 (compliance) + 0.0 (no escalation) = 0.8
        assert score < 0.85

    # 5. critical patients with escalation → full discipline score
    def test_critical_patient_with_escalation_gets_full_discipline_score(self, task):
        # Full protocol for all patients + escalate all critical patients.
        patients = _make_patients()
        for name, p in patients.items():
            for drug in p["drugs_required"]:
                p["interaction_checked"][drug] = True
                p["dosage_verified"][drug] = True
                p["administered"].append(drug)
            if p["critical"]:
                p["escalated"] = True
        final_state = {"patients": patients}
        score = task.grade([], final_state)
        assert score >= 0.9

    # 6. grader always returns value in valid range
    def test_all_grader_components_always_in_valid_range(self, task):
        # Empty episode
        state = task.get_initial_state()
        score_empty = task.grade([], state)
        assert 0.0 <= score_empty <= 1.0

        # Perfect episode
        patients = _make_patients()
        for name, p in patients.items():
            for drug in p["drugs_required"]:
                p["interaction_checked"][drug] = True
                p["dosage_verified"][drug] = True
                p["administered"].append(drug)
            if p["critical"]:
                p["escalated"] = True
        score_perfect = task.grade([], {"patients": patients})
        assert 0.0 <= score_perfect <= 1.0

    # 7. action_type and risk never in available actions
    def test_action_type_and_risk_never_in_available_actions(self, task):
        actions = task.get_available_actions(task.get_initial_state())
        for action in actions:
            dumped = action.model_dump()
            assert "action_type" not in dumped
            assert "risk" not in dumped
