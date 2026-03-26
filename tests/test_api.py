"""Phase 5 — API layer behaviour tests.
Integration tests that start a real TestClient against the app.
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from server.app import app
    return TestClient(app, raise_server_exceptions=False)


# ── Group 1: openenv standard endpoints ──────────────────────

class TestStandardEndpoints:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_schema_returns_action_and_observation(self, client):
        response = client.get("/schema")
        assert response.status_code == 200
        data = response.json()
        assert "action" in data
        assert "observation" in data

    def test_reset_returns_observation(self, client):
        response = client.post("/reset",
            json={"task_name": "easy"})
        assert response.status_code == 200
        data = response.json()
        assert "observation" in data
        assert "done" in data

    def test_reset_with_invalid_task_returns_error(self, client):
        response = client.post("/reset",
            json={"task_name": "nonexistent"})
        assert response.status_code in (400, 422, 500)


# ── Group 2: hackathon custom endpoints ──────────────────────

class TestHackathonEndpoints:

    def test_tasks_returns_all_three_tasks(self, client):
        response = client.get("/tasks")
        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data
        task_names = [t["name"] for t in data["tasks"]]
        assert "easy" in task_names
        assert "medium" in task_names
        assert "hard" in task_names

    def test_tasks_each_has_action_schema(self, client):
        response = client.get("/tasks")
        data = response.json()
        for task in data["tasks"]:
            assert "action_schema" in task
            assert "required" in task["action_schema"]

    def test_tasks_each_has_max_steps(self, client):
        response = client.get("/tasks")
        data = response.json()
        for task in data["tasks"]:
            assert "max_steps" in task
            assert task["max_steps"] > 0

    def test_grader_returns_score_in_range(self, client):
        from server.tasks.easy import FileCleanupTask
        task = FileCleanupTask()
        initial_state = task.get_initial_state()
        response = client.post("/grader", json={
            "task_name": "easy",
            "history": [],
            "final_state": initial_state,
        })
        assert response.status_code == 200
        data = response.json()
        assert "score" in data
        assert 0.0 <= data["score"] <= 1.0

    def test_grader_with_invalid_task_returns_error(self, client):
        response = client.post("/grader", json={
            "task_name": "nonexistent",
            "history": [],
            "final_state": {},
        })
        assert response.status_code in (400, 422)

    def test_baseline_returns_score_structure(self, client):
        response = client.post("/baseline", json={})
        assert response.status_code == 200
        data = response.json()
        assert "scores" in data
        assert "easy" in data["scores"]
        assert "medium" in data["scores"]
        assert "hard" in data["scores"]


# ── Group 3: Gradio web interface ─────────────────────────────

class TestWebInterface:
    """Verify Gradio web UI is mounted."""

    def test_web_interface_is_accessible(self, client):
        response = client.get("/web")
        # Gradio mounts at /web — should return 200 or redirect (307)
        assert response.status_code in (200, 307)

    def test_web_interface_not_404(self, client):
        response = client.get("/web")
        assert response.status_code != 404
