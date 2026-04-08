"""
FastAPI application for SafeAct-Env.
Uses create_fastapi_app for WebSocket (OpenEnv protocol).
Adds hackathon-required HTTP endpoints: /tasks, /grader, /baseline.
"""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openenv.core.env_server import create_fastapi_app

from models import (
    AgentAction,
    BaselineResult,
    GraderResult,
    SystemObservation,
    TaskInfo,
)
from server.environment import SafeActEnvironment
from server.tasks import ALL_TASKS


# ---------------------------------------------------------------------------
# Create the OpenEnv FastAPI app (handles WebSocket /ws endpoint)
# ---------------------------------------------------------------------------

app = create_fastapi_app(
    env=SafeActEnvironment,
    action_cls=AgentAction,
    observation_cls=SystemObservation,
)

# CORS for HuggingFace Spaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Shared environment instance for HTTP endpoints (non-WebSocket)
# For stateful HTTP testing — WebSocket sessions get their own instances.
# ---------------------------------------------------------------------------

_http_env = SafeActEnvironment()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# GET /tasks — list all tasks with action schemas
# ---------------------------------------------------------------------------

@app.get("/tasks", response_model=List[TaskInfo])
def list_tasks() -> List[TaskInfo]:
    return [task.get_task_info() for task in ALL_TASKS]


# ---------------------------------------------------------------------------
# HTTP endpoints for reset/step/state (for curl testing and baseline)
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: Optional[int] = None


@app.post("/http/reset")
def http_reset(req: ResetRequest) -> Dict[str, Any]:
    """HTTP reset for curl testing. The main /reset is handled by OpenEnv protocol."""
    obs = _http_env.reset(seed=req.seed, task_id=req.task_id)
    return obs.model_dump()


class StepRequest(BaseModel):
    action_name: str
    parameters: Dict[str, Any] = {}
    reasoning: str = ""


@app.post("/http/step")
def http_step(req: StepRequest) -> Dict[str, Any]:
    """HTTP step for curl testing. The main /step is handled by OpenEnv protocol."""
    action = AgentAction(
        action_name=req.action_name,
        parameters=req.parameters,
        reasoning=req.reasoning,
    )
    obs = _http_env.step(action)
    return obs.model_dump()


@app.get("/http/state")
def http_state() -> Dict[str, Any]:
    """HTTP state for curl testing. The main /state is handled by OpenEnv protocol."""
    return _http_env.state.model_dump()


# ---------------------------------------------------------------------------
# POST /grader — grade the current/last episode
# ---------------------------------------------------------------------------

@app.post("/grader", response_model=GraderResult)
def grade_episode() -> GraderResult:
    return _http_env.grade_episode()


# ---------------------------------------------------------------------------
# POST /baseline — run baseline inference (stub — actual logic in baseline.py)
# ---------------------------------------------------------------------------

@app.post("/baseline", response_model=BaselineResult)
def run_baseline() -> BaselineResult:
    """Run a simple random baseline against all tasks and return scores."""
    import random as rng

    results = []
    for task in ALL_TASKS:
        env = SafeActEnvironment()
        obs = env.reset(seed=42, task_id=task.task_id)

        # Simple heuristic baseline: prefer safe actions, escalate sometimes
        while not obs.done:
            actions = obs.available_actions
            if not actions:
                break
            # Pick a safe action if available, otherwise random
            safe_actions = [a for a in actions if "read" in a.name or "check" in a.name or "list" in a.name or "query" in a.name or "analyze" in a.name or "ping" in a.name]
            if safe_actions and rng.random() < 0.6:
                chosen = rng.choice(safe_actions)
            else:
                chosen = rng.choice(actions)

            action = AgentAction(
                action_name=chosen.name,
                parameters=_make_default_params(chosen.required_params, task.task_id, env),
                reasoning="Baseline heuristic agent",
            )
            obs = env.step(action)

        grade = env.grade_episode()
        results.append(grade)

    return BaselineResult(
        results=results,
        model_used="heuristic-baseline",
    )


def _make_default_params(
    required: Dict[str, str],
    task_id: str,
    env: SafeActEnvironment,
) -> Dict[str, Any]:
    """Generate plausible default parameters for an action."""
    params: Dict[str, Any] = {}
    state = env._system_state

    for key, type_str in required.items():
        if key == "path":
            # Pick a random file/dir from state
            files = state.get("files", {})
            if files:
                params[key] = list(files.keys())[0]
            else:
                params[key] = "/temp"
        elif key == "table":
            tables = state.get("tables", {})
            if tables:
                params[key] = list(tables.keys())[0]
            else:
                params[key] = "users"
        elif key == "service":
            services = state.get("services", {})
            if services:
                params[key] = "order-service"
            else:
                params[key] = "unknown"
        elif key == "server":
            params[key] = "app-server-1"
        elif key == "database":
            params[key] = "primary-db"
        elif key == "reason":
            params[key] = "Baseline agent flagging for review"
        elif key == "column":
            params[key] = "id"
        elif key == "before_date":
            params[key] = "2025-01-01"
        elif key == "where_clause":
            params[key] = "status = 'inactive'"
        elif key == "domain":
            params[key] = "api.example.com"
        elif key == "new_ip":
            params[key] = "10.0.2.20"
        elif key == "message":
            params[key] = "Scheduled maintenance window"
        elif key == "from_server":
            params[key] = "app-server-1"
        elif key == "to_server":
            params[key] = "app-server-2"
        elif key == "target_server":
            params[key] = "app-server-2"
        elif key == "config_key":
            params[key] = "log_level"
        elif key == "config_value":
            params[key] = "info"
        else:
            params[key] = "default"

    return params


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run the server with uvicorn."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
