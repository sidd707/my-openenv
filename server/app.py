"""
FastAPI application entry point for SafeAct-Env.
Uses openenv-core's create_app for standard endpoints + static demo UI.
Adds hackathon-required endpoints: /tasks, /grader, /baseline.
"""

import importlib.metadata
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from openenv.core.env_server.http_server import create_app
from pydantic import BaseModel

from models import ActionRecord, AgentAction, SystemObservation
from server.environment import _TASK_CONFIG, IrreversibleActionEnv
from server.tasks.cloud_infra import CloudInfraTask
from server.tasks.easy import FileCleanupTask
from server.tasks.hard import ServerMigrationTask
from server.tasks.medical import MedicalTriageTask
from server.tasks.medium import DatabaseMaintenanceTask

# ── Task registry for grader endpoint ────────────────────────
_GRADER_REGISTRY = {
    "easy": FileCleanupTask,
    "medium": DatabaseMaintenanceTask,
    "hard": ServerMigrationTask,
    "medical": MedicalTriageTask,
    "cloud_infra": CloudInfraTask,
}

_ACTION_SCHEMA = {
    "required": ["action_name", "parameters", "reasoning"],
    "properties": {
        "action_name": {
            "type": "string",
            "description": "Name of the action to execute",
        },
        "parameters": {"type": "object", "description": "Action-specific parameters"},
        "reasoning": {
            "type": "string",
            "minLength": 1,
            "description": "Agent must explain its choice",
        },
    },
}

# ── Create base app from openenv-core ────────────────────────
os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")
app = create_app(
    env=IrreversibleActionEnv,
    action_cls=AgentAction,
    observation_cls=SystemObservation,
    max_concurrent_envs=4,
)

# ── Replace stateless /reset and /step with session-aware versions ──
# The openenv-core creates a fresh env per request which breaks multi-step
# episodes over HTTP. We store envs keyed by episode_id.
_ENV_SESSIONS: dict[str, tuple[IrreversibleActionEnv, float]] = {}

_SESSION_TTL = 300  # 5 minutes


def _cleanup_stale_sessions() -> None:
    now = time.time()
    stale = [k for k, (_, ts) in _ENV_SESSIONS.items() if now - ts > _SESSION_TTL]
    for k in stale:
        del _ENV_SESSIONS[k]


# Remove the default /reset and /step routes so ours take precedence
app.router.routes = [
    r
    for r in app.router.routes
    if getattr(r, "path", None) not in ("/reset", "/step", "/state")
]


_VERSION = importlib.metadata.version("my-openenv")

logger = logging.getLogger(__name__)


@app.get("/health")
def health():
    return {"status": "ok", "environment": "safeact-env", "version": _VERSION}


class ResetRequest(BaseModel):
    task_name: str = "easy"
    episode_id: str | None = None
    seed: int | None = None


class StepRequest(BaseModel):
    action: dict[str, Any]
    episode_id: str | None = None


def _serialize_observation(obs: SystemObservation) -> dict:
    data = obs.model_dump()
    return {
        "observation": data,
        "reward": data.get("reward", 0.0),
        "done": data.get("done", False),
    }


@app.post("/reset")
def reset_episode(request: ResetRequest):
    _cleanup_stale_sessions()
    episode_id = request.episode_id or str(uuid.uuid4())
    env = IrreversibleActionEnv()
    obs = env.reset(
        seed=request.seed, episode_id=episode_id, task_name=request.task_name
    )
    _ENV_SESSIONS[episode_id] = (env, time.time())
    return _serialize_observation(obs)


@app.post("/step")
def step_episode(request: StepRequest):
    # Find env by episode_id, or fall back to most recent session
    env = None
    if request.episode_id and request.episode_id in _ENV_SESSIONS:
        env, _ = _ENV_SESSIONS[request.episode_id]
    elif _ENV_SESSIONS:
        env, _ = next(reversed(_ENV_SESSIONS.values()))

    if env is None:
        raise HTTPException(
            status_code=400, detail="No active episode. Call /reset first."
        )

    action = AgentAction(**request.action)
    obs = env.step(action)

    # Clean up completed episodes
    if getattr(obs, "done", False):
        _ENV_SESSIONS.pop(request.episode_id, None)
    else:
        _ENV_SESSIONS[request.episode_id] = (env, time.time())

    return _serialize_observation(obs)


@app.get("/state")
def get_state():
    if _ENV_SESSIONS:
        env, _ = next(reversed(_ENV_SESSIONS.values()))
        return env.state
    return {}


@app.get("/demo", response_class=HTMLResponse)
def demo_ui():
    return HTMLResponse(
        content=(Path(__file__).parent.parent / "static" / "demo.html").read_text()
    )


# ── Request/Response models ───────────────────────────────────


class GraderRequest(BaseModel):
    task_name: str
    history: list[dict[str, Any]]
    final_state: dict[str, Any]


class GraderResponse(BaseModel):
    score: float


class BaselineRequest(BaseModel):
    task_name: str | None = None


class BaselineResponse(BaseModel):
    scores: dict[str, float | None]


# ── Hackathon-required endpoints ──────────────────────────────


@app.get("/tasks")
def get_tasks():
    """List all tasks with their action schemas."""
    tasks = []
    for name, config in _TASK_CONFIG.items():
        tasks.append(
            {
                "name": name,
                "description": config["description"],
                "max_steps": config["max_steps"],
                "difficulty": config["difficulty"],
                "action_schema": _ACTION_SCHEMA,
            }
        )
    return {"tasks": tasks}


@app.post("/grader", response_model=GraderResponse)
def run_grader(request: GraderRequest):
    """Score a completed episode deterministically."""
    if request.task_name not in _GRADER_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task: {request.task_name!r}. "
            f"Valid: {list(_GRADER_REGISTRY)}",
        )
    task = _GRADER_REGISTRY[request.task_name]()
    history = [ActionRecord(**r) for r in request.history]
    score = task.grade(history, request.final_state)
    return GraderResponse(score=score)


@app.post("/baseline", response_model=BaselineResponse)
def run_baseline(request: BaselineRequest):
    """Trigger baseline agent run via subprocess."""
    tasks = [request.task_name] if request.task_name else list(_GRADER_REGISTRY.keys())

    # Early exit if no LLM credentials are available
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get(
        "AZURE_OPENAI_API_KEY"
    ):
        return BaselineResponse(scores={t: None for t in tasks})

    scores: dict[str, float | None] = {}

    for task in tasks:
        try:
            result = subprocess.run(
                [sys.executable, "scripts/baseline.py", "--task", task, "--json"],
                capture_output=True,
                text=True,
                timeout=90,
            )
            data = json.loads(result.stdout.strip())
            scores[task] = data["score"]
        except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError) as exc:
            logger.warning("Baseline run failed for task %s: %s", task, exc)
            scores[task] = None

    return BaselineResponse(scores=scores)


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
